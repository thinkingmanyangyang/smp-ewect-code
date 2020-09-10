from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertLayerNorm, BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu
from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from net.utils.focal_loss import FocalLoss
from transformers import BertForSequenceClassification

class BertForSequenceClassificationDE(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.args = args
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.nums = 1
        hidden_size_changed = config.hidden_size * self.nums
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size_changed, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.attention = Attention(config)
        # if args.fl_gamma:
        #     self.label_focal_loss = FocalLoss(num_class=config.num_labels,
        #                                    gamma=args.fl_gamma)
        self.label_focal_loss = FocalLoss(num_class=config.num_labels,
                                          gamma=1)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        original_logits = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        hidden_output = outputs[2]
        # last_cat = ()
        # for i in range(self.nums):
        #     last_cat += (hidden_output[-i-1][:, 0],)
        # # last_cat = torch.stack(last_cat, 1)
        # # last_cat = torch.matmul(torch.nn.functional.softmax(self.avg_weight, dim=-1),
        #                         # last_cat)
        # last_cat = torch.cat(last_cat, 1)
        # last_cat = torch.squeeze(last_cat)
        # pooled_output = self.pooler(last_cat)
        pooled_output = self.attention(hidden_output[-1], attention_mask)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.pooler(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # loss = self.label_focal_loss(logits.view(-1, self.num_labels), labels.view(-1))
                probs = torch.nn.functional.softmax(logits.view(-1, self.num_labels), dim=-1)
                # loss +=  -2e-2 * torch.sum(probs * torch.log(probs))
                # loss += -3e-2 * torch.norm(probs, 'nuc')
            if original_logits is not None:
                original_probs = torch.nn.functional.softmax(original_logits.view(-1, self.num_labels), dim=-1)
                probs = torch.nn.functional.softmax(logits.view(-1, self.num_labels), dim=-1)
                loss = torch.nn.functional.kl_div(original_probs, probs)
                loss = 0.5 * loss
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

import math
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_weight = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_num = 1
    def forward(self, H, mask):
        # mask (batch_size, seq_length)
        mask = (mask > 0).unsqueeze(-1).repeat(1, 1, self.head_num)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = self.attn_weight(H)
        hidden_size = H.size(-1)
        scores /= math.sqrt(float(hidden_size))
        scores += mask
        probs = nn.Softmax(dim=-2)(scores)
        H = H.transpose(-1, -2)
        output = torch.bmm(H, probs)
        output = torch.reshape(output, (-1, hidden_size * self.head_num))
        return output