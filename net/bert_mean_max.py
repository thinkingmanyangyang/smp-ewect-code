from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertLayerNorm, BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu
from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from net.utils.focal_loss import FocalLoss
from net.bert_transfer_learning import BertPreTrainedModelTransferLearning
from transformers import BertForSequenceClassification

class BertForSequenceClassificationMeanMax(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.args = args
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.nums = 3
        hidden_size_changed = config.hidden_size * self.nums
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size_changed, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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

        pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        hidden_output = outputs[2]
        # last_cat = ()
        # for i in range(self.nums):
        #     last_cat += (hidden_output[-i-1][:, 0],)
        # # last_cat = torch.stack(last_cat, 1)
        # # last_cat = torch.matmul(torch.nn.functional.softmax(self.avg_weight, dim=-1),
        #                         # last_cat)
        # last_cat = torch.cat(last_cat, 1)
        last_mean = torch.mean(hidden_output[-1][:, 1:, :], dim=1)
        last_max, _ = torch.max(hidden_output[-1][:, 1:, :], dim=1)
        last_cat = torch.cat((pooled_output, last_max, last_mean), dim=-1)
        last_cat = torch.squeeze(last_cat)
        pooled_output = self.pooler(last_cat)
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
                loss = torch.nn.functional.kl_div(
                    torch.nn.functional.softmax(original_logits.view(-1, self.num_labels), dim=-1),
                    torch.nn.functional.softmax(logits.view(-1, self.num_labels), dim=-1),
                    reduction='mean'
                )
                loss = 0.5 * loss
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertForSequenceClassificationMeanMaxTransferLearning(BertPreTrainedModelTransferLearning):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.args = args
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.nums = 3
        hidden_size_changed = config.hidden_size * self.nums
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size_changed, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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

        pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        hidden_output = outputs[2]
        # last_cat = ()
        # for i in range(self.nums):
        #     last_cat += (hidden_output[-i-1][:, 0],)
        # # last_cat = torch.stack(last_cat, 1)
        # # last_cat = torch.matmul(torch.nn.functional.softmax(self.avg_weight, dim=-1),
        #                         # last_cat)
        # last_cat = torch.cat(last_cat, 1)
        last_mean = torch.mean(hidden_output[-1][:, 1:, :], dim=1)
        last_max, _ = torch.max(hidden_output[-1][:, 1:, :], dim=1)
        last_cat = torch.cat((pooled_output, last_max, last_mean), dim=-1)
        last_cat = torch.squeeze(last_cat)
        pooled_output = self.pooler(last_cat)
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
                loss = torch.nn.functional.kl_div(
                    torch.nn.functional.softmax(original_logits.view(-1, self.num_labels), dim=-1),
                    torch.nn.functional.softmax(logits.view(-1, self.num_labels), dim=-1),
                    reduction='mean'
                )
                loss = 0.5 * loss
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
