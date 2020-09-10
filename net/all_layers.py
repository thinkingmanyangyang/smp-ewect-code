from transformers import BertPreTrainedModel, BertModel
from transformers import BertForSequenceClassification
from transformers.activations import gelu
from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss

class BertForSequenceClassificationLast2Embedding(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequenceClassificationLast2Embedding, self).__init__(config)
        hidden_size_changed = config.hidden_size * 3
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertModel(config)

        self.classifier = nn.Linear(hidden_size_changed , self.config.num_labels)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
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
        hidden_output = outputs[2]
        last_cat = torch.cat(
            (pooled_output, hidden_output[-1][:, 0], hidden_output[-2][:, 0]),
            1,
        )
        last_cat = gelu(last_cat)
        logits = self.classifier(last_cat)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]

        return outputs  # (loss), logits, (hidden_states), (attentions)
