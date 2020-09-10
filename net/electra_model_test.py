from transformers import ElectraForSequenceClassification, ElectraModel
from transformers.modeling_electra import ElectraClassificationHead, ElectraPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.num_labels = config.num_labels
        print(self.num_labels)
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

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
        output_attentions=None,
        output_hidden_states=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        # print(self.num_labels, logits.shape)
        outputs = (logits,) + discriminator_hidden_states[1:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
