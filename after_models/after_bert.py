import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel

from transformers.activations import gelu, gelu_new, swish
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING, BertModel

from after_models.after_modeling_utils import GradientReversal

logger = logging.getLogger(__name__)


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class AfterBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, lambd, mean_pool=False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.lambd = lambd
        print("Gradient reversal Prarameter is: {}".format(self.lambd))
        self.mean_pool = mean_pool
        if self.mean_pool: print("Using mean pooling instead of CLS for domain classifier")
        self.grl = GradientReversal(self.lambd)
        self.domain_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            aux=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.mean_pool:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs[0].size()).float()
            sum_embeddings = torch.sum(outputs[0] * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            dom_pooled_output = sum_embeddings / sum_mask
        else:
            dom_pooled_output = outputs[1]

        dom_pooled_output = self.dropout(dom_pooled_output)
        reversed_pooled_output = self.grl(dom_pooled_output)

        logits = self.classifier(pooled_output)
        dom_logits = self.domain_classifier(reversed_pooled_output)

        outputs = (logits,) + (dom_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels[:, 0].view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels[:, 0].view(-1)) if not aux else None
            dom_loss_fct = CrossEntropyLoss()
            dom_loss = dom_loss_fct(dom_logits.view(-1, 2), labels[:, 1].view(-1))
            outputs = (loss,) + (dom_loss,) + outputs

        return outputs  # (main_loss), (dom_loss), logits, domain_logits (hidden_states), (attentions)
