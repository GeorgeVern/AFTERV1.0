from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.file_utils import add_start_docstrings_to_callable, add_start_docstrings
from transformers.modeling_utils import SequenceSummary
from transformers.modeling_xlnet import XLNET_START_DOCSTRING, XLNetPreTrainedModel, XLNetModel, XLNET_INPUTS_DOCSTRING

from after_models.after_modeling_utils import GradientReversal


@add_start_docstrings(
    """XLNet Model with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    XLNET_START_DOCSTRING,
)
class AfterXLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config, lambd):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.lambd = lambd
        print("Gradient reversal Prarameter is: {}".format(self.lambd))
        self.grl = GradientReversal(self.lambd)
        self.domain_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            aux=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`)
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.XLNetConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import XLNetTokenizer, XLNetForSequenceClassification
        import torch

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        reversed_output = self.grl(output)

        logits = self.logits_proj(output)
        dom_logits = self.domain_classifier(reversed_output)

        outputs = (logits,) + (dom_logits,) + transformer_outputs[
                                              1:]  # Keep mems, hidden states, attentions if there are in it

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

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)
