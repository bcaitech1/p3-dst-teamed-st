import torch.nn as nn


class SOMDST(nn.Module):
    """Some Information about SOMDST"""

    def __init__(self, config, n_domain, n_op):
        super(SOMDST, self).__init__()
        self.encoder = BertEncoder(config, bert, 5, 4)
        self.decoder = Decoder(
            config, self.encoder.bert.embeddings.word_embeddings.weight
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        slot_positions,
        attention_mask,
        op_ids=None,
        max_update=None,
        teacher=None,
    ):
        enc_outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            state_positions=state_positions,
            attention_mask=attention_mask,
            op_ids=op_ids,
            max_update=max_update,
        )
        (
            domain_scores,
            state_scores,
            decoder_inputs,
            sequence_output,
            pooled_output,
        ) = enc_outputs
        gen_scores = self.decoder(
            input_ids,
            decoder_inputs,
            sequence_output,
            pooled_output,
            max_value,
            teacher,
        )

        return (domain_scores,)

        return x


class BertEncoder(nn.Module):
    """Some Information about BertEncoder"""

    def __init__(self, config, bert, n_domain, n_op):
        super(BertEncoder, self).__init__()
        self.config = config
        self.bert = bert
        self.domain_classifier = nn.Linear(config.hidden_size, n_domain)
        self.op_classifier = nn.Linear(config.hidden_size, n_op)

    def forward(self, input_ids, token_type_ids, attention_mask, state_mask):
        self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return x
