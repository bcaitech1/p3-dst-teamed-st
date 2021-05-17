from transformers import ElectraModel
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertOnlyMLMHead


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


# class TRADE(nn.Module):
#     def __init__(self, config, slot_vocab, slot_meta, pad_idx=0):
#         super(TRADE, self).__init__()
#         self.slot_meta = slot_meta
#         if config.model_name_or_path:
#             self.encoder = BertModel.from_pretrained(config.model_name_or_path)
#         else:
#             self.encoder = BertModel(config)
#
#         self.decoder = SlotGenerator(
#             config.vocab_size,
#             config.hidden_size,
#             config.hidden_dropout_prob,
#             config.n_gate,
#             None,
#             pad_idx,
#         )
#
#         # init for only subword embedding
#         self.decoder.set_slot_idx(slot_vocab)
#         self.tie_weight()
#
#
#     def tie_weight(self):
#         """
#         decoder embed weight랑 encoder embed weight를 일치시켜주는 함수
#         """
#         self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight
#         # if self.decoder.proj_layer:
#         #     self.decoder.proj_layer.weight = self.encoder.proj_layer.weight
#
#     def forward(
#         self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None
#     ):
#
#         encoder_outputs, pooled_output = self.encoder(input_ids=input_ids) # pooled_output -> gru의 마지막 hidden state
#         all_point_outputs, all_gate_outputs = self.decoder(
#             input_ids,
#             encoder_outputs,    # batch, vocab_size, emb_size
#             pooled_output.unsqueeze(0),     # 1, batch, emb_size
#             attention_mask,     # batch, vocab_size
#             max_len,
#             teacher,
#         )
#
#         return all_point_outputs, all_gate_outputs
#
#
#
# class SlotGenerator(nn.Module):
#     def __init__(
#         self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
#     ):
#         super(SlotGenerator, self).__init__()
#         self.pad_idx = pad_idx
#         self.vocab_size = vocab_size
#         self.embed = nn.Embedding(
#             vocab_size, hidden_size, padding_idx=pad_idx
#         )  # shared with encoder
#
#         if proj_dim:
#             self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
#         else:
#             self.proj_layer = None
#         self.hidden_size = proj_dim if proj_dim else hidden_size
#
#         self.gru = nn.GRU(  # pointer Generator
#             self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
#         )
#         self.n_gate = n_gate # none, doncare, ptr
#         self.dropout = nn.Dropout(dropout)
#         self.w_gen = nn.Linear(self.hidden_size * 3, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.w_gate = nn.Linear(self.hidden_size, n_gate) # slot Gate Classifier
#
#     def set_slot_idx(self, slot_vocab_idx):     # packing
#         whole = []
#         max_length = max(map(len, slot_vocab_idx))
#         for idx in slot_vocab_idx:
#             if len(idx) < max_length:
#                 gap = max_length - len(idx)
#                 idx.extend([self.pad_idx] * gap)
#             whole.append(idx)
#         self.slot_embed_idx = whole  # torch.LongTensor(whole)
#
#     def embedding(self, x):
#         x = self.embed(x)
#         if self.proj_layer:
#             x = self.proj_layer(x)
#         return x
#
#     def forward(
#         self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
#     ):
#         input_masks = input_masks.ne(1)     # not equal , (batch, vocab_size)
#         # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
#         # J,2
#         batch_size = encoder_output.size(0)
#         # slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)  # 45, 4
#         slot = torch.tensor(self.slot_embed_idx, device=input_ids.device)
#         slot_e = torch.sum(self.embedding(slot), 1)  # J, 4, 768 -> J , 768
#         J = slot_e.size(0)
#
#         all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size, device=input_ids.device)
#
#         # Parallel Decoding
#         w = slot_e.repeat(batch_size, 1).unsqueeze(1)   # (BS * J, 1, embed_size)
#         hidden = hidden.repeat_interleave(J, dim=1) # (1, BS, embed_size) -> (1, BS * J, embed_size)
#         encoder_output = encoder_output.repeat_interleave(J, dim=0) # (BS, vocab, embed) -> (BS * J, vocab, embed)
#         input_ids = input_ids.repeat_interleave(J, dim=0) # (BS, vocab) -> (BS * J, vocab)
#         input_masks = input_masks.repeat_interleave(J, dim=0) # (BS, vocab) -> (BS * J, vocab)
#         # input_masks = input_masks.to(dtype=hidden.dtype)
#         # input_masks = input_masks * -10000.0
#         for k in range(max_len):    # B = BS * J, D = embed
#             w = self.dropout(w)
#             _, hidden = self.gru(w, hidden)  # 1,B,D
#
#             # B,T,D * B,D,1 => B,T
#             attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
#             attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e4)
#             attn_history = F.softmax(attn_e, -1)  # B,T
#
#             if self.proj_layer:
#                 hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
#             else:
#                 hidden_proj = hidden # (1, BS * J, embed_size)
#
#             # B,D * D,V => B,V
#             attn_v = torch.matmul(
#                 hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
#             )  # B,V
#             attn_vocab = F.softmax(attn_v, -1) # p_vocab / 어떤 vocab을 예측할건지
#
#             # B,1,T * B,T,D => B,1,D
#             context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
#             p_gen = self.sigmoid(
#                 self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)) # B , 1, 3 * D를 w_gen에 넣어줌
#             )  # B,1
#             p_gen = p_gen.squeeze(-1) # 720, 1 -> 720
#
#             p_context_ptr = torch.zeros_like(attn_vocab, device=input_ids.device)
#             p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
#             p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
#             _, w_idx = p_final.max(-1)
#
#             if teacher is not None:
#                 w = self.embedding(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
#             else:
#                 w = self.embedding(w_idx).unsqueeze(1)  # B,1,D
#             if k == 0:
#                 gated_logit = self.w_gate(context.squeeze(1))  # (B,3)
#                 all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate) # (16, 45, 3)
#             all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)
#
#         return all_point_outputs, all_gate_outputs # (16, 45, 9, 35000) , (16, 45 ,3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TRADE(nn.Module):
    def __init__(self, config, slot_vocab, slot_meta, pad_idx=0):
        super(TRADE, self).__init__()
        self.slot_meta = slot_meta
        if config.model_name_or_path:
            self.encoder = BertModel.from_pretrained(config.model_name_or_path)
        else:
            self.encoder = BertModel(config)

        if config.vocab_size != self.encoder.embeddings.word_embeddings.num_embeddings:
            self.encoder.resize_token_embeddings(config.vocab_size)

        self.decoder = SlotGenerator(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.n_gate,
            None,
            pad_idx,
        )

        # init for only subword embedding
        self.decoder.set_slot_idx(slot_vocab)
        self.tie_weight()

    def tie_weight(self):
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None):

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids, encoder_outputs, pooled_output.unsqueeze(0), attention_mask, max_len, teacher
        )

        return all_point_outputs, all_gate_outputs


class SlotGenerator(nn.Module):
    def __init__(
            self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
    ):
        super(SlotGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = proj_dim if proj_dim else hidden_size

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate)

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
            self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        # slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)  ##
        slot = torch.tensor(self.slot_embed_idx, device=input_ids.device, dtype=input_ids.dtype)
        slot_e = torch.sum(self.embedding(slot), 1)  # J,d
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size, device=input_ids.device)

        # Parallel Decoding
        w = slot_e.repeat(batch_size, 1).unsqueeze(1)
        hidden = hidden.repeat_interleave(J, dim=1)
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        input_ids = input_ids.repeat_interleave(J, dim=0)
        input_masks = input_masks.repeat_interleave(J, dim=0)
        for k in range(max_len):
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D

            # B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e4)
            attn_history = F.softmax(attn_e, -1)  # B,T

            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            )  # B,V
            attn_vocab = F.softmax(attn_v, -1)

            # B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  # B,1
            p_gen = p_gen.squeeze(-1)

            p_context_ptr = torch.zeros_like(attn_vocab, device=input_ids.device)
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            _, w_idx = p_final.max(-1)

            if teacher is not None:
                w = self.embedding(teacher[:, :, k]).transpose(0, 1).reshape(batch_size * J, 1, -1)
            else:
                w = self.embedding(w_idx).unsqueeze(1)  # B,1,D
            if k == 0:
                gated_logit = self.w_gate(context.squeeze(1))  # B,3
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)

        return all_point_outputs, all_gate_outputs


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # q == 3060, 4, 1, 192
        # k == 3060, 4, 64, 192
        # scores == 3060, 4, 1, 64
        if mask is not None:
            mask = mask.unsqueeze(1) # 3060, 1, 1, 64

            # fp 16 호환용
            mask = mask.to(dtype=scores.dtype)
            mask = (1.0 - mask) * -10000.0
            scores = scores + mask

            # scores = scores.masked_fill(mask == 0, -1e4)
        scores = F.softmax(scores, dim=-1) # 3060, 4, 1, 64

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # batch_size

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # batch_size, sl, head_num, uttr_hidden // head_num
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # 3060, 768 -> 3060, 1, 4, 192
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # 3060, 64, 768 -> 3060, 64, 4, 192

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)   # 3060, 64, 4, 192 -> 3060, 4, 64, 192
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class SUMBT(nn.Module):
    def __init__(self, args, num_labels, device):
        super(SUMBT, self).__init__()

        self.hidden_dim = args.hidden_dim # 300
        self.rnn_num_layers = args.num_rnn_layers # 1
        self.zero_init_rnn = args.zero_init_rnn # False
        self.max_seq_length = args.max_seq_length # 64
        self.max_label_length = args.max_label_length # 12
        self.num_labels = num_labels # 45 label value length
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head # 4
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder: # default = False
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim) # (45, utter hidden_size)
        self.value_lookup = nn.ModuleList(
            [nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels] # (slot의 value 갯수, utter hidden size)
        ) # 45, slot_value_num, uttr_hidden_size

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0) # 4, utter hidden_size

        ### RNN Belief Tracker
        self.nbt = nn.GRU(
            input_size=self.bert_output_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_num_layers,
            dropout=self.hidden_dropout_prob,
            batch_first=True, # batch_size를 맨앞 shape로
        )
        self.init_parameter(self.nbt)

        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob),
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long, device=slot_ids.device)

        slot_mask = slot_ids > 0
        hid_slot, _ = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        )
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long, device=label_id.device)
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(
                label_id.view(-1, self.max_label_length),
                label_type_ids.view(-1, self.max_label_length),
                label_mask.view(-1, self.max_label_length),
            )
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")
        self.sv_encoder = None

    def forward(
        self,
        input_ids,
        token_type_ids, # segment_ids
        attention_mask, # input_masks
        labels=None, # target_ids
        n_gpu=1,
        target_slot=None,
    ):
        # input_ids: [B, M, N] batch size, Max turn size, max sequence length
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # labels: [B, M, J]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots)) # [0, 1, ...., 45]

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden, _ = self.utterance_encoder( # B * M, N, 768
            input_ids.view(-1, self.max_seq_length),    # (B * M , N)
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        )
        hidden = torch.mul(
            hidden, attention_mask.view(-1, self.max_seq_length, 1).expand(hidden.size()).float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*M*B, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [J*M*B, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        ) # J*M*B, 1, H
        hidden = hidden.squeeze()  # h [J*M*B, H] Aggregated Slot Context
        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim, device=self.device)
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1) # 1, 360, 768
            h = self.rnn_init_linear(h) # 1, 360, 300

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [J*B, M, H_GRU]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim, device=self.device)
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out))) # [360, 34, 768]

        hidden = rnn_out.view(slot_dim, ds, ts, -1)  # [J, B, M, H]

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight # [4, H]
            num_slot_labels = hid_label.size(0)

            _hid_label = (
                hid_label.unsqueeze(0)
                .unsqueeze(0)
                .repeat(ds, ts, 1, 1)
                .view(ds * ts * num_slot_labels, -1)
            ) # [B * M * num_slot_labels, H]
            _hidden = (
                hidden[s, :, :, :]
                .unsqueeze(2)
                .repeat(1, 1, num_slot_labels, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            # if self.distance_metric == "euclidean":
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
            _dist = -_dist # [B, M, num_slot_labels]
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        pred_slot = torch.cat(pred_slot, 2) # slot별로 [B, M, 1] 45개 -> [B, M, 45]
        if labels is None:
            return output, pred_slot

        # calculate joint accuracy
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = (
            torch.sum(accuracy, 0).float()
            / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        )
        acc = (
            sum(torch.sum(accuracy, 1) / slot_dim).float()
            / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()
        )  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return (
                loss.unsqueeze(0),
                None,
                acc.unsqueeze(0),
                acc_slot.unsqueeze(0),
                pred_slot.unsqueeze(0),
            )

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, dropout, proj_dim=None, pad_idx=0):
        super(GRUEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if proj_dim:
            self.proj_layer = nn.Linear(d_model, proj_dim, bias=False)
        else:   # default
            self.proj_layer = None

        self.d_model = proj_dim if proj_dim else d_model
        self.gru = nn.GRU(
            self.d_model,
            self.d_model,
            n_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        mask = input_ids.eq(self.pad_idx).unsqueeze(-1) # (batch_size, vocab_size, 1)
        x = self.embed(input_ids)   # (bath_size, vocab_size, embedding_size)
        if self.proj_layer:     # default == None
            x = self.proj_layer(x)
        x = self.dropout(x)
        o, h = self.gru(x)
        o = o.masked_fill(mask, 0.0)
        output = o[:, :, : self.d_model] + o[:, :, self.d_model :] # summation
        hidden = h[0] + h[1]  # n_layer 고려
        return output, hidden   # output -> (batch_size, vocab_size, embedding_size) , hidden -> (batch_size, embedding_size)
