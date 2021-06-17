import torch
import numpy as np
from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict


class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
        word_drop=0.0,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2, "yes": 3, "no": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length
        self.word_drop = word_drop
        print(f"Word drop: {self.word_drop}")

    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)

        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:]

        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        firt_sep_idx = input_id.index(self.src_tokenizer.sep_token_id)
        segment_id = [0] * len(input_id[: firt_sep_idx + 1]) + [1] * len(
            input_id[firt_sep_idx + 1 :]
        )
        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
                continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        if self.word_drop > 0.0:
            input_ids = []
            for b in batch:
                drop_mask = (
                    np.array(
                        self.src_tokenizer.get_special_tokens_mask(
                            b.input_id, already_has_special_tokens=True
                        )
                    )
                    == 0
                ).astype(int)
                word_drop = np.random.binomial(drop_mask, self.word_drop)
                input_id = [
                    token_id if word_drop[i] == 0 else self.src_tokenizer.unk_token_id
                    for i, token_id in enumerate(b.input_id)
                ]
                input_ids.append(input_id)
        else:
            input_ids = [b.input_id for b in batch]
        input_ids = torch.LongTensor(
            self.pad_ids(
                [b for b in input_ids],
                self.src_tokenizer.pad_token_id,
                max_length=512,
            )
        )
        segment_ids = torch.LongTensor(
            self.pad_ids(
                [b.segment_id for b in batch],
                self.src_tokenizer.pad_token_id,
                max_length=512,
            )
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids


OP2ID = {
    4: {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3},
    6: {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3, "yes": 4, "no": 5},
}


class SOMDSTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
        n_op=4,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.op2id = OP2ID[n_op]
        self.id2op = {v: k for k, v in self.op2id.items()}
        # self.domain2id = {"관광": 0, "숙소": 1, "식당": 2, "지하철": 3, "택시": 4}
        self.domain2id = {
            "관광": 0,
            "숙소": 1,
            "식당": 2,
            "지하철": 3,
            "택시": 4,
            "관광, 숙소": 5,
            "관광, 식당": 6,
            "관광, 지하철": 7,
            "관광, 택시": 8,
            "숙소, 식당": 9,
            "숙소, 지하철": 10,
            "숙소, 택시": 11,
            "식당, 지하철": 12,
            "식당, 택시": 13,
            "None": 14,
        }

        self.id2domain = {v: k for k, v in self.domain2id.items()}
        self.prev_example = None
        self.prev_state = {}
        self.prev_domain_id = None
        self.slot_id = self.src_tokenizer.convert_tokens_to_ids("[SLOT]")
        self.max_seq_length = max_seq_length
        self.n_op = n_op

    def _convert_example_to_feature(self, example):
        if not example.context_turns:
            self.reset_state()
        if self.prev_example:
            d_prev = " ; ".join(self.prev_example.current_turn)

        else:
            d_prev = ""
        d_t = " ; ".join(example.current_turn)
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        b_prev_state = []
        op_ids = []
        target_ids = []
        for slot in self.slot_meta:
            prev_value = self.prev_state.get(slot, "[NULL]")
            value = state.get(slot, "[NULL]")

            if value == prev_value:
                operation = self.op2id["carryover"]
            elif value == "[NULL]":
                operation = self.op2id["delete"]
            elif value == "doncare":
                operation = self.op2id["dontcare"]
            elif "yes" in self.op2id and value == "yes":
                operation = self.op2id["yes"]
            elif "no" in self.op2id and value == "no":
                operation = self.op2id["no"]
            else:
                operation = self.op2id["update"]
                target_id = self.trg_tokenizer.encode(
                    value + " [EOS]", add_special_tokens=False
                )
                target_ids.append(target_id)
            if prev_value == "dontcare":
                prev_value = "dont care"
            b_prev_state.extend(["[SLOT]"])
            b_prev_state.extend(slot.split("-"))
            b_prev_state.extend(["-", prev_value])
            op_ids.append(operation)
        b_prev_state = " ".join(b_prev_state)
        tokenized = self.src_tokenizer(
            d_prev,
            d_t + " [SEP] " + b_prev_state,
            padding=True,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )

        slot_positions = []
        for i, input_id in enumerate(tokenized.input_ids):
            if input_id == self.slot_id:
                slot_positions.append(i)

        if not self.prev_example:
            domain_slot = list(state.keys())
            if domain_slot:
                # domain_id = self.domain2id[domain_slot[0].split("-")[0]]
                doms = ", ".join(sorted(set([d.split("-")[0] for d in domain_slot])))
                domain_id = self.domain2id[doms]
            else:
                domain_id = self.domain2id["None"]
        else:
            update_state = set(example.label) - set(self.prev_example.label)
            delete_state = set(self.prev_example.label) - set(example.label)
            diff_state = update_state | delete_state
            if not diff_state:
                domain_id = self.domain2id["None"]
            else:
                doms = ", ".join(sorted(set([d.split("-")[0] for d in diff_state])))
                domain_id = self.domain2id[doms]
        self.prev_example = example
        self.prev_state = state
        self.prev_domain_id = domain_id
        return OpenVocabDSTFeature(
            example.guid,
            tokenized.input_ids,
            tokenized.token_type_ids,
            op_ids,
            target_ids,
            slot_positions,
            domain_id,
        )

    def reset_state(self):
        self.prev_example = None
        self.prev_state = {}
        self.prev_domain_id = 0

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, pred_ops, gen_list):
        recovered = []
        gid = 0
        for slot, op in zip(self.slot_meta, pred_ops):
            if op == "dontcare":
                self.prev_state[slot] = "dontcare"
            elif op == "delete" and slot in self.prev_state:
                self.prev_state.pop(slot)
            elif "yes" in self.op2id and op == "yes":
                self.prev_state[slot] = "yes"
                recovered.append(f"{slot}-yes")
            elif "no" in self.op2id and op == "no":
                self.prev_state[slot] = "no"
                recovered.append(f"{slot}-no")
            elif op == "update":
                tokens = self.trg_tokenizer.convert_ids_to_tokens(gen_list[gid])
                gen = []
                for token in tokens:
                    if token == "[EOS]":
                        break
                    gen.append(token)
                gen = " ".join(gen).replace(" ##", "")
                gid += 1
                gen = gen.replace(" : ", ":").replace("##", "")
                if gen == "[NULL]" and slot in self.prev_state:
                    self.prev_state.pop(slot)
                else:
                    self.prev_state[slot] = gen
                    recovered.append(f"{slot}-{gen}")
            else:
                prev_value = self.prev_state.get(slot)
                if prev_value:
                    recovered.append(f"{slot}-{prev_value}")
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor(
            self.pad_ids(
                [b.input_id for b in batch],
                self.src_tokenizer.pad_token_id,
                max_length=self.max_seq_length,
            )
        )
        segment_ids = torch.LongTensor(
            self.pad_ids(
                [b.segment_id for b in batch],
                self.src_tokenizer.pad_token_id,
                max_length=self.max_seq_length,
            )
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        domain_ids = torch.LongTensor([b.domain_id for b in batch])
        target_ids = [b.target_ids for b in batch]
        slot_position_ids = torch.LongTensor([b.slot_positions for b in batch])
        max_update = max([len(b) for b in target_ids])
        max_value = max([len(t) for b in target_ids for t in b] + [10])
        for bid, b in enumerate(target_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            target_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        target_ids = torch.LongTensor(target_ids)
        return (
            input_ids,
            input_masks,
            segment_ids,
            slot_position_ids,
            gating_ids,
            domain_ids,
            target_ids,
            max_update,
            max_value,
            guids,
        )
