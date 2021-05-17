import torch
import numpy as np
from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict

class SOMDSTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.op2id = {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3}
        self.id2op = {v: k for k, v in self.op2id.items()}
        self.domain2id = {"관광": 0, "숙소": 1, "식당": 2, "지하철": 3, "택시": 4}
        self.id2domain = {v: k for k, v in self.domain2id.items()}
        self.prev_example = None
        self.prev_state = {}
        self.prev_domain_id = None
        self.slot_id = self.src_tokenizer.convert_tokens_to_ids("[SLOT]")
        self.max_seq_length = max_seq_length

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
                operation = self.op2id["carryover"]     # 변경되지 않는 value
            elif value == "[NULL]":
                operation = self.op2id["delete"]    # 이전 turn에 NULL이 아니였는데 NULL이 되어야 할 때
            elif value == "doncare":
                operation = self.op2id["dontcare"]
            else:
                operation = self.op2id["update"]    # 업데이트 해야하는 경우
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
                domain_id = self.domain2id[domain_slot[0].split("-")[0]]
            else:
                domain_id = self.prev_domain_id
        else:
            diff_state = set(example.label) - set(self.prev_example.label)
            if not diff_state:
                domain_id = self.prev_domain_id
            else:
                domain_id = self.domain2id[list(diff_state)[0].split("-")[0]]

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
            elif op == "delete":
                if not slot in self.prev_state:
                    print("delete error")
                    continue
                self.prev_state.pop(slot)
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
