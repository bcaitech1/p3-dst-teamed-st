import torch
import numpy as np
from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict, _truncate_seq_pair, OntologyDSTFeature

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


class TRADEPreprocessor(DSTPreprocessor):
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
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length
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
        segment_id = [0] * len(input_id)

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
        input_ids = torch.LongTensor(
            self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids


class SUMBTPreprocessor(DSTPreprocessor):
    def __init__(
            self,
            slot_meta,
            src_tokenizer,
            trg_tokenizer=None,
            ontology=None,
            max_seq_length=64,
            max_turn_length=14,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq_length = max_seq_length  # N
        self.max_turn_length = max_turn_length  # M

    def _convert_example_to_feature(self, example):
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        num_turn = None
        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2
            uttrs = []
            for segment_idx, uttr in enumerate(turn.current_turn):
                token = self.src_tokenizer.encode(uttr, add_special_tokens=False)
                uttrs.append(token)

            _truncate_seq_pair(uttrs[0], uttrs[1], self.max_seq_length - 3)
            tokens = (
                    [self.src_tokenizer.cls_token_id]
                    + uttrs[0]
                    + [self.src_tokenizer.sep_token_id]
                    + uttrs[1]
                    + [self.src_tokenizer.sep_token_id]
            )
            token_type = [0] * (len(uttrs[0]) + 2) + [1] * (len(uttrs[1]) + 1)
            if len(tokens) < self.max_seq_length:
                gap = self.max_seq_length - len(tokens)
                tokens.extend([self.src_tokenizer.pad_token_id] * gap)
                token_type.extend([0] * gap)
            turns.append(tokens)
            token_types.append(token_type)
            label = []
            if turn.label:
                slot_dict = convert_state_dict(turn.label)
            else:
                slot_dict = {}
            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")
                # TODO
                # raise Exception('label_idx를 ontology에서 꺼내오는 코드를 작성하세요!')
                if value in self.ontology[slot_type]:
                    label_idx = self.ontology[slot_type].index(value)
                else:
                    label_idx = self.ontology[slot_type].index("none")
                label.append(label_idx)  # 45
            labels.append(label)  # turn length, 45
        num_turn = len(turns)
        if num_turn < self.max_turn_length:
            gap = self.max_turn_length - num_turn
            for _ in range(gap):
                dummy_turn = [self.src_tokenizer.pad_token_id] * self.max_seq_length
                turns.append(dummy_turn)
                token_types.append(dummy_turn)
                dummy_label = [-1] * len(self.slot_meta)
                labels.append(dummy_label)
        return OntologyDSTFeature(
            guid=guid,
            input_ids=turns,
            segment_ids=token_types,
            num_turn=num_turn,
            target_ids=labels,
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, pred_slots, num_turn):
        states = []
        for pred_slot in pred_slots[:num_turn]:
            state = []
            for s, p in zip(self.slot_meta, pred_slot):
                v = self.ontology[s][p]
                if v != "none":
                    state.append(f"{s}-{v}")
            states.append(state)
        return states

    def collate_fn(self, batch):
        # list를 batch level로 packing
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor([b.input_ids for b in batch])
        segment_ids = torch.LongTensor([b.segment_ids for b in batch])
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)
        target_ids = torch.LongTensor([b.target_ids for b in batch])
        num_turns = [b.num_turn for b in batch]
        return input_ids, segment_ids, input_masks, target_ids, num_turns, guids
