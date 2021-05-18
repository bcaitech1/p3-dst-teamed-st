from data_utils import DSTPreprocessor
import torch
from data_utils import CoCoClassifierInputFeature, CoCoGenInputFeature

class CoCoPreprocessor:

    def __init__(
            self,
            slot_meta,
            gen_tokenizer,
            cls_tokenizer,
            config,
            max_seq_length=512,
    ):
        self.slot_meta = slot_meta
        self.gen_tokenizer = gen_tokenizer
        self.cls_tokenizer = cls_tokenizer
        self.max_seq_length = max_seq_length
        self.config = config
        self.slot2idx = {slot: i for i, slot in enumerate(slot_meta)}
        self.idx2slot = {i: slot for i, slot in enumerate(slot_meta)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def gen_convert_example_to_feature(self, example):
        """ CoCoGenInputExamples -> CoCoGenInputFeature """
        sys = self.gen_tokenizer.tokenize(example.system_utter)
        turn_state = ', '.join([s.replace('-', ' ') for s in example.turn_state])
        state = self.gen_tokenizer.tokenize(turn_state)
        user = [self.gen_tokenizer.bos_token] + self.gen_tokenizer.tokenize(example.user_utter) + [self.gen_tokenizer.eos_token]

        input_tokens = [self.gen_tokenizer.bos_token] + sys + [self.gen_tokenizer.eos_token] + state + [self.gen_tokenizer.eos_token]
        input_id = self.gen_tokenizer.convert_tokens_to_ids(input_tokens)
        target_id = self.gen_tokenizer.convert_tokens_to_ids(user)

        return CoCoGenInputFeature(input_id=input_id, target_id=target_id)


    def gen_collate_fn(self, batch):
        input_ids = torch.LongTensor(self.gen_tokenizer.pad_ids([b.input_id for b in batch], self.gen_tokenizer.pad_token_id))
        target_ids = torch.LongTensor(self.gen_tokenizer.pad_ids([b.target_id for b in batch], -100))
        input_masks = input_ids.ne(self.gen_tokenizer.pad_token_id).float()
        return input_ids, target_ids, input_masks


    def cls_convert_example_to_feature(self, example):
        """ CoCoClassifierInputExamples -> CoCoClassifierInputFeature """
        sys = self.cls_tokenizer.tokenize(example.system_utter)
        user = self.cls_tokenizer.tokenize(example.user_utter)
        sys_token = [self.cls_tokenizer.cls_token] + sys + [self.cls_tokenizer.sep_token]
        user_token = user + [self.cls_tokenizer.sep_token]

        sys_id = self.cls_tokenizer.convert_tokens_to_ids(sys_token)
        user_id = self.cls_tokenizer.convert_tokens_to_ids(user_token)
        input_id = sys_id + user_id

        segment_id = [0] * len(sys_id) + [1] * len(user_id)
        input_mask = [1] * len(input_id)

        slots = [0] * len(self.slot2idx)
        for state in example.turn_state:
            slot = '-'.join(state.split('-')[:-1])
            slots[self.slot2idx[slot]] = 1

        return CoCoClassifierInputFeature(input_id=input_id, input_mask=input_mask, segment_id=segment_id,
                                          target_id=slots)


    def cls_collate_fn(self, batch):
        input_ids = torch.tensor(self.pad_ids([b.input_id for b in batch], self.cls_tokenizer.pad_token_id), dtype=torch.long,
                                 device=self.device)
        input_masks = torch.tensor(input_ids.ne(self.cls_tokenizer.pad_token_id), dtype=torch.long, device=self.device)
        segment_ids = torch.tensor(self.pad_ids([b.segment_id for b in batch], self.cls_tokenizer.pad_token_id), dtype=torch.long,
                                   device=self.device)
        target_ids = torch.tensor([b.target_id for b in batch], dtype=torch.float32, device=self.device)
        return input_ids, input_masks, segment_ids, target_ids


    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays


