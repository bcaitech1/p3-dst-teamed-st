from data_utils import DSTPreprocessor
import torch
from data_utils import CoCoClassifierInputFeature, CoCoGenInputFeature

class CoCoClassifierPreprocessor:

    def __init__(
            self,
            slot_meta,
            tokenizer,
            config,
            max_seq_length=512,
    ):
        self.slot_meta = slot_meta
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.config = config
        self.slot2idx = {slot: i for i, slot in enumerate(slot_meta)}
        self.idx2slot = {i: slot for i, slot in enumerate(slot_meta)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def convert_example_to_feature(self, example, tokenizer):
        """ CoCoClassifierInputExamples -> CoCoClassifierInputFeature """
        sys = tokenizer.tokenize(example.system_utter)
        user = tokenizer.tokenize(example.user_utter)
        sys_token = [tokenizer.cls_token] + sys + [tokenizer.sep_token]
        user_token = user + [tokenizer.sep_token]

        sys_id = tokenizer.convert_tokens_to_ids(sys_token)
        user_id = tokenizer.convert_tokens_to_ids(user_token)
        input_id = sys_id + user_id

        segment_id = [0] * len(sys_id) + [1] * len(user_id)
        input_mask = [1] * len(input_id)

        slots = [0] * len(self.slot2idx)
        for state in example.turn_state:
            slot = '-'.join(state.split('-')[:-1])
            slots[self.slot2idx[slot]] = 1

        return CoCoClassifierInputFeature(input_id=input_id, input_mask=input_mask, segment_id=segment_id,
                                          target_id=slots)


    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays


    def collate_fn(self, batch):
        input_ids = torch.tensor(self.pad_ids([b.input_id for b in batch], self.tokenizer.pad_token_id), dtype=torch.long,
                                 device=self.device)
        input_masks = torch.tensor(input_ids.ne(self.tokenizer.pad_token_id), dtype=torch.long, device=self.device)
        segment_ids = torch.tensor(self.pad_ids([b.segment_id for b in batch], self.tokenizer.pad_token_id), dtype=torch.long,
                                   device=self.device)
        target_ids = torch.tensor([b.target_id for b in batch], dtype=torch.float32, device=self.device)
        return input_ids, input_masks, segment_ids, target_ids


class CoCogenPreprocessor:

    def __init__(
            self,
            tokenizer,
    ):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def convert_example_to_feature(self, example, tokenizer):
        """ CoCoGenInputExamples -> CoCoGenInputFeature """
        sys = tokenizer.tokenize(example.system_utter)
        turn_state = ', '.join([s.replace('-', ' ') for s in example.turn_state])
        state = tokenizer.tokenize(turn_state)
        user = [tokenizer.bos_token] + tokenizer.tokenize(example.user_utter) + [tokenizer.eos_token]

        input_tokens = [tokenizer.bos_token] + sys + [tokenizer.eos_token] + state + [tokenizer.eos_token]
        input_id = tokenizer.convert_tokens_to_ids(input_tokens)
        target_id = tokenizer.convert_tokens_to_ids(user)

        return CoCoGenInputFeature(input_id=input_id, target_id=target_id)

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    def collate_fn(self, batch, tokenizer):
        input_ids = torch.LongTensor(tokenizer.pad_ids([b.input_id for b in batch], tokenizer.pad_token_id))
        target_ids = torch.LongTensor(tokenizer.pad_ids([b.target_id for b in batch], -100))
        input_masks = input_ids.ne(tokenizer.pad_token_id).float()
        return input_ids, target_ids, input_masks