import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from evaluation import evaluate


@dataclass
class CoCoGenInputExample:
    guid: str
    system_utter: str
    turn_state: List[str]
    user_utter: str


@dataclass
class CoCoGenInputFeature:
    input_id: List[int]
    target_id: List[int]


@dataclass
class CoCoClassifierInputExample:
    guid: str
    system_utter: str
    user_utter: str
    turn_state: List[str]


@dataclass
class CoCoClassifierInputFeature:
    input_id: List[int]
    input_mask: List[int]
    segment_id: List[int]
    target_id: List[int]


class CoCoClassifierDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


class CoCoGenDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


class DSTPreprocessor:
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [
            array + [pad_idx] * (max_length - min(len(array), 512)) for array in arrays
        ]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def _convert_example_to_feature(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def recover_state(self):
        raise NotImplementedError


def split_slot(dom_slot_value, get_domain_slot=False):
    try:
        dom, slot, value = dom_slot_value.split("-")  # 온전한 데이터
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:  # domain만 있을경우
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]  # domain, slot만 있을경우
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def coco_generator(example, slot_value_dict, slot_comb_dict={}, verbose=False):
    if not example.turn_state:
        return example

    coco = deepcopy(example)
    num_state = len(coco.turn_state)
    is_drop = False

    # drop: dialogue state 중 하나를 제거합니다. (e.g., [식당-종류-양식당, 식당-예약 시간-18:00] -> [식당-종류-양식당])
    if num_state > 1:
        drop_idx = random.choice(range(num_state))
        coco.turn_state.pop(drop_idx)
        num_state -= 1
        is_drop = True

    # change: dialogue state의 value 중 하나를 다른 value로 대체합니다. (e.g., [식당-종류-양식당] -> [식당-종류-중식당])
    change_idx = random.choice(range(num_state))
    origin_slot_value = coco.turn_state[change_idx]
    st, sv = split_slot(origin_slot_value, True)
    candidates = slot_value_dict.get(st, [sv])
    new_sv = random.choice(candidates[2:])
    new_slot_value = f"{st}-{new_sv}"
    coco.turn_state[change_idx] = new_slot_value

    # add: slot_comb_dict에서 하나의 slot-value를 생성합니다. (e.g., [식당-종류-중식당] -> [식당-종류-중식당, 식당-예약 인원-2])
    combinations = slot_comb_dict.get(st)
    slots = [c[0] for c in combinations]
    counts = [c[1] for c in combinations]
    weights = [sum(counts) / c for c in counts]
    if not combinations:
        return coco

    co_st = random.choice(slots)
    # co_st = random.choices(slots, weights=weights)[0]
    candidates = slot_value_dict.get(co_st, [])
    co_sv = random.choice(candidates[2:])
    new_slot_value = f"{co_st}-{co_sv}"
    coco.turn_state.append(new_slot_value)

    if verbose:
        print("Before:", example.turn_state)
        print("After:", coco.turn_state)

    return coco


def convert_example_to_feature(example, tokenizer):
    """CoCoGenInputExamples -> CoCoGenInputFeature"""
    sys = tokenizer.tokenize(example.system_utter)
    turn_state = ", ".join([s.replace("-", " ") for s in example.turn_state])
    state = tokenizer.tokenize(turn_state)
    user = (
        [tokenizer.bos_token]
        + tokenizer.tokenize(example.user_utter)
        + [tokenizer.eos_token]
    )

    input_tokens = (
        [tokenizer.bos_token]
        + sys
        + [tokenizer.eos_token]
        + state
        + [tokenizer.eos_token]
    )
    input_id = tokenizer.convert_tokens_to_ids(input_tokens)
    target_id = tokenizer.convert_tokens_to_ids(user)

    return CoCoGenInputFeature(input_id=input_id, target_id=target_id)


def get_coco_examples_from_dialogue(dialogue):
    """Dialogue 데이터셋 파일 -> CoCoGenInputExamples"""
    guid = dialogue["dialogue_idx"]
    examples = []
    d_idx = 0
    previous_state = []
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")

        turn_state = sorted(list(set(state) - set(previous_state)))
        examples.append(
            CoCoGenInputExample(
                guid=f"{guid}-{d_idx}",
                system_utter=sys_utter,
                turn_state=turn_state,
                user_utter=user_utter,
            )
        )

        d_idx += 1
        previous_state = state

    return examples


def get_coco_examples_from_dialogues(dialogue, dialogue_level=False):
    examples = []
    for d in tqdm(dialogue):
        example = get_coco_examples_from_dialogue(d)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples
