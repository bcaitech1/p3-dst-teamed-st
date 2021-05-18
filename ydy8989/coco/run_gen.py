from data_utils import (
    get_coco_examples_from_dialogues,
    coco_generator,
    convert_example_to_feature,
    CoCoClassifierInputExample,
    CoCoClassifierDataset,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from evaluation import evaluate
import json
import os
import pickle
import torch
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast, BertConfig, BertTokenizer
from model import BertForMultiLabelSequenceClassification
from preprocessor import CoCoClassifierPreprocessor


def get_augmented_uttrs(
    model, tokenizer, new_turn, slot_value_dict, slot_comb_dict, device
):
    x = convert_example_to_feature(new_turn, tokenizer)
    input_id = torch.LongTensor([x.input_id]).to(device)
    o = model.generate(
        input_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_length=30,
        early_stopping=True,
        num_beams=8,
        top_k=30,
        temperature=1.5,
        do_sample=True,
        num_return_sequences=8,
    )
    uttrs = tokenizer.batch_decode(o.tolist(), skip_special_tokens=True)
    return uttrs


def classifier_filtering(model, tokenizer, uttrs, example, device, processor):
    filtered = []
    examples = [
        CoCoClassifierInputExample(
            guid=example.guid,
            system_utter=example.system_utter,
            user_utter=uttr,
            turn_state=example.turn_state,
        )
        for uttr in uttrs
    ]
    features = []
    for example in examples:
        features.append(processor.convert_example_to_feature(example, tokenizer))
    data = CoCoClassifierDataset(features)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data, sampler=sampler, batch_size=32, collate_fn=processor.collate_fn
    )
    flags = evaluate(model, device, dataloader, is_query=True)
    for idx, flag in enumerate(flags):
        if flag:
            filtered.append(uttrs[idx])
    return filtered


def match_filtering(new_turn, ori_turn, sentences):
    ori_turn_label_set = ori_turn.turn_state
    new_turn_label_set = new_turn.turn_state

    if ori_turn_label_set == new_turn_label_set:
        return ""
    missing_values = []
    for state in ori_turn.turn_state:
        value = state.split("-")[-1]
        if (value in ori_turn.system_utter) and (value not in ori_turn.user_utter):
            missing_values.append(value)

    value_list = []
    best_sent = ""
    for state in new_turn.turn_state:
        domain, slot, value = state.split("-")
        if value in ["yes", "no"]:
            value = slot.split(" ")[0]
        if value not in missing_values:
            value_list.append(value)

    for sent in sentences:
        flag = True
        for value in value_list:
            if value not in sent:
                flag = False
                break

        if flag == True:
            best_sent = sent
            break

    return best_sent


def convert_examples_to_dialogue(examples):
    guid: str
    system_utter: str
    turn_state: List[str]
    user_utter: str

    dialogue = {}
    dialogue_idx = "augmented-" + "-".join(examples[0].guid.split("-")[:-1])
    dialogue = []

    states = defaultdict()
    for example in examples:
        for state in example.turn_state:
            domain, slot, value = state.split("-")
            slot = "-".join([domain, slot])
            states[slot] = value
        state_list = sorted([f"{slot}-{value}" for slot, value in states.items()])
        sys = {"role": "sys", "text": example.system_utter}
        user = {"role": "user", "text": example.user_utter, "state": state_list}
        if example.system_utter:
            dialogue.append(sys)
        dialogue.append(user)
    domains = sorted(set([slot.split("-")[0] for slot in states]))

    return {"dialogue_idx": dialogue_idx, "domains": domains, "dialogue": dialogue}


def get_augmented_dialogue(
    generator,
    classifier_filter,
    processor,
    gen_tokenizer,
    cls_tokenizer,
    dialogue,
    slot_value_dict,
    device,
    slot_comb_dict={},
):
    new_turns = []
    for turn in dialogue:

        new_turn = coco_generator(turn, slot_value_dict, slot_comb_dict)

        uttrs = get_augmented_uttrs(
            generator, gen_tokenizer, new_turn, slot_value_dict, slot_comb_dict, device
        )

        filter_uttrs = classifier_filtering(
            classifier_filter, cls_tokenizer, uttrs, new_turn, device, processor
        )
        best_uttr = match_filtering(new_turn, turn, filter_uttrs)
        if best_uttr:
            new_turn.user_utter = best_uttr
        else:
            new_turn = deepcopy(turn)
        new_turns.append(new_turn)

    new_dialogue = convert_examples_to_dialogue(new_turns)
    return new_dialogue


if __name__ == "__main__":
    device = torch.device('cuda')
    data_dir = "../../input/data/train_dataset"
    model_path = "hyunwoongko/kobart"
    generator_ckpt = "./checkpoint/gen_model.bin"
    classifier_ckpt = "./checkpoint/classifier_train.bin"

    data = json.load(open(os.path.join(data_dir, "train_dials.json"), "rt", encoding='UTF8'))
    coco_examples = get_coco_examples_from_dialogues(data, dialogue_level=True)

    generator = BartForConditionalGeneration.from_pretrained(model_path)
    gen_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    state_dict = torch.load(generator_ckpt)
    generator.load_state_dict(state_dict)
    generator.to(device)
    cls_tokenizer = BertTokenizer.from_pretrained("dsksd/bert-ko-small-minimal")

    # generator_keys = generator.state_dict().keys()
    # state_dict = {
    #     k: v for k, v in torch.load(generator_ckpt).items() if k in generator_keys
    # }

    slot_meta = json.load(open("../../input/data/train_dataset/slot_meta.json", "rt", encoding='UTF8'))

    bert_config = BertConfig.from_pretrained(
        "dsksd/bert-ko-small-minimal", num_labels=len(slot_meta)
    )
    bert_config.model_name_or_path = "dsksd/bert-ko-small-minimal"
    bert_config.num_labels = 45
    classifier_filter = BertForMultiLabelSequenceClassification.from_pretrained(
        "dsksd/bert-ko-small-minimal", config=bert_config
    )
    classifier_filter.load_state_dict(torch.load(classifier_ckpt))
    slot_value_dict = json.load(open("../../input/data/train_dataset/ontology.json", "rt", encoding='UTF8'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    classifier_filter.to(device)

    processor = CoCoClassifierPreprocessor(slot_meta, cls_tokenizer, bert_config)

    with open("slot_comb_dict.pkl", "rb", encoding='UTF8') as f:
        slot_comb_dict = pickle.load(f)
    augmented = []
    for dialogue in tqdm(coco_examples):
        new_dialogue = get_augmented_dialogue(
            generator,
            classifier_filter,
            processor,
            gen_tokenizer,
            cls_tokenizer,
            dialogue,
            slot_value_dict,
            device,
            slot_comb_dict,
        )
    with open("new_train.json", "w", encoding='UTF8') as f:
        json.dump(new_dialogue, f)
