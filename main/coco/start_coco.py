import json
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import BartForConditionalGeneration, BertConfig, BertTokenizer
from transformers import PreTrainedTokenizerFast
from data_utils import *
from preprocessor import CoCoPreprocessor
import pickle
from evaluation import evaluate
from model import BertForMultiLabelSequenceClassification
from collections import defaultdict


def get_augmented_uttrs(
    model, tokenizer, new_turn, device
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


def classifier_filtering(model, uttrs, example, device, processor):
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
        features.append(processor.cls_convert_example_to_feature(example))
    data = CoCoClassifierDataset(features)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data, sampler=sampler, batch_size=32, collate_fn=processor.cls_collate_fn
    )
    flags = evaluate(model, device, dataloader, is_query=True)
    for idx, flag in enumerate(flags):
        if flag:
            filtered.append(uttrs[idx])
    return filtered


def match_filtering(new_turn, ori_turn, sentences): # ori -> old로 네이밍 변경이 나을듯?
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
    dialogue,
    slot_value_dict,
    device,
    slot_comb_dict={},
):
    new_turns = []
    for turn in dialogue:

        new_turn = coco_generator(turn, slot_value_dict, slot_comb_dict)

        uttrs = get_augmented_uttrs(
            generator, processor.gen_tokenizer, new_turn, device
        )

        filter_uttrs = classifier_filtering(
            classifier_filter, uttrs, new_turn, device, processor
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model_path = "/opt/ml/model/gen_model.bin"
    gen_model_name = "hyunwoongko/kobart"
    gen_model = BartForConditionalGeneration.from_pretrained(gen_model_name)
    gen_tokenizer = PreTrainedTokenizerFast.from_pretrained(gen_model_name)
    ckpt = torch.load("/opt/ml/model/gen_model.bin")
    gen_model.load_state_dict(ckpt)
    gen_model.to(device)

    cls_model_path = "/opt/ml/model/cls_model.bin"
    cls_model_name = "dsksd/bert-ko-small-minimal"
    data = json.load(open('/opt/ml/input/data/train_dataset/train_dials.json'))

    slot_meta = json.load(open('/opt/ml/input/data/train_dataset/slot_meta.json'))
    cls_tokenizer = BertTokenizer.from_pretrained(cls_model_name)
    bert_config = BertConfig.from_pretrained(cls_model_name, num_labels=len(slot_meta))
    bert_config.model_name_or_path = cls_model_name
    bert_config.num_labels = len(slot_meta)
    cls_model = BertForMultiLabelSequenceClassification.from_pretrained(cls_model_name, config=bert_config)
    ckpt = torch.load("/opt/ml/model/cls_model.bin")
    cls_model.load_state_dict(ckpt)
    cls_model.to(device)

    coco_examples = get_coco_examples_from_dialogues(data, dialogue_level=True)
    coco_examples = coco_examples[:1500]
    processor = CoCoPreprocessor(slot_meta, gen_tokenizer, cls_tokenizer, bert_config)

    slot_value_dict = json.load(open('/opt/ml/input/data/train_dataset/new_ontology.json'))
    with open("../somdst/somdst_data/slot_comb_dict.pkl", "rb") as f:
        slot_comb_dict = pickle.load(f)

    augmented = []
    for dialogue in tqdm(coco_examples):
        new_dialogue = get_augmented_dialogue(
            gen_model,
            cls_model,
            processor,
            dialogue,
            slot_value_dict,
            device,
            slot_comb_dict,
        )
        augmented.append(new_dialogue)

    with open("new_train.json", "w") as f:
        json.dump(augmented, f)