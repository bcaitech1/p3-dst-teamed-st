import json
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from data_utils import *
from preprocessor import CoCoPreprocessor

def train_generation_model(model_path, data_path):

    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    data = json.load(open(data_path))

    processor = CoCoPreprocessor(tokenizer)

    examples = []
    for dialogue in tqdm(data):
        examples.extend(get_coco_examples_from_dialogue(dialogue))

    features = []
    for example in tqdm(examples):
        features.append(processor.gen_convert_example_to_feature(example, tokenizer))



    num_train_epochs = 10
    batch_size = 1
    lr = 5e-5
    warmup_ratio = 0.1
    weight_decay = 0.01

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = CoCoGenDataset(features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)

    t_total = len(train_loader) * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * warmup_ratio), num_training_steps=t_total
    )

    model.to(device)
    print("")

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, target_ids, input_masks = (b.to(device) for b in batch)
            decoder_input_ids = target_ids[:, :-1].contiguous()
            decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
            labels = target_ids[:, 1:].clone()
            outputs = model(input_ids,
                            attention_mask=input_masks,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels)
            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"[{epoch}/{num_train_epochs}][{step}/{len(train_loader)}] {loss}")

    torch.save(model.state_dict(), "/opt/ml/model/coco.bin")




if __name__ == "__main__":
    model_path = "hyunwoongko/kobart"
    data_path = '/opt/ml/input/data/train_dataset/train_dials.json'

    train_generation_model(model_path, data_path)

