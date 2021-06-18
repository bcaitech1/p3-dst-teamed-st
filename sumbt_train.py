import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from data_utils import (WOSDataset, get_examples_from_dialogues, load_dataset,
						set_seed, tokenize_ontology)
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import sumbt_inference, increment_path
from model.sumbt import SUMBT
from preprocessor import SUMBTPreprocessor

from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--save_dir", type=str, default=None)

	parser.add_argument("--train_batch_size", type=int, default=8)
	parser.add_argument("--eval_batch_size", type=int, default=8)
	parser.add_argument("--learning_rate", type=float, default=5e-5)
	parser.add_argument("--adam_epsilon", type=float, default=1e-8)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--warmup_ratio", type=int, default=0.1)
	parser.add_argument("--random_seed", type=int, default=42)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--distance_metric", type=str, default="euclidean")
	
	parser.add_argument("--model_name_or_path", type=str, default="dsksd/bert-ko-small-minimal")
	
	# Model Specific Argument
	parser.add_argument("--hidden_dim", type=int, help="GRU의 hidden size", default=300)
	parser.add_argument("--vocab_size", type=int, default=None)
	parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
	parser.add_argument("--proj_dim", type=int, default=None)
	parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
	parser.add_argument("--fix_utterance_encoder", type=bool, default=False)
	parser.add_argument("--attn_head", type=int, default=4)
	parser.add_argument("--max_label_length", type=int, default=12)
	parser.add_argument("--max_seq_length", type=int, default=64)
	parser.add_argument("--zero_init_rnn", type=bool, default=False)
	parser.add_argument("--num_rnn_layers", type=int, default=1)
	parser.add_argument("--use_amp", type=bool, default=True)
	args = parser.parse_args()
	print(args)

	save = False
	if args.save_dir:
		save = True
		save_dir = increment_path(args.save_dir)

	set_seed(args.random_seed)

	# Data Loading & processor
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))  # 45개의 slot
	ontology = json.load(open(f"{args.data_dir}/ontology.json"))
    train_data_file = f"{args.data_dir}/wos-v1_train.json"
	train_data, dev_data, dev_labels = load_dataset(train_data_file)  # item별로 분류 6301개 , 699개

	train_examples = get_examples_from_dialogues(
		train_data, user_first=True, dialogue_level=True
	)
	dev_examples = get_examples_from_dialogues(
		dev_data, user_first=True, dialogue_level=True
	)

	# Define Preprocessor
	tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
	max_turn = max([len(e['dialogue']) for e in train_data])
	processor = SUMBTPreprocessor(
		slot_meta,
		tokenizer,
		ontology=ontology,
		max_seq_length=64,
		max_turn_length=max_turn,
	)

	# Extracting Featrues
	train_features = processor.convert_examples_to_features(train_examples)
	dev_features = processor.convert_examples_to_features(dev_examples)

	# Ontology pre encoding
	slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, 12)
	num_labels = [len(s) for s in slot_values_ids]  # 각 Slot 별 후보 Values의 갯수

	# Model 선언
	n_gpu = 1 if torch.cuda.device_count() < 2 else torch.cuda.device_count()
	print(n_gpu)


	model = SUMBT(args, num_labels, device)
	model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)
	model.to(device)
	print("Model is initialized")

	train_data = WOSDataset(train_features)
	train_sampler = RandomSampler(train_data)
	train_loader = DataLoader(
		train_data,
		batch_size=args.train_batch_size,
		sampler=train_sampler,
		collate_fn=processor.collate_fn,  # feature를 tensor로 변경
	)
	print("# train:", len(train_data))

	dev_data = WOSDataset(dev_features)
	dev_sampler = SequentialSampler(dev_data)
	dev_loader = DataLoader(
		dev_data,
		batch_size=args.eval_batch_size,
		sampler=dev_sampler,
		collate_fn=processor.collate_fn,
	)
	print("# dev:", len(dev_data))

	# Optimizer 및 Scheduler 선언
	n_epochs = args.epochs
	t_total = len(train_loader) * n_epochs
	no_decay = ["bias", "LayerNorm.weight"]

	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		}
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=int(t_total * args.warmup_ratio), num_training_steps=t_total
	)

	if save:
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		json.dump(
			vars(args),
			open(f"{save_dir}/exp_config.json", "w"),
			indent=2,
			ensure_ascii=False,
		)

	scaler = GradScaler(enabled=args.use_amp)

	idx = 0
	best_score, best_checkpoint = 0, 0
	for epoch in range(n_epochs):
		start = time.time()
		batch_loss = []
		model.train()
		for step, batch in enumerate(train_loader):
			input_ids, segment_ids, input_masks, target_ids, num_turns, guids = [
				b.to(device) if not isinstance(b, list) else b for b in batch
			]
			# Forward
			if args.use_amp:
				with autocast(enabled=args.use_amp):
					if n_gpu == 1:
						loss, loss_slot, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids,
																  n_gpu)
					else:
						loss, _, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
			else:
				if n_gpu == 1:
					loss, loss_slot, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
				else:
					loss, _, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
			batch_loss.append(loss.item())

			# Backward
			if not args.use_amp:
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
			else:
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)

				scale = scaler.get_scale()
				scaler.update()
				step_scheduler = scaler.get_scale() == scale

				if step_scheduler:
					scheduler.step()

			if step % 100 == 0:
				print(
					f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} time: {time.time() - start}"
				)

		predictions = sumbt_inference(model, dev_loader, processor, device)
		eval_result = _evaluation(predictions, dev_labels, slot_meta)
		for k, v in eval_result.items():
			print(f"{k}: {v}")
		print(f" 걸린 시간 : {time.time() - start}")

		if best_score < eval_result['joint_goal_accuracy']:
			print("Update Best checkpoint!")
			best_score = eval_result['joint_goal_accuracy']
			best_checkpoint = epoch
			if save:
				idx = (idx + 1) % 3
				torch.save(model.state_dict(), f"{save_dir}/best_model{idx}.bin")
				save_info = {"model_name": f"best_model{idx}.bin", "epoch": epoch, "JGA": best_score}
				json.dump(save_info, open(f"{save_dir}/best_model{idx}.json", "w"), indent=2, ensure_ascii=False)
	if save:
		torch.save(model.state_dict(), f"{save_dir}/last_model.bin")
		print(f"Best checkpoint: {save_dir}/model-{best_checkpoint}.bin")

