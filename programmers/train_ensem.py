# 재원이가 고침 ㅎㅎㅎ
import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from inference import direct_inference
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from dataset import *
from adamp import AdamP
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, shuffle=False, n=16):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def custom_loss(output, target, criterion): # mask, age, gender 클래스 별 분류에 필요한 loss function
    mask_loss = criterion(output[0], target[0])
    gender_loss = criterion(output[1], target[1])
    age_loss = criterion(output[2], target[2])
    return (0.2 * mask_loss) + (0.2 * gender_loss) + (0.6 * age_loss),\
            mask_loss.item(), gender_loss.item(), age_loss.item()

def train(data_dir, model_dir, args): # /opt/ml/input/data/train/images/ , ./model/exp , args
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- test data config
    submission = pd.read_csv(os.path.join(args.test_dir, 'info.csv'))
    test_image_dir = os.path.join(args.test_dir, 'images')

    image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((224, 224), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    test_dataset = TestDataset(image_paths, transform)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False
    )

    # -- data_loader
    # train_set, val_set = dataset.split_dataset()
    counter = 0
    patience = 10
    oof_pred = None
    for fold, (train_set, val_set) in enumerate(skf.split(dataset, dataset.multi_class_labels)):
        is_multiclass = (fold >= 2)

        train_set = torch.utils.data.Subset(dataset,
                                            indices=train_set)
        val_set = torch.utils.data.Subset(dataset,
                                          indices=val_set)

        train_set.is_multiclass = is_multiclass
        val_set.is_multiclass = is_multiclass
        if fold == 0:
            print(
                f"fold : {fold} \t model : efficientnet_b3 \t criterion : focal"
            )

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_set,
                batch_size=args.valid_batch_size,
                num_workers=1,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )

            # -- model
            model_module = getattr(import_module("model"), "efficientnet_b3")  # default: BaseModel
            model = model_module(
                num_classes=num_classes
            ).to(device)

            model = torch.nn.DataParallel(model)

            # -- loss & metric
            criterion = create_criterion("focal")  # default: cross_entropy

            optimizer = AdamP(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5,
                weight_decay=1e-2)

        elif fold == 1:
            print(
                f"fold : {fold} \t model : resnet50 \t criterion : cross_entropy"
            )
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_set,
                batch_size=args.valid_batch_size,
                num_workers=1,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )

            # -- model
            model_module = getattr(import_module("model"), "resnet50")  # default: BaseModel
            model = model_module(
                num_classes=num_classes
            ).to(device)

            model = torch.nn.DataParallel(model)

            # -- loss & metric
            criterion = create_criterion("cross_entropy")  # default: cross_entropy

            optimizer = AdamP(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5,
                weight_decay=1e-2)

        elif fold == 2:
            print(
                f"fold : {fold} \t model : custom_resnet50 \t criterion : cross_entropy"
            )
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_set,
                batch_size=args.valid_batch_size,
                num_workers=1,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )

            # -- model
            model_module = getattr(import_module("model"), "custom_resnet50")  # default: BaseModel
            model = model_module(
                num_classes=num_classes
            ).to(device)

            model = torch.nn.DataParallel(model)

            # -- loss & metric
            criterion = create_criterion("cross_entropy")  # default: cross_entropy
            optimizer = AdamP(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=1e-2)

        elif fold == 3:
            print(
                f"fold : {fold} \t model : custom_efficientnet_b3 \t criterion : focal"
            )
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_set,
                batch_size=args.valid_batch_size,
                num_workers=1,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )

            # -- model
            model_module = getattr(import_module("model"), "custom_efficientnet_b3")  # default: BaseModel
            model = model_module(
                num_classes=num_classes
            ).to(device)

            model = torch.nn.DataParallel(model)

            criterion = create_criterion("focal")  # default: cross_entropy
            optimizer = AdamP(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=1e-2)
            # -- loss & metric
        elif fold == 4:
            print(
                f"fold : {fold} \t model : custom_resnet50 \t criterion : focal"
            )
            # transform_module = getattr(import_module("dataset"), "CustomAugmentation")  # default: BaseAugmentation
            # transform = transform_module(
            #     resize=args.resize,
            #     mean=dataset.mean,
            #     std=dataset.std,
            # )
            # train_set.set_transform(transform)

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_set,
                batch_size=args.valid_batch_size,
                num_workers=1,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )

            # -- model
            model_module = getattr(import_module("model"), "custom_resnet50")  # default: BaseModel
            model = model_module(
                num_classes=num_classes
            ).to(device)

            model = torch.nn.DataParallel(model)

            # -- loss & metric
            criterion = create_criterion("focal")  # default: cross_entropy

            optimizer = AdamP(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5,
                weight_decay=1e-2)



        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config{}.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        if is_multiclass:
            for epoch in range(args.epochs):
                # train loop
                model.train()
                mask_loss_value = 0
                gender_loss_value = 0
                age_loss_value = 0
                mask_matches = 0
                gender_matches = 0
                age_matches = 0
                for idx, train_batch in enumerate(train_loader):

                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    mask_labels = labels[0].to(device)
                    gender_labels = labels[1].to(device)
                    age_labels = labels[2].to(device)


                    optimizer.zero_grad()

                    outs = model(inputs)
                    mask_preds = torch.argmax(outs[0], dim=-1)
                    gender_preds = torch.argmax(outs[1], dim=-1)
                    age_preds = torch.argmax(outs[2], dim=-1)

                    loss, mask_loss, gender_loss, age_loss = custom_loss(outs, (mask_labels, gender_labels, age_labels), criterion)
                    loss.backward()
                    optimizer.step()

                    mask_loss_value += mask_loss
                    gender_loss_value += gender_loss
                    age_loss_value += age_loss

                    mask_matches += (mask_preds == mask_labels).sum().item()
                    gender_matches += (gender_preds == gender_labels).sum().item()
                    age_matches += (age_preds == age_labels).sum().item()

                    if (idx + 1) % args.log_interval == 0:
                        mask_train_loss = mask_loss_value / args.log_interval
                        gender_train_loss = gender_loss_value / args.log_interval
                        age_train_loss = age_loss_value / args.log_interval

                        mask_train_acc = mask_matches / args.batch_size / args.log_interval
                        gender_train_acc = gender_matches / args.batch_size / args.log_interval
                        age_train_acc = age_matches / args.batch_size / args.log_interval
                        current_lr = get_lr(optimizer)
                        print(
                            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"mask_training loss {mask_train_loss:4.4} || mask_training accuracy {mask_train_acc:4.2%} || lr {current_lr}\n"
                            f"gender_training loss {gender_train_loss:4.4} || gender_training accuracy {gender_train_acc:4.2%} || lr {current_lr}\n"
                            f"age_training loss {age_train_loss:4.4} || age_training accuracy {age_train_acc:4.2%} || lr {current_lr}\n"

                        )
                        mask_loss_value = gender_loss_value = age_loss_value = 0
                        mask_matches = gender_matches = age_matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    mask_val_loss_items = []
                    gender_val_loss_items = []
                    age_val_loss_items = []

                    mask_val_acc_items = []
                    gender_val_acc_items = []
                    age_val_acc_items = []
                    multi_class_acc_items = []
                    f1_score_items = []

                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        # labels = labels.to(device)
                        mask_labels = labels[0].to(device)
                        gender_labels = labels[1].to(device)
                        age_labels = labels[2].to(device)
                        multi_class_labels = 6*mask_labels + 3*gender_labels + age_labels

                        outs = model(inputs)
                        mask_preds = torch.argmax(outs[0], dim=-1)
                        gender_preds = torch.argmax(outs[1], dim=-1)
                        age_preds = torch.argmax(outs[2], dim=-1)
                        preds = 6*mask_preds + 3*gender_preds + age_preds

                        (loss, mask_loss, gender_loss, age_loss) = custom_loss(outs, (mask_labels, gender_labels, age_labels), criterion)
                        mask_acc_item = (mask_labels == mask_preds).sum().item()
                        gender_acc_item = (gender_labels == gender_preds).sum().item()
                        age_acc_item = (age_labels == age_preds).sum().item()
                        multi_class_acc_item = (multi_class_labels == preds).sum().item()
                        preds = preds.cpu().numpy()
                        multi_class_labels = multi_class_labels.cpu().numpy()
                        f1_score_item = f1_score(multi_class_labels, preds, average='macro')

                        mask_val_loss_items.append(mask_loss)
                        gender_val_loss_items.append(gender_loss)
                        age_val_loss_items.append(age_loss)


                        mask_val_acc_items.append(mask_acc_item)
                        gender_val_acc_items.append(gender_acc_item)
                        age_val_acc_items.append(age_acc_item)
                        multi_class_acc_items.append(multi_class_acc_item)
                        f1_score_items.append(f1_score_item)

                    mask_val_loss = np.sum(mask_val_loss_items)
                    gender_val_loss = np.sum(gender_val_loss_items)
                    age_val_loss = np.sum(age_val_loss_items)
                    avg_val_loss = (mask_val_loss + gender_val_loss + age_val_loss)/3

                    best_val_loss = min(best_val_loss, avg_val_loss)

                    mask_val_acc = np.sum(mask_val_acc_items) / len(val_set)
                    gender_val_acc = np.sum(gender_val_acc_items) / len(val_set)
                    age_val_acc = np.sum(age_val_acc_items) / len(val_set)
                    multi_class_val_acc = np.sum(multi_class_acc_items) / len(val_set)
                    f1_score_ = np.sum(f1_score_items) / len(val_set)

                    avg_val_acc = (mask_val_acc + gender_val_acc + age_val_acc) / 3

                    if avg_val_loss > best_val_acc:
                        print(f"New best model for val accuracy : {avg_val_acc:4.2%}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best{fold}.pth")
                        best_val_acc = avg_val_acc
                        counter = 0
                    else:
                        counter += 1
                    torch.save(model.module.state_dict(), f"{save_dir}/last{fold}.pth")
                    print(
                        f"[Val] acc : {avg_val_acc:4.2%}, loss: {avg_val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                        f"real acc : {multi_class_val_acc:4.2%}, f1 score : {f1_score_}"
                    )
                    if counter > patience:
                        print("Early Stopping...")
                        break

                print()
        else:
            for epoch in range(args.epochs):
                # train loop
                model.train()
                loss_value = 0
                matches = 0
                for idx, train_batch in enumerate(train_loader):
                    inputs, (mask_labels, gender_labels, age_labels) = train_batch
                    labels = (mask_labels * 6 + gender_labels * 3 + age_labels).to(device)
                    inputs = inputs.to(device)

                    optimizer.zero_grad()

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    optimizer.step()

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        train_acc = matches / args.batch_size / args.log_interval
                        current_lr = get_lr(optimizer)
                        print(
                            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                        )

                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []
                    figure = None
                    for val_batch in val_loader:
                        inputs, (mask_labels, gender_labels, age_labels) = val_batch
                        inputs = inputs.to(device)
                        labels = (mask_labels * 6 + gender_labels * 3 + age_labels).to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_set)
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best{fold}.pth")
                        best_val_acc = val_acc
                    torch.save(model.module.state_dict(), f"{save_dir}/last{fold}.pth")
                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                    )
                    print()

    print('test inference is done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=12, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamP', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    # parser.add_argument('--pretrained', type=bool, default=False, help='pretrained default is False')
    parser.add_argument('--direct_inference', type=bool, default=True, help='direct inference (default: True)')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/eval'))
    args = parser.parse_args()
    print(args)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    data_dir = args.data_dir
    model_dir = args.model_dir
    test_dir = args.test_dir
    train(data_dir, model_dir, args)


