import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import numpy as np

def multi_to_ont_convert(mask_pred, gender_pred, age_pred, num_classes):
    batch_size = mask_pred.shape[0]
    pred = torch.zeros(batch_size,num_classes)
    for i in range(batch_size):
        for mask_idx in range(3):
            for gender_idx in range(2):
                for age_idx in range(3):
                    j = 6*mask_idx + 3*gender_idx + age_idx
                    pred[i][j] = mask_pred[i][mask_idx]*gender_pred[i][gender_idx]*age_pred[i][age_idx]
    return pred



def load_model(saved_model, num_classes, device, fold, model_name):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f'best{fold}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
def kfold_load_model(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    print(loader)
    oof_pred = None
    for fold in range(5):
        if fold == 0:
            model_name = "efficientnet_b3"
        elif fold == 1:
            model_name = "resnet50"
        elif fold == 2:
            model_name = "custom_resnet50"
        elif fold == 3:
            model_name = "custom_efficientnet_b3"
        elif fold == 4:
            model_name = "custom_resnet50"
        model = load_model(model_dir, num_classes, device, fold, model_name).to(device)
        model.eval()

        print("Calculating inference results..")
        preds = []
        if fold >= 2:
            with torch.no_grad():
                for idx, images in enumerate(loader):
                    images = images.to(device)
                    pred = model(images)
                    mask_pred = pred[0]
                    gender_pred = pred[1]
                    age_pred = pred[2]

                    pred = multi_to_ont_convert(mask_pred, gender_pred, age_pred, num_classes)

                    preds.extend(pred.cpu().numpy())
        else:
            with torch.no_grad():
                for idx, images in enumerate(loader):
                    images = images.to(device)
                    pred = model(images)
                    preds.extend(pred.cpu().numpy())

        fold_pred = np.array(preds)
        if oof_pred is None:
            oof_pred = fold_pred / 5
        else:
            oof_pred += fold_pred / 5


    print(info)
    info['ans'] = np.argmax(oof_pred, axis=1)
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    if args.custom_classifier: # mask, gender, age 별 분류
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                mask_pred = pred[0].argmax(dim=-1)
                gender_pred = pred[1].argmax(dim=-1)
                age_pred = pred[2].argmax(dim=-1)
                pred = MaskBaseDataset.encode_multi_class(mask_pred,gender_pred,age_pred)

                preds.extend(pred.cpu().numpy())
    else:
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)

                preds.extend(pred.cpu().numpy())

    print(info['ans'])
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')

def direct_inference(model, test_dir, args):
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((224, 224), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)

            mask_pred = pred[0].argmax(dim=-1)
            gender_pred = pred[1].argmax(dim=-1)
            age_pred = pred[2].argmax(dim=-1)
            label = MaskBaseDataset.encode_multi_class(mask_pred,gender_pred,age_pred)
            all_predictions.extend(label.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(args.test_dir, 'submission.csv'), index=False)
    print('test inference is done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--k_fold', type=bool, default=False, help='k-fold using for k_fold True (default: False)')
    parser.add_argument('--custom_classifier', type=bool, default=False, help='True : 18 classifier , False : mask, gender, age 별 분류 (default: False)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/real last'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))


    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir


    os.makedirs(output_dir, exist_ok=True)
    if args.k_fold:
        kfold_load_model(data_dir, model_dir, output_dir, args)
    else:
        inference(data_dir, model_dir, output_dir, args)
