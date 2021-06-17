import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt
import seaborn as sns


def get_ext(img_dir, img_id):
    """
    학습 데이터셋 이미지 폴더에는 여러 하위폴더로 구성되고, 이 하위폴더들에는 각 사람의 사진들이 들어가있다. 하위폴더에 속한 이미지의 확장자를 구하는 함수이다.

    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로
        img_id: 학습 데이터셋 하위폴더 이름

    Returns:
        ext: 이미지의 확장자
    """
    filename = os.listdir(os.path.join(img_dir, img_id))[0]
    ext = os.path.splitext(filename)[-1].lower()

    return ext


def get_img_stats(img_dir, img_ids):
    """
    데이터셋에 있는 이미지들의 크기와 RGB 평균 및 표준편차를 수집하는 함수입니다.

    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로
        img_id: 학습 데이터셋 하위폴더 이름

    Returns:
        img_info: 이미지들의 정보 (크기, 평균, 표준편차)
    """
    img_info = dict(heights=[], widths=[], means=[], stds=[])
    for img_id in tqdm(img_ids):
        for path in glob(os.path.join(img_dir, img_id, '*')):
            img = np.array(Image.open(path))
            h, w, _ = img.shape
            img_info['heights'].append(h)
            img_info['widths'].append(w)
            img_info['means'].append(img.mean(axis=(0, 1)))
            img_info['stds'].append(img.std(axis=(0, 1)))
    return img_info

def custom_imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, 1, 2, 0))
    plt.show()
