# 다중 도메인 대화 상태 추적
미리 정의된 시나리오의 대화에서 (System발화, User발화)를 하나의 턴으로 둘 때, 턴마다 순차적으로 유저 발화의 Dialogue state(대화 상태)를 추적하는 Task

## Dataset
- `train_dials.json` : 7000개의 대화 (label 포함)

- `public/eval_dials.json` : 1000개의 대화 (label 미포함 / public test set)

- `private/eval_dials.json` : 1000개의 대화 (label 미포함 / private test set)

- `ontology.json` : Ontology-based DST model을 위한 pre-defined ontology입니다.

- `data_utils.py` : 각 대화를 공통적으로 전처리하기 위한 코드입니다.

## File Explorer

## Team Score

[comment]: <> "아래 이미지는 주석"
[comment]: <> "![image]&#40;https://user-images.githubusercontent.com/38639633/119125512-d0f6c680-ba6c-11eb-952e-fdc6de36fef9.png&#41;"
![image](https://user-images.githubusercontent.com/48181287/119263872-c9c1eb00-bc1b-11eb-916c-f6e171f1ba79.png)



## Baseline Code



refactoring 후 작성



## Usage

### Installation

```
pip install -r requirements.txt
```



### Train

```
SM_CHANNEL_TRAIN=data/train_dataset SM_MODEL_DIR=[model saving dir] python train.py
```

학습된 모델은 epoch 별로 `SM_MODEL_DIR/model-{epoch}.bin`으로 저장됩니다. 
추론에 필요한 부가 정보인 configuration들도 같은 경로에 저장됩니다. 
Best Checkpoint Path가 학습 마지막에 표기됩니다.



### Inference

```
SM_CHANNEL_EVAL=data/eval_dataset/public SM_CHANNEL_MODEL=[Model Checkpoint Path] SM_OUTPUT_DATA_DIR=[Output path] python inference.py
```





## Contributors

[윤도연(ydy8989)](https://github.com/ydy8989) | [전재열(Jayten)](https://github.com/Jayten) | [설재환(anawkward)](https://github.com/anawkward) | [민재원(ekzm8523)](https://github.com/ekzm8523) | [김봉진(BongjinKim)](https://github.com/BongjinKim) | [오세민(osmosm7)](https://github.com/osmosm7)







