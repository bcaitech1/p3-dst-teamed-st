

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







