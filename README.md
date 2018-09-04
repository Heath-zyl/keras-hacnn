# keras-ha-cnn

## Introduction
Paper - https://arxiv.org/pdf/1802.08122.pdf

2가지 방법에 대하여 구현 및 테스트
1. categorize를 이용한 feature extraction
2. Siamese network를 이용한 2가지 이미지 비교

### categorize
* input : image (164,60,3)
* output
    * training : softmax (category id)
    * evaluate : feature (global feature + local feature or global feature)

### Siamese network
* input : image A,B (164,60,3)
* output : 0 ~ 1

## Training
### categorize
```
$ python train.py \
    --dataset videotag \
    --dataset-path /Users/luke/Documents/ml_datasets/person_re_id/videotag_scene/dataset_7 \
    --height 160 \
    --width 64 \
    --batch-size 32 \
    --snapshot /Users/luke/Documents/ml_models/person-reid/ha-cnn-keras/global_local/snapshot/videotag_07.h5 \
    --log-dir /Users/luke/Documents/ml_models/person-reid/ha-cnn-keras/global_local/log \
    --epochs 1 \
    --learn-region 1
```

### Siamese network
```
python train_similarity.py \
    --dataset videotag \
    --dataset-path /Users/luke/Documents/ml_datasets/person_re_id/videotag_scene/dataset_7 \
    --height 160 \
    --width 64 \
    --batch-size 32 \
    --epochs 300 \
    --learn-region 1
```

## Testing
### categorize
...
### Siamese network
...

## Notebook

## 이슈
* keras에서 복잡한 Lambda 함수가 네트워크가 포함되어 있을 시, 모델이 저장이 안되는 버그가 있어 weight만 저장함.
    * https://github.com/keras-team/keras/issues/8343#issuecomment-392445490