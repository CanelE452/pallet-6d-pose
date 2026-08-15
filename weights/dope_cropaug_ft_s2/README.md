---
license: mit
tags:
  - pose-estimation
  - 6dof
  - dope
  - pallet
  - keypoint-detection
---

# pallet-dope-cropaug-truncation (DOPE, crop-aug, truncation-best)

팔레트 6D 포즈 추정용 DOPE 모델. **crop + padding 증강**으로 fine-tune 하여,
실제 카메라가 좌우로 패닝하며 팔레트가 화면 밖으로 잘리는(truncation) 상황에서
강건하게 동작하는 것이 특징.

- **best checkpoint**: `final_net_epoch_0180.pth` (누적 epoch 180)
- **convention**: camera-facing 0123 (0~3 앞면, {0,1,4,5}=위 / {2,3,6,7}=아래, 8=centroid)
- **backbone**: VGG-19, 9 belief maps + 16 affinity fields
- **input**: 448×448, sigma=4.0

## 왜 이 모델인가 (truncation best)

baseline DOPE 대비 real truncation 환경에서:

```
metric              baseline    crop-aug (this)
─────────────────────────────────────────────
detection rate      13%         94%
PnP success         23%         99%
```

YOLO-pose 의 crop+padding 증강 방식을 DOPE 로 이식하여 얻은 결과.

## 학습 설정

```
net_path  : dope_cropaug_ft_s1/final_net_epoch_0150.pth (이어서 fine-tune)
epochs    : 180 (누적 목표, base 150 + 추가 30)
lr        : 5e-5
batchsize : 4
imagesize : 448
sigma     : 4.0
seed      : 3709
data      : capturepallet{02~09,cad} + capturenight{04~09}
            + forklift_raw + truncation_crops_dope/ft_real (manual GT)
```

전체 학습 인자는 `header.txt` 참조.

## 평가 (synthetic val, 200 frames)

`eval_summary.json`:

```
PCK@3px   : 0.224     PnP success : 0.555
PCK@5px   : 0.277     reproj median: 164.6 px*
PCK@10px  : 0.416     volume ratio median: 0.836
```

\* reproj 수치는 evaluate_on_val 의 convention 불일치 버그로 과대평가됨
(실제 reproj 는 한 자리수 px). 검출률·PnP success 는 신뢰 가능.
truncation 강건성은 real 환경(위 표)에서 확인.

## 사용

```python
import torch
# DOPE DreamNetwork 정의는 Deep_Object_Pose/common/models.py 참조
state = torch.load("final_net_epoch_0180.pth", map_location="cpu")
model.load_state_dict(state)
```

## 주의

- 이 모델은 **논문용 truncation 강건성** 트랙 산출물. challenge(과제용) 트랙의
  `pallet-dope-challenge0123-ft-manual` / `pallet-dope-challengenight` 와는 별개.
- YOLO backend 와 혼용 시 R_fix=diag(-1,1,-1) 보정 필요(+Z 부호 반대). DOPE 단독은 불필요.
