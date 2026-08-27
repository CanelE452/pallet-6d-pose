# EVALUATOR AUDIT — 세 모델을 같은 자로 잴 수 있는가

## 공통 evaluator (Y0 가 쓴 것 그대로 재사용, 새 solver·threshold 도입 0)

```
cf_real_eval.py     pad=100 + BORDER_REFLECT_101, imgsz 640, conf=0.001(threshold-free)
                    top-1 by box conf, correct_box = IoU >= 0.5
                    9kp error = 예측 keypoint 와 GT 의 L2 (원본 이미지 px)
night_cand_one.py   any-cbox / top1-cbox / candidates per frame / wrong-candidate presence
                    / ranking margin
neg_eval_one.py     POS 128 + NEG 2,689 max_conf 덤프 -> AP·AUROC·FPR@TPR95·FP/image
```

## ★ 직접 비교 가능성 — postprocess 가 구조적으로 다르다

```
YOLOv8n-Pose   Detect  + NMS        predict 기본 iou=0.7, max_det=300
YOLO11n-Pose   Detect  + NMS        동일
YOLO26n-Pose   Pose26  end2end      one2one branch, NMS 없음
```

따라서 **candidate 수에 의존하는 지표는 같은 자가 아니다**:

- `candidates/frame` — v8/11 은 NMS 후 남은 박스 수, 26 은 one2one 출력 수. 정의가 다르다.
- `wrong-candidate presence` — 위 후보 집합 위에서 세므로 같은 영향.

이 둘은 표에 싣되 **모델 간 직접 비교 금지**로 표시한다.

**primary 지표는 정의가 동일하므로 비교 가능하다** — 셋 다 "top-1 by box conf" 하나만
쓰고 후보 집합 크기와 무관하다:

```
cbox                 top-1 박스의 IoU >= 0.5 여부          비교 가능
9kp median / p90     correct-box 프레임의 keypoint L2      비교 가능
NIGHT top1-cbox      top-1 이 correct 인 프레임 수 / 28    비교 가능
negative FP/image    conf >= 0.40 인 박스 수의 프레임 평균  ★아래 단서
```

`FP/image@0.40` 은 박스 **개수**를 세므로 NMS 유무의 영향을 받는다. 같은 정의·같은
코드로 재지만, v8/11 은 NMS 로 중복이 제거된 뒤 세어지고 26 은 one2one 출력이라
중복이 애초에 적다. **절대값 비교보다 순서·자릿수 비교로 읽을 것.**
보조로 프레임 단위 `neg_detect_rate`(개수 무관)를 함께 싣는다.

## PnP / 6D

Y0 의 정본 real 평가(`cf_real_eval.py`)는 **2D keypoint 전용**이다 — R/t/5cm5 항목이
없다. 즉 "Y0 에서 사용 가능한 동일 PnP evaluator" 가 존재하지 않는다.
지시상 새 solver 도입이 금지이므로 **6D 평가는 이번 표에서 제외**한다.

## synthetic 9kp median / p90

기존 `Y0_SYNTH.json` 은 ultralytics val 의 mAP 만 담고 있어 9kp 항목이 없다.
real 쪽과 **같은 정의**(top-1 by box conf, correct_box IoU>=0.5, keypoint L2 px)를
G38 val 1,998 에 적용해 세 모델 모두 새로 계산한다. real 과 달리 synthetic val 이미지는
이미 PAD100 캔버스이므로 **추가 padding 을 하지 않는다**(이중 패딩 방지).
mAP 계열은 기존 정의(ultralytics val) 그대로 둔다.
