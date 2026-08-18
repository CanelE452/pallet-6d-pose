# PURPOSE — 과제용 YOLO26n-pose 단일 모델 (합성 Stage A)

[소비처] 리프터 실배포. 정지 후 RealSense RGB 한 장 → 팔레트 bbox + 9 keypoint →
         기존 배포 후처리(`depth_cam/calib/pose6d_adapter.py`)의 PnP + depth 로
         yaw / lateral offset / forward distance 를 계산한다.
         이번 라운드 산출물은 real finetune 의 **출발 weight** 가 된다.

[문장]   "여러 팔레트 자산으로 만든 범용 합성 40k(G)와 과제 팔레트 합성 20k(T)를
         1:1 로 섞어 100px reflect padding 계약으로 학습하면, real 데이터를 한 장도
         쓰지 않고도 과제 팔레트의 9 keypoint 를 합성 val 에서 안정적으로 예측하는
         base weight 를 얻는다."

## 범위 (사용자 지시 2026-08-14)
- **합성만**. real 133장(13세션)은 이번에 쓰지 않고 이후 finetune 단계로 미룬다.
- 따라서 Stage B, real validation, yaw/PnP task evaluator 도 이번 범위 밖이다.
- ★ real val 이 없으므로 checkpoint 선택 근거는 **synthetic val 뿐**이다.
  real task metric(PnP valid rate, yaw error)은 측정하지 않았으므로 주장하지 않는다.

## 판단 지표 (이번 라운드)
```
1. 합성 val pose mAP50-95 / mAP50            (T val 우선, G val 은 sanity)
2. keypoint 정규화 오차 (T val)
3. 학습 안정성 — NaN 없음, loss 단조 감소, VRAM 여유
```
real 관련 지표(catastrophic yaw, PnP 성공률)는 **다음 단계**의 지표다.

## 고정 상수
```
model      yolo26n-pose.pt (challenge/weights/pretrained_yolo/)
imgsz 640  batch 32  nbs 64  seed 42  deterministic  AMP
padding    100px BORDER_REFLECT_101, 전 도메인 동일
fliplr     0.0 (flip_idx 는 계약 기록용으로만 보존)
env        conda activate pallet-yolo26   (ultralytics 8.4.60)
```

## 봉인 / 금지
- `split == "eval"` real 161장은 정본 평가셋이다. 학습에 넣지 않는다.
- 기존 `challenge/yolo_pose`, 논문용 DOPE weight, 기존 데이터 원본을 수정하지 않는다.
- `convert_to_camera_facing_v4.py` 를 G/T 에 다시 돌리지 않는다
  (reports/02 §7 — 2D 휴리스틱이 저앙각에서 오판).
