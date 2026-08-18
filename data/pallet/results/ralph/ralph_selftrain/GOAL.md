# Ralph Loop — Self-Training 개선 목표 (2026-07-15)

## 목적 (한 줄)
Self-training 모델이 **자기 base(R0) 대비 eval GT(사용자 어노, held-out)에서 실제 개선**을 달성.
(지금까지 강한 base 위 모든 self-train이 R0 미개선 → 벽을 깰 수 있는 가설공간을 자율 탐색)

## 판단 지표 (eval GT held-out, 동일 하네스: squash400/belief50/THRESH0.3/N_DET_MIN6)
- **주지표**: eval GT 전체 corner_med(px) 감소 (order-free Hungarian median).
- **보조**: NN<20px per-frame % 증가, det% 유지/증가.
- **성공선**: 어떤 라운드든 R0 대비 corner_med 유의 감소 **또는** NN20 +3pt↑ (소표본이니 방향+크기 함께 판단).

## 가설공간 (사용자 확정 방향: 과거 성공 재현 먼저)
- **H1 (현재)**: weak base(pallet_category) + self_train.py(ransac_loo, synthetic_ratio 0.5, strong aug).
  - 개선 재현 O → "파이프라인/base weakness가 원인" 확정 → 강한 base 개선 방향으로.
  - 개선 재현 X → "데이터/신호 자체가 원인" 쪽 → 직교신호(mask-coverage/temporal/TTA)로.
- confound 기록: 과거 synthetic(train/ 2000장 Isaac) 소실 → paper_4pallet_mask_v1(10k)로 대체.
  net_pallet_best.pth 소실 → final_net_epoch_0060.pth 사용. image_size 448(train) vs 400(eval harness).

## 비교 통제 (강한 base 실패 실험과 직접 비교되도록)
- real unlabeled pool = 최근 full-pool과 동일(INF.DOMAINS, wood 제외, eval GT 프레임 홀드아웃).
- eval GT = outside117/night43/noapril18/cad44 (전체 GT, 누수 홀드아웃).
- 바뀐 변수만: base(weak vs strong s2) + pipeline(self_train.py vs DOPE train.py) + filter(ransac_loo vs FULL7/reproj).

## 정지 조건
- H1에서 개선 재현 또는 명확한 미재현 결론 → 결과 Discord 통지 후 다음 가설.
- 사용자가 "멈춰" 할 때까지 자율 반복.
