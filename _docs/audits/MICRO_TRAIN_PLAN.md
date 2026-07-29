# MICRO TRAIN PLAN — 계획만 작성, 이번 작업에서 실행하지 않음

공통 조건: 200~500 training frames / mechanism-val 평가 / 5~10 epoch / seed 1 / 동일 initialization / 동일 sampler order / 동일 optimizer / **한 요소만 변경**.

승격 gate: primary target failure class 에서 pose success +5%p **또는** keypoint gross error −10%p **또는** yaw/reprojection 10% 이상 개선이 있고, clean/non-target subset 성능 하락이 3%p 미만일 때만 1-seed full smoke 후보로 승격.


## 후보 1 — B_far_depth_structured_refinement

- hypothesis: far/depth face 는 response 는 있는데 위치가 틀린다 (confident-wrong).  dimension-conditioned bounded residual 이 far face 를 교정한다.
- exact code change: final belief 위에 bounded residual heatmap head + FiLM(dims) 추가, far/depth edge consistency loss 를 heatmap anchor 와 함께 사용.
- trainable parameters: residual head + FiLM
- frozen parameters: backbone, stage1~6 belief
- train subset: non-truncated 400 frames
- validation failure class: F2_CONFIDENT_WRONG (mechanism-val N=35)
- metrics: pose success rate, yaw median, fixed-GT reproj median, matched 2D median, gross(>20px) 비율
- expected result: F2_CONFIDENT_WRONG 에서 pose success +5%p 이상 또는 yaw median 10% 이상 감소
- failure result: target class 무변화이거나 non-target 하락 3%p 이상 → 후보 폐기
- stop condition: 10 epoch 도달, 또는 val loss 2 epoch 연속 악화, 또는 gate 미달 확정
- runtime estimate: 400 frames × 10 epoch ≈ 25~40 분 (RTX 3080, batch 8)

## 후보 2 — A_visibility_partial_keypoint

- hypothesis: F1 미검출 corner 의 대부분은 GT 가 화면 **안**이므로(진단 §2) 원인은 '화면 밖 supervision' 만이 아니다.  단 화면 밖/경계 keypoint 를 background-negative 로 감독하면 경계 근처 response 가 함께 억제된다는 가설은 남는다.  outside 를 loss 에서 빼고 crop-only 로 경계 사례를 보여주면 in-frame response 도 함께 회복된다.
- exact code change: train.py belief target 생성에서 outside keypoint 를 loss mask 로 제외하고 (9ch belief + 9ch validity mask), PnP 는 valid correspondence 만 사용. 증강은 reflect-pad 가 아닌 **crop-only** 로 실제 frame-edge 분포를 준다.
- trainable parameters: m6_2 final belief stage + 새 validity head
- frozen parameters: vgg backbone, stage1~5
- train subset: crop-only 증강 포함 400 frames (truncation 비율 40%)
- validation failure class: F1_NO_RESPONSE (mechanism-val N=24)
- metrics: pose success rate, yaw median, fixed-GT reproj median, matched 2D median, gross(>20px) 비율
- expected result: F1_NO_RESPONSE 에서 pose success +5%p 이상 또는 yaw median 10% 이상 감소
- failure result: target class 무변화이거나 non-target 하락 3%p 이상 → 후보 폐기
- stop condition: 10 epoch 도달, 또는 val loss 2 epoch 연속 악화, 또는 gate 미달 확정
- runtime estimate: 400 frames × 10 epoch ≈ 25~40 분 (RTX 3080, batch 8)

## 후보 3 — C_calibrated_uncertainty_robust_pnp

- hypothesis: 작은 2D error 가 PnP 에서 증폭된다.  learned localization sigma 로 correspondence 를 가중하면 yaw 가 안정된다.
- exact code change: per-corner log-sigma head (cornerQuality) 학습 + bounded inverse-variance weighted PnP (raw covariance 금지, calibration 후 사용).
- trainable parameters: corner quality head
- frozen parameters: backbone + belief 전부
- train subset: mixed 500 frames
- validation failure class: F3_GEOMETRY_AMPLIFIED (mechanism-val N=4)
- metrics: pose success rate, yaw median, fixed-GT reproj median, matched 2D median, gross(>20px) 비율
- expected result: F3_GEOMETRY_AMPLIFIED 에서 pose success +5%p 이상 또는 yaw median 10% 이상 감소
- failure result: target class 무변화이거나 non-target 하락 3%p 이상 → 후보 폐기
- stop condition: 10 epoch 도달, 또는 val loss 2 epoch 연속 악화, 또는 gate 미달 확정
- runtime estimate: 400 frames × 10 epoch ≈ 25~40 분 (RTX 3080, batch 8)
