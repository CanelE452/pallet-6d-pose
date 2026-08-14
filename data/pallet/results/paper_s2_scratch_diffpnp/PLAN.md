# PAPER_S2_SCRATCH_SIGMA2_DIFFPNP3D — 실험 계획 (canonical)

> 작성: 2026-07-08. 사용자 지시문 기반. 이 문서가 계획 canonical.

## 목적 (한 줄)
논문-safe 데이터만으로 scratch 학습하되, heatmap keypoint 위에 **dimensions-aware DiffPnP3D 3D-corner geometry regularization**을 붙여
진단된 병목(**rear corner 4~7 저앙각 depth collapse**)을 완화한다. 새 decoder가 아니라 loss regularizer 실험.

## 판단 지표 (go/no-go)
- **1차(primary)**: synthetic val `rear_med` ↓ 또는 `honest8` ↓
- **guard**: `front_med` 악화 없음 / `det`·`good` 급락 없음 / `gross` 증가 없음 / PnP-only 좋고 honest8 나쁨 금지
- 최종 checkpoint = **synthetic validation composite best** (real eval로 best epoch 선택 금지)

## ★ 컴퓨트 게이팅 (사용자 결정 2026-07-08)
**cheap-first**: Q0/Q1 quick screen만 먼저 수행. **Q1에서 rear_med 또는 honest8 개선이 있을 때만** full scratch(Stage A/B) 진행.
개선 없으면 **STOP → Windows(FoundationPose) 저앙각/rear 데이터 생성 트랙으로 전환** (memory: rear 레버 = 데이터/appearance).

## memory 정합성 (인지하고 진행)
- 출하 모델(paper_base_v2, paper_s1_maskaux) 모두 `geo_loss=False`·`struct_loss=False`·`coord=False` [확인] — 기하/구조 loss는 스크리닝 후 OFF.
- STAGE22/23 결론: "rear 레버는 데이터/appearance지 loss표현 아님". → **이번 실험은 2D loss가 아닌 3D(dims-aware) loss라 다른 각도**. depth를 직접 제약. dims-free 2D 불가 memory와도 정합(dims 사용).
- 리스크: DiffPnP3D는 `pnp_valid_3d ∧ V=8`(clean full-view)에만 적용 → 정작 hard rear 저앙각 레짐엔 안 닿을 수 있음. **Q1이 이 가설을 검증**.

## 데이터 (총 29,308)
- Arm A (paper-original, mask 없음, 19,308): mixed_v8_train 9000 / v4_split_base 4000 / aug_squash_v2 2212 / aug_trunc_v2 2971 / aug_scale_v2 1125
- Arm B (paper_4pallet_mask_v1, mask 있음, 10,000): RGB + mask_rle, 4 pallet, V=8 100%
- Stage A = Arm A만 (heatmap + DiffPnP3D). Stage B = A+B 60:40 (B는 +mask_aux).

## JSON 스키마 사실 (2026-07-08 확인 — 위임 입력)
- K: `camera_data.intrinsics` {fx≈615, fy≈615, cx=320, cy=240}, 이미지 640×480. **imagesize=448 리사이즈와 K 스케일 정합 주의.**
- GT pose: `pose_transform`(4x4) 또는 `quaternion_xyzw`+`location`.
- 3D cuboid: `cuboid`(8 corners) — **모든 프레임 존재**. `dimensions_m`(dict)은 **일부 프레임 결측**(mixed_v8 ~75%만) → 결측 시 `cuboid` edge에서 dims/diag 유도.
- V=8: **`V_geom` 필드 없음** → projected_cuboid inside-image count로 산출.
- 2D-only aug: aug_squash_v2 / aug_scale_v2 는 `aug` 필드 존재, dims stale → **DiffPnP3D 제외**(pnp_valid_3d=false 예상).

## 조건 (고정)
- scratch (net_path=None, base/v3/addon/palletobj checkpoint 금지)
- batchsize 12, sigma 2, no-pad/aspect, lr 1e-4
- model = DOPE heatmap(9) + affinity + mask_aux(Stage B) + DiffPnP3D loss
- 금지: vector/edge/sparse/face-center offset/center-offset head, mask hard-gate
- DiffPnP3D: `pnp_valid_3d==true ∧ V=8 ∧ belief-interior`에만, pred_xy = local 7x7 soft-argmax(미분가능, detach 금지), 3D corner Huber/diag-normalize, GN unroll(GT-pose init), guards(NaN/positive-depth/condition/grad-norm)
  - ★ **belief-interior gate (Q0 발견, 2026-07-08)**: `CreateBeliefMap`는 keypoint가 belief 경계에서 `2*sigma`px 미만이면 그 채널을 **빈 상태로** 그린다(edge corner=empty→soft-argmax garbage). `V8`(이미지 안)로는 부족. 로더가 변환된 8 corner가 belief `[2σ, size-2σ)` 안에 있을 때만 diffpnp_valid=1. 이 게이트 후 GT-belief soft-argmax=K-proj(X_i) <0.5px(전 17~51px). full-set pool=14,329(valid&V8의 76.3%: mixed 94%/v4 71%/paper 70%).
  - aspect_resize + eligible 프레임은 **rotate skip**(이미지 중심 회전≠카메라 roll이라 기하 파괴). belief↔orig=고정 anisotropic(640×480→400→50, scale 12.8/9.6).
- λ_pnp warmup(ep0~ramp), 후보 0/0.001/0.003/0.005/0.01, 시작 0.001
  - ★ **Q0 캘리브레이션(2026-07-08)**: raw belief-grad ratio ~1097% → 5~30% effective λ∈[0.0046,0.0273]. λ=0.005=5.3%(band 안), 0.003=2.5%(약함). **Q1 권고 λ≈0.005~0.008**. sa_err 수렴 ~4.5px(0.4 belief px)=belief 해상도 floor(1 belief px≈12.8 orig px), <1 orig px 불가(정상).

## Phase (cheap-first)
- **P0** (완료 중): 계획 기록 + 산출물 골격 + val(1500) 확인.
- **P1 (3d-expert 위임)**: ① pnp_valid_3d 감사 스크립트(reproj≤1px) ② 로더 dims/K/pose/pnp_valid/V plumbing(기존 학습 불변) ③ DiffPnP3D loss(GN unroll·3D corner·guards) ④ local 7x7 soft-argmax + confidence.
- **P2**: Q0 32-sample overfit — 미분/grad 도달/NaN·depth guard/grad-norm 비율(heatmap의 5~30%) 검증.
- **P3**: Q1 1k screen (λ 0/0.001/0.003/0.005/0.01, 2~3ep) — **rear_med·honest8 go/no-go**. 통과 못 하면 STOP.
- (통과 시) **P4**: Stage A scratch 60ep → Stage B mask FT 10~20ep, synthetic-val best.
- **P5**: real eval 1회 + 비교표 + dims/ratio 입력 추론 API.

## 산출물
`data/pallet/results/paper_s2_scratch_diffpnp/`: PLAN.md, pnp_valid_3d_audit.json, quick_screen_results.md, train_config_stage{A,B}.yaml, *_eval_summary.md, final_comparison_table.md, failure_montage/
`data/pallet/eval_results/paper_s2_scratch_diffpnp/`: eval 산출.
