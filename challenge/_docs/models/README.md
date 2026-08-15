# Challenge 모델 카탈로그

challenge series 학습 모델 목록. 각 모델의 학습 설정/데이터/메모를 별도 파일로 분리.
(`_docs/models/` 의 mixed_v* / v8_ablation 카탈로그와 같은 구조)

## 문서 구조

```
파일                                 내용
────────────────────────────────────────────────────────────────────────────────
baseline_v8_A.md                     mixed_v8 → v9_A_coord → 3ep ft (depth_cam 기본 모델)
challenge.md                         scratch, mixed_v8 + chal_v1 + chal_v2 (60ep)
challenge_camfacing_scratch.md       challenge 와 동일 셋업 재학습 (seed 다름)
challenge_camfacing_ft.md            challenge 위 10ep refine (lr=1e-5)
challenge_ft_pallet07.md             baseline_v8_A 위 pallet07 단일 capture 91ep ft
challenge0123.md                     scratch, mixed_v8 + chal_v1 + chal_v2 (60ep, v4 convention)
challenge0123_ft_manual.md           challenge0123 위 6 manual GT 20ep ft
challenge0123_ft_v2.md               challenge0123 위 14 manual GT (낮+야간) 60ep ft
```

## 모델 요약

```
모델                            Weight 경로                                                     Ep      LR       시작점                                      비고
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
baseline_v8_A                   challenge/weights/baseline_v8_A.pth                              68      1e-5     v9_ablation_A_coord/ep65                    struct_coord=0.003, depth_cam MODEL_PATH
challenge                       weights/challenge_track/challenge/final_net_epoch_0060.pth                       60      1e-4     scratch                                     첫 camfacing 학습
challenge_camfacing_scratch     weights/challenge_track/challenge_camfacing_scratch/final_net_epoch_0060.pth     60      1e-4     scratch                                     challenge 와 같은 설정, seed 만 다름
challenge_camfacing_ft          weights/challenge_track/challenge_camfacing_ft/final_net_epoch_0070.pth          70      1e-5     challenge/ep60                              10 ep refine
challenge_ft_pallet07           weights/challenge_track/challenge_ft_pallet07/final_net_epoch_0091.pth           91      5e-5     challenge/weights/baseline_v8_A             symmetric+struct, pallet07 단일 capture
challenge0123                   weights/challenge_track/challenge0123/final_net_epoch_0060.pth                   60      1e-4     scratch                                     header outf=challenge_camfacing_v4 → rename
challenge0123_ft_manual         weights/challenge_track/challenge0123_ft_manual/final_net_epoch_0080.pth         80      1e-4     challenge0123/ep60                          6 manual GT (pallet 03/04/05/07/09/cad)
challenge0123_ft_v2             weights/challenge0123_ft_v2/net_epoch_0080.pth (..0120 학습 중)  120     1e-4     challenge0123/ep60                          14 manual GT (낮 8 + 야간 6)
```

## 공통 학습 하이퍼

```
imagesize : 448
sigma     : 4.0   (belief Gaussian)
intrinsic : fx=614.18, fy=614.31, cx=329.28, cy=234.53, img=640x480 (D435i)
object    : pallet
geo_loss  : False (모든 모델)
```

## 추론 시 contract (depth_cam 통합)

`depth_cam/tools/twin_pnp_check.py` 50/50 frame 검증 결과 — 모든 challenge 시리즈 공통:

```
PALLET_WIDTH_M  = 1.0    (mixed_v8_train 의 실제 라벨 dim)
PALLET_DEPTH_M  = 1.2
PALLET_HEIGHT_M = 0.15
PALLET_PNP_CONTRACT_Z180 = True    (Cuboid3d.vertices @ diag([-1,-1,+1]))
```

`task.yaml` 의 `(1.1, 1.3, 0.11)` 은 spec 값이며 학습 라벨과 다름.

## 참고

- 상위 카탈로그 (mixed_v*, v8_ablation 등): `_docs/models/README.md`
- camera-facing convention v1~v5 결정 흐름: memory `project_keypoint_convention_v4_conversion.md`, `..._v5_yaw_dist.md`
- annotate.py PnP fix v4 (gravity invariant): memory `project_annotate_pnp_fix_v4_gravity.md`
