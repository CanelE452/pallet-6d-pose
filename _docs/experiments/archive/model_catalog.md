# 모델 카탈로그

상태: **갱신** (2026-04-15)

본 프로젝트에서 학습된 모든 DOPE 모델의 요약. 상세 카드는 `_docs/models/`
에 있을 수 있음. 여기는 실험 문서에서 reference 하기 쉬운 요약본.

## 현행 모델

```
모델                    학습 데이터                      이미지 수   초기 weight                           비고
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v8                Isaac + Blender (1:1)             9,000      scratch                               Phase 0 baseline 60 ep
mixed_v8 / ep60         위 scratch 60 ep final            —          —                                     **pretrain baseline**
v8_ablation_A_coord     mixed_v8 + coord ft 5 ep          9,000      mixed_v8/ep60                         **primary anchor** ★ (λ=0.003, seed 8452)
v8_coord_ft_reproduce   위 재현 (다른 seed)                9,000      mixed_v8/ep60                         seed 9587, 결과 일치 확인
v8_exp3_coord_scratch   belief+coord joint 60ep           9,000      scratch (ImageNet)                    warmup=10, **실패** (pretrain 보다 나쁨)
v8_ablation_B_edge      + edge only                       9,000      mixed_v8/ep60                         Loss ablation
v8_ablation_C_coord_edge + coord + edge                   9,000      mixed_v8/ep60                         Loss ablation (실패)
v8_ablation_D_flip      + flip only                       9,000      mixed_v8/ep60                         Loss ablation (실패)
v8_ablation_E_rel       + reliability loss                9,000      mixed_v8/ep60                         noapril good, middle bad
v8_vis                  + visibility-aware reweighting    9,000      mixed_v8/ep60                         ablation (결과 보류)
v8_A_control            v9_A_coord + 3ep ft on v8         9,000      v9_ablation_A_coord/ep65              legacy, 이전 anchor
selftrain_r1            v8_A → ST R1 (legacy filter)      +188 PL    v8_A_control/ep68                     pre-RANSAC 시절
ST_8only                ep60 + 8 PL (B∧C) real-only       8 PL       mixed_v8/ep60                         91 ep, NN <20px 37.3% (middle 440)
T1_none                 ep60 + 64 PL (no filter) mixed    64 PL      mixed_v8/ep60                         5 ep, mixed training 실패
T1_ransac               ep60 + 6 PL (RANSAC) mixed        6 PL       mixed_v8/ep60                         5 ep, oversampling 문제 (5625회/장)
F3                      ep65 + RANSAC+LOO (2,324 pool)    ?          v8_A_coord/ep65                       real-only, 평가 미완
F4                      ep65 + 9 PL (RANSAC only)         9 PL       v8_A_coord/ep65                       real-only 90ep, NN <20px 21.8%
F5 ★                    ep65 + 2 PL (RANSAC+LOO)          2 PL       v8_A_coord/ep65                       real-only 96ep, NN <20px 60.5% (seed=4165)
f5_reproduce            F5 재학습 (다른 seed)              2 PL       v8_A_coord/ep65                       NN <20px 53.9% — seed 민감도 ~6-7pp
mixed_v9                Isaac + Blender + indoor           ~8,500     scratch                               mid-term (test_indoor_v1 포함)
mixed_v10               v8 + test_indoor_v1               10,000     scratch                               **폐기** — annotation broken
v10_exp1_coord_ft       mixed_v10 + coord ft              10,000     mixed_v10/ep60                        **폐기** — pretrain 오염
```

## Legacy 모델 (v1 ~ v7)

```
모델               학습 데이터                      이미지 수   비고
──────────────────────────────────────────────────────────────────────
pallet_category    Isaac Sim train/                 ~2,000     very first baseline
pallet_v11         Isaac Sim train/                 4,000      fine-tune 91 ep
pallet_v11_far     Isaac Sim train/ + far           6,000      far distance 포함
blender_v1         Blender only                     3,600      multi-source ablation (T9)
combined_v1        Isaac 6K + Blender 3.6K          9,600      multi-source ablation (T9)
mixed_v1           Isaac 4K + Blender 4K (1:1)      8,000      first mixed, Fair eval 1 위
mixed_v2 ~ v7      (iteration 단계)                  —          렌더 / DR 실험 반복
```

## 명명 규칙

```
mixed_v{N}                  Isaac + Blender 1:1 혼합, N 차 renderer / DR iteration
mixed_v{N}_train            학습용 train split
mixed_v{N}_val              val split (동일 seed 별도 생성)
v{N}_ablation_{X}           mixed_v{N} base 에 대한 구조적 loss ablation
v{N}_A_control              해당 iteration 의 coord only winner (ablation anchor)
v{N}_a_coord                간략 표기 (v10_a_coord = v10 + coord)
mixed_v{N}_st_{source}      v{N} + self-training on {source}
selftrain_r{k}              legacy ST round 결과 재명명
```

## Primary anchor 방침

- 현재 (2026-04-15): **v8_ablation_A_coord (mixed_v8/ep60 + coord ft 5 ep, λ=0.003)** — Phase 1 ablation 기준 ★
- 재현 실험(seed 9587) 에서 결과 일치 확인 (noapril PnP 72.3% vs 75.5%)
- v10 계열은 test_indoor_v1 annotation broken 으로 전부 폐기
- joint scratch (v8_exp3_coord_scratch) 는 pretrain 보다 나쁨 → sequential ft 가 유일 전략

## 최종 best model (2026-04-14)

- **F5 (`weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth`)**
- 설계: ep65 anchor + RANSAC+LOO 필터로 선별한 2 PL + **real-only** fine-tune 96 ep
- NN matching <20px = **60.5%** (seed=4165), 재현 53.9% → 발표 범위 **54~60%**
- ep65 기준 baseline(21.6%) 대비 +38.7pp
- Mixed training (synthetic+PL) 은 전부 실패 — real-only가 유일한 성공 전략

## 관련

- Loss ablation: [`loss/ablation.md`](./loss/ablation.md)
- Multi-source: [`synthetic/multisource.md`](./synthetic/multisource.md)
- Coord strategy: [`loss/coord_strategy.md`](./loss/coord_strategy.md)
- 메모리: `project_ablation_baseline_setup.md`

---

# YOLO26-pose 트랙 (challenge/) — DOPE와 별개 라인

DOPE keypoint→DA 파이프라인과 독립적으로, `challenge/`에서 YOLO26n-pose로 팔레트 9 keypoint를 직접 회귀하는 트랙. env=`pallet-yolo26` (ultralytics 8.4.60).

## 모델 목록

| 모델 | 방식 | 데이터 | weight | 비고 |
|------|------|--------|--------|------|
| `yolo26n_pose_v1` | — | 합성 pretrain | `challenge/weights/yolo26n_pose_v1/` | base |
| `yolo26n_pose_v1_ft_manual` | padding | manual 219 | `challenge/weights/yolo26n_pose_v1_ft_manual/` | 기존 A |
| `yolo26n_pose_v1_ft_manual_nopad` | 비패딩 | manual 219 | `runs/pose/challenge/weights/.../` | 기존 B |
| `yolo26n_pose_v1_ft_pad_ho` | padding | holdout 177 | `runs/pose/challenge/weights/.../` | **비교 A — 권장 ★** |
| `yolo26n_pose_v1_ft_nopad_ho` | 비패딩 | holdout 177 | `runs/pose/challenge/weights/.../` | 비교 B |

## 결론: padding vs 비패딩 (truncation 강건성, 2026-06-02)

- leakage 없는 holdout(frame 80/20, seed42) 재학습 + crop 강도별 평가로 비교.
- **심한 truncation(화면 밖 코너 3~4개)에서 padding(A) 압승**: PnP 성공률 76.2% vs 45.2%, reproj median 21px vs 31.5px.
- 원인: 비패딩(B)은 truncation 코너를 v=0 학습 → 추론 시 유효 keypoint가 6점 미만으로 떨어져 EPnP 붕괴. padding(A)은 화면 밖 코너까지 9점 회귀해 PnP 안정.
- 원본·가벼운 잘림에서는 A·B 동등 — 차이는 심한 truncation에서만.
- **권장: 팔레트가 자주 잘리는 환경(forklift)엔 padding 학습.**
- 단 절대 6D 정확도(ADD/5cm5°)는 양쪽 다 낮음 — flat 팔레트 광축 depth가 PnP로 약제약되는 task 특성(padding 무관, 별도 과제).
- 상세: `_docs/history/2026-06-02.md`, 재현 `challenge/scripts/eval_ab_crop.py`, 결과 `challenge/data/04_results/ab_crop_eval/`.

---

# DOPE crop+padding 증강 트랙 (dope_cropaug) — truncation 강건 DOPE

YOLO crop-aug 방식을 DOPE에 이식 (2026-06-02). pretrain=synthetic-only crop, ft=real 2단계.

| 모델 | base | epoch(누적) | 데이터 | weight |
|------|------|------|--------|--------|
| `dope_cropaug_pretrain` | scratch | 60 | mixed_v8 9000 + synth crop 8831 | `weights/dope/dope_cropaug_pretrain/final_net_epoch_0060.pth` |
| `dope_cropaug_ft_s1` | pretrain | 150(+90) | real GT 251 | `weights/dope/dope_cropaug_ft_s1/final_net_epoch_0150.pth` |
| **`dope_cropaug_ft_s2` ★** | ft_s1 | 180(+30) | real 251 + crop 485 | `weights/dope/dope_cropaug_ft_s2/final_net_epoch_0180.pth` |

## 성능 (real truncation, order-free PnP)
| model | det≥6 | PnP% | reproj med |
|------|-------|------|-----------|
| baseline_v8_A | 12.8% | 22.7% | 134.5px |
| ft_s2 | **94.2%** | **98.8%** | **19.3px** |

- truncation에서 baseline은 거의 붕괴(0/9 kp), ft_s2는 화면 밖 corner까지 복원. clean에선 s1≈s2.
- **주의**: `evaluate_on_val.py` reproj는 convention 불일치로 무의미(baseline도 135px). order-free PnP 필요.
- 재현: 데이터 `gen_truncation_crops.py`→`pad_truncation_crops.py`(MARGIN_FRAC=0.20), 학습 `scripts/ft_dope_cropaug.sh`. train.py finetune은 EPOCHS=누적 목표.
- 카드: `challenge/_docs/models/dope_cropaug.md`. 메모리: `dope-cropaug-truncation-success`.
