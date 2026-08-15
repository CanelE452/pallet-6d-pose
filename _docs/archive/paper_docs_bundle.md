# Pallet 6D Pose — 논문 docs 전체 번들 (LLM 컨텍스트용)

> 생성일: 2026-06-05
> 포함: `_docs/` 논문 트랙 (archive=폐기 v8 / history=변경로그 / migration 제외)
> 파일 수: 28개
> 주의: convention = **camera-facing 0123** (v8 object-frame 폐기). challenge(v1/v2) 트랙 제외, 논문용(일반화) 트랙.

---

## 목차

- experiments/data/A1_paper_base_perf.md
- experiments/data/A2_squash_ratio_ablation.md
- experiments/data/A3_truncation_padding_ablation.md
- experiments/eval/D1_generalization_seen_unseen.md
- experiments/eval/D2_real_test.md
- experiments/eval/D3_pnp_solver.md
- experiments/eval/F2_qualitative.md
- experiments/filter/B2_filter_selftraining.md
- experiments/filter/pr_screening.md
- experiments/README.md
- experiments/related_work.md
- experiments/self_training/C1_rounds.md
- experiments/self_training/C2_pl_quality_vs_quantity.md
- filter/2026-06-02_survey_pseudolabel_filtering.md
- filter/README.md
- method/evaluation.md
- method/overview.md
- method/step1_synthetic_data.md
- method/step2_geometric_filter.md
- method/step3_selftraining.md
- models/dope_architecture.md
- models/paper_base.md
- models/README.md
- models/training_loss.md
- preprocessing/keypoint_definition.md
- README.md
- survey/compare-self6dpp-vs-ours.md
- survey/survey-6d-pose-estimation.md

---


================================================================================
FILE: experiments/data/A1_paper_base_perf.md
================================================================================

# A1. paper_base 합성학습 성능 (base)  (논문 Table)

> 상태: **미시작** (paper_base 학습은 다른 머신) | 의존: paper_base 학습 완료
> 구분: **다시** (모델 자체가 camera-facing 새 base)

## 목적 (한 줄)
논문용 base 모델 `paper_base`(camera-facing 합성 + squash + truncation padding, v1/v2 제외)의 기준선 성능 확립.

## 판단 지표
합성 val에서 **PCK@3/5/10 · 검출률 · reproj(9kp)**. self-training 전 출발점.

## 설정
- 모델: `paper_base` (학습 데이터 = `mixed_v8_train`(camfacing) + `aug_squash` + `aug_trunc` + `aug_scale`)
- 평가: 합성 val (TBD: held-out 합성셋) + real held-out 일부
- 비교 레퍼런스: `dope_cropaug_pretrain`(squash 없는 전신)

## 방법
1. paper_base 학습 (scratch, train_dope.sh)
2. 합성 val + real 평가
3. dope_cropaug_pretrain 대비 squash 효과 1차 확인

## 결과 (TBD)
```
model                PCK@5   det%   reproj_9kp_med
─────────────────────────────────────────────────
dope_cropaug_pretrain  TBD    TBD    TBD
paper_base             TBD    TBD    TBD
```

## 결론 (TBD)


================================================================================
FILE: experiments/data/A2_squash_ratio_ablation.md
================================================================================

# A2. Squash 비율 강건성 Ablation  (논문 Table)

> 상태: **미시작** | convention: camera-facing 0123 | 의존: paper_base, paper_base_nosquash 학습
> 구분: **새로** (논문 핵심 — 처음 본 비율 일반화)

## 목적 (한 줄)
squash(찌부/늘림) 증강이 **학습에 없던 aspect ratio** 파렛트의 keypoint 추정 일반화를 실제로 높이는가.

## 판단 지표
비율이 다양한 테스트셋에서 **keypoint reproj(전체 9kp 평균) + 검출률**. squash 유 vs 무 비교.
(부차적 매력 주의: clean 정밀도만 보지 말고 unseen 비율에서의 일반화가 본질)

## 설정
- 모델: `paper_base`(squash O) vs `paper_base_nosquash`(squash X, 나머지 동일)
- 데이터: 학습=mixed_v8 camera-facing (±squash) / 평가=비율 다양한 파렛트 (TBD: 합성 비율 grid + real)
- convention: camera-facing 0123, 9kp

## 방법
1. squash 유/무 두 모델 동일 조건 학습 (squash만 차이)
2. aspect ratio 구간별(예: 0.6/0.8/1.0/1.3/1.6) 테스트셋에서 9kp reproj·검출
3. 학습 분포 밖 비율에서 격차 확인

## 결과 (TBD)
```
ratio    nosquash 9kp_med   squash 9kp_med   검출률 Δ
─────────────────────────────────────────────────────
0.6      TBD                TBD              TBD
...
```

## 결론 (TBD)

## 산출물 (예정)
- 스크립트/결과 경로 TBD, 비율별 곡선 figure


================================================================================
FILE: experiments/data/A3_truncation_padding_ablation.md
================================================================================

# A3. Truncation Padding Ablation  (논문 Table)

> 상태: **미시작** (dope_cropaug에서 일부 검증됨 — 논문용 재정리) | 의존: paper_base(padding) vs nopadding
> 구분: **새로**

## 목적 (한 줄)
DOPE padding(잘린 코너의 belief를 padding 영역에 supervise)이 truncation 이미지의 keypoint/PnP 강건성을 높이는가.

## 판단 지표
crop level(0/1/2)별 **검출률 · PnP 성공률 · reproj(전체 9kp)**. padding 유 vs 무.

## 설정
- 모델: `paper_base`(padding O) vs padding 미적용 동일 모델
- 데이터: GT known 프레임을 crop해 truncation 합성 (L/R 측면 위주, `eval_ab_crop` 방식 재활용)
- 참고: 기존 YOLO/DOPE 검증 — padding이 심한 truncation에서 PnP 76% vs 45%, DOPE PnP 23→99% (memory)

## 방법
1. padding 유/무 모델 학습
2. crop level별 GT-보정 평가 (화면 밖 코너도 offset 보정)
3. truncation 강도↑에서 격차 확인

## 결과 (TBD)
```
lvl   pad?   det%   PnP%   reproj_9kp_med
──────────────────────────────────────────
0     no     TBD    TBD    TBD
0     yes    TBD    TBD    TBD
...
```

## 결론 (TBD)

## 산출물 (예정)
- `challenge/scripts/eval_ab_crop.py` 계열 재활용, level별 곡선


================================================================================
FILE: experiments/eval/D1_generalization_seen_unseen.md
================================================================================

# D1. 일반화 — Seen vs Unseen 파렛트  (논문 Table)

> 상태: **미시작** | 의존: paper_base, real test 데이터
> 구분: **새로/다시** (v1/v2 제외 학습이라 unseen 정의가 바뀜 — 본 연구 핵심 주장)

## 목적 (한 줄)
논문용 모델은 **인터넷 합성만 학습(내 파렛트 v1/v2 제외)** → 내 실제 파렛트가 곧 **unseen**. 처음 본 파렛트 일반화를 정량화.

## 판단 지표
- **seen**(학습에 쓴 합성 파렛트 유형) vs **unseen**(처음 본 real 파렛트)의 keypoint reproj(9kp)·검출률
- PnP-free 2D 필터가 unseen에서도 작동하는지 (비율 unknown 적용성)

## 설정
- 모델: `paper_base` / `paper_r1`(self-train 후)
- seen 셋: 학습 분포 합성 파렛트
- unseen 셋: real(capturepallet/night/forklift) — 학습에 미사용 = unseen
- convention: camera-facing 0123

## 방법
1. seen/unseen 각각 추론 → 9kp reproj·검출
2. self-training 전후(R0 vs R1) unseen 개선폭
3. 일반화 갭(seen−unseen) 보고

## 결과 (TBD)
```
set       model        9kp_med   det%
──────────────────────────────────────
seen      paper_base   TBD       TBD
unseen    paper_base   TBD       TBD
unseen    paper_r1     TBD       TBD
```

## 결론 (TBD)


================================================================================
FILE: experiments/eval/D2_real_test.md
================================================================================

# D2. Real Test — ADD / 5cm5° / Reproj  (논문 Table)

> 상태: **미시작** | 의존: paper_base 또는 paper_r1, real GT(치수 known)
> 구분: **다시** (camera-facing + SQPnP로 재)

## 목적 (한 줄)
실제 파렛트에서 6D pose 정확도(metric). 치수 known 데이터에서만 (PnP 용도분리 B).

## 판단 지표
**ADD · 5cm5° · reproj(9kp)** — SQPnP, dims known.
(주의: monocular라 5cm5°는 약할 수 있음 — reproj median이 keypoint 품질의 깨끗한 신호)

## 설정
- 모델: paper_base / paper_r1 (+비교: challenge 과제 모델)
- GT: real manual GT (outside_combined·night_combined·forklift, dims per-frame), `_exclude.txt` 반영
- PnP: **SQPnP** (`cv2.SOLVEPNP_SQPNP`+RefineLM), order-free 비교
- 메모리: evaluate-on-val convention 버그 주의 (order-free PnP로)

## 방법
1. 추론 → 9kp → SQPnP → 6D
2. ADD/5cm5°/reproj 집계 (도메인별 + 전체)
3. R0 vs R1 개선

## 결과 (TBD)
```
model        ADD_med   5cm5°   reproj_med
──────────────────────────────────────────
paper_base   TBD       TBD     TBD
paper_r1     TBD       TBD     TBD
```

## 결론 (TBD)


================================================================================
FILE: experiments/eval/D3_pnp_solver.md
================================================================================

# D3. PnP Solver — SQPnP vs EPnP+RANSAC  (논문 Table, 부가)

> 상태: **부분 (challenge에서 검증됨, 논문용 재기록)** | 의존: real GT
> 구분: **새로** (평가/거리용 solver 확정)

## 목적 (한 줄)
near-planar 팔레트에서 SQPnP가 EPnP+RANSAC보다 정확한지 — 평가/거리추정 solver 선택 근거.

## 판단 지표
**reproj median · ADD median · PnP 성공률** (동일 keypoint, solver만 교체).

## 설정
- 동일 예측 9kp, solver만 EPnP+RANSAC vs SQPnP(+RefineLM, median reproj>12px reject)
- GT: real (dims known)
- 기존 검증(2026-06-02 YOLO 경로): reproj 5.27→3.12px, ADD 96.6→90.7mm

## 방법
1. 같은 keypoint에 두 solver 적용
2. reproj/ADD/성공률 비교
3. `scripts/self_training/pnp_solver.py`(현 EPnP) → SQPnP 교체 반영

## 결과 (TBD)
```
solver           reproj_med   ADD_med   성공률
─────────────────────────────────────────────
EPnP+RANSAC      TBD          TBD       TBD
SQPnP            TBD          TBD       TBD
```

## 결론 (TBD)


================================================================================
FILE: experiments/eval/F2_qualitative.md
================================================================================

# F2. Qualitative / Failure Analysis  (논문 Figure)

> 상태: **미시작** (일부 overlay 존재) | 의존: 위 실험들
> 구분: **다시** (camera-facing 케이스로)

## 목적 (한 줄)
정성 결과와 실패 유형을 그림으로 — 강점(통과 PL 정합)과 한계(diag scale-skew, 뒷면 오차, truncation).

## 판단 지표
대표 케이스 overlay (성공/필터가 거른 bad/필터가 놓친 케이스). 정량 아님, 정성 전달.

## 설정
- 모델: paper_base / paper_r1
- 케이스: 도메인별 good / diag가 거른 catastrophic / diag가 놓친 scale-skew / 뒷면 오차 / truncation 복원
- 기존 자산: `filter_domain_analysis/overlays_s2/`, `diag_pass_overlays/`, `pl_gt_diff/exp4_overlays/`

## 알려진 실패 유형 (채울 것)
- diag scale-skew: 중심 맞고 균일 스케일 틀림 → diag 통과 (indoor)
- 뒷면(4-7) 오차: monocular depth ambiguity, top-down에서 큼
- 검출 붕괴: held-out 모델 real에서 빈약

## 결과 (TBD)
- figure 경로 TBD

## 결론 (TBD)


================================================================================
FILE: experiments/filter/B2_filter_selftraining.md
================================================================================

# B2. 필터별 Self-training Downstream  (논문 Table)

> 상태: **미시작** | 의존: paper_base, C1(rounds 인프라)
> 구분: **다시** (v8 폐기 → camera-facing 필터로 재). B1(Stage1 P/R)은 `filter/pr_screening.md` 참조.

## 목적 (한 줄)
Stage1 P/R로 선정한 필터(diag 등)가 **실제 self-training downstream 향상에서도 최선인가** — P/R proxy ↔ 실제 향상 상관 검증 (4월 빗나감 교정).

## 판단 지표
필터별 **R1 도메인 향상폭**. + Stage1 9kp 오차 랭킹 ↔ downstream 향상 랭킹 상관(산점도).

## 설정
- 후보 필터: diag / ratio / diag∧ratio / fullkp / ransac_loo (대조군 none)
- anchor = paper_base, 도메인별
- Stage1 결과(pr_screening.md): outside diag 9.9px / night diag∧ratio 7.9px

## 방법
1. 각 필터로 PL 추출 → R1 학습
2. 도메인별 향상폭 측정
3. **Stage1 9kp오차 랭킹 vs downstream 향상 랭킹** 상관 그림 (P/R proxy 신뢰도 검증)

## 결과 (TBD)
```
필터          Stage1 9kp_med   R1 향상(outside)   R1 향상(night)
─────────────────────────────────────────────────────────────
diag          9.9              TBD                TBD
diag∧ratio    (night 7.9)      TBD                TBD
ransac_loo    낮음(물량사망)    TBD                TBD
```

## 결론 (TBD)
- P/R proxy가 downstream을 예측하는가? (4월엔 빗나감 — 이번엔 명시 검증)


================================================================================
FILE: experiments/filter/pr_screening.md
================================================================================

# Filter P/R Screening — 2D 기하 필터 (camera-facing)

> Stage 1: 학습 없이 기존 camera-facing 모델 추론으로 필터 P/R 비교.
> 폐기 v8 필터 실험(selection/ablation/consensus_sweep)은 `archive/`.

## 목적

camera-facing 0123 모델이 뱉은 9 keypoint 예측에 2D 기하 필터를 걸어,
신뢰도 높은 pseudo-label 을 얼마나 잘 거르는지 Precision/Recall 로 비교 →
self-training 에 쓸 최적 필터 선정. **모델 학습 불필요.**

## 방법

```
camera-facing 모델 → GT 평가셋 추론 → 예측 9 keypoint
  → 각 필터 적용 (통과/탈락)
  → 예측이 실제 good 인지 GT 대비 판정 (order-free 비교, reproj 거리 임계)
  → 필터별 Precision / Recall / F1 / 통과 PL 수
```

## 설정

- **모델**: 1차 `dope_cropaug_ft_s2` (검출 풍부 → P/R 통계 두텁게). 논문 최종은 `paper_base` 재확인.
- **GT 평가셋**: `outside_combined`(129) + `night_combined`(90) + 합성 val.
- **good 기준**: 예측-GT order-free 2D reproj 평균 < threshold (px).

## 필터 후보

| ID | 필터 | 비고 |
|----|------|------|
| baseline | no filter / confidence only | 대조군 |
| diag | 공간 대각선 교점 ≈ centroid(8) | projective invariant ★ |
| topbot | {0,1,4,5} 위 / {2,3,6,7} 아래 순서 | |
| ratio | 변 비율 (0-1≈4-5, 0-4≈1-5) | perspective 보정 |
| fullkp | 9 keypoint 전부 검출 시에만 | strict pre-filter |
| combo | 위 AND 결합 + per-domain adaptive | 서베이 권장 best |
| (논문발) | conf × geometry, μ+σ adaptive threshold | survey 참조 |

- 정확한 인덱스/불변량/임계값은 `3d-expert` 설계 (`../../method/step2_geometric_filter.md`).

## 선정 규칙 (4월 교훈 반영)

- 통과 PL 수 ≥ 최소치 충족 후 **precision 우선** 랭킹 (recall 후함 금지).
- P/R 1등 ↔ Stage 2 downstream 향상 상관 검증 (P/R proxy 빗나감 방지).

## 산출물 (예정)

- `data/.../filter_pr_camfacing/` summary CSV/JSON + P/R scatter.
- 상위 필터 → Step 3 downstream (R1/R2) 검증.

## 상태

- [x] 2D 기하 필터 구현 (3d-expert, `scripts/data_prep/eval/filter_pr_camfacing.py`)
- [x] dope_cropaug_ft_s2(ep180) 추론 → P/R (outside 129 + night 90 = 219)
- [x] 상위 필터 선정 → diag (PnP-free) + fullkp pre-gate

## 결과 (2026-06-04, 3d-expert)

스크립트: `scripts/data_prep/eval/filter_pr_camfacing.py`
산출물: `data/pallet/eval_results/filter_pr_camfacing/{summary,per_frame}_s2.json`

평가셋 219 프레임 중 검출 가능(>=6kp) pool = **115** (나머지 104 = 심한 occlusion/truncation,
0-5 kp → 자동 not-good). **good = order-free Hungarian mean reproj < 10px vs GT projected_cuboid**.
detectable pool 의 base rate(good 비율) = **0.530** (61/115).

| filter | type | pass | P | R | F1 | gross(>20px) reject |
|--------|------|------|------|------|------|------|
| none | 대조군 | 115 | 0.530 | 1.000 | 0.693 | 0/5 |
| conf>0.5 | 대조군 | 111 | 0.532 | 0.967 | 0.686 | — |
| **ransac** (c≥6) | PnP | 22 | 0.500 | 0.180 | 0.265 | reject |
| **ransac_loo** | PnP | 5 | 0.800 | 0.066 | 0.121 | **5/5** |
| **cf_strict** (B∧C∧D) | PnP | 6 | 0.667 | 0.066 | 0.119 | reject |
| **diag** ★ | 2D 기하 | 37 | 0.568 | 0.344 | 0.429 | **5/5** |
| topbot | 2D 기하 | 180 | 0.339 | 1.000 | 0.506 | 0/5 (무의미) |
| ratio | 2D 기하 | 114 | 0.272 | 0.508 | 0.354 | 2/5 |
| fullkp(9 검출) | 2D 기하 | 66 | 0.561 | 0.607 | 0.583 | 3/5 |
| combo (diag∧topbot∧ratio∧8kp) | 2D 기하 | 11 | 0.727 | 0.131 | 0.222 | 5/5 |

(P/R 은 detectable pool 기준; 전체 219 기준 표는 summary_s2.json overall.)

### 핵심 해석 — P/R 표만 보면 오독함

base rate 가 이미 0.53 라 어떤 필터도 precision 을 극적으로 못 올린다. **이유는 필터
실패가 아니라 good/bad 경계(10px)가 구조적 오류가 아닌 upscale jitter(448→640) 로
채워져 있어서**다. mean_match median = 9.9px 로 threshold 바로 위. bucket 분석:

| bucket | n | diag pass | combo | ransac_loo |
|--------|---|-----------|-------|-----------|
| good <10px | 61 | 34% | 13% | 7% |
| borderline 10-20px | 49 | 33% | 6% | 2% |
| **gross >20px** | 5 | **0%** | **0%** | **0%** |

→ diag/combo/ransac_loo 는 **gross 구조 오류(flip/collapse)를 100% 제거**한다(필터의
진짜 임무). 단 good↔borderline(둘 다 구조적으로 valid, jitter 차이)은 기하로 분리 불가.
self-training 에서 중요한 건 catastrophic PL 제거 → 이 목적엔 diag 가 정확히 작동.

### 선정: **diag** (primary) + **fullkp** (volume pre-gate)

- **diag** = 공간 대각선(0-6,1-7,2-4,3-5) 교점 ≈ centroid(8), norm by diag len, τ=0.05.
  **PnP 불필요 → 비율 unknown 처음 본 파렛트에도 적용**(본 연구 필터 contribution).
  GT 자체 검증 median 1.8-2.7% of diagonal. gross 5/5 reject, pass 37(≥30 충족).
- **ransac_loo / combo** = 고정밀(P 0.73-0.80) 저물량(pass 5-11). volume 부족해 R2 학습엔
  부적합. ablation 표의 "precision 상한" 레퍼런스로 보존.
- **topbot** 단독 무의미(거의 항상 통과), **ratio** 는 precision 떨어뜨림 → AND 결합에서 제외.
- 기존 ransac(c≥6) 단독은 P 0.50 으로 base rate 이하 = 폐기 타당(4월 v8 결론 재확인,
  단 이번엔 camera-facing canonical 순서 + SQPnP + 올바른 dims 로 정합).

---

## Held-out 재평가 (2026-06-04, 3d-expert) — 누수 교정

### 누수 발견
위 ft_s2 결과는 **train-set 평가(누수)**. 평가모델 `dope_cropaug_ft_s2`의 학습데이터
(capturepallet/night/forklift manual GT)가 평가 GT(outside_combined 129 + night_combined 90)와
동일 → base rate 0.53·P/R 전부 낙관적. held-out 모델로 재평가하여 diag 선정의 일반성 검증.

### 설정 (held-out)
- **평가모델 = `weights/dope/dope_cropaug_pretrain/final_net_epoch_0060.pth`** (final, ep60).
  학습데이터 = `mixed_v8_train`(합성) + `truncation_crops_dope/pretrain`(truncation). **manual GT 미포함 = held-out** (header.txt 확인).
- **GT pool 확대 = 251**: outside_combined(129) + night_combined(90) + **forklift gt_manual 32 추가**.
  forklift도 object-frame canonical convention 확인됨(HEIGHT-edge shortest 32/32) → 포함.
  스크립트에 `--include_forklift` 인자 추가, forklift는 `rgb/` 서브디렉 이미지 경로 처리.
- 동일 필터 후보 9종, 동일 order-free Hungarian good 판정(10px).
- 산출물: `data/pallet/eval_results/filter_pr_camfacing/{summary,per_frame}_heldout_pretrain.json`

### 검출 빈약 — 핵심 발견
| | ft_s2 (누수) | pretrain (held-out) |
|---|---|---|
| 전체 프레임 | 219 | 251 (forklift +32) |
| detectable (≥6 kp) | 115 | 119 (비슷) |
| **good (<10px)** | **61** | **14** |
| **base rate (detectable)** | **0.530** | **0.118** |
| mean_match median (det) | 9.9px | **16.4px** |
| gross >20px bucket | 5 | **43** |

held-out 모델은 키포인트를 **검출은 비슷하게 하지만(119) 정확도가 무너진다**(median 9.9→16.4px,
gross 5→43). 도메인별: outside good 9/51(0.18), **night good 1/42(0.02, 거의 사망)**, forklift 4/26(0.15).
→ **일반화(합성+trunc only) 모델의 real 검출 한계가 정량 확인됨**. 논문에 그대로 보고 가치 있음
(self-training 이 필요한 이유 = pretrain real 성능 빈약).

### Held-out 필터 P/R (detectable pool n=119, base rate 0.118)
| filter | pass | P | R | F1 | gross(>20px) reject |
|--------|------|------|------|------|------|
| none | 119 | 0.118 | 1.000 | 0.211 | 0/43 (0%) |
| conf>0.5 | 92 | 0.130 | 0.857 | 0.226 | 16/43 (37%) |
| ransac (c≥6) | 47 | 0.149 | 0.500 | 0.230 | 30/43 (70%) |
| ransac_loo | 8 | 0.000 | 0.000 | 0.000 | 41/43 (95%) |
| cf_strict | 1 | 0.000 | 0.000 | 0.000 | 43/43 (100%) |
| **diag** ★ | 40 | 0.150 | 0.429 | 0.222 | **31/43 (72%)** |
| topbot | 115 | 0.122 | 1.000 | 0.217 | 4/43 (9%, 무의미) |
| ratio | 58 | 0.121 | 0.500 | 0.194 | 22/43 (51%) |
| fullkp | 65 | 0.123 | 0.571 | 0.203 | 19/43 (44%) |
| combo | 10 | 0.100 | 0.071 | 0.083 | 40/43 (93%) |

### Bucket 분석 (gross-reject = 필터 진짜 임무)
bucket: good 14 / borderline(10-20px) 62 / gross(>20px) 43

| filter | good_pass | border_pass | **gross reject%** | catastrophic(>40px, n=15) pass |
|--------|-----------|-------------|----------|----------|
| diag | 6/14 | 22/62 | **72%** | 4/15 |
| combo | 1/14 | 6/62 | **93%** | 1/15 |
| ransac_loo | 0/14 | 6/62 | **95%** | 1/15 |
| cf_strict | 0/14 | 1/62 | **100%** | 0/15 |
| ratio | 7/14 | 30/62 | 51% | — |

### 결론 — diag held-out 검증
1. **diag 선정 방향성 유지(상대 우위 보존), 단 절대 강도는 약화.** ft_s2에서 diag는 gross 5/5(100%) 제거
   였으나 held-out gross 43개 표본에서 **72%(31/43)** 제거. catastrophic >40px 15개 중 4개 통과 — 누수표본(5개)
   에선 안 보이던 한계. ransac_loo(95%)/cf_strict(100%)/combo(93%)가 gross 제거는 더 강하나 **pass 1-8개·P=0
   = self-training 물량 사망**. PnP-free·비율 unknown 적용 가능성까지 종합하면 diag가 여전히 "volume vs gross-reject"
   최선 trade-off → **diag primary 유지 타당**. 단 "gross 100% 제거" 주장은 누수 산물이므로 철회, "gross 다수(~72%) 제거"로 정정.
2. **base rate 0.118** — held-out에선 어떤 필터도 P를 0.13-0.15 이상 못 올림(좋은 PL 자체가 14개뿐).
   P/R 표는 의미 약하고 **gross-reject%가 유일하게 정보성 있는 지표**(ft_s2 교훈 재확인, 표본 5→43으로 신뢰도↑).
3. **gross 오류 제거 능력 유지(diag 72%)** 하나, real 검출 자체가 빈약(good 14)해 단독 pretrain으론 PL pool 부족
   → self-training 1라운드 후 모델 개선 → 재필터 사이클 필요성 정량 근거 확보.
4. **paper_base 재확인 필요**: 본 held-out도 cropaug 계열. 논문 최종 보고는 paper_base 모델로
   동일 251-pool·diag 재검증 권장(검출 더 빈약할 수 있음 = 그 자체로 보고 대상).

> 주의: diag의 "gross 100%" 같은 absolute 수치는 누수 5-표본 산물. held-out 43-표본 기준으로 재서술할 것.

---

## 도메인별 분석 (2026-06-04, 3d-expert)

> 목적 = 절대 성능 X. **각 필터가 도메인(indoor/outside/night)별로 어떤 실패를 거르는지** 패턴.
> 스크립트: `scripts/data_prep/eval/filter_domain_analysis.py` (+ `filter_domain_overlay.py`).
> 산출물: `data/pallet/eval_results/filter_domain_analysis/{summary,per_frame}_{s2,pretrain}.json`,
> overlay `overlays_s2/`.

### 설정
- **도메인 3개**: indoor=`capture0403middle/gt_final`(440, AprilTag GT, **ft_s2 held-out**) /
  outside=`outside_combined`(129) / night=`night_combined`(90). outside·night은 ft_s2 학습데이터 = **누수**
  (절대성능 아닌 필터 패턴 목적이므로 OK).
- **convention**: indoor도 object-frame canonical 확정 (HEIGHT-edge 최단 **440/440**). dimensions_m은
  per-frame (indoor W=1.1/D=1.3, outside·night W=1.3/D=1.1 swap) — good 판정이 order-free Hungarian이라 자동 흡수.
- **필터(사용자 재정의)**: fullkp(9검출) / diag(공간대각선 교점≈centroid) /
  **ratio(가로변4 {0-1,3-2,4-5,7-6} 일관 AND 세로변4 {0-4,1-5,2-6,3-7} 일관)** /
  ransac_loo(RANSAC+LOO) / **combo = fullkp∧diag∧ratio∧ransac_loo (4-way AND)**.
- 평가모델 = `dope_cropaug_ft_s2`(검출 풍부). held-out 비교용으로 `dope_cropaug_pretrain`도 동일 실행.

### 도메인 × 필터 통과율 (ft_s2)
good = order-free Hungarian mean reproj < 10px. `/total`=전체분모, `/det`=detectable(≥6kp)분모.

| 도메인 (total / detect / good / gross>20px) | filter | pass | /total | /det | good_of_pass | gross_rej% |
|---|---|---|---|---|---|---|
| **indoor** (440 / 87 / 17 / 22) base=0.195 | fullkp | 32 | 7% | 37% | 1/32 | 73% |
| | diag | 7 | 2% | 8% | 0/7 | **95%** |
| | ratio | 234 | 53% | 269%† | 16/234 | 59% |
| | ransac_loo | 0 | 0% | 0% | 0/0 | 100% |
| | **combo** | **0** | 0% | 0% | 0/0 | 100% |
| **outside** (129 / 64 / 31 / 4) base=0.484 | fullkp | 37 | 29% | 58% | 20/37 | 75% |
| | diag | 27 | 21% | 42% | 15/27 | **100%** |
| | ratio | 59 | 46% | 92% | 11/59 | 25% |
| | ransac_loo | 3 | 2% | 5% | 3/3 | 100% |
| | **combo** | **2** | 2% | 3% | 2/2 | 100% |
| **night** (90 / 51 / 30 / 1) base=0.588 | fullkp | 29 | 32% | 57% | 17/29 | 0%‡ |
| | diag | 10 | 11% | 20% | 6/10 | **100%** |
| | ratio | 55 | 61% | 108%† | 20/55 | 100% |
| | ransac_loo | 2 | 2% | 4% | 1/2 | 100% |
| | **combo** | **1** | 1% | 2% | 1/1 | 100% |

† ratio pass가 detectable 수를 초과 = **ratio는 검출 게이트가 없어** 부분검출(2-5kp, 한 그룹 변 2개만 있어도)도 통과.
  indoor 저검출 353프레임 중 185개가 ratio 통과 → ratio는 단독으로 거의 무의미(약필터).
‡ night gross 표본 1개 → gross_rej% 신뢰 불가.

### 도메인별 필터 특성 해석

- **indoor (held-out, 최난도)**: 440 중 353(80%)이 **<6kp** (n_det 히스토그램 4kp=174, 2kp=88로 몰림).
  ft_s2가 indoor 미학습 → **검출 단계에서 이미 붕괴**. fullkp 통과 32(7%)뿐. diag_score는 good 0.053 vs
  gross 0.196으로 명확히 분리되어 **gross 95% 제거**. 그러나 ransac_loo/combo는 **통과 0** — flat 파렛트 +
  부분검출 + indoor 시점에서 PnP-LOO가 전혀 안정 못 함(ransac_loo의 구조적 약점이 도메인 중 가장 극단).
- **outside (누수, 중간)**: 검출 양호(detectable 64, fullkp 37). diag가 **gross 4/4 전부 제거(100%)**하면서
  pass 27로 물량도 확보 = outside에서 diag가 가장 균형. ratio는 gross_rej 25%로 거의 못 거름(약필터 재확인).
- **night (누수)**: 저대비라 **0kp/2kp 프레임 다수**(detectable 51)이나, 검출된 건 의외로 정확(base 0.588 최고,
  night_good overlay reproj 7px 청정). diag_score가 good 0.065 vs gross 0.074로 **분리 실패**(gross 1개라 통계
  무의미). 대신 ratio_score가 good 0.175 vs gross 1.136으로 night에선 ratio가 분리력 있음(단 gross 1개 한계).
  night의 주 실패는 "검출 자체 누락"이지 "구조 오류"가 아님 → 구조필터로 거를 대상이 적음.

→ **도메인 민감도 요약**: indoor=검출 붕괴(필터 이전 단계 실패), outside=구조 오류를 diag가 잘 거름,
  night=검출 누락 위주(구조 필터 거를 대상 적음). **diag만 세 도메인에서 일관되게 gross를 다수 제거**
  (indoor 95 / outside 100 / night 100%, night은 표본주의). ratio는 검출 게이트 없어 단독 약함.

### combo (4-way AND) — 도메인별 통과량
combo 통과 = **indoor 0 / outside 2 / night 1**. 사실상 ransac_loo의 통과량(0/3/2)에 묶임 =
**ransac_loo가 combo의 병목**. held-out 누수교정(위 섹션)에서 본 "ransac_loo·combo는 물량 사망"이
**도메인별로도 재현**, 특히 held-out 도메인 indoor에서 정확히 0. → combo는 P는 높아도(통과 전부 good에 가까움)
self-training 물량으로 부적합, primary는 diag.

### held-out (dope_cropaug_pretrain) 도메인 통과율 — 누수 없는 확인
세 도메인 모두 검출 붕괴(good: indoor 2 / outside 9 / night 1). gross 폭증(56/24/19).
diag gross_rej = indoor 89 / outside 67 / night 79%. combo 통과 = 0/2/1 (ft_s2와 동일 패턴, 더 빈약).
→ **합성+trunc only 모델은 모든 real 도메인에서 검출 한계**, self-training 1라운드 후 재필터 필요 정량 재확인.

### 대표 overlay (`overlays_s2/`)
- `outside_good_0_*.jpg` — 5필터 전부 PASS, reproj 9.8px (정렬 양호 = combo가 잡는 이상적 케이스).
- `night_good_0_*.jpg` — 저조도에도 5필터 PASS, reproj 7px (night 검출은 되면 정확).
- `indoor_caught_0_*.jpg` — reproj 23px collapse(6kp), diag·ratio·combo가 **정확히 거름**(GT녹색 vs 붕괴pred황색).
- `indoor_missed_0_*.jpg` — reproj 20px이나 **diag·fullkp는 PASS**(near-symmetric skew → 대각선 교점은 centroid 유지,
  scale만 틀림). ratio·ransac_loo·combo는 fail. = **diag 단독의 한계 케이스**(대칭 왜곡은 못 거름, ratio/loo 보완).

### 결론 — 어느 필터가 어느 도메인에 강/약
- **diag = 전 도메인 공통 강함**(gross 다수 제거, PnP-free라 indoor held-out에서도 작동). night는 표본부족 주의.
- **ratio = 약함**(검출 게이트 없어 부분검출 남발). 단 night에선 score 분리력 있음 → AND 보조로만.
- **ransac_loo / combo = 도메인 불문 물량 사망**, indoor(held-out·flat·부분검출)에서 통과 0으로 가장 극단.
- **fullkp = 검출 풍부 도메인(outside/night)에선 good_of_pass 높지만**(volume pre-gate로 유용),
  indoor처럼 검출 붕괴 도메인에선 통과 자체가 적어 pre-gate 효과 제한적.
- 종합: **primary diag + fullkp pre-gate** 유지가 도메인 robust. combo는 ablation 상한 레퍼런스로만 보존.

---

# PL-GT 차이 실험 (2026-06-04, 3d-expert)

> 직전 도메인별 필터 분석(filter_domain_analysis)의 후속. **필터가 통과시킨
> pseudo-label(PL)이 실제 GT와 얼마나 차이 나는지**를 4 실험으로 정량/정성 비교.
> 학습 불필요 — `_full_s2.json`(예측 9kp + GT projected_cuboid 8 corner + 도메인별
> 필터 통과 + order-free Hungarian reproj)을 재활용한 순수 후처리.
> 스크립트: `scripts/data_prep/eval/pl_gt_diff_analysis.py`
> 산출물: `data/pallet/eval_results/pl_gt_diff/`

## 설정
- 모델: `dope_cropaug_ft_s2` (직전과 동일, 일관성. indoor held-out, outside/night 누수 인지 — 목적은 필터별 상대 비교).
- 오차 = PL 예측 8 corner ↔ GT projected_cuboid 8 corner의 **order-free Hungarian mean reproj(px)**. dims/convention/W-D swap 흡수.
- per-keypoint(실험3)는 Hungarian **assignment 거리**를 예측 corner slot(0..7)별로, centroid(8)는 예측 vs GT cuboid 중심 거리.
- detectable = ≥6 corner(유한 reproj). `ratio`는 검출 게이트가 없어 <6kp 프레임도 통과 → 오차 산출 불가분만 제외(실험1·3은 detectable 통과만).

## 실험1 — 통과 PL의 GT reproj 오차 분포
그림: `exp1_passed_error_dist.png` (도메인 3패널 box+strip, gross 20px/good 10px 가이드선).

| domain | filter | n | median(px) | IQR(q1–q3) | gross%(>20px) |
|--------|--------|---|-----------|-----------|---------------|
| indoor | fullkp | 32 | 13.9 | 11.8–16.9 | 19% |
| indoor | diag | 7 | 12.2 | 11.5–15.3 | 14% |
| indoor | ratio | 49 | 12.6 | 8.6–14.5 | 18% |
| indoor | ransac_loo/combo | 0 | — 통과 0 (검출 붕괴) | | |
| outside | fullkp | 37 | 9.9 | 8.9–12.9 | 3% |
| outside | diag | 27 | 9.9 | 9.0–11.6 | **0%** |
| outside | ratio | 23 | 10.1 | 8.9–11.7 | 13% |
| outside | ransac_loo | 3 | 9.3 | 9.2–9.6 | 0% |
| outside | combo | 2 | 9.6 | 9.4–9.7 | 0% |
| night | fullkp | 29 | 9.2 | 7.4–13.1 | 3% |
| night | diag | 10 | 8.3 | 7.4–13.3 | **0%** |
| night | ratio | 32 | 8.7 | 7.2–11.2 | 0% |
| night | ransac_loo | 2 | 8.9 | 8.0–9.8 | 0% |
| night | combo | 1 | 7.0 | — | 0% |

- **indoor 통과 PL 오차가 전 필터 12–14px로 가장 나쁨** (held-out + flat top-down + 검출 붕괴). outside/night는 8–10px.
- **diag/combo/ransac_loo는 outside·night에서 gross%=0** — gross(>20px) PL을 한 장도 통과 안 시킴. ratio는 outside 13%·indoor 18%로 gross 누수.
- ratio가 indoor에서 통과 49장으로 가장 많지만(검출 게이트 없음) 56px·53px 같은 극단 outlier 포함.

## 실험2 — 통과 vs 탈락 PL의 GT 오차 분리도
그림: `exp2_pass_vs_reject_separability.png`. 지표: Δmed = median(탈락) − median(통과), AUC = P(탈락 오차 > 통과 오차).

| domain | filter | Δmed(px) | AUC |
|--------|--------|---------|-----|
| indoor | ratio | **+2.6** | **0.67** |
| indoor | diag | +1.3 | 0.53 |
| indoor | fullkp | −0.8 | 0.45 |
| outside | ransac_loo | +0.9 | 0.64 |
| outside | diag | +0.4 | 0.54 |
| night | combo | +1.9 | **0.78** |
| night | ratio | +1.2 | 0.60 |

- **분리도가 전반적으로 약함(Δmed ≤ 3px, AUC 0.45–0.78).** 직전 P/R 스크리닝 교훈과 일치 — good/bad 경계가 upscale jitter(~10px≈threshold)라 통과·탈락 오차 분포가 크게 겹친다. 필터의 가치는 "통과 중앙값을 크게 낮추기"가 아니라 **catastrophic(gross) PL 제거**(실험1 gross%)에 있다.
- fullkp는 indoor에서 Δmed 음수(AUC<0.5) — 검출만 충족하면 통과시켜 오차 큰 PL도 함께 통과. **fullkp 단독은 품질 필터가 아니라 pre-gate**임을 재확인.

## 실험3 — per-keypoint GT 오차 (앞/뒤/centroid)
그림: `exp3_per_keypoint_heatmap.png`(13행×9열 히트맵), `exp3b_front_back_centroid_bars.png`.

| domain | filter | front(0-3) | back(4-7) | ctr(8) |
|--------|--------|-----------|-----------|--------|
| indoor | fullkp | 6.9 | **19.6** | 16.6 |
| indoor | diag | 6.9 | **18.1** | 18.1 |
| indoor | ratio | 7.6 | 10.1 | **20.0** |
| outside | fullkp | 7.6 | 11.2 | 8.2 |
| outside | diag | 8.0 | 10.8 | 8.1 |
| night | ratio | 8.2 | 7.6 | 7.4 |
| night | diag | 7.7 | 9.0 | 8.4 |

- **오차는 앞면(0-3)이 아니라 뒷면(4-7)에 몰린다.** 히트맵에서 indoor c5/c6(back) = 30px vs front c0 = 5px. depth 방향 ambiguity + top-down에서 뒷면이 가려져 belief가 부정확.
- indoor에서 **centroid(8)도 16–20px로 나쁨** — diag 필터가 "대각선 교점≈centroid"를 보지만 indoor 통과 PL의 centroid 자체가 어긋나 있음(그럼에도 gross는 거름).
- outside/night는 front/back 격차가 작음(7–13px) — 측면뷰라 뒷면이 덜 가려짐.

## 실험4 — PL↔GT 겹침 overlay
그림: `exp4_overlay_contact_sheet.png` + 개별 `exp4_overlays/{dom}_{best|worst}_*.png`.
GT=magenta, PL/pred=cyan(centroid=★). 도메인별 best 2 + worst 2(통과 샘플 중) 대비.

- **good 통과**: outside/night 노점·고깔 배경의 측면뷰에서 cyan과 magenta cuboid가 거의 일치(err 5–9px).
- **misleading 통과**: indoor err=51px 케이스 — `ratio` 통과했지만 PL cuboid가 얇은 sliver로 scale-skew 붕괴(near-symmetric skew, 직전 메모리의 diag 한계와 동일 유형). night err=20px — 뒷면이 depth-flip 되어 통과.

## 결론
1. **GT에 가장 가까운 통과 PL = diag** (outside/night median 8.3–9.9px, **gross%=0**). combo/ransac_loo도 깨끗하나 통과 물량이 2–3장으로 self-training 무의미.
2. **필터는 "통과 중앙값을 낮추기"보다 "gross PL 제거"로 일한다.** Δmed·AUC 분리도는 약함(경계가 jitter라 겹침). 실험1의 gross% / 실험4의 misleading 케이스가 진짜 평가 축.
3. **오차는 뒷면(4-7)+centroid에 집중** (특히 indoor top-down). PL 학습 시 뒷면 corner 신뢰도를 낮추거나 front-weighted loss를 고려할 근거.
4. **ratio는 통과 물량은 많지만 gross/scale-skew 누수**(indoor 18%) → 단독 사용 금지, diag와 AND 보조로만. 직전 selection(primary diag + fullkp pre-gate)과 정합.

---

## 조합별 전체 9kp 평균오차 (2026-06-04, 3d-expert) — 최종 선정 기준

> ⚠️ 선정 기준 정정: 직전 exp3은 "뒷면(4-7) 오차"로 조합을 골랐으나, 올바른 기준은
> **전체 9 키포인트(8 corner + centroid) 평균 order-free 오차**다. 앞/뒤/centroid는 참고 컬럼.
> 스크립트: `scripts/data_prep/eval/filter_combo_9kp.py` (+ `_overlay.py`).
> 산출물: `data/pallet/eval_results/filter_combo_9kp/combo_9kp_s2.{json,txt}`, `overlays/`.
> 모델 `dope_cropaug_ft_s2`, inference-free(`_full_s2.json` 재활용). `_exclude.txt` 1프레임 제외.

### 9kp 오차 정의
- 8 corner: pred↔GT projected_cuboid **order-free Hungarian** 매칭 거리.
- centroid(idx8): pred centroid ↔ GT 8 corner 평균(중심) 거리.
- **9kp_err = 그 9개(가용분) 평균 px**. mean은 outlier skew → **median으로 선정**.
- 통과량 viability: indoor/outside ≥20, night ≥8, ALL ≥30.

### 조합 × [N / 9kp_med / 9kp_mean / good%(<10px) / gross(>20px) / (참고)front·back·ctr]

```
SCOPE=outside  total=129  detectable=64  good_overall_9kp=31
combo                  N  9kp_med 9kp_mn good% gross | front  back  ctr
diag                  27    9.9    10.5   52%    0      7.3  11.3   8.1   <- BEST viable
fullkp                37   10.0    11.2   49%    2      7.4  11.8   8.2
topbot                63   10.0    11.3   43%    4      7.5  11.8   9.8
ratio                 22   10.3    11.3   46%    2      9.2  10.2  12.7
diag+ransac_loo        2    9.1     9.1  100%    0  *low(2)
ransac_loo             3    9.3     9.2  100%    0  *low(3)
diag+ratio             3    9.4     9.7   67%    0  *low(3)
(diag+fullkp, diag+topbot, diag+fullkp+topbot == diag: 27, 동일)

SCOPE=night  total=90  detectable=51  good_overall_9kp=31
combo                  N  9kp_med 9kp_mn good% gross | front  back  ctr
diag+ratio             8    7.9     9.2   75%    0      7.8   9.0   7.9   <- BEST viable
ratio+fullkp          11    8.6     9.0   73%    0      7.2   8.6   8.2
diag                  10    8.8    10.5   60%    0      8.7   9.0   8.4
ratio                 32    8.8     9.6   66%    0      8.6   8.2   7.4
topbot                51    9.0    10.1   61%    0      8.8   8.3   8.6
fullkp                29    9.1    10.4   59%    0      8.5   9.5   9.0
diag+ransac_loo        1    7.2     7.2  100%    0  *low(1)
ransac_loo             2    8.4     8.4  100%    0  *low(2)

SCOPE=indoor  total=440  detectable=87  good_overall_9kp=10  (held-out, 최난도)
combo                  N  9kp_med 9kp_mn good% gross | front  back  ctr
ratio                 49   13.7    19.3   20%    8      7.5  12.0  20.0   <- BEST viable
fullkp                32   14.1    15.6    0%    5      7.1  20.8  16.6
topbot                86   14.1    22.7   12%   19      7.4  15.9  18.9
diag                   7   13.2    14.4    0%    1  *low(7)  7.6  16.4  18.1
ransac_loo / combo+loo:  통과 0 (검출 붕괴 + flat PnP 불안정)

SCOPE=ALL  total=658  detectable=201  good_overall_9kp=68
combo                  N  9kp_med 9kp_mn good% gross | front  back  ctr
diag                  44   10.0    11.1   46%    1      7.6  11.9   9.0   <- BEST viable
ratio                103   11.1    14.6   40%   10      7.9  10.2  15.0
fullkp                98   11.5    12.4   36%    7      7.4  14.2  11.6
topbot               200   11.5    15.9   34%   23      7.5  12.1  14.1
diag+ratio            13    9.0    10.3   62%    0  *low(13) 8.4   9.5   8.6
diag+ransac_loo        3    8.8     8.5  100%    0  *low(3)
```
(`*low` = viability 미달 통과량. ransac_loo 계열은 9kp_med는 최저(8.5~9.3)이나 통과 1~5장 → self-training 불가.)

### 최적 조합 (전체 9kp 평균오차 기준) — 도메인별로 다름

| scope | **최적(viable)** | N | 9kp_med | good% | gross | 근거 |
|-------|------------------|---|---------|-------|-------|------|
| outside | **diag** | 27 | **9.9** | 52% | 0 | 9kp_med 최저(viable) + gross 0. fullkp/topbot은 N↑이나 gross 누수. |
| night | **diag∧ratio** | 8 | **7.9** | 75% | 0 | 단독 diag(8.8)·ratio(8.8)보다 AND가 9kp_med 0.9px↓ & good 75%. N=8로 viability 충족. 물량 더 필요시 ratio∧fullkp(11장, 8.6). |
| indoor | **ratio** | 49 | **13.7** | 20% | 8 | 검출 붕괴라 diag는 N=7(미달). ratio가 유일하게 viable N 확보. 단 절대오차 13.7px·gross 8 = indoor PL은 품질 낮음(self-train 신중). |
| ALL | **diag** | 44 | **10.0** | 46% | 1 | 전 도메인 통합 시 diag가 viable 중 9kp_med 최저 + gross 거의 0(1장). |

### 핵심 결론
1. **diag가 전체 9kp 평균오차 기준에서도 일관 최적** (outside 9.9 / ALL 10.0, gross≈0). 직전 P/R·도메인 분석의 diag 선정이 9kp 기준으로도 재확인됨.
2. **night만 diag∧ratio(7.9px, good75%)가 diag 단독(8.8)을 능가** — night는 검출되면 정확해 ratio AND가 잔여 scale 오차를 추가로 거름. night PL은 이 조합 권장.
3. **ransac_loo 계열은 9kp_med 최저(7~9px)지만 통과 1~5장** → 품질 상한 레퍼런스일 뿐 self-training 물량 부적합(직전 결론 유지).
4. **diag에 fullkp/topbot AND는 outside/ALL에서 통과집합·오차가 diag와 동일**(diag⊂fullkp,topbot 관계) → diag 단독으로 충분, 추가 AND 불필요.
5. **indoor는 어떤 viable 조합도 9kp_med 13px+ / good ≤20%** — held-out·top-down·검출붕괴라 PL 신뢰 낮음. self-train 1라운드 후 재필터 필요(직전 결론 재확인).

### 대표 overlay (`filter_combo_9kp/overlays/`, GT=magenta·PL=cyan·노란선=Hungarian 9kp 매칭)
- `outside_diag_0_6px.jpg` ~ `_2_7px.jpg` — diag 통과, cyan/magenta cuboid 거의 완전 일치(9kp 5.9~7px).
- `night_diag_ratio_0_6px.jpg` ~ — diag∧ratio 통과, 저조도에도 9kp 6~7px 정합.
- `indoor_ratio_0_6px.jpg` ~ — ratio 통과 best 3장(6~8px). 단 통과 전체 median은 13.7px(best만 청정, 나머지 산포 큼).


================================================================================
FILE: experiments/README.md
================================================================================

# Experiments — camera-facing 논문 실험 인덱스

각 실험은 **하나의 Table/Figure 단위** 파일. 양식(빈 틀)을 미리 만들어두고, 실험하면 그 파일을 채운다.

> 2026-06-04 재편: v8(object-frame) 실험은 `archive/` 로 격리. 아래는 **camera-facing 0123 / 논문용(v1/v2 제외, 일반화)** 새 실험들.
> 방향: CLAUDE.md "핵심 방향" + memory 4종. 검증할 주장: `_docs/method/{overview,step1~3,evaluation}.md`.

## 실험 인덱스 (구분 / 논문 / 상태 / 의존)

```
#    파일                                    구분   논문    상태      의존
──────────────────────────────────────────────────────────────────────────────────────
A1   data/A1_paper_base_perf.md              다시   T       미시작    paper_base 학습
A2   data/A2_squash_ratio_ablation.md        새로   T       미시작    paper_base ±squash
A3   data/A3_truncation_padding_ablation.md  새로   T       미시작    paper_base ±padding
B1   filter/pr_screening.md                  다시   T1      ★일부완료 ft_s2/pretrain (paper_base 재확인)
B2   filter/B2_filter_selftraining.md        다시   T       미시작    paper_base, C1
C1   self_training/C1_rounds.md              다시   ★F1     미시작    paper_base, 필터선정
C2   self_training/C2_pl_quality_vs_quantity 다시   T       미시작    C1
D1   eval/D1_generalization_seen_unseen.md   새로   T       미시작    paper_base, real GT
D2   eval/D2_real_test.md                    다시   T       미시작    paper_base/r1, real GT
D3   eval/D3_pnp_solver.md                   새로   T       부분      real GT (challenge 검증됨)
F2   eval/F2_qualitative.md                  다시   F       미시작    위 실험들
T10  related_work.md                         유지   T10     예정      논문 draft
```

★ = 핵심 / 부분·일부완료 = 채울 데이터 일부 있음

## 의존 순서 (실행 경로)

```
[다른 머신] paper_base 학습
      │
      ├─→ A1 base 성능, A2 squash, A3 padding  (데이터/학습 검증)
      ├─→ B1 필터 P/R 재확인 (paper_base)
      │      │
      │      └─→ 필터 선정 (outside diag / night diag∧ratio)
      │             │
      │             └─→ C1 self-training R0→R1→R2 ──→ B2 필터별 downstream
      │                        │                       C2 PL 품질vs수량
      │                        └─→ D1 일반화, D2 real test, D3 PnP, F2 정성
      └─→ (논문 draft) T10 related work
```

## Metric 정책 (camera-facing)

```
필터 P/R 스크리닝   통과 PL의 전체 9kp order-free(Hungarian) 평균오차 — 필터 목적=믿을만한 PL
self-training       도메인별 per-frame 검출(NN<20px) + reproj(9kp)
real 6D (dims known) ADD, 5cm5°, reproj — SQPnP, order-free
주의                evaluate_on_val convention 버그 → order-free PnP 필수 (memory)
                    monocular라 5cm5° 약함 → reproj median이 keypoint 품질 신호
```

## 데이터셋 정책

```
GT 평가셋     outside_combined(129) + night_combined(90) + forklift(32) + capture0403middle(440)
제외          data/_eval_sets/_exclude.txt (1778652125245035520 = bad manual GT)
unseen 정의   논문용은 v1/v2(내 파렛트) 제외 학습 → real 파렛트가 곧 unseen
누수 주의     평가모델이 GT를 학습했는지 확인 (ft_s2=누수 / pretrain·paper_base=held-out)
```

## 관련 폴더
- `_docs/method/` — 검증할 주장 (overview / step1~3 / evaluation)
- `_docs/filter/` — 필터 전용 (pr_screening, survey)
- `_docs/models/paper_base.md` — 논문 base 모델 명세
- `data/pallet/eval_results/` — 평가 결과 원본
- `archive/` — 폐기 v8 실험 (참고용)


================================================================================
FILE: experiments/related_work.md
================================================================================

# T10. Related Work 비교 표

상태: **예정 (Phase 5, 논문 draft)**

## 목적

본 연구의 contribution 차별성을 관련 연구들과 수치로 대비.

## Table 10 (draft)

```
Method                  Sensor     Synth Size    Real Adaptation      Eval Environment         Metric
─────────────────────────────────────────────────────────────────────────────────────────────────────────
DOPE (2018)             RGB        ~60K          ✗                    lab + YCB                 ADD / ADD-S
Knitt et al. (2022)     RGB        ~50K          ✗                    lab, single pallet        2D IoU / custom
Mueller et al. (2023)   RGB        ~30K          ✗                    synthetic + simple real   PCK / reproj
UDA-COPE (2022)         RGB-D      —             ✅ (ST + depth)       category-level NOCS       NOCS mAP
PseudoFlow (2023)       RGB        YCB / LM      ✅ (optical flow ST)  YCB-V / LM-O              ADD-S
Ours (2026)             RGB        ~10K          ✅ (RANSAC filter ST) real warehouse + DR       PnP% / ADD / 5 cm 5°
```

## Contribution 차별화 포인트

1. **Real self-training with a geometric (not flow / depth) filter**
   - UDA-COPE 는 depth 의존, PseudoFlow 는 영상 광류 의존
   - Ours 는 단일 RGB frame + RANSAC subset consensus 만 사용 → 센서 / 데이터
     요구 최소

2. **더 작은 합성 데이터셋으로 동등 또는 더 좋은 real 성능**
   - DOPE / Knitt / Mueller 대비 1/3 ~ 1/5 크기의 합성 데이터
   - Isaac + Blender 혼합으로 domain gap 완화

3. **엄밀한 필터 P/R 비교를 contribution 자체로 승격**
   - 23 필터 후보 GT 기반 비교 ([`filter/selection.md`](./filter/selection.md))
   - canonical filter 의 negative result 도 contribution
   - Related work 에서는 "필터를 고른 근거" 를 이 수준으로 제시한 논문 없음

4. **Industrial domain (plastic pallet in warehouse)**
   - YCB / LM / NOCS 는 tabletop 객체 위주
   - Knitt / Mueller 는 pallet 이지만 lab 환경 / 단순 배경

## 논문 draft 문구

> "Unlike prior pseudo-label self-training methods that rely on depth
> supervision (UDA-COPE) or temporal optical flow (PseudoFlow), our
> pipeline uses only RGB frames and validates pseudo-labels via RANSAC
> subset consensus — a single-frame geometric filter selected via
> a rigorous 23-candidate precision/recall comparison. With a third to
> a fifth of the synthetic data of prior industrial pallet methods
> (~10K images vs 30K–50K), we achieve pose estimation in a real
> warehouse environment, on pallet types unseen during training."

## 확장 검토 대상 (optional)

```
- Self6D (2020)           : RGB self-supervised, no-SSS
- GDR-Net (2021)          : synth-to-real direct
- RKHSPose (2024)         : latest self-supervised 6DoF
- 3DUDA (2024)            : source-free category UDA
```

필요 시 Appendix 로 확장.

## 관련

- Survey: `_docs/survey/survey-6d-pose-estimation.md`
- Implementation contribution: `_docs/method/implementation.md` §13


================================================================================
FILE: experiments/self_training/C1_rounds.md
================================================================================

# C1. Self-training R0→R1→R2 × 도메인  (★논문 핵심 Figure F1 + Table)

> 상태: **미시작** | 의존: paper_base 학습, 필터 선정(diag/diag∧ratio)
> 구분: **다시** (v8 발표 셋업을 camera-facing + paper_base로 재현)

## 목적 (한 줄)
2D 기하 필터로 선별한 신뢰 PL로 self-training 반복 시, **도메인별(indoor/outside/night) 성능이 R0→R1→R2로 향상**되는가.

## 판단 지표
도메인별 **per-frame 검출 정확도(NN<20px) + reproj(9kp)**, R0/R1/R2 곡선.
(발표 교훈: PL 수보다 품질. indoor 소량 PL로 R1↑, outdoor/night 다량인데 R2↓ → 좋은 필터로 재현 검증)

## 설정
- anchor R0 = `paper_base`
- 필터: outside=`diag`, night=`diag∧ratio` (indoor=PL 신뢰 낮음 → 1라운드 후 재필터)
- unlabeled pool: outside 9894 / night 9134 / indoor(noapril) 188 (TBD: camera-facing 재확인)
- GT 평가셋: outside_combined(129)·night_combined(90)·capture0403middle(440) [exclude.txt 반영]
- 학습: train.py finetune, 누적 epoch (memory `dope-finetune-cumulative-epoch`)

## 방법
1. R0(paper_base) → unlabeled 추론 → 필터 → PL 추출
2. PL로 R1 finetune → 도메인별 평가
3. R1 → PL 재추출 → R2 → 평가
4. R0/R1/R2 매트릭스 + 곡선

## 결과 (TBD)
```
도메인     R0      R1      R2      필터
──────────────────────────────────────────
indoor     TBD     TBD     TBD     (재필터)
outside    TBD     TBD     TBD     diag
night      TBD     TBD     TBD     diag∧ratio
```

## 결론 (TBD)

## 산출물 (예정)
- round figure(F1), PL pool 증가표, 도메인 cross 평가


================================================================================
FILE: experiments/self_training/C2_pl_quality_vs_quantity.md
================================================================================

# C2. PL 품질 vs 수량 trade-off  (논문 Table)

> 상태: **미시작** | 의존: C1(rounds)
> 구분: **다시/새로** (발표 교훈을 camera-facing 필터로 정량화)

## 목적 (한 줄)
필터 strict 정도(통과량↓·순도↑)가 self-training 향상에 미치는 영향 — **"PL 수보다 품질"** 가설 검증.

## 판단 지표
필터별(느슨~빡셈: none / diag / diag∧ratio / ransac_loo) **통과 PL 수 vs R1 도메인 향상폭**.
좋은 PL 절대수(통과량×순도)가 향상과 상관되는지.

## 설정
- anchor = paper_base, 도메인별 동일 pool
- 필터 strict 단계: none → diag → diag∧ratio → ransac_loo(고순도 저물량)
- 평가: 각 필터로 R1 학습 후 도메인 검출/reproj

## 방법
1. 필터 strict 단계별 PL 추출(수·순도 기록)
2. 각각 R1 학습 → 향상폭
3. (통과량, 순도, 좋은PL 절대수) vs 향상 산점도

## 결과 (TBD)
```
필터          통과PL수   순도(9kp good%)   R1 향상(pp)
─────────────────────────────────────────────────────
none          많음       낮음              TBD
diag          중간       중간              TBD
diag∧ratio    적음       높음              TBD
ransac_loo    매우적음   매우높음(물량사망) TBD
```

## 결론 (TBD)


================================================================================
FILE: filter/2026-06-02_survey_pseudolabel_filtering.md
================================================================================

# Pseudo-Label Filtering 기법 서베이 (Self-Training 6D Pose)

작성일: 2026-06-02
목적: 팔레트 6D pose self-training의 pseudo-label(PL) filter 개선. 현재 geometric RANSAC subset-consensus 필터(n_iter=50, k=5, τ=5px, c≥6) + size sanity가 unlabeled pool 확대에 따라 noisy PL을 너무 많이 통과시킴. **precision(통과 PL 순도) 향상**이 핵심.

---

## 개요

SSL/self-training PL filtering 문헌은 크게 4계열로 정리된다.

1. **Confidence thresholding** (FixMatch → FlexMatch → Dash → FreeMatch): 고정 임계 → class/시점 adaptive 임계로 진화. 핵심 교훈은 "단일 고정 threshold는 학습 진행에 따라 항상 최적이 아니다".
2. **Uncertainty-aware selection** (UPS, Seq-UPS, deep ensemble, MC-dropout): confidence만으로는 mis-calibration 때문에 noise가 새어 들어가므로, uncertainty를 **AND 게이트**로 추가해 false-positive PL을 제거.
3. **6D pose self-training** (Self6D, ECCV'22 bin-picking, Pseudo-Flow-Consistency): classification confidence가 없는 pose 문제에서는 **render-and-compare(2D appearance) + 3D geometry consistency**를 PL 품질 proxy로 사용. 우리 RANSAC reproj consensus와 직접 대응.
4. **Curriculum / adaptive threshold** (Dash, CPL): 초기엔 strict(high-precision) → 후기엔 relax. 또는 데이터 분포 기반 `μ+σ` 동적 임계.
5. **TTA consistency**: weak/strong aug 또는 multi-view 예측 일치도를 PL 신뢰 척도로.

우리에게 가장 직접적인 건 **(3) 6D pose self-training의 adaptive consensus** + **(2) uncertainty AND 게이트**의 결합이다.

---

## 논문별 요약

| # | 제목 / 저자 / 연도 | 핵심 아이디어 | 우리 필터에 적용할 take-away |
|---|---|---|---|
| 1 | **FixMatch** (Sohn et al., NeurIPS 2020) | weak-aug 예측이 고정 confidence threshold(0.95) 넘으면 그 hard label을 strong-aug에 supervision으로 사용 | 단일 strict cutoff가 baseline. 우리 `c≥6 / τ=5px`가 이에 대응하나, 고정값이라 pool 분포 변화에 둔감 |
| 2 | **FlexMatch / Curriculum Pseudo-Labeling** (Zhang et al., NeurIPS 2021) | class별로 학습 난이도를 추정해 class-specific threshold를 동적 조정. 쉬운 시점/클래스는 임계 ↑, 어려운 건 ↓ | "한 종류의 PL에 한 임계"는 비효율. 우리는 **per-domain(scene/팔레트 종류/시점)** 으로 threshold를 쪼갤 근거 |
| 3 | **Dash** (Xu et al., ICML 2021) | 학습 진행에 따라 global threshold를 **점진적으로 강화(grow)**. 초기엔 loose, 후기엔 strict하게 손실 기준 cutoff 상승 | curriculum 방향의 정량 근거. self-training round가 진행될수록 strict하게 죄는 schedule이 confirmation bias를 줄임 |
| 4 | **FreeMatch** (Wang et al., ICLR 2023) | global + class-local threshold를 unlabeled confidence의 **EMA로 self-adaptive** 조정. class-fairness regularization 추가. FlexMatch 대비 CIFAR-10(1-label) 5.78%↓ | **EMA 기반 분포 추종 threshold**가 hand-tuned 고정값보다 강함. 우리도 reproj-error/consensus 분포의 EMA로 cutoff를 자동 산출 가능 |
| 5 | **UPS: In Defense of Pseudo-Labeling** (Rizve et al., ICLR 2021) | confidence와 **uncertainty를 분리한 이중 게이트**: `p_c > τ AND u(p_c) < κ`. MC-dropout으로 uncertainty 추정. mis-calibrated high-confidence noise 제거. negative learning 도입 | **핵심 차용 포인트**: confidence(=우리 belief peak/PnP inlier)와 uncertainty(=consistency 분산)를 **AND로 결합**. 둘 다 통과해야 PL 채택 → precision 직접 상승 |
| 6 | **Seq-UPS** (Patel et al., 2022) | UPS를 시퀀스(text recognition) 도메인으로 확장. MC-dropout 다중 forward의 분산을 uncertainty로, teacher-forcing으로 sample 간 prediction consistency 강제 | keypoint sequence(8 corner + centroid)에도 forward 분산 기반 uncertainty 적용 가능. dropout 다중 추론으로 keypoint별 분산 측정 |
| 7 | **Self6D** (Wang et al., ECCV 2020) | neural rendering 기반 visual + geometric alignment로 6D pose self-supervision (PL 없는 self-sup) | render-and-compare가 pose 품질의 강한 proxy. 우리 reproj consensus의 상위호환 신호로 silhouette/mask IoU 추가 고려 |
| 8 | **Sim-to-Real 6D Pose via Iterative Self-Training** (Chen et al., ECCV 2022, bin-picking) | **우리와 가장 유사.** teacher가 real unlabeled에 pose 예측 → (a) 2D appearance: mask overlap × perceptual distance, (b) 3D geometry: Chamfer distance, 두 신호를 **AND 게이트**로 PL 선별. 임계는 **분포 기반 `τ = μ + σ`**(고정값 아님). student→new teacher 반복으로 PL 품질·정확도 동반 상승. ADD(-S) +11.49%/+22.62%, bin-picking 성공률 +19.54% | **가장 직접적 청사진**: ① 2D(appearance) + 3D(geometry) **상보적 다중 metric AND 합의**, ② threshold를 unlabeled metric 분포의 `μ+σ`로 **adaptive 산출**, ③ iterative re-labeling. 우리의 reproj-consensus는 geometry축, 여기에 **appearance축(렌더 silhouette IoU / crop perceptual dist)을 추가하면 precision 상승** |
| 9 | **Pseudo Flow Consistency for Self-Sup 6D Pose** (Hai et al., ICCV 2023) | pure RGB. pixel-level **flow consistency**를 학습 이미지 쌍 사이 geometry 제약으로. 동적 PL 생성 | 보조 정보 없이 RGB만으로 geometry consistency 측정. multi-view/연속 프레임이 있다면 flow consistency가 추가 필터 신호 |
| 10 | **Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017) / **MC-Dropout** (Gal & Ghahramani, ICML 2016) | 다중 모델/다중 dropout forward의 예측 분산 = predictive uncertainty. ensemble이 MC-dropout보다 calibration 우수하나 비용 큼 | uncertainty 신호 구현 옵션. 비용이 부담이면 MC-dropout(저비용), 정확도 우선이면 small ensemble. keypoint heatmap의 forward 간 분산으로 PL 신뢰도 정량화 |

---

## 우리 4방향에 대한 시사점

### A. top-K / 백분위 quality 수량 제어
- **근거**: Dash·FreeMatch·CPL 모두 "절대 임계 고정"보다 **분포 기준 상대 선택**이 강함을 보임. 백분위(top-K%) 선택은 pool 크기/난이도가 바뀌어도 통과 PL의 *순도 분포*를 일정하게 유지.
- **시사점**: 현재 `c≥6 / τ=5px` 고정 cutoff는 pool이 커지면 절대 통과 수만 늘 뿐 순도가 떨어진다. **reproj-consensus 점수 상위 K%만 채택**(예: per-round top 30%)하면 noisy tail을 구조적으로 잘라낸다. 단, top-K는 "전부 나쁜 round"에서도 K%를 강제로 뽑는 위험 → **절대 floor(c≥6)와 AND**로 안전판 결합 권장.
- **주의**: FlexMatch 교훈상 K를 **per-domain/시점별로** 잡아야 한 시점이 PL을 독식하지 않음.

### B. consensus / strict threshold 강화
- **근거**: UPS·bin-picking 모두 **단일 metric → 다중 metric AND**로 precision을 끌어올림. FixMatch 0.95처럼 strict cutoff는 recall은 줄지만 self-training에선 precision이 더 중요(confirmation bias).
- **시사점**: 현재 RANSAC consensus(c≥6, τ=5px)를 **τ를 더 죄거나(예 3px)** consensus 비율 기준(8개 중 7개 이상)으로 강화. 다만 단순 강화는 recall 급감 → 차원을 늘리는 **C(결합 게이트)** 가 더 효율적.

### C. confidence × geometry 결합 게이트 ★ (가장 유망)
- **근거**: UPS의 `p_c>τ AND u<κ`, bin-picking의 `d_a<τ_a AND d_g<τ_g` — **서로 다른 실패 모드를 잡는 상보적 신호의 AND 게이트**가 문헌의 일관된 best practice. confidence는 in-plane/texture 오류, geometry는 out-of-plane/scale 오류를 잡음.
- **시사점**: 우리는 geometry축(RANSAC reproj consensus)만 있고 **confidence축이 약하다**. 추가 신호 후보:
  - **belief-map peak sharpness/height** (DOPE heatmap 신뢰도) → confidence축
  - **PnP RANSAC inlier ratio** → geometry confidence
  - **렌더 silhouette IoU / crop perceptual distance** (Self6D·bin-picking식 appearance축)
  - **MC-dropout keypoint 분산** (UPS식 uncertainty축)
  - 게이트: `(geometry consensus 통과) AND (confidence ≥ τ_c) AND (uncertainty ≤ κ)`. 셋 다 통과만 PL 채택.

### D. per-domain adaptive threshold
- **근거**: FlexMatch(class-local), FreeMatch(global+local EMA), bin-picking(`μ+σ` 분포 기반) 모두 단일 global 임계의 한계를 보임.
- **시사점**: 팔레트 종류(scene 1~4)/카메라 시점/배경 난이도별로 reproj-error 분포가 다르므로, **각 domain의 통과 metric 분포에서 `μ+σ` 또는 백분위로 threshold를 자동 산출**. 고정 5px를 모든 도메인에 적용하면 쉬운 도메인은 noise 통과, 어려운 도메인은 PL 고갈. EMA로 round마다 갱신(FreeMatch식).

**결론 우선순위**: C(결합 게이트) ≈ D(adaptive) > A(top-K) > B(단순 강화). C+D를 함께 가는 것이 문헌상 가장 검증된 조합이며 bin-picking 논문이 정확히 그 형태(다중 metric AND + `μ+σ` adaptive + iterative).

---

## 추가 제안 필터 후보 (구체적 구현)

### 후보 1: Dual-Gate Adaptive Consensus Filter (DGAC) — 최우선 추천
bin-picking(ECCV'22) + UPS를 우리 keypoint 파이프라인에 이식. **C + D 동시 충족**.

```
입력: teacher가 unlabeled 프레임 i에 예측한 keypoints K_i, PnP pose T_i, belief maps B_i
도메인 d = domain_of(i)   # 팔레트 종류 / 시점 bin

# --- 축 1: Geometry consensus (기존 RANSAC, 강화) ---
g_i = ransac_reproj_inlier_ratio(K_i, T_i)        # 8 corner 중 inlier 비율 (0~1)

# --- 축 2: Confidence (DOPE belief) ---
c_i = mean_peak_height(B_i)                         # keypoint별 belief 피크 평균

# --- 축 3: Uncertainty (MC-dropout, M회 forward) ---
u_i = mean_keypoint_std(MC_dropout_forward(frame_i, M=5))   # px 단위 분산

# --- 도메인별 adaptive threshold (EMA로 round마다 갱신) ---
τ_g[d] = ema(percentile(g_dist[d], 50))            # 또는 μ_g[d] (상위 절반)
τ_c[d] = ema(μ_c[d] - σ_c[d])
κ_u[d] = ema(μ_u[d] + σ_u[d])

# --- AND 게이트 ---
accept_i = (g_i ≥ max(τ_g[d], 6/8))   # 절대 floor와 AND
           AND (c_i ≥ τ_c[d])
           AND (u_i ≤ κ_u[d])
```
- 핵심: 세 축(geometry/confidence/uncertainty)을 **AND**로, threshold는 **per-domain EMA 분포 기반**.
- 비용: MC-dropout M=5회 추가 forward(unlabeled 1회성). 부담되면 축 3 생략하고 geometry×confidence 2축만으로도 UPS 대비 핵심은 유지.
- 기대: 절대 통과 수 대신 *순도*를 도메인별로 제어 → precision 직접 상승. iterative round마다 τ 강화(Dash식)로 confirmation bias 억제.

### 후보 2: Render-Consistency Re-Ranking Gate (RCR) — 보조/검증용
Self6D·bin-picking의 appearance축을 our geometry축에 **상보적으로 추가**. geometry consensus를 통과한 PL만 대상으로 2차 검증.

```
geometry 통과한 후보 PL에 대해:
  render_i = silhouette_render(pallet_USD, T_i, K_cam)   # 예측 pose로 USD 렌더
  iou_i    = mask_IoU(render_i, segmask_i)               # 관측 마스크와 silhouette IoU
  accept_2 = iou_i ≥ τ_iou[d]      # 도메인별 μ+σ 또는 top-K%
최종 PL = geometry_pass AND accept_2
```
- 이유: reproj consensus는 keypoint 위치만 보지만 **silhouette IoU는 scale/out-of-plane 오류(reproj가 놓치는 모드)** 를 잡음 → bin-picking의 "2D appearance + 3D geometry 상보성" 그대로.
- 비용: 후보당 1회 렌더(이미 USD 모델 보유). geometry 통과분에만 적용하므로 저렴.
- top-K(A방향)를 여기에 결합: IoU 상위 K%만 채택하면 수량 제어까지 동시.

---

## 참고문헌

1. Sohn et al. **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.** NeurIPS 2020. https://arxiv.org/abs/2001.07685
2. Zhang et al. **FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.** NeurIPS 2021. https://arxiv.org/abs/2110.08263
3. Xu et al. **Dash: Semi-Supervised Learning with Dynamic Thresholding.** ICML 2021. https://arxiv.org/abs/2109.00650
4. Wang et al. **FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning.** ICLR 2023. https://arxiv.org/abs/2205.07246
5. Rizve et al. **In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework (UPS).** ICLR 2021. https://arxiv.org/abs/2101.06329 · code: https://github.com/nayeemrizve/ups
6. Patel et al. **Seq-UPS: Sequential Uncertainty-aware Pseudo-label Selection for Semi-Supervised Text Recognition.** 2022. https://arxiv.org/abs/2209.00641
7. Wang et al. **Self6D: Self-Supervised Monocular 6D Object Pose Estimation.** ECCV 2020. https://arxiv.org/abs/2004.06468
8. Chen et al. **Sim-to-Real 6D Object Pose Estimation via Iterative Self-training for Robotic Bin Picking.** ECCV 2022. https://arxiv.org/abs/2204.07049
9. Hai et al. **Pseudo Flow Consistency for Self-Supervised 6D Object Pose Estimation.** ICCV 2023. https://arxiv.org/abs/2308.10016
10. Lakshminarayanan et al. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.** NeurIPS 2017. https://arxiv.org/abs/1612.01474
11. Gal & Ghahramani. **Dropout as a Bayesian Approximation (MC-Dropout).** ICML 2016. https://arxiv.org/abs/1506.02142
12. (survey) **A Review of Pseudo Labeling for Semi-Supervised Learning.** 2024. https://arxiv.org/abs/2408.07221


================================================================================
FILE: filter/README.md
================================================================================

# Filter Research

Self-training pseudo-label 필터 연구 폴더. camera-facing 0123 기반 2D 기하 필터의
설계·P/R 분석·ablation 을 시간순 누적.

> 폐기: v8(object-frame) 시절의 RANSAC c≥6 선정(`archive/2026-04-11_selection.md`,
> `archive/2026-04-11_design_rationale.md`)은 object-frame 기준이라 무효.
> object-frame 에선 코너 순서가 시점마다 달라 직사각형 기하 필터가 안 먹혔음.

## 현행 방향 (camera-facing 2D 기하 필터)

camera-facing 0123 이라 직사각형 cuboid 의 2D 기하 관계가 image 상에서 일관 →
**PnP 없이 2D 만으로** pseudo-label 신뢰도 판정 (처음 본 파렛트, 비율 unknown 대응).

핵심 후보:
- **공간 대각선 교점 ≈ centroid(8)** — projective invariant ★
- {0,1,4,5} 위 / {2,3,6,7} 아래 순서
- 변 비율 (0-1≈4-5, 0-4≈1-5) — perspective 보정
- 9 keypoint 전부 검출 strict, conf×geometry, per-domain adaptive (서베이 권장)

정확한 인덱스/불변량/임계값 설계는 `3d-expert` 위임 예정.

## 문서 목록

| 날짜 | 문서 | 요약 |
|------|------|------|
| 2026-06-02 | [2026-06-02_survey_pseudolabel_filtering.md](2026-06-02_survey_pseudolabel_filtering.md) | pseudo-label filtering 서베이 — conf×geometry AND, 분포기반 adaptive threshold, per-domain best |
| (예정) | `../experiments/filter/pr_screening.md` | 2D 기하 필터 P/R 스크리닝 (학습 불필요) |
| archive | `archive/` | 폐기 v8 필터 (RANSAC selection/rationale) |

## 평가 방법 (학습 불필요)

```
camera-facing 모델(dope_cropaug_ft_s2 등) → GT 평가셋 추론 → 예측 9 keypoint
  → 2D 기하 필터 → GT 대비 good 판정(order-free) → 필터별 P/R
```

## 동기 (발표 교훈)

indoor 는 소량 PL 로 R1 크게↑, outdoor/night 는 다량 PL 인데 ~1%↑·R2 에서↓.
→ **PL 수보다 품질**. 좋은 기하 필터로 신뢰도 높은 PL 을 만드는 것이 핵심.

## 관련

- method: `../method/step2_geometric_filter.md`
- convention: `../preprocessing/keypoint_definition.md`
- 코드(재설계 예정): `scripts/self_training/geometric_filter.py`


================================================================================
FILE: method/evaluation.md
================================================================================

# Evaluation — 메트릭 + PnP 용도 분리 (camera-facing)

> 폐기 v8 평가(formulation/implementation)는 `archive/`.

## PnP 용도 분리 (핵심)

비율 unknown 인 처음 본 파렛트 때문에 PnP 를 무조건 쓸 수 없다. 용도별로 분리:

| 용도 | 치수 필요? | 방법 |
|------|-----------|------|
| A. self-training PL 필터 | ❌ | 2D 기하 (PnP 불필요) → 처음 본 파렛트 가능 |
| B. 정확도 평가 (ADD) | ✅ | 치수 known GT(내 파렛트)에서 **SQPnP** |
| C. 거리(z) 추정 | ✅ | 과제용 challenge(내 파렛트)에서 **SQPnP** |

## PnP solver = SQPnP

- `cv2.SOLVEPNP_SQPNP` + RefineLM. EPnP+RANSAC 대비 reproj median 5.27→3.12px,
  ADD 96.6→90.7mm 개선 (2026-06-02 YOLO 경로 검증).
- 팔레트는 얇은 near-planar 직육면체라 globally optimal SQPnP 가 유리.
- `scripts/self_training/pnp_solver.py` (현재 EPnP+RANSAC) → 평가/거리용 SQPnP 교체 필요.

## 메트릭

- **keypoint**: PCK, reproj (order-free 비교 — convention 차이 흡수)
- **6D (치수 known)**: ADD, 5cm5°
- **필터**: Precision / Recall / F1 (GT 대비 PL good 판정)
- **self-training**: 도메인별 per-frame 검출 정확도 (R0→R1→R2 매트릭스)

## GT 평가셋

- `data/_eval_sets/outside_combined` (129), `night_combined` (90), 합성 val.
- ⚠️ convention 정합(camera-facing vs object-frame) 확인 필요 → order-free 비교로 흡수.

## 주의 (기록된 버그)

- `evaluate_on_val.py` reproj 130px 는 convention 삼중 불일치 (memory
  `evaluate-on-val-convention-bug`). same-index 아닌 **order-free PnP** 로 풀어야
  진짜 reproj(한 자리수)가 나옴. 검출률/PnP success 는 신뢰 가능.


================================================================================
FILE: method/overview.md
================================================================================

# Method Overview — 논문용 일반화 파이프라인 (camera-facing)

> 2026-06-04 재작성. 폐기된 v8(object-frame) 설계는 `archive/overview.md` 참조.
> memory `two-tracks-paper-vs-challenge`, `camera-facing-0123-convention`.

## 문제 정의

처음 보는 파렛트(비율·외형·조명 제각각)의 6D pose(=9 keypoint)를 추정한다.
**내 실제 파렛트(v1/v2)는 학습에 쓰지 않고**, 인터넷 무료 3D 팔레트 모델 기반
합성 데이터만으로 학습해 **일반화**를 달성하는 것이 논문 핵심.

## 두 트랙 (구분 필수)

| | 논문용 (`paper_*`) | 과제용 (`challenge*`) |
|---|---|---|
| 목표 | 처음 본 파렛트 일반화 | 내 파렛트 과적합, forklift 배포 |
| 데이터 | v1/v2 제외, 합성(mixed_v8) | v1/v2 포함 |
| convention | camera-facing 0123 | camera-facing 0123 |

본 문서는 **논문용** 파이프라인.

## 파이프라인

```
Step 1  합성 데이터 + DOPE 학습
        - camera-facing 0123 합성 (mixed_v8, v1/v2 제외)
        - squash: 여러 aspect ratio 로 비율 강건성 (처음 본 파렛트 비율 대응)
        - truncation padding: 잘린 코너의 belief 를 padding 영역에 supervise
        → paper_base

Step 2  기하 필터 + Pseudo-label
        - paper_base 로 unlabeled 추론 → 9 keypoint 예측
        - 2D projective 기하 필터로 신뢰도 높은 PL 만 선별 (PnP 불필요)
        → 처음 본 파렛트(비율 unknown)도 필터링 가능

Step 3  Self-training Finetuning (반복)
        - 선별된 PL 로 finetune → paper_r1 → PL 재추출 → paper_r2 ...
        - 핵심 교훈: PL 수보다 품질(신뢰도). 좋은 필터가 성공 열쇠.
```

## 핵심 설계 결정

- **convention = camera-facing 0123** → 직사각형 2D 기하 필터 가능 (이전 object-frame 에선 불가)
- **PnP 용도 분리**: 필터 = 2D 기하(PnP 불필요) / 평가·거리 = SQPnP(치수 known 데이터)
- **비율 강건성 = squash** (고정 치수 가정 폐기 → 일반화)
- **truncation = padding** (잘린 이미지 강건)

## 관련 문서

- Step 1: `step1_synthetic_data.md`
- Step 2: `step2_geometric_filter.md`
- Step 3: `step3_selftraining.md`
- 평가: `evaluation.md`
- keypoint: `../preprocessing/keypoint_definition.md`
- 모델: `../models/paper_base.md`


================================================================================
FILE: method/step1_synthetic_data.md
================================================================================

# Step 1 — 합성 데이터 + DOPE 학습 (paper_base)

> camera-facing 0123. 폐기 v8 버전은 `archive/step1_synthetic_data.md`.

## 목표

인터넷 무료 3D 팔레트 모델 기반 합성 데이터로 DOPE 를 학습하되,
**비율 강건성**(처음 본 파렛트 대응)과 **truncation 강건성**(잘린 이미지)을
확보한 논문용 base 모델 `paper_base` 를 만든다.

## 학습 데이터 (경로 확정)

```
합성 base    data/pallet/training_data/mixed_v8_train      9,000장  camera-facing v4
truncation   challenge/data/03_derived/truncation_crops_dope/pretrain 8,831장  crop+padding
squash       [미생성] 비율 강건 증강
제외         challenge/data/02_synthetic/training/v1·v2 (내 파렛트)
```

## 1) 비율 강건성 — squash 증강 [TODO]

처음 본 파렛트는 aspect ratio 가 제각각인데 우리는 특정 비율 합성만 학습 →
일반화 약함. 해결: 학습 이미지를 여러 비율로 **squash(찌부)/stretch** 증강.

- ⚠️ 이미지 변형 시 **JSON 꼭짓점(projected_cuboid)도 동일 변형 동기** 필수.
- 좌표 변환이라 `3d-expert` 위임으로 증강 스크립트 작성 + 검증.
- 변형 범위/분포(어느 비율까지), 원본:증강 비중은 실험으로 결정.

## 2) truncation 강건성 — crop + padding

9 keypoint 다 보이는 이미지를 crop 해 일부 코너가 화면 밖으로 나간 상황 합성 →
DOPE 로더가 padding 영역 확보 후 **화면 밖 코너의 belief map 을 padding 영역에
그려 supervise** (8/8 supervised 검증). 잘려도 9점 회귀 → PnP 6점 안정.

- 기존 자산 `truncation_crops_dope/` 재활용 (dope_cropaug 방식).
- 효과(과제 트랙 검증): real truncation PnP 23→99%, det 13→94%.
- 측면(L/R) 잘림 위주 (top 잘림은 비현실적·degenerate) — memory `truncation-side-cut-bias`.

## 3) 학습 설정

- DOPE VGG-19, 9 belief + 16 affinity, sigma=4.0 (sigma<1 gradient vanishing).
- finetune 은 누적 epoch (memory `dope-finetune-cumulative-epoch`).
- 중간 산출물 `dope_cropaug_pretrain`(squash 없음) → squash 추가 후 재학습 = paper_base.

## 체크리스트

- [ ] squash 증강 데이터 생성 (JSON 동기) — 3d-expert
- [ ] camera-facing v4 변환 정합성 최종 검증 — 3d-expert
- [ ] paper_base 학습 (합성 + squash + truncation)


================================================================================
FILE: method/step2_geometric_filter.md
================================================================================

# Step 2 — 2D 기하 필터 + Pseudo-label (camera-facing)

> camera-facing 0123 기반 2D projective 기하 필터. 폐기 v8(RANSAC c≥6 object-frame)
> 버전은 `archive/step2_geometric_filter.md`. 정확한 설계는 3d-expert 위임 예정.

## 핵심 아이디어

camera-facing 0123 이라 직사각형 cuboid 의 2D 기하 관계가 image 상에서 일관되게
성립한다. 이를 이용해 **PnP 없이 2D 만으로** pseudo-label 의 신뢰도를 판정 →
처음 본 파렛트(비율 unknown)도 필터링 가능. (PnP 용도 분리: 필터엔 PnP 불필요.)

이전 object-frame 에선 코너 순서가 시점마다 달라 이런 기하 제약이 안 먹혔다.
camera-facing 으로 비로소 가능해진 것이 본 연구의 필터 contribution.

## 후보 기하 제약 (사용자 제안 — 3d-expert 가 정확한 인덱스/임계값 확정)

1. **위/아래 순서**: {0,1,4,5} 가 {2,3,6,7} 보다 image y 위쪽
2. **변 비율 일관성**: 앞면 위변(0-1) ≈ 뒷면 위변(4-5), 좌 depth(0-4) ≈ 우 depth(1-5)
   — perspective foreshortening 영향 → 느슨하게 또는 vanishing point 보정
3. **공간 대각선 교점 ≈ centroid(8)**: 0-6, 2-4 등 cuboid 공간 대각선의 교점이
   centroid keypoint 와 가까운지. **직선 교점은 projective invariant** → 비율/거리/
   스케일/시점 무관 ★ 가장 강력
4. (옵션) confidence × geometry 결합, per-domain adaptive threshold — 서베이 권장
   (`../filter/2026-06-02_survey_pseudolabel_filtering.md`)

## 평가 방법 (학습 불필요)

기존 camera-facing 모델 추론만으로 필터 P/R 연구 가능 (Stage 1):
```
camera-facing 모델 → GT 평가셋 추론 → 예측 9 keypoint
  → 2D 기하 필터 적용 → GT 대비 good 판정(order-free 비교) → 필터별 P/R
```
- 상세: `../experiments/filter/pr_screening.md`

## 설계 원칙

- PL 수보다 **품질(precision)** 우선 (발표 교훈: 다량 noisy PL → R2 악화).
- "9 keypoint 전부 검출 시에만" 같은 strict pre-filter 도 후보 (신뢰도↑).

## 체크리스트

- [ ] 2D projective 기하 필터 정확한 인덱스/불변량/임계값 설계 — 3d-expert
- [ ] 기존 모델로 필터 P/R 스크리닝 (학습 불필요)
- [ ] 상위 필터 → Step 3 downstream 검증


================================================================================
FILE: method/step3_selftraining.md
================================================================================

# Step 3 — Self-training Finetuning (camera-facing)

> 폐기 v8 버전은 `archive/step3_finetuning.md`.

## 루프

```
paper_base → unlabeled 추론 → 2D 기하 필터로 PL 선별 (Step 2)
  → PL 로 finetune → paper_r1
  → paper_r1 로 PL 재추출 → finetune → paper_r2 → ...
```

- finetune = 누적 epoch (memory `dope-finetune-cumulative-epoch`).
- 도메인: indoor / outdoor / night (도메인 갭 robustness 검증).

## 핵심 교훈 (이전 발표에서 확인 — 필터 재실험의 동기)

- **indoor**: PL 수 적었지만 R1 에서 성능 크게 ↑
- **outdoor/night**: PL 수 많았지만 ~1% ↑ 에 그치고, **R2 에서 오히려 ↓**
- → **PL 수보다 품질(신뢰도)이 핵심.** 다량의 noisy PL 은 self-training 을 악화시킴.
  좋은 2D 기하 필터로 신뢰도 높은 PL 을 만드는 것이 성공의 열쇠.

## 평가 (2단계)

- **Stage 1 (P/R 스크리닝)**: 학습 없이 기존 모델 추론 → 필터별 P/R 로 후보 선별.
- **Stage 2 (downstream)**: 상위 필터로 실제 R1/R2 학습 → 도메인별 성능 매트릭스.
  - 4월 교훈: P/R 랭킹 ↔ downstream 향상 상관을 명시 검증 (P/R proxy 가 빗나갈 수 있음).

## 체크리스트

- [ ] (Stage 1) 필터 P/R 스크리닝
- [ ] (Stage 2) 상위 필터 R0→R1 도메인별 학습 + 평가
- [ ] R2 (R1 승자만) — 발표 매트릭스 재현


================================================================================
FILE: models/dope_architecture.md
================================================================================

# DOPE 모델 구조 및 학습 설정

## 모델 구조

```
입력: RGB 이미지 (448 × 448)
      ↓
VGG-19 Backbone (ImageNet pre-trained)
→ feature map (50 × 50 × 512)
      ↓
Multi-Stage CNN Heads (Stage 1~6)
→ Belief maps: 9채널 (8 꼭짓점 + 1 centroid)
→ Affinity fields: 16채널
```

## 학습 설정

```
파라미터           값              비고
──────────────────────────────────────────────────────────────────────
optimizer          Adam            DOPE 기본
learning_rate      1e-4 (pretrain) finetune 시 5e-5
                   5e-5 (finetune)
weight_decay       0               DOPE 기본 (no weight decay)
batch_size         4               GPU 메모리 제약
epochs             60 (pretrain)   finetune은 용도에 따라 조정
input_size         448 × 448       정사각형 리사이즈
output_size        50 × 50         belief map 해상도
sigma              4.0             belief map Gaussian std
loss               MSE             belief + affinity (상세: training_loss.md)
```

> **sigma 설정**: belief map GT 생성 시 각 keypoint에 sigma=4.0인 Gaussian을 찍는다.
> 50×50 output에서 ~25×25 픽셀 영역(전체의 25%)을 커버하여 충분한 gradient signal을 제공한다.
> sigma=0.5는 거의 1픽셀 peak만 생성하여 gradient vanishing 문제를 일으킨다.

## Annotation 형식

NDDS 호환 포맷. 각 이미지에 대해 JSON 파일 자동 생성:

```
필드                         내용
──────────────────────────────────────────────────────────────────────
projected_cuboid             8개 꼭짓점 2D 좌표
projected_cuboid_centroid    중심 2D 좌표
cuboid                       8개 꼭짓점 3D 좌표
pose_transform               4×4 포즈 행렬
```

데이터 로더: `Deep_Object_Pose/common/utils.py` CleanVisiiDopeLoader
- `{i:06d}.png` + `{i:06d}.json` 쌍으로 읽음

## Keypoint 추출 (Inference)

DOPE 공식 sub-pixel 방식: Gaussian filter + NMS + 11×11 weighted average

## 평가 메트릭

`scripts/data_prep/eval/evaluate_on_val.py`로 종합 평가:

```
메트릭              설명                              용도
──────────────────────────────────────────────────────────────────────
PCK@3/5/10px       keypoint 위치 정확도              Synthetic val screening
PnP 성공률         EPnP+RANSAC 성공 비율             기본 감지 성능
Reproj error       PnP 재투영 오차 (px)              Pose 품질
Volume Ratio       3D cuboid 부피 비 (1.0=perfect)   Pose 정밀도
ADD                3D 모델 포인트 평균 거리           Real test 최종 평가
5cm-5°             병진 5cm + 회전 5° 이내 비율       Real test 최종 평가
```

수학적 정의 → `_docs/method/formulation.md` Section 10 참조

## 코드 위치

```
파일                                    역할
──────────────────────────────────────────────────────────────────────
Deep_Object_Pose/train/train.py         학습 루프
Deep_Object_Pose/train/geo_loss.py      Geometric loss (soft-argmax + BPnP)
Deep_Object_Pose/common/models.py       DopeNetwork 모델 정의
Deep_Object_Pose/common/utils.py        데이터 로더 (CleanVisiiDopeLoader)
scripts/train_dope.sh                   학습 실행 스크립트
config/default.yaml                     설정 중앙 관리
```


================================================================================
FILE: models/paper_base.md
================================================================================

# paper_base — 논문용 base 모델 (명세 + 로드맵)

> **상태: 미학습 (다음 작업).** squash 비율 강건성을 포함해 새로 학습할 논문용 base.
> 작성일 2026-06-04. 결정: prefix=`paper_`, base=squash 포함 신규 학습 (사용자).

## 논문용 모델 명명 체계

```
paper_base   합성(camera-facing) + squash 비율강건 + truncation padding, v1/v2 제외
  └ paper_r1   기하 필터 self-training Round 1 PL finetune
      └ paper_r2 ...
```

- prefix `paper_` = 논문용 (일반화). 과제용 `challenge*` / `dope_cropaug_ft*` 와 명확히 구분.
- 자세한 트랙 구분: memory `two-tracks-paper-vs-challenge`, `_docs/` 방향 문서.

## 목적

처음 보는 파렛트(비율·외형 제각각)도 6D pose 를 잘 추론하는 **일반화** 모델.
내 실제 파렛트(v1/v2 = palletobj)는 학습에 쓰지 않는다 → 인터넷 무료 3D 팔레트
모델 기반 합성 데이터로만 학습.

## convention

**camera-facing 0123** (v4). 0~3 앞면, {0,1,4,5}=위 / {2,3,6,7}=아래, 8=centroid.
object-frame v8 은 폐기. memory `camera-facing-0123-convention` 참조.

## 학습 데이터 (경로 확정 2026-06-04)

```
합성 base   data/pallet/training_data/mixed_v8_train        9,000장
            (Isaac+Blender, 인터넷 무료 팔레트 모델 렌더)
            ✅ camera-facing v4 변환 적용 확인됨 (.json = camera-facing,
               .json.orig = object-frame 원본 7,205 백업). json≠orig 검증 완료.
+ truncation challenge/data/03_derived/truncation_crops_dope/pretrain  8,831장
            (mixed_v8 기반 crop+padding, camera-facing) — 잘린 이미지 강건성
            메커니즘: 9 kp 다 보이는 이미지를 crop 해 코너가 화면 밖으로 나간
            truncation 합성 → DOPE 로더가 padding 영역 확보 후 **화면 밖 코너의
            belief map(히트맵)을 padding 영역에 그려 supervise** (8/8 supervised
            검증). 잘려도 9점 회귀 → PnP 6점 안정 충족. (PnP 23→99%, det 13→94%,
            memory `dope-cropaug-truncation-success` / `yolo-padding-truncation-wins`)
+ squash    [미생성] 비율 강건성: 여러 aspect ratio 로 찌부(squash)/stretch 증강.
            ⚠️ 이미지 변형 시 JSON 꼭짓점(projected_cuboid)도 동기 변형 필수.
제외        challenge/data/02_synthetic/training/v1·v2 (내 실제 파렛트 palletobj) — 절대 미사용
```

- 위 base+truncation = `dope_cropaug_pretrain` 이 학습한 데이터 (논문 트랙 부합, squash만 빠짐).
- TODO: squash 증강 데이터 생성 스크립트 (변형+JSON 동기) → 3d-expert 검증.
- camera-facing v4 변환 logic: `challenge/scripts/convert_to_camera_facing_v4.py` (`compute_perm_v4`).

## 중간 산출물 (참고)

`weights/dope/dope_cropaug_pretrain` (2026-06-02, scratch 60ep): mixed_v8(camera-facing)
+ truncation crop 8,831 학습. **truncation padding 은 적용됐으나 squash 비율강건은 없음.**
→ paper_base 의 전신/중간 산출물. squash 추가 후 재학습한 것이 정식 paper_base.

## 평가 (계획)

- self-training PL 필터링용 신뢰도: 2D 기하 필터 (PnP 불필요).
- 정확도: 치수 known GT(내 파렛트)에서 ADD/reproj — 단 논문 핵심은 일반화라
  처음 본 파렛트 정성/keypoint reproj 위주.
- **PnP solver = SQPnP** (`cv2.SOLVEPNP_SQPNP` + RefineLM). EPnP+RANSAC 대비
  reproj median 5.27→3.12px, ADD 96.6→90.7mm 개선(2026-06-02 YOLO 경로 검증).
  팔레트는 얇은 near-planar 직육면체라 globally optimal SQPnP 가 유리.
  현재 `scripts/self_training/pnp_solver.py` 는 EPnP+RANSAC → 평가/거리용 SQPnP 교체 필요.
- PnP 용도 분리: memory `camera-facing-0123-convention` 참조.

## 상태 체크리스트

- [ ] squash 비율강건 증강 데이터 생성 (JSON 꼭짓점 동기) — 3d-expert
- [ ] camera-facing v4 변환 정합성 최종 검증 — 3d-expert
- [ ] paper_base 학습 (camera-facing 합성 + squash + truncation padding)
- [ ] 2D 기하 필터 설계 (공간대각선 교점≈centroid 등)
- [ ] paper_r1 self-training (기하필터 PL)


================================================================================
FILE: models/README.md
================================================================================

# 모델 카탈로그

학습된 DOPE 모델 목록. 각 모델의 학습 설정, 데이터, 평가 결과 기록.

> ⚠️ **트랙/convention 안내 (2026-06-04)**: 프로젝트는 **논문용(`paper_*`)** / **과제용(`challenge*`, `dope_cropaug_ft*`)** 2 트랙. 현행 convention = **camera-facing 0123**.
> 아래 `mixed_v*` / `v8_ablation` / `selftrain_r1` 및 "Active 모델 요약"·"v8 Ablation"·평가표는 전부 **폐기된 v8(object-frame) 자산** — 새 작업에 사용 금지, history 참고용으로만 보존.
> 현행 논문용 모델 = **`paper_base`** (아래 `paper_base.md`). 자세히는 CLAUDE.md "핵심 방향" + memory 3종.

## 문서 구조

```
파일                     내용
──────────────────────────────────────────────────────────────────────
paper_base.md            ★ 논문용 base (camera-facing, 합성+squash+truncation, v1/v2 제외) — 명세+로드맵
dope_architecture.md     DOPE 모델 구조, 학습 설정, annotation 형식, 평가 메트릭
training_loss.md         Loss 함수 상세 (기본 MSE + Geometric Loss + Symmetric Loss)
── 이하 폐기 v8(object-frame) 자산 (history 참고용) ──
mixed_v1.md              baseline (Isaac 4K + Blender 4K)
mixed_v2.md              mixed_v1 + blender_manydir 2K (개선 없음)
mixed_v3.md              geo loss 최초 적용 (cuboid 형태 개선, 감지율 하락)
mixed_v6_full.md         dark + view 데이터 추가, augmentation 조정 (PCK 최고)
mixed_v8.md              test_blender 데이터, Real PnP 최고, 8장 self-training
v8_ablation.md           Structural/Reliability loss ablation (A/B/C/D/E) — coord(A) B∧C 최고, rel(E) PnP/B 최고
selftrain_r1.md          Self-training Round 1 (pseudo-label 751장)
archive.md               초기 실험 모델 (pallet_category, pallet_v11, blender_v1, combined_v1 등)
```

## Active 모델 요약

```
모델               Weight 경로                                      Epochs   학습 데이터                              이미지 수    초기 weight       특수 loss
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1           weights/mixed_v1/final_net_epoch_0060.pth         60      Isaac 4K + Blender 4K                     8,000      scratch           MSE only
mixed_v2           weights/mixed_v2/final_net_epoch_0091.pth         91      mixed_v1 8K + manydir 2K                 10,000      mixed_v1 ep60     MSE only
mixed_v3           weights/mixed_v3/final_net_epoch_0091.pth         91      mixed_v2_train 10K                       10,000      mixed_v1 ep60     geo loss
mixed_v4_aug       weights/mixed_v4_aug/final_net_epoch_0060.pth     60      mixed_v1_train 8K (강한 aug)              8,000      scratch           MSE only
mixed_v6_full      weights/mixed_v6_full/final_net_epoch_0060.pth    60      mixed_v1 8K + dark 100 + view 1K          9,100      scratch           MSE only
mixed_v7_sym       (학습 중)                                         91      mixed_v6_full_train 9.1K                  9,100      mixed_v6 ep60     symmetric
selftrain_r1       weights/selftrain/selftrain_r1/final_net_epoch_0070.pth     70      mixed_v1 8K + pseudo-label 751            8,751      mixed_v1 ep60     MSE only
```

## v8 Ablation 모델 (Structural / Reliability Loss)

모두 mixed_v8 (ep60) 위에 mixed_v8_train (9000장)으로 5 epoch finetune. LR=5e-5, batch=4.

```
Ablation   Loss 설정                                         Weight 경로
─────────────────────────────────────────────────────────────────────────────────────
A (coord)  struct_coord=0.003                                weights/v9_ablation_A_coord/
B (edge)   struct_edge=0.003                                 weights/v8_ablation_B_edge/
C (co+ed)  struct_coord=0.003, struct_edge=0.002             weights/v8_ablation_C_coord_edge/
D (flip)   struct_flip=0.02                                  weights/v8_ablation_D_flip/
E (rel)    rel_loss, rel_lambda=0.005, rel_lambda_log=0.5    weights/v8_ablation_E_rel/
```

### noapril 추론 결과 (capture0403noapril, 188장)

```
             PnP Rate    A pass    B pass    C pass    B∧C
v8 (base)    49.5%       43장      1장       0장       0장
A (coord)    62.2%       19장      16장      6장       6장  << B∧C 최고
B (edge)     54.3%       43장      20장      4장       4장
C (co+ed)    55.3%       31장      13장      2장       0장
D (flip)     29.3%       19장      7장       2장       1장
E (rel)      62.8%       24장      25장      4장       2장  << PnP/B 최고
```

## 평가 결과 비교 (Synthetic Val, mixed_v1_val 200장)

```
모델               PCK@3px ↑   PCK@10px ↑   PnP Rate ↑   Reproj mean ↓   Vol Ratio   Vol<20% ↑
──────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1           0.469       0.731        72.5%        88.1 px         1.159       55.3%
mixed_v2           0.466       0.693        77.5%        112.4 px        1.048       54.6%
mixed_v3           0.470       0.719        70.5%        88.6 px         0.764       33.0%
mixed_v4_aug       0.439       0.612        63.0%        215.2 px        0.901       51.0%
mixed_v6_full      0.495       0.632        66.0%        143.8 px        -           52.1%
```

## Real Data 추론 결과 (capture0403noapril, 188장 어두운 팔레트)

```
모델               Avg KP ↑   PnP Rate ↑   비고
───────────────────────────────────────────────────────────────────────
mixed_v1           3.2/9      30.9%        밝은 팔레트 OK, 어두운 팔레트 0/9
mixed_v2           2.9/9      27.1%        데이터 추가했으나 악화
mixed_v3           2.3/9      27.1%        cuboid 3D 형태 개선, 감지율 하락
mixed_v4_aug       2.6/9      22.3%        aug 과도, 어두운 팔레트 centroid 1개 감지
mixed_v6_full      2.9/9      26.6%        어두운 팔레트 3/9 감지, PCK 최고
mixed_v7_sym       (학습 중)               symmetric loss로 앞뒤 혼란 해소 기대
```


================================================================================
FILE: models/training_loss.md
================================================================================

# DOPE Training Loss

## 기본 Loss (항상 활성)

```
total_loss = loss_belief + loss_affinities [+ geo_lambda × loss_geo]
```

```
Loss               수식                  대상                                          채널 수
──────────────────────────────────────────────────────────────────────────────────────────────────
loss_belief        mean((pred - gt)²)    9개 keypoint 히트맵 (8 corner + 1 centroid)   9
loss_affinities    mean((pred - gt)²)    16개 affinity field (8 edge × 2 방향)         16
```

## Symmetric Loss (`--symmetric_loss` 플래그로 활성화)

팔레트처럼 앞/뒤가 시각적으로 동일한 대칭 물체를 위한 loss.
GT belief map의 180° swap 버전도 정답으로 인정.

```
swap 매핑: 0↔5, 1↔4, 2↔7, 3↔6, centroid(8) 유지  (180° Y축 회전, 좌우 뒤집힘 반영)

loss_belief = min(MSE(pred, gt_orig), MSE(pred, gt_swapped))
```

- 앞/뒤 방향 구분 포기 (180° yaw 모호성)
- 앞 vs 옆은 991mm vs 1192mm로 시각적 구분 가능 → 90° swap 불필요
- `--symmetric_loss` 플래그 없으면 기존과 동일

### 멀티스케일 Supervision

```python
for stage in range(len(output_aff)):
    loss_affinities += mean((output_aff[stage] - target_affinities)²)
    loss_belief += mean((output_belief[stage] - target_belief)²)
```

VGG-19의 각 stage 출력마다 loss를 누적 — 중간 레이어에서도 학습 신호 전달.

### GT Belief Map 생성

- 각 keypoint 위치에 `sigma=4.0`의 2D Gaussian을 찍어서 GT 히트맵 생성
- 출력 해상도: 50×50 (input 448 → output 50)
- `sigma < 1`이면 gradient vanishing 발생

## Structural Loss (`--struct_loss` 플래그로 활성화)

Belief map MSE와 병행하여 keypoint 좌표/구조를 직접 최적화하는 loss.
DOPE 모델 구조 변경 없음 — loss 계산용으로만 사용.

### Soft-Argmax vs Argmax

```
argmax (기존 DOPE inference):
  heatmap에서 가장 큰 값의 픽셀 인덱스 반환
  (x, y) = argmax(belief_map) -> 정수 좌표
  미분 불가능 -> gradient가 흐르지 않아서 loss로 사용 불가

soft-argmax (structural loss에서 사용):
  heatmap을 softmax로 확률 분포로 변환 후, 좌표의 가중 평균 계산
  weights = softmax(belief_map / temperature)  # 합=1
  x = sum(weights * x_coords)  # 기대값
  y = sum(weights * y_coords)
  실수 좌표 반환 (예: 23.7, 41.2)
  미분 가능 -> "좌표가 틀리면 heatmap을 고쳐라" gradient 전달 가능
```

기존 DOPE MSE는 heatmap 전체 픽셀 값을 맞추는 loss라 peak 위치가 약간 밀려도 loss가 크게 변하지 않는다.
Soft-argmax 기반 coord loss는 추출된 (x,y) 좌표를 직접 비교하므로 위치 정밀도를 강제한다.

### 구성 요소 (3가지, 각각 독립 on/off 가능)

```
Loss               수식                                              효과
───────────────────────────────────────────────────────────────────────────────────────────
struct/coord       Huber(soft_argmax(pred) - soft_argmax(gt)) / D    좌표 정밀도 직접 강제
struct/edge        Huber(pred edge lengths - gt edge lengths) / D    cuboid 변 길이 보존
struct/flip        FlipEquivariance(pred, flip(input))               좌우 반전 일관성
```

D = object diagonal (크기 불변 정규화)

### 파라미터

```
파라미터             CLI flag           기본값     설명
───────────────────────────────────────────────────────────────────────────────────
활성화               --struct_loss      off       플래그 추가 시 활성화
coord 가중치        --struct_coord     0.10      좌표 Huber loss 스케일
edge 가중치         --struct_edge      0.05      edge length loss 스케일
flip 가중치         --struct_flip      0.02      flip equivariance loss 스케일
Huber delta         --struct_delta     0.03      Huber loss transition point
warmup              --struct_warmup    10        활성화 시작 epoch (이후 10 epoch ramp-up)
```

### Ablation 실험 (v9, base=mixed_v8)

```
Ablation    coord   edge    flip   noapril PnP   B pass   C pass   B^C
─────────────────────────────────────────────────────────────────────────
v8 (base)   -       -       -      49.5%          1        0        0
A (coord)   0.003   0       0      62.2%          16       6        6
B (edge)    0       0.003   0      54.3%          20       4        4
C (co+ed)   0.003   0.002   0      55.3%          13       2        0
D (flip)    0       0       0.02   29.3%          7        2        1
E (rel)     rel_lambda=0.005       62.8%          25       4        2
```

coord loss만으로 PnP 49.5% -> 62.2%, B^C 0 -> 6장. keypoint 양쪽 분포 + PnP 안정성 대폭 개선.

### 사용법

```bash
# coord-only (ablation A)
bash scripts/train_dope.sh --finetune --exp_name v9_A \
    --struct_loss --struct_coord 0.003 --struct_edge 0 --struct_flip 0

# 전체 structural loss
bash scripts/train_dope.sh --finetune --exp_name v9_full \
    --struct_loss --struct_coord 0.003 --struct_edge 0.05 --struct_flip 0.02
```

코드: `Deep_Object_Pose/train/geo_loss.py` (StructuralLoss 클래스)

## Geometric Loss (`--geo_loss` 플래그로 활성화)

Soft-argmax + BPnP(Backpropagatable PnP)로 3D 기하학적 제약 추가.
DOPE 모델 구조는 변경 없음 — loss 계산용으로만 사용, inference 시 제거.

```
학습 시:  이미지 → DOPE → belief map → soft-argmax → BPnP → 3D loss
                     ↑                                          ↓
                     └──────────── gradient 전달 ────────────────┘

추론 시:  이미지 → DOPE → belief map → argmax → PnP (기존 그대로)
```

### Geometric Loss 구성

```
Loss               단계       수식                                    필요 기술
───────────────────────────────────────────────────────────────────────────────────────────
geo/kp_l2          soft-argmax    ||pred_kp - gt_kp||²                    soft-argmax
geo/diagonal       soft-argmax    cuboid 대각선 중점 불일치                soft-argmax
geo/reproj         BPnP           reproject(R,t) vs gt_kp 거리           soft-argmax + BPnP
geo/volume         BPnP           ||V_pred/V_gt - 1||²                   soft-argmax + BPnP
geo/add            BPnP           avg 3D point distance (pred vs gt)     soft-argmax + BPnP
```

### 파라미터

```
파라미터             CLI flag           기본값     설명
───────────────────────────────────────────────────────────────────────────────────
활성화               --geo_loss         off       플래그 추가 시 활성화
전체 가중치          --geo_lambda       0.1       geometric loss 전체 스케일
BPnP warmup         --geo_warmup       5         PnP 기반 loss 활성화 시작 epoch
soft-argmax 온도    --geo_temperature  1.0       낮을수록 sharp, 높을수록 smooth
카메라 내부 파라미터  --geo_fx/fy/cx/cy  D435i     원본 이미지 기준 intrinsics
원본 해상도          --geo_img_w/h      640/480   합성 데이터 원본 크기
```

### BPnP (Backpropagatable PnP) 작동 원리

```
Forward:  cv2.solvePnP(EPnP) — 기존 OpenCV PnP 그대로 사용
Backward: implicit function theorem으로 gradient 계산
          d(pose)/d(kp2d) = (J^T J + λI)^{-1} J^T
          (J = projection Jacobian, λ = damping)
```

- 학습 파라미터 없음 (순수 수학 연산)
- PnP 실패 시 해당 샘플의 geometric loss 자동 skip (validity mask)
- 코드: `Deep_Object_Pose/train/geo_loss.py`

### 사용법

```bash
# 기본 학습 (geometric loss 없음 — 기존과 동일)
bash scripts/train_dope.sh --exp_name mixed_v1

# Geometric loss 포함 학습
bash scripts/train_dope.sh --exp_name mixed_v3 --geo_loss --geo_lambda 0.1

# Geometric loss + finetune
bash scripts/train_dope.sh --finetune --exp_name mixed_v3_geo --geo_loss --geo_lambda 0.1 --geo_warmup 0
```


================================================================================
FILE: preprocessing/keypoint_definition.md
================================================================================

# Keypoint Definition — camera-facing 0123

> 현행 convention (2026-05-22 사용자 결정, v4). 폐기된 object-frame Y=UP 정의는
> `archive/keypoint_definition.md` 참조. memory `camera-facing-0123-convention`.

## 9 keypoint

팔레트 cuboid 8 코너 + centroid = 9 keypoint. DOPE belief map 9채널.

```
인덱스   위치
──────────────────────────────────────────────
0~3     카메라에 가까운 큰 앞면 (FRONT, image polygon area 최대)
          0 = 좌상, 1 = 우상, 2 = 우하, 3 = 좌하
4~7     뒷면 (REAR), 앞면과 대응: 0-4, 1-5, 2-6, 3-7
8       centroid (3D 중심의 투영)
```

- **TOP (위) = {0, 1, 4, 5}**, **BOTTOM (아래) = {2, 3, 6, 7}**
- 앞↔뒤 대응 edge (depth): 0-4, 1-5, 2-6, 3-7
- 앞면 위 edge: 0-1, 뒷면 위 edge: 4-5
- 좌측 depth edge: 0-4, 우측: 1-5

## camera-facing 의 의미

매 frame 카메라에서 본 앞면(가장 큰 면)이 항상 0-1-2-3 으로 라벨된다. 물체
고정(object-frame)이 아니라 **시점 기준**. 따라서 직사각형의 2D 기하 관계(앞면
4점이 한 사각형, 좌우 대칭 등)가 image 상에서 일관되게 성립 → 2D 기하 필터 가능.

## 변환

- 학습 데이터: `challenge/scripts/convert_to_camera_facing_v4.py` (`compute_perm_v4`).
  origin frame 3D 좌표 기준 top/bot split → vertical pairing → image polygon area
  최대 face = FRONT → image x 로 LR. (cam-frame 부호 의존 X)
- 적용 확인: `data/pallet/training_data/mixed_v8_train` 의 `.json`(변환) ≠ `.json.orig`(원본).

## 2D 기하 필터에 쓰는 관계 (정확한 인덱스/불변량은 3d-expert 확정 예정)

- 위/아래 순서: {0,1,4,5} 가 {2,3,6,7} 보다 image y 위쪽
- 변 비율: 앞면 위변(0-1) ≈ 뒷면 위변(4-5), 좌 depth(0-4) ≈ 우 depth(1-5) — perspective 보정 필요
- 공간 대각선(0-6, 2-4 등) 교점 ≈ centroid(8) — projective invariant


================================================================================
FILE: README.md
================================================================================

# 연구 가이드 — Pallet 6D Pose Geometry-aware Self-Training

> **논문 제목:** 파렛트 6D 포즈 추정을 위한 기하학적 제약 기반 준지도 도메인 적응
> **핵심 키워드:** 6D pose estimation, geometry-aware self-training, synthetic data, geometric filter, unsupervised domain adaptation
> **작성일:** 2026-03-25 (v5) / **2026-06-04 v8(camera-facing 전환)**
> **작성자:** 민재
> **중요** 이거는 논문과 github에 코드를 올려서 다른사람들도 테스트하거나 실험할수 있도록 재현성이 있어야됨 그래서 파일 구조와 정리가 중요

> ⚠️ **2026-06-04 방향 전환**: 폐기된 v8(object-frame)을 각 폴더 `archive/` 로 격리.
> 현행 = **camera-facing 0123** convention, 논문용 `paper_*` 트랙(v1/v2 제외, 일반화).
> 2D 기하 필터(PnP 불필요) + squash 비율강건 + truncation padding. CLAUDE.md "핵심 방향" + memory 3종 참조.

---

## 문서 구조

### 전처리 (`preprocessing/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
keypoint_definition.md          키포인트 ID 매핑, camera-facing 0123 convention ({0,1,4,5}위/{2,3,6,7}아래)
archive/                        폐기 v8 (구 Y=UP keypoint_definition, data_pipeline)
```

### 방법론 (`method/`) — camera-facing 재작성 (2026-06-04)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
overview.md                     연구 개요, 두 트랙, 전체 파이프라인 (camera-facing)
step1_synthetic_data.md         Step 1: 합성 + squash 비율강건 + truncation padding → paper_base
step2_geometric_filter.md       Step 2: 2D projective 기하 필터 (PnP 불필요)
step3_selftraining.md           Step 3: 기하필터 PL self-training (R0→R1→R2)
evaluation.md                   메트릭 + PnP 용도분리(필터 2D / 평가·거리 SQPnP)
archive/                        폐기 v8 설계 (구 overview/step1~3/generalization/formulation/implementation)
```

### 모델 카탈로그 (`models/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
README.md                       모델 요약, 평가 비교 테이블, 상세 카드 링크
{model_name}.md                 개별 모델 카드 (학습 설정, 데이터, 평가 결과, 비고)
```

### 필터 연구 (`filter/`)

```
파일                                    내용
──────────────────────────────────────────────────────────────────────────────
README.md                               필터 인덱스, 현행 = camera-facing 2D 기하 필터
2026-06-02_survey_pseudolabel_filtering.md  pseudo-label filtering 서베이 (conf×geo, adaptive)
archive/                                폐기 v8 (RANSAC c≥6 selection/rationale)
실험계획: ../experiments/filter/pr_screening.md  (2D 기하 필터 P/R, 학습 불필요)
```

### 실험 (`experiments/`)

실험 단위로 파일 분할 후 5 개 분야 서브폴더로 재구성 (2026-04-12). 각 파일
은 하나의 Table 또는 Figure 에 대응. 전체 인덱스와 진행 상태는
`experiments/README.md` 참조.

```
폴더 / 파일                                내용                                 상태
──────────────────────────────────────────────────────────────────────────────────────
README.md                                  인덱스 + 평가 프로토콜                 —
model_catalog.md                           모델 카탈로그 (cross-cutting)          갱신
related_work.md                            T10 Related Work 비교                 예정
filter/
├── ablation.md                            T1 Filter Ablation main                예정
├── selection.md                           T3 Filter Selection P/R                ★ 완료
└── consensus_sweep.md                     T7 RANSAC consensus sweep              ★ 완료
loss/
├── ablation.md                            T2 Loss Ablation — coord               ★ 완료
└── coord_strategy.md                      T4 Coord Loss 학습 전략                예정
self_training/
├── rounds.md                              F1 Self-Training Round Figure          예정
├── alpha.md                               T6 α 민감도                            예정
└── forgetting.md                          T8 Catastrophic Forgetting             예정
eval/
├── seen_unseen.md                         T5 Real Seen vs Unseen                 촬영 대기
├── inference_speed.md                     Inference Speed breakdown              예정
└── qualitative.md                         Qualitative Failure Analysis           예정
synthetic/
├── multisource.md                         T9 Multi-source (legacy)               부분
└── sigma_sensitivity.md                   Sigma Sensitivity                      optional
```

### 서베이 (`survey/`)

```
파일                                    내용
──────────────────────────────────────────────────────────────────────────────
survey-6d-pose-estimation.md            6D Pose Estimation 분야 서베이 (방법론/학습 전략/메트릭 비교)
```

### 데이터 (`preprocessing/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
keypoint_definition.md          키포인트 ID 매핑, camera-facing 0123 convention ({0,1,4,5}위/{2,3,6,7}아래)
archive/                        폐기 v8 (구 Y=UP keypoint_definition, data_pipeline)
```

### Real Test Data

```
파일                                            내용
──────────────────────────────────────────────────────────────────────────────
data/pallet/real_data/README.md                 Real data split 정의, 촬영 프로토콜, AprilTag GT, 평가 메트릭
```

### 작업 기록 (`history/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
changelog.md                    과거 작업 이력 (렌더링 개선, 학습, 트러블슈팅)
```

---

## 변경 이력

```
날짜          버전    변경 내용
──────────────────────────────────────────────────────────────────────────────
2026-03-10    v1      초안 작성
2026-03-10    v2      팔레트 일반화 전략, NVIDIA 워크플로우 기반 Stage 1 보강
2026-03-10    v3      실전 렌더링 가이드, 품질 체크리스트 추가
2026-03-13    v3.2    Stage 1 코드 기준 동기화, DR 상세 파라미터
2026-03-19    v4      전면 구조 변경: FixMatch 제거, 3-Step Geometry-aware Self-Training으로 전환. 3단계 Geo Filter 신규 설계. 수식 정의 추가.
2026-03-25    v5      문서 구조 재편: preprocessing/method/experiments/survey/history 하위 폴더 분리. 키포인트 정의 복원. 합성 데이터 파이프라인 문서 추가. 작업 이력 정리.
2026-03-30    v6      멀티소스 학습: Blender 데이터 학습, 실험 관리 체계(compare_experiments.py), 3D 부피 비교 메트릭, 멀티소스 비교 실험 결과 추가
2026-04-11    v7      Filter 재선정: 23 후보 GT 기반 P/R 비교 후 canonical A∧B∧C → RANSAC subset consensus (c≥6) 교체. `filter_type` dispatcher + _docs/filter/ 전용 폴더 신설. overview/formulation/implementation/step2 전면 동기화.
```


================================================================================
FILE: survey/compare-self6dpp-vs-ours.md
================================================================================

# Self6D++ vs Ours (Pallet Pose) — 방법론 & 실험 설계 비교

> 작성: 2026-06-01
> 비교 대상: `~/Documents/github/self6dpp` (Self6D++, TPAMI 2021) vs 본 프로젝트(pallet-pose)
> 목적: ① 두 방법론의 구조·철학 정리, ② 평가 메트릭/실험 설계 관점 비교 및 차용 가능 지점 도출

두 방법 모두 **"PBR/합성으로 pretrain → 라벨 없는 real에서 self-improve"** 라는 동일한 큰 골격을 공유한다.
차이는 ① real 적응 신호를 **무엇으로** 만드는가, ② pose를 **어떻게** 표현/복원하는가 두 축에 집중된다.

---

## 1. 방법론 정리

### 1.1 한눈 비교표

| 축 | **Self6D++ (논문)** | **Ours (pallet-pose)** |
|---|---|---|
| Pose 표현 | **Dense 2D-3D correspondence** (GDR-Net: XYZ map + region + mask) | **Keypoint** (DOPE: 9 belief + 16 affinity, cuboid 8코너+centroid) |
| Backbone | ResNeSt50d / ResNet34 (256×256 in) | VGG-19 + 6-stage CPM (448×448 in, 56×56 out) |
| Pose 복원 | PnP-Net(ConvPnPNet) **direct regression** (6D rot + SITE trans) | **EPnP + RANSAC** (closed-form, gradient 불필요) |
| Refiner | **DeepIM**(FlowNetS, render-and-compare iterative) | 없음 (PnP 결과가 최종) |
| Real 적응 신호 | **Differentiable rendering(DIBR) self-supervision** — render vs real 일관성 | **Geometric Filter(RANSAC consensus)로 pseudo-label 선별 → finetune** |
| Teacher-Student | **Mean Teacher (EMA)**, teacher가 pseudo pose/mask/xyz 생성 | Teacher = 직전 round 모델, hard pseudo-label만 사용 |
| 센서 | **RGB-D** (depth로 chamfer/geom loss) | **RGB only** (depth 미사용) |
| 합성 데이터 | PBR synthetic | Isaac Sim 4.5 + Replicator (NDDS, structured DR) |
| 대상 | LM / LMO / YCBV (소형 가정용 물체 13~21종) | KS T-11 팔레트 1종, 1100×1100×150mm |
| 학습 단계 | Stage I (det+pose+refiner) → Stage II (self-sup) | Step1(pretrain) → Step2(filter+pseudo) → Step3(finetune), 순환 |

### 1.2 Self6D++ 핵심 (근거: `self6dpp/`)

- **Stage I** — PBR synthetic으로 3개 모듈 각각 supervised 학습
  - Detector: YOLOv4 (`det/yolov4/train_yolov4.sh`)
  - Pose estimator: GDR-Net (`core/gdrn_modeling/train_gdrn.sh`) — dense XYZ/region/mask → ConvPnPNet → R(6D)+T(centroid_z, SITE)
  - Refiner: DeepIM (`core/deepim/train_deepim.sh`) — FlowNetS 기반 render-and-compare
- **Stage II** — self-supervised (`core/self6dpp/engine/self_engine.py`)
  - Teacher(EMA mean-teacher)가 real 이미지에서 pseudo pose/mask/xyz 생성 → DIBR 렌더 → real과 일관성 loss로 student 학습
  - **Loss 구성** (`self_engine_utils.py`, config `SELF_LOSS_CFG`):

    | Loss | weight(예) | 비교 대상 |
    |---|---|---|
    | MASK_INIT_REN | 1.0 | pseudo mask ↔ rendered mask (edge-weighted BCE/Dice) |
    | MS_SSIM | 1.0 | real RGB ↔ rendered RGB (structural) |
    | LAB | 0.2 | real ↔ render, Lab의 a,b채널 (조명 불변 photometric) |
    | PERCEPT | 0.15 | AlexNet perceptual |
    | **GEOM (chamfer)** | **100.0** | real depth ↔ rendered depth point cloud (**occlusion-aware**) |
    | SELF_PM | 10.0 | pseudo pose ↔ pred pose (point matching, sym/disentangled) |

  - **Occlusion-aware**: DIBR가 내는 per-pixel prob map + real depth로 가시 영역만 마스킹하여 loss 계산
  - **Renderer**: DIBR(VertexColorBatch), `lib/dr_utils/dib_renderer_x/` — color/depth/mask/xyz/prob 동시 출력
- **철학**: pseudo-label을 *버리지 않고*, 미분 가능 렌더링으로 real 관측과의 **광학적·기하학적 정합**을 직접 backprop. depth가 강한 supervision을 제공.

### 1.3 Ours 핵심 (근거: `scripts/self_training/`, `_docs/method/`)

- **Step1** — Isaac Sim 합성(~6k train) → DOPE scratch 학습 60ep (MSE, 6-stage intermediate sup, σ=4.0)
- **Step2** — real unlabeled inference → keypoint peak(sub-pixel) → **Geometric Filter 3-gate**
  - Pre: keypoint ≥ 5
  - Main: **RANSAC subset consensus** (n_iter=50, subset=5, reproj τ=5px, consensus ≥ 6)
  - Sanity: 복원 팔레트 너비 0.5~2.5m
  - 통과 프레임만 hard pseudo-label로 저장 (`geometric_filter.py`)
- **Step3** — synthetic(GT) + pseudo real(strong aug) 혼합 finetune, `L = L_syn + λ·L_real`, round 반복
  - 수렴: acceptance rate 변화 < 1% 3 round 연속
- **철학**: pseudo-label을 **선별(filter)** 하는 데 집중. 미분 불가능한 EPnP+기하 제약을 *게이트*로만 쓰고, 학습 신호는 여전히 keypoint MSE. depth 없이 RGB+기하 지식으로 신뢰 프레임을 고름.

### 1.4 장단점 대비

| | Self6D++ | Ours |
|---|---|---|
| 강점 | 모든 pseudo 프레임 활용(버림 없음), 미분 렌더로 정밀 정합, refiner로 추가 보정, occlusion 견고 | 단순/안정, RGB-only(센서 부담↓), 기하 제약이 잘못된 라벨을 원천 차단, gradient 불필요로 디버깅 쉬움 |
| 약점 | RGB-D 필요, 미분 렌더러+3 모듈로 파이프라인 무겁고 학습 불안정 위험, mesh 텍스처 품질 의존 | filter가 strict하면 acceptance rate 매우 낮음(초기 ~3%), 버린 프레임의 정보 손실, 미세 pose 보정 메커니즘 없음, EPnP가 keypoint 노이즈에 민감 |

---

## 2. 평가 메트릭 & 실험 설계 비교

### 2.1 메트릭 대응표

| 측정 의도 | Self6D++ | Ours | 비고 |
|---|---|---|---|
| 3D 자세 정확도 | **ADD(-S)** @0.02/0.05/0.1d | **ADD** < 0.1·diameter (real only) | 동일 계열. 우리는 단일 임계(0.1d)만 사용 |
| 회전+병진 동시 | **ReTe** (2°2cm/5°5cm/10°10cm) | **5cm-5°** | Self6D++가 다중 임계로 더 세분화 |
| 2D 투영 정확도 | **Proj** @2/5/10px | **Reproj error (mean px)** | 우리는 비율(%) 아닌 평균값 → 임계 기반 %로 바꾸면 직접 비교 가능 |
| Keypoint 정확도 | (없음, dense) | **PCK@3/5/10px** | keypoint 표현 고유 메트릭 |
| 크기 타당성 | (없음) | **Volume Ratio** (pred/GT cuboid) | 우리 고유, 물리 sanity |
| GT 출처 | 데이터셋 GT pose | **AprilTag** real GT | 우리는 real GT를 직접 취득 |

근거: Self6D++ `core/self6dpp/engine/gdrn_custom_evaluator.py` (`add/adi/re/te/arp_2d`, `lib/pysixd/pose_error.py`) /
Ours `scripts/data_prep/eval/evaluate_on_val.py`, `evaluate_real.py`, `scripts/self_training/metrics.py`

### 2.2 실험 설계 관점 비교

| 관점 | Self6D++ | Ours |
|---|---|---|
| 벤치마크 | 공개 표준(LM/LMO/YCBV) → SOTA 직접 비교 가능 | 자체 팔레트 데이터 → 외부 비교 불가, ablation 중심 |
| 실험 축 | object별, with/without refiner, self-sup loss ablation | round(R0/R1/R2), filter type(ransac/none/loo), domain(night/outside/indoor), epoch(ep65/96) |
| 네이밍 | config 파일명에 인코딩 (`ss_v1_dibr_..._ape`) | 폴더명에 인코딩 (`pl_[domain]_R[round]_[filter]_[variant]`) |
| 결과 관리 | per-object metric 표 (config 주석에 baseline 기록) | `eval_summary.json` + `compare_experiments.py`로 표 생성 |
| 대표 수치 | APE(LM): AD@0.1d 75.7%, ReTe@5°5cm 95.5%, Proj@2px 86.7% | f5 best: PCK@3px 60.5% / syn-only baseline 18.9% (self-training으로 +41.6%p) |

### 2.3 우리 프로젝트에 일반화 가능한 점 (차용 후보)

> 아래는 "방법 차용"이 아니라 **실험 설계/평가 차원**에서 우리 repo에 바로 적용 가능한 것들.

1. **다중 임계 metric 도입** — 현재 5cm-5°, ADD<0.1d는 단일 임계. Self6D++처럼 ADD@0.02/0.05/0.1d, ReTe(2/5/10) 다단계로 보고하면 모델 개선 폭을 더 민감하게 추적 가능. (`metrics.py`에 임계 배열만 추가)
2. **Proj을 %-기반으로** — 현재 Reproj는 mean px. Self6D++의 Proj@2/5/10px처럼 임계 통과율로 함께 보고하면 outlier에 덜 휘둘리는 지표 확보.
3. **AUC(ADD) 추가** — 단일 임계 대신 0~0.1d 구간 AUC를 쓰면 임계 선택 편향 제거, 논문류 비교에 유리.
4. **per-difficulty 분해** — Self6D++의 LMO(occlusion) 분리 평가처럼, 우리도 domain(night/outside/indoor)·가림 정도별로 metric 분해 보고 (이미 실험은 domain별 분리되어 있으니 평가 표만 분해).
5. **(방법 차용, 선택) Soft self-supervision 신호** — filter로 *버리는* 프레임이 많은 게 약점(초기 acceptance ~3%). Self6D++식 mask/silhouette 일관성을 *보조 loss*로 추가하면 버려진 프레임도 약한 신호로 활용 가능. 단 미분 렌더러 도입 비용 큼 → PCK 정체 시에만 검토.
6. **EMA mean-teacher** — 현재 teacher=직전 round 체크포인트(hard switch). EMA로 부드럽게 갱신하면 self-training 안정성↑, round 간 진동↓ (구현 비용 낮음, 우선 검토 권장).

---

## 3. 결론 요약

- **같은 가족, 다른 신호**: 둘 다 self-supervised domain adaptation이지만, Self6D++는 *미분 렌더링 정합*(soft, depth 활용, 전 프레임 사용)이고 우리는 *기하 필터 선별*(hard, RGB-only, 신뢰 프레임만).
- **우리 설계의 정체성**: RGB-only + 기하/물리 제약 + keypoint는 센서·구현 부담이 낮고 잘못된 라벨을 원천 차단하는 게 강점. 대신 정보 손실(낮은 acceptance)과 미세 보정 부재가 약점.
- **즉시 적용 권장**: 메트릭 다단계화(2.3-1~3) + EMA teacher(2.3-6)는 방법론을 안 바꾸고도 비교성·안정성을 올리는 저비용 개선.
- **장기 검토**: PCK/ADD가 정체되면 Self6D++식 보조 silhouette/photometric loss로 "버린 프레임"을 약한 신호로 재활용하는 hybrid를 고려.

---

### 근거 파일 색인

**Self6D++** (`~/Documents/github/self6dpp`)
- 학습 엔트리: `core/self6dpp/engine/self_engine.py`, `main_self6dpp.py`, `train_self6dpp.sh`
- Loss: `core/self6dpp/engine/self_engine_utils.py`, `losses/depth_bp_chamfer_loss.py`, `losses/pm_loss.py`
- Renderer: `lib/dr_utils/dib_renderer_x/renderer_dibr.py`, `configs/_base_/renderer_base.py`
- 평가: `core/self6dpp/engine/gdrn_custom_evaluator.py`, `lib/pysixd/pose_error.py`
- Config: `configs/self6dpp/ssLM|ssLMO|ssYCBV/`

**Ours** (`~/Documents/github/pallet-pose`)
- Self-training: `scripts/self_training/self_train.py`, `self_train_pseudo.py`, `geometric_filter.py`, `pnp_solver.py`
- 모델: `Deep_Object_Pose/common/models.py`
- 평가: `scripts/data_prep/eval/evaluate_on_val.py`, `evaluate_real.py`, `scripts/self_training/metrics.py`
- 비교 유틸: `scripts/compare_experiments.py`
- 문서: `_docs/survey/survey-6d-pose-estimation.md`, `_docs/method/overview.md`


================================================================================
FILE: survey/survey-6d-pose-estimation.md
================================================================================

# 6D Pose Estimation Field Survey

> 6D object pose estimation 분야의 주요 논문/프로젝트 접근법을 비교 정리한다.
> 새로운 실험 설계나 구현 결정 시 참고 자료로 활용한다.
> 생성일: 2026-03-21

## 현재 프로젝트 설정

```
항목          설정
────────────────────────────────────────────────────────────────────────────────
방법론        DOPE (keypoint-based) + Geometric Self-Training
Backbone      VGG-19 (ImageNet pretrained)
출력          9 belief maps + 16 affinity fields, 50×50
Loss          MSE (belief + affinity, 6-stage intermediate supervision)
Sigma         4.0 (belief map Gaussian std)
PnP           EPnP + RANSAC
학습          Synthetic pretrain → Geometric filter pseudo-label → Mixed finetuning
합성 데이터   Isaac Sim 4.5 + Replicator, 도메인 랜덤화
평가          PCK@3px (val), ADD / 5cm5° / Reproj (test)
```

---

## 1. Pose Representation

6D 포즈를 어떻게 표현하고 추출하느냐에 따라 크게 4가지 패러다임으로 나뉜다.

```
패러다임                대표 방법                    출력                          포즈 복원          장점                              단점
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Keypoint-based          DOPE, PVNet                  2D keypoint heatmap/voting    PnP solver         해석 가능, 멀티인스턴스, 가볍다   keypoint 수에 의존, 대칭 객체 어려움
Dense correspondence    CDPN, ZebraPose, GDR-Net     픽셀별 3D 좌표 (NOCS map)     PnP/RANSAC         폐색에 강함, 정밀도 높음          계산량 큼, 대칭 ambiguity
Direct regression       PoseCNN                      쿼터니언 + 평행이동           없음 (end-to-end)  단순, 빠름                        정밀도 낮음, 비선형 회전 공간 학습 어려움
Render-and-compare      FoundationPose, MegaPose     SE(3) delta                   반복 정제          새 객체 일반화, CAD만 있으면 됨   느림, 렌더링 필요
```

### 현재 프로젝트와 비교
- **현재**: Keypoint-based (DOPE) — 팔레트는 비대칭 직육면체로 keypoint 방식에 적합
- **대안**: Dense correspondence (GDR-Net)는 정밀도 더 높지만 학습 복잡도 증가
- **Render-and-compare** (FoundationPose)는 CAD 모델 있으면 zero-shot 가능하나 실시간성 부족

---

## 2. 학습 전략 (Supervised vs Self-Training vs DA)

```
전략                          대표 방법                    Real 라벨 필요     핵심 메커니즘                       성능 수준
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Supervised                    PVNet, GDR-Net, ZebraPose    전량               GT 포즈 직접 학습                   최고 (상한선)
Synthetic-only                DOPE (original)              없음               도메인 랜덤화로 sim-to-real gap 극복 하한선
Differentiable rendering      Self6D, Self6D++             없음 (RGB-D 필요)  렌더링 일관성 loss로 자기지도       Supervised에 근접
Pseudo-label self-training    DSC-PoseNet, Ours            없음               모델 예측 → 필터링 → 재학습         Syn-only 대비 큰 향상
Foundation model              FoundationPose, MegaPose     없음               대규모 사전학습 + 일반화            객체별 학습 불필요
```

### 주요 Self-Training 방법 비교

```
방법          Pseudo-label 생성            필터링                                      필요 센서   특징
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Self6D        Differentiable rendering     렌더링 loss 수렴                            RGB-D      깊이 정보 필수
Self6D++      Noisy student + rendering    Teacher-student 일관성                      RGB-D      폐색 인식 추가
DSC-PoseNet   Dual-scale mask 비교         Scale consistency                           RGB        약지도 (bbox만 필요)
Ours          DOPE + EPnP                  3단계 기하학적 필터 (Flip/Diagonal/LOO-PnP) RGB        도메인 지식 활용, 깊이 불필요
```

### 현재 프로젝트와 비교
- **현재**: RGB-only pseudo-label self-training + geometric filter
- **차별점**: Self6D/Self6D++는 RGB-D 필요, 우리는 RGB-only
- **차별점**: DSC-PoseNet은 mask 기반, 우리는 keypoint + 물리적 규격 기반 필터링
- **고려사항**: Self6D++의 noisy student 프레임워크를 우리 파이프라인에 결합 가능성

---

## 3. 합성 데이터 & Domain Randomization

```
항목          Unstructured DR (Tremblay 2018)   Structured DR (Prakash 2019)   현재 프로젝트
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
배경          랜덤 텍스처/이미지                맥락에 맞는 장면               3모드 혼합 (창고 40%, 실내 30%, 야외 30%)
객체 배치     균일 랜덤                         맥락 인식 (차는 도로 위)       바닥 수평 고정, yaw만 자유
조명          랜덤 HDR                          물리 기반                      DomeLight + RectLight 3개, 랜덤화
디스트랙터    랜덤 형상                         의미론적으로 적절              54종 6카테고리 (primitive + 다양화)
가림          랜덤                              맥락에 맞는 적재물             60% 프레임에 팔레트 위 적재물
데이터량      매우 많이 필요                    중간                           5,000~15,000장
```

### 주요 합성 데이터 도구

```
도구                            사용 방법                특징
─────────────────────────────────────────────────────────────────────────────────────
NVIDIA Isaac Sim + Replicator   DOPE, 현재 프로젝트     PathTracing, USD 기반, 프로그래밍 가능
NVISII                          DOPE (이전 버전)        빠른 레이트레이싱, Python API
BlenderProc                     BOP Challenge 데이터 생성 Blender 기반, 유연함
Kubric                          Google, 다양한 객체     물리 시뮬레이션 포함
```

### 현재 프로젝트와 비교
- **현재**: Isaac Sim 4.5 + Replicator, Structured DR에 가까움 (도메인 특화 배치)
- **강점**: 카메라를 리프터 시점으로 제한, 팔레트 물리적 배치 반영
- **개선 가능**: LLM-aided 텍스처 다양화 (FoundationPose 방식)

---

## 4. 네트워크 구조

```
방법              Backbone               Head 구조                       출력 해상도   파라미터
─────────────────────────────────────────────────────────────────────────────────────────────────
DOPE              VGG-19 (24층)          6-stage CPM                     50×50         ~60M
PVNet             ResNet-18              Encoder-decoder                 입력 해상도   ~12M
CDPN              ResNet                 Encoder-decoder × 2             64×64         ~25M
GDR-Net           ResNet-34 / ConvNeXt   3 geometric maps + Patch-PnP   64×64         ~35M
ZebraPose         ResNet / EfficientNet  FCN encoder-decoder             64×64         ~25M
FoundationPose    Transformer            Refiner + Scorer                patch-based   ~100M+
```

### Backbone 선택 가이드

```
Backbone        특징                       적합한 경우
──────────────────────────────────────────────────────────────────────────
VGG-19          큰 receptive field, 느림   multi-stage refinement (DOPE)
ResNet-18/34    가볍고 효율적              실시간 필요 시
ResNet-50/101   더 높은 표현력             정밀도 우선
ConvNeXt        최신 CNN, ResNet 대체      GDRNPP에서 검증
EfficientNet    효율-성능 최적화           엣지 디바이스 배포
Transformer     전역 attention             Foundation model, 대규모 학습
```

### 현재 프로젝트와 비교
- **현재**: VGG-19 + 6-stage CPM (DOPE 원본 그대로)
- **고려사항**: ResNet-34로 교체 시 속도↑ + 성능 유지 가능 (GDR-Net 참고)

---

## 5. Loss Function & Belief Map 설계

### Loss Function 비교

```
방법              Loss                                      특징
──────────────────────────────────────────────────────────────────────────────────────
DOPE              MSE (belief + affinity)                   단순, 6-stage 중간 감독
PVNet             Smooth L1 (voting) + CE (seg)             견고한 voting
CDPN              L1 (coord) + regression (trans)           회전/평행이동 분리
GDR-Net           L1 + CE + PM loss (6D rotation)           기하학적 가이드
ZebraPose         BCE (hierarchical bits) + L1 (mask)       Coarse-to-fine 가중치
Self6D            MSE + rendering losses (RGB/depth/mask)   미분가능 렌더링
FoundationPose    Contrastive triplet + SE(3) regression    포즈 스코어링
```

### Belief Map Sigma 설정

```
설정     Sigma      커버리지 (50×50 기준)   효과
────────────────────────────────────────────────────────────────────────────────
극소     0.5        ~1px (0.04%)            Gradient vanishing, 학습 실패
소       2.0        ~13×13 (7%)             학습 가능, 정밀도 높음
표준     4.0        ~25×25 (25%)            DOPE 공식 기본값, 안정적 학습
대       7.0        ~43×43 (74%)            OpenPose 기본값, 쉬운 학습 but 낮은 정밀도
적응형   거리 비례  가변                    최신 연구, 스케일 변화에 강함
```

> **일반 원칙**: sigma↑ = 학습 용이 + 정밀도↓, sigma↓ = 학습 어려움 + 정밀도↑

### 현재 프로젝트와 비교
- **현재**: MSE loss + sigma=4.0 (표준)
- **self-training 단계**: sigma=2.0 (config), pretrain보다 작은 sigma로 정밀도 높임
- **고려사항**: 적응형 sigma (거리 기반)는 팔레트 크기 변화가 클 때 유용

---

## 6. PnP Solver & 후처리

```
Solver           필요 점 수   복잡도   미분 가능   사용처
──────────────────────────────────────────────────────────────────────────────
P3P              3            O(1)     No          RANSAC 내부
EPnP             ≥4           O(n)     No          DOPE, PVNet, 현재 프로젝트
DLT              ≥6           O(n)     No          고전적 방법
Iterative (LM)   ≥4           반복     No          ICP 정제
BPnP             N            반복     Yes         End-to-end 학습
EPro-PnP         N            반복     Yes         CVPR 2022 Best Student Paper
Progressive-X    N            반복     No          ZebraPose, 멀티인스턴스
```

### 후처리 방법

```
방법                            입력              효과                        비용
────────────────────────────────────────────────────────────────────────────────────────
ICP (Iterative Closest Point)   RGB-D + CAD       PoseCNN에서 +17% ADD-S      느림, 깊이 필요
RANSAC outlier 제거             2D keypoints      잘못된 keypoint 필터링      빠름
Pose hypothesis 랭킹            복수 후보         FoundationPose 스코어러     중간
Geometric filter                2D/3D keypoints   현재 프로젝트 핵심 기여     빠름
```

### 현재 프로젝트와 비교
- **현재**: EPnP + RANSAC (threshold=8px, 100 iter) + 3단계 geometric filter
- **차별점**: 대부분의 방법은 confidence threshold만 사용, 우리는 기하학적 일관성 검증 추가
- **고려사항**: EPro-PnP 적용 시 end-to-end 학습 가능 (연구 확장)

---

## 7. 평가 메트릭

### 메트릭 정의

```
메트릭          정의                                     임계값             사용처
──────────────────────────────────────────────────────────────────────────────────────────
ADD             모델 점들의 평균 3D 거리                 <0.1d (직경 10%)   비대칭 객체
ADD-S           최근접 점 매칭 평균 거리                 <0.1d              대칭 객체
ADD(-S) AUC     ADD/ADD-S 커브 아래 면적                 —                  YCB-Video 표준
5cm-5°          평행이동 <5cm AND 회전 <5°               5cm, 5°            로봇 조작
Reproj          2D 재투영 오차                           <5px               빠른 평가
PCK@Npx         N px 이내 keypoint 비율                  3px, 5px, 10px     Keypoint 정확도
VSD             Visible Surface Discrepancy              τ, δ               BOP Challenge
MSSD            Max Symmetry-aware Surface Distance      —                  BOP 2022+
MSPD            Max Symmetry-aware Projection Distance   —                  BOP 2022+
```

### BOP Challenge 표준 (AR = Average Recall)
```
AR = (AR_VSD + AR_MSSD + AR_MSPD) / 3
```
7개 코어 데이터셋: LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V

### 현재 프로젝트와 비교
- **현재**: PCK@3px (val), ADD + 5cm5° + Reproj (test)
- **BOP 표준과 차이**: BOP는 AR(VSD+MSSD+MSPD) 사용
- **팔레트 특성**: 비대칭 → ADD 적합, 대형 → 5cm5°가 실용적 지표

---

## 8. Pseudo-Label 필터링 전략

Self-training의 핵심은 pseudo-label 품질. 필터링 전략 비교:

```
방법                        필터링 기준                  센서    도메인 지식
─────────────────────────────────────────────────────────────────────────────────
Confidence threshold        모델 출력 confidence         RGB     없음
Self6D++ (noisy student)    Teacher-student 일관성       RGB-D   없음
DSC-PoseNet                 Dual-scale pose 일관성       RGB     없음
Ours (A)                    Flip consistency             RGB     없음
Ours (B)                    Diagonal concurrency         RGB     직육면체 가정
Ours (C)                    Leave-one-out PnP stability  RGB     PnP 기반 pose 검증
```

### 필터링 품질 vs 채택률 트레이드오프

```
엄격한 필터 → 높은 PL 정확도 / 낮은 채택률 → 학습 데이터 부족 위험
느슨한 필터 → 낮은 PL 정확도 / 높은 채택률 → 노이즈 라벨로 성능 저하
```

- **Confidence-only**: 가장 단순하지만 overconfident 예측 필터링 못함
- **기하학적 필터 (Ours)**: 모델 confidence와 독립적으로 물리적 일관성 검증 → 상호 보완
- **Self6D++ 방식**: Teacher가 좋아야 student도 좋음 → 초기 품질에 민감

### 현재 프로젝트와 비교
- **현재**: 3단계 기하학적 필터 (A: flip consistency, B: diagonal concurrency, C: LOO PnP)
- **강점**: RGB-only, 도메인 지식 활용, confidence와 독립적
- **약점**: 객체 형태에 의존 (직육면체 가정), 다른 객체에는 C 필터 재설계 필요

---

## Summary: 현재 프로젝트 vs 일반적 접근

```
항목            현재 프로젝트              일반적 접근                   비고
──────────────────────────────────────────────────────────────────────────────────────────────
Pose 표현       Keypoint (DOPE)            Dense correspondence가 주류  팔레트에는 keypoint 충분
Backbone        VGG-19                     ResNet-34/50이 주류          VGG는 무거움, 교체 고려
Loss            MSE                        Task-specific loss 조합      MSE로 충분 (keypoint 방식)
Sigma           4.0                        2~7 범위                     표준 설정
PnP             EPnP + RANSAC              EPnP + RANSAC (동일)         표준
Self-training   Geometric filter           Rendering loss / Noisy student  RGB-only가 차별점
PL 필터         3단계 기하학적             Confidence threshold         핵심 기여
합성 데이터     Isaac Sim + Structured DR  BlenderProc / Isaac Sim      표준
평가            ADD + 5cm5°                BOP AR 또는 ADD              표준에 가까움
```

---

## 추가 조사 필요 항목

- [ ] ResNet-34 backbone으로 DOPE 교체 시 성능/속도 비교
- [ ] 적응형 sigma (거리 기반) 적용 가능성
- [ ] EPro-PnP (differentiable PnP) 적용으로 end-to-end 학습 가능성
- [ ] Self6D++ noisy student + 우리 geometric filter 결합 실험
- [ ] BOP 표준 메트릭 (AR) 추가 구현 및 벤치마크 비교
- [ ] FoundationPose의 LLM-aided 텍스처 다양화를 합성 데이터에 적용

---

## 참고 문헌

```
약칭              논문                                                                                  학회
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DOPE              Tremblay et al., "Deep Object Pose Estimation"                                        CoRL 2018
PoseCNN           Xiang et al., "PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation" RSS 2018
PVNet             Peng et al., "PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation"              CVPR 2019
CDPN              Li et al., "CDPN: Coordinates-based Disentangled Pose Network"                        ICCV 2019
Self6D            Wang et al., "Self6D: Self-Supervised Monocular 6D Object Pose Estimation"            ECCV 2020
GDR-Net           Wang et al., "GDR-Net: Geometry-Guided Direct Regression Network"                     CVPR 2021
DSC-PoseNet       Yang et al., "DSC-PoseNet: Learning 6DoF Object Pose via Dual-Scale Consistency"      CVPR 2021
Self6D++          Wang et al., "Occlusion-Aware Self-Supervised Monocular 6D Object Pose Estimation"    TPAMI 2022
ZebraPose         Su et al., "ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation" CVPR 2022
EPro-PnP          Chen et al., "EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points"    CVPR 2022
MegaPose          Labbé et al., "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare"    CoRL 2022
FoundationPose    Wen et al., "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects" CVPR 2024
BOP 2023          Sundermeyer et al., "BOP Challenge 2023 on Detection, Segmentation and Pose Estimation" CVPRW 2024
```

