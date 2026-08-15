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

---

## ransac_loo reproj-threshold sweep (2026-06-08, **paper_base** 모델)

> 목적: ransac_loo 순도(good%)는 s2에서 거의 완벽했으나 통과량(N)이 사망(1~3) → reproj threshold 완화로 양을 늘리며 순도 유지점(sweet spot)이 있는지 탐색.
> 판단지표: 도메인별 [통과 N] + [good%(order-free 9kp reproj <10px)] + [9kp_med]. 양·순도 동시.
> 모델: **weights/paper_base/paper_base/final_net_epoch_0060.pth** (논문 트랙 base, GT셋 held-out — 누수 없음). 직전 pr_screening 표는 ft_s2(누수) 기준이라 직접 비교 불가.
> 코드: `scripts/data_prep/eval/filter_loo_sweep.py` (캐시 `_full_paper_base.json` + per-frame GT json에서 K·dimensions_m 재독). dims=per-frame GT(W/D swap 자동 흡수). good = Hungarian 9kp(8코너+centroid) reproj <10px.
> detectable(>=6코너): indoor 149 / outside 110 / night 57.

### [ransac_loo threshold sweep]  (LOO_tau=0.05, consensus c>=6)
```
domain    tau_px    N   good%  9kp_med
--------------------------------------
indoor         3    0      --       --
indoor         5    7     0.0     11.0
indoor         8   17     0.0     12.1
indoor        10   18     0.0     11.8
indoor        12   18     0.0     11.8
indoor        15   18     0.0     11.8
--------------------------------------
outside        3    1     0.0     13.9
outside        5    3     0.0     15.0
outside        8    4     0.0     16.7
outside       10    6     0.0     24.7
outside       12    6     0.0     24.7
outside       15    7     0.0     20.0
--------------------------------------
night          3    1     0.0     14.7
night          5    2     0.0     19.6
night          8    3     0.0     21.6
night         10    4     0.0     22.1
night         12    4     0.0     22.1
night         15    4     0.0     22.1
```

### [diag] (threshold-free)
```
domain       N   good%  9kp_med
-------------------------------
indoor      12    25.0     11.1
outside      6    16.7     13.2
night       13     7.7     20.8
```

### [fullkp] (threshold-free)
```
domain       N   good%  9kp_med
-------------------------------
indoor      36    11.1     12.3
outside     31    12.9     17.6
night       16     6.2     23.0
```

### sanity 대조군 — 동일 sweep을 s2 캐시로 (코드 검증 + s2엔 sweet spot 존재 확인)
```
[ransac_loo, s2]  tau   N   good%  9kp_med      [diag,s2]                [fullkp,s2]
outside            3    1   100.0    9.4        outside 27 51.9% 9.9     outside 37 48.6% 10.0
outside            5    3   100.0    9.3        night   10 60.0% 8.8     night   29 58.6% 9.1
outside            8    4    75.0    9.4        indoor   7  0.0% 13.2    indoor  32  0.0% 14.1
night              5    2   100.0    8.4
night              8   10    70.0    9.0   ← s2 night sweet spot: tau=8, N 2→10, 순도 70% 유지
night             15   15    73.3    8.6
indoor(s2도 붕괴)  15   12    33.3   23.5
```

### 핵심 결론 (paper_base 기준 — 직전 s2 결론과 정반대)
1. **paper_base에서는 ransac_loo good% = 모든 threshold·모든 도메인 0.0%.** threshold를 3→15로 완화하면 N은 늘지만(outside 1→7, night 1→4, indoor 0→18) 9kp_med가 11~25px로 그대로/악화. **sweet spot 없음.** 완화는 나쁜 PL만 더 통과시킴.
2. **원인 = 모델 keypoint 정확도.** s2(누수 self-train) 대조군에서는 같은 코드로 outside/night가 tau 5~8px에서 순도 70~100%·9kp_med ~9px·N이 4~10으로 늘어 명확한 sweet spot 존재. 즉 "ransac_loo 순도 완벽"은 **s2의 성질이지 paper_base의 성질이 아니다.** paper_base는 코너 예측 자체가 9kp_med 11px+ 라 어떤 기하필터도 <10px PL을 못 만듦.
3. **diag/fullkp 단독도 paper_base에선 good% 6~25%** (indoor diag 25%가 최고이나 N=12·9kp_med 11.1px). paper_base 단독으로 self-training용 청정 PL 확보 불가.
4. **함의**: paper_base를 self-training 1라운드의 PL 소스로 직접 쓰는 건 부적합. (a) 합성 데이터로 base를 더 끌어올리거나, (b) good 기준 px를 12~13으로 올려 "약간 부정확하나 방향 맞는" PL을 허용하거나(품질 저하 감수), (c) 누수 없는 약한 self-train을 거쳐 코너 정확도를 먼저 올린 뒤 재필터하는 단계적 접근이 필요.
5. unlabeled pool 추출량 추정: paper_base+ransac_loo(tau=8)는 detectable 대비 indoor 17/149(11%)·outside 4/110(4%)·night 3/57(5%) 통과하나 **good%=0** → 추출해도 학습에 해로움. 권장 추출량 = 0 (현 base·현 good기준).

> 저장: `data/pallet/eval_results/filter_loo_sweep/sweep_paper_base.json`, `sweep_s2.json`, 콘솔표 `.txt`. paper_base 추론 캐시 `data/pallet/eval_results/filter_domain_analysis/_full_paper_base.json`.

---

## Flip-consistency 필터 (2026-06-08, paper_base)

**질문:** 좌우 flip TTA 일관성이 `diag`(공간대각선 교점≈centroid, projective invariant — centroid만 봄)가 못 잡는 **back(4-7)/centroid 불안정**을 잡아 더 믿을만한 PL을 거르는가?

**방법** (`scripts/data_prep/eval/filter_flip_consistency.py`, A 추론은 `_full_paper_base.json` 캐시 재활용, flip 추론만 신규):
- A = 원본 9kp 예측. B = 이미지 좌우 flip 추론 → x un-flip(W-x) + camera-facing swap `0↔1,3↔2,4↔5,7↔6`(centroid 8 고정).
- flip score = index-aligned 평균 per-kp ||A−B|| (px). A·B는 같은 모델의 같은 물리 코너 예측이라 swap 후 index 정합이 맞아 order-free 불필요. 판정(통과 PL 품질)은 GT 대비 order-free 9kp(Hungarian 8코너+centroid).

```
[reference: pool & diag-alone]
scope       det good_pool%  diagN diag_med diag_good%
indoor      117       6.8%     12     11.1      25.0%
outside      58       6.9%      6     13.2      16.7%
night        27       3.7%     13     20.8       7.7%
ALL         202       6.4%     31     13.7      16.1%

[flip consistency  tau-sweep]   (N / 9kp_med / good%)
scope     tau=8           tau=10          tau=15
indoor    8/13.2/0%       24/13.2/4%      89/12.6/6%
outside   1/10.6/0%       4/10.7/25%      27/14.5/11%
night     0/--/--         7/19.7/14%      10/20.2/10%
ALL       9/13.1/0%       35/13.5/9%      126/13.0/7%

[diag AND flip — flip이 diag 통과분의 나쁜 PL을 거르나]
scope  tau=10 (combo N/9kp_med/good%)   diag-only  flip이 떨군 diag-pass(N, e9_med)
indoor 4/10.6/25%                       N12 25%    drop 8 (med 11.4)
night  5/19.7/20%                       N13 7.7%   drop 8 (med 23.0)
ALL    10/12.3/20%                      N31 16.1%  drop 21 (med 13.9)
```

**상관:** Spearman(flip, GT 9kp err)=0.37(전체 196), **diag 통과분 안에서 0.48**, Pearson 0.58. → flip은 PL 품질과 단조 상관. diag-pass 안에서 flip이 떨군 21개 e9 median 13.9 vs 남긴 10개 12.3 (떨군 쪽이 더 나쁨 — 방향 맞음).

### 결론 (정직하게)
1. **flip은 품질의 유효한 프록시** — score가 GT 오차와 상관(0.37~0.58)하고, 특히 **diag 통과분 안에서 0.48로 diag가 못 본 변동을 잡음**. diag∧flip(tau=10)은 indoor good% 25%→(N4 유지)·night 7.7%→20%로 순도를 올림(다만 N 급감).
2. **그러나 paper_base 천장이 결정적 제약.** detectable pool 자체 good%가 6.4%뿐(코너 예측 9kp_med 11px+). flip<=10이어도 good% 8.6%·9kp_med 13.5px. flip은 *상대적으로* 나쁜 걸 골라내지만, 모집단에 좋은 PL이 없어 **절대 순도(good%)는 여전히 한 자릿수**. (앞 ransac_loo/diag/fullkp 섹션과 동일 천장.)
3. **systematic depth 붕괴 우려는 데이터상 부분적으로만 맞음.** flat-view depth 붕괴가 좌우대칭이면 flip해도 일관(score 작음)→못 거름이 예상이나, 실제로 flip score와 GT err가 양의 상관 → 붕괴 케이스 다수는 flip시 불일치도 동반(완전 systematic은 아님). 단 flip<=10인데 good 아닌 32개는 이 잔여(일관되게 틀림)로 추정.
4. **함의:** flip은 diag/ransac_loo 대비 **버릴 이유 없는 보완 신호**(특히 diag 통과분의 2차 게이트로 0.48 상관). 하지만 단독으로도 조합으로도 **현 paper_base에선 청정 PL을 못 만든다**. 필터 개선이 아니라 base 코너 정확도(합성 보강/단계적 self-train)가 선결. flip의 진짜 가치는 base가 9kp_med<10px로 올라온 뒤 재평가해야 의미 있음(s2 캐시에선 다른 필터가 sweet spot 보였듯).

> 저장: `data/pallet/eval_results/filter_flip_consistency/flip_consistency_paper_base.{txt,json}` (records에 frame별 flip/e9/diag 포함). 코드 `scripts/data_prep/eval/filter_flip_consistency.py`.
