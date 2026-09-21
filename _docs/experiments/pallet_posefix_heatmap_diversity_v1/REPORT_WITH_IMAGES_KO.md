# 보정기 개선 방향 검증 — 완료 보고

**DIVERSITY_PILOT_POSITIVE**. 실제 SINGLE32/MULTI32 각300step 학습 완료. MULTI32의 PRIMARY PCK10 차이는 +0.70pp입니다.

| 항목 | 한 세션32장 | 두 세션32장 |
|---|---:|---:|
| PRIMARY PCK10% | 49.79 | 50.49 |
| 20–40px → ≤10px 복구 | 5 | 5 |
| BASE 정답 손실 ↓ | 22 | 19 |
| GREEN PCK10% | 78.85 | 78.85 |
| 합성 clean PCK10% | 91.59 | 91.69 |
| 합성 stress PCK10% | 68.15 | 67.90 |

앞선 무학습 진단에서는 큰 잔여 오류 163개 중 출력 범위상 복구 불가능 29개, global argmax로 해결 가능한 코너 0개였습니다. crop이 부분 병목인 것은 맞지만 전체 오류의 설명은 아니며, 단순 좌표 읽기 방식 변경만으로 복구되는 신호는 없었습니다.

중요: 두 조건은 같은32장 수·teacher/필터·loss·학습량을 사용하지만 데이터 내용이 다릅니다. 유효 코너256/246, 인공가림21/22장, teacher의 낮 recording 노출·다른 세션 미노출 차이도 있습니다. 따라서 순수한 세션 수 효과나 독립 일반화로 주장하지 않습니다. 새 어노테이션 없이 수행했고 현재 최종 모델을 교체하지 않았습니다.

최종 이미지 확인에서 추가한 밤16장은 저양각·일부 잘림도 포함했습니다. 이 결과를 clean/top-visible/no-truncation 조건까지 맞춘 다양성 실험이라고 부르지 않습니다. [실제 학습 이미지와 육안 확인](VISUAL_DATA_AUDIT_KO.md)에 이 제한을 명시했습니다. 표식이 보이는 RGB도 있지만 표식 기반 GT를 새로 만들거나 학습하지 않았습니다.

비교의 범위: 과거253장 FULL은 PCK10 50.35%, B3 복구 6개였습니다. MULTI32가 기존 FULL보다 전반적으로 우수하다고 확정한 결과가 아닙니다. 이번 신호는 같은32장 예산에서 일부 촬영조건 대응이 좋아진 것이며 큰 오류 복구 한계는 남았습니다.

[다음 단계](NEXT_STAGE_PLAN.md). 아래에 heatmap 오류 사례, 학습32장 전체 contact sheet, 개선/악화/무작위 평가 이미지를 포함했습니다. 보고서·이미지만 공개하고 checkpoint, 원본 heatmap, 비공개 mapping/후보 manifest는 제외합니다.

---

# Crop·heatmap 진단과32장 세션 비교

GT/reference는 모델 입력이 아니라 추론 후 진단·평가·그림에만 사용했다. 기존278장 출력을 모두 동결하고 비교했다. 새로운 어노테이션·깊이·CAD 없이 실행했으며 기존 알려진 치수 기반 PnP는 기존 pseudo recipe 그대로 재사용했다.

## 1. 큰 오류의 위치와 heatmap

| 구분 | 수 |
|---|---:|
| n | 163 |
| crop_outside | 34 |
| unreachable10 | 29 |
| expectation_good10 | 0 |
| argmax_good10 | 0 |
| argmax_rescue | 0 |
| top5_oracle_good10 | 16 |
| GT_peak_atleast10pct | 33 |
| GT_mass_median | 0.0016266757156699896 |

PRIMARY matched R0>20 & FP>20의 163개 중 crop 밖 reference 34개, 출력 지지영역 때문에10px 이내 도달불가 29개 (17.8%). 따라서 crop 문제가 없다는 결론이 아니다. 다만 사전20%·최소5개 중단 gate에는 미달했고, 평균 좌표 대신 global argmax를 쓰면 복구되는 코너는0개였다.

top5 local peak 중 reference10px 이내가 있는 것은 16/163. 이는 GT로 후보를 골라 보는 진단 상한이지 실제 선택기의 성능이 아니다. 후보가 없다는 사실만으로 RGB에 증거가 없다고 단정하지 않는다. 대부분 큰 오류에서 현재 학습된 채널이 reference 근처를 강하게 지지하지 않는다는 관찰이다.

추가 bounds 감사에서 도달불가29코너의 reference는 모두 원본640×480 이미지 안에 있었다. 즉 원본 밖 정답 때문이 아니라 예측 bbox 기반 crop/출력 지지영역의 제한이다. crop 확대가 정확도까지 개선할지는 별도이며, 이번에는 crop을 변경하거나 평가 reference로 맞추지 않았다.

BASE 재현은 최초 새로 설정한1e-4px 검사에서0.00018081px 차이로 중단됐다. 과거 코드의 실제0.01px 계약을 확인하고 BASE legacy mode로 재현했다. 최종 BASE 최대차이0.0001744px, FULL/FP는 약1.4e-14px. 실패 캐시와 수정 기록을 보존했고, weight·진단 gate·성능 threshold는 변경하지 않았다.

### 주요 오류 그림

초록x=reference, 청록=기존 FP expectation, 자홍+=argmax, 노랑원=top5 peaks. 범위 밖 reference는 crop 패널에서 보이지 않을 수 있으므로 원본 패널과 범위 제한 수치를 함께 본다.

#### crop_limited

![crop_limited](figures/diagnostic_001.jpg)

![crop_limited](figures/diagnostic_002.jpg)

![crop_limited](figures/diagnostic_014.jpg)

![crop_limited](figures/diagnostic_015.jpg)


#### in_crop_no_candidate

![in_crop_no_candidate](figures/diagnostic_005.jpg)

![in_crop_no_candidate](figures/diagnostic_004.jpg)

![in_crop_no_candidate](figures/diagnostic_011.jpg)

![in_crop_no_candidate](figures/diagnostic_006.jpg)


#### candidate_present_but_not_selected

![candidate_present_but_not_selected](figures/diagnostic_003.jpg)

![candidate_present_but_not_selected](figures/diagnostic_007.jpg)

![candidate_present_but_not_selected](figures/diagnostic_012.jpg)

![candidate_present_but_not_selected](figures/diagnostic_010.jpg)


#### random_hard

![random_hard](figures/diagnostic_013.jpg)

![random_hard](figures/diagnostic_009.jpg)

![random_hard](figures/diagnostic_016.jpg)

![random_hard](figures/diagnostic_008.jpg)


### 다른 코너 채널에는 위치 증거가 있는가?

잔여 큰 오류 163개 중 다른 채널의 argmax가 reference10px 이내인 사례 17개, 다른 채널 expectation이 가까운 사례 13개, 다른 채널 top5 중 가까운 사례 41개다. 자신의 top5에는 없지만 다른 채널 argmax에는 있는 사례는 8개다.

이는 일부 코너 identity 혼동 가능성을 보여 주지만 모든 큰 오류를 설명하지는 않는다. 수치는 서로 겹치며 더해서는 안 된다. GT로 다른 채널을 골랐고1:1 assignment 제약도 없으므로 실제 보정기 성능이 아니며, 정답 채널·공식 symmetry·학습 target을 바꾸지 않았다.

## 2. 동일조건32장 학습 비교

**DIVERSITY_PILOT_POSITIVE**. MULTI32−SINGLE32의 PRIMARY PCK10 차이 +0.701pp.

SINGLE32: 낮32장. MULTI32: 동일 낮16장 + capturenight01의16장. 같은 일반 플라스틱, 기존 고정 R0→flip/LOO→Replay→기존 self-occlusion PnP→LOO를 양쪽 모두 새로 적용. 평가 recording/hash 제외. 낮253/다른세션52장이 통과한 후 경로 순서 기반 midpoint로 선택했으며 평가 오차로 고르지 않았다.

PRIOR1 동일 초기화 FULL, seed1, TFAdam1e-4, 300step, real8+source8/micro2, BN 통계·affine 고정, preserveOFF. 각 이미지75노출로 실사2400, source2400. source 순서·corruption은 기존 FULL과도 전 step 동일. 최종 checkpoint만 평가하고 추가 sweep은 하지 않았다. 학생 detector self-training이 아니라 보정기 학습이다.

실제 유효 코너 수 SINGLE32=256, MULTI32=246. 인공가림 이미지 수 {'SINGLE32': 21, 'MULTI32': 22}. 다른 이미지에 동일 mask를 강제로 씌우지 않았다. 따라서 동일 이미지수·recipe·update 비교이지 동일 유효 코너수 또는 순수한 session 개수 인과실험은 아니다. 낮은 teacher 학습 recording 노출이 있고 다른 세션에는 없으며, 시점·빛·자연 가림도 함께 달라진다. NIGHT의 clean 여부는 별도 검증하지 않았다.

### 실제 학습 이미지 (노랑은 pseudo target, GT 아님)

![SINGLE32](figures/train_SINGLE32.jpg)

![MULTI32](figures/train_MULTI32.jpg)

### PLASTIC194 — 실제 128장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 54.82 | 75.13 | 8.353 | 40.653 | 0/118 | 0 | 57 |
| N2 | 57.66 | 76.45 | 8.311 | 40.793 | 0/118 | 29 | 0 |
| FULL | 57.56 | 75.53 | 7.526 | 41.265 | 6/118 | 32 | 56 |
| FULL_PRESERVE | 58.38 | 76.24 | 7.594 | 40.275 | 5/118 | 23 | 48 |
| SINGLE32 | 57.87 | 76.35 | 7.699 | 43.059 | 6/118 | 30 | 54 |
| MULTI32 | 58.38 | 76.35 | 7.512 | 42.458 | 6/118 | 31 | 54 |

### GREEN150_MANUAL — 실제 150장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 80.18 | 89.43 | 3.953 | 10.172 | 0/3 | 0 | 17 |
| N2 | 81.35 | 89.28 | 4.122 | 9.749 | 0/3 | 9 | 0 |
| FULL | 79.15 | 88.84 | 4.042 | 10.387 | 0/3 | 19 | 29 |
| FULL_PRESERVE | 79.44 | 88.40 | 3.898 | 10.396 | 1/3 | 18 | 27 |
| SINGLE32 | 78.85 | 88.69 | 4.002 | 10.797 | 0/3 | 22 | 32 |
| MULTI32 | 78.85 | 88.40 | 3.952 | 10.539 | 0/3 | 20 | 30 |

### DEV72_REFERENCE_UNKNOWN — 실제 72장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 57.84 | 75.68 | 7.684 | 50.524 | 0/79 | 0 | 20 |
| N2 | 59.46 | 75.14 | 7.831 | 50.910 | 0/79 | 11 | 0 |
| FULL | 60.90 | 75.68 | 7.244 | 52.362 | 6/79 | 13 | 22 |
| FULL_PRESERVE | 61.26 | 75.68 | 7.189 | 52.000 | 5/79 | 10 | 19 |
| SINGLE32 | 61.26 | 76.58 | 7.162 | 51.953 | 5/79 | 10 | 16 |
| MULTI32 | 60.54 | 76.22 | 7.230 | 52.641 | 5/79 | 12 | 21 |

### PRIMARY_OCC96 — 실제 93장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 47.69 | 67.04 | 9.638 | 52.649 | 0/99 | 0 | 29 |
| N2 | 49.09 | 68.02 | 9.448 | 51.193 | 0/99 | 19 | 0 |
| FULL | 50.35 | 67.88 | 8.554 | 53.178 | 6/99 | 17 | 31 |
| FULL_PRESERVE | 50.35 | 67.88 | 8.645 | 53.174 | 5/99 | 16 | 31 |
| SINGLE32 | 49.79 | 68.44 | 8.980 | 53.572 | 5/99 | 22 | 34 |
| MULTI32 | 50.49 | 68.16 | 8.755 | 53.438 | 5/99 | 19 | 33 |

### CLEAN_NONCAD69 — 실제 17장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 92.65 | 97.79 | 3.991 | 8.636 | 0/4 | 0 | 2 |
| N2 | 93.38 | 97.06 | 3.750 | 8.642 | 0/4 | 1 | 0 |
| FULL | 90.44 | 98.53 | 4.221 | 9.434 | 0/4 | 4 | 5 |
| FULL_PRESERVE | 91.91 | 99.26 | 3.802 | 8.943 | 0/4 | 2 | 3 |
| SINGLE32 | 92.65 | 98.53 | 4.121 | 8.467 | 0/4 | 2 | 3 |
| MULTI32 | 91.18 | 98.53 | 3.984 | 8.912 | 0/4 | 3 | 4 |

### CAD18 — 실제 18장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 54.41 | 94.85 | 9.829 | 18.584 | 0/15 | 0 | 26 |
| N2 | 66.91 | 100.00 | 8.690 | 13.891 | 0/15 | 9 | 0 |
| FULL | 62.50 | 92.65 | 7.861 | 18.544 | 0/15 | 11 | 20 |
| FULL_PRESERVE | 66.91 | 97.06 | 8.057 | 16.551 | 0/15 | 5 | 14 |
| SINGLE32 | 65.44 | 95.59 | 7.945 | 15.828 | 1/15 | 6 | 17 |
| MULTI32 | 66.91 | 97.06 | 7.746 | 15.706 | 1/15 | 9 | 17 |

### SELECTED_CLEAN8 — 실제 8장

| model | PCK10% | PCK20% | med px | P90 px | B3 복구 | BASE 손실 | N2 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 43.75 | 93.75 | 10.671 | 19.299 | 0/6 | 0 | 17 |
| N2 | 64.06 | 100.00 | 9.065 | 14.356 | 0/6 | 4 | 0 |
| FULL | 73.44 | 93.75 | 7.391 | 18.431 | 0/6 | 0 | 3 |
| FULL_PRESERVE | 71.88 | 100.00 | 7.729 | 14.911 | 0/6 | 0 | 3 |
| SINGLE32 | 70.31 | 96.88 | 7.758 | 13.475 | 0/6 | 0 | 5 |
| MULTI32 | 71.88 | 100.00 | 7.488 | 14.332 | 0/6 | 0 | 4 |

### 전이 검산

SINGLE→MULTI GOOD/GOOD=346, GOOD/BAD=9, BAD/GOOD=14, BAD/BAD=344. 순정답=5. PRIMARY의 실패8장54코너 벌점800px 유지; median/P90만 matched코너를 사용했다.

### Source heldout256

| arm | clean PCK10% | stress PCK10% |
|---|---:|---:|
| SINGLE32 | 91.59 | 68.15 |
| MULTI32 | 91.69 | 67.90 |

### 사전 screen

| 조건 | 통과 |
|---|---|
| PCK10 | True |
| B3 | True |
| BASE_preservation | True |
| GREEN | True |
| source_clean | True |

### Crop 제한과 후보 유무별 후속 변화

아래 하위집합은 기존FP 진단으로 고정했고, 공식 평가를 거르거나 학습샘플을 선택하는 데 사용하지 않았다.

| subgroup | n | SINGLE ≤10 / ≤20 | MULTI ≤10 / ≤20 |
|---|---:|---|---|
| unreachable10 | 29 | 0/0 | 0/0 |
| reachable10 | 134 | 0/8 | 0/7 |
| top5_present | 16 | 0/0 | 0/0 |
| top5_absent | 147 | 0/8 | 0/7 |

### PRIMARY 세션별 변화

| session | frames/corners | SINGLE PCK10% | MULTI PCK10% | B3 SINGLE/MULTI |
|---|---:|---:|---:|---:|
| eval_night08 | 12/92 | 42.39 | 44.57 | 0/0 |
| eval_night09 | 16/122 | 36.07 | 40.16 | 0/0 |
| eval_outside | 5/40 | 70.00 | 75.00 | 0/0 |
| eval_pallet07 | 27/197 | 52.28 | 51.27 | 5/5 |
| eval_pallet09 | 33/262 | 53.82 | 53.05 | 0/0 |

개선의 위치: eval_night 두 세션은 정답 +7개, 나머지 PRIMARY 세션 합계 -2개다. 따라서 모든 가림/시점에 고르게 일반화됐다는 결과가 아니라, 추가한 촬영조건과 관련된 일부 domain coverage 개선에 부합하는 제한적 결과다.

과거253장 FULL은 PRIMARY 50.35%, B3 6개, BASE 손실 17개였다. 새 MULTI32가 그 모델을 모든 면에서 이긴 것은 아니며, 이번의 직접 통제 비교는 SINGLE32 vs MULTI32다. 큰 오차 복구 증대나 최종 모델 승격으로 해석하지 않는다.

### 개선·악화·고정 무작위 평가 이미지


#### MULTI_better

`eval_pallet09:1778653630038417664`

![MULTI_better](figures/comparison_012.jpg)

`eval_pallet07:1778652168786111744`

![MULTI_better](figures/comparison_011.jpg)

`eval_pallet09:1778653634641026304`

![MULTI_better](figures/comparison_013.jpg)

`eval_pallet07:1778652154608392192`

![MULTI_better](figures/comparison_010.jpg)


#### MULTI_worse

`eval_pallet07:1778652150610404864`

![MULTI_worse](figures/comparison_009.jpg)

`eval_pallet09:1778653674184865536`

![MULTI_worse](figures/comparison_014.jpg)

`eval_night09:1779449634044385536`

![MULTI_worse](figures/comparison_006.jpg)

`eval_pallet09:1778653832794714368`

![MULTI_worse](figures/comparison_016.jpg)


#### random_primary

`eval_pallet09:1778653674184865536`

![random_primary](figures/comparison_014.jpg)

`eval_pallet07:1778652144496057088`

![random_primary](figures/comparison_008.jpg)

`eval_pallet09:1778653823253508096`

![random_primary](figures/comparison_015.jpg)

`eval_pallet07:1778652136533958400`

![random_primary](figures/comparison_007.jpg)

`eval_night09:1779449580573721600`

![random_primary](figures/comparison_005.jpg)

`eval_night08:1779449479095927552`

![random_primary](figures/comparison_004.jpg)


#### random_GREEN

`capture_20260902_kimjihoon__006169`

![random_GREEN](figures/comparison_002.jpg)

`capture_20260902_kimjihoon__005869`

![random_GREEN](figures/comparison_001.jpg)

`capture_20260902_kimjihoon__008369`

![random_GREEN](figures/comparison_003.jpg)


## 해석 제한

가림 이미지를 평가했지만 외부가림 코너인지 사람 visibility 검토는 REVIEW_PENDING이다. 비초록은 기존 legacy reference, GREEN은 manual이며 합쳐 독립 GT로 주장하지 않는다. 반복DEV·single seed·상관된pseudo teacher 실험이다. 신규 annotation/최종 모델/논문표 변경 없음. 추가 학습은 이 두 조건으로 종료했다.

[다음 단계](NEXT_STAGE_PLAN.md) · [전체 그림 HTML](GALLERY.html)
