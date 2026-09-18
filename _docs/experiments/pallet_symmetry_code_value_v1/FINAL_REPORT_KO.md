# Explicit C1/C2/C4 code value isolation v1 — 완료 보고

판정: **KEEP_CODE_DEV_EVIDENCE**

두 모델은 dimensions, symmetry target, architecture, parameters, training budget가 같고 explicit group-code information만 다르다.

동일 8-D 모델(20,307 parameters), seed별 모든 초기 tensor·batch order 일치, T=1/lambda=1/cap=0.01 고정. Blind는 항상 [1/3,1/3,1/3], aware는 approved one-hot. R0는 frozen이며 모든 과거 DCP 파일을 보존했다.

## 판정의 적용 범위

사전 등록 규칙의 형식 판정과 모든 환경에서 코드가 필수라는 주장은 다르다. A synthetic-only matched-capacity 비교는 전체 synthetic/REAL_DEV 모두 CI가 0을 포함했다. B mixed 비교에서는 C1/C2/C4 모두 개선됐지만 domain/group confounding이 있는 작은 개발 효과이다.
중요한 반례: A1과 B1 모두 REAL_DEV에서는 NEUTRAL이 CORRECT보다 낮은 E_sym을 보였다(각 CI가 0 아래). A1의 WRONG도 CORRECT보다 유의하게 좋았고, B1 WRONG의 점추정도 더 좋지만 CI는 0을 포함한다. 이를 숨기거나 mapping을 다시 고르지 않았다. 올바른 코드가 synthetic/square에서 유용하다는 결과를 실사 C2 배포의 보편적 필요성으로 확대하지 않는다.
따라서 KEEP_CODE_DEV_EVIDENCE가 나오더라도 그것은 명시된 mixed/개발 조건에서의 유지 근거이다. Synthetic-only paper A의 추가 이득이나 독립 실사 일반화 증명이 아니며, 이번 결과로 DEV에 맞춰 코드 모드를 새로 선택하지 않았다.

## A exact-capacity paper

| split | arm | E_sym | median px | P90 px | PCK10 |
|---|---|---:|---:|---:|---:|
| SYNTH | blind | 0.0088710 | 1.7687 | 6.7777 | 0.93754 |
| SYNTH | aware | 0.0088700 | 1.7622 | 6.7679 | 0.93788 |
| DEV | blind | 0.0484165 | 5.7278 | 41.9715 | 0.68707 |
| DEV | aware | 0.0484273 | 5.7842 | 41.7889 | 0.68681 |

모든 표는 3-seed 평균. median/P90은 매칭 관측 corner, E_sym/PCK는 missing-detection penalty 포함 전체 분모. E_sym은 이미지 대각선으로 정규화되어 px가 아니다.

| split | aware − blind | CI95 | seed별 delta | 개선 seed |
|---|---:|---|---|---:|
| SYNTH | -0.0000010 | [-0.0000031, 0.0000012] | [6.856958817120186e-07, -2.619206163792076e-07, -3.275135100235656e-06] | 2/3 |
| DEV | 0.0000108 | [-0.0000089, 0.0000317] | [2.515364687134124e-05, 1.2788326913048463e-05, -5.552867083808495e-06] | 1/3 |

## B mixed routing — DIAGNOSTIC ONLY

| eval population | group coverage | blind | aware | delta | CI95 |
|---|---|---:|---:|---:|---|
| SYNTH | C1, C2 | 0.0092365 | 0.0092062 | -0.0000303 | [-0.0000364, -0.0000242] |
| DEV | C2 | 0.0483153 | 0.0482918 | -0.0000235 | [-0.0001474, 0.0000584] |
| SQUARE | C4 | 0.0027356 | 0.0027226 | -0.0000129 | [-0.0000221, -0.0000032] |

## Group contrasts

| track/pop | group | n | delta | CI95 | improved seeds |
|---|---|---:|---:|---|---:|
| A/SYNTH | C1 | 511 | -0.0000021 | [-0.0000069, 0.0000027] | 2/3 |
| A/SYNTH | C2 | 1474 | -0.0000005 | [-0.0000030, 0.0000019] | 2/3 |
| A/DEV | C2 | 319 | 0.0000108 | [-0.0000089, 0.0000317] | 1/3 |
| B/SYNTH | C1 | 511 | -0.0000146 | [-0.0000230, -0.0000059] | 3/3 |
| B/SYNTH | C2 | 1474 | -0.0000358 | [-0.0000432, -0.0000281] | 3/3 |
| B/DEV | C2 | 319 | -0.0000235 | [-0.0001474, 0.0000584] | 2/3 |
| B/SQUARE | C4 | 155 | -0.0000129 | [-0.0000221, -0.0000032] | 3/3 |

## C inference code perturbation

E_sym. Correct checkpoint 및 dimensions/evaluator를 고정하고 network code만 변경했다. 새 optimizer update는 0.

| model/pop | correct | neutral | zero | wrong | shuffled |
|---|---:|---:|---:|---:|---:|
| A1_CODE_AWARE/SYNTH | 0.0088700 | 0.0088723 | 0.0088725 | 0.0088750 | 0.0088750 |
| A1_CODE_AWARE/DEV | 0.0484273 | 0.0483994 | 0.0484041 | 0.0484047 | N/A(single group) |
| B1_CODE_AWARE/SYNTH | 0.0092062 | 0.0092174 | 0.0092215 | 0.0092909 | 0.0092113 |
| B1_CODE_AWARE/DEV | 0.0482918 | 0.0482334 | 0.0482352 | 0.0481767 | N/A(single group) |
| B1_CODE_AWARE/SQUARE | 0.0027226 | 0.0027719 | 0.0027653 | 0.0027912 | N/A(single group) |

| model/pop/code | delta vs correct | CI95 | mean move px | max move px | JS | top1 change |
|---|---:|---|---:|---:|---:|---:|
| A1_CODE_AWARE/SYNTH/NEUTRAL | 0.0000023 | [0.0000001, 0.0000045] | 0.1178 | 1.7982 | 0.0002933 | 0.11539 |
| A1_CODE_AWARE/SYNTH/ZERO | 0.0000025 | [0.0000002, 0.0000047] | 0.1079 | 2.4881 | 0.0002835 | 0.10808 |
| A1_CODE_AWARE/SYNTH/WRONG | 0.0000049 | [0.0000020, 0.0000078] | 0.1153 | 2.6468 | 0.0004355 | 0.11463 |
| A1_CODE_AWARE/SYNTH/SHUFFLED | 0.0000050 | [0.0000021, 0.0000078] | 0.0982 | 2.7174 | 0.0004934 | 0.08621 |
| A1_CODE_AWARE/DEV/NEUTRAL | -0.0000280 | [-0.0000435, -0.0000144] | 0.1206 | 1.4454 | 0.0001645 | 0.07158 |
| A1_CODE_AWARE/DEV/ZERO | -0.0000232 | [-0.0000365, -0.0000117] | 0.1017 | 1.5144 | 0.0001335 | 0.06139 |
| A1_CODE_AWARE/DEV/WRONG | -0.0000226 | [-0.0000344, -0.0000134] | 0.0902 | 1.5886 | 0.0000942 | 0.05956 |
| B1_CODE_AWARE/SYNTH/NEUTRAL | 0.0000112 | [0.0000072, 0.0000152] | 0.2157 | 4.7402 | 0.0009018 | 0.19528 |
| B1_CODE_AWARE/SYNTH/ZERO | 0.0000153 | [0.0000108, 0.0000197] | 0.2350 | 5.9265 | 0.0010694 | 0.20802 |
| B1_CODE_AWARE/SYNTH/WRONG | 0.0000847 | [0.0000735, 0.0000957] | 0.5483 | 11.4030 | 0.0052040 | 0.39652 |
| B1_CODE_AWARE/SYNTH/SHUFFLED | 0.0000051 | [0.0000022, 0.0000080] | 0.1078 | 3.4003 | 0.0005841 | 0.09832 |
| B1_CODE_AWARE/DEV/NEUTRAL | -0.0000584 | [-0.0000903, -0.0000169] | 0.4378 | 9.9750 | 0.0020599 | 0.22845 |
| B1_CODE_AWARE/DEV/ZERO | -0.0000565 | [-0.0000927, -0.0000109] | 0.4650 | 10.8471 | 0.0023943 | 0.23746 |
| B1_CODE_AWARE/DEV/WRONG | -0.0001151 | [-0.0002051, 0.0000045] | 0.9777 | 15.9196 | 0.0084335 | 0.43103 |
| B1_CODE_AWARE/SQUARE/NEUTRAL | 0.0000492 | [0.0000259, 0.0000820] | 0.2316 | 4.3006 | 0.0020903 | 0.18065 |
| B1_CODE_AWARE/SQUARE/ZERO | 0.0000426 | [0.0000207, 0.0000733] | 0.2186 | 3.5591 | 0.0018327 | 0.17473 |
| B1_CODE_AWARE/SQUARE/WRONG | 0.0000685 | [0.0000426, 0.0001074] | 0.2731 | 6.3280 | 0.0030645 | 0.20269 |

## Damage / runtime / 6D

| track/pop | net good<5 → bad>10 | gross20 delta | safety guard |
|---|---:|---:|---|
| A/SYNTH | 0 | -0.0000212 | True |
| A/DEV | 0 | 0.0000000 | True |
| B/SYNTH | 0 | 0.0000849 | False |
| B/DEV | -1 | -0.0026677 | True |
| B/SQUARE | 0 | 0.0000000 | True |

| arm | params | P-only median ms | RGB-to-PnP median ms |
|---|---:|---:|---:|
| A0_CODE_BLIND | 20307 | 3.209 | 14.705 |
| A1_CODE_AWARE | 20307 | 3.208 | 14.846 |

동일 PnP adapter의 6D secondary는 `POSE_SECONDARY_RESULTS.json`에 보존했다. 실사/square pose reference는 재구성 geometry이며 독립 물리 GT가 아니다. 6D는 primary 판정 변경에 사용하지 않았다.

## Decision / interpretation

`KEEP_CODE_DEV_EVIDENCE`. 사전 고정 판정 게이트: `{'result': 'KEEP_CODE_DEV_EVIDENCE', 'headline_benefit': True, 'A_and_B_real_safety_pass': True, 'B_square_not_clearly_worse': True, 'same_multigroup_population_neutral_and_wrong_advantage': True, 'any_matched_capacity_benefit': True, 'any_correct_code_value': True, 'thresholds_source': {'path': '_docs/experiments/pallet_symmetry_code_value_v1/PROTOCOL_LOCK.json', 'sha256': 'f57624ca89687bf2249235767d07576cd64e94d23894df6da64209e5d39839f8', 'bytes': 5628}, 'additional_sweep': False}`.

Code 유지 개발 근거를 확보했다. Deployment에서 dimensions와 approved group code를 외부에서 알아야 한다.

## 제한 및 재현

- A에는 C4가 없다. 실제 source TRAIN C1=19,268/C2=36,712. Heldout C1=511/C2=1,474.
- 여기서 검증한 code-free 대조군은 동일한 8-D 모델의 마지막 3채널을 neutral로 고정한 버전이다. 기존 5-D N3로 교체했을 때의 수치를 이 실험에서 새로 주장하지 않는다.
- REAL_DEV319는 모두 C2: 그룹 간 routing이 아니라 constant-token 영향. Wood는 physical symmetry UNREVIEWED와 기존 C2 benchmark convention을 구분했다.
- B는 C1/C2 synthetic, C4 real-supervised의 domain/group confounding이 있다. Pooled 성공 지표나 symmetry causal concept 학습 주장 금지.
- Bootstrap 10,000, seed20260918. Synthetic frame primary/scenario secondary; real session-cluster 및 LOSO. Seed는 3개의 paired mean이며 seed population의 불확실성을 별도 resample하지 않았다.
- Session subgroup에 cluster가 하나뿐이면 CI N/A. Empty C1/C2/C4 group에 수치를 만들지 않았다. Multiplicity-adjusted confirmatory evidence가 아니며 DEV는 재사용되었다.
- Shuffled donor는 score 보기 전 고정. DEV/SQUARE는 단일 group이므로 N/A. 같은 code/frame donor 수는 CODE_PERTURBATION_RESULTS.json에 기록.
- Neutral/zero는 aware 학습에서 보지 않은 code 값이고 A의 cyclic C2→C4도 미학습 group 입력이다. 따라서 교란 민감도만으로 대칭 개념이나 인과적 routing 학습을 주장하지 않는다.
- 온도 calibration, seed/step/lr sweep, 추가 architecture/loss, 신규 데이터, FINAL/sealed 접근, R0 학습은 없다.
- 사전 검사 첫 시도는 결측 GT의 NaN==NaN 비교 때문에 실패했고 equal_nan=True, rtol=atol=0 검사로 바로잡았다. 데이터/모델 수정이나 optimizer update는 없었다.
- `python -B scripts/research/pallet_symmetry_code_value_v1/runner.py`가 고정 순서 실행/완료 checkpoint 재사용을 제공한다. Python 환경: pallet-yolo26. 소스/모델/metadata SHA는 SOURCE_BINDING과 각 예측 파일에 보존.
- 코드·통계·보고서는 Git으로 공유한다. 기존 저장소 정책상 `data/**`, checkpoint, logits는 Git 제외 로컬 산출물이다. RAW_ARTIFACT_INDEX.json에 경로·크기·SHA를 남겼으며 다른 PC 재현에는 명시된 기존 데이터/cache가 필요하다.

주학습 72,000 + 폐기 smoke 200 update. 회귀검사 37개 통과. 원본 경로는 read-only 감사 통과.
