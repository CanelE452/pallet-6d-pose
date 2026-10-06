# 고정 seed1 SOFT6D / EXPECT6D 비교

[확인] SOFT6D와 EXPECT6D를 각각 원래 seed1 초기값부터 6,000 update씩 실행하고 마지막 checkpoint만 평가했다. 두 실행은 첫 방법의 성능에 따라 바뀌지 않았다.

[확인] 판정: SOFT6D **NO_CLEAR_GAIN**, EXPECT6D **OLD_ONLY_GAIN**. N3 대비 판정과 OLD 대비 개선 신호를 구분한다.

[확인] 모든 표는 seed1이다. T는 cm, R은 degree, ADDsym는 m이다. pose 통계는 실제 F 성공 프레임 조건부이며 실패 수는 전체 등록 프레임으로 따로 표시한다. 2D median/P90은 매칭된 관측 코너를 모은 조건부 값이고 PCK10/gross20은 GT 코너 전체에 누락 페널티를 적용한 값이다.

## SYNTH_HELDOUT

[확인] 전체 1985프레임. 주 구간은 frame, 보조 구간은 scenario 재표집이다. 10,000회, seed 20260917, 동일 원 ID/단위 추출을 모든 방법·지표에 공유했다.

| 방법 | pose 성공/전체 | T mean / med / P90 | R mean / med / P90 | ADD mean / med / P90 |
|---|---:|---:|---:|---:|
| RAW | 1985/1985 | 9.2202 / 2.5036 / 21.2224 | 29.0824 / 1.2631 / 178.2558 | 0.319577 / 0.034781 / 1.414363 |
| N3 | 1985/1985 | 9.2206 / 2.4172 / 20.7454 | 29.2704 / 1.1913 / 178.3449 | 0.321640 / 0.032058 / 1.423361 |
| PoseFix | 1985/1985 | 11.7924 / 1.8903 / 19.3337 | 30.2197 / 0.9779 / 178.5652 | 0.344229 / 0.027033 / 1.454860 |
| SOFT2D_GEO | 1985/1985 | 9.4101 / 2.6695 / 20.2316 | 29.2094 / 1.3843 / 178.2491 | 0.322139 / 0.038514 / 1.418639 |
| HARD6D_GEO | 1985/1985 | 11.3948 / 2.8951 / 27.0489 | 29.8678 / 1.6824 / 175.1280 | 0.341129 / 0.043770 / 1.428497 |
| SOFT6D | 1985/1985 | 11.9895 / 2.9557 / 27.4511 | 29.8261 / 1.9777 / 174.0545 | 0.346560 / 0.047000 / 1.429918 |
| EXPECT6D | 1985/1985 | 9.2202 / 2.5036 / 21.2224 | 29.0824 / 1.2631 / 178.2558 | 0.319577 / 0.034781 / 1.414363 |
| GEO_ORACLE | 1985/1985 | 6.1219 / 1.1290 / 10.0675 | 25.3857 / 0.9190 / 168.9424 | 0.265803 / 0.016024 / 1.357334 |

| 방법 | 2D med / P90 px | PCK10 / gross20 비율 | GT / 관측 코너 | RAW good5→bad10 / bad20→good10 | NoOp |
|---|---:|---:|---:|---:|---:|
| RAW | 1.9784 / 7.0566 | 0.936390 / 0.027270 | 15658 / 15658 | 0 / 0 | NA |
| N3 | 1.7638 / 6.7106 | 0.939967 / 0.026440 | 15658 / 15658 | 7 / 0 | NA |
| PoseFix | 1.3893 / 6.3952 | 0.943032 / 0.025865 | 15658 / 15658 | 15 / 1 | NA |
| SOFT2D_GEO | 2.1889 / 7.8229 | 0.931153 / 0.027526 | 15658 / 15658 | 63 / 0 | 1036 |
| HARD6D_GEO | 2.7440 / 10.1018 | 0.897113 / 0.027717 | 15658 / 15658 | 450 / 0 | 1093 |
| SOFT6D | 2.8787 / 9.9648 | 0.900434 / 0.030080 | 15658 / 15658 | 440 / 0 | 784 |
| EXPECT6D | 1.9784 / 7.0566 | 0.936390 / 0.027270 | 15658 / 15658 | 0 / 0 | 1985 |
| GEO_ORACLE | 2.4989 / 9.0588 | 0.916975 / 0.026504 | 15658 / 15658 | 325 / 0 | 145 |

[확인] 2D evaluable 1978, matched 1978, detected 1985프레임. good5→bad10은 프레임 수가 아니라 동일 canonical GT 코너 수다.

| NEW − 대조군 | T paired mean [95% CI] cm | R paired mean [95% CI] degree | ADD paired mean [95% CI] m | C 손상 보존 실패 |
|---|---:|---:|---:|---|
| SOFT6D − N3 | 2.768941 [1.847967, 3.837297] | 0.555719 [-0.126992, 1.268191] | 0.024920 [0.014186, 0.036866] | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease |
| SOFT6D − HARD6D_GEO | 0.594753 [-0.373000, 1.640660] | -0.041730 [-0.560261, 0.489909] | 0.005431 [-0.003579, 0.015318] | gross20_nonincrease, matched_2D_median_nonincrease |
| SOFT6D − SOFT2D_GEO | 2.579427 [1.738968, 3.536771] | 0.616717 [0.145071, 1.090195] | 0.024420 [0.016161, 0.033672] | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease |
| EXPECT6D − N3 | -0.000429 [-0.386113, 0.325949] | -0.187944 [-0.755711, 0.391970] | -0.002063 [-0.009045, 0.004672] | gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease |
| EXPECT6D − HARD6D_GEO | -2.174617 [-2.921468, -1.492592] | -0.785393 [-1.293215, -0.293381] | -0.021552 [-0.028765, -0.014607] | 없음 |
| EXPECT6D − SOFT2D_GEO | -0.189944 [-0.352970, -0.048904] | -0.126946 [-0.387955, 0.124149] | -0.002562 [-0.005553, 0.000294] | 없음 |

[확인] 음의 paired mean은 낮은 오차 방향이며 CI가 0을 포함하면 확정 개선 신호로 부르지 않는다. paired median과 보조 재표집 CI, T/R/ADD median·P90 차이 및 각각의 보존 flag는 SUMMARY.json에 모두 있다.

[확인] SOFT6D: oracle action 적중 165/1985; headroom>1e-7m 1840프레임에서 rho 중앙값 0.0000, 음수 764, >1 0, 허용오차 초과 oracle bound 위반 0. RAW→oracle 최종 W/D 전환 68프레임 중 실제 최종 가설 회수 8/68.
[확인] EXPECT6D: oracle action 적중 145/1985; headroom>1e-7m 1840프레임에서 rho 중앙값 0.0000, 음수 0, >1 0, 허용오차 초과 oracle bound 위반 0. RAW→oracle 최종 W/D 전환 68프레임 중 실제 최종 가설 회수 0/68.

## REAL_DEV

[확인] 전체 319프레임. 주 구간은 session, 보조 구간은 frame 재표집이다. 10,000회, seed 20260917, 동일 원 ID/단위 추출을 모든 방법·지표에 공유했다.

| 방법 | pose 성공/전체 | T mean / med / P90 | R mean / med / P90 | ADD mean / med / P90 |
|---|---:|---:|---:|---:|
| RAW | 319/319 | 39.8968 / 7.8969 / 40.5302 | 21.3519 / 2.5389 / 86.5273 | 0.561092 / 0.087137 / 1.198890 |
| N3 | 319/319 | 39.9262 / 7.0392 / 38.1358 | 18.6999 / 2.1104 / 85.9305 | 0.540371 / 0.080549 / 1.192205 |
| PoseFix | 319/319 | 39.8187 / 7.1854 / 43.8098 | 18.1246 / 1.9540 / 85.9950 | 0.529985 / 0.078512 / 1.185009 |
| SOFT2D_GEO | 319/319 | 39.6771 / 7.2888 / 39.2921 | 21.3039 / 2.4026 / 86.1368 | 0.557689 / 0.082638 / 1.188831 |
| HARD6D_GEO | 319/319 | 40.7974 / 8.1259 / 41.7765 | 21.5081 / 3.0948 / 85.9181 | 0.568309 / 0.090390 / 1.196645 |
| SOFT6D | 319/319 | 39.5927 / 7.5986 / 48.6806 | 21.0266 / 2.8121 / 85.8032 | 0.552521 / 0.089195 / 1.177470 |
| EXPECT6D | 319/319 | 39.8968 / 7.8969 / 40.5302 | 21.3519 / 2.5389 / 86.5273 | 0.561092 / 0.087137 / 1.198890 |
| GEO_ORACLE | 319/319 | 31.4662 / 2.9126 / 24.8766 | 17.5060 / 1.9250 / 84.1527 | 0.451205 / 0.037372 / 1.153861 |

| 방법 | 2D med / P90 px | PCK10 / gross20 비율 | GT / 관측 코너 | RAW good5→bad10 / bad20→good10 | NoOp |
|---|---:|---:|---:|---:|---:|
| RAW | 6.7207 / 43.8900 | 0.634254 / 0.199280 | 2499 / 2445 | 0 / 0 | NA |
| N3 | 5.7446 / 42.3204 | 0.688275 / 0.175670 | 2499 / 2445 | 0 / 0 | NA |
| PoseFix | 5.5626 / 43.5346 | 0.685474 / 0.179672 | 2499 / 2445 | 2 / 1 | NA |
| SOFT2D_GEO | 6.4291 / 44.7775 | 0.648259 / 0.186475 | 2499 / 2445 | 6 / 1 | 82 |
| HARD6D_GEO | 8.3764 / 42.5522 | 0.563826 / 0.204482 | 2499 / 2445 | 52 / 1 | 91 |
| SOFT6D | 7.6827 / 45.2805 | 0.600240 / 0.197679 | 2499 / 2445 | 32 / 0 | 49 |
| EXPECT6D | 6.7207 / 43.8900 | 0.634254 / 0.199280 | 2499 / 2445 | 0 / 0 | 319 |
| GEO_ORACLE | 6.9304 / 44.5959 | 0.649460 / 0.184074 | 2499 / 2445 | 23 / 1 | 0 |

[확인] 2D evaluable 319, matched 311, detected 319프레임. good5→bad10은 프레임 수가 아니라 동일 canonical GT 코너 수다.

| NEW − 대조군 | T paired mean [95% CI] cm | R paired mean [95% CI] degree | ADD paired mean [95% CI] m | C 손상 보존 실패 |
|---|---:|---:|---:|---|
| SOFT6D − N3 | -0.333480 [-4.649863, 1.612451] | 2.326670 [0.760111, 3.576723] | 0.012150 [-0.035768, 0.039923] | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease |
| SOFT6D − HARD6D_GEO | -1.204734 [-2.797701, 0.269896] | -0.481578 [-1.039603, 0.030889] | -0.015788 [-0.035444, 0.000672] | matched_2D_P90_nonincrease |
| SOFT6D − SOFT2D_GEO | -0.084387 [-1.376134, 0.893444] | -0.277368 [-0.886173, 0.257172] | -0.005169 [-0.025659, 0.008855] | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease |
| EXPECT6D − N3 | -0.029411 [-3.176162, 1.567099] | 2.652003 [0.853527, 4.280609] | 0.020721 [-0.011989, 0.044875] | gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease |
| EXPECT6D − HARD6D_GEO | -0.900665 [-2.114497, 0.035630] | -0.156246 [-0.963237, 0.647966] | -0.007216 [-0.022191, 0.006306] | matched_2D_P90_nonincrease |
| EXPECT6D − SOFT2D_GEO | 0.219682 [-0.035486, 0.449001] | 0.047964 [-0.083364, 0.231795] | 0.003403 [0.002044, 0.005170] | gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease |

[확인] 음의 paired mean은 낮은 오차 방향이며 CI가 0을 포함하면 확정 개선 신호로 부르지 않는다. paired median과 보조 재표집 CI, T/R/ADD median·P90 차이 및 각각의 보존 flag는 SUMMARY.json에 모두 있다.

[확인] SOFT6D: oracle action 적중 16/319; headroom>1e-7m 319프레임에서 rho 중앙값 0.0000, 음수 142, >1 0, 허용오차 초과 oracle bound 위반 0. RAW→oracle 최종 W/D 전환 16프레임 중 실제 최종 가설 회수 3/16.
[확인] EXPECT6D: oracle action 적중 0/319; headroom>1e-7m 319프레임에서 rho 중앙값 0.0000, 음수 0, >1 0, 허용오차 초과 oracle bound 위반 0. RAW→oracle 최종 W/D 전환 16프레임 중 실제 최종 가설 회수 0/16.

## 손실 설정과 학습 관측

[확인] TRAIN 전역 scale s=0.028436526m, SOFT6D teacher tau=0.186269437. tau는 미리 고정한 1,024 TRAIN ID의 원래 2D teacher 정규화 entropy를 맞춰 정했으며 heldout/REAL 성능을 사용하지 않았다. 모델 softmax 온도는 1이다.

[확인] EXPECT6D는 고정 은행 전체에 대한 확률 가중 비용 합이다. 후보 좌표/F를 미분하거나 후보를 샘플링하지 않는다. 실패 후보의 유한 비용은 (최악 유효 regret/s)+1이라는 새 설계 근사이며 원 cost cache의 +inf를 바꾸지 않았다.

[확인] SOFT6D: 6000 update / 96000 반복 exposure; 누적 oracle action 정확도 0.064385, 마지막 100 update 정확도 0.064375, 마지막 hardJ cached ADD 평균 0.390458m, 마지막 oracle gap 평균 0.085615m. 학습 기록의 변하는 checkpoint·pre-update 반복 exposure 통계이며 최종 whole-TRAIN 정확도가 아니다.
[확인] SOFT6D 최종 checkpoint의 동일 고정 256 TRAIN ID probe: oracle action 25/256 (0.097656), NoOp 104, 선택 F 실패 0, 유효 선택 cached ADD 평균 0.411174m, gap 평균 0.076339m, 실제 expected ADD가 무한인 프레임 0, 별도 유한 페널티 근사 평균 0.397933m. probe는 F 추가 호출 없이 기존 비용을 조회하며 성능 선택에 쓰지 않았다.
[확인] EXPECT6D: 6000 update / 96000 반복 exposure; 누적 oracle action 정확도 0.056052, 마지막 100 update 정확도 0.048750, 마지막 hardJ cached ADD 평균 0.369201m, 마지막 oracle gap 평균 0.064358m. 학습 기록의 변하는 checkpoint·pre-update 반복 exposure 통계이며 최종 whole-TRAIN 정확도가 아니다.
[확인] EXPECT6D 최종 checkpoint의 동일 고정 256 TRAIN ID probe: oracle action 17/256 (0.066406), NoOp 256, 선택 F 실패 0, 유효 선택 cached ADD 평균 0.382841m, gap 평균 0.048006m, 실제 expected ADD가 무한인 프레임 0, 별도 유한 페널티 근사 평균 0.382841m. probe는 F 추가 호출 없이 기존 비용을 조회하며 성능 선택에 쓰지 않았다.

## 해석 범위

[확인] 한 training seed와 이미 여러 번 본 DEV의 진단이다. 합성 frame 구간은 원래 singleton/pair scenario 의존성을 모두 해결하지 않으므로 scenario 보조 구간을 함께 제공한다. REAL session 구간은 13개 세션의 재표집 불확실성이며 새로운 데이터나 training seed 변동의 구간이 아니다. 다중 비교 보정은 하지 않았다.

[확인] REAL 319장은 2D/치수 기반 재구성 pose reference를 사용한다. 독립 물리 계측은 0쌍이고 BLOCKED_DATA 상태이므로 실제 물리 자세의 개선을 확인한 결과가 아니다.

[확인] 같은 후보 은행·scorer·초기값·순서·optimizer·6000 update에서 손실만 바꿨다. soft2D→hard6D는 목적 정렬과 target hardness가 함께 바뀌어 두 원인을 독립 식별하지 못한다. 이번 SOFT6D/EXPECT6D의 tau·scale·실패 페널티도 신규 고정 설계이다. 현재 결과로 전체 6D 손실군의 불가능, 모델 크기나 최적화 실패의 원인을 단정하지 않는다.

[확인] 추가 seed/손실/규모 확대·subgroup·그림·runtime benchmark·원고 작업은 실행하지 않았다. 실행 시간은 실제 작업 wall time이며 모델 latency 비교가 아니다. private 데이터·feature·가중치와 외부 원본 cache가 필요하므로 공개 checkout만으로 완전 재현된다고 주장하지 않는다.

## 핵심 판독

[확인] EXPECT6D는 SYNTH_HELDOUT 1,985/1,985장과 REAL_DEV 319/319장, 합계 **2,304/2,304장에서 NoOp(index 0)**를 선택했다. 기존 은행의 native NoOp 좌표를 그대로 최종 F에 전달했고 RAW와 canonical 코너 오차·최종 가설·T/R/ADD가 원행 단위로 정확히 일치한다. 따라서 이 평가에서 얻은 OLD_ONLY_GAIN은 새로운 보정 동작의 이득이 아니라 RAW를 유지해 이전 hard6D-GEO의 손상을 피한 결과다.

[확인] 아래 REAL 수치는 모두 seed1이다. ADD 차이는 NEW−N3의 동일 프레임 평균이며 구간은 전체 319장·13개 원래 세션을 공유 재표집한 주 95% CI다.

| 방법 | REAL T 중앙값 cm | REAL R 중앙값 degree | REAL ADD 평균차이 − N3 m [session 95% CI] |
|---|---:|---:|---:|
| N3 | 7.039188 | 2.110365 | 대조군 |
| SOFT6D | 7.598561 | 2.812113 | +0.0121497 [-0.0357682, 0.0399226] |
| EXPECT6D | 7.896852 | 2.538878 | +0.0207213 [-0.0119885, 0.0448749] |

[확인] 두 새 방법 모두 REAL T/R 중앙값이 N3보다 높으며, N3 대비 ADD 평균차이는 양수이고 주 CI가 0을 포함한다. EXPECT6D의 OLD_ONLY_GAIN 근거는 합성에서 이전 hard6D-GEO 대비 ADD 평균차이 -0.0215518m [frame 95% CI -0.0287649, -0.0146072]라는 제한된 신호다. REAL의 이전 hard6D 대비 ADD 구간도 0을 포함하므로 이를 REAL 개선이나 N3 개선으로 확장하지 않는다. SOFT6D는 NO_CLEAR_GAIN이다. 한 seed·반복 사용 DEV의 이번 두 고정 손실에 대한 결과이며 모든 6D 손실군의 실패를 뜻하지 않는다.
