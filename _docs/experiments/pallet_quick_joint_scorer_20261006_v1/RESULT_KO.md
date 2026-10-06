# 같은 SOFT6D 감독의 LOCAL_CAP / JOINT8 비교

[확인] N3보다 REAL T/R 중앙값이 함께 낮고 모든 고정 보존 조건을 만족한 방법: 없음. 한 seed의 반복 사용 DEV 진단이다.

[확인] LOCAL_CAP: **NO_N3_GAIN**; JOINT8: **NO_N3_GAIN**. CI가 0을 포함하면 방향 신호·미확증으로 표시하며 안정적 개선으로 부르지 않는다. 개별 개선·손상 flag는 최종 라벨과 함께 읽는다.

[확인] REAL의 동급 용량 비교에서 LOCAL_CAP가 JOINT8보다 T/R 중앙값·전체 보존 조건에 모두 우세한가: False. JOINT8가 LOCAL_CAP보다 같은 조건에 우세한가: False.

## SYNTH_HELDOUT

[확인] 전체 1985장; 주 bootstrap frame. 동일 원 ID/원 세션·시나리오의 draw를 공유한 10,000회, seed20260917이다. 합성은 원 scenario, REAL은 frame 보조 구간도 RESULTS.json에 남긴다.

| seed1 방법 | T mean / med / P90 cm | R mean / med / P90 degree | ADD mean / med / P90 m | F 성공/전체 | NoOp |
|---|---:|---:|---:|---:|---:|
| RAW | 9.2202 / 2.5036 / 21.2224 | 29.0824 / 1.2631 / 178.2558 | 0.319577 / 0.034781 / 1.414363 | 1985/1985 | NA |
| N3 | 9.2206 / 2.4172 / 20.7454 | 29.2704 / 1.1913 / 178.3449 | 0.321640 / 0.032058 / 1.423361 | 1985/1985 | NA |
| PoseFix | 11.7924 / 1.8903 / 19.3337 | 30.2197 / 0.9779 / 178.5652 | 0.344229 / 0.027033 / 1.454860 | 1985/1985 | NA |
| SOFT2D_GEO | 9.4101 / 2.6695 / 20.2316 | 29.2094 / 1.3843 / 178.2491 | 0.322139 / 0.038514 / 1.418639 | 1985/1985 | 1036 |
| HARD6D_GEO | 11.3948 / 2.8951 / 27.0489 | 29.8678 / 1.6824 / 175.1280 | 0.341129 / 0.043770 / 1.428497 | 1985/1985 | 1093 |
| SOFT6D | 11.9895 / 2.9557 / 27.4511 | 29.8261 / 1.9777 / 174.0545 | 0.346560 / 0.047000 / 1.429918 | 1985/1985 | 784 |
| EXPECT6D | 9.2202 / 2.5036 / 21.2224 | 29.0824 / 1.2631 / 178.2558 | 0.319577 / 0.034781 / 1.414363 | 1985/1985 | 1985 |
| LOCAL_CAP | 11.1239 / 3.2400 / 25.2055 | 29.7800 / 1.9199 / 176.9496 | 0.339735 / 0.049369 / 1.405046 | 1985/1985 | 783 |
| JOINT8 | 11.3660 / 3.1043 / 26.7763 | 30.3722 / 1.9429 / 172.6543 | 0.346321 / 0.048396 / 1.413340 | 1985/1985 | 726 |
| GEO_ORACLE | 6.1219 / 1.1290 / 10.0675 | 25.3857 / 0.9190 / 168.9424 | 0.265803 / 0.016024 / 1.357334 | 1985/1985 | 145 |

| 방법 | 2D med / P90 px | PCK10 / gross20 | GT/관측 코너 | RAW good5→bad10 / bad20→good10 | RAW native / pose metric 같음 |
|---|---:|---:|---:|---:|---:|
| RAW | 1.9784 / 7.0566 | 0.936390 / 0.027270 | 15658/15658 | 0/0 | NA / 1985 |
| N3 | 1.7638 / 6.7106 | 0.939967 / 0.026440 | 15658/15658 | 7/0 | NA / 0 |
| PoseFix | 1.3893 / 6.3952 | 0.943032 / 0.025865 | 15658/15658 | 15/1 | NA / 0 |
| SOFT2D_GEO | 2.1889 / 7.8229 | 0.931153 / 0.027526 | 15658/15658 | 63/0 | NA / 1036 |
| HARD6D_GEO | 2.7440 / 10.1018 | 0.897113 / 0.027717 | 15658/15658 | 450/0 | NA / 1093 |
| SOFT6D | 2.8787 / 9.9648 | 0.900434 / 0.030080 | 15658/15658 | 440/0 | NA / 784 |
| EXPECT6D | 1.9784 / 7.0566 | 0.936390 / 0.027270 | 15658/15658 | 0/0 | NA / 1985 |
| LOCAL_CAP | 3.1281 / 10.5147 | 0.887023 / 0.030591 | 15658/15658 | 595/0 | 783 / 783 |
| JOINT8 | 2.9846 / 10.2044 | 0.893920 / 0.029378 | 15658/15658 | 511/0 | 726 / 726 |
| GEO_ORACLE | 2.4989 / 9.0588 | 0.916975 / 0.026504 | 15658/15658 | 325/0 | NA / 145 |

[확인] pose 분모는 1985장, 2D evaluable 1978·matched 1978장이다. 2D median/P90은 매칭 관측 코너 조건부, PCK/gross20은 전체 GT 코너+결측 페널티. 손상·복구는 프레임 수가 아닌 같은 canonical GT 코너 수다. old RAW 저장행에 R/t/native 좌표가 없어 독립 R/t 재대조는 NA이며, pose 같음 표는 저장 T/R/ADD·최종 가설의 정확 일치다. 새 native 같음은 frame.q와 직접 비교했고 동일 native·동일 F의 pose 동일성은 별도 계약 근거다.

| NEW−대조 | T paired mean [95% CI] cm | R paired mean [95% CI] degree | ADD paired mean [95% CI] m | T/R/ADD difference-of-medians | T/R 동시개선 / 혼합 / 동시악화 / 동률포함 |
|---|---:|---:|---:|---:|---:|
| JOINT8_minus_LOCAL_CAP | 0.242106 [-0.492530, 0.977269] | 0.592203 [0.021271, 1.168906] | 0.006586 [-0.000893, 0.014046] | -0.135713 / 0.022992 / -0.000973 | 249 / 395 / 221 / 1120 |
| LOCAL_CAP_minus_N3 | 1.903297 [1.222306, 2.638218] | 0.509635 [-0.188420, 1.216132] | 0.018095 [0.009604, 0.026850] | 0.822801 / 0.728605 / 0.017311 | 357 / 873 / 755 / 0 |
| JOINT8_minus_N3 | 2.145404 [1.392528, 2.933219] | 1.101837 [0.320985, 1.880058] | 0.024681 [0.014888, 0.034480] | 0.687088 / 0.751597 / 0.016338 | 376 / 870 / 739 / 0 |
| LOCAL_CAP_minus_PoseFix | -0.668541 [-2.142611, 0.698063] | -0.439682 [-1.443471, 0.504222] | -0.004494 [-0.019424, 0.009651] | 1.349682 / 0.942058 / 0.022336 | 264 / 823 / 898 / 0 |
| JOINT8_minus_PoseFix | -0.426435 [-1.904293, 0.951907] | 0.152521 [-0.891292, 1.133918] | 0.002092 [-0.013275, 0.016683] | 1.213969 / 0.965050 / 0.021363 | 279 / 823 / 883 / 0 |

[확인] 음의 paired 차이는 낮은 오차 방향이다. marginal 중앙값 감소와 같은 프레임 T/R 동시 개선은 별개다. mean의 CI가 0을 포함한 경우 확정 개선이 아니며 pair별 pose·corner 보존 flag 및 악화 지표는 아래와 RESULTS.json에 유지했다.

| 비교 | T/R 중앙값 둘 다 낮음 | pose 보존 위반 | corner 보존 위반 | TRADEOFF |
|---|---|---|---|---|
| JOINT8_minus_LOCAL_CAP | False | ADD_P90_nonworse, T_P90_nonworse | 없음 | True |
| LOCAL_CAP_minus_N3 | False | ADD_median_nonworse, T_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | False |
| JOINT8_minus_N3 | False | ADD_median_nonworse, T_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | False |
| LOCAL_CAP_minus_PoseFix | False | ADD_median_nonworse, T_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | True |
| JOINT8_minus_PoseFix | False | ADD_median_nonworse, T_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | True |

[확인] LOCAL_CAP: RAW native 동일 783/1985, RAW 저장 pose 지표·가설 동일 783/1985. oracle action 174/1985, headroom>1e-7m 1840장에서 rho 중앙값 0.0000·음수 801·>1 0·오차허용 초과 oracle 위반 0. RAW→oracle 실제 최종 W/D 전환 68장 중 14/68 회수. 생성 가설 라벨로 회수를 세지 않았다.

[확인] JOINT8: RAW native 동일 726/1985, RAW 저장 pose 지표·가설 동일 726/1985. oracle action 184/1985, headroom>1e-7m 1840장에서 rho 중앙값 0.0000·음수 776·>1 0·오차허용 초과 oracle 위반 0. RAW→oracle 실제 최종 W/D 전환 68장 중 11/68 회수. 생성 가설 라벨로 회수를 세지 않았다.

## REAL_DEV

[확인] 전체 319장; 주 bootstrap session. 동일 원 ID/원 세션·시나리오의 draw를 공유한 10,000회, seed20260917이다. 합성은 원 scenario, REAL은 frame 보조 구간도 RESULTS.json에 남긴다.

| seed1 방법 | T mean / med / P90 cm | R mean / med / P90 degree | ADD mean / med / P90 m | F 성공/전체 | NoOp |
|---|---:|---:|---:|---:|---:|
| RAW | 39.8968 / 7.8969 / 40.5302 | 21.3519 / 2.5389 / 86.5273 | 0.561092 / 0.087137 / 1.198890 | 319/319 | NA |
| N3 | 39.9262 / 7.0392 / 38.1358 | 18.6999 / 2.1104 / 85.9305 | 0.540371 / 0.080549 / 1.192205 | 319/319 | NA |
| PoseFix | 39.8187 / 7.1854 / 43.8098 | 18.1246 / 1.9540 / 85.9950 | 0.529985 / 0.078512 / 1.185009 | 319/319 | NA |
| SOFT2D_GEO | 39.6771 / 7.2888 / 39.2921 | 21.3039 / 2.4026 / 86.1368 | 0.557689 / 0.082638 / 1.188831 | 319/319 | 82 |
| HARD6D_GEO | 40.7974 / 8.1259 / 41.7765 | 21.5081 / 3.0948 / 85.9181 | 0.568309 / 0.090390 / 1.196645 | 319/319 | 91 |
| SOFT6D | 39.5927 / 7.5986 / 48.6806 | 21.0266 / 2.8121 / 85.8032 | 0.552521 / 0.089195 / 1.177470 | 319/319 | 49 |
| EXPECT6D | 39.8968 / 7.8969 / 40.5302 | 21.3519 / 2.5389 / 86.5273 | 0.561092 / 0.087137 / 1.198890 | 319/319 | 319 |
| LOCAL_CAP | 40.5577 / 8.4509 / 42.4172 | 20.8146 / 2.4497 / 85.2061 | 0.561086 / 0.091055 / 1.187866 | 319/319 | 47 |
| JOINT8 | 40.5769 / 8.0642 / 40.8836 | 21.6164 / 3.1263 / 85.6860 | 0.562611 / 0.093102 / 1.187641 | 319/319 | 49 |
| GEO_ORACLE | 31.4662 / 2.9126 / 24.8766 | 17.5060 / 1.9250 / 84.1527 | 0.451205 / 0.037372 / 1.153861 | 319/319 | 0 |

| 방법 | 2D med / P90 px | PCK10 / gross20 | GT/관측 코너 | RAW good5→bad10 / bad20→good10 | RAW native / pose metric 같음 |
|---|---:|---:|---:|---:|---:|
| RAW | 6.7207 / 43.8900 | 0.634254 / 0.199280 | 2499/2445 | 0/0 | NA / 319 |
| N3 | 5.7446 / 42.3204 | 0.688275 / 0.175670 | 2499/2445 | 0/0 | NA / 0 |
| PoseFix | 5.5626 / 43.5346 | 0.685474 / 0.179672 | 2499/2445 | 2/1 | NA / 0 |
| SOFT2D_GEO | 6.4291 / 44.7775 | 0.648259 / 0.186475 | 2499/2445 | 6/1 | NA / 82 |
| HARD6D_GEO | 8.3764 / 42.5522 | 0.563826 / 0.204482 | 2499/2445 | 52/1 | NA / 91 |
| SOFT6D | 7.6827 / 45.2805 | 0.600240 / 0.197679 | 2499/2445 | 32/0 | NA / 49 |
| EXPECT6D | 6.7207 / 43.8900 | 0.634254 / 0.199280 | 2499/2445 | 0/0 | NA / 319 |
| LOCAL_CAP | 9.1444 / 43.8297 | 0.535814 / 0.214486 | 2499/2445 | 101/1 | 47 / 47 |
| JOINT8 | 8.5786 / 44.0157 | 0.565426 / 0.210884 | 2499/2445 | 55/0 | 49 / 49 |
| GEO_ORACLE | 6.9304 / 44.5959 | 0.649460 / 0.184074 | 2499/2445 | 23/1 | NA / 0 |

[확인] pose 분모는 319장, 2D evaluable 319·matched 311장이다. 2D median/P90은 매칭 관측 코너 조건부, PCK/gross20은 전체 GT 코너+결측 페널티. 손상·복구는 프레임 수가 아닌 같은 canonical GT 코너 수다. old RAW 저장행에 R/t/native 좌표가 없어 독립 R/t 재대조는 NA이며, pose 같음 표는 저장 T/R/ADD·최종 가설의 정확 일치다. 새 native 같음은 frame.q와 직접 비교했고 동일 native·동일 F의 pose 동일성은 별도 계약 근거다.

| NEW−대조 | T paired mean [95% CI] cm | R paired mean [95% CI] degree | ADD paired mean [95% CI] m | T/R/ADD difference-of-medians | T/R 동시개선 / 혼합 / 동시악화 / 동률포함 |
|---|---:|---:|---:|---:|---:|
| JOINT8_minus_LOCAL_CAP | 0.019202 [-0.598621, 0.591049] | 0.801734 [-0.391128, 1.868189] | 0.001526 [-0.010857, 0.010761] | -0.386684 / 0.676636 / 0.002047 | 45 / 90 / 48 / 136 |
| LOCAL_CAP_minus_N3 | 0.631545 [-3.413068, 2.945180] | 2.114749 [0.717879, 3.521855] | 0.020715 [-0.023392, 0.047050] | 1.411693 / 0.339322 / 0.010505 | 43 / 124 / 152 / 0 |
| JOINT8_minus_N3 | 0.650747 [-3.425327, 3.316668] | 2.916483 [1.165139, 4.325981] | 0.022240 [-0.023802, 0.050143] | 1.025010 / 1.015958 / 0.012553 | 45 / 129 / 145 / 0 |
| LOCAL_CAP_minus_PoseFix | 0.739056 [-1.800563, 2.434535] | 2.690047 [0.715407, 5.157280] | 0.031101 [-0.009192, 0.063604] | 1.265456 / 0.495647 / 0.012543 | 60 / 109 / 150 / 0 |
| JOINT8_minus_PoseFix | 0.758258 [-1.939718, 2.845935] | 3.491781 [1.636687, 5.479148] | 0.032627 [-0.009505, 0.064679] | 0.878773 / 1.172283 / 0.014590 | 59 / 116 / 144 / 0 |

[확인] 음의 paired 차이는 낮은 오차 방향이다. marginal 중앙값 감소와 같은 프레임 T/R 동시 개선은 별개다. mean의 CI가 0을 포함한 경우 확정 개선이 아니며 pair별 pose·corner 보존 flag 및 악화 지표는 아래와 RESULTS.json에 유지했다.

| 비교 | T/R 중앙값 둘 다 낮음 | pose 보존 위반 | corner 보존 위반 | TRADEOFF |
|---|---|---|---|---|
| JOINT8_minus_LOCAL_CAP | False | ADD_median_nonworse, R_P90_nonworse | matched_2D_P90_nonincrease | True |
| LOCAL_CAP_minus_N3 | False | ADD_median_nonworse, T_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | False |
| JOINT8_minus_N3 | False | ADD_median_nonworse, T_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | False |
| LOCAL_CAP_minus_PoseFix | False | ADD_median_nonworse, ADD_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | False |
| JOINT8_minus_PoseFix | False | ADD_median_nonworse, ADD_P90_nonworse | RAW_good5_to_bad10_nonincrease, gross20_nonincrease, full_PCK10_non_decrease, matched_2D_median_nonincrease, matched_2D_P90_nonincrease | False |

[확인] LOCAL_CAP: RAW native 동일 47/319, RAW 저장 pose 지표·가설 동일 47/319. oracle action 8/319, headroom>1e-7m 319장에서 rho 중앙값 -0.0230·음수 168·>1 0·오차허용 초과 oracle 위반 0. RAW→oracle 실제 최종 W/D 전환 16장 중 5/16 회수. 생성 가설 라벨로 회수를 세지 않았다.

[확인] JOINT8: RAW native 동일 49/319, RAW 저장 pose 지표·가설 동일 49/319. oracle action 16/319, headroom>1e-7m 319장에서 rho 중앙값 -0.0177·음수 167·>1 0·오차허용 초과 oracle 위반 0. RAW→oracle 실제 최종 W/D 전환 16장 중 3/16 회수. 생성 가설 라벨로 회수를 세지 않았다.

## 실행과 해석 범위

[확인] 저장된 동일 FP32 SOFT6D target을 재사용해 각각 원래 seed1 초기값부터 6000 update를 수행했다. 두 추가 head는 실제 각 4680 parameter이며 공통 projection과 기존 frontend가 같다. 파라미터 수가 같아도 연산량·메모리·최적화 조건까지 같다고 주장하지 않는다. NoOp descriptor는 RAW 위치에서 새 patch를 뽑은 것이 아니라 기존 이동후보 pooled/coverage 집계와 null_pool을 사용했다.

[확인] LOCAL_CAP: 6000 update/96000 exposure, 실제 학습 루프 395.118s, 마지막 checkpoint 25c9678ba90013f2ca148c16a282a1182702e3f84f66c5f7a067cd5020eab1da. 동일 고정 256 TRAIN ID probe 요약: {'frames': 256, 'NoOp': 109, 'oracle_exact': 21, 'selected_F_failures': 0, 'mean_gap_m': 0.0738597296404658, 'expected_infinite_frames': 0, 'mean_CE': 5.1129901660606265, 'mean_target_entropy': 3.6619175251677234, 'mean_CE_minus_target_entropy': 1.451072639785707, 'mean_NoOp_probability': 0.01643214371142676, 'mean_entropy': 5.114735953057471}. 기존 비용만 조회하고 F 추가0이며 전체 TRAIN 정확도·모델 선택 근거로 확대하지 않는다.
[확인] JOINT8: 6000 update/96000 exposure, 실제 학습 루프 392.565s, 마지막 checkpoint a75670f698c388b5f8436b1386d64eeaae35022f4d295e9bfa1fefc8b0e3af0c. 동일 고정 256 TRAIN ID probe 요약: {'frames': 256, 'NoOp': 103, 'oracle_exact': 23, 'selected_F_failures': 0, 'mean_gap_m': 0.081096583677519, 'expected_infinite_frames': 0, 'mean_CE': 5.0839618649333715, 'mean_target_entropy': 3.6619175251677234, 'mean_CE_minus_target_entropy': 1.4220443405210972, 'mean_NoOp_probability': 0.01686645875497561, 'mean_entropy': 5.101253742173845}. 기존 비용만 조회하고 F 추가0이며 전체 TRAIN 정확도·모델 선택 근거로 확대하지 않는다.

[확인] 한 seed와 반복 사용 DEV이며 다중 비교 보정이 없다. REAL은 2D 주석·치수에 의존한 재구성 참조다. 독립 물리 실측은 0쌍/BLOCKED_DATA이므로 실제 물리 T/R 개선의 확증이 아니다. 이번 두 고정 구조의 결과로 8코너 관계의 필요성이나 6D 학습의 불가능을 입증하지 않는다. 추가 seed·구조·손실·온도 검색, 비용 재계산, backbone 재추출, bank 재생성, 원고/PDF/bib 변경·빌드는 0이다.
