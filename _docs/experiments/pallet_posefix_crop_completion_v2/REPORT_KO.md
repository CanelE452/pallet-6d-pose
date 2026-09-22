# Crop completion v2 — 실제 학습·네 조건 비교

신규학습 **C 1개 ×300update**. 기전 **SUPPORT_ONLY**, 시스템 **TRADE_OFF**. 과거 E0 실패는 변경하지 않았으며 이번은 별도로 승인된 탐색 실험이다.

**결론: 이번 crop 확대만으로 큰 코너 오류가 해결되지는 않았다.** 새로 도달 가능해진 14개는 10px뿐 아니라 20px 기준에서도 복구가 0개다. primary 정답률은 50.35%→50.35%로, 새 정답 27개와 기존 정답 손실 27개가 상쇄됐다. C를 기존 최종 모델로 승격하지 않는다. [완료·검증 및 공개 범위](COMPLETION.md).

| 조건 | primary ≤10 | N14 | I11 | E3 | U15 | R134 | FULL 정답손실 | N2 정답손실 | GREEN ≤10 | source clean ≤10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 359 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 539 | 1847 |
| B | 307 | 0 | 0 | 0 | 0 | 0 | 88 | 88 | 560 | 1735 |
| C | 359 | 0 | 0 | 0 | 0 | 0 | 27 | 32 | 560 | 1817 |
| D | 354 | 0 | 0 | 0 | 0 | 11 | 51 | 57 | 557 | 1735 |

A=기존FULL125/infer125, B=같은weight/infer150, C=PRIOR1부터 새FULL150/infer150, D=Cweight/infer125. B/D는 진단조건이며 DEV 최고값으로 최종 모델을 선택하지 않았다.

주 분모: primary 93장/713코너. N14 등 이름은 기존 고정 집합이며 실제 counts={'H163': 163, 'U29': 29, 'N14': 14, 'U15': 15, 'R134': 134, 'I11': 11, 'E3': 3}. GREEN은 수동좌표집합으로 분리. source는 기존1.25 고정2022코너, 새지원은 별도 표. 겹치는 모집단을 독립 재현으로 세지 않는다.

## 핵심 질문과 판단

- 범위를 열어 준 14개 중 실제 ≤10 정답: A 0, B 0, C 0, D 0. 도달가능과 실제복구를 구분한다.
- C−A 전체 정답 순변화 +0개. A 정답손실 27, 새획득 27이며 BG−GB 일치.
- B−A는 같은 weight의 입력확대 효과. C−B는 동일확대입력에서 각자 PRIOR1부터 같은300step 학습한 weight 차이이며 FULL을 추가학습한 것이 아니다. C/D 차이는 아래 네조건 표에 그대로 보고한다.
- C의 고정 hard 분류: {'NO_TESTED_PEAK': 130, 'PEAK_PRESENT_DECODE_WRONG': 18, 'SUPPORT_FAIL': 15}. reachable 148개 중 top5 oracle rescue 18개 (12.16%). argmax rescue 1개. 다른채널 의심 20개는 중복 보조태그이며 합산하지 않는다.
- 고정 real OCC TRAIN probe의 PRIOR1_150→C: PCK10 95.31→100.00%, task(CE+coord) 2.1690→1.1735. 이것은 TRAIN 적합도이지 독립 일반화 성공이 아니다.
- 다른채널 후보는 승인순열 대응 가능 여부와 모델 전체물체 branch 일치 여부를 구분했다. 자유 점별 재대응/새 selector/정답변경 없음. 후보가 없다고 RGB 정보가 없거나 아키텍처가 원천적으로 불가능하다고 결론내리지 않는다.
- 사람검토 REVIEW_PENDING. 기존 U29 연결 1, 같은양식 추가 28코너. 새 GT/가시성 응답 생성 없음. 기존reference의 물리 좌표정확도와 external/self-occlusion subtype는 미확인이다.

## 전체 성능

| 모집단 | 조건 | PCK5% | PCK10% | PCK20% | matched med | matched P90 | >20 수 |
|---|---|---:|---:|---:|---:|---:|---:|
| PLASTIC194 | A | 26.09 | 57.56 | 75.53 | 7.53 | 41.27 | 241 |
| PLASTIC194 | B | 25.58 | 50.86 | 69.54 | 8.87 | 44.36 | 300 |
| PLASTIC194 | C | 30.25 | 57.97 | 76.55 | 7.79 | 40.77 | 231 |
| PLASTIC194 | D | 31.88 | 57.16 | 73.60 | 7.54 | 43.82 | 260 |
| GREEN150_MANUAL | A | 56.53 | 79.15 | 88.84 | 4.04 | 10.39 | 76 |
| GREEN150_MANUAL | B | 56.24 | 82.23 | 88.84 | 3.78 | 9.20 | 76 |
| GREEN150_MANUAL | C | 58.30 | 82.23 | 89.28 | 3.90 | 9.46 | 73 |
| GREEN150_MANUAL | D | 60.94 | 81.79 | 88.99 | 3.72 | 9.43 | 75 |
| DEV72_REFERENCE_UNKNOWN | A | 30.09 | 60.90 | 75.68 | 7.24 | 52.36 | 135 |
| DEV72_REFERENCE_UNKNOWN | B | 29.37 | 52.43 | 70.81 | 9.09 | 51.66 | 162 |
| DEV72_REFERENCE_UNKNOWN | C | 36.04 | 60.54 | 76.40 | 7.49 | 52.17 | 131 |
| DEV72_REFERENCE_UNKNOWN | D | 37.12 | 61.26 | 75.86 | 6.85 | 58.54 | 134 |
| PRIMARY_OCC96 | A | 20.90 | 50.35 | 67.88 | 8.55 | 53.18 | 229 |
| PRIMARY_OCC96 | B | 19.50 | 43.06 | 62.69 | 10.83 | 54.16 | 266 |
| PRIMARY_OCC96 | C | 24.82 | 50.35 | 68.02 | 8.80 | 52.47 | 228 |
| PRIMARY_OCC96 | D | 25.25 | 49.65 | 66.90 | 9.22 | 59.54 | 236 |
| CLEAN_NONCAD69 | A | 58.09 | 90.44 | 98.53 | 4.22 | 9.43 | 2 |
| CLEAN_NONCAD69 | B | 66.91 | 88.24 | 97.79 | 3.49 | 10.30 | 3 |
| CLEAN_NONCAD69 | C | 66.18 | 90.44 | 99.26 | 3.47 | 9.74 | 1 |
| CLEAN_NONCAD69 | D | 67.65 | 90.44 | 96.32 | 3.49 | 8.95 | 5 |
| CAD18 | A | 21.32 | 62.50 | 92.65 | 7.86 | 18.54 | 10 |
| CAD18 | B | 16.18 | 54.41 | 77.21 | 8.83 | 33.94 | 31 |
| CAD18 | C | 22.79 | 65.44 | 98.53 | 8.08 | 16.75 | 2 |
| CAD18 | D | 30.88 | 63.24 | 86.03 | 7.56 | 24.60 | 19 |
| SELECTED_CLEAN8 | A | 28.12 | 73.44 | 93.75 | 7.39 | 18.43 | 4 |
| SELECTED_CLEAN8 | B | 15.62 | 60.94 | 73.44 | 7.96 | 42.75 | 17 |
| SELECTED_CLEAN8 | C | 25.00 | 59.38 | 100.00 | 8.30 | 15.53 | 0 |
| SELECTED_CLEAN8 | D | 39.06 | 70.31 | 84.38 | 6.07 | 24.01 | 10 |

검출/매칭/bbox/score/후보 identity 고정. 실패8장/54코너의800px penalty는 official PCK에서 유지하고 matched median/P90에서만 제외한다. 가림태그93장 정확도를 외부가림 코너 정확도로 부르지 않는다.

## 고정 R0 band — primary

| band | 코너수 | A≤10 | B≤10 | C≤10 | D≤10 |
|---|---:|---:|---:|---:|---:|
| B0 | 116 | 115 | 102 | 113 | 95 |
| B1 | 192 | 166 | 135 | 172 | 161 |
| B2 | 155 | 72 | 63 | 68 | 79 |
| B3 | 99 | 6 | 7 | 6 | 18 |
| B4 | 97 | 0 | 0 | 0 | 1 |
| B5_MATCH_FAILURE | 54 | 0 | 0 | 0 | 0 |

## 보존 판정

| 조건 | 충족 |
|---|---|
| N14_gain_atleast1 | False |
| primary_no_loss | True |
| BASE_loss_no_worse | False |
| N2_loss_no_worse | False |
| GREEN_no_loss | True |
| CLEAN_no_loss | True |
| source_clean_no_loss | False |
| source_stress_no_loss | True |
| P90_no_worse | True |

## Source: 고정분모와 신규지원 분리

| 조건 | 입력 | 집합 | n | ≤10 | PCK10% | PCK20% | median | P90 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | clean | COMMON_SUPPORT | 2022 | 1847 | 91.35 | 96.44 | 1.88 | 9.12 |
| A | clean | NEW_SUPPORT | 6 | 0 | 0.00 | 50.00 | 20.78 | 27.33 |
| A | stress | COMMON_SUPPORT | 2022 | 1369 | 67.71 | 85.36 | 5.02 | 25.45 |
| A | stress | NEW_SUPPORT | 6 | 0 | 0.00 | 33.33 | 21.37 | 27.59 |
| B | clean | COMMON_SUPPORT | 2022 | 1735 | 85.81 | 94.16 | 2.79 | 13.55 |
| B | clean | NEW_SUPPORT | 6 | 2 | 33.33 | 50.00 | 19.90 | 28.80 |
| B | stress | COMMON_SUPPORT | 2022 | 1094 | 54.10 | 75.91 | 8.79 | 33.85 |
| B | stress | NEW_SUPPORT | 6 | 1 | 16.67 | 50.00 | 20.50 | 28.08 |
| C | clean | COMMON_SUPPORT | 2022 | 1817 | 89.86 | 96.29 | 2.20 | 10.12 |
| C | clean | NEW_SUPPORT | 6 | 0 | 0.00 | 50.00 | 19.97 | 28.30 |
| C | stress | COMMON_SUPPORT | 2022 | 1418 | 70.13 | 86.20 | 4.81 | 24.39 |
| C | stress | NEW_SUPPORT | 6 | 0 | 0.00 | 50.00 | 20.71 | 27.95 |
| D | clean | COMMON_SUPPORT | 2022 | 1735 | 85.81 | 93.62 | 2.57 | 13.62 |
| D | clean | NEW_SUPPORT | 6 | 0 | 0.00 | 16.67 | 24.64 | 32.44 |
| D | stress | COMMON_SUPPORT | 2022 | 1122 | 55.49 | 74.63 | 8.19 | 35.59 |
| D | stress | NEW_SUPPORT | 6 | 0 | 0.00 | 16.67 | 25.09 | 31.51 |

## TRAIN-fit (독립 평가 아님)

| 조건 | probe | common n | PCK10% | median | heatmap loss | coordinate loss | 신규감독 n |
|---|---|---:|---:|---:|---:|---:|---:|
| PRIOR1_125 | real_clean | 128 | 100.00 | 0.62 | 1.0625 | 0.1321 | 0 |
| PRIOR1_125 | real_OCC | 128 | 96.09 | 1.05 | 1.7626 | 0.3628 | 0 |
| PRIOR1_125 | source_clean | 148 | 85.14 | 1.60 | 2.4236 | 2.6542 | 4 |
| PRIOR1_125 | source_stress | 148 | 48.65 | 10.18 | 4.8399 | 3.3416 | 4 |
| PRIOR1_150 | real_clean | 128 | 100.00 | 1.45 | 1.2435 | 0.2116 | 0 |
| PRIOR1_150 | real_OCC | 128 | 95.31 | 1.68 | 1.8043 | 0.3647 | 0 |
| PRIOR1_150 | source_clean | 148 | 83.78 | 2.14 | 2.5841 | 2.5237 | 4 |
| PRIOR1_150 | source_stress | 148 | 45.95 | 12.01 | 5.0838 | 3.1217 | 4 |
| A | real_clean | 128 | 100.00 | 0.30 | 0.9073 | 0.0558 | 0 |
| A | real_OCC | 128 | 100.00 | 0.47 | 1.0257 | 0.0990 | 0 |
| A | source_clean | 148 | 83.11 | 2.08 | 2.7724 | 2.6969 | 4 |
| A | source_stress | 148 | 72.97 | 3.15 | 2.9312 | 1.9953 | 4 |
| C | real_clean | 128 | 100.00 | 0.53 | 0.9939 | 0.0740 | 0 |
| C | real_OCC | 128 | 100.00 | 0.62 | 1.0756 | 0.0979 | 0 |
| C | source_clean | 148 | 83.78 | 2.17 | 2.8807 | 2.5125 | 4 |
| C | source_stress | 148 | 75.00 | 2.77 | 2.8899 | 1.9295 | 4 |

## Heatmap 분해

| 집합 | 조건 | n | support fail | decoded correct | peak 있으나decode실패 | 시험한peak없음 | identity보조 |
|---|---|---:|---:|---:|---:|---:|---:|
| H163 | A | 163 | 29 | 0 | 14 | 120 | 23 |
| H163 | B | 163 | 15 | 0 | 29 | 119 | 20 |
| H163 | C | 163 | 15 | 0 | 18 | 130 | 20 |
| H163 | D | 163 | 29 | 11 | 23 | 100 | 14 |
| C_REMAINING_HARD | A | 160 | 29 | 0 | 13 | 118 | 21 |
| C_REMAINING_HARD | B | 160 | 15 | 1 | 26 | 118 | 20 |
| C_REMAINING_HARD | C | 160 | 15 | 0 | 17 | 128 | 20 |
| C_REMAINING_HARD | D | 160 | 29 | 9 | 22 | 100 | 15 |

5×5 local maxima와 고정top5 규칙 유지. native↔canonical은 기전진단에서 과거FP branch 고정, 공식평가는 모델별whole-object branch. raw10px probability mass뿐 아니라 grid면적/균일기대mass 대비비와 log(grid수) 정규화entropy를 원본진단에 저장. top5 nearest-GT는 oracle이지 배포성능이 아니다.

## 학습 계약·감독량·한계

| 구분 | 기존 감독 | 확대 감독 | 신규 | 손실 |
|---|---:|---:|---:|---:|
| real | 2024 | 2024 | 0 | 0 |
| source_train | 11201 | 11205 | 4 | 0 |
| real_exposures | 19200 | 19200 | 0 | 0 |
| source_exposures | 19050 | 19057 | 7 | 0 |
| source_heldout | 2022 | 2028 | 6 | 0 |

실사253장 동일OCC 원영상/초기점/pseudo target/order, 합성동일row/order와 원영상교란. crop1.25 baseline 모든재구성tensor와 기존300step source교란hash 일치. source normal/stress 입력고정, 새감독코너가 RNG소비를 바꾸지 않음. PRIOR1 seed1, TFAdam1e-4,300step,real8+source8,micro2,FULL+BN통계/affine동결,preserveOFF,last300만. 중간평가·재학습구제 없음.

crop확대는 support만 바꾸지 않는다: 입력상물체scale5/6, 원영상grid1.2배, crop좌표loss의 원영상당 크기 및 신규감독량도 달라진다. 따라서 총crop경로 효과로만 해석한다. 단일seed, 반복DEV, 사용자대략clean 구간과 pseudo신뢰의 한계도 유지한다.

학습 전 JSON 정수 직렬화 오류1회는 IO만 수정했다. 이전준비tensor/교란을 exact검증 재사용했고 학습은1회뿐이다.

## 세션별 paired 정답 변화

| 세션 | n코너 | A | C | A→C 손실 | A→C 획득 |
|---|---:|---:|---:|---:|---:|
| eval_night08 | 92 | 39 | 39 | 4 | 4 |
| eval_night09 | 122 | 47 | 50 | 2 | 5 |
| eval_outside | 40 | 30 | 30 | 0 | 0 |
| eval_pallet07 | 197 | 105 | 105 | 12 | 12 |
| eval_pallet09 | 262 | 138 | 135 | 9 | 6 |

## 다음 질문 하나

**TARGET_DATA_AND_TRAIN_FIT**. 추가 실행 없이 다음 계획만 남겼다. 기존 최종 모델·논문표는 변경하지 않았다.

## 이미지

[전체 갤러리](GALLERY.html). N14 전부(프레임별 통합), 고정U15/무작위대조, 정답획득·손실 및 남은오류를 함께 제시한다. 상단의 모델별대칭branch와 하단의고정branch를 구분해야 하며, 서로다른branch의 같은G표시를 물리적으로같은native점의 이동이라고 단정하지 않는다.

### 001 N14_ALL

![N14_ALL](figures/comparison_001.jpg)

### 002 N14_ALL

![N14_ALL](figures/comparison_002.jpg)

### 003 N14_ALL

![N14_ALL](figures/comparison_003.jpg)

### 004 N14_ALL

![N14_ALL](figures/comparison_004.jpg)

### 005 N14_ALL

![N14_ALL](figures/comparison_005.jpg)

### 006 N14_ALL

![N14_ALL](figures/comparison_006.jpg)

### 007 N14_ALL

![N14_ALL](figures/comparison_007.jpg)

### 008 U15_FIXED5

![U15_FIXED5](figures/comparison_008.jpg)

### 009 U15_FIXED5

![U15_FIXED5](figures/comparison_009.jpg)

### 010 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_010.jpg)

### 011 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_011.jpg)

### 012 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_012.jpg)

### 013 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_013.jpg)

### 014 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_014.jpg)

### 015 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_015.jpg)

### 016 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_016.jpg)

### 017 RANDOM_HARD_FIXED

![RANDOM_HARD_FIXED](figures/comparison_017.jpg)

### 018 A_TO_C_WINS

![A_TO_C_WINS](figures/comparison_018.jpg)

### 019 A_TO_C_WINS

![A_TO_C_WINS](figures/comparison_019.jpg)

### 020 A_TO_C_WINS

![A_TO_C_WINS](figures/comparison_020.jpg)

### 021 A_TO_C_WINS

![A_TO_C_WINS](figures/comparison_021.jpg)

### 022 A_TO_C_WINS

![A_TO_C_WINS](figures/comparison_022.jpg)

### 023 A_TO_C_DAMAGE

![A_TO_C_DAMAGE](figures/comparison_023.jpg)

### 024 A_TO_C_DAMAGE

![A_TO_C_DAMAGE](figures/comparison_024.jpg)

### 025 A_TO_C_DAMAGE

![A_TO_C_DAMAGE](figures/comparison_025.jpg)

### 026 A_TO_C_DAMAGE

![A_TO_C_DAMAGE](figures/comparison_026.jpg)

### 027 PEAK_PRESENT_DECODE_WRONG

![PEAK_PRESENT_DECODE_WRONG](figures/comparison_027.jpg)

### 028 NO_TESTED_PEAK

![NO_TESTED_PEAK](figures/comparison_028.jpg)

### 029 NO_TESTED_PEAK

![NO_TESTED_PEAK](figures/comparison_029.jpg)

### 030 NO_TESTED_PEAK

![NO_TESTED_PEAK](figures/comparison_030.jpg)

### 031 IDENTITY_SUSPECT

![IDENTITY_SUSPECT](figures/comparison_031.jpg)

### 032 IDENTITY_SUSPECT

![IDENTITY_SUSPECT](figures/comparison_032.jpg)

### 033 IDENTITY_SUSPECT

![IDENTITY_SUSPECT](figures/comparison_033.jpg)

### 034 RANDOM_GREEN_FIXED

![RANDOM_GREEN_FIXED](figures/comparison_034.jpg)

### 035 RANDOM_GREEN_FIXED

![RANDOM_GREEN_FIXED](figures/comparison_035.jpg)

### 036 RANDOM_GREEN_FIXED

![RANDOM_GREEN_FIXED](figures/comparison_036.jpg)

### 037 RANDOM_GREEN_FIXED

![RANDOM_GREEN_FIXED](figures/comparison_037.jpg)

### 038 RANDOM_GREEN_FIXED

![RANDOM_GREEN_FIXED](figures/comparison_038.jpg)
