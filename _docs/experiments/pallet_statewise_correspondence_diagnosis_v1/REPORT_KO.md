# 상태별 관측 전달·대응·자세 선택 진단 결과

실제238장 관측 계보에서 B/I 1252개는 원 e의 활성 full 836개, typed partial 344개, legacy partial 72개로 나뉜다. 그중 원 e owner의 partial이 E 후단에서 full로 연결된 것은 381개(typed 310+legacy 71)다. 출처는 예측 fallback 260개, 배경 경계선 107개, 내부 접합 12개, 경계 복구 2개다. 예측 fallback을 새 영상 교점 측정으로 읽으면 안 되며, partial 원관측과 새 full 패키지는 같은 관측이 아니다.

N3 채널과 e owner 사이에는 비항등 모델 vertex 매핑이 38장/304꼭짓점에서 필요했다. 모델 Q·등록 bijection은 검산했지만 영상에서 그 점·선이 실제로 어느 물리 코너인지의 정답은 **SEMANTIC_OWNER_UNVERIFIED**다. 이는 후단 패키지 변경과 손상을 연결하는 진단이며 381개 각각이 실패를 일으켰다는 순수 인과증명이나 재번호 하나의 효과가 아니다. [봉인된 전238 관측·owner 감사](evidence/OBSERVATION_TYPE_AND_OWNER_AUDIT.csv), [추출 집계·봉인](evidence/EXTRACTION_SEAL.json)을 먼저 확인한다.

PRIMARY192의 사전고정 주대비 TYPED_SINGLE−E0는 회복 0장, 손상 0장이다. 위치 평균 짝차는 0.000 [0.000, 0.000]cm, 회전 평균 짝차는 -0.000 [-0.000, 0.000]°다. 연속 오차는 낮을수록 좋다. 이 수치와 관측/목적함수 동등성·후보 bank 범위를 함께 해석해야 한다.

## 네 우선순위의 완료 범위

| 우선순위 | 이 보고서의 근거 | 판정 범위 |
|---|---|---|
| 1 관측 전달·owner | 전체238 계보·실제 consumer 활성항·CAL 잔차 | 모델 좌표 등록 검산과 semantic owner 정답을 분리. semantic owner는 미검증 |
| 2 SINGLE 대 MULTI | 고정4경로 결과와 parity·패킷 봉인 | N3/SubPix 유지, e 후 재관측 운영 채택 보류. typed는 복원 대조군이며 추가 이득 아님. 순수 시작정책 비교는 NOT_ISOLATED 한계 적용 |
| 3 후보 상한·선택 | 아래 consumer coverage와 단계별 기록 | 원 caller가 실제 읽은 후보의 전량성과 역사적 모든 생성 시도의 기록을 분리. consumer 공백은 상한 미확정으로 보존 |
| 4 전체 손익·PoseFix | 192/49/143/46 전량, 회복·손상·tail, 별도 PF+SubPix95 | 반복 개발 자료. 서로 다른 머리/초기화/후단 패키지의 비교이며 독립 test나 순수 보정기 비교가 아님 |

## 모집단·지표·기술 상태

PRIMARY192=TEST49+REMAINDER143, CAL46은 별도다. severe·새 데이터·새 정답·새 학습·새 seed는 추가하지 않았다. 성공은 같은 pose 하나의 중심 위치≤5cm AND physical/C2 회전≤5°다. 직사각형의 90° W/D 오류는 C2 동등성으로 지우지 않는다. 분산은 cm²/°², SD는 cm/°이며 표본분산·SD(ddof1)를 사용한다.

연속 통계는 관측 가능 N에 대한 값이며 pose 없음도 이진 실패의 전체 분모에 남긴다. 짝차는 두 경로 공통 available만 사용하고 그 N·CI·전체 N을 원자료에 함께 저장한다. 상태 열은 true/기록된 N이다. NA는0이 아니다. 수락·fallback·ambiguity는 정확도 성공과 다르다. SINGLE의 local rank 기반 ambiguity와 MULTI의 후보 탐색은 검출 범위가 달라 scope를 별도로 보존했다. E0 초기값을 사용하는 경로는 REOBS_MULTI·REOBS_SINGLE·TYPED_SINGLE이며, TYPED_MULTI는 원 e의 계층 pipeline을 재현한다. TS의 audit.E0_start_only와 SINGLE 범위는 봉인 예측을 바꾸지 않고 분석 메타데이터에서 운반했다. E0의 수락/fallback은 원래 공통 캐시의 NA와 별도로 봉인된 원 E 입력의 명시 상태를 복원했다. 정확히 같은 저장 pose와 source SHA를 검산하고 historical_common 및 restored_original_status를 함께 보존했으며, 성공 여부나 joint-feasible 값으로 상태를 추측하지 않았다.

## PRIMARY 192장 전체19경로

| 경로 | 공동 성공/N | T/R available N | 위치 평균±SD cm | 위치 분산 cm² | 위치 최대 cm | 회전 평균±SD ° | 회전 분산 °² | 회전 최대 ° | >50/>100cm | pose 없음 | 수락 | fallback | solver 거부 | 기술실패 | timeout | 모호·수락 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 · RAW YOLO + F | 66/192 | 192/192 | 14.536 ± 26.443 | 699.217 | 183.080 | 14.282 ± 28.965 | 838.964 | 88.800 | 10/6 | 0 | NA | NA | NA | NA | NA | NA |
| R1 · RAW YOLO + 최소 B/I | 82/192 | 192/192 | 14.282 ± 25.437 | 647.055 | 181.841 | 14.662 ± 28.875 | 833.746 | 89.587 | 13/5 | 0 | 143/192 | 49/192 | NA | NA | NA | NA |
| N0 · N3 + SubPix + F | 108/192 | 192/192 | 11.306 ± 26.877 | 722.358 | 174.758 | 10.631 ± 25.131 | 631.556 | 88.905 | 8/5 | 0 | NA | NA | NA | NA | NA | NA |
| N1 · N3 + SubPix + 최소 B/I | 108/192 | 192/192 | 13.384 ± 30.254 | 915.320 | 180.235 | 12.534 ± 27.630 | 763.429 | 138.042 | 14/6 | 0 | 151/192 | 41/192 | NA | NA | NA | NA |
| E0 · 기존 전체 e | 114/192 | 192/192 | 14.112 ± 30.077 | 904.618 | 260.412 | 13.001 ± 28.388 | 805.864 | 159.137 | 15/3 | 0 | 188/192 | 4/192 | NA | NA | NA | NA |
| E1 · e 뒤 재관측 B/I 재선택 | 109/192 | 192/192 | 16.207 ± 33.166 | 1099.959 | 177.715 | 17.296 ± 32.725 | 1070.951 | 159.137 | 18/7 | 0 | 144/192 | 48/192 | NA | NA | NA | NA |
| POSEFIX_RAW · PoseFix RAW PASS1 + F | 80/192 | 192/192 | 13.836 ± 27.578 | 760.559 | 185.783 | 11.640 ± 26.525 | 703.564 | 88.945 | 12/5 | 0 | NA | NA | NA | NA | NA | NA |
| N3_ONLY · N3 단독 + F | 75/192 | 192/192 | 13.059 ± 24.725 | 611.331 | 179.787 | 11.847 ± 26.557 | 705.249 | 89.072 | 9/5 | 0 | NA | NA | NA | NA | NA | NA |
| SUBPIX_ONLY · SubPix 단독 + F | 98/192 | 192/192 | 12.794 ± 26.530 | 703.853 | 162.969 | 12.952 ± 27.544 | 758.653 | 89.352 | 10/5 | 0 | NA | NA | NA | NA | NA | NA |
| RA · RAW + guard A | 85/192 | 192/192 | 13.562 ± 25.893 | 670.471 | 183.080 | 13.899 ± 28.778 | 828.148 | 89.285 | 10/6 | 0 | 128/192 | 64/192 | NA | NA | NA | NA |
| RB · RAW + guard B | 68/192 | 192/192 | 14.775 ± 27.186 | 739.101 | 183.080 | 14.229 ± 28.930 | 836.918 | 88.800 | 10/7 | 0 | 10/192 | 182/192 | NA | NA | NA | NA |
| NA · N3/SubPix + guard A | 111/192 | 192/192 | 12.555 ± 29.129 | 848.507 | 180.235 | 10.807 ± 25.091 | 629.556 | 89.060 | 12/5 | 0 | 134/192 | 58/192 | NA | NA | NA | NA |
| NB · N3/SubPix + guard B | 109/192 | 192/192 | 11.341 ± 27.265 | 743.387 | 180.235 | 10.553 ± 25.148 | 632.437 | 88.905 | 8/5 | 0 | 13/192 | 179/192 | NA | NA | NA | NA |
| EA · e + guard A | 115/192 | 192/192 | 14.587 ± 32.281 | 1042.041 | 260.412 | 13.522 ± 28.865 | 833.200 | 159.137 | 17/4 | 0 | 113/192 | 79/192 | NA | NA | NA | NA |
| EB · e + guard B | 114/192 | 192/192 | 14.066 ± 30.078 | 904.680 | 260.412 | 13.001 ± 28.376 | 805.194 | 159.137 | 15/3 | 0 | 2/192 | 190/192 | NA | NA | NA | NA |
| REOBS_MULTI · 재관측 / 기존 다중 시작 | 109/192 | 192/192 | 16.207 ± 33.166 | 1099.959 | 177.715 | 17.296 ± 32.725 | 1070.951 | 159.137 | 18/7 | 0 | 144/192 | 48/192 | NA | 0/192 | 0/192 | 18/144 (NA 48) |
| REOBS_SINGLE · 재관측 / E0 단일 시작 | 110/192 | 192/192 | 14.698 ± 29.177 | 851.287 | 177.715 | 15.586 ± 30.732 | 944.430 | 159.137 | 16/4 | 0 | 136/192 | 56/192 | NA | 0/192 | 0/192 | 0/192 |
| TYPED_MULTI · 원 typed 관측 / 기존 다중 시작 | 114/192 | 192/192 | 14.112 ± 30.077 | 904.618 | 260.412 | 13.001 ± 28.388 | 805.864 | 159.137 | 15/3 | 0 | 188/192 | 4/192 | NA | 0/192 | 0/192 | 24/188 (NA 4) |
| TYPED_SINGLE · 원 typed 관측 / E0 단일 시작 | 114/192 | 192/192 | 14.112 ± 30.077 | 904.618 | 260.412 | 13.001 ± 28.388 | 805.864 | 159.137 | 15/3 | 0 | 188/192 | 4/192 | NA | 0/192 | 0/192 | 11/188 (NA 4) |

## TEST 49장 전체19경로

| 경로 | 공동 성공/N | T/R available N | 위치 평균±SD cm | 위치 분산 cm² | 위치 최대 cm | 회전 평균±SD ° | 회전 분산 °² | 회전 최대 ° | >50/>100cm | pose 없음 | 수락 | fallback | solver 거부 | 기술실패 | timeout | 모호·수락 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 · RAW YOLO + F | 30/49 | 49/49 | 4.140 ± 3.487 | 12.163 | 15.105 | 6.625 ± 19.972 | 398.898 | 87.885 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| R1 · RAW YOLO + 최소 B/I | 36/49 | 49/49 | 4.239 ± 6.770 | 45.831 | 44.890 | 6.940 ± 20.475 | 419.215 | 88.967 | 0/0 | 0 | 24/49 | 25/49 | NA | NA | NA | NA |
| N0 · N3 + SubPix + F | 40/49 | 49/49 | 3.363 ± 3.487 | 12.162 | 19.957 | 6.527 ± 20.003 | 400.137 | 88.164 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| N1 · N3 + SubPix + 최소 B/I | 38/49 | 49/49 | 4.263 ± 8.660 | 74.990 | 58.445 | 8.343 ± 23.356 | 545.504 | 102.047 | 1/0 | 0 | 29/49 | 20/49 | NA | NA | NA | NA |
| E0 · 기존 전체 e | 49/49 | 49/49 | 1.299 ± 1.082 | 1.172 | 4.339 | 1.110 ± 0.773 | 0.598 | 3.950 | 0/0 | 0 | 49/49 | 0/49 | NA | NA | NA | NA |
| E1 · e 뒤 재관측 B/I 재선택 | 42/49 | 49/49 | 7.058 ± 25.269 | 638.542 | 163.021 | 7.616 ± 22.043 | 485.892 | 102.044 | 2/1 | 0 | 30/49 | 19/49 | NA | NA | NA | NA |
| POSEFIX_RAW · PoseFix RAW PASS1 + F | 34/49 | 49/49 | 3.931 ± 3.142 | 9.875 | 14.323 | 6.339 ± 20.074 | 402.946 | 87.953 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| N3_ONLY · N3 단독 + F | 35/49 | 49/49 | 3.911 ± 3.679 | 13.532 | 20.202 | 6.482 ± 20.041 | 401.658 | 88.145 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| SUBPIX_ONLY · SubPix 단독 + F | 35/49 | 49/49 | 3.943 ± 3.630 | 13.174 | 17.947 | 6.643 ± 19.946 | 397.858 | 87.802 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| RA · RAW + guard A | 38/49 | 49/49 | 3.258 ± 3.177 | 10.091 | 13.986 | 6.702 ± 20.469 | 418.994 | 88.967 | 0/0 | 0 | 22/49 | 27/49 | NA | NA | NA | NA |
| RB · RAW + guard B | 30/49 | 49/49 | 4.140 ± 3.487 | 12.163 | 15.105 | 6.625 ± 19.972 | 398.898 | 87.885 | 0/0 | 0 | 0/49 | 49/49 | NA | NA | NA | NA |
| NA · N3/SubPix + guard A | 40/49 | 49/49 | 3.869 ± 8.474 | 71.809 | 58.445 | 6.246 ± 18.881 | 356.488 | 88.898 | 1/0 | 0 | 26/49 | 23/49 | NA | NA | NA | NA |
| NB · N3/SubPix + guard B | 40/49 | 49/49 | 3.349 ± 3.495 | 12.218 | 19.957 | 6.520 ± 20.005 | 400.215 | 88.164 | 0/0 | 0 | 1/49 | 48/49 | NA | NA | NA | NA |
| EA · e + guard A | 47/49 | 49/49 | 1.664 ± 1.468 | 2.154 | 7.626 | 1.258 ± 1.268 | 1.609 | 8.326 | 0/0 | 0 | 25/49 | 24/49 | NA | NA | NA | NA |
| EB · e + guard B | 49/49 | 49/49 | 1.299 ± 1.082 | 1.172 | 4.339 | 1.110 ± 0.773 | 0.598 | 3.950 | 0/0 | 0 | 0/49 | 49/49 | NA | NA | NA | NA |
| REOBS_MULTI · 재관측 / 기존 다중 시작 | 42/49 | 49/49 | 7.058 ± 25.269 | 638.542 | 163.021 | 7.616 ± 22.043 | 485.892 | 102.044 | 2/1 | 0 | 30/49 | 19/49 | NA | 0/49 | 0/49 | 4/30 (NA 19) |
| REOBS_SINGLE · 재관측 / E0 단일 시작 | 42/49 | 49/49 | 3.916 ± 11.073 | 122.621 | 75.733 | 6.084 ± 18.995 | 360.791 | 89.152 | 1/0 | 0 | 27/49 | 22/49 | NA | 0/49 | 0/49 | 0/49 |
| TYPED_MULTI · 원 typed 관측 / 기존 다중 시작 | 49/49 | 49/49 | 1.299 ± 1.082 | 1.172 | 4.339 | 1.110 ± 0.773 | 0.598 | 3.950 | 0/0 | 0 | 49/49 | 0/49 | NA | 0/49 | 0/49 | 4/49 |
| TYPED_SINGLE · 원 typed 관측 / E0 단일 시작 | 49/49 | 49/49 | 1.299 ± 1.082 | 1.172 | 4.339 | 1.110 ± 0.773 | 0.598 | 3.950 | 0/0 | 0 | 49/49 | 0/49 | NA | 0/49 | 0/49 | 2/49 |

## REMAINDER 143장 전체19경로

| 경로 | 공동 성공/N | T/R available N | 위치 평균±SD cm | 위치 분산 cm² | 위치 최대 cm | 회전 평균±SD ° | 회전 분산 °² | 회전 최대 ° | >50/>100cm | pose 없음 | 수락 | fallback | solver 거부 | 기술실패 | timeout | 모호·수락 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 · RAW YOLO + F | 36/143 | 143/143 | 18.098 ± 29.771 | 886.311 | 183.080 | 16.906 ± 31.088 | 966.465 | 88.800 | 10/6 | 0 | NA | NA | NA | NA | NA | NA |
| R1 · RAW YOLO + 최소 B/I | 46/143 | 143/143 | 17.724 ± 28.427 | 808.113 | 181.841 | 17.308 ± 30.856 | 952.114 | 89.587 | 13/5 | 0 | 119/143 | 24/143 | NA | NA | NA | NA |
| N0 · N3 + SubPix + F | 68/143 | 143/143 | 14.027 ± 30.631 | 938.282 | 174.758 | 12.037 ± 26.579 | 706.427 | 88.905 | 8/5 | 0 | NA | NA | NA | NA | NA | NA |
| N1 · N3 + SubPix + 최소 B/I | 70/143 | 143/143 | 16.509 ± 34.165 | 1167.276 | 180.235 | 13.970 ± 28.885 | 834.335 | 138.042 | 13/6 | 0 | 122/143 | 21/143 | NA | NA | NA | NA |
| E0 · 기존 전체 e | 65/143 | 143/143 | 18.502 ± 33.769 | 1140.318 | 260.412 | 17.076 ± 31.910 | 1018.233 | 159.137 | 15/3 | 0 | 139/143 | 4/143 | NA | NA | NA | NA |
| E1 · e 뒤 재관측 B/I 재선택 | 67/143 | 143/143 | 19.341 ± 34.999 | 1224.900 | 177.715 | 20.613 ± 35.112 | 1232.846 | 159.137 | 16/6 | 0 | 114/143 | 29/143 | NA | NA | NA | NA |
| POSEFIX_RAW · PoseFix RAW PASS1 + F | 46/143 | 143/143 | 17.230 ± 31.212 | 974.211 | 185.783 | 13.456 ± 28.233 | 797.118 | 88.945 | 12/5 | 0 | NA | NA | NA | NA | NA | NA |
| N3_ONLY · N3 단독 + F | 40/143 | 143/143 | 16.194 ± 27.909 | 778.937 | 179.787 | 13.686 ± 28.275 | 799.500 | 89.072 | 9/5 | 0 | NA | NA | NA | NA | NA | NA |
| SUBPIX_ONLY · SubPix 단독 + F | 63/143 | 143/143 | 15.826 ± 30.100 | 905.986 | 162.969 | 15.113 ± 29.454 | 867.515 | 89.352 | 10/5 | 0 | NA | NA | NA | NA | NA | NA |
| RA · RAW + guard A | 47/143 | 143/143 | 17.092 ± 29.142 | 849.233 | 183.080 | 16.365 ± 30.794 | 948.290 | 89.285 | 10/6 | 0 | 106/143 | 37/143 | NA | NA | NA | NA |
| RB · RAW + guard B | 38/143 | 143/143 | 18.420 ± 30.621 | 937.622 | 183.080 | 16.834 ± 31.050 | 964.091 | 88.800 | 10/7 | 0 | 10/143 | 133/143 | NA | NA | NA | NA |
| NA · N3/SubPix + guard A | 71/143 | 143/143 | 15.531 ± 32.895 | 1082.075 | 180.235 | 12.370 ± 26.770 | 716.655 | 89.060 | 11/5 | 0 | 108/143 | 35/143 | NA | NA | NA | NA |
| NB · N3/SubPix + guard B | 69/143 | 143/143 | 14.080 ± 31.083 | 966.182 | 180.235 | 11.935 ± 26.606 | 707.854 | 88.905 | 8/5 | 0 | 12/143 | 131/143 | NA | NA | NA | NA |
| EA · e + guard A | 68/143 | 143/143 | 19.016 ± 36.380 | 1323.513 | 260.412 | 17.725 ± 32.411 | 1050.475 | 159.137 | 17/4 | 0 | 88/143 | 55/143 | NA | NA | NA | NA |
| EB · e + guard B | 65/143 | 143/143 | 18.441 ± 33.778 | 1140.945 | 260.412 | 17.076 ± 31.896 | 1017.331 | 159.137 | 15/3 | 0 | 2/143 | 141/143 | NA | NA | NA | NA |
| REOBS_MULTI · 재관측 / 기존 다중 시작 | 67/143 | 143/143 | 19.341 ± 34.999 | 1224.900 | 177.715 | 20.613 ± 35.112 | 1232.846 | 159.137 | 16/6 | 0 | 114/143 | 29/143 | NA | 0/143 | 0/143 | 14/114 (NA 29) |
| REOBS_SINGLE · 재관측 / E0 단일 시작 | 68/143 | 143/143 | 18.393 ± 32.400 | 1049.729 | 177.715 | 18.842 ± 33.265 | 1106.535 | 159.137 | 15/4 | 0 | 109/143 | 34/143 | NA | 0/143 | 0/143 | 0/143 |
| TYPED_MULTI · 원 typed 관측 / 기존 다중 시작 | 65/143 | 143/143 | 18.502 ± 33.769 | 1140.318 | 260.412 | 17.076 ± 31.910 | 1018.233 | 159.137 | 15/3 | 0 | 139/143 | 4/143 | NA | 0/143 | 0/143 | 20/139 (NA 4) |
| TYPED_SINGLE · 원 typed 관측 / E0 단일 시작 | 65/143 | 143/143 | 18.502 ± 33.769 | 1140.318 | 260.412 | 17.076 ± 31.910 | 1018.233 | 159.137 | 15/3 | 0 | 139/143 | 4/143 | NA | 0/143 | 0/143 | 9/139 (NA 4) |

## CAL 46장 전체19경로

| 경로 | 공동 성공/N | T/R available N | 위치 평균±SD cm | 위치 분산 cm² | 위치 최대 cm | 회전 평균±SD ° | 회전 분산 °² | 회전 최대 ° | >50/>100cm | pose 없음 | 수락 | fallback | solver 거부 | 기술실패 | timeout | 모호·수락 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 · RAW YOLO + F | 31/46 | 46/46 | 4.989 ± 4.841 | 23.435 | 20.429 | 8.110 ± 20.871 | 435.598 | 87.509 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| R1 · RAW YOLO + 최소 B/I | 30/46 | 46/46 | 5.527 ± 7.549 | 56.993 | 44.838 | 9.899 ± 23.753 | 564.210 | 87.509 | 0/0 | 0 | 11/46 | 35/46 | NA | NA | NA | NA |
| N0 · N3 + SubPix + F | 34/46 | 46/46 | 3.982 ± 4.302 | 18.510 | 17.766 | 7.805 ± 20.905 | 437.025 | 87.868 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| N1 · N3 + SubPix + 최소 B/I | 34/46 | 46/46 | 5.085 ± 7.953 | 63.256 | 42.187 | 9.739 ± 23.997 | 575.835 | 87.868 | 0/0 | 0 | 17/46 | 29/46 | NA | NA | NA | NA |
| E0 · 기존 전체 e | 37/46 | 46/46 | 5.379 ± 11.288 | 127.427 | 51.978 | 10.599 ± 26.024 | 677.240 | 89.658 | 1/0 | 0 | 45/46 | 1/46 | NA | NA | NA | NA |
| E1 · e 뒤 재관측 B/I 재선택 | 36/46 | 46/46 | 4.793 ± 10.340 | 106.905 | 51.978 | 8.683 ± 23.071 | 532.259 | 87.819 | 1/0 | 0 | 17/46 | 29/46 | NA | NA | NA | NA |
| POSEFIX_RAW · PoseFix RAW PASS1 + F | 32/46 | 46/46 | 5.249 ± 7.636 | 58.306 | 46.482 | 9.759 ± 24.066 | 579.168 | 89.057 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| N3_ONLY · N3 단독 + F | 34/46 | 46/46 | 4.235 ± 4.415 | 19.497 | 17.736 | 7.816 ± 20.919 | 437.584 | 87.983 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| SUBPIX_ONLY · SubPix 단독 + F | 31/46 | 46/46 | 5.572 ± 7.828 | 61.270 | 47.713 | 9.997 ± 23.867 | 569.645 | 87.392 | 0/0 | 0 | NA | NA | NA | NA | NA | NA |
| RA · RAW + guard A | 32/46 | 46/46 | 4.399 ± 4.521 | 20.438 | 20.429 | 8.030 ± 20.899 | 436.753 | 87.509 | 0/0 | 0 | 9/46 | 37/46 | NA | NA | NA | NA |
| RB · RAW + guard B | 31/46 | 46/46 | 4.934 ± 4.871 | 23.731 | 20.429 | 8.098 ± 20.875 | 435.778 | 87.509 | 0/0 | 0 | 1/46 | 45/46 | NA | NA | NA | NA |
| NA · N3/SubPix + guard A | 37/46 | 46/46 | 3.577 ± 4.334 | 18.781 | 17.766 | 7.655 ± 20.952 | 438.966 | 87.868 | 0/0 | 0 | 14/46 | 32/46 | NA | NA | NA | NA |
| NB · N3/SubPix + guard B | 34/46 | 46/46 | 3.937 ± 4.317 | 18.633 | 17.766 | 7.769 ± 20.916 | 437.495 | 87.868 | 0/0 | 0 | 2/46 | 44/46 | NA | NA | NA | NA |
| EA · e + guard A | 37/46 | 46/46 | 4.637 ± 10.173 | 103.498 | 51.978 | 8.687 ± 23.157 | 536.233 | 88.864 | 1/0 | 0 | 14/46 | 32/46 | NA | NA | NA | NA |
| EB · e + guard B | 38/46 | 46/46 | 4.577 ± 10.179 | 103.618 | 51.978 | 8.662 ± 23.168 | 536.735 | 88.864 | 1/0 | 0 | 1/46 | 45/46 | NA | NA | NA | NA |
| REOBS_MULTI · 재관측 / 기존 다중 시작 | 36/46 | 46/46 | 4.793 ± 10.340 | 106.905 | 51.978 | 8.683 ± 23.071 | 532.259 | 87.819 | 1/0 | 0 | 17/46 | 29/46 | NA | 0/46 | 0/46 | 1/17 (NA 29) |
| REOBS_SINGLE · 재관측 / E0 단일 시작 | 37/46 | 46/46 | 3.866 ± 8.688 | 75.473 | 51.978 | 6.843 ± 19.760 | 390.466 | 84.000 | 1/0 | 0 | 12/46 | 34/46 | NA | 0/46 | 0/46 | 0/46 |
| TYPED_MULTI · 원 typed 관측 / 기존 다중 시작 | 37/46 | 46/46 | 5.379 ± 11.288 | 127.427 | 51.978 | 10.599 ± 26.024 | 677.240 | 89.658 | 1/0 | 0 | 45/46 | 1/46 | NA | 0/46 | 0/46 | 5/45 (NA 1) |
| TYPED_SINGLE · 원 typed 관측 / E0 단일 시작 | 37/46 | 46/46 | 5.379 ± 11.288 | 127.427 | 51.978 | 10.599 ± 26.024 | 677.240 | 89.658 | 1/0 | 0 | 45/46 | 1/46 | NA | 0/46 | 0/46 | 2/45 (NA 1) |

## 복원·재관측의 손익과 수치적 동등성

TYPED_SINGLE−E0의 PRIMARY 위치는 192/192장에서 정확히 같다. 회전의 원시 비교에는 개선 20장·악화 7장·동률 165장이 남지만 최대 차이는 3.04256619899e-12°다. 기존 replay 허용치1e-9를 넘은 경우는 0장이며, 이 부동소수 차이를 실질적인 정확도 이득으로 세지 않는다. 원시 비교 수와 값을 지우거나 새 임곗값으로 성공 여부를 바꾸지 않았다.

| PRIMARY192 재관측−e | 회복 | 손상 | 양쪽 실패 | 위치 평균 짝차 cm | 회전 평균 짝차 ° |
|---|---:|---:|---:|---:|---:|
| REOBS_MULTI | 10 | 15 | 68 | 2.094 | 4.295 |
| REOBS_SINGLE | 11 | 15 | 67 | 0.586 | 2.585 |

두 재관측 경로의 손상 ID 집합은 실제로 같다. 단순히 개수가 같다는 이유로 같은 사례라고 간주하지 않았으며 ID 전량은 전이표에 보존했다.

| PRIMARY192 경로 | 위치 최대 cm | >50cm N | >100cm N |
|---|---:|---:|---:|
| E0 | 260.412 | 15 | 3 |
| TYPED_SINGLE | 260.412 | 15 | 3 |
| REOBS_MULTI | 177.715 | 18 | 7 |
| REOBS_SINGLE | 177.715 | 16 | 4 |

typed는 기존 e의 최대오차와 실패를 함께 복원한다. 재관측이 최대 위치오차를 더 낮추는 이득도 남으므로 typed가 모든 지표에서 우월하다고 주장하지 않는다. 평균·공동 성공·tail 빈도·최대값을 함께 해석한다. 선택손실이 모두 큰 W/D 뒤집힘이라는 결론도 아니다. family 선택과 동일 family 내부의 작은 임계값 손실을 구분한다.

## 사전고정 주대비·양방향 전이

| 분모 | 주대비: 새−기존 | 양쪽 성공 | 회복 | 손상 | 양쪽 실패 | 참조 미확정 | 성공률 Δpp [95% CI] | 위치 Δcm [95% CI] | 회전 Δ° [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY192 | TYPED_SINGLE-E0 | 114 | 0 | 0 | 78 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] |
| PRIMARY192 | TYPED_SINGLE-TYPED_MULTI | 114 | 0 | 0 | 78 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] |
| PRIMARY192 | TYPED_SINGLE-REOBS_SINGLE | 99 | 15 | 11 | 67 | 0 | 2.083 [-1.326, 16.949] | -0.586 [-4.272, 1.821] | -2.585 [-7.575, -0.279] |
| TEST49 | TYPED_SINGLE-E0 | 49 | 0 | 0 | 0 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] |
| TEST49 | TYPED_SINGLE-TYPED_MULTI | 49 | 0 | 0 | 0 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] |
| TEST49 | TYPED_SINGLE-REOBS_SINGLE | 42 | 7 | 0 | 0 | 0 | 14.286 [8.108, 33.333] | -2.617 [-15.538, -0.590] | -4.974 [-19.402, -0.294] |
| REMAINDER143 | TYPED_SINGLE-E0 | 65 | 0 | 0 | 78 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] |
| REMAINDER143 | TYPED_SINGLE-TYPED_MULTI | 65 | 0 | 0 | 78 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] |
| REMAINDER143 | TYPED_SINGLE-REOBS_SINGLE | 57 | 8 | 11 | 67 | 0 | -2.098 [-2.597, -1.515] | 0.110 [-2.809, 2.611] | -1.766 [-3.870, 0.037] |
| CAL46 | TYPED_SINGLE-E0 | 37 | 0 | 0 | 9 | 0 | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] | 0.000 [0.000, 0.000] |
| CAL46 | TYPED_SINGLE-TYPED_MULTI | 37 | 0 | 0 | 9 | 0 | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] | 0.000 [0.000, 0.000] |
| CAL46 | TYPED_SINGLE-REOBS_SINGLE | 35 | 2 | 2 | 7 | 0 | 0.000 [-12.500, 4.762] | 1.513 [-0.277, 9.458] | 3.756 [-0.259, 22.336] |

세 주대비는 결과 전에 고정한 TYPED_SINGLE−E0, TYPED_SINGLE−TYPED_MULTI, TYPED_SINGLE−REOBS_SINGLE다. TYPED_SINGLE−TYPED_MULTI의 시작정책 인과분리는 **NOT_ISOLATED**다. MULTI의 G10 full/legacy family 선택 뒤 G13와 SINGLE의 이미 선택된 family 고정·별도 B/I seed fit 생략이 함께 달라, 단일 시작 효과만으로 귀속하지 않는다. 마지막은 관측 패키지 전체의 차이이며 재번호 하나의 순수 인과효과가 아니다. CI는 해당 ID 집합의 실제 원녹화 cluster를 whole-recording 단위로 10,000회, seed20260917로 새로 재표집한 frame-weighted 짝차다. 새 독립 평가나 다중비교 확증적 유의성·동등성 검증으로 쓰지 않는다.

## 양쪽 실패 집단의 오차 기여

| 주대비 | 양쪽 실패 N | 위치 Δ총합 cm | 전체 공통분모 평균에 대한 기여 cm | 회전 Δ총합 ° | 평균 기여 ° |
|---|---:|---:|---:|---:|---:|
| TYPED_SINGLE-E0 | 78 | 0.000 | 0.000 | -0.000 | -0.000 |
| TYPED_SINGLE-TYPED_MULTI | 78 | 0.000 | 0.000 | -0.000 | -0.000 |
| TYPED_SINGLE-REOBS_SINGLE | 67 | 41.427 | 0.216 | -270.722 | -1.410 |

회복−손상은 성공 수 차이와 같고, 네 전이군의 연속 짝차 합은 같은 공통 available 분모의 평균 차이를 재현한다. 양쪽 실패의 큰 악화도 포함했다. 전체 실패 ID·>50/>100cm ID·최대값과 모든 보조 대비는 [전체 결과](metrics/RESULTS.json), [행별 오차](metrics/ROW_METRICS.csv), [전량 전이표](metrics/PAIR_TRANSITIONS.csv)에 있다.

## PoseFix RAW와 두 기존 SubPix cap의 공통95

전체192의 PF comparator는 synthetic-only PRIOR1 last6000의 uncapped RAW PASS1+F다. 실사 replay last300이나 synthetic selection의 1% wrapper와 다르다. Gaussian 입력·좌표 loss는 이미 구현되어 있다. 아래 두 SubPix cap은 기존 CAL46/TEST49의 고정 결과를 모두 직접 재채점했으며 승자를 선택하지 않았다.

| 공통분모 | 경로 | 공동 성공/N | 위치 평균±SD cm | 위치 최대 cm | 회전 평균±SD ° | 회전 최대 ° |
|---|---|---:|---:|---:|---:|---:|
| CAL | POSEFIX_RAW | 32/46 | 5.249 ± 7.636 | 46.482 | 9.759 ± 24.066 | 89.057 |
| CAL | POSEFIX_SUBPIX_PF_ANCHOR | 32/46 | 5.022 ± 7.369 | 44.472 | 9.766 ± 24.091 | 89.174 |
| CAL | POSEFIX_SUBPIX_YOLO_ANCHOR | 32/46 | 5.025 ± 7.369 | 44.472 | 9.772 ± 24.086 | 89.174 |
| CAL | N0 | 34/46 | 3.982 ± 4.302 | 17.766 | 7.805 ± 20.905 | 87.868 |
| CAL | E0 | 37/46 | 5.379 ± 11.288 | 51.978 | 10.599 ± 26.024 | 89.658 |
| TEST | POSEFIX_RAW | 34/49 | 3.931 ± 3.142 | 14.323 | 6.339 ± 20.074 | 87.953 |
| TEST | POSEFIX_SUBPIX_PF_ANCHOR | 39/49 | 3.216 ± 2.751 | 11.865 | 6.350 ± 20.017 | 87.787 |
| TEST | POSEFIX_SUBPIX_YOLO_ANCHOR | 38/49 | 3.243 ± 2.771 | 11.865 | 6.355 ± 20.016 | 87.787 |
| TEST | N0 | 40/49 | 3.363 ± 3.487 | 19.957 | 6.527 ± 20.003 | 88.164 |
| TEST | E0 | 49/49 | 1.299 ± 1.082 | 4.339 | 1.110 ± 0.773 | 3.950 |
| COMMON95 | POSEFIX_RAW | 66/95 | 4.569 ± 5.779 | 46.482 | 7.995 ± 22.045 | 89.057 |
| COMMON95 | POSEFIX_SUBPIX_PF_ANCHOR | 71/95 | 4.090 ± 5.539 | 44.472 | 8.004 ± 22.031 | 89.174 |
| COMMON95 | POSEFIX_SUBPIX_YOLO_ANCHOR | 70/95 | 4.106 ± 5.542 | 44.472 | 8.010 ± 22.028 | 89.174 |
| COMMON95 | N0 | 74/95 | 3.663 ± 3.895 | 19.957 | 7.146 ± 20.346 | 88.164 |
| COMMON95 | E0 | 86/95 | 3.275 ± 8.112 | 51.978 | 5.705 ± 18.634 | 89.658 |

**POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED**: 전체192에 사용할 단일 기존 규칙이 확립되지 않았으므로 이 두 cap의 공통95를 RAW192에 합치거나 더 좋은 cap으로143장을 채우지 않았다. RAW와의 비교만으로 기존 Table II 최상 경쟁군보다 우수하다고 일반화하지 않는다. [두 cap의 전량 결과·전이·CI](metrics/POSEFIX_SUBPIX_COMMON95.json), [95장 행별 수치](metrics/POSEFIX_SUBPIX_COMMON95.csv), [원문·구현·과거 실패 감사](POSEFIX_TRANSFER_AUDIT.md).

## CAL46의 잔차·상태·물리 대응과 좋은 입력 손상

아래 px 오차는 기존 기하 재구성 참조의 투영에 대한 잔차다. 독립적으로 클릭한 영상점 정답이나 semantic owner 정답이 아니다. 영상별 초기점 묶음에 한 번 정한 C2 phase만 분석에 쓰고, phase0/180 잔차를 모두 보존한다. 점별 가까운 정답 재배정이나 90° 동등성은 추가하지 않았다.

| CAL 단계 | 상태 | 관측 가능한 점 N | px 평균±SD | px 최대 |
|---|---|---:|---:|---:|
| N3_INITIAL | B | 245/245 | 36.655 ± 99.287 | 563.597 |
| N3_INITIAL | I | 22/22 | 42.686 ± 109.658 | 446.158 |
| N3_INITIAL | S | 68/68 | 29.626 ± 71.429 | 323.496 |
| N3_INITIAL | O | 9/9 | 17.005 ± 15.864 | 38.600 |
| N3_INITIAL | T | 0/0 | NA ± NA | NA |
| N3_INITIAL | U | 24/24 | 10.620 ± 5.490 | 21.497 |
| ORIGINAL_E_COMPLETE | B | 177/177 | 55.889 ± 134.806 | 609.223 |
| ORIGINAL_E_COMPLETE | I | 15/15 | 44.636 ± 129.550 | 512.319 |
| ORIGINAL_E_COMPLETE | S | 0/0 | NA ± NA | NA |
| ORIGINAL_E_COMPLETE | O | 0/0 | NA ± NA | NA |
| ORIGINAL_E_COMPLETE | T | 0/0 | NA ± NA | NA |
| ORIGINAL_E_COMPLETE | U | 0/0 | NA ± NA | NA |
| POST_E_COMPLETE | B | 245/245 | 33.691 ± 92.689 | 563.597 |
| POST_E_COMPLETE | I | 21/21 | 32.738 ± 95.655 | 446.158 |
| POST_E_COMPLETE | S | 0/0 | NA ± NA | NA |
| POST_E_COMPLETE | O | 0/0 | NA ± NA | NA |
| POST_E_COMPLETE | T | 0/0 | NA ± NA | NA |
| POST_E_COMPLETE | U | 0/0 | NA ± NA | NA |
| POST_E_REINJECTED_N3 | B | 245/245 | 34.716 ± 92.389 | 563.597 |
| POST_E_REINJECTED_N3 | I | 21/21 | 33.161 ± 95.514 | 446.158 |
| POST_E_REINJECTED_N3 | S | 68/68 | 36.436 ± 100.903 | 539.814 |
| POST_E_REINJECTED_N3 | O | 11/11 | 13.416 ± 14.043 | 38.600 |
| POST_E_REINJECTED_N3 | T | 10/10 | 11.830 ± 5.158 | 21.497 |
| POST_E_REINJECTED_N3 | U | 13/13 | 30.520 ± 72.460 | 270.960 |

PRIMARY192의 R0→PoseFix 고정 입력 owner 비교: 좋은 입력(<5px) 684점 중 출력>10px 손상 0점. 입력>20px 274점 중 ≤10px 회복 0점. 좋은 입력의 이동량은 1.675 ± 1.134px다.

CAL 원 e의 활성 complete 192점에서는 N3 입력<5px 57점 중 출력>10px 9점, 입력>20px 21점 중 출력≤10px 1점이었다.

이 점 단위 손상 수를 pose 성공→실패 이미지 수와 합치지 않는다. PF 상태별 표는 원 N3 상태를 모델의 물리 꼭짓점으로 옮긴 문맥이며 PF 자체 가시성 정답이 아니다. S/O/T/U는 실제 B/I 점으로 승격하지 않았고, partial 선의 잔차는 별도의 1차원 값이다.

| PRIMARY192 PoseFix/e 패턴 | 이미지 N |
|---|---:|
| E_ONLY | 49 |
| BOTH_SUCCESS | 65 |
| BOTH_FAILURE | 63 |
| PF_ONLY | 15 |

| 경로 | 기존 e 성공 입력 N | 그중 pose 손상 N | 위치 평균 짝차 cm | 회전 평균 짝차 ° |
|---|---:|---:|---:|---:|
| REOBS_MULTI | 114 | 15 | 4.098 | 4.322 |
| REOBS_SINGLE | 114 | 15 | 2.747 | 3.663 |
| TYPED_MULTI | 114 | 0 | 0.000 | 0.000 |
| TYPED_SINGLE | 114 | 0 | 0.000 | -0.000 |

현재 학습 오류지원 판정: **COVERAGE_NOT_VERIFIED**. 과거의 5–15% bbox대각선 위치 교란과 현재 잔차 크기를 비교하는 것은 의미 대응·상태·오류빈도의 학습지원 검증이 아니다. 새 학습은 실행하지 않았다.

[전체 CAL 점 잔차](diagnostics/CAL_POINT_RESIDUALS.csv), [PF 좋은 입력 이동·손상 전량](diagnostics/PRIMARY_POSEFIX_GOOD_INPUT_MOVEMENTS.csv), [상태·owner 감사](diagnostics/CAL_OWNER_AND_STATE_AUDIT.json), [전체 진단](diagnostics/DIAGNOSTICS.json).

## 후보 상한·실패 단계와 남는 공백

**원 e가 실제 읽은 consumer 후보 bank는 전238장에서 확인됐다.** 이는 원 caller가 읽는 accepted winner·전체 feasible·명시 초기값 범위의 완전성이다.

원 caller가 읽지 않은 과거 W/D·P1 rejected 전량은 consumer 후보 상한의 분모가 아니다. 역사적 모든 생성 시도의 기록 완전성과 실제 consumer bank의 완전성은 별도로 보존했다. 고정 consumer에서 좋은 후보가 없다는 결과를 전역 물리 자세 불가능성으로 확대하지 않는다. after_support는 계측 후보의 사후 원 support 검사이며 새 순차 pruning 정책이 아니다.

| 분모 | consumer 생성 완전 이미지 N | 미확정 이미지 N | 해당 consumer bank 완전 |
|---|---:|---:|---|
| PRIMARY192 | 192 | 0 | True |
| TEST49 | 49 | 0 | True |
| REMAINDER143 | 143 | 0 | True |
| CAL46 | 46 | 0 | True |

| 분모 | 저장 단계 | 같은 후보의 joint 성공 가능 N | 후보가 저장된 이미지 N | 후보 없음 N | support 미확정 이미지 N |
|---|---|---:|---:|---:|---:|
| PRIMARY192 | generated | 146 | 188 | 4 | NA |
| PRIMARY192 | after_support | 146 | 188 | 4 | 0 |
| PRIMARY192 | feasible | 146 | 188 | 4 | NA |
| PRIMARY192 | family | 136 | 188 | 4 | NA |
| PRIMARY192 | refinement | 114 | 192 | 0 | NA |
| PRIMARY192 | selected | 114 | 192 | 0 | NA |
| TEST49 | generated | 49 | 49 | 0 | NA |
| TEST49 | after_support | 49 | 49 | 0 | 0 |
| TEST49 | feasible | 49 | 49 | 0 | NA |
| TEST49 | family | 49 | 49 | 0 | NA |
| TEST49 | refinement | 49 | 49 | 0 | NA |
| TEST49 | selected | 49 | 49 | 0 | NA |
| REMAINDER143 | generated | 97 | 139 | 4 | NA |
| REMAINDER143 | after_support | 97 | 139 | 4 | 0 |
| REMAINDER143 | feasible | 97 | 139 | 4 | NA |
| REMAINDER143 | family | 87 | 139 | 4 | NA |
| REMAINDER143 | refinement | 65 | 143 | 0 | NA |
| REMAINDER143 | selected | 65 | 143 | 0 | NA |
| CAL46 | generated | 40 | 45 | 1 | NA |
| CAL46 | after_support | 40 | 45 | 1 | 0 |
| CAL46 | feasible | 40 | 45 | 1 | NA |
| CAL46 | family | 37 | 45 | 1 | NA |
| CAL46 | refinement | 37 | 46 | 0 | NA |
| CAL46 | selected | 37 | 46 | 0 | NA |

| 분모 | 원 저장 패키지 | 단계 | 같은 후보의 joint 성공 가능 N | 후보가 저장된 이미지 N |
|---|---|---|---:|---:|
| PRIMARY192 | G10 | generated | 143 | 188 |
| PRIMARY192 | G10 | after_support | 143 | 188 |
| PRIMARY192 | G10 | feasible | 143 | 188 |
| PRIMARY192 | G10 | selected | 116 | 192 |
| PRIMARY192 | G13 | generated | 146 | 188 |
| PRIMARY192 | G13 | after_support | 146 | 188 |
| PRIMARY192 | G13 | feasible | 146 | 188 |
| PRIMARY192 | G13 | selected | 112 | 192 |
| TEST49 | G10 | generated | 47 | 49 |
| TEST49 | G10 | after_support | 47 | 49 |
| TEST49 | G10 | feasible | 47 | 49 |
| TEST49 | G10 | selected | 47 | 49 |
| TEST49 | G13 | generated | 49 | 49 |
| TEST49 | G13 | after_support | 49 | 49 |
| TEST49 | G13 | feasible | 49 | 49 |
| TEST49 | G13 | selected | 47 | 49 |
| REMAINDER143 | G10 | generated | 96 | 139 |
| REMAINDER143 | G10 | after_support | 96 | 139 |
| REMAINDER143 | G10 | feasible | 96 | 139 |
| REMAINDER143 | G10 | selected | 69 | 143 |
| REMAINDER143 | G13 | generated | 97 | 139 |
| REMAINDER143 | G13 | after_support | 97 | 139 |
| REMAINDER143 | G13 | feasible | 97 | 139 |
| REMAINDER143 | G13 | selected | 65 | 143 |
| CAL46 | G10 | generated | 39 | 45 |
| CAL46 | G10 | after_support | 39 | 45 |
| CAL46 | G10 | feasible | 39 | 45 |
| CAL46 | G10 | selected | 36 | 46 |
| CAL46 | G13 | generated | 40 | 45 |
| CAL46 | G13 | after_support | 40 | 45 |
| CAL46 | G13 | feasible | 40 | 45 |
| CAL46 | G13 | selected | 36 | 46 |

후보·coverage flag는 전체 이미지에 기록되므로 최종 성공 영상에도 존재할 수 있다. 최종 실패의 원인 수와 혼동하지 않도록 아래 두 분모를 나눈다.

| PRIMARY 전체192 coverage/후보 flag | 이미지 N |
|---|---:|
| REFINEMENT_LOSS | 124 |
| SELECTION_LOSS | 32 |
| OBSERVATION_OR_PROPOSAL_LIMIT | 46 |

최종 실패 78장, 성공 114장, 회전 참조 미확정 0장.

| PRIMARY 최종 실패 영상 안의 flag | 이미지 N |
|---|---:|
| SELECTION_LOSS | 32 |
| REFINEMENT_LOSS | 31 |
| OBSERVATION_OR_PROPOSAL_LIMIT | 46 |

flag는 중복될 수 있다. G10 family 선택과 G13 내부 refinement의 관측 패키지가 달라 두 단계를 한 목적함수의 단일 후보 사다리로 해석하지 않는다. 좋은 후보가 없는 이유가 관측/제안 한계인지 단순 누락인지 기록의 범위 안에서만 나눈다. partial 선 잔차 파일은 실제 선 관측이 있을 때 생성하며 [진단 봉인 파일 목록](diagnostics/DIAGNOSTIC_SEAL.json)을 기준으로 확인한다. [SINGLE 초기값·fallback 감사](oracle/SINGLE_FALLBACK_AUDIT.json), [고정6사례 단계별 진단](oracle/DIAGNOSTIC_CASE_STAGES.json)을 함께 보존한다.

다른 후보의 좋은 위치와 회전을 합쳐 공동 성공을 만들지 않았다. oracle은 고정된 기하 재구성 참조 기준의 저장 후보 도달 가능성이지 물리 참자세의 절대 상한이 아니다. 후보 누락은 실패0건이 아니라 판정 불가로 보존한다. [후보 범위·진단](ORACLE_LIMITS.md), [oracle 전체 결과](oracle/ORACLE_SUMMARY.json).

## 구현 유지·보류와 다음 판단

| 항목 | 권고 | 근거·한계 |
|---|---|---|
| N3/SubPix | 유지 | 기존 고정 비교 결과와 함께 유지한다. 이번 실험은 새 제거 실험이 아니며 typed 복원에서 이 초기 경로를 버릴 근거를 얻지 못함 |
| E 후 B/I 재관측 | 운영 채택 보류 | PRIMARY 기존 성공 손상 MULTI/SINGLE 15/15장·TEST 7/7장. 일부 회복·더 낮은 최대값을 포함해도 무손상 개선이 아님 |
| typed SINGLE/MULTI | 복원 대조군으로 유지 | e와 같은 114/192·49/49이며 새 정확도 향상이 아님. SINGLE 시작정책의 순수 인과분리는 NOT_ISOLATED |
| 다음 판단 | 선택손실과 관측·제안 제한 분리 | 고정 consumer의 선택손실32와 관측·제안 제한46을 각 근거 범위 안에서 다룬다. 선택손실이 모두 큰 W/D 오류라는 결론이나 사후 GT 선택기를 만들지 않음 |

운영 기본값·논문·가중치는 교체하지 않았다. TYPED_MULTI가 원 e와 같은 관측·후보·선택을 복원했다면 동등한 복원 대조군으로 해석하며, E0로 복귀한 것을 e보다 향상된 새 방법으로 세지 않는다. 전체 손상·원49 보존·평균과 tail을 함께 확인하고 유리한 영상별 routing이나 새 비열등성 margin을 만들지 않는다.

후속 판단은 좋은 후보 부재, 중간 pruning, 최종 선택, refinement 이동을 기록이 허용하는 범위에서 나누는 것이다. 큰 실패가 남았다는 사실만으로 상태별 처리의 기여를 폐기하거나 같은 Gaussian·좌표 loss·교란/replay 학습을 새 제안으로 반복하지 않는다. 현재 학습 오류지원은 미확인이며 새 학습안은 채택하지 않았다.

## 실제 실행량·시간

| 계측 | 실제 저장값 |
|---|---:|
| 본 분석 새 학습 update | 0 |
| 본 분석 새 neural forward | 0 |
| 본 분석 새 pose solve | 0 |
| 분석 프로세스 elapsed s | 23.299 |
| 분석 process CPU s | 23.297 |
| 잔차 진단 elapsed s | 17.717 |
| 잔차 진단 process CPU s | 17.706 |
| 실행 장부 cpu_solver_analysis_stage_elapsed_seconds | 3176.841 |
| 실행 장부 gpu_process_elapsed_seconds | 0.000 |
| 실행 장부 whole_task_wall_seconds | 5638.630 |
| 실행 장부 worker_cpu_seconds | 7676.278 |
| 실행 장부 top_level_new_pose_slots | 952.000 |
| 실행 장부 unique_slot_ID_arm_pairs | 952.000 |
| 실행 장부 pilot_slots_reused_by_full_run | 24.000 |
| 실행 장부 additional_original_e_logging_slots | 0.000 |
| 실행 장부 original_e_logging_shared_with_TYPED_MULTI | 238.000 |

위 분석 프로세스의 solve0은 네 진단 경로의 자세 재계산0이라는 뜻이 아니다. 재계산 슬롯·내부 후보 기록·GPU/CPU 누적량은 별도 실행 장부를 기준으로 한다. 겹쳐 실행된 단계 시간을 임의로 더해 전체 wall time이나 core-time으로 바꾸지 않았다. 미기록 값은 NA이며, 저장 pose 재집계로 full-RGB FPS를 추정하지 않는다. 범위는 캐시된 관측 이후의 pose replay·저장 결과 분석이며 원래 E0 상위 pipeline 비용은 추가다. 과거 소스 읽기·첫 실패 추출의 CPU 정밀 계측은 복원하지 않았다. 입력 materialization은 whole-task wall에 들어가지만 runner 단계 타이머에서는 빠져 있다. pilot24는 전체952에 재사용된 것으로 추가952+24로 세지 않는다. 원 e 로깅238도 TYPED_MULTI와 공유하며 별도 pose 슬롯을 중복 추가하지 않는다.

| 경로 | 새 상위 슬롯 N | 슬롯 elapsed 합 s | 실제 worker CPU s | 슬롯 중앙/최대 s | SLSQP 후보 기록 N | seed least-squares 시도 N |
|---|---:|---:|---:|---:|---:|---:|
| REOBS_MULTI | 238 | 642.151 | 642.074 | 2.641/11.165 | 13882 | 13501 |
| REOBS_SINGLE | 238 | 10.024 | 10.021 | 0.032/0.180 | 222 | 0 |
| TYPED_MULTI | 238 | 6941.057 | 6936.885 | 15.448/581.786 | NA | NA |
| TYPED_SINGLE | 238 | 87.312 | 87.298 | 0.265/15.466 | NA | 0 |

후보 기록·seed 시도는 optimizer function evaluation 수나 전체 내부 solve 수와 같지 않다. function evaluation의 미계측을0으로 채우지 않았으며 TYPED 경로의 해당 분모는 NA다.

| 저장 자원 장부 단계 | 종류 | elapsed s |
|---|---|---:|
| identity_freeze | cpu_solver_analysis | 1.361 |
| pilot_REOBS_MULTI_REOBS_SINGLE | cpu_solver_analysis | 6.793 |
| pilot_TYPED_MULTI_TYPED_SINGLE | cpu_solver_analysis | 61.446 |
| lineage_full_extract | cpu_solver_analysis | 72.953 |
| lineage_finalcopy | cpu_solver_analysis | 1.124 |
| lineage_preflight | cpu_solver_analysis | 8.722 |
| original_candidate_bank_extract | cpu_solver_analysis | 24.820 |
| pure_test_initial_fixture_failure | cpu_solver_analysis | 1.730 |
| pure_contract_tests_24 | cpu_solver_analysis | 2.159 |
| pilot_typed_parity | cpu_solver_analysis | 0.687 |
| actual_factor_continuity | cpu_solver_analysis | 6.739 |
| one_CAL_missing_raw_start_logging_repair | cpu_solver_analysis | 5.422 |
| full_REOBS_MULTI_REOBS_SINGLE_TYPED_MULTI_TYPED_SINGLE | cpu_solver_analysis | 2746.572 |
| actual_six_case_gallery_render | cpu_solver_analysis | 19.772 |
| independent_Rt_reaggregation | cpu_solver_analysis | 6.563 |
| actual_eight_case_gallery_render | cpu_solver_analysis | 19.471 |
| actual_gallery_source_numeric_HTML_QA | cpu_solver_analysis | 5.003 |
| final_frozen_input_source_contract_verification | cpu_solver_analysis | 5.764 |
| postseal_score_v1 | cpu_solver_analysis | 25.773 |
| postseal_metadata_only_score_v2 | cpu_solver_analysis | 23.299 |
| postseal_state_and_posefix_diagnostics | cpu_solver_analysis | 17.717 |
| postseal_original_consumer_candidate_oracle | cpu_solver_analysis | 110.472 |
| final_new_package_tests | cpu_solver_analysis | 2.479 |

[실제 비용과 정의](evidence/COST.json), [자원·실행 증거](evidence/RESOURCE_LEDGER.jsonl), [분석 봉인](metrics/SCORE_SEAL.json), [진단 봉인](diagnostics/DIAGNOSTIC_SEAL.json).

## 재현·독립 검산·게시 상태

독립 식(SciPy 회전벡터·별도 중심 norm)으로 4522개 행별 오차를 재집계했고 저장 discrepancy는 0개, 공동 성공 전량 일치는 True다. 허용치는 위치1e-09cm/회전1e-06°다. 실제 기록은 [독립 행 검산](evidence/INDEPENDENT_REAGGREGATION.json)에 있다.

4모집단×19경로=76개 집계의 전체N·성공·availableN·평균·표본분산 검산 all_pass는 True다. [독립 집계 확인](evidence/INDEPENDENT_SUMMARY_CHECK.json)을 별도로 보존한다. 이는 bootstrap CI나 의미 owner 정답까지 독립 검증했다는 뜻이 아니다.

SCORE v2는 TS의 E0 초기화·SINGLE ambiguity 범위만 바로잡은 메타데이터 수정이다. 4,522개 metric dict·76개 수치 집계·모든 짝대비/CI의 v1/v2 exact 일치를 확인했다. 기존 diagnostics v1·gallery·독립 metric 영수증은 [동등성 연결 영수증](evidence/SCORE_V1_V2_METADATA_BRIDGE.json)으로 연결하며 재추론하지 않았다.

이 보고서는 저장 결과에서 생성했다. [재현 안내](REPRODUCE.md), [실제 단계별 갤러리](gallery/index.html), [선행·기여 범위](CONTRIBUTION_EVIDENCE.md), [문헌·소스 접근 장부](LITERATURE_SOURCE_AUDIT.json)를 연결한다. 실제 push SHA와 원격 게시 확인은 별도의 게시 영수증으로 확인하며, 이 생성기가 게시 완료를 선언하지 않는다.

| 봉인·소스 | SHA256 |
|---|---|
| PREDICTION_SEAL | `b7028b2ce160298e274a7035f3ea6367bcf0e14bcf3c8a5b372823c4d3b4b148` |
| CANDIDATE_SEAL | `21673626100d4fb589a9d922844fe30a78bd4c68c73d95b5f6dd991437fce8cb` |
| SCORE_SEAL | `f6302dfa19289d794cc2173e64f7d97bc358276fc1662293782efd3f70240d54` |
| DIAGNOSTIC_SEAL | `e04856d62dc4d81813bc022fe83ab234716ee4975feff741c459f854f048eac9` |
| EXTRACTION_SEAL | `54bdc62546a1e35ed558cd3086563970c58c258fa4bf3acb684238740ae9651d` |
| analysis.py | `71580dd79c2ae62a03325ff7bd3e05adde9f0b6234a0e77f64a34c168d88da97` |
| report.py | `e37babf6372db2db83741aee2e5274edc53ee40b7b88a3aa924eade0fe3458be` |
| SCORE_V1_V2_METADATA_BRIDGE | `b859bba99b9741dfcb95b56b4552407a6749dd57333728693ca3c21c871aab26` |

## 요구 산출물 확인

| 산출물 | 저장 증거 상태 |
|---|---|
| 전체19경로4모집단 | 실제 봉인 score에서 생성 |
| PoseFix+SubPix 공통95 두 cap | 둘 다 보고; 전체192 미확립 |
| CAL 잔차·good 입력 진단 | 실제 진단 봉인 확인 |
| 후보 oracle | 결과 파일 존재; coverage 제한 확인 필요 |
| 독립 검산 | 영수증 존재; 실제 실패·범위 확인 필요 |
| 통합 COST | 명시 COST 기록 존재 |

파일 존재·소스 감사·수치 검산과 방법 효과 입증은 별개다. 누락/불완전 항목을 완료로 승격하지 않았다. 원 RGB·대형 저널·기존 실패 시도·원참조·사용자 수정은 보존하며, 원 참조 좌표 파일은 분석 전용 로컬 증거다.
