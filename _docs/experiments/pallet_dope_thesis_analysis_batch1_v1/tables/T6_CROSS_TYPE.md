# T6 기존 모델의 교차 종류 점수

두 실험의 SCORES.json에 이미 존재하는 모든 모델×3종류 그룹을 그대로 옮겼다. 새 추론·영상·원본 주석·가중치 읽기는 없다. PCK/coverage 표시는 %, 거리 통계는 px다. 누락 예측은 전체 기준점 분모에서 PCK 실패다.

diagonal은 재료별 자기학습 모델의 학습종류와 평가종류가 같은지 뜻한다. 공통 R0 및 합성 추가학습 대조군은 SHARED이며 diagonal 적용 대상이 아니므로 NA로 표시한다. 같은 캐시가 두 실험에 다시 포함되어도 독립 모델로 합산하지 않는다.

| 실험 | 모델 | 학습종류 | 평가종류 | Diagonal | 영상 | 기준점 | PCK5 % | PCK10 % | PCK20 % | coverage % | 조건부 median px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOO_flip | R0_before_selftraining | SHARED | PLASTIC | NA | 128 | 985 | 7.2081 | 26.1929 | 44.8731 | 77.3604 | 14.9531 |
| LOO_flip | R0_before_selftraining | SHARED | WOOD | NA | 45 | 346 | 5.7803 | 29.7688 | 64.4509 | 85.2601 | 12.4620 |
| LOO_flip | R0_before_selftraining | SHARED | GREEN | NA | 119 | 943 | 11.4528 | 47.7200 | 77.4125 | 98.5154 | 10.2374 |
| LOO_flip | GREEN_R1 | GREEN | PLASTIC | False | 128 | 985 | 18.9848 | 37.9695 | 57.0558 | 82.9442 | 10.8480 |
| LOO_flip | GREEN_R1 | GREEN | WOOD | False | 45 | 346 | 23.1214 | 48.8439 | 73.1214 | 82.0809 | 8.4880 |
| LOO_flip | GREEN_R1 | GREEN | GREEN | True | 119 | 943 | 39.3425 | 69.1410 | 82.7147 | 98.4093 | 6.0632 |
| LOO_flip | GREEN_R2 | GREEN | PLASTIC | False | 128 | 985 | 20.3046 | 42.1320 | 60.6091 | 91.0660 | 11.1229 |
| LOO_flip | GREEN_R2 | GREEN | WOOD | False | 45 | 346 | 19.6532 | 49.7110 | 72.2543 | 82.3699 | 8.6708 |
| LOO_flip | GREEN_R2 | GREEN | GREEN | True | 119 | 943 | 39.3425 | 69.6713 | 82.3966 | 99.1516 | 6.2432 |
| LOO_flip | PLASTIC_R1 | PLASTIC | PLASTIC | True | 128 | 985 | 19.0863 | 38.0711 | 53.5025 | 82.0305 | 11.2501 |
| LOO_flip | PLASTIC_R1 | PLASTIC | WOOD | False | 45 | 346 | 22.8324 | 51.1561 | 73.9884 | 83.5260 | 8.0477 |
| LOO_flip | PLASTIC_R1 | PLASTIC | GREEN | False | 119 | 943 | 42.9480 | 69.0350 | 83.3510 | 98.1972 | 5.8223 |
| LOO_flip | PLASTIC_R2 | PLASTIC | PLASTIC | True | 128 | 985 | 21.1168 | 39.4924 | 59.3909 | 93.6041 | 12.2543 |
| LOO_flip | PLASTIC_R2 | PLASTIC | WOOD | False | 45 | 346 | 19.0751 | 47.9769 | 71.9653 | 82.9480 | 9.0479 |
| LOO_flip | PLASTIC_R2 | PLASTIC | GREEN | False | 119 | 943 | 40.9332 | 67.7625 | 84.4115 | 99.1516 | 5.9734 |
| LOO_flip | WOOD_R1 | WOOD | PLASTIC | False | 128 | 985 | 16.2437 | 35.7360 | 52.1827 | 76.5482 | 10.7266 |
| LOO_flip | WOOD_R1 | WOOD | WOOD | True | 45 | 346 | 14.7399 | 48.2659 | 82.9480 | 91.6185 | 9.6786 |
| LOO_flip | WOOD_R1 | WOOD | GREEN | False | 119 | 943 | 30.3287 | 64.4751 | 83.1389 | 98.3033 | 7.1766 |
| LOO_flip | WOOD_R2 | WOOD | PLASTIC | False | 128 | 985 | 20.4061 | 39.4924 | 58.6802 | 86.8020 | 11.0295 |
| LOO_flip | WOOD_R2 | WOOD | WOOD | True | 45 | 346 | 20.2312 | 60.1156 | 87.2832 | 96.5318 | 8.2431 |
| LOO_flip | WOOD_R2 | WOOD | GREEN | False | 119 | 943 | 36.7975 | 66.7020 | 82.9268 | 98.5154 | 6.4442 |
| LOO_flip | SYNTHETIC_CONTROL_R1 | SHARED | PLASTIC | NA | 128 | 985 | 19.8985 | 36.3452 | 48.7310 | 70.3553 | 9.4038 |
| LOO_flip | SYNTHETIC_CONTROL_R1 | SHARED | WOOD | NA | 45 | 346 | 21.6763 | 52.8902 | 73.4104 | 81.5029 | 7.6146 |
| LOO_flip | SYNTHETIC_CONTROL_R1 | SHARED | GREEN | NA | 119 | 943 | 40.7211 | 68.5048 | 80.9120 | 95.2280 | 5.7098 |
| LOO_flip | SYNTHETIC_CONTROL_R2 | SHARED | PLASTIC | NA | 128 | 985 | 22.2335 | 43.0457 | 56.0406 | 80.0000 | 9.1283 |
| LOO_flip | SYNTHETIC_CONTROL_R2 | SHARED | WOOD | NA | 45 | 346 | 20.8092 | 53.4682 | 75.4335 | 86.4162 | 8.3276 |
| LOO_flip | SYNTHETIC_CONTROL_R2 | SHARED | GREEN | NA | 119 | 943 | 42.7359 | 70.2015 | 83.0329 | 97.3489 | 5.6603 |
| LOO_only | R0_before_selftraining | SHARED | PLASTIC | NA | 128 | 985 | 7.2081 | 26.1929 | 44.8731 | 77.3604 | 14.9531 |
| LOO_only | R0_before_selftraining | SHARED | WOOD | NA | 45 | 346 | 5.7803 | 29.7688 | 64.4509 | 85.2601 | 12.4620 |
| LOO_only | R0_before_selftraining | SHARED | GREEN | NA | 119 | 943 | 11.4528 | 47.7200 | 77.4125 | 98.5154 | 10.2374 |
| LOO_only | GREEN_LOO_flip_R1 | GREEN | PLASTIC | False | 128 | 985 | 18.9848 | 37.9695 | 57.0558 | 82.9442 | 10.8480 |
| LOO_only | GREEN_LOO_flip_R1 | GREEN | WOOD | False | 45 | 346 | 23.1214 | 48.8439 | 73.1214 | 82.0809 | 8.4880 |
| LOO_only | GREEN_LOO_flip_R1 | GREEN | GREEN | True | 119 | 943 | 39.3425 | 69.1410 | 82.7147 | 98.4093 | 6.0632 |
| LOO_only | GREEN_LOO_flip_R2 | GREEN | PLASTIC | False | 128 | 985 | 20.3046 | 42.1320 | 60.6091 | 91.0660 | 11.1229 |
| LOO_only | GREEN_LOO_flip_R2 | GREEN | WOOD | False | 45 | 346 | 19.6532 | 49.7110 | 72.2543 | 82.3699 | 8.6708 |
| LOO_only | GREEN_LOO_flip_R2 | GREEN | GREEN | True | 119 | 943 | 39.3425 | 69.6713 | 82.3966 | 99.1516 | 6.2432 |
| LOO_only | PLASTIC_LOO_flip_R1 | PLASTIC | PLASTIC | True | 128 | 985 | 19.0863 | 38.0711 | 53.5025 | 82.0305 | 11.2501 |
| LOO_only | PLASTIC_LOO_flip_R1 | PLASTIC | WOOD | False | 45 | 346 | 22.8324 | 51.1561 | 73.9884 | 83.5260 | 8.0477 |
| LOO_only | PLASTIC_LOO_flip_R1 | PLASTIC | GREEN | False | 119 | 943 | 42.9480 | 69.0350 | 83.3510 | 98.1972 | 5.8223 |
| LOO_only | PLASTIC_LOO_flip_R2 | PLASTIC | PLASTIC | True | 128 | 985 | 21.1168 | 39.4924 | 59.3909 | 93.6041 | 12.2543 |
| LOO_only | PLASTIC_LOO_flip_R2 | PLASTIC | WOOD | False | 45 | 346 | 19.0751 | 47.9769 | 71.9653 | 82.9480 | 9.0479 |
| LOO_only | PLASTIC_LOO_flip_R2 | PLASTIC | GREEN | False | 119 | 943 | 40.9332 | 67.7625 | 84.4115 | 99.1516 | 5.9734 |
| LOO_only | WOOD_LOO_flip_R1 | WOOD | PLASTIC | False | 128 | 985 | 16.2437 | 35.7360 | 52.1827 | 76.5482 | 10.7266 |
| LOO_only | WOOD_LOO_flip_R1 | WOOD | WOOD | True | 45 | 346 | 14.7399 | 48.2659 | 82.9480 | 91.6185 | 9.6786 |
| LOO_only | WOOD_LOO_flip_R1 | WOOD | GREEN | False | 119 | 943 | 30.3287 | 64.4751 | 83.1389 | 98.3033 | 7.1766 |
| LOO_only | WOOD_LOO_flip_R2 | WOOD | PLASTIC | False | 128 | 985 | 20.4061 | 39.4924 | 58.6802 | 86.8020 | 11.0295 |
| LOO_only | WOOD_LOO_flip_R2 | WOOD | WOOD | True | 45 | 346 | 20.2312 | 60.1156 | 87.2832 | 96.5318 | 8.2431 |
| LOO_only | WOOD_LOO_flip_R2 | WOOD | GREEN | False | 119 | 943 | 36.7975 | 66.7020 | 82.9268 | 98.5154 | 6.4442 |
| LOO_only | SYNTHETIC_CONTROL_R1 | SHARED | PLASTIC | NA | 128 | 985 | 19.8985 | 36.3452 | 48.7310 | 70.3553 | 9.4038 |
| LOO_only | SYNTHETIC_CONTROL_R1 | SHARED | WOOD | NA | 45 | 346 | 21.6763 | 52.8902 | 73.4104 | 81.5029 | 7.6146 |
| LOO_only | SYNTHETIC_CONTROL_R1 | SHARED | GREEN | NA | 119 | 943 | 40.7211 | 68.5048 | 80.9120 | 95.2280 | 5.7098 |
| LOO_only | SYNTHETIC_CONTROL_R2 | SHARED | PLASTIC | NA | 128 | 985 | 22.2335 | 43.0457 | 56.0406 | 80.0000 | 9.1283 |
| LOO_only | SYNTHETIC_CONTROL_R2 | SHARED | WOOD | NA | 45 | 346 | 20.8092 | 53.4682 | 75.4335 | 86.4162 | 8.3276 |
| LOO_only | SYNTHETIC_CONTROL_R2 | SHARED | GREEN | NA | 119 | 943 | 42.7359 | 70.2015 | 83.0329 | 97.3489 | 5.6603 |
| LOO_only | GREEN_LOO_only_R1 | GREEN | PLASTIC | False | 128 | 985 | 17.9695 | 38.5787 | 56.3452 | 84.9746 | 11.2962 |
| LOO_only | GREEN_LOO_only_R1 | GREEN | WOOD | False | 45 | 346 | 23.1214 | 50.0000 | 71.9653 | 80.0578 | 8.2254 |
| LOO_only | GREEN_LOO_only_R1 | GREEN | GREEN | True | 119 | 943 | 35.3128 | 67.9745 | 82.3966 | 98.3033 | 6.3632 |
| LOO_only | GREEN_LOO_only_R2 | GREEN | PLASTIC | False | 128 | 985 | 19.0863 | 40.9137 | 59.8985 | 92.0812 | 11.7923 |
| LOO_only | GREEN_LOO_only_R2 | GREEN | WOOD | False | 45 | 346 | 19.9422 | 47.9769 | 71.9653 | 82.3699 | 8.9141 |
| LOO_only | GREEN_LOO_only_R2 | GREEN | GREEN | True | 119 | 943 | 38.7063 | 67.7625 | 82.7147 | 99.0456 | 6.3966 |
| LOO_only | PLASTIC_LOO_only_R1 | PLASTIC | PLASTIC | True | 128 | 985 | 15.0254 | 33.8071 | 48.0203 | 81.2183 | 14.0923 |
| LOO_only | PLASTIC_LOO_only_R1 | PLASTIC | WOOD | False | 45 | 346 | 21.0983 | 53.1792 | 72.5434 | 82.3699 | 8.0732 |
| LOO_only | PLASTIC_LOO_only_R1 | PLASTIC | GREEN | False | 119 | 943 | 35.9491 | 63.6267 | 80.3818 | 97.9852 | 6.8524 |
| LOO_only | PLASTIC_LOO_only_R2 | PLASTIC | PLASTIC | True | 128 | 985 | 15.3299 | 35.1269 | 51.2690 | 86.2944 | 13.5923 |
| LOO_only | PLASTIC_LOO_only_R2 | PLASTIC | WOOD | False | 45 | 346 | 19.3642 | 51.1561 | 73.1214 | 83.8150 | 8.7852 |
| LOO_only | PLASTIC_LOO_only_R2 | PLASTIC | GREEN | False | 119 | 943 | 36.7975 | 64.2630 | 80.9120 | 98.8335 | 6.7875 |
| LOO_only | WOOD_LOO_only_R1 | WOOD | PLASTIC | False | 128 | 985 | 16.4467 | 36.3452 | 53.2995 | 81.7259 | 11.7777 |
| LOO_only | WOOD_LOO_only_R1 | WOOD | WOOD | True | 45 | 346 | 16.4740 | 49.7110 | 81.2139 | 89.3064 | 9.0488 |
| LOO_only | WOOD_LOO_only_R1 | WOOD | GREEN | False | 119 | 943 | 26.7232 | 64.1569 | 82.3966 | 97.0308 | 7.2654 |
| LOO_only | WOOD_LOO_only_R2 | WOOD | PLASTIC | False | 128 | 985 | 20.5076 | 40.2030 | 58.5787 | 87.2081 | 11.1395 |
| LOO_only | WOOD_LOO_only_R2 | WOOD | WOOD | True | 45 | 346 | 21.0983 | 54.9133 | 85.5491 | 93.6416 | 8.4375 |
| LOO_only | WOOD_LOO_only_R2 | WOOD | GREEN | False | 119 | 943 | 33.6161 | 65.7476 | 83.7752 | 98.0912 | 6.8349 |

CSV에는 관측/누락 개수, 조건부 P90/평균 및 source별 점 수도 포함한다. 대칭 정렬을 하지 않은 의미 채널0–7 기준의 반복 DEV 비교이며, 출처 미상/PnP 생성 기준점이 포함되고 독립 TEST 또는 물리적6D 정확도를 뜻하지 않는다.
