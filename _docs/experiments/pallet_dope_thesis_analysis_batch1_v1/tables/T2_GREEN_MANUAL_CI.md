# T2 GREEN_MANUAL paired bootstrap

모든 변화량은 후보−비교군의 percentage point(%p)다. 영상/녹화 단위로 같은 복원추출 가중치를 두 모델에 적용하고, 선택된 모든 기준점·관측점·정답점의 합으로 비율을 계산했다. 영상당 비율을 단순 평균하지 않는다.

10000회, seed20261026, PCG64 multinomial, percentile95 및 linear quantile. 같은 panel×unit 안의 모든 모델·비교는 동일 표본을 사용한다. 다중 비교 보정은 하지 않았다. CI하한>0은 “높았다”, 상한<0은 “낮았다”, 그 외는 “차이 불명확”이다.

조건부 정밀도는 error≤10 정답점 수/관측점 수다. 0분모 반복은 미정의로 기록하며 재추출하지 않는다. 미정의가 있으면 정의된 반복만의 조건부 percentile을 표시하고 유효/미정의 반복 수를 함께 보고한다. 녹화 ID가 하나라도 없거나 확인된 녹화가3개 미만이면 녹화CI를 계산하지 않는다. 완전한 녹화가5개 이상이면 녹화CI가 본문 기준이다.

| Panel | Unit | Known rec | Missing IDs | Pair | Metric | Δ %p | CI low | CI high | 유효/미정의 | 판정 | 상태 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R1 | PCK5 | 0.0000 | -4.0336 | 4.0543 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R1 | PCK10 | 2.1667 | -0.6579 | 5.0679 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R1 | PCK20 | 1.6667 | -0.5051 | 4.2304 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R1 | coverage | 3.5000 | 1.1706 | 6.3519 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R1 | conditional_precision10 | -0.6241 | -3.4261 | 1.9745 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R2 | PCK5 | -2.8333 | -6.4356 | 0.8239 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R2 | PCK10 | 0.6667 | -2.0033 | 3.3333 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R2 | PCK20 | 0.6667 | -1.1609 | 2.5085 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R2 | coverage | 1.6667 | 0.3350 | 3.3167 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_control_R2 | conditional_precision10 | -0.6728 | -3.3183 | 1.9423 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R1 | PCK5 | 4.3333 | 1.6920 | 6.8852 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R1 | PCK10 | -0.3333 | -2.7028 | 1.8740 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R1 | PCK20 | -0.6667 | -2.4511 | 1.0033 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R1 | coverage | 0.3333 | -0.3356 | 1.1905 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R1 | conditional_precision10 | -0.6082 | -3.0304 | 1.6892 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R2 | PCK5 | 0.8333 | -1.3675 | 2.8523 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R2 | PCK10 | 0.6667 | -1.1865 | 2.4752 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R2 | PCK20 | -0.5000 | -1.8707 | 0.6623 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R2 | coverage | -0.1667 | -1.5050 | 0.9853 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_vs_loo_R2 | conditional_precision10 | 0.8055 | -0.9702 | 2.5287 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_R2_vs_R1 | PCK5 | 1.0000 | -1.4706 | 3.5058 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_R2_vs_R1 | PCK10 | 1.0000 | -1.5203 | 3.6789 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_R2_vs_R1 | PCK20 | 1.0000 | -0.6504 | 2.7166 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_R2_vs_R1 | coverage | 0.6667 | -0.6590 | 2.1776 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | flip_R2_vs_R1 | conditional_precision10 | 0.4794 | -2.1740 | 3.1700 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | control_R2_vs_R1 | PCK5 | 3.8333 | 1.0000 | 6.7128 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | control_R2_vs_R1 | PCK10 | 2.5000 | 0.8278 | 4.4143 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | control_R2_vs_R1 | PCK20 | 2.0000 | 0.1634 | 4.1806 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | control_R2_vs_R1 | coverage | 2.5000 | 0.8223 | 4.6205 | 10000/0 | 높았다 | COMPUTED |
| GREEN_MANUAL | frame | 0 | 119 | control_R2_vs_R1 | conditional_precision10 | 0.5281 | -1.2714 | 2.1809 | 10000/0 | 차이 불명확 | COMPUTED |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R1 | PCK5 | 0.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R1 | PCK10 | 2.1667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R1 | PCK20 | 1.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R1 | coverage | 3.5000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R1 | conditional_precision10 | -0.6241 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R2 | PCK5 | -2.8333 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R2 | PCK10 | 0.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R2 | PCK20 | 0.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R2 | coverage | 1.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_control_R2 | conditional_precision10 | -0.6728 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R1 | PCK5 | 4.3333 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R1 | PCK10 | -0.3333 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R1 | PCK20 | -0.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R1 | coverage | 0.3333 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R1 | conditional_precision10 | -0.6082 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R2 | PCK5 | 0.8333 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R2 | PCK10 | 0.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R2 | PCK20 | -0.5000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R2 | coverage | -0.1667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_vs_loo_R2 | conditional_precision10 | 0.8055 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_R2_vs_R1 | PCK5 | 1.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_R2_vs_R1 | PCK10 | 1.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_R2_vs_R1 | PCK20 | 1.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_R2_vs_R1 | coverage | 0.6667 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | flip_R2_vs_R1 | conditional_precision10 | 0.4794 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | control_R2_vs_R1 | PCK5 | 3.8333 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | control_R2_vs_R1 | PCK10 | 2.5000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | control_R2_vs_R1 | PCK20 | 2.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | control_R2_vs_R1 | coverage | 2.5000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| GREEN_MANUAL | recording | 0 | 119 | control_R2_vs_R1 | conditional_precision10 | 0.5281 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |

반복 DEV·단일 optimizer seed의 기술적 비교다. 영상 단위CI는 같은 녹화의 연속 프레임 의존성을 반영하지 못한다. GREEN manual600 및 PLASTIC184는 주 평가와 분리된 보조 패널이며 독립 표본으로 합산하지 않는다. PLASTIC184의 추가56은 FINAL역할 자료의 이미 계산된 점수만 사용했고 원본 영상/주석은 열지 않았다.
