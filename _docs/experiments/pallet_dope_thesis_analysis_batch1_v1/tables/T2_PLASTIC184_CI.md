# T2 PLASTIC184 paired bootstrap

모든 변화량은 후보−비교군의 percentage point(%p)다. 영상/녹화 단위로 같은 복원추출 가중치를 두 모델에 적용하고, 선택된 모든 기준점·관측점·정답점의 합으로 비율을 계산했다. 영상당 비율을 단순 평균하지 않는다.

10000회, seed20261026, PCG64 multinomial, percentile95 및 linear quantile. 같은 panel×unit 안의 모든 모델·비교는 동일 표본을 사용한다. 다중 비교 보정은 하지 않았다. CI하한>0은 “높았다”, 상한<0은 “낮았다”, 그 외는 “차이 불명확”이다.

조건부 정밀도는 error≤10 정답점 수/관측점 수다. 0분모 반복은 미정의로 기록하며 재추출하지 않는다. 미정의가 있으면 정의된 반복만의 조건부 percentile을 표시하고 유효/미정의 반복 수를 함께 보고한다. 녹화 ID가 하나라도 없거나 확인된 녹화가3개 미만이면 녹화CI를 계산하지 않는다. 완전한 녹화가5개 이상이면 녹화CI가 본문 기준이다.

| Panel | Unit | Known rec | Missing IDs | Pair | Metric | Δ %p | CI low | CI high | 유효/미정의 | 판정 | 상태 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R1 | PCK5 | -1.6277 | -3.4629 | 0.1393 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R1 | PCK10 | 0.0000 | -2.3256 | 2.2615 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R1 | PCK20 | 2.5478 | -0.3554 | 5.4853 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R1 | coverage | 7.4310 | 3.7695 | 11.2376 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R1 | conditional_precision10 | -5.7266 | -8.4966 | -2.9741 | 10000/0 | 낮았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R2 | PCK5 | -1.4154 | -3.6827 | 0.7841 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R2 | PCK10 | -2.8309 | -5.0883 | -0.5654 | 10000/0 | 낮았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R2 | PCK20 | 3.2555 | 0.7773 | 5.7625 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R2 | coverage | 11.9604 | 8.4517 | 15.6184 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_control_R2 | conditional_precision10 | -11.4211 | -14.2771 | -8.6636 | 10000/0 | 낮았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R1 | PCK5 | 3.2555 | 1.1291 | 5.3042 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R1 | PCK10 | 1.5570 | -1.2922 | 4.3203 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R1 | PCK20 | 1.7693 | -1.4957 | 4.9577 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R1 | coverage | -3.5386 | -8.1178 | 0.9181 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R1 | conditional_precision10 | 4.2460 | 1.0855 | 7.4990 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R2 | PCK5 | 4.8832 | 2.4647 | 7.2650 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R2 | PCK10 | 2.4062 | -0.4255 | 5.1883 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R2 | PCK20 | 4.1047 | 0.8499 | 7.2034 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R2 | coverage | 3.6801 | 0.7032 | 6.6904 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_vs_loo_R2 | conditional_precision10 | 0.5750 | -2.2385 | 3.3877 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_R2_vs_R1 | PCK5 | 2.8309 | 0.9066 | 4.8468 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_R2_vs_R1 | PCK10 | 3.1847 | 0.7812 | 5.7307 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_R2_vs_R1 | PCK20 | 6.5110 | 3.7323 | 9.3465 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_R2_vs_R1 | coverage | 11.6773 | 8.0929 | 15.4173 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | flip_R2_vs_R1 | conditional_precision10 | -3.6062 | -6.2732 | -0.9828 | 10000/0 | 낮았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | control_R2_vs_R1 | PCK5 | 2.6185 | 0.9233 | 4.4586 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | control_R2_vs_R1 | PCK10 | 6.0156 | 3.8924 | 8.3042 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | control_R2_vs_R1 | PCK20 | 5.8033 | 3.5663 | 8.2027 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | control_R2_vs_R1 | coverage | 7.1479 | 4.0874 | 10.2346 | 10000/0 | 높았다 | COMPUTED |
| PLASTIC184 | frame | 9 | 5 | control_R2_vs_R1 | conditional_precision10 | 2.0883 | -0.3948 | 4.6789 | 10000/0 | 차이 불명확 | COMPUTED |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R1 | PCK5 | -1.6277 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R1 | PCK10 | 0.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R1 | PCK20 | 2.5478 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R1 | coverage | 7.4310 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R1 | conditional_precision10 | -5.7266 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R2 | PCK5 | -1.4154 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R2 | PCK10 | -2.8309 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R2 | PCK20 | 3.2555 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R2 | coverage | 11.9604 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_control_R2 | conditional_precision10 | -11.4211 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R1 | PCK5 | 3.2555 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R1 | PCK10 | 1.5570 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R1 | PCK20 | 1.7693 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R1 | coverage | -3.5386 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R1 | conditional_precision10 | 4.2460 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R2 | PCK5 | 4.8832 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R2 | PCK10 | 2.4062 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R2 | PCK20 | 4.1047 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R2 | coverage | 3.6801 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_vs_loo_R2 | conditional_precision10 | 0.5750 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_R2_vs_R1 | PCK5 | 2.8309 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_R2_vs_R1 | PCK10 | 3.1847 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_R2_vs_R1 | PCK20 | 6.5110 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_R2_vs_R1 | coverage | 11.6773 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | flip_R2_vs_R1 | conditional_precision10 | -3.6062 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | control_R2_vs_R1 | PCK5 | 2.6185 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | control_R2_vs_R1 | PCK10 | 6.0156 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | control_R2_vs_R1 | PCK20 | 5.8033 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | control_R2_vs_R1 | coverage | 7.1479 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PLASTIC184 | recording | 9 | 5 | control_R2_vs_R1 | conditional_precision10 | 2.0883 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |

반복 DEV·단일 optimizer seed의 기술적 비교다. 영상 단위CI는 같은 녹화의 연속 프레임 의존성을 반영하지 못한다. GREEN manual600 및 PLASTIC184는 주 평가와 분리된 보조 패널이며 독립 표본으로 합산하지 않는다. PLASTIC184의 추가56은 FINAL역할 자료의 이미 계산된 점수만 사용했고 원본 영상/주석은 열지 않았다.

## 보호 FINAL 파생 점수 사용 범위

[확인] PLASTIC184 중56장은 보호 FINAL과 선언 이미지 SHA가 같다. 기존 파생 SCORES의 reference.xy는 모델 간 대응 검사에, error_px는 CI 계산에 사용했다. 원본 FINAL RGB·annotation 본문 열람은0회다. 이 사실을 공개하고 2026-10-27 사용자의 “기존 파생 SCORES 사용만 허용하고 push” 답변에 따라 포함했다. preflight/GATE5_RESOLUTION.json의 좁은 허용 범위이며 주 DEV292 결과와 분리한다.
