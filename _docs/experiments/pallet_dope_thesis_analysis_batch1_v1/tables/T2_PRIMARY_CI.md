# T2 PRIMARY paired bootstrap

모든 변화량은 후보−비교군의 percentage point(%p)다. 영상/녹화 단위로 같은 복원추출 가중치를 두 모델에 적용하고, 선택된 모든 기준점·관측점·정답점의 합으로 비율을 계산했다. 영상당 비율을 단순 평균하지 않는다.

10000회, seed20261026, PCG64 multinomial, percentile95 및 linear quantile. 같은 panel×unit 안의 모든 모델·비교는 동일 표본을 사용한다. 다중 비교 보정은 하지 않았다. CI하한>0은 “높았다”, 상한<0은 “낮았다”, 그 외는 “차이 불명확”이다.

조건부 정밀도는 error≤10 정답점 수/관측점 수다. 0분모 반복은 미정의로 기록하며 재추출하지 않는다. 미정의가 있으면 정의된 반복만의 조건부 percentile을 표시하고 유효/미정의 반복 수를 함께 보고한다. 녹화 ID가 하나라도 없거나 확인된 녹화가3개 미만이면 녹화CI를 계산하지 않는다. 완전한 녹화가5개 이상이면 녹화CI가 본문 기준이다.

| Panel | Unit | Known rec | Missing IDs | Pair | Metric | Δ %p | CI low | CI high | 유효/미정의 | 판정 | 상태 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R1 | PCK5 | -0.8122 | -2.8427 | 1.2085 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R1 | PCK10 | 1.7259 | -0.8155 | 4.2946 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R1 | PCK20 | 4.7716 | 1.3171 | 8.3590 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R1 | coverage | 11.6751 | 7.0263 | 16.4813 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R1 | conditional_precision10 | -5.2486 | -8.6132 | -1.9134 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R2 | PCK5 | -1.1168 | -3.7640 | 1.4344 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R2 | PCK10 | -3.5533 | -6.3572 | -0.8089 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R2 | PCK20 | 3.3503 | 0.1015 | 6.6734 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R2 | coverage | 13.6041 | 9.3910 | 18.0062 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_control_R2 | conditional_precision10 | -11.6162 | -14.9951 | -8.2830 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R1 | PCK5 | 4.0609 | 1.8443 | 6.2249 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R1 | PCK10 | 4.2640 | 1.2170 | 7.3220 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R1 | PCK20 | 5.4822 | 1.8072 | 9.2765 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R1 | coverage | 0.8122 | -4.3260 | 5.7204 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R1 | conditional_precision10 | 4.7859 | 1.2550 | 8.3513 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R2 | PCK5 | 5.7868 | 3.0999 | 8.5085 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R2 | PCK10 | 4.3655 | 1.3265 | 7.4414 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R2 | PCK20 | 8.1218 | 4.5918 | 11.8355 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R2 | coverage | 7.3096 | 4.0568 | 10.8300 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_vs_loo_R2 | conditional_precision10 | 1.4850 | -1.6549 | 4.6423 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_R2_vs_R1 | PCK5 | 2.0305 | -0.1011 | 4.1280 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_R2_vs_R1 | PCK10 | 1.4213 | -1.0102 | 3.7926 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_R2_vs_R1 | PCK20 | 5.8883 | 2.8629 | 8.8686 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_R2_vs_R1 | coverage | 11.5736 | 7.9716 | 15.5671 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | flip_R2_vs_R1 | conditional_precision10 | -4.2200 | -7.1935 | -1.3246 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | control_R2_vs_R1 | PCK5 | 2.3350 | 0.5076 | 4.1710 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | control_R2_vs_R1 | PCK10 | 6.7005 | 4.0816 | 9.4225 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | control_R2_vs_R1 | PCK20 | 7.3096 | 4.7235 | 10.1112 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | control_R2_vs_R1 | coverage | 9.6447 | 6.1411 | 13.3540 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_PLASTIC | frame | 7 | 5 | control_R2_vs_R1 | conditional_precision10 | 2.1477 | -0.8531 | 5.1747 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R1 | PCK5 | -0.8122 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R1 | PCK10 | 1.7259 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R1 | PCK20 | 4.7716 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R1 | coverage | 11.6751 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R1 | conditional_precision10 | -5.2486 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R2 | PCK5 | -1.1168 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R2 | PCK10 | -3.5533 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R2 | PCK20 | 3.3503 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R2 | coverage | 13.6041 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_control_R2 | conditional_precision10 | -11.6162 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R1 | PCK5 | 4.0609 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R1 | PCK10 | 4.2640 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R1 | PCK20 | 5.4822 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R1 | coverage | 0.8122 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R1 | conditional_precision10 | 4.7859 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R2 | PCK5 | 5.7868 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R2 | PCK10 | 4.3655 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R2 | PCK20 | 8.1218 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R2 | coverage | 7.3096 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_vs_loo_R2 | conditional_precision10 | 1.4850 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_R2_vs_R1 | PCK5 | 2.0305 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_R2_vs_R1 | PCK10 | 1.4213 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_R2_vs_R1 | PCK20 | 5.8883 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_R2_vs_R1 | coverage | 11.5736 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | flip_R2_vs_R1 | conditional_precision10 | -4.2200 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | control_R2_vs_R1 | PCK5 | 2.3350 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | control_R2_vs_R1 | PCK10 | 6.7005 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | control_R2_vs_R1 | PCK20 | 7.3096 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | control_R2_vs_R1 | coverage | 9.6447 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_PLASTIC | recording | 7 | 5 | control_R2_vs_R1 | conditional_precision10 | 2.1477 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R1 | PCK5 | -6.9364 | -12.3167 | -1.6895 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R1 | PCK10 | -4.6243 | -10.2639 | 0.8772 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R1 | PCK20 | 9.5376 | 5.1724 | 14.3275 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R1 | coverage | 10.1156 | 5.9829 | 14.7826 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R1 | conditional_precision10 | -12.2122 | -17.9046 | -6.4571 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R2 | PCK5 | -0.5780 | -5.2790 | 4.0936 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R2 | PCK10 | 6.6474 | 1.7341 | 11.5820 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R2 | PCK20 | 11.8497 | 6.8571 | 17.9412 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R2 | coverage | 10.1156 | 5.1724 | 16.1290 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_control_R2 | conditional_precision10 | 0.4025 | -4.7214 | 5.4759 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R1 | PCK5 | -1.7341 | -4.0936 | 0.5780 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R1 | PCK10 | -1.4451 | -5.9829 | 2.8986 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R1 | PCK20 | 1.7341 | -0.8671 | 4.8991 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R1 | coverage | 2.3121 | -0.2857 | 5.5882 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R1 | conditional_precision10 | -2.9820 | -7.6249 | 1.6559 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R2 | PCK5 | -0.8671 | -4.3353 | 2.5862 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R2 | PCK10 | 5.2023 | 2.6237 | 7.8947 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R2 | PCK20 | 1.7341 | 0.2849 | 3.6827 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R2 | coverage | 2.8902 | 1.1236 | 5.2786 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_vs_loo_R2 | conditional_precision10 | 3.6335 | 0.9000 | 6.4514 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_R2_vs_R1 | PCK5 | 5.4913 | 1.1461 | 10.0865 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_R2_vs_R1 | PCK10 | 11.8497 | 6.9164 | 16.8605 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_R2_vs_R1 | PCK20 | 4.3353 | 0.8646 | 8.0931 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_R2_vs_R1 | coverage | 4.9133 | 2.2792 | 7.9882 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | flip_R2_vs_R1 | conditional_precision10 | 9.5941 | 4.3423 | 14.7833 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | control_R2_vs_R1 | PCK5 | -0.8671 | -4.3356 | 2.6239 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | control_R2_vs_R1 | PCK10 | 0.5780 | -2.8902 | 4.0936 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | control_R2_vs_R1 | PCK20 | 2.0231 | -0.5900 | 4.8991 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | control_R2_vs_R1 | coverage | 4.9133 | 2.2921 | 7.8035 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_WOOD | frame | 2 | 0 | control_R2_vs_R1 | conditional_precision10 | -3.0207 | -6.8300 | 0.6991 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R1 | PCK5 | -6.9364 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R1 | PCK10 | -4.6243 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R1 | PCK20 | 9.5376 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R1 | coverage | 10.1156 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R1 | conditional_precision10 | -12.2122 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R2 | PCK5 | -0.5780 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R2 | PCK10 | 6.6474 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R2 | PCK20 | 11.8497 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R2 | coverage | 10.1156 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_control_R2 | conditional_precision10 | 0.4025 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R1 | PCK5 | -1.7341 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R1 | PCK10 | -1.4451 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R1 | PCK20 | 1.7341 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R1 | coverage | 2.3121 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R1 | conditional_precision10 | -2.9820 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R2 | PCK5 | -0.8671 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R2 | PCK10 | 5.2023 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R2 | PCK20 | 1.7341 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R2 | coverage | 2.8902 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_vs_loo_R2 | conditional_precision10 | 3.6335 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_R2_vs_R1 | PCK5 | 5.4913 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_R2_vs_R1 | PCK10 | 11.8497 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_R2_vs_R1 | PCK20 | 4.3353 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_R2_vs_R1 | coverage | 4.9133 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | flip_R2_vs_R1 | conditional_precision10 | 9.5941 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | control_R2_vs_R1 | PCK5 | -0.8671 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | control_R2_vs_R1 | PCK10 | 0.5780 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | control_R2_vs_R1 | PCK20 | 2.0231 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | control_R2_vs_R1 | coverage | 4.9133 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_WOOD | recording | 2 | 0 | control_R2_vs_R1 | conditional_precision10 | -3.0207 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_FEWER_THAN3_RECORDINGS |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R1 | PCK5 | -1.3786 | -4.8832 | 2.0170 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R1 | PCK10 | 0.6363 | -2.3183 | 3.7275 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R1 | PCK20 | 1.8028 | -0.4242 | 4.2553 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R1 | coverage | 3.1813 | 1.0593 | 5.7571 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R1 | conditional_precision10 | -1.6790 | -4.5988 | 1.1608 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R2 | PCK5 | -3.3934 | -6.7653 | -0.1056 | 10000/0 | 낮았다 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R2 | PCK10 | -0.5302 | -3.9404 | 2.8572 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R2 | PCK20 | -0.6363 | -3.1712 | 1.9007 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R2 | coverage | 1.8028 | 0.4242 | 3.6055 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_control_R2 | conditional_precision10 | -1.8459 | -5.3757 | 1.6191 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R1 | PCK5 | 4.0297 | 1.7003 | 6.2634 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R1 | PCK10 | 1.1665 | -1.6061 | 3.8218 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R1 | PCK20 | 0.3181 | -1.9169 | 2.5424 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R1 | coverage | 0.1060 | -0.5291 | 0.7455 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R1 | conditional_precision10 | 1.1108 | -1.6285 | 3.7487 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R2 | PCK5 | 0.6363 | -1.1628 | 2.3355 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R2 | PCK10 | 1.9088 | 0.0000 | 3.7194 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R2 | PCK20 | -0.3181 | -1.8124 | 1.0616 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R2 | coverage | 0.1060 | -0.6363 | 0.9534 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_vs_loo_R2 | conditional_precision10 | 1.8520 | 0.0000 | 3.6969 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_R2_vs_R1 | PCK5 | 0.0000 | -2.5505 | 2.5560 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_R2_vs_R1 | PCK10 | 0.5302 | -2.4237 | 3.5979 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_R2_vs_R1 | PCK20 | -0.3181 | -2.3479 | 1.9027 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_R2_vs_R1 | coverage | 0.7423 | -0.2130 | 2.0556 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | flip_R2_vs_R1 | conditional_precision10 | 0.0088 | -2.9292 | 2.9999 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | control_R2_vs_R1 | PCK5 | 2.0148 | -0.6383 | 4.5696 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | control_R2_vs_R1 | PCK10 | 1.6967 | -0.3168 | 3.7037 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | control_R2_vs_R1 | PCK20 | 2.1209 | 0.2123 | 4.2373 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | control_R2_vs_R1 | coverage | 2.1209 | 0.7447 | 3.7194 | 10000/0 | 높았다 | COMPUTED |
| PRIMARY_GREEN | frame | 0 | 119 | control_R2_vs_R1 | conditional_precision10 | 0.1757 | -1.8236 | 2.1688 | 10000/0 | 차이 불명확 | COMPUTED |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R1 | PCK5 | -1.3786 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R1 | PCK10 | 0.6363 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R1 | PCK20 | 1.8028 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R1 | coverage | 3.1813 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R1 | conditional_precision10 | -1.6790 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R2 | PCK5 | -3.3934 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R2 | PCK10 | -0.5302 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R2 | PCK20 | -0.6363 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R2 | coverage | 1.8028 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_control_R2 | conditional_precision10 | -1.8459 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R1 | PCK5 | 4.0297 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R1 | PCK10 | 1.1665 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R1 | PCK20 | 0.3181 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R1 | coverage | 0.1060 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R1 | conditional_precision10 | 1.1108 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R2 | PCK5 | 0.6363 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R2 | PCK10 | 1.9088 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R2 | PCK20 | -0.3181 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R2 | coverage | 0.1060 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_vs_loo_R2 | conditional_precision10 | 1.8520 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_R2_vs_R1 | PCK5 | 0.0000 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_R2_vs_R1 | PCK10 | 0.5302 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_R2_vs_R1 | PCK20 | -0.3181 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_R2_vs_R1 | coverage | 0.7423 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | flip_R2_vs_R1 | conditional_precision10 | 0.0088 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | control_R2_vs_R1 | PCK5 | 2.0148 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | control_R2_vs_R1 | PCK10 | 1.6967 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | control_R2_vs_R1 | PCK20 | 2.1209 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | control_R2_vs_R1 | coverage | 2.1209 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |
| PRIMARY_GREEN | recording | 0 | 119 | control_R2_vs_R1 | conditional_precision10 | 0.1757 | NA | NA | 0/0 | 미계산 | NOT_COMPUTED_MISSING_RECORDING_ID |

반복 DEV·단일 optimizer seed의 기술적 비교다. 영상 단위CI는 같은 녹화의 연속 프레임 의존성을 반영하지 못한다. GREEN manual600 및 PLASTIC184는 주 평가와 분리된 보조 패널이며 독립 표본으로 합산하지 않는다. PLASTIC184의 추가56은 FINAL역할 자료의 이미 계산된 점수만 사용했고 원본 영상/주석은 열지 않았다.
