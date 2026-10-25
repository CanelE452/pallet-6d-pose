# V239 canonical SOURCE 두 팔 저장 결과

같은 공통 특징으로 CONTROL 1,000회와 TREAT 1,000회를 각각 새로 학습한 저장 출력의 산술입니다. SYN은 intended-native witness0 MAIN, YCB는 associated-r AUX이며 서로 다른 참조 대상을 하나의 과업으로 합치지 않습니다.

| 가족 | 역할 | 팔 | N | D recall | E recall | D precision | E precision | U |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SYN_MAIN | FIT | control | 4752 | 99.12% | 100.00% | 100.00% | 82.38% | 0 |
| SYN_MAIN | FIT | treatment | 4752 | 99.19% | 100.00% | 100.00% | 83.48% | 0 |
| SYN_MAIN | DEV | control | 1344 | 98.84% | 91.07% | 99.61% | 77.27% | 0 |
| SYN_MAIN | DEV | treatment | 1344 | 98.76% | 91.07% | 99.61% | 76.12% | 0 |
| YCB_AUX | FIT | control | 1124 | 99.88% | 100.00% | 100.00% | 99.65% | 0 |
| YCB_AUX | FIT | treatment | 1124 | 100.00% | 100.00% | 100.00% | 100.00% | 0 |
| YCB_AUX | DEV | control | 397 | 97.94% | 0.00% | 85.38% | 0.00% | 0 |
| YCB_AUX | DEV | treatment | 397 | 97.65% | 1.75% | 85.57% | 11.11% | 0 |

고정 SOURCE gate: FAIL. 평균 DEV balanced recall 변화 0.3457pp. 기준은 +3pp / 각 family·class 하락 최대 2pp / YCB E recall 50% / SYN D·E 각각 90%입니다.

원 15,040행 중 알려진 D/E 7,617행만 이 출력 분모이며 masked 7,423행은 모델 U가 아닙니다. S/U 문맥 노드는 원 입력 지원에 따라 보존됩니다. q-reference 거리 sidecar를 이 wrapper가 읽지 않으므로 거리별 표는 UNKNOWN으로 남깁니다.

SOURCE DEV는 이미 노출·재사용된 자료입니다. 추가 residual 9,956개 파라미터의 효과를 완전한 동일용량 인과 실험으로 해석하지 않습니다. 현재/held 평가·PnP는 새로 수행하지 않았고 current admission=false, 선택 모델 없음입니다.

[전체 CM·P/R·게이트](SUMMARY.json) · [수치표](METRICS.csv) · [행별 paired 출력](ALL7617_PAIRED_ROWS.jsonl)
