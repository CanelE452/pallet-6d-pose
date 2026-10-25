# V245 동일 장면 CF: 저장 SOURCE 결과

고정 SOURCE gate FAIL. 새 treatment1000과 과거 baseline byte재사용0newupdates의 제한된 SOURCE 비교입니다. SOURCE DEV는 노출·재사용됐으며 새 blind 검증이나 현재 팔레트 실사 성공이 아닙니다.

SYN intended-native witness0와 YCB associated-r AUX(native UNKNOWN)를 별도 표로 유지합니다. 자연 known7617과 CF FIT10812를 합치지 않으며, 원15040 중 masked7423은 보존됩니다. 문턱은각family scalar0입니다.

| family | role | arm | N | DD | DE | DU | ED | EE | EU | D_precision | D_recall | E_precision | E_recall | U_rate | balanced_recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SYN_MAIN | FIT | control | 4752 | 4528 | 37 | 0 | 0 | 187 | 0 | 100.00% | 99.19% | 83.48% | 100.00% | 0.00% | 99.59% |
| SYN_MAIN | FIT | treatment | 4752 | 4485 | 80 | 0 | 0 | 187 | 0 | 100.00% | 98.25% | 70.04% | 100.00% | 0.00% | 99.12% |
| SYN_MAIN | DEV | control | 1344 | 1272 | 16 | 0 | 5 | 51 | 0 | 99.61% | 98.76% | 76.12% | 91.07% | 0.00% | 94.91% |
| SYN_MAIN | DEV | treatment | 1344 | 1263 | 25 | 0 | 5 | 51 | 0 | 99.61% | 98.06% | 67.11% | 91.07% | 0.00% | 94.57% |
| YCB_AUX | FIT | control | 1124 | 841 | 0 | 0 | 0 | 283 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| YCB_AUX | FIT | treatment | 1124 | 840 | 1 | 0 | 0 | 283 | 0 | 100.00% | 99.88% | 99.65% | 100.00% | 0.00% | 99.94% |
| YCB_AUX | DEV | control | 397 | 332 | 8 | 0 | 56 | 1 | 0 | 85.57% | 97.65% | 11.11% | 1.75% | 0.00% | 49.70% |
| YCB_AUX | DEV | treatment | 397 | 296 | 44 | 0 | 51 | 6 | 0 | 85.30% | 87.06% | 12.00% | 10.53% | 0.00% | 48.79% |

고정 gate는 family DEV balanced recall 평균 +3pp 이상·각 family/class recall 감소 2pp 이하·YCB E50% 이상·SYN D/E 각각90% 이상입니다. CF target UNKNOWN은 CM 밖에 남기며 input unavailable U와 구분합니다. CF 성적은 이미 노출된 FIT 진단으로 자연 DEV 일반화 성적이 아닙니다. q/L/native/U 또는 라벨을 수정하지 않았습니다.

[전체 수치](SUMMARY.json) · [metric CSV](NATURAL_METRICS.csv) · [paired 자연행](PAIRED_NATURAL7617.csv) · [CF 진단행](CF_DIAGNOSTIC10812.csv) · [HTML](PORTABLE_REPORT_KO.html)
