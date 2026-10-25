# 저장 q/r footprint 진단

원 FIT known-D 부모 5,406개를 그대로 보존했습니다. r proof PASS 5,367개만 아래 실패율 분모로 사용하며, unconstructed 39개는 삭제하거나 재라벨하지 않았습니다.

| family | q−r | r PASS | footprint | COVER 불투명 실패 | SHAM clear 실패 | 둘 중 하나 실패 | 비율 |
|---|---|---:|---|---:|---:|---:|---:|
| 0 | ≤3 px | 2901 | continuous positive taps | 224 | 80 | 296 | 10.20% |
| 0 | ≤3 px | 2901 | half-up 3×3 | 437 | 155 | 570 | 19.65% |
| 0 | >3 px | 1636 | continuous positive taps | 557 | 225 | 708 | 43.28% |
| 0 | >3 px | 1636 | half-up 3×3 | 673 | 274 | 844 | 51.59% |
| 1 | ≤3 px | 489 | continuous positive taps | 56 | 36 | 84 | 17.18% |
| 1 | ≤3 px | 489 | half-up 3×3 | 93 | 52 | 126 | 25.77% |
| 1 | >3 px | 341 | continuous positive taps | 106 | 40 | 132 | 38.71% |
| 1 | >3 px | 341 | half-up 3×3 | 131 | 51 | 164 | 48.09% |

`둘 중 하나 실패`는 각 footprint에서 COVER all-opaque 실패 또는 SHAM all-clear 실패의 합집합입니다. 두 실패의 중복은 CSV의 both_FAIL에 보존합니다. SHAM RGB 불변 검사와 두 footprint 전체 PASS 여부는 JSON/CSV의 별도 열이며 위 opacity 합집합과 구분합니다. unsupported는 UNKNOWN으로 별도 집계합니다.

family0 SYN의 intended-native witness0와 family1 YCB의 associated-r AUX는 서로 다른 교사입니다. r에서의 증명이 q 위치의 같은 opacity 증명을 자동 보장하지 않는다는 저장 관측이며, **q footprint 불일치는 literal pixel의 실사 가시성 GT 증거가 아닙니다.** 공개 YCB의 q−r GT jitter를 실제 detector 오차로 해석하지 않습니다.

새 RGB·배열·가중치·원 source GT·모델·보호 payload를 열지 않았습니다. 기존 labels/parents/threshold/feature architecture는 바꾸지 않았고 학습·모델 선택·인과 원인 확정은 하지 않았습니다. current admission은 false입니다.

[세부 수치](SUMMARY.json) · [CSV](FOOTPRINT_FAILURES_BY_FAMILY_DISTANCE.csv) · [고정 집계 코드](summarize_saved_qr.py)
