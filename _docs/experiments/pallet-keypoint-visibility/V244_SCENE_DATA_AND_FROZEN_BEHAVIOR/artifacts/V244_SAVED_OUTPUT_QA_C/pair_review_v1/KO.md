# 이전 SOURCE FIT pair 행동의 저장 산술 검산

원 5,406행을 유지했습니다. 성공 5,367개(SYN 4,537/YCB 830)만 pair 판정 분모이며, 실패 39개(SYN 28/YCB 11)의 child target−1·null logit·U는 그대로 보존했습니다. 원 teacher D는 유지됩니다.

| 그룹 | pair 분모 | COVER E | SHAM D | 둘 다 | COVER>SHAM / = / < |
|---|---:|---:|---:|---:|---:|
| ALL | 5367 | 5025 | 5079 | 4794 | 5234 / 0 / 133 |
| SYN_MAIN | 4537 | 4236 | 4294 | 4043 | 4417 / 0 / 120 |
| YCB_ASSOCIATED_R_AUX | 830 | 789 | 785 | 751 | 817 / 0 / 13 |

YCB의 원 saved natural D 정답 829개 중 COVER E 전환은 788개, SHAM D 보존은 785개입니다.

이는 이미 학습에 사용한 SOURCE FIT pair의 함수적 재생 진단입니다. 독립 자연 DEV 일반화나 실사 성공이 아닙니다. SYN intended-native witness0와 YCB associated-r AUX를 별도 의미로 유지하며 native physical/literal-q gold로 바꾸지 않습니다. 원 natural saved logit은 old-prefix route, pair는 old-CF fullforward 재구성이므로 원 D delta는 서술 통계이며 저장되지 않았던 과거 CF 특징과 byte 비교한 증거가 아닙니다. object/aux mapping은 UNKNOWN 그대로이고 모델·데이터·문턱 선택은 없습니다. current admission은 false입니다.

[REVIEW](results/REVIEW.json) · [stdlib 검산기](verify_saved_pairs.py) · [실행 안내](README_KO.md)
