# 기하 grazing B/I→U 저장 후보 대조군

기존2° 기하 불확실 대역의31점만 U로 바꾸고 정확 B/I 비용·부분선을 유보했습니다. 원래 S와 이전 U/O/T, K/Q/모델/옛 covariance·reliability·마스크·전체 physical 후보 집합을 유지했습니다. 새 모델·RGB·자세 최적화는 없으며, 전체238 예측 봉인 후 기존 참조로 평가했습니다.

| 동일 방법 | TEST49 T≤5cm / joint5cm5° | 평균/최대 T(cm) | 추가143 T≤5cm / joint | 평균/최대 T(cm) |
|---|---:|---:|---:|---:|
| SAME_FROZEN_E | 49/49 | 1.299/4.339 | 66/65 | 18.502/260.412 |
| GRAZING_U_NATIVE_DMERGED | 44/42 | 2.681/40.408 | 74/73 | 13.969/184.072 |
| GRAZING_U_NATIVE_PLUS_MASK_DMERGED | 45/45 | 1.940/16.793 | 74/74 | 14.549/195.406 |

채택하지 않습니다. 원49장49/49 목표가 보존되지 않았습니다. native arm은 U 변경이 없는207장의 이전v17 결과와 자세·family·accepted가 exact 일치했습니다. baseline e는238장 모두 원본과 exact 같습니다. zero-BI/fallback도 제외하지 않았습니다.

noapril142853의 I7이 U로 바뀌어도 native 결과는40.408cm로 남았습니다. 초기 visibility가 불확실했다는 사실과, 그 점을 제외하면 WD 오류가 해결된다는 주장은 구분해야 합니다. 이번 결과는 후자를 지지하지 않습니다. U는 기존 CAL264 max 범위를 빌린 것이며 정확 점이나 확률 교정 보장이 아닙니다.

native와 native+mask는 별도 고정 대조군입니다. mask arm은 원래 outline_factor_weight1을 그대로 더했으며 native-only보다 contrary native cost를 이길 수 있다는 정의를 숨기지 않았습니다. 원본 source/관측의 과거 노출과 재사용 TEST·143 평가라는 한계, 수동 주석 PnP 참조의 한계를 유지합니다. 집단별·프레임별 GT 선택은 없습니다.

[완료/207 parity/31사례](COMPLETION_AND_PAIRED_RECEIPT_V1.json) · [TEST49](TEST_SCORE.json) · [추가143](REMAINDER_SCORE.json) · [실행 protocol](PROTOCOL.json)
