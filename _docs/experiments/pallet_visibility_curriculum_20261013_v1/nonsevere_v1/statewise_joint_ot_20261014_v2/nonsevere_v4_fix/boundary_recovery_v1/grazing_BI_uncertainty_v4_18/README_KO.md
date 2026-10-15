# 기하 grazing B/I→U 제안

원래 초기 coarse 기하에서 max incident face cosine이 기존 solver tolerance ±2° 안인 현재 B/I만 U로 유보합니다. 초기 S와 기존 U/O/T는 그대로 둡니다. 새 threshold 학습, GT·프레임 ID·집단별 라우팅은 없습니다. 자세 비교는 아직 실행하지 않았고 기본 방법에 채택하지 않았습니다.

| 자료 | 전체 | 영향 영상/새 U점 | 이전 I/B | 이전 관측 source |
|---|---:|---:|---:|---|
| CAL | 46 | 11/11 | 7/4 | 전부 VISIBLE_BACKBONE_FALLBACK |
| TEST | 49 | 9/9 | 7/2 | 전부 VISIBLE_BACKBONE_FALLBACK |
| REMAINDER | 143 | 11/11 | 8/3 | 전부 VISIBLE_BACKBONE_FALLBACK |

총31장31점입니다. 원49 TEST에서는 noapril 영상의 native7만9개 해당합니다. 추가143에서는031344의 B7이 포함되지만031426/038630은 해당하지 않으므로 이 규칙이 남은 전체 문제를 해결한다고 주장하지 않습니다. source가 fallback인 것은 집계 결과이며 flag 조건에 source를 사용하지 않았습니다.

새 U receiver의 complete2D likelihood/partial 선은 좌표·공분산·선 payload를 보기 전에 제외합니다. 다른 B/I로 전파된 선도 query source가 새 U이면 지웁니다. retained B/I의 whole-edge 선이 새 U의 과거 정확 endpoint를 사용한 경우도 지웁니다. retained B/I가 U mate의 등록3D 방향만 쓴 local 선은 유지합니다. typed I와 legacy B를 모두 지원합니다.

U는 기존 fullCAL264 max 비율0.544750754221928 × 기존 nonS scale의 원을 빌립니다. 새 U의 원래 nonS raw 중심만 읽고, 과거 U domain 객체는 재생성하지 않습니다. 캔버스 clipping/visibility/정확 point·line 비용은 추가하지 않습니다. 이 범위는 U 전용 확률 교정이 아닙니다. S raw 좌표와 영상은 읽지 않습니다.

새 U 원이 없는 missing raw는 전체238에서0개입니다. 어떤 영상도 새로 visible BI2개 미만이 되지는 않았습니다. 다만 complete2D 관측 수는 줄어들 수 있으므로 실제 비교에서 rank/모호성/실패를 그대로 보고해야 합니다. wood_day_01:026085에서는 새 U6에서 전파된 선을 제외한 뒤 남은 B의 원래 complete2D likelihood 한 개가 복원됩니다. 새 점이나 새 보정 좌표를 만든 것은 아닙니다.

향후 control은 동일하게238장 전체에 적용하고, 선택한 parent 알고리즘의 그 외 가중치·calibration·K/Q·모델·mask를 고정해야 합니다. O/T bundle의 actual-image/state 바인딩을 갱신한다면 수치 영역과 scale이 원본과 exact임을 확인해야 합니다. 과거 cached local query가 없었다고 소급 주장하지 않습니다. 새 입력·소스와 전체 예측을 봉인한 뒤에만 재사용 참조 점수를 계산합니다. 원49와 추가143별로 다른 방법을 배포하는 선택은 하지 않습니다.

새 pure source 및8 focused tests가 준비됐습니다. 기존 v49 geometry 계산을 재사용하되 O/T scope를 보존하고 typed I 및 whole-edge endpoint 간접 재사용을 보완한 별도 버전입니다.

[전체238 inventory](INVENTORY_V1.json) · [실행 전 제안 계획](PROPOSED_CONTROL_PLAN_V1.json) · [관측 수 상세](COUNTS_DETAIL_V1.json) · [source snapshot](grazing_BI_uncertainty_v4_18.py) · [tests](test_grazing_BI_uncertainty_v4_18.py)
