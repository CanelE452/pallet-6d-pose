# V30 고정 두 사례: 실제 선 후보와 저장된 최종 자세

[실제 과정 HTML](index.html) · [목재 9단계](case0_montage.png) · [플라스틱 9단계](case1_montage.png) · [저장 원장](PROCESS_DATA.json) · [렌더 봉인](RENDER_SEAL.json)

두 사례와 다섯 방법 순서는 점수 전에 고정했다. ①–④는 공통 저장 관측이며 ⑤–⑨는 서로 다른 최종 방법 비교다. 중간 optimizer 궤적이 아니다. 원 RGB 밝기와 색을 유지했고 S/U 점과 GT wire는 그리지 않았다. U는 범위, S는 모델 wire로만 표시하며 기존 bounded S 정규화 항은 계속 활성이다. 독립 other-object mask가 없어 target mask 부재만으로 실제 가림을 확정하지 않는다.

V30은 원 V29 raw chain을 유지하고 모든 eligible B/I q와 무한 TLS 선의 법선 거리를 기존 43px 범위로 검사한 새 선택 모집단이다. 정책별 새 CAL 잡음을 별도로 적합했다. 이 검사는 along-line native 대응을 확정하지 않는다. 그래프 선정 후보와 AUTH/source 정책을 통과한 실제 활성 선을 구분한다.

## wood_night_01:031426

| 방법 | WD | T (cm) | R (°) | joint 수용 | 활성 선 |
|---|---|---:|---:|---|---:|
| 기존 e | ORIGINAL_WD | 208.2242 | 83.7512 | 예 | 0 |
| ALL native | SWAPPED_WD | 13.0346 | 8.8504 | 예 | 0 |
| ALL containment | ORIGINAL_WD | 191.3135 | 82.3232 | 예 | 0 |
| SOURCE native | SWAPPED_WD | 13.0346 | 8.8504 | 예 | 0 |
| SOURCE containment | ORIGINAL_WD | 191.3135 | 82.3232 | 예 | 0 |

## eval_pallet09:1778653661653195264

| 방법 | WD | T (cm) | R (°) | joint 수용 | 활성 선 |
|---|---|---:|---:|---|---:|
| 기존 e | SWAPPED_WD | 2.4542 | 3.4591 | 예 | 0 |
| ALL native | SWAPPED_WD | 237.2541 | 55.5067 | 예 | 3 |
| ALL containment | SWAPPED_WD | 237.2541 | 55.5067 | 예 | 3 |
| SOURCE native | SWAPPED_WD | 4.5924 | 6.0398 | 예 | 0 |
| SOURCE containment | SWAPPED_WD | 4.6976 | 6.2471 | 예 | 0 |

목재: 그래프 후보 3개가 있어도 새 방법의 활성 선 요인은 모두 0개다. native의 208.2242→13.0346cm 차이를 새 선 보정의 인과 증거로 해석하지 않는다. containment에서는 191.3135cm가 남는다. 이전 176.5351cm는 V24의 다른 MASK_FIT 방법 결과이며 기존 e가 아니다.

플라스틱: 그래프 후보 7개 중 ALL 방법에는 실제 활성 선 3개가 있고 joint 수용되지만 2.4542→237.2541cm로 크게 악화했다. SOURCE에는 활성 선 0개이며 4.5924/4.6976cm, 회전 6.0398/6.2471°로 5cm·5° 동시 기준을 통과하지 못한다. feasible은 관측상 유일하거나 정확하다는 뜻이 아니다.

V30 전체 평가도 성공 기준을 충족하지 않아 기본값 e는 유지한다. 이 두 사진은 전체 성능을 대표하지 않는다. 실제 포즈·지지점·참조는 새로 적합하거나 측정하지 않았고, 수치는 봉인된 동일 기준 점수만 옮겼다. 새 V31 조사는 별도 모집단이며 이 보고서의 방법·모델·결과를 바꾸지 않는다.
