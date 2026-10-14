# V4.1 봉인 후 동일 프레임 효과 감사

선택 설정은 바꾸지 않았습니다. 모든 비교는 같은 TEST49와 물리 C2 기준을 사용하며, fallback·native 불일치 프레임을 포함합니다. CAL46은 fullCAL 모델 점수이며 LOGO 선택 점수와 구분합니다.

| 방법 | TEST 성공/49 | 평균 T(cm) | 평균 R(deg) |
|---|---:|---:|---:|
| FINAL | 42 | 2.644967 | 6.519532 |
| ORIGINAL_N3_PNP | 40 | 3.363034 | 6.527092 |
| FROZEN_STATEWISE_V3 | 39 | 3.513931 | 6.567535 |
| OLD_RGB_SINGLE_FAMILY | 42 | 2.618628 | 6.592330 |
| BOUNDARY_PROGRESSIVE | 43 | 3.004005 | 6.524340 |
| BASE_V4_ZERO_MEAN | 41 | 2.779464 | 6.484917 |
| NO_OT | 43 | 2.565237 | 6.508123 |
| NO_PRIOR | 37 | 3.488761 | 6.878460 |
| COARSE_MEAN_ONLY | 42 | 2.967720 | 7.198638 |

고정 수치 목표 충족: **False**. 이 결과만으로 통계적 우월성이나 새 환경 일반화를 주장하지 않습니다.

B/I 관측 교정의 순수 효과는 현재 대조군으로 분리할 수 없습니다. NO_PRIOR 대비 초기값 차이는 상태·soft fitting·검색·O/T가 섞인 효과입니다. zero-mean prior, CAL median bias, O/T, mean-only 차이는 JSON의 조건부 paired 비교에 따로 기록했습니다.

O/T 효과는 동일 B/I·유효 공분산·prior·bias를 유지한 NO_OT 비교로 판단했습니다. source EXTERNAL O와 초기 기하에 따른 conditional T를 구분했고, BI2의 수락은 모호한 pose family입니다.

초기 S의 이미지 질의는 0입니다. annotation SELF가 B/I로 들어간 사례는 초기 coarse self-visibility와 과거 annotation 간 불일치로 별도 기록했습니다.

4개 세션을 복원추출한 256개 모든 ordered draw에서 구간을 계산했습니다. 재사용 TEST의 탐색적 구간이며 독립 표본의 유의성 검정이 아닙니다.

전체 프레임·고정 사진 사례·초기 성공→최종 실패·활성 O/T는 FRAME_DETAILS.json에 있습니다. 원본 봉인 소스·예측·선택은 수정하지 않았습니다.
