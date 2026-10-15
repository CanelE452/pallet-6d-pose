# V4.2 동일 프레임 봉인 후 효과 감사

모든 비교는 같은 TEST49·물리 C2 기준입니다. 전체95의 fallback과 native 불일치를 유지하고 crop3은 별도 기록했습니다.

| 방법 | 성공/49 | 평균 T(cm) | 평균 R(deg) |
|---|---:|---:|---:|
| FINAL | 43 | 2.565237 | 6.508123 |
| ORIGINAL_N3_PNP | 40 | 3.363034 | 6.527092 |
| FROZEN_STATEWISE_V3 | 39 | 3.513931 | 6.567535 |
| OLD_RGB_SINGLE_FAMILY | 42 | 2.618628 | 6.592330 |
| BOUNDARY_PROGRESSIVE | 43 | 3.004005 | 6.524340 |
| BASE_V4_ZERO_MEAN | 41 | 2.779464 | 6.484917 |
| FROZEN_BIAS_V4_1 | 42 | 2.644967 | 6.519532 |
| NO_OT | 43 | 2.565237 | 6.508123 |
| NO_PRIOR | 38 | 3.377651 | 6.895333 |
| COARSE_MEAN_ONLY | 42 | 2.967720 | 7.198638 |

고정 숫자 목표 충족: **True**. 조건부 T를 U로 보류한 효과와 남은 O/T의 효과를 구분했습니다. 원래 B/I·유효 공분산·nonS scale·O 숫자 영역·λ1β1은 그대로입니다.

재사용 TEST의 후속 탐색적 수정이며 새 일반화나 통계적 우월성을 증명하지 않습니다. CAL full모델 점수는 세션 제외 LOGO 선택 점수와 다릅니다. 원과 캔버스 교차는 CAL 최대 오차에 따른 보류이며, 미래 포함률이나 물리 T 인증을 뜻하지 않습니다.

모든 세션별 paired 차이와 256개 exact 세션 복원추출 구간, prior·bias·interval 기여는 EFFECT_AUDIT.json에 있습니다. 실제 사진 및 실패/변화 사례는 FRAME_DETAILS.json에 있습니다.
