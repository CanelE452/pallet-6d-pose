# Translation / full rotation 평가 계약

[확인] 새 지시문 §3을 새 학습 전에 잠갔다. 기존 원고·AUC 판정·참조·후보 생성은 수정하지 않았다. 새 T/R 주집합은 반복 DEV인 Plastic 자연 Moderate21+Severe78의 실제99 frame 합집합이다. 아래 과거 C2/C3 표는 새 기준의 사후 기술 통계이며, 새 성공 후보로 선정하지 않았다.

## 지표와 참조

T는 원점 중심 cuboid의 centroid 두 위치를 같은 OpenCV camera 좌표계에서 비교한 L2 norm×100(cm)이다. full R은 object-to-camera physical registry rotation의 geodesic angle(deg)을 기존 C2={I,Ry180}에서 최소화한다. camera-facing W/D→registry 변환은 예측/참조에 동일하게 고정되며 전치나 inverse로 바꾸지 않는다. 직사각형 90° 혼동은 대칭으로 제거하지 않는다. yaw는 같은 C2에서 상대회전 atan2의 wrap 최솟값이며 full R을 대체하지 않는다.

참조는 같은 annotation/치수로 복원된 geometry-derived pose다. 180° signed-axis 등가류이지 독립 physical6D 또는 삽입면 방향 검증이 아니다. camera-x/z는 예측−참조 signed bias와 절대오차로 보고하며, 차량 extrinsic이 확인되지 않아 forklift lateral/forward라고 부르지 않는다.

기존 conditional median/P90은 valid pose만 집계하고 coverage를 병기한다. 실패 포함 표는 실패를 +∞로 순서화한 extended-real linear quantile이다. JSON null은 POSITIVE_INFINITY/NA_EMPTY 상태와 구분한다. 임의 cm/deg failure penalty와 0오차 대체는 없다. common-valid 교집합 표·실패 전이·T/R 9칸 개선/동률/악화 교차표를 함께 제공한다. 기존 W/D extent axis-mismatch를 세고 신규 각도 합격선을 발명하지 않았다.

## 선택 규칙

primary99에서 OLD_REF 대비 T/R median이 둘 다 감소하고 coverage가 감소하지 않은 후보만 joint로 표시한다. R0 및 matched RAW 비교는 별도다. T/R Pareto 비지배 후보 중 추가 추론시간→신규 학습 GPU시간→사전 card ID 순으로 반복 후보를 택한다. cm+degree 합산이나 AUC/PCK tie-break는 없다. 하나만 좋아지거나 tradeoff인 경우 축을 명시하며, 작은 부호를 실질적 향상으로 자동 인정하지 않는다.

## Plastic primary99 baseline / 과거 참고선

| arm | valid/N | T median / P90 (cm) | R median / P90 (deg) | yaw median / P90 (deg) | axis mismatch |
|---|---:|---:|---:|---:|---:|
| R0 | 99/99 | 12.147922 / 120.471370 | 6.037944 / 89.218262 | 5.563436 / 88.888879 | 45 |
| OLD_RAW | 99/99 | 12.429008 / 119.308678 | 9.074374 / 89.309873 | 8.998087 / 89.150269 | 47 |
| OLD_REF | 99/99 | 12.423010 / 127.143068 | 8.694696 / 89.197272 | 7.694101 / 88.913988 | 46 |
| SYN | 99/99 | 12.432426 / 121.105137 | 6.203132 / 88.986411 | 5.533137 / 88.842928 | 45 |
| C2_RAW | 99/99 | 12.443805 / 118.905324 | 8.199379 / 89.122805 | 7.109047 / 88.986279 | 46 |
| C2_REF | 99/99 | 12.363877 / 127.590771 | 5.810111 / 89.238941 | 5.156649 / 88.959955 | 44 |
| C3_RAW9 | 99/99 | 12.190881 / 118.116249 | 5.856465 / 89.049729 | 5.249190 / 88.905600 | 45 |
| C3_MANUAL9 | 99/99 | 12.461408 / 120.472514 | 5.751853 / 89.481725 | 5.128505 / 89.357203 | 43 |

## 서로 다른 T/R oracle

| material/group | old REF T / R | T-best pose T / R | R-best pose T / R | same-candidate joint gain frames | distinct T/R choices |
|---|---:|---:|---:|---:|---:|
| PLASTIC/PRIMARY_MODERATE_PLUS_SEVERE | 12.423010 / 8.694696 | 9.221832 / 3.364443 | 9.412364 / 3.151069 | 29 | 20 |
| PLASTIC/ALL | 9.185879 / 3.766166 | 6.194534 / 2.733403 | 6.529108 / 2.628810 | 29 | 20 |
| WOOD/ALL | 2.071426 / 1.604867 | 2.071426 / 1.604867 | 2.071426 / 1.604867 | 2 | 1 |

T-best의 R과 R-best의 T도 각각 동일하게 선택된 한 pose에서 산출했다. 서로 다른 최솟값을 하나의 가능한 pose로 합치지 않는다. whole-output R0/teacher/RAW/REF pool의 별도 oracle도 JSON에 보존했다. GT로 고른 선택은 private diagnostic이며 학생 타깃·일반 추론에 제공하지 않는다.

모든 severity/recording/전체128·45·paired/LORO 수치와 camera-x/z는 [BASELINE_POSE_RESULTS.json](BASELINE_POSE_RESULTS.json), oracle은 [TR_ORACLE_RESULTS.json](TR_ORACLE_RESULTS.json)에 있다. Wood Severe는 N=0, NA_EMPTY_POPULATION이며 성능0으로 해석하지 않는다.

재현: `python -m scripts.research.pallet_pose_objective_followup_v2.metric_baseline all`. 잠긴 산출물이 있으면 hash를 검증하고 재학습 없이 재사용한다.
