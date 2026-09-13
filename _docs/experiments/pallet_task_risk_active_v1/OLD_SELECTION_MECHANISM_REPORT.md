# 기존 선택의 hard-frame enrichment

기존 verdict는 `RETROSPECTIVE_AL_NO_SIGNAL` 그대로다. 아래는 pool174에 대한
retrospective R0 오차 진단이며, 기존 학생 성능을 재판정하지 않는다.

diversity30과 geometry-weighted30의 공통은13장, 각 고유 집합은17장이다.
hard20은 pool의80th percentile 이상이며 각 target에서35장이다.

| 고유 집합 | kp mean 오차 중앙값 px | 위치 오차 중앙값 cm | yaw 중앙값 ° | kp/위치/yaw hard 수 | 축 오류 |
|---|---:|---:|---:|---|---:|
| diversity-only17 | 8.286 | 8.167 | 0.700 | 3 / 4 / 4 | 1 |
| geometry-only17 | 18.034 | 12.625 | 5.082 | 7 / 7 / 9 | 5 |

geometry-only 집합이 더 어려운 프레임을 포함했다는 관찰과 일관된다.
하지만 이것만으로 이전 P90 개선의 인과적 매개가 입증되지는 않는다.

9개 pool 세션을 paired cluster bootstrap10,000회(seed20260913)한
geometry-only minus diversity-only 차이의95% 구간:

- 평균 위치 오차: [2.400, 295.499]cm. 극단치 영향이 매우 크다.
- 중앙 위치 오차: [-2.019, 15.902]cm. 0을 포함한다.
- 평균 yaw 오차: [1.309, 24.554]°.
- 평균 frame-kp 오차: [-2.627, 49.013]px. 0을 포함한다.
- 축 오류 비율: [0.061, 0.444].

전체 분포, 세션·재질·주야 분포와 모든 metric의 mean/median/P90 및
bootstrap은 `OLD_SELECTION_HARDNESS_AUDIT.json`과
`OLD_SELECTION_UNIQUE_SET_ANALYSIS.json`을 참조한다.

이번 task-risk 선택30은 diversity와18장, old geometry와18장 겹친다.
단순 R_task 상위30과 k-center 선택30은 다른 집합이다. 전자는 위험도
precision@30 진단이고 후자만 학생 학습에 사용된다.
