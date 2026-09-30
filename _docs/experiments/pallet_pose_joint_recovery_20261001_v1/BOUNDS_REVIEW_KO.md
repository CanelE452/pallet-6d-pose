# 고정 후보 집합 하한의 독립 검토

2026-10-01. `candidate_bounds.py`와 생성된 `CANDIDATE_BOUNDS.json`을 읽고 수학·범위를 검토했다. 코드를 수정하거나 결과를 재생성하지 않았다. 추가로 기존 `POSE_PREDICTIONS_LOCK.json`의 **23개 binding 전부**를 재귀 검증해 파일·코드·registry/geometry·GEO 선택 lock/checkpoint SHA가 일치함을 확인했다. 경로·SHA는 [BOUNDS_REVIEW.json](BOUNDS_REVIEW.json)에 남겼다.

각 frame에서 기존 후보의 T 최솟값은 그 집합에서 어느 하나를 고르는 선택기의 T보다 작거나 같다. 같은 frame 분모를 유지한 empirical median/P90은 이 성분별 순서를 보존하며, seed별 quantile의 평균도 순서를 보존한다. `T_best`와 `R_best`는 각각 완전한 pose 한 개를 선택한다. T와 R 최솟값을 서로 다른 후보에서 가져와 가상 pose로 만들지 않았다. 이전 3모델×2모집단×2oracle×2metric×2statistic의 **48개 수치 일치 검사** 기록도 확인했다.

**조건부 quantile 사용의 전제는 이번 실제 데이터에서 충족된다.** 자연99·clean29는 모든9개 운영 모델에서 pose가 전부 유효하고, `T_best`/`R_best`도 같은 분모가 전부 유효하다. 원래 실패 수 증가 금지 기준과 R0 실패0이 있기 때문에 어려운 행을 실패시켜 조건부 분모에서 빼는 방법은 허용되지 않는다. 다른 데이터에서 실패가 존재한다면 조건부 중앙값만으로 동일한 증명을 일반화할 수 없으며, 동일 분모·실패 포함 extended error 또는 coverage 조건을 다시 확인해야 한다.

| 고정 후보 집합 | clean T median 하한 cm | 자연 T P90 하한 cm | 현재 상한 기준에 관한 결론 |
|---|---:|---:|---|
| R0 | 2.929862 | 120.471370 | 두 하한으로 배제되지 않음. 공동 달성의 증명은 아님 |
| PRIOR1 | 3.198106 | 123.679073 | clean 상한3.076355보다 높음 |
| FULL125 | 3.432615 | 120.824080 | clean 상한3.076355보다 높음 |
| SINGLE251 seed별 quantile 평균 | 3.061082 | 124.782532 | 이 두 평균 하한으로 배제되지 않음 |
| DIVERSE251 seed별 quantile 평균 | 3.076372 | 129.502732 | 자연 상한126.494939 초과. clean도 미세 초과 |

DIVERSE clean 초과량은 **0.000016957cm**로 매우 작다. 판정을 뒤집거나 실질 차이로 과장하지 않으며, 자연 T P90 하한이 이미 상한을 명확히 넘는 사실을 중심으로 해석한다. DIVERSE seed3나 SINGLE3개 평균까지 배제됐다고 말하지 않는다. JSON의 각 단일 모델 행은 해당 단일 후보 집합의 scalar ceiling 검사다. 원래 clean/tail 기준은3개 seed quantile의 평균이므로 **단일 seed의 초과만으로 원래 평균 기준 실패를 증명할 수 없다**. 그 판단은 `*_mean3seeds` 행으로 한다.

증명의 범위는 동일 박스·좌표·카메라/치수·기존 W/D 후보·metric/failure 계약이다. 새 좌표, 다른 박스, 다른 연속 solver, 모델 간 후보 합집합은 이 하한 밖에 있다. 범위 밖의 방법이 성공한다는 뜻은 아니다. GT-dependent oracle은 설명용이며 배포 가능한 정확도나 새 감독 타깃으로 승격하지 않는다. 기존6-fit 주 결과와 성공 기준은 그대로다.
