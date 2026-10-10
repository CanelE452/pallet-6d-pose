가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**현재 v2의 우위는 확인하지 못했다. 이번 수치 검산도 기존 결론을 바꾸지 않는다.** 기존 자세 원행과 같은 세션 bootstrap 자료로 보고된 신뢰구간 자체를 독립 계산했다. 새 추론·자세 fit·채점·설정 변경은 0이다.

기존 v2 검산은 평균·분산·분모·ID를 확인했지만 신뢰구간의 유한성·정렬·재표집 해시만 검사했다. 기존 `CI95_numeric_replay=false` receipt를 그대로 두고 이번 검산을 별도 파일로 추가했다. 이전 검사가 이미 수치 CI를 검증했다고 소급 주장하지 않는다.

쉬움 153+중간 92=245장의 saved predictions 1,225행과 fixed predictions 490행을 읽었다. 3 strata × 8 contrasts × 2 scopes × 3 metrics = 144개 distinct CI, combined 별칭 48개, 총 384개의 끝점 비교가 완료됐다. paired ID·분모·비어 있지 않은 재표집 수를 정확하게 비교하고, 756개 exact 검사와 528개 numeric 검사를 통과했다. 최대 수치 차이는 8.88e-16이다. subgroup/scope에 따라 비어 있지 않은 재표집 수는 8,817–10,000이며 빈 재표집을 숨기지 않았다.

단위는 T·ADDsym cm, R degree다. Δ는 후보−대조이며 양수는 평균 악화다. 아래는 전체 운용 245장의 비교이고 fallback을 제외하지 않았다.

| 후보 / 대조 | 지표 | 평균 Δ | 동일 세션 bootstrap 95% 구간 |
| --- | --- | ---: | --- |
| N3_VALIDATED_ROLE / 고정 N3_SUBPIX | T(cm) | +0.58804 | [-0.06584, +1.17817] |
| N3_VALIDATED_ROLE / 고정 N3_SUBPIX | R(°) | +0.04149 | [-0.01861, +0.10706] |
| N3_VALIDATED_ROLE / 고정 N3_SUBPIX | ADDsym(cm) | +0.54455 | [-0.03466, +1.06240] |
| N3_VALIDATED_ROLE / 같은 자세 정책의 N3_BASIN_ROBUST | T(cm) | +0.54779 | [+0.02956, +1.03165] |
| N3_VALIDATED_ROLE / 같은 자세 정책의 N3_BASIN_ROBUST | R(°) | +0.00645 | [-0.03614, +0.04683] |
| N3_VALIDATED_ROLE / 같은 자세 정책의 N3_BASIN_ROBUST | ADDsym(cm) | +0.53243 | [+0.02796, +1.01933] |

고정 N3 대조의 세 구간은 0을 포함한다. 우위가 입증되지 않았다는 판단이며 동등성의 증명이 아니다. 같은 자세 정책 대조에서는 T·ADDsym 구간이 양수지만, 이미 여러 번 개발에 사용한 proxy DEV와 다수 절제라는 한계를 유지한다. 독립 일반화나 물리 실측 정확도로 확대하지 않는다.

독립 구현은 session별 delta를 `math.fsum`으로 합산하고, 기존 multiplicity를 곱한 분자·paired frame 수 분모로 재표집 평균을 계산했다. 분위수는 Python 정렬과 선형 order-statistic 보간을 사용했다. 원래 통계 함수·모델·GT·PnP 모듈을 import하지 않았다. before/after 입력 SHA 및 materialized prior tracked 624파일의 bytes가 일치했다.

실제 실행은 freeze 1회와 산술 run 1회, 실패·재시도 0이다. [CHECKS](CHECKS.json)의 실행량과 [ROWS](ROWS.jsonl.gz)의 48 paired group을 확인할 수 있다. 이 검사는 선의 실제 소유권, source→real 전이, 코너 상대 정확도를 해결하는 새 방법이 아니다. [현재 메커니즘 진단](../pallet_corner_mechanism_audit_20261010_v1/RESULT_KO.md)에서 남은 정확도 문제와 API의 실제 사용 검증을 구분한다.
