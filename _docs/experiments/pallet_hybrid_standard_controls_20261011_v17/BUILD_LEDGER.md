# V17 실제 실행량과 완료 범위

완료 receipt가 있는 실행만 기록한다. Report는 저장 JSON 전사이며 추가 score·통계·모델·원행 산술을 실행하지 않았다. 새 head/RGB/seed/학습/선·LOCAL/좌표선택은0회. 부모 V16의 원환경 실패와 READY564FAIL을 보존하며 이를 geometryPASS로 표시하지 않는다.

| 실제 블록 | 완료량 | authority |
| --- | --- | --- |
| 고정 protocol·preflight·기하 | 245frame·490ordinary solves·490geometry·245ledger, exception0/cleanup완료 | [PROTOCOL](PROTOCOL.json), [PREFLIGHT](PREFLIGHT.json), [CONTROL_RECEIPT](CONTROL_RECEIPT.json), [seal](GEOMETRY_SEAL.json) |
| 독립 기하 scalar | ownfreeze1·실제run1,394552checks/0fail | [VALIDATION_PROTOCOL](VALIDATION_PROTOCOL.json), [STARTED](VALIDATION_STARTED.json), [CHECKS](VALIDATION_CHECKS.json) |
| 실제 score | 490새 score, 두 방법 각245 | [EVALUATION_PROTOCOL](EVALUATION_PROTOCOL.json), [SCORING_AUTHORITY](SCORING_AUTHORITY.json), [SCORING_RECEIPT](SCORING_RECEIPT.json) |
| 통계 | fixed490+new490=980;108분포/648값;27그룹/81CI;새draw0 | [STATISTICS_STARTED](STATISTICS_STARTED.json), [METRICS](METRICS.json) |
| 독립 통계 | ownfreeze1·실제run1 PASS980/648/108/27/81,63유효CI·18None | [STATISTICS_PROTOCOL](STATISTICS_VERIFICATION_PROTOCOL.json), [STARTED](STATISTICS_VERIFICATION_STARTED.json), [VERIFICATION](VERIFICATION.json) |
| Hybrid fresh 전체 경로 |150호출·공식130회·26RGB·quiet/cleanup완료,8구간56값 | [RUNTIME_PROTOCOL](RUNTIME_PROTOCOL.json), [RECEIPT](RUNTIME_RECEIPT.json) |
| 저장 시간 독립 검산 |실제49993검사/0FAIL·150행/130공식/56slot | [RUNTIME_CHECKS](RUNTIME_CHECKS.json) |
| 그림·실제 열람 | renderer freeze1/run1,8PNG·고정6case24panel·980saved display·root view_image8 | [FIGURE_PROTOCOL](FIGURE_PROTOCOL.json), [FIGURE_BINDINGS](FIGURE_BINDINGS.json), [ROOT_VISUAL_REVIEW](ROOT_VISUAL_REVIEW.json) |
| 공개 재현·게시 | 이번 V17 bundle/push 아직 미완료 | 실제 hybrid 비용을 native 시간으로 대신하지 않음 |

원행과 frozen 코드는 덮어쓰지 않았다. main 변경·자동 병합·force push를 하지 않으며 V17 게시·원격SHA 일치는 별도 실제 receipt 후 기록한다. 알려진 DEV의 GEOMETRIC_PROXY이고 물리 실측·unseen 결과가 아니다. 종합 N3 대비 양평균 개선=true, 같은backend native 대비 hybrid 양축 우위=false는 별도 판정이며 전체 목표 완료로 표시하지 않는다.
