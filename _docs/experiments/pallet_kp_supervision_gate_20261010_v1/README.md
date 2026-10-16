# 고정 IMAGE_ROLE 선 관측 절제와 실제 메시 감독 검산

먼저 [RESULT_KO.md](RESULT_KO.md)를 읽는다. **전체319장의 위치·회전 개선은 달성하지 못했다.** 보고서는 실패·fallback·직접 가시/숨은 점 손상과 분모를 포함하며, 이번 과학 그림과 기존 실제 이미지의 역할을 구별한다.

| 검토할 내용 | 파일 |
|---|---|
| 결과·그림·해석·미완료 | [RESULT_KO.md](RESULT_KO.md) |
| 실사 실행 전 고정 조건 | [PROTOCOL.json](PROTOCOL.json) |
| GT 읽기 전 봉인된319 pose/관측/H/국소기하 | [GEOMETRY_SEALED.jsonl.gz](GEOMETRY_SEALED.jsonl.gz) |
| 실제 채점 원행319장 | [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz) |
| 4경로1276행의 후행 reference·코너·마스크 비교 | [POSTHOC_ROWS.jsonl.gz](POSTHOC_ROWS.jsonl.gz) |
| 모든5통계·최대·공통집합·13세션CI | [METRICS.json](METRICS.json), [METRICS.csv](METRICS.csv) |
| 실제 optimizer 실행량·GT 봉인·코드SHA | [LINE_EXECUTION.json](LINE_EXECUTION.json) |
| point/line 중복·H 출력·rank 무결성 검사 | [ALL_LINES_CHECKS.json](ALL_LINES_CHECKS.json) |
| source128의49→90 국소 관측 가능성 | [SOURCE_OBSERVATION_GATE_ROWS.jsonl.gz](SOURCE_OBSERVATION_GATE_ROWS.jsonl.gz), [SOURCE_OBSERVATION_GATE.json](SOURCE_OBSERVATION_GATE.json) |
| 실제 wire의6463 query 수정 후보·독립검토 | [CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz](CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz), [READ_ONLY_WIRE_REVIEW.json](READ_ONLY_WIRE_REVIEW.json) |
| 실제 경계 소유권을 독립 재계산할 작은 face 증거 | [WIRE_FACE_WITNESSES.json](WIRE_FACE_WITNESSES.json) |
| 1024가족의 분할별 보수적 감독 배열·원행 | [FULL_SOURCE_PREPARATION.json](FULL_SOURCE_PREPARATION.json), [FULL_SOURCE_TARGET_ROWS.jsonl.gz](FULL_SOURCE_TARGET_ROWS.jsonl.gz), [PREPARED_TARGETS.npz](PREPARED_TARGETS.npz) |
| 원래 실행경로와 공개 hash 대응 | [PUBLICATION_RELOCATION.json](PUBLICATION_RELOCATION.json), [EXTERNAL_DEPENDENCIES.json](EXTERNAL_DEPENDENCIES.json) |
| 기존189파일 보존 | [PRIOR_PUBLICATION_BINDINGS.json](PRIOR_PUBLICATION_BINDINGS.json) |

여기에 저장된 `new_pose_estimated`, `fallback_used`, `no_pose`, model-line rank, proposal 필드는 이 실험이 정의한 결과 스키마다. 과거 캐시에 같은 key가 있었다고 가정하지 않는다. private 데이터가 필요한 전체 추론 재실행과 공개 원행만으로 가능한 검산은 [REPRODUCE.md](REPRODUCE.md)에 구분한다. 재학습·최종 목표의 달성은 미완료이며 이를 검산PASS와 혼동하지 않는다.
