가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**이번 고정 실험에서는 아니요.** 최신 요청에 따라 쉬움 153장·중간 92장만 사용했습니다. 어려움 74장은 새 평가에서 제외했습니다. IMAGE_ROLE은 위치 15.20239cm·회전 14.56122°, 고정 N3→cornerSubPix는 9.75455cm·10.91484°였습니다. 기존 GEOMETRIC_PROXY 참조의 결과이며 독립 실측 자세 인증은 아닙니다.

읽기 시작할 파일은 [상세 결과 보고서](RESULT_KO.md)입니다. 24방법의 전체 운용 결과, 쉬움/중간 분리, 평균·분산·SD·중앙값·P90, 새 자세/기본 반환/실패, 마스크 오판과 자세 성능, 직접 가시점 손상과 숨은 점 재투영, source 학습과 전이 실패를 설명합니다. 보고서에는 **실제 영상 12패널과 수치 그림 9개**가 있습니다. 사전에 고른 사례이므로 새 모델의 성공 사례만 골라 보여 주지 않았습니다.

검토자는 [검산 방법](REPRODUCE.md)의 표준 Python 명령으로 공개 원행을 다시 계산할 수 있습니다. 핵심 자료는 다음과 같습니다.

| 검토할 내용 | 파일 |
| --- | --- |
| 고정된 포함/제외 영상, 분할, 시간 패널, 사례 선정 | [COHORT.json](COHORT.json), [SUBSET_PROTOCOL.json](SUBSET_PROTOCOL.json), [RUNTIME_PANEL.json](RUNTIME_PANEL.json), [VISUAL_CASE_PROTOCOL.json](VISUAL_CASE_PROTOCOL.json) |
| 실제 추가 3×3000 학습과 같은 초기화/배치 순서 | [FORMAL_UPDATE_ROWS.jsonl.gz](FORMAL_UPDATE_ROWS.jsonl.gz), [TRAINING_COMPLETION.json](TRAINING_COMPLETION.json), [CHECKPOINT_METADATA.json](CHECKPOINT_METADATA.json) |
| 정답을 읽기 전 관측 735개와 geometry 1470개 | [LEARNED_OBSERVATIONS.jsonl.gz](LEARNED_OBSERVATIONS.jsonl.gz), [LEARNED_GEOMETRY_SEALED.jsonl.gz](LEARNED_GEOMETRY_SEALED.jsonl.gz) |
| 실제 채점 원행과 대응점 검산 | [LEARNED_PREDICTIONS.jsonl.gz](LEARNED_PREDICTIONS.jsonl.gz), [POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz](POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz) |
| 같은 Base/N3 좌표의 필수 무학습 12대조 | [HISTORICAL_FILTERED_ROWS.jsonl.gz](HISTORICAL_FILTERED_ROWS.jsonl.gz), [METRICS.json](METRICS.json) |
| 실제 1·2점/동시오류 6860경로와 oracle 한계 | [REAL_STRESS_ROWS.jsonl.gz](REAL_STRESS_ROWS.jsonl.gz), [REAL_STRESS_STATISTICS.json](REAL_STRESS_STATISTICS.json) |
| 새 전체 경로 600회, 캐시 재생 없는 시간 측정 | [RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz), [RUNTIME.json](RUNTIME.json) |
| 실패와 채점만 재개한 기록 | [EVALUATION_INTERRUPTION.json](EVALUATION_INTERRUPTION.json), [SCORING_RESUME_PROTOCOL.json](SCORING_RESUME_PROTOCOL.json) |
| 실제 실행량·독립 검산·변조 거부 검사 | [BUILD_LEDGER.json](BUILD_LEDGER.json), [REVIEW_CHECKS.json](REVIEW_CHECKS.json), [REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json) |
| 모든 새 공개 파일 SHA 및 Git 게시 증거 | [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json), [PUBLICATION.json](PUBLICATION.json) |

이 폴더는 이전 319장 결과와 감독 수정의 실패 기록을 덮어쓰지 않은 후속 실험입니다. 원본 보호 331파일의 목록은 [PRIOR_PUBLICATION_BINDINGS.json](PRIOR_PUBLICATION_BINDINGS.json)에 있습니다. 배포 방법과 사람이 정답으로 점을 고르는 oracle 진단을 구분했습니다. 새 자세가 나오면 예측 자기 가림으로 제외한 코너를 최종 재투영 좌표로 교체했고, 이 재투영점으로 다시 PnP를 수행하지 않았습니다.
