# KP 보정의 어려움과 고정 인과 비교 — 검토 자료

전체319장의 새8조건에서 기존 N3→SubPix보다 위치·회전을 함께 개선하지 못했다. 성공한 척하지 않고 source 감독·대응 존재·코너 조립·전역 다중해의 병목을 실제로 검사했다.

먼저 [그림5개·실사12패널 포함 상세 보고서](RESULT_KO.md), [source 감독 세부 증거](SOURCE_FINDINGS_KO.md)를 읽는다. 기존 보고서와 원행111개는 변경하지 않았다.

공개 Git만 있으면 Python3.9+ 표준라이브러리로 다음 명령을 실행할 수 있다. 추론·학습·원본RGB·가중치·GPU·private환경변수는 필요하지 않다.

```bash
python3 scripts/research/pallet_kp_difficulty_20261010_v1/review_verify.py --require-manifest
```

`--root`는 repo 또는 이 문서 디렉터리, `--output`은 별도 검산 JSON이다. 오류는 nonzero exit로 표시한다. 출력 `REVIEW_CHECKS.json`은 재생성 가능한 검산 영수증이며 원행을 덮어쓰지 않는다.

| 파일 | 검토할 내용 |
|---|---|
| [PROTOCOL.json](PROTOCOL.json) | 사전고정8조건·성공기준·별도prior/decoder절제·DEV319한계 |
| [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz), [CAUSAL_GEOMETRY_SEALED.jsonl.gz](CAUSAL_GEOMETRY_SEALED.jsonl.gz) | 채점 전 봉인과 채점 후2552행·상태·R,t·native출력 |
| [METRICS.csv](METRICS.csv), [METRICS.json](METRICS.json) | 5통계·산출률·75paired비교·집합ID·숨은/가시손상 |
| [BOOTSTRAP_SESSION_DRAWS.json.gz](BOOTSTRAP_SESSION_DRAWS.json.gz) | 기존과 같은13세션10000행을 직접CI재계산 |
| [POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz](POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz) | 참조정확pool·배치·finalinlier·오판과자세결과 분리 |
| [MATCH_MASS_OBSERVATIONS.jsonl.gz](MATCH_MASS_OBSERVATIONS.jsonl.gz) | 원래957logits에서별도고정decoder·없음자동채움false |
| [SOURCE_CEILING_ROWS.jsonl.gz](SOURCE_CEILING_ROWS.jsonl.gz), [SOURCE_LOGIT_DIFFICULTY.json](SOURCE_LOGIT_DIFFICULTY.json) | 실제기존target1024와존재/위치확률분리 |
| [SOURCE_RAY_VALIDATION_ROWS.jsonl.gz](SOURCE_RAY_VALIDATION_ROWS.jsonl.gz), [SOURCE_RAY_PROTOCOL.json](SOURCE_RAY_PROTOCOL.json) | 실제메시고정128광선검사·NONE수치민감도·앞표면대조 |
| [VISUAL_REVIEW.json](VISUAL_REVIEW.json), [SOURCE_RAY_VISUAL_CASES.json](SOURCE_RAY_VISUAL_CASES.json) | 실제이미지hash·ID·query·crop·사후선택근거 |
| [RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz), [RUNTIME.json](RUNTIME.json) | fresh전체경로600회·일치·경쟁guard·5통계 |
| [BUILD_LEDGER.json](BUILD_LEDGER.json), [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json) | 실제실행량·보존·공개전체binding |
| [PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json), [PUBLICATION.json](PUBLICATION.json) | 보호상태·게시commit·원격SHA영수증 |

자세 원행 key는 `(method,id)`이며 gzip JSONL을 읽어 한 프레임을 추적할 수 있다. source ceiling/ray는 `id`, source logits는 `(arm,id)`, runtime은 `(arm,id,phase,repeat 또는 warmup_index)`이며 각 schema를 참고한다. `solver.used`는 scoring pool, `fit_input_ids`는 실제fit, `inliers`는 최종합의다. 서로 같다고 가정하지 않는다. 새R,t가 있으면 초기hidden을재투영교체하며 재투영점을 다시fit에 넣지 않는다. fallback은전체 초기출력이다. `GEOMETRIC_PROXY`는독립물리GT가아니다.

원본RGB와featurecache·체크포인트를 공개하지 않았다. 공개검산은저장결과의일관성·연산을 확인하며 실제촬영GT나원본학습실행을 독립인증하는것은아니다. 실제재실행요건은 [REPRODUCE.md](REPRODUCE.md)를 참고한다.
