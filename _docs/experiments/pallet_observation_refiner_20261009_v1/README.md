# Observation refiner / robust PnP — 검토 자료

전체319장 실험에서 기존 단순 대조보다 위치·회전을 함께 개선했다는 근거를 얻지 못했다. 세 소형 학습 모델의 새 point 자세는0/3/0장이며 fallback을 새 복원으로 세지 않았다.

먼저 [그림 포함 결과 보고서](RESULT_KO.md)를 읽고, 방법·단위·원행 key·프레임 추적은 [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md)를 참고한다. 원본 데이터 없이 저장된 결과를 검산하는 명령은 아래와 같다. Python 3.9 이상 표준 라이브러리만 사용하며 추론·학습·GPU·private 환경변수는 필요하지 않다.

```bash
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py \
  --frame-id eval_noapril:1775201415399297536
```

`--root`는 repo 또는 이 결과 디렉터리, `--output`은 별도 JSON, `--require-manifest`는 최종 manifest 필수 검사를 지정한다. 결과는 기본 `REVIEW_CHECKS.json`에 저장된다. 원본 RGB·Base/N3 가중치를 이용한 전체 재실행은 [REPRODUCE.md](REPRODUCE.md)의 별도 범위다.

| 파일 | 다른 연구자가 확인할 내용 |
|---|---|
| [RESULT_KO.md](RESULT_KO.md) | 23방법의 전체/공통/신규 결과, 산출률, 실패·그림·제한 |
| [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md) | solver 계약·가설 공유·mask/oracle·학습 target/loss·key와 drilldown |
| [METRICS.json](METRICS.json), [METRICS.csv](METRICS.csv) | 원행 평균·표본분산·SD·중앙값·P90·최대값·denominator |
| [PAIRED_COMPARISONS.json](PAIRED_COMPARISONS.json) | 같은 ID의 new−comparator 오차와 공통집합·세션 CI |
| [FIXED_CONTROLS.jsonl.gz](FIXED_CONTROLS.jsonl.gz) | BASE/SUBPIX/N3/N3_SUBPIX의 변경하지 않은 과거4×319행 |
| [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz) | Base/N3SubPix의 무학습12조건과 shared 경계13×319행 |
| [LEARNED_PREDICTIONS.jsonl.gz](LEARNED_PREDICTIONS.jsonl.gz) | 학습3조건과 ROLE mask/solver/point+line 절제6×319행 |
| [INPUTS.json](INPUTS.json), [OBSERVATIONS_LOCK.json](OBSERVATIONS_LOCK.json), [OBSERVATION_SEAL.json](OBSERVATION_SEAL.json) | native 입력·고정 metadata·GT 채점 전 선택 봉인 |
| [LEARNED_OBSERVATIONS.jsonl.gz](LEARNED_OBSERVATIONS.jsonl.gz), [OBSERVATION_STORAGE_AUDIT.json](OBSERVATION_STORAGE_AUDIT.json) | 957행의 query/bin/none/line/corner와 float32 logits 무손실 저장 |
| [REAL_CORRESPONDENCE_AUDIT.json](REAL_CORRESPONDENCE_AUDIT.json), [REAL_CORRESPONDENCE_ROWS.jsonl.gz](REAL_CORRESPONDENCE_ROWS.jsonl.gz) | mask 오판과 참조상 accurate pool/inlier/배치·실제 T/R 사건의 분리 |
| [GEOMETRY_STRESS.jsonl.gz](GEOMETRY_STRESS.jsonl.gz), [GEOMETRY_STRESS_SUMMARY.json](GEOMETRY_STRESS_SUMMARY.json) | 렌더0인2,816개 analytic 경로; 실사 증거와 구분 |
| [REAL_MASK_STRESS.jsonl.gz](REAL_MASK_STRESS.jsonl.gz), [REAL_MASK_STRESS_SUMMARY.json](REAL_MASK_STRESS_SUMMARY.json) | 사람 mask의 한 점 오제외/오잔류638행 |
| [VISIBILITY_DAMAGE.json](VISIBILITY_DAMAGE.json) | 직접 가시 손상과 숨은 투영 오차를 별도 집계 |
| [SOURCE_FAMILY_SPLIT.json](SOURCE_FAMILY_SPLIT.json), [SUPERVISION_PREPARATION.json](SUPERVISION_PREPARATION.json) | 실제 source1,024개·분할768/128/128·true mask/depth target |
| [LEARNING_PROTOCOL.json](LEARNING_PROTOCOL.json), [TRAIN_LOGS.jsonl](TRAIN_LOGS.jsonl), [TRAINING_COMPLETION.json](TRAINING_COMPLETION.json) | 동일5,890 parameter·초기값·배치·3×3,000 update·실제 gradient/log |
| [REVIEW_LOSS_MASK_CHECK.json](REVIEW_LOSS_MASK_CHECK.json) | 실제 첫 배치 target의 per-logit ignore0 확인; 추가 loss-only 검산 |
| [RUNTIME.json](RUNTIME.json), [RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz) | detector·초기/후단 자세 포함 실제 전체 경로600실행 |
| [EXECUTION_LEDGER.json](EXECUTION_LEDGER.json), [SOURCE_EXECUTION_COUNTS.json](SOURCE_EXECUTION_COUNTS.json) | 실제 실행량·재시도·실패 비용과 NA primitive 카운터 |
| [VERIFICATION.json](VERIFICATION.json), [CHECKS.json](CHECKS.json), [RUNTIME_VERIFICATION.json](RUNTIME_VERIFICATION.json) | 당시 private 원본 포함 독립 검산·무결성·runtime 재집계 |
| [VISUAL_REVIEW_CASES.json](VISUAL_REVIEW_CASES.json), [figures/](figures/) | 저장 결과에서 파생한 그림의 입력·ID·사후 선택 기준·해시 |
| [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json), `REVIEW_CHECKS.json` | 현재 확장 bundle binding과 public-only 검산 결과 |
| [PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json), [PUBLICATION.json](PUBLICATION.json) | 앞서 완료한 원 실험 게시 스냅샷; 현재 확장 manifest와 구분 |

이번 파일·key·CLI는 이 실험의 스키마다. 과거 다른 실험 파일에 같은 key가 있다고 가정하지 않는다. `used`는 scoring pool, `inliers`는 최종 합의, `fit_input_ids`는 실제 fit 입력이며 서로 다를 수 있다. `actual_pose`는 fallback이면 초기 자세다. ADDsym 원행은 meter, 표는 cm다. 사람 mask와 고정 평가 phase를 쓰는 oracle은 배포 결과가 아니다.

실사 reference는 기존 기하 재구성값이고 독립 물리 측정은 아니다. 고정 대조586행의 과거 R/t 누락과 E6 네 통제 변형 미실행은 그대로 남아 있다. 그림은 사후 설명 자료이며 전체 성능 증거는 전체319 원행과 비교 분모다. 원본 RGB·대형 cache·checkpoint는 공개 Git에 넣지 않았다.

이번 보고서 보강의 실제 실행량과 원 실험 보존 여부는 [REVIEW_BUILD.json](REVIEW_BUILD.json)에, 공개 파일만 복사한 격리 검사와 의도적 오류 검출 결과는 [REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json)에 기록했다. [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)은 현재 코드·원행·문서·그림의 SHA256 목록이며 자체 파일과 재생성 가능한 REVIEW_CHECKS만 순환을 피하려고 제외한다.
