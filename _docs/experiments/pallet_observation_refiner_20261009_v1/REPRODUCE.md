# 실제 실행과 재집계

Python3.10, OpenCV4.9, 고정 기존 detector/N3를 사용했다. private source/cache/checkpoint 경로를 공개 결과에 넣지 않고 환경변수로 연결한다. 학습 checkpoint 세 개는 source root 아래 data/pallet/results/pallet_observation_refiner_20261009_v1/weights에 해시 그대로 보존했다. tmpfs가 사라진 뒤에는 이를 private scratch의 learned_fits로 복사하여 재집계/추론에 연결할 수 있다. PRIVATE_CHECKPOINT_RETENTION.json이 상대경로·해시를 기록한다. 실행 시 PALLET_SOURCE_ROOT는 원본 읽기 전용 checkout, PALLET_OBSERVATION_SCRATCH는 private cache다.

```bash
export PALLET_SOURCE_ROOT=/path/to/read-only-source
export PALLET_BASELINE_ROOT=/path/to/immutable-a22-research-code
export PALLET_INSTRUCTION_PATH=/path/to/pallet_cli_observation_refiner_robust_pnp_20261009.md
export PALLET_OBSERVATION_SCRATCH=/path/to/private-experiment-cache
export PALLET_TEX_ARCHIVE=/path/to/read-only-legacy-TEX.zip
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_docs=_docs/experiments/pallet_observation_refiner_20261009_v1
python -m scripts.research.pallet_observation_refiner_20261009_v1.audit
python -m scripts.research.pallet_observation_refiner_20261009_v1.test_solver --output "$task_docs/SOLVER_CHECKS.json"
python -m scripts.research.pallet_observation_refiner_20261009_v1.contract_audit audit
python -m scripts.research.pallet_observation_refiner_20261009_v1.pilot
python -m scripts.research.pallet_observation_refiner_20261009_v1.evaluate
python -m scripts.research.pallet_observation_refiner_20261009_v1.contract_audit repair_replay
python -m scripts.research.pallet_observation_refiner_20261009_v1.geometry_stress --scenes 128 --output "$task_docs"
python -m scripts.research.pallet_observation_refiner_20261009_v1.source_audit --tex-archive "$PALLET_TEX_ARCHIVE"
python -m scripts.research.pallet_observation_refiner_20261009_v1.source_audit --tex-archive "$PALLET_TEX_ARCHIVE" --archived-only
python -m scripts.research.pallet_observation_refiner_20261009_v1.training prepare
python -m scripts.research.pallet_observation_refiner_20261009_v1.training train
python -m scripts.research.pallet_observation_refiner_20261009_v1.learned_infer --output LEARNED_OBSERVATIONS.jsonl.gz
python -m scripts.research.pallet_observation_refiner_20261009_v1.learned_evaluate
python -m scripts.research.pallet_observation_refiner_20261009_v1.statistics
python -m scripts.research.pallet_observation_refiner_20261009_v1.benchmark measure --image-role-adapter scripts.research.pallet_observation_refiner_20261009_v1.benchmark:benchmark_adapter --learned-reference "$task_docs/LEARNED_PREDICTIONS.jsonl.gz"
python -m scripts.research.pallet_observation_refiner_20261009_v1.verify
python -m scripts.research.pallet_observation_refiner_20261009_v1.report
```

완료 raw 출력을 덮어쓰는 추론 명령은 거부한다. 새 worktree/출력에서 재실행하거나 완료 동일 해시 단계를 재사용한다. source_audit는 --tex-archive 필수 인수를 받고 --archived-only로 기존 TEX 32장 보조 감사를 실행한다. contract_audit의 stage는 audit와 repair_replay다. 기록된 구현 수리와 재실행은 REPAIR_LOG/LEARNED_DTYPE_FIX/LEARNED_OUTPUT_COLLISION에 있다. contract_audit repair_replay는 canary 수리 후 수행한 전체319개 수치 패리티 재검사다. compact_observations는 학습 관측 봉인 후 적용한 무손실 저장 단계이며 기존 원본을 private cache에 보존했다. driver CLI의 audit→solver_tests→contract_audit→cpu_pilot→pose_diagnostics→mask_stress→source_audit→prepare_supervision→train→infer_observations→solve_evaluate→statistics→benchmark→verify→report 단계와 동일 해시 완료 receipt를 함께 사용한다. 게시 작업은 driver와 별개로 전용 브랜치에 정상 git commit/push하고 원격 SHA를 확인한다.

원행만 재집계할 때 statistics→verify→report를 실행한다. verify는 이전 검산과 달라진 파생 대응 감사를 해시 suffix 파일로 보존한다. 모델 가중치 선택·임계값 선택·추가 seed를 실행하지 않는다. 실제 전체 시간은 격리된 고정 패널 benchmark 실행만 사용한다.

## 원본 영상 없이 공개 파일만 검산

Python 3.9 이상의 표준 라이브러리만 필요하다. detector/학습 가중치, CUDA, PALLET_SOURCE_ROOT 또는 원본 영상은 필요하지 않다. 체크아웃 루트에서 실행한다. 이 명령은 저장된 숫자·ID·상태·투영·해시를 다시 검사하며 실제 추론이나 학습을 다시 실행하지 않는다. 원본 영상과 평가 정답의 물리적 정확성까지 증명하는 명령은 아니다.

```bash
python scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py --require-manifest --output /tmp/pallet-public-review-checks.json
python scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py --require-manifest --frame-id eval_noapril:1775201415399297536 --output /tmp/pallet-frame-review-checks.json
```

세부 파일·필드·검토 순서는 [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md)에 있다. 그림01–05는 저장된 숫자만으로 다시 만들 수 있다. 그림06–07은 기존 원본 영상과 자산이 있어야 재생성할 수 있으며 공개 PNG와 [VISUAL_REVIEW_CASES.json](VISUAL_REVIEW_CASES.json)에서 사례 ID·크롭·그림 원천을 확인할 수 있다. 시각화 실행은 모델 학습·후단 PnP 재실행과 구별된다.
