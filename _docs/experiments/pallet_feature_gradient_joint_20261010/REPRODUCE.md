# 재현: 기존 환경과 공개 원행

설치·재학습·합성/실사 데이터 생성 없이 기존 환경을 사용한다. 정확도 실행은 기존 비공개 입력이 필요하고, 공개 원행의 통계·그림 검산은 원본 RGB/checkpoint/GPU 없이 가능하다.

```bash
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/pallet-pose"
export PALLET_BASELINE_ROOT="/path/to/immutable/a22-baseline-worktree"
export PALLET_JOINT_PRIVATE="/path/to/private/replay-records"
export PALLET_PRIVATE_OUTPUT="$PALLET_JOINT_PRIVATE"
export PALLET_JOINT_OUTPUT="/path/to/new-joint-output"
export PYTHONDONTWRITEBYTECODE=1
```

게시된 입력/출력을 덮어쓰지 않고 **새 출력 디렉터리**로 전체 Gate A→B→C를 재실행:

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.replay --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_JOINT_PRIVATE" --output "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.review_pose
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.visibility
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_posterior --docs "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_auxiliary --docs "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -c 'from scripts.research.pallet_feature_gradient_joint_20261010.finalize import mechanism_proof; mechanism_proof()'
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.summarize --doc "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.figures --doc "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify --doc "$PALLET_JOINT_OUTPUT" --require-figures
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.report --doc "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -c 'from scripts.research.pallet_feature_gradient_joint_20261010.finalize import artifact_manifest; artifact_manifest()'
```

replay 내부 순서는 `preflight`(원본 hash와 metadata 인증) → `capture`(고정 N3의 seed별 319회 forward 및 native parity) → `test_fusion`(합성 배열을 이용한 8개 단위검사와 실행 영수증) → `coordinates`(16장 smoke와 두 방법의 1,914개 좌표를 참조 읽기 전에 봉인) → `evaluate`(각 새 좌표에 실제 F를 1회 호출)다. `UNIT_TESTS.json`과 방법 lock을 새 출력에 저장하고 기존 코드의 수치 설정은 변경하지 않는다. 아래 명령은 단위검사만 실행하므로 전체 재현에 필요한 실행 영수증은 replay가 생성한다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.test_fusion
```

공개 원행만으로 검산·재집계·PNG 재생성:

```bash
export PALLET_PUBLIC_DOC="_docs/experiments/pallet_feature_gradient_joint_20261010"
mkdir -p "$PALLET_JOINT_PRIVATE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_posterior --docs "$PALLET_PUBLIC_DOC" --output "$PALLET_JOINT_PRIVATE/public_posterior_check.json"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_auxiliary --docs "$PALLET_PUBLIC_DOC" --output "$PALLET_JOINT_PRIVATE/public_auxiliary_check.json"
PALLET_JOINT_OUTPUT="$PALLET_PUBLIC_DOC" "$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.review_pose
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.summarize --doc "$PALLET_PUBLIC_DOC"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.figures --doc "$PALLET_PUBLIC_DOC"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify --doc "$PALLET_PUBLIC_DOC" --require-figures
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.report --doc "$PALLET_PUBLIC_DOC"
```

독립 검산기의 `--docs`는 공개 자료 디렉터리, `--output`은 새 검산 영수증 파일이다. 기존 파일을 덮어쓰지 않는 배타적 생성 방식이므로 다시 실행할 때는 새로운 출력 파일명을 지정한다. 통계와 그림 생성기의 옵션은 단수형 `--doc`이다. 저장된 자료만 검산할 때에는 비공개 RGB, checkpoint, CUDA가 필요하지 않다.

위의 전체 재현은 새로운 출력에 posterior와 보조 검산 영수증, `MECHANISM_PROOF.json`을 생성한다. `finalize.mechanism_proof()`는 저장된 좌표의 차이와 계산 계약만 검산하며 추론이나 F를 다시 호출하지 않는다. 게시용 `finalize`의 main은 이번 실행 세션의 START와 원래 checkout snapshot을 요구하므로 일반 재현에서는 호출하지 않는다. 게시 세션 전용 SHA manifest와 원격 게시 영수증도 일반 정확도 재현과 구분한다.

원본 의존성은 N3 seed 1·2·3의 last.pt, REAL_DEV의 기존 N3 좌표 JSON, detector/neck feature cache, 원영상 RGB, K와 등록 치수, 기하 참조다. [INPUT_LOCK.json](INPUT_LOCK.json)의 `models`, `input_manifest`, `source_bindings`에 상대 경로와 SHA256이 있다. checkpoint `data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed{1,2,3}/last.pt`와 예측 `predictions/REAL_DEV/N3_DIM_SYM_seed{1,2,3}.json`은 기존 TRAINING_COMPLETE/DEV_INFERENCE_COMPLETE 기록과 대조한다. logits가 cache에 있다고 가정하지 않고 기존 feature cache에서 N3만 다시 forward한다. 누락, 변조, native 복원 불일치가 있으면 BLOCKED로 종료하며 새 분포, 가중치, GT 보정으로 채우지 않는다.

정확도 재실행의 N3 capture에는 CUDA GPU와 기존 torch 2.1.1+cu118 환경이 필요하다. joint, 기존 F, 통계는 CPU 계산이며 OpenCV 4.9.0, NumPy 1.26.4, matplotlib 3.10.9를 사용했다. 추론 단계는 RGB와 추론 입력만 사용하며 참조는 좌표 봉인 후 평가에서만 읽는다. 공개 통계 검산은 표준 Python 통계 함수와 NumPy의 동일한 고정 bootstrap draw를 사용한다.

새 배포 파이프라인의 end-to-end latency는 측정하지 않았다. capture, joint, F의 실행 총시간을 검출부터의 배포 latency로 해석하거나 기존 직렬 15.8 ms를 JOINT의 시간으로 사용하지 않는다. 논문, LaTeX, PDF, 초록, PPT는 수정하지 않았다.
