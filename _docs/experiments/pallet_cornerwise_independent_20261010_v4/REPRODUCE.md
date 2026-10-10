# v4 원행 검토와 실행 재현

현재 결과를 독립 검토하는 경로와 원본 자산으로 새 전체 경로를 실행하는 경로를 구분한다. 아래 명령은 이 문서 작성 중 실행한 명령이 아니다. 실제 실행 완료 증거는 RESULT_KO.md와 각 receipt다. 기존 완료 파일은 덮어쓰지 않는다.

## 공개 저장소만으로 원행 복원·검토

저장소 루트에서 시작한다. Python3 표준 라이브러리만 있으면 원행 복원과 public moment 검토를 할 수 있다. 원본 RGB·가중치·GT를 다시 읽거나 모델/PnP/학습을 수행하지 않는다.

```bash
python3 -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.restore_archives
```

[ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)에 묶인 40MiB 조각을 확인한 뒤 `GEOMETRY_SEALED.jsonl.gz`, `PREDICTIONS.jsonl.gz`, `RUNTIME_ROWS.jsonl.gz`를 해당 증거 디렉터리에 복원한다. 이미 파일이 있으면 byte 수와 SHA256을 검사하고 그대로 둔다. 재압축을 하지 않는다. 세 원래 seal의 SHA를 그대로 사용할 수 있다. 추가 디스크 공간은 원래 세 파일 크기의 합 약342MB가 필요하다. 완료된 [ARCHIVE_RESTORE_CHECKS.json](ARCHIVE_RESTORE_CHECKS.json)은 세 파일을 실제 새 위치에 복원해 검증한 기록이다.

새 receipt 경로는 기존 파일과 겹치지 않게 지정한다. 다음 `$PWD`는 이 저장소 루트의 절대 경로다.

```bash
mkdir -p "$PWD/_docs/experiments/pallet_cornerwise_independent_review_local"
python3 -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.public_review   --input "$PWD/_docs/experiments/pallet_cornerwise_independent_20261010_v4"   --output "$PWD/_docs/experiments/pallet_cornerwise_independent_review_local/PUBLIC_REVIEW.json"   --export-csv "$PWD/_docs/experiments/pallet_cornerwise_independent_review_local/PER_FRAME.csv"
python3 -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.reprojection_check   --input "$PWD/_docs/experiments/pallet_cornerwise_independent_20261010_v4"   --output "$PWD/_docs/experiments/pallet_cornerwise_independent_review_local/REPROJECTION_CHECKS.json"
```

`public_review`는 봉인·scoring binding, 모든 공개 frozen input, 245 ID·6방법·1,470행·972 moment를 확인한다. private 자산을 열지 못하는 항목은 `external_frozen_dependencies_not_opened`에 남긴다. 이 공개 검사 자체는 bootstrap CI, 실제 물리 GT, GPU 재추론을 검증하지 않는다. `reprojection_check`는 저장된 R,t/K와 출력의 H 재투영·non-H·중심 유지만 scalar 산술로 확인한다. 원본 reference의 정확도를 인증하지 않는다.

각 명령은 `x` 생성 또는 동등한 기존 파일 보호 조건을 사용한다. 이미 local receipt가 생겼다면 새 빈 경로를 선택한다. 완료 receipt를 삭제하고 같은 실행인 척하지 않는다. 새 그림 생성은 원본 RGB가 필요하므로 공개 CSV 검토 경로에 포함하지 않는다.

## CI·posthoc 전체 검산의 원본 의존성

[verify.py](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/verify.py)는 statistics.py를 import하지 않고 저장된 원행·reference 필드·기존 10,000×13 session draw로 189개 CI와 posthoc를 재계산한다. 원래 code·가중치·calibration·baseline의 바이트 binding과 보호 snapshot도 확인하므로, 공개 clone만으로 private 입력 검사를 통과한 척하지 않는다. 실제 [VERIFICATION.json](VERIFICATION.json)은 원본 자산이 있는 환경에서 한 번 실행해 PASS한 기록이다.

필요한 자산은 고정 Base와 N3 checkpoint, 수정 감독 IMAGE_ROLE step3000 checkpoint·TRAINING_COMPLETION, 원본 source repository, immutable baseline repository, COHORT245·기존 camera/body registry·기존 평가 reference다. 경로보다 protocol에 기록된 바이트·SHA가 권위다. 외부 파일의 임의 대체, 새 head 학습, 새 calibration은 이 재현의 일부가 아니다.

다음 환경변수는 사용자가 보유한 별도 원본 디렉터리를 가리켜야 한다. source 및 baseline은 서로 다른 고정 입력이다. `$CORRECTED_FITS_ROOT`는 세 수정 head와 완료 receipt가 있는 디렉터리다.

```bash
export PALLET_SOURCE_ROOT=/absolute/path/to/original/source
export PALLET_BASELINE_ROOT=/absolute/path/to/immutable/baseline
export CORRECTED_FITS_ROOT=/absolute/path/to/completed/corrected/checkpoints
```

CLI의 `--source-root`·`--baseline-root`·`--fits`는 명시적으로 전달한다. 보고서의 원본 사용자 홈 경로를 새 공개 파일에 복사하지 않는다. 실행 환경에는 기존 PyTorch/CUDA, OpenCV4.9.0, NumPy, PIL, SciPy와 baseline import 의존성이 필요하다. 현재 프로토콜과 다른 버전은 우선 binding/출력 계약 차이를 기록해야 한다.

## 별도 새 실행 디렉터리의 전체 경로 재현

고정 코드로 재실행하려면 기존 결과를 덮어쓰지 않는 private subtree와 새 protocol/보호 receipt를 사용한다. 런타임 code는 출력 위치를 현재 v4 DOC 또는 아래 private root의 새 하위 디렉터리로 제한한다. `/tmp`로 `--output`을 바꾸면 이 검사를 통과하지 않는다. freeze 전에 확인할 source·baseline·fits 파일은 protocol에 그대로 묶인다. 원래 완료된 CPU 검사와 calibration을 재사용하며, 새 head·새 source feature cache를 만들지 않는다.

```bash
export CORNERWISE_REPLAY_OUTPUT=/dev/shm/pallet-cornerwise-independent-private-20261010-v4/replay_local
export CORNERWISE_REPLAY_METADATA="$PWD/_docs/experiments/pallet_cornerwise_independent_replay_local"
mkdir -p "$CORNERWISE_REPLAY_OUTPUT" "$CORNERWISE_REPLAY_METADATA"

python -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.run freeze   --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"   --fits "$CORRECTED_FITS_ROOT" --output "$CORNERWISE_REPLAY_OUTPUT"   --protocol "$CORNERWISE_REPLAY_METADATA/PROTOCOL.json"   --prior-bindings "$CORNERWISE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.run infer   --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"   --fits "$CORRECTED_FITS_ROOT" --output "$CORNERWISE_REPLAY_OUTPUT"   --protocol "$CORNERWISE_REPLAY_METADATA/PROTOCOL.json"   --prior-bindings "$CORNERWISE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.evaluate score   --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"   --fits "$CORRECTED_FITS_ROOT" --output "$CORNERWISE_REPLAY_OUTPUT"   --protocol "$CORNERWISE_REPLAY_METADATA/PROTOCOL.json"   --prior-bindings "$CORNERWISE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.statistics   --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"   --fits "$CORRECTED_FITS_ROOT" --input "$CORNERWISE_REPLAY_OUTPUT" --output "$CORNERWISE_REPLAY_OUTPUT"   --protocol "$CORNERWISE_REPLAY_METADATA/PROTOCOL.json"   --prior-bindings "$CORNERWISE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.verify   --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"   --fits "$CORRECTED_FITS_ROOT" --input "$CORNERWISE_REPLAY_OUTPUT" --output "$CORNERWISE_REPLAY_OUTPUT"   --protocol "$CORNERWISE_REPLAY_METADATA/PROTOCOL.json"   --prior-bindings "$CORNERWISE_REPLAY_METADATA/PROTECTION_BEFORE.json"
```

scoring은 geometry980+fixed490+observations245의 완전 seal과 `INFERENCE_RECEIPT.complete=True`, cleanup_error 없음, 기존 v3 control parity를 먼저 확인한다. 그 뒤에만 기존 GT/reference를 읽고, 새 PnP/모델 호출을 금지한다. 실제 보존된 v4 control은 수치 값 모두 정확히 일치했다. 다른 장치에서 허용오차1e−7 밖의 차이나 categorical 차이가 발생하면 receipt를 실패로 보존하며 자동 설정 변경·재시도를 하지 않는다.

원본 실사와 GT에 접근하는 이 경로는 새로운 실제 실행이다. 위 공개 산술 검토와 실행량이 다르다. 공개 CI를 확인하려고 이 경로 전체를 다시 실행할 필요는 없다.

## 실제 전체 경로 시간 재현

다른 GPU/CPU 연구 작업을 멈춘 quiet window에서만 별도로 실행한다. baseline·source·fits와 replay protocol은 위와 동일하게 전달한다. 측정 경로는 detector, N3, 초기 자세, ROLE와 feature 자세, 실제 LOO, 최종 PnP와 H 재투영까지 포함한다.

```bash
python -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.runtime measure   --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"   --fits "$CORRECTED_FITS_ROOT" --output "$CORNERWISE_REPLAY_OUTPUT"   --accuracy-output "$CORNERWISE_REPLAY_OUTPUT"   --protocol "$CORNERWISE_REPLAY_METADATA/PROTOCOL.json"   --prior-bindings "$CORNERWISE_REPLAY_METADATA/PROTECTION_BEFORE.json"
```

4경로×(20 warmup+26×5)=600회다. 출력 parity와 모델 entry counter, 자원 snapshot을 기록한다. 기존 좌표 cache 시간·기존 평균의 합산·모델 로드 시간을 전체 경로 시간으로 바꾸지 않는다. resource guard나 cleanup이 실패하면 해당 시작/실패 receipt를 보존하고 완료로 보고하지 않는다.

## 실제 기록의 확인 순서

1. [PROTOCOL.json](PROTOCOL.json)과 [EVALUATION_CONTRACT_KO.md](EVALUATION_CONTRACT_KO.md): 방법·245 ID·GT-free 경계·검산 정책.
2. [INDEPENDENCE_CHECKS.json](INDEPENDENCE_CHECKS.json): 제외 좌표·초기 prior 차단의 11개 실제 synthetic 검사.
3. [INFERENCE_RECEIPT.json](INFERENCE_RECEIPT.json), [GEOMETRY_SEAL.json](GEOMETRY_SEAL.json), [V3_CONTROL_PARITY.json](V3_CONTROL_PARITY.json): fresh245·980+490·control 실제 parity·봉인 SHA.
4. [SCORING_RECEIPT.json](SCORING_RECEIPT.json), [METRICS.json](METRICS.json), [VERIFICATION.json](VERIFICATION.json): 같은 raw 점수·표본통계·189 CI.
5. [FAULT_ISOLATION_CHECKS.json](FAULT_ISOLATION_CHECKS.json): 이미 저장된 원행만 읽은 추가 산술 한 번; branch/경계/산출집합 연관 분해.
6. [RUNTIME.json](RUNTIME.json), [REPROJECTION_CHECKS.json](REPROJECTION_CHECKS.json), [PUBLIC_REVIEW.json](PUBLIC_REVIEW.json): 실제600 전체 경로·scalar 계약·공개 moment 확인.
7. [FIGURE_BINDINGS.json](FIGURE_BINDINGS.json), [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json), [BUILD_LEDGER.json](BUILD_LEDGER.json): 고정 사례·원행 공개 바이트·실행량.

재현 성공은 같은 부정 결과와 원행 계약을 확인한다는 뜻이다. 주 방법이 고정 N3보다 정확해졌다는 성능 성공을 뜻하지 않는다.
