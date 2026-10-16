# v6 양끝점 heldout 선 절제의 원행 검토와 재현

이 문서는 실제 완료된 결과를 검토하거나 별도 새 실행으로 재현하는 안내다. 문서 작성 중 모델·PnP·optimizer·학습·GT 점수화를 다시 실행하지 않았다. 현재 목표 판정은 [RESULT_KO.md](RESULT_KO.md)의 부정 결과다. 기존 파일을 덮어쓰거나 실패 뒤 설정을 바꿔 자동 반복하지 않는다. formal [PROTOCOL.json](PROTOCOL.json) SHA는 `1de41781e280963c16486e828990ba9d4c4e1556c45faedd5ddc7edd2b55ce2c`이며 frozen core와 source-only ROLE CAL을 유지한다.

실제 inference245·geometry980+fixed490·score1470·VERIFY189CI·VALIDATION3500726check·scalarH980행·fresh600runtime·8PNG·gate품질245행·공개 byte 복원이 모두 완료됐다. 정확한 시도 수와 primitive는 [BUILD_LEDGER.json](BUILD_LEDGER.json)을 따른다. runtime CLI는2회였지만 첫 stage 누락은 model/timer 전 argparse 실패였으며 numerical600측정은1회다. 아래 명령을 적었다는 이유로 새 호출을 실행한 것으로 세지 않는다.

## 공개 large raw 복원

protocol의 unchanged control input이 v5 geometry이고 v5는 v4 geometry를 binding한다. 공개 clone의 저장소 루트에서 **v4→v5→v6** 순서로 원 gzip 바이트를 복원한다. 압축 해제·재압축이 아니라 순서 있는 byte part의 연결이며 기존 복원 파일이 있으면 SHA/byte수를 검사하고 그대로 둔다.

```bash
python3 -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.restore_archives
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.restore_archives
python3 -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.restore_archives
```

v6는12part·총476488594byte이며 원geometry198198150byte, predictions199275445byte, runtime79014999byte다. [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)이 원/part SHA를 기록한다.94개 공개 파일만으로 만든 별도 새 evidence bundle은 Git clone이 아니며 private RGB·weight·GT를 포함하지 않았다. 처음에 v4/v5/v6 원gzip9개가 모두 없는 상태에서 위 세 stage를 실제 각1회 실행해9개 모두 새 byte 복원·SHA 확인 PASS였다. [준비](PUBLIC_ARCHIVE_PREPARATION.json), [V4](PUBLIC_ARCHIVE_RESTORE_V4.json), [V5](PUBLIC_ARCHIVE_RESTORE_V5.json), [V6](PUBLIC_ARCHIVE_RESTORE.json), [fresh 공개 검토](PUBLIC_ARCHIVE_REVIEW.json)가 실제 기록이며 복원 실패0회다. raw seal의 SHA와 바이트가 authority이고 archive를 새 모델 결과로 보지 않는다. 별도 review 폴더에는 PROTOCOL·GEOMETRY_SEAL·INFERENCE_RECEIPT·SCORING_RECEIPT·METRICS·fixed geometry/predictions·OBSERVATIONS·BASE_N3_PARITY·V5_CONTROL_PARITY 및 공개 bound dependency도 원 byte로 함께 있어야 한다.

## 표준 라이브러리 공개 산술 검토

`$PWD`는 publication 저장소 루트의 절대 경로다. 기존 receipt와 겹치지 않는 새 in-repository 디렉터리를 쓴다. 공개 moment·status·ID/SHA 검토는 private RGB/weights/reference를 다시 열거나 모델/PnP를 호출하지 않는다. physical GT의 진실성과 bootstrap CI를 public_review 하나로 인증하지 않는다.

```bash
mkdir -p "$PWD/_docs/experiments/pallet_endpoint_line_public_review_local"
python3 -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.public_review \
  --input "$PWD/_docs/experiments/pallet_partial_line_heldout_20261010_v6" \
  --output "$PWD/_docs/experiments/pallet_endpoint_line_public_review_local/PUBLIC_REVIEW.json" \
  --export-csv "$PWD/_docs/experiments/pallet_endpoint_line_public_review_local/PER_FRAME.csv"
python3 -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.reprojection_check \
  --input "$PWD/_docs/experiments/pallet_partial_line_heldout_20261010_v6" \
  --output "$PWD/_docs/experiments/pallet_endpoint_line_public_review_local/REPROJECTION_CHECKS.json"
```

`--input`는 gzip 파일이 아니라 결과 **디렉터리**다. scalar reprojection checker는 저장 R,t/K/등록 vertex, H교체·non-H 관측/center 유지와 fallback의 실제 output 계약만 확인한다. 모델 fit이나 accuracy scoring을 반복하지 않는다.

별도 `validation_checks.py`는 destination을 v6 DOC의 새 단일 receipt 또는 고정 v6 private subtree로 제한한다. 현재 완료 receipt를 덮어쓰지 않으려면 다음 새 하위 디렉터리를 사용한다.

```bash
mkdir -p /tmp/pallet-partial-line-heldout-private-20261010-v6/validation_review_local
python3 -I -S scripts/research/pallet_partial_line_heldout_20261010_v6/validation_checks.py \
  --input "$PWD/_docs/experiments/pallet_partial_line_heldout_20261010_v6" \
  --protocol "$PWD/_docs/experiments/pallet_partial_line_heldout_20261010_v6/PROTOCOL.json" \
  --output /tmp/pallet-partial-line-heldout-private-20261010-v6/validation_review_local/VALIDATION_CHECKS.json
```

이 checker는 저장된 native bank·H/두endpoint 제외·prior-free witness·candidate/scoring/refit IDs·8px decision·deepcopy filtering·consumed support·최종 같은factorpool·NEW/fallback·operation count를 검사한다. Jacobian/SVD·optimizer·새 pose를 실행하거나 physical ownership을 인증하지 않는다. active candidate 제외와 global cached primitive의 excluded ID 포함 가능성을 구분한다. output symlink/ancestry·existing file guard를 우회하지 않는다.

## 원본 의존성을 가진 독립 검산

[verify.py](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/verify.py)는 statistics.py를 import하지 않고 saved1470score·posthoc980·기존10000×13 draw로 moments·mask/corner·subset·189CI를 확인한다. 원본 input binding도 확인하므로 public clone만으로 private source·checkpoint·reference binding이 확인된 척하지 않는다.

원본 RGB·Base/N3 weight·immutable baseline, corrected3head checkpoint와 TRAINING_COMPLETION, source ROLE CAL, existing cohort245·camera/body registry·reference가 필요하다. 실제 사용자 홈 경로를 공개 문서에 복사하지 않는다. 환경 변수의 예는 다음과 같다.

```bash
export PALLET_SOURCE_ROOT=/absolute/path/to/original/source
export PALLET_BASELINE_ROOT=/absolute/path/to/immutable/baseline
export CORRECTED_FITS_ROOT=/absolute/path/to/completed/corrected/checkpoints
```

checkpoint는 corrected last IMAGE_ROLE step3000이며 새 학습·새 CAL·새 feature cache를 요구하지 않는다. 이 head의 SHA는 `882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d`다. 원래 미수정 IMAGE_ROLE을 대용하지 않는다. 기록된 PyTorch/CUDA·OpenCV·NumPy·SciPy·PIL·Ultralytics/source import 환경과 byte/SHA를 확인한다.

## 별도 fresh245 전체 실행

출력은 현재 v6 DOC 또는 `/tmp/pallet-partial-line-heldout-private-20261010-v6`의 **새 하위 디렉터리**로 제한된다. protocol/protection metadata는 binding helper에 맞는 저장소 안 새 **절대 경로**다. `/tmp/PROTOCOL.json`처럼 metadata를 repository 밖에 두면 freeze 이후 binding에서 실패할 수 있다. 기존 frozen CPU receipt·code·EVAL의 SHA는 그대로 사용하며 별도 CPU/학습을 반복하지 않는다.

```bash
export ENDPOINT_LINE_REPLAY_OUTPUT=/tmp/pallet-partial-line-heldout-private-20261010-v6/replay_local
export ENDPOINT_LINE_REPLAY_METADATA="$PWD/_docs/experiments/pallet_endpoint_line_replay_local"
mkdir -p "$ENDPOINT_LINE_REPLAY_OUTPUT" "$ENDPOINT_LINE_REPLAY_METADATA"

python -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.run freeze \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --protocol "$ENDPOINT_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$ENDPOINT_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.run infer \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --protocol "$ENDPOINT_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$ENDPOINT_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.evaluate score \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --protocol "$ENDPOINT_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$ENDPOINT_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.statistics \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --input "$ENDPOINT_LINE_REPLAY_OUTPUT" --output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --protocol "$ENDPOINT_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$ENDPOINT_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.verify \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --input "$ENDPOINT_LINE_REPLAY_OUTPUT" --output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --protocol "$ENDPOINT_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$ENDPOINT_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"
```

complete inference receipt·cleanup_error=None·980+490geometry·245OBS·protocol/seal population와 GT-free field를 확인한 뒤에만 reference를 읽는다. v5 C2 geometry parity도 reference 전에 수행한다. actual C2 numeric field 존재 여부와categorical exact, atol1e−7/rtol0를 검사하며 nonzero/within-tolerance와 bitexact는 다르다. 이번245control은 비교한 수치가 모두 exact였다. 경계를 넘는 차이는 실패 receipt로 남기고 무시하거나 다른 policy로 반복하지 않는다.

후단 score는 새로운 fit·model을 금지하며 fixed N3 phase의 기존 target/permutation/human state만 사용한다. unmatched proxy를 negative known correspondence로 바꾸지 않는다. severe74·새 label·유리한 reference phase를 추가하지 않는다. 이는 새 실제 accuracy 실행의 재현 안내이며 공개 saved-row 산술과 실행량이 다르다.

## 실제 전체 경로 시간

다른 CPU/GPU 연구 workload가 없는 quiet window에서만 실행한다. 완전한 inference/scoring 결과가 먼저 필요하다. **`measure` stage를 반드시 명시**한다. 실제 첫 CLI는 이를 누락해 argparse exit2로 model/timer 전에 실패했고, 해당 operator 실패를 보존했다. 이 실패는 알고리즘 실행이나600 numeric 측정으로 세지 않는다.

```bash
python -B -m scripts.research.pallet_partial_line_heldout_20261010_v6.runtime measure \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --accuracy-output "$ENDPOINT_LINE_REPLAY_OUTPUT" \
  --protocol "$ENDPOINT_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$ENDPOINT_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"
```

same26 eligible panel·4arm×(20warmup+130measured)=600의 detector/N3/초기pose/featurepose/ROLE/코너LOO/native dualendpointLOO/finalC2/H재투영이 full interval 안에 있다. 초기 model load·RGB decode·GT scoring·parity/journal은 receipt의 실제 경계를 따른다. cached좌표 replay나 기존 stage평균합산으로 latency를 대체하지 않는다. 실제 official latency와 model/OpenCV 호출·interference 결과는 완료된 [RUNTIME.json](RUNTIME.json)에 있다. 주 경로 measured130 평균147.363178323ms이며 초기·heldout·최종 자세를 모두 포함한다. cached coordinate replay는False다.

## 후행 gate 품질 감사와 검토 순서

`gate_quality.py`는 같은8px 정의로 이미 봉인·점수화된 선/점·native reference를 읽는 별도 supplemental code다. core 성능 protocol과 달리 **score/statistics 완료 후** own protocol/code/input을 산술 전에 freeze한다. GT proxy endpoint RMS≤8/>8/UNKNOWN과 SUPPORTED/CONTRADICTED/UNVERIFIED의 관계, validator native point proxy품질·기존 frame pose차이를 확인한다. 새 pose·model·ray·threshold tuning은0이며 실제 visible ownership이나 along-edge3D fraction을 인증하지 않는다.

```bash
export ENDPOINT_GATE_REVIEW_OUTPUT=/tmp/pallet-partial-line-heldout-private-20261010-v6/gate_review_local
python3 -I -S scripts/research/pallet_partial_line_heldout_20261010_v6/gate_quality.py freeze \
  --input "$PWD/_docs/experiments/pallet_partial_line_heldout_20261010_v6" \
  --output "$ENDPOINT_GATE_REVIEW_OUTPUT"
python3 -I -S scripts/research/pallet_partial_line_heldout_20261010_v6/gate_quality.py run \
  --input "$PWD/_docs/experiments/pallet_partial_line_heldout_20261010_v6" \
  --output "$ENDPOINT_GATE_REVIEW_OUTPUT"
```

예시 출력은 반드시 없는 새 경로여야 한다. 현재 공개 DOC의 gate receipt는 덮어쓰지 않는다. 실제 gate freeze1회·산술1회는 score/statistics 후 수행됐고793unused선은504SUPPORTED·179REJECT·110UNVERIFIED였다. reject130개는 proxy≤8에 맞았고45개는>8였으며4개는UNKNOWN이다. 이 후행 분류가 core 사전등록이나 실제 physical ownership truth가 되지 않는다. 다른 saved replay를 감사할 때도 명시 CLI·새 exclusive output guard를 따른다. 현재 문서에서 추가 산술을 실행한 것이 아니다.

검토 순서는 corePROTOCOL/평가계약→CPU receipt→complete inference/seal/controlparity→score/METRICS/독립VERIFY→저장 endpoint/factor/H산술→fresh600→고정 그림·archive/복원→BUILD_LEDGER/publication receipt다. raw line source/support query, consumed/rejected/retained/finalinlier와 fallback 후보를 구분한다. 실제 RGB를 표시하는 그림은 원 RGB가 필요하므로 공개 숫자 검사와 분리한다.

기존 corrected3head의245장66-way decoder 평가는 이미 완료됐지만 현재 새로운 confidence/cornerwise/C2는ROLE만 평가했다. GEOM/NO_ROLE용 새 decoder CAL이나 새 성능 비교가 실행된 것처럼 쓰지 않는다. source controlled4variants와 source role3의 독립 semantic replay 제한도 이 재현이 해결한 것으로 보고하지 않는다. 최상위 목표 미달과 실행 재현 가능성은 별개다.
