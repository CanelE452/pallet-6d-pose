# v5 C2 원행 검토와 실행 재현

아래 명령은 완료된 결과의 검토·재현 안내다. 문서 작성 중 모델/PnP/학습을 다시 실행한 명령이 아니다. 실제 실행은 [BUILD_LEDGER.json](BUILD_LEDGER.json)과 receipt에 묶었다. 기존 완료 파일을 덮어쓰지 않으며, 실패하면 새 설정으로 자동 재시도하지 않는다. 성능 성공과 같은 부정 결과의 재현을 구별한다.

## 공개 원행 복원

저장소 루트에서 Python3 표준 라이브러리로 실행한다. protocol이 기존 v4 geometry도 binding하므로 **v4를 먼저**, 이어 v5를 복원한다. 이 명령은 압축 해제·재압축이 아니라 원 gzip의 순서 있는 바이트 조각을 연결한다. 이미 복원 파일이 있으면 원 byte수/SHA를 검사하고 그대로 둔다.

```bash
python3 -B -m scripts.research.pallet_cornerwise_independent_20261010_v4.restore_archives
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.restore_archives
```

v5 [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)은 12조각/3파일435252323B를 기록한다. 추가 약435MB의 복원 공간과 v4 복원 공간이 필요하다. geometry181979774B, predictions183057104B, runtime70215445B의 SHA는 원 seal과 그대로 일치한다. [ARCHIVE_RESTORE_CHECKS.json](ARCHIVE_RESTORE_CHECKS.json)은 실제 새 위치에 복원한3파일 PASS다. 원행이 공개 저장소에 원래 큰 gzip filename으로 보이지 않는 경우에도 조각 전체를 보존한다.

공개 review 디렉터리를 별도로 구성한다면 큰3파일만 복원해서는 충분하지 않다. `PROTOCOL.json`, `GEOMETRY_SEAL.json`, `SCORING_RECEIPT.json`, `METRICS.json`, `FIXED_GEOMETRY_SEALED.jsonl.gz`, `FIXED_PREDICTIONS.jsonl.gz`, `OBSERVATIONS.jsonl.gz`, `BASE_N3_PARITY.json`도 같은 원 바이트로 필요하다. 실제 처음 archive review는 작은3파일을 빠뜨려 raw 읽기 전 실패했고 [ARCHIVE_PUBLIC_REVIEW_FAILURE.json](ARCHIVE_PUBLIC_REVIEW_FAILURE.json)에 보존했다. 필요한 파일을 채운 [ARCHIVE_REVIEW_INPUTS_COMPLETE.json](ARCHIVE_REVIEW_INPUTS_COMPLETE.json)과 성공 [ARCHIVE_PUBLIC_REVIEW.json](ARCHIVE_PUBLIC_REVIEW.json)이 별도다.

## 공개 moment·재투영·line factor 검토

다음 `$PWD`는 이 publication 저장소 루트의 절대 경로다. 기존 receipt와 겹치지 않는 새 in-repository 경로를 선택한다. private RGB·weights·GT를 다시 읽거나 detector/PnP를 실행하지 않는다.

```bash
mkdir -p "$PWD/_docs/experiments/pallet_partial_line_review_local"
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.public_review \
  --input "$PWD/_docs/experiments/pallet_partial_line_independent_20261010_v5" \
  --output "$PWD/_docs/experiments/pallet_partial_line_review_local/PUBLIC_REVIEW.json" \
  --export-csv "$PWD/_docs/experiments/pallet_partial_line_review_local/PER_FRAME.csv"
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.reprojection_check \
  --input "$PWD/_docs/experiments/pallet_partial_line_independent_20261010_v5" \
  --output "$PWD/_docs/experiments/pallet_partial_line_review_local/REPROJECTION_CHECKS.json"
```

`--input`는 gzip filename이 아니라 결과 **디렉터리**다. `public_review`는 245ID·6방법·1470행·972moment·46공개 input binding을 확인한다. private 의존성은 `external_frozen_dependencies_not_opened`로 남긴다. bootstrap CI나 실제 물리 GT를 이 검사로 인증하지 않는다. `reprojection_check`는 저장 R,t/K와 H 교체, non-H 좌표·중심 유지 및 fallback을 scalar 산술로 확인한다. 원래 reference의 진실성을 증명하지 않는다.

line factor checker의 외부 출력 guard는 **새 빈 임시 디렉터리**를 요구한다. 위 review 디렉터리나 기존 결과의 임의 다른 파일로 출력하려고 하지 않는다.

```bash
mkdir -p /tmp/pallet-v5-line-factor-review-local
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.line_checks \
  --input "$PWD/_docs/experiments/pallet_partial_line_independent_20261010_v5" \
  --protocol "$PWD/_docs/experiments/pallet_partial_line_independent_20261010_v5/PROTOCOL.json" \
  --output /tmp/pallet-v5-line-factor-review-local/LINE_FACTOR_CHECKS.json
```

이미 receipt가 있다면 새 빈 디렉터리를 선택한다. 이 checker는 같은pool/consumededge/semantic factor count·point/line residual·최소4 pointinlier·저장localrank label과 NEW/fallback을 검토한다. Jacobian/SVD·optimizer를 다시 실행하거나 실사 물리 소유권을 인증하지 않는다.

## 독립 CI와 source/weights binding

[verify.py](../../../scripts/research/pallet_partial_line_independent_20261010_v5/verify.py)는 statistics.py를 import하지 않고 원행1470·posthoc980·기존10000×13 session draw로 189CI·moment·subset·mask/corner를 확인한다. 실제 [VERIFICATION.json](VERIFICATION.json)은 원본 의존성을 가진 환경에서 한 번 수행해 PASS했다. 공개 clone만으로 private source·checkpoint binding 검사를 통과한 척하지 않는다.

원본 자산이 있는 독자는 고정 Base/N3, 수정 감독 마지막 IMAGE_ROLE step3000 head와 TRAINING_COMPLETION, 기존 source/baseline, cohort245·camera/body registry·평가 reference가 필요하다. 경로보다 protocol의 byte수/SHA가 authority다. 기존 head 재사용이며 새 학습·새 calibration·새 feature cache는 이 재현의 일부가 아니다. 환경은 기록된 PyTorch/CUDA·OpenCV4.9.0·NumPy1.26.4·SciPy·PIL·Ultralytics와 source import 의존성을 포함한다.

```bash
export PALLET_SOURCE_ROOT=/absolute/path/to/original/source
export PALLET_BASELINE_ROOT=/absolute/path/to/immutable/baseline
export CORRECTED_FITS_ROOT=/absolute/path/to/completed/corrected/checkpoints
```

실제 원본 사용자 홈 경로를 공개 문서에 복사하지 않는다. source와 baseline은 서로 다른 고정 입력이다. `$CORRECTED_FITS_ROOT`는 corrected head3개와 완료 receipt를 보유한 디렉터리다.

## 별도 새 전체 실행

출력은 현재 v5 DOC 또는 `/tmp/pallet-partial-line-independent-private-20261010-v5`의 **새 하위 디렉터리**로 제한된다. protocol과 protection receipt는 binding helper에 맞게 저장소 안 새 절대 경로에 둔다. `/tmp/PROTOCOL.json` 같은 경로로 freeze하면 in-repository binding 계약을 통과하지 않는다. 기존 검사 실패/성공·frozen renderer·source inputs는 보존한다.

```bash
export PARTIAL_LINE_REPLAY_OUTPUT=/tmp/pallet-partial-line-independent-private-20261010-v5/replay_local
export PARTIAL_LINE_REPLAY_METADATA="$PWD/_docs/experiments/pallet_partial_line_replay_local"
mkdir -p "$PARTIAL_LINE_REPLAY_OUTPUT" "$PARTIAL_LINE_REPLAY_METADATA"

python -B -m scripts.research.pallet_partial_line_independent_20261010_v5.run freeze \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --protocol "$PARTIAL_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$PARTIAL_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_independent_20261010_v5.run infer \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --protocol "$PARTIAL_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$PARTIAL_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_independent_20261010_v5.evaluate score \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --protocol "$PARTIAL_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$PARTIAL_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_independent_20261010_v5.statistics \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --input "$PARTIAL_LINE_REPLAY_OUTPUT" --output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --protocol "$PARTIAL_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$PARTIAL_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"

python -B -m scripts.research.pallet_partial_line_independent_20261010_v5.verify \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --input "$PARTIAL_LINE_REPLAY_OUTPUT" --output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --protocol "$PARTIAL_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$PARTIAL_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"
```

infer는 245frame·980method+490fixed를 seal한 뒤 GT 접근을 허용한다. scoring은 complete inference·cleanup_error=None·같은 protocol/seal·고정 Base/N3 parity·기존 v4 control parity를 먼저 확인한다. v4 control은 numeric tolerance1e−7와 categorical exact를 요구하며, 실제 이번 245는 모든 수치가 정확히 같았다. 다른 장치의 nonzero 차이는 within-tolerance와 bit-exact를 구분해 기록한다. 범위를 벗어나면 실패 receipt를 보존한다.

GT scoring 단계는 새 PnP/model을 금지한다. 입력 recipe는 기존 frozen245·reference·N3 phase이며 새 label이나 severe74를 끼워 넣지 않는다. 재현 명령은 새로운 실제 실행이며 공개 원행 산술 검토와 실행량이 다르다.

## 고정 line quality 산술 재현

별도 준비된 replay 결과에서 같은8px 정의로 실행하려면 아래와 같이 원행이 있는 디렉터리 안에 **새** protocol/receipt를 생성한다. 기존 완료 LINE_QUALITY 파일이 있으면 같은 경로를 재사용하지 않는다. query/line/point 후보를 다시 예측하지 않는다.

```bash
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.line_quality freeze \
  --input "$PARTIAL_LINE_REPLAY_OUTPUT" --output "$PARTIAL_LINE_REPLAY_OUTPUT"
python3 -B -m scripts.research.pallet_partial_line_independent_20261010_v5.line_quality run \
  --input "$PARTIAL_LINE_REPLAY_OUTPUT" --output "$PARTIAL_LINE_REPLAY_OUTPUT"
```

실제 LINE_QUALITY_PROTOCOL freeze는 완전 기하 봉인·점수화·통계 완료 후이며, 기존 core policy의8px 정의와 별도 산술 code/입력을 산술 전에 묶은 supplemental 감사다. accuracy 사전등록이나 score 이전 감사로 보고하지 않는다. 새로운 score 기반 설정 변경은0이다. 이 산술은 저장된 fixed-N3-phase native reference에 대한 무한선 RMS와 input point 오차를 확인한다. matched=False/끝점미확인은UNKNOWN, fallback rawcandidate는 승인finalinlier와 구분한다. 실제 edge 소유권·along-edge3D visible fraction을 인증하거나 threshold를 조정하지 않는다.

## 실제 전체 경로 시간 재현

다른 CPU/GPU 연구 workload를 멈춘 quiet window에서만 실행한다. fresh600은 4경로×(warmup20+26×5)다. inference/scoring은 이미 완료해야 한다. resource/parity 검사와 durable journal은 타이머 밖이며 초기 pose·ROLEfeaturepose·LOO·최종C2·H재투영은 안이다.

```bash
python -B -m scripts.research.pallet_partial_line_independent_20261010_v5.runtime measure \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT" \
  --fits "$CORRECTED_FITS_ROOT" --output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --accuracy-output "$PARTIAL_LINE_REPLAY_OUTPUT" \
  --protocol "$PARTIAL_LINE_REPLAY_METADATA/PROTOCOL.json" \
  --prior-bindings "$PARTIAL_LINE_REPLAY_METADATA/PROTECTION_BEFORE.json"
```

기존 cache 시간을 전체 경로 latency로 사용하거나 stage 평균을 합산하지 않는다. 이번 실제 runtime600은 한 번 complete/PASS였으며 detector601(초기화1 포함)·N3450·ROLE150이 기록됐다. runtime schema의 legacy v4 label은 actualv5 route/protocol binding과 구분한다.

## 검토 순서와 한계

1. PROTOCOL·EVALUATION_CONTRACT:245범위,같은관측C2,최소4point,prior없음,중복제거.
2. PARTIAL_LINE_CHECKS 실패+REPAIR1:첫순서버그와별도12PASS,failedsource보존.
3. INFERENCE_RECEIPT·GEOMETRY_SEAL·V4_CONTROL_PARITY:GT이전완전봉인·245동일control.
4. SCORING_RECEIPT·METRICS·VERIFICATION:동일원행moment·subset·189CI.
5. LINE_QUALITY_CHECKS·LINE_FACTOR_CHECKS:proxy선품질과수치계약,physicaltruth인증아님.
6. RUNTIME·REPROJECTION·PUBLIC_REVIEW:600fresh전체경로,scalarH교체,972moment.
7. FIGURE_BINDINGS_REPAIR1·ARCHIVE_MANIFEST·BUILD_LEDGER:고정6사례·원바이트조각·실제실행량.

새 이미지 표시에는 원 RGB가 필요하므로 공개 산술 검토 명령에 포함하지 않았다. frozen render.py 실패는 RENDER_FAILURE로 보존했고 별도 render_repair.py가 metadata None 처리만 고쳐 최종8PNG를 만들었다. 이 문서는 재현 안내이며 추가 성능실험·학습 허가나 목표 완료 선언이 아니다.

## 저장 선-heldout packet 감사

[LINE_HELDOUT_CHECKS.json](LINE_HELDOUT_CHECKS.json)은 geometry245와 저장447cornerrecord를 새 fit 없이 검사한 추가산술1회다. 자체 LINE_HELDOUT_PROTOCOL은 점수화 완료 후 기존 code·원행·정의를 별도 산술 전에 묶었으며 accuracy 사전등록 protocol은 아니다.793선 중17선만 양끝점과H가 모든 scoring/fit/generator pool에서 빠진 storedNEW witness를 갖고,776선은UNVERIFIED다. qualified17은12frame에서 한끝점temporaryheldout+다른끝점ineligible인 경우이며, 두끝점을 명시적으로 함께heldout한 line-specific 실행은0이다. NONE이나 물리소유권정답으로 바꾸지 않았다. 공개 큰geometry를 복원한 뒤 checker를 원격GT/model 없이 검토할 수 있다. 기존 완료파일을 덮어쓰지 않는 새 디렉터리와 CLI 정의는 [line_heldout_audit.py](../../../scripts/research/pallet_partial_line_independent_20261010_v5/line_heldout_audit.py)를 따른다. 실제 새로운 양끝점제외 PnP 경로는 이번 재현에 포함되지 않는다.
