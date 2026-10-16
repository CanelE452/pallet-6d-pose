# v7 세 head·두 관측 공급의 검토와 재현 안내

source calibration16forward/256exposure·ROLE exact-byte reuse0, source 계약9개·CPU pipeline13개·source 독립 산술198848개·stream4그룹 검사, GT 없는 fresh245장 inference·scoring2450행·통계·432CI 슬롯 검산·1500회 실제 전체 경로 시간·8PNG 생성과 시각 검토는 완료됐다. 별도 공개 CI 산술1회와 runtime scalar B도 PASS했고, 원 gzip6개가 없는 fresh dependency bundle에서 byte 복원·public moment 재검토까지 완료했다. 실제 시도와 실행량은 [BUILD_LEDGER.json](BUILD_LEDGER.json), 부정 정확도 판정과 제한은 [RESULT_KO.md](RESULT_KO.md)를 따른다. 게시 완료는 별도 publication 영수증으로 구분한다. 아래 재현 명령을 적었다는 사실을 새로운 실행 PASS로 세지 않는다.

## 입력과 의미

corrected step3000 GEOMETRY_ONLY·IMAGE_NO_ROLE·IMAGE_ROLE, 완료된9000update 영수증·READY 감독 SHA, source CAL128(index768..895)·기존 cached feature·family/cache identity, fixed Base/N3·baseline·camera/body registry·동일 쉬움153/중간92 cohort가 필요하다. original RGB·weights·reference는 외부 read-only 의존성이고 공개 saved-row 산술과 별개다. 새 학습/RGB/feature 재생성은 필요하지 않다.

`GEOMETRY_ONLY`가 role25:28을 유지한다는 구현 의미를 보존한다. GEOM−NO_ROLE은 image와 role을 함께 바꾸므로 순수 영상 절제로 부르지 않는다. ROLE−NO_ROLE은 같은 image channels에서 role을 바꾸지만 각자 source-only CAL을 가진 전체 head 방법 비교다. 실제 점수로 head·checkpoint·계수를 바꿔 재실행하지 않는다.

## 실제 source calibration 기록

[source_calibration/PROTOCOL.json](source_calibration/PROTOCOL.json)은 source pass 이전 code/input을 고정했고 [COMPLETION](source_calibration/COMPLETION.json)은 총16head forward·256exposure, ROLE 새0·historical8, INPUT before/after equality·보호17017·원 사용자 checkout449 상태 보존과 no model-feature/pose/ray/training/RGB 재실행을 기록한다. 기존 READY·features를 원본과 byte/SHA로 확인한다. 공개 ROLE9파일은 exact-byte 재사용이며 과거 실행 영수증의8을 이번8로 잘못 세지 않는다.

source CAL을 새로 재현하려면 원 cache·manifest·family split·READY·corrected weight가 필요하고 별도 빈 output에freeze→run 한 번을 실행해야 한다. 지금 완료된 경로를 덮어쓰지 않는다. `calibration.py freeze/run`의 필수 입력은 `--features`, `--fits`, `--source-root`이며 선택적으로 `--cache-manifest`, `--prior-bindings`, `--output`을 명시한다. 공개 수치 검토에는 이미 완료된 CAL 파일을 그대로 사용하므로 새 forward가 필요하지 않다. SOURCE_CALIBRATION_CONTRACT_CHECKS는 scalar/list fixture와 mask-prefix 계약만 검사하며 whole head numerical parity를 주장하지 않는다.

완료된 SOURCE_CALIBRATION_CHECKS는 source CAL 원 logits/READY scalar를 stdlib로 독립 계산한198848개 PASS다. 새 source head forward가 아니라 저장자료 산술1회이며 ROLE9byte exact reuse/historical8/current0을 함께 검사한다. actual batch source calibration16과 독립 산술1의 비용을 합쳐 head17로 세지 않는다. source-only 조건부 precision을 실사 guarantee로 해석하지 않는다.

## 공개 saved-row 산술

공개 archive는 original gzip의 byte parts를 연결해 원 SHA를 유지한다. [PUBLIC_REVIEW.json](PUBLIC_REVIEW.json)의 실제2450행·735관측·1620moment·59public binding PASS와 [REPROJECTION_CHECKS.json](REPROJECTION_CHECKS.json)의1960행·1238NEW scalar PASS는 저장자료 검사다. public_review 자체는 모든 bootstrap CI를 검사하지 않는다. 기존 verify는 statistics.py를 import하지 않는 독립 stdlib 산술 본문으로432CI 슬롯을 검산했으나 wrapper가 production POLICY를 읽으며 NumPy/OpenCV를 import하고 원 private dependency bindings도 필요하다. 따라서 그 전체 entrypoint를 순수 stdlib 공개 재현이라고 부르지 않는다.

이를 구분하기 위해 새 `public_ci_review.py`는 production 모듈을 import하지 않고 private weights·RGB·GT를 열지 않는다. 완성된2450개의 saved scalar와 cohort·기존10000×13 draw를 읽고1620moment,144paired group/432CI 슬롯을 독립 계산한다. 339nonempty/93None의 빈 CI 구분, paired ID·NEW/fallback·bootstrap 빈 resample 분모도 확인한다. 새로운 draw나 pose/score를 만들지 않는다. 정확도 protocol을 바꾸지 않는 별도 사후 공개 감사이며, 자기 코드와 공개 입력을 **자기 산술 전에** PUBLIC_CI_PROTOCOL로 고정하고 기존 protocol/receipt를 덮어쓰지 않는다. [실제 현재 검사](PUBLIC_CI_CHECKS.json)는 freeze1/run1·25078exact/2730numeric·0FAIL, 최대 차이4.55e−13 PASS였다.

아래는 parser와 정적으로 대조한 공개 saved-row 명령이다. 전용 research branch를 clone한 repository root에서 시작하고 새 review directory를 쓴다. `--input`은 증거 디렉터리이며 gzip 파일 경로가 아니다. 첫 restore는 byte 복원이며 private 영상·모델이 없어도 실행할 수 있다. 먼저 CI 전용 경로를 실행하면 다른 private 의존성 없이 핵심 숫자를 확인할 수 있다.

```bash
export THREE_HEAD_PUBLIC_DOC="$PWD/_docs/experiments/pallet_three_head_observation_20261010_v7"
export THREE_HEAD_PUBLIC_SCRIPT="$PWD/scripts/research/pallet_three_head_observation_20261010_v7"
export THREE_HEAD_PUBLIC_REVIEW=/tmp/pallet-three-head-observation-private-20261010-v7/review_local
mkdir -p "$THREE_HEAD_PUBLIC_REVIEW"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/restore_archives.py" \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --receipt "$THREE_HEAD_PUBLIC_REVIEW/RESTORE.json"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/public_ci_review.py" freeze \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_PUBLIC_REVIEW"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/public_ci_review.py" run \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_PUBLIC_REVIEW"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/restore_archives.py" \
  --input "$PWD/_docs/experiments/pallet_cornerwise_independent_20261010_v4" \
  --receipt "$THREE_HEAD_PUBLIC_REVIEW/RESTORE_V4.json"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/public_review.py" \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_PUBLIC_REVIEW/PUBLIC_REVIEW.json" \
  --export-csv "$THREE_HEAD_PUBLIC_REVIEW/PER_FRAME.csv"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/reprojection_check.py" \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_PUBLIC_REVIEW/REPROJECTION_CHECKS.json"
```

public_review의59binding에는 이전V4의 `GEOMETRY_SEALED.jsonl.gz`가 포함되므로 위 V4 restore를 먼저 실행한다. V4 manifest는 해당 geometry를 포함한 original3개를 일괄 복원한다. 59binding에 그3개 모두가 직접 포함된다고 주장하지 않는다. `public_ci_review` 자체는 V7 scored 원행 두 개·cohort·draw만 필요하다. 기존 파일이 있으면 byte/SHA 일치만 확인해 보존하며 새 파일은 `xb`로 한 번 만든다. 부분을 decompress/recompress하지 않는다. 그 재생성은 기존 raw byte의 복원이지 새로운 detector/pose/score 실험이 아니다.

[실제 fresh bundle 검사](PUBLIC_FRESH_BUNDLE_CHECKS.json)는207파일·944333100B를 복사한 dependency bundle에서 V4/V7의6original gzip이 없음을 먼저 확인하고,6개·873655209B를 새로 복원했다. [fresh public review](PUBLIC_FRESH_REVIEW.json)는2450행·735관측·1620moment·59binding PASS다. public_review는 원환경1회+fresh bundle1회, 총2회이며 fresh bundle CI 산술은 반복하지 않았다. 실제 git clone 전체환경/GPU 재실행으로 바꾸어 부르지 않는다.

추가 all-candidate scalar 검산을 새 빈 private output에서 확인하려면 같은 공개 입력·coefficients에 새 checker protocol을 먼저 고정한 뒤 실행한다. frozen accuracy protocol이나 완료된 validation receipt는 덮어쓰지 않는다.

```bash
python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/validation_checks.py" freeze \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_PUBLIC_REVIEW" \
  --protocol "$THREE_HEAD_PUBLIC_DOC/PROTOCOL.json" \
  --calibration-root "$THREE_HEAD_PUBLIC_DOC/source_calibration"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/validation_checks.py" run \
  --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_PUBLIC_REVIEW" \
  --protocol "$THREE_HEAD_PUBLIC_DOC/PROTOCOL.json" \
  --calibration-root "$THREE_HEAD_PUBLIC_DOC/source_calibration"
```

이 checker의 실제 현재 run은1회9095709check PASS였다. 모든 packet의 후보와 선택 후보의 추가 재검사 횟수를 구분하며 stored residual/projection·active ID·final actual_pose 연결·계수·stored rank를 확인한다. JSON null에서 원 NaN payload/hash를 복원할 수 없다는 제한, 절대+상대 tolerance와 global physical uniqueness/ownership 미증명을 유지한다.

공개 CI reader는 원 gzip을 한 행씩 읽고 id/session/method/pose scalar/status/NEW/fallback만 RAM에 유지한다. 기존 전체 verify/statistics reader는 unused candidate/alternative packet을 RAM에서만 버리고 필요한 active pool/heldout ID와 좌표를 유지한다. 두 경로 모두 full raw 파일·seal은 수정하지 않는다. 이 방식은 메모리 사용을 줄이는 저장자료 읽기 정책이며 numerical setting 변경이나 accuracy 반복이 아니다.

STREAMING_CHECKS의 실제 두 시도는 보존한다. 첫 harness는 기존 guard의 RuntimeError를 AssertionError로 잘못 예상하여2/4FAIL이었고 기대 타입만 고친 두 번째4/4PASS였다. original first test byte와 이유 영수증은 남아 있으며 production writer는 두 시도 동일 SHA다. 별도 static doc CLI의 sys.path 실패도 numerical 실행 전에 발생했다. 이 실패들을 accuracy 실패나 성능에 의한 재시도로 합치지 않는다.

`reprojection_check --input`는 gzip 파일이 아니라 결과 디렉터리다. 새 pose의 실제 H만 최종 R,t로 대체했는지, 관측 non-H 좌표와 center·fallback·unobserved non-H native display를 보존했는지 저장 scalar 식으로 확인한다. displayed native 좌표를 sparse-only fit 관측으로 바꾸지 않는다. 모델/PnP/GT 점수화를 새로 호출하지 않는다.

## 저장 runtime의 독립 공개 검산

공식1500회 측정은1회다. 사후 runtime checker는 first A에서 known primitive의0회 key도 존재한다고 잘못 가정했고, 실패 evidence의 set 직렬화도 실패했다. [A 이유와 원 bytes](RUNTIME_VALIDATION_ATTEMPT_A/REASON.json)는 보존했으며 A의 partial JSON을 PASS 영수증으로 사용하지 않는다. checker의 absent-known-key/실패 serializer만 고친 [B protocol](RUNTIME_VALIDATION_ATTEMPT_B/RUNTIME_VALIDATION_PROTOCOL.json)을 새로freeze한 [B 결과](RUNTIME_VALIDATION_ATTEMPT_B/RUNTIME_VALIDATION_CHECKS.json)는148824check·1500행·70block·490moment·0FAIL PASS다. 자체freeze와 scalar 시도는2회(A실패/B성공)이고 detector/PnP/측정을 다시 실행하지 않았다.

현재 standalone checker로 다시 검토하려면 default DOC의A/B 영수증을 덮어쓰지 말고 새 PRIVATE 하위 output을 명시한다. `--stage`는 positional stage가 아니라 필수 옵션이다.

```bash
export THREE_HEAD_RUNTIME_REVIEW=/tmp/pallet-three-head-observation-private-20261010-v7/review_runtime_local
mkdir -p "$THREE_HEAD_RUNTIME_REVIEW"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/runtime_scalar_check.py" \
  --stage freeze --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_RUNTIME_REVIEW" \
  --protocol "$THREE_HEAD_PUBLIC_DOC/PROTOCOL.json"

python3 -I -S "$THREE_HEAD_PUBLIC_SCRIPT/runtime_scalar_check.py" \
  --stage run --input "$THREE_HEAD_PUBLIC_DOC" \
  --output "$THREE_HEAD_RUNTIME_REVIEW" \
  --protocol "$THREE_HEAD_PUBLIC_DOC/PROTOCOL.json"
```

이 검사는 original full_ms를 직접 검산한다. 원 absolute t0/t1, center/box 원값, fallback 초기 R,t는 저장되지 않아 일부 항목은 recorded flag 검토이며 독립 hardware 인증이 아니다. 저장 candidate 개수는 세지만 새 Jacobian/SVD/후단 solver를 실행하지 않는다.

## 새 별도 실제 평가

publication repo root의 `$PWD`가 absolute path라는 전제로, protocol/protection metadata는 새 in-repository 경로를 사용한다. 결과는 fixed v7 private root의 새 하위 디렉터리만 쓴다. 이미 있는 output·symlink·prior source 겹침을 우회하지 않는다. source/base/baseline/fits를 선언하는 환경 변수는 실제 사용자 홈 경로를 이 공개 문서에 복사하지 않는다.

```bash
export PALLET_SOURCE_ROOT=/absolute/path/to/original/source
export PALLET_BASELINE_ROOT=/absolute/path/to/immutable/baseline
export CORRECTED_FITS_ROOT=/absolute/path/to/completed/corrected/checkpoints
export PALLET_EXPERIMENT_PYTHON=/absolute/path/to/python-with-locked-dependencies
export THREE_HEAD_REPLAY_OUTPUT=/tmp/pallet-three-head-observation-private-20261010-v7/replay_local
export THREE_HEAD_REPLAY_METADATA="$PWD/_docs/experiments/pallet_three_head_replay_local"
mkdir -p "$THREE_HEAD_REPLAY_OUTPUT" "$THREE_HEAD_REPLAY_METADATA"
```

다음 명령은 새 directory에서 동일 설정의 실제 평가를 재현하는 형태다. `run freeze`가 새 protection metadata를 만든다. 각 단계가 PASS할 때만 다음으로 진행하며 기존 guard를 우회하지 않는다. 공개 숫자 검토에 이 GPU 실험을 실행할 필요는 없다.

```bash
THREE_HEAD_REPLAY_ARGS=(
  --source-root "$PALLET_SOURCE_ROOT"
  --baseline-root "$PALLET_BASELINE_ROOT"
  --fits "$CORRECTED_FITS_ROOT"
  --output "$THREE_HEAD_REPLAY_OUTPUT"
  --protocol "$THREE_HEAD_REPLAY_METADATA/PROTOCOL.json"
  --prior-bindings "$THREE_HEAD_REPLAY_METADATA/PROTECTION_BEFORE.json"
  --calibration-root "$THREE_HEAD_PUBLIC_DOC/source_calibration"
)

"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.run freeze "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.run preflight "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.run infer "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.evaluate preflight "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.evaluate score "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.statistics --input "$THREE_HEAD_REPLAY_OUTPUT" "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.verify --input "$THREE_HEAD_REPLAY_OUTPUT" "${THREE_HEAD_REPLAY_ARGS[@]}"
```

complete245/OBS735/geometry1960/fixed490·cleanup_error=None·같은 byte seal과 V4 ROLE hybrid preGT parity를 확인한 뒤에만 existing N3-phase reference를 읽는다. no-fit scoring canary를 유지하고 새 annotation·GT phase·난도 subset은 선택하지 않는다. 검산 자체를 freeze/run하고 싶다면 위 geometry complete 뒤, score 전에 새 validation output에서 앞 절의 두 명령을 실행한다.

preflight CLI는 서로 다른 foreign CPU workload 때문에2회, 첫 infer CLI는 자체 guard에서1회 model/PnP 전에 거절됐다. formal protocol과 세 실패 snapshot을 보존하며 기존 사용자 실험을 kill하거나 frozen guard를 우회하지 않았다. 이후 세 번째 preflight가 quiet window에서 통과했고 numerical inference는1회 실제 완료됐다. [INFERENCE_RECEIPT](INFERENCE_RECEIPT.json)의245/735/1960/490·완료/cleanup 상태와 [GEOMETRY_SEAL](GEOMETRY_SEAL.json)의 원 gzip byte·SHA가 authority다. 이 pre-model 거절 수와 실제 numerical inference1회를 분리한다. wall115.371703859초는 accuracy 실행 벽시계이며 full-path latency로 인용하지 않는다.

## 실제 전체 경로 시간

실제 [RUNTIME.json](RUNTIME.json)은 공식1500full path를1회 완료했다. 고정10method 각각20warm/130measured이며,24resource snapshot 모두 quiet·foreign CPU/GPU0이었다. 세 learned arm의 요청 head만 매번 한 번 forward하며 NativeH/no-mask 대조에는 불필요한 head/feature를 넣지 않는다. RAM BGR 입력부터 detector·initial pose·필요한N3/feature/head·hybridLOO·finalrobust/H/metadata를 full interval에 포함한다. load·decode·GT/parity·journal·resource는 timer 밖이며 이미지 I/O를 포함한 end-to-end 수치로 확대하지 않는다. 모델 hook/close·실제 OpenCV 호출 수와10경로의 mean/variance/SD/median/P90/max는 receipt와 RESULT 표에 기록했다. cache replay나 기존 stage latency 단순 합산이 아니다.

새 runtime은 GPU accuracy/scalar 작업이 끝난 별도 quiet window에서 다음 두 단계만 직렬 실행한다. report·검산·다른 성능 측정과 병렬로 실행하지 않는다.

```bash
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.runtime preflight --accuracy-output "$THREE_HEAD_REPLAY_OUTPUT" "${THREE_HEAD_REPLAY_ARGS[@]}"
"$PALLET_EXPERIMENT_PYTHON" -B -m scripts.research.pallet_three_head_observation_20261010_v7.runtime measure --accuracy-output "$THREE_HEAD_REPLAY_OUTPUT" "${THREE_HEAD_REPLAY_ARGS[@]}"
```

이전 scalar validation/scoring 중 별도 사용자의 timing job이 뒤늦게 관찰된 사건은 [POST_GEOMETRY_RESOURCE_INCIDENT.json](POST_GEOMETRY_RESOURCE_INCIDENT.json)에 숨기지 않고 기록했다. 그 후 모든 bulk 작업을 멈추고 [QUIET_RECOVERY](POST_GEOMETRY_QUIET_RECOVERY.json)를 확인했다. 이번1500회 측정은 그 이후 별도 quiet window다. 모든 실행 시간대에 다른 작업이 없었다고 주장하지 않는다.

## 보호와 공개 재현의 범위

기존 published v2–v6·source-neck/bootstrap 완료 감사·원사용자 변경·원RGB/weight·마감 실험을 보호한다. 연구 branch만 정상 commit/push하고 main/force push는 하지 않는다. remote SHA는 실제 게시 영수증으로 확인한다. 공개 숫자 재현 가능성과 physical GT/real transfer/최상위 정확도 성공은 서로 다른 판정이다. 새 실사 촬영·수동 annotation·논문/PPT/LaTeX/PDF 변경은0이다.
