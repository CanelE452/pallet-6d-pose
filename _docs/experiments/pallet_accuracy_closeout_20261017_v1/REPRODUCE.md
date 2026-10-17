# 재현 절차와 실패 보존

이 절차는 기존 RGB·가중치·neural cache·봉인 pose·참조가 로컬에 존재하는 연구 환경용이다. 공개 저장소에 대형 원자료나 가중치를 새로 올리지 않았다. 결과가 좋은 설정을 고르거나 학습/threshold/weight 탐색을 실행하는 절차가 아니다. 원본 사용자 checkout을 수정하지 말고 이 연구 커밋의 별도 checkout에서 실행한다. 기존 출력은 exclusive-create로 보존하므로 재현 결과를 기존 시도 위에 덮어쓰지 않는다.

## 환경과 고정 입력

실행 환경은 `/path/to/existing/pallet-yolo26/python`, Python3.10, NumPy와 기존 OpenCV이며 BLAS/OpenCV CPU thread는1이다. 새 패키지 설치·학습·NN forward는 필요하지 않았다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PYTHON=/path/to/existing/pallet-yolo26/python
TASK_PRIVATE=/absolute/path/to/new_accuracy_closeout_reproduction
TASK_BASELINE=/path/to/existing/STATE_CACHE/score_v2
mkdir "$TASK_PRIVATE"
```

`TASK_PRIVATE`는 사용자가 재현을 위해 지정한 **새 디렉터리**다. 이미 존재하는 실험 디렉터리를 삭제하여 다시 사용하는 방식은 허용하지 않는다. 기존 실행은 `pallet-accuracy-closeout-20261017-v1`에 보존돼 있다.

핵심 입력은 checkpoint SHA `b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581`, `seed1_REAL_DEV.npz` SHA `683b7395860865f7e10cba5e1dfb7c04e84234dd296340416c4759b671c95f06`, 원 baseline pose SHA `372a021ef4ebced48428ffdd3f685fdf50bdd42ede2ed57da423f710a8018934`, 고정 참조 SHA `d488564817a8cac8d0103350f03ac61d78bdeb56163c6a5a4daffe414f0f2af1`다. `SOURCE_AND_INPUT_BINDINGS.json`과 protocol/seal이 실제 소스·RGB·입력·출력을 연결한다.

## 1. 원결과 hard gate와 평가 프로토콜 고정

```bash
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.baseline_audit \
  --sealed-baseline-folder "$TASK_BASELINE" --output "$TASK_PRIVATE/baseline_audit_v1"
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation \
  --write-protocol "$TASK_PRIVATE/EVALUATOR_PROTOCOL_LOCK.json"
```

첫 gate가 R0/N0/E0/PF_RAW=66/108/114/80, E0−N0 교차89/25/19/59를 모두 확인해야 뒤 단계로 진행한다. 모든 원 row metric도 재계산하며 반회전C2 이외90°는 허용하지 않는다. bootstrap은 기존 recording cluster, seed20260917,10,000회로 고정되며 모델 seed를 만들지 않는다.

## 2. 두 PoseFix 고정 정의 복원·봉인

```bash
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay prepare \
  --repo "$PWD" --private "$TASK_PRIVATE"
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay infer95 \
  --repo "$PWD" --private "$TASK_PRIVATE"
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay parity95 \
  --repo "$PWD" --private "$TASK_PRIVATE"
```

95장×5경로의 점·최종 pose를 원 봉인 출력과 직접 비교한다. 원 실행에서는475/475 점 배열·475/475 전체 pose dictionary가 정확히 같았고 최대 절대 차이는0이었다. 허용차1e-10이며 parity가 어긋나면192장으로 진행하지 않는다. 후처리 selector는 참조를 읽지 않는다.

```bash
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay extend192 \
  --repo "$PWD" --private "$TASK_PRIVATE"
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay parity143 \
  --repo "$PWD" --private "$TASK_PRIVATE"
```

238장×세 PF 경로를 한 번 실행하고 PRIMARY192와 CAL46 gzip도 별도로 봉인한다. 두 cap은 같은 native SubPix 결과에서 함께 만들어지며 결과에 따른 winner 선정이 없다. 추가143의 `raw_points`는 N3+SubPix이므로 YOLO anchor로 쓰지 않는다. 원 prediction-only YOLO cache를 사용한다. 캐시가 이미319ID를 포함하여 신규 forward는0이다.

## 3. 별도 정확도 프로세스

```bash
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation \
  --baseline-folder "$TASK_BASELINE" \
  --predictions "$TASK_PRIVATE/posefix192_v1/PREDICTIONS.jsonl.gz" \
  --prediction-seal "$TASK_PRIVATE/posefix192_v1/PREDICTION_SEAL.json" \
  --protocol-lock "$TASK_PRIVATE/EVALUATOR_PROTOCOL_LOCK.json" \
  --output "$TASK_PRIVATE/fair_evaluation_v1"
```

이 evaluator는 predictor seal과 모든238ID를 먼저 확인한 뒤 numeric 참조를 해독한다. 원 prediction-order가 달라도 exact immutable ID로만 결합하고 점·pose를 바꾸지 않는다. 원 실행에서 이 순서 차이가 조기에 검출되어 scientific rule을 바꾸지 않는 ID join을 추가했다. 기존 lock과 기술 수정 lock 및 실패 receipt는 전부 보존했다. 현재 소스는 처음부터 이 결합을 지원하므로 새 재현 lock 하나로 실행된다.

예상 PRIMARY192 성공 수는 R0=66/N0=108/E0=114/PF RAW=80/PF-anchor=112/YOLO-anchor=111이다. COMMON95는 PF RAW66/PF-anchor71/YOLO-anchor70/N074/E086이며 CAL46은 별도다. 새 통계의 유불리에 따라 ID·분모·참조·좌표·대칭을 수정하지 않는다. `FAIR_RESULTS.json`에 원49/추가143·세션·분포·status unknown·paired 공동실패 tail까지 보존한다.

## 4. E0 정보 계약 감사

원 typed inventory와 trace는 기존 봉인 자료이며 새 candidate generation이나 score 최적화가 아니다. 실제 경로는 `E0_INFORMATION_CONTRACT_AUDIT_KO.md`와 감사 입력 binding을 확인한다.

```bash
TASK_TYPED_INVENTORY="$TASK_BASELINE/../e_trace_v2/TYPED_PACKETS.jsonl.gz"
TASK_LINEAGE_TRACE="$TASK_BASELINE/../e_trace_v2/E_TRACE.jsonl.gz"
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.e_contract_audit \
  --repo "$PWD" --typed "$TASK_TYPED_INVENTORY" --trace "$TASK_LINEAGE_TRACE" \
  --output "$TASK_PRIVATE/e_contract_audit_v1"
```

`TASK_TYPED_INVENTORY`와 `TASK_LINEAGE_TRACE`는 감사 문서에 기록된 기존 source files다. 238장1,252관측의 단위·원 covariance·fit override·source·native owner·partial revocation·Q를 검사한다. 이 감사는 pose solve·새 후보 채점·교정 fit을 실행하지 않는다. 확인된 우발적 구현 결함이 없고 새 출처의 covariance 재적용이 정당화되지 않아 `NO_SUPPORTED_CHANGE`로 종료했다. 따라서 새 E 수정안192 실행 명령은 없다.

## 5. 검사와 게시 증거

```bash
$TASK_PYTHON -B -m unittest \
  scripts.research.pallet_accuracy_closeout_20261017_v1.test_posefix_replay \
  scripts.research.pallet_accuracy_closeout_20261017_v1.test_evaluation -v
$TASK_PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.independent_check \
  --task-private "$TASK_PRIVATE" --baseline-folder "$TASK_BASELINE" \
  --output "$TASK_PRIVATE/INDEPENDENT_ARITHMETIC.json"
```

독립 수치 재집계는 주 evaluator의 metric/aggregate 함수를 import하지 않고 원 saved poses+참조에서 수행한다. 최종 코드와 실제 실행 명령·대조 범위는 `VALIDATION.json` 및 해당 독립 checker에 보존한다. 테스트가 참조를 읽지 않는 predictor의 I/O probe, 두 anchor의 구분, native center/unsupported slot 보존, 동일 pose의 joint metric·C2·missing policy·cluster weighting을 확인한다.

소스와 보고서만 신규 namespace에서 commit한다. 원본 user dirty 파일을 stash/reset/clean하거나 기존 branch를 강제로 덮어쓰지 않는다. remote branch/main을 fast-forward로 게시한 후 `git ls-remote`와 remote commit 파일 SHA를 다시 읽어 실제 게시물을 확인한다. 운영 기본값·모델·원 예측·논문은 변경하지 않는다.
