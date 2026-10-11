고정된 N3·YOLO를 이용한 B1/B2와 여덟 선택기 C의 실제 실행 명령, 입력 계약, 검산 경로를 기록한다.

[확인] C는 TRAIN4096/VAL1024/TEST1024를 사용했고, REAL173장은 seed1·2·3을 모두 평가했다. C의 OpenCV 호출은 30,748회, B1은 3,114회다. 여덟 분류기는 각각 한 번만 적합했다. [검산](C_B1_VERIFICATION.json), [계산 기록](C_B1_COMPUTE_RECEIPT.json), [실행 코드 연결](C_B1_EXECUTION_SOURCE_VERIFICATION.json)을 함께 확인한다.

[확인] 다음 단계 명령은 이 실험에서 실제 실행했다. 공개 문서에서는 로컬 절대 경로를 사용자가 지정하는 환경 변수로 대체했다. `PALLET_PYTHON`은 CPU 환경의 Python, `PALLET_GPU_PYTHON`은 GPU 환경의 Python이다. 원본 데이터·모델·결과를 읽기만 하고 새 작업 폴더와 private 폴더에 썼다. 게시된 산출물과 같은 경로에서 전체 명령을 다시 실행하면 기존 파일 보존 게이트가 막는다. 새 실행은 출력이 비어 있는 작업 복사본과 별도 private 폴더에서 입력 감사·PLAN 및 단계별 소스 잠금을 먼저 만들어야 한다. 기존 잠금 해시를 임의로 덮어써서 재실행하지 않는다.

```bash
cd "$PALLET_NEW_ROOT"
# PALLET_SOURCE_ROOT, PALLET_PUBLIC_ROOT, PALLET_FAST_PRIVATE는
# 소유자의 기존 원본/공개 기준/새 비공개 출력 위치로 미리 지정한다.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/contract_audit.py \
  --source-root "$PALLET_SOURCE_ROOT" --public-root "$PALLET_PUBLIC_ROOT" \
  --private-output "$PALLET_FAST_PRIVATE" \
  --output _docs/experiments/pallet_wd_info_fast_20261011

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py lock

"$PALLET_GPU_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py extract

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py geometry

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py labels

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py fit

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py evaluate

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/classifier.py synthetic-evaluate

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/dimension_intervention.py

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/verify_bc.py

"$PALLET_PYTHON" -B \
  scripts/research/pallet_wd_info_fast_20261011/ablation_reuse.py
```

[확인] 실제 단계별 소스는 다음처럼 잠겼다. 과거 잠금은 실행 순서를 보존하므로 최신 파일 해시와 모두 같지 않다. 학습 후 모델이나 선택식은 바꾸지 않았다.

| 단계 | 소스 SHA256 | 기록 |
|---|---|---|
| C 특징·확률 입력 추출 | `f6ce5015cf6932fa76b35c8beacd4fde1e1dd4e17cd94229ea2c1e8c495f6263` | [첫 잠금](C_CODE_LOCK.json) |
| C 기하 특징 계산 | `54db1fa92137cc71015ef12c319c334ffe5f7f2a250f8772cfd86a8652ffaf8f` | [집계 수정 전후 연결](C_REPORTING_PREEXEC_AMENDMENT.json), [잠금](C_CODE_FINAL_LOCK.json) |
| C 라벨·여덟 적합·REAL·합성 TEST | `c706a276928a649b10a1b46a2dd61e4b45d55d2277fa669232243a4fa5a1213d` | [I/O 잠금](C_IO_PREFIT_LOCK.json), [수정 기록](C_IO_PREFIT_AMENDMENT.json), [전체 차이](C_IO_PREFIT_SOURCE_DIFF.patch) |
| B1 반사실 치수 | B1_CODE_LOCK의 SHA256 | [잠금](B1_CODE_LOCK.json) |

[확인] 라벨 단계의 처음 실행은 NPZ를 영상마다 다시 압축 해제하는 버그 때문에 중단했다. 여덟 적합을 시작하기 전에 배열을 한 번 읽도록 고쳤다. 새·구 소스는 로더와 잠금 파일명만 정규화하면 전체 AST가 동일하다. 값·라벨식·분할·목적함수·설정은 유지했다. 재적합과 성능을 보고 하는 설정 변경은 없었다.

[확인] 기존 캐시에 94개 기하 특징에 필요한 키포인트 confidence 9개가 없었다. [사전 정정](C_PREEXEC_AMENDMENT.json) 후 같은 YOLO를 합성 영상마다 한 번씩만 실행해 confidence를 별도 저장했다. 처음 8장은 전체 6,144장에 포함되어 재추론하지 않았다. 좌표·상자·FP16 P3/P4 최대 절대차는 모두 0이었다. V와 N3 입력에는 기존 P3/P4 캐시를 계속 썼다. 새 REAL·DOPE·ResNet 영상 특징 추출은 0회다.

[확인] [B1 검산](B1_PREDICTION_SEAL.json)은 실제 222개 전체 logits를 복원해 기존 N3 좌표와 최대 1e-4 px 이내에서 일치시켰다. 그 뒤에만 참조를 읽었다. Stage1의 canonical 치수 입력만 바꾸었고, PnP에는 매번 올바른 canonical 치수를 넣었다. TRAIN 평균 치수와 정규화 0은 서로 달랐다. seed1 외의 새 반사실 조건은 실행하지 않았다.

[확인] C 모델·확률 배열과 원본 RGB는 공개하지 않는다. 소유자의 기존 로컬 데이터가 있어야 전체 계산을 재실행할 수 있다. private 예측 seal은 `C_SYNTH_PREDICTIONS_SEALED.npz`, `C_SYNTH_PROBABILITIES_SEALED.npz`, `C_REAL_PROBABILITIES_SEALED.npz`, `B1_PREDICTIONS_SEALED.json`이다. 공개 원행은 [C REAL](C_FRAME_METRICS.jsonl.gz), [C 합성 TEST](C_SYNTH_TEST_ROWS.jsonl.gz), [B1](B1_FRAME_ROWS.jsonl.gz)이다.

[확인] C의 후보 입력은 이름에 따라 long·short 순서로 고정한다. 후보 컨테이너의 열거 순서를 바꾸어도 직렬화된 입력·물리 선택은 동일했다. 이름 없이 G의 두 94개 슬롯을 교환하면 G 모델의 선택이 실제로 바뀌므로, 임의 슬롯 교환에 대한 구조적 불변성은 주장하지 않는다. 확률·정답을 임의로 반전해 통과시키지 않았다.

[확인] 정사각형 C4의 90°는 동등하고 직사각형 C2의 90°는 동등하지 않다. 모두 proper rotation이며 중심 8을 고정한다. 기존 합성 C1의 물리 회전 roundtrip 오류는 그대로 남겼으므로 합성 6D 수치를 검출 잡음의 식별성 상한으로 해석하지 않는다. 합성 TEST는 기존 모델이 이미 사용한 개발 자료이며, D TRAIN이 C VAL253·TEST271장과 겹쳐 통합된 독립 TEST라고 부를 수 없다.

[확인] [기존 N0/N1/N2/N3 짝 절제 재집계](B1_ABLATION_REUSE.json)는 허용173 ID의 per-ID 기존 logits/decoded points 2,076개만 읽었다. 다른 학습 checkpoint의 여섯 대조와 모델별 세 seed 쌍, Base 이동량을 동일 좌표계에서 실제 계산했다. 동일 checkpoint의 치수 반사실 결과도 같은 표에 분리했다. protected146의 per-ID 캐시와 전체319 GT/예측 파일은 읽지 않았으며 새 fit·forward·solve는 0회다. 과거319장 집계와 모집단·집계가 달라 사용자 주장 수치의 직접 재현으로 부르지 않는다.
