# 재현·재개와 공개 범위

이번 실행의 결과는 개발에 반복 노출된 PRIMARY192, 원 TEST49, 추가 REMAINDER143, 별도 CAL46을 같은 고정 physical C2로 집계한다. 학습용1003장은 source별 원 대칭(C2 502장, C1 501장)을 유지하고 validation235장은 별도 C1 asset이다. 공개 숫자 파일만으로 물리 owner·pose의 독립 검증이 가능해지는 것은 아니다.

## 실제 경로와 환경

실행 작업트리는 `$SOURCE_ROOT`, private run은 `$RUN_PRIVATE`이다. 원 사용자 작업트리 `$PRIVATE_HOME/Documents/github/pallet-pose`의 기존 수정·모델·논문은 보호한다. 최종 Git 원격 SHA는 root의 게시 확인 기록이 권위이며 이 문서가 push를 수행하지 않는다.

실제 Python은 `$PRIVATE_HOME/anaconda3/envs/pallet-pose/bin/python`이다. 이번 실행에서 확인한 주요 버전은 NumPy1.26, SciPy1.12, OpenCV4.9, PyTorch2.1 계열이다. 정확한 환경/모델 SHA는 SOURCE_BINDINGS, mask protocol, 원 실행 receipt를 따른다. 전체 solver worker 최대4, BLAS/OpenMP1thread, 고정CPU head 학습 최대6회와 승인된 자원 예산을 유지한다.

```bash
cd $SOURCE_ROOT
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PALLET_PY=$PRIVATE_HOME/anaconda3/envs/pallet-pose/bin/python
PALLET_RUN=$RUN_PRIVATE
```

게시 과정에서 로컬 경로는 `$RUN_PRIVATE`, `$OLD_STATE_PRIVATE`, `$SOURCE_ROOT`, `$PRIVATE_HOME` 등으로 바뀔 수 있다. **공개 CONFIG/PLAN/receipt의 별칭은 제시용이다. byte-exact 재개는 경로를 다시 치환한 공개 JSON이 아니라 보존된 private 원본을 사용한다.** private 원본 SHA와 공개 파생물 SHA의 대응은 PUBLICATION_MANIFEST와 private PUBLICATION_RECEIPT에 남긴다. 수치·영상 ID·원 private 파일은 변경하지 않는다.

## 현재 실행을 중복 계산 없이 재개

```bash
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.campaign run --config "$PALLET_RUN/RUNTIME_CONFIG_V3.json"
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.campaign resume --run-dir "$PALLET_RUN"
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.campaign report --run-dir "$PALLET_RUN"
```

첫 명령은 실제 존재한 runtimeV3 인터페이스의 예시다. `resume`은 CONTROLLER_STATE에 현재 연결된 runtime 경로를 사용하므로 이후 root가 봉인/최종 단계의 command hook만 연결한 버전도 따른다. 수식·seed·임계값을 바꾸는 파일은 CONFIG_FROZEN이며 runtime command/경로와 구분한다. worker scheduling amendment는 pilot/real 종료 뒤 확장을4worker로 실행하는 자원 배치 변경이다. 입력·42 local attempts/frame·동작 중 batch lock은 바꾸지 않는다.

제어기는 현재 live PID/start-ticks/argv와 proof hash를 검증하며 기존 작업을 다시 시작하거나 임의로 종료하지 않는다. 유효한 BATCH_SEAL/FRAME_SEAL이 있으면 그 결과를 재사용한다. 완료 디렉터리 존재만으로 DONE을 기록하지 않는다. 독립 branch 하나가 입력 대기여도 다른 branch를 계속한다. source/hash가 변하면 기존 파일을 덮어쓰지 않고 기술 오류를 진단한다.

root의 사전 정해진 **후봉인 GO**는 새 사용자 허가 요청이 아니다. 새 모델·출력을 모두 봉인한 뒤 평가자가 고정 참조를 읽을 수 있게 하는 프로세스 경계다. root가 아직 이 GO를 주지 않은 run에서 `--scoring-GO`, `--checking-GO`, `--postseal-GO`, `--final-GO`를 임의로 붙이지 않는다.

## 입력과 결과 해시 확인

PLAN_LOCK은 원 source/config/arm registry, synthetic split·none-prior와 자원 범위를 고정한다. 실제 수정 입력은 `inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz`이고 INPUT_CONTRACT_REPAIR와 독립 INPUT_COVERAGE_INDEPENDENT가 원래 가용 R/t 역할29,564개, positive-depth 진입29,556개, archive 보존8개를 연결한다. 회전 자체가 기록되지 않은 outcome은 복원했다고 주장하지 않는다.

`real_batch_v1`의52개 sealed frame은 누락된 가용 후보 입력을 발견하기 전의 중단 시도다. 이 시도를 최종 arm으로 사용하지 않고 지우지도 않는다. 최종 real batch는 `real_batch_all_available_v1`이다. feature bridge는 수치 feature·pose를 바꾸지 않고 dedup 과정에서 겹친 N0/E0 source 역할과 원래 eligibility를 복원한다. source symmetry prose correction은 원 label·GT·error 함수·CONFIG bytes를 바꾸지 않는다.

최종 PREDICTION_SEAL은 모든238개 출력과 모든 실제 실패를 먼저 봉인한다. EVALUATION_SEAL은 이를 참조 해시와 연결한다. independent_v1/INDEPENDENT_CHECK는 공통 evaluator를 import하지 않고 저장 pose·분포·paired·단계별 숫자를 독립 계산한다. REPORT_SEAL과 PUBLICATION_MANIFEST는 그 뒤의 설명·공개 파생물 경계다.

## 고정 cache512를 유지한1003 확장

제어기가 아래 단계를 이미 수행했다면 다시 실행하지 않는다. 외부 재현에서는 **새 private namespace**를 쓰고 해당 run의 봉인된 원 입력·config를 연결한다.

```bash
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.extend_batch \
  --pilot-inputs "$PALLET_RUN/synthetic_v1/PILOT512_ROOT_FRAMES_AUTO_MASK.jsonl.gz" \
  --expanded-inputs "$PALLET_RUN/synthetic_v1/EXPANSION1003_ROOT_FRAMES_AUTO_MASK.jsonl.gz" \
  --expanded-input-seal "$PALLET_RUN/synthetic_v1/EXPANSION1003_ROOT_FRAMES_AUTO_MASK_SEAL.json" \
  --config "$PALLET_RUN/CONFIG_FROZEN.json" \
  --pilot-batch "$PALLET_RUN/synthetic_pilot_batch_v1" \
  --output "$PALLET_RUN/synthetic_expansion_batch_v1"
```

전체 canonical input row SHA·frame seal·config·5개 batch core SHA가 같을 때만 pilot512의 실제 RESULT/BUNDLE/SEAL을 해당1003 index로 hardlink한다. ID만 같은 것을 근거로 재사용하지 않는다. 남은491개만 실제 solve하고, 최종1003 per-frame CPU 장부에는 pilot512 비용이 포함되므로 pilot CPU를 다시 더하지 않는다.

## 라벨·학습·추론 경계

synthetic_data의 select/provision/packets/frames, synthetic_masks, 고정 batch, feature_bridge를 거친 후에만 learned_head의 다음 세 subcommand를 순서대로 사용한다. 각 입력 경로/해시는 실제 runtime config와 원 seal을 따른다.

```bash
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head prepare-labels --help
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head train --help
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head predict --help
```

synthetic RGB·원 upstream cache·source별 target 계약을 검증하며 renderer GT mask/visibility를 feature로 사용하지 않는다. 예측·기하 B/I/S/T와 고정 MobileSAM 자동 mask를 사용하지만 별도 synthetic external-O classifier는 실행하지 않았다. real O/T/U coverage와 동등한 synthetic 감독이라고 주장하지 않는다. 실사 CAL/PRIMARY의 참조는 새 head 라벨·normalize·학습·선택에 들어가지 않는다.

선형과 작은 head는 같은62개 특징·동일 bank/eligibility·none fallback N0를 사용한다. seed20261018/19/20 각각1회씩, 총6회만 등록했다. 대표는 첫 seed20261018이며 PRIMARY 최고 seed를 고르지 않는다. 검증 epoch 선택과 새로운 구조/seed 탐색을 혼동하지 않는다. 고정 label-support gate가 실패하면 TRAINING_NOT_RUN을 남기고6개 행 모두 N/A로 보고한다. 학습 비용·step·노출수·best/last SHA는 TRAINING_MANIFEST가 권위다.

학습을 실행한 경우 `learned_synthetic_v1`의 고정6개 예측을 먼저 봉인하고 `synthetic_head_evaluation distributions`가 추가6,018 tiny-head forward로 전체 log-softmax만 보존한다. `evaluate`는 이 두 봉인 뒤에만 원 합성 target을 읽어 train/validation 공동성공·보류·oracle·전체/good/no-good CE를 집계한다. 이는 추가 학습·새 모델 선택이 아니다. 작은 SYNTHETIC_LEARNED_EVALUATION.json/KO.md와 실제 SYNTHETIC_HEAD_EVALUATION_SEAL을 보고서에 연결하며 학습 N/A라면 결과도 없는 상태로 유지한다.

## 봉인 뒤 평가·독립검산·그림·최종 게시 준비

자동 파이프라인의 실제 root hook이 이 명령들을 연결한다. 여기서는 재현 가능한 인터페이스만 적으며 실행하지 않은 성공 로그를 만들지 않는다.

```bash
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.evaluator --help
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.independent_check --help
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.gallery --help
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.mechanism_diagnosis --help
"$PALLET_PY" -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.finalize \
  --run-dir "$PALLET_RUN" \
  --independent-receipt "$PALLET_RUN/independent_v2/INDEPENDENT_CHECK.json" \
  --gallery-receipt "$PALLET_RUN/gallery_v3/GALLERY_RECEIPT.json" \
  --mechanism-dir "$PALLET_RUN/mechanism_v1" \
  --output-dir _docs/experiments/pallet_statewise_end_to_end_recovery_v1 \
  --final-GO
```

finalize는 현재 독립PASS와 gallery receipt를 확인하고 report→작은 공개 파생물을 생성한다. 유효한 REPORT_SEAL/PUBLICATION_RECEIPT/FINALIZATION_RECEIPT는 검증해 재사용하고 계산을 반복하지 않는다. git commit/push는 하지 않으며 root가 보호 대상·차이·독립검산을 확인한 뒤 게시한다.

최종 검산 receipt는 독립v2다. v1의86개 strict-zero 방향 계수 차이는5영상·37개 pair/metric 부동소수점 부호 차이로 설명하고 원 evaluator 계수를 그대로 재현했다. stable atan2/fsum의 성공·오차 크기·분포·CI를 바꾸지 않았으며 원 v1 FAIL·상세·소스와 ROUNDING_DIRECTION_AUDIT를 보존한다. 산술PASS는 유한 projected-edge support나 실제 물리 owner의 독립 인증을 뜻하지 않는다. B/edge 극단 출력과 양성확률질량 부록도 새 후보/선택기를 만들지 않은 사후 진단으로 구분한다.

최종 private `gallery_v3/index.html`은 실제RGB의 추론→상태→실제 관측/분포→실제 선택→모델 투영/참조를 비교한다. 최종 공개 `metrics_gallery_v3/index.html`은 전체238개 수치표·위치cm 대 회전° 산점도이며 RGB를 포함하지 않는다. 이전 gallery_v1/v2와 metrics_gallery/metrics_gallery_v2는 보존한다. 표시 수정은 sentinel[-1,-1]의 ROI 확대, 미사용 분포/unknown head의 의미 표기와 긴 텍스트 배치만 다루며 원 관측·pose·metric을 바꾸지 않는다. renderer receipt는 출력한 증거이며 사람이 이미지를 직접 확인한 증거는 별도 검사 기록을 참조한다.

## 공개 파일만으로 가능한 확인과 private 의존

```bash
"$PALLET_PY" -B -m unittest \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.test_owner_model \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.test_boundary_model \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.test_learned_head \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.test_evaluator \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.test_campaign_and_extension \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.test_report_and_publish
```

새 수식·mask/RGB 구간·Jacobian·full/partial·S 제외·후보 순서/결측·firewall·cache 재사용·게시 안전성 fixture를 확인한다. solver의 실제 CAL fixture는 private cache가 없는 환경에서 skip하는 범위를 명시하고 그 skip을 실제 데이터 검산PASS로 바꾸지 않는다. 기존 큰 cache가 필요한 검사는 공개 파일만으로 완전히 재현할 수 없다.

COMPARISON_ALL_ARMS/ROW_METRICS/PAIRED_TRANSITIONS/RESULTS/단계 진단의 작은 공개 수치와 해시를 배포한다. 원RGB, 전체 후보pose bank, checkpoint, 원 참조 파일, upstream 실행cache는 private에 둔다. 공개 artifact의 완전성과 private 데이터의 입수 가능성을 구분한다. 과거 실패·중단·source 수정·미확정 physical owner를 성공한 것으로 바꾸지 않는다.
