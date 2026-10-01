# 실제 실행과 재현 범위

**방향18 입력을 추가한 네 모델을 실제 학습했고, 합성 검증은 43/45로 전체 실패했다. 현재 모델의 실사 T/R 평가는 실행하지 않았다.** GitHub 공개와 검산 통과는 성능 개선 판정과 별개다.

작업 경로는 `/home/minjae/Documents/github/pallet-pose`, Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. 실행 환경은 `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MPLBACKEND=Agg`, `MPLCONFIGDIR=/tmp/pallet-mpl`이며 `-B`로 bytecode 쓰기를 막았다. source freeze 때만 `MPLCONFIGDIR`가 생략되어 Matplotlib 임시 경로 경고가 있었지만 실행은 exit 0으로 끝났다. 새 이미지 신경망 forward나 PnP는 실행하지 않았다.

아래는 실제 수행한 순서다. `MOD`는 명령의 패키지 이름을 줄인 표기다. 완료된 산출물을 지우고 재실행하는 안내가 아니다.

```bash
MOD=scripts.research.pallet_pose_signed_axes_direction_20261001_v1
python -B -m "$MOD.prefit_review" write
python -B -m "$MOD.seal_training"
python -B -m "$MOD.convex_train" all
python -B -m "$MOD.verify_train" verify
python -B -m "$MOD.evaluate_source" freeze
python -B -m "$MOD.evaluate_source" score
python -B -m "$MOD.verify_source" verify
python -B -m "$MOD.evaluate_real" not_run
python -B -m "$MOD.report"
```

| 단계 | 실제 확인 결과 | 근거 |
|---|---|---|
| 학습 전 검산 | 이전 253차원의 6개 해시 보존, 추가 방향·차분·확장 입력 3개 해시 확인 | [PREFIT](PREFIT_REVIEW.json) |
| 계약 고정 | 데이터·18개 정규화·코드·손실·solver·평가 기준을 학습 전에 고정 | [TRAIN_PROTOCOL](TRAIN_PROTOCOL.json) |
| 실제 학습 | zero 초기화 4회, objective 187회, 승인 반복 60회, 거절 trial 123회 | [완료 기록](TRAINING_COMPLETE.json), [모든 호출](TRAINING_OBJECTIVE_LOG.csv) |
| 독립 TRAIN 검산 | 4개 모델의 gradient·Hessian·수렴 인증 PASS, 동일 목적식에서 직전253 대비 손실 감소 | [검산](TRAIN_CONVERGENCE_KO.md) |
| 합성 선택·평가 | 선택 4,096행을 정답 채점 전에 고정, 전체 결과 8,192행, 조건 43/45 | [선택 잠금](SOURCE_VAL_ROUTING_LOCK.json), [전체 CSV](SOURCE_VAL_FRAME_RESULTS.csv) |
| 독립 합성 검산 | 선택 4,096행 재현, T/R 오차 16,384개 일치, 이전 고정 기준 모델 오차 8,192개 bit 일치 | [검산](SOURCE_VAL_VERIFICATION_KO.md) |
| 실사 | 현재 learned 선택·성능 평가 0회, 관련 산출물 12개의 부재 확인 | [미실행 기록](REAL_EVALUATION_NOT_RUN_KO.md) |
| 공개 이미지 | 그래프 3장과 역사적 R0 입력 RGB 6장을 담은 패널 3장 | [상세 보고서](REPORT_KO.md) |

`freeze`는 참조 오차를 읽기 전에 선택을 고정한다. `score`의 정상 종료는 성능 기준 PASS와 다르다. `not_run`은 실사 평가가 아니라 합성 실패로 실사 평가를 하지 않았음을 기록하는 명령이다. `seal_real`, 실사 `freeze`·`score`는 실행하지 않았다. 한 모델·시드만 선택하거나 실패 행을 제거하지 않았다.

그림의 110×11×130cm는 물리 Width/Height/Depth다. 실제 입력 사진에 겹친 빨간 자세와 T/R는 과거 R0 결과이며 현재271 모델의 실사 성능이 아니다. source 각 행의 실제 치수는 source 입력 계보를 따른다. 모든 데이터를 동일한 110×11×130cm로 바꾸지 않았다.

[공개 최종 파라미터](model_parameters/)에는 모델별 271×2 계수, 기존94 정규화, 고정 RBF basis와 추가18 정규화를 포함한다. 독립 검산이 실제 checkpoint와 값을 대조한다. 원본 대용량 이미지·캐시 전체는 공개 묶음에 포함하지 않으므로 GitHub clone만으로 전체 학습을 재현할 수 있다는 뜻은 아니다. 같은 로컬 데이터 계보를 갖춘 별도 작업 공간에서 입력·코드 SHA를 먼저 만족해야 한다.

원본 작업 공간의 Git index와 관련 없는 수정은 유지한다. 공개 파일만 별도 checkout에 복사하고 manifest의 SHA·크기를 확인한 뒤 commit/push한다. push 이후 GitHub main의 commit과 모든 공개 파일의 Git blob SHA·크기를 대조한다. 공개 검산과 원격 게시 여부를 성능 성공으로 바꾸지 않는다.

## 고정 선택 진단의 실행 중단과 보존

학습·합성 평가가 끝난 뒤 `source_transfer_diagnostic seal`·`analyze`로 현재와 직전의 고정 선택만 비교했다. 출력 이름을 정리하기 전에 만든 `SOURCE_TRANSFER_DIAGNOSTIC_PROTOCOL`은 계산을 실행하지 않은 상태로 보존했고, 새 `SOURCE_FIXED_CHOICE_DIAGNOSTIC_PROTOCOL`이 이름 변경만을 기록한다. 학습·평가 결과에 따른 계산 범위 변경은 없다.

`analyze`는 모든 계산과 JSON·한국어 문서 저장을 마친 뒤, 마지막 출력 로그에서 자신의 결과 JSON 해시를 읽다가 접근 검사에 걸려 **exit 1**로 끝났다. 봉인한 코드와 완성된 두 결과는 수정하거나 재실행하지 않았다. 별도 검산 프로세스는 고정 선택·오차로 분류, 전환, 선택 변경, 행별 개선·악화, 중앙값·P90 차이의 **178개 비교를 확인하고 exit 0/PASS**로 종료했다. 기록의 `complete/PASS`는 보존된 진단 계산의 검산 결과이며 원래 생산 프로세스가 exit 0이었다는 뜻이 아니다. [실제 실행·오류·검산 기록](SOURCE_FIXED_CHOICE_DIAGNOSTIC_EXECUTION.json)

추가 진단을 보고서에 연결할 때는 `report --refresh-text`만 사용했다. 기존 수치 표·CSV·네 모델 파라미터·여섯 그림의 byte를 유지하고 보고서 코드, 두 한국어 본문과 연결 영수증만 갱신했다.

[상세 결과](REPORT_KO.md) · [고정 선택 실패 진단](SOURCE_FIXED_CHOICE_DIAGNOSTIC_KO.md) · [공개 파일 검산](PUBLIC_REVIEW_KO.md)
