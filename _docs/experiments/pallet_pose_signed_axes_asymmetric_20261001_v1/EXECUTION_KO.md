# 실제 실행과 검산기 수정 기록

**고정 2:1 과소예측 비용으로 네 모델을 실제 학습했다. 모두 수렴했지만 합성 검증은 43/45로 실패했으며 이번 실사 평가는 실행하지 않았다.** GitHub 공개·수치 검산 통과와 성능 성공은 별개다.

작업 경로는 `/home/minjae/Documents/github/pallet-pose`, Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. 실행 환경은 `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MPLCONFIGDIR=/tmp/pallet-mpl`, `MPLBACKEND=Agg`이며 `-B`로 bytecode 쓰기를 막았다. 이번 학습·평가는 캐시와 CPU 수치 연산을 사용했고 새 이미지 신경망 forward·PnP는 0회다.

## 실행 순서

```bash
MOD=scripts.research.pallet_pose_signed_axes_asymmetric_20261001_v1
python -B -m "$MOD.prefit_review" write
python -B -m "$MOD.seal_training"
python -B -m "$MOD.convex_train" all
python -B -m "$MOD.verify_train" verify
python -B -m "$MOD.evaluate_source" freeze
python -B -m "$MOD.evaluate_source" score
python -B -m "$MOD.verify_source" verify
python -B -m "$MOD.evaluate_real" not_run
```

`freeze`는 참조 오차를 읽기 전에 4×1,024행의 선택을 고정한다. `score`의 정상 종료는 성능 기준 PASS와 다르다. `not_run`은 실사를 평가하는 명령이 아니라 합성 기준 실패와 실사 산출물 12개의 부재를 기록한다. 실사 `seal_real`, `freeze`, `score`, `verify_real`의 실제 데이터 실행은 하지 않았다. 실사 검산기의 검사는 가상 배열만 사용하는 selfcheck에 한정했다.

| 단계 | 실제 결과 | 근거 |
|---|---|---|
| 사전 입력·수식 검산 | 직전 P의 9개 해시 및 이전 N의 6개 해시 유지, 독립 미분·Hessian·경계 검사 PASS | [PREFIT](PREFIT_REVIEW_KO.md) |
| 실제 학습 | zero 초기화 4회, objective 160회, 승인 반복 57회, 거절 trial 99회 | [완료 기록](TRAINING_COMPLETE.json) |
| 독립 학습 검산 | 네 모델의 gradient·Hessian·수렴·동일 새 목적식 비교 PASS | [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) |
| 합성 선택·평가 | 선택 4,096행, 결과 8,192행, 43/45·전체 FAIL | [선택 잠금](SOURCE_VAL_ROUTING_LOCK.json), [판정](SOURCE_VAL_GATE.json) |
| 수정 후 독립 합성 검산 | exit 0/PASS, 선택 4,096행 재현·T/R 오차 16,384개 일치·고정 기준 오차 8,192개 bit 일치 | [검산](SOURCE_VAL_VERIFICATION_KO.md) |
| 실사 | 현재 모델의 실사 선택·성능 평가 0회 | [미실행 기록](REAL_EVALUATION_NOT_RUN_KO.md) |

평가의 분모·기준·비용비·seed를 결과에 따라 바꾸지 않았다. 학습 budget 증가, 재시작, 가중치 이어 학습, 최적 checkpoint 선택도 하지 않았다. 이전 대칭 손실의 J와 새 비대칭 손실의 J를 직접 비교하지 않으며, 이전 가중치를 같은 새 목적식에 넣은 사후 비교만 별도로 공개한다.

## 독립 source 검산기의 형식 오류

첫 `verify_source verify`는 이전 P와 현재 선택의 비교 단계에서 `old_choices['protocol']`를 읽으려다 **KeyError로 exit 1**이 됐다. P의 choices에는 그 키가 없으며, P의 routing lock가 protocol과 choices SHA를 함께 연결한다. 이 오류는 추가한 검산기의 입력 형식 가정에 있었다. 이미 고정된 선택·오차·평가 결과에는 변경이 없다.

실패한 검산기 원본을 byte 그대로 보존하고, 존재하지 않는 키 검사만 `old_lock.protocol == P의 실제 TRAIN_PROTOCOL binding` 검사로 고쳤다. 기존 lock→gate 및 lock→choices SHA 검사를 유지했다. 학습·source 평가 코드는 바꾸지 않았고 학습이나 후보 선택·채점을 다시 실행하지 않았다. 수정 후 가상 selfcheck를 통과한 검산기만 다시 실행했다. [원본 SHA·보존 경로·정확한 수정 기록](SOURCE_VERIFICATION_SCHEMA_INCIDENT.json), [최종 독립 검산](SOURCE_VAL_VERIFICATION_KO.md)

학습 전에 수행한 가상 solver fixture에도 수정이 있었다. 새 목적식의 해를 기계 정밀도까지 요구하던 검사는 고정 gradient 기준 1e−8에서 정상 종료한 해를 잘못 거부했다. 실제 gradient/곡률에 따른 해 오차 상계로 fixture만 고쳤다. solver, tolerance, budget은 바꾸지 않았으며 이때 실제 TRAIN 읽기·fit은 0회였다.

## 공개 범위

최종 [보고서](REPORT_KO.md)는 전체 CSV·모든 seed·실패 조건·손실의 대칭/과소예측 성분·안전 개선/악화/anchor 선택을 공개한다. [파라미터 네 개](model_parameters/)는 실제 마지막 승인 checkpoint와 byte 동일하며 기존94·방향18 정규화와 고정 RBF basis를 포함한다.

실제 RGB 6장에 표시한 **110×11×130cm는 물리 Width/Height/Depth**다. 빨간 투영 자세와 T/R는 역사적 R0 결과이며 이번 모델의 실사 개선 그림이 아니다. 합성 각 행은 원래 source 치수를 유지했다. 대용량 원본 이미지·캐시 전체는 공개 묶음에 포함하지 않으므로 clone만으로 전체 학습을 재현할 수 있다는 의미는 아니다. 별도 작업 공간에서 프로토콜의 데이터·코드 SHA를 먼저 만족해야 한다.

원본 작업 공간의 Git index와 관련 없는 수정은 유지하고 공개 파일만 별도 checkout에 복사한다. 공개 검산·manifest·byte 대조 후 commit/push하며 원격 main과 모든 공개 파일의 Git blob SHA·크기를 확인한다. 완료된 namespace의 START·TRACE·checkpoint·결과를 지워 재실행하지 않는다.

[상세 결과](REPORT_KO.md) · [공개 검산](PUBLIC_REVIEW_KO.md)
