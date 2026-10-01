# 실제 실행 순서와 재현 범위

이 기록은 실행한 명령과 산출물을 연결한다. **학습 네 번과 source 평가를 실제 실행했으며, source 43/45·전체 FAIL 이후 실사 평가는 실행하지 않았다.** 공개 파일 검산이나 GitHub push는 성능 개선 판정과 별개다.

작업 경로는 `/home/minjae/Documents/github/pallet-pose`이고 Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. Python 3.10.20, NumPy 1.26.4, SciPy 1.12.0, Torch 2.1.1+cu118, Matplotlib 3.10.9, Pillow 12.2.0을 사용했다. 실행 환경은 `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MPLCONFIGDIR=/tmp/pallet-mpl`, `MPLBACKEND=Agg`이며 `-B`로 bytecode 쓰기를 막았다. 고정 특징의 수치 최적화와 검산 단계에서 새 이미지 신경망 forward는 실행하지 않았다.

다음 순서의 각 명령이 exit code 0으로 종료된 것을 확인했다. `MOD`는 아래의 실제 패키지 이름을 줄인 표기다.

```bash
MOD=scripts.research.pallet_pose_signed_axes_sign_20261001_v1
python -B -m "$MOD.evaluate_real" self_check
python -B -m "$MOD.prefit_review" write
python -B -m "$MOD.seal_training"
python -B -m "$MOD.convex_train" all
python -B -m "$MOD.verify_train" verify
python -B -m "$MOD.evaluate_source" freeze
python -B -m "$MOD.evaluate_source" score
python -B -m "$MOD.verify_source" verify
python -B -m "$MOD.evaluate_real" not_run
```

`self_check`는 가상 배열만 검사한다. `freeze`는 source 참조 오차를 읽기 전에 4개 모델×1,024행 선택을 고정한다. `score`는 이를 채점하며, 프로세스의 정상 종료와 성능 gate PASS는 다르다. `not_run`은 실사 평가를 실행하는 명령이 아니라 source 실패와 실사 산출물 9개의 부재를 기록하는 명령이다. `seal_real`, 실사 `freeze`·`score`는 실행하지 않았다.

| 단계 | 실제 확인 결과 | 근거 |
|---|---|---|
| 입력·타깃·수식 검산 | 이전 Newton과 모델별 6개 데이터 해시 동일 | [PREFIT](PREFIT_REVIEW.json) |
| 실제 학습 | zero 초기화 4회, 185호출, 승인 60반복, 거부 trial 121회 | [완료 기록](TRAINING_COMPLETE.json), [objective CSV](TRAINING_OBJECTIVE_LOG.csv) |
| 독립 TRAIN 검산 | 네 모델의 gradient·Hessian·수렴 인증 PASS | [독립 검산](TRAIN_CONVERGENCE_KO.md) |
| source 선택·평가 | 선택 4,096행, 전체 결과 8,192행, 조건 43/45 | [선택 잠금](SOURCE_VAL_ROUTING_LOCK.json), [평가 CSV](SOURCE_VAL_FRAME_RESULTS.csv) |
| 독립 source 검산 | 오차 16,384개와 기준 모델 오차 8,192개 대조 | [독립 검산](SOURCE_VAL_VERIFICATION_KO.md) |
| 실사 | 현재 learned 선택·성능 평가 0회 | [미실행 확인](REAL_EVALUATION_NOT_RUN_KO.md) |

공개한 [네 모델 파라미터](model_parameters/)는 실제 최종 checkpoint와 byte 단위로 동일하며 고정 RBF basis와 정규화도 포함한다. 실행 코드, 해시, 전체 요약·CSV와 이미지도 공개한다. 대용량 원본 이미지·학습 캐시 전체는 이 결과 묶음에 포함되지 않으므로 **GitHub clone만으로 전체 학습을 재현할 수 있다는 뜻은 아니다.** 같은 로컬 데이터 계보를 갖춘 환경에서 [학습 프로토콜](TRAIN_PROTOCOL.json)의 입력·코드 SHA를 먼저 만족해야 한다.

공식 산출물은 덮어쓰기를 막는다. 위 명령은 완료된 namespace에 재학습을 권하는 안내가 아니며, 기존 START·TRACE·checkpoint·평가 영수증을 삭제해서 재실행하면 이 실험의 단 한 번 실행 기록을 보존할 수 없다. 독립 재현은 원본을 보존한 별도 작업 공간에서 수행하고 재현 기록을 따로 남겨야 한다.

[상세 결과](REPORT_KO.md) · [실제 선택 진단](SOURCE_TRANSFER_DIAGNOSTIC_KO.md) · [공개 파일 검산](PUBLIC_REVIEW_KO.md)
