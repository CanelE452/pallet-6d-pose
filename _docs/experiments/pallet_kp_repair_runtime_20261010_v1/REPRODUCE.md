# 공개 준비 검사와 실제 측정의 구분

아래 공개 검산은 Python3.9+ 표준 라이브러리만 사용한다. 새 JSON 파일로 출력하고 원본 파일을 덮어쓰지 않는다. manifest의 현재 파일과 이전311개의 SHA/크기를 확인한다. 모델·source RGB·Torch·OpenCV·GPU·PnP·학습·timing은 실행하지 않는다.

```bash
python scripts/research/pallet_kp_repair_runtime_20261010_v1/public_review.py \
  --root "$PWD" --output /tmp/pallet-runtime-readiness-review.json
```

반환0·`passed:true`가 게시 무결성 PASS다. **새 모델이나 실제600회 측정이 성공했다는 뜻은 아니다.** 실제 pre-GPU 오류 거부는 [RUNTIME_READINESS_CHECKS.json](RUNTIME_READINESS_CHECKS.json)과 [독립 CODE_REVIEW.json](CODE_REVIEW.json)의 command·exit·fixture·import audit를 확인한다.

순수 schedule 재현은 원래 benchmark의 `schedules` 함수 AST와 원래 constants만 실행한다. 원래 heavy module·모델을 import하지 않고 warmup80/measured520/total600, 경로당20+130을 출력한다. 이600은 **실행 예정 순서**이며 실제 detector600회가 아니다.

```bash
python -m scripts.research.pallet_kp_repair_runtime_20261010_v1.adapter schedule
```

현재 추가 학습 완료 파일과 승인 기록·가중치·geometry seal이 없으므로 다음 preflight는 `RUNTIME_PENDING: corrected TRAINING_COMPLETION missing`을 내고0이 아닌 값으로 종료해야 한다. 이 종료는 예상한 준비 중단이며, GPU 시간 측정 실패·600회 실행 시도로 세지 않는다. published protocol을 다시 freeze하지 않는다.

```bash
python -m scripts.research.pallet_kp_repair_runtime_20261010_v1.adapter preflight
```

# 승인과 실제 학습·자세 평가 이후의 측정 명령

현재 아래 measure 명령은 **미실행**이다. 먼저 [고정 재학습 명령](../pallet_kp_supervision_repair_20261010_v1/TRAINING_COMMANDS.md)과 추가9000update 승인이 필요하다. 같은3개 마지막checkpoint, authorized9000 completion,957observations,1914GT-free geometry, 두 downstream adapter receipt를 만들어야 한다. geometry는 `LEARNED_GEOMETRY_SEALED.jsonl.gz`를 사용하고 GT-scored predictions를 대체 입력으로 주지 않는다. hash·모델·frame·session·초기/finalpose/H·center·fallback 연계를 guard가 검사한다.

원래 source checkout·immutable handoff baseline·실사319RGB·기존가중치가 있는 pallet 환경에서 실행한다. 지정한 output/scratch는 이전 실험·source·baseline·fits와 겹치지 않는 새 위치여야 한다. 이전 완료/중단 runtime 파일이나 claim을 삭제해서 재시도하지 않는다.

```bash
task_python=/path/to/pallet/environment/bin/python
task_fits=/path/to/authorized/repaired/learned_fits
task_pose=/path/to/repaired/downstream/output
task_approval=/path/to/APPROVED_ADDITIONAL_9000.json

"$task_python" -m scripts.research.pallet_kp_repair_runtime_20261010_v1.adapter measure \
  --source-root /path/to/original/source/checkout \
  --baseline-root /path/to/immutable/pallet-pose-handoff-20261006 \
  --fits "$task_fits" --completion "$task_fits/TRAINING_COMPLETION.json" \
  --authorization "$task_approval" \
  --observations "$task_pose/LEARNED_OBSERVATIONS.jsonl.gz" \
  --observation-seal "$task_pose/OBSERVATION_SEAL.json" \
  --geometry "$task_pose/LEARNED_GEOMETRY_SEALED.jsonl.gz" \
  --pose-execution "$task_pose/LEARNED_POSE_EXECUTION.json" \
  --infer-receipt "$task_pose/REPAIR_INFER_ADAPTER_RECEIPT.json" \
  --evaluate-receipt "$task_pose/REPAIR_EVALUATE_ADAPTER_RECEIPT.json" \
  --output /path/to/new/runtime/output --scratch /path/to/new/runtime/scratch \
  --remaining-seconds 3600
```

진짜 timing은 원래 untouched observation benchmark의 loop가 새로600개의 전체 경로를 수행한다. CUDA 동기화·환경/온도/경합·thread·원행·초기/최종solver 호출·parity 결과를 기록한다. resource guard가 중단하면 다른 작업은 보존하고 그 partial timing을 정식 결과로 쓰지 않는다. measured520개가 모두 완료한 결과만 official이며 warmup80개는 별도로 계수한다. 모델/파일decode·journal·parity는 구간 밖이다. cached point/pose replay·기존 시간 단순 합산은0이다.
