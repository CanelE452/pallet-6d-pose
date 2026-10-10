# 공개 자료의 독립 검산

research 브랜치의 게시 commit을 checkout한 repository root에서 Python3.9+와 NumPy로 실행한다. 공개 파일만 읽으며 private dataset·GT·mesh를 찾지 않는다. GPU·PyTorch·OpenCV·Open3D·학습·ray·PnP 실행은 없다.

```bash
python scripts/research/pallet_kp_supervision_repair_20261010_v1/review_verify.py \
  --root "$PWD" --require-manifest \
  --output /tmp/pallet-supervision-repair-review.json
```

output은 보호된 입력 파일과 겹치지 않는 새 파일이어야 한다. 반환0과 output의 `passed: true`를 확인한다. 저장된 `REVIEW_CHECKS.json`은 실제 게시 전 검산 결과다. 파일 hash만 확인하는 것이 아니라 다음 산술을 검사한다.

1. 이전257파일·실행 byte-exact 복사·현재 manifest의 SHA/크기.
2. 고정896scene/13757query 목록, 원래 tolerance, V1/V2 cast0, 기록된1844입력 전후 보존.
3. finite ray/triangle ID/barycentric/normal·기존 rayparameter와 실제camera-Z·front NONE13754/무한대IGNORE3.
4. 공개 실제triangle vertex에서 hit point와 camera-Z 재구성, 원래깊이허용 범위, 모든13754개 front조건.
5. 전체1024×84target 원행/NPZ join·train/cal NONE복구·test와 기타array 보존·mixed supervision 분모.
6. 고정3모델 protocol과 CPU 검사에 기록된 첫batch/target/loss/gradient/state/실행량의 join. CPU head를 다시 실행하는 검산은 아니다.

검산기 의미 음성대조는 isolated public copy의 stored camera-Z에+0.02m만 주어 recomputed-camera-Z 검사에서 실제 거부되는지 확인했다. 원래 공개 파일은 수정하지 않았다. [REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json)에 command·exit·오류·verifier SHA가 있다. 모델 실행 승인 없는 상태도 [AUTHORIZATION_GUARD_CHECKS.json](AUTHORIZATION_GUARD_CHECKS.json)에서 optimizer 생성 전에 실제 거부됐다.

이 검산은 공개 witness의 산술을 검사한다. private 전체 메시의 first-hit 순서·원본mask/asset 동일성·실사 자세 정답·학습 전이를 독립 인증하지 않는다.1844입력 보존도 기록된 before/after hash의 동일성을 확인하며 private 파일을 다시 읽지는 않는다.

# 그림 재생성

Python과 NumPy·Matplotlib이 있는 환경에서 새 output directory로 실행한다. 새로운 ray/model/학습은 없다. 원래 게시그림을 덮어쓰지 않는다. Matplotlib 버전/font에 따라 PNG bytes는 달라질 수 있으며 수치 입력이 같은지 확인한다.

```bash
python scripts/research/pallet_kp_supervision_repair_20261010_v1/plot_ready_supervision_v2.py \
  --preparation _docs/experiments/pallet_kp_supervision_gate_20261010_v1/FULL_SOURCE_PREPARATION.json \
  --recovery _docs/experiments/pallet_kp_supervision_repair_20261010_v1/DEPTH_RECOVERY_VALIDATION.json \
  --triangle-check _docs/experiments/pallet_kp_supervision_repair_20261010_v1/FRONT_TRIANGLE_DEPTH_VALIDATION.json \
  --output-dir /tmp/pallet-ready-supervision-figure
```

# 학습·실사 재실행과 비용

현재 게시상태에서는 추가 학습 승인 기록이 없다. `retrain.py`는 그런 상태에서 종료하며 이를 우회하지 않는다. 원래 정식9000update가 이미 소비됐고, 같은3×3000추가 실행은 별도9000update다. 새 batch·seed·설정·source-test 기반 checkpoint 선택은 허용되지 않는다. 데이터 원본이 없는 공개 clone만으로 학습 완료를 재현했다고 주장할 수 없다.

승인 후 필요한 SHA와 argument·output 구조는 [RETRAINING_PROTOCOL.json](RETRAINING_PROTOCOL.json) 및 [학습 실행 설명](TRAINING_COMMANDS.md)에 있다. readonly 외부 feature313MB/order·원래 Base/N3·실사 입력 hash를 먼저 확인한다. 원래 decode/solver를 호출하는 [downstream.py](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/downstream.py)는319장×6경로1914행을 reference 읽기 전에 봉인하도록 한다. cached accuracy 평가 시간은 latency가 아니다. 승인 후 actual600whole-path time 측정도 별도 필요하다.
