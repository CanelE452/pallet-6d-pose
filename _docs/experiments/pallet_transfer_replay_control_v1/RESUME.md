# 재개 — GPU 외부 상태 복구 뒤

이 저장소에는 현재 **preflight와 계약 helper만** 있다. 학습 runner나 one-command
training resume가 구현됐다는 뜻이 아니다. 미실행 항목은 `TRAINING_AUDIT.json`에 있다.
드라이버·전력·시스템 변경이나 재부팅을 자동 수행하는 명령은 제공/실행하지 않는다.

GPU가 외부에서 사용 가능한 상태로 복구되면 실제 호스트에서 `nvidia-smi`로
GPU ID/VRAM/다른 작업을 확인한다. 기존 immutable 감사는 덮어쓰지 않고 새 경로로
다음 CPU 재감사 명령을 실행할 수 있다.

```bash
cd /home/minjae/Documents/github/pallet-pose
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B scripts/research/pallet_transfer_replay_control_v1/preflight.py --out _docs/experiments/pallet_transfer_replay_control_v1/recovery_01
```

`recovery_01`이 이미 존재하고 바인딩이 달라졌으면 보존하고 새 recovery 경로를
사용한다. 버전/원본 SHA가 변했으면 변경 사실을 숨기지 않고 학습 전에 감사한다.

다음 작업은 동일 승인 실험의 구현 계속이다.

1. 기존 `simulation.py`의 export_target/augmentation/HYP/optimizer를 정확히 재사용할
   격리 loader를 구현한다. T30만 export하고 각 stream 전용 buffer/RNG를 둔다.
2. 실제 batch8에서 gradient 항등식, BN running 불변/affine gradient, base tensor SHA,
   Mosaic 보조참조 allowlist, save/load output parity, criterion schedule을 검사한다.
   smoke 총16update 이내; 현재 사용0. 학습 전에 최종 실행코드와 gate 결과를 추가 lock한다.
3. resume 가능한 model/optimizer/RNG/sampler/criterion 상태와 노출·clip log를 저장하는
   runner로 seed1/2/3×T8_FULL/T8_QUARTER/REPLAY/T32_COMPUTE 각300update를 수행한다.
   각 fit은 원래 R0에서 시작. 기존 다른 실험의 학생을 초기값이나 새 대조군으로 쓰지 않는다.
4. 모든12개 last가 저장된 뒤13모델을 real145/negative2689/source512에서 실제 추론한다.
   source의 prepared100px padding을 중복 적용하지 않고 기존 원본좌표 변환을 사용한다.
   frozen R0 feature cache를 적응모델의 feature로 재사용하지 않는다.
5. 고정 full-GT PCK와 기존2D/6D 평가를 통합하고4session/frame paired bootstrap,
   source scenario bootstrap,4회 LOSO, 사전contrast를 계산한다. 성능 기반 재학습 금지.
6. 새 보고서로 CURRENT_RESULT를 갱신하되 최초 기술중단 기록은 이력에 보존한다.
   main만 사용하고 작은 산출물만 commit/push, 실제 원격SHA를 검증한다.
