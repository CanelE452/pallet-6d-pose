# 실행 경로와 재현 조건

이 폴더는 완료 결과를 검토하기 위한 기록이다. GitHub의 표·그림·계약 파일은 별도 GPU 없이 읽을 수 있다. 실제 학습과 추론을 재현하려면 원본 데이터·고정 모델·source cache·prepared pair가 필요하다. 큰 checkpoint와 crop tensor는 Git에 포함하지 않으며 경로와 SHA256을 INPUTS/각 FIT 파일에 기록한다. GitHub clone만으로 모든 학습 입력을 얻는 구조는 아니다.

## 검토 순서

1. [REPORT_KO.md](REPORT_KO.md)의 전체 판정과 모든 seed 결과를 읽는다.
2. [FRAME_RESULTS.csv](FRAME_RESULTS.csv)에서 9모델×173장=1,557행을 확인한다. pose 실패도 행을 남긴다.
3. [GALLERY_NATURAL99.md](GALLERY_NATURAL99.md)에서 자연 가림99장을 모두 확인한다. 그림의 seed1은 결과 전에 고정했으며 수치표에는 seed1·2·3이 모두 있다.
4. [PROTOCOL_EFFECTIVE.json](PROTOCOL_EFFECTIVE.json), [PROTOCOL_AMENDMENT_01.json](PROTOCOL_AMENDMENT_01.json), [INPUTS.json](INPUTS.json)에서 252→251 변경, 고정 입력·seed·기준을 확인한다.
5. [PRETRAIN_TESTS.json](PRETRAIN_TESTS.json), [TRAINING_COMPLETE.json](TRAINING_COMPLETE.json), [PREDICTIONS_LOCK.json](PREDICTIONS_LOCK.json), [RESULTS.json](RESULTS.json)의 파일 연결과 hash를 확인한다.

## 실행 환경

원래 작업 경로는 `/home/minjae/Documents/github/pallet-pose`이며 Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. 패키지 버전은 [RUN_ENVIRONMENT.json](RUN_ENVIRONMENT.json)에 있다. GPU 작업은 호스트 RTX3080에서 실행했다. CPU 검토 환경에서 CUDA가 보이지 않는 현상과 실제 GPU 작업 성공을 혼동하지 않는다. 환경·드라이버 설치나 변경은 하지 않았다.

아래 명령은 저장소 루트에서 수행한 단계의 실행 진입점이다. 이미 만들어진 실험 폴더에서 새 학습을 다시 시작하면 안 된다. START/trace/checkpoint가 있으면 trainer는 중복 실행을 거부한다. 기록을 삭제해 재학습하거나 실패 seed를 교체하는 경로는 제공하지 않는다. 별도 재현은 원본 산출물을 보존한 새 namespace와 동일 계약이 필요하다.

```bash
MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.prepare_diverse cpu

MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.prepare_diverse gpu
```

처음 준비에서 IoU 기준에 실패한1쌍은 DATA_PREPARATION_GPU.json에 남아 있다. 이때 INPUTS252를 만들지 않았다. 학습 전 명시적 수정 이후 `finalize_inputs.py`가 실패 DIVERSE 입력과 정해진 SINGLE 입력을 각각1개 제거하고 세 seed의 order를 잠갔다. IoU 기준을 낮추거나 새 사진을 뽑아 대체하지 않았다.

CPU `train preflight`에서 실제 backward를 검사하되 optimizer step은0이었다. GPU 실행 직전에는 TRAIN_TRANSITIVE_CODE_LOCK의104개 코드 hash를 검증했다. 실제 학습 wrapper의 핵심은 다음과 같다.

```python
from scripts.research.pallet_pose_stable_improvement_20261001_v1 import common as C
from scripts.research.pallet_pose_stable_improvement_20261001_v1 import train as T

for binding in C.read(C.DOC / 'TRAIN_TRANSITIVE_CODE_LOCK.json')['files']:
    C.verify(binding)
for seed in C.SEEDS:
    for arm in C.ARMS:
        T.train(arm, seed)
T.audit_pairing()
```

각 fit은 동일 PRIOR1에서300step을 학습한다. optimizer state를 군이나 seed 사이에 이어 쓰지 않는다. 미완료 학습을 새 실행으로 조용히 대체하지 않으며, 실행 중이면 원래 process/trace를 먼저 확인해야 한다.

## 추론·평가·보고서 생성

다음 순서를 유지한다. `infer`는 RGB와 고정 R0 입력으로9모델 출력을 전부 동결한다. `evaluate`는 그 뒤 GEO pose 후보를 고정하고, 마지막으로 기존 참조 file hash를 검사한 후 정답을 읽는다. 평가 값으로 checkpoint·seed·타깃·입력을 고르지 않는다.

```bash
MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.infer all --device cuda

MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.evaluate all

MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.mechanism score

MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.figures all

MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.report

MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_pose_stable_improvement_20261001_v1.validate
```

프레임별 추론 cache는 같은 context hash일 때만 이어 읽는다. 실제 새 forward 수는 이미지 수와 구별하며, 검출이 없어 refiner 입력이 없는 프레임도 평가 모집단에 남긴다.

## GitHub 게시 경계

원래 작업 저장소의 기존 변경·미추적 산출물은 유지하고 별도 checkout에서 명시된 publication manifest만 commit/push한다. 새 문서·표·실제 결과 그림·스크립트와 직접 참조하는 작은 이전 증거 파일을 포함한다. model weight, RGB 원본, 큰 crop tensor, 무관한 실험 결과는 게시 목록에 포함하지 않는다. push 이후 원격 commit과 파일 blob이 로컬 publication payload와 같은지 확인해야 게시 완료다. 체크포인트 학습 완료나 로컬 보고서 생성만으로 push 완료라고 표시하지 않는다.
