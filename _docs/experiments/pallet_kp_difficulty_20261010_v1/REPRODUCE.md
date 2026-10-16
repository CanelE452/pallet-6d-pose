# 저장 결과 검산과 실제 재실행

공개 검산은 새 원행과 기존 게시111파일, logits, frozen bootstrap 행렬, 통계와 그림 binding을 읽는다. 원본 영상이나 가중치를 요구하지 않는다. Python3.9+ 표준라이브러리만 사용한다.

```bash
python3 scripts/research/pallet_kp_difficulty_20261010_v1/review_verify.py --require-manifest
python3 scripts/research/pallet_kp_difficulty_20261010_v1/review_verify.py \
  --output /tmp/pallet-kp-review.json --require-manifest
```

출력의 전체 PASS와 각 검사 범위를 확인한다. 해시·원행 수·통계·CI·관측 없음·fit·hidden replacement·실제 runtime 원행을 재검산하지만 원본 촬영의 물리 GT나 과거 학습 실행을 독립 인증하는 명령은 아니다. 의도적 오류 검출과 공개 파일만 있는 격리 실행은 `REVIEW_VALIDATION_TESTS.json`에 기록한다.

실제 재실행은 원본 자료가 있는 연구 환경에서 **게시 결과와 다른 전용 재현 tree**를 사용한다. 완료된 원행·ray·runtime 시도를 덮어쓰는 동작은 guard가 거부한다. 원래 결과가 들어 있는 이 브랜치에서 같은 실행 명령을 무조건 재실행하지 않는다. 대부분 DOC가 checkout의 고정 경로이며 임의 --output-dir 옵션은 없다. 재현 tree에서 새 DOC가 비도록 구성하고 필요한 PROTOCOL·앞 단계 검사 영수증·기존 입력 의존성만 준비해야 한다. private scratch의 새 journal도 기존과 분리한다. 원래 게시 checkout의 완료 파일을 삭제해서 시작하지 않는다. 대형 비공개 cache를 Git에 추가하거나 모델을 다시 학습할 필요도 없다.

이 환경의 실제 Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. Python3.10, OpenCV4.9, PyTorch/CUDA·Ultralytics·Open3D·USD·NumPy·Matplotlib 등의 버전과 device는 원 실험 및 새 `RUNTIME.json`에 기록했다. 실사 source repository와 과거 baseline handoff가 필요하다.

```bash
export PALLET_SOURCE_ROOT=/home/minjae/Documents/github/pallet-pose
export PALLET_BASELINE_ROOT=/home/minjae/Documents/github/pallet-pose-handoff-20261006
export PALLET_OBSERVATION_SCRATCH=/dev/shm/pallet-observation-private-20261009
export PALLET_KP_DIFFICULTY_SCRATCH=/dev/shm/pallet-kp-difficulty-private-20261010
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

`PALLET_OBSERVATION_SCRATCH`에는 기존 `learned_cache`(features/lo/hi/weight/valid/order), manifest, 실제 메시 캐시와 기존 `learned_fits/{GEOMETRY_ONLY,IMAGE_NO_ROLE,IMAGE_ROLE}.pt`가 있다. 기존3개 마지막 가중치는 source의 ignored `data/pallet/results/pallet_observation_refiner_20261009_v1/weights/`에도 보존되어 있으며 기존 `PRIVATE_CHECKPOINT_RETENTION.json`의 SHA와 일치한다. 새 학습은 실행하지 않았다. feature 캐시는 공개 logits 검산에는 필요하지 않지만 source fixed-head 재추론에는 필요하다.

선택과 fit의 입력은 frozen `INPUTS.json`·`OBSERVATIONS.jsonl.gz`·`LEARNED_OBSERVATIONS.jsonl.gz`와 원본 K·치수다. source 감독은 실제 기존 P0 RGB/JSON·USD·전달된 mask와 원래 cache다. 후행 reference annotation은 `data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json` 및 기존 `load_real` 의존성을 필요로 하며 GT를 최초로 여는 시점은 `CAUSAL_GEOMETRY_SEALED` 저장 이후다.

실행 의존 순서는 다음과 같다. 모델·설정을 골라 바꾸는 반복이 아니라 고정 한 번의 단계들이다. `source_ceiling`은 기본 감독 조립, `source_ceiling --probe-frozen-heads`는 기존 head의 source-test 재추론이며, `source_mass`는 고정 calibration/test source-head 진단이다. `--help`는 argparse를 지원하는 모듈에만 사용한다. source_mass/source_ray_validation/source_ray_review는 argparse 없이 run하므로 --help를 주어도 실제 작업을 시작할 수 있다. 이 모듈은 먼저 코드와 고정 프로토콜을 읽고 완료-file guard·경로를 확인한다.

1. `source_ceiling`, `decode_difficulty`, `geometry_difficulty`: 기존 감독과 관측/해의 어려움 분해.
2. `source_mass`, `source_logit_difficulty`: 기존 frozen head의 source logits·확률 검사.
3. `source_ray_validation`, `source_ray_review`: 봉인된 source-test128의 actual-mesh ray 민감도·기계적 조립 진단과 그림. 새 감독을 승인하거나 원래 target을 바꾸지 않는다.
4. `test_dimension_prior`, `match_mass`: numerical prior 계약과 고정 decoder의 기존957행 재현.
5. `causal_evaluate --pilot`, `causal_evaluate`: GT-free 선택/fit→봉인→319채점 및 기존4조건 재실행 일치.
6. `statistics`: 후행 GT annotation·전체/공통/조건부 통계·같은 frozen bootstrap draw.
7. `benchmark measure`: 다른 numeric/GPU/시간 측정 작업이 없는 창에서 실제600전체 경로. 캐시나 기존 latency 합산을 사용하지 않는다.
8. `visual_review`, `report`, public `review_verify`: 그림·문서·검산. 새 결과 manifest를 생성하고 다시 검산한 뒤 게시한다.

Package 명령은 `python -m scripts.research.pallet_kp_difficulty_20261010_v1.<module>` 형식을 권장한다. 이 폴더의 `statistics.py`를 표준라이브러리 `statistics`로 잘못 import하는 직접 script 실행을 피한다. public verifier의 직접 실행은 이 충돌을 피하도록 구현되어 있다.

원래12무학습 비교·oracle5/6·가림 오제외/오잔류·학습3조건·동일 IMAGE_ROLE point+line 절제의 전체 재현은 [기존 REPRODUCE.md](../pallet_observation_refiner_20261009_v1/REPRODUCE.md)에 있다. 이번 결과를 그 실험의 설정 변경이나 새 holdout 확인으로 혼동하지 않는다. 기존 미실행 E6 네 변형과 FIXED_CONTROLS의 BASE293+SUBPIX293=586행의 과거 R/t 미저장은 그대로 남아 있다. 이586행의 저장 metric 재집계는 가능하지만 독립 pose 재계산은 불가능하다. 새2552행의 R/t 누락을 뜻하지 않는다.

Git에 있는 파생 RGB 그림은 입력 SHA·crop·ID가 manifest에 기록된 검토 이미지다. source 감독·실사 정답 입력으로 쓰지 않았다. 원본 RGB·대형 cache·가중치는 공개 Git에 넣지 않았다.
