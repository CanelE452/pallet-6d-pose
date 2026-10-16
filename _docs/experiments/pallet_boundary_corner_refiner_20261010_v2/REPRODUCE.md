공개 원행 검산과 실제 모델 실행을 구분합니다. 검산은 이미 저장한 출력의 수치·기하·파일 연결을 확인하며 GPU 실행이나 물리 GT를 독립 인증하지 않습니다. 모든 명령은 해당 연구 브랜치의 저장소 루트에서 실행합니다. import만으로 추론·학습·시간 측정을 시작하지 않습니다.

공개 source calibration 검산은 NumPy가 필요하며 원본 영상·모델·GT·ray caster를 사용하지 않습니다. 출력 기본값은 표준 출력입니다.

```bash
python3 -B -m scripts.research.pallet_boundary_corner_refiner_20261010_v2.verify_calibration
```

전체 공개 검산은 표준 라이브러리만 사용합니다. 공개 결과 폴더가 기본 입력이며 새 저장소 외부 JSON 파일에 영수증을 저장합니다. 이미 게시된 검산 영수증을 덮어쓰지 않습니다. source calibration 원행, 245개의 관측, 1225개의 새 geometry/score, 490개의 fresh 대조, 통계, 시간 원행과 protocol SHA가 검토 대상입니다.

```bash
python3 -I -S scripts/research/pallet_boundary_corner_refiner_20261010_v2/review_verify.py \
  --require-runtime --output /new/output/boundary-review.json
```

6개 그룹 PASS/complete=true가 완료 조건입니다. 평균·분산·표준편차·중앙값·P90·paired 평균, R,t의 재투영, fit 제외, 실제600회 측정 interval과 saved output parity를 독립 산술로 검산합니다. bootstrap draw SHA와 CI 형식은 확인하지만 CI 수치 자체를 재생하지 않는 한계를 영수증에 명시합니다. posthoc 대응/손상 검산은 NumPy를 사용합니다.

```bash
python3 -B -m scripts.research.pallet_boundary_corner_refiner_20261010_v2.posthoc_check \
  --output /new/output/boundary-posthoc-review.json
```

CPU 기하·관측·인터페이스 테스트는 `test_observations.py`, `test_pose.py`, `pipeline_checks.py`에 있습니다. 4·5점 표준 해법, 잘못된 대응·마스크, 부족한 관측·분기 이동·수치 동점, 최종 H 제외/재투영, 선 지지/불확실성 채택, metadata와 기본 반환 상태를 검사합니다. 테스트 영수증은 실제 수행 횟수와 OpenCV 호출량을 기록합니다.

실제 배포에는 SHA가 일치하는 원본 Base/N3 가중치, 수정 IMAGE_ROLE 마지막 체크포인트와 완료 영수증, 원래 실행 코드·registry·N3 설정, immutable `a22fb14beb5e8df08076385000e0d53503c1ae29` baseline 의존성, 호환 CUDA 환경이 필요합니다. 원본 데이터를 가진 실행자는 의존 경로를 명시합니다. baseline은 기존 9개 코드·증거 바이트를 검증하는 읽기 전용 의존성입니다.

```bash
export PALLET_SOURCE_ROOT=/path/to/original/source
export PALLET_BASELINE_ROOT=/path/to/immutable/baseline
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

단일 실제 영상은 K를 JSON 3×3 배열, 치수를 JSON `[width,height,depth]` 미터 배열로 제공합니다. 결과는 새 JSON 파일에만 씁니다. 이 명령은 evaluator와 사람 마스크를 사용하지 않습니다.

```bash
python -B -m scripts.research.pallet_boundary_corner_refiner_20261010_v2.pipeline predict \
  --fits _docs/experiments/pallet_boundary_corner_refiner_20261010_v2/deployment \
  --calibration _docs/experiments/pallet_boundary_corner_refiner_20261010_v2/CALIBRATION.json \
  --image /path/to/image.png --K-json /path/to/K.json \
  --dimensions-json /path/to/physical_whd.json --output /new/output/prediction.json
```

정식 연구 실행 순서는 `pipeline freeze → pipeline preflight → pipeline infer → evaluate score → statistics → diagnostics → runtime measure → report → review_verify`입니다. 시간 측정은 실제 최초0호출 모듈 경로 실패를 보존하고 `runtime_resume freeze/preflight/measure`의 한 outer context에서 동일한 고정 본문을 한 번 실행했습니다. 완료된 protocol/원행을 보존하므로 그대로 재실행하면 기존 파일 보호 검사가 거절합니다. 새 실험을 실행할 때는 별도 namespace와 입력 보호 목록이 필요합니다. 게시 protocol이 완료된 한 번의 실행을 자동으로 반복하거나 추가 학습을 승인하지 않습니다.

geometry와 고정 대조는 전체245의 fit을 끝낸 뒤 봉인합니다. 채점은 봉인 SHA 검증 뒤 reference를 읽으며 PnP 호출을 금지합니다. 통계·그림·posthoc 진단은 저장된 행만 읽습니다. 시간 측정은 같은26장·13세션에서 4경로 각각20 warmup+130 측정, 총600회의 fresh 호출이며 모델 로드·RGB decode와 채점은 측정 밖입니다. detector·N3·경계 관측·초기 자세·강건 PnP·H 재투영은 해당 경로의 실제 측정 안에 있습니다.

정확한 checkpoint와 입력·실행 코드 SHA는 PROTOCOL/CALIBRATION/GEOMETRY_SEAL/실행 영수증에 있습니다. NumPy/OpenCV/Torch/CUDA/device/thread 설정은 RUNTIME.json에 있습니다. 공개 검산은 이 실행 기록과 원행의 일관성을 확인합니다. 비공개 checkpoint tensor, 당시 GPU 동작과 순간 자원 상태, 실제 물리 자세를 다시 측정하지 않습니다.
