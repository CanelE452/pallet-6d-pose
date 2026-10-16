공개 원행의 산술 검산과 GPU 경로 재현은 서로 다르다. 원행·protocol·영상을 함께 보고 동작과 실패 이유를 확인할 수 있다. 실제 GPU 추론에는 원래 원본 source checkout, 고정 baseline checkout, byte-identical Base/N3 체크포인트와 수정 감독의 완료 IMAGE_ROLE checkpoint가 필요하다. [이전 배포 의존성 설명](../pallet_boundary_corner_refiner_20261010_v2/REPRODUCE.md)을 따른다. 경계 선택기는 공개 v2 deployment 경로에서도 제공한다. 경로를 바꿀 때 입력 SHA가 같아야 한다.

**비공개 영상·가중치 없이 공개 원행을 검산하는 명령**은 아래다. Python 표준 라이브러리만 필요하다. 모든1470행·방법별245개 ID·공개 frozen-input SHA와 쉬움/중간/전체의 평균·분산·SD·중앙값·P90를 독립 계산한다. GPU 추론과 물리 GT를 다시 인증하거나 CI를 다시 계산하는 명령은 아니다. CI의 별도 독립 산술 실행은 VERIFICATION.json에 보존했다.

```bash
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.public_review \
  --output /tmp/cornerwise_public_review.json \
  --export-csv /tmp/cornerwise_per_frame.csv
```

게시된 [PER_FRAME.csv](PER_FRAME.csv)에는 영상 ID·난도·방법·새 자세/반환 상태·T/R/ADDsym·H·fit/inlier·선택 코너를 한 행씩 제공한다. 더 자세한 후보 및 LOO 근거는 gzip 원행에 있다.

LOO와 최종 fit에서 self-occ 좌표를 제외했는지, 선택한 좌표가 실제 경계 후보인지 확인하는 공개 검산도 표준 라이브러리로 실행한다. 저장된 실행 기록을 검사하며 가림 정답·자세 성능을 새로 인증하지 않는다. 최종 재투영의 별도 scalar 산술 검산은 REPROJECTION_CHECKS.json에 있다.

```bash
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.loo_exclusion_check \
  --output /tmp/cornerwise_loo_exclusions.json
```

원래 실행은 다음 순서로 새 namespace에 한 번 수행한다. 완료 파일을 덮어쓰지 않는다. 재현자는 기존 publication 데이터를 복사한 별도 checkout과 새로운 private subtree를 사용해야 한다. `freeze`는 새 protocol/protection을 생성하므로 게시된 파일에 실행하지 않는다.

```bash
export PALLET_SOURCE_ROOT=/path/to/source
export PALLET_BASELINE_ROOT=/path/to/frozen/baseline
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
```

실제 실행 모듈은 다음이다. `--output`은 이 실험의 정확한 DOC 또는 `/dev/shm/pallet-cornerwise-private-20261010-v3` 아래 새 디렉터리만 허용한다. GPU 재현에서 protocol·protection의 생성 경로를 분리하고 같은 파일을 이후 단계에 지정한다.

```text
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.test_selection
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.run freeze
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.run preflight
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.run infer
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.evaluate score
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.statistics
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.verify
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.runtime measure
python -B -m scripts.research.pallet_cornerwise_refiner_20261010_v3.render
```

CLI의 기본 output은 모듈별로 다르므로 실제 명령은 BUILD_LEDGER의 argv와 함께 확인한다. 새 inference는245개 Base/N3 좌표·초기 자세·경계 관측을 실제 생성하고 네 최종 방법980행과 고정 대조490행을 GT 전에 봉인한다. score는 이후 기존 참조만 읽으며 pose fitting 진입을 금지한다. bootstrap은 기존10000×13세션 multiplicity를 재사용하며 새 draw는 생성하지 않는다.

실제 첫 실행은160장 뒤 자원 guard에서 중단됐으며 [원래 실패 기록](INFERENCE_FAILURE.txt)과 INTERRUPTED 원행을 보존했다. [별도 재개 계약](RESUME_PROTOCOL.json)의 `resume freeze`와 `resume resume`는 남은85장만 처리한다. 완료한160장을 다시 계산하지 않고 gzip 원본 member를 보존해 최종245장을 구성한다. 이 재개는 알고리즘·체크포인트·성능 설정을 바꾼 재실험이 아니다.

runtime은 별도 자원 창에서4방법×(20warmup+26영상×5반복)=600회 새 전체 API 경로를 실행한다. 초기 자세·LOO 검증·최종 강건 PnP·H 재투영이 실제 interval에 포함된다. RGB decode, 모델 load, GT 채점, parity, journal과 자원 스냅샷은 interval 밖이다. 캐시 좌표 재생이나 다른 측정값을 더한 시간이 아니다.

그림6사례는 이전에 고정한 ID를 그대로 사용한다. 성능 개선 사례만 새로 고르지 않는다. 흰 점은 기존 proxy 참조이며 새 수동 어노테이션이 아니다. 원본 영상·가중치·실패 원행·사용자 변경과 main은 보호 대상이다. 추가 실험으로 설정을 다시 고르는 명령은 없다.
