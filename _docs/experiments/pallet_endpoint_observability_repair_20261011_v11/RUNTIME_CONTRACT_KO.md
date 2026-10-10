# V11 안전 경로의 전체 처리시간 계약

이 계약은 `FILTERED_LOCAL_WEAK_SAFE` 한 경로를 원본 RGB에서 새로 실행해 비용을 측정한다. V11 정확도 실험의 관측, 임계값, 모델, 초기화, 해 선택을 바꾸지 않는다. 실행 결과가 생기기 전에는 처리시간이나 성능 개선을 주장하지 않는다. `NEW`는 수치적으로 새 자세를 얻었다는 뜻이며 실제 위치·회전의 개선을 뜻하지 않는다.

## 실행 전 권위와 고정 입력

[runtime.py](../../../scripts/research/pallet_endpoint_observability_repair_20261011_v11/runtime.py)는 [evaluation_review.py](../../../scripts/research/pallet_endpoint_observability_repair_20261011_v11/evaluation_review.py)의 V11 기하 완료·cleanup·독립 검산 성공 gate와 완료된 `SCORING_RECEIPT.json`을 먼저 확인한다. 원 검산이 대응 부족 상태의 선택 후보 생략 필드를 직접 참조해 `KeyError('best_candidate')`로 멈춘 자료는 보존한다. 별도 검토는 이 접근 한 개만 `.get()`으로 바꾼 검사기의 전체 성공을 인수한다. 원 실패, 별도 검사기·protocol/started/checks, scoring 권위 wrapper와 원 평가 kernel을 SHA로 함께 묶는다. 정확도 프로토콜이나 원행을 수정해서 성공으로 만들지 않는다.

측정 전에 별도 `RUNTIME_PROTOCOL.json`을 한 번 동결한다. 동결 입력에는 V11 `runner.inputs`의 전체 바이트 결합, V7 생성기의 원본 코드·Base/N3 가중치·세 학습 모델과 보정 자료의 전체 결합, 실제 로드할 IMAGE_ROLE의 확정 가중치, 이 코드·계약, 변경하지 않은 V9 프레임 정규화/자세 비교 helper, 정확도 원행·seal·receipt·검산·scoring 바이트가 포함된다. 사용한 경로는 명시적인 `--source-root`, `--baseline-root`, `--fits`, `--cohort`, `--calibration`, `--calibration-root`, 선택적인 `--base-weights`, `--N3-weights`로 기록한다. 모델은 IMAGE_ROLE 하나만 로드한다. 다른 모델 파일의 SHA 결합은 기존 V7 생성기 계약의 일부이며 해당 모델의 추가 추론을 의미하지 않는다.

기존 corrected `RUNTIME_PANEL.json`의 13개 세션에서 각 두 장, 총 26장을 그대로 사용한다. V9 `normalized_frames`는 기존 245장 전체의 ID·세션·원본 영상 경로/SHA·K·등록 치수를 대조한 뒤 추론 입력을 RGB, K, 치수, ID/세션/물체 유형으로 제한한다. helper가 읽는 기존 메타데이터에는 GT 필드가 포함될 수 있으나 해당 수치나 라벨을 추론에 전달하지 않는다. 저장한 Base/N3 좌표와 V11 기하 원행은 반환 후 비교에만 사용한다. 새 RGB, seed, 학습, 난이도 선택, GT에 따른 시간 측정 영상 선택은 없다.

## 150회 실제 실행과 시간 경계

고정 순서는 warmup 20회(`image_index = index % 26`) 이후 같은 26장 순서를 다섯 번 실행하는 측정 130회다. 각 호출마다 새 검출과 새 N3/경계 모델 forward를 수행한다. 한 호출 안에서 새 sparse point 강건 자세, native 좌표의 endpoint LOO, 모순된 관측만 제외하는 필터, filtered LOCAL 한 번과 기존 약한 관측 gate를 계산한다. 기존 `Pipeline.predict`가 동일한 LOCAL 제안에서 두 운용 view의 메타데이터를 함께 조립하는 비용도 포함된다. 별도 두 번째 LOCAL fit이나 다른 경로의 시간 측정은 하지 않는다.

CUDA를 동기화한 뒤 `perf_counter_ns`의 `t0`를 기록하고 실제 V11 `deployment.Pipeline.predict(..., method=PRIMARY)`를 호출한다. 반환 직후 다시 CUDA를 동기화하고 `t1`을 기록한다. 전체 처리시간은 `(t1-t0)/1e6` ms다. RAM RGB/K/치수에서 시작해 검출기, N3→cornerSubPix, 초기 자세, Base에서 만든 역할 특징과 IMAGE_ROLE 관측, sparse point 강건 PnP, native endpoint LOO, 관측 필터, LOCAL, 약한 관측 gate, 자기 가림 코너 재투영, 원 검출의 다른 후보/점수/박스/중심 보존과 결과 메타데이터 반환까지 포함한다.

측정 안에서 동기화하는 단계 callback은 다음 여섯 개다. callback·동기화 비용 역시 전체 처리시간에 들어간다.

1. `detector`
2. `N3_correction`
3. `initial_pose`
4. `boundary_observation`
5. `endpoint_validation_and_filtered_local`
6. `final_pose_reprojection_and_metadata`

다섯 번째 구간은 fresh sparse point 자세와 endpoint LOO·필터·LOCAL 및 두 view 조립을 함께 포함한다. 마지막 callback 이후 반환까지는 `return_ms`로 따로 기록한다. 원본 파일 SHA 검사·26회 RGB decode·가중치 로드·모델 초기화·반환 후 parity·GPU/CPU 자원 검사·SHA/보존 검사·durable logging·원행 gzip 저장·GT 평가를 시간 구간 밖에서 수행한다. 저장한 관측 또는 자세의 재생 시간, 과거 시간의 합산을 사용하지 않는다.

## 원행과 검산 증거

`RUNTIME_ROWS.jsonl.gz`에 warmup을 포함한 모든 150개 호출의 원행을 기록한다. 각 원행에는 `t0_ns`/`t1_ns`, 실제 단계 clock tick, 전체/단계 시간, 반환된 전체 결과와 후보·solver·재투영·상태, 원본 fresh 검출의 전체 후보, fresh point bank와 repair ledger, endpoint/LOCAL 실행 event, 실제 모델/헤드 forward·OpenCV·SciPy·local/native 연산의 호출 전후 차이를 보존한다. 원행 저장은 반환 후 수행하며 다음 호출의 입력으로 쓰지 않는다.

IMAGE_ROLE의 실제 forward pre/post hook을 함께 센다. OpenCV 여섯 진입점의 실제 entry/return/exception, SciPy `least_squares`의 실제 entry/return/성공 보고·residual/J callback을 감싼 뒤 원 함수를 그대로 호출한다. SciPy 계수는 LOCAL ledger의 실제 optimizer/residual/J 횟수와 대조한다. 실제 entry는 return과 exception의 합과 같아야 한다. 생산 solver가 초기화별 수치 예외를 처리해 다른 초기화나 typed fallback으로 정상 반환하는 경우에는 해당 예외 횟수와 최종 상태를 그대로 보존한다. 처리된 수치 예외 하나만으로 전체 benchmark를 실패시키지 않으며, 처리하지 못해 호출이 중단되면 실패 prefix를 보존한다. 모델 로딩 중 검출기 내부 초기화 forward는 별도로 기록하고 150회 측정 호출의 forward와 분리한다. endpoint native bank의 실제 계수 차이와 전체 provenance를 보존한다. 선언한 수를 실제 호출 수처럼 복사하지 않는다.

동일한 26개 정확도 기하 원행과 반환 후 좌표·실제 자세·H·NEW/fallback·메타데이터를 비교한다. Base/N3 원 좌표도 반환 후 대조한다. pose helper는 기존에 존재하는 필드만 비교하며 LOCAL에 없는 필드를 새로 만들지 않는다. 허용치는 기존 절대 `1e-7`, 상대 `0`이며 H 재투영 좌표 대조는 절대 `1e-12`다. NEW에서 H가 최종 fit IDs와 겹치지 않는지 확인한다. H 표시가 달라졌다는 이유로 정확도 원행을 재계산하거나 모델 출력을 교체하지 않는다. 자기 가림 점을 재투영한 뒤 다시 독립 관측으로 fit하지 않는다.

전 호출·parity·cleanup이 성공한 경우에만 130개 측정 원행에서 전체 시간, 여섯 단계와 반환 구간의 `n`, 평균, 표본 분산(`ddof=1`), 표준편차, 중앙값, P90, 최대값을 계산한다. warmup은 원행에 유지하되 요약에 넣지 않는다. 중간 실패의 작은 완료 집합으로 공식 latency를 발표하지 않는다. 새 정확도·bootstrap·GT 통계 계산을 이 benchmark에서 하지 않는다.

## 자원·보존·실패

모델/영상 작업 전, 모델 로딩 후, 측정 중 최소 5초마다 및 각 measured block 시작, 마지막 측정 뒤, cleanup 뒤 자원 snapshot을 남긴다. snapshot의 quiet와 GPU 온도 80°C 미만 조건을 모두 요구한다. 자원 검사는 시간 구간 밖에서 수행한다. `RUNTIME_RESOURCES.json`과 receipt에 실제 snapshot을 보존한다. 기록한 snapshot의 quiet 여부와 실제 경쟁 작업 발견 여부를 구분하며 관측하지 않은 전체 시간의 무경합을 인증하지 않는다.

원본 체크아웃·사용자 변경·완료 실험의 보호를 시작과 cleanup 이후에 확인한다. 게시 HEAD는 benchmark 도중 바꾸지 않는다. 후크·OpenCV·SciPy wrapper·OpenCV thread 설정·원본 환경변수·모델 lifecycle을 정리한 뒤 최종 receipt를 쓴다. `RUNTIME_STARTED.json`은 이번 실행이 처음 소유한 journal만 durable하게 갱신한다. 나머지 출력은 독점 생성하며 기존 파일, pending prefix 또는 실패 receipt가 있으면 덮어쓰지 않는다.

실패하면 첫 오류와 traceback을 보존하고 cleanup 오류는 별도로 기록한다. 완료된 prefix와 가능한 마지막 전체 capture/point/repair/event를 `INTERRUPTED_RUNTIME_ROWS.jsonl.gz` 및 `INTERRUPTED_RUNTIME_PACKET.json`에 남긴다. 자동 재시도, threshold 변경, 대체 candidate 재선택, 두 번째 fit은 없다. STARTED 이전 CLI/preflight 오류도 외부 실행 ledger에서 따로 보존한다. 최종 `RUNTIME_RECEIPT.json`의 `complete=True`와 공식 통계 여부가 실제 완료를 판단하는 권위이며 코드 준비만으로 완료를 보고하지 않는다.

## 재현 명령

기존 실험과 같은 Python 환경과 두 원본 root를 사용한다. 아래의 positional stage를 순서대로 실행하며 실패 시 같은 출력 폴더에서 자동 재시도하지 않는다. 정확도 기하·독립 검산·scoring 완료 전에 실행할 수 없다.

```bash
python -m scripts.research.pallet_endpoint_observability_repair_20261011_v11.runtime freeze --source-root /absolute/source --baseline-root /absolute/baseline
python -m scripts.research.pallet_endpoint_observability_repair_20261011_v11.runtime preflight --source-root /absolute/source --baseline-root /absolute/baseline
python -m scripts.research.pallet_endpoint_observability_repair_20261011_v11.runtime measure --source-root /absolute/source --baseline-root /absolute/baseline
```

개인 GPU/원본 가중치와 영상이 없는 공개 환경에서는 이 실제 추론을 재실행했다고 주장할 수 없다. 공개 runtime 원행·receipt·별도 독립 검사 자료로 실행량·계수·시간 요약의 일관성을 확인할 수 있다. 속도 측정 자체는 개선된 대응의 물리적 정확성이나 장치 간 동일성을 인증하지 않는다.
