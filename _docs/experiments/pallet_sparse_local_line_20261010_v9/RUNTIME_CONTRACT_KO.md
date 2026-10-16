# V9 fresh 전체 경로 runtime 계약

이 보충 계약은 V9 정확도 코드·원행·설정을 변경하지 않는다. 해당 정확도의 봉인, 독립 기하 검산 PASS와 채점 완료 후, runtime의 자체 코드·가중치·source/calibration·고정 panel·정확도 참조 SHA를 별도 `RUNTIME_PROTOCOL.json`에 freeze한다. 결과를 보고 optimizer·rank·threshold·시작점·이미지를 재선택하지 않는다. 아래 코드는 작성 단계이며 실제 실행량과 PASS 여부는 `RUNTIME_STARTED.json`, `RUNTIME_ROWS.jsonl.gz`, `RUNTIME.json`을 확인한다.

네 경로는 `BASE`, `N3_SUBPIX`, `ROLE_BOUNDARY_H_ROBUST`, `ROLE_BOUNDARY_LOCAL_POINT_LINE`이다. 기존13세션의2장씩 고정26 panel을 사용하고, 각 경로20 warmup+5×26 measured의150회, 전체600회의 fresh capture를 수행한다. 매 호출마다 detector를 새로 실행한다. 다른 경로 또는 accuracy의 저장 관측을 공유·재생하지 않는다. 모델 로딩·원본 RGB hash/decode는 시간 밖에 있지만 RAM BGR/K/등록 W/H/D부터 detector, N3, initial N3 pose/H, Base-query feature의 initial pose, ROLE head/고정 decode/admission, sparse robust point pose, LOCAL optimizer, H 교체와 detector candidate metadata 보존·반환까지 같은 synchronized interval에 들어간다.

[deployment.py](../../../scripts/research/pallet_sparse_local_line_20261010_v9/deployment.py)는 unchanged V7 `Pipeline(head_arms=('IMAGE_ROLE',))`를 합성한다. 학습 완료의 세 checkpoint 및 source-calibration identity를 검증하지만 모델로 로딩하는 head는 IMAGE_ROLE 하나다. fresh sparse point-only pose를 LOCAL 호출 안에서 계산해 available일 때만 추가 start로 전달한다. frozen `solve_local_packet`을 호출하거나 sealed parent completion을 fresh provenance로 위장하지 않는다. frozen solver의 `SEALED_INITIAL_N3`/`SEALED_AVAILABLE_POINT_ONLY` 문자열은 기존 start source identifier이며 runtime에서는 각각 **이 호출의 fresh N3**와 **이 호출의 fresh point-only pose**라고 별도 필드에 연결한다. 둘 다 residual prior 또는 측정 관측은 아니다.

실제 partial line과 point의 source-edge 소비, H 초기2D fit 제외, rank/다중해/soft_l1 cost, NEW 뒤 H 재투영과 재fit 금지 규칙은 frozen 정확도 solver 그대로다. 결측을 native로 채우지 않는다. 원행의 full candidate/J/rank/start/factor witness와 fresh point bank ledger를 보존한다. 수치 LOCAL 산출은 정확한 자세나 global uniqueness의 증명이 아니다.

예정된 정상 전체 실행 카운트는 detector600, initial pose600, N3 route450, ROLE head/feature Base pose300, fresh sparse robust point path300, local bank/path150이다. detector model 내부 initialization은 로딩 단계 호출과 별도로 기록한다. 독립 head pre/post hooks, 여섯 OpenCV primitive entry, SciPy 실제 진입/완료/예외 및 residual/J callback을 route마다 실제 기록한다. local optimizer 호출은 start/cache/관측 부족에 따라 달라지며 고정 수량을 지어내지 않는다. CUDA synchronize stage marker와 full interval은 실제 측정하고 mean/sample variance/std/median/P90/max를 측정 원행에서 계산한다.

[runtime.py](../../../scripts/research/pallet_sparse_local_line_20261010_v9/runtime.py)의 `freeze`, `preflight`, `measure`를 순서대로 사용한다. 환경 변수와 `--source-root`/`--baseline-root`는 REPRODUCE의 서로 다른 원래 checkout을 사용한다. `--accuracy-output`은 완료된 V9 정확도 자료이고 `--parent`/`--point-parent`는 unchanged V7/V8이다. `--protocol`은 accuracy protocol, `--runtime-protocol`은 새 자체 runtime protocol이므로 혼동하지 않는다. source image의 hash, K/registry dimension과 detector candidate metadata를 확인한다. 관측·pose/reference parity는 timed call의 **완전한 반환 뒤** 수행하고 참조 좌표·자세·GT를 predict input으로 전달하지 않는다.

```bash
python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.runtime freeze \
  --accuracy-output "$v9_output" --output "$v9_output" --protocol "$v9_protocol" \
  --parent "$v7_input" --point-parent "$v8_input" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.runtime preflight \
  --accuracy-output "$v9_output" --output "$v9_output" --protocol "$v9_protocol" \
  --parent "$v7_input" --point-parent "$v8_input" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.runtime measure \
  --accuracy-output "$v9_output" --output "$v9_output" --protocol "$v9_protocol" \
  --parent "$v7_input" --point-parent "$v8_input" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"
```

독립 quiet preflight와 model 로딩 전/측정 중/완료 뒤 CPU/GPU workload·온도 snapshot을 남긴다. 경쟁 작업·thermal·parity·cleanup 실패시 실제 prefix, active fresh packet과 receipt를 보존하고 official statistics를 무효화한다. 기존 파일을 덮어쓰거나 자동 retry하지 않는다. completed latency는 initial/feature/point/local fit을 포함한 fresh 전체 경로이며 accuracy의 replay wall이나 기존 시간 합산은 쓰지 않는다. 원래 영상·가중치·사용자 작업본·게시 V7/V8·frozen V9 core는 보호 snapshot으로 전후 검증한다. 공개 독립 runtime 검산은 별도의 code/input protocol을 고정해 실제 저장 원행과 호출량·시간 산술을 확인해야 한다. 그 검산은 기록된 resource와 실행 receipt의 일관성을 확인하며 과거 GPU 수행의 독립적인 인증은 아니다.
