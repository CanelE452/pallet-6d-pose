# V9 fresh runtime 입력 schema 수정 보충 계약

이 보충 실행은 정확도·관측 선택·optimizer·threshold·시작점·timing 경계·600회 schedule을 바꾸지 않는다. 원본 [runtime.py](../../../scripts/research/pallet_sparse_local_line_20261010_v9/runtime.py), `RUNTIME_PROTOCOL.json`, `RUNTIME_PREFLIGHT.json`, `RUNTIME_CLI_ATTEMPT_A.json`은 보존한다. 첫 실행은 cohort의 `image` 객체를 경로 문자열로 취급해 이미지 decode·모델 생성·timed 호출·STARTED 이전에 실패했다. 해당 실제 실행량은 모두 0이며 이후 성공을 원본 실행의 성공으로 표시하지 않는다.

새 [runtime_review.py](../../../scripts/research/pallet_sparse_local_line_20261010_v9/runtime_review.py)는 immutable 원본에서 아래 schema 경계만 수정한 별도 supplement다.

1. label-only COHORT를 직접 inference frame으로 쓰던 부분을 unchanged `V7.cohort_frames(args)`로 연결한다. 그 함수는 원래 prediction-only `INPUTS.json`의 319장 authority와 selected245 cohort의 ID·session·image SHA·분할 및 registry dimensions를 검증한다. 보충 adapter는 245개의 image path도 cohort와 일치하는지 확인하고, `id/session/object_type/image/image_sha256/raw_hw/K/xyz/points.BASE/points.N3_SUBPIX`만 보존한다. 원래 helper가 반환하는 그 밖의 필드는 전달하지 않는다. 저장 Base/N3 좌표는 반환 뒤 parity에만 쓰며 `predict`에는 RAM RGB, K, xyz와 `id/session/object_type`만 전달한다.
2. legacy `_pose_parity`가 모든 available pose에 `reprojection_px`를 요구하던 검사 대신, 실제 공통 `R_cf/R_physical/centroid/cf_extents` 및 availability·dimension hypothesis를 검사한다. 실제 존재하는 projection·SSE·soft-loss·residual 필드는 양쪽 존재 여부와 값도 검사한다. LOCAL solver에 없는 `reprojection_px`를 만들어 넣지 않는다. 이 검사는 complete timed return 뒤에 수행하며 허용 오차는 기존 `1e-7`, 상대 오차 0이다.
3. 원래 fresh detector의 full prediction을 반환 뒤 원행에 별도로 저장한다. 최종 prediction과 selected score/class/box/confidence·center·비선택 candidate 보존을 독립 검산할 증거이며 detector나 fitting을 다시 실행하지 않는다.

모든 새 artifact는 `RUNTIME_REVIEW_*` 이름을 쓴다. 원본 code/protocol/preflight/실패 receipt와 새 code/계약, authoritative helper·INPUTS/cohort·source/checkpoint/calibration·완료 정확도 SHA를 새 `RUNTIME_REVIEW_PROTOCOL.json`에 고정한다. 이 계약을 작성한 시점에는 보충 freeze·model 실행·timed 호출을 아직 수행하지 않았으며 실제 상태는 별도 protocol/STARTED/receipt가 결정한다. 원래 실패는 `original_runtime_history`에 남는다. 자동 retry는 하지 않는다.

실행 순서는 같은 명시적 split roots를 사용한 `runtime_review freeze`, 독립 quiet `preflight`, `measure`다. 각 명령은 `--protocol`에 V9 accuracy protocol, `--accuracy-output`과 `--output`에 V9 자료, `--parent`에 V7, `--point-parent`에 V8을 전달한다. 새 runtime protocol 기본값은 V9 DOC의 `RUNTIME_REVIEW_PROTOCOL.json`이다. 출력은 `RUNTIME_REVIEW_PREFLIGHT.json`, `RUNTIME_REVIEW_STARTED.json`, `RUNTIME_REVIEW_ROWS.jsonl.gz`, `RUNTIME_REVIEW.json`이며 경쟁 작업·thermal·parity·cleanup 실패시 새 interrupted prefix/packet을 보존한다. 원본 실패 artifact를 덮어쓰지 않는다.

네 route 각각 20 warmup+5×26 measured, 전체600 fresh 호출과 실제 detector/N3/head/point bank/LOCAL/primitive/optimizer 진입 카운트, 초기 pose·Base feature pose·robust point solve·LOCAL·H 교체까지 포함한 전체 synchronized latency 계약은 원본 [RUNTIME_CONTRACT_KO.md](RUNTIME_CONTRACT_KO.md) 그대로다. 새 측정 전에 모델과 이미지 작업을 포함한 독립 quiet preflight가 필요하다. 저장 accuracy는 post-return parity용이며 GT·저장 좌표·pose가 inference input이 아니다. 별도 독립 runtime checker의 자체 freeze와 원행 검산은 측정 완료 후 수행한다. timestamp/hardware의 독립 인증과 기록된 duration·호출량·metadata 검산을 구분한다.
