# V233 crop 데이터 조립 코드

현재 CODEONLY이며 실제 SOURCE·RGB·NPZ·가중치·원GT를 읽지 않았다. Root의 true entry 이후만 실행한다.

`python -B prepare_crop_samples.py --entry /absolute/ROOT_ACTUAL_ENTRY.json`

`ENTRY_TEMPLATE.json`의 `approved_source_payload_read_and_crop_assembly`를 실제 Root entry에서 true로 바꾸고 `code`/`protocol`/13개 `inputs` binding을 정확히 유지한다. `ycb_rgb_read_allowlist`에는 TARGET8000의 원 D/E와 연관된 image binding 집합만 넣는다. sealed700 whitelist와 정확히 일치하는 subset인지 검증하고 SHA 검증 후 PIL RGB로 연다. 실제 입력·whitelist·이미지의 경로를 이번 준비에서 열거하지 않았다. `output_dir`은 `V233_ROOT_CROP_ADAPTATION/crop_samples_v1` 고정·새 디렉터리다. 실패 시 기존 출력을 보존한다. 자동 재시도는 없다.

SYN7040은 원 component TRAIN/NOMINAL/lineage와 META6/actual own-image teacher key를 검증한다. 기존 point_loss_target은 D5853/E243=6096, 나머지944는−1이다. q/L/native는 유지하며 condition32는 C65 원 함수와 row hash를 맞춘다. 원 initializer q8만 입력이며 reference·target·ROI는 feature에 넣지 않는다. SYN 내부 role은 사전 V233 component hash, YCB는 사전 scene0..4 FIT/6,7 DEV다. cross-family donor 독립성은 UNKNOWN이다.

`ALL15040_ROWS.jsonl`은 SYN7040+YCB8000의 원 D/E/S/U·−1·role·q/L·availability·reference 진단을 모두 보존한다. YCB에는 r 좌표가 없으므로 distance는 null이고 추가 GT를 따라가지 않는다. YCB unknown 행의 validity는 원 raw_hw/q/L 수학이며 해당 RGB를 추가 열지 않는다.

학습용 `samples.npz`는 known7617(SYN6096+YCB1521) 전수다. availablefalse도 삭제하지 않는다. keys는 `full_view`,`central_view` int64(N), `fractions` float32(N,2:full96/centre32), `central4` bool(N), `condition` float32(N,32), `target` int8(N), `family` int8(N:0SYN/1YCB), `fit`,`available` bool(N), `row_id` Unicode(N)다. view 인덱스는2j/2j+1이고 centre32는[32:64,32:64]다. availability는 original availability & centre[47:49] allvalid & anyfullvalid다. target 권위와 crop 사용 가능 여부를 혼동하지 않는다.

같은 순서의 `raw_rgb96.npy` uint8(N,3,96,96), `raw_valid96.npy` bool(N,1,96,96), `SAMPLE_DIAGNOSTICS.jsonl`(원q/reference/픽셀SHA), `SUMMARY.json`,`COMPLETE.json`을 쓴다. SYN HWC→CHW, YCB 기존V90 F64 crop만 사용한다. 입력 row equality/condition hash, 저장 pixel/valid row equality와 output SHA를 검사한다. DINO·resize·정규화·특징·모델·fit은 이 코드 역할 밖이다.
