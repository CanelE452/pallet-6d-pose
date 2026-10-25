# Pallox SOURCE 라벨 연결 준비성

원 initializer는 first-argmax detector의 raw q0..7과 원 L을 그대로 저장합니다. centroid8·GT-best·재정렬·recenter는 없습니다. 이것은 채널 순서의 계보이며 Pallox expected part의 의미 대응은 UNKNOWN입니다.

원 `ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl`은 q/L, 입력 지원·reason, split, slot, `native_semantic_state=UNKNOWN`을 담는 입력 전용 schema입니다. crop pack은 같은 800행 순서의 RGB96/valid96/P3 및 원 query lineage입니다. 별도 `ALL800_QUERY_METADATA_AND_CUE_AVAILABILITY_GT_ONLY.jsonl`과 query×cue 4832행에는 지역 teacher가 있으며 입력 전용 파일과 다릅니다. 이 파일들의 행본문은 열지 않았습니다.

기존 paired 교사는 같은 카메라·정적 bin·정합 가정 아래 SAM interior와 mesh/측정 depth의 양성 합치로 V를, 반대 view에서 보인 target보다 destination depth가 충분히 가까운 경우로 E를 정의합니다. 순수 helper는 변경하지 않은 q에서 nearest `floor(q+.5)`의 1픽셀과 별도 5×5 관측을 지원합니다. SELF는 생성하지 않으며 mask0·depth 누락·미확정 geometry는 UNKNOWN입니다. full-negative/O는 비활성입니다.

따라서 주소와 불확실성을 명시한 유한 cue/지역 또는 같은 픽셀의 조건부 V/E weak AUX 근거는 보존할 수 있습니다. 완벽 mesh/native gold가 없다는 이유로 모든 weak SOURCE 사용이 막히는 것은 아닙니다. 다만 crop에 cue bbox가 FULL로 들어온 사실을 native expected-part D/E/SELF나 모든 ROI 픽셀의 정답으로 승격할 수 없습니다. 이 감사는 학습 목적·자료·모델을 선택하지 않습니다.

선언된 TRAIN은 22쌍/44 frames/352 slots이며 원 continuous-available은 297 slots입니다. 기존 readiness의 known-FULL 81(지역 D-only70/E-only11)은 native scalar 라벨 수가 아닙니다. 검토한 seal/protocol은 혼합 100-frame/800-slot 파일만 bind하며, 별도 ONLY_TRAIN44/352 allowlist의 정확 경로·SHA는 확인되지 않았습니다. 전역적 부재를 주장하지 않습니다. 후속 연결에 빠진 것은 원 frame/query ID·혼합 원 ordinal·이미지 binding·원 q/L/native/support·chunk offset을 가진 명시적 TRAIN-only 명단입니다. 현재 권한에서는 혼합 행을 해석하지 않았습니다.

정확 CODE/metadata SHA와 line, 입력/교사 schema 및 미열람 포인터는 `LABEL_READINESS.json`에 기록했습니다. 원 RGB/depth/mask/GT/배열/PT, 보호 행·점수, model import/forward/fit, 재귀 검색, 네트워크와 Git 접근은 0입니다. 허용 intake/protocol의 역사적 split 집계는 접했으나 held/VAL per-row 본문은 열지 않았습니다.
