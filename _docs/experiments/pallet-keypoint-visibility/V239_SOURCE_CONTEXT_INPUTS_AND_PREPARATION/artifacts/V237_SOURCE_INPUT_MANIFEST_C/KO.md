# V237 SOURCE 入力 whitelist・metadata 정합

새 입력/모델/분할을 고르지 않고 기존 SOURCE 전체15,040행과 known7,617행을 원 q8 context 및 실제 RGB 선언 binding에 연결했다. raw NPZ/PNG의 parent·leaf stat/hash/header/decode는0이며 Root pixel scope 이전에는 이 manifest도 읽기 권한을 부여하지 않는다.

SYN은440개 actual scene의 raw NPZ path/SHA를 RGB_uint8 한 member로만 닫았다. YCB는 input1000의 image metadata에서700개 exact RGB path/SHA를 닫았다. 실제 영상 크기는 SYN에서미확인, YCB는기록된480×640이며 이번에 픽셀로재확인하지않았다. 원7040 source metadata/mapping과 V236표를 연결했으며 raw source pointer는따라가지않았다.

|가족|전체 FIT / DEV|known FIT / DEV|원 q available / 불가|기존 crop available / 불가|
|---|---:|---:|---:|---:|
|SYN|5504 / 1536|4752 / 1344|6976 / 64|6976 / 64|
|YCB|5600 / 2400|1124 / 397|7723 / 277|7721 / 279|

known7,617은 원 sample_index0..7616·row_id 순서를 진단 sidecar와 전수 일치시켰다. target−1의7,423행은 전체표에 유지하며 모델 U와 구분한다. YCB의 추가 unsupported2행도 원 original_query_available과 crop available을 따로 보존하여 삭제/라벨변환하지 않는다. 원 TRAIN role와 이미 고정된 internal FIT/DEV를 복사했으며 새 split을 만들지 않았다.

원 context는 SYN770개(actual scene+원q8계보)와 YCB1000개(instance)로 분리 저장한다. 숫자가같은q8라도 원 identity를 함부로합치지않는다. SYN2640행에서 actual image와query anchor가다르며, box는해당원q8·RGB는actual scene를유지한다. V107 box는 finite/non[-1,-1] q8 min/max이며 availability mask를 적용하지않았다. SYN unavailable이지만finite/non-sentinel q8slot 반복512회도 그대로보존했다.

YCB activeq는initial_points[logical_aux_slot], L은roi_side와전수같고 저장 query_box는같은q8 min/max와exact일치한다. logical aux는native가아니며native UNKNOWN을유지한다. SYN activeq/L/availability는원q8/L8/available8의native slot과일치한다. GT ownership/pose/reference를context입력으로쓰지않았다. 다만YCB q자체의GT-jitter emulation 출처를숨기지않는다.

[입력 manifest](INPUT_MANIFEST.json) · [context1770 input-only](ORIGINAL_CONTEXTS1770_INPUT_ONLY.jsonl) · [전체15040행 연결](ALL15040_ROW_INPUT_JOIN.csv) · [원known7617 sample 연결](KNOWN7617_SAMPLE_INPUT_JOIN.csv)

표의target/original_state 열은원감독보존용이며forward feature가아니다. 실제 RGB/header 검증·특징추출·학습·current평가·성능성공·모델채택은이번작업에없다.
