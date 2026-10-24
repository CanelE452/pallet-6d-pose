# V216 SOURCE 부위·query-support 두 관찰 조인

고정 SOURCE TRAIN12프레임/native96의 두 독립 봉인 약관찰을 원ID/qL/U11 그대로 조인했다. 아래 D/E/SELF와 witness 수는 원관찰/기계적 합의후보이며 모델 성능이나 최종 semantic admission이 아니다. 같은 부위·query addressability의 root 검토는 pending이고 학습 admission은 false다.

원role TRAIN과 internal SOURCE_FIT64/SOURCE_DEV32를 분리한다. DEV는 미래 모든 loss/pool/normalization 및 sibling/pairedvariant 학습에서 제외한다. 이전 appearance·oldaux 노출이 있어 freshblind/OOD 또는 물리적 독립 검증이 아니다.

원 U11과 의미 UNKNOWN은 다른 축이다. 배경·다른 목재/물체·qOFF를 native E/D로 복사하지 않는다. ROI는 교정 keypoint/physical gold가 아니다. 두 원 ROI와 sealed wholeJSON을 보존하고 exactOR를 rowmajor0/1 mask SHA/area로 기록한다. legacy bbox는 표시용일 뿐 alpha/proof mask가 아니다.

|cohort|N|원U|local witness 합의후보|join visibility 후보|
|---|---:|---:|---:|---|
|ALL|96|11|62|{"UNKNOWN": 85, "D": 11}|
|SOURCE_FIT|64|9|40|{"UNKNOWN": 55, "D": 9}|
|SOURCE_DEV|32|2|22|{"UNKNOWN": 30, "D": 2}|

|cohort|reader / raw후보|D / E / SELF / UNKNOWN|ON / NEAR / OFF / UNKNOWN|
|---|---|---|---|
|ALL|A|14 / 0 / 0 / 82|33 / 28 / 22 / 13|
|ALL|C|17 / 0 / 0 / 79|33 / 31 / 15 / 17|
|ALL|JOIN_RAW_CANDIDATE|11 / 0 / 0 / 85|24 / 17 / 11 / 44|
|SOURCE_FIT|A|12 / 0 / 0 / 52|22 / 19 / 12 / 11|
|SOURCE_FIT|C|15 / 0 / 0 / 49|22 / 21 / 10 / 11|
|SOURCE_FIT|JOIN_RAW_CANDIDATE|9 / 0 / 0 / 55|16 / 11 / 8 / 29|
|SOURCE_DEV|A|2 / 0 / 0 / 30|11 / 9 / 10 / 2|
|SOURCE_DEV|C|2 / 0 / 0 / 30|11 / 10 / 5 / 6|
|SOURCE_DEV|JOIN_RAW_CANDIDATE|2 / 0 / 0 / 30|8 / 6 / 3 / 15|

|native|FIT N / 원U|DEV N / 원U|ALL N / 원U|
|---:|---|---|---|
|0|8 / 1|4 / 1|12 / 2|
|1|8 / 3|4 / 0|12 / 3|
|2|8 / 3|4 / 0|12 / 3|
|3|8 / 2|4 / 1|12 / 3|
|4|8 / 0|4 / 0|12 / 0|
|5|8 / 0|4 / 0|12 / 0|
|6|8 / 0|4 / 0|12 / 0|
|7|8 / 0|4 / 0|12 / 0|

[전체 조인96](ALL96_JOINED_CANDIDATES.json) · [두ROI exactOR sidecar](ALL96_EXACT_WITNESS_MASK_SIDECAR.json) · [reader/role/native/축 count CSV](SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv) · [모든 축 차이 CSV](ALL96_AXIS_DIFFERENCES.csv) · [전체12 overlay gallery](PORTABLE_GALLERY_KO.md)

원 관찰 각축/증거와 차이는 삭제하지 않았다. ROI overlap/같은 category는 같은 semantic feature 증명이 아니며, helper proposal을 승인된 학습 target으로 승격하지 않는다. valid96를 확인하지 않았거나 exactOR가 invalid를 건드리면 proof 미완료/실패를 그대로 기록하고 ROI를 줄이거나 이동하지 않는다.

## 고정 설명 위치0/6/10

다음3장은 사전 지정 설명 카드이며 대표성·prevalence나 점수 기반 실패선별이 아니다.

![고정 위치0](FIXED_FRAME_00_TWO_READER_OVERLAY.png)

![고정 위치6](FIXED_FRAME_06_TWO_READER_OVERLAY.png)

![고정 위치10](FIXED_FRAME_10_TWO_READER_OVERLAY.png)

[A sealed96 observations](../actual_sameframe_observations_a_v1/ALL96_OBSERVATIONS.json) · [A 실제view receipt](../actual_sameframe_observations_a_v1/ACTUAL_IMAGE_VIEW_RECEIPT.json)

[C sealed96 observations](../actual_sameframe_observations_c_v1/ALL96_OBSERVATIONS.json) · [C 실제view receipt](../actual_sameframe_observations_c_v1/ACTUAL_IMAGE_VIEW_RECEIPT.json)

[ROOT JSON](ROOT_RESULTS_KO.json) · [HTML](ROOT_READOUT_KO.html) · [SUMMARY](SUMMARY.json) · [조인 코드](../join_preparation_b_v2/join_and_render_saved_two_reader_observations_v2.py) · [조인 protocol](../join_preparation_b_v2/PROTOCOL_BEFORE_SEALED_OBSERVATIONS_OR_GENERATED_PNG_BODIES.json)

실제 entry와 양쪽 wholeJSON SHA는 COMPLETE/sidecar에 봉인했다.
