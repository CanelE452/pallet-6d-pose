# SOURCE TRAIN 라벨 의미와 coverage

공개 SOURCE TRAIN 원frame은155개지만 weak44는3frame의 원case8개를 반복한 것이다. 나머지152개가 RGB 부재나 불가판정인 것은 아니다. 기존 center-owner 보조 라벨을 intended-part/native D/E로 바꾸지 않았고, 고정된 소수 관찰과 paired 개입 가능성이 현재 weak 감독 범위를 제한했다.

## 의미를 나누기

|감독|실제 의미|의미하지 않는 것|
|---|---|---|
|Point7040 / loss6096|실제 입력영상 own-scene의 선언 native→canonical point0. target-aloneD이고 combinedD/E인 저장 기하 상태|q픽셀 owner, region 상태, real human native gold|
|Weak44|공개TRAIN 보이는 local-feature 두 ROI를 덮거나 보존한 post-sampler96 개입|정확한 native point0, 3D외부가림·physicalgold|
|PUBLIC old5 owner|239known rows/81frame의 query-center TARGET/FOREIGN/BG 약한 관찰 보조|intended part/native visibility D/E·quality|

Point target은 원q와 같거나 특정거리 안이어야 한다는 규칙이 없다. `47.5+96*(reference-q)/L`이 crop canvas[-.5,95.5)에 들어오는지 등을 확인한다. 실제 pure helper에서 regionU여도 pointD가 허용되고, q로부터 output20pixel 떨어진 reference도 허용됨을 가상검사했다. 이는 라벨 오류를 뜻하지 않는다. 입력에 그 점/부위의 판별 가능한 RGB 근거가 어떻게 들어오는지는 별도의 SOURCE 관찰 질문이다.

Point의 source_image scene과 BEFORE-anchor q는 분리된다. q/L/native/availability와 initializer를 그대로 두며, 저장 reference는 target 적격성·표시 진단에만 쓰고 모델 특징이나 recenter에 쓰지 않는다.

## SOURCE 분모

- Point: TRAIN110bases×4variants=440장면, own3520native 상태 → 7040입력 대응. Own combined D2957/E123/S440; mapped D5914/E246/S880. Loss-eligible6096 D5853/E243, UNKNOWN-loss944. 원inputU own32/mapped64도 보존한다.
- 원 point와 region의 marginals는 다르다: region D1639/E115/S440/U1326. 이 검토는 개별 joint를 재계산하지 않았고, regionU를 pointU로 바꾸지 않는다.
- 공개 raw pool1480행은185frame 컨테이너명이다. Immutable TRAIN은155frame/1240native rows/inputU161/103origin-graph components이다. Prefix는65/38/24/28개다. 서로 다른 frame key와 graph component는 물리적 독립 사례 인증이 아니다.
- Weak44: factual8+SHAM18 D, COVER18 weakE. 원case8은 native0×2,1×3,2×1,3×1,4×1이다. native5/6/7은 선택8에서 없으나 이것만으로 upstream SELF 필터 이후 필요한 D/E coverage 부족을 단정하지 않는다.
- Source native6/7 own point E는1/2개이며 동일 point의 두 input-view 대응에서2/4행이 된다. canonical index와 native slot을 혼동하지 않는다.
- Weak pair는10 selected YCB-V TRAIN donor로80개를 시도해18개만 구성됐다(6case,4유효donor; object1/7/21/9). 62geometry-exhausted와 factual-only2case를 보존한다. YCB-V는foreign donor이고 pallet target dataset이 아니다.

Synthetic material은 PROCEDURAL_WOOD1696/DARK_POLYMER1824 own-row이고 aspect는 RECTANGULAR1760/SQUARE1760이다. Material·aspect·geometry는 별도 factor이다. Weak44 metadata에는 검증된 material/nativecanonical crosswalk가 없어 이를 natural WOOD/SQUARE 전체 감독으로 세지 않는다.

## 왜 나머지 frame을 아직 direct 약감독으로 세지 않는가

Weak bridge descends from fixed6-frame48slot source pilot ->8 independent visible-feature witnesses on3frames ->18/80 geometry-feasible pairs; feature/part-to-intended-native evidence and legal paired construction, not perfect-mesh certification or absent155 RGB, limit current direct weak target scope.
152개에는 미관찰 frame뿐 아니라 이미 관찰했어도 부위대응/관찰합의/개입요건을 갖추지 않은 경우가 섞인다. 이 검토에서 각152개를 개별 불가판정한 것은 아니다. Old PUBLIC/CF는 owner 보조로 사용될 수 있으나 ownerF나 BG가 곧 queried native external E라는 뜻은 아니다. PUBLIC은 prior owner observation이며, SYNTH의 central4 raster owner mass와도 권한이 다르다.

## 최소 다음 SOURCE-only 단계

새 weak donor나 새 학습팔보다 먼저, 기존155 TRAIN에서 이미 관찰된 frame을 제외하고 네 prefix당1개씩 새4frame을 stableSHA로 사전 고정해 전체32native를 원fullimage+crop/q8로 독립 관찰하는 pilot을 권고한다. 원q/L/native/U/96 sampler를 유지하고 UNKNOWN·불일치를 남긴다. q포함/중심/완벽한 물리점 인증을 gate로 만들지 않는다. selected part/visible feature와 semantic state, location 상태를 분리하며 center-owner를 D/E로 복사하지 않는다.

별도의 작은 합성 TRAIN native8 고정 표시에서 원q·저장 point0 reference·실제 보이는 부위 위치를 나란히 확인하면 point 목표와 feature 목표의 일치/차이/UNKNOWN을 구분할 수 있다. 저장 point를 입력으로 넣거나 GT recrop·새 ray/projection·PnP를 하지 않는다. A의 frozen SOURCE modality ablation은 입력조건 의존성을 측정하지만 이 semantic 대응을 단독 증명하지 못한다. OOD 개입에서 나온 변화도 실사 실패 원인의 증명으로 쓰지 않는다.

이 권고는 coverage 준비이며 새 라벨 admission·학습·보호 평가 튜닝·모델 채택이 아니다. V162/165의 pretrained-qpatch 재시도도 제안하지 않는다.

## 근거와 다음 loader 경계

- [원native coverage CSV](SOURCE_TRAIN_NATIVE_COVERAGE.csv)
- [본 audit JSON](REVIEW.json)
- [V212 target contract](../../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/source_target_contract.py)
- [V212 trainer](../../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/train_dense_point_weak_bridge.py)
- [V211 TRAIN teacher summary](../../query_visibility_dense_source_native_point_teacher_20261025_v211/dense_source_point_teacher_v1/SUMMARY.json)
- [Point admission](../../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/ROOT_SOURCE_PROCEDURAL_POINT_TEACHER_ADMISSION.json)
- [Weak44 metadata](../../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json)
- [SOURCE TRAIN155 aggregate](../../query_visibility_source_train_intervention_query_alignment_20261025_v204/public_source_coverage_b/SUMMARY.json)
- [PUBLIC owner cache code](../../query_visibility_local_support_raw_head_20261024_v181/raw_cache_preparation/build_source_raw96_cache.py)

원 PUBLIC metadata와 V194 role sidecar는 JSON의 INERT 포인터다. 향후 loader는 먼저 immutable roleTRAIN 행ID를 고정한 뒤 해당 source metadata와 선택된 원RGB/crop binding만 읽어야 한다. 이 audit는 이미 TRAIN-only인 aggregate와 weak metadata만 계산했으며 공유 컨테이너의 HOLDOUT 값·원PNG/NPZ/GT/weight/header를 열지 않았다.
