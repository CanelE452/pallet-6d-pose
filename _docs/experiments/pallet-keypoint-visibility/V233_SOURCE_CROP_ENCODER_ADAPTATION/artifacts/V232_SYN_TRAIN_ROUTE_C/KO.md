# V232: C65 SYN TRAIN의 기존 DATA 경로

코드·봉인 metadata상 **순수 component-TRAIN/NOMINAL 7040행 packet**이 있다. C65 actual entry(91035379…)는 V219/source_fit_rows_v1/SYNTH_AUG65536_INPUTS.npz(eb8b2401…,23,487,096B), TARGETS.npz(5c3cb88e…,25,368B), FILTERED_METADATA.jsonl(57190254…,10,067,991B)을 pin한다. source_fit_rows COMPLETE85ae1bd4…와 extractor7376b180…는 selected row_ids가 point7040 전체와 같고 모든 fit mask가 참임을 봉인한다. 이번 확인에서 파일 본체·NPZ stat/hash/header나7040행 본문은 읽지 않았다.

이7040 안의 direct point loss는 D5853/E243=6096행이며944행은−1로 남는다(원 inputU64 포함). **별도6096-only 물리 packet은 검토한 closure에 선언되지 않았다.** 원65536 packet의 historical SOURCE_TRAIN_only는 이후 component split과 다르다. saved LINEAGE_QA는 원 component TRAIN56320/HOLDOUT9216, TRAIN nominal7040/non-nominal49280을 구분한다. 원mixed 전체를 새7040 packet 권한으로 읽으면 안 된다.

입력 member는 RGB96 uint8[7040,96,96,3], valid bool[7040,96,96], 원 row_id/global/native/q/L/availability다. 형상은 extractor CODE 계약이며 새header검사 결과가 아니다. V183는 원 synthetic scene RGB에서 원 sampler로 crop을 만들었고, 자기 nominal crop/valid는 이전 저장본과 byteparity를 선언한다. SELF는 own anchor, BEFORE_ANCHOR는 원 BEFORE q/L을 실제 variant image에 적용한다. SAM은 원fullpacket에 존재하지만 selected INPUT8에는 복사되지 않았다.

선택7040의 RGB96는 **원 anchor query를 중심**으로 만든 NOMINAL crop이다. 좌표식은 `q + ([u,v]−47.5)·L/96`이며 q는 중앙 네 sample 사이 연속 중심이다. SELF는 own initializer, BEFORE_ANCHOR는 원 BEFORE initializer를 실제 variant 영상에 적용한다. GT reference로 crop을 옮기지 않는다.

filtered metadata는 V194 SPLIT_ROWS의 row_id/META6/cache_index로 component_id와 원 TRAIN/HOLDOUT role을 join한다. C65 fit mask는 TRAIN·nominal·lineage_bound·유효 component·새DEV4 제외를 모두 요구한다. SYN의 새DEV descendant는0이다. 원q8/L8/available8은 anchor initializer에서 오며, condition32의 native8/othermask8/relativeXY16에 GT reference를 넣지 않는다. 원global/cache index는 추출 후에도 유지되므로 새NPZ row j와 원index를 혼동하면 안 된다.

**TARGETS.npz의5member는 direct point6096 라벨이 아니다.** 그것은 기존 visible/foreign support 및 owner mass 축이다. direct label은 V211 ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl(aeb958d6…)와 point_loss_target(d8925908…)에서 만든다. 실제 image의 teacher key=[source_image_original_global_index//8,native_slot]를 쓰고 q-anchor와 분리한다. 저장 records_CF/선언perm의 point0 target-aloneD·combinedD/E를 읽되 region은별도다. finite/available 원qL과 frame/원crop 내 saved reference만 losseligible, 나머지는−1이다. reference는 target 측 대수 locality에만 쓰며 recrop하지 않는다. 실제 사람/실사의 native gold 주장은 아니다.

따라서 strong point0는 **literal q-center 픽셀 owner가 아니다.** own-image `records_CF`의 저장 `target_witness_states[0]`/`combined_witness_states[0]` 상태이며, 같은 record의 projected reference가 원 frame/crop 안인지 검사한다. `47.5+96·(reference−q)/L`가 crop 안이라는 조건은 reference=q라는 조건이 아니다. central-four owner mass는 TARGET5의 별도 축이다.

q/L/native는 INPUT8과 metadata에 있고 q8/L8/availability8은 FILTERED_METADATA의 `context_*`에 있다. 이 filtered schema는 raw_hw나 fullRGB binding을 넣지 않는다. POINT_MAPPING7040은 원q/L과 own-image/anchor ID·reference/locality, `actual_image_source_binding_inert`를 보존하지만 raw_hw/전체q8를 명시 출력하지 않는다. 별도 V183 `SCENE512_ORIGINALS_INPUT_ONLY.jsonl`의 코드 schema가 fullRGB packet/key/decodedSHA/raw_hw와 original8을 연결한다(이 manifest 본문도 이번에 열지 않음).

**C11식 crop 밖 contextRGB는 selected7040 packet만으로 복원할 수 없다.** RGB96/valid/q/L/q8에는 crop 밖 픽셀이 없고 CLS/token도 저장돼 있지 않다. 기존 scene manifest의 정확한 own-image fullRGB packet/member/key·decodedSHA·raw_hw를 별도 미래 scope로 pin해야 한다. 경로가 inert로 있다는 사실과 실제 fullcontext 접근 권한은 구분한다.

다음 최소 접근 대상은 이미필터된 INPUTS, FILTERED_METADATA, point7040 JSONL 및 그 contract/admission/COMPLETE다. old5를 그대로 소비할 경우에만 그 TARGET5가 별도입력이다. 원GT·원mixed·PUBLIC weakDEV·weights·이미지 접근은 이 경로확인으로 허가되지 않는다. 과거 extractor는 별도rootscope로 여섯 원ZIP의opaqueSHA/NPYheader를 검사하고 선택행만수치decode했으며, 제외행 압축해제bytes는opaque통과할수있었다. 이를 “과거HOLD bytes0”로 바꾸어쓰지 않는다.

모든 추가읽기 파일과 mixed 여부, exact path/SHA, shape·join·제외조건은 ROUTE.json에 있다. 실제 새학습/데이터선택/모델설계는 하지 않았다.
