Selected TRAIN110 GT JSON의 ancestry 실제 확인을 마쳤다. 원 scope의110개 SHA를 각각 확인한 뒤24,496,847B를 parse했으며440개 own scene/variant와152개 normalized key/type를 조사했다. historical all4096/512나 새 pointer는 열지 않았다.

110개 모두에 **배경 asset의 원 dataset/scene/frame 계보, donor의 원 dataset/scene/frame 계보, renderer CODE/config/catalog binding은 기록돼 있지 않았다.** 모든 문자열에서 YCB/train_real 표기를 찾았지만0개였고, YCB000006/000007의 literal assetidentity도0개였다. 이는 이110개 JSON이 그 계보를 제공하지 않는다는 직접 증거이지 실제 RGB에 해당 asset이 없다는 증거는 아니다. 따라서 cross-family overlap/disjoint 판정은 UNKNOWN으로 유지한다.

원 generic keyword SUMMARY의 background110은 `/rows/*/nearest_pixel_diagnostic_CF/*/nearest_background` 한 필드를 세어 생긴 값이다(3520scalar/110parents). 이것은 픽셀 진단이며 배경 원본 계보가 아니다. 원 추출 결과는 그대로 보존하고 ASSET_LINEAGE_CLASSIFICATION_ADDENDUM.json과 새 parent표에서 이 구분을 명시했다. GT·원 추출물 수정이나 GT 재조회는 없었다.

실제로 기록된 pathlike pointer는 각440개 variant의 `geometry`, `foreign_geometry`, `raw_buffers`3종, 총1320참조다. foreign_geometry는 가림물 geometry의 저장 NPZ pointer를 제시하지만 donor의 원 dataset/장면 계보를 제시하지 않는다. 이 문자열은 INERT_POINTER_STRINGS.json에 보존했으며 존재확인·SHA계산·body조회0이다. geometry/pose/native point 숫자로 새 데이터·모델·분할을 선택하거나 기하를 재계산하지 않았다.

base_index·scene_index·variant는440행 identity표에 원대로 남겼다(BEFORE/COVER/NEAR_SHAM/FAR_SHAM). 이 GT schema에는 anchor identity 필드가 없어, 이를 원 q-anchor 계보로 대신 해석하지 않는다. 현재 SOURCE role·원q/L·native·target·문턱은 바꾸지 않았다.

[110 parent표](ALL110_PARENT_ASSET_LINEAGE.csv), [정확 분류 addendum](ASSET_LINEAGE_CLASSIFICATION_ADDENDUM.json), [전체 key/type](actual_read_v1/ALL_SCHEMA_FIELDS.csv), [own scene/variant identity](actual_read_v1/ALL_VARIANT_IDENTITY_AND_ANCESTRY.json), [inert pointer](actual_read_v1/INERT_POINTER_STRINGS.json), [원 실제 읽기 manifest](actual_read_v1/COMPLETE.json).
