# V192: source 끝부위 감독의 범위와 SHAM 배치 진단

source 감사와 단 한 번의 반대 SHAM bbox 진단을 완료했다. 기존 지원은 합성97/1024·공개 weak5/66 그대로이며, 반대 bbox 가능26건은 새 유효 가림쌍이나 학습 승인26건을 뜻하지 않는다.

## 이미 확인된 source 범위

표의 geometry 실패 열은 저장된 **최상위 상태명** GEOMETRY_FAILURE_RETAINED의 수다. 합성에서는 geometry 실패 사유도 최상위 UNKNOWN_RETAINED 안에 저장되므로, 합성의 0은 geometry 실패가 없다는 뜻이 아니다. 세부 사유와 중복 집계는 아래 전체 분모 CSV에 보존했다.

|source|원 시도|기존 지원 triplet|기존 geometry 실패|기존 UNKNOWN|
|---|---|---|---|---|
|SYNTHETIC|1024|97|0|927|
|REAL|66|5|43|18|


직접 정답은 D204/E102, 총306행이며 물리 입력2792행 중 UNKNOWN2486행을 보존한다. 전체 시도는 합성128bases×8native=1024와 공개22사례×3donor=66이다. known은 합성97triplet/47bases, 공개5triplet/2frames다. 65536 증강 crop을 독립 영상65536장으로 세지 않는다.

|native|합성 시도|합성 지원 triplet|공개 weak 시도|공개 weak 지원 triplet|
|---|---|---|---|---|
|0|128|29|15|5|
|1|128|24|15|0|
|2|128|17|6|0|
|3|128|23|3|0|
|4|128|2|6|0|
|5|128|2|15|0|
|6|128|0|6|0|
|7|128|0|0|0|


[모든 native·재질·aspect·geometry/foreign family와 실패/UNKNOWN 분모](source_metadata_diagnostic_v1/POPULATION_AND_STORED_STATUS_BY_STRATUM.csv) · [전체 lineage](source_coverage_readiness_b/SOURCE_TRAIN_LINEAGE_COVERAGE.json)

## 원96 입력에 남은 저장 신호량

합성 accepted97의 end-related sample 수는 16~49/9216(약 0.17~0.53%)다. 이는 끝 후보에 양의 보간 기여가 있는 output sample 수이며 RGB 차이 크기·시각적 판독성·독립 정보량이 아니다.

|source|저장 수량|분모|0|1|2~4|5~8|9~16|17~32|33~64|65이상|missing|
|---|---|---|---|---|---|---|---|---|---|---|---|
|SYNTHETIC|end_related96_samples|97|0|0|0|0|3|82|12|0|0|
|SYNTHETIC|COVER_changed_RGB96_samples|97|0|0|0|0|3|82|12|0|0|
|SYNTHETIC|SHAM_changed_RGB96_samples|97|97|0|0|0|0|0|0|0|0|
|REAL|end_related96_samples|5|0|0|0|0|0|0|0|5|0|
|REAL|COVER_changed_RGB96_samples|5|0|0|0|0|0|0|0|0|5|
|REAL|SHAM_changed_RGB96_samples|5|0|0|0|0|0|0|0|0|5|


공개 weak accepted5는 end-related sample 수만 정량 저장했고 COVER 변화/SHAM 불변은 boolean proof다. 그 boolean을 변화 sample 수나 RGB 강도로 바꾸지 않는다. missing은0으로 채우지 않았다. [전체 고정 bins](source_metadata_diagnostic_v1/STORED_END96_FIXED_BIN_DISTRIBUTIONS.csv) · [공개 변화 boolean 전체 분모](source_metadata_diagnostic_v1/STORED_REAL_CHANGE_BOOLEAN_COUNTS.csv)

## 새 계산: SHAM x방향만 한 번 반전

|기존 상태|원 시도|반대 bbox 가능|반대 bbox 불가|관측 UNKNOWN 유지|
|---|---|---|---|---|
|GEOMETRY_FAILURE_RETAINED|43|22|21|0|
|SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED|5|4|1|0|
|UNKNOWN_RETAINED|18|0|0|18|


26건은 기존 geometry 실패22건과 기존 지원4건이다. 기존 지원1건 Q1088_D2는 반대 bbox가 불가능하다. 기존 UNKNOWN18건은 None/UNKNOWN 그대로다. 실패43건 전체와 과거 보고서의 newly-entered33 subset을 혼동하지 않는다.

COVER/donor/scale/후보/witness/원 q·L·native는 고정하고 기존 정수 dx의 부호만 뒤집었다(dy=0 확인). 모든 bbox의 frame 내부 여부, 끝 enclosure와 SHAM의 비교차, 동일 witness의 원96 positive tap 참여·양 bbox 밖 보존, SHAM의 원96 sample 기회만 확인했다. bbox 가능은 실제 donor alpha가 후보를 덮는지나 RGB 변화·same-part 보존을 인증하지 않는다.

[전체66 신규 bbox 진단](source_metadata_diagnostic_v1/ALL66_REAL_BBOX_ONLY_DIAGNOSTICS.jsonl) · [중복 가능한 원인별 집계](source_metadata_diagnostic_v1/REAL_OVERLAPPING_GEOMETRY_COUNTS.csv)

## 실제 사진이 아닌 raw-coordinate 도식4개

선택은 source 결과 후·도식 전 고정했다. 각 지정 집단에서 원래 source row 순서의 첫 사례만 사용하며 빈 집단은 대체하지 않는다. 표시 확대는 geometry diagram 축 범위일 뿐 입력 crop/원 q/L/GT를 바꾸지 않는다.

### OLD_GEOMETRY_FAIL_OPPOSITE_TRUE

기존 geometry 실패 중 반대 SHAM bbox 가능인 첫 사례. 실제 alpha·RGB·유효 학습쌍은 검증하지 않았다.

![geometry schematic 1](source_geometry_schematics_v1/GEOMETRY_01.png)

[SVG](source_geometry_schematics_v1/GEOMETRY_01.svg)

### OLD_GEOMETRY_FAIL_OPPOSITE_FALSE

기존 geometry 실패 중 반대 SHAM bbox도 불가능한 첫 사례. 다른 배치를 탐색하거나 witness를 바꾸지 않았다.

![geometry schematic 2](source_geometry_schematics_v1/GEOMETRY_02.png)

[SVG](source_geometry_schematics_v1/GEOMETRY_02.svg)

### OLD_SUPPORTED_OPPOSITE_FALSE

기존 지원 쌍이지만 반대 SHAM bbox는 불가능한 첫 사례. 원래 지원 판정은 그대로다.

![geometry schematic 3](source_geometry_schematics_v1/GEOMETRY_03.png)

[SVG](source_geometry_schematics_v1/GEOMETRY_03.svg)

### FROZEN_WEAK_UNKNOWN

기존 weak UNKNOWN의 첫 사례. q와 원 frame만 표시하며 후보 끝·witness·대체 배치를 만들지 않았다.

![geometry schematic 4](source_geometry_schematics_v1/GEOMETRY_04.png)

[SVG](source_geometry_schematics_v1/GEOMETRY_04.svg)

[도식 전체 gallery](source_geometry_schematics_v1/PORTABLE_GALLERY_KO.md) · [그림 전 고정 선택표](source_geometry_schematics_v1/SELECTION_BEFORE_DRAW.json)

## 해석과 남은 검증

- 기존 source 구성·집계 재요약과 새 dx 반전 bbox 계산을 구분한다. alpha, 실제 RGB 변화, 앞뒤 관계 또는 끝부위 identity를 새로 검증하지 않았다.
- 공개 source의 direct E는 실제 촬영된 자연 가림 정답이 아니라 YCB cutout 합성 차폐와 기존 RGB weak end 관찰의 결합이다. bbox 배치 문제만 풀어도 native gold나 자연 가림 전이를 증명하지 않는다.
- synthetic finite registered-end와 공개 RGB semantic-end weak 관찰은 서로 다른 권위다. center owner T/F/BG나 broad support2를 native D/E/S 또는 direct-end 정답으로 바꾸지 않았다.
- 직접 감독306행 모두 C54/C55 학습 pool에 포함됐다. 기존 source holdout은 없으며 사후 부분집합을 독립 검증으로 부를 수 없다. 향후 새 학습의 독립 검증은 source base/frame/donor 연결 계보를 먼저 분리해 사전 동결해야 한다.
- 합성 native6/7의 known direct 감독은0이고 공개 known은 두 프레임의 native0뿐이다. 전체 WOOD/SQUARE·재질/aspect/geometry·native0..7 목표를 유지하며 좁은 범위의 성공으로 끝내지 않는다.
- 4도식은 source 결과를 본 후 정한 설명용 집단별 원순서 첫 사례다. random·blind·전66 대표표본이 아니며 실제 사진·원crop·실제 paste도 아니다.
- 이 단계에서 새 학습/forward/평가/라벨·좌표수정·문턱/모델/데이터 선택/PnP는 없었다. 실사 전이와 전체 가림 목표는 미해결이며 채택 모델은 없다.

다음 source 검증은 원 witness/enclosure를 바꾸지 않고도 반대 bbox 후보가 실제 alpha·샘플 RGB·주변 보존 proof까지 통과하는지를 별도 사전규약으로 확인할 수 있다. 이 보고서는 그 새 생성/학습을 실행하거나 승인하지 않는다.

[원1024/66 집계](source_metadata_diagnostic_v1/SUMMARY.json) · [독립 수치검산](independent_metadata_diagnostic_review_b/REVIEW.json) · [기존 source 제어 원문](../query_visibility_sampled_domain_direct_end_20261024_v189/PORTABLE_REPORT_KO.md) · [source 학습 단계 원문](../query_visibility_direct_end_paired_source_training_20261024_v190/PORTABLE_REPORT_KO.md)

[진단 코드](source_metadata_diagnostic_preparation_a/diagnose_stored_source_metadata.py) · [보고서/도식 코드](source_report_preparation_a/build_source_report_and_schematics.py) · [정확 결과JSON](ROOT_RESULTS_KO.json) · [단계 범위](ROOT_PHASE_RESULT.json)
