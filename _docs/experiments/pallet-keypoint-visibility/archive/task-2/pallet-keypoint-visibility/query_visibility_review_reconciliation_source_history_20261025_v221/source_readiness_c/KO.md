# SOURCE 원 query / GT-query / 유한 부위 제어쌍 정리

기존 공개 SOURCE metadata와 동결 코드만 읽어 표를 닫았다. 이번 작업은 첨부 검토에 대한 읽기·감사 결과이며 새 학습 지시나 데이터 생성 실행이 아니다. 기존 목표는 active이고, 현재 허용 범위에는 public real + GT가 포함된다. 과거 검토서의 synthetic-only 전제를 현재 제한으로 적용하지 않는다.

| 자료군 | SOURCE_FIT | SOURCE_DEV | 관찰/증강 단위 |
|---|---:|---:|---|
| GT-query nominal D | 15 | 4 | 19 nominal weak 판독, 8 frame/component |
| 같은 nominal 판독에서 support 검사된 D variants | 39 | 9 | 48 RGB, nominal/jitter 상관 표본 |
| 기존 original-query finite-part COVER | 7 | 2 | 9 제어쌍, BEFORE/COVER/SHAM 총27 |
| V219 실제 정의된 weak 학습 pool | D39/E6 | fit 제외 | FIT45; DEV는 별도12(D9/E3) |

D19의 해부학 표기는 DECK17/BOTTOM_RUNNER2다. D19 FIT15/DEV4와 D48 FIT39/DEV9는 같은 부모의 관계이며 독립 표본19/48개라는 뜻이 아니다. D군은 FIT6 frame/component, DEV2 frame/component다. 기존 COVER9는 FIT4 frame/component·DEV1 frame/component다.

## 코드가 바로 읽을 표

- [48행 exact ROI union·valid96·원q/GTq·PNG binding CSV](D48_EXACT_ROI_UNION_VALIDITY_AND_QUERY_PROVENANCE.csv)
- [nominal D19 CSV](NOMINAL_D19_SOURCE_ROLE_AND_QUERY_PROVENANCE.csv)
- [원 query COVER9 CSV](ORIGINAL_QUERY_COVER9_FINITE_PART_AND_OCCLUDER_PROVENANCE.csv)
- [V219 FIT45와 natural E6 원장](V219_ACTUAL_DEFINED_FIT45_AND_NATURAL_E6.csv)
- [전체 source provenance JSON](SOURCE_ROI_IMAGE_OCCLUDER_PROVENANCE.json)
- [48행 validity bounds/exact union JSON](D48_VALID96_BOUNDS_AND_EXACT_ROI_UNION.json)

`exact_union_RLE_rowmajor96` / `exact_ROI_union_RLE`는 [start,length]이고 flat index의 (y,x)=divmod(index,96)이다. A/C inclusive 정수 ROI를 집합 OR로 합친 결과이며, enclosing bbox는 DISPLAY_ONLY다. D48 전부 exact union이 저장된 strict-valid bitmap 안에 있다. 이는 저장 metadata의 수학 검사이며 PNG를 다시 검증한 것은 아니다. ROI는 약한 bag/region이고 ROI 내부 모든 픽셀이 실제 부재라는 segmentation gold가 아니다.

## nominal D19 전체 주소

| row | role | frame/candidate/native | anatomy | A ROI / C ROI |
|---|---|---|---|---|
| F00_C000000_S0_N4 | SOURCE_FIT | 0/0/4 | DECK | [38, 38, 63, 63] / [36, 46, 73, 65] |
| F00_C000000_S0_N5 | SOURCE_FIT | 0/0/5 | DECK | [32, 28, 58, 72] / [35, 44, 56, 65] |
| F02_C000000_S0_N0 | SOURCE_FIT | 2/0/0 | DECK | [47, 40, 75, 67] / [49, 40, 77, 68] |
| F02_C000000_S0_N4 | SOURCE_FIT | 2/0/4 | DECK | [36, 39, 64, 62] / [35, 45, 66, 63] |
| F02_C000000_S0_N5 | SOURCE_FIT | 2/0/5 | DECK | [20, 29, 50, 61] / [25, 35, 49, 62] |
| F02_C000001_S0_N5 | SOURCE_FIT | 2/1/5 | DECK | [18, 25, 43, 62] / [14, 38, 41, 61] |
| F03_C000000_S0_N0 | SOURCE_FIT | 3/0/0 | DECK | [48, 43, 78, 67] / [49, 47, 71, 65] |
| F03_C000000_S0_N3 | SOURCE_FIT | 3/0/3 | BOTTOM_RUNNER | [45, 41, 78, 62] / [48, 42, 78, 58] |
| F04_C000000_S0_N0 | SOURCE_FIT | 4/0/0 | DECK | [45, 42, 76, 65] / [47, 42, 69, 58] |
| F04_C000000_S0_N3 | SOURCE_FIT | 4/0/3 | BOTTOM_RUNNER | [44, 41, 73, 61] / [47, 39, 70, 56] |
| F05_C000000_S0_N1 | SOURCE_DEV | 5/0/1 | DECK | [15, 46, 45, 76] / [20, 46, 42, 64] |
| F05_C000000_S0_N4 | SOURCE_DEV | 5/0/4 | DECK | [45, 41, 74, 69] / [48, 43, 74, 66] |
| F06_C000003_S0_N0 | SOURCE_FIT | 6/3/0 | DECK | [46, 40, 74, 69] / [48, 43, 77, 65] |
| F06_C000003_S0_N1 | SOURCE_FIT | 6/3/1 | DECK | [22, 37, 51, 69] / [32, 42, 51, 68] |
| F07_C000003_S0_N0 | SOURCE_FIT | 7/3/0 | DECK | [22, 32, 52, 64] / [33, 35, 62, 61] |
| F07_C000003_S0_N1 | SOURCE_FIT | 7/3/1 | DECK | [16, 15, 45, 45] / [25, 20, 48, 42] |
| F07_C000003_S0_N4 | SOURCE_FIT | 7/3/4 | DECK | [46, 36, 76, 68] / [47, 44, 78, 69] |
| F08_C000003_S0_N1 | SOURCE_DEV | 8/3/1 | DECK | [15, 14, 44, 45] / [24, 24, 46, 47] |
| F08_C000003_S0_N4 | SOURCE_DEV | 8/3/4 | DECK | [46, 36, 74, 68] / [47, 43, 72, 66] |

## 기존 original-query COVER9 전체 주소

| frame/native | role | exact union pixels | 원 A/C ROI | 의미 제한 |
|---|---|---:|---|---|
| 0/6 | SOURCE_FIT | 1130 | [44, 26, 65, 70] / [40, 27, 58, 61] | DECK_VS_BOTTOM_RUNNER_UNRESOLVED |
| 2/0 | SOURCE_FIT | 1169 | [51, 40, 76, 75] / [47, 37, 77, 67] | WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED |
| 2/4 | SOURCE_FIT | 1053 | [36, 47, 64, 76] / [31, 54, 64, 78] | WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED |
| 4/0 | SOURCE_FIT | 939 | [63, 44, 87, 71] / [57, 39, 81, 62] | WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED |
| 5/4 | SOURCE_DEV | 957 | [46, 44, 74, 76] / [49, 44, 74, 71] | WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED |
| 5/7 | SOURCE_DEV | 1205 | [42, 16, 68, 46] / [40, 31, 63, 60] | DECK_VS_BOTTOM_RUNNER_UNRESOLVED |
| 6/4 | SOURCE_FIT | 953 | [51, 40, 79, 70] / [49, 44, 70, 70] | F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE |
| 6/5 | SOURCE_FIT | 859 | [65, 54, 83, 84] / [56, 58, 82, 85] | F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE |
| 6/6 | SOURCE_FIT | 1418 | [47, 30, 79, 62] / [64, 23, 86, 53] | F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE |

기존9개는 top-role 부위 정합 재검토4, deck-vs-bottom-runner 미해결2, f6 local-board 비native-addressable3이다. 같은 local feature가 보인다는 약한 근거는 유효하지만 이9개를 원 native point D/E로 복사하지 않는다. 저장된 `native_DE_target=-1`, variant에서 query-support를 추론하지 않는 규칙도 그대로다.

## 원 q와 GT-derived training q는 다른 family다

기존9개는 detector/initializer의 원 q·L·native·availability를 그대로 유지한 RAW96에 post-render 개입을 더했다. 두 reader의 보이는 feature ROI를 exact OR로 합쳤으며, 기존 PNG/valid binding과 source 입력을 연결한 정보가 표에 있다. 원 q8/condition32가 필요하면 그 원 initializer metadata를 정확히 join해야 한다. 이 표는 원 q/L/row lineage를 보존하며 GT-derived condition32를 원9개에 대신 넣지 않는다.

D19/D48은 `PUBLIC_SOURCE_GT_QUERY_JITTER_V1`다. GT 선언 후보의 nominal q8와 (0,0),(.25L,0),(0,.25L) 세 offset으로 별도 생성한 학습 입력이다. L은 원본과 같고 원 query는 변경하지 않았다. 저장 JSON에는 원 q와 생성 q 및 delta(raw96 단위)를 함께 남겼다. q8 자체가 GT-derived라는 정보를 숨기지 않는다. 추가 GT pose/ref/ROI/mask 배포 입력이 없다는 말과 q8의 GT 유래는 별개다. 원 detector→GT instance는 여전히 UNKNOWN이며 첫/가장 가까운 GT 후보를 고르지 않았다.

## 개입 물체와 중복

기존 COVER9의 occluder는 실제 외부 물체 donor가 아니라 두 색 [32,160,230]/[230,64,160]의 4픽셀 checker를 원 RAW96 후단에서 불투명하게 덮은 패턴이다. SHAM은 같은 RGB/alpha pattern과 질량을 original-valid·target-disjoint 위치로 옮긴다. 선택은 RGB/모델/라벨이 아니라 기하·validity의 최소 L1 이동과 고정 tie 순서다. 저장 proof는 COVER가 exact target OR를 덮고 SHAM은 OR와 겹치지 않으며 target RGB를 유지하고 밖은 그대로임을 기록한다. 이번에는 그 proof를 재실행하거나 PNG를 열지 않았다. 물리 depth/owner를 changedRGB에서 추론하지 않는다.

V219 actual fit entry의 weak_rows360 binding은 이 canonical360 SHA와 정확히 같다. 동결 contract의 `weak_loss_candidate & SOURCE_FIT` 필터가 FIT45(D39/E6)를 정의한다. D48의 FIT39와 nominal D19의 FIT15는 이미 그 pool에 포함됐다. 기존27 제어 PNG의 선언 encoded SHA와 FIT45의 선언 encoded SHA 교집합은0이다. 원 (frame,native) row는 FIT끼리3개가 겹치므로 부모 source 재사용은 있다. 이는 선언 PNG identity 비교이며 decoded-pixel 비교나 새 독립 이미지 dedup은 아니다. V219 E6은 저장 semantic foreign-occlusion 약관찰/지원검사를 통과한 GT-query family 입력이지 기존9개 checker COVER가 아니다.

## 후속 대조를 허용할 명시적 계약

COVER/SHAM 생성 자체는 이미 했다. 새로 확인할 수 있는 것은 expected-part와 native/q8 주소가 같은 부위를 묻도록 정렬한 대조다. 현재 감사는 이를 실행하지 않는다.

1. D19/D48를 재사용한다면 새 named family에서 동일 GT 후보·native/q8·L·32조건을 BEFORE/COVER/SHAM 모두 고정한다. 타깃은 해당 후보의 관찰된 finite expected-part 약영역이다. COVER는 exact 두 ROI OR를 가리고 SHAM은 같은 패턴을 target-disjoint 위치에 둔 뒤, 이후 별도 실제 image/alpha proof로 검증한다. FIT39의 BEFORE는 새 독립 데이터가 아니며 이미 사용한 D다. 새 variant 구성 여부/성공 수는 아직 정하지 않았다.
2. 원9개를 그대로 쓴다면 task authority를 original-query local finite-witness auxiliary로 둔다. 임의 목재 feature를 가려서 native E로 우회하지 않는다. 입력 q/native/영상이 그 witness를 주소화하는 근거가 부족한5개는 그 한계를 그대로 보존한다. ROI를 감독 selector로 쓰는 것과 배포 입력에 추가하는 것은 다르다.
3. 모든 DEV 부모와 파생형은 모든 fit/aux pool/정규화에서 제외한다. DEV는 사전 노출된 작은 SOURCE 진단이며 fresh blind·human/physical point gold가 아니다. UNKNOWN·원 U·semantic unresolved를 삭제하거나 E로 강제하지 않는다.

완벽한 mesh/물리 corner 인증은 이 weak finite-part 감독의 일괄 선결조건이 아니다. 동시에 region visibility proxy를 native point0 truth라고 이름만 바꾸지는 않는다. 학습·새 모델·새 real 평가·PnP·선택·채택은 이번 감사에서0이다.
