# V216 · 공개 SOURCE 부위와 query support를 분리한 관찰

같은 12 SOURCE TRAIN 프레임의 원 96개 native 입력을 유지해 두 관찰자가 독립 판단하고, root가 같은 부위 근거를 다시 확인했습니다. **원시 D 후보 11개 중 9개를 약한 동일 팔레트 부위 visible 근거로 남겼습니다(7 SOURCE_FIT / 2 SOURCE_DEV). 나머지 87개는 UNKNOWN입니다.** 이는 물리 native-point gold나 새로운 학습 허용이 아닙니다. 외부 E와 SELF의 양성 확정은 각각 0개입니다.

진전은 데이터·질문·감독 권위의 구분입니다. 새 모델·학습·추론·평가·문턱 변경은 없고 안정적인 외부 가림 판별 개선을 입증하지 않았습니다. 최신 실제 실사 비교 V213은 그대로입니다. 이 보고서 작성에서는 그 보호된 결과 본문을 열지 않았습니다.

## 범위와 원시 합의 이후의 의미 검토

| 범위 | 프레임/행 | 원 U | 원시 D 후보 | 의미 검토 후 weak 부위 D | UNKNOWN | 공간 witness 후보 |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | 12/96 | 11 | 11 | 9 | 87 | 62 |
| SOURCE_FIT | 8/64 | 9 | 9 | 7 | 57 | 40 |
| SOURCE_DEV | 4/32 | 2 | 2 | 2 | 30 | 22 |

원래 외부 역할은 전부 TRAIN입니다. 사전에 고정한 내부 SOURCE_FIT은 8프레임 64행(U9), SOURCE_DEV는 4프레임 32행(U2)입니다. 각 prefix에 2 FIT/1 DEV이며 4개 prefix를 유지했습니다. 메타데이터 component 12개가 물리적으로 독립된 12개 대상이라는 뜻은 아닙니다. 이미 appearance가 노출된 내부 개발 분할이며 새 blind 검증이나 기존 HOLD가 아닙니다. 미래에는 DEV 및 모든 파생 pair/variant를 모든 loss·sampler·정규화 분모에서 제외해야 합니다. **현재 primary·auxiliary training admission은 모두 false입니다.**

A는 D14, C는 D17을 각각 독립 기록했고 raw helper의 D 합의는 11개였습니다. 원시 관찰값과 helper 결과를 수정하지 않았습니다. root는 아래 두 행에서 ROI가 같은 팔레트의 서로 다른 feature를 포함한다고 보아 weak D 후보만 보류했습니다. 완벽한 mesh/정확한 3D corner를 모든 weak 부위 관찰의 선결조건으로 요구한 것은 아닙니다.

| 보류 행 | 서로 다른 feature 근거 | 기존 panel |
| --- | --- | --- |
| frame 4 / native 3 | EPAL 블록 면의 위쪽과 아래 코너·바닥 경계 | [원 CLEAN96](../query_visibility_public_train_visible_part_coverage_expansion_20261025_v215/fixed12_public_TRAIN_input_previews_v1/FRAME_04_0124/NATIVE_3_CLEAN96.png) |
| frame 6 / native 7 | 위 판 끝면과 아래 변·틈/다음 판 시작 | [원 CLEAN96](../query_visibility_public_train_visible_part_coverage_expansion_20261025_v215/fixed12_public_TRAIN_input_previews_v1/FRAME_06_0174/NATIVE_7_CLEAN96.png) |

## native별 완전 분모

| native | N | 원 U | 공간 witness | 원시 D 후보 | 검토 후 weak 부위 D | UNKNOWN | ON/NEAR/OFF/UNKNOWN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 2 | 8 | 2 | 2 | 10 | 1/4/1/6 |
| 1 | 12 | 3 | 8 | 0 | 0 | 12 | 5/0/1/6 |
| 2 | 12 | 3 | 7 | 0 | 0 | 12 | 2/1/1/8 |
| 3 | 12 | 3 | 6 | 1 | 0 | 12 | 3/0/2/7 |
| 4 | 12 | 0 | 6 | 3 | 3 | 9 | 2/5/3/2 |
| 5 | 12 | 0 | 8 | 1 | 1 | 11 | 3/3/2/4 |
| 6 | 12 | 0 | 10 | 2 | 2 | 10 | 4/3/1/4 |
| 7 | 12 | 0 | 9 | 2 | 1 | 11 | 4/1/0/7 |

모든 native0..7의 12행을 보존합니다. 표의 D는 원 q/native·프레임 맥락에서 설명 가능한 **같은 팔레트 부위의 assistant weak visible 관찰**이며 정확한 native 3D점의 gold가 아닙니다. native1/2/3의 D0도 해당 물리 부위가 없거나 가려졌다는 음성 정답은 아닙니다.

## 공간 witness와 query support는 다른 축

| 범위 | ON | NEAR | OFF | UNKNOWN | ROI 유효/실패/없음 |
| --- | --- | --- | --- | --- | --- |
| ALL | 24 | 17 | 11 | 44 | 79/4/13 |
| SOURCE_FIT | 16 | 11 | 8 | 29 | 49/4/11 |
| SOURCE_DEV | 8 | 6 | 3 | 15 | 30/0/2 |

**공간 witness 후보 62개는 팔레트 라벨 62개가 아닙니다.** 목재·플라스틱 상자의 모서리, 카트 손잡이도 포함합니다. 원 q/native만으로 의도 팔레트 부위를 주소화하지 못하고 explicit ROI가 추가로 필요한 관찰은 LOCAL_FEATURE_ONLY로 보존하며 unconditioned native head에 복사하지 않습니다. ROI 종류·겹침이나 두 reader의 동의만으로 같은 intended part/instance가 증명되지는 않습니다.

ON24/NEAR17/OFF11/UNKNOWN44는 **관찰 ROI feature에 대한 원 query의 관계**입니다. native 좌표 정확도·가시성 정답이 아닙니다. OFF, 바닥/배경, 다른 물체가 q에 있다는 이유로 E를 만들지 않았습니다. E에는 의도 부위를 실제로 가리는 외부 전경의 양성 근거가 필요하고 SELF와도 구분합니다.

ROI는 원96 좌표의 inclusive 정수 영역입니다. A와 C의 원 ROI를 모두 보존하고 **정확한 픽셀 OR**만 기록했습니다. display용 enclosing bbox는 alpha/proof mask가 아닙니다. 79개는 원 valid96 내부, 4개는 실패, 13개는 ROI 없음으로 남았습니다. 실패 ROI를 줄이거나 이동시키지 않았고, valid는 동일 semantic feature나 실제 COVER/SHAM proof를 인증하지 않습니다.

## 실제 시각 근거를 읽는 방법

두 관찰자는 봉인 전 상대방 새 값을 보지 않고 같은 12프레임의 full context/all8 contact를 실제 확인했습니다. root는 저장된 12 overlay와 위의 추가 clean96 두 장을 실제 보았습니다. 아래 설명은 그 봉인된 시각 receipt에 근거합니다. 이번 집계 builder의 새 PNG 열람/렌더는 0입니다.

### 고정 frame 0 · val/000000:51

n6은 같은 팔레트 상판 끝 윤곽의 weak 근거가 남습니다. n5의 판 끝/판 면 선택 차이와 원 U0..3은 그대로 보존했습니다.

[두 원 ROI overlay 보기](joined_part_query_support_v2/FIXED_FRAME_00_TWO_READER_OVERLAY.png)

### 고정 frame 6 · val/000005:8

뒤의 세운 팔레트와 앞 목재 상자가 함께 있습니다. n4/5/6의 팔레트 판 끝 근거와 n0..3의 다른 상자 feature를 분리했고 n7은 서로 다른 feature 범위로 보류했습니다.

[두 원 ROI overlay 보기](joined_part_query_support_v2/FIXED_FRAME_06_TWO_READER_OVERLAY.png)

### 고정 frame 10 · val/000006:28

q8이 카트 손잡이에 놓였습니다. 선명한 edge와 query ON 합의가 있어도 pallet native D/E로 바꿀 수 없음을 보여줍니다.

[두 원 ROI overlay 보기](joined_part_query_support_v2/FIXED_FRAME_10_TWO_READER_OVERLAY.png)

[전체 고정 12프레임 갤러리](joined_part_query_support_v2/PORTABLE_GALLERY_KO.md)

## 내부 역할 × native

| 내부 역할 | native | N | 원 U | 원시 D 후보 | 검토 후 weak 부위 D | UNKNOWN | 공간 witness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SOURCE_FIT | 0 | 8 | 1 | 2 | 2 | 6 | 6 |
| SOURCE_FIT | 1 | 8 | 3 | 0 | 0 | 8 | 5 |
| SOURCE_FIT | 2 | 8 | 3 | 0 | 0 | 8 | 5 |
| SOURCE_FIT | 3 | 8 | 2 | 1 | 0 | 8 | 4 |
| SOURCE_FIT | 4 | 8 | 0 | 2 | 2 | 6 | 3 |
| SOURCE_FIT | 5 | 8 | 0 | 1 | 1 | 7 | 5 |
| SOURCE_FIT | 6 | 8 | 0 | 2 | 2 | 6 | 6 |
| SOURCE_FIT | 7 | 8 | 0 | 1 | 0 | 8 | 6 |
| SOURCE_DEV | 0 | 4 | 1 | 0 | 0 | 4 | 2 |
| SOURCE_DEV | 1 | 4 | 0 | 0 | 0 | 4 | 3 |
| SOURCE_DEV | 2 | 4 | 0 | 0 | 0 | 4 | 2 |
| SOURCE_DEV | 3 | 4 | 1 | 0 | 0 | 4 | 2 |
| SOURCE_DEV | 4 | 4 | 0 | 1 | 1 | 3 | 3 |
| SOURCE_DEV | 5 | 4 | 0 | 0 | 0 | 4 | 3 |
| SOURCE_DEV | 6 | 4 | 0 | 0 | 0 | 4 | 4 |
| SOURCE_DEV | 7 | 4 | 0 | 1 | 1 | 3 | 3 |

## 기존 공개 GT 준비도와 다음 한 가지 점검

공개 SOURCE GT가 없다는 전제는 철회합니다. V153의 과거 보존 영수증은 RGB/depth와 전체·가시 마스크, scene GT/camera 관련 자료를 기록합니다. 입력 전용 manifest에서 GT를 뺀 것은 feature 계약이며 미보유를 뜻하지 않습니다. V154의 instance/role-query weak bridge, V159의 nominal-part·mask 진단 및 owner bridge도 이미 있으므로 첫 GT 활용으로 재포장하지 않습니다. **보존은 역사적 영수증 근거이며 이번 phase에서 GT 본체·현재 payload 존재를 재확인한 것은 아닙니다. 실제 GT 값 접근은 0입니다.**

다음 한 가지는 기존 고정 SOURCE TRAIN의 선택 인스턴스/native 부위/query/crop 대응을 retained 전체·가시 마스크와 depth 계약에 대조하는 bounded audit입니다. 먼저 정확한 TRAIN whitelist를 고정하고 별도 실제 접근 범위를 둡니다. 원 q/L/native/U, 순열 방향을 유지하고 GT-best 대응·PnP·q 재중심화는 하지 않습니다. 마스크 음성/타 물체/부위 불명확을 자동 E로 바꾸지 않으며 완벽한 surface gold를 blanket gate로 삼지도 않습니다. [공개 GT text 준비도와 증거](public_GT_text_readiness_a/KO.md)

## 보존 이력·재현·한계

join V1은 실행되지 않았습니다. 실행 전 artifact binding의 optional size 유무 때문에 whole-dict 비교가 부적합함을 발견하여 V1 원문을 보존하고 V2의 path+SHA 정규화 어댑터만 수리했습니다. 기존 수치·ROI·관찰 판단은 바꾸지 않았습니다. V2 join은 한 번 실제 실행되어 exit0으로 완료됐고 독립 저장 검산을 통과했습니다. 보존된 4 invalid ROI는 이 메타데이터 수리와 별개인 실제 실패 상태입니다.

재현은 아래 봉인 코드와 exact entry를 사용합니다. 출력 경로는 exclusive이므로 기존 산출물 위로 재실행하지 않습니다. 이미 완료된 annotation/join을 다시 실행할 필요는 없습니다.

```bash
python -B join_preparation_b_v2/join_and_render_saved_two_reader_observations_v2.py --entry ROOT_ACTUAL_TWO_READER_JOIN_AND_OVERLAY_V2_ENTRY.json
python -B report_preparation_a/build_saved_v216_reports.py
```

이 phase는 assistant의 약한 SOURCE 부위/feature 관찰입니다. 사람 gold, 물리 native-point gold, 자연 외부 가림의 완전 정답 집합, 무작위 대표 표본, 인과적 모델 개선 증거가 아닙니다. 모든 U11과 UNKNOWN을 남겼고 새 fit/admission/paired controls/model/threshold/PnP 작업을 하지 않았습니다. 미래 loss/입력 계약은 실제로 주소화한 weak 질문과 맞아야 합니다.

## 원문과 실행 근거

- [전체 96 원시 join](joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json)

- [전체 96 root 의미 검토 CSV](ROOT_ALL96_SEMANTIC_REVIEW.csv)

- [전체 12 overlay 갤러리](joined_part_query_support_v2/PORTABLE_GALLERY_KO.md)

- [원 역할/native/관찰자 수표](joined_part_query_support_v2/SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv)

- [모든 축 차이 CSV](joined_part_query_support_v2/ALL96_AXIS_DIFFERENCES.csv)

- [두 원 ROI와 exact OR sidecar](joined_part_query_support_v2/ALL96_EXACT_WITNESS_MASK_SIDECAR.json)

- [독립 저장 join QA](independent_saved_join_review_c/REVIEW.json)

- [root 실제 12+2 시각·의미 검토](ROOT_ACTUAL_ALL12_VISUAL_AND_SEMANTIC_QA.json)

- [annotation 계약](contract_preparation_a/ANNOTATION_AND_SUPERVISION_CANDIDATE_CONTRACT.json)

- [순수 helper](contract_preparation_a/part_query_support_contract.py)

- [내부 FIT/DEV 고정 역할](component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json)

- [실제 두 관찰자 진입 기록](ROOT_ACTUAL_SAMEFRAME_TWO_READER_ANNOTATION_ENTRY.json)

- [실제 V2 join 진입 기록](ROOT_ACTUAL_TWO_READER_JOIN_AND_OVERLAY_V2_ENTRY.json)

- [V1 보존·V2 최소 메타데이터 수리](join_preparation_b_v2/V1_PRESERVATION_AND_MINIMAL_BINDING_REPAIR.json)

- [V2 코드 준비·검사 바인딩](join_preparation_b_v2/FINAL_READY.json)

- [기존 공개 GT text 준비도](public_GT_text_readiness_a/KO.md)

- [보고서 재현 코드](report_preparation_a/build_saved_v216_reports.py)

- [A 봉인 관찰 원문](actual_sameframe_observations_a_v1/ALL96_OBSERVATIONS.json)

- [C 봉인 관찰 원문](actual_sameframe_observations_c_v1/ALL96_OBSERVATIONS.json)

- [기계가 읽는 결과와 전체 수표](ROOT_RESULTS_KO.json)

- [HTML 읽기 화면](ROOT_READOUT_KO.html)

