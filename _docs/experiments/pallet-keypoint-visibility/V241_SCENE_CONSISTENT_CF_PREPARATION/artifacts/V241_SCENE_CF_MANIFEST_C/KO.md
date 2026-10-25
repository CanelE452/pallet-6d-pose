# V241 원 장면 CF 메타데이터 closure

Root가 고정한 원 ORIGINAL FIT known-D 전체 5,406행(SYN 4,565·YCB 841)을 포함했습니다. 기존 V235 성공 5,367행으로 선별하지 않았으며 실패 39행도 유지했습니다. 새 이미지·CF 생성·특징 추출·학습은 0입니다.

| 항목 | 수량 |
|---|---:|
| 부모 원 query | 5,406 |
| 부모 q8 group | 1,198 (SYN593/YCB605) |
| 부모 actual RGB | 804 (SYN RGB_uint8 NPZ340/YCB PNG464) |
| 기존 고정 donor | 64 instance / RGB61 / visible-mask64 |
| 부모 대 원 DEV group·image 중복 | 각각0 |
| donor 대 원 DEV group·image 중복 | 각각0 |

이 중복 검사는 선언된 literal asset identity와 원 내부 FIT/DEV에 한정합니다. donor64는 FIT 부모의64group/61image와 겹치는 SOURCE 재사용이며 독립 자료라고 주장하지 않습니다. 원 SOURCE15040/known7617/masked7423의 분모는 보존된 입력 manifest를 참조합니다.

`PARENTS_ALL5406.jsonl`에는 원 행·sample_index/group/slot·q/L·teacher_reference_xy와 감독 권한, 원 BEFORE RGB96(CHW)/valid bool SHA, 같은 V235 donor 선택 및 전체 donor_plan을 담았습니다. `PARENT_GROUPS_INPUT_ONLY.jsonl`은 **actual image**와 원 q8/L8/availability8/box/raw_hw를 그대로 보존합니다. GT/anchor 이미지로 바꾸거나 clipped condition32에서 q8을 역복원하지 않습니다. S/U·원 unavailable인 다른 슬롯도 q8에서 삭제하지 않습니다.

teacher reference는 원 V235 FIT-D metadata의 좌표를 exact join했습니다. SYN은 intended-native witness0 MAIN, YCB는 associated-r AUX이며 YCB native는 UNKNOWN입니다. r·object·source ID는 감독/샘플링/계보용이고 배포 입력에 추가하지 않습니다. 원 SYN/YCB 진단 파일의 exact path/SHA는 provenance로만 적었으며 그 원문을 새로 읽지 않았습니다. physical pallet native gold로 승격하지 않았습니다.

Root가 지정한 기존 V235 donor64·different-object rule·선택·후보 순서를 유지합니다. all5406의 teacher_ref는 null이 없고 원 q/L/row_id·donor RGB/mask binding이 일치했습니다. 이전 postcrop 성공/teacherchecks는 새로운 full-scene opacity·visible support proof가 아닙니다. 새 BEFORE5406의 active RGB96/valid SHA를 원 hash와 맞추고 새 장면에서 teacher proof를 다시 실행해야 합니다. 새 COVER/SHAM 상태는 그 전까지 UNKNOWN입니다.

실제 RGB/NPZ/NPY/GT body·weights/model/current/held에는 stat/hash/header/decode를 하지 않았습니다. 선언된 metadata와 gzip metadata만 읽었고 원 payload pointer는 inert입니다.

[선언 manifest](MANIFEST.json) · [부모5406](PARENTS_ALL5406.jsonl) · [원 q8 group](PARENT_GROUPS_INPUT_ONLY.jsonl) · [부모 이미지 whitelist](PARENT_IMAGE_WHITELIST.json) · [donor64](DONOR64_CLOSED_METADATA.json) · [수량·중복](CENSUS_AND_OVERLAP.json) · [CSV](COUNTS.csv)
