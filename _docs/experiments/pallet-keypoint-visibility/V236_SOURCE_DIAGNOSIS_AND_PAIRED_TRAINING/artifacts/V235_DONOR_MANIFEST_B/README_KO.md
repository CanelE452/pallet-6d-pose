# V235 SOURCE FIT donor 바인딩 선정

메타데이터 선정은 PASS입니다. 기존 FIT D witness 841행에서 unique instance 605개를 얻었고, 사전 cap16을 모든 eligible 물체 ID에 동일하게 적용해 **64 donor**를 선정했습니다. “약48” 추정에 맞춰 물체나 donor를 줄이지 않았습니다.

| object_id | FIT D가 있는 unique instance 후보 | 고정 선택 |
|---|---:|---:|
| 2 | 153 | 16 |
| 3 | 96 | 16 |
| 7 | 181 | 16 |
| 8 | 175 | 16 |

한 instance의 canonical witness는 원 FIT D의 최소 `sample_index`입니다. 각 물체 안에서 `sha256('V235:donor:'+input_id)` 오름차순, `input_id` 동률순으로 앞16개를 택했습니다. cap16은 root가 픽셀 열람 전에 고정한 균등 상한이며 DEV의 물체별 성능이나 label 분포로 조정하지 않았습니다. D witness가 없는 instance는 후보가 아닙니다.

선택은 scene000000–000004의 기존 TRAIN/INTERNAL_SOURCE_FIT에만 속합니다. 64 instance가 61 frame에 걸쳐 있어 future exact whitelist는 RGB61개·visible mask64개, **총125개 PNG**입니다. 각 `(scene,frame)`의 원 mask metadata에서 `visible_masks[source_instance_id]`를 직접 인계했고 `instance_object_ids[source_instance_id]==object_id`를 검산했습니다. 같은 물체를 찾아 임의의 다른 mask나 첫 instance로 대체하지 않았습니다.

`DONOR_WHITELIST.json`은 원 RGB 및 visible-mask 경로·기록 SHA, object/scene/frame/instance, canonical D witness q/L·aux slot, 고정 정렬 SHA를 담습니다. native mapping은 계속 UNKNOWN입니다. associated-r D witness와 GT sampling metadata는 SOURCE provenance이며 모델 forward에 추가하는 GT feature가 아닙니다. D witness 하나가 donor 전체의 무가림·품질을 인증하지는 않습니다.

이번에는 입력1000·target8000·진단7617·mask frame metadata700개의 텍스트를 파싱했습니다. 그 안에는 내부 DEV 기록도 있지만 후보와 순위에는 FIT D만 사용했고 DEV raw pointer를 따라가지 않았습니다. 원 donor RGB/mask·depth·scene GT/camera JSON에는 stat/hash/read/decode를 하지 않았습니다. 인계 SHA는 상속한 기록이며 새 실파일 해시 검증이 아닙니다. mask의 픽셀·bbox·비어 있지 않음·opacity는 미확인입니다.

**픽셀 접근·합성·훈련은 승인하지 않습니다.** Root의 별도 exact PNG scope/entry와 파일 안전성·SHA 확인 이후에만 decode할 수 있습니다. YCB base에는 다른 object의 donor를 쓰는 조건도 후속 compositor가 지켜야 하며 이번 목록 선정이 그 실행을 대신하지 않습니다. 새 데이터/모델/threshold 선택이나 보호 평가 접근은 없었습니다.

- [64 donor와 future PNG 바인딩](DONOR_WHITELIST.json)
- [선정 분모와 접근 한계](SUMMARY.json)
- [메타데이터 선정 코드](select_donor_metadata.py)
- [사전 범위](SCOPE_BEFORE_METADATA_SELECTION.json)
