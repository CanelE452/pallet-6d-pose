# 배포 가능한 query 조건과 SOURCE 목표 권한

실제 사용 가능한 **96×32 조건값**을 저장했습니다. 원 initializer q8·native ID·원 L·원 availability만 사용하며, 이는 C61에 이미 있던 정보입니다. 인접 방향이나 q8 범위를 다시 계산했다고 새 관측정보나 새 모델을 얻은 것은 아닙니다. 다음 SOURCE 구성은 같은 feature를 새 것처럼 반복하기보다 **선택된 instance·의도한 part·목표 라벨의 대응**을 분리해 정리하는 데 써야 합니다.

[실제 조건 JSONL](actual_conditions_v1/ALL96_DEPLOYABLE_CONDITION32.jsonl)은 각 행의 `features.condition32`만 모델에 전달할 수 있게 저장했습니다. 순서는 native one-hot 8, other mask 8, other XY 16입니다. XY는 원 좌표에서 F64 차분/원 L 나눗셈 후 [-8,8]로 제한하고 float32로 저장합니다. self XY·mask는 0입니다. [원 ID/q/L/flags/역할 sidecar](actual_conditions_v1/ALL96_ORIGINAL_IDENTITIES_ADMIN_ONLY.json)는 행 대조용이며 모델 feature가 아닙니다.

|보존 항목|실제 수량|
|---|---:|
|원 frame / component|12 / 12|
|원 query / native별 행|96 / 각 12|
|SOURCE_FIT / SOURCE_DEV|64 / 32|
|원 input U: FIT / DEV|11: 9 / 2|
|GT candidate 확장 입력 / 중복 제거 후|120 / 96|
|유한 float32 조건값|3,072|
|새 이미지 문맥·native 라벨·학습|0 / 0 / 0|

원 input U11은 그대로입니다. U 행은 native category만 유지하고 다른 query의 mask/XY를 모두 0으로 만들며, 남은 geometry가 유효하다는 이유로 입력을 승격하지 않습니다. GT 후보가 여러 개인 행도 같은 원 row를 한 번만 사용합니다. 원 dynamic native 순서를 다시 permute하지 않습니다. [실제 identity/availability CSV](actual_conditions_v1/ALL96_CONDITION_IDENTITY_AND_AVAILABILITY.csv)와 [요약](actual_conditions_v1/SUMMARY.json)에 분모를 남겼습니다.

**무엇이 있고, 무엇이 없는가.** 초기 q8은 같은 instance의 여러 query 사이 방향·비율을 조건으로 제공합니다. [인접 방향과 initializer span](actual_conditions_v1/ALL96_INITIALIZER_GEOMETRY_DIAGNOSTICS_NOT_EXTRA_FEATURES.json)은 이 정보의 설명용 재표현이며 추가 입력 채널이 아닙니다. native ID는 category이고 물리 part의 관찰 증거가 아닙니다. 기존 RAW96 밖의 실제 instance·전체 part 픽셀, 잘못 잡힌 instance를 교정할 관찰 증거, 현재 local wood가 의도한 native part인지의 증거는 이 32값에서 새로 생기지 않습니다. q8-defined 추가 영상 문맥은 새 픽셀 정보를 줄 수 있지만 이번에 읽거나 선택·구현하지 않았습니다.

**목표를 입력과 분리합니다.** [목표 권한 계약](TARGET_AUTHORITY_AND_MINIMAL_CONDITION_CONTRACT.json)은 instance 대응, intended-native/part 대응, 원 q의 ON/NEAR/LOCATION_UNSUPPORTED, native D/E/SELF/UNKNOWN, local-part 보조 관찰을 별도 축으로 둡니다. 보이는 목재·crop-centre owner·약한 local witness·nominal reference가 crop 안에 있다는 사실만 native D가 되지 않습니다. 다른 instance, OFF query, mask/depth 부재나 padding도 E가 아닙니다. E에는 의도한 같은 part를 앞에서 가리는 foreign evidence가 필요하고 SELF는 별도 근거를 유지합니다. 완벽한 mesh/corner gold가 없다는 이유로 유효한 약한 part 보조목표까지 막지는 않되, 그 목표를 native gold로 복사하지 않습니다.

[V217 저장 진단](../../query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217/ROOT_RESULTS_KO.json)은 원 96과 후보 120을 구분하며 새 native gold/학습 admission은 0입니다. [query-instance-part 종합](../../query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217/ROOT_QUERY_INSTANCE_PART_SUPPORT_SYNTHESIS.json)의 nominal 거리·standing 사례는 진단 근거이며 물리 좌표오류나 E의 자동 판정이 아닙니다. GT pose/perm/projection/mask/depth, witness ROI와 라벨, 점수, FIT/DEV 역할은 feature API가 거부합니다.

**작은 다음 사용.** 기존 C61의 RAW96+condition32 계약을 재사용하면서 SOURCE_FIT에서 의도한 instance/part 대응 및 약한 보조목표 권한을 실제 행별로 연결하는 것이 우선입니다. native bias만으로는 image×native를 표현하지 못하고, 인접 방향 추가만으로는 관측정보가 늘지 않습니다. GAP 대신 작은 조건부 공간 pooling을 쓰는 가설도 기존 공간정보를 다르게 보존할 뿐, 누락된 픽셀이나 목표 권한을 만들어 주지 않습니다. 이 단계는 새 head·모델 선택이 아닙니다.

SOURCE_DEV 4개 component의 모든 native/sibling/파생 variant는 향후 모든 학습 branch에서 제외합니다. frame/component, native0..7, 의도한 part 및 standing 정의의 적용 범위, 목표 authority, known/UNKNOWN와 원 U를 각각 보고해야 합니다. 후보 GT나 개입 복제행을 독립 표본으로 세지 않습니다. 기존 노출이 있었던 작은 SOURCE_DEV는 fresh-blind/OOD 검증이 아니며 보호 평가 점수를 조건·목표·모델 선택에 쓰지 않습니다.

[실행 코드](build_deployable_query_conditions.py) · [사전 프로토콜](PROTOCOL_BEFORE_DEPLOYABLE_CONDITION_METADATA_GENERATION.json) · [6개 순수검사](INVENTED_CONDITION_CHECKS.json) · [실제 metadata COMPLETE](actual_conditions_v1/COMPLETE.json). 첫 가상 fixture의 필수 frame_key 누락은 보존 후 fixture만 고쳤고 production 코드는 바꾸지 않았습니다. 이미지·원 배열·원 GT·모델·추론·PnP·학습은 실행하지 않았으며, 목표는 여전히 미해결입니다.
