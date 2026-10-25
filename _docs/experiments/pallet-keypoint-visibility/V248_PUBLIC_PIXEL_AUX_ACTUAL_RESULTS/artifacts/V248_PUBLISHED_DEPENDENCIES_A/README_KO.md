V248 기존 공개 dependency 재사용 안내

기준 commit은 b9ec7535175e7be918cd5cd990e1d2639cac1a29입니다. dependency_map.json의17개 SOURCE dependency는 그 commit의 정확 tracked blob을 opaque stream으로 SHA256/size 검산했습니다. local 원본 array/PT/RGB/GT를 열거나 모형을 실행하지 않았습니다. 새 V248 package copy/stage/commit은 하지 않았습니다.

기존 파일은 V240_CANONICAL_CONTEXT_ACTUAL_RESULTS(6개 frozen feature·warm core·baselineCSV·COMPLETE·CODE), V233_SOURCE_CROP_ENCODER_ADAPTATION(samples), V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION(geometry/head)에 이미 있습니다. 새 package에서 상대 경로는 map의 relative_from_new_package를 사용합니다. old trainer/producer는 provenance 또는 코드 binding 검증용이며 cached features로 head를 재생·재학습할 때 공식 DINO weight 다운로드나 원 RGB/GT는 필요하지 않습니다.

1. 기존 commit과 새 V248_PUBLIC_PIXEL_AUX_ACTUAL_RESULTS를 함께 받습니다. map의 path/sha256/bytes로17개 기존 dependency와 새 package MANIFEST를 검산합니다.
2. 저장 head 출력 재검산에는 V248 final CP/코드·위 main6features/samples/geometry·새 AUXfeatures/samples를 사용합니다. teacher certificate나 RGB는 forward 입력이 아닙니다. threshold는 scalar logit0입니다.
3. 학습 자체 재실행은 별도 실행입니다. frozen entry/CODE에는 원 absolute binding과 fixed fresh output 정책이 있으므로 원 경로를 보존하는 격리 filesystem layout(예: container bind mount)에 artifact 하위 구조를 재구성해야 합니다. metadata를 무단 편집하거나 기존 완료 output에 다시 실행하지 않습니다. Root가 명시적으로 승인한 fresh true entry와 원 고정 recipe/환경을 써야 합니다.
4. 원 main 수량은 정확 known7617/FIT5876/DEV1741, masked7423입니다. public AUX는 새 conditional pixel 목표이며 원 native352를 대체하지 않습니다. 내부 SOURCE DEV 재사용을 blind/current 성공으로 해석하지 않습니다.

이 파일은 byte dependency map이며 실제 재실행·성능 검증 완료 주장이 아닙니다. 원 weight를 deserialize하거나 array를 수치 decode하지 않았습니다.
