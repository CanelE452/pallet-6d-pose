# V211 저장 합성 장면의 native point 상태 조회

고정 TRAIN110 base·440 원 장면의 native point3520개와 기존 C61 입력7040개의 실제 image-source 대응을 확인했다. 새 렌더·키포인트 좌표 추정·학습을 실행하지 않았다. 기록된 합성 내부 기하 상태이며 실사 사람 native gold와 구분한다.

|원 장면 변형|point D|point E|point SELF|region UNKNOWN|
|---|---:|---:|---:|---:|
|BEFORE|770|0|110|330|
|COVER|658|112|110|332|
|NEAR_SHAM|759|11|110|334|
|FAR_SHAM|770|0|110|330|

전체 point는 D2957/E123/SELF440이다. 같은 자료의 region은 D1639/E115/SELF440/U1326으로, 주변 영역의 상태를 점 자체의 상태로 대체하면 서로 다른 감독을 섞게 된다. **NEAR_SHAM에도 실제 point E11개가 있으므로 변형 이름만 보고 D를 붙이지 않는다.**

|input native|point E|
|---|---:|
|0|29|
|1|31|
|2|18|
|3|15|
|4|16|
|5|11|
|6|1|
|7|2|

native6/7에도 E가 있지만 각각1·2개뿐이다. 모든8개가 등장한다는 사실이 충분한 감독 범위나 일반화의 증거는 아니다. 원 U32개, C61 입력 U64개를 유지했다.

각 own teacher는 nominal/BEFORE-anchor의 두 입력에 연결되지만 teacher는 항상 **실제 RGB가 나온 image scene**의 상태다. 원 q/L/flags의 anchor와 별도 키로 보존했다. 7040 입력 중2640개는 image scene과 BEFORE query anchor가 다르며, BEFORE 상태를 다른 변형 RGB에 복사하지 않는다.

V212의 사전 고정된 SOURCE loss eligibility를 독립 적용하면7040 중6096(D5853/E243)이 대상이고944는 제외된다. SELF·입력 U·crop 밖 등의 이유는 중첩되므로 원인 건수를 더하지 않는다. 이 단계 자체는 새 target array나 학습 admission을 생성하지 않았다. 후속 admission은 별도 V212 문서에 기록한다.

- [전체3520 point 상태](dense_source_point_teacher_v1/ALL3520_OWN_SCENE_NATIVE_POINT_TEACHER_METADATA.jsonl)
- [전체7040 image/query-anchor 대응](dense_source_point_teacher_v1/ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl)
- [variant/native/canonical/material/aspect/geometry/camera/group 집계](dense_source_point_teacher_v1/SUMMARY.json)
- [독립 strata CSV](independent_actual_review_c/SOURCE_POINT_STRATA_INDEPENDENT.csv)
- [독립 실제 검산](independent_actual_review_c/KO.md)
- [실제 조회 COMPLETE](dense_source_point_teacher_v1/COMPLETE.json)
- [조회 코드](dense_teacher_preparation_b/lookup_dense_source_points.py)
- [GT 열람 전 protocol](dense_teacher_preparation_b/PROTOCOL_BEFORE_DENSE_TRAIN_GT.json)

정확110 GT JSON만 조회했고 region·point·reference를 분리했다. 원 q/L/native/U/declared permutation/RAW96·initializer를 보존했다. 실제 RGB·NPZ·config·새 ray/projection·weights·실사 평가 자료를 읽지 않았다. 실사 성능 개선·채택·전체 문제 해결을 주장하지 않는다.
