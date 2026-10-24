# TRAIN 원q8 계보와 입력 계약 확인

원 합성 q8은 **GT 투영이 아니라 실제 고정 initializer 예측**이다. V132는 원 RGB를 YOLO에 넣고 confidence 첫 argmax 후보의 keypoints 0..7을 그대로 쓴다. V132 TRAIN 앞4096행의 q/L을 원512scene 메타데이터와 직접 비교해 전부 일치했다. GT pose·끝점·최적 permutation을 사용해 q를 교정하는 절차는 없다. initializer를 재실행하지 않았다.

|입력 모집단|읽은 전체 행|TRAIN 행|TRAIN inputU|TRAIN 원context 그룹|
|---|---:|---:|---:|---:|
|SELF_SCENE15|61440|52800|480|440|
|BEFORE_ANCHOR4|4096|3520|32|440|
|PUBLIC|1480|1240|161|155|
|CF|3816|3816|360|477|
|DIRECT|3144|2784|112|348|

합계 TRAIN64160행·inputU1145는 중복 가능한 학습 입력 occurrence 수다. 독립 프레임 수가 아니다. 전체10개 JSONL container148857행에는 HOLDOUT **role 메타데이터**도 물리적으로 포함된다. 실제 q 비교는 V132 TRAIN prefix4096행만 읽었고 VAL suffix는 읽지 않았다. 픽셀·배열·헤더·타깃·점수·가중치·모델은 열지 않았다.

SELF의 q8은 원scene 초기값이다. 현재 crop q 하나에는15개 고정 변환이 적용된다. TRAIN52800행 모두 원anchor와의 dxdy·scale 복원이 가능하며, 이 중 inputU480을 제외한 사용가능 행은52320개다. 원anchor와 현재crop q/L를 함께 넣으면 결정적 augmentation 변환을 알려주는 통로가 된다. **실제 끝점 GT 오차를 안다는 뜻은 아니다.** 증강된 이미지에서 재추론된 동시 jitter8은 저장돼 있지 않다. BEFORE는 원BEFORE8을4개 이미지에 고정하며 TRAIN440 image groups는110 BEFORE origins를 공유한다.

PUBLIC는 원MR frame별8개, CF는 pair/variant별8개다. CF477개 payload occurrence는 해당PUBLIC 원q/L·availability를 정확히 계승하며 새편집 RGB의 재추론이 아니다. DIRECT는 과거V190 rows를 따르지 않고 최신V1943144행·393groups(합성291/실사weak102)를 사용했다. 원metadata8·all8슬롯·q/L·availability와 split join이 일치했다.

`permCF_to_canonical[k]`는 **camera-facing convention 슬롯 k가 가리키는 canonical index**다. 계산식은 `CF_xy[k]=project(canonical_anchors)[perm[k]]`다. V129가 미리 정한 signed_quarter의 Y축 회전과 고정 부호 순서에서 만든 대응이며, RGB/query-nearest/GT-best matching이 아니다. 실제 initializer 출력슬롯을 이 permutation으로 다시 배열하지도 않는다.

|native 슬롯|TRAIN128 설정에서 대응한 canonical index|
|---|---|
|0|0, 1, 4, 5|
|1|0, 1, 4, 5|
|2|2, 3, 6, 7|
|3|2, 3, 6, 7|
|4|0, 1, 4, 5|
|5|0, 1, 4, 5|
|6|2, 3, 6, 7|
|7|2, 3, 6, 7|

따라서 nativeID 하나가 모든 장면의 단일 canonical 물리코너를 뜻하지 않는다. native6/7 direct UNKNOWN을 canonical6/7 부위 전체의 감독 부재로 해석하면 안 된다. 이 규약 변화가 모델 결과의 원인이라는 증거는 없다.

다음 입력 계약은 stored nativeID와 **같은 origin의 원q8/L8/availability8**을 명시적으로 분리해 전달하는 범위로 잡을 수 있다. active crop q/L와 원anchor context는 다른 이름을 유지하고 SELF shortcut 통제는 fit 전에 고정해야 한다. TRAIN의 q/L는 전부 finite이지만1145행의 availability는 false다. finite 여부로 U를 덮어쓰면 안 된다. GT end·enclosure·pose·canonical 대응·reference·base/factor 정보는 입력에서 제외한다. 여기서는 새 입력 텐서·정규화·학습·성능선택을 만들지 않았다.

PUBLIC의 upstream 실행 자체는 새로 감사하지 않았고, V181→V153 입력 메타데이터 계보와 저장값 일치를 확인했다. 원q8의 저장 정합만으로 selected-instance/end 의미가 항상 맞거나 실사 성능이 개선됨을 증명하지 않는다.

고정V19712개는 새선택 없이 그대로 [원q8 입력메타](SELECTED12_ORIGINAL_Q8_METADATA.json)로 내보냈다. 3변형의 입력좌표는 같고, 라벨·end enclosure·pose·예측은 포함하지 않는다. ID/domain/base/frame은 출처 표시이며 네트워크 허용입력 목록이 아니다.

첫 실행은 export 정렬에서 일부 real group의 선택적 `variant_index` 누락으로 중단됐다. [원code/trace](provenance/BEFORE_SELECTED12_CANONICAL_VARIANT_SORT_ADAPTER/FAILURE.txt)를 보존하고 canonical BEFORE/COVER/SHAM 이름순서만 적용했다. 같은 입력·검사·선택의 명시 재실행은 exit0/PASS였다.

[전수 결과](CENSUS_RESULT.json) · [모집단 CSV](POPULATION_TRAIN_Q8_CENSUS.csv) · [native별 CSV](TRAIN_NATIVE_Q8_CENSUS.csv) · [상세 검토/코드 포인터](REVIEW.json) · [실행 코드](census_q8_provenance.py)
