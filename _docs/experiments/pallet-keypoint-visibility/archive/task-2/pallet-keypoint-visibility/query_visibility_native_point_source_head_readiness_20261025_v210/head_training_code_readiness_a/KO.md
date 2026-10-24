# V210 · native-point source head 준비 근거

C61의 입력·head는 재사용할 수 있다. 다만 C61이 학습한 direct target은 finite-end/공개 RGB semantic-weak D/E이며, native point0의 자체 상태가 아니다. 다음 후보는 구조를 바꾸기보다 **같은 입력·head에서 SOURCE point-state 감독 범위를 바꾸는 시험**이다. 이 문서는 학습·새 라벨·admission을 만들지 않았다.

## 현재 모델 계약

- RAW 입력: RGB/255 3 + valid1 + 고정 XY2 + zero SAM2 =8채널. 원96 sampler와 q/L/U 유지, 새 crop·RGB 정규화 없음.
- 공유 CNN: Conv16/32/32/48/48/64 + GN/SiLU,96→12→AvgPool3×3→shared64. support2·owner3·direct1은 별도 출력이다.
- C61 direct residual: shared64+native8+othermask8+relativeXY16 →32 SiLU→1, 마지막층0초기화. 총118,359 parameters. C60은 동일 입력 구조에서 XY16만0이다.
- q8는 원 initializer 입력이다. F64 상대좌표/L을[-8,8]clip후FP32, self/미가용 mask0. GT point/reference·선택부위·role을 head에 넣지 않는다.
- 손실: 기존5손실 평균 + balanced direct BCE_D/BCE_E 평균(weight1). old192 + direct16D/16E; seed1810/1811,AdamW3e-4/wd1e-4,clip5,FP32,양4000final이 모두 동결된 뒤 source 출력.
- 모든 loss pool에서 immutable TRAIN∩nominal eligibility. support/owner에 조건을 직접 넣지는 않지만 direct-loss의 공유 backbone 영향은 가능하다.

## TRAIN 분모와 원 source

|구분|원 component TRAIN 행|C61 fit 행|근거|
|---|---:|---:|---|
|SYNTH_AUG65536|56320|7040|원 role / U 보존|
|PUBLIC_TRAIN1480|1240|1240|원 role / U 보존|
|CF3816|3816|3816|원 role / U 보존|
|DIRECT_CONTROL|2784|2784|원 role / U 보존|

SYNTH56320=SELF52800+BEFORE_ANCHOR3520이다. C61은 SELF의 NOMINAL3520와 BEFORE_ANCHOR3520만 학습했다. 제외 jitter는 여전히 TRAIN이며 HOLD로 전환되지 않았다. 전체110base의 4scene가 모두 같은 TRAIN role임을 확인했다. 원440scene/native3520은 모두 V129 출처이고, 현재 variant는 BEFORE/COVER/NEAR_SHAM/FAR_SHAM이다. TARGET_SHIFT를 현재 view에 임의 대응하지 않았다.

## 기존 direct TRAIN 노출

|authority|D|E|완전 D/E/D triplet|부모/영상|component|
|---|---:|---:|---:|---|---:|
|SYNTH_FINITE_END|164|82|82|41 bases|41|
|REAL_SEMANTIC_END_WEAK|54|27|27|12 frames|1|

현재 expanded3144의 TRAIN2784행을 최신 control_join_key로 연결했다. 알려진327행 외2457행은 UNKNOWN이다. 역사적 split.inputs.direct_rows의 V190 cache_index를 사용하지 않았다. 공개27triplet/12frame은 연결된 component1개로 독립27영상이 아니다.

|native|SYN D|SYN E|공개weak D|공개weak E|
|---:|---:|---:|---:|---:|
|0|52|26|18|9|
|1|38|19|14|7|
|2|26|13|0|0|
|3|44|22|0|0|
|4|4|2|8|4|
|5|0|0|14|7|
|6|0|0|0|0|
|7|0|0|0|0|

SYN COVER의 canonical0..7 노출은 12/12/11/11/11/12/9/4다. native6/7 직접감독0을 물리 canonical6/7 부재라고 해석하지 않는다. 두 auxiliary support BCE나 owner CE를 native D/E로 이름만 바꿀 수 없다.

## 더 넓은 point-teacher 경로

110 TRAIN base의 BASE_COMPLETE metadata에서 고유 GT_scenes 경로·기록 SHA110개를 확보했다. **GT 원문은0회 열었다.** 이 원440scene/native3520 각각의 records_CF point0·region·projected reference를 분리 조회할 수 있는 다음 단계의 명시 whitelist다. 실제 dense D/E/S/U와 native2..7 외부가림 수는 아직 UNKNOWN이다. V173의 state8_CF는 region이므로 point target으로 복사하지 않는다.

V129 pure code는 signed-quarter 정확 모서리 permutation을 쓰며 GT-best/PnP/nearest-query가 아니다. 선택 CF는 카메라 부호·고정 role 규칙으로0..5에서 정해지고, selected canonical patch에 COVER와 동일 foreign 물체의 near/far translation을 만든다. 따라서 다른 native가 실제 E인지 추측할 수 없다. CF_records는 각scene records[perm[k]]와 해당 camera로 point/patch reference를 저장한다. RGB shade도 scene buffer·같은 camera·foreign translation을 받지만, 이번 literal chain에서 writer의 정확 호출 파일 binding은 발견하지 못했다. 이 부분은 UNKNOWN으로 남기고 추측 경로를 열지 않았다.

## 다음 하나의 제한된 source 시험

새 generation보다 기존110base·440own scene 경로를 먼저 확인한다. 별도 root entry로 제안110 GT만 조회하여 native/canonical·재질/aspect/pose별 point D/E/S/U와 원 crop reference locality를 세고, 정해진 source 사례에서 실제 native/part 대응을 확인한다. 그 뒤 같은 C61 원입력·head와 원 component split을 유지한 채 direct target 권한을 own-scene point0 D/E로 바꾸는 후보를 사전고정한다. S/U/UNKNOWN과 inputU는 별도 유지하며 D/E에 강제 편입하지 않는다. 예산·admission은 그 source 증거 뒤 root가 별도로 동결한다.

V208 BEFORE 상태를 새 V189 COVER/SHAM에 alias하지 않는다. V209 requested point footprint가 COVER에서 덮이고 SHAM에서 clear였다는 증거는 기존 중심 pixel 불변만으로 E가 틀렸다는 주장을 막지만, 그 자체가 새 native label admission은 아니다. point source 내부 기하 권한은 향후 명시한 source 감독으로 사용할 수 있으며 완벽한 실사 mesh gold를 선결조건으로 요구하지 않는다.

## 근거와 한계

- [실행 가능한 metadata census](census_source_train_metadata.py) · [전수 집계](SOURCE_TRAIN_COVERAGE_CENSUS.json) · [native별 CSV](DIRECT_TRAIN_NATIVE_EXPOSURE.csv)
- [110 GT binding 제안—원문 미열람](SOURCE110_TRAIN_GT_BINDING_PROPOSAL_NOT_OPENED.json) · [정확 code 의존](CODE_DEPENDENCY_BINDINGS.json) · [검토 JSON](REVIEW.json)
- C의 병렬 CODE-only 확인을 결합했다. operational import/모델·가중치·원영상·NPZ·GT config·protected evaluation 점수 접근은0이다.
- source metadata 공유 container는 물리적으로 읽고 원 TRAIN role로 먼저 필터했다. HOLDOUT 내용/점수를 집계·선택하지 않았다.
- source point 상태는 물리 실사 gold나 일반화 성공이 아니다. 새 학습/성능/채택/원인 확정/전체 해결 주장은 없다.
