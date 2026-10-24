# V211 독립 실제 saved-GT·metadata 검산

PASS. root와 동일한 TRAIN110 GT JSON만 읽어 own3520 point와 실제 image-source7040 대응, CF→canonical, 원 q/L/native/U/flags, frame/crop locality 및 9종 strata를 독립 검산했다. producer import·재실행은 없다.

|구분|D|E|S|U|UNKNOWN|
|---|---:|---:|---:|---:|---:|
|own point0|2957|123|440|0|0|
|own region|1639|115|440|1326|0|
|mapped point0|5914|246|880|0|0|

NEAR_SHAM에도 저장 point E11이 있다. variant 이름으로 D를 강제하지 않는다. 원 own inputU32와 mapped inputU64는 geometry state와 별도로 유지된다. 실제 image와 BEFORE query anchor가 다른 입력2640개도 teacher는 image scene에서, q/L/flags는 원 anchor에서 유지된다.

V212에서 실제값 조회 전에 고정한 조건(target-alone D·combined D/E·원 available/finite qL·finite reference·원 frame 및 실제 입력 crop canvas 안)을 독립 대입하면6096행(D5853/E243)이다. 제외944행의 사유는 중첩되므로 항목 합계를 새 분모로 쓰지 않는다. 이는 source loss eligibility 수이며 새 target 배열·학습 admission이 아니다.

[전 strata CSV](SOURCE_POINT_STRATA_INDEPENDENT.csv) · [독립 readout](INDEPENDENT_READOUT.json) · [QA receipt](REVIEW.json) · [원 source summary](../dense_source_point_teacher_v1/SUMMARY.json)

실사 native gold·fresh blind 일반화·모델개선을 인증하지 않는다. RGB/NPZ/config/원 mask·depth/weights/forward/보호 평가 접근0이다. saved reference의 원 crop 좌표는 대수적 association이며 실제 pixel 가시성이나 keypoint 좌표 보정의 증거가 아니다.
