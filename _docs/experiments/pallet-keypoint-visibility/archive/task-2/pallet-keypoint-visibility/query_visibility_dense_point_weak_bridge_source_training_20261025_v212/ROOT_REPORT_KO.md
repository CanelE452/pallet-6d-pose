# V212 SOURCE dense-point 학습 결과

C63은 사전 지정 주 모델, C62는 point-only 대조 팔이다. 같은 RAW/native/original q8 구조118,359개 파라미터를 새 초기값으로 각4000회 학습했다. SOURCE apparent fit 결과이며 실사 개선·모델 채택·목표 해결을 주장하지 않는다.

POINT7040 전체를 보존하고 fit-eligible6096(D5853/E243), UNKNOWN-loss944를 분리했다. Weak44는 D26/E18의 source local-feature 약한 감독이다. C62도 old auxiliary에서 같은 공개3frame을 보았으므로 이 weak44 점수는 독립 검증이 아니다.

|domain|model|all_rows|known_N|D_N|E_N|CM_D_E_by_D_E_U|D_precision|D_recall|E_precision|E_recall|U_known|U_known_rate|UNKNOWN_target_rows|UNKNOWN_target_predictions|original_input_U|all_prediction_U|input_U_in_known|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|POINT7040|C62|7040|6096|5853|243|[[5847, 6, 0], [0, 243, 0]]|1.0|0.9989748846745259|0.9759036144578314|1.0|0|0.0|944|{'D': 847, 'E': 33, 'U': 64}|64|64|0|
|WEAK44|C62|44|44|26|18|[[18, 8, 0], [4, 14, 0]]|0.8181818181818182|0.6923076923076923|0.6363636363636364|0.7777777777777778|0|0.0|0|{'D': 0, 'E': 0, 'U': 0}|0|0|0|
|POINT7040|C63|7040|6096|5853|243|[[5849, 4, 0], [0, 243, 0]]|1.0|0.9993165897830173|0.9838056680161943|1.0|0|0.0|944|{'D': 866, 'E': 14, 'U': 64}|64|64|0|
|WEAK44|C63|44|44|26|18|[[26, 0, 0], [0, 18, 0]]|1.0|1.0|1.0|1.0|0|0.0|0|{'D': 0, 'E': 0, 'U': 0}|0|0|0|


## 감독과 실험 범위

Own-image procedural point teacher. All7040 retained; loss-eligible6096 D5853/E243; 944 UNKNOWN-loss preserved. Source TRAIN apparent, not human native gold.
44 assistant local visible-feature weak rows: factual8 +SHAM18 D +COVER18 E; 3 SOURCE public frames. C62 also sees these frames in old auxiliary source pools, so not independent validation.
C63 primary a priori vs C62 point-only; additional weak branch .25 only. Earlier finite-end327 versus densepoint6096 changes definition and exposure, not label-only causality.
SOURCE point HOLDOUT18 is a separate V213 lane; never read or merge here.

CM 행 D/E, 열 D/E/U. Recall은 예측 U를 포함한 원 known 분모를 사용하고 UNKNOWN target은 정확도에서 분리한다. InputU와 known-predictionU도 별개이며 U0은 confidence가 아니다.

이전 finite-end327에서 densepoint6096으로의 전환은 감독 정의와 노출량도 달라졌다. C63과C62만 같은 point/old192 draw의 추가 weak branch(.25) 대조다. 두 팔은 SOURCE 점수와 무관하게 모두 별도 V213으로 이어간다.

## paired source 변화

|domain|known_N|both_correct|recovered_by_C63|damaged_by_C63|both_wrong|same_prediction|different_prediction|net_correct_change|direction|
|---|---|---|---|---|---|---|---|---|---|
|POINT7040|6096|6088|4|2|2|6090|6|2|C62 to predeclared C63; source apparent only, no adoption|
|WEAK44|44|32|12|0|0|32|12|12|C62 to predeclared C63; source apparent only, no adoption|


## 고정 학습·감사 범위

총8000 updates, runtime 394.74초. AdamW3e-4/wd1e-4/clip5, init1810, old RNG1810·point1811·weak1812. old192+point16D16E, C63만 weak8D8E 가중치.25. 원q/L/native/U·zeroSAM·기존role/nominal fitmask를 유지하고 oldDIRECT input/target NPZ는 제외했다. 두 최종 상태를 먼저 봉인한 뒤 SOURCE 예측을 저장했다.

독립 감사는 draw/hash·로그 scalar 산식·opaque checkpoint/봉인을 검토했다. 신경망 loss/gradient/activation 전체를 독립 재학습한 것은 아니다. 원GT·source RGB·input/target NPZ·checkpoint 모델 해석은 이 보고서에서 수행하지 않았다.

## 저장 loss 요약

|arm|field|updates|first|last|mean|minimum|maximum|scope|
|---|---|---|---|---|---|---|---|---|
|C62_DENSE_POINT_RAW_Q8|old_five_mean|4000|0.9460678100585938|0.019707614555954933|0.09227525335445534|0.006400112994015217|0.9460678100585938|Logged scalars only; not independent neural-loss replay|
|C62_DENSE_POINT_RAW_Q8|point_loss|4000|0.6931471824645996|9.819496881391387e-06|0.05025172813715215|2.3953261916176416e-06|0.8565079569816589|Logged scalars only; not independent neural-loss replay|
|C62_DENSE_POINT_RAW_Q8|weak_loss|4000|0.0|0.0|0.0|0.0|0.0|Logged scalars only; not independent neural-loss replay|
|C62_DENSE_POINT_RAW_Q8|weak_loss_weight|4000|0.0|0.0|0.0|0.0|0.0|Logged scalars only; not independent neural-loss replay|
|C62_DENSE_POINT_RAW_Q8|total_loss|4000|1.6392149925231934|0.01971743442118168|0.14252698149706702|0.006414358038455248|1.6392149925231934|Logged scalars only; not independent neural-loss replay|
|C62_DENSE_POINT_RAW_Q8|grad_norm|4000|0.6700244545936584|1.2604756355285645|9.768583151292056|0.15072575211524963|272.6828918457031|Logged scalars only; not independent neural-loss replay|
|C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8|old_five_mean|4000|0.9460678100585938|0.02078728750348091|0.09350884033530019|0.0036329682916402817|0.9460678100585938|Logged scalars only; not independent neural-loss replay|
|C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8|point_loss|4000|0.6931471824645996|1.825035724323243e-05|0.05040676178083842|9.126947588811163e-07|0.6931697130203247|Logged scalars only; not independent neural-loss replay|
|C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8|weak_loss|4000|0.6931471824645996|2.7864959974976955e-06|0.02449305924472001|2.8312194899626775e-07|0.6938420534133911|Logged scalars only; not independent neural-loss replay|
|C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8|weak_loss_weight|4000|0.25|0.25|0.25|0.25|0.25|Logged scalars only; not independent neural-loss replay|
|C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8|total_loss|4000|1.8125017881393433|0.02080623432993889|0.15003886679490097|0.0042284647934138775|1.8125017881393433|Logged scalars only; not independent neural-loss replay|
|C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8|grad_norm|4000|0.6698909401893616|0.9476461410522461|10.300930683154613|0.15933789312839508|242.42774963378906|Logged scalars only; not independent neural-loss replay|


## 연결

- [全source CM/P/R/U](source_fit_readout_b_v1/SOURCE_APPARENT_CM_P_R_U.csv)
- [native8](source_fit_readout_b_v1/SOURCE_NATIVE8_CM_P_R_U.csv)
- [원source 그룹 수](source_fit_readout_b_v1/SOURCE_GROUP_COUNTS.csv)
- [paired 변화](source_fit_readout_b_v1/SOURCE_PAIRED_C62_TO_C63.csv)
- [loss 요약](source_fit_readout_b_v1/LOSS_SCALAR_SUMMARY.csv)
- [8000 logged loss](source_fit_readout_b_v1/ALL8000_LOSS_SCALARS.csv)
- [전체7084 source scalar](source_fit_readout_b_v1/ALL7084_SOURCE_TARGET_PREDICTION_READOUT.csv)
- [공개용 owned binary 명세](source_fit_readout_b_v1/OWNED_BINARY_PUBLICATION_BINDINGS.json)
- [SOURCE COMPLETE](runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/COMPLETE.json)
- [두 최종 봉인](runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/ALL2_FINAL4000_FROZEN_BEFORE_ANY_SOURCE_OUTPUTS.json)
- [독립 fit 감사](independent_actual_fit_review_c/actual_review_v1/REVIEW.json)
- [입력 검증](runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/FIT_INPUT_VALIDATION.json)
- [source 감독 protocol](fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json)
- [보고서 코드](source_fit_report_preparation_b/build_source_fit_readout.py)
