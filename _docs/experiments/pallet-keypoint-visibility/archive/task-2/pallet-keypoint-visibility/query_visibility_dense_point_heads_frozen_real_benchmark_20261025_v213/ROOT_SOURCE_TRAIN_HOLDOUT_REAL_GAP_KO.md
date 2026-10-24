# SOURCE TRAIN·SOURCE HOLDOUT·실사 평가의 간격

학습에 사용한 SOURCE point에서는 거의 모두 맞혔지만, SOURCE HOLDOUT과 실사에서는 특히 E recall이 낮다. 이 수치 간격은 관측됐으나 감독 의미·입력 분포·노출 차이가 함께 있어 원인이나 실사 개선 효과를 분리해 입증하지는 못한다. C63은 결과 전 지정한 주 모델이고 C62는 point-only 대조 팔이며, 두 모델을 모두 보고한다.

CM 행은 D/E, 열은 예측 D/E/U다. Recall은 각 기존 known 클래스의 전체 분모를 사용한다. 서로 다른 집합의 정확도를 합치지 않는다.

|집합|모델|N (D/E)|CM [D행; E행]|D precision / recall|E precision / recall|known U|
|---|---|---|---|---|---|---|
|SOURCE TRAIN point|C62|6096 (5853/243)|`[[5847,6,0],[0,243,0]]`|100.00% / 99.90%|97.59% / 100.00%|0|
|SOURCE weak44|C62|44 (26/18)|`[[18,8,0],[4,14,0]]`|81.82% / 69.23%|63.64% / 77.78%|0|
|SOURCE HOLD point|C62|938 (899/39)|`[[881,18,0],[9,30,0]]`|98.99% / 98.00%|62.50% / 76.92%|0|
|실사 paired|C62|1237 (1147/90)|`[[709,438,0],[64,26,0]]`|91.72% / 61.81%|5.60% / 28.89%|0|
|SOURCE TRAIN point|C63|6096 (5853/243)|`[[5849,4,0],[0,243,0]]`|100.00% / 99.93%|98.38% / 100.00%|0|
|SOURCE weak44|C63|44 (26/18)|`[[26,0,0],[0,18,0]]`|100.00% / 100.00%|100.00% / 100.00%|0|
|SOURCE HOLD point|C63|938 (899/39)|`[[885,14,0],[9,30,0]]`|98.99% / 98.44%|68.18% / 76.92%|0|
|실사 paired|C63|1237 (1147/90)|`[[843,304,0],[71,19,0]]`|92.23% / 73.50%|5.88% / 21.11%|0|

SOURCE TRAIN point는 원7,040행을 보존하고, fit-eligible6,096(D5,853/E243)과 UNKNOWN-loss944를 분리한 apparent fit이다. 전체 원 inputU64는 UNKNOWN 쪽에 남는다. Weak44는 공개 SOURCE TRAIN 3frame의 factual8·SHAM18 D와 COVER18 weak E다. C63의44/44는 직접 학습한 약한 감독 적합도이며, C62도 같은 원frame을 old auxiliary에서 보았으므로 C62의32/44 역시 독립 검증으로 부르지 않는다.

SOURCE HOLDOUT은18 component/base groups·72장면·576개 own native 상태에서 나온1,152 input rows다. Known938(D899/E39)과 UNKNOWN214를 구분하며, 제외사유는 중복될 수 있다. 원 inputU80도 제외사유에 보존된다. 같은 base·variant·native와 anchor 문맥에서 반복된 행이므로938개 독립 실사 사례가 아니다. 기존에 알려진 SOURCE 설계와 역사적 노출이 있는 합성 내부 holdout으로, fresh blind 검증이나 공개 실사 holdout으로 일반화하지 않는다.

실사 paired는 기존 human D/E1,237(D1,147/E90)이다. C62→C63에서 D 정답은709→843(+134), E 정답은26→19(−7)로 바뀌었다. SOURCE HOLDOUT에서는 D881→885(+4), E30→30이었다. SOURCE TRAIN의 E100%가 실사 E100%를 뜻하지 않으며, C63의 D 증가와 E 감소를 함께 보아야 한다. Legacy보다 개선됐다는 결론이나 더 좋은 팔의 사후 채택은 이 문서에서 하지 않는다.

이전 finite-end327에서 dense point6,096으로 바뀔 때 target 정의와 데이터 노출도 함께 바뀌었다. 이를 순수 라벨 하나의 효과로 해석하지 않는다. C62/C63는 같은 구조·fresh init·old192/point32 draw·4,000회 갱신에 대해 C63만 추가 weak branch(.25)를 쓰는 사전 대조다. SOURCE procedural point와 public local-feature weak target은 인간 native visibility gold와 다르며, source→real 간격만으로 shortcut·국소 정보 소실·라벨 문제 등 특정 원인을 확정하지 않는다.

표의 known U0는 availability/유한 logit 이후 고정0 문턱으로 이진 응답을 했다는 뜻이다. Confidence·calibration·불확실성 해결의 증거가 아니다. 실사 전체1784의 원 inputU45와 주평가 U0도 별개다. 반복 노출된 실사/HOLDOUT 수치로 모델·데이터·문턱·checkpoint를 조정하거나 선택하지 않았고, 이 문서도 새로운 최적화를 수행하지 않는다.

## 저장 보고서·표

- [SOURCE TRAIN/weak 보고서](../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/PORTABLE_REPORT_KO.md)
- [SOURCE CM/P/R/U 표](../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/source_fit_readout_b_v1/SOURCE_APPARENT_CM_P_R_U.csv)
- [SOURCE native8 표](../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/source_fit_readout_b_v1/SOURCE_NATIVE8_CM_P_R_U.csv)
- [SOURCE HOLDOUT V2 요약](source_point_holdout_evaluation_v2/SUMMARY.json)
- [SOURCE HOLDOUT 완료](source_point_holdout_evaluation_v2/COMPLETE.json)
- [실사 전체 모델 저장 metric](saved_real_head_evaluation_v1/current1784/METRICS.csv)
- [실사 독립 수치 QA](independent_saved_result_review_a/actual_review_v1/REVIEW.json)
- [고정12 갤러리](fixed12_dense_point_real_visual_v1/PORTABLE_GALLERY_KO.md)
