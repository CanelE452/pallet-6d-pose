# V197: TRAIN 입력과 부위 감독의 대응

확인한 것은 감독 정의와 입력의 관계다. 새 예측·학습성능·실사 성능은 측정하지 않았다. 기존 TRAIN327 weak 라벨을 바꾸지 않고 전체109개 metadata를 확인했으며, 모델 점수 없이 미리 정한 hash 규칙의12triplet만 그림으로 만들었다.

## 전체 TRAIN109의 범위

|권한|triplet|기존 known D/E|원본 범위|저장 component|canonical metadata|q→enclosure 중심 거리96 최소/중앙/최대|
|---|---|---|---|---|---|---|
|구성적 합성 finite-end|82|164/82|41 bases|41|82제공/0UNKNOWN|0.445/6.759/33.709|
|공개 실사 RGB-weak end +YCB 합성차폐|27|54/27|12 frames|1|0제공/27UNKNOWN|4.027/11.709/35.614|

거리96은 원crop의 q중심(47.5,47.5)에서 저장된 finite-enclosure 중심까지의 설명용 거리다. 물리 corner 오차나 q 보정 정답이 아니다. 합성 enclosure의 폭/높이는 각각0.918~1.567 /1.024~2.036, 공개실사 weak enclosure는6.651~19.170 /8.539~23.093이다. 두 권한의 영역 정의가 다르므로 같은 물리 부위 면적이나 난이도의 직접 비교로 해석하지 않는다.

|native|0|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|---|
|합성 triplet|26|19|13|22|2|0|0|0|
|공개실사 weak triplet|9|7|0|0|4|7|0|0|

native6/7의 direct known 감독은0이며 없는 사례를 보충하지 않았다. native0..7 원입력을 보존한 것과 모든 native의 감독·성능이 검증된 것은 다르다. canonical 값82개는 기존 구성적 metadata이고, real27개의 canonical UNKNOWN은 그대로다. 이를 추정 permutation으로 채우거나 RGB-weak 라벨을 물리 native gold로 승격하지 않는다. 또한 canonical UNKNOWN만을 이유로 기존 약한 부위 관찰을 폐기하거나 완벽 mesh 인증을 새 필수조건으로 삼지 않는다.

27개 real triplet은12개 원frame이지만 저장된 component는1개다.109개를109개 독립 영상으로 세지 않는다. 이 real control의 E는 촬영된 자연가림 정답이 아니라 기존 RGB-weak 끝 관찰과 YCB donor 합성차폐의 권한이다.

## 고정 화면의 범위와 확인된 점

- 선택12=합성8+공개실사4. 3variant×native8의288행을 유지했고 D24/E12/UNKNOWN252, 원inputU24(8.33%)다. U24는 선택된 real96행에 포함되며 unknown/padding을 자연 배경이나 D로 채우지 않는다. 이는 전체109의 all8 가용성 비율이 아니다.
- 같은 exact TRAINpacket의 full_RGB가36/36개 있었고 다른 영상으로 대체하지 않았다. clean96·valid96·q/L·all8은 원본 그대로이며 색 표시는 별도 display 사본이다.
- Root는 고정00~07 contact8+BEFORE문맥8, B는08~11 contact4+BEFORE문맥4를 실제 보았다. A는 생성metadata만 검토했으며 추가 실제뷰는0이다. 288개 개별 crop이나36개 문맥 전부를 눈으로 보았다고 주장하지 않는다.
- Root/B 관찰에서는 일부 합성 제어가 작고 목재·배경과 비슷한 색으로 보이며, 다른 합성은 구분되는 색도 있었다. 공개실사 control에서는 큰 YCB donor와 분리된 weak끝 영역이 보이는 예가 있었다. 전체문맥은 부재·끝 선언을 이해하는 데 도움을 주지만 물리 정체성 인증은 아니다.

## 아직 확정하지 못한 원인

q중심과 감독 부위는 일치할 필요가 없으며, 본 metadata에서도 큰 차이가 존재한다. 이것만으로 좌표·라벨 오류라고 할 수는 없다. literal q 소속, crop 안 팔레트 존재, 선택된 finite-end의 차폐는 다른 질문이다.

작은 부위·색 대비·q와의 거리·재질/기하·감독 권한이 함께 달라져 어떤 요인이 학습이나 전이에 영향을 줬는지 분리되지 않았다. 마지막3×3 공간 표현에서 작은 단서가 약해질 수 있다는 설명도 검증하지 않은 가설이다. 이번 화면에는 예측이 없으므로 그 설명으로 모델 원인을 확정하거나 아키텍처를 선택하지 않는다.

## 다음 한 가지 TRAIN-only 확인 제안

같은 고정12 TRAIN 사례를 그대로 두고, 먼저 clean96만으로 선언된 동일 instance/part/end를 식별할 근거가 있는지 supported/ambiguous/UNKNOWN으로 기술한 뒤, 이미 생성된 동일 BEFORE 전체문맥과 기존 근거를 함께 보았을 때 달라지는지를 기록한다. 새 native 라벨·q 보정·자료 필터는 만들지 않는다. 문맥에서만 분명해지면 입력-목표 식별성의 공백을 의심할 근거가 되고, 문맥에서도 불분명하면 weak 부위 대응 자체의 미확정으로 남긴다. 어느 경우도 학습가능성이나 모델 원인을 증명하지 않는다.

이는 제안만이며 실행하지 않았다. 이전 문맥 노출이 있는 assistant의 기술적 관찰이므로 새 독립 human gold로 부르지 않는다. 새 모델·추론·학습·instrumentation 파이프라인·문턱·아키텍처·자료 선택과 HOLDOUT/current/held 평가 사용은 없다. 모델 미채택·전체 목표 미해결이다.

[고정 TRAIN 갤러리](../fixed_train_target_contract_viewer_v1/PORTABLE_GALLERY_KO.md) · [109개 기존 대응 metadata](../fixed_train_target_contract_viewer_v1/ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json) · [B 독립 검산/뷰 범위](../independent_saved_train_viewer_review_b/REVIEW.json) · [root 실제뷰 범위](../ROOT_ACTUAL_FIXED8_TRAIN_CONTACT_AND_CONTEXT_VISUAL_QA.json)

[기존 V194 source 학습 권한·분모](../../query_visibility_source_group_split_direct_end_training_20261024_v194/ROOT_REPORT_KO.md) · [기존 V193 source 제어 갤러리](../../query_visibility_source_end_pairs_opposite_sham_20261024_v193/source_pair_visual_readout_v2_caption_fit/PORTABLE_GALLERY_KO.md)

기존 source credit/권한을 유지하며 새 라이선스 확인을 수행한 것으로 주장하지 않는다.
