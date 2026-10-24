# V121 동결 특징·paired 감독 이력 감사

**DINOv2와 same-query COVER/SHAM margin은 이미 사용됐다.** 새 SOURCE 실험의 차이는 원 query가 의도한 instance/native finite part를 주소화하는지와 개입이 그 부위를 덮거나 보존하는지에 있어야 한다. 이번 감사는 성능을 다시 평가하지 않았다.

| 항목 | 코드·메타에서 확인한 이력 | 새 실험과의 관계 |
|---|---|---|
| 사전학습 특징 | 공식 DINOv2 ViT-S/14-reg weight를 strict load하고 eval/no-grad로 사용. V121 receipt는 666×1928 특징, 85 encoder 호출, 상태 불변을 기록 | DINO 재도입 자체는 신규 아님 |
| query/context 특징 | context CLS와 q 위치 patch token, local96·central32 CLS, 중심 patch 평균 및 8 scalar. 관계 abs/product를 거쳐 64-unit head(221,825개 파라미터) | 단순 local/context·query 관계 추가도 기존과 겹침 |
| paired 외부가림 감독 | C14는 TRAIN samewood 62개 COVER/SHAM pool에서 16쌍을 뽑고 기존 8쌍과 합쳐 24쌍의 margin을 사용 | paired loss 자체는 신규 아님 |
| intended native part와 original query 주소 | V121 probe는 weak reference±3px를 q8 carrier0에 넣으며 native 의미 UNKNOWN. C14 source도 native mapping UNKNOWN/AUX | 이 권한·주소 정렬은 기존 이력으로 입증되지 않음 |

V121은 새 fit 없이 이미 동결된 C14(step1200)를 사용했다. `pretrained=False`는 공장 함수의 자동 다운로드 설정이며, 바로 뒤에서 공식 checkpoint를 로드하므로 random feature 실험이 아니다. SOURCE receipt의 encoder/head 반복 호출과 전후 상태 동일성도 확인했다. 검토한 실행경로에는 ResNet·CLIP·SAM 특징 호출이 없지만 다른 phase 전체의 미시도 여부까지 뜻하지는 않는다.

V121의 PUBLIC probe는 37 weak cue 기록/22 물리 frame pair/44 RGB에 9개의 상관된 위치 이동을 적용한 666행이라는 **선언된** 구성이다. BEFORE q8의 slot0을 reference 이동점으로 바꾸고 그 tuple에서 L과 context를 계산해 BEFORE/AFTER에 동일하게 쓴다. slot0은 native ID가 아니다. 라벨은 BEFORE D/AFTER E인 기존 structural-cue **region**의 weak 가림 의미를 유지하며, 각 이동점의 target-pixel membership·정확한 native corner·좌표 정확성은 아니다.

C14 원학습은 frozen 1928 특징의 작은 head만 학습했다. supplied q에서 D 대 external E를 다루고 S/U는 binary loss에서 제외한다. 실제 trainer는 128행(D64/E64), old8+samewood16 matched pair를 만들며, 사전 설계의 손실은 `mean BCE + 0.25·mean max(0, 2−(z_cover−z_sham))`이다. q/L/crop96를 고정한 same-query control과 initial-q8 minmax context 정렬도 이미 있었다. 이는 logit ranking이며 DINO embedding을 새 contrastive loss로 학습했다는 뜻은 아니다. 역사적 SOURCE-VAL checkpoint selector 규칙은 코드에서 확인했으나 해당 점수는 열지 않았다.

최소 다음 후보는 동일 frozen DINO/head/decoder와 기존 BCE+hinge를 기준선으로 유지하면서, SOURCE에서 original q/q8/L에 의도한 instance/finite part 근거 및 COVER의 덮음·SHAM의 보존을 연결하는 자료 계약을 비교하는 것이다. 불명확한 대응은 UNKNOWN으로 남기고 cue/carrier 라벨을 native gold로 복사하지 않는다. 추가 contrastive 목적을 시험한다면 같은 입력·용량·분모·SOURCE FIT/DEV에서 목적만 분리해야 한다. 이는 새 학습 실행이나 실제 개선을 승인·입증하는 결과가 아니다.

현재 사용자 계약은 공개 실사 task fit을 허용한다. 예전 synth-only START 전제는 후속 승인으로 대체됐으며, current223 등 보호 평가를 학습·자료/모델 선택에 쓰지 않는 경계는 유지한다.

근거: [V121 실행코드](../../query_visibility_natural_cue_query_baseline_20261023_v121/feature_baseline_preparation/run_frozen_public666.py), [V121 입력 생성](../../query_visibility_natural_cue_query_baseline_20261023_v121/input_preparation/build_emulated_input_metadata.py), [특징 완료 메타](../../query_visibility_natural_cue_query_baseline_20261023_v121/frozen_C14_public_baseline/FEATURE_COMPLETE.json), [예측 봉인 메타](../../query_visibility_natural_cue_query_baseline_20261023_v121/frozen_C14_public_baseline/PREDICTION_SEAL.json), [V111 특징 정의](../../query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json), [V111 head](../../query_visibility_spatial_target_query_features_20261023_v111/train_spatial_target_query_head.py), [V119 원 trainer](../../query_visibility_consistent_query_context_20261023_v119/source_fit_preparation/train_source_once.py), [V119 사전 설계](../../query_visibility_consistent_query_context_20261023_v119/ROOT_C14_DESIGN_BEFORE_SOURCE_FORWARD.json). 정확 SHA·행 근거는 [REVIEW.json](REVIEW.json)에 있다.

실제 범위: 코드·사전정책·실행 봉인 메타 13개만 읽었다. current223/publicheld/oldVAL/HOLD 행·점수, ROOT score 보고서/HTML, 배열·PT·모델·RGB·GT payload는 stat/hash/read하지 않았다. 모델 import/forward/fit·전체 P 검색·기존 phase 쓰기도 0이다. 과거 결과의 좋고 나쁨이나 실제 실패 원인은 이 감사로 판단하지 않았다.
