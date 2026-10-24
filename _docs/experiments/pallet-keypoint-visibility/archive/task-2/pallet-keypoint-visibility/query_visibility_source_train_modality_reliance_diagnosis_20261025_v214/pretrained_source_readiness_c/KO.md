# V214 pretrained SOURCE 코드 readiness

pretrained 자원이 없었던 것은 아니다. **ResNet18·DINO·SAM의 기존 SOURCE 경로가 있고, 현재 C61/C62/C63은 그 encoder를 이어받지 않은 작은 RAW96 CNN이다.** 이번 검토는 코드·기록 metadata만으로 이 차이를 확인했다. 기존 VAL/HOLD/실사 점수로 성공·실패를 판정하지 않았다.

| 경로 | 실제 확인한 계약 | 이번에 말할 수 있는 한계 |
|---|---|---|
| [V105 ResNet18](../../query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py) | ImageNet 원본 frozen. full96·central32를 각각224로 변환하고 global512 두 개+valid2=1026. feature COMPLETE 존재 | 공간 격자가 저장 head 입력에 남지 않는다. source 성능 판정은 하지 않음 |
| [V78 DINO](../../query_visibility_dino_query_support_20261023_v78/feature_preparation/pooling.py) | S/14 마지막16×16 token의 중심2×2/전체 평균, target 평균. 실제 feature seal 존재 | pretrained 미시도는 아님. 현재 point0/weak44 계약과 다름 |
| [V111→V162/V165 DINO reg4](../../query_visibility_counterfactual_frozen_features_20261024_v162/feature_preparation/PROTOCOL_BEFORE_ANY_ENCODER_OR_INPUT_BODIES.json) | T=contextCLS, Q=초기q의 context patch bilinear 표본, F=full96CLS, C=central32CLS, P=중심2×2 token 평균, 총1928. 실제 추출 terminal 존재 | **q-aware pretrained도 이미 시도됨.** 선택된 descriptor이며 전체 격자를 유지한 입력은 아님 |
| [V68 SAM2.1](../../query_visibility_sam21_frozen_20261023_v68/PROTOCOL.json), [V180 SAM1](../../query_visibility_frozen_sam_source_baseline_20261024_v180/runner_preparation/PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json) | 고정 box prompt의 mask/logit 경로 | 기존 mask cue는 latent spatial feature가 아니다. mask 음성을 native E/S로 바꿀 수 없음 |
| [현재 RAW 기반](../../query_visibility_local_support_raw_head_20261024_v181/head_preparation/local_support_head.py) | 96→stride8의12×12→AvgPool4의3×3→shared64. q8/native residual은 별도 | pretraining이 없다는 코드 사실은 어떤 실사 오류의 원인 증명이 아님 |

V75의 “pretrain”도 generic ImageNet encoder 학습이 아니라 frozen specialist P3 위의 SOURCE proxy head 학습이다. 전문 YOLO P3 65×9×9 경로 역시 과거에 있었으므로 “공간 특징을 한 번도 쓰지 않았다”는 넓은 주장은 맞지 않는다.

다음 작은 준비의 근거는 기존 ImageNet ResNet18 원본 body의 `features_only(layer2,layer3)`이다. 기록상 stride8/16·128/256채널이므로 raw96에서는12×12/6×6 격자가 된다(대수적 shape, 이번 runtime 검증 아님). **같은 frozen encoder의 pooled 읽기와 공간 격자 읽기**를 현재 SOURCE point6096와 public weak44를 분리한 상태에서 고정 비교하는 계약을 준비할 수 있다. 단순 C8 global descriptor 반복이나 DINO q-patch 추가를 새 실험으로 포장할 수 없다. q/L·native/U·원 TRAIN 역할을 보존하고 GT reference는 입력하지 않는다. head 용량까지 바뀌면 공간정보만의 인과효과라고 해석할 수 없다. 이 후보는 아직 채택·실행되지 않았다.

DINO224/14의16×16은 full96에서 명목 격자 간격6픽셀, central32에서2픽셀이다. 업샘플링이 원 RGB 증거를 늘리지는 않으며 이 간격은 receptive field나 물리적 해상도 인증이 아니다. SAM encoder latent는 새 cache·좌표 계약이 필요하고 기존 mask 출력 cache를 그대로 재명명할 수 없다.

모든 checkpoint 경로·SHA·size는 [기존 자원 inventory](../../query_visibility_controlled_pairs_20261023_v77/generic_backbone_inventory/GENERIC_BACKBONE_INVENTORY.json)와 protocol의 **기록값**이다. 이번 weight stat/hash/decode, NPZ/header, 이미지, 모델 import/forward/fit, 다운로드·Git은0이다. source/VAL가 함께 명시된 옛 protocol은 방법만 읽었고 실제 score 파일·배열은 열지 않았다. 전 실험에 대한 미시도/실패 전수판정은 아니다. 정확한 자원 경로·SHA와 text evidence bindings는 [READINESS.json](READINESS.json)에 있다.
