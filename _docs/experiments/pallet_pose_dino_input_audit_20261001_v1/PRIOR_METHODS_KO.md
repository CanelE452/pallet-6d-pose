# DINO 후보 위치 입력의 선행·계약 감사

2026-10-01. **권고는 아래 한 가지 385차원 입력의 TRAIN 계약 감사 진행이며, selector 학습이나 T/R 개선 판정이 아니다.** 기존 DINO를 처음 도입하는 실험도, RGB로 전체 후보를 고르는 첫 실험도 아니다. 이번 작성자는 로컬 코드·봉인된 과거 보고서·checkpoint 바이트만 읽었다. 새 이미지 forward, fit, GT 조회, pose 생성, 후보 선택 정책 실행은 모두 0회다. 현재 Q의 source 43/45 실패와 실사 미실행 상태는 바뀌지 않는다.

## 확인된 선행과 반대 근거

| 선행 | 실제 입력·선택 단위 | 기록과 이번 해석 |
|---|---|---|
| S0/S1 Stage4 router | 기하·confidence·두 expert 불일치와 frozen S1 pose-head GAP448, MLP64/32, 두 expert의 전체 pose 선택 | 단순 raw RGB만 쓴 선행이 아니다. synthetic TEST mean ADDnorm은 S0 0.09621724, routed 0.09768815로 더 좋은 S0를 이기지 못했다. 현재 R0/PoseFix·T/R 목적식과는 다르다. [입력 코드](../../../scripts/research/pallet_selector_recovery_v1/router_features.py), [기존 독립 감사](../pallet_pose_union_selection_20261001_v1/PRIOR_AND_METHOD_AUDIT_KO.md). |
| DINO localization / wide | frozen DINO final patch tokens, learned heatmap head, R0 점 prior; wide는 같은 픽셀 배율에서 crop만 가로·세로 2배 | 두 실험 모두 일부 큰 코너 오차를 복원했지만 자체 damage·source 조건을 통과하지 못했다. 토큰이 존재한다는 사실이 안전한 보정을 보장하지 않는다. [localization 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization/RESULTS.json), [wide 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_wide/RESULTS.json). |
| DINO wide_visual / visual_long / source_diversity | 점 prior·hint를 모두 끈 영상 전용 heatmap head; 더 긴 학습 및 8,192 source 이미지로 확대한 후속 | 각 기록의 SYN/MIX는 recovery 조건을 만족해도 damage·PCK20·source PCK10/P90 조건은 FAIL이다. 단순 학습 연장이나 source 수 증가가 전이를 해결했다는 근거가 없다. [영상 head 코드](../../../scripts/research/pallet_type_selftrain_v1/dino_wide_visual_model.py), [visual 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_wide_visual/RESULTS.json), [long 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_visual_long/RESULTS.json), [diversity 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_source_diversity/RESULTS.json). |
| DINO mid_feature / RGB detail | 같은 backbone block6의 추가 특징 또는 stride4 RGB detail branch를 기존 head에 추가 | 두 결과 모두 자체 damage·PCK20·source PCK10/P90 FAIL이다. 고해상도·중간 특징을 더하면 해결된다고 단정할 수 없다. 정보와 head 용량이 함께 바뀌었으므로 원인 분리도 아니다. [mid 계약](../../../scripts/research/pallet_type_selftrain_v1/dino_mid_feature.py), [mid 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_mid_feature/RESULTS.json), [RGB detail 코드](../../../scripts/research/pallet_type_selftrain_v1/dino_rgb_detail_model.py), [detail 결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_rgb_detail/RESULTS.json). |
| DINO evidence gate | 학습된 SYN/MIX heatmap의 5×5 질량·entropy·이동·경계 등 16개 특징, 점별 MLP 수락 | **SYN/MIX 모두 자체 6개 조건 PASS**다. 따라서 DINO 선행이 전부 실패했다고 요약하면 틀린다. 다만 source 점별 이득 감독과 threshold calibration이며 현재 whole-pose T/R 안정성 45/5 기준과 다르다. 결과 자체도 `goal_complete=false`, `independent_confirmation=false`다. [특징 코드](../../../scripts/research/pallet_type_selftrain_v1/dino_evidence_gate_model.py), [감독·계약](../../../scripts/research/pallet_type_selftrain_v1/dino_evidence_gate.py), [결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_evidence_gate/RESULTS.json). |
| **DINO joint** | R0/SYN/MIX의 전체 8코너 세트 × 4채널 순열 = 12후보. 후보 위치의 두 heatmap 질량과 R0 대비 차이 등 98개 특징, MLP64/32, 하나의 전체 세트 선택 | **후보 위치의 영상 근거로 whole-hypothesis를 선택한 선행이 이미 있다.** recovery5·frames3 FAIL, 나머지 보호 조건 PASS다. source threshold를 쓰며 감독은 픽셀 복원·손상, 입력에 치수/K가 없다. 이번 제안과 가장 가까운 음성 근거다. [descriptor·ranker](../../../scripts/research/pallet_type_selftrain_v1/dino_joint_model.py), [후보·감독 계약](../../../scripts/research/pallet_type_selftrain_v1/dino_joint.py), [결과](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_joint/RESULTS.json). |

Structured DHT의 soft cost·회귀와 RGB utility selector도 이미 실행됐다. source 성공 후 실사 identity 유지 또는 clean 손상 기록을 보존해야 한다. 이번 제안의 근거를 단순히 “분류 대신 회귀”, “RGB 추가”, “whole candidate”라는 이름으로 새롭게 포장하지 않는다. [기존 목적함수 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md).

이번 고정 후보는 **학습된 DINO heatmap과 새 좌표를 만들지 않고, 공식 frozen raw patch-token을 현재 R0/PoseFix의 이미 존재하는 물리 pose 투영 위치에서 읽는다**는 구체적 차이가 있다. 현재 기하271·후보 pool·물리 T/R 감독을 유지한다면 변화는 이 입력 그룹 하나다. 기존보다 우수하다는 증거는 아직 없다.

## 실제 로컬 backbone과 정보 범위

[로더](../../../scripts/research/pallet_type_selftrain_v1/dino_localization_assets.py)의 `load`는 official repository commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` 및 `dinov2_vits14`를 사용한다. `eval().requires_grad_(False)`이며 patch14, embedding384를 검사한다. [BACKBONE 영수증](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization/BACKBONE.json)의 checkpoint는 88,283,115 bytes, SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`다. 이번 읽기에서 실제 로컬 바이트의 SHA와 일치를 확인했다. 다운로드·모델 로드는 하지 않았다.

원본 checkpoint와 pinned upstream 소스는 로컬 `outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets/` 아래에 있다. 공개 저장소에 대용량 checkpoint가 포함된다는 뜻은 아니다. pinned `dinov2/layers/patch_embed.py`는 kernel=stride=14인 Conv2d 뒤 행 우선 flatten을 사용하고, `models/vision_transformer.py:forward_features`는 모든 block을 거친 뒤 final LayerNorm의 `x_norm_patchtokens`를 반환한다. 이는 분류 확률·코너 confidence가 아니며, self-attention을 지난 토큰이라 샘플 위치 주변만의 독립적 관측도 아니다. class/register token을 섞지 않는다.

과거 cache의 이미지 ID가 같아도 재사용이 자동으로 성립하지 않는다. RGB SHA, 준비영상 크기·padding, 선택된 R0 box, affine, backbone/runtime, crop/resize, token layout·dtype를 대조해야 한다. source cache에는 `target`·`target_valid` 배열도 들어갈 수 있으므로 입력 감사는 승인된 feature·matrix 항목만 읽어야 한다. [cache 생성 코드](../../../scripts/research/pallet_type_selftrain_v1/dino_wide.py), [diversity cache 영수증](../pallet_type_selftrain_v1/selftrain_recovery_v1/dino_source_diversity/CACHE_COMPLETE.json). 전체 8,505행에는 source와 과거 real이 함께 있으므로 8,505를 현재 source 일치 수로 부르면 안 된다. 정확한 현재 일치 수와 affine 검산은 별도 cache identity 감사에 따른다.

## 제안하는 단일 고정 입력: projected-corner mean384 + support1

이 절은 구현할 입력 계약 하나다. 다른 layer·crop·token 수·pooling·threshold를 비교해 고르는 탐색은 제안하지 않는다.

1. **영상과 crop은 frame당 하나다.** 현재 봉인된 R0 selected detection의 `box_xyxy`에서 기존 `axis_aligned_crop_matrix(box, input_shape=(384,288), expansion=1.25)`를 그대로 계산하고 affine translation에 `(144,192)`를 더한다. 모든 expert/W-D 후보와 R0 anchor가 동일 affine을 공유한다. 출력 crop은 **H768×W576**이다. 후보별 crop, GT box, 다른 detector 선택, 추가 padding을 만들지 않는다. source의 이미 준비된 RGB와 그 좌표계 K를 그대로 사용한다. [원 affine](../../../scripts/research/pallet_sensors_submission_v1/posefix_contract_math.py), [실제 R0 crop](../../../scripts/research/pallet_posefix_large_error_v1/core.py), [wide affine·warp](../../../scripts/research/pallet_type_selftrain_v1/dino_wide.py).

2. **기존 DINO 전처리를 고정한다.** OpenCV `INTER_LINEAR`, `BORDER_CONSTANT`의 wide warp 후 BGR→RGB, FP32를 사용한다. 기존 mean subtraction/addition 경로까지 그대로 유지한 뒤 `/255`, `torch.interpolate(size=(784,588), mode='bilinear', align_corners=False)`, ImageNet mean `[.485,.456,.406]`/std `[.229,.224,.225]`를 적용한다. final patch tokens를 **384×56×42**의 contiguous 배열로 만들고 기존처럼 FP16으로 저장한다. 입력 감사에서 읽을 때 FP32로 올린다. 별도의 token L2 normalization, learned heatmap head, layer 선택, fine-tuning은 없다. 이전 narrow 384×288→392×294의 384×28×21 cache는 wide cache의 대체물이 아니다.

3. **샘플 위치는 기존 whole-pose의 물리적 8코너 투영이다.** 승인된 각 후보의 `R_cf`, `centroid`, `cf_extents`와 현재 K를 사용한다. O의 고정 cuboid signs 순서로 `X_camera=X_cf R_cfᵀ+t`, `p=(K X_camera)xy/(K X_camera)z`를 계산한다. 새 PnP/fit을 실행하지 않는다. 원래 observed `q9`는 새 appearance 입력의 샘플 위치·visibility를 정하는 데 쓰지 않는다. center index8도 사용하지 않는다. center는 8코너 표면 근거와 같지 않으며 기존 feature271·후보 생성에는 이미 남아 있다. [기존 순수 투영식](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/direction_features.py).

4. **support는 접근 가능한 위치만 뜻한다.** 각 코너가 finite이고 camera depth `z>0`이며 prepared native image의 `0≤x<W, 0≤y<H`와 crop의 `0≤x<576, 0≤y<768` 안에 있으면 supported다. GT visibility, semantic mask, confidence threshold, 뒷면/가림 판정은 넣지 않는다. 이미지 안의 다른 물체·실제 가림·letterbox/padding도 이 조건만으로 구별되지 않는다. 따라서 supported를 visible/correct라고 표시하면 안 된다.

5. **half-pixel과 patch center를 정확히 연결한다.** crop 좌표 `(x_c,y_c)`의 연속 token 좌표는

   `u=(x_c+0.5)×42/576−0.5`, `v=(y_c+0.5)×56/768−0.5`다.

   이는 resize의 half-pixel mapping과 14×14 patch의 중심을 합친 것이다. `x_c/14`나 heatmap stride4의 좌표식과 섞지 않는다. supported crop 내부지만 첫/마지막 patch 중심 밖인 위치에는 **border bilinear interpolation**을 적용한다. 즉 token 좌표를 `[0,41]×[0,55]`로 clip한 뒤 FP32 보간한다. crop 밖의 코너를 clip해 supported로 바꾸지 않는다.

6. **descriptor385를 만든다.** supported 코너들의 FP32 sampled values를 채널별로 독립 정렬한 뒤 FP32 평균을 구한다. 같은 비중으로 합쳐 **supported 개수**로 나눈 평균384, 그리고 `supported_count/8` 한 값을 연결한다. 정렬은 같은 값의 multiset에 대한 합산 순서만 고정하며 새로운 정보를 더하지 않는다. 0지원이면 descriptor는 finite zero384 + fraction0이다. unsupported 코너가 있다고 기존 pose 후보를 invalid로 바꾸거나 행을 버리지 않는다. all-invalid source1행도 그대로 유지한다. 개별 mask와 projection은 감사 기록에 남기되 추가 predictor 입력으로 확대하지 않는다.

7. **C2와 corner order를 GT 없이 처리한다.** 8코너 평균은 C2에 따른 cuboid 코너 순열에도 같은 집합을 보므로, 별도 GT 대칭 선택·quarter-turn 후보·의미적 index 재정렬 없이 동작한다. 수학적으로는 C2보다 강한 모든 코너 순열 불변성이다. 채널별 정렬로 **이미 계산된 동일 sampled values의 순열**에는 bitwise 동일한 평균을 얻는다. C2 회전으로 투영 좌표를 다시 계산할 때 생기는 부동소수점 차이까지 제거한다는 뜻은 아니다. 이 pooling은 앞/뒤·좌/우 대응과 코너 간 topology를 버린다. 이것은 복잡도를 제한하는 선택인 동시에 실패 가능한 정보 손실이다.

8. **TRAIN-only 정규화 뒤 anchor 차분을 붙인다.** 기존 eligible TRAIN2,598행의 원래 valid인 R0 두 가설 descriptor, 즉 기존 유효 R05,194개를 기준으로 FP32 mean/std를 구하고 std floor `float32(1e-6)`를 사용한다. 지원 수가 작다는 이유로 정규화 표본을 다시 고르지 않는다. `(a−mean)/std`는 FP32, 그 후 FP64로 올려 `normalized(candidate)−normalized(R0 operational GEO anchor)`를 만든다. 기존271의 뒤에 이385만 붙여 **656차원**이다. 기존 anchor identity는 GT-free GEO와 원래 candidate valid에서 정한다. any-valid인데 anchor가 없으면 계약 오류로 STOP, all-invalid는 추가 입력0이다. 평균과 std를 VAL/실사에서 다시 계산하지 않는다.

현재 source/real 비교의 후보 pool은 R0_ONLY의 두 W/D 또는 각 seed별 R0+paired DIVERSE의 네 whole pose다. 세 refiner를 한 pool로 합치거나, token feature로 새 좌표·pose·검출을 만들지 않는다. 런타임 정보는 한 장 RGB·물리 치수·기존 K와 그 영상에서 나온 기존 예측에 한정된다.

## 진행 판단과 중단 경계

**위 385 입력의 구현·재사용성 감사는 근거가 있다. 곧바로 새 성능 실험이 정당화되었다는 뜻은 아니다.** raw271은 직접적인 RGB appearance를 포함하지 않고, 이전 O 감사의 residual18도 투영과 관측좌표에서 정해진 기하다. 제안한 token descriptor는 동일 기하에 대해 영상 내용이 달라질 때 달라질 수 있는 정보경로다. 그러나 이론적 정보 추가와 현재 데이터에서 유용한 분별력은 다르다.

우선 label을 읽지 않는 단계에서 checkpoint/image/cache/affine SHA, TRAIN membership, layout, support/zero-support, scalar bilinear 대조, anchor 차분0, all-invalid 유지, C2 순열 fixture를 검증하고 입력을 동결한다. 누락 cache는 임의 대체하지 않는다. 새 TRAIN forward가 필요한 경우 root가 별도 예산·runtime·실행을 승인하고 기록해야 한다. source VAL의 quality와 real GT·새 routing은 이 입력 감사에서 읽거나 계산하지 않는다. 실제 데이터에 대한 support/분별 통계는 현재 문서의 미실행 제안을 결과로 바꾸지 않고 별도 영수증에 기록한다.

향후 학습 여부를 결정하더라도 Q의 loss·target·scale·originalvalid·solver·zero initialization·예산·source45 및 original AND matched real5 기준을 동시에 바꾸지 않아야 입력 변화의 결과를 해석할 수 있다. 385추가는 파라미터 수·conditioning도 바꾸므로 순수하게 “RGB 원인”만을 분리하는 실험은 아니다. 모든 seed와 전체 failure 분모를 유지한다.

가장 강한 반대 근거는 이미 존재하는 **whole-candidate DINO joint의 복원 실패**, 여러 DINO head의 clean 손상, source 확대 후에도 유지된 전이 문제다. 14-pixel token과 전역 attention, corner mean pooling은 작은 위치·orientation 차이를 충분히 구별하지 못할 수 있다. 잘못된 R0 box가 다른 물체를 가리키면 같은 crop과 고정 pose pool 밖의 팔레트를 이 입력이 복구하지 못한다. 반대로 evidence gate의 일부 성공 때문에 pretrained appearance가 원천적으로 무용하다고 단정할 수도 없다. **권고 결론은 한 계약의 입력 감사 진행이며, 학습 승인·목표 달성·T/R 향상은 아직 아니다.**
