# 고정 DINO 후보 입력: 식별 가능성과 계산 비용 검토

이 문서는 코드와 입력 계약을 검토한 설계 의견이다. 새 학습·가중치 probe·VAL/실사 성능 조회를 통한 선택·새 argmin·PnP·DINO forward는 실행하지 않았다. 현재 권고는 **고정 pose의 투영 코너 8곳에서 DINO384 토큰을 샘플링한 평균384개와 support 비율1개**를 후보별로 만들고 R0 anchor와 차분하는 단일 입력 감사다. 이 입력이 실제 T/R 회귀를 구별하거나 개선한다는 결과는 아직 없다.

## 기존 코드에서 확인한 차이

- [DINO backbone](../../../scripts/research/pallet_type_selftrain_v1/dino_localization_assets.py)은 고정 commit의 `dinov2_vits14`, 384채널·patch14를 사용한다. [wide 추출](../../../scripts/research/pallet_type_selftrain_v1/dino_wide.py)의 crop은 **높이768×너비576**, resize는 높이784×너비588이며 final `x_norm_patchtokens`를 `[384,56,42]` FP16으로 저장한다. 이 검토는 새 layer·backbone·해상도 선택을 권하지 않는다.
- [기존 evidence sample](../../../scripts/research/pallet_type_selftrain_v1/dino_evidence_gate_model.py)의 `sample`은 DINO patch token이 아니라 학습된 **corner heatmap의 log mass**를 stride4, `align_corners=True`로 읽는다. [whole-corner ranker](../../../scripts/research/pallet_type_selftrain_v1/dino_joint_model.py)도 그 함수를 사용한다. 이를 raw `[384,56,42]` 토큰에 그대로 적용하면 좌표계가 달라진다. 새 입력은 이 학습된 heatmap head나 새 corner 후보를 사용하지 않는다.
- 과거 [RGB context 추출](../../../scripts/research/pallet_selector_recovery_v1/extract_features.py)은 YOLO head 입력의 공간 평균을 구했고, [models.pack_inputs](../../../scripts/research/pallet_selector_recovery_v1/models.py)는 같은 expert의 두 W/D 후보에 같은 context를 broadcast했다. 후보별 투영 위치에서 읽는 이번 입력과는 구조가 다르다. 이 차이는 새 성능의 증거가 아니다.
- 기존 [whole-pose geometry](../../../scripts/research/pallet_selector_recovery_v1/features.py)와 [source 후보 고정](../../../scripts/research/pallet_pose_union_selection_20261001_v1/source_features.py)은 같은 expert의 q9에서 long/short 두 pose를 만든다. [고정 pose 재투영](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/direction_features.py)은 `cf_extents`, `R_cf`, `centroid`, 기존 K를 쓴다. 새 샘플링도 이 pose를 그대로 읽으며 PnP나 코너 좌표를 바꾸지 않는다.

관측 q9 또는 공통 crop 전체 평균만 읽으면 동일 expert의 두 W/D pose에 같은 새 값이 생긴다. 특히 공유 선형 scorer에서 공통 입력을 candidate−anchor 차분하면 그 공통 성분이 소거된다. **가설별 고정 pose의 투영 위치**를 사용해야 W/D와 expert 간 후보에 따른 시각 정보가 생길 수 있다. 서로 다른 pose가 같은 2D 투영 집합을 만들면 이 입력도 둘을 구분하지 못한다.

## 권고하는 한 가지 385차원 계약

한 이미지에는 기존 R0 bbox로 고정한 wide crop과 DINO feature map 하나를 공유한다. 후보마다 crop을 바꾸거나 GT·score·오차를 보고 crop을 선택하지 않는다. 원본 prepared image, crop affine matrix, resize·backbone·FP16 cache의 binding을 고정해야 한다. source의 기존 padding을 다시 추가하면 안 된다.

후보 pose의 camera-facing cuboid 코너8개만 투영한다. center9번째와 관측 q9는 시각 pooling에 넣지 않는다. support는 **유한 좌표, 양의 camera z, prepared image 내부, crop 내부**의 교집합이다. 이는 GT visibility나 가림 여부가 아니다. 픽셀 중심의 좌표 변환은 다음과 같다.

```text
uv_image = project(cf_corners8 @ R_cf.T + centroid, K)
uv_crop  = affine_matrix @ homogeneous(uv_image)
token_x = (crop_x + 0.5) * 42/576 - 0.5
token_y = (crop_y + 0.5) * 56/768 - 0.5

# 같은 위치를 grid_sample(align_corners=False)로 표현
grid_x = 2*(crop_x + 0.5)/576 - 1
grid_y = 2*(crop_y + 0.5)/768 - 1

descriptor_c = [mean_supported8(sample_FP32(FP16_DINO_map)), support_count/8]
visual_difference_c = normalize_TRAIN(descriptor_c) - normalize_TRAIN(descriptor_anchor)
x656_c = concatenate(existing_x271_c, visual_difference_c)
```

crop 내부이지만 token 중심의 convex hull 바깥인 위치는 **border bilinear**를 사용한다. crop/image 바깥을 border feature로 바꾸면 안 되며 먼저 support에서 제외한다. 음수 NumPy index의 wraparound, 음의 깊이 투영, 새 visibility filter를 허용하지 않는다. FP16→FP32 변환 뒤 별도 L2 정규화·PCA·채널 선택·학습된 projection은 하지 않는다. support가0이면 평균384와 support 비율을 모두0으로 두되 기존 기하 후보 valid는 유지한다. unavailable pose의 descriptor는0이고 기존 invalid mask·실패행은 그대로다.

평균은 **채널마다 supported FP32 샘플을 정렬한 뒤 같은 순서로 FP32 mean**을 계산한다. 수학적으로 같은 평균이며 성능을 고르는 하이퍼파라미터가 아니다. 동일한 코너 샘플 집합의 순열이 FP32 합산 순서 차이로 descriptor와 동률 결정을 바꾸지 않도록 하는 고정 수치 구현이다.

후속 학습을 별도로 허용할 경우 정규화는 기존 방식대로 R0의 eligible TRAIN 유효 후보5194개에서 FP32 mean/std와 floor1e−6으로 한 번 고정한다. 시각 support0 후보를 이 모집단에서 임의로 제거하지 않는다. 정규화 뒤 FP64 후보−anchor 차분을 수행하며 anchor는 정확0, 원래 invalid 후보·전체 실패행은 명시적으로0을 유지한다. 실제 지도값이나 VAL/실사 성과에 따라 이 정규화를 고르면 안 된다.

## C2와 식별 가능성의 한계

물리 Y축180° C2는 이 cuboid의 코너 순열이다. 동일 feature map, 좌표별 support, 순열 대칭적인 평균과 count를 사용하면 C2-equivalent pose가 같은 descriptor를 갖는다. 채널별 정렬 합산으로 **동일 샘플 집합의 임의 코너 순열은 byte 동일한 descriptor**를 만든다. identity 및 비항등 회전의 C2 합성 pose도 descriptor exact equality로 검사한다. 서로 독립적으로 계산한 pose나 projection 자체가 미세하게 다르면 그 차이까지 정렬이 제거한다고 주장하지 않는다. 평균은 모든 코너 순열에 불변이므로 C2를 지키는 대신 **코너 역할·모서리 연결·어떤 두 지점이 함께 어긋났는지**도 버린다. W/D에90° 대칭을 새로 허용하는 것은 아니다.

DINO token은 관측 RGB에 따른 고정 descriptor이며 cm/degree, GT visibility, 오차 부호 또는 보정된 신뢰도가 아니다. 가려진 코너가 배경을 샘플링할 수 있고, 작은 pose 차이는 같은 저해상도 token 구역 안에 남을 수 있다. 두 pose의 투영이 같거나 외관이 모호하면 이 추가 정보만으로 실제 T/R를 식별할 수 없다. 단일 RGB·치수·K의 물리적 모호성이나 기존 참조의 2D/K/치수 유도 한계도 사라지지 않는다.

새 채널을 선형 signed-axis 회귀에 단순 연결하면 시각 효과는 `w_visual · mean(tokens)`이다. 위치 샘플링 자체는 pose에 의존하지만 최종 회귀는 여전히 additive하며, geometry에 따라 채널을 선택하는 학습된 곱셈 상호작용·attention·코너별 신뢰도 집계가 생기는 것은 아니다. 새 loss가 강볼록하고 최적점까지 수렴하더라도 표현의 충분성이나 일반화가 보장되지는 않는다.

## 계산 비용: 작은 편이지만 무시할 수 없음

다음은 dense float64 배열의 이론적 크기이며 실측 실행시간이 아니다. UNION의 design tensor는2598×4×d로 계산했다. 두 축의 Newton block은 d×d이지만 [현재 trainer](../../../scripts/research/pallet_pose_signed_axes_asymmetric_20261001_v1/convex_train.py)는 full `(2d)×(2d)` Hessian을 만들고 마지막에 full `eigvalsh`를 호출한다.

| 구성 | 입력 d | 계수 수 | full Hessian MB | 두 block 합 MB | UNION X MB | d³ 비율 |
|---|---:|---:|---:|---:|---:|---:|
| 기존271 | 271 | 542 | 2.350112 | 1.175056 | 22.529856 | 1.000 |
| 평균384+support1 추가 | 656 | 1312 | 13.770752 | 6.885376 | 54.537216 | 14.184 |
| 비교용: 코너8×384를 전부 연결 | 3343 | 6686 | 357.620768 | 178.810384 | 277.923648 | 1877.160 |

표의 마지막 행은 실행 제안이 아니다. 385평균 descriptor는 그보다 훨씬 작고 압축 학습·차원 선택도 추가하지 않는다. 다만 Hessian 조립의 d² 항은 기존 대비약5.86배, dense solve/eigendecomposition의 d³ 항은약14.18배이므로 기존 호출 예산 안의 시간 증가를 예상해야 한다. 두 축 독립성을 이용한 추후 저장 최적화와 이번 입력 비교를 섞지 않는다. 실제 optimizer나 데이터에 대한 시간 probe는 하지 않았다.

한 `[384,56,42]` FP16 map은1.806336 MB, TRAIN2598장을 모두 저장하면4.692861 GB의 비압축 feature bytes다. 같은 이미지의 R0/D1/D2/D3와 두 pose는 map을 공유할 수 있다. 중복 map을 후보별로 저장하거나 한번에 전체를 RAM에 올릴 필요가 없다. 기존 cache가 같은 crop/image/backbone 조건을 모두 만족한다는 확인은 별도 cache coverage 감사가 맡으며, 오래된 cache 파일에 함께 든 target/GT 배열을 읽는 것은 이 입력 감사 범위가 아니다.

## 학습 전에 감사로 답할 수 있는 것

이번 단계는 descriptor·support·후보 차분의 유한성, 기존 valid/2598 IDs/anchor/pose 유지, 동일 입력 재현, C2/좌표계/경계 처리, 후보간 비상수성 및 기존271 열공간과의 입력 중복 여부를 확인할 수 있다. raw descriptor가 전부 같거나 후보−anchor 차분이 모두0이면 현재 선형 scorer에 추가될 정보가 없다는 명확한 반례다. 반대로 비중복성·높은 rank만으로 T/R 효과를 주장하지 않는다. support 밖 점의 숫자를 줄이기 위한 사후 crop 변경도 하지 않는다.

후속 실행 여부는 이 고정된 입력 계약의 적합성 감사 뒤 별도로 판단해야 한다. 본 문서는 새 학습·VAL gate·실사 route를 실행하거나 그 통과를 승인하지 않는다. 향후 기존 계수 뒤385개0을 붙인 모델은 수학적으로 기존 예측을 포함하지만, 실제 구현의 reduction 순서 차이는 합성 수치 검산으로 확인해야 한다. 참조 오차나 이전 weight를 이용해 입력을 선택하는 절차는 권하지 않는다.
