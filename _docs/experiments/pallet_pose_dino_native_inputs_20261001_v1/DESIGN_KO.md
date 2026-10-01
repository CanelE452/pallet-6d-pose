# 원본 이미지 영역만 지원하는 고정 DINO 입력

이 단계는 기존 TRAIN token의 재풀링만 수행합니다. 새 backbone 추론, 학습, 목표 오류 조회, VAL·실사 평가, 새 pose 추정은 모두 0회입니다. 생성 완료와 독립 검산 완료를 구분하며, 이후 별도 독립 검산 전에는 학습 준비가 검증되었다고 판정하지 않습니다.

직전 입력 감사에서 준비된 source 이미지의 반사 패딩 영역에 지원 투영점이 놓이는 사례를 확인했습니다. 이번 변경은 기존 메타데이터의 `pad`를 이용해 지원 영역을 `[pad, width-pad) × [pad, height-pad)`로 제한하는 한 가지입니다. 기존 양의 깊이·유한 좌표·준비된 이미지 범위·crop 범위 조건과 교집합을 취합니다. 지원점을 가림 여부의 정답으로 해석하지 않습니다.

기존 2,598개 TRAIN 행과 원래 실패 1행, 모델별 5,194개의 유효 후보를 그대로 유지합니다. `pad`는 각 행에 명시된 정수 값을 사용하며 누락 시 임의 값을 대입하지 않습니다. 원래 유효하지만 원본 영역 지원점이 없는 후보도 삭제하지 않고 385차원 0 벡터로 유지합니다. 기존 이미지 좌표·K·고정 pose·R0 crop 행렬·FP16 DINO token은 바꾸지 않습니다.

기존 8개 투영 코너에서 같은 half-pixel bilinear border sampler로 원시 384채널을 읽고, 지원점만 채널별 정렬한 뒤 FP32 평균을 계산합니다. 마지막 값은 지원점 수/8입니다. 센터 점, L2 정규화, 새 채널 선택, 새로운 시각 단서는 추가하지 않습니다. `pad=0`이면 이전 descriptor와 정확히 동일해야 합니다.

새 정규화는 R0의 기존 유효 TRAIN 후보 5,194개 전체에서 FP32 mean/std를 계산하고 std 하한을 1e-6으로 둡니다. 0지원 후보도 원래 valid이면 정규화 분모에 포함합니다. 이후 같은 656차원 입력을 구성할 때 사용할 385차원 원시 특징과 mean/std schema를 유지하며, 현재 단계에서는 모델이나 점수 함수를 만들지 않습니다.

`TRAIN_APPEARANCE.npz`는 이전 18개 키를 유지합니다. `ids`, `source_index`, `anchor_index`, `crop_matrices`, `mean385`, `std385`와 모델별 `_appearance385`, `_valid`, `_support8`입니다. 별도 `NATIVE_SUPPORT_DETAILS.npz`에는 행별 `padding_px`, 원본 영역 좌표 및 이전/현재/제거된 지원점 mask를 저장합니다. 원본 `TRAIN_TOKENS.npy`는 복사하지 않고 이전 파일의 SHA binding으로 재사용합니다. 봉인 프로토콜과 영수증은 이전 프로토콜·descriptor·token·독립 검산·패딩 감사의 정확한 binding을 포함합니다.

지원 위치를 원본 영역으로 제한해도 DINO token 자체의 문맥은 이전 crop과 반사 패딩을 보았습니다. 따라서 이 변경은 패딩 영향 전체를 제거하거나 실제 T/R 식별 가능성을 입증하지 않습니다. 입력만 바꾸는 단계이며 성능 성공을 주장하지 않습니다.

실행 전 합성 검사는 `pad=0`의 이전 입력과 exact equality, `pad=100`의 패딩 제외, 반열린 경계 조건, 0지원 유효 후보 보존, C2 동일 샘플의 exact equality를 포함합니다. 실제 재풀링은 프로토콜 봉인 후 한 번 수행하며, 별도 담당자의 독립 검산을 다음 단계로 남깁니다.

[이전 입력 감사](../pallet_pose_dino_input_audit_20261001_v1/INPUT_VERIFICATION_KO.md) · [패딩 지원점 감사](../pallet_pose_dino_input_audit_20261001_v1/PADDING_SUPPORT_AUDIT.json) · [새 pure 연산](../../../scripts/research/pallet_pose_dino_native_inputs_20261001_v1/visual_features.py) · [봉인·재풀링 코드](../../../scripts/research/pallet_pose_dino_native_inputs_20261001_v1/freeze_train.py)
