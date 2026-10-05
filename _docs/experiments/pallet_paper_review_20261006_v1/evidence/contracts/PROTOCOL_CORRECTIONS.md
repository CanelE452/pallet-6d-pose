# ResNet 명세 정정

원본 protocol/receipt와 가중치는 수정하지 않았다. 실제 사용 모델은 10-epoch CONSTANT-fold RGB이다.

- `checkpoint_selection`: 과거 `fixed final epoch60; no real-based threshold selection` → 실제 `fixed final epoch10; step34990; synthetic calibration only`.
- `decoder`: 과거 `channel argmax; if both coordinates are interior 1..size-2, add .25*sign(right-left,down-up)` → 실제 `Spatial softmax expectation for all nine keypoints; single-pallet source contract.`.
- `mse`: 과거 `mean_frames(sum9(mask * 0.5 * mean_pixels((prediction-target)^2)) / 9)` → 실제 `0.1*normalized Gaussian spatial cross entropy + smooth-L1 spatial-expectation coordinate loss.`.
- `confidence_threshold`: 과거 `0.1` → 실제 `not used in DSNT decoder; nine finite expectations; validity all nine`.

CONSTANT는 치수 조건 z=0을 뜻한다. 고정 FiLM을 마지막 1×1 convolution에 접어 RGB만 받는다. N3는 이미지 특징과 등록 W,D,H를 입력받으며 각 seed 6,000 update 완료 기록을 재사용한다. 기본 추정기까지 같은 총학습량을 통제한 백본 인과 비교가 아니다. 실제 checkpoint 헤더 epoch10/step34990과 strict folded load를 재확인했다. 상세 원본 해시는 RESNET_EFFECTIVE_PROTOCOL.json에 있다.
