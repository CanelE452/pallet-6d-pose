# 예측 점과 영상 선 분포의 학습 중 연결

2026-09-08. 구현은 `incidence.py`, CPU 검증은 `test_incidence.py`에 있습니다. 기존 공동학습 모델과 모든 v1 파일은 수정하지 않았습니다. 이 문서는 손실의 기하 계약이며 성능 개선 결과가 아닙니다.

## 고정된 범위

공동학습 대조군과 동일한 전역 DHT→Hough 처리→역투표→YOLO 특징 경로, 동일한 원래 YOLO 손실, 동일한 선 감독 계수 0.1을 유지합니다. 추가하는 것은 **실제 예측 점과 영상에서 예측한 선 분포 사이의 incidence 손실 하나**입니다. Gradient surgery와 손실 가중 균형 arm에는 이 손실을 추가하지 않습니다. Incidence 계수 0.1은 초기 제안이며 본 학습 전에 합성 데이터만으로 규모를 확인하고 루트 프로토콜에 확정해야 합니다.

추론 경로는 변경하지 않습니다. GT crop, 정답으로 만든 예측선, 점에서 다시 만든 예측선, hard argmax 선 선택, 코너 후처리 또는 PnP 변경은 없습니다. 이 손실은 GT를 받는 학습 criterion에서만 실행됩니다.

이번 세션에서 확정된 `camera_dynamic_0123_v4`와 기존 12개 semantic role을 그대로 사용합니다. 별도로 읽은 일반 3D 스킬의 object-frame/order-free 지침으로 정본 라벨을 재정의하지 않습니다. 코너 0–7만 선과 연결하며 centroid 8에는 incidence 경사가 없습니다.

## 실제 예측과 매칭

설치된 `ultralytics/utils/loss.py`의 `PoseLoss26`를 직접 확인했습니다. 원래 Task-Aligned Assigner가 반환한 `fg_mask`, `target_gt_idx`, `anchor_points`, `stride_tensor`를 **동일 loss 호출에서 캐시**합니다. 재매칭 또는 정답 오차에 따른 positive 선택을 추가하지 않습니다.

실제 dense 예측 좌표는 다음과 같습니다.

```text
pred_points_px = (raw_kpts_xy + anchor_points) * stride_tensor
```

현재 PoseLoss26 decoder에는 이전 버전의 `*2` 또는 `-0.5`가 없습니다. `decode_points_px()`는 원래 decoder와 CPU에서 bit-exact로 일치했습니다. 입력 픽셀 좌표는 원본 사진 좌표가 아니라 증강과 YOLO 입력 변환을 거친 network input 좌표입니다. P4 좌표로 바꿀 때는 `p_feature = p_input / stride_P4 - [W/2,H/2]`를 사용합니다. 이는 고정 Hough 셀 좌표 계약이며 실제 convolution receptive-field 중심을 주장하지 않습니다.

`incidence_loss()`는 one2many 또는 one2one 한 branch를 받습니다. 상위 criterion이 두 결과를 원래 E2ELoss의 `o2m/o2o` 값으로 합칩니다. 추가 계수와 batch-size 곱은 각각 상위에서 한 번만 적용합니다. 원래 one2one 특징 detach는 유지되며, 추가 incidence의 line 쪽 경사는 영상의 Hough branch로 전달됩니다.

## 전체 선 분포에 대한 동일 선 일치

각 역할의 physical-footprint 유효 bin 전체에 대해 영상 line logit `l_k`를 다음 상대 질량으로 정규화합니다.

```text
p_k = sigmoid(l_k) / sum_j sigmoid(l_j)
h_k = (cos(theta_k), sin(theta_k), -rho_k)
sigma_input = max(1 pixel, 0.01 * hypot(input_width, input_height))
sigma_feature = sigma_input / stride_P4
```

`p_k`는 독립 sigmoid 점수를 정규화한 상대 질량입니다. 보정된 정답 확률 또는 물리 가시성 확률이라고 해석하지 않습니다. 확률 합은 `logsigmoid`와 `logsumexp`로 안정적으로 계산합니다.

한 semantic edge에 연결된 **실제 예측 끝점 두 개**를 `q1,q2`라 할 때:

```text
r_ek = (normal_k dot q_e - rho_k) / sigma_feature
c_k = 0.5 * sum_e [sqrt(1 + r_ek^2) - 1]       # pseudo-Huber, delta=1
L_pair = -log sum_k p_k * exp(-c_k)
```

두 끝점의 cost를 먼저 더하고 그다음 후보 선에 대해 marginalize합니다. 따라서 한 끝점은 왼쪽 선, 다른 끝점은 오른쪽 선을 골라 둘 다 일치한 것처럼 만드는 계산이 아닙니다. 분포의 평균선을 만들지 않으므로 두 실제 mode 사이의 존재하지 않는 선을 정답처럼 취급하지 않습니다. `(normal,rho)→(-normal,-rho)`는 residual 부호만 바꾸고 cost는 동일하므로 θ antipodal seam에서 정의가 보존됩니다.

두 predicted endpoints가 겹쳐도 loss를 끄지 않습니다. 예측 길이·신뢰도·정답 오차로 손실을 회피하는 gate는 없습니다. 상대적으로 큰 오차에서는 pseudo-Huber가 선형으로 증가하여 제곱 오차의 과도한 경사를 줄입니다. 그래도 한 무한선은 접선 방향 위치나 선분 길이를 결정하지 못하며, 점 감독을 대체하지 않습니다.

각 이미지에서 역할별 모든 원래 foreground anchor를 평균하고, 유효 역할을 평균한 뒤, 전체 batch 이미지를 평균합니다. 빈 이미지 또는 전부 무감독인 이미지는 0으로 batch 분모에 남습니다. 추가 선이나 쉬운 positive만 선택하지 않습니다.

## GT가 허용되는 위치와 가시성 한계

GT는 기존 object identity와 line 지원 여부에만 사용합니다. 기존 `build_line_targets()`의 mask/instance usability를 재사용하지만 **GT target heatmap 값, GT θ/ρ 또는 GT 좌표 거리를 incidence 값과 분포 가중치에 넣지 않습니다.** 동일한 지원 상태에서 GT 위치만 옮겨도 이 손실은 bit-exact로 변하지 않는 CPU 테스트를 포함했습니다.

두 GT 끝점은 유한하고 `v>0`이어야 하며, GT 선분이 실제 network input footprint와 양의 길이로 교차하고 기존 lattice로 표현 가능해야 합니다. 물리적으로 가려졌더라도 감독되는 구조 좌표는 포함합니다. Padding은 기존 입력의 일부이며 원본 내용 영역을 GT로 찾아 자르는 mask는 없습니다.

동결된 `pallet_line_pose_v1/SOURCE_MANIFEST.json`의 60,000장 모두 annotation object 수가 정확히 하나입니다. 총 540,000개 keypoint의 v는 2가 535,066개, 0이 4,934개이며 v=1 구분은 없습니다. 학습 target 필드도 box, class, keypoints뿐입니다. 원 renderer 예제의 `front_visibility_cos`/`facing_margin`은 per-edge 물리적 가시성 정답이 아니며 현재 학습 입력에 전달되지 않습니다. 따라서 선은 **amodal structural role**이고 `v>0`를 실제 보이는 edge로 재명명하지 않습니다.

전역 role map에는 instance 축이 없습니다. 현재 자료에 맞춰 이미지당 annotation 0개 또는 1개만 허용하고, 2개 이상이면 명시적으로 실패합니다. 여러 객체의 선을 정답으로 나누거나 가까운 line mode를 임의로 선택하는 확장 기능은 구현하지 않았습니다. 이는 추론 모델의 다중 검출 능력을 제한하는 규칙이 아니라 이번 추가 학습 손실의 데이터 범위입니다.

## CPU 검증과 해석 범위

다음 명령으로 17개 검사를 통과했습니다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m pytest -q scripts/research/pallet_dht_coupling_v2/test_incidence.py
```

검사에는 실제 stock decoder 일치, 하나의 잘못된 끝점, 끝점 순서 반전, antipodal seam, 다중 mode의 동일 선 조건, 평균선 허상 방지, 이동/스케일 불변성, 양쪽 예측으로의 finite gradient, FP64 gradcheck, centroid 제외, GT 위치 누출 부재, 부분/전체 무감독·빈 이미지·잘린 선, 전체 batch 분모, 예측 끝점 붕괴, 다중 객체·잘못된 assignment·비유한 예측 실패가 포함됩니다.

이 검증은 정의와 미분이 맞음을 확인합니다. 점과 선이 함께 잘못된 해석에 맞춰질 가능성, 대칭 role의 모호함, 실제로 보이지 않는 구조의 감독 부담, 낮은 Hough 해상도, 원래 point/line GT 손실과의 충돌은 남아 있습니다. 기존 두 GT 손실을 유지하고 별도 합성 규모 검사와 동일 예산 실험으로 효과를 판단해야 합니다. 실사 결과로 계수·mask·선택 규칙을 바꾸지 않습니다.
