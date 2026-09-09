# 소스 확인과 설계의 범위

기준 repository: `CanelE452/pallet-6d-pose`, commit `6a68452ea9a17a69860981fea1c5a3dc234e3dfa` (2026-09-09 01:46:51 UTC / 10:46:51 KST).

## 실제 읽은 핵심 구현

1. `scripts/research/pallet_dht_structured_v2/model.py`, Git blob `23cffddb7f8f79f3b92539f1e4a4b01291d99179`.
   - [확인] corner patch/edge interior readout과 semantic tokens가 이미 있다.
   - [확인] same-ID baseline position, displacement, confidence 및 ordered baseline context를 사용한다.
   - [추정] 번호 오류에 대한 original-ID shortcut 가능성은 아직 원인으로 입증되지 않았다.
2. `scripts/research/pallet_dht_structured_v2/data_ops.py`, blob `6979f448b82f9e64869bc753c120fb66d8ad6ad7`.
   - [확인] 후보별 정답 오차, listwise target, 직접 비용 회귀 및 코너별 회귀가 이미 있다.
   - [확인] clean/point_c4/point_and_line_c4/local_deformation의 학습 상태 혼합을 구현했다.
   - 이번 native-only 작은 비교는 그 인위적 학습 교란을 제거한 새 실험이다. 따라서 과거 v2와의 비교는 단일 요인 비교가 아니다.
3. `scripts/research/pallet_dht_structured_v2/infer.py`, blob `490d0ef6ec4955785688b2c48f6e460a39429cd6`.
   - [확인] 실제 raw inference의 공통 backbone SHA, content affine, 후보 생성·scoring·margin 경로를 확인했다.
4. `scripts/research/pallet_dht_structured_v2/SEMANTICS.md`, blob `5d2b3ca2ee96e9c1f364433bd57bd63b01958a85`.
   - [확인: 문서] camera-facing 번호/물리 대칭/구조선과 영상 가시성의 구분, DLT만으로 번호를 판별할 수 없는 한계를 기록한다.
5. `scripts/research/pallet_dht_structured_v2/train.py`, blob `a13715ca551e80a832bbb4b66385e6fba0818aa8`.
   - [확인] source freeze, 실제 step, seed/batch 순서 기록과 고정 마지막 checkpoint를 확인했다.
6. `scripts/research/pallet_dht_joint_v1/hough_block.py` 및 `scripts/research/deep_hough_side_v1/dht.py`.
   - [확인] 최신 모델은 실제 feature voting/backprojection 경로다. 기존 Direct-Hough의 전역 descriptor scoring과 구별한다.

원문 주소의 기본형은 다음과 같다.
`https://github.com/CanelE452/pallet-6d-pose/blob/6a68452ea9a17a69860981fea1c5a3dc234e3dfa/<repository-path>`

## 선행연구

**Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning(2023/IEEE TPAMI).** 끝점과 선분의 관측·제안·검증을 구분하는 설계의 근거다. 이 패키지는 HAT field/HAWP 학습의 재현이 아니고, 물리 wireframe GT를 사용한다고 주장하지 않는다. 공식 구현은 TPAMI 게재를 명시한다. 논문의 arXiv HTML은 열람본이며 arXiv-only 논문이 아니다.

- 저자 공식 구현: https://github.com/cherubicXN/hawp
- 열람한 본문: https://arxiv.org/html/2210.12971v2
- DOI: 10.1109/TPAMI.2023.3312749

**Deep Hough-Transform Line Priors(2020/ECCV).** Hough/역투표를 통한 특징 결합의 선행 근거다. 이미 해당 계보의 공동학습이 저장소에 있으므로 “처음 DHT를 연결한다”는 실험으로 되돌리지 않는다.

- 저자 공식 구현: https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors

위 선행연구는 제안의 원리적 근거일 뿐 이 팔레트 실험의 성공률·1% gate·2,000-step 예산을 뒷받침하지 않는다. 그 값은 새 제한 예산의 [미검증 제안]이다.

## 제공 코드의 설계 차이

[구현] native P3/P4를 각각 읽고 원래 점 집합에 대해 순열 불변인 distance-weighted reference를 만든다. HA는 동일한 입력 차원에서 이를 same-ID reference로 바꾼다. 후보 자체는 기존 bank를 공통 재사용한다. 직접 품질 감독은 기존 아이디어이며 새 기여로 주장하지 않는다.

[한계] P3 추가, anchor 표현, native-only 훈련 등 과거 v2와 달라진 항목이 여럿이다. 인과적으로 해석 가능한 비교는 이번 새 네 군 안에서 고정한 대비다. native P3/P4 사용 자체의 효과를 과거와 단순 차감해서 주장하지 않는다.
