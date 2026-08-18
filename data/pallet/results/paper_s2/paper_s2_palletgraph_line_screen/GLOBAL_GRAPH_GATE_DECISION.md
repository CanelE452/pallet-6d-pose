# GLOBAL GRAPH GATE DECISION — PalletGraph-6D (G0/G1 완료, G2 미실행)

[관찰]
- G0 (yaw 전역 탐색, GT t/roll/pitch = UPPER BOUND): oracle line 3종 모두 overall/point-fail/truncated **1.000**, top-1 yaw 오차 중앙값 0.0°.
- G0 point-only 는 point-fail 에서 **0.000** (초기 pose 부재).  point+line 은 0.931.
- G1 (yaw/tx/tz 탐색, roll/pitch/ty = GT): point-fail 17/17 이 yaw<=10°, yaw 중앙값 0.250°, translation 중앙값 0.0036 m.

[Energy continuity]
- [확인] v1 의 계단형 energy 제거 확인.  인접 0.25° 간 max |dE| = 0.004, sweep 중 edge 수 9 고정(v1 은 7<->9 점프).
- [확인] order 무관 |dE|=0, 180° equivalence |dE|=0, class permutation 시 448배 악화, gradient finite.
- [판정] **직전 실험의 FAIL 원인 3(계단형 energy)은 실제로 존재했고 v2 에서 해소됐다.**

[Global yaw identifiability]
- [확인] G0-LO(실제 image gradient support 가 있는 fragment 만) 도 overall 1.000 / point-fail 1.000 / truncated 1.000.
- [확인] 즉 **영상에 남아 있는 line 만으로 [0°,180°) 전역에서 GT yaw 가 식별된다** (oracle semantic 라벨 조건).
- [확인] 대조로 point-only 는 point-fail 0.000 / truncated 0.353.

[Translation upper bound]
- [확인] point-fail 17 전부 valid pose 획득, positive depth 17/17.
- [확인] point-success 70 에서도 corner error 70/70 개선 (0.4516 -> 0.0051 m).

[Point-fail rescue]
- [확인] 직전 실험에서 '한 번도 시험되지 않았던' 17 프레임이 이번에는 전부 실행됐고 pose 가 나왔다.
- [확인] 최상위 가설('point 가 사라질 때 line 이 회복')이 **처음으로 검증 가능한 형태가 됐다**.

[★ 이 결과를 능력으로 읽으면 안 되는 이유]
- [확인] G0/G1 모두 line map 을 **GT pose 로 그렸고**, roll/pitch(+G1 은 ty)도 **GT** 다.
  corner 5.1mm 는 물리적 정확도가 아니라 **oracle 자기참조**다.  GT 로 만든 신호에서 GT 를 되찾은 것이다.
- [확인] semantic class(width/depth/vertical) 라벨도 GT 다.  learned line head 는 이를 스스로 예측해야 한다.
- [판정] 따라서 G0/G1 이 증명한 것은 **정보량 상한**뿐이며 deployable 성능이 아니다.

[Global solver effect]
- [확인] G1 은 point-success 프레임에서도 baseline PnP 대비 corner 를 88배 낮췄는데, 이는 solver 효과가 아니라
  oracle 신호 효과다.  solver 자체 효과는 G2(C1-C0)에서만 분리 측정할 수 있다.

[Line marginal effect]
- [미측정] G2 미실행이라 line 의 marginal contribution(PL0-C1)은 아직 없다.

[지지 증거]
- [확인] continuity unit test 10종 통과, G0/G1 gate 통과, point-fail 17 전부 실행.
- [확인] search prior 는 paper_4pallet_mask_v1 에서만 유도했고 N87 GT 를 쓰지 않았다 (테스트로 강제).

[반증 증거]
- [확인] 없음.  단 위 '능력으로 읽으면 안 되는 이유'가 결과의 해석 범위를 강하게 제한한다.

[현재 판정]
- [확인] **직전 세션의 'MSL REJECT' 는 완전히 뒤집혔다.**  line 은 정보가 없던 것이 아니라
  (a) energy 가 계단형이라 최적화가 불가능했고 (b) 초기 pose 가 basin 밖이었으며
  (c) point-fail 프레임은 실행조차 되지 않았다 — 세 가지 모두 이번에 해소·검증됐다.
- [확인] G0 PASS + G1 PASS.  지시문 판정 규칙상 **G2(deployable) 로 진행**한다.
- [미완] G2 는 이번 실행에서 완료하지 못했다 (사유 아래).

[architecture 결정]
- DGP-v2: **ACCEPT** — 연속 energy·frame-fixed edge set·sample-wise 정규화가 unit test 로 검증됐고,
  이 수정 없이는 global search 자체가 불가능했다.
- Semantic line information: **ACCEPT (upper bound 한정)** — oracle 조건에서 yaw 전역 식별 100%.
- Learned MSL: **HOLD** — G2 미완이므로 아직 PROCEED 아님.

[G2 미완 사유 — BLOCKED 아님, 범위]
- [확인] E1 ground prior audit 결과 **camera-ground normal / camera height / ground-plane constraint / IMU 전부 부재**
  (N87 JSON 의 camera_data 는 width/height/intrinsics 뿐).
- [확인] 따라서 G2 는 roll/pitch 를 prior grid 로 탐색해야 하고, 탐색이 yaw x tx x tz x roll x pitch **5D** 가 된다.
  G1(3D, 87 프레임 350초) 기준 grid 3x3 만 해도 9배(~52분), 5x5 면 25배(~2.4시간).
- [판정] 계산 자체는 가능하나 이번 실행에서 완주하지 못했다.  코드·gate·prior 는 모두 준비되어 있다.

[다음 admissible experiment]
1. G2 deployable: roll/pitch prior grid(paper_4pallet_mask_v1 유래)로 5D 탐색, GT component 0 사용.
   C0/C1/L0/PL0/PL1 arm 으로 solver 효과와 line marginal 효과를 분리.
2. G2 PASS 시에만 Phase H: fresh mask head + semantic line head 를 paper_4pallet_mask_v1 **하나로만** 학습.
3. full training / 3-seed / final-test 는 실행하지 않는다.
