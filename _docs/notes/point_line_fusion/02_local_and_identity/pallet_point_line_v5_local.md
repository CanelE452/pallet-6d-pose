# 치수·대칭 일관 점·선 공동 자세 — 로컬 재구현 (pallet_point_line_v5_local)

> 이것은 외부 v5 번들의 재현이 **아니다**. 저자 코어 `plpose_v5/` 가 전달되지 않아
> (`data/pallet/results/pallet_point_line_v5/BLOCKED_MISSING_BUNDLE.json`) 같은 질문에
> 답하기 위한 독립 구현을 만들어 돌린다. 범위·한계는
> `data/pallet/results/pallet_point_line_v5_local/SCOPE_DISCLAIMER.md`.

## 1. 제안

- 가설: 치수와 유효 대칭을 준 공동 point-line pose 추정에서, 선 증거를 추가하면
  (D = direct readout, H = DHT readout) 점만 쓰는 P 보다 pose 가 좋아진다.
- 방법: stock R0(`970a0913…`, YOLO26n-pose, `Pose26` 입력 P3/P4 훅)의 고정 feature 에서
  24x24 ROI 를 뽑고, 새 object frame(+x=W, +y=up, +z=D, 원점=기하중심)에서 point head +
  line head(P 없음 / D / H) + 다중 시작점 + 언롤 LM 공동 solver 를 학습한다.
  세 군은 backbone·ROI·K·dims·GT·초기값·batch 계획·예산이 동일하고 line 경로만 다르다.
- 판정 지표: 유효 대칭 하나를 프레임 전체에 적용한 8코너 평균오차 / raw diagonal,
  point 당 1 로 cap, pose 실패는 1. 조건부 raw px median/P90 과 실패 coverage 별도 보고.
- 진입 기준(외부 지시문 §8 그대로): 세 seed 각각 H 가 P 대비 primary 1% 이상 개선 +
  median/P90·coverage 비악화 + 원래 프레임 평균 ≤10px 가 >50px 로 바뀐 새 catastrophic
  frame 0. 이 값들은 미검증 예비 기준이지 안전 인증이 아니다.
- 예상 실패 모드: (i) object frame 회전 A 미확정 → 전 단계 무의미, (ii) 다중 시작점이
  같은 basin 으로 collapse, (iii) 선 격자 해상도가 정밀도 천장을 만듦, (iv) 합성에서만 개선.
- 중단 기준: 상류 게이트(A 검증, GT 재투영) 실패 시 학습 0. 예산 상한 3군×3seed×2,000 =
  18,000 optimizer steps. 성능 실패 후 다른 설정으로 반복하지 않는다.
- 대칭: 로컬 계약상 **C4 표본 0** 이므로 C1/C2 만 실행된다.

## 2. 결과

(진행 중)

### 게이트 1 — 생성 fixture (통과)

`tests/test_geometry_solver.py` **10/10 PASS**. 투영 Jacobian 6자유도 전부 유한차분과 일치,
회전 미분이 `−skew(RX)` 이고 `−skew(RX+t)` 와 구분됨, C2 는 임의 footprint 에서 코너를
순열하고 C4 는 정사각에서만 성립, reframe 이 반사·스케일을 거부, 점 전용/점+선 모두
섭동된 시작점에서 자세 복구, 한 edge 의 두 끝점이 **하나의 mode 책임을 공유**(decoy mode 가
공짜 에너지를 못 만듦), 선택이 에너지만 쓰고 카메라 뒤 후보를 배제.

개발 중 잡은 결함 1건: 혼합 질량을 정규화하지 않아 mode 를 추가할수록 에너지가 내려갔다
(decoy 추가 시 −0.35). 허용오차를 키우는 대신 edge 별 질량을 log-softmax 로 정규화해
고쳤고, 그 뒤 decoy 는 에너지를 **올린다**.

### 게이트 2 — 실제 라벨의 object frame 매핑 (통과)

`OBJECT_FRAME_A.json` (3d-expert 위임, 산출 검증은 내가 수행).

- **A = identity, det=+1.** 렌더러 고정 축이 이미 목표 계약(+x=W, +y=up, +z=D)과 같다.
  치수 입력 순서는 `(fixed_renderer_dimensions_m_xyz[0], [2], [1])` = (W, D, H).
  중심이 같으므로 translation 불변, `R_new = R_old @ A`.
- **재투영 max 0.00040620605 px / median 7.8e-6 px**, 1,200 프레임(4 asset 각 300장), 전부
  0.1px 기준 통과. asset 별 max 는 2.4e-4 ~ 4.1e-4 px. cuboid extent 대 고정 치수 최대차 9.5e-7 m.
  지시문이 요구한 32프레임보다 큰 표본으로 확인했다.
- 통과 후보가 4개(identity, Rx180, Ry180, Rz180)인 것은 **상자 코너 집합이 각 축 180° 회전에
  불변**이기 때문이다. `+y_up` 1200/1200 로 y 축을 확정해 Rx180·Rz180 을 배제했고, 남은
  identity vs Ry180 은 정확히 **C2 모호성**이다. 렌더러 자체 고정 프레임을 일관되게 쓰므로
  자기정합적이며, C2 asset 에서는 대칭 손실이 이를 동치로 처리한다. C1 asset 에서는
  **임의이지만 고정된 대표 선택**임을 명시한다.

**부수 발견 — 저장 코너 순서는 프레임마다 돈다** [확인]. 저장된 `projected_cuboid` 순서를
내 SIGNS 에 맞추면 1,200 프레임이 네 순열에 거의 균등하게 흩어진다:
`[0,1,2,3,4,5,6,7]` 269 / `[1,5,6,2,0,4,7,3]` 316 / `[5,4,7,6,1,0,3,2]` 308 /
`[4,0,3,7,5,1,2,6]` 306 (+ 정체불명 1건 `[0,4,7,3,1,5,6,2]`).
이 넷은 `pallet_dht_structured_v2/proposals.py` 의 **C4 군과 정확히 같다**. 즉
`camera_dynamic_0123_v4` 가 매 프레임 앞면을 다시 고른다는 `SEMANTICS.md` 기술이 수치로
확인된다. 따라서 **매핑을 저장 순서로 하면 안 되고**, pose GT 는 `pose_transform` 을 쓰고
2D point GT 는 고정 프레임 코너를 직접 투영해 만든다(지시문 §4.1 이 허용하는 경로).

이는 v4 의 "C4 후보" 가 물리 대칭이 아니라 **카메라 기준 앞면 재선택의 산물**임을 다시
보여준다 — 물리 대칭은 로컬 계약상 C2 까지만 확인돼 있다.
