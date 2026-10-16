# 평가 solver 교체 — SQPnP+RefineLM → 미분가능 Gauss-Newton PnP (solver_swap_v1)

계열 코드: `solver_swap_v1`
착수: 2026-09-06

## 1. 제안

**가설.** 논문 pose 평가는 예측된 2D keypoint 에 `cv2.SOLVEPNP_SQPNP + RefineLM` 을 걸어
pose 를 읽는다 (`metric_split_lock.md` §3.1 LOCKED). 이 읽기 단계를 미분가능한 unrolled
Gauss-Newton PnP 로 바꾸면 같은 keypoint 에서 pose 수치가 달라지는가, 달라진다면 어느
방향인가.

**범위.** 학습 0 step, 새 추론 0 회. 캐시된 `predictions/{ARM}.json` 의 keypoint 를 그대로
쓰고 **pose read-out solver 만** 바꾼다. 축 가설 선택기(`select_pnp_hypotheses`)·GT·metric·
프레임 모집단은 전부 고정 — 유일 변수는 solver.

**solver arm (사전 고정).**
```
S0_SQPNP_LM       cv2.SOLVEPNP_SQPNP + solvePnPRefineLM      정본 baseline
D1_GN_LS          EPnP init + unrolled GN, 제곱오차          solver 전면 교체
D2_GN_HUBER       EPnP init + unrolled GN, Huber(12px)       robust 교체
D3_SQPNP_GN       SQPnP init + unrolled GN, 제곱오차         refiner 만 교체
D4_GN_HUBER_CONF  D2 + keypoint confidence 가중              가중 robust 교체
```
Huber delta = 12px 는 튜닝값이 아니라 `metric_split_lock` 이 이미 못박은 reject 임계
(median reproj > 12px) 를 그대로 가져온 것이다.

**판정 지표.** rotation median deg · translation median cm · IoU3D median ·
symmetry-aware ADD AUC. 짝지은 프레임(모든 solver arm 이 성공한 교집합)에서 비교.
게이트는 `SOLVER_SWAP_METHOD_LOCK.json` 에 결과 보기 전에 고정한다.

**예상 실패 모드.**
- 예측 2D 에 더 잘 맞출수록 pose 가 나빠진다 — predseed DiffPnP 스크린(2026-08)에서
  observed reproj -47% 인데 GT reproj +4.5%, rotation +13.8% 로 이미 관측된 방향.
- SQPnP 가 반환하는 pose 는 9 점 reprojection 의 최소점이 아니다. 순수 2D 적합으로 밀면
  translation 은 당겨지고 rotation 이 깨진다.

**중단 기준.** S0 이 기존 `POSE_EVALUATION_{ARM}.json` 의 ALL 을 정확히 재현하지 못하면
하네스가 무효이므로 전부 폐기한다. 재현되면 그때만 D1~D4 수치를 읽는다.

**누수 경고.** PAPER_EVAL 319 장은 반복 사용된 development set 이다. 어느 solver 가 좋아도
held-out / final 로 부르지 않는다.

## 2. 결과

`FINAL = SOLVER_SWAP_DOES_NOT_IMPROVE_POSE` · SQPnP LOCK 유지 · DEVELOPMENT_METHOD_SIGNAL.
산출물 `data/pallet/results/paper_pose_metric_closure_v1/solver_swap_v1/`.

### 한 문장

미분가능 GN 은 자기 목적함수(예측 2D 적합도)를 실제로 낮추지만, 그 대가로 회전이
나빠지고 총 pose 품질은 개선되지 않는다 — solver 는 레버가 아니다.

### 게이트

```
게이트                                    판정      근거
─────────────────────────────────────────────────────────────────────────────────
0 하네스 유효 (S0 이 정본 재현)           PASS      7개 arm × 7개 키 전부 1e-9 이내 일치
1 solver 정확성 (무잡음 pose 회복)        PASS      회전 Frobenius 1e-16, t 1e-15 m
                                                    Jacobian = 수치미분 (atol 1e-4)
                                                    무잡음에서 SQPnP+LM 과 동일해
                                                    이상점 1개면 Huber 가 실제로 이김
                                                    2D 입력으로 grad 가 흐름
```

### solver arm (R0, 짝지은 319 프레임)

| solver | rot med° | trans med cm | IoU3D | ADD AUC | 판정 |
|---|---:|---:|---:|---:|---|
| S0_SQPNP_LM (정본) | 2.262 | 7.897 | 0.6032 | 0.4285 | baseline |
| D1_GN_LS | 2.296 | 7.897 | 0.6000 | 0.4272 | REJECT |
| D2_GN_HUBER | 2.296 | 7.897 | 0.6000 | 0.4281 | REJECT |
| D3_SQPNP_GN | 2.262 | 7.897 | 0.6032 | 0.4285 | 변화 없음 |
| D4_GN_HUBER_CONF | 2.296 | 7.897 | 0.6000 | 0.4279 | REJECT |

### 읽는 법 세 가지

1. **D3 의 ACCEPT 는 실체가 없다.** 최대 상대변화 3.67e-08 — SQPnP 로 초기화한 GN 은
   SQPnP+LM 의 답으로 되돌아간다. 사전등록 규칙이 개선을 부등호로만 정의해서 1e-9 차이를
   개선으로 셌다. 규칙은 고치지 않고 산출물에 `indistinguishable_from_baseline` 을 덧붙여
   구분되게 했다. 앞으로 이런 게이트에는 **의미 있는 차이의 하한**을 같이 등록할 것.
2. **전면 교체(D1)는 회전을 1.48% 악화**시키고 IoU3D −0.53%, ADD AUC −0.30%. translation
   중앙값은 그대로. 즉 solver 를 바꿔서 얻는 것이 없다.
3. **robust/가중 arm(D2·D4)은 같은 맞교환을 재현**한다. translation 은 arm 에 따라 개선
   (R5 8.827→8.690cm, −1.54%)되지만 회전과 IoU3D 는 나빠진다. R2_CONF 에서도
   trans 7.785→7.532 / rot 2.477→2.579 로 방향이 같다.

### 기존 결론과의 정합

2026-08 predseed DiffPnP 스크린(DOPE ep57, 70 프레임)의 방향을 **다른 모델 계열(YOLO26n)과
다른 모집단(319 프레임)에서 재현**했다: 예측 2D 적합도를 낮추면 translation 은 당겨지고
rotation 이 깨진다. 그때 [추정]으로 남겼던 이유 — canonical solve_pose 가 순수 reprojection
이 아니라 degeneracy guard 를 포함한 복합 점수로 고르기 때문에 순수 2D 적합이 보호장치를
깬다 — 와 부합한다. 단 이번 경로는 selector 를 고정했으므로 그 기전을 직접 검증한 것은
아니다.

### 남은 범위

축 가설 선택기(`select_pnp_hypotheses`)는 내부적으로 여전히 SQPnP 다. 완전한 교체는
선택기까지 바꾸는 것이지만, 그러면 axis_accuracy 가 같이 흔들려 solver 효과와 교란된다.
이번 실험은 read-out 만 바꾼 깨끗한 비교로 한정했다.
