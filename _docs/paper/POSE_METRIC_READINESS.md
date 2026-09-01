# POSE_METRIC_READINESS

작성 2026-09-01.  `POSE_METRICS_STATUS = **BLOCKED**`.

논문의 6D pose 열(`R med`, `Yaw`, `t`, `IoU3D`, `AUC`)을 **채우지 않는다**.
2D 성능이 좋아져도 "6D pose improved" 라고 쓰지 않는다.

self-training 트랙 자체는 2D pseudo-keypoint 기반이므로 이 blocker 와 무관하게
진행됐다 — M2/M3/M4/M5 는 이미 채워져 있다.

## 상태 원천

이 판정은 추정이 아니라 evaluator 실행 결과다.

```text
artifact   data/pallet/results/paper_eval_v1/EVALUATOR_CONTRACT.json
           data/pallet/results/paper_eval_v1/arms/*.json  (metrics.pose.status)
runner     challenge/evaluation_v2/paper_real_eval.py
```

전 모델(R0 · R0-CONT · R1~R5 · replicate)의 `metrics.pose.status` 가 `BLOCKED` 다.

## blocker 를 축별로 분리

하나로 뭉뚱그리면 무엇을 고쳐야 하는지 알 수 없으므로 나눠 적는다.

```text
축                          상태      근거
──────────────────────────────────────────────────────────────────────────────
Plastic axis selector       BLOCKED   prediction-only W/D hypothesis selector FAIL
                                      (83/140, NIGHT 13/28)
                                      challenge/evaluation_v2/selector_diagnostic/
Wood symmetry               BLOCKED   registry symmetry_status = UNREVIEWED,
                                      selector NOT_RUN
Wood canonical migration    BLOCKED   CANONICAL_MIGRATION_NOT_PASS
신규 어노 146 signed axis   BLOCKED   pose_status = UNCONFIRMED_SIGNED_AXIS 146/146
                                      camera_facing_pnp.axis_assignment_confirmed
                                      = False 146/146
final manifest              NOT_FROZEN  PAPER_EVAL 은 DEV role 이다 (held-out 아님)
intrinsics                  OK        camera_data.intrinsics 가 전 프레임 존재
```

## 핵심 제약 — GT 로 prediction 을 고를 수 없다

pose metric 을 내려면 예측된 cuboid 의 **W/D 축 배정**을 정해야 한다.
카메라를 향한 면이 1.1 m 쪽인지 1.3 m 쪽인지가 미결이기 때문이다.

GT 의 `axis_assignment` 를 써서 그 모호성을 풀면 평가가 문제를 대신 풀어 주는 것이
되고, 그 수치는 배포 가능한 성능이 아니다.  이 저장소에는 그렇게 해서 5cm5°
30.4% 가 나왔다가 배포가능 정보만 쓰면 19.3% 로 떨어진 이력이 있다.

따라서 prediction-only selector 가 신뢰 가능해지기 전에는 pose 를 열지 않는다.

## 이번 트랙에서 확인된 것 — filter 는 pose 를 풀지 못한다

self-training 필터는 registry 의 두 hypothesis 를 각각 풀고 **score 최소값**만 쓴다.
이긴 hypothesis 의 pose 는 filtering 을 위한 latent check 일 뿐이고 저장하지 않는다.

```text
pseudo supervision 으로 저장한 것   2D box + 2D keypoint + visibility
저장하지 않은 것                   pseudo 6D pose
```

`min(hypothesis_A, hypothesis_B)` 는 "둘 중 하나로는 기하가 성립한다" 만 말한다.
어느 쪽인지는 말하지 않으므로 selector 문제를 풀어 주지 않는다.

## 해제 조건

아래가 모두 참이 되어야 `READY` 로 바꾼다.

```text
1. Plastic prediction-only W/D selector 가 합의된 기준을 통과
2. Wood symmetry_status 가 FROZEN 이 되고 selector 가 실행됨
3. Wood canonical migration 이 PASS
4. 신규 어노 146 의 axis_assignment 가 확정 (어노테이션 단계 산출물)
```

## 해제되면 할 일

```text
1. R0 · R0-CONT · R1~R5 · replicate 를 같은 pose evaluator 로 한 번에 재평가
2. M1 / M2 / M3 / M5 의 pose 열 자동 갱신
3. _docs/paper/ABSTRACT_RESULT_SLOTS.md 의 YAW_RESULT_SLOT 갱신
```

## 초록에 미치는 영향

`YAW_RESULT_SLOT = BLOCKED` 이므로

> "reduces median yaw error by [Y]"

문장은 **사용할 수 없다**.  다른 2D metric 을 yaw 라고 바꿔 쓰지 않는다.
자세한 것은 `_docs/paper/ABSTRACT_RESULT_SLOTS.md`.
