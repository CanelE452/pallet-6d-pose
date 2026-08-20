# FINAL FUSION DECISION

```
FINAL FUSION    NOT_ESTABLISHED   (사전등록 gate 의 엄격 연언 기준)
ARCHITECTURE    NOT_LOCKED
최종 pose 경로  당분간 CORNER_PNP_PRIMARY 유지
```

## 그러나 F3 는 지금까지 나온 것 중 유일하게 작동하는 후보다

이번 screen 의 가설 — **line 은 rotation 만, point 는 translation** — 은 의도한 대로
작동했다.

```
                ALL R        ALL t        5cm5      판정
F1 (기존 joint)  +24.1/+15.9  -4.3/+1.8   -1.0/+1.6pp   s1 FAIL
F3 (역할 분리)   +22.6/+12.7  +4.9/+1.9   +1.4/+0.9pp   두 seed PASS
```

F1 은 seed1 에서 translation 을 4.3% 잃는다. F3 는 **같은 rotation 이득을 얻으면서
translation 을 오히려 개선한다.** 그 차이는 corner-only t-refit 하나에서 나온다
(F2 는 새 R 에 옛 t 를 붙여 5cm5deg 를 떨어뜨리고, F3 의 재적합이 그것을 되돌린다).

rotation 이득은 견고하다 — ALL 과 partial-visibility(Vvis<=5)에서 **두 seed 모두
CI 가 0 을 배제**한다(+12.5~22.5%, +15.0~31.5%, P=1.000).

## 왜 그래도 LOCK 하지 않는가

gate 는 hard cell 도 요구하고, `LA_HARD` seed2 에서 R −1.3% / t −7.1% 로 떨어진다.
결과를 보고 gate 를 완화하지 않는다.

다만 그 cell 은 n=51 이고 bootstrap CI 가 [−31.4, +16.0] 으로 0 을 크게 포함한다
(P=0.198). 같은 cell 의 seed1 은 +11.1% / +5.1% 로 반대다. 직전 감사가 보인 대로
이 규모의 cell 에서는 **측정 잡음이 효과와 같거나 크다**. 즉 F3 를 떨어뜨린 것은
"F3 가 hard 에서 나쁘다" 는 증거가 아니라 **그 cell 을 판정에 쓸 만큼 재지 못한다**는 뜻이다.

## 다음 (자동 실행 금지, 승인 필요)

### 1순위 — gate 를 바꾸지 말고 측정을 고친다

`LA_HARD` 를 n=51 → 수백 규모로 키우면 F3 판정이 실제로 갈린다. 지금 dev 에서 뽑을 수
있는 LA_HARD 는 185 가 전부이고 train 쪽 931 장은 E3@18k 가 이미 봤다.
→ **평가 전용 저앙각 holdout** 이 필요하다. 합성팀에 학습용보다 이쪽이 먼저다.

### 2순위 — PHASE 12 조건부 ablation

브리프가 허용한 `ENTROPY_WEIGHTED_F3` 는 "rotation 은 강하게 통과, translation safety 가
근소하게 실패" 일 때의 후보다. 지금은 그 패턴이 아니다(ALL translation 은 통과,
실패는 hard cell 의 rotation·translation 동시). **제안하지 않는다.**

## 확정된 사실 (재사용 가능)

```
rho 는 objective 에 없다             rho 13px 이동에도 residual 변화 6.2e-15
yaw 축은 convention 에서 유도됨      +5도 주입 → -5.0000도 복구, 잔차 0.0
F2 의 t 는 F0 와 정확히 동일          설계상 동결, max_abs 0.0 으로 확인
GT 입력에서 pose 복구                 R 2e-06도, t 2e-08 m
```

## NEXT

negative dataset 도착 후 negative suppression qualification.
그 전에 평가 전용 저앙각 holdout 이 생기면 F3 를 같은 gate 로 재판정한다.
