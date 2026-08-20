# PARAMETER DRIFT AUDIT (training 0)

`Δθ_C = θ_C(C1_RESCUE) − θ_C(C0)`, 같은 seed·같은 마크.

```
relative L2                    250      500     1000     2000     3000
seed1 corner_late            0.1130   0.1741   0.2618   0.3687   0.4429
seed2 corner_late            0.1157   0.1773   0.2553   0.3663   0.4383
seed1 belief_head            0.0449   0.0702   0.1058   0.1470   0.1780
seed2 belief_head            0.0452   0.0706   0.1021   0.1441   0.1729
seed1 line_late              0.0000   0.0000   0.0000   0.0000   0.0000
seed2 line_late              0.0000   0.0000   0.0000   0.0000   0.0000
```

## 읽기

1. **corner private late 가 belief head 보다 2.5배 크게 움직인다.** 데이터 개입이
   주로 private late block 을 바꾸고 head 는 따라간다.

2. **두 seed 의 drift 궤적이 사실상 겹친다** (3,000 에서 0.4429 vs 0.4383, 1.0% 차).
   같은 스케줄·같은 dose 를 주면 파라미터는 재현 가능하게 움직인다는 뜻이다.

3. 그런데 같은 두 run 의 cell 지표는 최종 마크에서 부호가 갈린다
   (LA_HARD R −26.0 vs +22.7). 파라미터가 재현되는데 지표가 안 되면
   **원인은 학습 쪽이 아니라 읽는 쪽(cell n=134~185)** 이다.

4. `line_late` 가 전 마크 정확히 0 — two-stream 배선 불변식이 3,000 step 내내 유지됐다.
   percentage guard 가 아니라 구조적 보장이라는 주장이 학습 후에도 성립한다.

## 함의

seed 확장은 between-run 분산을 다루는 도구인데, 여기서 파라미터 수준의 between-run
분산은 이미 작다(1%). 흔들리는 것은 지표다. 따라서 **seed 를 늘리기 전에 평가 쪽을
고치는 것이 비용 대비 효과가 크다.**
