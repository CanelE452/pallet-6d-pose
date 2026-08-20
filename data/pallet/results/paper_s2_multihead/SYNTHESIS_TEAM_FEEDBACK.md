# SYNTHESIS TEAM FEEDBACK

## A. LA_HARD(Y30+) targeted enrichment benefit = **NOT_ESTABLISHED**

근거 — C1_RESCUE(노출 완벽 보존 + canonical Y30 1,500) vs C0, 최종 3,000 step:

```
seed1   obs_rms -23.5%   R -26.0%   t  -9.0%      전부 악화
seed2   obs_rms +29.3%   R +22.7%   t -23.1%      rms/R 개선, t 악화
seed3~6 미실행 — E3@18k source 부재로 HARD BLOCK (SEED_REPLICATION_PROTOCOL.md)
```

두 seed 가 rms·R 에서 정면충돌하고, t 는 둘 다 악화다. 학습곡선을 보면 이건
"seed 마다 학습 방향이 다르다" 가 아니다 — **파라미터 drift 는 두 seed 가 1% 차이로
동일**한데(corner_late relative L2 0.4429 vs 0.4383), n=185 짜리 cell 에서 읽는 지표만
인접 마크 사이에 최대 51 포인트씩 흔들린다. **효과가 측정 잡음보다 작다.**

## B. LA_FRONTAL residual failure = **SEED_UNSTABLE**

```
최종 3,000 step   seed1 R +2.2  t  -3.1        seed2 R -33.5  t -53.1
곡선              두 seed 모두 250~2000 구간에서 부호가 여러 번 뒤집힘
```

`STILL_PRESENT` 로 올리지 않는 이유: seed2 는 확실히 나쁘지만(직전 판정에서
t −58.9% P=0.015 확립) seed1 은 0 근처다. 재현되는 손상이라고 말하려면 최소한
두 seed 가 같은 방향이어야 하는데 아니다.
`RESOLVED` 도 아니다 — 노출을 완벽히 보존했는데도 seed2 에서 손상이 남았다.

## C. CORNER_LA_FRONTAL_V1 2.5K main-training value = **UNRESOLVED**

`YES` 가 아닌 이유: 같은 실험 설계에서 **hard-cell enrichment 조차 효과를 확립하지
못했다.** enrichment 가 이 파이프라인에서 작동한다는 증거가 없는 상태에서 frontal
enrichment 의 가치를 판정할 수 없다. 데이터가 이미 생성됐다는 사실은 판정 근거가 아니다.

`NO` 가 아닌 이유: frontal 이 병목인 것 자체는 여러 번 관측됐다(canonical failure map
에서 LA_FRONTAL 이 rms 두 번째로 나쁜 cell). 다만 지금 측정 장치로는 개입 효과를
읽을 수 없다.

**해석**: 지금 막힌 것은 데이터가 아니라 **평가 해상도**다.

## D. Additional synthetic data request = **NONE (지금은)**

새 렌더를 요청하지 않는다. 렌더 0장.

다만 **평가용**으로는 실질적 필요가 있고, 그건 학습 데이터 요청과 성격이 다르다.
현재 held-out 저앙각 프레임이 dev 전체에 이것뿐이다.

```
LA_FRONTAL   134       LA_EASY   170       LA_HARD   185
```

train split 에는 같은 cell 이 706 / 950 / 931 장 있으나 **E3@18k 가 이미 학습에 사용**해
평가로 쓸 수 없다. 이 n 으로는 20~30% 미만 효과를 가릴 수 없다는 것이 세 번 반복 확인됐다.

만약 합성팀에 여력이 있다면, 학습용 frontal 2.5K 보다 **평가 전용 저앙각 holdout** 이
지금 더 가치가 크다. 사양은 정해지면 별도로 제안하겠다 — 이번 phase 에서는 요청하지 않는다.

---

## 다음 실험 하나 (자동 실행 안 함)

seed 확장(15~17시간)을 권하지 않는다. 파라미터 수준 between-run 분산이 이미 1% 인데
seed 를 늘리는 것은 지배적 잡음(within-run)을 건드리지 못한다.

대신 **평가 설계를 먼저 고친다** — LA cell 의 held-out n 을 키우거나, 단일 마크가
아닌 다중 마크 추정량으로 바꾼다(후자는 사전등록 필요). 둘 다 새 학습이 아니다.
