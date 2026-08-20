# C1_RESCUE_CANONICAL 결과 — H_curriculum 반증

```
CURRICULUM_INDUCED_FRONTAL_EXPOSURE_LOSS_SUPPORTED = True   (R1: 노출은 실제로 줄었다)
H_CURRICULUM_SUPPORTED                              = False  (R2: 보존해도 손상이 남는다)
TARGET_ENRICHMENT_EFFECT_NOT_ESTABLISHED            = True
→ CASE C
```

새 이미지 렌더 0. line parity 4개 비교 전부 `0.000e+00`.

---

## 스케줄이 요구대로 됐는지

```
             C0      RESCUE    변화
LA_FRONTAL   516  →   516      +0     ← 보존됨
LA_EASY      689  →   689      +0     ← 보존됨
LA_HARD      662  →   662      +0     ← 보존됨
NON_LA     22133  → 20633   -1500     ← 유일한 donor
NEW_LA_HARD    0  →  1500   +1500     ← old C1 실측 dose 와 동일
```

해상도 매칭 100%(미스 0), V_vis 평균 절대차 0.099, 24,000 슬롯 중 22,500 이 C0 와
같은 위치에 동일 프레임. schedule sha256 `558dbf41ea6c1555...`.

```
line param max|diff| C0 vs C1        seed1 0.000e+00   seed2 0.000e+00
line param max|diff| C0 vs C1_RESCUE seed1 0.000e+00   seed2 0.000e+00
```

즉 **설계는 의도대로 됐다.** 아래 결과는 배선 결함이 아니다.

---

## 핵심 — frontal 노출을 완벽히 보존했는데 frontal 이 회복되지 않았다

vs C0, MH_DEV 6,242, paired bootstrap 10,000. `★` = CI 가 0 배제.

```
LA_FRONTAL (n=134)          obs_rms          R                t
  seed1 C1-old              +1.67%        +7.30%          -4.21%
  seed1 C1-RESCUE           +0.44%        +3.35%          -5.84%
  seed2 C1-old             -13.75%        -8.52%      ★  -66.98%  P<0.001
  seed2 C1-RESCUE          -14.01%   ★   -31.09%      ★  -58.89%  P=0.015
                                          P=0.008
```

seed2 의 frontal translation 손상이 **거의 그대로 남았고**(−67.0% → −58.9%),
rotation 은 오히려 **새로 확립된 손상으로 나빠졌다**(−8.5% 미확립 → −31.1% P=0.008).

**노출 손실은 원인이 아니었다.** R1 이 측정한 −58/−78 draws 를 0 으로 되돌렸는데도
손상이 유지된다.

## 그리고 노출을 안 건드린 cell 에서도 새 손상이 생긴다

```
★ seed1 C1-RESCUE  LA_EASY  R  -32.13%  CI[-61.87, -9.12]  P=0.003
```

`LA_EASY` 슬롯은 C0 와 **완전히 동일**(689 → 689, +0)하다. 그런데 C1-old 에서는 없던
손상이 RESCUE 에서 확립된다. 노출 회계로는 설명되지 않는다 — **NEW 프레임 자체가
corner head 를 흔들거나, 이 예산에서 seed 잡음이 그만큼 크다는 뜻**이다.

## Y30 hard 이득도 확립되지 않는다

```
LA_HARD (n=185)             obs_rms          R                t
  seed1 C1-RESCUE          -21.88%       -24.67%         -19.48%     전부 악화
  seed2 C1-RESCUE          +23.12%   ★  +24.42%      ★  -25.41%
                                        P=0.992          P=0.017
```

seed1 은 세 지표 모두 나빠지고, seed2 는 회전이 확립 개선(+24.4%)인데 translation 이
확립 손상(−25.4%)이다. **두 seed 가 정면충돌**하며, 이는 C1-old 의 LA_HARD 와 같은 양상이다
(seed1 R −21.7% vs seed2 R +14.4%).

즉 hard cell 을 정확히 겨냥해 같은 dose 를 넣고 다른 모든 노출을 보존해도
**targeted enrichment 가 두 seed 같은 방향으로 모델을 움직이지 못한다.**

---

## 판정

```
FRONTAL   C1-RESCUE 가 C1-OLD 보다 회복?          아니오 (seed2 rotation 은 오히려 악화)
          C0 대비 새 손상 없음?                    아니오 (seed2 R -31.1% 신규 확립)
Y30 HARD  두 seed 같은 방향 개선?                  아니오 (seed1 전부 악화, seed2 R/t 반대)
SAFETY    NON_LA 새 실질 열화 없음?                통과 (RESCUE 는 오히려 seed1 t +4.6% 개선)
LINE      exact parity                             통과 (0.000e+00 × 4)
```

브리프 결정 트리에서 **CASE C** 다.

```
TARGETED_ENRICHMENT_EFFECT_NOT_ESTABLISHED = True
frontal 2.5K 도 만들지 않는다.
먼저 targeted enrichment 자체가 이 설정에서 모델을 움직이는지 재검토한다.
```

CASE B(H_DATA_REMAINS_PLAUSIBLE)로 가지 않는 이유: frontal 이 병목인 것은 맞지만,
**같은 실험에서 hard cell enrichment 조차 효과를 확립하지 못했다.** enrichment 가
작동한다는 증거 없이 frontal enrichment 를 정당화할 수 없다.

---

## 한계

- 3,000 step × 2 seed. 이 예산이 20~30% 미만 효과를 못 가른다는 것은 이미 두 번 드러났다.
  LA cell 은 n=134~185 라 CI 폭이 ±30~70% 다.
- seed 충돌이 반복된다(LA_HARD 에서 C1-old·RESCUE 둘 다). seed 2개로는 못 가른다.
  이것이 "효과 없음" 인지 "seed 분산이 효과보다 큼" 인지 구분하려면 seed 수를 늘려야 한다.
- NEW 의 V_vis 분포 편향(V_vis=4 가 9~10% vs BROAD 32~36%)은 matched substitution 으로
  donor 쪽에만 맞췄고 pool 자체 편향은 남아 있다.
- MH_DEV 는 학습에 쓰이지 않았지만 데이터 설계 단계에서 risk map 산출에 쓰였다.

산출: `R0_CANONICAL_REEVAL.json`, `R0_bootstrap.json`, `R1_EXPOSURE_AUDIT.json`,
`R2_FULL_REEVAL.json`, `C1_RESCUE_bootstrap.json`, `C1_RESCUE_SCHEDULE.json`
(+sha256), `CANONICAL_YAW_TRANSITION.json`, `canonical_yaw_frames.csv`,
`new_canonical_y30_pool.json`, `branch_curriculum_C1_RESCUE_seed{1,2}.json`.
