# DATA RESAMPLING CAUSAL SCREEN — PHASE 10-11

```
LOW_ANGLE_COVERAGE_CAUSAL_SIGNAL = False        (사전등록 gate)
```

**단 떨어진 절이 주효과가 아니다.** 아래를 "저앙각 노출은 효과가 없다" 로 읽으면 안 된다.

---

## 설계

```
target   저앙각 (<8도) — risk map 결과를 보고 실행 전에 고정
DCTRL    자연 분포 (<8 = 7.66%)
DANGLE   저앙각 상향 (<8 = 30%, 자연 대비 4배)
source   E3 @18k, 두 arm 동일
budget   3,000 step × batch 8 = 24,000 노출 × 2 seed
동일      optimizer(AdamW, fresh) / LR 1e-3 / WD 1e-4 / batch / steps / source
새 렌더   없음 — 기존 40k 안에서만
```

**control 도 재학습했다.** 3,000 step × 8 = 24,000 은 pool 33,758 보다 작아 한 pass 도
돌지 않는다. 즉 pool 의 **앞부분이 곧 노출 분포**이고, 기존 P0 는 자연 순서로 스트리밍
했으므로 그대로 control 로 쓰면 재가중과 재정렬이 섞인다. 두 arm 을 같은 draw 절차로
만들었다.

### 노출 감사 (학습 전)

```
             <8      8-15    15-30   >=30    V=8     V=7     V<=6    unique
DCTRL      0.0766  0.0913  0.2240  0.6081  0.6647  0.1109  0.2244   24,000
DANGLE     0.3000  0.0697  0.1706  0.4597  0.6477  0.0998  0.2525   19,387
size(mid/near-large/small)  DCTRL 0.268/0.277/0.455   DANGLE 0.266/0.275/0.459
```

**V 분포와 size 분포가 거의 그대로다** — low-angle 개입이 몰래 V 개입이나 크기 개입이
되지 않았다. ⚠ DANGLE 은 unique 19,387 (vs 24,000) 로 다양성이 낮다. 저앙각 프레임을
2.78회 반복하므로 불가피하며 한계로 기록한다.

---

## 결과 (step 3000, dev = D2+D3+D4)

```
seed1  low-angle    frs      scale     R        t          | V=8 R/t      | line angle/offset
       DCTRL      1.5347   0.1317   11.645   0.4769
       DANGLE     1.2769   0.1367    8.170   0.4404
       변화       +16.8%    −3.8%   +29.8%   +7.7%        | +4.2% / +3.0% | +0.4% / +4.1%   PASS

seed2  DCTRL      1.7899   0.1761   10.364   0.6628
       DANGLE     1.3785   0.1503    9.044   0.5086
       변화       +23.0%   +14.6%   +12.7%  +23.3%        | +2.9% / +9.4% | −4.8% / −6.4%   FAIL
```

### gate 별 판정

```
기하 (frs 또는 scale ≥ +10%)        seed1 +16.8%  seed2 +23.0%     PASS  PASS
R/t 같은 방향 개선                   seed1 +29.8/+7.7  seed2 +12.7/+23.3  PASS  PASS
full-view 안전 (R/t 열화 ≤5%)        seed1 +4.2/+3.0  seed2 +2.9/+9.4    PASS  PASS  (개선)
line guard (angle/offset 열화 ≤5%)   seed1 +0.4/+4.1  seed2 −4.8/−6.4    PASS  FAIL
```

**주효과 세 절은 두 seed 모두 통과한다.** 유일한 탈락은 seed2 의 **line offset −6.4%**
로, 허용치 −5% 를 1.4pp 넘겼다.

기전은 분명하다 — DANGLE 은 corner branch 만이 아니라 **line branch 의 학습 분포도**
바꾼다(같은 프레임을 두 head 가 함께 본다). 저앙각 프레임의 line 통계가 다르므로
dev 의 offset 이 밀린다. guard 는 정확히 그걸 잡으라고 있는 것이고, 제 역할을 했다.

⚠ 결과를 보고 guard 를 완화하지 않는다. 재시험이 필요하면 **새 사전등록**으로 돌려야
한다(예: line 노출을 고정한 변형).

---

## paired frame bootstrap (10,000, seed 분리)

```
low-angle (n=130)              seed1                          seed2
front_rear_shift        +17.89 [ −1.97, +40.62] P.964   +22.86 [ −1.49, +42.12] P.968
affine_scale_gap         −2.34 [−26.65, +15.80] P.411   +14.80 [ −6.85, +31.33] P.925
corner_error            +13.67 [ −6.73, +34.50] P.908   +12.98 [ −6.21, +31.63] P.907
R                       +28.38 [ +3.90, +45.33] P.989   +15.06 [ −3.13, +36.05] P.955
t                        +7.86 [−22.76, +28.57] P.716   +24.25 [ −4.48, +50.09] P.956

full-view V=8 (n=1024)         seed1                          seed2
R                        +3.89 [ −3.28, +10.55] P.865    +3.61 [ −4.55, +11.64] P.826
t                        +2.33 [−11.45, +12.52] P.650    +8.47 [ −3.72, +19.23] P.909
affine_scale_gap         −7.31 [−23.13,  +5.36] P.140    −8.89 [−23.04,  +3.53] P.084
```

`seed1 low-angle R` 하나가 CI 로 0 을 배제한다(+3.90 ~ +45.33). 나머지는 CI 가 0 을
포함하지만 **10개 low-angle 지표 중 9개가 양의 방향이고 P(better) 가 0.90~0.99** 다.
full-view 에서 확립된 열화는 없다.

⚠ **저앙각 표본이 seed 당 130 프레임뿐**이라 CI 가 넓다. 이 신호를 "확립됨" 으로
쓰면 안 된다.

---

## 이 결과의 위치

이번 세션에서 시도한 개입들과 나란히 두면 차이가 분명하다.

```
Ridge scale 보정        pose 악화, 5cm5deg −7.0/−4.3pp
pose-aware corner       28개 비교 전부 CI 가 0 포함, 부호도 섞임
theta-only solver       rotation 은 강하나 translation gate 를 못 넘음
저앙각 resampling       기하·R·t 가 두 seed 같은 방향, P 0.90~0.99, full-view 무해
```

**목표 기하를 두 seed 같은 방향으로 움직인 것은 이 개입이 유일하다.** 그래도
사전등록 gate 는 FALSE 이고, 브리프의 중단 기준 6 에 따라 **fresh synthetic 생성은
승인되지 않는다.**

---

## 한계

- 저앙각 dev 표본 seed 당 130 프레임. CI 가 넓다.
- DANGLE 은 unique 19,387 (2.78회 반복) — 노출 다양성이 control 보다 낮다.
- 3,000 step, seed 2개, synthetic 전용. real 전이 미평가, sealed 미접근.
- line guard 탈락은 개입이 두 head 의 분포를 함께 바꾼 결과다. corner 쪽 이득과
  line 쪽 손실을 분리하려면 별도 설계가 필요하고, 그건 새 사전등록 대상이다.

산출: `data_resampling_{DCTRL,DANGLE}_seed{1,2}.json`,
`data_resampling_report_step3000.json`, `data_resampling_bootstrap.json`,
frames npz 4개.  스크립트 `scripts/stage0/multihead/mh_resample.py`.
