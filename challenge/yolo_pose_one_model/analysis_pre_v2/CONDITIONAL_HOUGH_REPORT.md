# CONDITIONAL HOUGH — 최종 판정

**VERDICT = `HOUGH_TRACK_CLOSED`**  (PHASE 1 oracle: `HEADROOM_SUFFICIENT_PROCEED`)

Direct-Hough 를 닫기 전 마지막 확인. "점이 좋을 땐 point-only, 점이 불안정할 때만
Hough/F3" 라는 **조건부 fallback** 이 catastrophic pose 실패를 회수하는가.

- 재추론 **0**. 저장된 YOLO 60ep 키포인트 + FINAL40K seed1 의 line theta/rho 재사용.
- 평가 대상 YOLO = `yolo26n_paper_generic_v1` **60 epoch** (`last.pt`).
  ★ 기존 hybrid 매트릭스의 B1 은 5 epoch 판이라 그대로 쓰지 않고 다시 계산했다.
- arm 정의 — `Y0` point-only(SQPnP+refineLM) / `YH` 항상 Hough+F3 /
  `YG` gate 로 조건부 / `ORACLE` GT 를 보고 프레임마다 더 좋은 쪽.

---

## PHASE 1 — oracle headroom (gate 를 만들 가치가 있는가)

★ oracle 은 **deployment 결과가 아니다.** 완벽한 gate 가 존재한다는 가정의 **상한**이다.

### REAL_DEV_OPEN_56 (검출 n=50)
```
arm         R med    R p90    t med    ADD-S     IoU    5cm5
───────────────────────────────────────────────────────────
Y0           2.25     5.54   0.0396   0.0437   0.685   0.640
YH           5.00     7.84   0.0427   0.0708   0.529   0.380
ORACLE       2.25     5.14   0.0396   0.0437   0.680   0.680
```
Hough R 승률 0.100 ·
oracle R 개선 0.0% ·
5cm5 이득 +0.040

### REAL_CHALLENGE_DEV_105 (검출 n=66)
```
arm         R med    R p90    t med    ADD-S     IoU    5cm5
───────────────────────────────────────────────────────────
Y0           5.63    79.34   0.1189   0.1618   0.517   0.242
YH           8.06    90.11   0.1186   0.1707   0.359   0.182
ORACLE       4.78    51.91   0.1221   0.1638   0.491   0.288
```
Hough R 승률 0.394 ·
oracle R 개선 15.0% ·
5cm5 이득 +0.045

**STOP RULE 판정**: 사전등록 3 조건(oracle R 개선 <10% AND 5cm5 이득 <+5pp AND
Hough 승률 <0.25)이 **동시에** 성립해야 종료인데, challenge 에서 R 개선 15.0% ·
승률 0.394 로 두 개를 통과 → `HEADROOM_SUFFICIENT_PROCEED`. PHASE 2 진행.

핵심 관찰 — `YH`(항상 Hough) 는 두 population 모두 `Y0` 보다 **나쁘다**
(open 2.25→5.00°, challenge 5.63→8.06°). 즉 Hough 는 평균적으로 해롭고,
가치가 있다면 오직 **선택적으로 켤 때만** 있다. 그래서 gate 가 전부다.

Hough 가 이기는 31 프레임의 성격: Y0 R median **17.15°**(전체 5.63°),
reproj median **11.36 px**(전체 8.68 px) — 점이 이미 크게 틀린 프레임에 몰려 있다.
분포로는 gate 가 될 것처럼 보인다.

---

## PHASE 2~3 — 예측 가능한 gate + session LOSO CV

gate feature 는 `pnp_reproj` **하나만** 썼다 (추론 시 즉시 계산, GT 미사용).
결과를 보고 feature 를 늘리지 않았다.

tau 는 **session Leave-One-Out** 으로 골랐다 — 같은 DEV 에서 고르고 같은 DEV 로
평가하면 낙관 편향이므로, held-out session 을 빼고 나머지 6 개에서 고른 뒤
held-out 에서만 평가했다. 선택 목적함수는 사전등록: *R median 최소화, 단 t 악화
≤5% 이고 5cm5 악화 ≤0pp*.

```
held-out       n     tau     활성    Y0 R    YG R     비악화
───────────────────────────────────────────────
cad           22   17.83   0.14    2.71    2.71       예
night08        7   17.56   0.29    2.46    6.04     아니오
night09       14   17.94   0.14   60.68   60.68       예
noapril       12   19.41   0.00    1.12    1.12       예
outside       16   18.11   0.12    3.34    3.34       예
pallet07      27   13.02   0.22    3.53    3.74     아니오
pallet09      18   17.38   0.22    9.53    9.98     아니오
```

R 비악화 fold = **4/7** (게이트 기준 ≥5/7).

---

## PHASE 4 — Y0 vs YH vs YG (CV 예측)

### REAL_DEV_OPEN_56
```
arm         R med    R p90    t med    ADD-S     IoU    5cm5
───────────────────────────────────────────────────────────
Y0           2.25     5.54   0.0396   0.0437   0.685   0.640
YH           5.00     7.84   0.0427   0.0708   0.529   0.380
YG           2.25     5.55   0.0396   0.0437   0.680   0.660
```
활성화율 0.100

### REAL_CHALLENGE_DEV_105
```
arm         R med    R p90    t med    ADD-S     IoU    5cm5
───────────────────────────────────────────────────────────
Y0           5.63    79.34   0.1189   0.1618   0.517   0.242
YH           8.06    90.11   0.1186   0.1707   0.359   0.182
YG           5.99    89.93   0.1250   0.1675   0.469   0.273
```
활성화율 0.212

### 도메인별
```
domain         n    Y0 R    YG R  Y0 5cm5  YG 5cm5     활성
─────────────────────────────────────────────────────────
cad           22    2.71    2.71    0.818    0.818   0.14
night08        7    2.46    6.04    0.429    0.429   0.29
night09       14   60.68   60.68    0.071    0.071   0.14
noapril       12    1.12    1.12    0.917    0.917   0.00
outside       16    3.34    3.34    0.188    0.250   0.12
pallet07      27    3.53    3.74    0.407    0.481   0.22
pallet09      18    9.53    9.98    0.056    0.056   0.22
```

---

## PHASE 5 — 사전등록 판정

```
지표                      측정        게이트
──────────────────────────────────────────────────
challenge R 개선        -0.0644     >= +0.10 (지지) / < +0.05 (기각)
challenge 5cm5 이득     +0.0303     >= +0.03
challenge t 악화        +0.0510     <= +0.05 (기각선 +0.10)
open 5cm5 악화          -0.0200     <= +0.02
open R 악화             +0.0000     <= +0.05 (기각선 +0.05)
CV fold R 비악화        4/7          >= 5/7
```

**`challenge R 개선 = -0.0644` 이 기각선 +0.05 미달** →
`HOUGH_TRACK_CLOSED`.

정직하게 적어 둘 것 — **5cm5 는 올랐다**(challenge +3.0pp, open +2.0pp) 그리고
open 은 손상되지 않았다(R 동일, t 100% 비악화). 그러나 사전등록 1 차 지표는
rotation median 이고 그게 악화됐다. 결과를 보고 지표를 바꾸지 않는다.

---

## PHASE 6 — 왜 oracle 15% 가 gate 에서 −6.4% 가 됐는가

1. **gate feature 의 판별력이 거의 없다.** `pnp_reproj` 로 "Hough 가 이길
   프레임" 을 예측한 AUC = **0.597**
   (0.5 = 무작위). PHASE 1 에서 본 median 차이(11.36 vs 8.68 px)는 분포 겹침을
   가린 집계 통계였다. — 집계 통계가 per-frame 정확성으로 전이되지 않는 패턴이
   이 프로젝트에서 또 반복됐다.
2. **oracle 이득이 소수에 몰려 있다.** 31 개 승리 프레임 중 **상위 5 개가 전체
   이득의 61%**
   를 차지한다. 그 5 개를 정확히 집지 못하면 이득은 0 이고 손해만 남는다.
3. **켠 프레임의 손익이 동전던지기다.** CV gate 가 켠 19 프레임 중
   도움 10 · 손해 9,
   도움일 때 median +0.807° ·
   손해일 때 median −0.546°.
   거의 대칭이라 median 이 개선될 이유가 없다.

**결론**: oracle headroom 은 실재하지만 GT 없이는 접근할 수 없다.
`HOUGH_TRACK_CLOSED`.

---

## 적용 범위 (넘어서 주장하지 말 것)

- synthetic 아님. real DEV 161 장 중 60ep YOLO 가 검출한 116 장 기준.
- `REAL_DEV_OPEN_56` · `REAL_CHALLENGE_DEV_105` 는 **DEV** 다. final test 아님.
- oracle 수치는 진단용 상한이며 **deployment 결과가 아니다**.
- 이 판정은 *조건부 Hough* 를 닫은 것이다. line branch 자체(SPLIT_LATE 의 구조적
  분리, theta-only 회전 이득)에 대한 과거 판정을 뒤집지 않는다.
- gate feature 는 `pnp_reproj` 하나만 시험했다. 다른 feature 로 다시 열려면 **새
  brief** 가 필요하다 — 결과를 본 뒤 feature 를 늘리는 것은 이 판정의 무효화다.

## 산출물

`CONDITIONAL_HOUGH_ORACLE.json` · `CONDITIONAL_HOUGH_CV.json` ·
`CONDITIONAL_HOUGH_PHASE6.json` · `CONDITIONAL_HOUGH_PER_FRAME.csv` ·
`CONDITIONAL_HOUGH_PLOTS.png` · 스크립트 `conditional_hough.py` ·
`conditional_gate_cv.py` · `conditional_phase6.py`
