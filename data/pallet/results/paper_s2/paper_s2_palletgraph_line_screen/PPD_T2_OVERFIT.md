# PPD T2 32-frame overfit — polarity PASS, pixel line gate FAIL

> candidate set 은 oracle unsigned SAI-U 로 고정했고, learned map 은 **re-ranking 만** 한다.
> SAI candidate 는 **원본 해상도 T2 support** 로 생성했다 (100x100 nearest-resize 경로 금지).
> learned map 의 candidate scoring 은 **native 100x100 map 에서 bilinear sampling** 한다.

## Target gate v2 — PASS

`top_and_base_rate >= 0.90` 항목은 **삭제**했다 (숫자 완화가 아니라 proxy 정의 수정).
사유: observed-fragment target 에서는 top 또는 base 한쪽 evidence 만으로도 upright/inverted
를 구분할 수 있고, candidate pair 가 있는 48/48 frame 을 정확히 선택했으므로
`both present` 는 polarity 식별의 **필요조건이 아니다**.

```
nonempty_rate                 1.000  >=0.95   PASS
polarity_evidence_rate        1.000  >=0.95   PASS   (top* 또는 base* 중 하나 이상)
class_derivation_mismatch     0      ==0      PASS
yaw180_invariance_mismatch    0      ==0      PASS
nan_inf_count                 0      ==0      PASS
O0 helper numeric parity      20/20  identical PASS
conditional polarity acc      1.000  >=0.95   PASS   (n=48, candidate-pair frames)
conditional inversion         0.000  <=0.05   PASS
```

[별도 upstream metric] candidate-pair availability = **48/60 = 0.800**.
PPD selection gate 의 분모에 강제로 넣지 않는다.
따라서 "utility perfect" 라고 단독 서술하지 않는다 — 정확히는
**candidate-pair 가 존재하는 프레임에 조건부로 1.000** 이다.

## Overfit 설정

- 32 frames, train split 내부 **candidate-pair** frame 에서 결정적 추출
  (sha `480e7fa624694dbd`, asset Pallet_0~3 = 12/6/6/8, mode 6종 전부)
- L0/M0/M1 **동일** initial state_dict / sample order / optimizer / 800 steps / seed 1
- high-res feature: runtime 탐색 결과 `vgg[17]`, **256 ch, 100x100 shape assert 통과**
- loss calibration (train 20 batch, update 없음):
  L_line median 1.8359 -> lambda_pol 0.1259,
  lambda_mask 0.6071, lambda_out 0.1216

## 결과

```
arm   maskIoU   dice   macroF1   semAcc   polAcc   inversion   dist_reduction
L0     0.113   0.203    0.848    1.000    0.969      1/32          0.934
M0     0.942   0.970    0.663    0.999    1.000      0/32          0.878
M1     0.942   0.970    0.883    1.000    1.000      0/32          0.939
```

per-class precision@2 (recall@2 는 세 arm 모두 **전 클래스 1.000**):

```
L0  0.741 0.687 0.806 0.826 0.633   min 0.633
M0  0.560 0.454 0.778 0.416 0.340   min 0.340
M1  0.775 0.801 0.856 0.829 0.702   min 0.702
```

## Gate 판정

```
arm  H1 mask   H2 line   H3 candidate polarity   overall
L0   N/A       FAIL      PASS                    FAIL
M0   FAIL      FAIL      PASS                    FAIL
M1   FAIL      FAIL      PASS                    FAIL
```

- **H3 (candidate polarity) 는 세 arm 모두 PASS** — 목적 지표다.
  L0 0.969(1/32 inversion), M0/M1 1.000(0/32).  기준은 acc>=0.95, inversion<=2.
- **H2 (pixel line) 는 전부 FAIL** — recall 은 전 클래스 1.000 인데 **precision 만 미달**
  (기준 0.85).  macro F1 도 0.848/0.663/0.883 로 0.90 미달.
- **H1 (mask) 은 M0/M1 이 IoU 0.942 / Dice 0.970** — Dice 는 통과(>=0.97), IoU 는 0.95 에
  0.008 미달.

## H2 실패 해석

[확인] recall 이 전 클래스 1.000 이므로 **GT line 을 놓치는 것이 아니다**.
precision 만 낮다 = 예측이 GT 보다 **두껍거나 넓게** 퍼진다.

[확인] loss 는 step 400 부근에서 최저였다가 다시 오른다
(L0 0.2047 -> 0.1017 -> 0.1059 -> 0.1236).  단조 하강 중 중단된 것이 아니므로
**단순 예산 부족이 아니다**.

[추정] class-balanced positive weight(pos_weight 최대 200)가 positive 를 넓게 예측하는
쪽으로 균형점을 옮긴 것으로 보인다.  tol=2 cells 기준에서 '위치는 맞고 두께가 큰' 예측은
recall 을 1.0 으로 유지하면서 precision 만 떨어뜨린다.

[확인] 이것이 목적에 미치는 영향은 **없다시피 하다**: 두꺼운 line 이어도 candidate
polarity 는 0.969~1.000 이다.  H2 는 pixel-level proxy 이고 H3 가 목적 지표다.

## Mask ablation (overfit 단계 한정)

```
L0 (mask 없음)         polAcc 0.969   macroF1 0.848
M0 (mask regular.)     polAcc 1.000   macroF1 0.663   <- line precision 이 가장 나쁨
M1 (mask feature)      polAcc 1.000   macroF1 0.883   <- 전 지표 최고
```

[확인] M1 이 L0 보다 macro F1 과 polarity 모두 낫다.
[확인] M0 는 outside penalty 만 있고 gate 가 없어 line precision 이 오히려 나빠졌다.
[추정] soft mask **feature support**(M1)가 유효하고, outside penalty 단독(M0)은 해롭다.
단 32 frame overfit 이므로 **일반화 주장 금지**.

## 판정

- [확인] Overfit gate 는 지시문 기준으로 **세 arm 모두 FAIL** 이다 (H2 미달).
- [확인] 그러나 **목적 지표 H3 는 세 arm 모두 PASS** 이며, 이는
  "5-class polarity representation 을 학습해 candidate 를 고를 수 있다" 는 첫 학습 증거다.
- [판정] 지시문 Phase H4 "모든 arm FAIL -> 3k/1k 금지" 를 따라 **held-out 학습을 실행하지
  않았다**.  H2 기준을 낮추거나 loss weight 를 바꾸면 통과시킬 수 있으나, 그것은
  결과를 보고 설정을 바꾸는 것이라 하지 않았다.

## 사용자 결정이 필요한 지점

1. H2(pixel line precision/F1)를 **목적 지표가 아닌 진단 지표로 강등**하고 H3 기준으로
   3k/1k 를 진행할지.  근거: recall 1.000, polarity 0.969~1.000, precision 저하가
   목적에 영향을 주지 않음.
2. positive weight 상한(현재 200)을 낮춰 line 을 얇게 만들고 H2 를 다시 볼지.
   이 경우 calibration 을 다시 고정해야 한다.
3. 현행 기준을 유지하고 여기서 중단할지.

어느 쪽도 이번 실행에서 임의로 고르지 않았다.
