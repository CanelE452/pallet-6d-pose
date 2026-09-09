# PURPOSE — LEGACY_V1V2_P0_10K FINE-TUNE ON G38

[소비처] 과제(challenge) 트랙 배포 모델 선정. 현재 target-free BEST 인 G38 을
        그대로 쓸지, legacy v1v2 대체 셋으로 FT 한 버전을 쓸지 결정한다.
        논문 트랙에는 넣지 않는다 (데이터셋 자신의 PURPOSE.md 가 그렇게 못 박음).

[문장] G38(generic 38K, target-free)에 LEGACY_V1V2_P0_10K(P0 플라스틱 1만 장,
       구 train_palletobj_v1/v2 대체분)를 15 epoch FT 하면, target 팔레트에 대한
       real 검출·코너 정확도가 G38 대비 개선된다.

## 왜 이 계열을 또 하는가 (매몰 방지)

`G38 + SUPPORT` 는 hits 0/6 로 HARM 판정났고 memory 는 "generic support 를 더하면
검출이 나빠진다(2회 독립)" 로 닫아 뒀다. 이번이 다른 이유는 **더하는 데이터의 성격**이다.

```
G38+SUPPORT   generic 팔레트 support 1,933  →  target 과 무관한 분포 확장 → HARM
이번(LV1V2)   P0 target 팔레트 10,000       →  target 분포 자체를 주입
```

같은 표의 참조행이 근거다 — target 학습본은 G38 을 크게 앞선다.

```
model   cbox    median     p90    night_top1
────────────────────────────────────────────
G38    0.852    12.03    66.66         0.536
OLD    0.969     9.68    40.99         0.929   ← 구 v1/v2 학습본
FT     0.984     6.47    25.40         0.964   ← real FT (upper)
```

★ 단 OLD/FT 는 학습 데이터 조건이 달라 **REFERENCE/UPPER DIAGNOSTIC** 이지
  같은 실험군이 아니다. 이번 arm 이 목표하는 건 G38→OLD 구간의 회수다.

## 판정 지표 — 사전등록 (GATE_PREREG.json 에 하드코딩, 결과 보고 못 고침)

`G38_PLUS_SUPPORT_GATE_PREREG.json` 과 **동일한 문턱**을 쓴다. 같은 질문("G38 에
데이터를 더하면 좋아지나")이므로 문턱을 새로 만들면 비교가 깨진다.

```
hits (6개 중 3개 이상 통과해야 GAIN)
  all_cbox_pp              correct_box_recall  +0.02 이상
  all_median_rel           corner_median       -8% 이상
  all_p90_rel              corner_p90          -10% 이상
  night_top1_cbox_frames   night top1_cbox     +3 프레임 이상
  night_p90_rel            night corner_p90    -15% 이상
  night_margin_abs         night margin_median +0.10 이상

guards (하나라도 걸리면 GAIN 취소 → HARM)
  all_cbox_pp        -0.03 미만으로 떨어짐
  night_any_cbox     -2 프레임 이상 떨어짐
  all_p90_rel_worse  corner_p90 이 +15% 이상 악화
```

## 교락 (미리 적어 둔다)

- FT 는 G38 대비 **추가 step 을 쓴다**(15ep x 10K). "데이터 성격" 과 "추가 학습량" 이
  분리되지 않는다 — causal ablation 아님. 순수 성능 후보 비교다.
- NIGHT n=28, seed 1개. night 지표의 단일 seed 판정은 약하다.
- real 평가셋은 DEV(개발용)다. FINAL 은 미동결이라 이 결과로 논문 수치를 못 만든다.
