# Table F — 다른 모델과의 비교 (같은 평가 계약)

population   PAPER_EVAL 319 positive + 2,689 negative — 위 표들과 **같은 셋**
evaluator    동일.  그래서 Table A/B 와 같은 줄에 놓고 읽어도 된다
source       data/pallet/results/paper_eval_v1/arms/<ARM>.json + ARM_RESULTS.json

```text
model                                    det@.5 ↑  AP50-95 ↑   AUROC ↑   FPR95 ↓  kp med px ↓  gross20 ↓
--------------------------------------------------------------------------------------------------------
R0  합성만 학습 (YOLO26n)                  0.9749     0.7688    0.9921    0.0417        6.616     0.1720
R5  ST 전체일관성 (제안)                   0.9843     0.7585    0.9953    0.0283        7.210     0.1972
DOPE  합성만 학습 (VGG19)                  0.7367     0.3412    0.9903    0.0409       10.916     0.2193
실GT FT  legacy v1v2                       0.9843     0.7172    0.9896    0.0513        9.324     0.2728
실GT FT  A (real157+neg259+synth12k)       0.9937     0.8399    0.9991    0.0041        5.990     0.1316
실GT FT  B (patience0 40ep)                0.9937     0.8293    0.9992    0.0000        5.628     0.1398
```

★ 읽을 때 반드시 붙일 단서

- DOPE 는 box head 가 없다. AP 는 검출된 코너의 bounding box 에서 유도한 값이라
  YOLO 의 학습된 box AP 와 **같은 양이 아니다**. score 도 belief peak vs box
  confidence 라 AUROC/FPR95 척도가 다르다. 직접 비교가 성립하는 열은
  **kp med px 와 det** 뿐이다.
- REALFT 3 개는 **실제 manual GT 로 학습**했다. 통제된 비교가 아니라 상한선이며,
  논문에서 unlabeled adaptation arm 과 같은 열 블록에 넣지 않는다.
- REALFT_LV1V2 는 실제 GT 로 학습했는데도 kp 9.324 px 로 R0 보다 나쁘다 —
  '실제 라벨이면 무조건 낫다' 가 아니라는 반례다.
- 6D 수치는 이 표에 없다. POSE_EVALUATION_* 는 R0~R5 일곱 arm 만 존재한다.
  ARM_RESULTS 의 pose_status BLOCKED 는 pose closure 이전의 옛 표기다.

Paper role: **appendix / baseline reference**.
