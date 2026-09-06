# SELFTRAIN_DECISION — 저앙각 self-training 을 할 것인가

지시문 §16 · §19 · §20 · §21 · §22.
**새 학습 0회.**  이 판정은 이미 돌아간 실험과 새로 층화한 pseudo-label 품질로만 낸다.

## 1. §16 의 전제부터 — 저앙각 개선 근거는 human GT 쪽이다

[확인] 저앙각에서 실제로 위치추정을 움직인 개입은 **human-GT real supervision**
하나다.  self-training 은 아니다.

[확인] `paper_selftrain_v1` 6 arm (R0_CONT / R1_NAIVE / R2_CONF / R3 / R4 / R5_PROPOSED)
결과, PAPER_EVAL 319장:

```text
축                       R0        최고 adapted arm      판정
2D kp median px       6.6157      R4  6.9987         6/6 이 R0 보다 나쁘다
6D IoU3D              0.60318     R2  0.59947        아무도 못 넘는다
6D ADDsym AUC         0.42847     R1  0.42045        아무도 못 넘는다
ranking AUROC         0.9921      R5  0.9953         R5 만 이긴다 (frame-level 만)
```

출처: `_docs/advising/2026-09-professor-consult/02_MAIN_RESULTS_SUMMARY.md`.
6D 는 24개 metric block 중 개선 방향으로 세션클러스터 CI 가 0 을 배제한 것이 **0개**다.

즉 self-training 은 **앞단(랭킹)은 움직이는데 뒷단 기하로 전파되지 않는다.**

## 2. §17·§18 이 더한 것 — 강조하려는 층이 가장 나쁜 층이다

`LOW_ANGLE_PL_QUALITY.md` 요약:

```text
필터 통과한 <8  의 gross(>20px)    20.3%    coverage 48.8%
필터를 안 건 >=15 의 gross           9.4%    coverage  100%
```

`ST_LOW`(저앙각 강조)는 노출 슬롯을 이 층으로 옮기는 것이다.
그러면 학생이 보는 gross 코너 비율이 올라간다.
**teacher 가 모르는 좌표를 학생에게 더 많이 가르치는 것에 근거가 없다** (§35, Case D).

## 3. §19 의 실험 설계가 성립하는가

지시문 §19 는 `ST_ALL` 대 `ST_LOW` 를 총 노출 고정으로 비교하라고 한다.
설계 자체는 성립한다 — `SELFTRAIN_EXPOSURE_LOCK.json` 이 pseudo:synthetic = 50:50,
슬롯 14,400 을 이미 얼려 놓아서, 필터마다 통과 장수가 달라도 **수량이 아니라 품질**
비교가 된다 [확인].

성립하지 않는 것은 **전제**다.  `ST_ALL` 이 이미 R0 를 못 넘었다(6/6).
그 계열에서 구성만 바꾼 변형은 지시문 §26 의 "duplicate already-rejected experiment" 다.

## 4. §20 의 draw 산포 요건도 비용을 키운다

[확인] pseudo-label 표집 draw 자체의 산포가 크다 — 대조군끼리도 rotation 6.7% ·
translation 14.3% 흔들린다 (memory `selftrain-pseudo-draw-noise-floor`).
그리고 ultralytics `seed` 는 dataloader 표집에 도달하지 않으므로
(memory `ultralytics-seed-does-not-reach-dataloader`) `args.seed` 만 바꾼 것을
replicate 라고 부를 수 없다 — **membership 이 실제로 다른 draw** 를 만들어야 한다.

즉 §19 를 제대로 하려면 arm 2개 x 독립 draw 3개 = 6 run 이고,
그 위에 §21 의 네 조건(저앙각 개선 + gross tail 감소 + draw 재현 + 고앙각 무손상)을
동시에 만족해야 한다.  전제가 이미 부정된 계열에 그 예산을 쓸 근거가 없다.

## 판정

```text
LOW_ANGLE_SELFTRAIN = NOT_JUSTIFIED   (지시문 §18 의 STOP, Case D)
```

§22 대로 다음으로 넘어간다 — `TARGETED_SYNTHETIC` 또는 `CAPACITY`.
어느 쪽인지는 §11 의 corrected real-FT 결과와 `SOURCE_DIVERSITY_AUDIT.md` 가 정한다.

## 무엇이 이 판정을 뒤집을 수 있나 (미리 적어 둔다)

1. teacher 를 R0 가 아닌 **corrected real-FT 모델**로 바꿨을 때 저앙각 PL 품질이
   달라진다면 — §11 이 개선을 보이면 재측정 가치가 있다.  그때는 teacher 를 바꾼
   것이지 필터를 바꾼 것이 아니므로 중복이 아니다.
2. adaptation pool 에 지금 없는 시점(정사각 근접 시야)이 들어온다면 —
   memory `adaptation-pool-lacks-near-square-viewpoints` 가 지적한 구조적 결손이다.

둘 다 지금은 성립하지 않는다.
