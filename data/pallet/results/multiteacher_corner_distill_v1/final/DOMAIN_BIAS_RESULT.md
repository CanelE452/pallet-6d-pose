# GATE E — 합성/실제 도메인 편향 진단

R0 의 동결 backbone/neck 3 level feature 를 global average pooling 하고,
SOURCE_DEV(합성 val) 500 장과 TARGET_UNLABELED 500 장을 로지스틱 회귀로 가른다.
classifier 는 **PAPER_EVAL 을 보지 않는다** — 그 위에 점수를 얹기만 한다.

## 판정

```
TARGET_BIAS_SIGNAL = DOMAIN_SEPARABLE_BUT_NOT_ERROR_LINKED
TARGET_ADAPTER     = NOT_RUN
```

## 수치

```
feature level     차원     domain AUROC (5-fold)
────────────────────────────────────────────────
level0              64     1.0000   [0.9998, 1.0, 1.0, 1.0, 1.0]
level1             128     0.9999
level2             256     0.9999
```

도메인은 **완벽하게 분리된다.** 세 level 어디서든 합성과 실제를 거의 오차 없이 가른다.

```
DEV_EVAL 319 프레임에서 domain score 와 오차의 연관
────────────────────────────────────────────────────────────
spearman(score, keypoint median)     -0.0238   (p = 0.672)
gross 프레임 분리 AUC                  0.4820   (무작위 0.5)
gross 프레임 비율                      0.4389
```

연관은 **없다.** 상관계수는 0 과 구분되지 않고, gross 프레임 분리 AUC 는 0.482 로
동전던지기보다 오히려 조금 아래다.

## 그래서 adapter 를 돌리지 않았다

사전등록 조건은 `domain AUROC >= 0.85` **그리고** `gross 분리 AUC >= 0.65` 였다.
앞은 압도적으로 넘고 뒤는 못 넘는다. METHOD_LOCK 규칙에 따라 E0 AdaBN 도 E1 residual
adapter 도 실행하지 않았다.

이건 "adapter 가 실패했다" 가 아니라 **"고칠 대상이 잘못 지목됐다"** 는 뜻이다.
feature 는 "이건 합성이고 저건 실제다" 를 큰 소리로 말하지만, 그 축은 모델이 어디서
틀리는지와 무관하다. 그 축을 지우거나 정렬해도 keypoint 실패는 그대로일 것이다 `[추정]`.

이 결과는 Gate A·B·D 와 같은 방향을 가리킨다 — 남은 실패는 도메인 외형의 문제가 아니라
**어느 코너가 어느 코너인지에 대한 구조적 오배정**이다.
