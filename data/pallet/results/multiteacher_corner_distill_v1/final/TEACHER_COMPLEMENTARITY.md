# GATE A — 여러 교사 상보성 감사

학습 0 · 새 checkpoint 0 · 새 지표 0.
모집단은 DEV_EVAL 319 (supervised 코너 2,499 · 그중 visible 1,594).
추론 recipe 는 `INFERENCE_REPLAY_LOCK` 그대로이고, 기준 모델(R0) 재추론이
동결 캐시를 **비트 단위로 재현**했다 (box/conf/keypoint max|Δ| = 0).

## 판정

```
MULTI_TEACHER_HEADROOM = STRONG
```

사전등록 임계 두 개를 모두 넘었다.

```
기준                                          임계      실측      통과
──────────────────────────────────────────────────────────────────────
oracle 이 최고 단독 교사 대비 visible p90 개선   >= 15%    69.6%     예
   (또는 gross20 상대감소 >= 20%)              >= 20%    59.8%     예
R0 의 gross20 코너 중 다른 교사가 <=10px 인 비율 >= 20%    30.3%     예
```

## 1. 교사 단독 성능 — 기준 모델이 모든 축에서 최고다

visible 코너 기준.

```
teacher                   축                          검출   median     p90   gross20  gross40
──────────────────────────────────────────────────────────────────────────────────────────────
T0 R0  YOLO26n g38+legacy REFERENCE                  1.000     6.36   43.89    0.157    0.102
T4 YOLO26n g38-only 60ep  TRAINING_SOURCE_COVERAGE   0.994     6.71   68.85    0.180    0.123
T5 YOLO26n broad40k       TRAINING_SOURCE_COVERAGE   1.000     6.62  133.84    0.206    0.154
T6 YOLO26n g38-only 30ep  ARCHITECTURE_CONTROL       1.000     7.19  135.89    0.231    0.155
T1 YOLOv8n g38-only       ARCHITECTURE               0.994     8.10  108.69    0.219    0.162
T2 YOLO11n g38-only       ARCHITECTURE               1.000     9.25  109.82    0.257    0.166
T3 DOPE heatmap           REPRESENTATION             0.870    13.54   73.38    0.283    0.151
```

교사를 단독 성능으로 고르면 전부 탈락한다. 그런데도 제외하지 않은 이유는
이 트랙의 질문이 "누가 더 정확한가" 가 아니라 **"R0 가 틀린 자리에서 다른 모델이 맞는가"** 이기 때문이다.

## 2. 조건부 구조 — 그렇다, 다른 모델이 맞는다

R0 가 20px 넘게 틀린 코너 498 개에서:

```
teacher                     n     <=10px 로 구제   <=20px 로 구제
──────────────────────────────────────────────────────────────────
T1 YOLOv8n                496          0.087           0.220
T2 YOLO11n                498          0.088           0.163
T3 DOPE heatmap           383          0.094           0.305   ← 단독 최약체가 조건부 최강
T4 YOLO26n g38-only 60ep  494          0.093           0.190
T5 YOLO26n broad40k       498          0.094           0.213
T6 YOLO26n g38-only 30ep  498          0.080           0.241
어느 하나라도                498          0.303           0.578
```

교사 하나당 8~9% 이고 합집합이 30.3% 다. 완전 독립이면 1-(1-0.09)^6 = 0.43 이 나오므로
**부분적으로 독립적인 오차**다. 같은 방향으로만 틀리는 것이 아니다.

가장 눈에 띄는 것은 T3(DOPE)다. 단독으로는 median 13.5px 로 제일 나쁜데,
`<=20px 구제` 에서는 0.305 로 제일 높다. representation 축이 실제로 다른 실패를 한다는 뜻이다.

## 3. oracle 상한 — 교사 수로 부풀린 값이 아니다

oracle 상한은 교사 수에 단조 증가한다. 그래서 중첩해서 본다 (visible 코너).

```
K   median     p90   gross20   그 단계에서 추가된 교사
──────────────────────────────────────────────────────────────
1     6.36   43.89     0.157   T0 R0 (기준)
2     5.17   23.73     0.119   T4 YOLO26n g38-only 60ep
3     4.91   17.91     0.090   T3 DOPE heatmap
4     4.43   15.18     0.076   T1 YOLOv8n
5     4.09   14.56     0.070   T2 YOLO11n
6     3.64   14.13     0.069   T5 broad40k
7     3.44   13.34     0.063   T6 g38-only 30ep
```

교사를 **하나만** 더해도(K=2) p90 −45.9%, gross20 −24.2% 로 두 임계를 이미 넘는다.
따라서 STRONG 판정은 admit 개수의 산물이 아니다.

## 4. ★ 그런데 GT 없는 융합은 R0 보다 나쁘다

이게 이번 게이트에서 제일 중요한 결과다. visible 코너.

```
arm                              median     p90   gross20  gross40
────────────────────────────────────────────────────────────────────
F0  R0 단독                         6.36   43.89    0.157    0.102
F1  좌표 성분별 median                6.31   72.66    0.173    0.129
F2  geometric medoid                6.36   69.91    0.173    0.129
F3  불확실성 가중 평균                 BLOCKED_NOT_COMPARABLE
ORACLE (배포 불가 상한)               3.44   13.34    0.063    0.040
```

F1·F2 는 median 을 거의 그대로 두면서 **꼬리를 크게 악화시킨다** (p90 43.9 → 70~73).
이유는 2번 표와 3번 표를 겹쳐 보면 분명하다 — 다른 교사들은 평균적으로 훨씬 나쁘고,
7개 좌표의 median/medoid 는 그 나쁜 예측 쪽으로 끌려간다.
정답이 후보 안에 **있지만**, 파라미터 없는 규칙으로는 **못 고른다**.

F3 는 측정하지 않았다. `SIGMA_STATUS = DIAGNOSTIC_ONLY` — Pose26 의 `kpts_sigma` 는
`self.training` 일 때만 생성되고 `fuse()` 가 head 를 None 으로 지운다. 값은 sigmoid 로
(0,1) 상한이고 단위가 anchor scale 의존 grid unit 이라 교사 간 comparable 하지 않으며,
DOPE 에는 sigma 자체가 없다.

## 5. 그러나 불일치는 "어디가 틀렸는지" 를 안다

R0 코너가 20px 넘게 틀렸는지를 예측하는 AUC (n=2,499, 양성률 0.199).
가중치를 fit 하지 않았고 rank average 도 정규화 없이 단순 평균이다.

```
신호                          AUC
────────────────────────────────────
teacher spread trace         0.7929
simple rank average          0.7990
max disagreement             0.7875
median pair disagreement     0.7798
R0 자신의 keypoint confidence  0.6582
```

교사 불일치는 R0 자신의 confidence(0.658)보다 **확실히 잘** 오류를 짚는다.
"무엇이 맞는지" 는 못 골라도 "여기는 믿지 마라" 는 안다.

## 이 게이트가 하류에 넘기는 것

```
Gate C / D 의 multi-teacher 학습    허용 (headroom STRONG)
융합 좌표를 그대로 target 으로 쓰기   금지 — F1/F2 가 R0 보다 나쁘다
사전등록된 abstention 설계          살아 있다 (METHOD_LOCK gate_c.real_consensus_gate,
                                   gate_d 의 usable/abstain). 불일치 AUC 0.79 가 그 근거다
```

Gate D 의 offline target quality 감사에서 **융합 target 이 R0 보다 나쁘면 student 학습을
하지 않는다** 는 규칙이 이미 잠겨 있다. 위 F1/F2 결과는 그 규칙이 실제로 발동할 수 있는
상황임을 미리 알려준다.
