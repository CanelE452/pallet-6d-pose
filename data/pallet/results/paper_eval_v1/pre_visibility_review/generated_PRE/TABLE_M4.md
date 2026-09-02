# Table M4 — Pseudo-label filter quality

population `PAPER_EVAL_PLASTIC_POS`  N=194  detected=194  CORRECT_2D=54

CORRECT_2D = detected AND no supervised keypoint error > 20 px  (gross 20.0 px, metric_split_lock.md §2.2 [LOCKED])

```text
Filter                                        Pass   Ret.  Pass~px   Rej~px    Sep↑  Gross↓  Prec↑   Rec↑    F1↑
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
No filter                                      194  1.000     4.28        —       —   0.103  0.278  1.000  0.435
Confidence                                     150  0.773     4.24     4.70    0.47   0.099  0.333  0.926  0.490
Confidence + Reprojection                      143  0.737     4.24     4.70    0.47   0.099  0.350  0.926  0.508
Confidence + Keypoint-removal consistency      149  0.768     4.26     4.53    0.27   0.099  0.336  0.926  0.493
Confidence + Horizontal-flip consistency       143  0.737     4.08     6.10    2.02   0.073  0.343  0.907  0.497
Proposed                                       142  0.732     4.10     5.88    1.78   0.072  0.345  0.907  0.500
```

## Confidence bin 진단

"0.7~0.8 을 넘으면 실제로 더 맞는가" 에 답한다.

`src` 는 keypoint 통계 출처다.  `strict` 는 evaluator 의 supervision mask,
`diag` 는 all-annotated (visibility 가 unknown 인 legacy 점 포함).
저신뢰 bin 은 전부 legacy 프레임이라 strict 가 비어 diag 로 채웠다.
**두 출처의 절대값을 직접 비교하지 않는다.**

```text
box_conf bin         N     src   n_kp  corner~px       p90    gross
──────────────────────────────────────────────────────────────────
[0.00,0.70)         33    diag    293      18.82    137.62    0.485
[0.70,0.80)          7  strict     25       4.59      7.93    0.000
[0.80,0.90)         18  strict     50       3.99    139.19    0.120
[0.90,1.01)        136  strict    499       4.30     25.62    0.106
```

confidence 가 TAU_BOX 아래인 검출은 눈에 띄게 나쁘다 — corner 가 한 자릿수에서
두 자릿수로 뛰고 gross rate 도 몇 배가 된다.  confidence pre-filter 가 하는
일이 여기서 보인다.  다만 그 위 구간(0.70~1.00) 안에서는 단조 개선이 아니다.

## Pseudo-label funnel (unlabeled pool)

```text
total                            1000
detected                         954
candidate_min_valid_corners      926
confidence                       272
confidence_reprojection          251
confidence_keypoint_removal      267
confidence_flip                  263
proposed                         259
```

threshold 는 이 결과를 보기 전에 동결됐고, 보고 나서 바꾸지 않았다.
