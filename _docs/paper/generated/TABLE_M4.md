# Table M4 — Pseudo-label filter quality

population `PAPER_EVAL_PLASTIC_POS`  N=194  detected=194  CORRECT_2D=105

CORRECT_2D = detected AND no supervised keypoint error > 20 px  (gross 20.0 px, metric_split_lock.md §2.2 [LOCKED])

```text
Filter                                        Pass   Ret.  Pass~px   Rej~px    Sep↑  Gross↓  Prec↑   Rec↑    F1↑
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
No filter                                      194  1.000     7.32        —       —   0.198  0.541  1.000  0.702
Confidence                                     150  0.773     6.56    12.79    6.23   0.139  0.593  0.848  0.698
Confidence + Reprojection                      143  0.737     6.31    12.82    6.51   0.127  0.622  0.848  0.718
Confidence + Keypoint-removal consistency      149  0.768     6.58    11.95    5.37   0.139  0.597  0.848  0.701
Confidence + Horizontal-flip consistency       143  0.737     6.46    11.47    5.01   0.132  0.594  0.810  0.685
Proposed                                       142  0.732     6.49    11.44    4.95   0.132  0.599  0.810  0.688
```

## Confidence bin 진단

"0.7~0.8 을 넘으면 실제로 더 맞는가" 에 답한다.

`src` 는 keypoint 통계 출처다.  `strict` 는 evaluator 의 supervision mask,
`diag` 는 all-annotated (visibility 무시, 좌표가 있는 점 전부).
저신뢰 bin 은 전부 legacy 프레임이라 strict 가 비어 diag 로 채웠다.
**두 출처의 절대값을 직접 비교하지 않는다.**

```text
box_conf bin         N     src   n_kp  corner~px       p90    gross
──────────────────────────────────────────────────────────────────
[0.00,0.70)         33  strict    255      18.82    129.68    0.482
[0.70,0.80)          7  strict     54       5.77    203.15    0.222
[0.80,0.90)         18  strict    145       6.84     41.97    0.241
[0.90,1.01)        136  strict   1174       6.50     23.61    0.130
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
