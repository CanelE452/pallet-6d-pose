# CELL COVERAGE REPORT

bin 경계와 UNDER/OVER 판정 규칙은 결과를 보기 전에 고정했다. 판정은 절대 개수가 아니라 **BROAD 가 같은 cell 에 준 비중** 대비다 (모든 cell 을 같은 N 으로 만들지 않는다).

```
status                     ADEQUATE  INSUFFICIENT_N  NOT_IN_PRIOR  OVERCONCENTRATED
dataset_id                                                                         
BROAD_40K                        60               0             0                 0
CORNER_LA_Y15_30                  0               0             0                 4
CORNER_LA_Y30_PLUS                0               0             0                 4
EDGE_HARD_CLEAN_UNTOUCHED         8              51             0                 1
EDGE_HARD_TRUNC_DEV               0               4            11                 0
EDGE_HARD_TRUNC_TRAIN             0               0            15                 0
EDGE_HARD_TRUNC_UNTOUCHED         0               6             9                 0
```

## line-hard 영역

```
  ('<=3', 'full', '0.25-0.40')
  ('<=3', 'truncated', '<0.25')
  ('<=3', 'truncated', '0.25-0.40')
  ('<=3', 'truncated', '0.40-0.60')
  ('<=3', 'truncated', '0.60-0.85')
  ('<=3', 'truncated', '>=0.85')
```

이 cell 들의 BROAD 프레임 수는 전부 **0** 이다. 그래서 '`BROAD 대비 2배`' 조항은 분모가 0 이라 어떤 비율에서도 통과한다 — 비율을 가르지 못한다. 이 사실을 통과로 적지 않고 `clause_1_is_vacuous=True` 로 기록했다.
