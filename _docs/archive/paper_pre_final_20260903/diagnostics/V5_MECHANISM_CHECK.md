# V5 mechanism check — 학습 전에 한 번만

고정된 `R_total` 이 가중치를 옳은 방향으로 주는지 본다.  score 는 이미
`RELIABILITY_SCORE_LOCK.json` 으로 동결됐고 이 결과를 보고 바꾸지 않는다.

모집단 PAPER_EVAL positive, daytime+nighttime, box_conf >= 0.85, n=79, frame gross 51.9%

## 분리력

```text
signal             AUC
----------------------
R_total          0.763
box_conf         0.653
s_reproj         0.745
s_remove         0.694
s_flip           0.583
```

## 학생이 기대상 보게 되는 라벨 품질

노출 횟수를 가중치로 한 기대값이다 — 같은 프레임이 3 번 나오면 3 배로 센다.

```text
metric              uniform V3-B  weighted V5        변화
--------------------------------------------------------
frame_gross               0.5190       0.4615   -0.0574
corner_gross              0.2078       0.1823   -0.0255
median_error_px          20.0479      18.2491   -1.7988
p90_error_px             31.4206      28.7036   -2.7170
```

## Gate

```text
PASS  M1_score_discriminates: AUC(R_total) = 0.763
PASS  M2_corner_gross_improves: uniform 0.2078 -> weighted 0.1823
PASS  M3_frame_gross_improves: uniform 0.5190 -> weighted 0.4615
```

**PASS** — `OK`

> pool 프레임에는 GT 가 없다.  여기 수치는 같은 teacher·같은 규칙을 GT 가 있는 PAPER_EVAL 에 적용한 대리 측정이다.

