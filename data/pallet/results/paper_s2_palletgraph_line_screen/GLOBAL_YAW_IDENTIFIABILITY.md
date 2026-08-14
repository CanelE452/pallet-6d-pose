# G0 — Global Yaw Identifiability (UPPER BOUND)

> **UPPER BOUND**: translation·roll·pitch 를 GT 로 고정하고 yaw 만 [0°,180°) 전역 탐색했다.
> line map 은 GT pose 로 그린 **oracle** 이며 semantic class 라벨도 GT 다.  inference 결과가 아니다.


탐색: 2.0° coarse over [0,180) -> top-5 주변 ±3° 를 0.25° 로 refine.  180° 대칭 적용.


## Gate
```
기준: overall top3<=5° >= 0.80 / point-fail >= 0.60 / truncated >= 0.60
판정: PASS  (G0-LV 또는 G0-LO 기준)
```


## Arm x slice
```
arm      slice                   n  top3<=5°  top1<=5°  top1_err_med  gt_rank_med
──────────────────────────────────────────────────────────────────────────────────
G0-LA    overall                87     1.000     1.000         0.000          1.0
G0-LA    point_fail             17     1.000     1.000         0.000          1.0
G0-LA    point_success          70     1.000     1.000         0.000          1.0
G0-LA    truncated              17     1.000     1.000         0.000          1.0
G0-LA    non_truncated          70     1.000     1.000         0.000          1.0
G0-LA    close_range            22     1.000     1.000         0.000          1.0
G0-LA    F1_NO_RESPONSE         24     1.000     1.000         0.000          1.0
G0-LA    F2_CONFIDENT_WRONG     35     1.000     1.000         0.000          1.0
G0-LA    outside                44     1.000     1.000         0.000          1.0
G0-LA    night                  43     1.000     1.000         0.000          1.0
G0-LO    overall                87     1.000     1.000         0.000          1.0
G0-LO    point_fail             17     1.000     1.000         0.250          1.0
G0-LO    point_success          70     1.000     1.000         0.000          1.0
G0-LO    truncated              17     1.000     1.000         0.250          1.0
G0-LO    non_truncated          70     1.000     1.000         0.000          1.0
G0-LO    close_range            22     1.000     1.000         0.250          1.0
G0-LO    F1_NO_RESPONSE         24     1.000     1.000         0.250          1.0
G0-LO    F2_CONFIDENT_WRONG     35     1.000     1.000         0.000          1.0
G0-LO    outside                44     1.000     1.000         0.000          1.0
G0-LO    night                  43     1.000     1.000         0.250          1.0
G0-LV    overall                87     1.000     1.000         0.000          1.0
G0-LV    point_fail             17     1.000     1.000         0.000          1.0
G0-LV    point_success          70     1.000     1.000         0.000          1.0
G0-LV    truncated              17     1.000     1.000         0.000          1.0
G0-LV    non_truncated          70     1.000     1.000         0.000          1.0
G0-LV    close_range            22     1.000     1.000         0.000          1.0
G0-LV    F1_NO_RESPONSE         24     1.000     1.000         0.000          1.0
G0-LV    F2_CONFIDENT_WRONG     35     1.000     1.000         0.000          1.0
G0-LV    outside                44     1.000     1.000         0.000          1.0
G0-LV    night                  43     1.000     1.000         0.000          1.0
G0-P     overall                87     0.678     0.678         3.000          1.0
G0-P     point_fail             17     0.000     0.000        23.000          1.0
G0-P     point_success          70     0.843     0.843         2.375          1.0
G0-P     truncated              17     0.353     0.353        23.000          1.0
G0-P     non_truncated          70     0.757     0.757         3.000          1.0
G0-P     close_range            22     0.500     0.500         4.625          1.0
G0-P     F1_NO_RESPONSE         24     0.208     0.208        23.000          1.0
G0-P     F2_CONFIDENT_WRONG     35     0.800     0.800         3.000          1.0
G0-P     outside                44     0.705     0.705         3.000          1.0
G0-P     night                  43     0.651     0.651         3.000          1.0
G0-PL    overall                87     0.931     0.897         1.000          1.0
G0-PL    point_fail             17     0.882     0.882         0.250          1.0
G0-PL    point_success          70     0.943     0.900         1.250          1.0
G0-PL    truncated              17     1.000     1.000         0.250          1.0
G0-PL    non_truncated          70     0.914     0.871         1.375          1.0
G0-PL    close_range            22     1.000     1.000         0.625          1.0
G0-PL    F1_NO_RESPONSE         24     0.917     0.917         0.500          1.0
G0-PL    F2_CONFIDENT_WRONG     35     0.914     0.857         1.250          1.0
G0-PL    outside                44     0.909     0.886         1.000          1.0
G0-PL    night                  43     0.953     0.907         1.250          1.0
```


## 해석

- [확인] oracle line arm 3종(LA/LV/LO)은 **overall·point-fail·truncated 전부 1.000**, top-1 yaw 오차 중앙값 0.0°.
- [확인] 특히 **G0-LO** 는 실제 image gradient support 가 있는 구간만 남긴 fragment 인데도 100% 다 —
  '영상에 실제로 남아 있는 line 만으로도 yaw 가 식별된다' 는 뜻이다.
- [확인] point-only(G0-P)는 point-fail 프레임에서 **0.000** 이다 (초기 pose 자체가 없으므로 당연).
- [주의] oracle 은 semantic class 라벨(width/depth/vertical)이 GT 에서 온다.  learned line head 는
  이 라벨을 스스로 예측해야 하므로, 이 100% 는 달성 가능한 **상한**이지 예상 성능이 아니다.

