# Paper-S1 pick overlays — 도메인별 검출/오차 요약

weights: `/home/minjae/Documents/github/pallet-pose/weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth`  
전처리: reflect-pad100 (official 아님)  
corner_med = per-frame order-free Hungarian vs GT8 (px)  
★ challenge/palletobj 는 S1(paper-track)엔 unseen → 도메인갭 존재  
★ N 소표본(도메인당 6~44) → 과결론 금지

```
domain     N  det>=6  no-det  det_mean  cm_med  cm<10px
────────────────────────────────────────────────────────────
noapril    6       5       1      6.67     9.0      3/5
outside   44      38       6      7.27    17.3     6/38
cad       22       4      18      3.64    13.7      0/4
────────────────────────────────────────────────────────────
TOTAL     72      47
```

컬럼: det>=6=검출성공 프레임수, no-det=미검출(<6)프레임수, det_mean=평균 검출 코너수(/8), cm_med=corner_med 중앙값, cm<10px=corner_med<10px 프레임/유효프레임.
