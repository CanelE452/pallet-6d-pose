# Table E — what the filters actually keep (label quality, not student)

population   PAPER_EVAL_PLASTIC_POS, 194 frames (these HAVE manual GT)
criterion    detected AND no supervised keypoint error > 20 px
source       data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json

★ 이 표는 **라벨 품질**이다. 학생 모델 성능이 아니다. 둘을 같은 표에 섞지 않는다.

```text
filter                    kept  retain  pass med px ↓  reject med px ↑  precision ↑  recall ↑
---------------------------------------------------------------------------------------------
F0 no filter               194   1.000          7.689                —       0.5258    1.0000
F1 confidence              150   0.773          6.615           13.780       0.5867    0.8627
F2 + reprojection          143   0.737          6.388           14.080       0.6154    0.8627
F3 + kp removal            149   0.768          6.633           13.063       0.5906    0.8627
F5 + flip                  143   0.737          6.516           12.880       0.5874    0.8235
F4 proposed (all)          142   0.732          6.551           12.811       0.5915    0.8235
```

How to read it in the meeting

- The filters do work at the label level: the proposed filter raises the
  kept-label median error and separates pass from reject.
- That improvement does not appear in the student (Table B) or in the
  downstream pose (Table A). That gap is the paper.

Paper role: **main table** (the label-quality half of the story).
