# V21 사전 설계

```json
{
  "arms": [
    "unit_old_sum",
    "unit_old_pcgrad",
    "unit_new_sum",
    "unit_new_pcgrad",
    "unit_new_sum_local"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "lr": 0.0003,
  "weight_decay": 0.0001,
  "clip": 5,
  "float32": true,
  "design": "2x2: old versus new naturally generated residuals x original gradient sum versus PCGrad; new sum local extra-line control. Same architecture, initialization and original-image grouped batches within population. No point jitter or new real GT training.",
  "population": "253 new training and253 prospective check images, unique pair_group_id disjoint from original YOLO train and every original refinement train/calibration/eval ID group. They come from YOLO val used in historical model selection, so not an untouched final holdout or certified scene-disjoint set.",
  "input": "Actual image-level clean/occlude20/40/crop20/40/corner20 observations. Rerun YOLO and all frozen parents with consistent features/context/base/delta. GT never selects detections or edits base/context.",
  "optimizer": "Original movement-weighted BCE decomposed into alpha_star<=.05 stop and remaining move terms, each divided by total valid-point count. Sum exactly original objective; PCGrad projects each conflicting group gradient off the ORIGINAL other group gradient and sums. No class balancing. If either group empty/zero, ordinary sum. This is stop/move group PCGrad, not point-vs-line multi-task PCGrad.",
  "source": "https://arxiv.org/abs/2001.06782 ; https://github.com/tianheyu927/PCGrad",
  "acceptance": "Every real seed clean andcrop10/20/30outside<=1.01base; cornerhidden andcrop40outside preserve>=90% V18 vector gain. Pairwise everyseed>=1% improvement AND9session95%CI upper<0. Fixed last checkpoint; report all arms.",
  "eval": "Unchanged real297/9sessions/1251manual corners and synth505DEV x8conditions. New check253 x6conditions is prospective for retention but historical YOLO val. No selection on any eval metrics.",
  "budget": "5arms x3seeds x2000=30000 updates; old sum rerun provides functional V20 reproduction. Same number originals and grouped draws across populations, detected observation counts can differ.",
  "stage": "Exploratory follow-up to V20; frozen full parent, no end-to-end retraining."
}
```
