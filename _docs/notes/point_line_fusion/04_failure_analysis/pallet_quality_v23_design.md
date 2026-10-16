# V23 설계와 학습 데이터 진단 후 추가 대조

{
  "initial": {
    "arms": [
      "unit_expand",
      "unit_quality",
      "unit_expand_local",
      "unit_quality_local"
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
    "hypothesis": "Intact image observations with genuinely accurate frozen parent points under mild cropping are underrepresented. Expanding independent synthetic source groups and balancing training-only quality bins may preserve accurate points without sacrificing hard errors.",
    "design": "All arms use original253 plus new512 image groups. 2x2 original movement-weighted BCE versus training-only crop20-outside quality weights x extra-line evidence versus local control. Same batches, initialization, 2000 final updates. Compare existing V22 unit_bce on old253 as historical data-expansion reference; that comparison changes data exposure.",
    "weights": "Only valid outside points in crop20: frozen-base error bins <=5px, (5,15]px, >15px. For each seed compute bin counts from TRAIN GT only. Inverse-frequency weights N/(3*n), clipped to [0.125,8], renormalized to mean1 over the targeted points. Other points weight1. All originals, hard points, GT and features retained. No GT edit, no inference filter or pseudo-label.",
    "input": "Image-level six conditions, rerun YOLO and all frozen parents. Parent inference does not receive GT. Fixed raw coordinates and IDs.",
    "population": "New512 train and128 prospective check groups disjoint from original YOLO train, original requested refinement all splits, and V21 both pools. Historical YOLO validation, not untouched whole-pipeline holdout or certified asset/scene independence.",
    "evaluation": "Original real297/synth505 and V21new_check253 reused DEV plus new128 synthetic check. Real GT evaluation only; no selection by evaluation results.",
    "acceptance": "Every real seed clean/crop10/20/30outside <=1.01base; cornerhidden/crop40outside retain>=90% V18 vector gain. Pairwise everyseed>=1% and real9session bootstrap95%CI upper<0. Fixed last checkpoint; report all arms.",
    "budget": "4arms x3seeds x2000=24000updates. No further loss sweep or PCGrad. Fixed frozen V18 direction; gate only adjusts its magnitude."
  },
  "training_probe_amendment": {
    "arms": [
      "unit_mass",
      "unit_mass_local"
    ],
    "seeds": [
      1,
      2,
      3
    ],
    "steps": 2000,
    "updates": 12000,
    "registered_before_V23_evaluation": true,
    "training_probe_sha256": "7bd0e281a04ca3011e28ea6aec4100998057637f2fdef1a5f30fe4a5a2b223e8",
    "reason": "Accurate points already28% of mild outside train points; count balancing only mildly changes weights, while movement-weighted BCE mass remains dominated by large corrections. Add synthetic-only bin coefficient-mass balance to distinguish counting from effective loss emphasis.",
    "weights": "Same fixed base-error bins and same target crop20outside. Per bin factor=sum movement/(3*bin movement), clamp[.125,8], renormalize so total weighted movement coefficient equals original. No negative/zero weights; other targets exactly1. Coefficient mass, not actual loss/gradient magnitude, is balanced.",
    "control": "Same intact old+new observations, parent predictions, initialization and2000group batches as unit_expand. Extra-line/local paired. Real and all evaluation GT unused for registration/weight fitting.",
    "source_sha256": "ae79a80144c51c2e7cb6886ce5757b9c4c3d83e4876eafc7517273bd5a6f5457"
  }
}