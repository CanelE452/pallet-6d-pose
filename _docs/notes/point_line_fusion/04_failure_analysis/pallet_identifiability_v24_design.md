# V24 사전 설계

{
  "seeds": [
    1,
    2,
    3
  ],
  "arms": [
    "unit_sign",
    "unit_sign_local",
    "unit_sign_parent"
  ],
  "tiny_arms": [
    "unit_tiny",
    "unit_tiny_local"
  ],
  "hypothesis": [
    "Can unchanged gate memorize accurate-point preservation on12intact synthetic originals?",
    "Can current point/line features predict whether fixed proposal has positive dot product with GT residual, and does it transfer to real?"
  ],
  "tiny": {
    "train_originals": [
      "G38__G__f10263",
      "G38__G__f22876",
      "G38__G__f10492",
      "P0__shard_06_f0451",
      "G38__G__f9669",
      "G38__G__f19694",
      "G38__G__f16381",
      "G38__G__f34568",
      "G38__G__f18598",
      "G38__G__f7262",
      "P0__shard_06_f1102",
      "G38__G__f7327"
    ],
    "check_originals": [
      "G38__G__f16451",
      "G38__G__f15616",
      "G38__G__f35610",
      "P0__shard_06_f0531",
      "P0__shard_06_f1171",
      "G38__G__f36584",
      "G38__G__f7915",
      "G38__G__f28329",
      "G38__G__f34416",
      "G38__G__f30540",
      "G38__G__f29731",
      "G38__G__f31834",
      "G38__G__f15210",
      "G38__G__f8396",
      "G38__G__f33230",
      "P0__shard_00_f1036",
      "P0__shard_03_f0236",
      "G38__G__f35384",
      "P0__shard_00_f0851",
      "G38__G__f19549",
      "G38__G__f21760",
      "P0__shard_04_f0095",
      "G38__G__f8365",
      "P0__shard_01_f0030",
      "G38__G__f28422",
      "G38__G__f25808",
      "G38__G__f35569",
      "G38__G__f8465",
      "G38__G__f13542",
      "G38__G__f39644",
      "G38__G__f21410",
      "G38__G__f18398",
      "G38__G__f26158",
      "G38__G__f2931",
      "P0__shard_01_f0822",
      "G38__G__f20780",
      "G38__G__f34508",
      "G38__G__f37589",
      "G38__G__f31421",
      "G38__G__f26857",
      "G38__G__f28410",
      "P0__shard_06_f0082",
      "G38__G__f3222",
      "G38__G__f16364",
      "P0__shard_06_f1156",
      "G38__G__f38599",
      "G38__G__f17128",
      "P0__shard_07_f0607"
    ],
    "conditions": [
      "clean",
      "occlude20",
      "occlude40",
      "crop20",
      "crop40",
      "corner20"
    ],
    "steps": 6000,
    "batch": "3original groups and all detected conditions; cycle seeded permutations",
    "objective": "Original movement-weighted BCE on continuous oracle alpha; same initialization, batches, optimizer for line/local.",
    "positive_control": "Per-observation/per-point free trainable logits, same continuous target,2000Adam updates; explicitly privileged GT-fitted lookup, no transfer claim.",
    "acceptance": "All seeds: final TRAIN point error minus GT-optimal scalar error <=.25px and accurate(base<=5px) train mean <=base mean+.1px. Compare early2000 vs final6000. Tiny check48 originals never used by tiny gate; already part of V23/full sign train, not independent pipeline holdout."
  },
  "direction": {
    "steps": 2000,
    "batches": "Exactly V23 expanded764original/4409observation plans",
    "target": "1 iff dot(GT-base,proposal)>0; exclude degenerate residual/proposal norms<=.1px from classification target only, never from geometric metrics.",
    "loss": "Binary BCE weighted to equal total positive/negative coefficient on synthetic TRAIN only. No real target fitting, tuning or temperature calibration.",
    "arms": "Full line; remove extra line tokens; parent geometry control zeroes current192D feature map and64D context but retains geometry, probability, amodal, line coordinates and content mask. Thus parent control is not wholly independent of images.",
    "downstream": "Exploratory unchanged p=base+sigmoid(logit)*d with no threshold/filter. Sign probability is not optimal step size or calibrated probability; direction classification can succeed without geometric improvement.",
    "signal_acceptance": "Real crop20outside nondegenerate AUROC>=.70 allseeds and9session95%CI lower>.5; extra-line increment allseeds AUROC gain>=.02 and paired95%CI lower>0. Thresholds fixed before training; no tuned decisions."
  },
  "optimizer": {
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "gradient_clip": 5,
    "float32": true
  },
  "budget": "6tiny x6000 +9direction x2000=54000neural updates;3privileged lookup controls x2000=6000additional tiny parameter updates.",
  "evaluation": "All V23 real297/synth505/V21check253/V23check128 now reusedDEV, six or eight conditions. No new real holdout or new natural occlusion labels.",
  "stability": "Every real seed clean/crop10/20/30outside<=1.01base; cornerhidden/crop40outside retain>=90%V18 full-vector gain.",
  "scope": "Frozen parent proposal and trainable gate, no YOLO/DHT end-to-end, no self-training or inference GT. Failure of these probes is not information-theoretic impossibility.",
  "parent_completion_sha256": "f9342f37f6f6c237f687583201864ae04f9452b9b242b6f9c6bb24b5aac66303"
}