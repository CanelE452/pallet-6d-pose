# V22 사전 대조 설계

```json
{
  "arms": [
    "unit_bce",
    "unit_pixel",
    "unit_nocontext_bce",
    "unit_nocontext_pixel",
    "unit_pixel_local",
    "unit_nocontext_pixel_local"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "lr": 0.0003,
  "weight_decay": 0.0001,
  "gradient_clip": 5,
  "float32": true,
  "updates": 36000,
  "hypotheses": [
    "Direct final-point Euclidean loss better weights attainable error reduction than movement-weighted alpha BCE.",
    "Removing frozen64D parent context discourages a synthetic contextual shortcut; current image/geometry still available."
  ],
  "factors": "BCE versus Euclidean final-point error x retain64D parent context versus zero64D. Two pixel-loss local controls remove only extra line evidence. Identical tensors, initialization, batches, learning rate, final checkpoint.",
  "objective": "BCE is original V20 unit_weighted. Pixel loss is mean raw-pixel Euclidean error over all original valid points. No per-type weighting, preservation penalty, GT filter, pseudo-labels, or self-training. Pixel gradient traverses sigmoid and may saturate; explicitly diagnose.",
  "context": "Only rep.context64D is zeroed. Frozen observability probabilities, amodal delta, point ID, original features, geometry and proposed direction/length are retained. This is not a complete removal of parent-derived information.",
  "architecture": "Frozen V18 vector d, ObjectiveGate derives from CauseGate, p=base+sigmoid(logit)*d. No new direction or end-to-end YOLO/DHT training.",
  "training": "Original253synthetic/1462detections/6conditions,2000updates per model, same original-image grouped plan as V20. Use original pool to isolate objective/context without new source composition change.",
  "evaluation": "Reused real297/9sessions/manual1251 and synth505DEV x8conditions, plus V21 new_check253 x6 now also reused. No untouched final holdout or new real-GT fitting.",
  "acceptance": "All3seeds clean andcrop10/20/30outside<=1.01base; cornerhidden/crop40outside retain>=90% V18 full-vector gain. Pairwise everyseed>=1% and real9session95%CI upper<0; no multiple comparison correction.",
  "interventions": "no_context and no_extra_lines probe are OOD sensitivity only; trained 2x2 required for causal comparison.",
  "probe_sha256": "e37e8a66368e03243f4d3d2b85507a9b6fe4a4b6017f85b2dd8efdc24f9afc56"
}
```

## 결과 확인 후 탐색적 추가

```json
{
  "source_sha256": "eb0be4c8d0fee71c02662c395dfb8632626da7af109b3bd5993bcf1a625af688",
  "model_sha256": "2880a6107874fb4e14e825782bf7f1e88dc71d7598e4f8a98be7e42557986cc3",
  "arms": [
    "unit_warm_bce",
    "unit_warm_pixel"
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
  "hypothesis": "Early full-retention saturation may prevent direct-distance learning of useful context. Start from already-trained BCE features/alpha, then compare another2000 BCE versus pixel updates.",
  "initialization": "Same V22 unit_bce final state per seed, fresh AdamW for both continuations, exactly original2000 grouped updates repeated. Same retained64D context and line samples. No real GT training or checkpoint selection.",
  "scope": "Exploratory after viewing initial V22 results; six new models12000updates. Not preregistered before initial results or independent final holdout. Reset optimizer in both controls."
}
```
