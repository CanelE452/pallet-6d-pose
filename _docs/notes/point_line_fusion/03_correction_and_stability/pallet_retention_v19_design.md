# V19 적용률 아키텍처 설계

초기 계획과 실패 진단 후 추가한 BCE 대조군을 구분한다.

```json
{
  "question": "Can frozen successful V18 corrections retain hard-case gains while suppressing mild-crop direction errors using observed image/line context?",
  "arms": [
    "local_gate",
    "area_gate",
    "line_gate"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "batch": "Three original image IDs per step; include all available detected variants among6 training conditions (up to18observations). Same original-ID permutation for all arms; no condition token or GT at inference.",
  "proposal": "Frozen V18 line_vector seed-matched; points=base+sigmoid(logit)*immutable_delta. No reversed direction or extra vector, no test GT or heuristic gate.",
  "architecture": "Shared local3x3 patch+parent64context+pointID+observability+coordinates+amodal_delta+proposal_delta ->64query. local gate adds zero32token. area gate reads6x6 samples from box intersect raw image. line gate reads12samples from each top3 associated DHT line clipped to that same rectangle. Both context arms use36samples, identical24channel projection/32token MLP/query attention and same final scalar head. Canonical line signs; empty segments masked. Pixel support is not physical edge correctness or visibility through occluders.",
  "supervision": "Original synthetic GT only. alpha_star=clip(dot(GT-base,delta)/norm(delta)^2,0,1). Loss=mean(norm(delta)*abs(alpha-alpha_star)) over all original valid points, equal to coordinate distance to projected attainable GT. Zero vectors have zero consequential loss; no self-training/filter/discard. Fixed final checkpoint; no real fitting.",
  "controls": "Same parameter tensors/initialization (alpha=.5), same grouped batches/2000steps/AdamW3e-4 wd1e-4 float32 clip5. area versus line isolates line-guided context at same sample budget; local versus context tests added nonlocal evidence. All use the same frozen line-based parent lineage, not pure YOLO controls.",
  "acceptance": "All3seeds clean andcrop10/20/30 outside <=1.01 base; corner20hidden andcrop40outside retain>=90% seed-matched V18 line_vector gain over base. Context incremental effect requires all3seeds>=1% gain and real9session paired95%CI upper<0. Multiplicity uncorrected.",
  "population": "Training253synthetic/1462detected observations6conditions already seen by frozen V18, fixed parents. Eval reused real297/9sessions/1251manual corners and synth505/8conditions; crop10/30 evaluation-only. Original IDs/GT/misses. No independent new final holdout or physical manual-click reannotation.",
  "interventions": "Same checkpoint: zero additional context token, or replace only context feature source by next detected-image features retaining point patch/geometry/proposal. OOD diagnostic. Fixed alpha.5 control. No performance-based choice of arm/seed."
}
```

## 탐색적 손실 대조군

```json
{
  "stage": "Exploratory followup after original train/DEV results; not part of initial preregistration.",
  "hypothesis": "Movement-weighted L1 on sigmoid alpha has logit gradient proportional to alpha*(1-alpha), weak at wrong saturation. Soft-target BCE-with-logits gradient is alpha-alpha_star. Compare same architecture/initial/batches/budget to distinguish this optimizer/objective limitation from absent context information. Conditional median versus mean objective also changes; not an isolated proof of saturation causality.",
  "arms": [
    "local_bce",
    "area_bce",
    "line_bce"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "loss": "mean(detached proposal movement * binary_cross_entropy_with_logits(logit, synthetic GT alpha_star)) over original valid points; no clamp on logits, no filtering.",
  "selection": "No real GT fitting, seed/checkpoint selection or further parameter search. Same final checkpoint, acceptance and evaluation populations. All outcomes reported, initial runs kept.",
  "original_protocol_sha256": "9548bd9dada379c5aa9ff8a06a2e71d265150da95d0f857db8d074159c62f1e3",
  "source_sha256": "9ad1bd39713f37e1e8b4c70d31d26efd47b6841ccd5a19edfadefb1ff4d83540",
  "model_sha256": "72976f260a26f94a5b4123024651d8c0de08a9702c1170d4e3ae791775a85634",
  "context_sha256": "55327ab7de74ef56d2d5deb8b2655bebf8ee3b9c175d09ca5bb8c1dcab9b453a"
}
```
