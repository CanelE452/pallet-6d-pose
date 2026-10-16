# V20 설계와 단계별 계획

최종 구조: p = p0 + sigmoid(g(point context, line evidence, direction, length)) * frozen_delta.

초기 원인 분리 이후 두 후속 단계는 탐색적이다. 단계 사이 평가 결과를 확인했으며 독립 최종 holdout은 없다.

## PROBE_PROTOCOL.json

```json
{
  "scope": "Training-only source feature scales and target/movement mass. No final real evaluation for design selection.",
  "hypotheses": [
    "Movement weighting favors large helpful corrections.",
    "Feature block scale obscures geometric/proposal evidence.",
    "Point-wise representation lacks trainable cross-point context."
  ],
  "source_sha256": "bd0fd578e248ece8437aed2ff60389dac90e2394bc60c6b33f732d9efee69bf4"
}
```

## PROTOCOL.json

```json
{
  "stage": "New exploratory DEV experiment chosen using training-only V20 probe and earlier V19 findings; no independent final holdout.",
  "hypotheses": [
    "Movement weighting suppresses stop-target contribution.",
    "Tiny proposal coordinates are not encoded strongly enough relative to appearance/context."
  ],
  "probe_sha256": "aef29ba7f2ebe66ab6ab2b9bf19c3e3d83144e400db8b8a29d57ed15902b0b3a",
  "arms": [
    "raw_weighted",
    "raw_equal",
    "unit_weighted",
    "unit_equal",
    "unit_equal_area",
    "unit_equal_local"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "factors": "Raw versus unit-direction/radius-fraction encoding x movement-weighted versus equal-point BCE. All4 use same line context. unit_equal_area and unit_equal_local test extra line context against same sample-budget area and no extra context.",
  "encoding": "Original 228descriptor unchanged except amodal_delta/boxdiag scaled by1/.25, proposal d/boxdiag replaced with d/|d| (epsilon1e-8 normalized length), proposal length/boxdiag scaled by1/.15. Fixed .25/.15 are existing amodal/vector radius definitions, not GT-fitted scales. For nonzero d information is preserved; no new GT information.",
  "architecture": "Frozen V18 line_vector, same RetentionGate weights/tensors as V19. points=base+sigmoid(logit)*frozen_delta. New trainable head learns retained amount, not an external filter. No self-training or real GT forward.",
  "training": "253synthetic/1462detected variants/6conditions. Same original-ID grouped batches (3originals, up to18observations), initialization seed19000+seed and2000updates as V19 BCE. AdamW3e-4 wd1e-4 float32 clip5; last checkpoint. Every original valid point retained.",
  "loss": "BCE-with-logits against original synthetic GT projected alpha_star. Weight either proposal movement or1 per valid point. Equal weights alter relative emphasis; no benefit/harm GT selection or sampling.",
  "acceptance": "Everyseed clean andcrop10/20/30 outside <=1.01 base; corner20hidden andcrop40outside retain>=90% V18 line_vector benefit. Pairwise effect requires everyseed>=1% and real9session paired95%CI upper<0. Multiplicity uncorrected.",
  "evaluation": "Immutable real297/9sessions/1251manual corners + synth505,8conditions. Original IDs/detections/point GT; artificial stress. crop10/30 evaluation-only. All arms evaluated/reported, no real-driven model selection.",
  "interventions": "unit_equal actual versus zero line token and donor line-source features. Point patch/base/proposal/line geometry stay actual. OOD diagnostic, not proof of useful semantics.",
  "reproduction": "raw_weighted must reproduce V19 line_bce final state with same seed and batches before conclusions."
}
```

## RESUME_PROTOCOL.json

```json
{
  "source_sha256": "45921d3685c4c07da2002fba8fd7e6f0fbea5deb370f0679fc123be0d605bacf",
  "reason": "Parameter exactness check stopped at seed2: max1.1474e-6 versus1e-6 bound; training loss differences begin at float32 roundoff size ~1.19e-7. Completed checkpoints kept. Require output agreement on all synthetic training observations: <.001px and alpha<1e-4; check parameter difference<1e-4 as supporting evidence. No GT/performance threshold adjustment."
}
```

## CF_PROTOCOL.json

```json
{
  "stage": "Exploratory followup after initial2x2 results; not original preregistration.",
  "hypothesis": "One frozen proposal per point lets image-conditioned mean retention substitute for direction comparison. Present same image/base with original, reverse and two perpendicular directions, recompute alpha_star from existing synthetic GT; compare more original-only optimization.",
  "arms": [
    "unit_weighted_long",
    "unit_weighted_cf",
    "unit_weighted_cf_local"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "training": "CF2000updates with four copies of each original grouped batch: d,-d,rot90(d),rot-90(d). All GT and point IDs retained. Same movement lengths; weighted BCE averaged across copies. Long control8000updates using the same grouped batches cyclically four times. Exactly same source-observation forward count per arm, but update counts differ. Same init/optimizer/lr/fixed final checkpoint. CF local zeros new line token.",
  "inference": "Always original frozen V18 proposal; no augmented candidates, target or selection at inference. Learned scalar still only0..1. No GT filtering or selftraining.",
  "acceptance": "Everyseed clean andcrop10/20/30 outside <=1.01 base; corner20hidden andcrop40outside retain>=90% V18 line_vector benefit. Pairwise effect requires everyseed>=1% and real9session paired95%CI upper<0. Multiplicity uncorrected.",
  "evaluation": "Immutable real297/9sessions/1251manual corners + synth505,8conditions. Original IDs/detections/point GT; artificial stress. crop10/30 evaluation-only. All arms evaluated/reported, no real-driven model selection.",
  "source_sha256": "29e8790e21da27399165601bf85a3d40a3b8e3ef4b66c6c33f5b12e8b1ba5103",
  "model_sha256": "9b609c9c184b31311d773e2dbeea95561a4cc338385f4a12cbd2429c3650ff04",
  "initial_protocol_sha256": "f6f18477649a765e96488921c8e45af9fc89495e029538cc8517b49ea39b9a64"
}
```

## QUALITY_PROTOCOL.json

```json
{
  "stage": "Exploratory followup after point-ID matched residual distribution diagnosis; not confirmatory holdout.",
  "hypothesis": "Synthetic baseline errors are much larger than real mild-crop errors. Include already correct and slightly wrong synthetic base points so the retention model learns no-op cases at appropriate query positions.",
  "augmentation": "Four views of each synthetic observation: original base, GT base, GT+.01*boxdiag*unit_random_direction, GT+.03*boxdiag*unit_random_direction. Fractions fixed to span correct/near-correct points, not fitted to real optimal scales. Directions fixed per seed/observation/point. Only synthetic valid points are changed. Frozen V18 proposal is recomputed for each changed base. Original features/global parent context/amodal output/line associations remain frozen; this is counterfactual base augmentation, not a claim to reproduce the entire parent distribution.",
  "training": "2000updates, each original grouped batch expanded4times; same original-ID plan/seed/optimizer as previous. Movement-weighted BCE from original synthetic GT, all original valid points kept. Same observation-forward count as CF and8000step original-only long control; optimizer step counts differ versus long. local zeros extra line token.",
  "arms": [
    "unit_weighted_quality",
    "unit_weighted_quality_local"
  ],
  "seeds": [
    1,
    2,
    3
  ],
  "steps": 2000,
  "source_sha256": "adb8d90f8d5b23d43a2d2ed910e288c64d28091830d82e47f46d1ec0354d5ab1",
  "model_sha256": "9b609c9c184b31311d773e2dbeea95561a4cc338385f4a12cbd2429c3650ff04",
  "diagnosis_sha256": "08f9f1f49a18c74a5246aca44470ac3bdb13fe4f92fe9e7b1e9c1a1f07746f88",
  "acceptance": "Everyseed clean andcrop10/20/30 outside <=1.01 base; corner20hidden andcrop40outside retain>=90% V18 line_vector benefit. Pairwise effect requires everyseed>=1% and real9session paired95%CI upper<0. Multiplicity uncorrected.",
  "evaluation": "Immutable real297/9sessions/1251manual corners + synth505,8conditions. Original IDs/detections/point GT; artificial stress. crop10/30 evaluation-only. All arms evaluated/reported, no real-driven model selection.",
  "inference": "Only original frozen parent/V18 outputs, no GT, synthetic perturbation or fallback rule at inference. All models evaluated, no real checkpoint selection."
}
```
