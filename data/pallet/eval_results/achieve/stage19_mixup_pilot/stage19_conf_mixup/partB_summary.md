# PART B — domain-mixup pilot (2-arm) verdict

**VERDICT = HONEST NEGATIVE + REGRESSION → soft pixel-interpolation folds.**

DM-ADA core (domain linear interpolation + soft PL supervision, NO adversarial
discriminator). A-PASS branch: FRONT confidence-selected PL (peak>=0.9 & flip<=10)
+ REAR spatial-masked. Both arms init from B2, identical sim data/order/schedule
(600 steps, lr 1e-4); only mixup adds ~50% real-blend samples.

## Paired eval (same-frame, both-detect n=80; filter-val outside+night + manual; SEALED)
```
arm       front_med  rear_med  reproj_med  good<10  gross>20
control     10.86     20.07      17.83       176      223
mixup       12.20     21.14      19.23       153      243     <- worse on ALL
V=8 control 10.86     19.90      18.12       172      215
V=8 mixup   12.20     21.10      19.23       150      233     <- same regression
```

- mixup degrades FRONT (+1.34px), REAR (+1.07px), honest full-8 reproj (+1.40px),
  good −23, gross +20. Full-view (V=8) shows the identical regression → not a
  truncation artifact.
- control ≈ B2 (front ~10.9 vs PART A B2 front 11.9) → control is a clean B2
  continuation; the regression is caused by the mixup blend, not the schedule.

## Interpretation (matches §5 suspicion)
- **Pixel ghosting hurts px-precise pose**: λ·x_sim+(1-λ)·x_real overlays two
  pallets; the faint real pallet acts as structured noise that shifts belief
  peaks → precision loss. Predicted a-priori as a risk; confirmed.
- **Appearance mixup does NOT close the rear depth-cue gap** (PART A finding:
  rear is confidently-wrong = a geometry/depth problem, not appearance). Rear got
  worse, not better.
- Sparse, noisy front-PL (only 36 real frames, 1–2 accepted corners each) gives
  little upside while the blend adds noise everywhere.

## Caveats (small-sample, honest)
- Real GT paired n=80; front-PL source 36 frames / 1–2 corners → pilot signal.
- Short pilot (600 steps). But the comparison is paired + same-schedule + same
  sim data, so the *relative* regression is a fair read.

## Conclusion for the thesis
Naive soft domain-mixup is the wrong lever here — consistent with the master
thesis (suppression + structural + co-primary dataset, NOT naive PL/mixup).
Do NOT promote to full training. The productive levers remain: (1) rear/low-angle
flat-view DATA (STAGE17), (2) front-only confidence PL used as *suppression*
signal rather than pixel-mixup supervision.
