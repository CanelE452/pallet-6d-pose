# Explicit point–line decoder pilot

Updated 2026-09-09: both 1000-update runs and evaluation are complete. The prespecified exploratory continuation criteria were not met. The independent audit, actual visual review, browser opening and requested Discord delivery are also complete; [COMPLETION.json](../../../data/pallet/results/pallet_dht_decoder_probe_v1/COMPLETION.json) records execution integrity, not an accuracy-improvement PASS.

This experiment checks a small learned output module on top of a frozen, previously trained pallet Deep Hough network. It is separate from the completed joint-training and gradient-coupling experiments. It uses synthetic GT for decoder training and the existing 319-image real DEV population for exploratory 2D evaluation. It is not self-training or an acceptance filter.

The same 55,118-parameter module is trained with either point-derived candidates or semantic-line intersection candidates. Both controls can use the original predicted points to correct their spatial placement; the point-only candidate bank also permits ID reassignment. The line branch uses a direct learned coordinate mixture, a signed gate, and a small residual. Centroid point8 is retained. The frozen candidate extraction is discrete; this pilot does not test gradients into the DHT/backbone.

The fixed protocol, source manifests, smoke receipts, measured counts and all results live in `data/pallet/results/pallet_dht_decoder_probe_v1/`. The [experiment note](../../../_docs/notes/pallet_dht_decoder_probe.md) states the hypothesis, metrics, failure modes and decision criteria before results. No bound v1/v2 source or result is overwritten.

Pipeline (after the synthetic preflight and source binding):

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_dht_decoder_probe_v1.driver --run-dir /home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_dht_decoder_probe_v1
```

The driver performs the saved14×3 no-training diagnostic, frozen feature extraction, two1000-update decoder runs, real/synthetic-validation head inference, evaluation, HTML generation and browser QA. It then waits for the agent's actual screenshot inspection receipt before verifying the independent audit, visibly opening the report, and sending the user's authorized Discord result notice. `COMPLETION.json` prevents repeating delivery. An interrupted stage resumes only against its bound inputs; completed old experiments must not be restarted.

The oracle nearest-GT candidate diagnostic is explicitly separate from GT-free predictions. One newly trained seed, a previously used backbone validation pool, a reused real DEV population, unchanged centroid, and the absence of 6D evaluation limit the claims this pilot can support.

Actual results on reused real DEV319, with original semantic IDs and the fixed centroid included:

| Output | Median error (px) | P90 error (px) | Mean error (px) |
|---|---:|---:|---:|
| Frozen joint-network baseline | 6.897 | 41.487 | 22.454 |
| Learned point/image candidate module | 6.514 | 44.928 | 22.346 |
| Learned line-intersection/image module | 6.661 | 41.856 | 22.323 |
| Same line module with another image's lines | 7.372 | 42.996 | 23.142 |

The point-only module uses the same frozen joint backbone; it is not the historical point-only YOLO training arm. All outputs retain the same 2738/2818 observable-point coverage on 309 matched frames, with all 319 frames retained. The line-minus-point frame-mean difference was −0.023px, session bootstrap 95% CI [−0.642, +0.305]px. The line module also moved more originally ≤10px points beyond 10px than the point module (3.97% versus3.39%). The required-case median remained 265.213px versus 270.065px before refinement. Useful nearby candidates existed for some corners, but the learned mixture/gate/residual did not repair the large error; other corners lacked a close candidate.

The 512-image synthetic decoder validation is disjoint from the 2048-image decoder training subset but comes from the backbone's previously used validation pool. Both 55,118-parameter modules used identical initialization and 1000×64 sample order, with no checkpoint selection. Feature extraction took 76.035s and candidate/feature preparation 463.715s; the two small-head training runs including periodic validation took 22.390s combined. These are measured stage times, not full-pipeline inference latency.

See [actual results](../../../data/pallet/results/pallet_dht_decoder_probe_v1/RESULTS.json), [training completion](../../../data/pallet/results/pallet_dht_decoder_probe_v1/TRAINING_COMPLETION.json), [saved-output interpretation](../../../data/pallet/results/pallet_dht_decoder_probe_v1/provenance/docs_interpretation/RESULTS.json), and the [experiment note](../../../_docs/notes/pallet_dht_decoder_probe.md). No independent FINAL, 6D, negative-image evaluation, or new full-backbone training was performed in this pilot.

The [interactive report](../../../data/pallet/results/pallet_dht_decoder_probe_v1/index.html) contains the saved319-frame gallery. The note embeds an immutable [problem-case screenshot](../../../_docs/assets/pallet_dht_decoder_probe/problem_case_20260909.png). A separately declared [post-hoc fixed-gate diagnostic](../../../data/pallet/results/pallet_dht_decoder_probe_v1/provenance/docs_interpretation/gate_fixed_one/RESULTS.json) improved this case partially but substantially worsened overall and originally good points; it does not replace the registered result.
