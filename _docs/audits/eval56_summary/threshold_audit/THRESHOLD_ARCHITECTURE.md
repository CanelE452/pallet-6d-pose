# Where the 0.30 actually lives

The evaluation never calls the decoder people usually mean by "the canonical
decoder".  `evaluate_belief_maps` and `evaluate_cache` both take
`MD.decode_all(...)["D0"]`, and `D0` is built from `FZ.heatmap_stats`.

```
scripts/stage0/paper_s2_frozen_diagnostic.py:73    BELIEF_THRESHOLD = 0.3
scripts/stage0/paper_s2_frozen_diagnostic.py:74    LOCAL_RADIUS = 3
scripts/stage0/paper_s2_frozen_diagnostic.py:75    LOCAL_TEMPERATURE = 0.1
scripts/stage0/paper_s2_frozen_diagnostic.py:544   flat_index = argmax(work)      raw map
scripts/stage0/paper_s2_frozen_diagnostic.py:546   peak = work[arg_y, arg_x]      raw peak
scripts/stage0/paper_s2_frozen_diagnostic.py:558   patch = 7x7 around the argmax
scripts/stage0/paper_s2_frozen_diagnostic.py:661   "detected": peak >= BELIEF_THRESHOLD
scripts/stage0/paper_s2_mechanism_diagnostic.py:406 D0 = soft_px if detected else None
```

So on the evaluation path there is exactly **one** threshold, it is compared
against the **raw, unsmoothed** peak, and the comparison is `>=`.

Four things that are commonly confused with it are **not** on this path:

- the Gaussian `sigma=2` blur and the 4-neighbour NMS live in
  `scripts/data_prep/eval/filter_pr_camfacing.py:132-139`, inside
  `extract_keypoints_from_belief`, which produces `D2`.  The evaluation does
  not use `D2`.
- the 11x11 weighted centroid (`RAN = 5`) is also `D2`
  (`filter_pr_camfacing.py:125,146-152`).  The evaluation path uses a **7x7**
  local softargmax at temperature 0.1 instead (`LOCAL_RADIUS = 3`).
- the `+0.4395` offset is `D2` only.
- there is no affinity grouping anywhere on this path.  `current_solve`
  (`paper_s2_frozen_diagnostic.py:1140-1153`) takes the nine indexed points
  directly and only checks `valid < 4`.  "Association" and "acceptance" are the
  same event here, which is why the corner table records
  `affinity_association = None` rather than inventing a value.

The audit therefore varies one number, per channel, and nothing else.
`decode_thresholded` calls `FZ.heatmap_stats` unmodified and re-applies the
comparison to the `peak` it returns; an accepted corner receives exactly the
`_soft_px` the frozen decoder already computed.  At 0.30 on all nine channels
this is bit-identical to `decode_all(...)["D0"]`, which is what the Phase A
parity check confirms.

Centroid (channel 8) is held at the canonical 0.30 in every arm.

## Provenance

```
HEAD                 769041102221c012d6666f7a471e69e3b8197837
ep57 checkpoint SHA  c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
threshold source     paper_s2_frozen_diagnostic.py:73  BELIEF_THRESHOLD = 0.3
acceptance site      paper_s2_frozen_diagnostic.py:661
audit decoder        scripts/stage0/paper_s2_eval56.py  decode_thresholded
training steps       0
optimizers created   0
```
