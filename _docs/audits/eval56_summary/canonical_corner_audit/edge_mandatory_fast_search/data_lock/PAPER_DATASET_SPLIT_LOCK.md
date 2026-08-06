# Paper dataset split lock

The split lives under `data/`, which is gitignored, so nothing there is under
version control.  These copies are the protected record.  After this commit the
split does not change.

## What the split is

```
holdout name   APPEARANCE_COMBINATION_HOLDOUT
group key      (background, preset, floor_texture, floor_mode, pallet_type)
group_id       sha256(canonical_json(key))
split sha256   9a755438dcb55e0f...

               frames    groups
train          16,011      306      80.1%
validation      1,995       39      10.0%
untouched       1,994       39      10.0%
total          20,000      384
```

Never part of the key: stage seeds, camera seeds, run id, archive id, shard id,
loader index, frame id, filename.  `run1` and `run2` frames land in the same
group whenever their appearance tuple matches, and all 384 groups span both runs.

## Why it is not called a scene holdout

Every frame in this dataset carries its own independent render seed.  A 2,000
frame probe found 2,000 distinct background seeds and no tuple shared by two
frames, so there are no camera sequences, no appearance variants and no derived
frames.  Source-scene leakage is structurally impossible here, which means a
scene-group holdout has nothing to hold out.  Calling this an
`APPEARANCE_COMBINATION_HOLDOUT` states what it actually withholds.

```
validates            an unseen exact appearance combination
                     robustness to background / floor / preset / pallet change

does not validate    unseen asset class
                     unseen pallet type
                     temporal or camera-trajectory generalization
                     derived-frame leakage (not applicable, see above)
```

Reading a result from this split as scene-level generalization would overstate
it.

## Full 20,000-frame audit

```
unique groups        384        singletons 0
group size           min 21   median 46   p90 65   p99 154   max 166
largest group        0.83% of eligible          (gate <= 5%)
groups spanning both runs   384 / 384
field cardinality    background 2 · preset 4 · floor_texture 12 · floor_mode 2 · pallet_type 4
missing group field  0
image hashes         20,000 distinct, 0 shared across splits
```

All ten pre-registered gates passed.

## canonical null

`floor_texture` is JSON `null` on 4,023 frames, every one of them
`floor_mode == native`.  The audit reports zero missing group fields, so null is
a valid category rather than a defect, and it is never rewritten into the string
`"None"`, `"null"` or `"missing"`.

## Allocation

Deterministic greedy over groups ordered by descending size then group id.
Lexicographic objective: lowest fill ratio against the frame-count target, then
marginal deviation across pallet_type, preset, background and diagnostic_mode,
then group id.  No random retry, no re-draw.

A first attempt used relative deviation as the leading term and produced
60.5 / 19.7 / 19.8, because the two small targets always looked further from
their goal.  That was a defect in the objective rather than a property of the
data, and it was corrected to fill ratio before anything downstream was built.

## Verify

```
cd _docs/audits/eval56_summary/canonical_corner_audit/edge_mandatory_fast_search/data_lock
sha256sum -c SHA256SUMS
```
