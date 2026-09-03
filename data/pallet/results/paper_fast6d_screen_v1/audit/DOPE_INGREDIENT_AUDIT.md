# DOPE ingredient audit

```text
checkpoint      weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth
sha256 prefix   0de80490cb3b4f9b
inference path  scripts/stage0/selftrain/s1_cad_9filters.py infer_belief
coordinate map  eval_capturecad_b2.belief_to_orig_pad, the authoritative helper
pad             100
belief threshold 0.3 (THRESH)
existing cache  data/pallet/results/paper_eval_v1/baselines/DOPE_R0_PREDICTIONS.json
                covers 319 of 319 frames, 297 with detected corners
```

The decode here uses the repository's own belief-to-original mapping rather than a
private scale calculation. An earlier draft of this screen computed the mapping
itself and was replaced once the authoritative helper was located.

Padding is BORDER_REFLECT_101. The instruction offered constant 127 as a fallback
if an audit showed reflect to be worst; `MECHANISM_DIAGNOSTIC_REPORT.md` compares
both without condemning reflect, and the recorded convention is that DOPE
inference needs reflect because plain squash systematically under-detects
truncated and close-range pallets. The authoritative convention is followed and
the deviation from the fallback is recorded.

Raw DOPE on this population gives detection coverage 0.850 and pooled
keypoint median 11.42 px. The separately recorded DOPE reference is about
10.9 px at 0.737 detection; the difference comes from a different decode and
population and is noted rather than reconciled, since no claim rests on it.
