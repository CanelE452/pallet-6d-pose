# DOPE actual-resize control

The frozen legacy DOPE preprocessing reflects 100 pixels, resizes the padded image back to original `W×H`, then resizes to a short side of 400. Its actual tensor sizes are rounded down to multiples of eight:

```python
sc = 400 / min(H, W)
nw = max(8, int(round(W * sc)) & ~7)
nh = max(8, int(round(H * sc)) & ~7)
```

Sources: `scripts/stage0/eval_harness/eval_pvnet_heads.py:preprocess` and `data/pallet/eval_results/stage16_truncation_addon/capturecad_b2_eval/eval_capturecad_b2.py:pad_frame,belief_to_orig_pad`, reused by `dope_baseline.py`.

For decoded belief coordinate `b_x`, belief width `bw`, and pad `P`, the existing inverse is:

```text
legacy_x = (b_x * nw / bw) / sc * (W + 2P) / W - P
```

The actual resize uses width ratio `nw/W`, rather than nominal `sc`. Undoing that actual ratio, while keeping every other existing decoder convention, gives:

```text
corrected_x = (b_x * nw / bw) * W / nw * (W + 2P) / W - P
            = (legacy_x + P) * (sc * W / nw) - P
corrected_y = (legacy_y + P) * (sc * H / nh) - P
```

For a 640×480 original, the actual network input is 528×400 rather than 533.333×400. Thus `corrected_x=(legacy_x+100)*100/99-100`, while y is unchanged. A stored x=0 moves +1.0101 pixels, x=320 moves +4.2424, and x=640 moves +7.4747. These values are geometric consequences of the fixed input sizes, determined before any accuracy analysis.

`resize_control.py` applies this affine correction to all finite stored coordinates and keeps semantic IDs and confidence masks. The center undergoes the same baseline-coordinate correction; subsequent corner/line fusion leaves this corrected center unchanged. Nonfinite missing points remain missing. The actual `input_shape_chw` field is checked against the established preprocessing formula, so an incompatible cache fails explicitly.

The `DOPE_exact_resize` arm is an **exact inverse of resize sizes under the existing decoder convention**. It keeps the canonical Gaussian/local-peak decoder, weighted patch, `+0.4395` offset and all pixel-center conventions. It introduces no extra half-pixel correction and does not claim to resolve every possible belief-to-image alignment bias. DHT's separately verified VGG feature mapping is unchanged.

This arm is a required control: a DHT fusion gain over legacy DOPE could partly compensate for the inherited inverse-scale bias. Report both legacy and corrected DOPE baselines, each with synthetic-validation-only lambda selection, and assess any integration recommendation against the corrected control. No GT, prediction error or DHT output enters the correction. Tests compare arbitrary decoded belief coordinates against the direct inverse using actual per-axis sizes; no real images or annotations are needed.
