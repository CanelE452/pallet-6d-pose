# Canonical dimension gate — completed before any DIM main fit

Status: COMPLETE. All40,000 G38 fixed axes and unchanged20,000 legacy dimensions were verified. Corrected N4 GPU smoke, actual-cache adapter audit and25 regression tests passed. DIM main training resumed only afterwards. Detailed evidence is in `CANONICAL_DIMENSION_CORRECTION.json`; the narrative below records the historical gate.

The source-code audit found that G38 `objects[0].dimensions_m` is **camera-facing**, not fixed renderer XYZ. `build_probe_metadata.py` explicitly labels it forbidden as model input. The user-requested example exporter was therefore unsuitable for G38 conditioning, despite being appropriate for its previous limited use.

N0 and N1 do not read dimension context. Their six fixed-budget runs remain valid if completed under the unchanged rows/targets/order. At discovery N1 seed3 was at step2800; N2/N3/N4 main fits had not begun. The protocol file was moved intact into history as a deliberate gate: the current N1 fit can finish, but the next run must fail before initialization/updates. No process or optimizer state is killed, and no incomplete fit updates need rollback.

Next: use the already audited **fixed_renderer_dimensions_m_xyz_model_input** for all G38 records, verify its provenance and agreement with the immutable geometry side table, keep legacy fixed XYZ, recompute train-only normalization, preserve the pre-correction sidecar/locks/discarded smoke evidence, and rerun corrected N4 smoke before DIM main training. No new architecture, loss, order, seed, budget or DEV selection.

Correction supersedes the preliminary statement that the exporter dimensions were canonical. Sorted extent equality alone cannot certify axis semantics. Neither old nor new per-frame pose, clicked corners or evaluation-selected phase may be used by the inference wrapper to swap dimensions.
