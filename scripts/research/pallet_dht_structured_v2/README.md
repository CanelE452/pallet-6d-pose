# Learned point/segment layout verification

Active development toward the user's requested point + Deep Hough architecture with verified accuracy improvement. This directory is separate because prior completed experiments and their hashes must remain immutable.

The first implementation will test learned finite-segment and endpoint evidence using cached P4 feature maps and DHT predictions. Synthetic labels supervise training only; real original-ID GT is evaluation-only. Data filtering for self-training is outside scope. A failed pilot is progress evidence, not completion of the active goal.

The verifier samples 3x3 corner patches and eight positions with three normal offsets along each finite edge. A two-layer Transformer combines eight corner and twelve edge tokens. The DHT arm adds four role-specific line modes with absolute sigmoid scores and signed endpoint distances. All three arms have131,458 parameters and receive the same GT-free whole-layout candidate bank. The point-only name refers to its scoring evidence; that shared bank still contains DHT intersections.

Run the frozen pilot with the `pallet-pose` Python environment:

```bash
python -m scripts.research.pallet_dht_structured_v2.driver --run-dir data/pallet/results/pallet_dht_structured_v2 --device cuda:0
```

The driver requires a completed cache and `SOURCE_FREEZE.json`. It trains three matched final-checkpoint models and calibrates the selection margin on synthetic calibration256 before synthetic validation512. `PILOT_RESULTS.json` describes synthetic advancement only; it is not a real accuracy or completed-user-goal claim. Existing completed cells are validated and reused, never silently overwritten.

See `_docs/notes/pallet_dht_structured.md`, `_docs/handoff/2026-09-09_pallet_dht_structured_v2.md` and the run-directory protocol/specification for decisions and current progress.
