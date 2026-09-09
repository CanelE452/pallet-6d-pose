# Whole-pallet candidate decoding pilot

Select eight semantic corners jointly using twelve role-specific Deep Hough lines and projective-cuboid consistency. Reuses frozen evidence from `pallet_dht_decoder_probe_v1`; no CNN retraining, GT rewriting, self-training, or pseudo-label filtering.

Protocol and outputs: `data/pallet/results/pallet_dht_global_layout_v1/`. Scientific intent: `_docs/notes/pallet_dht_global_layout.md`. The original camera-facing same-ID convention and 319-frame DEV population remain authoritative. Qualitatively reviewed GT points are an additional diagnostic, not certified exact coordinates.

Actual result: global median/P90 **11.444/79.682px**, original joint output **6.897/41.487px**. The registered continuation criteria fail. Synthetic validation also worsens. Correct-looking layouts can be present but misranked by the fixed scoring rule; this is not evidence that all learned point/line architectures are impossible.

Reproduction uses the `pallet-pose` environment, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Run modules from the repository root:

```
python -m scripts.research.pallet_dht_global_layout_v1.preflight --run-dir RUN_DIR
python -m scripts.research.pallet_dht_global_layout_v1.runner --run-dir RUN_DIR --phase all
python -m scripts.research.pallet_dht_global_layout_v1.report --run-dir RUN_DIR
python -m scripts.research.pallet_dht_global_layout_v1.visual_qa --run-dir RUN_DIR
```

The run directory must contain PURPOSE, PROTOCOL and a SHA-bound SOURCE_FREEZE before calibration. `driver` connects actual computation, rendering, audit/review receipts and final delivery; it stops on missing or stale evidence and can resume completed stages. Do not overwrite the completed original run or replay its completion notification. Geometry tests: `python -m pytest -q scripts/research/pallet_dht_global_layout_v1/test_geometry.py`.
