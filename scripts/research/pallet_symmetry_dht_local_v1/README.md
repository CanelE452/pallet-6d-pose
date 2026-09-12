# Symmetry-aware reliability-gated DHT local fusion v1

This bounded pilot keeps the stock R0 detector and all nine stock keypoints
frozen. Direct and DHT heads consume the same cached native P3/P4 forward,
predict 12 cuboid-edge line roles, and can move corners 0–7 only through a
reliability-gated local normal-direction WLS update. Centre 8 is copied exactly.
There is no PnP, pose solver, global candidate search, learned point head, or
backbone update in this package.

The data exporter derives the approved 1,792/256/512 membership from the v4
split manifests but never reuses the v4 model tensors: every observation is
rebuilt from the canonical stock-R0 cache whose checkpoint SHA256 is
`970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`.
The historical cache is FP16-rounded by design; exported 24x24 ROI samples are
FP32. This precision boundary is recorded rather than described as original
full-precision features.

Typical commands (from this directory):

```bash
python verify_bundle.py
python -m pytest tests -q
python -m symdht_local.export
python -m symdht_local.audit --export ../../../../data/pallet/results/pallet_symmetry_dht_local_v1/export --output ../../../../data/pallet/results/pallet_symmetry_dht_local_v1/INTEGRITY_AUDIT.json
python -m symdht_local.runner --help
python -m symdht_local.assessment --help
```

`runner score` and `score-point` instantiate the dataset with target loading
disabled. Training is the only runner path that opens supervision. A requested
CUDA device must exist; there is no implicit CPU fallback.
