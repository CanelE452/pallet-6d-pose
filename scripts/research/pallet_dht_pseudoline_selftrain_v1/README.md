# DHT pseudo-line self-training Stage A

This bounded track stopped at its preregistered teacher-source mechanism gate.
It does **not** include a trained student or a validated full mosaic student
trainer. No 9-fit was run, and no real DHT cache was generated after gate failure.
Stock student graph/exposure/optimizer tests are static contracts, not fit results.

`geometry.py` implements normal-only incidence loss and exact inverse-transpose
line transforms. `audit_mechanism.py` locks canonical sources and runs synthetic
calibration/test diagnostics using existing frozen P/DHT predictions. `independent.py`
checks every output-coordinate derivative against an independent NumPy formula.

Executed phases (repository root; original artifacts are immutable):

```bash
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.audit_mechanism lock
python -m pytest scripts/research/pallet_dht_pseudoline_selftrain_v1/tests -q
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.audit_mechanism mechanism --device cuda:0
# Initial diagnostic stopped on strict JSON NaN rejection; narrowly corrected:
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.checks correction
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.checks tests
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.audit_mechanism mechanism --device cuda:0
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.independent
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.checks close
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.checks baseline
python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.report
```

The original lock hashes are preserved. `IMPLEMENTATION_CORRECTION.json` binds
the narrow ignored-GT-NaN gradient fix and unchanged line caches; no gate, weight,
teacher or architecture was changed. These commands describe provenance and do
not authorize another student experiment or a post-failure parameter search.

CUDA runs used the existing `pallet-yolo26` environment and process-local
`LD_LIBRARY_PATH=/tmp/nvidia-580.173.02-userspace`. GPU monitoring is available via
`python -m scripts.research.pallet_dht_pseudoline_selftrain_v1.checks gpu`.
Other GPU jobs were not interrupted. No reboot/system driver modification occurred.

See [REPORT_KO.md](../../../_docs/experiments/pallet_dht_pseudoline_selftrain_v1/REPORT_KO.md).
