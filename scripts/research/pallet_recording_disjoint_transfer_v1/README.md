# Frozen recording-disjoint transfer

Read REPORT_KO.md in the matching `_docs/experiments/` namespace.
No training entry point. Only original PLASTIC S0/S1 (seed42,320 updates each), with R0 supplementary.

Run in order with the existing `pallet-yolo26` environment:

```bash
python -m scripts.research.pallet_recording_disjoint_transfer_v1.preflight
python -m scripts.research.pallet_recording_disjoint_transfer_v1.role_scan
python -m scripts.research.pallet_recording_disjoint_transfer_v1.infer
python -m scripts.research.pallet_recording_disjoint_transfer_v1.evaluate
python -m scripts.research.pallet_recording_disjoint_transfer_v1.report decision
python -m scripts.research.pallet_recording_disjoint_transfer_v1.render
python -m scripts.research.pallet_recording_disjoint_transfer_v1.report
python -m scripts.research.pallet_recording_disjoint_transfer_v1.audit
```

First three locks/results are single-run outputs. On completed artifacts run `audit` only.
Hash-bound private image, annotation, checkpoint and cache files are required; they are not uploaded.
S0/S1 predictions are reused bit-for-bit from the frozen historical checkpoint caches.
Production D9 candidate inference is rerun on CPU without any reference input before scoring.
Oracle WD is posthoc/nondeployable. Fixed-ID visible66 are final V2 only.
Historical ALL300 and HELDOUT128 denominators must never be pooled.

The historical S1 policy has supervised coverage and matched-S2-feasibility conditions.
The evaluated difference is this conditional policy, not general unconstrained erasing.
No S2, LoRA, new loss/filter, human relabeling, seed search or training is performed.
