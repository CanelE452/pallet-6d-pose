# Existing-data hard transfer

Plastic-only, three fixed320-update students, same original R0 and source512. Existing artifact paths are read-only. No new teacher fits, annotation, capture, loss, selector or seed search.

Execute from repository root in the existing `pallet-yolo26` environment. All output files use exclusive creation; do not rerun completed stages into an existing namespace.

```
python -m scripts.research.pallet_existing_data_transfer_v1.prepare split
python -m scripts.research.pallet_existing_data_transfer_v1.prepare teacher
python -m scripts.research.pallet_existing_data_transfer_v1.cache
python -m scripts.research.pallet_existing_data_transfer_v1.contracts
python -m scripts.research.pallet_existing_data_transfer_v1.train T0_EASY_PSEUDO
python -m scripts.research.pallet_existing_data_transfer_v1.train T1_HARD_PSEUDO
python -m scripts.research.pallet_existing_data_transfer_v1.train T2_HARD_MANUAL
python -m scripts.research.pallet_existing_data_transfer_v1.evaluate infer
python -m scripts.research.pallet_existing_data_transfer_v1.evaluate score
python -m scripts.research.pallet_existing_data_transfer_v1.evaluate diagnostics
python -m scripts.research.pallet_existing_data_transfer_v1.report
python -m scripts.research.pallet_existing_data_transfer_v1.audit
```

T0=C0+E pseudo; T1=C0+H pseudo; T2=same H/RGB/box/mask with existing manual coordinates. Masks are intersected after the actual installed image transform. Additional E/H are same-recording budget pairs, not same-pose occlusion pairs. Frozen S1 rectangle application, fill, area fraction and aspect are reused; the rectangle center is mapped using normalized predicted-box coordinates. Source and C0 tensors remain exact.

Recording-disjoint fitting but reused DEV; legacy-reference and verified-manual metrics are separate. Manual heldout is N/A when provenance is absent. Pose oracles are posthoc and nondeployable. No hidden-coordinate truth is fabricated.

Generated report figures crop the pallet ROI and anonymize detected faces only for publication; model/evaluation input images remain untouched. Inspect figures before push. Stage only this code namespace and its public docs, never RAW/cache/checkpoints/full original images.
