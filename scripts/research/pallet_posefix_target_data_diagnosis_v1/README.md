# Frozen target-data / train-fit diagnosis

No new training, optimizer steps, checkpoint updates or target changes. [Illustrated results](../../../_docs/experiments/pallet_posefix_target_data_diagnosis_v1/RESULTS_KO.md).

This implementation depends on the local historical `pallet_posefix_crop_completion_v2` and its imported research modules, frozen models, paired RGB tensors and private input bindings. It is not a standalone public reproduction package. Private predictions, pseudo coordinates and source-image mappings are intentionally excluded from publication.

Executed module order from repository root, in the `pallet-yolo26` environment:

1. `python -m scripts.research.pallet_posefix_target_data_diagnosis_v1.prepare`
2. `python -m scripts.research.pallet_posefix_target_data_diagnosis_v1.probe` (CUDA inference only)
3. `python -m scripts.research.pallet_posefix_target_data_diagnosis_v1.analyze`
4. `python -m scripts.research.pallet_posefix_target_data_diagnosis_v1.report`
5. `python -m scripts.research.pallet_posefix_target_data_diagnosis_v1.complete`

Outputs use exclusive creation. Do not delete existing outputs or rerun a completed stage to bypass guards. The analysis adapter supports exact verified reuse of already completed TRAIN aggregation after handling legacy eight-corner masks/null error entries; it never overwrites predictions. Initial attempt logs are retained locally.

The initial prepare loader was adapted to the actual accepted-only253 manifest plus264 candidate frame records, and current278-record subset of an older300-record pool. No population was newly selected. All253 natural inputs are evaluated. Controlled inference uses128 corners from128 frames selected from TRAIN pseudo targets before inference, not DEV references. P1/P2 hold crop/point inputs fixed and change RGB only; P3 additionally has actual OCC point/bbox changes.

Frozen models run eval+inference_mode with gradients disabled. Optimizer construction is blocked, state hashes are checked before/after, original artifacts are hash-verified. Reports are descriptive, not physical GT validation. Final human-edited explanation and coupled20/40/60px follow-up design refine the generated draft; this design was not executed and60px was not measured in the current probe.
