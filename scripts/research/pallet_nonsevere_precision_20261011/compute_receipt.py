"""Account recorded execution scopes without pretending wall time is GPU occupancy."""
import argparse, hashlib, json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-dir', type=Path, required=True)
    args = parser.parse_args()
    doc = args.public_dir
    out = doc / 'COMPUTE_RECEIPT.json'
    assert not out.exists(), 'Existing receipt retained'
    training = read(doc / 'TRAIN_RECEIPTS.json')
    assert training['total_updates'] == 5 * 2 * training['common_steps']
    device = [dict(stage='train_and_auxiliary_calibration',
                   seconds=training['actual_device_seconds'],
                   scope='Synchronized training loops plus one-time auxiliary calibration; excludes some setup')]
    for name in ('selection', 'accuracy'):
        paths = sorted(doc.glob(f'STAGE_{name}*.json'))
        assert paths and any(read(path)['status'] == 'DONE' for path in paths)
        for path in paths:
            receipt = read(path)
            device.append(dict(stage=receipt['stage'], seconds=receipt['device_reserved_upper_seconds'],
                               status=receipt['status'], receipt=path.name,
                               scope='Whole process wall including CPU work/imports; failed attempts counted'))
    latency = read(doc / 'LATENCY.json')
    routes = latency.get('methods', latency.get('routes'))
    if isinstance(routes, dict):
        routes = list(routes.values())
    if routes is None:
        routes = [read(p) for p in sorted(doc.glob('LATENCY_*.json'))]
    assert len(routes) == 18
    # Use individual executable receipts, rather than summing timing repetitions.
    for path in sorted(doc.glob('LATENCY_*.json')):
        value = read(path)
        seconds = value.get('actual_device_seconds', value.get('device_active_seconds'))
        if seconds is None:
            seconds = value.get('GPU_active_seconds')
        if seconds is None:
            seconds = value.get('GPU_device_job_wall_seconds_conservative')
        assert seconds is not None, (path.name, list(value))
        device.append(dict(stage=path.stem, seconds=seconds,
                           scope='Independent process model load, warmup and measured full pipelines'))
    dry = read(doc / 'MODEL_DRY_RUN.json')
    device.append(dict(stage='forward_backward_dry_run', seconds=dry['total_setup_seconds'],
                       scope='Whole setup including no-update dry-run; recorded failed first import separately'))
    # Setup/failures without their own stopwatch are an explicit upper allowance.
    margin = 300
    observed = sum(float(v['seconds']) for v in device)
    cpu = []
    for name, keys in (
        ('DATA_CONTRACT.json', ('wall_seconds',)),
        ('BASELINE_COMPUTE_RECEIPT.json', ('wall_seconds',)),
        ('NEW_EVALUATION_RECEIPT.json', ('wall_seconds',)),
        ('STATISTICS_RECEIPT.json', ('wall_seconds',)),
        ('FIGURE_RECEIPT.json', ('wall_seconds',)),
        ('INDEPENDENT_VERIFICATION.json', ('wall_seconds',)),
        ('DETECTOR_MATCH_REFERENCE_LOCK.json', ('wall_seconds',)),
        ('DETECTOR_MATCH_PARITY.json', ('wall_seconds',)),
    ):
        path = doc / name
        if path.exists():
            value = read(path)
            for key in keys:
                if key in value:
                    cpu.append(dict(stage=name, seconds=value[key]))
                    break
    for stage in ('statistics', 'figures', 'independent'):
        path = doc / f'STAGE_{stage}.json'
        if path.exists():
            value = read(path)
            assert value['status'] == 'DONE'
            cpu.append(dict(stage=f'whole_process_{stage}', seconds=value['wall_seconds']))
    decoder = read(doc / 'DECODER_EXECUTION_RECEIPT.json')
    for key in ('geometry_completed_wall_seconds', 'geometry_contract_completed_wall_seconds',
                'decoder_finalizer_wall_seconds', 'figure_wall_seconds',
                'conservative_extra_draft_cpu_wall_seconds'):
        cpu.append(dict(stage=key, seconds=decoder[key]))
    selection = read(doc / 'SYNTHETIC_SELECTION.json')
    cpu.append(dict(stage='synthetic_pose_scoring_also_in_device_wall',
                    seconds=selection['CPU_scoring_seconds']))
    analysis_margin = 300
    cpu_upper = sum(float(v['seconds']) for v in cpu) + analysis_margin
    result = dict(status='PASS' if observed + margin < 7200 and cpu_upper < 3600 else 'BUDGET_EXCEEDED',
                  device_timing_definition='Reserved GPU stage wall; conservative upper bound, not CUDA kernel active time',
                  device_stage_receipts=device, measured_device_stage_seconds=observed,
                  setup_and_failed_startup_upper_allowance_seconds=margin,
                  accounted_device_upper_seconds=observed + margin,
                  accounted_device_upper_minutes=(observed + margin) / 60,
                  device_budget_seconds=7200, CPU_analysis_stage_receipts=cpu,
                  CPU_uninstrumented_contract_publication_and_later_verification_upper_allowance_seconds=analysis_margin,
                  accounted_CPU_analysis_upper_seconds=cpu_upper,
                  CPU_analysis_budget_seconds=3600,
                  CPU_timing_definition='Stage wall sum including interrupted-draft allowances; excludes waiting during fitting',
                  actual_optimizer_updates=training['total_updates'],
                  common_updates_per_arm_seed=training['common_steps'],
                  actual_new_corrector_fits=10, new_backbone_updates=0,
                  new_full_PoseFix_updates=0, new_synthetic_images=0,
                  REAL_Severe_training_selection_evaluation_rows=0,
                  protected_FINAL_training_selection_evaluation_rows=0,
                  simultaneous_owned_GPU_jobs=1, hyperparameter_search=False,
                  new_REAL_F_calls=read(doc / 'NEW_EVALUATION_RECEIPT.json')['actual_F_calls'],
                  new_REAL_OpenCV_calls=read(doc / 'NEW_EVALUATION_RECEIPT.json')['total_OpenCV_calls'],
                  synthetic_selection_F_calls=selection['pose_F_calls'],
                  synthetic_selection_OpenCV_calls=selection['OpenCV_calls'],
                  synthetic_roundtrip_preflight_F_calls=16,
                  synthetic_preflight_OpenCV_calls='Not instrumented; do not claim an actual counted total',
                  no_source_files_modified=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    assert result['status'] == 'PASS'
    print('COMPUTE_BUDGET_PASS', round(observed + margin, 3), round(cpu_upper, 3))


if __name__ == '__main__':
    main()
