"""Inventory actual S3/S4 generation versions and completed H1 costs.

The sums below are explicitly completed, instrumented work. Interrupted-row
overhead and uninstrumented setup are unknown, never silently assigned zero.
This report does not run scientific production or read real references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from . import image_batch as I, material_assets as A


def _tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def report(run, output):
    run, output = map(Path, (run, output))
    output.mkdir(parents=True, exist_ok=True)
    if not (run/'s3_s4_postprocess/FINISH_RECEIPT.json').exists():
        raise ValueError('Completed five-arm postprocessing seal required')
    versions = {}
    total_resource, total_calls = Counter(), Counter()
    inputs = {}
    for name in ('s3_s4_technical_attempt1', 's3_s4_technical_attempt2', 's3_s4_production'):
        folder = run/name
        if not folder.exists():
            continue
        resource, calls = Counter(), Counter()
        count, peak = 0, 0
        for path in sorted((folder/'frames').glob('*/*/SEAL.json')):
            seal = I.read(path)
            if not seal['complete'] or A.sha(path.parent/'FRAME.jsonl.gz') != seal['frame_sha256']:
                raise ValueError('Cost archive includes an invalid row seal')
            count += 1
            row = seal['resource']
            for key in ('wall_seconds','CPU_seconds','logical_render_calls','actual_raster_passes','renderer_wall_seconds'):
                resource[key] += row.get(key, 0)
            calls.update(row.get('actual_calls', {}))
            peak = max(peak, row.get('process_peak_RSS_kib', 0))
        receipts = {}
        for mode in ('PILOT','FULL'):
            path = folder/(mode+'_RECEIPT.json')
            if path.exists():
                r = I.read(path)
                receipts[mode] = dict(path=str(path), sha256=A.sha(path), complete=r['complete'],
                    invocation_wall_seconds=r['invocation_wall_seconds'],
                    invocation_CPU_seconds=r['invocation_CPU_seconds'],
                    invocation_counts=r['current_invocation_counts'])
                inputs[str(path)] = A.sha(path)
        status = 'CURRENT_ACCEPTED_SCIENTIFIC_VERSION' if name == 's3_s4_production' else 'PRESERVED_TECHNICAL_PREDECESSOR_NOT_FINAL_ACCURACY'
        versions[name] = dict(status=status, unique_completed_rows=count,
            completed_row_resource_totals=dict(resource), completed_row_actual_calls=dict(calls),
            process_peak_RSS_kib=peak, completed_invocation_receipts=receipts,
            interrupted_partial_row_overhead=('UNKNOWN_NOT_ZERO; no completed seal for interrupted work'
                if (folder/'FULL_EXECUTION_LOCK.json').exists() and not (folder/'FULL_RECEIPT.json').exists()
                else 'NO_INTERRUPTED_FULL_ROW_RECORDED_IN_THIS_VERSION'),
            apparent_artifact_bytes=_tree_bytes(folder))
        total_resource.update(resource)
        total_calls.update(calls)
    ledgers = [I.read(p) for p in sorted((run/'s3_s4_training').glob('*/*/RUN_RECEIPT.json'))]
    if len(ledgers) != 15 or len({(r['arm'],r['seed']) for r in ledgers}) != 15:
        raise ValueError('Exactly fifteen completed own-distribution H1 fits required')
    fit_cost = Counter()
    for r in ledgers:
        fit_cost['internal_fit_wall_seconds'] += r['cost']['wall_seconds']
        fit_cost['internal_fit_CPU_seconds'] += r['cost']['CPU_seconds']
        fit_cost['optimizer_steps'] += r['optimizer_steps']
        fit_cost['training_frame_exposures'] += r['training_frame_exposures']
    post_ledger = run/'s3_s4_postprocess/POSTPROCESS_LEDGER.jsonl'
    completed = [r for r in I.rows(post_ledger) if r['event'] == 'COMPLETE']
    failed = [r for r in I.rows(post_ledger) if r['event'] == 'TECHNICAL_FAILURE']
    post_cost = dict(completed_command_invocations=len(completed), failed_command_invocations=len(failed),
        completed_child_wall_seconds=sum(r['wall_seconds'] for r in completed),
        completed_child_CPU_seconds=sum(r['child_CPU_seconds'] for r in completed),
        failed_child_wall_seconds=sum(r['wall_seconds'] for r in failed),
        failed_child_CPU_seconds=sum(r['child_CPU_seconds'] for r in failed),
        wall_includes_import_preparation_and_IO=True, internal_fit_cost_is_subset_do_not_add_twice=True)
    identities=list(I.rows(run/'s3_s4_production/MODEL_IDENTITY_INPUTS1241.jsonl.gz'))
    if len(identities)!=1241 or any(r['GT_pose_reads'] for r in identities):
        raise ValueError('Frozen no-GT model identity linkage differs')
    result = dict(schema='S3_S4_ACTUAL_COST_INVENTORY_V1', complete=True,
        generation_versions=versions,
        instrumented_completed_generation_all_version_resource_sum=dict(total_resource),
        instrumented_completed_generation_all_version_call_sum=dict(total_calls),
        old_original_D1_or_PF_production_cost_charged_again=False,
        exact_reused_CONTOUR_IMAGE_cost_charged_to_new_row=False,
        completed_generation_sum_is_lower_bound_on_entire_campaign_cost=True,
        uninstrumented_or_partial_setup_and_interruption_cost='UNKNOWN_NOT_ZERO; preserved logs and root wall ledger give further evidence',
        LP1_LP4_parity_receipt_sha256=A.sha(run/'S3_S4_RENDER_THREAD_PARITY_LP4.json'),
        renderer_hardware='Mesa llvmpipe CPU software rendering; no CUDA/hardware GPU render calls',
        frozen_source_asset_counts=dict(Counter(r['domain']+'/'+str(r['named_source_asset'])for r in identities)),
        frozen_asset_selection_reason_counts=dict(Counter(r['domain']+'/'+r['selection_reason']for r in identities)),
        model_identity_inputs_sha256=A.sha(run/'s3_s4_production/MODEL_IDENTITY_INPUTS1241.jsonl.gz'),
        all235_synthetic_validation_named_GLB_geometry_missing_uses_fixed_proxy=True,
        dataset_workers=1, BLAS_threads=1, final_graphics_internal_LP_threads=4,
        H1_actual_fits=15, H1_internal_fit_cost=dict(fit_cost), postprocessing_command_cost=post_cost,
        H1_training_receipts=[dict(arm=r['arm'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],
            optimizer_steps=r['optimizer_steps'],training_frame_exposures=r['training_frame_exposures'],cost=r['cost']) for r in ledgers],
        source_bindings=inputs, postprocessing_ledger_sha256=A.sha(post_ledger),
        full_generation_external_timing=I.read(run/'s3_s4_production/EXTERNAL_FINALIZATION_TIME_PROVENANCE.json'),
        full_generation_external_timing_sha256=A.sha(run/'s3_s4_production/EXTERNAL_FINALIZATION_TIME_PROVENANCE.json'),
        finish_receipt_sha256=A.sha(run/'s3_s4_postprocess/FINISH_RECEIPT.json'),
        report_source_sha256=A.sha(Path(__file__)),
        current_real_reference_reads=0)
    I.write(output/'ACTUAL_COST_INVENTORY.json',result)
    return dict(complete=True,H1_actual_fits=15,postprocessing_command_cost=post_cost,
        instrumented_completed_generation_all_version_resource_sum=dict(total_resource))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    print(I.canonical(report(a.run,a.output)),flush=True)
