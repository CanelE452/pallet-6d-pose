"""Aggregate measured costs without repeating inference or score evaluation."""
import argparse
from pathlib import Path
import time

from .publish import read, dump


def run(private_root, public_root):
    private, public = map(Path, (private_root, public_root))
    history = read(private/'KERNEL_ATTEMPT_HISTORY.json')
    measured = []
    for event in history['attempts']:
        measured.append(dict(stage=event['kind'], order=event['order'], exit_code=event['exit_code'],
            wall_seconds=event.get('elapsed_seconds', event.get('tool_wall_seconds')),
            CPU_seconds=event.get('process_cpu_seconds'), peak_RSS_KiB=event.get('maximum_resident_set_size_kib'),
            clock='process stage' if 'elapsed_seconds' in event else 'actual tool wall span',
            unknown_CPU_or_RSS_not_inferred=True))
    for path, stage in [
        (private/'eval_v1/EVALUATION_RECEIPT.json', 'postseal_24_evaluation'),
        (private/'diagnostic_v1/DIAGNOSTIC_RECEIPT.json', 'postseal_351_diagnosis'),
        (private/'observation_context_v1/OBSERVATION_CONTEXT_RECEIPT.json', 'postseal_observation_spatial_context'),
        (private/'INDEPENDENT_CHECK_V1.json', 'independent_geometry_cost_selection_error'),
        (private/'INDEPENDENT_MATCHED_CONTROL_CSV_CHECK.json', 'independent_six_matched_control_CSV'),
        (public/'gallery/GALLERY_RECEIPT.json', 'gallery_render_and_proof'),
        (public/'gallery/PACKAGING_NOTES.json', 'gallery_display_metadata_fix'),
        (public/'PUBLIC_DERIVATIONS.json', 'small_output_publication')]:
        obj = read(path)
        measured.append(dict(stage=stage, order=None, exit_code=0,
            wall_seconds=obj.get('elapsed_seconds', obj.get('packaging_elapsed_seconds')),
            CPU_seconds=obj.get('process_cpu_seconds', obj.get('packaging_process_cpu_seconds')),
            peak_RSS_KiB=obj.get('maximum_resident_set_size_kib', obj.get('maximum_resident_set_KiB')),
            clock='process stage', unknown_CPU_or_RSS_not_inferred=True))
    result = dict(schema='VISIBLE_BOUNDARY_PILOT_ACTUAL_COST_V1', measured_stages=measured,
        measured_computation_wall_sum_seconds=sum(row['wall_seconds'] or 0 for row in measured),
        measured_CPU_sum_seconds=sum(row['CPU_seconds'] or 0 for row in measured),
        maximum_measured_peak_RSS_KiB=max(row['peak_RSS_KiB'] or 0 for row in measured),
        computation_wall_cap_seconds=3600, CPU_heavy_workers_max=2,
        elapsed_since_goal_creation_seconds=time.time()-1792241033,
        elapsed_since_workspace_snapshot_seconds=time.time()-read(private/'START_BEFORE_FETCH.json')['start_epoch'],
        total_wall_bounds_computation_including_uninstrumented_setup=True,
        model_visual_input_authoring_seconds=None, per_view_times_not_recorded=True,
        model_input_review_scope='Six original full RGBs, their direct crops and one comparison sheet per image; all before freeze',
        human_review_seconds=0, human_review_performed=False,
        model_author_input_frozen_UTC='2026-10-17T12:50:08Z',
        downloads_or_file_restorations_seconds=None,
        downloads_or_file_restorations_note='No dataset/model download; only existing selected-six neutral package copied; exact copy duration not instrumented',
        new_training=0, new_seed=0, new_data=0, new_NN_forward=0,
        new_SAM_or_cone_input=0, new_PnP=0, new_pose_optimization=0, new_candidate_generation=0,
        new_weights_or_threshold_or_r_or_direction_search=0, external_RGB_upload=0,
        original_retained_records=351, original_record_copy_is_not_generation=True,
        successful_new_score_runs=1, failed_new_score_attempts=1,
        end_to_end_RGB_to_pose_FPS_measured=False,
        stage_timing_includes_only_its_recorded_process_interval=True,
        final_QA_cost_location='VALIDATION.json and evidence/INDEPENDENT_PUBLICATION_CHECK.json when available',
        failed_run_CPU_and_RSS_missing_reason='Process crashed in Cython destructor; not instrumented before exit, no rerun to fill missing values',
        input_semantic_versions=1, candidate_or_reference_selection_feedback_to_input=False)
    dump(public/'COST.json', result)
    return {key: value for key, value in result.items() if key != 'measured_stages'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-root', required=True)
    parser.add_argument('--public-root', required=True)
    args = parser.parse_args()
    print(run(args.private_root, args.public_root))
