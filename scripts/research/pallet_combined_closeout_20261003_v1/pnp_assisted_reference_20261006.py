"""Bind the existing twelve PnP-assisted annotations to a separate geometry panel.

This deliberately does not modify or approve the original directly-visible
reference, its evaluator, its 120-frame plan, or any prior result.  It creates
only a new immutable reference/queue/protocol; target decisions remain human.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
LIFTER = ROOT / 'scripts/research/pallet_lifter_case_review_20261003_v1'
RAW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1'
REVIEW = RAW / 'review'
OUTPUT = ROOT / 'data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1'
INTEGRATION = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/INTEGRATION_MANIFEST.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_frozen(path, value):
    """Repeated preparation validates the first frozen copy; never replaces it."""
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen new-contract input changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def valid_xy(xy):
    return (isinstance(xy, list) and len(xy) == 2
            and all(type(value) in (int, float) and math.isfinite(value) for value in xy))


def prepare_context(output=OUTPUT):
    """Return the original object-match Context bound to the new geometry panel."""
    started = time.perf_counter()
    output = Path(output).resolve()
    if output == REVIEW or output.is_relative_to(REVIEW):
        raise ValueError('New PnP-assisted panel must be outside original review inputs')
    sys.path.insert(0, str(LIFTER / 'combined_integration'))
    from bridge import QUEUE_SCHEMA, _selection
    from object_match_review import Context
    import cv2

    checks = Counter()
    def require(condition, name):
        if not condition:
            raise ValueError(name)
        checks[name] += 1

    manifest_path = REVIEW / 'MANIFEST.json'
    plan_path = RAW / 'LIFTER_EVALUATION_PLAN.json'
    corner_path = REVIEW / 'CORNER_CONTRACT.json'
    batch_path = REVIEW / 'SMALL_BATCH_12_V1.json'
    recovery_path = REVIEW / 'VIEWER_DRAFT_RECOVERY.json'
    predictions_path = RAW / 'raw_predictions/ALL_STORED_FRAMES.jsonl'
    identity_path = predictions_path.parent / 'RUN_IDENTITY.json'
    original_evaluator = LIFTER / 'metrics/evaluate.py'
    # Preserve every existing review artifact, not only the 12 selected sources.
    protected_paths = {path.resolve() for path in REVIEW.rglob('*') if path.is_file()}
    protected_paths.update(path.resolve() for path in
        (plan_path, predictions_path, identity_path, INTEGRATION,
         ROOT / 'scripts/annotate/annotate.py', original_evaluator))
    protected = [binding(path) for path in sorted(protected_paths)]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    batch, manifest, plan, recovery = map(read, (batch_path, manifest_path, plan_path, recovery_path))
    expected_raw = read(INTEGRATION)['inputs']['lifter_raw']
    raw_binding = binding(predictions_path)
    require(raw_binding['sha256'] == expected_raw['sha256']
            and raw_binding['bytes'] == expected_raw['bytes'], 'original_8910_raw_receipt_binding')
    input_bindings = dict(manifest_sha256=sha(manifest_path), plan_sha256=sha(plan_path),
                          corner_contract_sha256=sha(corner_path),
                          corner_definition_version=read(corner_path)['version'])
    require(batch['bindings'] == recovery['bindings'] == input_bindings,
            'original_batch_recovery_and_input_bindings')
    ids = batch['frame_ids']
    require(len(ids) == len(set(ids)) == 12 and batch['repeat_frame_ids'] == []
            and batch['frozen_plan_modified'] is False, 'original_fixed_twelve_frame_batch')
    frames = {frame['frame_id']: frame for frame in manifest['frames']}
    planned = {frame['frame_id']: frame for frame in plan['frames']}
    require(len(planned) == 8910 and set(ids) <= set(frames) <= set(planned),
            'original_8910_plan_and_manifest_scope')
    rows = {}
    for line in predictions_path.open(encoding='utf-8'):
        raw = json.loads(line)
        fid = raw['frame_id']
        require(fid not in rows and fid in planned, 'raw_unique_identity')
        rows[fid] = raw
    require(set(rows) == set(planned), 'complete_8910_raw_identity_set')

    records, queue_records = [], []
    point_sources = Counter()
    current_manual = current_hidden = 0
    for fid in ids:
        frame, planned_frame, raw = frames[fid], planned[fid], rows[fid]
        require(frame['decoded_bgr_sha256'] == planned_frame['decoded_bgr_sha256']
                == raw['decoded_bgr_sha256'], 'selected_frame_original_decoded_pixels')
        require(raw['session_id'] == frame['session_id']
                and raw['stored_index'] == frame['saved_frame_index']
                and raw['camera_frame_number'] == frame['camera_frame_number']
                and raw['sensor_timestamp_ms'] == frame['camera_sensor_timestamp_ms'],
                'selected_frame_original_time_index_identity')
        image_path = (manifest_path.parent / frame['image_path']).resolve()
        require(sha(image_path) == frame['image_sha256'], 'original_PNG_file_binding')
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        require(image is not None and image.shape == (frame['height'], frame['width'], 3)
                and hashlib.sha256(image.tobytes()).hexdigest() == frame['decoded_bgr_sha256'],
                'original_PNG_decoded_binding')
        source_path = REVIEW / 'native_annotations/primary' / (
            f"{frame['session_id']}_{frame['saved_frame_index']:05d}.PNP_ASSISTED.json")
        source = read(source_path)
        require(source['schema'] == 'lifter_native_pnp_assistance_v1'
                and source['source_kind'] == 'human_assisted_annotation'
                and source['frame_id'] == fid and source['review_pass'] == 'primary'
                and source['image_sha256'] == frame['image_sha256']
                and source['bindings'] == input_bindings, 'exact_original_PnP_source_identity')
        require(source['evaluation_use'] is False and source['independent_reference'] is False
                and source['human_provenance_confirmed'] is False,
                'original_PnP_evaluation_approval_flags_preserved')
        xy_list = source['editor_kps_2d'][:8]
        annotations = source['editor_keypoint_annotations'][:8]
        require(len(xy_list) == len(annotations) == 8 and all(valid_xy(xy) for xy in xy_list),
                'all_eight_original_saved_editor_coordinates')
        objects = source['annotation']['objects']
        require(len(objects) == 1 and objects[0]['projected_cuboid'] == xy_list
                and objects[0]['manual_kps'][:8] == xy_list,
                'original_editor_and_annotation_geometry_identical')
        points = []
        for index, (xy, annotation) in enumerate(zip(xy_list, annotations)):
            origin = annotation['source']
            require(origin in ('manual_click', 'pnp_projected') and annotation['xy'] == xy,
                    'actual_original_point_provenance_and_xy')
            point_sources[origin] += 1
            points.append(dict(id=index, x=xy[0], y=xy[1], source=origin,
                in_frame=(0 <= xy[0] < frame['width'] and 0 <= xy[1] < frame['height'])))
        draft = recovery['drafts'][fid + '|primary']['record']
        require(draft['frame_id'] == fid and draft['review_pass'] == 'primary'
                and draft['status'] == 'draft' and draft['reviewer'] is None,
                'existing_target_identity_record_not_promoted')
        target = draft['object']
        require(target['presence'] == 'present' and target['target_identity_confirmed'] is True
                and isinstance(target['target_object_id'], str) and target['target_object_id'],
                'actual_existing_target_identity')
        current_manual += sum(corner['visibility'] == 'direct_visible'
                              and valid_xy([corner['x'], corner['y']]) for corner in draft['corners'])
        current_hidden += sum(corner['self_occlusion'] is True for corner in draft['corners'])
        base, n3 = raw['methods']['Base'], raw['methods']['N3']
        require(base['keypoints_mask'] == n3['keypoints_mask'], 'Base_N3_same_original_point_masks')
        selection = _selection(raw)
        require(base['selected_object'] == n3['selected_object'], 'Base_N3_same_original_selected_object')
        require(selection['selected_index'] is not None and valid_xy(selection['selected_box_xyxy'][:2])
                and valid_xy(selection['selected_box_xyxy'][2:]), 'actual_frozen_selected_box_present')
        source_binding = binding(source_path)
        records.append(dict(frame_id=fid, session_id=frame['session_id'],
            saved_frame_index=frame['saved_frame_index'],
            camera_sensor_timestamp_ms=frame['camera_sensor_timestamp_ms'],
            camera_frame_number=frame['camera_frame_number'],
            target_object_id=target['target_object_id'], object=target,
            target_identity_source=dict(path=binding(recovery_path)['path'],
                sha256=sha(recovery_path), pointer=f"drafts.{fid}|primary.record.object"),
            image_sha256=frame['image_sha256'], decoded_bgr_sha256=frame['decoded_bgr_sha256'],
            image_path=frame['image_path'], width=frame['width'], height=frame['height'],
            points=points, source_pnp_binding=source_binding,
            source_original_evaluation_use=False, source_original_human_provenance_confirmed=False,
            independent_reference=False, formal_human_review_promoted=False,
            keypoint_frame=objects[0].get('keypoint_frame'),
            dimensions_m=objects[0]['dimensions_m'],
            original_reprojection_px=objects[0].get('reproj_error_px')))
        queue_records.append(dict(frame_id=fid, session_id=frame['session_id'],
            target_object_id=target['target_object_id'], image_sha256=frame['image_sha256'],
            image_path=frame['image_path'], **selection))
    require(dict(point_sources) == {'manual_click': 66, 'pnp_projected': 30},
            'original_66_manual_30_PnP_geometry_sources')
    require(len(records) == 12 and sum(len(record['points']) for record in records) == 96,
            'fixed_12_by_8_geometry_denominator')
    profile_path = REVIEW / 'NATIVE_REVIEWER_PROFILE.json'
    history_confirmation = None
    if profile_path.is_file():
        profile = read(profile_path)
        reviewer = profile.get('reviewer', {})
        require(reviewer.get('entered_by') == 'human' and reviewer.get('confirmation') is True
                and isinstance(reviewer.get('previous_prediction_exposure'), bool),
                'existing_actual_human_history_answer_preserved')
        history_confirmation = dict(binding=binding(profile_path), record=profile,
            scope='actual_first_save_history_answer_for_annotation_creation',
            previous_prediction_exposure_answer=reviewer['previous_prediction_exposure'],
            does_not_approve_per_frame_reference=True,
            does_not_supply_target_match_decisions=True)
    reference = dict(schema_version='lifter_pnp_assisted_geometry_reference_v1',
        source_kind='existing_human_assisted_pnp_annotations',
        evaluation_reference_type='PnP_assisted_geometry8',
        evaluation_scope='partial_12_frame_exploratory_assisted_2D_geometry_panel',
        independent_reference=False, independent_physical_TR='x',
        formal_human_review_promoted=False, prior_external_prediction_exposure='UNKNOWN',
        existing_human_history_confirmation=history_confirmation,
        user_workflow_clarification='User states manually clicked points, visually checked PnP alignment, '
            'and used G to complete corners; clarification is workflow-level, not fabricated per-frame approval.',
        original_direct_visible_contract_unchanged=True, original_reference_flags_unchanged=True,
        source_binding_policy='Exact first eight editor_kps_2d of original per-frame PNP_ASSISTED files; '
            'no later direct click replacement, nearest-point remapping, pose recalculation or generated point.',
        bindings=input_bindings, original_batch_binding=binding(batch_path),
        raw_prediction_binding=raw_binding, prediction_run_identity_binding=binding(identity_path),
        source_counts=dict(point_sources), frame_count=12, reference_point_count=96,
        later_current_manual_snapshot=dict(coordinate_count=current_manual, self_occluded_count=current_hidden,
            binding=binding(recovery_path), used_for_this_geometry_reference=False),
        records=records)
    reference_path = output / 'ASSISTED_REFERENCE.json'
    write_frozen(reference_path, reference)
    bindings = dict(predictions_sha256=sha(predictions_path),
        prediction_run_identity_sha256=sha(identity_path),
        reviewed_reference_sha256=sha(reference_path), manifest_sha256=sha(manifest_path))
    queue = dict(schema_version=QUEUE_SCHEMA, source_kind='machine_prepared_human_task',
        status='WAITING_HUMAN', bindings=bindings,
        policy=dict(show_selected_box_only=True, show_model_corners=False, show_method_name=False,
            show_error_or_improvement=False, decision_values=['same', 'different', 'undetermined']),
        human_decision_required_count=12,
        human_decision_not_required_counts=dict(no_selected_prediction=0,
            no_direct_visible_reference=0, no_predicted_points=0), records=queue_records,
        reference_contract='SEPARATE_PNP_ASSISTED_GEOMETRY8_PANEL',
        original_direct_visible_contract_unchanged=True)
    queue_path = output / 'OBJECT_MATCH_QUEUE.json'
    write_frozen(queue_path, queue)
    # Freeze analysis before opening any prediction error or improvement.
    protocol = dict(schema_version='lifter_pnp_assisted_geometry8_protocol_v1',
        analysis_scope='separate_partial_12_frame_exploratory_panel',
        bindings={**bindings, 'queue_sha256': sha(queue_path),
            'plan_sha256': sha(plan_path), 'corner_contract_sha256': sha(corner_path),
            'batch_sha256': sha(batch_path)},
        ordered_frame_ids=ids, methods=['Base', 'N3'], ordered_corner_ids=list(range(8)),
        reference_frame_count=12, fixed_reference_point_denominator=96,
        reference_point_sources=dict(point_sources),
        reference_kind='PnP_assisted_geometry8_not_independent_visible_or_physical_ground_truth',
        include_all_reference_corners=True, include_out_of_frame_reference_corners=True,
        include_self_hidden_reference_corners=True,
        matching_rule='canonical_fixed_id_0_to_7_no_nearest_or_symmetry_remapping',
        object_selection='exact_original_shared_Base_N3_selected_index_and_box',
        masks='exact_original_shared_Base_N3_validity_mask',
        pck_threshold_px=10.0,
        pck_denominator='all_96_locked_reference_slots_for_each_method',
        pck_failure_policy='Missing prediction or confirmed different target correspondence fails '
            'PCK for all affected fixed reference slots; no slot/frame exclusion. '
            'Undetermined correspondence is unresolved and never classified as an incorrect target.',
        undetermined_object_policy='WAITING_UNDETERMINED_OBJECT_MATCH; withhold final accuracy while '
            'any actual human object decision is undetermined; retain the full 12-frame/96-point '
            'reference population and unresolved counts separately from failures.',
        error_summary_policy='Median and P90 over finite eligible same-target prediction errors; '
            'finite count and full fixed denominator/failure counts disclosed alongside.',
        pending_object_decisions='No reported final accuracy until all 12 actual human decisions exist '
            'and none is undetermined; incomplete remains WAITING_HUMAN, no synthesized same decisions.',
        paired_analysis='Same original frame and canonical corner ID; median(after)-median(before) '
            'and median(after-before) remain separate.',
        no_new_training=True, no_new_model_forward=True, no_pose_recalculation=True,
        no_parameter_search=True, no_relabeling=True, no_original_result_replacement=True,
        full_original_120_24_completed=False, prior_external_prediction_exposure='UNKNOWN',
        independent_physical_TR='x', stationary_noise='x',
        formal_canonical_direct_visible_reference_approval_created=False)
    write_frozen(output / 'EVALUATION_PROTOCOL.json', protocol)
    write_frozen(output / 'PROTECTED_SOURCE_BINDINGS.json', dict(
        schema_version='pnp_assisted_source_preservation_v1', git_HEAD=head,
        protected_file_count=len(protected), files=protected))
    context = Context(queue_path, manifest_path, predictions_path, reference_path,
                      output / 'object_match_in_progress.json')
    context.native_scope_metadata = dict(partial_batch=True, batch_primary_denominator=12,
        analysis_scope=protocol['analysis_scope'], reference_point_denominator=96,
        reference_type='PnP_assisted_geometry8', full_frozen_plan_completed=False,
        original_direct_visible_contract_unchanged=True, formal_human_review_promoted=False)
    require(set(context.records) == set(ids) and len(context.records) == 12,
            'existing_object_match_Context_accepts_exact_new_bound_queue')
    for item in protected:
        require(binding(resolve(item['path'])) == item, 'all_original_input_files_preserved')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == head,
            'git_HEAD_preserved')
    result = dict(schema_version='pnp_assisted_preparation_receipt_v1',
        status='READY_FOR_ACTUAL_HUMAN_TARGET_MATCH', generated_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(output), frame_count=12, reference_point_count=96,
        reference_point_sources=dict(point_sources),
        later_current_manual_coordinate_count=current_manual,
        later_current_self_occluded_count=current_hidden,
        original_8910_predictions_reused=True, original_direct_visible_contract_unchanged=True,
        original_review_files_unchanged=True, original_source_evaluation_flags_unchanged=True,
        machine_created_human_decisions=0, formal_human_review_promoted=False,
        actual_target_review_counts=context.counts(),
        checks=dict(checks), passed_check_count=sum(checks.values()),
        reference_binding=binding(reference_path), queue_binding=binding(queue_path),
        protocol_binding=binding(output / 'EVALUATION_PROTOCOL.json'),
        script_binding=binding(Path(__file__)), cpu_wall_seconds=time.perf_counter() - started,
        training_runs=0, optimizer_updates=0, new_model_forward_frames=0,
        hardware_control_calls=0, pdf_generation=0, remote_uploads=0, pushes=0)
    receipt = output / 'PREPARATION_RECEIPT.json'
    if receipt.exists():
        with receipt.with_suffix('.history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(read(receipt), ensure_ascii=False, allow_nan=False) + '\n')
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                       encoding='utf-8')
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    context = prepare_context(args.output)
    print(json.dumps(read(context.store_path.parent / 'PREPARATION_RECEIPT.json'),
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
