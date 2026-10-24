"""Independent standard-library-only V115 metadata replay. No payload loaders."""
import collections
import datetime
import hashlib
import json
import math
import struct
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLOCKERS, CHECKS, BINDINGS = [], collections.Counter(), {}


def binding(path):
    path = Path(path)
    assert path.suffix in ('.json', '.jsonl', '.py', '.md', '.txt', '.log')
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def check(value, name):
    CHECKS[name] += 1
    if not value:
        BLOCKERS.append(name)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return binding(HERE / name)


def positive_box(box):
    return isinstance(box, list) and len(box) == 4 and all(math.isfinite(v) for v in box) and box[2] > 0 and box[3] > 0


def endpoints(box):
    x, y, width, height = [F(str(v)) for v in box]
    return x, y, x + width, y + height


def overlap_ratio(box1, box2):
    a, b = endpoints(box1), endpoints(box2)
    overlap = max(F(0), min(a[2], b[2]) - max(a[0], b[0])) * max(F(0), min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else F(0)


def ratio_record(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def main():
    receipt = read(ROOT / 'METADATA_PREPARATION_RECEIPT.json')
    policy = read(ROOT / 'ROOT_SAM_CANDIDATE_POLICY_BEFORE_PIXELS.json')
    seal = read(ROOT / 'METADATA_PREPARATION_SEAL.json')
    # Only named metadata/code descriptors are followed; tensor/image/checkpoint references are never opened.
    for group in [receipt['bindings'], receipt['outputs']]:
        for key, expected in group.items():
            actual = binding(expected['path'])
            BINDINGS[str(expected['path'])] = actual
            check(all(actual[k] == expected[k] for k in ('path', 'sha256', 'bytes')), 'metadata_binding_exact')
    for expected in [seal['receipt'], seal['policy'], seal['code'], *seal['outputs'], policy['metadata_receipt'], policy['request_plan']]:
        actual = binding(expected['path'])
        check(all(actual[k] == expected[k] for k in expected), 'seal_and_root_policy_binding_exact')
    for name in ['METADATA_PREPARATION_RECEIPT.json', 'METADATA_PREPARATION_SEAL.json', 'ROOT_SAM_CANDIDATE_POLICY_BEFORE_PIXELS.json']:
        BINDINGS[str(ROOT / name)] = binding(ROOT / name)
    source = {k: v['path'] for k, v in receipt['bindings'].items()}
    frames, slots = rows(ROOT / 'FRAME20_ORIGINAL_QUERY_INPUTS_ONLY.jsonl'), rows(ROOT / 'QUERY160_ORIGINAL_INPUTS_ONLY.jsonl')
    targets = rows(ROOT / 'TARGET20_EXISTING_ANNOTATION_ENTITIES_GT_ONLY.jsonl')
    requests = rows(ROOT / 'SAM_REQUEST_PLAN40_METADATA_ONLY.jsonl')
    witnesses = rows(ROOT / 'EXPLICIT_SAME_ANN_ID_WEAK_VISIBLE_WITNESSES_GT_ONLY.jsonl')
    check(binding(ROOT / 'FRAME20_ORIGINAL_QUERY_INPUTS_ONLY.jsonl')['sha256'] == binding(source['selected_frame_inputs'])['sha256'], 'V92_frame_manifest_byte_exact')
    check(binding(ROOT / 'QUERY160_ORIGINAL_INPUTS_ONLY.jsonl')['sha256'] == binding(source['original_slot_inputs'])['sha256'], 'V92_slot_manifest_byte_exact')
    check((len(frames), len(slots), len(targets), len(requests), len(witnesses)) == (20, 160, 20, 40, 10), 'all_cohort_lengths')
    reps = read(source['representative_instance_lock'])['records']
    proposals = rows(source['initializer_proposals'])
    compact = rows(source['compact_input_lineage'])
    rgb_provenance = read(source['rgb_provenance'])['frames']
    groups = read(source['source_group_freeze'])
    coco = read(source['official_coco'])
    ann_by_id = {a['id']: a for a in coco['annotations']}
    check(len(ann_by_id) == len(coco['annotations']), 'official_annotation_ID_unique')
    ann_by_image = collections.defaultdict(list)
    for a in coco['annotations']:
        ann_by_image[a['image_id']].append(a)
    image_by_id = {a['id']: a for a in coco['images']}
    candidate_cues = {a['cue_refID']: a for a in rows(source['final_weak_cue_candidates'])}
    old_cues = {a['index']: a for a in read(source['final_weak_cue_addendum'])['frames'] if a['review_origin'] == 'ORIGINAL20'}
    later_cues = rows(source['later_crop_review_comparison'])
    check(not any('ann_id' in a or 'representative_annotation_id' in a for a in later_cues), 'later_unbound_crop_cues_not_identity_certificates')
    expected_witnesses, competitor_replay = [], []
    status_counts, fold_counts = collections.Counter(), collections.defaultdict(collections.Counter)
    actual_ties, missing, half_ties, nearest_rule_differences = [], [], [], []
    for i, (frame, target, rep, prop) in enumerate(zip(frames, targets, reps, proposals)):
        ann = rep['representative_annotation']
        check(rep['selection_index'] == target['frame_index'] == frame['frame_index'] == i, 'representative_original_order')
        check(frame['input_id'] == target['input_id'] == prop['input_id'], 'frame_identity')
        check(frame['image_sha256'] == target['image_sha256'] == rep['image_sha256'] == prop['image_sha256'], 'image_identity')
        check(ann == ann_by_id[ann['id']] == target['official_annotation_payload'], 'exact_preexisting_representative_entity')
        check(target['existing_target_ann_id'] == ann['id'] and ann['image_id'] == rep['image_id'] and ann['category_id'] == 7, 'representative_ID_and_category')
        check(target['bbox_xywh_raw'] == ann['bbox'] and target['box_xyxy_raw'] == list(map(float, endpoints(ann['bbox']))), 'exact_raw_target_boxes')
        im = image_by_id[ann['image_id']]
        check([im['height'], im['width']] == frame['raw_hw'] == target['raw_hw'] == rep['raw_hw'], 'native_HW_metadata')
        check(frame['officialpath_lineage'] == rep['official_path'] and Path(rep['official_path']).name == im['file_name'], 'official_path_identity')
        expected_fold = 'TRAIN' if frame['source_subset'] in groups['TRAIN_subsets'] else 'VALIDATION'
        check(frame['source_subset'] in groups['TRAIN_subsets'] + groups['VALIDATION_subsets'] and frame['split'] == target['split'] == expected_fold, 'unchanged_environment_fold')
        check(target['initializer_to_annotation_association'] == 'UNKNOWN' and target['certified_positive_pixel'] is None and not target['witness_is_certified_positive_pixel'], 'no_target_identity_or_pixel_gold_promotion')
        candidates = prop['candidates']
        if candidates:
            first_max = candidates.index(max(candidates, key=lambda a: a['score']))
            selected = candidates[first_max]
            check(first_max == prop['selected_index'] == frame['selected_index'] and len(candidates) == frame['candidate_count'], 'original_first_argmax_candidate')
            check(frame['query8_xy'] == selected['keypoints_xy'][:8] and frame['selected_predictor_box_xyxy'] == selected['box_xyxy'] and frame['selected_score'] == selected['score'], 'selected_query_and_box_unchanged')
        else:
            check(prop['no_detection'] and frame['selected_index'] is None and frame['query8_xy'] is None and frame['L'] is None, 'no_detection_unchanged')
        check(rgb_provenance[i]['source_RGB'] == frame['decoded_RGB_input'] and rgb_provenance[i]['decoded_pixel_SHA256'] == frame['RGB_pixels_sha256'], 'canonical_RGB_descriptors_only')
        ratios = [(overlap_ratio(ann['bbox'], a['bbox']), a) for a in ann_by_image[ann['image_id']] if a['id'] != ann['id'] and positive_box(a.get('bbox'))]
        ratios = [(r, a) for r, a in ratios if r > 0]
        ratios.sort(key=lambda ra: (-ra[0], ra[1]['id']))
        best = ratios[0] if ratios else None
        tied_ids = sorted(a['id'] for r, a in ratios if best and r == best[0])
        if len(tied_ids) > 1:
            actual_ties.append(dict(frame_index=i, tied_ann_ids=tied_ids))
        c = target['competitor']
        if best:
            ratio, other = best
            check(c is not None and c['ann_id'] == other['id'] and c['bbox_xywh_raw'] == other['bbox'] and c['box_xyxy_raw'] == list(map(float, endpoints(other['bbox']))), 'competitor_exact_max_positive_IoU_lowest_ID')
            check(c['raw_bbox_IoU_to_target'] == ratio_record(ratio) and c['same_physical_instance'] == 'UNKNOWN', 'competitor_exact_rational_and_unknown_identity')
        else:
            missing.append(i)
            check(c is None, 'missing_competitor_preserved')
        competitor_replay.append(dict(frame_index=i, input_id=frame['input_id'], target_ann_id=ann['id'], target_bbox_xywh=ann['bbox'], eligible_positive_candidates=len(ratios), competitor_ann_id=best[1]['id'] if best else None, exact_max_IoU=ratio_record(best[0]) if best else None, equal_max_tie_ann_ids=tied_ids))
        local_witnesses = [a for a in old_cues[i]['reference_proposals'] if a.get('conditional_aux_candidate')]
        check(target['visible_own_cue_witnesses'] == (local_witnesses or None), 'weak_regions_exact_old_records')
        for cue in local_witnesses:
            check(cue == candidate_cues[cue['cue_refID']] and cue['representative_annotation_id'] == ann['id'] and cue['image_id'] == ann['image_id'] and cue['image_sha256'] == frame['image_sha256'], 'weak_region_explicit_same_ann_and_image')
            check(cue['native8_mapping'] == 'UNKNOWN' and cue['center_pixel_support'] == 'UNKNOWN' and not cue['human_gold'] and not cue['native_semantic_gold'] and not cue['admitted_training_teacher'] and cue['training_rows_created'] == 0 and cue['initial_query_xy'] is None and cue['initializer_detection_or_raw_index_used'] is None, 'weak_region_no_promotion_or_query_binding')
            expected_witnesses.append(dict(frame_index=i, target_ann_id=ann['id'], original_record=cue))
        for role_index, role in enumerate(('TARGET', 'COMPETITOR')):
            req = requests[2 * i + role_index]
            req_ann = ann if role == 'TARGET' else (best[1] if best else None)
            check((req['request_index'], req['request_id'], req['frame_index'], req['input_id'], req['role']) == (2*i+role_index, f'V115_{i:02d}_{role}', i, frame['input_id'], role), 'all40_request_order')
            check(req['point_coords'] is None and req['point_labels'] is None and not req['executed'] and not req['box_clipping_or_adjustment_applied'], 'null_points_unexecuted_no_box_adjustment')
            check(req['canonical_decoded_RGB'] == frame['decoded_RGB_input'] and req['decoded_RGB_pixels_sha256'] == frame['RGB_pixels_sha256'] and req['image_sha256'] == frame['image_sha256'] and req['raw_hw'] == frame['raw_hw'] and req['split'] == frame['split'], 'request_RGB_descriptors_and_fold')
            check(req['call_eligible_metadata'] == (req_ann is not None), 'request_call_eligibility')
            check(req['status'] == ('PLANNED_BOX_REQUEST' if req_ann else 'NO_CALL_MISSING_COMPETITOR'), 'request_status')
            check(req['ann_id'] == (req_ann['id'] if req_ann else None) and req['bbox_xywh_raw'] == (req_ann['bbox'] if req_ann else None) and req['box_xyxy_raw'] == (list(map(float, endpoints(req_ann['bbox']))) if req_ann else None), 'request_annotation_and_box')
            check(req['positive_witness_ref_ids_available_not_prompted'] == ([a['cue_refID'] for a in local_witnesses] if role == 'TARGET' else []), 'witnesses_not_point_prompts')
            if req_ann:
                check(all(struct.unpack('f', struct.pack('f', v))[0] == v for v in req['box_xyxy_raw']), 'raw_prompt_coordinates_exact_FP32_representability')
        fold_counts[frame['split']]['frames'] += 1
        fold_counts[frame['split']]['witness_regions'] += len(local_witnesses)
        for native_slot in range(8):
            n = 8*i+native_slot
            slot, cache = slots[n], compact[n]
            q = None if frame['query8_xy'] is None else frame['query8_xy'][native_slot]
            check((slot['frame_index'], slot['input_id'], slot['native_slot'], slot['split'], slot['image_sha256'], slot['query_xy'], slot['L']) == (i, frame['input_id'], native_slot, frame['split'], frame['image_sha256'], q, frame['L']), 'all160_order_qL_native_fold')
            check((cache['slot_index'], cache['input_id'], cache['native_slot_logical_only'], cache['split'], cache['query_xy'], cache['L'], cache['original_input_valid']) == (n, slot['input_id'], native_slot, slot['split'], q, slot['L'], slot['query_center_valid']), 'all160_base_cache_metadata_join')
            if q is None:
                valid, continuous, nearest, status = False, False, False, 'NO_DETECTION_QUERY_U'
            else:
                x, y = q
                h, w = frame['raw_hw']
                continuous = 0 <= x < w and 0 <= y < h
                nearest = 0 <= math.floor(x+.5) < w and 0 <= math.floor(y+.5) < h
                ties_even_nearest = 0 <= round(x) < w and 0 <= round(y) < h
                if any(v % 1 == .5 for v in q):
                    half_ties.append(n)
                if nearest != ties_even_nearest:
                    nearest_rule_differences.append(n)
                valid = continuous and nearest
                status = 'SAVED_UNLABELED_QUERY' if valid else 'OUTSIDE_OR_INVALID_QUERY_U'
            check(slot['query_exists'] == (q is not None) and slot['continuous_valid'] == continuous and slot['nearest_valid'] == nearest and slot['query_center_valid'] == valid and slot['input_status'] == status, 'all160_factual_validity_recomputed')
            check(slot['semantic_state'] == 'UNLABELED_UNKNOWN' and slot['native8_semantic_mapping'] == 'UNKNOWN' and not slot['new_training_label'], 'all160_unlabeled_native_unknown')
            status_counts[status] += 1
            fold_counts[frame['split']]['slots'] += 1
            fold_counts[frame['split']][status] += 1
    check(witnesses == expected_witnesses, 'all10_witness_rows_exact_order_and_contents')
    check(missing == [4,5,8,9,12,14,19] == receipt['missing_competitor_no_call_frames'], 'seven_missing_frames_exact')
    check(sorted(set(a['frame_index'] for a in witnesses)) == [0,1,8,16,17] == receipt['witness_frame_indices'], 'five_witness_frames_exact')
    check(dict(fold_counts) == receipt['fold_counts'], 'all_fold_counts_match_receipt')
    check(dict(status_counts) == {k:v for k,v in receipt['counts'].items() if k in status_counts}, 'query_status_counts_match_receipt')
    check(sum(a['call_eligible_metadata'] for a in requests) == receipt['planned_model_calls'] == 33, '33_primary_calls_and7_null_requests')
    check(not nearest_rule_differences, 'root_numpy_rint_nearest_bounds_match_fixed_metadata')
    check(not policy['training_admission'] and policy['current223_reads'] == 0 and policy['V112_reads'] == 0 and policy['coordinate_updates'] == 0 and policy['PnP'] == 0, 'root_policy_semantic_and_access_limits')
    check('20 encoder calls' in policy['model_operator'] and 'Total34 predict calls' in policy['repeat_check'] and 'Save33 native bool-mask NPZs' in policy['outputs'] and '16 noquery and22 OOB' in policy['query_readout'], 'root_policy_counts_match_locked_plan')
    computed = write('INDEPENDENT_COMPETITOR_REPLAY20.json', competitor_replay)
    inventory = write('VERIFIED_METADATA_BINDINGS.json', list(BINDINGS.values()))
    report = dict(schema='V115_INDEPENDENT_METADATA_REVIEW_V1', completed=True, PASS=not BLOCKERS, blockers=BLOCKERS, reviewer='A /root/head_implementation', created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        preserved_checker_wording_failure=binding(HERE/'provenance/FAILED_REVIEW_LITERAL_PHRASE_CHECK_V1/FAILURE_EXPLANATION.json'), metadata_receipt=binding(ROOT/'METADATA_PREPARATION_RECEIPT.json'), root_policy=binding(ROOT/'ROOT_SAM_CANDIDATE_POLICY_BEFORE_PIXELS.json'), checks=dict(CHECKS),
        counts=dict(frames=20, targets=20, competitors=13, missing_competitors=7, request_slots=40, primary_calls=33, predetermined_repeat_calls=1, total_predict_calls_prospective=34, queries=160, query_status=dict(status_counts), fold_counts=dict(fold_counts), weak_regions=10, weak_region_frames=[0,1,8,16,17], initializer_annotation_equivalence_certified=0, certified_positive_pixels=0),
        exact_rational_ties=actual_ties, nearest_half_integer_queries=half_ties, nearest_bounds_disagreement_queries=nearest_rule_differences,
        root_policy_inconsistencies=[], nonblocking_notes=[
            'The locked pre-policy request records retain full_execution_policy_fixed=false and the earlier receipt execution_ready=false. These correctly record preparation chronology; the later root policy fixes the prospective operator and requires a separate execution entry. Do not interpret those historical flags as execution approval or mutate the locked metadata.',
            'Exact rational selection uses all other valid official annotations in the same image, not only pallet annotations. Another ann_id can describe overlapping/merged/same physical material; its identity relation remains UNKNOWN.',
            'Raw XYXY coordinates in all33 eligible requests are exactly representable in FP32. No manual clipping or inward adjustment is required or authorized.',
            'No actual half-integer query occurs. Original nearest-bounds metadata and the prospective NumPy rint readout therefore agree on all160 rows.',
            'Whole V92 frame and slot manifests are byte-identical, including original crop-validity metadata. This metadata review does not inspect/recompute any cached crop pixels or mask pixels.'
        ],
        caveats=[
            'PASS certifies existing metadata bindings, numerical joins and policy consistency only; it does not certify SAM output quality, API implementation or execution.',
            'Ten weak same-ann-ID visible regions are preserved exactly. Their centre pixel remains UNKNOWN, they are not point prompts or human/native gold, and no association to initializer-selected instances is inferred.',
            'Payload descriptors are checked only against existing metadata. RGB/NPY/NPZ/checkpoint bytes were not read, hashed or decoded. No model/array/tensor library was imported.',
            'Public source folds are prior environment partitions, not a new unseen-site or blind generalization claim. Candidate mask absence cannot create external E or geometric S.',
            'SAM weights/runtime behavioral verification, RGB lineage replay and all mask/readout diagnostics remain future execution-stage responsibilities.'
        ], operations=dict(metadata_only=True, payload_reads=0, payload_rehashes=0, RGB_decodes=0, model_imports=0, model_calls=0, checkpoint_reads=0, current_or_V112_reads=0, network=0, fitting=0, coordinate_changes=0, PnP=0), independent_competitor_replay=computed, metadata_bindings=inventory, scope=binding(HERE/'SCOPE.json'), review_code=binding(__file__))
    report_binding = write('REVIEW.json', report)
    seal_binding = write('SEAL.json', dict(completed=True, PASS=not BLOCKERS, review=report_binding, files=[binding(HERE/'SCOPE.json'), binding(__file__), computed, inventory], old_files_modified=False, operations=report['operations']))
    print(json.dumps(dict(PASS=not BLOCKERS, blockers=BLOCKERS, review=report_binding, seal=seal_binding, counts=report['counts'], exact_rational_ties=actual_ties, check_count=sum(CHECKS.values())), indent=2))


if __name__ == '__main__':
    main()
