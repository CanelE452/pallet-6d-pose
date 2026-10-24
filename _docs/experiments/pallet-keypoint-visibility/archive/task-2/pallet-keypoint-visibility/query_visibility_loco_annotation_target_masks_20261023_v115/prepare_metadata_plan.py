"""Standard-library, metadata-only V115 plan; never opens pixel/weight payloads."""
import collections
import datetime
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
HERE = Path(__file__).resolve().parent
V73 = P / 'query_visibility_public_pallet_loco_20261023_v73'
V74 = P / 'query_visibility_loco_native_context_20261023_v74'
V92 = P / 'query_visibility_flat_pallet_cue_bridge_20261023_v92'
V94 = P / 'query_visibility_matched_query_scalar_20261023_v94'
V95 = P / 'query_visibility_same_q_counterfactual_20261023_v95'
PATHS = {
    'policy': HERE / 'METADATA_PLAN_POLICY_BEFORE_MASKS.json',
    'representative_instance_lock': V74 / 'public_reference_review/REPRESENTATIVE_INSTANCE_LOCK.json',
    'reference_geometry_lock': V74 / 'public_reference_review/RGB_REFERENCE_GEOMETRY_LOCK.json',
    'final_weak_cue_addendum': V74 / 'public_reference_review/COMBINED60_REFERENCE_REVIEW_ADDENDUM.json',
    'final_weak_cue_candidates': V74 / 'public_reference_review/COMBINED60_AUX_CANDIDATES.jsonl',
    'selected_frame_inputs': V92 / 'rgb96_preparation/FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl',
    'original_slot_inputs': V92 / 'rgb96_review/ROWS_INPUT_ONLY.jsonl',
    'rgb96_seal': V92 / 'rgb96_review/RGB96_REVIEW_SEAL.json',
    'rgb_provenance': V92 / 'rgb96_review/RGB_PROVENANCE_AND_FIGURES.json',
    'initializer_proposals': V74 / 'public_initializer/DECODED_RGB_V2_RUN/FROZEN_PROPOSALS.jsonl',
    'source_group_freeze': V74 / 'source_groups/SOURCE_GROUP_FREEZE.json',
    'official_coco': V73 / 'data/intake/annotations/loco-all-v1.json',
    'official_annotation_receipt': V73 / 'data/intake/ANNOTATION_DOWNLOAD_RECEIPT.json',
    'prior_metadata_review': V94 / 'source_metadata_loco/REVIEW.json',
    'later_crop_review_comparison': V94 / 'aux_cue_review_comparison/ALL160_B_C_LOCKED_METADATA_COMPARISON.jsonl',
    'compact_input_seal': V95 / 'loco_before_input_pack/INPUT_PACK_SEAL.json',
    'compact_input_lineage': V95 / 'loco_before_input_pack/BEFORE160_INPUT_LINEAGE.jsonl',
    'public_feasibility_before_C12_current': P / 'query_visibility_ycb_wood_foam_source_20261023_v113/public_pallet_source_feasibility_next/SOURCE_FEASIBILITY_NEXT.json',
    'cached_sam_runtime_seal': P / 'foundation_runtime_sam21_20261023/RUNTIME_SEAL.json',
    'preserved_prefix_assertion_failure': HERE / 'provenance/FAILED_PREFIX_PARITY_ASSERTION_V1/FAILURE.json',
}

def binding(p):
    b = p.read_bytes()  # This function is called only for listed metadata/code files.
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}

def read(p):
    return json.loads(p.read_text())

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]

def write(name, obj):
    p = HERE / name
    with p.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    return binding(p)

def write_rows(name, rr):
    p = HERE / name
    with p.open('x') as f:
        for r in rr:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    return binding(p)

def xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def iou(a, b):
    a, b = [Fraction(str(x)) for x in a], [Fraction(str(x)) for x in b]
    iw = max(Fraction(0), min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(Fraction(0), min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    it = iw * ih
    den = a[2] * a[3] + b[2] * b[3] - it
    return it / den if den > 0 else Fraction(0)

def valid_box(b):
    return len(b) == 4 and all(math.isfinite(x) for x in b) and b[2] > 0 and b[3] > 0

def fraction_record(f):
    return {'numerator': f.numerator, 'denominator': f.denominator, 'value': float(f)}

def main():
    assert not (HERE / 'METADATA_PREPARATION_SEAL.json').exists(), 'No overwrite/restart'
    binds = {k: binding(p) for k, p in PATHS.items()}
    binds['code'] = binding(Path(__file__))
    frames = rows(PATHS['selected_frame_inputs'])
    slots = rows(PATHS['original_slot_inputs'])
    proposals = rows(PATHS['initializer_proposals'])
    reps = read(PATHS['representative_instance_lock'])['records']
    provenance = read(PATHS['rgb_provenance'])['frames']
    compact = rows(PATHS['compact_input_lineage'])
    coco = read(PATHS['official_coco'])
    receipt = read(PATHS['official_annotation_receipt'])
    official = next(x for x in receipt['files'] if x['path'] == str(PATHS['official_coco']))
    assert binds['official_coco']['sha256'] == official['sha256']
    assert len(frames) == len(reps) == len(proposals) == len(provenance) == 20
    assert len(slots) == len(compact) == 160
    assert len({r['input_id'] for r in frames}) == 20
    rep_by_image = collections.defaultdict(list)
    for r in reps:
        rep_by_image[r['image_sha256']].append(r)
    image_by_id = {r['id']: r for r in coco['images']}
    ann_by_id = {r['id']: r for r in coco['annotations']}
    assert len(ann_by_id) == len(coco['annotations'])
    ann_by_image = collections.defaultdict(list)
    for a in coco['annotations']:
        ann_by_image[a['image_id']].append(a)
    categories = {r['id']: r['name'] for r in coco['categories']}
    assert categories[7].lower() == 'pallet'
    addendum = read(PATHS['final_weak_cue_addendum'])
    original_cues = {r['index']: r for r in addendum['frames'] if r['review_origin'] == 'ORIGINAL20'}
    final_candidates = {r['cue_refID']: r for r in rows(PATHS['final_weak_cue_candidates'])}
    later_reviews = rows(PATHS['later_crop_review_comparison'])
    assert len(later_reviews) == 160
    assert not any('representative_annotation_id' in r or 'ann_id' in r for r in later_reviews)
    rgblock = read(PATHS['rgb96_seal'])
    compact_seal = read(PATHS['compact_input_seal'])
    assert rgblock['PASS'] and compact_seal['PASS']
    # Payload bindings below are copied from existing seals, never read/rehashed here.
    rgb_pack = next(x for x in rgblock['files'] if x['path'].endswith('INPUT_RGB96_ONLY.npz'))
    compact_pack = compact_seal['inputs']
    assert Path(rgb_pack['path']).is_file() and Path(compact_pack['path']).is_file()
    planned, entities, witness_ledger = [], [], []
    counts = collections.Counter()
    fold_counts = collections.defaultdict(collections.Counter)
    for i, f in enumerate(frames):
        assert f['frame_index'] == i
        p = proposals[i]
        assert f['input_id'] == p['input_id'] and f['image_sha256'] == p['image_sha256']
        assert provenance[i]['input_id'] == f['input_id']
        assert provenance[i]['source_RGB'] == f['decoded_RGB_input']
        assert provenance[i]['decoded_pixel_SHA256'] == f['RGB_pixels_sha256']
        if p['candidates']:
            sel = max(range(len(p['candidates'])), key=lambda j: p['candidates'][j]['score'])
            c = p['candidates'][sel]
            assert sel == f['selected_index'] == p['selected_index']
            assert c['keypoints_xy'][:8] == f['query8_xy']
            assert c['box_xyxy'] == f['selected_predictor_box_xyxy']
            assert c['score'] == f['selected_score']
        else:
            assert p['no_detection'] and f['query8_xy'] is None and f['L'] is None
        for k in range(8):
            j = i * 8 + k
            s, cc = slots[j], compact[j]
            assert (s['input_id'], s['frame_index'], s['native_slot']) == (f['input_id'], i, k)
            assert s['image_sha256'] == f['image_sha256'] and s['split'] == f['split']
            expected = None if f['query8_xy'] is None else f['query8_xy'][k]
            assert s['query_xy'] == expected and s['L'] == f['L']
            assert cc['slot_index'] == j and cc['query_xy'] == expected and cc['L'] == f['L']
            assert cc['native_slot_logical_only'] == k and cc['split'] == f['split']
            assert cc['decoded_RGB_sha256'] == f['RGB_pixels_sha256']
            assert cc['original_input_valid'] == s['query_center_valid']
            if s['query_xy'] is not None:
                x, y = s['query_xy']; h, w = f['raw_hw']
                continuous = 0 <= x < w and 0 <= y < h
                nearest = 0 <= math.floor(x + .5) < w and 0 <= math.floor(y + .5) < h
                assert continuous == s['continuous_valid'] and nearest == s['nearest_valid']
                assert s['query_center_valid'] == (continuous and nearest)
            counts['slots'] += 1
            counts[s['input_status']] += 1
            fold_counts[f['split']]['slots'] += 1
            fold_counts[f['split']][s['input_status']] += 1
        candidates = rep_by_image[f['image_sha256']]
        target = candidates[0] if len(candidates) == 1 else None
        witness = []
        competitor = None
        entity = {'frame_index': i, 'input_id': f['input_id'], 'split': f['split'],
                  'source_subset': f['source_subset'], 'image_sha256': f['image_sha256'],
                  'raw_hw': f['raw_hw'], 'existing_target_ann_id': None,
                  'annotation_entity_status': 'MISSING_OR_NONUNIQUE_EXISTING_TARGET',
                  'initializer_to_annotation_association': 'UNKNOWN',
                  'visible_own_cue_witnesses': None, 'certified_positive_pixel': None}
        if target is not None:
            a = target['representative_annotation']; iid = target['image_id']
            assert target['selection_index'] == i and target['raw_hw'] == f['raw_hw']
            assert a == ann_by_id[a['id']] and a['image_id'] == iid and a['category_id'] == 7
            im = image_by_id[iid]
            assert [im['height'], im['width']] == f['raw_hw']
            assert f['officialpath_lineage'] == target['official_path']
            assert im['file_name'] == Path(f['officialpath_lineage']).name
            assert valid_box(a['bbox'])
            for c in original_cues[i]['reference_proposals']:
                assert c['representative_annotation_id'] == a['id'] and c['image_id'] == iid
                assert c['image_sha256'] == f['image_sha256']
                if c.get('conditional_aux_candidate'):
                    assert c['cue_refID'] in final_candidates and c == final_candidates[c['cue_refID']]
                    witness.append(c)  # Entire old weak record; no revised coordinate or state.
                    witness_ledger.append({'frame_index': i, 'target_ann_id': a['id'], 'original_record': c})
            eligible = [(iou(a['bbox'], other['bbox']), other) for other in ann_by_image[iid]
                        if other['id'] != a['id'] and valid_box(other.get('bbox', []))]
            eligible = [(v, other) for v, other in eligible if v > 0]
            if eligible:
                best_iou, other = min(eligible, key=lambda z: (-z[0], z[1]['id']))
                competitor = {'ann_id': other['id'], 'category_id': other['category_id'],
                              'category': categories[other['category_id']], 'bbox_xywh_raw': other['bbox'],
                              'box_xyxy_raw': xyxy(other['bbox']), 'raw_bbox_IoU_to_target': fraction_record(best_iou),
                              'same_physical_instance': 'UNKNOWN', 'selection_not_mask_quality_based': True}
            pb = f['selected_predictor_box_xyxy']
            diagnostic = None if pb is None else fraction_record(iou(a['bbox'], [pb[0], pb[1], pb[2]-pb[0], pb[3]-pb[1]]))
            entity.update(existing_target_ann_id=a['id'], annotation_entity_status='EXACT_PREEXISTING_OFFICIAL_ENTITY',
                          official_image_id=iid, official_COCO_file_name_raw=im['file_name'],
                          official_archive_path=f['officialpath_lineage'],
                          bbox_xywh_raw=a['bbox'], box_xyxy_raw=xyxy(a['bbox']),
                          official_annotation_payload=a, official_annotation_count=len(ann_by_image[iid]),
                          official_pallet_annotation_count=sum(x['category_id'] == 7 for x in ann_by_image[iid]),
                          original_lock_review_id=target['review_id'],
                          initializer_box_IoU_diagnostic_only=diagnostic,
                          visible_own_cue_witnesses=witness or None,
                          witness_status='EXPLICIT_SAME_ANN_ID_WEAK_REGION' if witness else 'NO_EXPLICIT_SAME_ANN_ID_WITNESS',
                          witness_is_certified_positive_pixel=False,
                          competitor=competitor)
            counts['targets'] += 1
            counts['witness_regions'] += len(witness)
            counts['frames_with_witness'] += bool(witness)
            counts['competitors'] += competitor is not None
            fold_counts[f['split']]['frames'] += 1
            fold_counts[f['split']]['witness_regions'] += len(witness)
            for role, ann in [('TARGET', {'ann_id': a['id'], 'bbox_xywh_raw': a['bbox'], 'box_xyxy_raw': xyxy(a['bbox'])}),
                              ('COMPETITOR', competitor)]:
                planned.append({'request_index': len(planned), 'request_id': f'V115_{i:02d}_{role}',
                                'frame_index': i, 'input_id': f['input_id'], 'role': role,
                                'status': 'PLANNED_BOX_REQUEST' if ann else 'NO_CALL_MISSING_COMPETITOR',
                                'call_eligible_metadata': ann is not None, 'executed': False,
                                'image': f['image'], 'image_sha256': f['image_sha256'],
                                'canonical_decoded_RGB': f['decoded_RGB_input'],
                                'decoded_RGB_pixels_sha256': f['RGB_pixels_sha256'], 'raw_hw': f['raw_hw'],
                                'split': f['split'], 'ann_id': ann['ann_id'] if ann else None,
                                'bbox_xywh_raw': ann['bbox_xywh_raw'] if ann else None,
                                'box_xyxy_raw': ann['box_xyxy_raw'] if ann else None,
                                'point_coords': None, 'point_labels': None,
                                'box_clipping_or_adjustment_applied': False,
                                'positive_witness_ref_ids_available_not_prompted': [x['cue_refID'] for x in witness] if role == 'TARGET' else [],
                                'full_execution_policy_fixed': False, 'root_execution_policy_required': True,
                                'output_meaning': 'CANDIDATE_ANNOTATION_ENTITY_OWNERSHIP_NOT_GT'})
        else:
            for role in ['TARGET', 'COMPETITOR']:
                planned.append({'request_index': len(planned), 'request_id': f'V115_{i:02d}_{role}',
                                'frame_index': i, 'input_id': f['input_id'], 'role': role,
                                'status': 'NO_CALL_MISSING_OR_NONUNIQUE_TARGET', 'call_eligible_metadata': False,
                                'executed': False, 'ann_id': None, 'point_coords': None, 'point_labels': None})
        entities.append(entity)
    assert counts['targets'] == 20 and counts['slots'] == 160
    assert counts['witness_regions'] == 10 and counts['frames_with_witness'] == 5
    assert len(planned) == 40
    outputs = {}
    # Exact bytes and schema retained for both input-only manifests.
    for name, source in [('FRAME20_ORIGINAL_QUERY_INPUTS_ONLY.jsonl', PATHS['selected_frame_inputs']),
                         ('QUERY160_ORIGINAL_INPUTS_ONLY.jsonl', PATHS['original_slot_inputs'])]:
        with (HERE / name).open('xb') as f:
            f.write(source.read_bytes())
        outputs[name] = binding(HERE / name)
        assert outputs[name]['sha256'] == binding(source)['sha256']
    outputs['target_entities'] = write_rows('TARGET20_EXISTING_ANNOTATION_ENTITIES_GT_ONLY.jsonl', entities)
    outputs['witnesses'] = write_rows('EXPLICIT_SAME_ANN_ID_WEAK_VISIBLE_WITNESSES_GT_ONLY.jsonl', witness_ledger)
    outputs['request_plan'] = write_rows('SAM_REQUEST_PLAN40_METADATA_ONLY.jsonl', planned)
    runtime = read(PATHS['cached_sam_runtime_seal'])
    report = {
        'schema': 'V115_METADATA_PREPARATION_RECEIPT_V1', 'completed': True, 'PASS': True,
        'counts': dict(counts), 'fold_counts': {k: dict(v) for k, v in fold_counts.items()},
        'request_slots': 40, 'planned_model_calls': sum(r['call_eligible_metadata'] for r in planned),
        'missing_competitor_no_call_frames': [r['frame_index'] for r in planned if r['status'] == 'NO_CALL_MISSING_COMPETITOR'],
        'witness_frame_indices': [r['frame_index'] for r in entities if r['visible_own_cue_witnesses']],
        'certified_positive_pixel_count': 0,
        'actual_initializer_target_to_annotation_certified_count': 0,
        'target_semantics': 'All20 exact preexisting representative annotation entities; physical target equivalence to actual initializer selected detection remains UNKNOWN. Diagnostic IoU does not resolve identity.',
        'witness_semantics': '10 prelocked same-ann_id weak visible regions on5frames; center_pixel_support UNKNOWN. No point prompt was invented. Fifteen frames retain None.',
        'later_crop_cues': 'V94 B/C conditional 2D crop cues carry no explicit ann_id. They remain unbound; no later D6/query-anchor cue group was treated as an ann_id certificate.',
        'source_partitions': 'Preserved official environments TRAIN subsets2/3/5, VALIDATION subsets1/4, not a new unseen-site guarantee.',
        'source_input_payloads': {'rgb96_V92': rgb_pack, 'three_tensor_V95': compact_pack,
                                  'verification': 'Existing seals and metadata joins only. Payload file existence stat; no payload bytes/arrays read or rehashed.'},
        'official_annotation_source': official,
        'official_mask_semantics': 'COCO bbox/segmentation metadata is not certified visible-own/amodal pixel ownership. SAM outputs will remain predictions, not labels.',
        'cached_later_runtime': {'runtime_seal': binds['cached_sam_runtime_seal'], 'python': runtime['python'],
                                 'repo_revision': runtime['repo_revision'], 'checkpoint_recorded_binding': runtime['checkpoint'],
                                 'checkpoint_loaded_or_rehashed_here': False, 'downloads': 0},
        'public_causal_provenance': {'prior_feasibility': binds['public_feasibility_before_C12_current'],
                                    'timing': 'Parent records this public feasibility plan as fixed before C12 current results. V115 selects no targets/competitors/witnesses using current or holdout metrics.'},
        'bindings': binds, 'outputs': outputs,
        'operations': {'metadata_reads': True, 'new_pixel_or_array_payload_reads': 0, 'model_or_SAM_imports': 0,
                       'checkpoint_deserializations': 0, 'model_calls': 0, 'network_downloads': 0,
                       'fit': 0, 'private_current_V112_reads': 0, 'PnP': 0, 'old_source_mutations': 0},
        'limitations': ['No official target-owned pixel gold', 'No certified initializer-to-ann_id correspondence',
                        'Competitor annotation may duplicate/merge the same physical pallet',
                        'Cue region visibility does not certify its exact center pixel',
                        'Mask absence is not external E; geometric S and unknown U remain separate',
                        'No native mapping, human gold, or training admission'],
        'training_admission': False, 'native_mapping': 'UNKNOWN', 'human_gold': False,
        'execution_ready': False, 'next_step': 'Root fixes complete prospective SAM policy and exact entry before pixels/model calls.',
        'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    review = write('METADATA_PREPARATION_RECEIPT.json', report)
    seal = write('METADATA_PREPARATION_SEAL.json', {'completed': True, 'PASS': True,
                 'receipt': review, 'policy': binds['policy'], 'code': binds['code'],
                 'outputs': list(outputs.values()), 'model_calls': 0, 'training_admission': False})
    print(json.dumps({'counts': dict(counts), 'planned_calls': report['planned_model_calls'],
                      'receipt': review, 'seal': seal}, indent=2))

if __name__ == '__main__':
    main()
