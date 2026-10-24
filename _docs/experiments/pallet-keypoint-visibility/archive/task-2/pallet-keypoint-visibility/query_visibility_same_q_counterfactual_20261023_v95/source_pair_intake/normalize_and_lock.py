"""Metadata-only exact960 normalization and one-donor lock before pixel intake."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
V94 = P / 'query_visibility_matched_query_scalar_20261023_v94'
V92 = P / 'query_visibility_flat_pallet_cue_bridge_20261023_v92'
V74 = P / 'query_visibility_loco_native_context_20261023_v74'
V3 = V94 / 'paired_source_preparation/v3_integrated_same_q_builder/integrated_same_q_v3.py'


def bind(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def jsonl(p, values):
    with Path(p).open('x') as f:
        for x in values:
            f.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    names = {
        'loco_rows': V92 / 'rgb96_review/ROWS_INPUT_ONLY.jsonl',
        'loco_frames': V92 / 'rgb96_preparation/FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl',
        'loco_gt': V94 / 'root_actual_cue_review/ALL160_ROOT_AUXILIARY_DECISIONS_GT_ONLY.jsonl',
        'pallox_gt': V94 / 'conditional_source_fit_contract_preparation/CLEAN_ALL_SAME_ALL800_PROPOSAL_GT_ONLY.jsonl',
        'pallox_frames': V94 / 'pallox_actual_initializer/FIXED100/FROZEN_PROPOSALS_ALL100.jsonl',
        'pallox_query_lineage': V94 / 'pallox_actual_query_input_pack/QUERY_INPUT_LINEAGE_ONLY.jsonl',
        'pallox_pack': V94 / 'pallox_actual_query_input_pack/PACK_MANIFEST.json',
        'loco_p3_receipt': V74 / 'public_initializer/DECODED_RGB_V2_RUN/INFERENCE_RESULT.json',
        'loco_rgb_receipt': V92 / 'rgb96_review/RGB96_REVIEW_SEAL.json',
        'donor_metadata': V94 / 'paired_source_preparation/YCB4200_DONOR_METADATA_CANDIDATES_NOT_ADMITTED.jsonl'}
    v3spec = importlib.util.spec_from_file_location('v95_pinned_v3_metadata', V3)
    v3 = importlib.util.module_from_spec(v3spec); v3spec.loader.exec_module(v3)
    loco, lf, lg = lines(names['loco_rows']), lines(names['loco_frames']), lines(names['loco_gt'])
    pg, pf, pq = lines(names['pallox_gt']), lines(names['pallox_frames']), lines(names['pallox_query_lineage'])
    assert len(loco) == len(lg) == 160 and len(pg) == len(pq) == 800 and len(lf) == 20 and len(pf) == 100
    lrgb = next(x for x in read(names['loco_rgb_receipt'])['files'] if x['path'].endswith('INPUT_RGB96_ONLY.npz'))
    lp3 = next(x for x in read(names['loco_p3_receipt'])['outputs'] if x['path'].endswith('P3_INPUT_FEATURES_ONLY.npz'))
    pm = bind(names['pallox_pack'])
    lg_bind, pg_bind = bind(names['loco_gt']), bind(names['pallox_gt'])
    normalized, lineage, decisions = [], [], []
    for i, (r, d) in enumerate(zip(loco, lg)):
        f = lf[r['frame_index']]
        assert d['query_key'] == [r['input_id'], r['native_slot']] and d['original_q'] == r['query_xy'] and d['original_L'] == r['L']
        row = dict(slot_index=i, frame_index=r['frame_index'], input_id=r['input_id'], native_slot_logical_only=r['native_slot'],
                   split=r['split'], image_sha256=r['image_sha256'], raw_hw=f['raw_hw'], query_xy=r['query_xy'], L=r['L'],
                   selected_box_xyxy=f['selected_predictor_box_xyxy'], original_input_valid=r['query_center_valid'])
        v3.checked_row(row); v3.decision_identity(row, d)
        normalized.append(row)
        lineage.append(dict(base_row_index=i, source='LOCO', original_row_index=i, frame_index=r['frame_index'],
                            selected_candidate_index=f['selected_index'], selected_candidate_score=f['selected_score'],
                            original_query_continuous_valid=r['continuous_valid'], query_nearest_valid=r['nearest_valid'],
                            original_input_status=r['input_status'], fullRGB_path=f['decoded_RGB_input']['path'],
                            fullRGB_format='CANONICAL_RGB_UINT8_HWC_NPY', fullRGB_file_binding=f['decoded_RGB_input'],
                            fullRGB_original_image=f['image'], original_image_sha256=r['image_sha256'],
                            before_decoded_sha256=f['RGB_pixels_sha256'], source_family='LOCO_EXISTING_SUBSET_FAMILY_SPLIT',
                            before_cache=dict(kind='LOCO_V92_RGB96_PLUS_V74_FULL_P3',
                                              rgb96=dict(file=lrgb, key='rgb96', index=i),
                                              crop_valid96=dict(file=lrgb, key='crop_valid96', index=i),
                                              full_p3=dict(file=lp3, key='p3', index=r['frame_index']),
                                              ROI_action='V90_FROZEN_SAMPLER_SAME_q_L_NOT_STORED_ROI')))
        dec = dict(d)
        dec.update(base_row_index=i, source='LOCO', original_natural_label_GT_only=0 if d['before_aux_state'] == 'D' else -1,
                   protected_regions_GT_only=[dict(kind='LOCO_HALF_OPEN_NATIVE_PIXEL_EDGE', support_bbox=d['support_bbox_xyxy'],
                                                   uncertainty_bbox=d['uncertainty_bbox_xyxy'])] if d['before_aux_state'] == 'D' else [],
                   null_unlocated_limit_count=None, GT_source=lg_bind)
        decisions.append(dec)
    for j, (g, q) in enumerate(zip(pg, pq)):
        i = 160 + j; f = pf[g['frame_index']]; k = g['native_slot']; s = f['selected_index']
        assert g['query_id'] == q['query_id'] and g['original_q'] == q['query_xy'] and g['original_L'] == q['roi_side']
        box = None if s is None else f['candidates'][s]['box_xyxy']
        available = bool(q['original_query_continuous_valid'] and q['query_nearest_pixel_valid'] and q['query_rgb96_centre_domain_valid'])
        row = dict(slot_index=i, frame_index=g['frame_index'], input_id=g['frame_input_id'], native_slot_logical_only=k,
                   split=g['split'], image_sha256=g['image_sha256'], raw_hw=g['raw_hw'], query_xy=g['original_q'], L=g['original_L'],
                   selected_box_xyxy=box, original_input_valid=available)
        v3.checked_row(row); normalized.append(row)
        lineage.append(dict(base_row_index=i, source='PALLOX', original_row_index=j, original_query_id=g['query_id'],
                            frame_index=g['frame_index'], pair_id=g['pair_id'], view=g['view'], seed_V84_reuse=g['seed_V84_reuse'],
                            selected_candidate_index=s, selected_candidate_score=None if s is None else f['candidates'][s]['score'],
                            original_query_continuous_valid=q['original_query_continuous_valid'], query_nearest_valid=q['query_nearest_pixel_valid'],
                            query_rgb96_centre_domain_valid=q['query_rgb96_centre_domain_valid'], original_input_status=q['reason'],
                            fullRGB_path=g['image'], fullRGB_format='PNG_RGB', fullRGB_file_binding=dict(path=g['image'], sha256=g['image_sha256']),
                            fullRGB_original_image=g['image'], original_image_sha256=g['image_sha256'], before_decoded_sha256=None,
                            source_family=g['source_family'], before_cache=dict(kind='PALLOX_V94_AUDITED_QUERY_PACK', manifest=pm, query_index=j)))
        beforeD = g['scalar_target_D0_E1_Uminus1'] == 0
        regions = []
        if beforeD:
            assert g['retained_FULL_GT_cues'] and all(x['teacher_label'] == 0 for x in g['retained_FULL_GT_cues'])
            regions = [dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY', bbox=x['cue_uncertainty_bbox_unchanged'],
                            cue_id=x['cue_id'], cue_view_index=x['cue_view_index'], r_unchanged=x['cue_reference_xy_unchanged'],
                            certified_duplicate_group=x['certified_duplicate_group']) for x in g['retained_FULL_GT_cues']]
        dec = dict(base_row_index=i, source='PALLOX', query_key=[row['input_id'], k], input_id=row['input_id'],
                   frame_index=g['frame_index'], native_slot=k, split=g['split'], image_sha256=g['image_sha256'],
                   original_q=g['original_q'], original_L=g['original_L'], before_aux_state='D' if beforeD else 'U',
                   paired_construction_eligible=beforeD, support_bbox_xyxy=None, uncertainty_bbox_xyxy=None,
                   protected_regions_GT_only=regions, protected_union_not_enclosing_rectangle=True,
                   original_natural_label_GT_only=g['scalar_target_D0_E1_Uminus1'],
                   original_natural_state_GT_only=g['candidate_state'], null_unlocated_limit_count=g['null_unlocated_count'],
                   retained_FULL_GT_cues=g['retained_FULL_GT_cues'], original_C1_ledger_row_index=j,
                   native_expected_part_state='UNKNOWN', human_gold=False, training_admitted=False, GT_source=pg_bind)
        v3.decision_identity(row, dec); decisions.append(dec)
    donors = lines(names['donor_metadata'])
    assigned = []
    for row in normalized:
        d = v3.choose_one_donor(row, donors)
        assigned.append(dict(base_row_index=row['slot_index'], query_key=[row['input_id'], row['native_slot_logical_only']],
                             assigned_donor=d, one_donor_only=True, replacement_allowed=False))
    assert len(normalized) == len(decisions) == len(lineage) == len(assigned) == 960
    assert collections.Counter(x['original_natural_label_GT_only'] for x in decisions) == {0: 143, 1: 20, -1: 797}
    v3.all_row_occurrence_plan(normalized, decisions)
    for name, obj in [('ALL960_NORMALIZED_INPUT_ONLY.jsonl', normalized), ('SOURCE960_LINEAGE_ONLY.jsonl', lineage),
                      ('CONSTRUCTION960_GT_ONLY.jsonl', decisions), ('ONE_DONOR_ASSIGNMENT960_LOCK_BEFORE_PIXELS.jsonl', assigned)]:
        jsonl(HERE / name, obj)
    write(HERE / 'METADATA_INPUT_LOCK.json', dict(completed=True, PASS=True, base_rows=960, occurrences_planned=2880,
          before_labels_D0_E1_Uminus1={'0':143,'1':20,'-1':797}, construction_D=143,
          files=[bind(HERE / n) for n in ['ALL960_NORMALIZED_INPUT_ONLY.jsonl','SOURCE960_LINEAGE_ONLY.jsonl','CONSTRUCTION960_GT_ONLY.jsonl','ONE_DONOR_ASSIGNMENT960_LOCK_BEFORE_PIXELS.jsonl']],
          inputs={k:bind(p) for k,p in names.items()}, code=bind(__file__), pixel_reads_at_lock=0,
          new_model_P3_fit_current_PnP=0, natural_E_only_before_not_V3_constructed=True))
    print(json.dumps(bind(HERE / 'METADATA_INPUT_LOCK.json')))


if __name__ == '__main__':
    main()
