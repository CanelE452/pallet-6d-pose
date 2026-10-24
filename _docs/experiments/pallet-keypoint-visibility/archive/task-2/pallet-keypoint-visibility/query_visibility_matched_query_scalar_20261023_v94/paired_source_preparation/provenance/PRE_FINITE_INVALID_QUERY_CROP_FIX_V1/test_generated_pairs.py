"""Invented arrays only; no real source RGB/mask/NPZ/checkpoint reader."""
from pathlib import Path
import ast
import copy
import hashlib
import json

import numpy as np

import same_q_pair_builder as core

ROOT = Path(__file__).resolve().parent
P = ROOT.parent.parent
CROP = P / 'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
CROP_SHA = '802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66'


def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def run():
    crop, cropast = core.frozen_rgb96_operator(CROP, CROP_SHA)
    target = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    donor = np.arange(40 * 44 * 3, dtype=np.uint8).reshape(40, 44, 3)
    mask = np.zeros((40, 44), bool); mask[3:37, 4:40] = True; mask[3:9, 4:12] = False
    row = dict(slot_index=0, frame_index=0, input_id='INVENTED_FRAME_0', native_slot_logical_only=0,
               split='TRAIN', image_sha256='a' * 64, raw_hw=[64, 64], query_xy=[32., 32.],
               L=48., selected_box_xyxy=[12., 10., 52., 54.], original_input_valid=True)
    cue = dict(consensus='AGREED_CLEAR_OWNED_LOCAL_STRUCTURAL_CUE',
               target_binding='AGREED_SAME_SELECTED_INSTANCE', support_usable=True,
               contradictory_multiplecue=False, uncertainty_bbox_xyxy=[29., 29., 31., 31.],
               visible_support_bbox_xyxy=[28., 28., 32., 32.])
    donor_meta = dict(split='TRAIN', image_sha256='b' * 64, object_id=3,
                      foreign_opaque_cutout_admitted=True)
    saved_row = copy.deepcopy(row)
    result = core.prepare_occurrences(target, row, cue, donor, mask, donor_meta, crop)
    assert result['pair_ready'] and row == saved_row
    occurrences = result['occurrences']
    assert [r['gt_only']['conditional_AUX_hidden_label'] for r in occurrences] == [0, 1, 0]
    assert [r['input_only']['variant'] for r in occurrences] == list(core.VARIANTS)
    for o in occurrences:
        assert o['input_only']['query_xy'] == row['query_xy'] and o['input_only']['L'] == row['L']
        assert o['input_only']['selected_box_xyxy'] == row['selected_box_xyxy']
        assert o['gt_only']['native8_state'] == 'UNKNOWN'
        assert o['gt_only']['operator_query_available'] is None
        assert o['feature_job']['initializer_or_query_reselection'] is False
        assert set(o['input_only']) == set(core.INPUT_FIELDS) | {'occurrence_id', 'variant'}
        c, v = crop(o['full_rgb'], np.asarray(row['query_xy']), row['L'], 96)
        assert np.array_equal(c, o['rgb96']) and np.array_equal(v, o['crop_valid96'])
    assert occurrences[0]['feature_job']['P3_action'] == 'REUSE_BOUND_AUDITED_BEFORE_P3'
    assert all(o['feature_job']['P3_action'] == 'RECOMPUTE_FROZEN_P3_FROM_THIS_FULL_RGB' for o in occurrences[1:])
    assert not np.array_equal(occurrences[0]['full_rgb'], occurrences[1]['full_rgb'])
    assert not np.array_equal(occurrences[0]['rgb96'], occurrences[1]['rgb96'])
    cert = result['gt_edit_certificate_only']; protected, _ = core.cue_certificate(row, cue)
    assert cert['cover_protected_cells_all_alpha1'] and cert['sham_protected_cells_all_alpha0']
    assert np.array_equal(occurrences[2]['full_rgb'][protected], target[protected])
    for index, key in [(1, 'cover_xy'), (2, 'sham_xy')]:
        alpha = core.translated_alpha(row['raw_hw'], cert['selected_cutout'], cert[key])
        assert np.array_equal(occurrences[index]['full_rgb'][~alpha], target[~alpha])

    # Bad cue, target ambiguity, unsupported / MISS and unavailable rows all survive.
    bad_cues = [dict(cue, contradictory_multiplecue=True),
                dict(cue, target_binding='UNRESOLVED'),
                dict(cue, support_usable=False),
                dict(cue, uncertainty_bbox_xyxy=[60., 60., 62., 62.]),
                dict(cue, uncertainty_bbox_xyxy=[-1., 2., 3., 4.])]
    for c in bad_cues:
        z = core.prepare_occurrences(target, row, c, donor, mask, donor_meta, crop)
        assert len(z['occurrences']) == 3 and not z['pair_ready']
        assert all(o['gt_only']['conditional_AUX_hidden_label'] == -1 for o in z['occurrences'])
    for dm, dm_mask in [(dict(donor_meta, split='VALIDATION'), mask),
                        (dict(donor_meta, object_id=2), mask),
                        (donor_meta, np.zeros_like(mask))]:
        z = core.prepare_occurrences(target, row, cue, donor, dm_mask, dm, crop)
        assert len(z['occurrences']) == 3 and not z['pair_ready']
        assert [o['gt_only']['conditional_AUX_hidden_label'] for o in z['occurrences']] == [0, -1, -1]
    # A cue occupying most of q crop has no same-shape clear sham, so no favorable replacement.
    huge = dict(cue, uncertainty_bbox_xyxy=[12., 12., 51., 51.], visible_support_bbox_xyxy=[12., 12., 51., 51.])
    z = core.prepare_occurrences(target, row, huge, donor, mask, donor_meta, crop)
    assert not z['pair_ready'] and len(z['occurrences']) == 3
    missing = dict(row, original_input_valid=False, query_xy=None, L=None)
    z = core.prepare_occurrences(target, missing, cue, donor, mask, donor_meta, crop)
    assert all(o['gt_only']['conditional_AUX_hidden_label'] == -1 for o in z['occurrences'])
    assert all(not o['crop_valid96'].any() for o in z['occurrences'])

    # 160 original rows, including all missing/invalid, always produce 480 ledger occurrences.
    rows = [dict(row, slot_index=i, frame_index=i // 8, input_id=f'FRAME{i // 8}',
                 native_slot_logical_only=i % 8, split='TRAIN' if i // 8 < 12 else 'VALIDATION',
                 original_input_valid=i < 122) for i in range(160)]
    plan = core.preserve_full160_plan(rows)
    assert len(plan) == 480 and sum(not o['original_input_valid'] for o in plan) == 114
    assert sum(o['split'] == 'TRAIN' for o in plan) == 288
    assert sum(o['split'] == 'VALIDATION' for o in plan) == 192
    donors = [dict(id=f'D{i}', split='TRAIN', object_id=i, image_sha256=str(i) * 64)
              for i in range(1, 10)]
    first = core.choose_one_donor_metadata(row, donors)
    assert first == core.choose_one_donor_metadata(row, donors[::-1]) and first['object_id'] % 2 == 1
    # Original validity and stricter operator support are explicitly different facts.
    p3 = np.zeros((65, 9, 9), np.float32); p3[64] = 1
    valid = np.ones((1, 96, 96), bool)
    assert core.operator_available_from_tensors(p3, valid)
    valid[0, 47, 47] = False
    assert row['original_input_valid'] and not core.operator_available_from_tensors(p3, valid)
    tree = ast.parse((ROOT / 'same_q_pair_builder.py').read_text())
    calls = {t.func.attr for t in ast.walk(tree) if isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute)}
    assert not calls.intersection({'load', 'open', 'predict', 'forward', 'load_state_dict'})
    return {'completed': True, 'PASS': True, 'code': bind(ROOT / 'same_q_pair_builder.py'),
            'generated_code': bind(Path(__file__)), 'crop_code': bind(CROP), 'crop_AST_sha256': cropast,
            'same_q_L_box_order': True, 'full_image_RGB_edit_then_exact_RGB96': True,
            'after_P3_requires_fresh_same_full_RGB_no_before_feature_reuse': True,
            'all160_480_occurrences_and_38x3_unavailable_U_retained': True,
            'same_donor_scale_shape_alphaarea_cover_vs_sham': True,
            'full_protected_cover_and_sham_bitexact_clear': True,
            'wrong_split_or_object_family_or_empty_or_impossible_remains_U': True,
            'one_minSHA_donor_no_success_replacement': True,
            'original_input_valid_operator_available_accounted_separately': True,
            'actual_source_array_pixel_mask_model_checkpoint_current_reads': 0,
            'actual_forward_fit_PnP': 0, 'native_or_human_semantic_gold_created': False}


if __name__ == '__main__':
    result = run()
    out = ROOT / 'GENERATED_SAME_Q_PAIR_TESTS.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, indent=2))
