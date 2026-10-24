"""Pure draft kernels for SAME-q full-image cover/sham pairs.

All arrays passed to these functions are caller-owned. There is deliberately no
source image/NPZ/checkpoint/model reader and no actual-source CLI in this draft.
A future entry must bind consensus, donor families and exact feature operators
before reading real pixels. Generated fixture arrays are allowed now.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_FIELDS = ('slot_index', 'frame_index', 'input_id', 'native_slot_logical_only',
                'split', 'image_sha256', 'raw_hw', 'query_xy', 'L',
                'selected_box_xyxy', 'original_input_valid')
VARIANTS = ('BEFORE', 'OPAQUE_COVER', 'CUE_PRESERVING_SHAM')
SCALE = .40
SEED = 20261023


def digest(*parts):
    return hashlib.sha256('|'.join(map(str, (SEED,) + parts)).encode()).hexdigest()


def check_input(row):
    if set(row) != set(INPUT_FIELDS):
        raise ValueError('Strict original input-only schema')
    if row['split'] not in ('TRAIN', 'VALIDATION'):
        raise ValueError('Frozen target partition required')
    q = row['query_xy']
    if row['original_input_valid']:
        if q is None or np.asarray(q).shape != (2,) or not np.isfinite(q).all():
            raise ValueError('Available original q must be finite')
        if not np.isfinite(row['L']) or row['L'] <= 0:
            raise ValueError('Unchanged positive L required')


def frozen_rgb96_operator(source_code, expected_sha256):
    """Code-only exact V74 AST extraction; never imports any current reader."""
    path = Path(source_code)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Frozen crop source mismatch')
    tree = ast.parse(path.read_text())
    fn = next(t for t in tree.body if isinstance(t, ast.FunctionDef) and t.name == 'crop')
    scope = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope['crop'], hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()


def bbox_native_cells(box, hw):
    """All NN raster cells touched by a closed continuous uncertainty/support box."""
    h, w = hw
    if box is None or np.asarray(box).shape != (4,) or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = map(float, box)
    if x0 > x1 or y0 > y1 or x0 < 0 or y0 < 0 or x1 > w - 1 or y1 > h - 1:
        return None
    # floor(x+.5) at both endpoints includes ties without moving the box.
    lo = np.floor(np.array([x0, y0]) + .5).astype(int)
    hi = np.floor(np.array([x1, y1]) + .5).astype(int)
    if (lo < 0).any() or hi[0] >= w or hi[1] >= h:
        return None
    result = np.zeros((h, w), bool)
    result[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = True
    return result


def cue_certificate(row, cue):
    """GT-only admission. No cue field is copied into model-input descriptors."""
    check_input(row)
    if not row['original_input_valid']:
        return None, 'ORIGINAL_INPUT_UNAVAILABLE_U'
    flags = (cue.get('consensus') == 'AGREED_CLEAR_OWNED_LOCAL_STRUCTURAL_CUE'
             and cue.get('target_binding') == 'AGREED_SAME_SELECTED_INSTANCE'
             and cue.get('support_usable') is True
             and cue.get('contradictory_multiplecue') is False)
    if not flags:
        return None, 'CUE_OR_TARGET_BINDING_NOT_ADMITTED_U'
    a = bbox_native_cells(cue.get('uncertainty_bbox_xyxy'), row['raw_hw'])
    b = bbox_native_cells(cue.get('visible_support_bbox_xyxy'), row['raw_hw'])
    if a is None or b is None or not a.any() or not b.any():
        return None, 'UNCERTAINTY_OR_SUPPORT_INVALID_U'
    q = np.asarray(row['query_xy'], np.float64)
    # Conservative actual RGB96 sample-centre envelope, not expanded nominal L.
    extent = row['L'] * 47.5 / 96
    boxes = np.asarray([cue['uncertainty_bbox_xyxy'], cue['visible_support_bbox_xyxy']])
    if not (np.all(boxes[:, :2] >= q - extent) and np.all(boxes[:, 2:] <= q + extent)):
        return None, 'CUE_SUPPORT_NOT_FULLY_IN_UNCHANGED_RGB96_ENVELOPE_U'
    return a | b, 'CONDITIONAL_AUX_BEFORE_VISIBLE'



def donor_family_allowed(object_id, split):
    if not isinstance(object_id, int) or not 1 <= object_id <= 21:
        return False
    return (split == 'TRAIN' and object_id % 2 == 1) or (split == 'VALIDATION' and object_id % 2 == 0)


def choose_one_donor_metadata(row, donor_rows):
    """One minSHA donor, no quality/pixel/feasibility filter or replacement."""
    check_input(row)
    candidates = [d for d in donor_rows if d['split'] == row['split']
                  and donor_family_allowed(d['object_id'], row['split'])
                  and d['image_sha256'] != row['image_sha256']]
    candidates.sort(key=lambda d: digest(row['input_id'], row['native_slot_logical_only'],
                                         'one_donor', d['id'], d['image_sha256']))
    return candidates[0] if candidates else None


def prepare_cutout(rgb, visible, longest):
    """Real donor RGB with conservative one-native-pixel eroded hard alpha.

    Boundary/material/opacity validity still needs actual donor admission. No
    rectangle/color fallback; empty masks remain an unusable donor occurrence.
    """
    if rgb.dtype != np.uint8 or rgb.shape != (*visible.shape, 3):
        raise ValueError('Donor RGB and native visible-mask shape/dtype')
    visible = np.asarray(visible, bool)
    padded = np.pad(visible, 1)
    core = np.logical_and.reduce([padded[y:y + visible.shape[0], x:x + visible.shape[1]]
                                  for y in range(3) for x in range(3)])
    yy, xx = np.nonzero(core)
    if not len(xx):
        return None
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    scale = max(1, int(longest)) / max(y1 - y0, x1 - x0)
    h = max(1, int(np.floor((y1 - y0) * scale + .5)))
    w = max(1, int(np.floor((x1 - x0) * scale + .5)))
    r = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).resize((w, h), Image.Resampling.BILINEAR))
    alpha = np.asarray(Image.fromarray(core[y0:y1, x0:x1]).resize((w, h), Image.Resampling.NEAREST)).astype(bool)
    return {'rgb': r.copy(), 'alpha': alpha.copy(), 'donor_native_bbox_xyxy': [x0, y0, x1, y1],
            'resized_hw': [h, w], 'longest': int(longest)}


def translated_alpha(hw, cutout, xy):
    """Require unclipped identical-size donor appearances in BOTH variants."""
    h, w = hw; ch, cw = cutout['alpha'].shape
    left, top = map(int, xy)
    if left < 0 or top < 0 or left + cw > w or top + ch > h:
        return None
    alpha = np.zeros((h, w), bool)
    alpha[top:top + ch, left:left + cw] = cutout['alpha']
    return alpha


def candidate_positions(row, cutout, protected):
    """Finite fixed positions; never moves q/L. GT only places the cover edit."""
    yy, xx = np.nonzero(protected)
    cy, cx = (yy.min() + yy.max()) / 2, (xx.min() + xx.max()) / 2
    ch, cw = cutout['alpha'].shape
    centre = np.array([cx - (cw - 1) / 2, cy - (ch - 1) / 2])
    cover = [tuple(np.floor(centre + [dx, dy] + .5).astype(int))
             for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]]
    q = np.asarray(row['query_xy']); L = row['L']
    sham = [tuple(np.floor(q + [dx * L, dy * L] - [(cw - 1) / 2, (ch - 1) / 2] + .5).astype(int))
            for dy in [-.35, -.175, 0, .175, .35] for dx in [-.35, -.175, 0, .175, .35]]
    return sorted(set(cover), key=lambda xy: digest(row['input_id'], row['native_slot_logical_only'], 'cover', *xy)), sorted(set(sham), key=lambda xy: digest(row['input_id'], row['native_slot_logical_only'], 'sham', *xy))


def fixed_crop_contains_cutout(row, cutout, xy):
    q = np.asarray(row['query_xy']); extent = row['L'] * 47.5 / 96
    h, w = cutout['alpha'].shape
    return bool(np.all(np.asarray(xy) >= q - extent)
                and np.all(np.asarray(xy) + [w - 1, h - 1] <= q + extent))


def placement_pair(row, cutout, protected):
    cover_candidates, sham_candidates = candidate_positions(row, cutout, protected)
    selected = []
    for candidates, must_cover in [(cover_candidates, True), (sham_candidates, False)]:
        found = None
        for xy in candidates:
            if not fixed_crop_contains_cutout(row, cutout, xy):
                continue
            alpha = translated_alpha(row['raw_hw'], cutout, xy)
            if alpha is None:
                continue
            correct = bool(alpha[protected].all()) if must_cover else not bool(alpha[protected].any())
            if correct:
                found = (xy, alpha)
                break
        selected.append(found)
    return selected if all(x is not None for x in selected) else None


def compose(native_rgb, cutout, xy, alpha):
    out = native_rgb.copy(); left, top = xy; h, w = cutout['alpha'].shape
    patch = out[top:top + h, left:left + w]
    patch[cutout['alpha']] = cutout['rgb'][cutout['alpha']]
    assert np.array_equal(out[~alpha], native_rgb[~alpha])
    return out


def prepare_occurrences(native_rgb, row, cue, donor_rgb, donor_visible, donor_meta, crop):
    """Pure per-query builder. Return all three occurrences even when all U.

    GT fields stay in gt_only; downstream feature_jobs expose only original
    input identity and edited full RGB requirement, never cue/donor masks.
    """
    check_input(row)
    if native_rgb.dtype != np.uint8 or native_rgb.shape != (*row['raw_hw'], 3):
        raise ValueError('Native target image shape/dtype')
    protected, reason = cue_certificate(row, cue)
    pair = None; selected_cutout = None
    donor_ok = (donor_meta.get('split') == row['split']
                and donor_meta.get('image_sha256') != row['image_sha256']
                and donor_meta.get('foreign_opaque_cutout_admitted') is True
                and donor_family_allowed(donor_meta.get('object_id'), row['split']))
    if protected is not None and donor_ok:
        cutout = prepare_cutout(donor_rgb, donor_visible, max(9, int(np.floor(SCALE * row['L'] + .5))))
        if cutout is not None:
            pair = placement_pair(row, cutout, protected)
            if pair is not None:
                selected_cutout = cutout
    images = [native_rgb.copy()] * 3
    labels = [0 if protected is not None else -1, -1, -1]
    reasons = [reason, 'NO_MATCHED_COVER_SHAM_PLACEMENT_U', 'NO_MATCHED_COVER_SHAM_PLACEMENT_U']
    certificates = {}
    if pair is not None:
        (cxy, ca), (sxy, sa) = pair
        images = [native_rgb.copy(), compose(native_rgb, selected_cutout, cxy, ca),
                  compose(native_rgb, selected_cutout, sxy, sa)]
        assert ca[protected].all() and not sa[protected].any()
        assert ca.sum() == sa.sum() == selected_cutout['alpha'].sum()
        assert np.array_equal(images[2][protected], native_rgb[protected])
        labels = [0, 1, 0]
        reasons = ['CONDITIONAL_AUX_BEFORE_VISIBLE', 'CONDITIONAL_OPAQUE_COVER', 'CONDITIONAL_CUE_PRESERVING_SHAM']
        certificates = {'cover_xy': list(cxy), 'sham_xy': list(sxy),
                        'same_donor_cutout_shape_style_scale': True,
                        'cover_protected_cells_all_alpha1': True, 'sham_protected_cells_all_alpha0': True,
                        'same_alpha_pixel_count': int(ca.sum()), 'selected_cutout': selected_cutout}
    elif protected is not None and not donor_ok:
        reasons[1:] = ['DONOR_PARTITION_OR_OPACITY_NOT_ADMITTED_U'] * 2
    occurrences = []
    for variant, rgb, label, why in zip(VARIANTS, images, labels, reasons):
        has_crop = (row['query_xy'] is not None and row['L'] is not None
                    and np.asarray(row['query_xy']).shape == (2,)
                    and np.isfinite(row['query_xy']).all() and np.isfinite(row['L']) and row['L'] > 0)
        if has_crop:
            rgb96, valid96 = crop(rgb, np.asarray(row['query_xy']), row['L'], 96)
        else:
            rgb96 = np.zeros((3, 96, 96), np.uint8); valid96 = np.zeros((1, 96, 96), bool)
        occurrence_id = f"{row['input_id']}::slot{row['native_slot_logical_only']}::{variant}"
        # No geometric/support/GT metadata is accepted by the feature forward.
        input_only = dict(row)
        input_only.update(occurrence_id=occurrence_id, variant=variant)
        feature_job = {'occurrence_id': occurrence_id, 'input_only': input_only,
                       'full_RGB_sha256_decoded': hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                       'P3_action': 'REUSE_BOUND_AUDITED_BEFORE_P3' if variant == 'BEFORE' else 'RECOMPUTE_FROZEN_P3_FROM_THIS_FULL_RGB',
                       'initializer_or_query_reselection': False,
                       'q_L_canvas_affine_ROI_operator': 'UNCHANGED_ORIGINAL_SOURCE_PROTOCOL'}
        occurrences.append({'input_only': input_only, 'full_rgb': rgb, 'rgb96': rgb96,
                            'crop_valid96': valid96, 'feature_job': feature_job,
                            'gt_only': {'conditional_AUX_hidden_label': label, 'known': label >= 0,
                                        'reason': why, 'native8_state': 'UNKNOWN',
                                        'original_input_valid': bool(row['original_input_valid']),
                                        'operator_query_available': None,
                                        'operator_extra_unavailable_pending_P3': True}})
    return {'occurrences': occurrences, 'gt_edit_certificate_only': certificates,
            'all_original_rows_retained': True, 'native_or_human_gold_created': False,
            'pair_ready': pair is not None}


def operator_available_from_tensors(p3_roi, valid96):
    """Same stricter support as draft operator; separate from original status."""
    return bool(p3_roi[64, 4, 4] == 1 and np.asarray(valid96)[0, 47:49, 47:49].all())


def materialize_head_inputs(occurrence, before_bound_roi, extract_after_full_rgb):
    """Inject frozen producer only after future entry; GT is never passed to it.

    The caller must bind before cache/ROI phase and after feature code/runtime.
    Passing an old feature array as the after extractor is not a valid API.
    Generated tests use a mock callback, never a real backbone/model call.
    """
    row = occurrence['input_only']
    q, L = row['query_xy'], row['L']
    finite_crop = (q is not None and L is not None and np.asarray(q).shape == (2,)
                   and np.isfinite(q).all() and np.isfinite(L) and L > 0)
    if not finite_crop:
        p3 = np.zeros((65, 9, 9), np.float32)
        action = 'MISSING_QUERY_PLACEHOLDER_NO_EXTRACTOR_CALL'
    elif row['variant'] == 'BEFORE':
        p3 = np.asarray(before_bound_roi).copy()
        action = 'BOUND_AUDITED_SAME_Q_BEFORE_P3_REUSED'
    else:
        if not callable(extract_after_full_rgb):
            raise TypeError('After requires callable frozen full-image extractor, not before P3')
        p3 = np.asarray(extract_after_full_rgb(occurrence['full_rgb'], np.asarray(q).copy(), L))
        action = 'FRESH_FROZEN_FULL_RGB_THEN_UNCHANGED_Q_ROI'
    if p3.shape != (65, 9, 9) or p3.dtype.kind != 'f' or not np.isfinite(p3).all():
        raise ValueError('Frozen producer must return finite floating 65x9x9 ROI')
    if not np.logical_or(p3[64] == 0, p3[64] == 1).all():
        raise ValueError('Factual P3 support must be binary')
    op = operator_available_from_tensors(p3, occurrence['crop_valid96'])
    original = bool(row['original_input_valid'])
    return {'head_inputs_only': {'p3_roi': p3, 'local_rgb96': occurrence['rgb96'],
                                 'local_valid96': occurrence['crop_valid96']},
            'input_accounting_only': {'original_input_valid': original,
                                      'operator_query_available': op,
                                      'additional_operator_unavailable': original and not op,
                                      'effective_available': original and op,
                                      'P3_action': action}}


def preserve_full160_plan(rows):
    """Metadata-only ledger; no known/valid/consensus filtering of occurrences."""
    if len(rows) != 160 or [r['slot_index'] for r in rows] != list(range(160)):
        raise ValueError('Frozen all160 order required')
    for row in rows:
        check_input(row)
    return [{'slot_index': r['slot_index'], 'input_id': r['input_id'], 'split': r['split'],
             'variant': v, 'original_input_valid': r['original_input_valid'],
             'status': 'WAIT_CONSENSUS_DONOR_AND_ACTUAL_ENTRY_NO_LABEL_CREATED'}
            for r in rows for v in VARIANTS]
