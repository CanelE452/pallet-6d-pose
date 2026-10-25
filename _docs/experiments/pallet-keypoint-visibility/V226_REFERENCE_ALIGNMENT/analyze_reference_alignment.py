"""Nominal provided-reference alignment at unchanged initial queries.

Continuous square support q +/- L/2 is a geometric diagnostic, not a replay
of the 96-sample raster or proof of physical keypoint visibility/quality.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V226_ROOT_ALIGNMENT')
SPLIT = R.parent / 'V225_GROUP_REVIEW_A/CANDIDATE_SOURCE_FIT_DEV_MANIFEST.json'


def quantile(values, p):
    a = sorted(values)
    if not a:
        return None
    k = (len(a) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return a[lo] * (hi - k) + a[hi] * (k - lo) if hi != lo else a[lo]


def stats(values):
    return {name: quantile(values, p) for name, p in
            [('min', 0), ('q25', .25), ('median', .5), ('q75', .75), ('q90', .9), ('max', 1)]}


def summarize(rows):
    valid = [r for r in rows if r['finite_reference']]
    return {'rows': len(rows), 'finite_reference': len(valid),
            'reference_positive_camera_Z': sum(r['reference_positive_camera_Z'] for r in rows),
            'reference_continuous_inframe': sum(r['reference_continuous_inframe'] for r in rows),
            'reference_in_continuous_crop_square': sum(r['reference_in_continuous_crop_square'] for r in rows),
            'reference_in_crop_and_image': sum(r['reference_in_crop_and_image'] for r in rows),
            'outside_continuous_crop_square': sum(r['finite_reference'] and not r['reference_in_continuous_crop_square'] for r in rows),
            'provided_reference_error_px': stats([r['provided_reference_error_px'] for r in valid]),
            'provided_reference_error_over_L': stats([r['provided_reference_error_over_L'] for r in valid]),
            'reference_error_le_3px_NOT_physical_gold': sum(r['provided_reference_error_px'] <= 3 for r in valid),
            'reference_error_le_10px': sum(r['provided_reference_error_px'] <= 10 for r in valid),
            'reference_error_le_30px': sum(r['provided_reference_error_px'] <= 30 for r in valid),
            'provided_convex_self_visible_proxy_true': sum(r['provided_convex_self_visible_proxy_only'] for r in rows),
            'states_unchanged': dict(Counter(r['existing_weak_role'] for r in rows))}


def main():
    src = R / 'SELECTED_TRAIN_REFERENCE368.jsonl'
    raw = src.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'b7a0fb16372d83f725e5b5c8ca4a69ffef361276d08f42fb0769a09bb8953cc1'
    selected = [json.loads(s) for s in raw.splitlines()]
    split = json.loads(SPLIT.read_text())
    dev = set(split['DEV_component_ids'])
    fit = set(split['FIT_component_ids'])
    assert len(dev) == 18 and len(fit) == 81 and not dev & fit
    out = []
    for item in selected:
        a = item['original_query_weak_record_unchanged']
        ref = item['provided_reference_record_unchanged']
        assert a['native_slot'] == ref['native_slot']
        assert a['frame_index'] == ref['frame_index'] and a['selected_gt_instance_index'] == ref['gt_instance_index']
        perm = ref['dynamic_to_fixed_permutation']
        assert sorted(perm) == list(range(8))
        assert ref['fixed_canonical_index'] == perm[ref['native_slot']]
        q = a['query_xy']; L = a['L']; xy = ref['reference_xy']; h, w = ref['raw_hw']
        assert all(math.isfinite(v) for v in q) and math.isfinite(L) and L > 0
        finite = isinstance(xy, list) and len(xy) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy)
        dx, dy = (xy[0] - q[0], xy[1] - q[1]) if finite else (None, None)
        error = math.hypot(dx, dy) if finite else None
        in_frame = bool(finite and 0 <= xy[0] <= w - 1 and 0 <= xy[1] <= h - 1)
        in_crop = bool(finite and abs(dx) <= L / 2 and abs(dy) <= L / 2)
        assert in_frame == ref['continuous_inframe']
        assert item['component_id'] in dev | fit
        out.append({'row_id': item['row_id'], 'frame_index': a['frame_index'], 'source_id': a['source_id'],
                    'native_slot_dynamic': a['native_slot'], 'fixed_canonical_index': ref['fixed_canonical_index'],
                    'selected_gt_instance_index': a['selected_gt_instance_index'], 'component_id': item['component_id'],
                    'candidate_role': 'DEV' if item['component_id'] in dev else 'FIT',
                    'existing_weak_role': a['role_state'], 'original_query_available': a['original_input_valid'],
                    'query_xy_unchanged': q, 'L_unchanged': L, 'provided_reference_xy': xy,
                    'finite_reference': finite, 'dx_over_L': dx / L if finite else None,
                    'dy_over_L': dy / L if finite else None,
                    'provided_reference_error_px': error, 'provided_reference_error_over_L': error / L if finite else None,
                    'reference_positive_camera_Z': ref['positive_camera_Z'],
                    'reference_continuous_inframe': in_frame, 'reference_in_continuous_crop_square': in_crop,
                    'reference_in_crop_and_image': in_frame and in_crop,
                    'provided_convex_self_visible_proxy_only': ref['convex_self_visible_proxy_only'],
                    'reference_semantic_state_unchanged': ref['semantic_state'],
                    'original_point_quality_label_unchanged': a['point_quality_label'],
                    'reference_point_uncertainty_unchanged': ref['reference_point_uncertainty'],
                    'authority': 'Nominal provided CAD/pose reference under existing weak instance/role correspondence, not physical point gold.',
                    'new_training_admission': False})
    assert len(out) == 368 and all(r['original_point_quality_label_unchanged'] == -1 for r in out)
    with (R / 'REFERENCE_ALIGNMENT368.jsonl').open('x') as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    groups = defaultdict(list)
    for row in out:
        for name in ('ALL', 'STATE_' + row['existing_weak_role'], row['candidate_role'],
                     row['candidate_role'] + '_' + row['existing_weak_role'], 'NATIVE_' + str(row['native_slot_dynamic'])):
            groups[name].append(row)
    summary = {'completed': True, 'PASS': True, 'groups': {k: summarize(v) for k, v in sorted(groups.items())},
               'all_q_L_native_inputU_point_quality_unchanged': True,
               'dynamic_to_fixed_mapping_applied_again': False,
               'GT_nearest_or_permutation_search': False,
               'reference_in_crop_definition': 'abs(reference_xy - original_q) <= L/2 on each axis; continuous square, not exact RAW96 raster support/valid-pixel replay.',
               'physical_3px_gold_or_semantic_visibility_certified': False,
               'reference_uncertainty': 'Nominal low-centimeter CAD/pose reference. Do not train physical point-quality labels or overwrite D/E/SELF from this diagnostic.',
               'new_training_forward_current223_PnP': 0, 'new_training_admission': False}
    with (R / 'ALIGNMENT_SUMMARY.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    fields = ['row_id', 'frame_index', 'source_id', 'native_slot_dynamic', 'candidate_role', 'existing_weak_role',
              'provided_reference_error_px', 'provided_reference_error_over_L', 'dx_over_L', 'dy_over_L',
              'reference_continuous_inframe', 'reference_in_continuous_crop_square', 'reference_in_crop_and_image']
    with (R / 'REFERENCE_ALIGNMENT368.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in out)
    print(json.dumps({k: v for k, v in summary['groups'].items() if k in ('ALL', 'STATE_D', 'STATE_E', 'FIT', 'DEV')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
