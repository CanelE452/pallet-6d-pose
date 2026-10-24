"""Prospective scale/mass tests on invented arrays, no source annotations/images."""
from pathlib import Path
import hashlib
import json

import numpy as np

import geometry_mass_v2 as v2

ROOT = Path(__file__).resolve().parent


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def scalar_alpha(mask, q, L, y, x):
    h, w = mask.shape
    xx = q[0] + ((x + .5) / 96 - .5) * L
    yy = q[1] + ((y + .5) / 96 - .5) * L
    if not (-.5 <= xx < w - .5 and -.5 <= yy < h - .5):
        return 0.
    x0, y0 = int(np.floor(xx)), int(np.floor(yy))
    dx, dy = xx - x0, yy - y0
    value = 0.
    for ax, ay, weight in [(0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)]:
        if 0 <= x0 + ax < w and 0 <= y0 + ay < h:
            value += float(mask[y0+ay, x0+ax]) * weight
    return value


def run():
    v1_before = hashlib.sha256(v2.V1_PATH.read_bytes()).hexdigest()
    row = dict(slot_index=0, frame_index=0, input_id='INVENTED_LARGE_SUPPORT', native_slot_logical_only=0,
               split='TRAIN', image_sha256='a'*64, raw_hw=[96, 96], query_xy=[32., 32.], L=64.,
               selected_box_xyxy=[1., 1., 70., 70.], original_input_valid=True)
    rgb = np.arange(32*60*3, dtype=np.uint8).reshape(32, 60, 3)
    visible = np.zeros((32, 60), bool); visible[4:26, 4:56] = True
    protected = np.zeros((96, 96), bool); protected[30:33, 2:30] = True
    old = v2.v1.prepare_cutout(rgb, visible, int(np.floor(.40*row['L']+.5)))
    assert old['alpha'].shape[1] < 28
    assert v2.v1.placement_pair(row, old, protected) is None
    cut, scale_facts = v2.support_scaled_cutout(rgb, visible, protected, row['L'])
    assert cut is not None
    pair, mass_facts = v2.core_aligned_pair(row, cut, protected)
    assert pair is not None and mass_facts['PASS']
    assert pair[0][1][protected].all() and not pair[1][1][protected].any()
    assert pair[0][1].sum() == pair[1][1].sum()
    assert row['query_xy'] == [32., 32.] and row['L'] == 64.

    # Largest allopaque rectangle is inside real alpha; holes cannot be filled.
    rng = np.random.default_rng(20261023)
    brute_cases = 0
    for h in range(1, 6):
        for w in range(1, 6):
            mask = rng.random((h, w)) > .35
            expected = []
            for y0 in range(h):
                for x0 in range(w):
                    for y1 in range(y0+1, h+1):
                        for x1 in range(x0+1, w+1):
                            if mask[y0:y1, x0:x1].all():
                                expected.append(((x1-x0)*(y1-y0), (x0, y0, x1, y1)))
            expected.sort(key=lambda t: (-t[0], t[1]))
            assert v2.largest_opaque_rectangle(mask) == (expected[0][1] if expected else None)
            brute_cases += 1
    huge = np.zeros((96, 96), bool); huge[2:60, 2:60] = True
    no_cut, facts = v2.support_scaled_cutout(rgb, visible, huge, 64.)
    assert no_cut is None and facts['reason'] == 'SCALE_OR_WHOLE_CUTOUT_BOUND_EXCEEDED_U'
    empty, _ = v2.support_scaled_cutout(rgb, np.zeros_like(visible), protected, 64.)
    assert empty is None

    # Equal global area can hide outside the qcrop; explicit native/96 mass rejects.
    c = np.zeros((96, 96), bool); c[25:35, 25:35] = True
    s = np.zeros_like(c); s[75:85, 75:85] = True
    ok, outside_facts = v2.mass_certificate(row, c, s)
    assert c.sum() == s.sum() and not ok
    # Even same shape entirely inside crop can alias differently at RGB96 phase.
    large_row = dict(row, raw_hw=[224, 224], query_xy=[112., 112.], L=192.)
    c = np.zeros((224, 224), bool); c[80, 30:110] = True
    s = np.zeros_like(c); s[81, 30:110] = True
    ok, alias_facts = v2.mass_certificate(large_row, c, s)
    assert alias_facts['native_in_crop_alpha_cells'] == [80, 80]
    assert not ok and alias_facts['bilinear_RGB96_alpha_pixel_equivalents'][0] == 0
    # Direct scalar oracle at every RGB96 sample, including edge/partial validity.
    mask = rng.random((17, 23)) > .5; q = np.array([.3, 15.9]); L = 31.7
    scalar = sum(scalar_alpha(mask, q, L, y, x) for y in range(96) for x in range(96))
    actual = v2.alpha_mass96(mask, q, L)
    assert abs(scalar - actual) < 1e-10
    assert hashlib.sha256(v2.V1_PATH.read_bytes()).hexdigest() == v1_before == v2.V1_SHA
    return {'completed': True, 'PASS': True, 'proposal_code': bind(ROOT/'geometry_mass_v2.py'),
            'generated_code': bind(Path(__file__)), 'V1_unchanged_code': bind(v2.V1_PATH),
            'generated_fixed_040L_fail_support_dependent_bounded_pair_success': True,
            'generated_scale': scale_facts, 'generated_matched_pair_mass': mass_facts,
            'largest_opaque_rectangle_bruteforce_masks': brute_cases,
            'bounded_scale_impossible_and_empty_donor_U': True,
            'same_global_mass_outside_crop_rejection': outside_facts,
            'same_shape_inside_crop_RGB96_aliasing_mass_rejection': alias_facts,
            'alpha96_all9216_scalar_oracle_abs_difference': abs(scalar-actual),
            'no_q_L_move_donor_replacement_or_source_data': True,
            'actual_source_annotation_array_pixel_model_current_reads': 0,
            'actual_raster_forward_fit_PnP': 0,
            'proposal_adopted_for_actual_source': False}


if __name__ == '__main__':
    result = run()
    output = ROOT/'GENERATED_GEOMETRY_MASS_TESTS.json'
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
