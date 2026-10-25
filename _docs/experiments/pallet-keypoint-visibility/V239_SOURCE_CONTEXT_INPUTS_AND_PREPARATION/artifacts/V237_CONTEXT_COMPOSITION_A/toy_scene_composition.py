"""Invented arrays only. No SOURCE pixels, GT, weights, teacher rows or models."""
import hashlib
import json
import os
from pathlib import PurePosixPath
import numpy as np

HERE = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V237_CONTEXT_COMPOSITION_A'
V107 = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_target_context_rgb_20261023_v107/target_context_math.py'


def read_code(path):
    low = path.lower()
    assert 'v112' not in low and '__pycache__' not in low and not low.endswith('.pyc')
    assert '..' not in PurePosixPath(path).parts
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in PurePosixPath(path).parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        child = os.open(PurePosixPath(path).name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(child, 'rb') as handle:
            return handle.read()
    finally:
        os.close(fd)


def run():
    code = read_code(HERE + '/context_scene_math.py')
    source = read_code(V107)
    assert hashlib.sha256(source).hexdigest() == 'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3'
    m, v = {}, {}
    exec(compile(code, '<own-pure-scene-math>', 'exec'), m)
    exec(compile(source, '<pinned-V107-pure-math>', 'exec'), v)
    rng = np.random.Generator(np.random.PCG64(237))
    base = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    q = np.array([64.2, 63.7], np.float64)
    L = np.float64(80.6)
    q8 = np.array([[20., 20.], [107., 20.], [20., 107.], [107., 107.],
                   [30., 80.], [85., 35.], [-1., -1.], [np.nan, np.nan]], np.float64)
    box = m['query_extent'](q8)
    assert box == v['query_extent'](q8)
    assert np.array_equal(m['context_grid'](box), v['target_context_grid'](box))
    expected, expected_valid = v['sample_target_context'](base, box)
    got, got_valid = m['sample_native_field'](base, m['context_grid'](box))
    assert np.array_equal(got, expected) and np.array_equal(got_valid, expected_valid)

    alpha = np.zeros((96, 96), bool)
    alpha[33:63, 33:63] = True
    layer = np.empty((96, 96, 3), np.uint8)
    layer[:] = [203, 41, 88]
    clean, _ = m['sample_native_field'](base, m['local_grid'](q, L))
    quantize = lambda x: np.clip(np.floor(x.astype(np.float64) + .5), 0, 255).astype(np.uint8)
    old_post_crop = quantize(clean)
    old_post_crop[alpha] = layer[alpha]
    new = m['compose_native_then_sample'](base, q8, q, L, old_post_crop, alpha, reference=q.copy())
    new_local_u8 = quantize(new['local_rgb_F32'])
    different = np.any(new_local_u8 != old_post_crop, axis=2)
    assert int(different.sum()) > 0  # A concrete noncommutativity counterexample.
    boundary_mixed = alpha & (new['local_layer_coverage_diagnostic_only'] < 1)
    assert int(boundary_mixed.sum()) > 0
    changed_native = np.any(new['native_composite'] != base, axis=2)
    assert not np.any(changed_native & ~new['native_alpha'])

    # Two view grids with exactly coincident native coordinates must agree.
    big = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    q2 = np.array([128., 128.], np.float64)
    L2 = np.float64(168.)
    q82 = np.array([[44., 44.], [212., 44.], [44., 212.], [212., 212.],
                    [80., 100.], [160., 100.], [80., 160.], [160., 160.]], np.float64)
    same = m['compose_native_then_sample'](big, q82, q2, L2, layer, alpha)
    li = np.arange(1, 96, 3)
    ci = np.arange(3, 224, 7)
    lg = m['local_grid'](q2, L2)[np.ix_(li, li)]
    cg = m['context_grid'](m['query_extent'](q82))[np.ix_(ci, ci)]
    assert np.array_equal(lg, cg)
    assert np.array_equal(same['local_rgb_F32'][np.ix_(li, li)], same['context_rgb_F32'][np.ix_(ci, ci)])
    assert np.array_equal(same['local_valid'][np.ix_(li, li)], same['context_valid'][np.ix_(ci, ci)])

    # Covered reference pixels cannot depend on the clean pixels behind them.
    zero = m['compose_native_then_sample'](np.zeros_like(base), q8, q, L, layer, alpha, q.copy())
    white = m['compose_native_then_sample'](np.full_like(base, 255), q8, q, L, layer, alpha, q.copy())
    assert zero['teacher_opacity_diagnostic_only']['continuous_reference']['all_covered']
    z, _ = m['sample_native_field'](zero['native_composite'], q[None, None])
    w, _ = m['sample_native_field'](white['native_composite'], q[None, None])
    assert np.array_equal(z, w) and np.array_equal(z[0, 0], layer[0, 0].astype(np.float32))

    # A nearest-pixel hit is insufficient evidence of four-tap opacity.
    sparse = np.zeros((128, 128), bool)
    nearest = np.floor(q + .5).astype(int)
    sparse[nearest[1], nearest[0]] = True
    sparse_check = m['reference_opacity'](sparse, q)
    assert sparse[nearest[1], nearest[0]] and not sparse_check['continuous_reference']['all_covered']
    assert sparse_check['native_gold_or_new_label'] is False

    # No input mutation/U promotion, and an unsupported q8 box has no label.
    qb, q8b, ab, bb = q.copy(), q8.copy(), alpha.copy(), base.copy()
    m['compose_native_then_sample'](base, q8, q, L, layer, alpha)
    assert np.array_equal(q, qb) and np.array_equal(q8, q8b, equal_nan=True)
    assert np.array_equal(alpha, ab) and np.array_equal(base, bb)
    invalid = np.full((8, 2), -1., np.float64)
    unsupported = m['compose_native_then_sample'](base, invalid, q, L, layer, alpha)
    assert unsupported['supported'] is False and unsupported['new_label'] is None

    # Zero-outside V107 behavior and validity agree, including fractional OOB.
    oob = [-1.75, -.25, 4.25, 5.75]
    ref_rgb, ref_valid = v['sample_target_context'](base, oob, (7, 9))
    own_rgb, own_valid = m['sample_native_field'](base, m['context_grid'](oob, (7, 9)))
    assert np.array_equal(ref_rgb, own_rgb) and np.array_equal(ref_valid, own_valid)
    assert np.any(~own_valid)

    result = {
        'completed': True, 'PASS': True, 'fixture_groups': 7,
        'code_sha256': hashlib.sha256(code).hexdigest(),
        'V107_CODE_sha256': hashlib.sha256(source).hexdigest(),
        'invented_post_crop_vs_native_first_counterexample': {
            'different_RAW96_pixels_after_same_declared_rounding': int(different.sum()),
            'old_alpha_pixels_with_partial_native_layer_coverage': int(boundary_mixed.sum()),
            'q': q.tolist(), 'L': float(L), 'actual_SOURCE_example': False,
            'counterexample_not_a_prevalence_or_teacher_error_estimate': True},
        'coincident_grid_sites_RGB_valid_bitexact': int(len(li)**2),
        'V107_context_RGB_grid_validity_bitexact': True,
        'four_tap_opaque_reference_clean_base_dependency_zero': True,
        'nearest_pixel_only_is_not_four_tap_opacity': True,
        'input_mutation_and_unsupported_to_E': 0,
        'actual_RGB_NPZ_GT_teacher_model_or_feature_payload_reads': 0,
        'model_forward_fit_or_optimizer_steps': 0,
        'toy_numpy_arrays_only': True,
    }
    with open(HERE + '/TOY_RESULT.json', 'x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    run()
