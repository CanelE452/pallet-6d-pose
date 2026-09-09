"""Read-only semantic geometry audit; not a deployed canonicalization rule.

The public function observes a predicted, same-ID layout only. Generated
fixtures below compare necessary screen-order conditions with the *legacy*
area converter. The current external renderer's exact face-selection code is
not present in the audited repository: its convention tag is not proof that
the legacy converter generated the current 60,000 labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
FACES = {"front": (0, 1, 2, 3), "rear": (4, 5, 6, 7),
         "left": (0, 3, 7, 4), "right": (1, 2, 6, 5),
         "top": (0, 1, 5, 4), "bottom": (3, 2, 6, 7)}
LR = ((0, 1), (3, 2), (4, 5), (7, 6))
TB = ((0, 3), (1, 2), (4, 7), (5, 6))
FR = ((0, 4), (1, 5), (2, 6), (3, 7))
YAW90 = np.array([1, 5, 6, 2, 0, 4, 7, 3])
C4 = np.stack([np.arange(8), YAW90, YAW90[YAW90], YAW90[YAW90[YAW90]]])
# Generated object only, same as annotation: x right, y down, front at -z.
UNIT = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def polygon_area(points: np.ndarray) -> float:
    p = np.asarray(points, float)
    return float(abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)
                            - p[:, 1]*np.roll(p[:, 0], -1)))*.5)


def layout_semantics(points_xy, diagonal: float, valid=None) -> dict:
    """GT-free, scale-normalized *soft observations*, with missingness explicit.

    No front-face selection, pixel clipping, camera depth inference, physical
    visibility inference or ID permutation occurs. Positive LR/TB margin is
    the annotation tool's necessary image-order condition. Face area requires
    all its four finite vertices; image-external amodal vertices are retained.
    """
    p = np.asarray(points_xy, dtype=np.float64)
    if p.shape != (8, 2) or not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Expected 8x2 points and positive finite image diagonal")
    known = np.isfinite(p).all(1)
    if valid is not None:
        v = np.asarray(valid, bool)
        if v.shape != (8,):
            raise ValueError("valid must have shape (8,)")
        known &= v
    def margins(pairs, axis):
        return [float((p[b, axis]-p[a, axis])/diagonal)
                if known[a] and known[b] else None for a, b in pairs]
    areas = {name: polygon_area(p[list(ids)])/diagonal**2
             if known[list(ids)].all() else None for name, ids in FACES.items()}
    side_known = all(areas[k] is not None for k in ("front", "rear", "left", "right"))
    return dict(
        uses_gt=False, points_valid=known.tolist(),
        lr_margins=margins(LR, 0), tb_margins=margins(TB, 1),
        face_areas_over_image_diagonal_squared=areas,
        front_minus_rear_area=(areas["front"]-areas["rear"]) if side_known else None,
        opposing_area_difference_advantage=(abs(areas["front"]-areas["rear"])
                                            - abs(areas["left"]-areas["right"]))
        if side_known else None,
        camera_depth_known=False, physical_edge_visibility_known=False,
        canonicalization_applied=False,
    )


def _project(points, camera, focal=500., orthographic=False):
    forward = -np.asarray(camera, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., -1., 0.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack([right, -up, forward])
    pc = (np.asarray(points)-camera) @ rotation.T
    scale = np.linalg.norm(camera) if orthographic else pc[:, 2:3]
    return pc[:, :2]/scale*focal + [320., 240.], pc[:, 2]


def generated_fixtures() -> dict:
    from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4
    from scripts.research.pallet_dht_global_layout_v1.geometry import projective_penalty

    records, checks = [], []
    scenarios = [
        ("rectangular_oblique", [1.2, .12, 1.0], [4., -2., -6.], False),
        ("square_oblique", [1., .12, 1.], [4., -2., -6.], False),
        ("square_diagonal_tie", [1., .12, 1.], [5., -2., -5.], False),
        ("square_diagonal_minus", [1., .12, 1.], [4.999, -2., -5.], False),
        ("square_diagonal_plus", [1., .12, 1.], [5.001, -2., -5.], False),
        ("rectangular_orthographic", [1.2, .12, 1.], [4., -2., -6.], True),
        ("frontal_orthographic", [1.2, .12, 1.], [0., -2., -6.], True),
    ]
    for name, scale, camera, ortho in scenarios:
        x = UNIT*np.array(scale)
        q, depth = _project(x, np.array(camera), orthographic=ortho)
        origin = np.column_stack([x[:, 0], x[:, 2], -x[:, 1]])
        canonical_perm = compute_perm_v4(origin, q)[:8]
        variants = []
        for yaw, perm in enumerate(C4):
            f = layout_semantics(q[perm], 800.)
            lr = sum(v <= 0 for v in f['lr_margins'])
            tb = sum(v <= 0 for v in f['tb_margins'])
            fr = sum(depth[perm[a]] >= depth[perm[b]] for a, b in FR)
            geometry = projective_penalty(q[perm], 800., 8.)
            variants.append(dict(yaw_quarters=yaw, permutation=perm.tolist(),
                                 lr_violations=lr, tb_violations=tb,
                                 true_depth_violations_generated_only=int(fr),
                                 all_annotation_pair_invariants_pass=not (lr or tb or fr),
                                 semantic_observations=f,
                                 dlt_valid=geometry['valid'],
                                 dlt_cost=geometry['cost']))
        areas = layout_semantics(q, 800.)['face_areas_over_image_diagonal_squared']
        differences = [abs(areas['front']-areas['rear']), abs(areas['left']-areas['right'])]
        records.append(dict(name=name, generated_dimensions=scale, camera=camera,
                            orthographic=ortho, points_xy=q.tolist(),
                            legacy_converter_permutation=canonical_perm,
                            opposing_area_differences=differences,
                            relative_axis_tie_gap=abs(differences[0]-differences[1]),
                            variants=variants))
        # A projective fit cannot distinguish the C4 semantic rotations.
        assert all(v['dlt_valid'] and abs(v['dlt_cost']) < 1e-10 for v in variants)
        checks.append(name+': all_C4_exact_projective_fit')
        a, b = layout_semantics(q, 800.), layout_semantics(q*3+[12, -5], 2400.)
        assert np.allclose(a['lr_margins'], b['lr_margins'], atol=1e-12)
        assert np.allclose(a['tb_margins'], b['tb_margins'], atol=1e-12)
        assert np.allclose(list(a['face_areas_over_image_diagonal_squared'].values()),
                           list(b['face_areas_over_image_diagonal_squared'].values()), atol=1e-12)
    checks.append('all_7_scenarios_translation_isotropic_scale_invariance')
    square = next(r for r in records if r['name'] == 'square_oblique')
    tie = next(r for r in records if r['name'] == 'square_diagonal_tie')
    assert square['relative_axis_tie_gap'] > 1e-6
    assert tie['relative_axis_tie_gap'] < 1e-12
    checks.append('square_shape_generic_view_is_not_automatically_semantic_tie')
    assert sum(v['all_annotation_pair_invariants_pass'] for v in square['variants']) == 2
    checks.append('two_C4_front_faces_can_both_pass_LR_TB_and_true_depth_order')
    before, after = records[3], records[4]
    assert set(before['legacy_converter_permutation'][:4]) != set(after['legacy_converter_permutation'][:4])
    checks.append('arbitrarily_close_to_square_diagonal_front_axis_can_switch')
    ortho = next(r for r in records if r['name'] == 'rectangular_orthographic')
    assert max(ortho['opposing_area_differences']) < 1e-12
    checks.append('orthographic_opposing_area_differences_collapse_even_non_square')
    q = np.array(square['points_xy'])
    features = layout_semantics(q, 800., [True]*7+[False])
    assert features['face_areas_over_image_diagonal_squared']['rear'] is None
    assert features['front_minus_rear_area'] is None
    assert features['face_areas_over_image_diagonal_squared']['front'] is not None
    checks.append('missing_corner_does_not_fabricate_face_area')
    q[0] = np.nan
    assert layout_semantics(q, 800.)['lr_margins'][0] is None
    checks.append('nonfinite_corner_is_unknown')
    # Infinite line equation is unchanged when endpoints slide on that line.
    a = np.cross([0., 0., 1.], [1., 0., 1.])
    b = np.cross([2., 0., 1.], [3., 0., 1.])
    assert np.array_equal(a, b)
    checks.append('infinite_line_does_not_identify_finite_segment_endpoints')
    return dict(complete=True, PASS=True, scope='generated_geometry_only',
                observed_model_accuracy_cause=False, checks=checks, records=records,
                n_scenarios=len(records), n_C4_layouts=len(records)*4,
                real_images_read=0, model_forwards=0, GT_or_predictions_modified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generated_fixtures()
    files = [Path(__file__).resolve(),
             REPO/'scripts/annotate/convert_to_camera_facing_v4.py',
             REPO/'scripts/annotate/annotate_pnp.py',
             REPO/'scripts/annotate/pallet_geometry.py',
             REPO/'scripts/research/pallet_dht_global_layout_v1/geometry.py']
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['limits'] = [
        'Legacy area converter is not established as exact writer of current external renderer labels.',
        'Image pair-order conditions are necessary, not proof of physical front-face identity.',
        'No physical visibility labels are inferred from v>0 or these 2D features.',
        'Generated fixtures establish mathematical possibilities, not observed causes of model error.',
        'No hard semantic reorder or new training/scoring threshold is selected by this audit.',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('PASS', 'n_scenarios', 'n_C4_layouts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
