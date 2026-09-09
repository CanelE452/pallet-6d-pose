"""Analytic contract tests; no real examples, GT, or model forward."""
import unittest

import numpy as np

from scripts.research.pallet_dht_structured_v2.proposals import build_proposals, C4, DEFAULT_CONFIG


def fixture():
    p = np.array([[120, 210], [430, 194], [430, 245], [119, 260],
                  [80, 122], [351, 108], [357, 161], [82, 176], [250, 181]], float)
    xy = np.broadcast_to(p[:8, None], (8, 49, 2)).copy()
    valid = np.zeros((8, 49), bool)
    valid[:, 0] = True
    for role in range(8):
        for quarter in range(4):
            # One distinct close intersection for each seed at this semantic role.
            xy[role, 1+quarter] = p[C4[quarter, role]] + [1.1+role*.07, 2.3+quarter*.11]
            valid[role, 1+quarter] = True
    return p, xy, valid


class TestProposals(unittest.TestCase):
    def test_c4_is_exact_twelve_edge_graph_automorphism(self):
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
        graph = {frozenset(e) for e in edges}
        for perm in C4:
            self.assertEqual(sorted(perm), list(range(8)))
            self.assertEqual({frozenset((perm[a], perm[b])) for a, b in edges}, graph)
            self.assertEqual(set(perm[[0, 1, 4, 5]]), {0, 1, 4, 5})
        np.testing.assert_array_equal(C4[1][C4[3]], np.arange(8))

    def test_fixed_count_seed_order_padding_and_center(self):
        p, xy, v = fixture()
        result = build_proposals(p, xy, v, 800.)
        self.assertEqual(result['layouts'].shape, (64, 9, 2))
        self.assertEqual(result['diagnostics']['n_generated_before_dedup'], 54)
        self.assertEqual(result['valid'].sum(), 54)
        for k in range(4):
            np.testing.assert_array_equal(result['layouts'][k, :8], p[C4[k]])
            self.assertEqual(result['kind'][k], f'baseline_c4_{k}')
        np.testing.assert_array_equal(result['layouts'][:, 8], np.repeat(p[8:9], 64, 0))
        np.testing.assert_array_equal(result['layouts'][~result['valid']], np.repeat(p[None], 10, 0))

    def test_rotated_seed_uses_new_semantic_role_intersections(self):
        p, xy, v = fixture()
        result = build_proposals(p, xy, v, 800.)
        i = result['kind'].index('snap_all_c4_1_alpha_1')
        # It must use pool[corner], not pool[C4[1,corner]].
        np.testing.assert_array_equal(result['layouts'][i, :8], xy[:, 2])
        np.testing.assert_array_equal(result['candidate_index'][i], np.full(8, 2))

    def test_snap_moves_one_corner_only_and_keeps_original_available(self):
        p, xy, v = fixture()
        result = build_proposals(p, xy, v, 800.)
        i = result['kind'].index('snap_corner_c4_2_corner_7_alpha_0.5')
        expected = p[C4[2]].copy()
        expected[7] += .5*(xy[7, 3]-expected[7])
        np.testing.assert_array_equal(result['layouts'][i, :8], expected)
        np.testing.assert_array_equal(result['layouts'][0], p)
        self.assertEqual(np.count_nonzero(result['candidate_index'][i] >= 0), 1)

    def test_slot_zero_is_not_an_intersection_and_ties_are_stable(self):
        p, xy, v = fixture()
        v[:] = False
        v[:, [0, 5, 9]] = True
        xy[:, 5] = p[:8] + [2., 0.]
        xy[:, 9] = p[:8] - [2., 0.]
        result = build_proposals(p, xy, v, 800.)
        nearest = result['diagnostics']['nearest_intersection_index_by_c4']
        self.assertEqual(nearest[0], [5]*8)
        self.assertTrue(all(i != 0 for row in nearest for i in row))

    def test_no_line_keeps_seeds_and_only_unique_point_perturbations(self):
        p, xy, v = fixture()
        v[:, 1:] = False
        result = build_proposals(p, xy, v, 800.)
        self.assertEqual(result['valid'].sum(), 10)
        self.assertEqual(result['diagnostics']['n_deduplicated'], 44)
        self.assertTrue(all(not k.startswith('snap') for k in result['kind']))

    def test_missing_prediction_fallback_preserves_every_original_coordinate(self):
        p, xy, v = fixture()
        point_valid = np.ones(9, bool)
        point_valid[4] = False
        result = build_proposals(p, xy, v, 800., point_valid=point_valid)
        self.assertEqual(result['valid'].sum(), 1)
        np.testing.assert_array_equal(result['layouts'], np.repeat(p[None], 64, 0))
        self.assertEqual(result['diagnostics']['fallback_reason'], 'missing_baseline_corner')

    def test_nonfinite_intersection_is_excluded_not_repaired_from_gt(self):
        p, xy, v = fixture()
        xy[0, 1] = [np.nan, 1.]
        result = build_proposals(p, xy, v, 800.)
        self.assertEqual(result['diagnostics']['excluded_nonfinite_intersections'], 1)
        self.assertTrue(np.isfinite(result['layouts']).all())
        self.assertNotEqual(result['diagnostics']['nearest_intersection_index_by_c4'][0][0], 1)

    def test_missing_centroid_does_not_disable_eight_valid_corners(self):
        p, xy, v = fixture()
        p[8] = [np.nan, np.nan]
        result = build_proposals(p, xy, v, 800.)
        self.assertEqual(result['valid'].sum(), 54)
        self.assertTrue(np.isnan(result['layouts'][:, 8]).all())

    def test_similarity_equivariance_original_pixel_units(self):
        p, xy, v = fixture()
        a = build_proposals(p, xy, v, 800.)
        b = build_proposals(p*3+[71., -19.], xy*3+[71., -19.], v, 2400.)
        np.testing.assert_array_equal(a['valid'], b['valid'])
        np.testing.assert_allclose(a['layouts']*3+[71., -19.], b['layouts'], atol=1e-10)
        self.assertEqual(a['kind'], b['kind'])

    def test_whole_translation_and_scale_do_not_move_centroid(self):
        p, xy, v = fixture()
        r = build_proposals(p, xy, v, 800.)
        i = r['kind'].index('translate_x_+1')
        np.testing.assert_array_equal(r['layouts'][i, :8], p[:8]+[4., 0.])
        center = (p[:8].min(0)+p[:8].max(0))*.5
        j = r['kind'].index('scale_corner_bbox_1.02')
        np.testing.assert_array_equal(r['layouts'][j, :8], center+1.02*(p[:8]-center))
        np.testing.assert_array_equal(r['layouts'][j, 8], p[8])

    def test_provenance_text_cannot_change_geometry_and_unknown_gt_arg_rejected(self):
        p, xy, v = fixture()
        a = build_proposals(p, xy, v, 800., source_tag='synthetic')
        b = build_proposals(p, xy, v, 800., source_tag='real')
        np.testing.assert_array_equal(a['layouts'], b['layouts'])
        np.testing.assert_array_equal(a['valid'], b['valid'])
        with self.assertRaises(TypeError):
            build_proposals(p, xy, v, 800., gt_points=p)

    def test_degenerate_screen_corners_do_not_remove_required_c4_seeds(self):
        p, xy, v = fixture()
        p[:8] = [250., 200.]
        v[:] = False
        r = build_proposals(p, xy, v, 800.)
        self.assertTrue(r['valid'][:4].all())
        np.testing.assert_array_equal(r['layouts'][:4], np.repeat(p[None], 4, 0))
        self.assertEqual(r['valid'].sum(), 8)  # C4 duplicates retained, four translations.

    def test_malformed_contract_fails_clearly(self):
        p, xy, v = fixture()
        for diag in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                build_proposals(p, xy, v, diag)
        with self.assertRaises(ValueError):
            build_proposals(p[:8], xy, v, 800.)
        self.assertFalse(DEFAULT_CONFIG['GT_used'])


if __name__ == '__main__':
    unittest.main()
