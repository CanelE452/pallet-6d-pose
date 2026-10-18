"""Independent analytic counterexamples for the proposed observation types."""
from copy import deepcopy
import unittest
import numpy as np
from .b_query import finite_intersection
from .typed_observation import evaluate, evaluate_fixed_universe


class IndependentGeometryContract(unittest.TestCase):
    polygon = [[0., 0.], [100., 0.], [100., 100.], [0., 100.]]
    binding = "8" * 64

    def known(self):
        return dict(id="q", type="KNOWN_PHYSICAL_EDGE", center=[15., 50.], normal=[0., 1.],
                    edge_evidence=dict(association_confirmed=True, basis="analytic_counterexample",
                                       edge_id="e", target_instance_id="pallet"))

    def edge(self, a, b):
        return {"e": dict(xy=[a, b], target_instance_id="pallet", fixed_pose_binding_sha256=self.binding)}

    def membership(self, owner="target"):
        return dict(id="m", type="TARGET_MEMBERSHIP_ONLY", semantic_owner=owner,
                    target_points=[[20., 30.], [70., 60.]], membership_evidence=dict(
                        basis="analytic_counterexample", target_instance_id="pallet"))

    def test_same_infinite_line_different_finite_segment(self):
        o = self.known()
        first = evaluate(o, self.polygon, image_hw=[200, 200], depth=[1., 1.],
                         candidate_edges=self.edge([10., 40.], [20., 40.]))
        second = evaluate(o, self.polygon, image_hw=[200, 200], depth=[1., 1.],
                          candidate_edges=self.edge([50., 40.], [60., 40.]))
        self.assertTrue(first["relation"]["finite_support"])
        self.assertFalse(second["relation"]["finite_support"])
        self.assertEqual(first["relation"]["lambda_segment"], .5)

    def test_observation_payload_cannot_supply_its_own_candidate_edge(self):
        o = self.known()
        o.update(edge_a=[10., 40.], edge_b=[20., 40.])
        missing = evaluate(o, self.polygon, image_hw=[200, 200], depth=[1., 1.])
        self.assertEqual(missing["status"], "UNBOUND_CANDIDATE_EDGE")
        expected = evaluate(o, self.polygon, image_hw=[200, 200], depth=[1., 1.], candidate_edges=self.edge([10., 40.], [20., 40.]))
        o["edge_a"] = [1e6, -1e6]
        actual = evaluate(o, self.polygon, image_hw=[200, 200], depth=[1., 1.], candidate_edges=self.edge([10., 40.], [20., 40.]))
        self.assertEqual(expected, actual)

    def test_virtual_corner_and_B_label_do_not_authenticate_material(self):
        result = evaluate(dict(id="fake", type="KNOWN_PHYSICAL_EDGE", B=True,
            reprojection_error=0., center=[0., 0.], normal=[1., 0.]), self.polygon, image_hw=[200, 200], depth=[1., 1.])
        self.assertFalse(result["constraint_applied"])
        self.assertFalse(result["independent_physical_ownership_verified"])

    def test_hole_and_occlusion_edges_inside_cuboid_are_not_outer_edges(self):
        result = evaluate(self.membership(), self.polygon, image_hw=[200, 200], depth=[1., 1.])
        self.assertTrue(result["relation"]["all_inside"])
        self.assertFalse(result["precise_edge_constraint"])
        self.assertFalse(result["relation"]["identifies_pose"])

    def test_cone_in_target_prediction_is_not_authenticated_target(self):
        result = evaluate(self.membership("occluder"), self.polygon, image_hw=[200, 200], depth=[1., 1.])
        self.assertFalse(result["constraint_applied"])
        self.assertEqual(result["status"], "OWNERSHIP_CONFLICT")

    def test_deleted_observations_cannot_win_with_zero_cost(self):
        result = evaluate_fixed_universe([], ["required"], self.polygon, image_hw=[200, 200], depth=[1., 1.])
        self.assertEqual(result["status"], "INCOMPLETE_OBSERVATION_UNIVERSE")
        self.assertIsNone(result["total_cost"])
        self.assertFalse(result["candidate_ranking_allowed"])

    def test_giant_projection_degeneracy_exposed_not_selected(self):
        result = evaluate(self.membership(), [[-1e5, -1e5], [1e5, -1e5], [1e5, 1e5], [-1e5, 1e5]], image_hw=[200, 200], depth=[1., 1.])
        self.assertTrue(result["relation"]["all_inside"])
        self.assertTrue(result["relation"]["large_projection_degeneracy_possible"])
        self.assertFalse(result["candidate_ranking_allowed"])

    def test_no_evidence_means_zero_new_correction_without_safety_claim(self):
        result = evaluate_fixed_universe([dict(id="u", type="UNRESOLVED")], ["u"], self.polygon, image_hw=[200, 200])
        self.assertEqual(result["new_correction_applied"], 0)
        self.assertTrue(result["identity_output_is_not_safety_evidence"])

    def test_independent_linear_solve_on_analytic_edges(self):
        # Fixed constructed coordinates, not a fit/seed/threshold search.
        for angle in np.linspace(.05, 6.2, 25):
            tangent = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-tangent[1], tangent[0]])
            a = np.array([400., 300.]); b = a + 73. * tangent
            for lam in (-2., -.1, 0., .3, 1., 1.1, 3.):
                center = a + lam * (b - a) + 13. * normal
                expected = np.linalg.solve(np.column_stack((b - a, -normal)), center - a)
                actual = finite_intersection(a, b, center, normal, image_hw=[1000, 1000], depth=[1., 1.])
                self.assertAlmostEqual(actual["lambda_segment"], expected[0], delta=1e-12)
                self.assertAlmostEqual(actual["normal_offset_px"], expected[1], delta=1e-9)
                self.assertEqual(actual["finite_support"], 0 <= lam <= 1)


if __name__ == "__main__":
    unittest.main()
