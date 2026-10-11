"""Synthetic CPU contracts; ROOT freezes and executes this module.

The pinhole/Rodrigues reference below does not call the production projection,
residual, optimizer or rank helpers. Natural recovery is explicitly separate
from the kernel counterexample. No real frame, image, annotation or GT is read.
The caller supplies the frozen research repository on sys.path, as solver.py
also requires. Importing this module is an actual CPU action reserved for ROOT.
"""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scripts.research.pallet_factor_subset_consensus_20261011_v15 import solver as P15
from scripts.research.pallet_finite_edge_consensus_repair_20261011_v13 import solver as P13
from scripts.research.pallet_consensus_decision_repair_20261011_v18 import solver as D18


PARENTS = (P15, P13, D18, P15.P, P13.V5, P13.V9, P15.G14)
PARENT_GLOBALS = {m: dict(vars(m)) for m in PARENTS}
PARENT_POLICIES = {m: copy.deepcopy(m.POLICY) for m in PARENTS}
SPEC = importlib.util.spec_from_file_location(
    'private_v19_observed_frame_solver', Path(__file__).with_name('solver.py'))
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
EMPTY = dict(corners=[], lines=[], queries=[])


def rotation(r):
    """Independent axis-angle exponential, used only in synthetic references."""
    r = np.asarray(r, float)
    angle = math.sqrt(math.fsum(float(x*x) for x in r))
    if angle == 0.:
        return np.eye(3)
    x, y, z = r
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)/angle*cross + (1.-math.cos(angle))/angle**2*(cross@cross)


def project(model, z, K):
    camera = np.asarray(model, float) @ rotation(z[:3]).T + np.asarray(z[3:], float)
    homogeneous = camera @ np.asarray(K, float).T
    return homogeneous[:, :2]/homogeneous[:, 2:3]


def fixture():
    xyz = np.array([1., .2, 1.])  # One dimension registry entry: W == D.
    # Exact existing V1 cuboid corner order, independent of its constructor.
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    model = signs*xyz/2.
    K = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    z = np.array([.18,-.12,.07,.05,.04,4.])
    q = project(model, z, K)
    return dict(xyz=xyz, model=model, K=K, z=z,
                points=np.vstack((q, [123.,234.])))


def numeric_line(q, edge=0, fractions=(-.25,.5,1.25)):
    a, b = S.EDGES[edge]
    direction = q[b]-q[a]
    normal = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
    support = [q[a]+fraction*direction for fraction in fractions]
    return dict(edge=edge, normal=normal, offset=float(normal@q[a]),
                support_query_ids=[edge*7+k for k in range(len(support))],
                source_line=dict(support_points=np.asarray(support).tolist(),
                                 query_radii_px=[1.]*len(support)))


def independent_residual(model, z, K, points, point_ids, lines):
    """Fixed observed-axis interval by scalar clipping, independent of helper."""
    q = project(model, z, K)
    rows = [float(q[k,i]-points[k,i]) for k in point_ids for i in range(2)]
    for line in lines:
        a, b = S.EDGES[line['edge']]
        normal = np.asarray(line['normal'], float)
        tangent = np.array([-normal[1], normal[0]])
        endpoint_u = q[[a,b]]@tangent
        lower, upper = min(endpoint_u), max(endpoint_u)
        rows.extend((q[[a,b]]@normal-line['offset'])/math.sqrt(2.))
        support = np.asarray(line['source_line']['support_points'])
        u = support@tangent
        rows.extend((u-np.clip(u, lower, upper))/math.sqrt(len(support)))
    return np.asarray(rows, float)


def closed_distances(support, A, B):
    direction = B-A
    location = ((support-A)@direction)/(direction@direction)
    closest = A + np.clip(location, 0., 1.)[:,None]*direction
    return np.linalg.norm(support-closest, axis=1)


class ObservedFrameGeometryContract(unittest.TestCase):
    def tearDown(self):
        # Test import and every exercised call, not only a declared proof flag.
        for module in PARENTS:
            self.assertEqual(set(vars(module)), set(PARENT_GLOBALS[module]))
            for key, value in PARENT_GLOBALS[module].items():
                self.assertIs(vars(module)[key], value, (module.__name__, key))
            self.assertEqual(module.POLICY, PARENT_POLICIES[module])

    def test_actual_3D_projection_reproduces_old_false_inlier_and_fixed_rejection(self):
        # Kernel-only synthetic 3D edge, not a full cuboid pose-recovery claim.
        data = fixture(); model = np.zeros((8,3)); a,b = S.EDGES[0]
        model[a] = [0.,-.00008,0.]; model[b] = [0.,.00008,0.]
        z = np.array([0.,0.,0.,0.,0.,4.]); q = project(model,z,data['K'])
        support = np.array([[220.,240.],[320.,240.],[420.,240.]])
        line = dict(edge=0,normal=np.array([0.,1.]),offset=240.,support_query_ids=[0,1,2],
                    source_line=dict(support_points=support.tolist(),query_radii_px=[1.,1.,1.]))
        old,_,old_q,_ = P13.finite_residual_jacobian(model,z,data['K'],data['points'],(),[line])
        fixed,_,fixed_q,blocks = S.finite_residual_jacobian(model,z,data['K'],data['points'],(),[line])
        np.testing.assert_allclose(old_q,q,rtol=0.,atol=1e-12)
        np.testing.assert_allclose(fixed_q,q,rtol=0.,atol=1e-12)
        np.testing.assert_allclose(q[[a,b]],[[320.,239.99],[320.,240.01]],rtol=0.,atol=1e-12)
        np.testing.assert_allclose(fixed,independent_residual(model,z,data['K'],data['points'],(),[line]),rtol=0.,atol=1e-12)
        self.assertLessEqual(float(np.linalg.norm(old)),8.)
        self.assertGreater(float(np.linalg.norm(fixed)),8.)
        np.testing.assert_allclose(closed_distances(support,*q[[a,b]]),[100.,0.,100.],rtol=0.,atol=1e-12)
        self.assertTrue(blocks[0]['interval_order_cusp'])
        self.assertEqual(blocks[0]['block_factor_count'],1)
        self.assertEqual(blocks[0]['new_independent_point_correspondences'],0)
        # At this interval cusp there is no smooth-J/rank assertion.

    def test_smooth_analytic_J_against_independent_projection_differences(self):
        data = fixture(); q = project(data['model'],data['z'],data['K'])
        lines = [numeric_line(q)]; ids = (0,4)
        residual,J,projected,blocks = S.finite_residual_jacobian(
            data['model'],data['z'],data['K'],data['points'],ids,lines)
        self.assertFalse(blocks[0]['interval_order_cusp'])
        self.assertFalse(any(blocks[0]['endpoint_hit_cusps']))
        self.assertEqual(len(set(blocks[0]['hinge_branches'])),3)
        np.testing.assert_allclose(projected,q,rtol=0.,atol=1e-12)
        np.testing.assert_allclose(residual,independent_residual(
            data['model'],data['z'],data['K'],data['points'],ids,lines),rtol=0.,atol=1e-10)
        numeric = np.empty_like(J); step = 1e-6  # Frozen parent CPU difference step.
        for k in range(6):
            plus,minus = data['z'].copy(),data['z'].copy(); plus[k]+=step; minus[k]-=step
            numeric[:,k] = (independent_residual(data['model'],plus,data['K'],data['points'],ids,lines)
                            -independent_residual(data['model'],minus,data['K'],data['points'],ids,lines))/(2*step)
        np.testing.assert_allclose(J,numeric,rtol=2e-5,atol=2e-5)

    def test_endpoint_reversal_and_observed_normal_sign_preserve_objective(self):
        data = fixture(); q = project(data['model'],data['z'],data['K'])
        line = numeric_line(q); a,b = S.EDGES[0]
        r,J,_,_ = S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(),[line])
        reversed_model = data['model'].copy(); reversed_model[[a,b]]=reversed_model[[b,a]]
        rr,JJ,_,_ = S.finite_residual_jacobian(reversed_model,data['z'],data['K'],data['points'],(),[line])
        np.testing.assert_allclose(rr[:2],r[:2][::-1],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(rr[2:],r[2:],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(JJ.T@JJ,J.T@J,rtol=1e-12,atol=1e-8)
        negative = copy.deepcopy(line); negative['normal']=-line['normal']; negative['offset']=-line['offset']
        rn,JN,_,_ = S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(),[negative])
        np.testing.assert_allclose(rn,-r,rtol=0.,atol=1e-10)
        np.testing.assert_allclose(JN.T@JN,J.T@J,rtol=1e-12,atol=1e-8)

    def test_point_only_callback_is_exact_parent_numpy_output(self):
        data = fixture(); points = data['points'].copy(); points[2]+=[40.,-40.]
        for modeled in (False,True):
            old = P13.finite_residual_jacobian(data['model'],data['z'],data['K'],points,(0,2,4,5),[],modeled_normals=modeled)
            actual = S.finite_residual_jacobian(data['model'],data['z'],data['K'],points,(0,2,4,5),[],modeled_normals=modeled)
            for left,right in zip(actual[:3],old[:3]):
                self.assertIsInstance(left,np.ndarray); self.assertEqual(left.dtype,right.dtype)
                self.assertEqual(left.shape,right.shape); np.testing.assert_array_equal(left,right)
            self.assertEqual(actual[3],old[3])

    def test_natural_pose_retained_bad_point_and_excluded_good_H(self):
        data = fixture(); results = []
        line = numeric_line(project(data['model'],data['z'],data['K']),fractions=(.2,.5,.8))
        raw = dict(edge=0,endpoints=list(S.EDGES[0]),normal=line['normal'].tolist(),
                   offset=line['offset'],queries=line['support_query_ids'],
                   support_points=line['source_line']['support_points'],query_radii_px=[1.,1.,1.])
        observation = dict(corners=[],lines=[raw],queries=[])
        for hidden_xy in ([10.,10.],[600.,450.]):
            points = data['points'].copy(); points[2]+=[40.,-40.]; points[6]=hidden_xy
            original = points.copy()
            solved = S.ObservedFrameFactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(
                observation,hidden=(6,),initial_pose=dict(available=False),point_result=dict(available=False))
            self.assertTrue(solved['available']); self.assertEqual(solved['state'],'NEW_POSE')
            self.assertIn(2,solved['used']); self.assertNotIn(6,solved['used'])
            self.assertGreaterEqual(len(solved['fit_input_ids']),4)
            self.assertTrue(set(solved['fit_input_ids']).isdisjoint((2,6,8)))
            self.assertTrue(set(solved['final_inliers']).isdisjoint((2,6,8)))
            np.testing.assert_allclose(solved['centroid'],data['z'][3:],rtol=0.,atol=1e-5)
            np.testing.assert_allclose(solved['R_cf'],rotation(data['z'][:3]),rtol=0.,atol=1e-5)
            self.assertEqual(solved['operation_counts']['hidden_reprojection_assignments'],1)
            self.assertFalse(solved['reprojected_points_reused_as_observations'])
            self.assertEqual(solved['fit_line_edges'],[0])
            self.assertGreater(solved['operation_counts']['refit_optimizer_calls'],0)
            self.assertEqual(solved['operation_counts']['direct_four_ID_requests'],math.comb(len(solved['used']),4))
            self.assertTrue(all(set(p['fit_input_ids']).isdisjoint((6,8)) for p in solved['proposal_solutions']))
            final = np.asarray(solved['points_final']); projected = np.asarray(solved['projected'])
            np.testing.assert_array_equal(final[6],projected[6])
            np.testing.assert_array_equal(final[[0,1,2,3,4,5,7,8]],original[[0,1,2,3,4,5,7,8]])
            np.testing.assert_array_equal(points,original)
            results.append(solved)
        for key in ('centroid','R_cf','fit_input_ids','final_inliers'):
            self.assertEqual(results[0][key],results[1][key])

    def test_source_isolation_proof_is_distinct_from_numerical_success(self):
        for key in ('finite_callback_inverse_whole_AST_equal',
                    'geometry_and_optimizer_method_bodies_whole_AST_unchanged',
                    'score_inverse_whole_AST_equal','solve_inverse_whole_AST_equal',
                    'point_only_callbacks_delegate_frozen_parent'):
            self.assertTrue(S.SOURCE_PROOF[key])
        self.assertFalse(S.SOURCE_PROOF['parent_globals_mutated'])
        self.assertTrue(S.SOURCE_PROOF['source_AST_checks_are_not_numerical_integration_tests'])
        self.assertFalse(S.POLICY['physical_ownership_certified'])
        self.assertFalse(S.POLICY['actual_integration_or_accuracy_verified'])


if __name__ == '__main__':
    unittest.main()
