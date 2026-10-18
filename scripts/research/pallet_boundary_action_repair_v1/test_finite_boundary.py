"""Expected conditions are fixed here before executing the numerical tests.

Ten registered categories: finite vs same infinite line; range/endpoint and
singular states; far collapsed projection; infeasible input; fixed universe;
no B input; analytic/one-sided derivatives; scale/Q/archive/duplicate
invariance; optimizer status distinctions; independent intersection audit.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import rows, read, write, sha
from . import finite_boundary as F

EXPECTED = {
 'finite_vs_infinite': 'Same line and identical old energy; only finite segment containing intersection supported.',
 'range_and_singular': 'lambda<0/>1, endpoints/internal, parallel, near-zero length and nonpositive depth distinguished; off canvas alone remains valid.',
 'far_shrink': 'A far shrunken edge missing the normal intersection is rejected, never rewarded by profile removal.',
 'infeasible_initial': 'A fixed profile outside initial segment remains rejected; no fabricated observation or GT repair.',
 'fixed_universe': 'Every candidate evaluates same original supported profiles and consumed groups.',
 'no_B_input': 'No fixed supported B input preserves the original observation policy.',
 'derivatives': 'Interior analytic dE/dq and finite-constraint Jacobian agree with FD; endpoint one-sided support is correct.',
 'invariances': 'Uniform resolution, physical coordinate-basis Q, duplicate group/query and complete/partial replacement preserve semantics.',
 'status_contract': 'Optimizer success, numeric validity, original feasibility, finite support and fallback remain distinct.',
 'independent_geometry': 'Independent linear-system intersection agrees with production lambda and finite predicate.',
}


def toy():
    projected = np.array([[0., 0.], [10., 0.], [10., 3.], [0., 3.], [0., 5.], [10., 5.], [10., 8.], [0., 8.]])
    grid = np.arange(-10., 11.)
    lp = -.15*(grid-.37)**2; prob = np.exp(lp-np.max(lp)); prob /= prob.sum()
    query = dict(support_id='fixed0', supported=True, native_corner=0, native_neighbor=1,
        center_xy=[5., -.37], outward_normal_xy=[0., 1.], offset_grid_base_px=grid.tolist(),
        offset_log_probability=np.log(prob).tolist(), offset_probability=prob.tolist(), edge_location_base_px=.37)
    group = dict(native_corner=0, supported=True, reliability_weight=.7, queries=[query], supported_query_count=1)
    bundle = dict(config=F.OLD.configuration_dict(), image_hw=[80, 100], pixel_scale=1.,
        initial_projection_xy=projected.tolist(), groups=[group], consumed_native_ids=[0],
        consumed_group_ids=['full:0'], replaces_global_outline=True)
    return bundle, projected


def independent(query, projected, depth=None):
    a, b = np.asarray(projected[query['native_corner']]), np.asarray(projected[query['native_neighbor']])
    c, n = np.asarray(query['center_xy']), np.asarray(query['outward_normal_xy'])
    mat = np.column_stack([n, a-b])
    try:
        offset, lam = np.linalg.solve(mat, a-c)
    except np.linalg.LinAlgError:
        return None, False
    valid = bool(-F.LAMBDA_EPS <= lam <= 1+F.LAMBDA_EPS and np.linalg.norm(b-a) > 1e-5 and
        (depth is None or np.min(depth) > 0))
    return float(lam), valid


class FiniteTests(unittest.TestCase):
    def test_01_finite_vs_same_infinite_line(self):
        b, q = toy(); moved = q.copy(); moved[[0, 1], 0] += 20
        e1 = F.energy_and_gradient(b, q)[0]; e2 = F.energy_and_gradient(b, moved)[0]
        self.assertAlmostEqual(e1, e2)
        self.assertTrue(F.support_audit(b, q)['finite_support'])
        self.assertFalse(F.support_audit(b, moved)['finite_support'])

    def test_02_range_singular_depth_and_canvas(self):
        b, q = toy(); query = b['groups'][0]['queries'][0]
        for x, status in [(-1., 'BEFORE_FINITE_ENDPOINT'), (11., 'AFTER_FINITE_ENDPOINT'),
                          (0., 'SUPPORTED_FINITE_SEGMENT'), (5., 'SUPPORTED_FINITE_SEGMENT'), (10., 'SUPPORTED_FINITE_SEGMENT')]:
            z = dict(query, center_xy=[x, -.37]); r = F.intersection(z, q, 1., image_hw=b['image_hw'])
            self.assertEqual(r['geometry_status'], status)
        z = q.copy(); z[1] = z[0]+[0., 1.]
        self.assertEqual(F.intersection(query, z, 1.)['geometry_status'], 'PARALLEL_NORMAL_AND_EDGE')
        z[1] = z[0]+[1e-12, 0.]
        self.assertEqual(F.intersection(query, z, 1.)['geometry_status'], 'DEGENERATE_PROJECTED_EDGE')
        self.assertEqual(F.intersection(query, q, 1., depth=np.zeros(8))['geometry_status'], 'NONPOSITIVE_DEPTH')
        self.assertEqual(F.intersection(query, q, 1., depth=-np.ones(8))['geometry_status'], 'NONPOSITIVE_DEPTH')
        z = q-200; qq = dict(query, center_xy=(np.asarray(query['center_xy'])-200).tolist())
        r = F.intersection(qq, z, 1., image_hw=b['image_hw']); self.assertTrue(r['finite_support']); self.assertTrue(r['off_canvas_intersection'])

    def test_03_far_shrink(self):
        b, q = toy(); q[:,0] = 100+q[:,0]*1e-9
        a = F.support_audit(b, q)
        self.assertFalse(a['finite_support']); self.assertEqual(a['fixed_profile_count'], 1)
        self.assertGreaterEqual(F.energy_and_gradient(b, q)[0], 0.)

    def test_04_infeasible_initial_is_not_repaired(self):
        b, q = toy(); b['groups'][0]['queries'][0]['center_xy'][0] = 50
        self.assertFalse(F.support_audit(b, q)['finite_support'])
        self.assertEqual(b['groups'][0]['queries'][0]['center_xy'][0], 50)

    def test_05_fixed_universe(self):
        b, q = toy(); before = F.universe(b)
        for delta in [0., 5., 1e6]:
            a = F.support_audit(b, q+delta)
            self.assertEqual(a['fixed_profile_count'], 1)
            self.assertEqual(F.universe(b), before)
        self.assertEqual(a['candidate_specific_observation_deletions'], 0)

    def test_06_no_B_input_original_policy(self):
        from . import finite_solver as S
        self.assertTrue(F.support_audit(None, np.zeros((8, 2)))['finite_support'])
        self.assertEqual(F.support_audit(None, np.zeros((8, 2)))['fixed_profile_count'], 0)
        fake = type('Model', (), {'ctx': type('Context', (), {'xyz': None, 'K': None, 'scale': 1.})(), 'bundle': None})()
        with patch.object(S.OLD.J, 'projection_and_jacobian', return_value=(np.zeros((8, 2)), np.ones(8), None, None)), patch.object(S.OLD, '_old_components', return_value=(12., np.ones(6), {'full': 12.})) as spy:
            value, grad, detail = S.OLD.objective(fake, np.zeros(6), 'B')
            self.assertEqual(value, 12.); self.assertFalse(detail['B_supported']); spy.assert_called_once_with(fake, unittest.mock.ANY)

    def test_07_analytic_and_one_sided_derivatives(self):
        b, q = toy(); q[[0,1],1] += .21
        value, grad = F.energy_and_gradient(b, q)
        signs = F.constraint_orientation(b, q)
        g, jac = F.constraints_and_jacobian(b, q, orientations=signs)
        h = 1e-5
        for i in range(16):
            hi, lo = q.copy(), q.copy(); hi.flat[i] += h; lo.flat[i] -= h
            fd = (F.energy_and_gradient(b, hi)[0]-F.energy_and_gradient(b, lo)[0])/(2*h)
            self.assertAlmostEqual(grad.flat[i], fd, delta=1e-7)
            df = (F.constraints_and_jacobian(b, hi, orientations=signs)[0]-F.constraints_and_jacobian(b, lo, orientations=signs)[0])/(2*h)
            np.testing.assert_allclose(jac[:,i], df, atol=1e-8, rtol=1e-6)
        from . import finite_solver as S
        fake = type('Model', (), {'ctx': type('Context', (), {'xyz': np.array([[.2, .1, -.3], [.4, .2, .1]]), 'scale': 2.3})()})()
        p0 = np.array([.1, -.2, .3, .4, -.5, 2.]); pp = p0+np.array([.03, -.04, .02, -.01, .02, .03])
        reg = dict(rotation_precision=3., translation_precision=5.)
        vv, gg = S.local_penalty(pp, p0, fake, reg)
        for i in range(6):
            hi, lo = pp.copy(), pp.copy(); hi[i] += h; lo[i] -= h
            df = (S.local_penalty(hi, p0, fake, reg)[0]-S.local_penalty(lo, p0, fake, reg)[0])/(2*h)
            self.assertAlmostEqual(gg[i], df, delta=1e-8)
        query = b['groups'][0]['queries'][0]
        endpoint = dict(query, center_xy=[0., 0.])
        self.assertTrue(F.intersection(endpoint, q, 1.)['finite_support'])
        self.assertFalse(F.intersection(dict(endpoint, center_xy=[-1e-4, 0.]), q, 1.)['finite_support'])

    def test_08_scale_Q_archive_and_duplicate(self):
        b, q = toy(); value = F.energy_and_gradient(b, q)[0]
        scaled = deepcopy(b); scaled['image_hw'] = [160, 200]; scaled['pixel_scale'] = 2.
        scaled['initial_projection_xy'] = (np.asarray(b['initial_projection_xy'])*2).tolist()
        for query in scaled['groups'][0]['queries']:
            query['center_xy'] = (np.asarray(query['center_xy'])*2).tolist()
        self.assertAlmostEqual(F.energy_and_gradient(scaled, q*2)[0], value)
        self.assertEqual(F.support_audit(scaled, q*2)['finite_support'], F.support_audit(b, q)['finite_support'])
        xyz = np.array([[i, i*i, i+3.] for i in range(8)])*.1
        Q = Rotation.from_rotvec([.2, -.3, .1]).as_matrix(); R = Rotation.from_rotvec([-.1, .4, .2]).as_matrix()
        np.testing.assert_allclose(xyz@R.T, (xyz@Q.T)@(R@Q.T).T, atol=1e-14)
        repeated = deepcopy(b); repeated['groups'] += deepcopy(repeated['groups']); repeated['groups'][0]['queries'] *= 2
        self.assertAlmostEqual(F.energy_and_gradient(repeated, q)[0], value)
        self.assertEqual(F.support_audit(repeated, q)['fixed_profile_count'], 1)
        from . import finite_solver as S
        duplicated_groups = deepcopy(b); duplicated_groups['groups'] *= 2
        with patch.object(S.OLD, 'compile_frame', return_value='compiled') as compile_spy:
            self.assertEqual(S.compile_models({}, {}, duplicated_groups), 'compiled')
            self.assertEqual(len(compile_spy.call_args.kwargs['bundle']['groups']), 1)
            self.assertEqual(compile_spy.call_args.kwargs['bundle']['groups'][0], b['groups'][0])
        fake = type('Model', (), {'ctx': type('Context', (), {'xyz': None, 'K': None, 'scale': 1.})(), 'bundle': b})()
        with patch.object(S.OLD.J, 'projection_and_jacobian', return_value=(q, np.ones(8), np.zeros((8,2,6)), None)), patch.object(S.OLD, '_old_components', return_value=(3., np.zeros(6), {})) as spy:
            S.OLD.objective(fake, np.zeros(6), 'B')
            self.assertEqual(spy.call_args.args[2], {0}); self.assertFalse(spy.call_args.kwargs['outline'])

    def test_09_optimizer_status_fields_not_conflated(self):
        from . import finite_solver as S
        frame = dict(fallback=dict(available=True, R_physical=np.eye(3).tolist(), centroid=[0.,0.,1.]))
        pred, scores = S._select([], {}, frame, 'B')
        self.assertTrue(pred['fallback']); self.assertIsNone(pred['finite_support'])
        self.assertNotEqual(pred['prediction_status'], 'SELECTED')
        self.assertTrue(pred['support_is_fallback_not_new_B_output'])

    def test_10_independent_linear_intersection(self):
        b, q = toy(); rng = np.random.default_rng(1234)
        for _ in range(30):
            candidate = q+rng.normal(0., .3, size=q.shape)
            query = b['groups'][0]['queries'][0]
            lam, support = independent(query, candidate)
            production = F.intersection(query, candidate, 1., image_hw=b['image_hw'])
            self.assertAlmostEqual(lam, production['lambda_segment'], delta=1e-12)
            self.assertEqual(support, production['finite_support'])


def regression(original_root):
    """Original known three failures: fixed poses only, no new error scoring."""
    from . import finite_solver as S
    root = Path(original_root); config = read(root/'CONFIG_FROZEN.json')
    wanted = {'wood_day_01:005249': 'B_FULL', 'plastic_day_01:012545': 'CTRL_EDGE_BOUNDARY', 'wood_184309:000432': 'AB_FULL'}
    found = []
    for i, frame in enumerate(rows(root/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz')):
        if frame['id'] not in wanted:
            continue
        folder = root/'real_batch_all_available_v1/frames'/f'{i:04d}'
        result = next(rows(folder/'RESULT.jsonl.gz')); bundle = next(rows(folder/'BOUNDARY_BUNDLE.jsonl.gz'))
        arm = wanted[frame['id']]; pred = result['predictions'][arm]
        model = S.OLD.compile_frame(frame, config, bundle=bundle)[pred['selected_family']]
        candidate = dict(R_physical=pred['actual_pose']['R_physical'], centroid=pred['actual_pose']['centroid'])
        q, depth, _, _ = S.projected(model, S.OLD._native_pose(candidate, model.ctx))
        hs = [0] if arm != 'AB_FULL' else list(range(4))
        for h in hs:
            perm = model.owner['permutations'][h]
            qq = q[list(perm)] if arm == 'AB_FULL' else q
            pp = depth[list(perm)] if arm == 'AB_FULL' else depth
            audit = F.support_audit(bundle, qq, pp)
            independent_rows = [independent(query, qq, pp) for query in F.fixed_queries(bundle)]
            for direct, prod in zip(independent_rows, audit['profiles']):
                if direct[0] is not None and prod['lambda_segment'] is not None:
                    np.testing.assert_allclose(direct[0], prod['lambda_segment'], atol=1e-4, rtol=1e-8)
            if audit['finite_support']:
                raise AssertionError('Known extension-line failure must be rejected')
            found.append(dict(id=frame['id'], old_arm=arm, original_selected_family=pred['selected_family'], global_owner_h=h,
                finite_support=False, fixed_profiles=audit['fixed_profile_count'], supported_profiles=audit['supported_profiles'],
                lambda_values=[p['lambda_segment'] for p in audit['profiles']],
                independent_geometry_agrees=True, geometry_only_no_new_accuracy_scoring=True))
    if len(found) != 6:
        raise ValueError('All three original regressions and four AB owners required')
    return found


if __name__ == '__main__':
    import argparse
    import io
    parser = argparse.ArgumentParser(); parser.add_argument('--output'); parser.add_argument('--original-root')
    args = parser.parse_args(); before = time.monotonic()
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FiniteTests))
    print(stream.getvalue())
    record = dict(schema='FINITE_B_GEOMETRY_TESTS_V1', expected_before_execution=EXPECTED,
        status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
        errors=[str(x) for x in result.errors], failures=[str(x) for x in result.failures],
        numeric_pass=result.wasSuccessful(), finite_support_pass=result.wasSuccessful(),
        actual_material_owner='UNVERIFIED', source_sha256=sha(__file__), elapsed_seconds=time.monotonic()-before)
    if args.original_root and result.wasSuccessful():
        try:
            record['original_fixed_pose_regressions'] = regression(args.original_root)
        except BaseException as exc:
            record.update(status='FAILED_TECHNICAL', regression_error=repr(exc))
            if args.output:
                write(args.output, record)
            raise
    if args.output:
        write(args.output, record)
    if not result.wasSuccessful():
        raise SystemExit(1)
