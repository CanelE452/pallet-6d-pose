"""Reference-diagnostic contract fixtures; constructed C1/C2, no GT reads."""
import unittest
from copy import deepcopy
import numpy as np
from . import reference_support as B


def target(order):
    return dict(id='toy',split='train',R_fixed=np.eye(3).tolist(),t=[0.,0.,3.],symmetry_order=order)


class ReferenceContracts(unittest.TestCase):
    def _constructed_model(self):
        from types import SimpleNamespace
        from scripts.research.pallet_visibility_curriculum_20261013_v1.test_statewise_partial_BI_u_solver_v4_13 import typed_fixture
        from scripts.research.pallet_visibility_curriculum_20261013_v1.test_statewise_partial_boundary_u_solver_v4_6 import region
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_u_solver_v4_13 as P
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_outline_solver_v4_13 as PREP
        a,b=typed_fixture();prepared=PREP._prepare(*a[:7],a[7],[b],[region(a)])
        xyz,K,states,ms,ds,hw,cfg,bs,us,Q,prior,requested=prepared
        for m in ms:m.update(state=states[m['native_corner']],source='CONSTRUCTED_IMAGE_OBSERVATION')
        c=SimpleNamespace(xyz=xyz,K=K,states=states,measurements=ms,domains=ds,image_hw=hw,cfg=cfg,
            boundary=bs,regions=us,Q=Q,prior=prior,scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3),
            backend=SimpleNamespace(P=P),outline={},use_outline=False)
        groups=[(ms,bs,ds,us)]*4
        model=SimpleNamespace(ctx=c,constraints_by_h=groups,replaced_constraints_by_h=groups)
        physical=dict(R_physical=np.asarray(a[6]['R'])@Q,centroid=np.asarray(a[6]['R'])@xyz.mean(0)+a[6]['t'])
        return model,B.F.OLD._native_pose(physical,c)

    def test_c1_only_original(self):
        ps=B.registered_reference_phases(target(1));self.assertEqual(len(ps),1)
        self.assertEqual(ps[0]['phase_index'],0)

    def test_c2_exact_original_and_half_turn(self):
        ps=B.registered_reference_phases(target(2));self.assertEqual(len(ps),2)
        np.testing.assert_allclose(ps[1]['reference_pose']['R_physical'],np.diag([-1.,1.,-1.]),atol=1e-15)

    def test_quarter_turn_not_allowed(self):
        for n in [0,3,4,1.5]:
            with self.assertRaises(ValueError):B.registered_reference_phases(target(n))

    def test_reference_unchanged(self):
        t=target(2);before=deepcopy(t);B.registered_reference_phases(t);self.assertEqual(t,before)

    def test_center_stays_physical(self):
        t=target(2);t['t']=[.25,-.13,4.12]
        for phase in B.registered_reference_phases(t):self.assertEqual(phase['reference_pose']['centroid'],t['t'])

    def test_invalid_reference_rotation_rejected(self):
        t=target(1);t['R_fixed'][0][0]=2.
        with self.assertRaises(ValueError):B.registered_reference_phases(t)

    def test_reference_infinite_rejected(self):
        t=target(1);t['t'][0]=float('nan')
        with self.assertRaises(ValueError):B.registered_reference_phases(t)

    def test_native_Q_origin_roundtrip_both_reference_phases(self):
        from types import SimpleNamespace
        xyz=np.array([[x,y,z] for x in [0.,1.4] for y in [0.,.18] for z in [0.,.8]])
        Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        for phase in B.registered_reference_phases(target(2)):
            for q in [np.eye(3),Q]:
                c=SimpleNamespace(xyz=xyz,Q=q,scale=2.)
                p=B.F.OLD._native_pose(phase['reference_pose'],c);back=B.F.OLD._actual_pose(p,c)
                np.testing.assert_allclose(back['R_physical'],phase['reference_pose']['R_physical'],atol=1e-14)
                np.testing.assert_allclose(back['centroid'],phase['reference_pose']['centroid'],atol=1e-14)

    def test_nan_constraint_never_passes(self):
        d=B._constraint_summary([1.,np.nan],1e-7)
        self.assertFalse(d['feasible']);self.assertIsNone(d['minimum_slack']);self.assertEqual(d['finite_count'],1)

    def test_no_constraint_is_vacuous_only_not_B_confidence(self):
        d=B._constraint_summary([],1e-7);self.assertTrue(d['feasible']);self.assertEqual(d['count'],0)

    def test_plain_nonfinite_explicit_missing(self):
        self.assertEqual(B.plain(dict(v=np.array([1.,np.nan,np.inf]))),dict(v=[1.,None,None]))
        self.assertIs(B.plain(np.bool_(True)),True)

    def test_no_production_reference_import(self):
        from pathlib import Path
        p=Path(B.F.__file__).read_text()
        self.assertNotIn('import reference_support',p)
        self.assertNotIn('from .reference_support',p)

    def test_constructed_observation_diagnostics_all_sources(self):
        model,p=self._constructed_model();d=B.observation_diagnostics(model,p)
        self.assertGreater(len(d['full_points']),0);self.assertEqual(len(d['partial']),1)
        self.assertEqual(len(d['OT_domains']),1);self.assertEqual(len(d['U_regions']),1)
        self.assertTrue(d['original_joint_constraints']['feasible'])
        self.assertEqual(sum(x['likelihood_replaced_by_partial'] for x in d['full_points']),1)

    def test_large_full_residual_is_soft_not_new_hard_budget(self):
        model,p=self._constructed_model();model.ctx.measurements[0]['xy']+=1000.
        d=B.observation_diagnostics(model,p)
        self.assertGreater(d['full_points'][0]['residual_px'],1000.)
        self.assertTrue(d['full_points'][0]['residual_is_soft_not_a_hard_acceptance_test'])
        self.assertTrue(d['original_joint_constraints']['feasible'])

    def test_partial_source_and_original_component_dictionary_preserved(self):
        model,p=self._constructed_model();d=B.observation_diagnostics(model,p)
        self.assertIn('association_provenance',d['partial'][0]['original_observation'])
        self.assertIn('coarse_pose_prior_loss',d['original_objective_components'])


if __name__=='__main__':unittest.main()
