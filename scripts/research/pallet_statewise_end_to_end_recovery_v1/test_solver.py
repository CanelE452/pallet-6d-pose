"""Independent integration contracts; two CAL input packets, never references.

CAL tests compare original objective arithmetic, not pose accuracy. Synthetic
OpenCV degeneracy fixtures do not use evaluation images or labels.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from scripts.research.pallet_selection_score_diagnosis_v1 import fixed_cost
from .common import rows
from . import owner_model as A
from . import solver as S
from .test_owner_model import XYZ,K,Q


INPUTS=Path('/home/minjae/Documents/pallet-research/pallet-statewise-end-to-end-recovery-v1/PREDICTION_INPUTS238.jsonl.gz')


def numerical_gradient(fun,p,step=1e-6):
    result=np.zeros_like(p)
    for i in range(len(p)):
        hi=p.copy();lo=p.copy();hi[i]+=step;lo[i]-=step
        result[i]=(fun(hi)-fun(lo))/(2.*step)
    return result


class SolverContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames=[]
        if INPUTS.exists():
            for frame in rows(INPUTS):
                if frame['partition']=='CAL':cls.frames.append(frame)
                if len(cls.frames)==2:break
        cls.models=[S.compile_frame(f,dict(none_prior=.05)) for f in cls.frames]

    def require_CAL(self):
        if len(self.frames)!=2:self.skipTest('Local sealed CAL input packet fixture is not public data')

    def test_fixed_original_cost_parity_both_families_two_saved_CAL_predictions(self):
        self.require_CAL()
        for frame,models in zip(self.frames,self.models):
            saved=frame['packet']['e0_pose']
            candidate=dict(R_physical=saved['R_physical'],centroid=saved['centroid'])
            for family,model in models.items():
                p=S._native_pose(candidate,model.ctx);R,t=S.V._pose(p,model.ctx.scale)
                original=fixed_cost.evaluate(frame['packet'],'G13',family,
                    dict(R=R,t=t,Q=model.ctx.Q,objective_scope='COUNTERFACTUAL_FIXED_POSE'),
                    context=model.ctx,strict=False)
                value,_,_=S.objective(model,p,'fixed')
                self.assertAlmostEqual(value,original['objective_recomputed'],places=8)

    def test_native_projection_chain_and_A_parameter_gradient(self):
        self.require_CAL();model=self.models[0]['ORIGINAL_WD'];c=model.ctx
        candidate=self.frames[0]['packet']['e0_pose'];p=S._native_pose(candidate,c)
        p=p+np.array([.001,-.002,.001,.0002,-.0003,.0001])
        q,_,jac,_=S.J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
        for i in range(8):
            for axis in range(2):
                numerical=numerical_gradient(lambda x:S.J.projection_and_jacobian(x,c.xyz,c.K,c.scale)[0][i,axis],p)
                np.testing.assert_allclose(jac[i,axis],numerical,rtol=2e-6,atol=1e-5)
        for h in (0,2):
            _,gradient,_=S.objective(model,p,'A',conditional_h=h)
            numerical=numerical_gradient(lambda x:S.objective(model,x,'A',conditional_h=h)[0],p)
            np.testing.assert_allclose(gradient,numerical,rtol=2e-4,atol=2e-4)

    def test_final_A_mask_and_owner_transport_use_same_h(self):
        self.require_CAL();frame=self.frames[0];models=self.models[0]
        saved=frame['packet']['e0_pose'];candidate=dict(R_physical=saved['R_physical'],centroid=saved['centroid'])
        scored=S.candidate_scores(candidate,models)
        for family,model in models.items():
            score=scored[family];detail=score['audits']['A'];per=detail['owner']['per_h_total_energies']
            good=np.asarray(detail['jointly_feasible_global_h'],bool)
            if good.any():
                expected=-2.*logsumexp(np.log(model.owner['normalized_global_priors'][good])-.5*np.asarray(per)[good])+detail['outline']
                self.assertAlmostEqual(score['scores']['A'],expected,places=10)
                responsibility=np.asarray(detail['owner']['global_owner_responsibilities'])
                np.testing.assert_array_equal(responsibility[~good],np.zeros(np.sum(~good)))
                self.assertAlmostEqual(responsibility.sum(),1.,places=12)
            for h,p in enumerate(model.owner['permutations']):
                ms,bs,domains,regions=model.constraints_by_h[h]
                self.assertEqual(len(ms),len(model.ctx.measurements));self.assertEqual(len(bs),len(model.ctx.boundary))
                for m,old in zip(ms,model.ctx.measurements):
                    self.assertEqual(m['native_corner'],p[old['native_corner']])
                    np.testing.assert_array_equal(m['object_xyz'],model.ctx.xyz[m['native_corner']])
                    expected_normals=S.V.registered_incident_normals(model.ctx.xyz)[m['native_corner']]
                    np.testing.assert_array_equal(m['incident_normals'],expected_normals)
                    np.testing.assert_array_equal(m['xy'],old['xy'])

    def test_B_consumption_does_not_resurrect_revoked_full_or_keep_old_hull(self):
        self.require_CAL();frame=deepcopy(self.frames[0]);base=self.models[0]['ORIGINAL_WD']
        partial={b['native_corner'] for b in base.ctx.boundary}
        full={m['native_corner'] for m in base.ctx.measurements}-partial
        consumed=sorted(partial|({next(iter(full))} if full else set()))
        if not consumed:self.skipTest('No B-like group available in fixture')
        model=S.compile_frame(frame,dict(none_prior=.05),bundle={'consumed_native_ids':consumed})['ORIGINAL_WD']
        p=S._native_pose(frame['packet']['e0_pose'],model.ctx)
        with patch.object(S,'_boundary_energy',return_value=(0.,np.zeros((8,2)))):
            value,_,_=S.objective(model,p,'B')
            for m in model.ctx.measurements:
                if m['native_corner'] in consumed:m['xy']=m['xy']+np.array([50.,-35.])
            for b in model.ctx.boundary:
                if b['native_corner'] in consumed:
                    for line in b['lines']:line['offset_px']+=50.
            model.ctx.outline_cfg['outline_factor_weight']*=1000.
            changed,_,detail=S.objective(model,p,'B')
            self.assertEqual(detail['parts']['outline'],0.)
            self.assertAlmostEqual(value,changed,places=10)

    def test_S_is_not_an_image_factor_and_allU_has_no_geometric_precision(self):
        self.require_CAL()
        for frame,models in zip(self.frames,self.models):
            for model in models.values():
                self.assertTrue(all(model.ctx.states[g['native_corner']] in ('B','I') for g in model.owner['groups']))
        model=A.compile_owner_groups(XYZ,K,[],[],(480,640),none_prior=.05,states=['U']*8)
        self.assertEqual(A.energy(model,Q),A.energy(model,Q+100.))
        self.assertEqual(A.energy(model,Q),0.)

    def test_same_eight_proposals_per_full_arm_preserve_all_original_poses(self):
        self.require_CAL();frame=deepcopy(self.frames[0]);models=self.models[0]
        frame['candidates']=frame['candidates'][:1]
        calls=[]
        def scored(candidate,unused):
            return {f:dict(scores={m:float(np.linalg.norm(candidate['centroid'])) for m in ('fixed','A','B','AB','edge')},
                feasible={m:True for m in ('fixed','A','B','AB','edge')},audits={},h_min_slack={}) for f in models}
        def solve(model,start,mode,h,config):
            calls.append((mode,model.ctx.family_name,h,start['candidate_id']))
            return dict(candidate_id=f'TEST/{len(calls)}',R_physical=start['R_physical'],centroid=start['centroid'],
                        source='NEW_'+mode,solve=dict(mode=mode,feasible=True,optimizer_success=True))
        with patch.object(S,'compile_frame',return_value=models),patch.object(S,'candidate_scores',side_effect=scored),\
             patch.object(S,'solve_one',side_effect=solve),patch.object(S,'ransac_control',return_value=dict(actual_pose=frame['fallback'],fallback=True)):
            result=S.generate_candidates(frame,dict(starts_per_family=4,max_iterations=2,optimizer_ftol=1e-8,none_prior=.05))
        counts=Counter(mode for mode,f,h,start in calls if start!='N0_LOCAL_START')
        self.assertEqual(counts,dict(fixed=8,A=8,B=8,AB=8,edge=8))
        original=result['carried'][0]
        for mode,bank in result['banks'].items():
            self.assertTrue(any(np.array_equal(c['R_physical'],original['R_physical']) and
                                np.array_equal(c['centroid'],original['centroid']) for c in bank))
        proposals=[start for mode,f,h,start in calls if start!='N0_LOCAL_START']
        self.assertTrue(all('/PROPOSED_NATIVE_YAW_' in p for p in proposals))

    def test_three_point_P3P_and_planar_IPPE_return_multiple_solutions(self):
        R=Rotation.from_rotvec([.13,.27,-.08]).as_matrix();q,_=A.project_native(XYZ,R,[.1,-.05,3.],K)
        ids=[0,1,4]
        count,rv,tv=cv2.solveP3P(np.ascontiguousarray(XYZ[ids]),np.ascontiguousarray(q[ids]),K,None,flags=cv2.SOLVEPNP_P3P)
        self.assertGreaterEqual(count,2)
        plane=[0,1,4,5]
        solutions=cv2.solvePnPGeneric(np.ascontiguousarray(XYZ[plane]),np.ascontiguousarray(q[plane]),K,None,flags=cv2.SOLVEPNP_IPPE)
        self.assertGreaterEqual(len(solutions[1]),2)
        # Multiple numerical roots are retained; rank/low residual never proves
        # globally unique physically correct 6D pose.

    def test_good_existing_CAL_initialization_is_retained_without_precision_claim(self):
        self.require_CAL();frame=self.frames[0];model=self.models[0]['ORIGINAL_WD']
        saved=frame['packet']['e0_pose'];p=S._native_pose(saved,model.ctx)
        before=S.objective(model,p,'fixed')[0]
        start=dict(candidate_id='CAL_GOOD_INIT_TECHNICAL',R_physical=saved['R_physical'],centroid=saved['centroid'])
        result=S.solve_one(model,start,'fixed',0,dict(max_iterations=5,optimizer_ftol=1e-8))
        self.assertTrue(result['solve']['feasible'])
        self.assertLessEqual(result['solve']['objective'],before+1e-7)
        R=np.asarray(result['R_physical'])
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0)
        self.assertGreater(np.linalg.det(R),0.)


if __name__=='__main__':unittest.main()
