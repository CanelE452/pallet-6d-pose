"""Synthetic composition contracts only; no image files, data labels or fits."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import continuous_outline_problem_v4_24 as NEW
from . import bounded_geometric_priors_problem_v4_22 as PRIORS
from .test_global_outline_analytic_factor_v4_11 import scene
from .test_native_edge_problem_v4_17 import problem as native_fixture


class DummyBase:
    """Separate differentiable N/O/S terms and an untouched hardguard API."""
    def __init__(self,xyz,K,scale):
        self.xyz,self.K,self.scale=xyz,K,scale
        self.sides=[('left',)];self.Q=np.eye(3)
        self.constraint_calls=[];self.evaluate_calls=0

    def objective(self,p):
        p=np.asarray(p,float);d=p-np.array([.2,-.3,.1,.15,-.2,1.])
        coefficients=np.array([2.,3.,4.,5.,6.,7.])
        native=float(np.sum(coefficients*d*d));ng=2.*coefficients*d
        ov=float(.7*(p[0]+.1)**2);og=np.array([1.4*(p[0]+.1),0.,0.,0.,0.,0.])
        sv=float(.4*(p[4]-.3)**2);sg=np.array([0.,0.,0.,0.,.8*(p[4]-.3),0.])
        self.last_parts=dict(native_costs=[dict(kind='COMPLETE_BI',native_corner=0,weighted_robust_cost=native)],
            regularizer_costs=[dict(kind=PRIORS.REGULARIZER_KINDS[0],weighted_cost=ov),dict(kind=PRIORS.REGULARIZER_KINDS[1],weighted_cost=sv)],
            native_observation_objective=native,regularizer_objective=ov+sv,total_regularized_objective=native+ov+sv,
            S_image_or_2D_terms=0,O_point_terms=0,U_image_terms=0,native_costs_exclude_regularizers=True)
        return native+ov+sv,ng+og+sg,self.last_parts

    def constraints(self,p,sides):
        p=np.asarray(p,float);self.constraint_calls.append((p.copy(),sides))
        return np.array([p[3]+2.,3.-p[4]]),np.array([[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,-1.,0.]])

    def evaluate(self,R,t):
        self.evaluate_calls+=1
        p=NEW.G._pose_vector(np.asarray(R),np.asarray(t),self.scale)
        value,_,parts=self.objective(p)
        self.last_evaluation=dict(accepted=True,objective=value,**parts,
            all_original_hard_guards_recomputed=True,scalar_and_vector_guard_token='EXACT_BASE_GUARDS',
            R=np.asarray(R).tolist(),t=np.asarray(t).tolist(),R_physical=(np.asarray(R)@self.Q).tolist())
        return self.last_evaluation


def numeric(fun,p,step=1e-7):
    return np.array([(fun(p+np.eye(6)[j]*step)-fun(p-np.eye(6)[j]*step))/(2.*step)for j in range(6)])


@pytest.mark.parametrize('translation',[[0.,.3,2.2],[-.9,.3,2.2]])
def test_total_N_plus_O_S_plus_one_weighted_outline_and_FD_gradient(translation):
    xyz,K,scale,p,outline=scene(translation,[.12,.65,.07]);base=DummyBase(xyz,K,scale)
    weight=2.75;cfg={'outline_factor_weight':weight};wrapped=NEW.ContinuousOutlineProblem(base,outline,cfg)
    p=p+np.array([.013,-.017,.011,.009,-.007,.019])
    nv,ng,na=base.objective(p);saved=deepcopy(na)
    mv,mg,_=NEW.OUTLINE.outline_value_and_gradient(p,xyz,K,scale,outline,wrapped.outline_configuration)
    value,gradient,audit=wrapped.objective(p)
    assert value==pytest.approx(nv+weight*mv,rel=1e-13,abs=1e-10)
    np.testing.assert_allclose(gradient,ng+weight*mg,rtol=1e-13,atol=1e-10)
    np.testing.assert_allclose(gradient,numeric(lambda x:wrapped.objective(x)[0],p),rtol=2e-5,atol=3e-4)
    assert na==saved  # Source native/O/S lists were not mutated.
    assert audit['native_costs']==saved['native_costs']and audit['regularizer_costs']==saved['regularizer_costs']
    assert audit['native_observation_objective']==saved['native_observation_objective']
    assert audit['regularizer_objective']==saved['regularizer_objective']
    assert audit['total_regularized_objective']==saved['total_regularized_objective']
    assert audit['mask_count_in_optimizer']==1 and audit['global_mask_weight']==weight
    assert audit['weighted_mask_objective']==pytest.approx(weight*mv)
    assert audit['native_plus_coarse_regularized_objective']==nv
    assert audit['total_optimization_objective']==pytest.approx(value)
    assert audit['mask_not_in_native_costs_or_O_S_regularizers']
    assert len(audit['native_costs'])==1 and len(audit['regularizer_costs'])==2
    R=Rotation.from_rotvec(p[:3]).as_matrix();t=p[3:]*scale
    evaluated=wrapped.evaluate(R,t)
    assert evaluated['objective']==pytest.approx(value,rel=1e-12,abs=1e-9)
    assert evaluated['total_optimization_objective']==pytest.approx(value,rel=1e-12,abs=1e-9)
    assert evaluated['native_costs']==audit['native_costs']and evaluated['regularizer_costs']==audit['regularizer_costs']
    assert evaluated['accepted']and evaluated['scalar_and_vector_guard_token']=='EXACT_BASE_GUARDS'


def test_real_native_and_bounded_prior_guards_are_delegated_exactly():
    args,_,_,native=native_fixture(typed=True)
    R=np.asarray(args[6]['R']);t=np.asarray(args[6]['t']);p=NEW.G._pose_vector(R,t,native.scale)
    oid=next(i for i,s in enumerate(native.states)if s=='O')
    q=NEW.G.project_native(native.xyz,R,t,native.K)[0]
    source=dict(native_corner=oid,state='O',shape='ellipse',localization_available=True,independent_precise_support=False,
        center_xy=(q[oid]-[24.,10.]).tolist(),half_axes_px=[4.,7.],angle_deg=17.)
    regions=PRIORS.O.prepare_o_localization_regions([source],native.states,[s=='S'for s in native.states])
    model=dict(schema=PRIORS.S.MODEL_SCHEMA,fit_partition='CAL',fixed_pseudocounts=PRIORS.S.PSEUDOCOUNTS,
        geometry_tolerance_deg=2.,epsilon=.3)
    model['model_digest_sha256']=PRIORS.S.model_digest(model)
    sp=PRIORS.S.prepare_bounded_coarse_visibility_prior(native.xyz,R,t,[s=='S'for s in native.states],model)
    base=PRIORS.BoundedGeometricPriorsProblem(native,regions,dict(epsilon=.2,huber_scale=3.),sp)
    mask=np.zeros(native.image_hw,np.uint8)
    cv2.fillConvexPoly(mask,np.rint(cv2.convexHull(q.astype(np.float32))).astype(np.int32),1)
    outline=NEW.OUTLINE.observe_global_outline(mask,native.image_hw)
    wrapped=NEW.ContinuousOutlineProblem(base,outline,{'outline_factor_weight':1.7})
    assert wrapped.Q is native.Q and wrapped.measurements is native.measurements
    assert wrapped.domains is native.domains and wrapped.regions is native.regions
    for sides in native.sides:
        sv,sj=base.constraints(p,sides);wv,wj=wrapped.constraints(p,sides)
        assert np.array_equal(sv,wv)and np.array_equal(sj,wj)
    value,_,audit=wrapped.objective(p);be=base.evaluate(R,t);we=wrapped.evaluate(R,t)
    assert we['objective']==pytest.approx(value,abs=1e-9,rel=1e-12)
    for key in ('accepted','minimum_joint_slack','visible_protection_audit','joint_constraint_audit',
            'U_region_constraint_audit','boundary_observation_audit','R_physical','all_original_hard_guards_recomputed'):
        assert we[key]==be[key]
    assert we['native_costs']==be['native_costs']and we['regularizer_costs']==be['regularizer_costs']
    assert audit['mask_count_in_optimizer']==1


@pytest.mark.parametrize('supported,weight',[(False,3.),(True,0.)])
def test_inactive_outline_has_exact_numeric_identity_without_geometry_or_projection(supported,weight,monkeypatch):
    xyz,K,scale,p,_=scene([0.,.3,2.2],[.12,.65,.07]);base=DummyBase(xyz,K,scale)
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('Disabled outline queried projection geometry')
    base.xyz=Poison();base.K=Poison()
    def forbidden(*args,**kwargs):raise AssertionError('Disabled outline projection called')
    monkeypatch.setattr(NEW.OUTLINE,'outline_value_and_gradient',forbidden)
    wrapped=NEW.ContinuousOutlineProblem(base,dict(supported=supported),{'outline_factor_weight':weight})
    bv,bg,bp=base.objective(p);value,gradient,audit=wrapped.objective(p)
    assert value==bv and np.array_equal(gradient,bg)
    assert audit['native_costs']==bp['native_costs']and audit['regularizer_costs']==bp['regularizer_costs']
    assert audit['global_mask_cost']==audit['weighted_mask_objective']==0.
    assert audit['mask_count_in_optimizer']==0 and not audit['continuous_mask_gradient_active']
    mv,mg,_=wrapped.mask_value_and_gradient(Poison())
    assert mv==0. and np.array_equal(mg,np.zeros(6))
    R=Rotation.from_rotvec(p[:3]).as_matrix();t=p[3:]*scale
    be=base.evaluate(R,t);we=wrapped.evaluate(R,t)
    assert we['objective']==be['objective']and we['accepted']==be['accepted']
    assert we['native_costs']==be['native_costs']and we['regularizer_costs']==be['regularizer_costs']
    sv,sj=base.constraints(p,base.sides[0]);wv,wj=wrapped.constraints(p,base.sides[0])
    assert np.array_equal(sv,wv)and np.array_equal(sj,wj)


def test_pre_rejected_base_and_invalid_outline_do_not_forge_acceptance(monkeypatch):
    xyz,K,scale,p,outline=scene([0.,.3,2.2],[.12,.65,.07]);base=DummyBase(xyz,K,scale)
    rejected=dict(accepted=False,objective=None,reason='base rejected',native_costs=[])
    monkeypatch.setattr(base,'evaluate',lambda R,t:rejected)
    def forbidden(*args,**kwargs):raise AssertionError('Previously rejected base should not evaluate outline')
    monkeypatch.setattr(NEW.OUTLINE,'outline_value_and_gradient',forbidden)
    wrapped=NEW.ContinuousOutlineProblem(base,outline,{'outline_factor_weight':2.})
    assert wrapped.evaluate(object(),object())is rejected
    monkeypatch.undo()
    base=DummyBase(xyz,K,scale);wrapped=NEW.ContinuousOutlineProblem(base,outline,{'outline_factor_weight':2.})
    monkeypatch.setattr(NEW.OUTLINE,'outline_value_and_gradient',lambda *a,**k:(1e12,np.zeros(6),dict(valid_projection=False)))
    R=Rotation.from_rotvec(p[:3]).as_matrix();t=p[3:]*scale
    evaluated=wrapped.evaluate(R,t)
    assert not evaluated['accepted']and evaluated['objective']is None
    assert evaluated['outline_factor_numeric_validity_failed_not_new_physical_guard']
    with pytest.raises(ValueError,match='valid existing whole-outline'):wrapped.objective(p)
