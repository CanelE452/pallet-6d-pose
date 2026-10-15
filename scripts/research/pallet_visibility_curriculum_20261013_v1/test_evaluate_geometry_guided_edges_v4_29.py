"""Uniform five-arm wiring/selection; no actual images, fits or metrics."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from . import evaluate_geometry_guided_edges_v4_29 as R
from .test_geometry_guided_edge_problem_v4_29 import fixture

STATES=['B','I','U','O','T','B','S','S']


def candidate(name,native=(1.,1.),prior=(0.,0.),shape=0.):
    return dict(name=name,family='ORIGINAL_WD',accepted=True,all_joint_constraints_satisfied=True,
        native_costs=[dict(kind='BI_COMPLETE',native_corner=k,weighted_robust_cost=v)for k,v in zip((0,1),native)],
        regularizer_costs=[dict(kind=k,weighted_cost=v)for k,v in zip(R.BOUNDED.REGULARIZER_KINDS,prior)],global_mask_cost=shape)


def test_selector_does_not_pareto_prefilter_and_counts_OS_C_once():
    candidates=[candidate('native_best',prior=(5.,3.),shape=2.),candidate('sum_best',native=(2.,2.),prior=(.2,.3),shape=.5)]
    result=R.select_candidates(STATES,candidates,containment_weight=1.)
    assert result['selected_name']=='sum_best'
    assert len(result['candidates'])==2 and result['candidates'][1]['total_energy']==pytest.approx(5.)
    assert result['native_Pareto_audit_not_prefilter']


def test_native_ties_use_stable_order_without_old_mask_tie():
    result=R.select_candidates(STATES,[candidate('first',shape=100.),candidate('later',native=(1.+1e-15,1.),shape=0.)],containment_weight=0.)
    assert result['selected_name']=='first' and not result['old_full_mask_cost_or_numeric_tie_break_used']


def test_hidden_full_pose_prior_or_unknown_new_cost_kind_rejected():
    c=candidate('bad');c['regularizer_costs'].append(dict(kind='CAL_COARSE_POSE_PRIOR',weighted_cost=0.))
    with pytest.raises(ValueError,match='no hidden'):
        R.select_candidates(STATES,[c],containment_weight=0.)
    with pytest.raises(KeyError):R._bridge([dict(kind='S_POINT',native_corner=6,weighted_robust_cost=1.)])


def test_empty_feasible_bank_uses_own_coarse_not_numeric_e_pose(monkeypatch):
    base,bundle,_,rotation,translation=fixture()
    class Rejected:
        plan=dict(new_owner_ids=[],active_groups=[],corner_plans=[])
        def evaluate(self,*args):return dict(accepted=False,objective=None,native_costs=[],reason='synthetic hardguard rejection')
    family=dict(name='ORIGINAL_WD',native_xyz=base.xyz.tolist(),physical_registration_Q=np.eye(3).tolist(),
        cf_extents=np.ptp(base.xyz,axis=0).tolist(),coarse_pose=dict(R=rotation.tolist(),t=translation.tolist()))
    swapped=deepcopy(family);swapped['name']='SWAPPED_WD'
    task=dict(id='synthetic',partition='CAL',session='synthetic',image='unused.png',image_sha256='a'*64,
        states=base.states,initial_self_hidden=[s=='S'for s in base.states],spec=dict(K=base.K.tolist(),visible=[
            deepcopy(a['source_record'])for a in base.prepartial_bound_source_archives.values()]),
        boundary_merged=[],families=dict(ORIGINAL_WD=family,SWAPPED_WD=swapped),proposals=[],
        baseline=dict(actual_pose=dict(centroid=[99.,99.,99.]),accepted=True,WD_family='SWAPPED_WD'),
        outline=dict(supported=False,configuration={}))
    before=deepcopy(task)
    monkeypatch.setattr(R,'_problem',lambda *a:(Rejected(),{}))
    monkeypatch.setattr(R.OLD,'_prior_inputs',lambda *a:([],dict(epsilon=.01,huber_scale=3.),None))
    monkeypatch.setattr(R.BOUNDED.S,'rebind_bounded_visibility_trial_geometry',lambda *a:None)
    monkeypatch.setattr(R.PARENT.V.V,'_REFINER',lambda *a,**kw:dict(solutions=[],candidates=[],search={},
        start_selection=dict(selected_start_names=['OWN_INITIAL_COARSE'])))
    row=R._run(task,{})
    assert set(row['arms'])==set(R.ARMS) and row['arms'][R.ARMS[0]]==task['baseline']
    for arm in R.ARMS[1:]:
        result=row['arms'][arm]
        assert not result['accepted'] and result['fallback_audit']['own_original_initial_coarse_only']
        assert np.array_equal(result['actual_pose']['centroid'],translation)
        assert not result['fallback_audit']['e_numeric_pose_used']
        assert result['keypoint_readout']['reported_points'][next(k for k,s in enumerate(base.states)if s=='S')]is not None
        assert not result['keypoint_readout']['OT_solved']
    assert task==before


def test_score_rejects_incomplete_seal_before_numeric_reference(monkeypatch,tmp_path):
    def read(path):
        if Path(path).name=='PREDICTION_SEAL.json':return dict(rows=237)
        if Path(path).name=='PROTOCOL.json':return {}
        raise AssertionError('No reference file read before full seal')
    monkeypatch.setattr(R.C,'read',read)
    with pytest.raises(ValueError,match='Full declared'):
        R.score(tmp_path)


def test_frozen_source_union_cannot_hide_parent_dependency_change():
    assert R._merge_frozen({'a':'1'},{'a':'1','b':'2'})=={'a':'1','b':'2'}
    with pytest.raises(ValueError,match='ancestry conflict'):
        R._merge_frozen({'a':'1'},{'a':'changed'})
