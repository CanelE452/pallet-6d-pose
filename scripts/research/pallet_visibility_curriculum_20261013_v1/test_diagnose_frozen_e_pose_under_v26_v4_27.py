"""Synthetic diagnosis semantics only; no real predictions/scores/references."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import diagnose_frozen_e_pose_under_v26_v4_27 as D


def test_native_WD_identity_is_primary_while_physical_transport_is_explicit():
    xyz=D.C.cuboid(.8,.4,.6)
    Q=Rotation.from_euler('y',90,degrees=True).as_matrix()
    family=dict(native_xyz=(xyz@Q.T).tolist(),physical_registration_Q=Q.tolist())
    physical=Rotation.from_rotvec([.2,-.5,.1]).as_matrix();t=np.array([.1,-.03,3.])
    pose=dict(R_physical=physical.tolist(),R_cf=(physical@Q.T).tolist(),centroid=t.tolist())
    D._validate_saved_e_basis(pose,family)
    R,tt,audit=D.transport_physical_pose_to_family(pose,family)
    assert np.array_equal(tt,t) and np.allclose(R,pose['R_cf'],atol=1e-15,rtol=0)
    assert audit['physical_rotation_max_absolute_difference']<1e-12
    with pytest.raises(ValueError,match='saved WD family'):
        D._validate_saved_e_basis({**pose,'R_cf':np.eye(3).tolist()},family)


def test_six_profile_mapping_and_filter_vs_AUTH_guard_search_cost_distinctions():
    profiles=D._profiles()
    assert list(profiles)==list(D.V.ARMS[1:])
    assert [p['locked']for p in profiles.values()]==[False,True,False,True,False,True]
    assert [p['containment']for p in profiles.values()]==[False,False,False,False,True,True]
    assert [p['mode']for p in profiles.values()]==['FULL','FULL','TRANSLATION','TRANSLATION','TRANSLATION','TRANSLATION']
    kw=dict(family_allowed=True,e_guard_accepted=True,e_AUTH_eligible=True,selected_accepted=True,e_total=1.,selected_total=2.)
    assert 'search_coverage_gap' in D.classify_comparison(**kw)
    assert 'objective_prefers' in D.classify_comparison(**{**kw,'e_total':3.})
    assert 'native_family_excluded' in D.classify_comparison(**{**kw,'family_allowed':False})
    assert 'hard_or_numeric_guard' in D.classify_comparison(**{**kw,'e_guard_accepted':False})
    assert 'native_AUTH' in D.classify_comparison(**{**kw,'e_AUTH_eligible':False})
    assert 'own_fallback' in D.classify_comparison(**{**kw,'selected_accepted':False,'selected_total':None})
    assert 'same_current_energy' in D.classify_comparison(**{**kw,'e_total':2.+1e-10})


def test_separate_native_O_S_P_containment_ledgers_and_noP_zero(monkeypatch):
    class Problem:
        xyz=D.C.cuboid(.8,.4,.6)
        K=np.eye(3);Q=np.eye(3);scale=1.
        def evaluate(self,R,t):return dict(accepted=True,minimum_joint_slack=.5)
        def objective(self,p):
            return 3.,np.zeros(6),dict(native_costs=[dict(kind='COMPLETE_BI',native_corner=0,weighted_robust_cost=3.)])
    monkeypatch.setattr(D.A.A.O,'bounded_o_localization_value_and_gradient',lambda *a,**kw:(1.,np.zeros(6),{'O_point_terms':0}))
    monkeypatch.setattr(D.A.A.S,'rebind_bounded_visibility_trial_geometry',lambda *a:{})
    monkeypatch.setattr(D.A.A.S,'coarse_visibility_prior_value_and_gradient',lambda *a:(2.,np.zeros(6),{'S_image_or_2D_terms':0}))
    monkeypatch.setattr(D.A,'direct_old_P1_prior',lambda p,xyz,s,original,Q:(4.*original['factor_weight'],np.zeros(6),{'factor_weight':original['factor_weight']}))
    monkeypatch.setattr(D.A.A.MASK_CORE,'_outline_config',lambda cfg:dict(outline_factor_weight=.7))
    calls=[]
    def containment(*args):calls.append(1);return 5.,np.zeros(6),{'valid_projection':True}
    monkeypatch.setattr(D.A.CONTAINMENT,'outline_value_and_gradient',containment)
    task=dict(outline=dict(supported=True,configuration={}))
    result=D._evaluate(Problem(),task,np.eye(3),np.array([0.,0.,3.]),[],{}, {},dict(factor_weight=1.),containment=True)
    assert result['total']==pytest.approx(13.5) and result['weighted_containment']==3.5
    assert result['native_by_kind']=={'COMPLETE_BI':3.} and len(calls)==1
    assert [r['weighted_cost']for r in result['regularizer_costs']]==[1.,2.,4.]
    noP=D._evaluate(Problem(),task,np.eye(3),np.array([0.,0.,3.]),[],{}, {},dict(factor_weight=0.),containment=False)
    assert noP['total']==6. and noP['P_prior']==0. and noP['containment']==0. and len(calls)==1


def test_family_excluded_diagnostic_never_enters_AUTH_or_runtime_bank(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Excluded family entered AUTH/runtime')
    monkeypatch.setattr(D.V.P,'select_candidates',forbidden)
    monkeypatch.setattr(D.V,'_run',forbidden)
    task=dict(states=['B']*8)
    evaluation=dict(accepted_current_hard_and_numeric_guards=True)
    checked=D._authenticate_diagnostic(task,evaluation,family='SWAPPED_WD',requested_family='ORIGINAL_WD',containment=True)
    assert checked==dict(family_allowed=False,AUTH_eligible=False,diagnostic_only=True)


def test_original_archived_source_gate_precedes_S_or_U_payload():
    class Poison(dict):
        def __getitem__(self,key):
            if key=='native_corner':return super().__getitem__(key)
            raise AssertionError('Forbidden S/U source/point read')
    for state in ('S','U'):
        task=dict(boundary_merged=[],initial_self_hidden=[state=='S']+[False]*7,
            states=[state]+['B']*7,spec=dict(visible=[Poison(native_corner=0)]))
        with pytest.raises(ValueError,match='initial-nonS BI'):
            D._source_inventory(task)


def test_source_inventory_distinguishes_restored_RGB_from_exact_fallback():
    task=dict(boundary_merged=[dict(native_corner=0),dict(native_corner=1)],initial_self_hidden=[False]*8,
        states=['B','I']+['U']*6,spec=dict(visible=[dict(native_corner=0,source='B_TARGET_BACKGROUND_LINES'),
            dict(native_corner=1,source='VISIBLE_BACKBONE_FALLBACK')]))
    before=deepcopy(task);inventory=D._source_inventory(task)
    assert inventory['restored_prepartial_ids']==[0] and task==before
    assert inventory['restored_prepartial_sources']==[dict(state='B',source='B_TARGET_BACKGROUND_LINES',count=1)]
    assert len(inventory['original_partial_BI_sources'])==2 and inventory['S_U_coordinate_reads']==0


def test_only_TEST49_JSON_decoded_from_bounded95_prefix(tmp_path):
    path=tmp_path/'synthetic.jsonl.gz';ids=['TEST:'+str(i)for i in range(49)]
    with gzip.open(path,'wt',encoding='utf-8')as f:
        for _ in range(46):f.write('CAL deliberately not JSON decoded\n')
        for ident in ids:f.write(json.dumps(dict(id=ident,partition='TEST'))+'\n')
        f.write('REMAINDER deliberately not requested or decoded\n')
    assert [r['id']for r in D._TEST_prefix(path,ids)]==ids


def test_missing_postseal_GO_rejects_before_reading_any_current_journal_or_score(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Current input/prediction/score accessed')
    monkeypatch.setattr(D.C,'read',forbidden)
    monkeypatch.setattr(D.C,'sha',forbidden)
    monkeypatch.setattr(D,'_verify_parent',forbidden)
    with pytest.raises(ValueError,match='separate root full-v26'):
        D.run()
