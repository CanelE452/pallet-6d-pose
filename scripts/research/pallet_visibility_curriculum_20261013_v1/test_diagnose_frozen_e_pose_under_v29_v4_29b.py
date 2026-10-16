"""Pure V29 diagnosis semantics. No actual models, rows, labels or images."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import diagnose_frozen_e_pose_under_v29_v4_29b as D


def scene():
    xyz = D.C.cuboid(.8,.4,.6)
    physical = Rotation.from_rotvec([.2,-.4,.1]).as_matrix()
    t = np.asarray([.1,-.02,3.])
    family = dict(name='ORIGINAL_WD',native_xyz=xyz.tolist(),physical_registration_Q=np.eye(3).tolist())
    pose = dict(R_cf=physical.tolist(),R_physical=physical.tolist(),centroid=t.tolist(),native_xyz=xyz.tolist())
    return xyz,physical,t,family,pose


def test_profiles_exact_four_arms_and_no_pose_prior():
    profiles = D.profiles()
    assert tuple(profiles)==D.ARMS
    assert [v['containment'] for v in profiles.values()]==[False,True,False,True]
    assert [v['source_policy'] for v in profiles.values()]==['ALL_BI_REPLACE']*2+['RGB_NONFALLBACK_PRESERVE']*2
    assert all(v['no_full_TP_rotation_pose_prior'] and v['both_WD_families_allowed'] for v in profiles.values())


def test_transport_preserves_unordered_body_but_exposes_changed_channel_identity():
    xyz,R,t,first,pose = scene()
    Q = Rotation.from_euler('y',90,degrees=True).as_matrix()
    permutation = [1,0,3,2,5,4,7,6]
    other = dict(name='SWAPPED_WD',native_xyz=(xyz[permutation]@Q.T).tolist(),physical_registration_Q=Q.tolist())
    saved = deepcopy(pose)
    rotated,translated,audit = D.transport_diagnostic_pose(pose,first,other)
    assert np.allclose(rotated@Q,R,atol=1e-14,rtol=0)
    assert np.array_equal(translated,t)
    assert audit['available'] and audit['not_the_original_saved_e_channel_correspondence']
    assert audit['unordered_physical_body_gap_m']<1e-12
    # The unordered physical body is preserved; corner k has a different named
    # correspondence in this trial. It cannot be presented as original e KPs.
    projected_original=xyz@R.T+t
    projected_other=np.asarray(other['native_xyz'])@rotated.T+t
    assert not np.allclose(projected_original,projected_other)
    assert np.allclose(projected_original[permutation],projected_other)
    assert pose==saved


def test_nonmatching_body_transport_abstains_and_invalid_saved_basis_fails():
    xyz,R,t,first,pose = scene()
    other={**first,'name':'OTHER','native_xyz':(xyz*1.1).tolist()}
    moved,translation,audit=D.transport_diagnostic_pose(pose,first,other)
    assert moved is None and translation is None and not audit['available']
    with pytest.raises(ValueError,match='correspondence'):
        D.validate_saved_pose_basis({**pose,'R_cf':np.eye(3).tolist()},first)
    with pytest.raises(ValueError,match='Centered'):
        D.validate_saved_pose_basis(pose,{**first,'native_xyz':(xyz+1.).tolist()})


class DummyProblem:
    scale=1.
    def __init__(self,accepted=True,containment=True):
        self.accepted=accepted;self.containment=containment
    def objective(self,p):
        shape=5. if self.containment else 0.
        weight=.7 if self.containment else 0.
        parts=dict(native_costs=[dict(kind='COMPLETE_BI',native_corner=0,weighted_robust_cost=1.),
            dict(kind='UNREPLACED_PARTIAL_NORMAL',native_corner=1,weighted_robust_cost=2.),
            dict(kind='PROJECTED_NATIVE_EDGE',native_corner=2,native_neighbor=3,weighted_robust_cost=3.),
            dict(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',native_corner=2,weighted_robust_cost=4.)],
            native_observation_objective=10.,regularizer_costs=[dict(kind=k,weighted_cost=v)
                for k,v in zip(D.M.BOUNDED.REGULARIZER_KINDS,(.2,.3))],
            global_containment_cost=shape,global_containment_weight=weight)
        return 10.5+shape*weight,np.zeros(6),parts
    def evaluate(self,R,t):
        total=self.objective(None)[0]
        return dict(accepted=self.accepted,objective=total,minimum_joint_slack=1. if self.accepted else -1.,
            actual_edge_support_constraint_audit=[dict(passed=self.accepted)])


def test_native_O_S_C_decomposition_and_no_native_pollution():
    _,R,t,_,_=scene()
    result=D.evaluate_parts(DummyProblem(),R,t)
    assert result['total']==14. and result['native']==10.
    assert [result[k] for k in ('O','S','P','C','C_weight','weighted_C')]==[.2,.3,0.,5.,.7,3.5]
    assert sum(result['native_by_kind'].values())==10.
    assert result['native_factor_counts']['PROJECTED_NATIVE_EDGE']==1
    assert result['maximum_internal_ledger_difference']==0.
    native=D.evaluate_parts(DummyProblem(containment=False),R,t)
    assert native['total']==10.5 and native['C']==native['C_weight']==0.


def test_hard_guard_failure_keeps_finite_cost_without_claiming_search_gap():
    _,R,t,_,_=scene()
    old=D.evaluate_parts(DummyProblem(accepted=False),R,t)
    selected=D.evaluate_parts(DummyProblem(),R,t)
    assert old['finite_objective'] and not old['accepted_current_guards']
    assert 'hard_or_numeric_guards' in D.comparison(old,selected,selected_accepted=True)
    assert 'own_fallback' in D.comparison(old,selected,selected_accepted=False)


def test_cost_preference_search_gap_ties_and_numeric_abstention_are_separate():
    good=dict(accepted_current_guards=True,finite_objective=True,total=1.)
    assert 'search_coverage_gap' in D.comparison(good,{**good,'total':2.},selected_accepted=True)
    assert 'objective_prefers_selected' in D.comparison({**good,'total':3.},good,selected_accepted=True)
    assert 'same_current_energy' in D.comparison(good,{**good,'total':1.+1e-10},selected_accepted=True)
    assert 'numeric_evaluation' in D.comparison(dict(finite_objective=False),good,selected_accepted=True)


def test_ledger_mismatch_is_reported_as_unavailable_not_repaired():
    class Bad(DummyProblem):
        def evaluate(self,R,t):return dict(accepted=True,objective=200.)
    _,R,t,_,_=scene()
    result=D.evaluate_parts(Bad(),R,t)
    assert not result['finite_objective'] and 'ledger mismatch' in result['reason']


def test_source_gate_precedes_S_U_payload_and_fallback_sources_remain_distinct():
    class Poison(dict):
        def __getitem__(self,key):
            if key=='native_corner':return super().__getitem__(key)
            raise AssertionError('Forbidden S/U payload read')
    for state in ('S','U'):
        task=dict(boundary_merged=[],initial_self_hidden=[state=='S']+[False]*7,
            states=[state]+['B']*7,spec=dict(visible=[Poison(native_corner=0)]))
        with pytest.raises(ValueError,match='NonS BI'):
            D.source_inventory(task)
    task=dict(boundary_merged=[dict(native_corner=0)],initial_self_hidden=[False]*8,
        states=['B','I']+['U']*6,spec=dict(visible=[dict(native_corner=0,source='RGB_COMPLETE'),
            dict(native_corner=1,source='VISIBLE_BACKBONE_FALLBACK')]))
    before=deepcopy(task);inventory=D.source_inventory(task)
    assert len(inventory['archived_BI_sources'])==2
    assert inventory['original_partial_BI_sources']==[dict(state='B',source='RGB_COMPLETE',count=1)]
    assert task==before


def test_TEST_prefix_decodes_only49_and_does_not_parse_CAL_or_remainder(tmp_path):
    path=tmp_path/'rows.jsonl.gz';ids=['TEST:'+str(i) for i in range(49)]
    with gzip.open(path,'wt') as stream:
        for _ in range(46):stream.write('CAL not JSON\n')
        for ident in ids:stream.write(json.dumps(dict(id=ident,partition='TEST'))+'\n')
        stream.write('REMAINDER not requested\n')
    assert [r['id'] for r in D._TEST_prefix(path,ids)]==ids


def test_no_actual_replay_without_separate_root_GO(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No actual file/model/score decode before GO')
    monkeypatch.setattr(D.C,'read',forbidden)
    monkeypatch.setattr(D.C,'sha',forbidden)
    monkeypatch.setattr(D.M,'_models',forbidden)
    with pytest.raises(ValueError,match='Separate root postscore replay GO'):
        D.run()
