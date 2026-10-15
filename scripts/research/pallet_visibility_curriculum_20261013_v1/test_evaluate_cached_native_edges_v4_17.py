"""Focused no-inference contracts for the uniform cached-candidate caller."""
from copy import deepcopy
import numpy as np
import pytest

from . import evaluate_cached_native_edges_v4_17 as V


class Poison:
    def __deepcopy__(self,memo):raise AssertionError('Target field queried')
    def __float__(self):raise AssertionError('Target field queried')


def test_proposals_ignore_old_status_cost_and_target_fields():
    bank={};solution=dict(R=np.eye(3).tolist(),t=[0,0,2],accepted=False,
        objective=Poison(),metric=Poison(),ground_truth=Poison(),
        feasible_pose_samples=[dict(R=np.eye(3).tolist(),t=[0,0,3],accepted=False)])
    result=V.collect_solution_poses(bank,solution,np.eye(3),'source')
    assert result==dict(saved_records=2,invalid_records=0)
    assert sorted(p['t'][2]for p in bank.values())==[2,3]


def test_registration_transport_deduplicates_physical_pose_not_native_R():
    bank={};Q=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
    assert V._add_pose(bank,np.eye(3),[0,0,2],np.eye(3),'original')
    assert V._add_pose(bank,Q.T,[0,0,2],Q,'swap')
    assert len(bank)==1
    assert next(iter(bank.values()))['origins']==['original','swap']


def test_fallback_and_seed_dictionaries_are_not_mined_as_saved_candidates():
    bank={};solution=dict(R=np.eye(3).tolist(),t=[0,0,2],
        fallback_pose=dict(R=np.eye(3).tolist(),t=[0,0,7]),
        BI_prior_seed=dict(R=np.eye(3).tolist(),t=[0,0,8]),
        candidates=[dict(R=np.full((3,3),np.nan),t=[0,0,3])])
    result=V.collect_solution_poses(bank,solution,np.eye(3),'source')
    assert len(bank)==1 and result['invalid_records']==1
    assert next(iter(bank.values()))['t']==[0,0,2]


def test_exact_cost_bridge_keeps_reliability_weighted_energies_unchanged():
    records=[dict(kind=key,native_corner=0,weighted_robust_cost=.375,provenance='fixed')for key in V.KIND_MAP]
    before=deepcopy(records);out=V._bridge_costs(records)
    assert records==before
    assert [r['kind']for r in out]==list(V.KIND_MAP.values())
    assert all(r['weighted_robust_cost']==.375 and r['provenance']=='fixed'for r in out)
    with pytest.raises(KeyError):V._bridge_costs([dict(kind='unregistered_cost')])


def test_fresh_joint_rejection_precedes_energy_and_mask_ranking():
    rows=[dict(name='old_accepted_but_current_invalid',family='ORIGINAL_WD',accepted=False,
        all_joint_constraints_satisfied=False,native_costs=Poison(),global_mask_cost=Poison()),
        dict(name='current_feasible',family='SWAPPED_WD',accepted=True,all_joint_constraints_satisfied=True,
            native_costs=[dict(kind='BI_COMPLETE',native_corner=0,weighted_robust_cost=2.)],global_mask_cost=9.)]
    selected=V.AUTH.select_hierarchical_native_candidates(['I']+['S']*7,rows)
    assert selected['selected_name']=='current_feasible'


def test_readout_preserves_S_projection_only_U_none_and_partial_not_point():
    xyz=np.array([[-.1,-.1,-.1],[.1,-.1,-.1],[.1,-.1,.1],[-.1,-.1,.1],
        [-.1,.1,-.1],[.1,.1,-.1],[.1,.1,.1],[-.1,.1,.1]])
    task=dict(states=['B','I','U','O','T','S','S','S'],spec=dict(K=[[100,0,50],[0,100,50],[0,0,1]],
        visible=[dict(native_id=0,xy=[40,40],source='archived'),dict(native_id=1,xy=[60,40],source='archived')]))
    selected=dict(family=dict(native_xyz=xyz.tolist()),accepted=False)
    result=V._readout(task,[dict(native_corner=0)],selected,dict(R_cf=np.eye(3).tolist(),centroid=[0,0,2]))
    assert result['reported_points'][:5]==[None,[60,40],None,None,None]
    assert all(result['reported_points'][k]is not None for k in (5,6,7))
    assert result['S_projection_only']and result['U_point_or_line_measurements']==0
    assert not result['OT_solved']
