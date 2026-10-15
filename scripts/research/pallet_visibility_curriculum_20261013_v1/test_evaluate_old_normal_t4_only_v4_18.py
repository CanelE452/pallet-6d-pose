"""Matched OFF control: exact old objective, no NMS/model field access."""
from copy import deepcopy

import numpy as np
import pytest

from . import evaluate_old_normal_t4_only_v4_18 as OFF
from .test_evaluate_refined_native_edges_v4_17 import task_fixture


def _task():
    task,_ = task_fixture(True)
    task['new_chain_observation'] = {'schema':'shared_input_only_not_used'}
    task['outline']['configuration'] = {}
    return task


def _base_problem(task,family):
    spec=task['spec'];config={**spec['config'],'prior_weight':0.}
    return OFF.ON.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],
        spec['states'],spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],
        config=config,boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],
        edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))


def test_OFF_problem_is_identical_old_normal_T4_object_and_analytic_objective():
    task=_task();family=task['families']['ORIGINAL_WD'];base=_base_problem(task,family)
    bundle=OFF._no_image_chain_groups(base,object(),initial_self_hidden=task['initial_self_hidden'],
        image_binding=object())
    result=OFF._unchanged_old_problem(base,bundle)
    assert result is base and result.remaining_normals==base.boundary
    p=OFF.ON.BASE.NATIVE.V._pose_vector(np.asarray(family['coarse_pose']['R']),
        np.asarray(family['coarse_pose']['t']),base.scale)
    a=base.objective(p);b=result.objective(p)
    assert a[0]==b[0]and np.array_equal(a[1],b[1])and a[2]==b[2]
    assert all(c['kind']!='PROJECTED_NATIVE_EDGE'for c in b[2]['native_costs'])


def test_NMS_observation_and_model_are_never_inspected_by_OFF_factory():
    class Poison:
        def __getattribute__(self,key):raise AssertionError('NMS/model field read: '+key)
    task=_task();base=_base_problem(task,task['families']['ORIGINAL_WD'])
    assert OFF._no_noise_parameters(Poison())=={}
    bundle=OFF._no_image_chain_groups(base,Poison(),initial_self_hidden=task['initial_self_hidden'],
        image_binding=Poison(),parameters=Poison())
    assert bundle['groups']==[]and bundle['replaces_normal_factor_refs']==[]
    assert bundle['NMS_observation_fields_read']==0


def test_worker_exact_old_objective_minimum_all_proposals_and_no_global_mutation():
    task=_task();before=deepcopy(task);original_CAL=OFF.ON.CAL;original_MIX=OFF.ON.MIX
    energies=[]
    for family in task['families'].values():
        problem=_base_problem(task,family);Q=np.asarray(family['physical_registration_Q'])
        for proposal in task['proposals']:
            checked=problem.evaluate(np.asarray(proposal['R_physical'])@Q.T,proposal['t'])
            if checked['accepted']:energies.append(checked['objective'])
    row=OFF._run(task,object())
    assert task==before and OFF.ON.CAL is original_CAL and OFF.ON.MIX is original_MIX
    assert row['arms']['SAME_FROZEN_E']==task['baseline']
    assert row['arms'][OFF.ARMS[1]]['solution']['objective']==pytest.approx(min(energies),abs=1e-12)
    for arm in OFF.ARMS[1:]:
        value=row['arms'][arm]
        assert value['old_normal_T4_only_audit']['NMS_observations_used_in_objective']is False
        assert value['old_normal_T4_only_audit']['NMS_noise_model_used_in_objective']is False
        assert not value['keypoint_readout']['OT_solved']
        assert row['old_normal_T4_only_audit'][arm]['evaluated_candidates']==len(task['families'])*len(task['proposals'])
    assert set(row['fixed_family'])==set(OFF.ARMS)
    assert set(row['tangent_control_audit'])==set(OFF.ARMS[1:])


def test_all_rejected_keep_frame_and_only_own_coarse_fallback():
    task=_task()
    for proposal in task['proposals']:proposal['t']=[0.,0.,-4.]
    row=OFF._run(task)
    coarse=task['families']['ORIGINAL_WD']['coarse_pose']
    for arm in OFF.ARMS[1:]:
        value=row['arms'][arm]
        assert not value['accepted']
        assert value['actual_pose']['R_cf']==coarse['R']and value['actual_pose']['centroid']==coarse['t']
        assert not value['fallback_audit']['old_best_pose_fallback']
    assert row['contract']['NMS_groups_disabled']is True
