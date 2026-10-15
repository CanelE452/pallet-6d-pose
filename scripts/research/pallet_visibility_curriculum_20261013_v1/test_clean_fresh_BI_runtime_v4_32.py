"""Clean-boundary and real pure-refiner wiring on synthetic geometry only."""
from copy import deepcopy
import inspect

import cv2
import numpy as np
import pytest

from . import clean_fresh_BI_runtime_v4_32 as V
from .test_fresh_observed_BI_pnp_proposals_v4_32 import fixture,Poison


def binding():
    return dict(path='SYNTHETIC/independent_first_input_receipt.json',
        sha256=V.ADMISSION.INITIAL_SOURCE_CLOSURE_SHA256,verified_by_caller=True,
        independent_initial_geometry_and_BI_q=True)


def prepare(ids=(0,1,4,6),**changes):
    inputs,R,t=fixture(ids)
    inputs.update(changes)
    return V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding()),R,t


class Problem:
    """Finite quadratic SE3 objective, preserving the real refiner interface."""
    def __init__(self,clean,family_name='ORIGINAL_WD',target=None):
        family=clean['families'][family_name]
        self.xyz=np.asarray(family['native_xyz']);self.K=np.asarray(clean['spec']['K'])
        self.Q=np.asarray(family['physical_registration_Q']);self.states=clean['states']
        self.scale=float(np.linalg.norm(np.ptp(self.xyz,axis=0)))
        coarse=family['coarse_pose']
        self.target=V.REFINE.N.V._pose_vector(np.asarray(coarse['R']),np.asarray(coarse['t']),self.scale) if target is None else target
        self.cfg=dict(positive_depth_m=1e-6,max_joint_iterations=4,optimizer_ftol=1e-10)
        self.sides=[()];self.constraint_calls=0
    @property
    def reference_start_problem(self):
        raise AssertionError('Old reference-start problem must never be accessed')
    @property
    def proposals(self):
        raise AssertionError('Opaque old pose bank must never be accessed')
    def evaluate(self,R,t):
        # Match the real native evaluator's rotvec chart roundtrip.
        p=V.REFINE.N.V._pose_vector(np.asarray(R),np.asarray(t),self.scale)
        rr,tt=V.REFINE.N.V._pose(p,self.scale)
        value,_,parts=self.objective(p)
        return dict(accepted=True,R=rr.tolist(),t=tt.tolist(),objective=value,
            native_costs=parts['native_costs'],minimum_joint_slack=1.)
    def objective(self,p):
        d=np.asarray(p)-self.target
        return float(d@d),2*d,dict(native_costs=[dict(kind='SYNTHETIC_COMPLETE_BI',weighted_robust_cost=float(d@d))])
    def constraints(self,p,sides):
        self.constraint_calls+=1
        return np.ones(1),np.zeros((1,6))


def test_clean_task_positive_fields_and_both_family_physical_body_centers():
    clean,_,_=prepare()
    assert clean['ready'] and not clean['accepted'] and not clean['pose_available']
    assert set(clean['families'])=={'ORIGINAL_WD','SWAPPED_WD'}
    original=clean['families']['ORIGINAL_WD'];R0=np.asarray(original['coarse_pose']['R'])
    xyz0=np.asarray(original['native_xyz']);t0=np.asarray(original['coarse_pose']['t'])
    for name,family in clean['families'].items():
        R=np.asarray(family['coarse_pose']['R']);t=np.asarray(family['coarse_pose']['t']);xyz=np.asarray(family['native_xyz'])
        assert V.WD._unordered_gap(xyz@R.T+t,xyz0@R0.T+t0)<1e-9
        assert np.allclose(R@family['physical_registration_Q'],R0@original['physical_registration_Q'])
        assert np.allclose(R@xyz.mean(axis=0)+t,R0@xyz0.mean(axis=0)+t0)
        assert 'initializations' not in family and not family['initializations_imported_from_legacy_task']
        assert all(p['source'] in (V.FRESH.COARSE_SOURCE,V.FRESH.FRESH_SOURCE)
            for p in clean['fresh_seed_banks'][name]['proposals'])
    assert not {'baseline','proposals','pose_guides','stage1_family_seed_bank'} & set(clean)


def test_excluded_payloads_never_enter_clean_task_or_family_seed_generators():
    inputs,_,_=fixture((0,1,4))
    inputs['states']=['B','I','S','U','B','O','T','B'];inputs['initial_self_hidden'][7]=True
    inputs['visible_measurements'] += [Poison(native_corner=k) for k in (2,3,5,6,7)]
    clean=V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding())
    assert [r['native_corner'] for r in clean['spec']['visible']]==[0,1,4]
    assert all(r['admitted_BI_ids']==[0,1,4] for r in clean['fresh_seed_banks'].values())


@pytest.mark.parametrize('bad',['nan','nonrotation','behind','missing','wrong_source'])
def test_invalid_initial_coarse_is_retained_no_pose_abstention_not_old_fallback(bad):
    inputs,_,_=fixture((0,1,4))
    if bad=='nan':inputs['initial_coarse_pose']['t'][2]=float('nan')
    elif bad=='nonrotation':inputs['initial_coarse_pose']['R'][0][0]+=1.
    elif bad=='behind':inputs['initial_coarse_pose']['t'][2]=-5.
    elif bad=='missing':del inputs['initial_coarse_pose']['R']
    else:inputs['initial_coarse_pose']['source']='SAME_FROZEN_E'
    clean=V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding())
    assert not clean['ready'] and clean['row_retained'] and not clean['pose_available']
    assert clean['families']=={} and clean['fresh_seed_banks']=={}
    assert not clean['old_method_or_baseline_fallback_used']
    packet=V.select_clean_fresh_native_starts(object(),clean,'ORIGINAL_WD')
    result=V.refine_clean_fresh_native_problem(object(),clean,'ORIGINAL_WD',packet)
    assert result['abstained'] and result['row_retained'] and not result['solutions']


def test_false_source_closure_and_opaque_legacy_fields_fail_before_payload():
    inputs,_,_=fixture((0,1,4));proof=binding();proof['sha256']='a'*64
    inputs['visible_measurements']=[Poison(native_corner=0)]
    with pytest.raises(ValueError,match='closure'):
        V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=proof)
    inputs,_,_=fixture((0,1,4));inputs['initial_coarse_pose']['initializations']=Poison(native_corner=0)
    with pytest.raises(ValueError,match='fields only'):
        V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding())
    with pytest.raises(TypeError):
        V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding(),baseline=Poison())
    names=inspect.signature(V.prepare_clean_fresh_BI_task).parameters
    assert not {'legacy_task','baseline','proposals','pose_guides','initializations'} & set(names)


def test_output_identifiers_do_not_route_geometry_or_seed_generation():
    inputs,_,_=fixture((0,1,4))
    a=V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding(),identifiers=dict(id='a',partition='CAL'))
    b=V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding(),identifiers=dict(id='b',partition='TEST'))
    assert a['families']==b['families'] and a['fresh_seed_banks']==b['fresh_seed_banks']
    with pytest.raises(ValueError,match='metadata'):
        V.prepare_clean_fresh_BI_task(**inputs,initial_source_binding=binding(),identifiers=dict(error=1.))


def test_direct_fresh_native_start_rank_and_frozen_refiner_never_use_old_reference():
    clean,_,_=prepare();before=deepcopy(clean);problem=Problem(clean)
    packet=V.select_clean_fresh_native_starts(problem,clean,'ORIGINAL_WD')
    assert 1<=len(packet['starts'])<=7 and packet['native_ranked_start_budget']==6
    assert not packet['legacy_reference_start_problem_used']
    result=V.refine_clean_fresh_native_problem(problem,clean,'ORIGINAL_WD',packet)
    assert problem.constraint_calls>0 and result['search']['cached_or_baseline_pose_inputs']==0
    assert result['search']['same_clean_start_packet_sha256']==packet['start_packet_sha256']
    assert 'mask_objective' not in result['search'] and 'pose_prior' not in result['search']
    assert clean==before


def test_same_policy_native_and_containment_can_share_exact_fresh_start_packet():
    clean,_,_=prepare();native=Problem(clean);containment=Problem(clean)
    packet=V.select_clean_fresh_native_starts(native,clean,'ORIGINAL_WD')
    a=V.refine_clean_fresh_native_problem(native,clean,'ORIGINAL_WD',packet)
    b=V.refine_clean_fresh_native_problem(containment,clean,'ORIGINAL_WD',packet)
    assert a['start_selection']['selected_start_names']==b['start_selection']['selected_start_names']
    assert a['search']['same_clean_start_packet_sha256']==b['search']['same_clean_start_packet_sha256']


def test_modified_bank_start_packet_and_wrong_physics_fail_closed():
    clean,_,_=prepare();problem=Problem(clean);packet=V.select_clean_fresh_native_starts(problem,clean,'ORIGINAL_WD')
    bad=deepcopy(clean);bad['fresh_seed_banks']['ORIGINAL_WD']['proposals'][0]['t'][0]+=.1
    with pytest.raises(ValueError,match='input changed'):V.select_clean_fresh_native_starts(problem,bad,'ORIGINAL_WD')
    changed=deepcopy(packet);changed['starts'][0]['t'][0]+=.1
    with pytest.raises(ValueError,match='packet'):V.refine_clean_fresh_native_problem(problem,clean,'ORIGINAL_WD',changed)
    changed['start_packet_sha256']=V._hash(changed,'start_packet_sha256')
    with pytest.raises(ValueError,match='generated fresh seed'):
        V.refine_clean_fresh_native_problem(problem,clean,'ORIGINAL_WD',changed)
    problem.Q=np.eye(3)
    with pytest.raises(ValueError,match='physics'):V.select_clean_fresh_native_starts(problem,clean,'ORIGINAL_WD')
