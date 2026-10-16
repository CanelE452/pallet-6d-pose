"""Pure independent raw-query audit fixtures, no actual rows/models/fit."""
from copy import deepcopy

import pytest

from . import audit_geometry_guided_edges_v4_31 as A


def fixture(mate='I',kind='OBSERVED_BI_REFERENCE_BAND',guide='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR'):
    states=['B',mate]+['S']*6
    reference=dict(native_corner=0,native_neighbor=1,actual_search_anchor_xy=[0.,0.],
        actual_search_direction_xy=[1.,0.],guide_kind=guide)
    if kind is not None:reference['search_anchor_kind']=kind
    reference['guide_reference_id']=A.PARENT._digest(reference)
    chain=dict(guide_reference=reference,guide_reference_id=reference['guide_reference_id'],
        generating_guide_direction_xy=[1.,0.])
    ref=dict(ray_index=0,chain_index=0,line_name='actual',native_corner=0,native_neighbor=1,
        guide_reference=deepcopy(reference),guide_reference_id=reference['guide_reference_id'])
    cfg=dict(normal_search_radius_px=43.)
    observation=dict(selection_profile=A.PROFILE,observation_generation=A.RAW_GENERATION,
        association_configuration=cfg,chain_observation=dict(configuration=deepcopy(cfg),
            edge_records=[dict(native_corner=0,native_neighbor=1,chains=[chain])]),
        selected_chain_refs=[ref],joint_graph=dict(line_candidates=[dict(name='actual',
            native_edge=[0,1],normal=[0.,1.],offset=-19.,provenance=['0:0'])]))
    task=dict(v29_observation=observation,states=states,initial_self_hidden=[s=='S'for s in states],
        spec=dict(visible=[dict(native_corner=0,xy=[0.,0.])]+([dict(native_corner=1,xy=[30.,0.])]if mate=='I'else[])),
        v29_initial_geometry=dict(initial_projection_xy=[[100.,100.],[140.,100.]]+[None]*6))
    plan=dict(selection_profile=A.PROFILE,observation_family=A.FAMILY,population=A.POPULATION,
        raw_observation_generation=A.RAW_GENERATION)
    prediction=dict(arms={arm:dict(accepted=False,solution=dict(v29_cost_plan=deepcopy(plan)))for arm in A.ARM_REPLACEMENTS.values()})
    return task,prediction


def run(task,row):
    checks=[]
    A._check_association_frame(task,row,lambda passed,reason,ident,arm=None:checks.append((passed,reason)),'synthetic')
    return checks


def refresh_reference(task):
    obs=task['v29_observation'];chain=obs['chain_observation']['edge_records'][0]['chains'][0]
    reference=chain['guide_reference'];reference.pop('guide_reference_id',None)
    reference['guide_reference_id']=A.PARENT._digest(reference)
    chain['guide_reference_id']=reference['guide_reference_id']
    obs['selected_chain_refs'][0].update(guide_reference=deepcopy(reference),guide_reference_id=reference['guide_reference_id'])


def test_only_exact_AST_versions_import_and_one_hook_parent_kernel_globals_unchanged():
    before=dict(A.PARENT.main.__globals__)
    function,transform=A._build_main()
    assert set(transform.replacements.values())=={1}
    assert transform.imports==transform.frame_hooks==1
    for name in('recompute_native','prepare_raw','observation_checks','cost_buckets'):
        assert function.__globals__[name]is getattr(A.PARENT,name)
    assert function.__globals__['_check_association_frame']is A._check_association_frame
    assert all(A.PARENT.main.__globals__[key]is value for key,value in before.items())
    assert not set(A.REPLACEMENTS)&{v for v in function.__code__.co_consts if isinstance(v,str)}


@pytest.mark.parametrize('mate',['I','O','T','U'])
def test_q_anchor_normal19_and_fixed_signed_ray_pass_without_unknown_observed_query(mate):
    task,row=fixture(mate)
    assert all(passed for passed,_ in run(task,row))


@pytest.mark.parametrize('kind,guide',[(None,'FROZEN17C_ORIGINAL_MODEL_SEED'),
    ('OBSERVED_BI_REFERENCE_BAND','MEASURED_BI_ENDPOINT_DIRECTION')])
def test_original_native_and_measured_BI_queries_remain_admissible(kind,guide):
    task,row=fixture(kind=kind,guide=guide)
    assert all(passed for passed,_ in run(task,row))


@pytest.mark.parametrize('change',['model_anchor','anchor_shift','opposite_ray','orthogonal','short_BI','normal_far','bad_alias'])
def test_independent_raw_query_rejections_even_when_all_pose_arms_fallback(change):
    task,row=fixture();obs=task['v29_observation'];chain=obs['chain_observation']['edge_records'][0]['chains'][0]
    ref=chain['guide_reference']
    if change=='model_anchor':ref['search_anchor_kind']='MODEL_PROJECTION_SEARCH_ONLY_BAND'
    elif change=='anchor_shift':ref['actual_search_anchor_xy'][0]=1e-5
    elif change in('opposite_ray','orthogonal'):
        direction=[-1.,0.]if change=='opposite_ray'else[0.,1.]
        ref['actual_search_direction_xy']=direction;chain['generating_guide_direction_xy']=direction[:]
    elif change=='short_BI':task['spec']['visible'][1]['xy']=[9.,0.]
    elif change=='normal_far':obs['joint_graph']['line_candidates'][0]['offset']=-100.
    else:chain['generating_guide_direction_xy']=[-1.,0.]
    refresh_reference(task)
    assert any(not passed for passed,_ in run(task,row))


def test_TLS_normal_offset_sign_reversal_does_not_change_admission():
    task,row=fixture();before=run(task,row)
    node=task['v29_observation']['joint_graph']['line_candidates'][0]
    node['normal']=[0.,-1.];node['offset']=19.
    assert before==run(task,row)


def test_both_BI_missing_source_measurement_abstains_without_model_ray_rescue():
    task,row=fixture();task['spec']['visible']=task['spec']['visible'][:1]
    with pytest.raises(ValueError,match='model mate cannot rescue'):
        run(task,row)


def test_initial_S_gate_before_poison_chain_node_and_projection_payload():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('S payload read')
    task,row=fixture('S');task['v29_observation']['joint_graph']['line_candidates']=[]
    task['v29_observation']['chain_observation']['edge_records'][0]['chains']=[Poison()]
    task['v29_initial_geometry']=Poison()
    with pytest.raises(ValueError,match='S/BI gate'):
        run(task,row)


def test_U_observed_payload_and_other_S_model_projection_slots_are_never_touched():
    class Unknown(dict):
        def __getitem__(self,key):
            if key!='native_corner':raise AssertionError('Unknown observed payload read')
            return super().__getitem__(key)
    class Projection:
        def __getitem__(self,key):
            if key==0:return[100.,100.]
            if key==1:return[140.,100.]
            raise AssertionError('Other S model projection read')
    task,row=fixture('U');task['spec']['visible'].append(Unknown(native_corner=1))
    task['v29_initial_geometry']['initial_projection_xy']=Projection()
    assert all(passed for passed,_ in run(task,row))


def test_unknown_signed_reference_uses_ORIGINAL_model_ray_not_generating_direction():
    task,row=fixture('U');task['v29_initial_geometry']['initial_projection_xy'][1]=[60.,100.]
    assert any(not passed and reason=='actual query positive fixed native hemisphere'for passed,reason in run(task,row))


def test_raw_reference_dictionary_digest_and_indices_are_not_fabricated():
    task,row=fixture();task['v29_observation']['selected_chain_refs'][0]['guide_reference']['actual_search_anchor_xy']=[1.,0.]
    assert any(not passed and 'exact indexed raw guide' in reason for passed,reason in run(task,row))
    task,row=fixture();task['v29_observation']['selected_chain_refs'][0]['ray_index']=True
    with pytest.raises(ValueError,match='integer index'):
        run(task,row)


def test_new_population_checked_for_every_fallback_arm_too():
    task,row=fixture();arm=next(iter(row['arms']))
    row['arms'][arm]['solution']['v29_cost_plan']['population']='OLD'
    assert any(not passed and 'including fallback'in reason for passed,reason in run(task,row))
