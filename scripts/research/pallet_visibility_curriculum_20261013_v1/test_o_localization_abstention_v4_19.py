"""Focused domain-set, immutable-input and worker/scorer contracts, no fits."""
from copy import deepcopy

import numpy as np
import pytest

from . import o_localization_abstention_v4_19 as A
from . import evaluate_o_localization_abstention_v4_19 as V
from .test_evaluate_grazing_image_chain_mixtures_v4_18b import fixture
from .test_image_chain_mixture_problem_v4_18 import setup
from .test_evaluate_refined_native_edges_v4_17 import no_optimizer


def domain(k=0, extras=()):
    return dict(native_corner=k, native_id=k, state='O', shape='ellipse',
        canvas_hw=[480, 640], center_xy=[100., 120.], half_axes_px=[8., 8.], angle_deg=0.,
        halfplanes=[[-1., 0., 0.], [1., 0., 640.], [0., -1., 0.], [0., 1., 480.], *extras],
        localization_available=True, binding={'calibration_model_sha256': 'c'*64},
        source='current_RGB_raw_image_predictor_plus_actual_external_O_CAL_uncertainty')


def test_only_ellipse_is_abstained_canvas_and_extra_condition_stay_and_original_is_archived():
    d=domain(extras=[[1., 0., 300.]])
    states=['O','B','I','U','T','B','S','S']
    t=dict(native_corner=4, state='T', shape='halfplane', outside_canvas_side='unknown', halfplanes=[])
    saved=deepcopy([d,t]);out=A.abstain_external_o_localization([d,t],states,[480,640])
    assert [d,t]==saved and out['domains'][1]==t and out['changed_native_ids']==[0]
    assert out['records'][0]['original_domain']==d
    box=out['domains'][0]
    assert box['shape']=='box' and box['bounds_xyxy']==[0.,0.,640.,480.]
    assert box['halfplanes']==[[1.,0.,300.]]
    assert not any(k in box for k in('center_xy','half_axes_px','angle_deg'))
    assert box['state']=='O' and not box['localization_available']


def test_actual_set_enlargement_preserves_boundary_and_rejects_outside_canvas_or_extra_plane():
    out=A.abstain_external_o_localization([domain(extras=[[1.,0.,300.]])],['O']+['U']*7,[480,640])
    b=out['domains'][0]
    def contains(x,y):
        x0,y0,x1,y1=b['bounds_xyxy']
        return x0<=x<=x1 and y0<=y<=y1 and all(a*x+c*y<=z for a,c,z in b['halfplanes'])
    assert contains(100.,120.) and contains(108.,120.)
    assert contains(200.,200.) and contains(0.,0.) and contains(300.,480.)
    assert not contains(-.1,100.) and not contains(301.,100.) and not contains(100.,480.1)


def test_S_and_non_OT_gate_before_copy_or_coordinate_reads():
    class Poison:
        def __deepcopy__(self,memo): raise AssertionError('forbidden copy')
        def __array__(self,*args): raise AssertionError('forbidden coordinate read')
    for state in('S','U','B','I'):
        states=[state]+['U']*7
        with pytest.raises(ValueError,match='S or non-OT'):
            A.abstain_external_o_localization([dict(native_corner=0,state=state,center_xy=Poison())],states,[480,640])


def test_existing_broad_O_box_is_exact_and_adding_previously_absent_canvas_is_rejected():
    d=dict(native_corner=0,state='O',shape='box',bounds_xyxy=[0.,0.,640.,480.],halfplanes=[],localization_available=False)
    assert A.abstain_external_o_localization([d],['O']+['U']*7,[480,640])['domains']==[d]
    bad=domain();bad['halfplanes']=bad['halfplanes'][:-1]
    with pytest.raises(ValueError,match='unestablished canvas'):
        A.abstain_external_o_localization([bad],['O']+['U']*7,[480,640])


def task_and_parameters():
    task=V.PARENT.PARENT._join_task(*fixture())
    _,_,_,parameters=setup()
    parameters={k:v for k,v in parameters.items()if k not in('initial_self_hidden','image_binding')}
    h,w=task['spec']['image_hw']
    for d in task['spec']['domains']:
        if d['state']=='O'and d['shape']=='ellipse':
            d.update(canvas_hw=[h,w],localization_available=True,
                halfplanes=[[-1.,0.,0.],[1.,0.,float(w)],[0.,-1.,0.],[0.,1.,float(h)]])
    return task,parameters


def test_adapter_preserves_actual_chain_bank_geometry_states_and_full_measurement_archives():
    task,_=task_and_parameters();saved=deepcopy(task);changed=V.adapt_task(task)
    assert task==saved
    for key in('states','families','proposals','boundary_merged','new_chain_observation','u_regions','outline','baseline'):
        assert changed[key]==task[key]
    for key in task['spec']:
        if key!='domains':assert changed['spec'][key]==task['spec'][key]


def test_domain_abstention_adds_no_objective_or_gradient_term(monkeypatch):
    task,parameters=task_and_parameters()
    monkeypatch.setattr(V.CAL,'mixture_noise_parameters',lambda model:deepcopy(parameters))
    changed=V.adapt_task(task);family=task['families']['ORIGINAL_WD']
    old,_=V.PARENT._problem(task,family,{})
    new,_=V.PARENT._problem(changed,family,{})
    p=V.BASE.NATIVE.V._pose_vector(np.asarray(family['coarse_pose']['R']),np.asarray(family['coarse_pose']['t']),old.scale)
    ov,og,oa=old.objective(p);nv,ng,na=new.objective(p)
    assert ov==nv and np.array_equal(og,ng) and oa==na


def test_worker_keeps_all_candidates_and_supplies_score_metadata_without_optimizer(monkeypatch):
    task,parameters=task_and_parameters();task=V.adapt_task(task);saved=deepcopy(task)
    monkeypatch.setattr(V.CAL,'mixture_noise_parameters',lambda model:deepcopy(parameters))
    monkeypatch.setattr(V.REFINE,'refine_native_edge_pose',no_optimizer)
    row=V._run(task,{})
    assert task==saved and row['arms'][V.ARMS[0]]==task['baseline']
    assert set(row['fixed_family'])==set(V.ARMS)
    assert set(row['tangent_control_audit'])==set(V.ARMS[1:])
    for arm in V.ARMS[1:]:
        assert row['native_refinement_audit'][arm]['evaluated_candidates']==len(task['families'])*len(task['proposals'])
        assert not row['arms'][arm]['keypoint_readout']['original_O_ellipse_constraints_claimed_solved']
        for k,s in enumerate(task['states']):
            if s=='U':assert row['arms'][arm]['keypoint_readout']['reported_points'][k]is None
            if s=='S':assert row['arms'][arm]['keypoint_readout']['point_sources'][k]=='POSE_ONLY_MODEL_PROJECTION'


def test_no_ellipse_worker_pose_and_family_parity_against_parent(monkeypatch):
    task,parameters=task_and_parameters()
    task['states']=['U' if s=='O' else s for s in task['states']]
    task['spec']['states']=deepcopy(task['states'])
    task['grazing_adaptation']['original_states']=deepcopy(task['states'])
    task['grazing_adaptation']['geometry_gate']['states']=deepcopy(task['states'])
    task['spec']['domains']=[]
    monkeypatch.setattr(V.CAL,'mixture_noise_parameters',lambda model:deepcopy(parameters))
    monkeypatch.setattr(V.REFINE,'refine_native_edge_pose',no_optimizer)
    old=V.PARENT._run(task,{})
    new=V._run(V.adapt_task(task),{})
    for a,b in zip(V.PARENT.ARMS[1:],V.ARMS[1:]):
        for key in ('R_cf','R_physical','centroid','cf_extents','physical_registration_Q'):
            assert old['arms'][a]['actual_pose'][key]==new['arms'][b]['actual_pose'][key]
        assert old['arms'][a]['WD_family']==new['arms'][b]['WD_family']
