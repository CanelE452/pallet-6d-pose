"""Pure case selection/physical projection/state gates; real rows/refs IO=0."""
from copy import deepcopy
import numpy as np
import pytest

from . import render_clean_fresh_BI_full_result_process_v4_32_r2_v1 as R
from .test_prepare_geometry_guided_native_metadata_v4_32 import Poison
from .test_score_clean_fresh_BI_poses_v4_32 import prediction,LEGACY


def reports():
    def make(n):
        rows=[dict(id='toy:'+str(i),metrics={a:dict(pose_available=True,translation_cm=float(i))for a in R.M.ARMS})for i in range(n)]
        return dict(count=n,rows=rows,summary={a:dict(max_translation_id=rows[-1]['id'])for a in R.M.ARMS})
    return make(192),make(49)


def test_fixed_maximum_union_deduplicates_four_methods_and_adds_test_case():
    p,t=reports();before=deepcopy((p,t));selected=R.choose_cases(p,t)
    assert [v['id']for v in selected]==['toy:191','toy:48']
    assert len(selected[0]['reasons'])==4 and selected[1]['reasons'][0]['arm']==R.SOURCE_ARM
    assert (p,t)==before


def test_no_pose_is_infinite_failure_and_canonical_first_tie_is_retained():
    p,t=reports();arm=R.M.ARMS[1]
    for i in(0,1):p['rows'][i]['metrics'][arm]=dict(pose_available=False,translation_cm=None)
    p['summary'][arm]['max_translation_id']='toy:0'
    selected=R.choose_cases(p,t)
    assert selected[0]['id']=='toy:0'and selected[0]['reasons'][0]['missing_pose_is_infinite_failure']
    p['summary'][arm]['max_translation_id']='toy:1'
    with pytest.raises(ValueError,match='summary'):R.choose_cases(p,t)


def test_zero_drop_population_and_finite_metrics_fail_closed():
    p,t=reports();p['count']=191
    with pytest.raises(ValueError,match='population'):R.choose_cases(p,t)
    p,t=reports();p['rows'][0]['metrics'][R.M.ARMS[1]]['translation_cm']=None
    with pytest.raises(ValueError,match='Finite'):R.choose_cases(p,t)


def test_observed_state_gate_before_S_U_poison_and_no_archive_rewrite():
    packet=dict(clean_input=dict(states=['B','S','U','O','T','I','S','U'],initial_self_hidden=[False]*8,
        visible_measurements=[dict(native_corner=0,state='B',xy=[1,2]),dict(native_corner=1,state='S',xy=Poison()),
            dict(native_corner=2,state='U',xy=Poison()),dict(native_corner=5,state='I',xy=[4,5])]))
    rows=R.archived_BI(packet)
    assert [p['native_corner']for p in rows]==[0,5]
    rows[0]['xy'][0]=99
    assert packet['clean_input']['visible_measurements'][0]['xy']==[1,2]


@pytest.mark.parametrize('family',R.M.FAMILIES)
def test_physical_body_projection_respects_nonzero_origin_and_Q_for_both_WD(family):
    arm=R.M.ARMS[1];entry=prediction(arm,mean=(.2,-.1,.3),family=family)
    g=entry['new_geometry_edge_cost_plans'][family]['initial_model_ray_binding']['geometry']
    packet=dict(clean_input=dict(native_xyz=g['native_xyz'],physical_registration_Q=g['physical_registration_Q'],K=np.eye(3).tolist()))
    actual=R.final_projection(packet,entry,arm,LEGACY)
    Q=np.asarray(g['physical_registration_Q']);xyz=np.asarray(g['native_xyz']);canonical=(Q.T@(xyz-xyz.mean(0)).T).T
    expected=R.C.project(canonical,np.asarray(entry['actual_pose']['R_physical']),np.asarray(entry['actual_pose']['centroid']),np.eye(3))
    assert np.array_equal(actual,expected)
    assert R.final_projection(Poison(),dict(accepted=False,pose_available=False,actual_pose=Poison()),arm,LEGACY)is None


def test_RGB_mapping_is_exact_inputmetadata_not_a_search():
    packet=dict(identifiers=dict(id='sessionA:frame1',session='sessionA',image_sha256='a'*64))
    path='data/evaluation/pallet_eval_v1/some_root/sessions/sessionA/rgb/frame1.png'
    assert R.image_path(packet,{path:'a'*64})==R.C.ROOT/path
    with pytest.raises(ValueError,match='Unique'):R.image_path(packet,{path:'b'*64})
    with pytest.raises(ValueError,match='Unique'):R.image_path(packet,{path:'a'*64,path.replace('some_root','another_root'):'a'*64})


def test_actual_and_metadata_GOs_before_experiment_IO(monkeypatch):
    def no(*a,**kw):raise AssertionError('No experimental IO')
    monkeypatch.setattr(R.C,'read',no);monkeypatch.setattr(R.C,'sha',no)
    with pytest.raises(PermissionError):R.declare()
    with pytest.raises(PermissionError):R.render()
