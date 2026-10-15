"""Historical-input/source gate contracts; no actual pixels, GT or pose fits."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import run_joint_geometry_guided_line_cases_v4_28 as R


class Poison(dict):
    def __getitem__(self,key):
        if key=='native_corner':return super().__getitem__(key)
        raise AssertionError('Forbidden S/U coordinate/field access')
    def __deepcopy__(self,memo):raise AssertionError('Forbidden S/U record copy')


def test_targeted_metadata_filter_keeps_both_requested_ids_without_other_decode(tmp_path):
    path=tmp_path/'synthetic.jsonl.gz'
    with gzip.open(path,'wt')as stream:
        stream.write('unrelated metadata deliberately not JSON\n')
        stream.write(json.dumps(dict(id='a',payload='allowed'))+'\n')
        stream.write('more unrelated invalid JSON\n')
        stream.write(json.dumps(dict(id='b',payload='allowed'))+'\n')
        stream.write('later rows must not be requested\n')
    rows,digests=R._target_rows(path,['a','b'])
    assert list(rows)==['a','b'] and len(digests)==2
    with pytest.raises(ValueError,match='Both exact'):
        R._target_rows(path,['missing'])


def test_source_gate_precedes_both_point_and_legacy_record_copy():
    states=['S','U','B','I','O','T','S','U'];hidden=[s=='S'for s in states]
    records=[Poison(native_corner=0),Poison(native_corner=1),
        dict(native_corner=2,state='B',xy=[12.,13.],source='VISIBLE_BACKBONE_FALLBACK'),
        dict(native_corner=3,state='I',xy=[16.,18.],source='I_INTERNAL_RGB_JUNCTION')]
    visible=R._visible(records,states,hidden)
    assert [r['native_corner']for r in visible]==[2,3]
    legacy=R._old_records(dict(records=records),states,hidden)
    assert [r['native_corner']for r in legacy]==[2,3]


def test_only_coarse_bank_used_not_later_recovered_q_or_e_or_winner():
    xyz=R.C.cuboid(.8,.4,.6);Q=Rotation.from_euler('y',90,degrees=True).as_matrix()
    physical=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();t=np.array([0.,0.,3.])
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    geometry=dict(native_xyz=xyz.tolist(),K=K.tolist(),R_cf=physical.tolist(),centroid=t.tolist(),
        R_physical=physical.tolist(),physical_registration_Q=np.eye(3).tolist(),cf_extents=[.8,.4,.6],
        initial_projection_xy=R.C.project(xyz,physical,t,K).tolist(),initial_self_hidden=[False]*8,
        initial_roles=['B']*8,image_hw=[480,640])
    task=dict(families={'SWAPPED_WD':dict(native_xyz=(xyz@Q.T).tolist(),physical_registration_Q=Q.tolist(),
        cf_extents=[.6,.4,.8],coarse_pose=dict(R=(physical@Q.T).tolist(),t=t.tolist()))},
        baseline=Poison(),proposals=Poison(),selected_method_winner=Poison())
    before=deepcopy(geometry);guides=R._coarse_guides(task,geometry)
    assert geometry==before and len(guides)==1
    trial=guides[0]['geometry']
    assert guides[0]['source_kind']=='ORIGINAL_TASK_COARSE'
    assert np.allclose(trial['R_physical'],physical,atol=1e-12,rtol=0)
    assert np.allclose(trial['initial_projection_xy'],geometry['initial_projection_xy'],atol=1e-10,rtol=0)
    assert trial['initial_self_hidden']==geometry['initial_self_hidden']


def test_geometry_whitelist_never_copies_raw_or_reference_coordinates():
    geometry={key:[]for key in R.GEOMETRY_KEYS}
    geometry.update(original_raw_points=Poison(),reference_pose=Poison(),human_points=Poison())
    copied=R._geometry(geometry)
    assert set(copied)==set(R.GEOMETRY_KEYS)
    assert 'original_raw_points'not in copied and 'reference_pose'not in copied


def test_guard_diagnostics_keep_zero_support_and_specific_failure_fields():
    observation=dict(edge_records=[dict(native_corner=0,native_neighbor=1,supported=False,
        reason='S_endpoint_excluded_before_projection_and_image_query'),
        dict(native_corner=2,native_neighbor=3,supported=False,reason='no_connected_image_peak_chain_passed_existing_line_guards',
            image_NMS_peak_count=10,RGB_context_peak_count=0,normal_search_radius_px=43.,
            rejected_chains=[dict(reason='angle_change_too_large',angle_change_deg=18.,inlier_count=8,
                inlier_fraction=8/15,inlier_span_fraction=.5,inlier_rms_px=.8)])])
    result=R._guard_summary(observation)
    assert len(result['rays'])==2 and sum(result['reason_counts'].values())==2
    assert result['rays'][1]['actual_RGB_context_peaks']==0
    assert result['rays'][1]['rejected_chains'][0]['inlier_span_fraction']==.5


def test_changed_input_seal_stops_before_any_actual_image_or_NMS_call(tmp_path,monkeypatch):
    def read(path):
        if Path(path).name=='INPUT_SEAL.json':return dict(inputs_sha256='expected',plan_sha256='plan')
        if Path(path).name=='PLAN.json':return {}
        raise AssertionError('Unsealed input opened')
    def forbidden(*args,**kw):raise AssertionError('Actual pixel/line search before input validation')
    monkeypatch.setattr(R.C,'read',read);monkeypatch.setattr(R.C,'sha',lambda path:'changed')
    monkeypatch.setattr(R.cv2,'imread',forbidden)
    monkeypatch.setattr(R.JOINT,'observe_joint_geometry_guided_lines',forbidden)
    with pytest.raises(ValueError,match='Frozen two-case'):
        R.observe(tmp_path)


def test_expected_visibility_uses_only_selected_actual_support_and_never_U_anchor():
    class Context:
        def __init__(self):self.calls=[]
        def classify_expected_edge(self,k,mate,*,observed_chain_points,mask_context_kind):
            self.calls.append((k,mate,np.asarray(observed_chain_points),mask_context_kind))
            assert k in (0,2) and mate not in (4,5,6,7)
            return dict(samples=[],counts=dict(observed_support=len(observed_chain_points),
                occluder_supported_gap=0,unsupported_unknown=1))
    row=dict(states=['B','U','B','U','S','S','S','S'],
        geometry=dict(initial_self_hidden=[False]*4+[True]*4))
    graph=dict(selected_line_names=['chosen'],line_candidates=[dict(name='chosen',native_edge=[0,1],
        actual_inliers=[[12.,20.],[13.,20.]],line_record=dict(mask_context_kind='TARGET_BACKGROUND_OUTLINE')),
        dict(name='unselected',native_edge=[2,3],actual_inliers=[[99.,99.]],line_record={})])
    context=Context();result=R._expected_visibility(row,graph,context)
    assert result['counts']==dict(observed_support=2,occluder_supported_gap=0,unsupported_unknown=4)
    supported=[c for c in context.calls if len(c[2])]
    assert len(supported)==1 and supported[0][3]=='TARGET_BACKGROUND_OUTLINE'
    assert np.array_equal(supported[0][2],[[12.,20.],[13.,20.]])
    assert result['independent_occluder_present']is False
