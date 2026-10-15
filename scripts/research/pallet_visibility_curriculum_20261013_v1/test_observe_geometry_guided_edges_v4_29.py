"""Prediction-only row joins; no real inputs, images, labels or models."""
from copy import deepcopy
import numpy as np
import pytest

from . import observe_geometry_guided_edges_v4_29 as V
from .test_image_edge_observations_v4_29 import fixture
from .test_joint_geometry_guided_lines_v4_28 import Poison


def rows():
    args,_=fixture();geometry={k:deepcopy(v)for k,v in args[2].items()if k!='initial_projection_xy'}
    xyz=np.asarray(geometry['native_xyz']);R=np.asarray(geometry['R_cf']);t=np.asarray(geometry['centroid'])-R@xyz.mean(0)
    # These are synthetic MODEL projections, not observed S/U points.
    camera=xyz@R.T+t;h=camera@np.asarray(geometry['K']).T
    geometry['initial_projection_xy']=(h[:,:2]/h[:,2,None]).tolist()
    initial=dict(R=R.tolist(),t=t.tolist());family=dict(native_xyz=geometry['native_xyz'],
        physical_registration_Q=geometry['physical_registration_Q'],coarse_pose=initial)
    domain=dict(native_corner=1,state='O',shape='box',lower_xy=[0,0],upper_xy=[180,120])
    original=dict(native_corner=1,state='O',shape='ellipse',center_xy=[130.,60.],half_axes_px=[9.,9.])
    task=dict(id='synthetic:0',partition='CAL',session='synthetic',image='synthetic.png',image_sha256='a'*64,
        states=args[3],spec=dict(states=args[3],visible=args[-1],K=geometry['K'],image_hw=geometry['image_hw'],domains=[domain]),
        u_regions=[],initial_self_hidden=geometry['initial_self_hidden'],families=dict(ORIGINAL_WD=family,SWAPPED_WD=family),
        proposals=[],boundary_merged=[],new_chain_observation=dict(edge_records=[]),
        O_localization_abstention=dict(records=[dict(native_corner=1,original_domain=original)]))
    observed={k:task[k]for k in('id','partition','session','image','image_sha256','states','u_regions')}
    observed.update(initial_geometry=geometry,visible_measurements=args[-1],domains=[original],
        chain_observation=task['new_chain_observation'],mask_binding=dict(path='synthetic.npz',sha256='b'*64),
        nonself_prediction_scale_px=100.)
    return task,observed


def test_join_retains_exact_runtime_q_guards_and_original_domain_archive_without_baseline_guide():
    task,observed=rows();before=deepcopy(task);result=V._join(task,observed)
    assert task==before
    assert result['domains']==task['spec']['domains']and result['original_observer_domains']==observed['domains']
    assert result['visible_measurements']==task['spec']['visible']and result['states']==task['states']
    assert 'baseline'not in result['guide_task']and 'finish_input'not in result['guide_task']
    assert result['unchanged_observation_math_sha256']==V.OLD._digest(V._math(result))


def test_join_rejects_query_source_state_before_poison_S_coordinate():
    task,observed=rows();record=dict(native_corner=2,state='S',xy=Poison(),source=Poison())
    task['spec']['visible'].append(record)
    with pytest.raises(ValueError,match='gated B/I'):
        V._join(task,observed)


def test_join_wrong_native_Q_or_changed_reference_start_chain_rejects_without_IO():
    task,observed=rows();observed['initial_geometry']['physical_registration_Q']=[[0,0,1],[0,1,0],[-1,0,0]]
    with pytest.raises(ValueError,match='Original registered geometry'):
        V._join(task,observed)
    task,observed=rows();observed['chain_observation']=dict(edge_records=[dict(reason='changed')])
    with pytest.raises(ValueError,match='exact original fresh17c'):
        V._join(task,observed)
