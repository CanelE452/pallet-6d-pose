"""Pure whitelisting and aggregate contracts; no data/GT/RGB/pose IO."""
from copy import deepcopy
import numpy as np
import pytest
from . import observe_native_edge_chains_v4_17 as D


class Poison:
    def __array__(self,*a,**k):raise AssertionError('Excluded data converted')
    def __deepcopy__(self,memo):raise AssertionError('Excluded data copied')


def fixture():
    geometry=dict(native_xyz=np.zeros((8,3)).tolist(),physical_registration_Q=np.eye(3).tolist(),
        initial_self_hidden=[False]+[True]*7)
    visible=[dict(native_corner=0,state='B',xy=[10.,10.],covariance_px2=[[4.,0.],[0.,4.]])]
    states=['B']+['S']*7
    return dict(id='synthetic:1',partition='CAL',session='synthetic',image='unused.png',image_sha256='0'*64,
        mask_binding=dict(path='unused.npz',sha256='0'*64),spec=dict(visible=visible,states=states,domains=[]),
        u_regions=[],boundary_all=[],stage1_family_seed_bank={'ORIGINAL_WD':{'family':deepcopy(geometry)}},
        finish_input=dict(_original_source=dict(initial_geometry=geometry,states=states,original_raw_points=Poison()),
            frozen_BI_fit=visible,partial_boundary_observation={'unchanged_nonself_prediction_scale_px':100.}),
        truth=Poison(),baseline=Poison(),initial_solutions=Poison())


def test_whitelist_never_copies_raw_S_reference_or_old_pose_containers():
    row=fixture();out=D._thin_task(row)
    assert out['id']==row['id'] and out['visible_measurements']==row['spec']['visible']
    assert all(key not in out for key in ('original_raw_points','truth','baseline','initial_solutions'))
    assert out['unchanged_observation_math_sha256']==D._digest(D._math_fields(out))


def test_malformed_S_visible_record_rejected_before_poisoned_fields():
    row=fixture();row['spec']['visible'].append(dict(native_corner=1,state='S',xy=Poison(),covariance_px2=Poison()))
    with pytest.raises(ValueError,match='permanent nonS'):D._thin_task(row)


def test_native_geometry_and_visible_covariance_binding_cannot_change():
    row=fixture();row['stage1_family_seed_bank']['ORIGINAL_WD']['family']['physical_registration_Q'][0][0]=2.
    with pytest.raises(ValueError,match='ORIGINAL'):D._thin_task(row)
    row=fixture();row['finish_input']['frozen_BI_fit']=deepcopy(row['spec']['visible'])
    row['finish_input']['frozen_BI_fit'][0]['covariance_px2'][0][0]=5.
    with pytest.raises(ValueError,match='visible math'):D._thin_task(row)


def test_zero_edge_frames_remain_in_summary_denominator():
    def row(identifier,edges):
        return dict(id=identifier,partition='CAL',summary=dict(zero_supported_edges=edges==0,
            supported_directed_edges=edges,supported_unique_native_edges=edges,actual_chain_hypotheses=edges,
            B_junction_candidates=0,linking_overflow_abstained_edges=0))
    out=D._summary([row('a',0),row('b',1)])
    assert out['rows']==2 and out['frames_with_zero_supported_edges']==1
    assert out['partition_counts']=={'CAL':2} and out['pose_calls']==out['NN_calls']==0
