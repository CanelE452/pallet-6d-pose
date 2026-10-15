from copy import deepcopy

import numpy as np
import pytest

from . import common as C
from . import select_frozen_wd_partial_refinement_v4_13e as S


def fixture():
    rotate=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    geometry=[]; arms={};states=['B','I','O','S','U','B','S','B']
    for name,dims,Q,depth in [('ORIGINAL_WD',[1.1,.11,1.3],np.eye(3),2.),
                               ('SWAPPED_WD',[1.3,.11,1.1],rotate,3.)]:
        geometry.append(dict(name=name,native_xyz=C.cuboid(*dims).tolist(),cf_extents=dims,physical_registration_Q=Q.tolist()))
        arms[name]=dict(WD_family=name,states=states,accepted=True,
            actual_pose=dict(R_cf=Q.T.tolist(),R_physical=np.eye(3).tolist(),physical_registration_Q=Q.tolist(),
                             centroid=[0.,0.,depth],cf_extents=dims),
            ambiguity_audit=dict(material_ambiguity=False),solution=dict(ambiguity=dict(detected=False),OT_solved=True),
            keypoint_readout=dict(reported_points=[[1.,2.]]*8+[None],point_sources=['SOURCE']*9,OT_solved=True))
    old=dict(id='synthetic',partition='CAL',session='scene',image_sha256='image',states=states,
        frozen_BI_fit=[dict(native_corner=0,xy=[1.,2.],covariance_px2=[[1.,0.],[0.,1.]])],
        u_region_observation=dict(u_regions=[]),selected_family='SWAPPED_WD',families=geometry,
        family_ambiguity=dict(detected=False))
    new={k:deepcopy(old[k]) for k in ('image_sha256','states','frozen_BI_fit','u_region_observation')}
    new.update(selected_family='ORIGINAL_WD',families=deepcopy(geometry),arms=arms,family_ambiguity=dict(detected=True))
    partial={k:old[k] for k in ('id','partition','session','image_sha256','states')}
    partial['arms']={S.D.ARMS[1]:new}
    return old,partial


def test_frozen_family_reads_new_same_family_pose_and_ignores_metrics():
    class Poison:
        def __deepcopy__(self,memo):raise AssertionError('Forbidden score copied')
    old,partial=fixture();old['metrics']=Poison();partial['metrics']=Poison()
    chosen=S.select_same_family(old,partial)
    assert chosen['WD_family']=='SWAPPED_WD'
    assert chosen['actual_pose']==partial['arms'][S.D.ARMS[1]]['arms']['SWAPPED_WD']['actual_pose']
    assert chosen['actual_pose']['centroid']==[0.,0.,3.]
    assert not chosen['hierarchical_correspondence_audit']['old_pose_used_as_output']


def test_old_family_uncertainty_preserved_and_no_false_OT_uniqueness():
    old,partial=fixture();old['family_ambiguity']['detected']=True
    before=deepcopy(partial)
    chosen=S.select_same_family(old,partial)
    assert chosen['ambiguity_audit']['material_ambiguity']
    assert not chosen['solution']['OT_solved'] and not chosen['keypoint_readout']['OT_solved']
    assert chosen['keypoint_readout']['point_sources'][2]=='AMBIGUOUS_FROZEN_FAMILY_JOINT_PROJECTION'
    assert chosen['keypoint_readout']['point_sources'][0]=='SOURCE'
    assert partial==before


def test_family_registration_mismatch_rejected_and_failed_family_retained():
    old,partial=fixture();partial['arms'][S.D.ARMS[1]]['arms']['SWAPPED_WD']['actual_pose']['R_physical']=np.diag([1.,-1.,-1.]).tolist()
    with pytest.raises(ValueError,match='registration differs'):S.select_same_family(old,partial)
    old,partial=fixture();arm=partial['arms'][S.D.ARMS[1]]['arms']['SWAPPED_WD'];arm['accepted']=False
    arm['actual_pose']['source']='EXPLICIT_SAME_FAMILY_CURRENT_FALLBACK'
    chosen=S.select_same_family(old,partial)
    assert not chosen['accepted'] and chosen['WD_family']=='SWAPPED_WD'
    assert chosen['actual_pose']['source']=='EXPLICIT_SAME_FAMILY_CURRENT_FALLBACK'
