"""Independent arithmetic contracts, deliberately no production guard imports."""
from copy import deepcopy
import math
import numpy as np
from . import independent as I


def test_line_distance_endpoint_reversal_and_weighted_huber():
    q=np.array([[0.,0.],[10.,0.]])
    points=np.array([[0.,0.],[2.,1.],[5.,3.]])
    weights=np.array([1.,2.,1.])
    value=I.ray_cost(points,weights,q,0,1)
    assert math.isclose(value[0],.9921875,abs_tol=1e-14)
    assert value==I.ray_cost(points,weights,q,1,0)
    assert math.isclose(value[0],I.ray_cost(points,7*weights,q,0,1)[0],abs_tol=1e-14)


def test_physical_C2_and_no_pose_failure_keep_denominator():
    truth=dict(t=[0.,0.,2.],R=np.eye(3).tolist(),xyz=[1.1,.11,1.3],order=2)
    p=dict(available=True,centroid=truth['t'],R_physical=I.signed_yaw('YAW_180').tolist())
    m=I.metric(p,truth)
    assert m['translation_cm']==m['rotation_deg']==0
    assert m['ADDsym_cm']<1e-12 and m['success_le5'] is True
    missing=I.metric(dict(available=False),truth)
    assert missing['success_le5'] is False and missing['translation_cm'] is None
    d=I.distribution([m['translation_cm'],missing['translation_cm']])
    assert (d['n'],d['denominator'],d['missing'])==(1,2,1)


def test_fixed_candidate_pair_requires_exact_center_copy():
    nominal=np.eye(4);nominal[:3,3]=[.1,-.2,2.]
    records=[]
    for assignment in ['YAW_90','YAW_270']:
        T=nominal.copy();T[:3,:3]=I.signed_yaw(assignment)
        records.append(dict(axis_assignment=assignment,pose_transform=T.tolist()))
    doc=dict(schema_version='real_pallet_gt_v2',objects=[dict(
        camera_facing_pnp=dict(pose_transform=nominal.tolist(),axis_assignment_confirmed=False),
        physical_dimensions_m=dict(x=1.1,y=.11,z=1.3),canonical_pose=None,canonical_pose_candidates=records)])
    ref=I.reference_document(doc)
    assert np.array_equal(ref['R'],I.signed_yaw('YAW_90'))
    tampered=deepcopy(doc)
    tampered['objects'][0]['canonical_pose_candidates'][0]['pose_transform'][0][3]+=1e-15
    assert I.reference_document(tampered)['R'] is None
