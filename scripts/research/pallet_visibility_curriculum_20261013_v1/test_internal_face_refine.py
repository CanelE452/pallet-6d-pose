"""Synthetic RGB face boundaries verify the independent-observation contract."""
import cv2
import numpy as np

from . import common as C
from . import internal_face_refine as I


def fixture(same_color=False):
    xyz=C.cuboid(1.,.5,1.)
    R=cv2.Rodrigues(np.array([.5,-.6,.1]))[0]
    t=np.array([0.,0.,3.])
    K=np.array([[420.,0.,160.],[0.,420.,160.],[0.,0.,1.]])
    q=C.project(xyz,R,t,K)
    H=C.self_hidden(xyz,R,t)
    camera=-R.T@t
    visible=[f for f,ids in enumerate(I.FACES) if I.NORMALS[f]@(camera-xyz[list(ids)].mean(0))>0]
    image=np.full((320,320,3),245,np.uint8)
    for index,face in enumerate(visible):
        color=100 if same_color else [70,115,170][index]
        cv2.fillConvexPoly(image,np.rint(q[list(I.FACES[face])]).astype(np.int32),(color,color,color))
    hull=cv2.convexHull(q.astype(np.float32),returnPoints=False).reshape(-1)
    junction=[k for k in range(8) if not H[k] and k not in hull and sum(k in I.FACES[f] for f in visible)==3]
    assert len(junction)==1
    predicted=q.copy();predicted[junction[0]]+=[4.,-3.]
    return image,predicted,H,xyz,R,t,K,q,junction[0]


def test_true_internal_boundary_intersection_and_immutability():
    image,q,H,xyz,R,t,K,truth,k=fixture()
    before=[a.tobytes() for a in (image,q,H)]
    for a in (image,q,H):a.flags.writeable=False
    result=I.infer_internal_face_points(image,q,H,xyz,R,t,K,dict(maximum_corner_shift_px=12.))
    assert k in result['observed_refined_ids'],result['corner_records'][k]
    refined=np.asarray(result['points'])
    assert np.linalg.norm(refined[k]-truth[k])<np.linalg.norm(q[k]-truth[k])
    assert [a.tobytes() for a in (image,q,H)]==before
    assert result['reprojected_points_used'] is False
    assert result['outer_material_background_edges_used'] is False
    assert result['physical3D_corner_certified'] is False
    supported=[r for r in result['edge_records'] if r['supported']]
    assert len(supported)>=2
    assert all(min(r['observed_sideA_material_support'])>=.55 and min(r['observed_sideB_material_support'])>=.55 for r in supported)
    for j in range(8):
        if j!=k:assert np.array_equal(refined[j],q[j])


def test_nonseparable_material_faces_preserve_all_points():
    image,q,H,xyz,R,t,K,_,_=fixture(same_color=True)
    result=I.infer_internal_face_points(image,q,H,xyz,R,t,K)
    assert result['observed_refined_ids']==[]
    assert np.array_equal(result['points'],q)
    assert any(r['reason']=='adjacent_face_palettes_not_separable' for r in result['edge_records'])


def test_incoherent_initial_pose_preserves_points():
    image,q,H,xyz,R,t,K,_,_=fixture()
    result=I.infer_internal_face_points(image,q,H,xyz,R,t+np.array([1.,0.,0.]),K)
    assert result['observed_refined_ids']==[]
    assert not result['diagnostics']['pose_coherence']['usable']
    assert np.array_equal(result['points'],q)


def test_shift_cap_and_fixed_hidden_exclusion():
    image,q,H,xyz,R,t,K,_,k=fixture()
    capped=I.infer_internal_face_points(image,q,H,xyz,R,t,K,dict(maximum_corner_shift_px=.1))
    assert capped['observed_refined_ids']==[]
    hidden=H.copy();hidden[k]=True
    result=I.infer_internal_face_points(image,q,hidden,xyz,R,t,K)
    assert result['observed_refined_ids']==[]
