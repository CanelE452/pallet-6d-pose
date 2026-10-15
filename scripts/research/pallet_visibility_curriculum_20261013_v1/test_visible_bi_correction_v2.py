"""State-specific image measurements and excluded-point invariance."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import common as C
from .visible_bi_correction_v2 import correct_visible_bi


def scene():
    X=C.cuboid(1.1,.25,1.1)
    R=Rotation.from_euler("xyz",[32,-27,7],degrees=True).as_matrix()
    t=np.array([0.,0.,3.])
    K=np.array([[650.,0.,240.],[0.,650.,210.],[0.,0.,1.]])
    P=C.project(X,R,t,K)
    image=np.full((420,480,3),235,np.uint8)
    faces=[([0,1,2,3],[0,0,-1]),([4,5,6,7],[0,0,1]),
           ([0,1,5,4],[0,-1,0]),([3,2,6,7],[0,1,0]),
           ([0,3,7,4],[-1,0,0]),([1,2,6,5],[1,0,0])]
    mask=np.zeros(image.shape[:2],np.uint8)
    colors=[(50,130,60),(110,190,60),(80,100,90),(140,150,60),(65,165,85),(125,100,80)]
    for (ids,n),color in zip(faces,colors):
        if np.array(n)@R.T@(-t)>0:
            polygon=np.rint(P[ids]).astype(np.int32)
            cv2.fillConvexPoly(image,polygon,color)
            cv2.fillConvexPoly(mask,polygon,1)
    hidden=C.self_hidden(X,R,t)
    hull=set(cv2.convexHull(P.astype(np.float32),returnPoints=False).ravel())
    states=["S" if hidden[i] else ("B" if i in hull else "I") for i in range(8)]
    assert states.count("I")==1
    raw=np.vstack([P+np.array([3.,-2.]),[240.,210.]])
    pose=dict(available=True,R_cf=R.tolist(),centroid=t.tolist(),cf_extents=[1.1,.25,1.1])
    return image,mask.astype(float),raw,K,X,pose,states,P


def test_B_and_I_measure_different_current_pixel_evidence():
    image,mask,q,K,X,pose,states,P=scene()
    output=correct_visible_bi(image,q,K,pose,states,pallet_mask=mask,native_xyz=X)
    changed=[r for r in output["records"] if r["corrected"]]
    assert any(r["state"]=="B" for r in changed)
    assert any(r["state"]=="I" for r in changed)
    for row in changed:
        k=row["native_corner"]
        assert np.linalg.norm(np.array(row["measured_xy"])-P[k])<np.linalg.norm(q[k]-P[k])
        expected="OBSERVED_TARGET_BACKGROUND_TRANSITION" if states[k]=="B" else "OBSERVED_INTERNAL_RGB_GRADIENT"
        assert all(r["measurement_source"]==expected for r in row["rays"])
    np.testing.assert_array_equal(output["measured_points"][8],q[8])


def test_excluded_raw_coordinates_cannot_change_BI_measurement_or_uncertainty():
    image,mask,q,K,X,pose,states,_=scene()
    boundary=[i for i,s in enumerate(states) if s=="B"]
    states[boundary[0]]="O";states[boundary[1]]="T"
    states[boundary[2]]="U"
    initial=correct_visible_bi(image,q,K,pose,states,pallet_mask=mask,native_xyz=X)
    perturbed=q.copy()
    for i,s in enumerate(states):
        if s not in ("B","I"):perturbed[i]=[1e9,-1e9]
    second=correct_visible_bi(image,perturbed,K,pose,states,pallet_mask=mask,native_xyz=X)
    assert initial["visible_measurements"]==second["visible_measurements"]
    assert initial["records"]==second["records"]
    assert set(initial["queried_native_ids"])=={i for i,s in enumerate(states) if s in ("B","I")}


def test_blank_pixels_do_not_turn_model_projection_into_corrected_observation():
    image,mask,q,K,X,pose,states,_=scene()
    image[:]=127;mask[:]=.5
    before=deepcopy(q)
    output=correct_visible_bi(image,q,K,pose,states,pallet_mask=mask,native_xyz=X)
    assert all(not r["corrected"] for r in output["records"])
    np.testing.assert_array_equal(output["measured_points"],before)
    assert all(v["source"]=="VISIBLE_BACKBONE_FALLBACK" for v in output["visible_measurements"])
    np.testing.assert_array_equal(q,before)


def test_registered_native_correspondence_is_validated():
    image,mask,q,K,X,pose,states,_=scene()
    with pytest.raises(ValueError,match="Registered native cuboid"):
        correct_visible_bi(image,q,K,pose,states,pallet_mask=mask,native_xyz=X[::-1])
