"""Synthetic behavioral tests; no real-frame pose/reference labels are read."""
from copy import deepcopy

import numpy as np
import pytest

from . import boundary_model as B


def scene(scale=1):
    xyz = np.array([[-2,-1,-1],[2,-1,-1],[2,1,-1],[-2,1,-1],
                    [-2,-1,1],[2,-1,1],[2,1,1],[-2,1,1]], float)
    K = np.array([[150,0,320],[0,150,240],[0,0,1]], float)
    K[:2] *= scale
    camera = xyz + [0,0,4]
    homogeneous = camera @ K.T
    projection = homogeneous[:,:2] / homogeneous[:,2:]
    mask = np.zeros((480,640), float)
    mask[190:291,220:421] = 1
    image = np.empty((480,640,3), np.uint8)
    image[:] = [10,30,220]
    image[mask > .5] = [220,30,10]
    if scale != 1:
        mask = np.repeat(np.repeat(mask, scale, 0), scale, 1)
        image = np.repeat(np.repeat(image, scale, 0), scale, 1)
    spec = dict(states=["B"]*4+["S"]*4, image_hw=list(mask.shape), xyz=xyz.tolist(), K=K.tolist(),
                coarse_pose=dict(R=np.eye(3).tolist(),t=[0,0,4]),
                visible=[dict(native_corner=k,xy=projection[k].tolist(),state="B",fit_reliability_weight=.8)
                         for k in range(4)], boundary_observations=[])
    packet = dict(solver_input=spec,activation=dict(complete_ids=[0,1,2,3],all_partial_ids=[]))
    return image, mask, packet, projection


def test_foreground_to_background_direction_and_no_precise_hidden_points():
    image, mask, packet, projection = scene()
    bundle = B.build_boundary_bundle(image, mask, packet)
    assert bundle["consumed_native_ids"] == [0,1,2,3]
    assert bundle["consumed_group_ids"] == ["full:0","full:1","full:2","full:3"]
    assert bundle["new_precise_O_T_S_U_I_observations"] == 0
    assert len(bundle["groups"]) == 4
    for group in bundle["groups"]:
        assert group["supported"] and group["supported_query_count"] == 8
        for query in group["queries"]:
            pf = np.asarray(query["foreground_probability"])
            assert pf[0] > .5 and pf[-1] < .5
            assert abs(sum(query["offset_probability"])-1) < 1e-12
    assert np.isfinite(B.candidate_energy(bundle, projection))


def test_normal_profile_intersects_physical_edge_not_corner():
    image, mask, packet, projection = scene()
    bundle = B.build_boundary_bundle(image, mask, packet)
    for group in bundle["groups"]:
        for q in group["queries"]:
            offset, grad = B._intersection_offset_and_gradient(q, projection, 1.)
            assert abs(offset) < 1e-12
            # Moving only the corner along its edge does not move the line.
            changed = projection.copy()
            k, m = q["native_corner"], q["native_neighbor"]
            changed[k] += .1 * (projection[m]-projection[k])
            d, _ = B._intersection_offset_and_gradient(q, changed, 1.)
            assert abs(d) < 1e-10


def test_uniform_resolution_scale_keeps_energy_and_actual_sample_coordinates():
    image, mask, packet, projection = scene()
    image2, mask2, packet2, projection2 = scene(2)
    one, two = B.build_boundary_bundle(image,mask,packet), B.build_boundary_bundle(image2,mask2,packet2)
    assert two["pixel_scale"] == 2 * one["pixel_scale"]
    for a, b in zip(one["groups"], two["groups"]):
        for p, q in zip(a["queries"], b["queries"]):
            np.testing.assert_allclose(q["sample_xy"], 2*np.asarray(p["sample_xy"]),atol=1e-12)
            np.testing.assert_allclose(p["foreground_probability"],q["foreground_probability"],atol=1e-12)
    for model in ("region","edge"):
        q, q2 = projection+[2.31,1.77], projection2+2*np.array([2.31,1.77])
        assert B.candidate_energy(one,q,model) == pytest.approx(B.candidate_energy(two,q2,model),abs=1e-11)


def test_multiple_transitions_and_flat_support_are_not_gaussianized():
    r=np.arange(-43,44,dtype=float); d=np.arange(-36,37,dtype=float)
    # Two equally supported foreground->background transitions with a reverse
    # transition between them. The two maxima must survive on the same grid.
    pf=np.where(r < -12,1.,np.where(r < 0,0.,np.where(r < 12,1.,0.)))
    p, logp=B.offset_distribution(pf,r,d)
    peaks=B._peak_indices(p)
    assert any(d[j] < -8 for j in peaks) and any(d[j] > 8 for j in peaks)
    assert abs(logp.max()-logp.min()) > 1
    flat, _=B.offset_distribution(np.full_like(r,.5),r,d)
    np.testing.assert_allclose(flat,np.full(len(d),1/len(d)),atol=1e-14)
    image,mask,packet,_=scene(); image[:]=[80,80,80]
    bundle=B.build_boundary_bundle(image,mask,packet)
    assert bundle["consumed_native_ids"] == []
    assert all(not g["supported"] for g in bundle["groups"])
    assert all(q["exact_flat"] for g in bundle["groups"] for q in g["queries"])


def test_S_O_T_U_I_are_gated_and_unsupported_is_explicit():
    image,mask,packet,projection=scene()
    packet["solver_input"]["states"]=["I","O","T","U","S","S","S","S"]
    packet["solver_input"]["visible"]=[]
    bundle=B.build_boundary_bundle(image,mask,packet)
    assert bundle["groups"] == [] and bundle["consumed_native_ids"] == []
    assert B.candidate_energy(bundle,projection) == 0
    image,mask,packet,projection=scene(); mask[:]=0
    bundle=B.build_boundary_bundle(image,mask,packet)
    assert not bundle["shared_appearance"]["supported"]
    assert bundle["consumed_native_ids"] == []


def test_actual_border_is_not_padded_and_wrong_direction_is_not_refit():
    image,mask,packet,projection=scene()
    packet["solver_input"]["visible"][0]["xy"]=[1.,1.]
    bundle=B.build_boundary_bundle(image,mask,packet)
    assert any("outside_actual_RGB" in r["reason"] for r in bundle["skipped"])
    reversed_mask=1-mask
    reversed_bundle=B.build_boundary_bundle(image,reversed_mask,packet)
    assert not reversed_bundle["consumed_native_ids"]


def test_candidate_cannot_change_support_or_appearance_and_duplicates_have_no_gain():
    image,mask,packet,projection=scene(); bundle=B.build_boundary_bundle(image,mask,packet)
    before=deepcopy(bundle)
    value=B.candidate_energy(bundle,projection+[1.27,2.83])
    B.candidate_energy(bundle,projection+200)
    assert bundle == before
    duplicated=deepcopy(bundle)
    duplicated["groups"][0]["queries"] *= 3
    assert B.candidate_energy(duplicated,projection+[1.27,2.83]) == pytest.approx(value,abs=1e-12)


@pytest.mark.parametrize("model",["region","edge"])
def test_projected_coordinate_gradient_matches_finite_difference(model):
    image,mask,packet,projection=scene(); bundle=B.build_boundary_bundle(image,mask,packet)
    q=projection+np.array([[.213,1.713],[1.103,-.723],[.733,-1.223],[-1.313,.433],
                           [.523,.173],[-.433,.313],[.733,-.143],[-.913,.533]])
    value,gradient=B.energy_and_gradient(bundle,q,model)
    numerical=np.zeros((8,2)); step=1e-5
    for i in range(8):
        for j in range(2):
            a,b=q.copy(),q.copy();a[i,j]+=step;b[i,j]-=step
            numerical[i,j]=(B.candidate_energy(bundle,a,model)-B.candidate_energy(bundle,b,model))/(2*step)
    np.testing.assert_allclose(gradient,numerical,rtol=3e-6,atol=3e-6)
    assert np.isfinite(value)


def test_reference_or_manual_fields_are_not_model_inputs():
    image,mask,packet,projection=scene(); a=B.build_boundary_bundle(image,mask,packet)
    packet["e0_pose"]="POISONED_NOT_A_POSE"
    packet["reference"]={"t":"POISONED"};packet["manual_boundaries"]="POISONED"
    packet["partition"]="POISONED"; packet["id"]="POISONED"
    b=B.build_boundary_bundle(image,mask,packet)
    assert a==b
    assert B.candidate_energy(a,projection)==B.candidate_energy(b,projection)
