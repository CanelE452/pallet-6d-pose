"""Prediction-only RGB-supported comparison of two registered W/D hypotheses.

Corner observations, their native channel order and the supplied initial self
mask remain fixed for every candidate. An alternate hypothesis uses standard
all-observation SQPnP+LM only as its prediction-derived initial pose, followed
by its own conservative curriculum. A common independently predicted RGB-visible
pool scores both branches; near ties retain the supplied family/physical basis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy

import cv2
import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.curriculum import solve_curriculum


@dataclass(frozen=True)
class BranchConfig:
    selection_margin_px2: float = 1.0
    minimum_support: int = 4
    support_visibility_threshold: float = .35
    minimum_accuracy: float = .10
    interior_weight: float = .60
    robust_px: float = 8.0
    minimum_spread_ratio: float = .01
    minimum_spread_px: float = 3.
    maximum_jacobian_condition: float = 15.
    curriculum: dict = field(default_factory=lambda:{"visible_threshold":.35})


def physical_from_cf(dimensions, registered_dimensions):
    """Return right pose factor Q: R_physical = R_cf @ Q.

    Physical object coordinates therefore equal Q.T @ x_cf. The historical
    name refers to the pose conversion, not a forward object-point transform.
    """
    dimensions, registry = np.asarray(dimensions), np.asarray(registered_dimensions)
    if np.allclose(dimensions, registry, rtol=0, atol=1e-9):
        return np.eye(3)
    if np.allclose(dimensions, registry[[2,1,0]], rtol=0, atol=1e-9):
        return np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    raise ValueError("Initial dimensions must be an exact registered W/D hypothesis")


def _huber(residual, weights, radius):
    residual = np.asarray(residual)
    losses = np.where(residual<=radius, residual**2, 2*radius*residual-radius**2)
    return float(np.sum(losses*weights)/weights.sum())


def _support(points, self_hidden, appearance, cfg):
    states = np.asarray(appearance["states"])
    visibility = np.asarray(appearance["visibility_score"], float)
    accuracy = np.asarray(appearance["local_accuracy_score"], float)
    independent = np.asarray(appearance.get("independent", np.ones(8,bool)), bool)
    eligible = np.isfinite(points).all(1) & ~np.all(points==-1,axis=1) & ~np.asarray(self_hidden,bool) & independent
    visible = eligible & np.isin(states,["VISIBLE_BOUNDARY","VISIBLE_INTERIOR"]) & (visibility>=cfg.support_visibility_threshold) & (accuracy>=cfg.minimum_accuracy)
    ids = np.flatnonzero(visible)
    weights = np.clip(visibility,0,1)*np.clip(accuracy,0,1)
    weights[states=="VISIBLE_INTERIOR"] *= cfg.interior_weight
    return ids,weights


def _geometry(model, points, ids, R, t, K, cfg):
    if len(ids)<cfg.minimum_support:
        return dict(usable=False, reason="insufficient_shared_RGB_visible_support")
    singular = np.linalg.svd(points[ids]-points[ids].mean(0),compute_uv=False)
    if singular[1]<cfg.minimum_spread_px or singular[1]/max(singular[0],1e-12)<cfg.minimum_spread_ratio:
        return dict(usable=False,reason="insufficient_shared_image_spread")
    _,J=cv2.projectPoints(model[ids],cv2.Rodrigues(R)[0],t,K,None)
    J=J[:,:6];J=J/np.maximum(np.linalg.norm(J,axis=0),1e-12)[None]
    s=np.linalg.svd(J,compute_uv=False);condition=float(s[0]/max(s[-1],1e-15))
    rank=int(np.sum(s>s[0]*1e-9))
    return dict(usable=bool(rank==6 and condition<=cfg.maximum_jacobian_condition),
                reason="usable" if rank==6 and condition<=cfg.maximum_jacobian_condition else "unstable_shared_support",
                condition=condition,numerical_rank=rank)


def _standard_initial(points, model, K):
    eligible=np.isfinite(points).all(1)&~np.all(points==-1,axis=1)
    if eligible.sum()<4:
        return None,"insufficient_prediction_points"
    try:
        ok,rv,tv=cv2.solvePnP(model[eligible],points[eligible],K,None,flags=cv2.SOLVEPNP_SQPNP)
        if not ok:return None,"SQPNP_returned_false"
        rv,tv=cv2.solvePnPRefineLM(model[eligible],points[eligible],K,None,rv,tv)
        R,t=cv2.Rodrigues(rv)[0],tv.reshape(3)
        if not np.isfinite(R).all() or not np.isfinite(t).all() or np.any((model@R.T+t)[:,2]<=1e-6):
            return None,"invalid_camera_pose"
        return (R,t),"all_prediction_SQPNP_LM"
    except cv2.error as exc:
        return None,str(exc)


def solve_branch_curriculum(points,K,registered_dimensions,initial_pose,self_hidden,appearance_result,config=None):
    cfg=BranchConfig(**config) if isinstance(config,dict) else (config or BranchConfig())
    points=np.array(points,float,copy=True);K=np.array(K,float,copy=True)
    registry=np.array(registered_dimensions,float,copy=True);hidden=np.array(self_hidden,bool,copy=True)
    if points.shape!=(8,2) or K.shape!=(3,3) or registry.shape!=(3,) or hidden.shape!=(8,):
        raise ValueError("Expected points8x2, K3x3, registry3, self_hidden8")
    if not np.isfinite(registry).all() or np.any(registry<=0):
        raise ValueError("Registered dimensions must be finite and positive")
    original_dimensions=np.asarray(initial_pose["cf_extents"],float)
    original_R=np.asarray(initial_pose["R_cf"],float);original_t=np.asarray(initial_pose["centroid"],float)
    physical_from_cf(original_dimensions,registry) # enforce family registration
    original_Q=original_R.T@np.asarray(initial_pose["R_physical"],float)
    dimensions=[registry]
    if not np.isclose(registry[0],registry[2],rtol=0,atol=1e-9):dimensions.append(registry[[2,1,0]])
    # Put the supplied family first: ties never depend on dimension enumeration.
    dimensions=sorted(dimensions,key=lambda d:not np.allclose(d,original_dimensions,rtol=0,atol=1e-9))
    ids,weights=_support(points,hidden,appearance_result,cfg)
    candidates=[]
    for index,dims in enumerate(dimensions):
        original=index==0;model=C.cuboid(*dims)
        Q=original_Q if original else physical_from_cf(dims,registry)
        if original:
            start=(original_R,original_t);initializer="supplied_original_pose"
        else:
            start,initializer=_standard_initial(points,model,K)
        base=dict(original=original,dimensions=dims.tolist(),physical_from_cf=Q.tolist(),initializer=initializer,
                  observations_generated_from_pose=False,self_hidden_used=hidden.tolist())
        if start is None:
            candidates.append(dict(base,available=False,cost=None,reason=initializer));continue
        R,t=start
        solution=solve_curriculum(points,model,K,R,t,hidden,appearance_result,cfg.curriculum)
        R,t=np.asarray(solution["R"]),np.asarray(solution["t"])
        projected=C.project(model,R,t,K)
        residual=np.linalg.norm(projected[ids]-points[ids],axis=1)
        geometry=_geometry(model,points,ids,R,t,K,cfg)
        cost=_huber(residual,weights[ids],cfg.robust_px) if len(ids) and weights[ids].sum()>0 else None
        candidates.append(dict(base,available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t.tolist(),
            projected=projected.tolist(),cost=cost,residuals_shared_px=residual.tolist(),shared_geometry=geometry,
            curriculum=solution,reason=solution["reason"],initial_R_cf=start[0].tolist(),initial_t=start[1].tolist()))
    original=candidates[0]
    selected=original;reason="square_W_equals_D_deduplicated" if len(candidates)==1 else "original_retained_near_tie"
    gap=None
    if len(candidates)>1:
        alternate=candidates[1]
        if not original["shared_geometry"]["usable"]:
            reason=original["shared_geometry"]["reason"]
        elif not alternate["available"]:
            reason="alternate_initialization_unavailable"
        elif not alternate["shared_geometry"]["usable"]:
            reason="alternate_shared_geometry_unstable"
        else:
            gap=float(original["cost"]-alternate["cost"])
            if gap>=cfg.selection_margin_px2:
                selected=alternate;reason="alternate_shared_RGB_visible_cost_improved"
    actual_pose=deepcopy(initial_pose)
    actual_pose.update(available=True,R_cf=selected["R_cf"],R_physical=selected["R_physical"],
                       centroid=selected["centroid"],cf_extents=selected["dimensions"],
                       selected_hypothesis="REGISTERED_RGB_WD_BRANCH" if not selected["original"] else initial_pose.get("selected_hypothesis","ORIGINAL"))
    return dict(available=True,branch_flip=not selected["original"],accepted=bool(not selected["original"] or selected["curriculum"]["accepted"]),
                actual_pose=actual_pose,solution=selected["curriculum"],candidates=candidates,selection_reason=reason,
                original_cost=original["cost"],alternate_cost=candidates[1].get("cost") if len(candidates)>1 else None,
                cost_improvement_px2=gap,common_visible_ids=ids.tolist(),common_weights=weights[ids].tolist(),
                observed_points=points.tolist(),self_hidden=hidden.tolist(),configuration=asdict(cfg),
                original_physical_basis_preserved_when_retained=True,independent_observations_only=True,
                observations_reprojected_or_permuted=False,ground_truth_used_for_selection=False)
