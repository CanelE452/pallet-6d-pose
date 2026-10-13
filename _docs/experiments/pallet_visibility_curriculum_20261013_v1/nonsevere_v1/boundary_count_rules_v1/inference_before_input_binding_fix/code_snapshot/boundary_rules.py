"""Optional count-based correction using independently observed boundary points.

Counts refer to RGB/material and two actual image-line support, conditional on
the original model/queries. They do not certify requested-corner semantics or
pose rank. Three supported points estimate translation only with exact fixed R;
two or fewer retain the supplied pose. Missing observations are never projected
into the fit, and excluded interiors/unknowns never supplement the count.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import curriculum as M
from .boundary_evidence import summarize_boundary_evidence


def _translation_geometry(xyz, points, K, R, t, ids, cfg):
    ids = np.asarray(ids, int)
    if len(ids) != 3:
        return dict(usable=False, reason="translation_rule_requires_exactly_three_points")
    shape = M._shape(points[ids])
    if shape["spread_ratio"] < cfg.minimum_spread_ratio or shape["singular_values"][1] < cfg.minimum_spread_px:
        return dict(usable=False, reason="insufficient_image_spread", image=shape)
    camera = xyz[ids] @ R.T + t
    homogeneous = camera @ K.T
    derivative = (K[None,:2,:]*homogeneous[:,2,None,None]
                  - homogeneous[:,:2,None]*K[None,2:3,:]) / homogeneous[:,2,None,None]**2
    jac = derivative.reshape(-1,3)
    jac /= np.maximum(np.linalg.norm(jac,axis=0),1e-12)[None]
    singular = np.linalg.svd(jac,compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-9))
    condition = float(singular[0]/max(singular[-1],1e-15))
    usable = rank == 3 and condition <= cfg.maximum_jacobian_condition
    return dict(usable=usable,reason="usable_translation_geometry" if usable else "unstable_translation_geometry",
        image=shape,jacobian_rank=rank,jacobian_condition=condition,jacobian_singular_values=singular.tolist(),
        rotation_parameters_estimated=0,translation_parameters_estimated=3)


def _fit_translation(xyz, points, K, R0, t0, ids, weights, cfg):
    """Three-parameter IRLS; no rotation variable or PnP completion exists."""
    ids = np.asarray(ids,int)
    X,target = xyz[ids] @ R0.T,points[ids]
    scale = max(np.linalg.norm(t0),np.linalg.norm(np.ptp(xyz,axis=0)),1e-6)
    x = np.zeros(3)
    robust = np.asarray(weights,float).copy()
    def residual(delta):
        camera = X+t0+delta*scale
        homogeneous = camera @ K.T
        pixel = (homogeneous[:,:2]/homogeneous[:,2:3]-target)*np.sqrt(robust[:,None])
        return np.r_[pixel.ravel(),cfg.prior_strength_px*delta/cfg.translation_prior_fraction]
    nfev = 0
    for _ in range(cfg.irls_steps):
        solved = least_squares(residual,x,method="trf",max_nfev=cfg.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        x = solved.x;nfev += solved.nfev
        t = t0+x*scale
        error = np.linalg.norm(M.project(xyz[ids],R0,t,K)-target,axis=1)
        updated = weights*np.minimum(1.,cfg.robust_px/np.maximum(error,1e-12))
        converged = np.max(np.abs(updated-robust)) < 1e-4
        robust = updated
        if converged:break
    return R0.copy(),t0+x*scale,dict(nfev=int(nfev),optimizer_success=bool(solved.success),
        optimizer_message=str(solved.message),robust_weights=robust.tolist(),estimated_dof=3,
        rotation_held_exactly_fixed=True)


def _pose_gate(xyz, points, K, R, t, R0, t0, ids, cfg, translation_only):
    projected = M.project(xyz,R,t,K)
    depth = (xyz @ R.T+t)[:,2]
    if not np.isfinite(projected).all() or not np.isfinite(t).all() or (depth<=1e-6).any():
        return dict(usable=False,reason="invalid_camera_pose")
    rotation = M._rotation_angle(R @ R0.T)
    translation = float(np.linalg.norm(t-t0)/max(np.linalg.norm(t0),1e-6))
    shift = float(np.max(np.linalg.norm(projected-M.project(xyz,R0,t0,K),axis=1)))
    if translation_only and not np.array_equal(R,R0):
        return dict(usable=False,reason="translation_rule_changed_rotation")
    geometry = (_translation_geometry(xyz,points,K,R,t,ids,cfg) if translation_only else
                M._geometry(xyz,points,ids,R,t,K,cfg))
    if rotation>cfg.maximum_rotation_deg or translation>cfg.maximum_translation_fraction or shift>cfg.maximum_projection_shift_px:
        reason = "exceeds_initial_pose_motion_limit"
    elif not geometry["usable"]:
        reason = "unstable_candidate_geometry"
    else:reason = "usable"
    return dict(usable=reason=="usable",reason=reason,geometry=geometry,rotation_change_deg=rotation,
        translation_fraction=translation,maximum_projection_shift_px=shift)


def _ambiguity_records(candidates, xyz, t0, cfg):
    """Finite probe ambiguity only, never a claim of global pose uniqueness."""
    if not candidates:return []
    best = min(candidates,key=lambda c:(c["objective"]["objective"],c["probe_index"]))
    scale = max(np.linalg.norm(t0),np.linalg.norm(np.ptp(xyz,axis=0)),1e-6)
    conflicts = []
    for candidate in candidates:
        rotation = M._rotation_angle(candidate["R"] @ best["R"].T)
        translation = float(np.linalg.norm(candidate["t"]-best["t"])/scale)
        gap = candidate["objective"]["objective"]-best["objective"]["objective"]
        if gap<=cfg.validation_tolerance_px2 and (rotation>cfg.rotation_prior_deg/2 or translation>cfg.translation_prior_fraction/2):
            conflicts.append(dict(best_probe=best["probe_index"],alternative_probe=candidate["probe_index"],
                objective_gap=gap,rotation_separation_deg=rotation,normalised_translation_separation=translation))
    return conflicts


def solve_boundary_rules(observation_result, xyz, K, initial_R, initial_t, config=None):
    """Return standard R/t plus count, support, DOF, fallback and native states.

    config: ``coordinate_source`` RAW/FUSED (default FUSED), actual ``image_hw``
    or observation_result.image_hw, ``strong_visibility_threshold`` (default .65),
    and optional ``pose`` fields of the existing CurriculumConfig. The RGB head,
    observed lines and original H are input evidence, never recomputed from fit.
    """
    config = deepcopy(config or {})
    unknown = set(config)-{"coordinate_source","image_hw","strong_visibility_threshold","pose"}
    if unknown:raise ValueError("Unknown boundary rule fields: "+str(sorted(unknown)))
    source = str(config.get("coordinate_source","FUSED")).upper()
    if source not in {"RAW","FUSED"}:raise ValueError("coordinate_source must be RAW or FUSED")
    cfg = M.CurriculumConfig(**dict(config.get("pose",{})))
    if min(cfg.robust_px,cfg.rotation_prior_deg,cfg.translation_prior_fraction,cfg.irls_steps,cfg.max_nfev)<=0:
        raise ValueError("Positive robust/prior/optimizer settings required")
    xyz,K,R0,t0 = np.asarray(xyz,float),np.asarray(K,float),np.asarray(initial_R,float),np.asarray(initial_t,float)
    if xyz.shape!=(8,3) or K.shape!=(3,3) or R0.shape!=(3,3) or t0.shape!=(3,):
        raise ValueError("Expected ordered xyz8, K/R3x3 and t3")
    if not all(np.isfinite(a).all() for a in (xyz,K,R0,t0)) or min(K[0,0],K[1,1])<=0:
        raise ValueError("Finite model/pose/camera with positive focal lengths required")
    if not np.allclose(R0.T@R0,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R0),1.,atol=1e-6):
        raise ValueError("Proper initial rotation required")
    if (xyz @ R0.T+t0)[:,2].min()<=1e-6:raise ValueError("Initial model must be in front of camera")
    image_hw = config.get("image_hw",observation_result.get("image_hw"))
    evidence = summarize_boundary_evidence(observation_result,image_hw=image_hw,
        strong_visibility_threshold=config.get("strong_visibility_threshold",.65),
        occluding_evidence=observation_result.get("occluding_evidence"))
    raw = np.asarray(observation_result["raw_points"],float)
    q = np.asarray(observation_result["raw_points" if source=="RAW" else "fused_points"],float)
    if q.shape!=raw.shape or q.shape not in ((8,2),(9,2)):
        raise ValueError("Fit coordinates must preserve original8/9slots")
    H = np.asarray(observation_result.get("self_hidden",observation_result.get("fixed_H")),bool)
    if H.shape!=(8,):raise ValueError("Fixed original H8 required")
    if "fixed_H" in observation_result and not np.array_equal(H,np.asarray(observation_result["fixed_H"],bool)):
        raise ValueError("Original self masks disagree")
    initial_projected = M.project(xyz,R0,t0,K)
    hull = cv2.convexHull(initial_projected.astype(np.float32),returnPoints=False).ravel()
    expected = np.isin(np.arange(8),hull)
    if not np.array_equal(expected,np.asarray(observation_result["boundary_visibility"]["expected_outline"],bool)):
        raise ValueError("Boundary evidence must retain supplied initial model/pose outline")
    unavailable = H | np.asarray([row["raw_prediction_missing"] or row["raw_prediction_out_of_frame"]
                                 for row in evidence["records"]],bool)
    if not np.array_equal(q[:8][unavailable],raw[:8][unavailable],equal_nan=True) or not np.array_equal(q[8:],raw[8:],equal_nan=True):
        raise ValueError("Independent observation provider changed hidden points or center, or missing/outside points")
    if source=="FUSED":
        valid = np.isfinite(raw[:8]).all(1)&~(raw[:8]==-1).all(1)
        if not np.array_equal(q[:8][~valid],raw[:8][~valid],equal_nan=True):
            raise ValueError("Observation provider changed missing points")
        if not np.isfinite(q[:8][valid]).all() or np.any(np.linalg.norm(q[:8][valid]-raw[:8][valid],axis=1)>6.+1e-8):
            raise ValueError("Expected bounded sigma2 fusion within6nativepixels")
    ids = evidence["strong_ids"]
    if any(H[k] for k in ids) or not np.isfinite(q[:8][ids]).all():
        raise ValueError("Strong observations must remain finite and independent of hidden completion")
    count = len(ids)
    app = observation_result["visibility_appearance"]
    visibility = M._array_field(app,"visibility_score",0.)
    accuracy = M._array_field(app,"local_accuracy_score",0.)
    states = np.asarray(app["states"])
    weights = np.maximum(visibility,.05)*np.maximum(accuracy,.05)
    weights[states=="VISIBLE_INTERIOR"]*=cfg.interior_weight
    before = M._aggregate_objective(xyz,q[:8],K,R0,t0,R0,t0,ids,weights,cfg)
    R,t = R0.copy(),t0.copy()
    branch = ("FIVE_OR_MORE_FULL_6DOF" if count>=5 else "FOUR_FULL_6DOF_INITIAL_PRIOR" if count==4
              else "THREE_TRANSLATION_ONLY" if count==3 else "TWO_OR_FEWER_INITIAL")
    stages = [dict(name="initial",accepted=True,reason="supplied_initial_pose",used_ids=[],R=R.tolist(),t=t.tolist())]
    candidates = [];conflicts = [];accepted=False;fallback="insufficient_image_supported_boundary_points"
    geometry = None
    if count>=3:
        translation_only = count==3
        geometry = (_translation_geometry(xyz,q[:8],K,R0,t0,ids,cfg) if translation_only else
                    M._geometry(xyz,q[:8],ids,R0,t0,K,cfg))
        fallback = geometry["reason"]
        starts = [("initial",R0)]
        if count==4:
            for axis in range(3):
                for sign in (1,-1):
                    delta = np.zeros(3);delta[axis]=sign*np.deg2rad(cfg.rotation_prior_deg)
                    starts.append(("axis"+str(axis)+( "+" if sign>0 else "-"),cv2.Rodrigues(delta)[0]@R0))
        if geometry["usable"]:
            for index,(name,start_R) in enumerate(starts):
                stage=dict(name="boundary_count_fit_"+name,used_ids=ids,accepted=False,estimated_dof=3 if translation_only else 6)
                try:
                    if translation_only:
                        candidate_R,candidate_t,optimizer=_fit_translation(xyz,q[:8],K,R0,t0,ids,weights[ids],cfg)
                    else:
                        candidate_R,candidate_t,optimizer=M._fit(xyz,q[:8],K,R0,t0,start_R,t0,ids,weights[ids],cfg)
                    gate=_pose_gate(xyz,q[:8],K,candidate_R,candidate_t,R0,t0,ids,cfg,translation_only)
                    objective=M._aggregate_objective(xyz,q[:8],K,candidate_R,candidate_t,R0,t0,ids,weights,cfg)
                    gain=before["objective"]-objective["objective"]
                    valid=gate["usable"] and gain>cfg.minimum_gain_px2
                    reason=gate["reason"] if not gate["usable"] else "valid_independent_candidate" if valid else "no_selected_boundary_aggregate_gain"
                    stage.update(reason=reason,optimizer=optimizer,gate=gate,aggregate_before=before,aggregate_after=objective,
                        initial_objective_gain=gain,candidate_R=candidate_R.tolist(),candidate_t=candidate_t.tolist())
                    if valid:candidates.append(dict(probe_index=index,stage_index=len(stages),R=candidate_R,t=candidate_t,objective=objective))
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,cv2.error) as error:
                    stage.update(reason="optimizer_failure",error=str(error))
                stages.append(stage)
            conflicts=_ambiguity_records(candidates,xyz,t0,cfg) if count==4 else []
            if conflicts:
                fallback="four_point_near_equal_distinct_pose_candidates"
            elif candidates:
                best=min(candidates,key=lambda c:(c["objective"]["objective"],c["probe_index"]))
                R,t=best["R"].copy(),best["t"].copy();accepted=True;fallback=None
                stages[best["stage_index"]].update(accepted=True,reason="accepted_boundary_count_rule")
            else:
                fallback=stages[-1]["reason"]
    after=M._aggregate_objective(xyz,q[:8],K,R,t,R0,t0,ids,weights,cfg)
    return dict(schema="boundary_count_rule_v1",available=True,accepted=accepted,new_pose_estimated=accepted,
        R=R.tolist(),t=t.tolist(),R_cf=R.tolist(),centroid=t.tolist(),used_ids=ids if accepted else [],
        reliable_boundary_ids=ids,reliable_boundary_count=count,branch=branch,
        estimated_dof=(3 if count==3 else 6) if accepted else 0,rotation_held_fixed=count==3,
        fallback_reason=fallback,reason="refined" if accepted else "initial_pose_preserved",
        evidence=evidence,states=[row["state"] for row in evidence["records"]],
        geometric_boundary_count=evidence["geometric_boundary_count"],
        RGB_material_boundary_count=evidence["RGB_material_supported_boundary_count"],
        initial_geometry=geometry,stages=stages,ambiguity_probe_conflicts=conflicts,
        aggregate_before=before,aggregate_after=after,frozen_selected_weights=weights[ids].tolist(),
        coordinate_source=source,observed_points=M._finite_list(q[:8]),raw_points=M._finite_list(raw),
        projected=M.project(xyz,R,t,K).tolist(),unknown_added_ids=[],interior_supplement_ids=[],
        configuration=dict(coordinate_source=source,strong_visibility_threshold=evidence["strong_visibility_threshold"],pose=asdict(cfg)),
        provenance=dict(ground_truth_inputs=False,projections_reused_as_observations=False,
            hidden_completion_used_for_fit=False,original_model_and_native_corner_ids_preserved=True,
            strong_counts_certify_pose_rank=False,line_covariance_is_semantic_confidence=False,
            unknown_or_unverified_interior_supplements=0,default_method_promoted=False))
