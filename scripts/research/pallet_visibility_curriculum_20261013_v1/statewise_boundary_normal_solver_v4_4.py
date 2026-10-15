"""Opt-in partial boundary-normal factors around the immutable bias solver.

No point is manufactured from a partial contour. Each caller-associated native
U/B vertex has one/two observed line-normal residuals and finite tangent support.
The association, line variances and any unknown-split choice are caller
hypotheses; this module does not certify native identity, physical coverage or
independence of two lines from a shared contour. It reads no image, labels or
files, performs no fitting of covariance, and changes no caller input.

Original BI/prior seed fitting and all OT constraints are preserved. Positive
factor weight adds soft normal information AND hard finite-support guards to
the final joint fit. Empty observations or zero weight delegate directly to the
immutable parent and return its exact result, with no extra hard rows.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import statewise_uncertain_solver_v4_bias as BASE
from . import joint_ot_pnp as V3
from . import joint_ot_jacobian_v2 as J

_configuration = BASE._configuration
_helper_config = BASE._helper_config
_constraints = BASE._constraints
_objective_pair = BASE._objective_pair
_visible_audit = BASE._visible_audit


def _boundary_inputs(records, states, image_hw, xyz=None):
    """Validate state before accessing any line/image-observation fields."""
    if records is None:
        return []
    if not isinstance(records, (list, tuple)):
        raise ValueError("boundary_observations must be a sequence of records")
    normalized=[]; seen=set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Boundary factor records must be mappings")
        k=record.get('native_corner')
        if isinstance(k, (bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 0 <= k < 8:
            raise ValueError("Boundary native_corner must be an integer in [0,7]")
        if k in seen:
            raise ValueError("Each native corner has at most one grouped boundary observation")
        if len(states)!=8 or V3._state(states[k]) not in ('U','B'):
            raise ValueError("Boundary observations accept only U/B, never S/O/T/I")
        state=V3._state(states[k])
        if record.get('state',state)!=state:
            raise ValueError("Boundary state must match authoritative native state")
        seen.add(k)
        binding=record.get('image_binding')
        if not isinstance(binding,Mapping):
            raise ValueError("Caller image_binding is required")
        image_sha=binding.get('sha256')
        if (not isinstance(image_sha,str) or len(image_sha)!=64
                or any(c not in '0123456789abcdef' for c in image_sha)):
            raise ValueError("image_binding sha256 must be a lowercase SHA256")
        if tuple(binding.get('image_hw',()))!=tuple(image_hw):
            raise ValueError("Boundary image binding must match image_hw")
        source_lines=record.get('lines')
        if not isinstance(source_lines,(list,tuple)) or not 1 <= len(source_lines) <= 2:
            raise ValueError("Each boundary factor needs one or two observed lines")
        lines=[]
        for line in source_lines:
            if not isinstance(line,Mapping):
                raise ValueError("Observed line must be a mapping")
            normal=V3._finite_array(line.get('normal_unit_xy'),(2,),'observed unit normal')
            if not np.isclose(np.linalg.norm(normal),1.,atol=1e-8,rtol=0):
                raise ValueError("Line normal must already have unit length")
            offset=float(line.get('offset_px',np.nan))
            variance=float(line.get('normal_variance_px2',np.nan))
            if not np.isfinite(offset) or not np.isfinite(variance) or variance<=0:
                raise ValueError("Line offset must be finite and normal variance strictly positive")
            interval=V3._finite_array(line.get('support_tangent_interval_px'),(2,),'observed tangent interval')
            margin=float(line.get('support_extrapolation_px',np.nan))
            if interval[0]>=interval[1] or not np.isfinite(margin) or margin<0:
                raise ValueError("Finite increasing tangent support and nonnegative explicit extrapolation required")
            lines.append(dict(normal_unit_xy=normal.copy(),offset_px=offset,normal_variance_px2=variance,
                tangent_unit_xy=np.array([-normal[1],normal[0]]),support_tangent_interval_px=interval.copy(),
                support_extrapolation_px=margin,provenance=deepcopy(line.get('provenance',{}))))
        normalized.append(dict(native_corner=int(k),state=state,lines=lines,image_binding=deepcopy(dict(binding)),
            association_provenance=deepcopy(record.get('association_provenance',{}))))
    if xyz is not None:
        normals=V3.registered_incident_normals(xyz)
        for record in normalized:
            k=record['native_corner'];record.update(object_xyz=np.asarray(xyz[k]).copy(),incident_normals=normals[k].copy())
    return normalized


def _normal_residual_and_jacobian(p,xyz,K,boundary,scale):
    xy,_,jacobian,_=J.projection_and_jacobian(p,xyz,K,scale)
    residuals=[];rows=[]
    for record in boundary:
        k=record['native_corner']
        for line in record['lines']:
            normal=line['normal_unit_xy'];std=np.sqrt(line['normal_variance_px2'])
            residuals.append((normal@xy[k]+line['offset_px'])/std)
            rows.append((normal/std)@jacobian[k])
    return np.asarray(residuals),np.asarray(rows).reshape((-1,6))


def _normal_objective_pair(p,xyz,K,boundary,scale,weight,huber_scale):
    residual,jacobian=_normal_residual_and_jacobian(p,xyz,K,boundary,scale)
    value=0.;gradient=np.zeros(6);cursor=0
    for record in boundary:
        count=len(record['lines']);r=residual[cursor:cursor+count];j=jacobian[cursor:cursor+count]
        denominator=np.sqrt(1+float(r@r)/huber_scale**2)
        value+=weight*2*huber_scale**2*(denominator-1)
        gradient+=(2*weight*r/denominator)@j
        cursor+=count
    return float(value),gradient


def _joint_objective_pair(p,xyz,K,measurements,scale,prior,cfg,boundary,weight):
    value,gradient,components=BASE._objective_pair(p,xyz,K,measurements,scale,prior,cfg)
    extra,extra_gradient=_normal_objective_pair(p,xyz,K,boundary,scale,weight,cfg['huber_scale'])
    return value+extra,gradient+extra_gradient,{**components,'partial_boundary_normal_robust_loss':extra,
        'boundary_factor_weight':weight,'objective_uses_any_S_or_U':any(r['state']=='U' for r in boundary),
        'boundary_factor_uses_S':False,'U_is_partial_observation_not_full_2D_measurement':True}


def _boundary_constraint_pair(p,xyz,K,boundary,image_hw,scale,cfg):
    xy,_,jacobian,_=J.projection_and_jacobian(p,xyz,K,scale)
    R=Rotation.from_rotvec(np.asarray(p[:3]).copy()).as_matrix()
    camera=-R.T@(np.asarray(p[3:])*scale)
    camera_jacobian=np.column_stack([J.skew(camera)@J.so3_right_jacobian(p[:3]),-scale*R.T])
    height,width=image_hw;values=[];rows=[]
    for record in boundary:
        k=record['native_corner'];q=xy[k]
        for line in record['lines']:
            tangent=line['tangent_unit_xy'];low,high=line['support_tangent_interval_px'];margin=line['support_extrapolation_px']
            coordinate=float(tangent@q)
            values.extend([coordinate-low+margin,high+margin-coordinate])
            derivative=tangent@jacobian[k];rows.extend([derivative,-derivative])
        # B already has exactly these original guards; add them only for U.
        if record['state']=='U' and cfg['enforce_visible_geometry']:
            ray=camera-record['object_xyz'];length=float(np.linalg.norm(ray));denominator=max(length,1e-12)
            cosines=record['incident_normals']@ray/denominator;selected=int(np.argmax(cosines))
            cosine=float(cosines[selected]);normal=record['incident_normals'][selected]
            derivative=normal/denominator
            if length>1e-12:derivative-=cosine*ray/length**2
            values.append(cosine+np.sin(np.deg2rad(cfg['geometry_tolerance_deg'])))
            rows.append(derivative@camera_jacobian)
        if record['state']=='U' and cfg['enforce_visible_canvas']:
            values.extend([q[0]/width,q[1]/height,(width-q[0])/width,(height-q[1])/height])
            rows.extend(np.array([[1/width,0],[0,1/height],[-1/width,0],[0,-1/height]])@jacobian[k])
    return np.asarray(values),np.asarray(rows).reshape((-1,6))


def _joint_constraint_pair(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg,boundary,analytic=True):
    values,jacobian=BASE._constraints(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg)
    extra,extra_jacobian=_boundary_constraint_pair(p,xyz,K,boundary,image_hw,scale,cfg)
    pair=(np.r_[values,extra],np.vstack([jacobian,extra_jacobian]))
    return pair if analytic else pair[0]


def _boundary_audit(p,xyz,K,boundary,image_hw,scale,cfg,weight):
    xy,_,_,_=J.projection_and_jacobian(p,xyz,K,scale)
    R,t=V3._pose(p,scale);camera=-R.T@t;result=[]
    for record in boundary:
        k=record['native_corner'];q=xy[k];lines=[];information=np.zeros((2,2))
        for line in record['lines']:
            normal=line['normal_unit_xy'];tangent=line['tangent_unit_xy']
            low,high=line['support_tangent_interval_px'];margin=line['support_extrapolation_px'];s=float(tangent@q)
            slacks=[s-low+margin,high+margin-s];residual=float(normal@q+line['offset_px'])
            information+=weight*np.outer(normal,normal)/line['normal_variance_px2']
            lines.append(dict(normal_unit_xy=normal.tolist(),offset_px=line['offset_px'],normal_variance_px2=line['normal_variance_px2'],
                support_tangent_interval_px=[float(low),float(high)],support_extrapolation_px=margin,
                projected_tangent_coordinate_px=s,support_slack_px=slacks,
                finite_support_pass=min(slacks)>=-cfg['feasibility_tolerance'],normal_residual_px=residual,
                normal_residual_standardized=residual/np.sqrt(line['normal_variance_px2'])))
        geometry=(not cfg['enforce_visible_geometry'] or V3._maximum_incident_cosine(camera,record)
            +np.sin(np.deg2rad(cfg['geometry_tolerance_deg']))>=-cfg['feasibility_tolerance'])
        h,w=image_hw;canvas=not cfg['enforce_visible_canvas'] or min(q[0]/w,q[1]/h,(w-q[0])/w,(h-q[1])/h)>=-cfg['feasibility_tolerance']
        result.append(dict(native_corner=k,state_unchanged=record['state'],projected_xy=q.tolist(),lines=lines,
            visible_geometry_pass=bool(geometry),visible_canvas_pass=bool(canvas),
            feasible=bool(geometry and canvas and all(line['finite_support_pass'] for line in lines)),
            image_binding=record['image_binding'],image_binding_authenticated_by_pure_solver=False,
            association_provenance=record['association_provenance'],native_association_is_caller_hypothesis=True,
            diagonal_normal_information_px_minus2=information.tolist(),normal_information_eigenvalues=np.linalg.eigvalsh(information).tolist(),
            diagonal_normal_model_assumes_independence_not_certified=True,normal_variance_is_calibrated_coverage_probability=False,
            no_measured_2D_corner_created=True))
    return result


def solve_boundary_normal_joint_ot_pnp(xyz: Any, K: Any, states: Sequence, visible: Any,
                                 domains: Any, image_hw: Sequence, coarse_pose: Mapping,
                                 initializations: Any = None, config: Mapping | None = None, *,
                                 boundary_observations: Any = None, boundary_factor_weight: float = 1.0) -> dict:
    """Fit full SE(3) using soft B/I, a coarse prior and every O/T inequality.

    Inputs match v3's native observations/domains. ``coarse_pose`` requires R/t
    in the supplied native basis and optionally physical_registration_Q. Each
    visible record may supply CAL-only ``fit_covariance_px2`` and positive
    ``fit_reliability_weight``; neither changes the observed xy/covariance.
    Otherwise effective Sigma = covariance_scale*observed Sigma + the B/I
    state_BI_total_error_floor_px2. No calibration or GT access occurs here.

    prior_bias6 is a caller-supplied CAL-training median in the same six
    coordinates, scaled by bias_strength. No bias is estimated at runtime.
    With prior_weight=0 the bias has no effect on the objective or pose search.

    Prior covariance ordering is camera rotation tangent radians[3], model
    center x/z and y/z, log center depth. Its default scales are declared
    generic assumptions, exposed for CAL-only selection, not fitted claims.
    accepted means a searched joint-feasible hypothesis, never proven unique.
    Two B/I with O/T is always reported as an ambiguous pose family.
    """
    if isinstance(boundary_factor_weight,(bool,np.bool_)) or not np.isfinite(boundary_factor_weight) or boundary_factor_weight<0:
        raise ValueError("boundary_factor_weight must be finite and nonnegative")
    boundary=_boundary_inputs(boundary_observations,states,image_hw)
    if not boundary or boundary_factor_weight==0:
        return BASE.solve_uncertain_joint_ot_pnp(xyz,K,states,visible,domains,image_hw,coarse_pose,
            initializations=initializations,config=config)
    cfg = _configuration(config)
    source_records = V3._records(visible)
    original = {V3._id(r): r for r in source_records}
    xyz, K, states, measurements, domains, image_hw = V3._inputs(
        xyz, K, states, source_records, domains, image_hw, _helper_config(cfg))
    boundary=_boundary_inputs(boundary_observations,states,image_hw,xyz)
    normals = V3.registered_incident_normals(xyz)
    if any(n.shape != (3, 3) or not np.allclose(n @ n.T, np.eye(3), atol=1e-6, rtol=0) for n in normals):
        raise ValueError("Registered native geometry must be the original eight cuboid vertices")
    for m in measurements:
        record = original[m["native_corner"]]
        state = states[m["native_corner"]]
        observed_covariance = m["covariance_px2"].copy()
        if "fit_covariance_px2" in record:
            covariance = V3._finite_array(record["fit_covariance_px2"], (2, 2), "fit_covariance_px2")
            source = "CALLER_CAL_TOTAL_ERROR_COVARIANCE_OVERRIDE"
        else:
            covariance = (cfg["covariance_scale"] * observed_covariance
                          + np.asarray(cfg["state_BI_total_error_floor_px2"][state], dtype=float))
            source = "OBSERVED_CONDITIONAL_COVARIANCE_PLUS_CAL_STATE_FLOOR"
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
            raise ValueError("fit covariance must be symmetric")
        try:
            whitening = np.linalg.inv(np.linalg.cholesky(covariance))
        except np.linalg.LinAlgError as exc:
            raise ValueError("fit covariance must be positive definite") from exc
        weight = float(record.get("fit_reliability_weight", cfg["state_loss_weights"][state]))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("fit_reliability_weight must be positive")
        m.update(observed_covariance_px2=observed_covariance, observed_whitening=m["whitening"].copy(),
                 covariance_px2=covariance.copy(), whitening=whitening,
                 fit_reliability_weight=weight, fit_covariance_source=source)
    coarse_R = V3._finite_array(coarse_pose["R"], (3, 3), "coarse R")
    coarse_t = V3._finite_array(coarse_pose["t"], (3,), "coarse t")
    Q = V3._finite_array(coarse_pose.get("physical_registration_Q", np.eye(3)), (3, 3), "physical registration Q")
    if not V3._valid_rotation(coarse_R) or not V3._valid_rotation(Q):
        raise ValueError("Coarse rotation and physical registration Q must be proper rotations")
    if "R_physical" in coarse_pose and not np.allclose(coarse_R @ Q, coarse_pose["R_physical"], atol=1e-7, rtol=0):
        raise ValueError("Coarse physical rotation must equal R_native @ Q")
    coarse_center = coarse_R @ xyz.mean(axis=0) + coarse_t
    if coarse_center[2] <= 0:
        raise ValueError("The initial registered model center must have positive depth")
    covariance6 = cfg["prior_covariance6"]
    if covariance6 is None:
        deviations = np.r_[np.repeat(np.deg2rad(cfg["prior_rotation_sigma_deg"]), 3),
                           np.repeat(cfg["prior_ray_sigma"], 2), cfg["prior_log_depth_sigma"]]
        covariance6 = np.diag(deviations**2)
    covariance6 = V3._finite_array(covariance6, (6, 6), "prior_covariance6")
    if not np.allclose(covariance6, covariance6.T, atol=1e-12, rtol=0):
        raise ValueError("Prior covariance must be symmetric")
    try:
        prior_whitening = np.linalg.inv(np.linalg.cholesky(covariance6))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Prior covariance must be positive definite") from exc
    prior = {"R": coarse_R, "center": coarse_center, "whitening": prior_whitening}
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    observations = [{"native_corner": m["native_corner"], "state": states[m["native_corner"]],
                     "xy": m["xy"].tolist(), "covariance_px2": m["observed_covariance_px2"].tolist(),
                     "protection_px": m["protection_px"], "provenance": m["provenance"],
                     "fit_covariance_px2": m["covariance_px2"].tolist(),
                     "fit_reliability_weight": m["fit_reliability_weight"]} for m in measurements]
    coarse_q, _ = V3.project_native(xyz, coarse_R, coarse_t, K)
    result = {"schema": "statewise_boundary_normal_solver_v4_4", "accepted": False, "status": "unresolved",
              "R": None, "t": None, "R_physical": None, "states": list(states),
              "projected_native": None, "rigid_projection8": None, "visible_observations": observations,
              "reported_BI_coordinates_preserved": True, "reported_native": [None] * 8,
              "fallback_reason": None, "BI_prior_seed": None, "corrected_BI_seed": None,
              "candidates": [], "feasible_pose_samples": [], "pose_family_candidates": [],
              "joint_constraint_audit": [], "rank": None, "ambiguity": None,
              "OT_constraints_satisfied": False, "OT_feasible_hypothesis_available": False,
              "OT_unique_solution_claimed": False, "OT_solved": False,
              "prior_audit": {"parameter_order": ["camera_rotation_x_rad", "camera_rotation_y_rad",
                               "camera_rotation_z_rad", "center_x_over_z", "center_y_over_z", "log_center_depth"],
                              "covariance6": covariance6.tolist(), "prior_weight": cfg["prior_weight"],
                              "enabled": bool(cfg["prior_weight"] > 0),
                              "prior_bias6": list(cfg["prior_bias6"]), "bias_strength": cfg["bias_strength"],
                              "effective_prior_bias6": (np.asarray(cfg["prior_bias6"]) * cfg["bias_strength"]).tolist(),
                              "bias_subtracted_before_whitening": True,
                              "bias_active_in_loss": bool(cfg["prior_weight"] > 0 and cfg["bias_strength"] > 0
                                                          and np.any(np.asarray(cfg["prior_bias6"]) != 0)),
                              "coarse_R": coarse_R.tolist(), "coarse_t": coarse_t.tolist(),
                              "physical_registration_Q": Q.tolist(), "default_scales_are_calibration_claims": False},
              "provenance": {"GT_inputs": False, "image_reads": 0, "S_image_terms": 0, "S_region_terms": 0,
                             "S_or_U_used_for_scale_or_initialization": False, "all_OT_simultaneous": True,
                             "OT_used_as_precise_correspondences": False,
                             "initial_pose_prior": bool(cfg["prior_weight"] > 0),
                             "coarse_pose_used_for_initialization": True,
                             "B_I_measurements_and_covariance_frozen": True,
                             "B_I_absolute_covariance_residual_constraints": False,
                             "seed_projection_protection_enabled": False,
                             "final_BI_model_geometry_enforced": cfg["enforce_visible_geometry"],
                             "final_BI_closed_canvas_enforced": cfg["enforce_visible_canvas"],
                             "rotation_fixed_for_three_points": False, "native_order_and_basis_preserved": True}}
    result["boundary_factor_audit"]={"factor_weight":float(boundary_factor_weight),"active_native_ids":[r["native_corner"] for r in boundary],
        "states_preserved":True,"original_BI_seed_fit_preserved":True,"finite_support_changes_positive_weight_feasible_set":True,
        "line_association_is_caller_hypothesis":True,"no_new_covariance_calibration":True,"no_fake_2D_measurement":True}
    result["boundary_observation_audit"]=[]
    result["provenance"].update(partial_boundary_normal_terms=len(boundary),S_boundary_terms=0,
        U_partial_lines_allowed_without_state_change=True,zero_or_empty_exact_parent_delegation=True,
        boundary_support_guards_only_in_final_positive_weight_joint_fit=True)
    result["fallback_pose"] = {"R": coarse_R.tolist(), "t": coarse_t.tolist(),
                               "projected_native": coarse_q.tolist(), "source": "INITIAL_REGISTERED_COARSE_PNP",
                               "valid_as_joint_OT_solution": False, "S_projection_only": True}
    for m in measurements:
        result["reported_native"][m["native_corner"]] = m["xy"].tolist()
    for native_id, state in enumerate(states):
        if state == "S":
            result["reported_native"][native_id] = coarse_q[native_id].tolist()
    if len(measurements) < 2 or (len(measurements) == 2 and not domains):
        result["fallback_reason"] = "two_BI_requires_OT_ranges" if len(measurements) == 2 else "fewer_than_two_BI"
        return result
    pose_family = len(measurements) == 2
    supplied = [{"name": "REGISTERED_COARSE_PRIOR", "R": coarse_R, "t": coarse_t},
                *list(initializations or [])]
    starts, search = V3._initial_poses(xyz, K, measurements, supplied, cfg)
    result["search"] = {**search, "supplied_initializations_are_priors": False,
                        "registered_coarse_pose_is_prior": bool(cfg["prior_weight"] > 0), "analytic_jacobian_used": cfg["analytic_jacobian"],
                        "joint_method": "SLSQP_full_SE3_soft_BI_partial_boundary_normals_coarse_prior_all_OT_inequalities"}
    seed_objective_pair = V3._cached_value_jacobian(lambda p: _objective_pair(p, xyz, K, measurements, scale, prior, cfg))
    objective_pair = V3._cached_value_jacobian(lambda p: _joint_objective_pair(p, xyz, K, measurements, scale, prior, cfg,boundary,boundary_factor_weight))
    objective = lambda p: objective_pair(p)[0]
    gradient = lambda p: objective_pair(p)[1].copy()
    analytic = cfg["analytic_jacobian"]

    def fit(start, subset_domains, sides, limit, use_boundary=True):
        evaluator=(lambda p: _joint_constraint_pair(p,xyz,K,measurements,subset_domains,sides,image_hw,scale,cfg,boundary)) if use_boundary else (lambda p: _constraints(p,xyz,K,measurements,subset_domains,sides,image_hw,scale,cfg))
        pair = V3._cached_value_jacobian(evaluator)
        fit_objective_pair=objective_pair if use_boundary else seed_objective_pair
        fit_objective=lambda p: fit_objective_pair(p)[0]
        fit_gradient=lambda p: fit_objective_pair(p)[1].copy()
        inequality = {"type": "ineq", "fun": lambda p: pair(p)[0].copy()}
        if analytic:
            inequality["jac"] = lambda p: pair(p)[1].copy()
        answer = minimize(fit_objective, V3._pose_vector(start["R"], start["t"], scale),
                          method="SLSQP", jac=fit_gradient if analytic else None, constraints=[inequality],
                          options={"maxiter": int(limit), "ftol": cfg["optimizer_ftol"], "disp": False})
        R, t = V3._pose(answer.x, scale)
        q, depth = V3.project_native(xyz, R, t, K)
        values = pair(answer.x)[0]
        feasible = bool(np.isfinite(q).all() and np.isfinite(values).all()
                        and np.min(values) >= -cfg["feasibility_tolerance"]
                        and V3._valid_rotation(R) and np.min(depth) >= cfg["positive_depth_m"] - 1e-10)
        return answer, {"R": R, "t": t, "q": q, "p": answer.x, "name": start["name"],
                        "objective": fit_objective(answer.x), "feasible": feasible, "sides": sides,
                        "minimum_constraint_slack": float(np.min(values))}

    seeds = []
    for start in starts:
        try:
            _, seed = fit(start, [], (), cfg["max_seed_nfev"], use_boundary=False)
            if seed["feasible"] and not any(V3._same_pose((seed["R"], seed["t"]), (s["R"], s["t"]), scale) for s in seeds):
                seeds.append(seed)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    seeds.sort(key=lambda s: (s["objective"], s["name"]))
    if seeds:
        seed = seeds[0]
        seed_rank = V3._rank(J.visible_residual_and_jacobian(seed["p"], xyz, K, measurements, scale)[1])
        underconstrained_seed = pose_family or seed_rank["rank"] < 6
        seed_record = {"R": seed["R"].tolist(), "t": seed["t"].tolist(), "projected_native": seed["q"].tolist(),
                       "objective": seed["objective"], "name": seed["name"],
                       "fitted_only_corrected_BI": bool(cfg["prior_weight"] == 0),
                       "fitted_BI_and_coarse_prior": bool(cfg["prior_weight"] > 0),
                       "coarse_prior_weight": cfg["prior_weight"], "visible_support_guards_pass": True,
                       "contains_underconstrained_pose_family": underconstrained_seed,
                       "admissible_as_joint_fallback": False, "visible_rank": seed_rank,
                       "visible_measurement_audit": _visible_audit(seed["q"], measurements, -seed["R"].T @ seed["t"], image_hw, cfg)}
        result["BI_prior_seed"] = seed_record
        result["corrected_BI_seed"] = seed_record
        if not underconstrained_seed:
            seed_source = ("SOFT_BI_AND_REGISTERED_COARSE_PRIOR_SEED" if cfg["prior_weight"] > 0
                           else "SOFT_BI_SEED_WITH_REGISTERED_COARSE_INITIALIZATION")
            result["fallback_pose"] = {**seed_record, "source": seed_source,
                                       "valid_as_joint_OT_solution": False, "S_projection_only": True}
            for i, state in enumerate(states):
                if state == "S":
                    result["reported_native"][i] = seed["q"][i].tolist()
    choices = [(list(d["outside_canvas_side"]) if isinstance(d["outside_canvas_side"], tuple)
                else [d["outside_canvas_side"]] if d["outside_canvas_side"] else list(V3._SIDES))
               if d["state"] == "T" else [None] for d in domains]
    branch_count = int(np.prod([len(c) for c in choices], dtype=int))
    result["search"].update(T_union_branches_required=branch_count, BI_prior_local_minima=len(seeds))
    if branch_count > cfg["maximum_union_branches"]:
        result["fallback_reason"] = "T_union_branch_limit_exceeded_no_partial_joint_claim"
        return result
    branches = list(product(*choices))
    joint_starts = [{"name": "BI_PRIOR_SEED_" + s["name"], "R": s["R"], "t": s["t"]} for s in seeds]
    for start in starts:
        if not any(V3._same_pose((start["R"], start["t"]), (s["R"], s["t"]), scale) for s in joint_starts):
            joint_starts.append(start)
    result["search"].update(T_union_branches_searched=len(branches), joint_initializations_per_branch=len(joint_starts))
    solutions = []
    for branch_index, sides in enumerate(branches):
        for start in joint_starts:
            try:
                answer, solved = fit(start, domains, sides, cfg["max_joint_iterations"])
                record = {"initialization": start["name"], "branch": branch_index,
                          "optimizer_success": bool(answer.success), "optimizer_message": str(answer.message),
                          "iterations": int(answer.nit), "joint_feasible": solved["feasible"],
                          "objective": solved["objective"], "objective_components": objective_pair(answer.x)[2],
                          "minimum_constraint_slack": solved["minimum_constraint_slack"],
                          "visible_measurement_audit": _visible_audit(solved["q"], measurements, -solved["R"].T @ solved["t"], image_hw, cfg),
                          "boundary_audit": _boundary_audit(answer.x,xyz,K,boundary,image_hw,scale,cfg,boundary_factor_weight),
                          "domain_audit": V3._domain_audit(solved["q"], domains, sides, image_hw, _helper_config(cfg))}
                result["candidates"].append(record)
                if solved["feasible"]:
                    record.update(R=solved["R"].tolist(), t=solved["t"].tolist())
                    if not any(V3._same_pose((solved["R"], solved["t"]), (s["R"], s["t"]), scale) for s in solutions):
                        solutions.append({**solved, "record": record})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                result["candidates"].append({"initialization": start["name"], "branch": branch_index,
                                             "joint_feasible": False, "error": str(exc)})
    result["search"].update(feasible_distinct_local_poses=len(solutions), global_feasibility_proved=False)
    if not solutions:
        result.update(status="no_joint_feasible_pose_found", fallback_reason="all_searched_joint_branches_infeasible_or_not_converged")
        result["fallback_diagnostics"] = {"no_global_infeasibility_proof": True, "hard_BI_residual_caps_used": False,
                                           "OT_output_unresolved": True, "fallback_source": result["fallback_pose"]["source"]}
        return result
    solutions.sort(key=lambda s: (s["objective"], s["name"]))
    chosen = solutions[0]
    R, t, q, p = chosen["R"], chosen["t"], chosen["q"], chosen["p"]
    alternatives = []
    for other in solutions[1:]:
        rotation = float(np.rad2deg(Rotation.from_matrix(other["R"] @ R.T).magnitude()))
        translation = float(np.linalg.norm(other["t"] - t) / scale)
        gap = other["objective"] - chosen["objective"]
        if (rotation > cfg["ambiguity_rotation_deg"] or translation > cfg["ambiguity_translation_fraction"]) and gap <= cfg["ambiguity_objective_gap"]:
            alternatives.append({"R": other["R"].tolist(), "t": other["t"].tolist(),
                                 "objective_gap": gap, "rotation_difference_deg": rotation,
                                 "translation_difference_fraction": translation})
    visible_jacobian = J.visible_residual_and_jacobian(p, xyz, K, measurements, scale)[1]
    values, guards_jacobian = _joint_constraint_pair(p, xyz, K, measurements, domains, chosen["sides"], image_hw, scale, cfg,boundary)
    active = np.flatnonzero(values <= 1e-5)
    boundary_jacobian=_normal_residual_and_jacobian(p,xyz,K,boundary,scale)[1]*np.sqrt(boundary_factor_weight)
    partial_image_jacobian=np.vstack([visible_jacobian,boundary_jacobian])
    joint_jacobian = np.vstack([partial_image_jacobian, guards_jacobian[active]])
    visible_rank, joint_rank = V3._rank(visible_jacobian), V3._rank(joint_jacobian)
    ambiguous = bool(pose_family or alternatives or joint_rank["rank"] < 6)
    result.update(accepted=True, status="joint_feasible_pose_family" if pose_family else "joint_feasible_ambiguous" if ambiguous else "joint_feasible",
                  R=R.tolist(), t=t.tolist(), R_physical=(R @ Q).tolist(), projected_native=q.tolist(), rigid_projection8=q.tolist(),
                  objective=chosen["objective"], objective_components=objective_pair(p)[2],
                  visible_protection_audit=chosen["record"]["visible_measurement_audit"],
                  boundary_observation_audit=chosen["record"]["boundary_audit"],
                  joint_constraint_audit=chosen["record"]["domain_audit"], minimum_joint_slack=chosen["minimum_constraint_slack"],
                  rank={"visible": visible_rank, "BI_and_partial_boundary":V3._rank(partial_image_jacobian), "visible_and_active_joint": joint_rank,
                        "active_constraint_count": len(active), "prior_not_counted_as_image_information": True},
                  ambiguity={"detected": ambiguous, "alternatives": alternatives, "pose_family_from_two_BI": pose_family,
                             "visible_data_underdetermined": visible_rank["rank"] < 6,
                             "local_underdetermined": joint_rank["rank"] < 6, "global_uniqueness_proved": False,
                             "prior_regularized_MAP_is_not_observation_uniqueness_proof": True},
                  OT_constraints_satisfied=True, OT_feasible_hypothesis_available=bool(domains),
                  OT_solved=bool(domains) and not ambiguous)
    result["feasible_pose_samples"] = [{"R": s["R"].tolist(), "t": s["t"].tolist(),
                                       "projected_native": s["q"].tolist(), "objective": s["objective"],
                                       "all_OT_individual_constraints_pass": True,
                                       "initialization": s["name"]} for s in solutions]
    result["pose_family_candidates"] = result["feasible_pose_samples"] if ambiguous else []
    for i, state in enumerate(states):
        if state in {"O", "T", "S"}:
            result["reported_native"][i] = q[i].tolist()
    result["output_corner_sources"] = ["IMMUTABLE_BI_OBSERVATION" if s in {"B", "I"}
        else "S_MODEL_PROJECTION_ONLY" if s == "S" else "UNRESOLVED" if s == "U"
        else "AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION" if ambiguous else "JOINT_FEASIBLE_REPRESENTATIVE_PROJECTION" for s in states]
    result["mixed_readout_is_rigid_pose"] = False
    camera = -R.T @ t
    self_ids = [i for i, normal in enumerate(normals) if np.max(normal @ (camera - xyz[i]) / np.linalg.norm(camera - xyz[i]))
                < -np.sin(np.deg2rad(cfg["geometry_tolerance_deg"]))]
    input_s = [i for i, state in enumerate(states) if state == "S"]
    result["final_model_visibility_audit"] = {"initial_S_ids": input_s, "final_model_self_ids": self_ids,
                                              "S_set_changed": set(input_s) != set(self_ids),
                                              "S_state_is_initial_geometry_conditional": True}
    return result


# Same old positional interface; extra factors remain keyword-only and opt-in.
solve_uncertain_joint_ot_pnp=solve_boundary_normal_joint_ot_pnp
