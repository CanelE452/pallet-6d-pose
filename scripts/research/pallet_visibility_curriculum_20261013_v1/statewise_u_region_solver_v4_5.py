"""Pure opt-in simultaneous uncertain-U ellipse constraints.

The immutable bias solver's BI robust likelihood, prior, OT constraints, Q,
geometry scale and BI-only seed fit stay unchanged. U contributes only a final
range inequality, never a point cost/correspondence, visibility/canvas guard,
seed or scale input. States and reported BI observations stay fixed; U readout
remains unresolved rather than a manufactured image KP. S is gated before any
region-center field is accessed and appears only as rigid pose projection.

An ellipse matrix is a geometric squared-radius shape matrix, not a calibrated
probability covariance. Region production/calibration belongs to the caller.
This module performs no image, annotation, model, network or file IO.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import statewise_uncertain_solver_v4_bias as OLD
from . import joint_ot_pnp as V3
from . import joint_ot_jacobian_v2 as J

_configuration=OLD._configuration
_helper_config=OLD._helper_config
_constraints=OLD._constraints
_objective_pair=OLD._objective_pair
_visible_audit=OLD._visible_audit


def _u_region_inputs(records,states):
    if records is None:
        return []
    if isinstance(records,Mapping) and ('native_corner' in records or 'native_id' in records):
        supplied=[records]
    elif isinstance(records,(list,tuple)):
        supplied=records
    else:
        raise ValueError('u_regions must be a record or a sequence of native U records')
    result=[]
    for index,record in enumerate(supplied):
        if not isinstance(record,Mapping):raise ValueError('U region records must be mappings')
        k=V3._id(record)
        # Reject S (and every nonU state) before inspecting any coordinate.
        if len(states)!=8 or V3._state(states[k])!='U':
            raise ValueError('Only final U may carry u_regions; S/O/T/B/I are excluded before center access')
        if V3._state(record.get('state','U'))!='U':
            raise ValueError('u_region state must be U')
        if record.get('shape','ellipse')!='ellipse':
            raise ValueError('u_regions currently support finite ellipses only')
        center=V3._finite_array(record.get('center_xy'),(2,),'U ellipse center').copy()
        if 'ellipse_matrix_px2' in record:
            if 'half_axes_px' in record or 'angle_deg' in record:
                raise ValueError('Supply ellipse_matrix_px2 or half_axes_px/angle_deg, not both')
            matrix=V3._finite_array(record['ellipse_matrix_px2'],(2,2),'U geometric ellipse matrix').copy()
            if not np.allclose(matrix,matrix.T,atol=1e-12,rtol=0):
                raise ValueError('U ellipse matrix must be symmetric positive definite')
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError('U ellipse matrix must be positive definite') from error
            values,basis=np.linalg.eigh(matrix);order=np.argsort(values)[::-1]
            axes=np.sqrt(values[order]);first=basis[:,order[0]]
            angle=float(np.rad2deg(np.arctan2(first[1],first[0])))
        else:
            axes=V3._finite_array(record.get('half_axes_px'),(2,),'U ellipse half axes').copy()
            if np.any(axes<=0):raise ValueError('U ellipse half axes must be strictly positive')
            angle=float(record.get('angle_deg',0.))
            if not np.isfinite(angle):raise ValueError('U ellipse angle must be finite degrees')
            phi=np.deg2rad(angle);basis=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
            matrix=basis@np.diag(axes**2)@basis.T
        result.append(dict(native_corner=k,state='U',shape='ellipse',u_region_index=index,
            center_xy=center,half_axes_px=axes,angle_deg=angle,ellipse_matrix_px2=matrix,
            precision=np.linalg.inv(matrix),provenance=deepcopy(record.get('provenance',{}))))
    return result


def _u_constraint_pair(p,xyz,K,regions,scale):
    xy,_,jacobian,_=J.projection_and_jacobian(p,xyz,K,scale)
    values=[];rows=[]
    for region in regions:
        k=region['native_corner'];delta=xy[k]-region['center_xy'];precision=region['precision']
        values.append(float(1-delta@precision@delta))
        rows.append((-2*precision@delta)@jacobian[k])
    return np.asarray(values),np.asarray(rows).reshape((-1,6))


def _joint_constraint_pair(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg,regions,analytic=True):
    values,jacobian=OLD._constraints(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg)
    extra,extra_jacobian=_u_constraint_pair(p,xyz,K,regions,scale)
    pair=(np.r_[values,extra],np.vstack([jacobian,extra_jacobian]))
    return pair if analytic else pair[0]


def _u_audit(q,regions,tolerance):
    result=[]
    for region in regions:
        k=region['native_corner'];delta=np.asarray(q[k])-region['center_xy'];value=float(delta@region['precision']@delta)
        result.append(dict(native_corner=k,state='U',u_region_index=region['u_region_index'],shape='ellipse',
            center_xy=region['center_xy'].tolist(),half_axes_px=region['half_axes_px'].tolist(),angle_deg=region['angle_deg'],
            ellipse_matrix_px2=region['ellipse_matrix_px2'].tolist(),projected_xy=np.asarray(q[k]).tolist(),
            normalized_ellipse_quadratic=value,slack=1-value,feasible=bool(value<=1+tolerance),
            U_visibility_or_canvas_guard=False,U_point_cost=False,U_used_as_PnP_correspondence=False,
            shape_matrix_is_calibrated_probability_covariance=False,provenance=deepcopy(region['provenance'])))
    return result


def _annotate_u_result(result,regions,parent,xyz,K,scale,cfg,*,reused_parent=False):
    """Attach U evidence while keeping OT flags semantically separate."""
    result['schema']='statewise_u_region_solver_v4_5'
    result['parent_baseline_solution']=parent
    result['U_region_search_audit']=dict(regions=len(regions),active_native_ids=sorted({r['native_corner'] for r in regions}),
        parent_winner_reused=reused_parent,parent_winner_reuse_is_not_global_optimality_proof=True,
        original_objective_and_BI_prior_seed_preserved=True,U_loss_terms=0,U_measurements=0,
        U_visibility_guards=0,U_canvas_guards=0,U_used_for_seeding_or_scale=False,
        every_retained_U_region_simultaneous=True,U_ellipse_is_caller_declared_range_not_probability=True)
    result['provenance'].update(U_region_constraints=len(regions),U_point_terms=0,U_visible_geometry_terms=0,
        U_canvas_terms=0,all_OT_and_U_simultaneous=True,U_used_as_precise_correspondence=False)
    result['U_region_constraint_audit']=[]
    result['U_constraints_satisfied']=False
    result['fallback_does_not_solve_U']=not result['accepted']
    if result['accepted']:
        q=np.asarray(result['projected_native'])
        result['U_region_constraint_audit']=_u_audit(q,regions,cfg['feasibility_tolerance'])
        result['U_constraints_satisfied']=all(r['feasible'] for r in result['U_region_constraint_audit'])
        if not result['U_constraints_satisfied']:raise RuntimeError('Accepted pose violates an individual U range')
        result['minimum_joint_slack']=min(result['minimum_joint_slack'],*[r['slack'] for r in result['U_region_constraint_audit']])
    return result


def _reuse_parent_winner(parent,regions,xyz,K,scale,cfg):
    """Filter the finite parent search; never claim an exhaustive global optimum."""
    if not parent['accepted']:
        return None
    if not all(r['feasible'] for r in _u_audit(parent['projected_native'],regions,cfg['feasibility_tolerance'])):
        return None
    result=deepcopy(parent)
    def feasible(R,t):
        q,_=V3.project_native(xyz,np.asarray(R),np.asarray(t),K)
        return all(r['feasible'] for r in _u_audit(q,regions,cfg['feasibility_tolerance']))
    for sample in result['feasible_pose_samples']:
        sample['all_U_individual_constraints_pass']=feasible(sample['R'],sample['t'])
    result['feasible_pose_samples']=[s for s in result['feasible_pose_samples'] if s['all_U_individual_constraints_pass']]
    result['pose_family_candidates']=result['feasible_pose_samples'] if result['ambiguity']['detected'] else []
    result['ambiguity']['alternatives']=[s for s in result['ambiguity']['alternatives'] if feasible(s['R'],s['t'])]
    result['ambiguity']['U_filtered_finite_samples_do_not_prove_uniqueness']=True
    for candidate in result['candidates']:
        candidate['parent_joint_feasible']=candidate['joint_feasible']
        if 'R' in candidate:
            q,_=V3.project_native(xyz,np.asarray(candidate['R']),np.asarray(candidate['t']),K)
            candidate['U_region_audit']=_u_audit(q,regions,cfg['feasibility_tolerance'])
            candidate['all_U_individual_constraints_pass']=all(r['feasible'] for r in candidate['U_region_audit'])
            candidate['joint_feasible']=bool(candidate['joint_feasible'] and candidate['all_U_individual_constraints_pass'])
            candidate['parent_minimum_constraint_slack']=candidate['minimum_constraint_slack']
            candidate['minimum_constraint_slack']=min(candidate['minimum_constraint_slack'],*[r['slack'] for r in candidate['U_region_audit']])
    result['search']['parent_feasible_distinct_local_poses']=result['search']['feasible_distinct_local_poses']
    result['search']['feasible_distinct_local_poses']=len(result['feasible_pose_samples'])
    result['search'].update(U_added_optimization_calls=0,U_parent_winner_reused=True,
        U_feasible_distinct_parent_samples=len(result['feasible_pose_samples']),global_feasibility_proved=False)
    result['rank']['parent_rank_retained_U_not_recomputed']=True
    return _annotate_u_result(result,regions,parent,xyz,K,scale,cfg,reused_parent=True)


def reuse_parent_solution_if_u_feasible(xyz,K,states,parent_solution,u_regions,config=None):
    """Reuse a caller-bound finite parent winner, or return None to request fit.

The caller MUST verify the complete unchanged solver-math specification before
passing a cached result: same BI coordinates/fit Sigma/reliability, domains,
coarse prior/bias/Q, starts and config. This pure helper cannot infer that cache
binding from a pose. It verifies states, R/t, projection, depth and every U
ellipse. Reuse remains a finite-search hypothesis, never global optimality.
Empty U returns the exact parent object unchanged, with no new metadata/rows.
"""
    regions=_u_region_inputs(u_regions,states)
    if not regions:
        return parent_solution
    cfg=_configuration(config)
    xyz=V3._finite_array(xyz,(8,3),'xyz_native')
    K=V3._finite_array(K,(3,3),'K')
    if list(map(V3._state,states))!=list(parent_solution['states']):
        raise ValueError('Cached parent states must match unchanged native states')
    if not parent_solution['accepted']:
        return None
    R=V3._finite_array(parent_solution['R'],(3,3),'cached parent R')
    t=V3._finite_array(parent_solution['t'],(3,),'cached parent t')
    if not V3._valid_rotation(R):raise ValueError('Cached parent rotation must be proper')
    q,depth=V3.project_native(xyz,R,t,K)
    if (not np.isfinite(q).all() or np.min(depth)<cfg['positive_depth_m']-1e-10
            or not np.allclose(q,parent_solution['projected_native'],atol=1e-7,rtol=0)):
        raise ValueError('Cached parent projection or positive-depth binding differs')
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    return _reuse_parent_winner(parent_solution,regions,xyz,K,scale,cfg)


def solve_uncertain_u_region_joint_ot_pnp(xyz: Any, K: Any, states: Sequence, visible: Any,
                                 domains: Any, image_hw: Sequence, coarse_pose: Mapping,
                                 initializations: Any = None, config: Mapping | None = None, *,
                                 u_regions: Any = None) -> dict:
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
    regions=_u_region_inputs(u_regions,states)
    if not regions:
        return OLD.solve_uncertain_joint_ot_pnp(xyz,K,states,visible,domains,image_hw,coarse_pose,
            initializations=initializations,config=config)
    cfg = _configuration(config)
    source_records = V3._records(visible)
    domain_records = V3._records(domains)
    supplied_initializations=list(initializations or [])
    parent=OLD.solve_uncertain_joint_ot_pnp(xyz,K,states,source_records,domain_records,image_hw,coarse_pose,
        initializations=supplied_initializations,config=config)
    original = {V3._id(r): r for r in source_records}
    xyz, K, states, measurements, domains, image_hw = V3._inputs(
        xyz, K, states, source_records, domain_records, image_hw, _helper_config(cfg))
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
    reused=reuse_parent_solution_if_u_feasible(xyz,K,states,parent,u_regions,config=cfg)
    if reused is not None:
        return reused
    observations = [{"native_corner": m["native_corner"], "state": states[m["native_corner"]],
                     "xy": m["xy"].tolist(), "covariance_px2": m["observed_covariance_px2"].tolist(),
                     "protection_px": m["protection_px"], "provenance": m["provenance"],
                     "fit_covariance_px2": m["covariance_px2"].tolist(),
                     "fit_reliability_weight": m["fit_reliability_weight"]} for m in measurements]
    coarse_q, _ = V3.project_native(xyz, coarse_R, coarse_t, K)
    result = {"schema": "statewise_u_region_solver_v4_5", "accepted": False, "status": "unresolved",
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
    result["fallback_pose"] = {"R": coarse_R.tolist(), "t": coarse_t.tolist(),
                               "projected_native": coarse_q.tolist(), "source": "INITIAL_REGISTERED_COARSE_PNP",
                               "valid_as_joint_OT_solution": False, "S_projection_only": True}
    for m in measurements:
        result["reported_native"][m["native_corner"]] = m["xy"].tolist()
    for native_id, state in enumerate(states):
        if state == "S":
            result["reported_native"][native_id] = coarse_q[native_id].tolist()
    if len(measurements) < 2 or (len(measurements) == 2 and not domains and not regions):
        result["fallback_reason"] = "two_BI_requires_OT_ranges" if len(measurements) == 2 else "fewer_than_two_BI"
        return _annotate_u_result(result,regions,parent,xyz,K,scale,cfg)
    pose_family = len(measurements) == 2
    supplied = [{"name": "REGISTERED_COARSE_PRIOR", "R": coarse_R, "t": coarse_t},
                *supplied_initializations]
    starts, search = V3._initial_poses(xyz, K, measurements, supplied, cfg)
    result["search"] = {**search, "supplied_initializations_are_priors": False,
                        "registered_coarse_pose_is_prior": bool(cfg["prior_weight"] > 0), "analytic_jacobian_used": cfg["analytic_jacobian"],
                        "joint_method": "SLSQP_full_SE3_same_soft_BI_coarse_prior_all_OT_U_inequalities"}
    objective_pair = V3._cached_value_jacobian(lambda p: _objective_pair(p, xyz, K, measurements, scale, prior, cfg))
    objective = lambda p: objective_pair(p)[0]
    gradient = lambda p: objective_pair(p)[1].copy()
    analytic = cfg["analytic_jacobian"]

    def fit(start, subset_domains, sides, limit, use_U=True):
        evaluator=(lambda p: _joint_constraint_pair(p,xyz,K,measurements,subset_domains,sides,image_hw,scale,cfg,regions)) if use_U else (lambda p: _constraints(p,xyz,K,measurements,subset_domains,sides,image_hw,scale,cfg))
        pair = V3._cached_value_jacobian(evaluator)
        inequality = {"type": "ineq", "fun": lambda p: pair(p)[0].copy()}
        if analytic:
            inequality["jac"] = lambda p: pair(p)[1].copy()
        answer = minimize(objective, V3._pose_vector(start["R"], start["t"], scale),
                          method="SLSQP", jac=gradient if analytic else None, constraints=[inequality],
                          options={"maxiter": int(limit), "ftol": cfg["optimizer_ftol"], "disp": False})
        R, t = V3._pose(answer.x, scale)
        q, depth = V3.project_native(xyz, R, t, K)
        values = pair(answer.x)[0]
        feasible = bool(np.isfinite(q).all() and np.isfinite(values).all()
                        and np.min(values) >= -cfg["feasibility_tolerance"]
                        and V3._valid_rotation(R) and np.min(depth) >= cfg["positive_depth_m"] - 1e-10)
        return answer, {"R": R, "t": t, "q": q, "p": answer.x, "name": start["name"],
                        "objective": objective(answer.x), "feasible": feasible, "sides": sides,
                        "minimum_constraint_slack": float(np.min(values))}

    seeds = []
    for start in starts:
        try:
            _, seed = fit(start, [], (), cfg["max_seed_nfev"], use_U=False)
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
        return _annotate_u_result(result,regions,parent,xyz,K,scale,cfg)
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
                          "U_region_audit": _u_audit(solved["q"],regions,cfg["feasibility_tolerance"]),
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
        return _annotate_u_result(result,regions,parent,xyz,K,scale,cfg)
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
    values, guards_jacobian = _joint_constraint_pair(p, xyz, K, measurements, domains, chosen["sides"], image_hw, scale, cfg,regions)
    active = np.flatnonzero(values <= 1e-5)
    joint_jacobian = np.vstack([visible_jacobian, guards_jacobian[active]])
    visible_rank, joint_rank = V3._rank(visible_jacobian), V3._rank(joint_jacobian)
    ambiguous = bool(pose_family or alternatives or joint_rank["rank"] < 6)
    result.update(accepted=True, status="joint_feasible_pose_family" if pose_family else "joint_feasible_ambiguous" if ambiguous else "joint_feasible",
                  R=R.tolist(), t=t.tolist(), R_physical=(R @ Q).tolist(), projected_native=q.tolist(), rigid_projection8=q.tolist(),
                  objective=chosen["objective"], objective_components=objective_pair(p)[2],
                  visible_protection_audit=chosen["record"]["visible_measurement_audit"],
                  joint_constraint_audit=chosen["record"]["domain_audit"], minimum_joint_slack=chosen["minimum_constraint_slack"],
                  rank={"visible": visible_rank, "visible_and_active_joint": joint_rank,
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
                                       "all_U_individual_constraints_pass": True,
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
    return _annotate_u_result(result,regions,parent,xyz,K,scale,cfg)

