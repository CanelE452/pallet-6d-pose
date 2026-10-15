"""Typed partial B/I normals with simultaneous immutable O/T and U ranges.

The archived complete B/I 2D likelihood is explicitly revoked for each partial
B/I, not
counted alongside the correlated RGB line. The line constrains its normal
position softly; its caller-declared finite corner tangent interval adds hard
rows. Original coordinates/covariances remain reference/seed/audit data, not a
new full 2D observation. Unaffected BI, the old full-BI seed, coarse prior/Q,
all O/T domains and every U ellipse remain unchanged. U is range-only: no U
line, objective, canvas or visibility term. S is rejected before line fields.

The parser, normal/Jacobian/finite-range primitives are reused from immutable
v4.4 and v4.5 modules. Only the finite search body is minimally adapted; no
monkeypatch or mutation of a parent's globals occurs. Zero/empty boundary
observations return the exact immutable v4.6 parent result. Positive weight changes
both the objective and feasible set, so parent winners cannot be reused as
optima. Native line association and borrowed noise/locality budgets remain
conditional assumptions, not calibrated probabilities or unique poses.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import statewise_uncertain_solver_v4_bias as BASE
from . import statewise_boundary_normal_solver_v4_4 as N
from . import statewise_u_region_solver_v4_5 as U
from . import joint_ot_pnp as V3
from . import joint_ot_jacobian_v2 as J
from . import statewise_partial_boundary_u_solver_v4_6 as LEGACY

_configuration=BASE._configuration
_helper_config=BASE._helper_config
_constraints=BASE._constraints
_objective_pair=BASE._objective_pair
_visible_audit=BASE._visible_audit
_normal_residual_and_jacobian=N._normal_residual_and_jacobian


REVOCATION_REASONS=frozenset(('no_unique_face_supported_junction',
    'grazing_incident_face_junction','only_local_visible_edge_no_neighbor_measurement'))


def _boundary_inputs(records, states, image_hw, xyz=None):
    """Gate state before fields, reuse immutable structural line validation.

    Legacy fallback-B records remain admissible unchanged. Typed B/I records
    require explicit revocation; neither corrected source nor authoritative I
    is relabelled as a runtime fallback. The temporary B structural parser
    adapter below only reuses v4.4's common finite-line/image checks; I state
    is restored immediately and its existing I geometry/canvas guards remain.
    """
    if records is None:
        return []
    if not isinstance(records,(list,tuple)):
        raise ValueError('boundary_observations must be a sequence')
    adapted=[];kinds=[]
    parser_states=list(states)
    for record in records:
        if not isinstance(record,Mapping):
            raise ValueError('Boundary records must be mappings')
        k=V3._id(record)
        if len(states)!=8 or V3._state(states[k]) not in ('B','I'):
            raise ValueError('Typed partial lines accept only B/I; S/U/O/T excluded before line fields')
        state=V3._state(states[k])
        if record.get('state',state)!=state:
            raise ValueError('Partial B/I must preserve its authoritative state')
        typed=record.get('explicit_complete_BI_likelihood_revocation') is True
        if typed:
            if record.get('replaces_complete_BI_likelihood') is not True:
                raise ValueError('Typed partial B/I must explicitly replace complete 2D likelihood')
            provenance=record.get('association_provenance')
            if not isinstance(provenance,Mapping) or provenance.get('revocation_reason') not in REVOCATION_REASONS:
                raise ValueError('Typed partial B/I requires an allowed explicit revocation reason')
            if provenance.get('display_coordinate_is_full_2D_measurement') is not False:
                raise ValueError('Normal display must explicitly be a partial, not full 2D, measurement')
        elif state!='B' or record.get('replaces_raw_B_likelihood') is not True:
            raise ValueError('I or corrected B requires explicit complete BI likelihood revocation')
        parser_states[k]='B'
        adapted.append({**record,'state':'B'})
        kinds.append((state,typed))
    result=N._boundary_inputs(adapted,parser_states,image_hw,xyz)
    for target,source,(state,typed) in zip(result,records,kinds):
        reliability=float(source.get('fit_reliability_weight',np.nan))
        if not np.isfinite(reliability) or reliability<=0:
            raise ValueError('Partial B/I requires its fixed positive CAL fit reliability')
        target['fit_reliability_weight']=reliability
        target['state']=state
        target['replaces_complete_BI_likelihood']=True
        target['explicit_complete_BI_likelihood_revocation']=typed
        target['legacy_fallback_B_record']=not typed
    return result


def validate_complete_BI_likelihood_revocations(boundary, visible, config=None):
    """Check exact old observation/noise archives before removing any 2D cost.

    Accept normalized ``_boundary_inputs`` output. This helper
    is shared with outline solvers, which must retain original full-BI seeds
    and use the same removal IDs in their final objective. It authenticates
    caller archives, not image-line/native identity or physical visibility.
    """
    cfg=_configuration(config)
    original={V3._id(r):r for r in V3._records(visible)}
    for record in boundary:
        k=V3._id(record);state=record.get('state')
        if state not in ('B','I'):
            raise ValueError('Revocation requires authoritative B/I before archived fields')
        if k not in original:
            raise ValueError('Partial B/I requires its immutable original visible reference/seed')
        source=original[k]
        if source.get('state',state)!=state:
            raise ValueError('Revocation and visible record state mismatch')
        typed=record.get('explicit_complete_BI_likelihood_revocation') is True
        if not typed:
            if state!='B' or source.get('source','VISIBLE_BACKBONE_FALLBACK')!='VISIBLE_BACKBONE_FALLBACK':
                raise ValueError('Legacy partial B permits original fallback source only')
            continue
        provenance=record.get('association_provenance',{})
        if (record.get('replaces_complete_BI_likelihood') is not True
                or provenance.get('revocation_reason') not in REVOCATION_REASONS
                or provenance.get('display_coordinate_is_full_2D_measurement') is not False):
            raise ValueError('Explicit complete B/I revocation audit required')
        observed=V3._finite_array(source.get('covariance_px2'),(2,2),'original observed covariance')
        effective=V3._finite_array(source.get('fit_covariance_px2',
            cfg['covariance_scale']*observed+np.asarray(cfg['state_BI_total_error_floor_px2'][state])),
            (2,2),'original effective covariance')
        weight=float(source.get('fit_reliability_weight',cfg['state_loss_weights'][state]))
        expected=(('original_observed_xy',source.get('xy'),(2,)),
            ('original_observed_covariance_px2',observed,(2,2)),
            ('original_effective_fit_covariance_px2',effective,(2,2)))
        for field,value,shape in expected:
            archive=V3._finite_array(provenance.get(field),shape,field)
            if not np.array_equal(archive,V3._finite_array(value,shape,'current '+field)):
                raise ValueError('Revocation archive differs from immutable '+field)
        if provenance.get('original_measurement_source')!=source.get('source','VISIBLE_BACKBONE_FALLBACK'):
            raise ValueError('Revocation must preserve corrected/fallback source exactly')
        if (provenance.get('original_fit_reliability_weight')!=weight
                or record.get('fit_reliability_weight')!=weight):
            raise ValueError('Revocation must preserve original reliability exactly')
    return sorted(V3._id(r) for r in boundary)


def _joint_objective_pair(p,xyz,K,measurements,scale,prior,cfg,boundary,weight):
    ids={b['native_corner'] for b in boundary}
    complete=[m for m in measurements if m['native_corner'] not in ids]
    value,gradient,components=BASE._objective_pair(p,xyz,K,complete,scale,prior,cfg)
    extra=0.;extra_gradient=np.zeros(6)
    for record in boundary:
        v,g=N._normal_objective_pair(p,xyz,K,[record],scale,
            weight*record['fit_reliability_weight'],cfg['huber_scale'])
        extra+=v;extra_gradient+=g
    return value+extra,gradient+extra_gradient,{**components,
        'partial_boundary_normal_robust_loss':float(extra),'boundary_factor_weight':float(weight),
        'removed_complete_BI_2D_likelihood_ids':sorted(ids),
        'removed_raw_B_2D_likelihood_ids':sorted(b['native_corner'] for b in boundary if b['legacy_fallback_B_record']),
        'complete_BI_likelihood_count':len(complete),
        'raw_B_normal_cost_duplicated':False,'raw_B_tangent_point_cost':False,
        'objective_uses_any_S_or_U':False,'tangent_is_range_only':True}


def _joint_constraint_pair(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg,boundary,regions,analytic=True):
    values,jacobian=U._joint_constraint_pair(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg,regions)
    extra,extra_jacobian=N._boundary_constraint_pair(p,xyz,K,boundary,image_hw,scale,cfg)
    pair=(np.r_[values,extra],np.vstack([jacobian,extra_jacobian]))
    return pair if analytic else pair[0]


def _boundary_audit(p,xyz,K,boundary,image_hw,scale,cfg,weight):
    result=[]
    for record in boundary:
        audit=N._boundary_audit(p,xyz,K,[record],image_hw,scale,cfg,
            weight*record['fit_reliability_weight'])[0]
        audit.update(original_complete_BI_2D_likelihood_replaced=True,
                     original_raw_2D_likelihood_replaced=record['legacy_fallback_B_record'],raw_tangent_cost=False,
                     raw_xy_is_reference_or_seed_not_measured_full_2D=True,
                     fixed_fit_reliability_weight=record['fit_reliability_weight'])
        result.append(audit)
    return result


def _finalize(result,regions,boundary,xyz,K,scale,cfg,weight):
    result['schema']='statewise_partial_BI_u_solver_v4_13'
    result['original_BI_xy_covariance_preserved']=True
    result['reported_BI_coordinates_preserved']=False
    result['unaffected_BI_coordinates_preserved']=True
    result['U_region_constraint_audit']=[]
    result['U_constraints_satisfied']=bool(result['accepted'] and not regions)
    result['fallback_does_not_solve_U']=not result['accepted']
    result['fallback_does_not_solve_partial_boundary']=not result['accepted']
    result['fallback_pose']['valid_as_joint_partial_boundary_solution']=False
    result['boundary_readout']={}
    for record in boundary:
        k=record['native_corner'];source=record['association_provenance']
        result['reported_native'][k]=None
        if 'output_corner_sources' in result:
            result['output_corner_sources'][k]='PARTIAL_BOUNDARY_NORMAL_AND_TANGENT_INTERVAL'
        result['boundary_readout'][str(k)]=dict(state=record['state'],observation_type='PARTIAL_NORMAL_PLUS_TANGENT_INTERVAL',
            original_raw_xy_reference=deepcopy(source.get('original_raw_xy')),
            normal_projected_display_xy=deepcopy(source.get('normal_projected_display_xy')),
            normal_display_is_precise_2D_measurement=False,reported_full_2D_xy=None,
            model_projection_xy=(result['projected_native'][k] if result['accepted'] else None),
            original_complete_BI_likelihood_removed=True,
            archived_complete_BI_measurement_source=source.get('original_measurement_source'),
            explicit_revocation_reason=source.get('revocation_reason'),
            original_raw_B_2D_likelihood_removed=record['state']=='B',lines=[dict(normal_unit_xy=l['normal_unit_xy'].tolist(),
                offset_px=l['offset_px'],normal_variance_px2=l['normal_variance_px2'],
                tangent_interval_px=l['support_tangent_interval_px'].tolist(),
                extrapolation_px=l['support_extrapolation_px']) for l in record['lines']])
    result['boundary_factor_audit'].update(raw_B_2D_likelihood_replaced=any(b['legacy_fallback_B_record'] for b in boundary),
        raw_normal_not_double_counted=True,raw_tangent_has_no_point_cost=True,
        original_xy_covariance_and_old_BI_seed_preserved=True,U_line_factors=0,
        U_canvas_terms=0,U_visibility_terms=0,all_OT_U_partial_BI_simultaneous=True,
        complete_BI_2D_likelihood_replaced=True,
        support_is_caller_corner_locality_range_not_inlier_segment_alone=True)
    result['U_region_search_audit']=dict(active_native_ids=sorted({r['native_corner'] for r in regions}),
        regions=len(regions),U_loss_terms=0,U_canvas_guards=0,U_visibility_guards=0,
        parent_winner_reused=False,normal_objective_change_prevents_parent_optimum_reuse=True)
    if result['accepted']:
        result['U_region_constraint_audit']=U._u_audit(result['projected_native'],regions,cfg['feasibility_tolerance'])
        result['U_constraints_satisfied']=all(a['feasible'] for a in result['U_region_constraint_audit'])
        if not result['U_constraints_satisfied']:
            raise RuntimeError('Accepted pose violates an individual U region')
        if regions:
            result['minimum_joint_slack']=min(result['minimum_joint_slack'],*[a['slack'] for a in result['U_region_constraint_audit']])
    result['partial_BI_constraints_satisfied']=bool(result['accepted'] and all(a['feasible'] for a in result['boundary_observation_audit']))
    result['partial_B_constraints_satisfied']=result['partial_BI_constraints_satisfied']
    return result


def solve_partial_BI_u_region_joint_ot_pnp(xyz: Any, K: Any, states: Sequence, visible: Any,
                                 domains: Any, image_hw: Sequence, coarse_pose: Mapping,
                                 initializations: Any = None, config: Mapping | None = None, *,
                                 boundary_observations: Any = None, boundary_factor_weight: float = 1.0,
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
    if isinstance(boundary_factor_weight,(bool,np.bool_)) or not np.isfinite(boundary_factor_weight) or boundary_factor_weight<0:
        raise ValueError("boundary_factor_weight must be finite and nonnegative")
    boundary=_boundary_inputs(boundary_observations,states,image_hw)
    if not boundary or boundary_factor_weight==0:
        return LEGACY.solve_boundary_u_region_joint_ot_pnp(xyz,K,states,visible,domains,image_hw,coarse_pose,
            initializations=initializations,config=config,boundary_observations=None,
            boundary_factor_weight=0.,u_regions=u_regions)
    regions=U._u_region_inputs(u_regions,states)
    cfg = _configuration(config)
    source_records = V3._records(visible)
    domain_records=V3._records(domains)
    supplied_initializations=list(initializations or [])
    original = {V3._id(r): r for r in source_records}
    xyz, K, states, measurements, domains, image_hw = V3._inputs(
        xyz, K, states, source_records, domain_records, image_hw, _helper_config(cfg))
    boundary=_boundary_inputs(boundary_observations,states,image_hw,xyz)
    partial_ids={r["native_corner"] for r in boundary}
    validate_complete_BI_likelihood_revocations(boundary,source_records,cfg)
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
    by_native={m['native_corner']:m for m in measurements}
    for record in boundary:
        if record['fit_reliability_weight']!=by_native[record['native_corner']]['fit_reliability_weight']:
            raise ValueError('Partial line reliability must preserve the original fixed BI fit weight')
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
    result = {"schema": "statewise_partial_BI_u_solver_v4_13", "accepted": False, "status": "unresolved",
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
        U_partial_lines_allowed_without_state_change=False,
        original_complete_BI_2D_likelihood_removed_in_joint=True,
        original_raw_B_2D_likelihood_removed_in_joint=any(b['legacy_fallback_B_record'] for b in boundary),
        all_OT_and_U_simultaneous=True,zero_or_empty_exact_parent_delegation=True,
        boundary_support_guards_only_in_final_positive_weight_joint_fit=True)
    result["fallback_pose"] = {"R": coarse_R.tolist(), "t": coarse_t.tolist(),
                               "projected_native": coarse_q.tolist(), "source": "INITIAL_REGISTERED_COARSE_PNP",
                               "valid_as_joint_OT_solution": False, "S_projection_only": True}
    for m in measurements:
        result["reported_native"][m["native_corner"]] = m["xy"].tolist()
    for native_id, state in enumerate(states):
        if state == "S":
            result["reported_native"][native_id] = coarse_q[native_id].tolist()
    if len(measurements) < 2:
        result["fallback_reason"] = "two_BI_requires_OT_ranges" if len(measurements) == 2 else "fewer_than_two_BI"
        return _finalize(result,regions,boundary,xyz,K,scale,cfg,boundary_factor_weight)
    pose_family = len(measurements) == 2
    supplied = [{"name": "REGISTERED_COARSE_PRIOR", "R": coarse_R, "t": coarse_t},
                *supplied_initializations]
    starts, search = V3._initial_poses(xyz, K, measurements, supplied, cfg)
    result["search"] = {**search, "supplied_initializations_are_priors": False,
                        "registered_coarse_pose_is_prior": bool(cfg["prior_weight"] > 0), "analytic_jacobian_used": cfg["analytic_jacobian"],
                        "joint_method": "SLSQP_full_SE3_soft_BI_partial_boundary_normals_coarse_prior_all_OT_U_inequalities"}
    seed_objective_pair = V3._cached_value_jacobian(lambda p: _objective_pair(p, xyz, K, measurements, scale, prior, cfg))
    objective_pair = V3._cached_value_jacobian(lambda p: _joint_objective_pair(p, xyz, K, measurements, scale, prior, cfg,boundary,boundary_factor_weight))
    objective = lambda p: objective_pair(p)[0]
    gradient = lambda p: objective_pair(p)[1].copy()
    analytic = cfg["analytic_jacobian"]

    def fit(start, subset_domains, sides, limit, use_boundary=True):
        evaluator=(lambda p: _joint_constraint_pair(p,xyz,K,measurements,subset_domains,sides,image_hw,scale,cfg,boundary,regions)) if use_boundary else (lambda p: _constraints(p,xyz,K,measurements,subset_domains,sides,image_hw,scale,cfg))
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
        return _finalize(result,regions,boundary,xyz,K,scale,cfg,boundary_factor_weight)
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
                          "U_region_audit": U._u_audit(solved["q"],regions,cfg["feasibility_tolerance"]),
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
        return _finalize(result,regions,boundary,xyz,K,scale,cfg,boundary_factor_weight)
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
    objective_measurements=[m for m in measurements if m["native_corner"] not in partial_ids]
    visible_jacobian = J.visible_residual_and_jacobian(p, xyz, K, objective_measurements, scale)[1]
    values, guards_jacobian = _joint_constraint_pair(p, xyz, K, measurements, domains, chosen["sides"], image_hw, scale, cfg,boundary,regions)
    active = np.flatnonzero(values <= 1e-5)
    boundary_jacobian=_normal_residual_and_jacobian(p,xyz,K,boundary,scale)[1]*np.sqrt(boundary_factor_weight)
    boundary_jacobian*=np.repeat([np.sqrt(b["fit_reliability_weight"]) for b in boundary],[len(b["lines"]) for b in boundary])[:,None]
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
                                       "all_U_individual_constraints_pass": True,
                                       "all_partial_BI_tangent_constraints_pass": True,
                                       "all_partial_B_tangent_constraints_pass": True,
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
    return _finalize(result,regions,boundary,xyz,K,scale,cfg,boundary_factor_weight)


solve_boundary_u_region_joint_ot_pnp=solve_partial_BI_u_region_joint_ot_pnp
solve_uncertain_joint_ot_pnp=solve_partial_BI_u_region_joint_ot_pnp
