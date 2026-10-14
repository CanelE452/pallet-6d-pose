"""Actual RGB observations and uncertainty-aware joint pose runtime.

No annotation access, calibration fitting or record-specific parameter choice
is available in this API. S is erased before any image prompt or local query.
"""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np

from .joint_ot_pnp import project_native
from .ot_domains_v2 import (build_ot_domains, validate_domain_bindings,
                            image_pixel_sha256, _array_digest)
from .statewise_joint_api_v2 import StatewiseJointV2, registered_geometry, _json
from .state_visibility_v4 import geometry_state_gates_v4, infer_state_visibility_v4
from .visible_bi_correction_v2 import correct_visible_bi


class StatewiseJointV4(StatewiseJointV2):
    def __init__(self, state_model=None, ot_model=None, mask_provider=None,
                 uncertainty_model=None, config=None, state_producer=None):
        super().__init__(state_model, ot_model, mask_provider, config,
                         state_producer or infer_state_visibility_v4)
        self.injected_state_producer = state_producer is not None
        self.uncertainty_model = uncertainty_model

    def observe(self, image_bgr, raw_points, K, native_xyz, coarse_pose, *,
                pallet_mask=None, state_pallet_mask=None,
                independent_truncation=None, crop_transform=None):
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Actual uint8 BGR image required')
        raw = np.array(raw_points, dtype=float, copy=True)
        if raw.shape not in ((8, 2), (9, 2)):
            raise ValueError('Native 8 or 9 points required')
        original_pose = deepcopy(coarse_pose)
        geometry = registered_geometry(image.shape[:2], K, native_xyz, coarse_pose)
        if crop_transform is not None and (
            crop_transform.get('schema') != 'REAL_RGB_TRANSLATION_CROP_V2'
            or tuple(crop_transform['canvas_hw']) != image.shape[:2]
            or crop_transform['crop_image_sha256'] != image_pixel_sha256(image)
            or crop_transform['crop_K_sha256'] != _array_digest(geometry['K'])):
            raise ValueError('Actual crop canvas, K and RGB bindings required')
        hidden = geometry['initial_self_hidden']
        gated = raw.copy()
        gated[:8][hidden] = [-1., -1.]
        _, eligible = geometry_state_gates_v4(
            image, gated, hidden, geometry['initial_roles'], geometry['initial_projection_xy'],
            independent_truncation=independent_truncation)
        correction_mask, state_mask, mask_audit, model = self._masks(
            image, gated, geometry, eligible, pallet_mask, state_pallet_mask)
        observed = self.state_producer(
            image, gated.copy(), hidden.copy(), list(geometry['initial_roles']),
            geometry['initial_projection_xy'].copy(), model, pallet_mask=state_mask,
            independent_truncation=deepcopy(independent_truncation))
        states = list(observed['states'])
        records = observed.get('records', [])
        if (len(states) != 8 or any(s not in {'S','B','I','O','T','U'} for s in states)
                or len(records) != 8 or any(
                    int(r['native_id']) != k or r['state'] != states[k]
                    for k,r in enumerate(records))):
            raise ValueError('Native state records must preserve order')
        if any(k not in eligible for k in observed.get('rgb_queried_ids', [])):
            raise ValueError('Excluded geometric gate queried')
        if any(records[k].get('rgb_evaluated') for k in np.flatnonzero(hidden)):
            raise ValueError('S received a local RGB query')
        if any((s == 'S') != bool(hidden[k]) for k,s in enumerate(states)):
            raise ValueError('Initial S gate must remain fixed')
        if any(s in {'B','I'} and s != geometry['initial_roles'][k]
               for k,s in enumerate(states)):
            raise ValueError('Visible role changed')
        normalized_pose = dict(original_pose, R_cf=geometry['R_cf'].tolist(),
            centroid=geometry['centroid'].tolist(), cf_extents=geometry['cf_extents'].tolist())
        correction = correct_visible_bi(
            image, gated.copy(), geometry['K'], normalized_pose, states,
            pallet_mask=correction_mask, config=self.config.get('bi'),
            native_xyz=geometry['native_xyz'])
        measurements = deepcopy(correction['visible_measurements'])
        domain_states = ['V' if s in {'B','I'} else s for s in states]
        sides = {}
        for record in records:
            if record['state'] == 'T' and record.get('physical_truncation_certified'):
                choices = record.get('outside_canvas_side', [])
                if isinstance(choices,str): choices = [choices]
                if len(choices) == 1: sides[record['native_id']] = choices[0]
        kwargs = dict(prediction_source=self.config.get('prediction_source','raw_backbone'),
            record_id=self.config.get('record_id','current'), self_hidden=hidden)
        domains = build_ot_domains(
            image, gated, domain_states, geometry['K'], geometry['native_xyz'],
            self.ot_model, truncation_sides=sides, crop_transform=crop_transform, **kwargs)
        validate_domain_bindings(
            domains, image, gated, domain_states, geometry['K'], geometry['native_xyz'], **kwargs)
        return _json(dict(schema='STATEWISE_RGB_OBSERVATIONS_V4', states=states,
            initial_geometry=geometry, state_observation=observed, mask_audit=mask_audit,
            visible_correction=correction, frozen_visible_measurements=measurements,
            ot_domains=domains, original_raw_points=raw.tolist(),
            original_coarse_pose=original_pose,
            provenance=dict(GT_inputs=False, S_image_queries=0, S_mask_prompt_queries=0,
                S_coordinates_removed_before_image_pipeline=True,
                current_RGB_state_inference=True, B_I_measured_before_joint_OT=True,
                initial_pose_coordinates_are_not_observations=True,
                conditional_T_cannot_override_available_nonS_image_query=True,
                original_native_model_and_registration_preserved=True,
                state_producer_injected_for_tests=self.injected_state_producer)))

    def solve_observations(self, observation):
        from .statewise_uncertainty_calibration_v4 import (
            load_uncertainty_model, apply_calibrated_measurements, solver_config_from_model)
        from .statewise_uncertain_solver_v4 import solve_uncertain_joint_ot_pnp
        source = deepcopy(observation)
        geometry = source['initial_geometry']
        model = self.uncertainty_model
        if model is None:
            raise ValueError('An explicitly fitted CAL-only uncertainty model is required')
        if not isinstance(model,Mapping): model = load_uncertainty_model(model)
        original = source['frozen_visible_measurements']
        fit = apply_calibrated_measurements(original, model, states=source['states'])
        weight = self.config.get('solver',{}).get('prior_weight',1.)
        config = solver_config_from_model(model, prior_weight=weight)
        config.update(self.config.get('solver',{}))
        solution = solve_uncertain_joint_ot_pnp(
            geometry['native_xyz'], geometry['K'], source['states'], fit,
            source['ot_domains']['domains'], geometry['image_hw'],
            coarse_pose={'R':geometry['R_cf'],'t':geometry['centroid']}, config=config)
        if original != source['frozen_visible_measurements']:
            raise RuntimeError('Frozen observation changed')
        return finish_observations(source, solution, fit, model)

    def refine(self, image_bgr, raw_points, K, native_xyz, coarse_pose, **kwargs):
        return self.solve_observations(self.observe(
            image_bgr, raw_points, K, native_xyz, coarse_pose, **kwargs))


def finish_observations(observation, solution, calibrated_measurements, model):
    result = deepcopy(observation)
    geometry = result['initial_geometry']
    accepted = bool(solution['accepted'])
    seed = solution.get('BI_prior_seed') or solution.get('corrected_BI_seed') or {}
    if accepted:
        R,t = np.asarray(solution['R']),np.asarray(solution['t'])
        pose_source = 'SOFT_BI_COARSE_PRIOR_ALL_OT_JOINT_SE3'
    elif seed.get('visible_support_guards_pass') and not seed.get('contains_underconstrained_pose_family'):
        R,t = np.asarray(seed['R']),np.asarray(seed['t'])
        pose_source = 'EXPLICIT_SOFT_BI_PRIOR_SEED_FALLBACK_S_ONLY'
    else:
        R,t = np.asarray(geometry['R_cf']),np.asarray(geometry['centroid'])
        pose_source = 'EXPLICIT_REGISTERED_COARSE_FALLBACK_S_ONLY'
    projected,_ = project_native(
        np.asarray(geometry['native_xyz']),R,t,np.asarray(geometry['K']))
    ambiguous = bool((solution.get('ambiguity') or {}).get('detected',False))
    points,sources = [None]*8,['UNRESOLVED']*8
    for m in result['frozen_visible_measurements']:
        k=m['native_corner'];points[k]=deepcopy(m['xy'])
        sources[k]='FROZEN_CURRENT_RGB_BI_OBSERVATION'
    for k,state in enumerate(result['states']):
        if state=='S':
            points[k]=projected[k].tolist();sources[k]='FINAL_PNP_S_PROJECTION'
        elif state in {'O','T'} and accepted:
            points[k]=projected[k].tolist()
            sources[k]='AMBIGUOUS_JOINT_FAMILY_PROJECTION' if ambiguous else 'ALL_OT_JOINT_RIGID_PROJECTION'
    center = result['original_raw_points'][8] if len(result['original_raw_points'])==9 else None
    if center is not None: points.append(deepcopy(center));sources.append('ORIGINAL_NINTH_CENTER_METADATA')
    result.update(schema='STATEWISE_UNCERTAINTY_RGB_V4',method='STATEWISE_SOFT_BI_PRIOR_ALL_OT_V4',
        accepted=accepted,calibrated_visible_measurements=deepcopy(calibrated_measurements),
        joint_pose=solution,solution=solution,
        actual_pose=dict(available=True,R_cf=R.tolist(),centroid=t.tolist(),
            R_physical=(R@np.asarray(geometry['physical_registration_Q'])).tolist(),
            physical_registration_Q=deepcopy(geometry['physical_registration_Q']),
            cf_extents=deepcopy(geometry['cf_extents']),source=pose_source,
            joint_feasible=accepted,fallback=not accepted,
            fallback_pose_does_not_solve_OT=not accepted),
        keypoint_readout=dict(reported_points=points,point_sources=sources,
            rigid_model_corners=projected.tolist(),native_order_preserved=True,
            BI_observations_preserved=True,ninth_center_metadata_unchanged=True,
            original_ninth_center_metadata=deepcopy(center),mixed_points_are_one_rigid_pose=False,
            OT_constraints_satisfied=bool(solution.get('OT_constraints_satisfied',accepted)),
            OT_feasible_hypothesis_available=accepted and bool(result['ot_domains']['domains']),
            OT_solved=accepted and not ambiguous and bool(result['ot_domains']['domains']),
            OT_unique_solution_claimed=False,
            pose_family_candidates=solution.get('pose_family_candidates',[])),
        fallback_audit=dict(applied=not accepted,pose_source=pose_source,
            reason=solution.get('fallback_reason'),O_T_reported_as_solved=accepted and not ambiguous),
        ambiguity_audit=dict(material_ambiguity=ambiguous,globally_unique_pose_proved=False,
            accepted_pose_is_one_joint_feasible_branch=accepted,
            solver_ambiguity=solution.get('ambiguity')),
        uncertainty_model_binding=deepcopy(model.get('provenance',{})))
    result['provenance'].update(all_OT_simultaneous=True,O_T_regions_used_as_precise_points=False,
        initial_pose_is_initialization_only=False,initial_pose_is_soft_regularizer=True,
        BI_hard_residual_cap=False,pose_projection_shift_clamp=False,
        effective_covariance_is_CAL_empirical_not_physical_coverage_guarantee=True)
    return _json(result)
