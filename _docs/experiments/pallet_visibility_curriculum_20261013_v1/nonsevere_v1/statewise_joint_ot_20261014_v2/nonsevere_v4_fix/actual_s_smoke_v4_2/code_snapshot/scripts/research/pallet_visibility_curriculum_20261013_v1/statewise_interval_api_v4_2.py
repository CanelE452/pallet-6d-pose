"""Opt-in CAL localization interval gate; frozen bias pose solver inherited."""
from copy import deepcopy

import numpy as np

from .statewise_bias_api_v4_1 import StatewiseBiasV4_1
from .statewise_joint_api_v2 import _json
from .ot_domains_v2 import build_ot_domains, validate_domain_bindings
from .truncation_interval_v4_2 import load_truncation_interval_model, apply_truncation_interval


class StatewiseIntervalV4_2(StatewiseBiasV4_1):
    def __init__(self, state_model=None, ot_model=None, mask_provider=None,
                 uncertainty_model=None, truncation_model=None, config=None,
                 state_producer=None):
        super().__init__(state_model, ot_model, mask_provider,
                         uncertainty_model, config, state_producer)
        self.truncation_model = load_truncation_interval_model(truncation_model)

    def observe(self, image_bgr, raw_points, K, native_xyz, coarse_pose, **kwargs):
        original = super().observe(image_bgr, raw_points, K, native_xyz, coarse_pose, **kwargs)
        return self.adapt_observation(image_bgr, original, crop_transform=kwargs.get('crop_transform'))

    def adapt_observation(self, image_bgr, observation, *, crop_transform=None):
        """Rebind cached observations using actual pixels, without any RGB forward.

        The current image is required only for exact existing/new domain image
        bindings. State, mask and B/I predictors are not invoked by this method.
        """
        original = deepcopy(observation)
        result = deepcopy(original); geometry = result['initial_geometry']
        if tuple(geometry['image_hw']) != tuple(np.asarray(image_bgr).shape[:2]):
            raise ValueError('Cached observation canvas differs from actual image')
        old_states = list(original['states'])
        gate = apply_truncation_interval(old_states, original['original_raw_points'],
            geometry['image_hw'], original['ot_domains']['nonself_prediction_scale_px'],
            self.truncation_model, original['state_observation']['records'],
            geometry['initial_self_hidden'])
        result['states'] = gate['states']
        result['state_observation']['states'] = gate['states']
        result['state_observation']['records'] = gate['records']
        result['state_observation']['truncation_interval_guard'] = gate['audit']
        gated = np.asarray(original['original_raw_points'], dtype=float).copy()
        gated[:8][np.asarray(geometry['initial_self_hidden'], bool)] = [-1., -1.]
        domain_states = ['V' if s in {'B','I'} else s for s in result['states']]
        sides = {}
        for record in gate['records']:
            if record['state'] == 'T' and record.get('physical_truncation_certified'):
                choices = record.get('outside_canvas_side', [])
                if isinstance(choices, str): choices = [choices]
                if len(choices) == 1: sides[record['native_id']] = choices[0]
        binding_args = dict(prediction_source=original['ot_domains']['binding']['prediction_source'],
            record_id=original['ot_domains']['binding']['record_id'], self_hidden=geometry['initial_self_hidden'])
        old_domain_states = ['V' if s in {'B','I'} else s for s in old_states]
        validate_domain_bindings(original['ot_domains'], image_bgr, gated, old_domain_states,
            geometry['K'], geometry['native_xyz'], **binding_args)
        # Rebuild all geometry and bindings; old domain hashes are never filtered.
        domains = build_ot_domains(image_bgr, gated, domain_states, geometry['K'],
            geometry['native_xyz'], self.ot_model, truncation_sides=sides,
            crop_transform=crop_transform, **binding_args)
        validate_domain_bindings(domains, image_bgr, gated, domain_states,
            geometry['K'], geometry['native_xyz'], **binding_args)
        if domains['nonself_prediction_scale_px'] != original['ot_domains']['nonself_prediction_scale_px']:
            raise RuntimeError('T interval gate changed frozen non-S scene scale')
        old_O = {d['native_corner']:d for d in original['ot_domains']['domains'] if d['state']=='O'}
        new_O = {d['native_corner']:d for d in domains['domains'] if d['state']=='O'}
        keys = ('shape','center_xy','half_axes_px','angle_deg','bounds_xyxy','halfplanes')
        if set(old_O) != set(new_O) or any(old_O[k].get(n) != new_O[k].get(n) for k in old_O for n in keys):
            raise RuntimeError('T interval gate changed O numeric geometry')
        old_T = {d['native_corner']:d for d in original['ot_domains']['domains'] if d['state']=='T'}
        new_T = {d['native_corner']:d for d in domains['domains'] if d['state']=='T'}
        tkeys = ('shape','outside_canvas_side','outside_margin_px','halfplanes','canvas_hw',
                 'source_crop_validity_intersection')
        if any(k not in old_T or any(old_T[k].get(n) != new_T[k].get(n) for n in tkeys) for k in new_T):
            raise RuntimeError('Interval gate changed retained T numeric geometry or crop-validity intersection')
        result['ot_domains'] = domains
        result['truncation_interval_guard'] = gate['audit'] | {
            'all_retained_OT_domains_rebuilt_and_bindings_validated': True,
            'O_numeric_geometry_unchanged': True,
            'retained_T_numeric_geometry_and_crop_halfplanes_unchanged': True,
            'original_domain_binding': deepcopy(original['ot_domains']['binding'])}
        for key in ('initial_geometry','mask_audit','visible_correction','frozen_visible_measurements',
                    'original_raw_points','original_coarse_pose'):
            if result[key] != original[key]:
                raise RuntimeError('Interval observation contract changed: '+key)
        if any(a != b for a,b in zip(old_states,result['states']) if a != 'T'):
            raise RuntimeError('Interval gate changed a non-T role')
        result.update(schema='STATEWISE_INTERVAL_RGB_OBSERVATIONS_V4_2')
        result['provenance'].update(CAL_raw_localization_interval_applied=True,
            initial_S_and_BI_roles_preserved=True, retained_OT_domains_rebuilt=True,
            interval_gate_RGB_reads=0, interval_gate_GT_inputs=False,
            inherited_bias_solver_and_CAL_prior_unchanged=True)
        return _json(result)

    def solve_observations(self, observation):
        guard = observation.get('truncation_interval_guard', {})
        if (guard.get('model_sha256') != self.truncation_model['model_sha256']
                or not guard.get('all_retained_OT_domains_rebuilt_and_bindings_validated')):
            raise ValueError('Call observe or adapt_observation before interval-gated pose fitting')
        # This is exactly the frozen bias solve; only result metadata differs.
        result = super().solve_observations(observation)
        model = self.truncation_model
        result.update(schema='STATEWISE_INTERVAL_CAL_BIAS_RGB_V4_2',
            method='STATEWISE_SOFT_BI_CAL_MEAN_PRIOR_INTERVAL_GATED_ALL_OT_V4_2',
            truncation_model_binding=dict(schema=model['schema'],fit_partition=model['fit_partition'],
                model_sha256=model['model_sha256'],radius_ratio=model['radius_ratio'],
                fit_ids=deepcopy(model['fit_ids']),heldout_session=model['heldout_session'],
                empirical_CAL_max_only=True,physical_truncation_certificate=False,
                future_coverage_guarantee=False,physical_probability_calibrated=False))
        return _json(result)
