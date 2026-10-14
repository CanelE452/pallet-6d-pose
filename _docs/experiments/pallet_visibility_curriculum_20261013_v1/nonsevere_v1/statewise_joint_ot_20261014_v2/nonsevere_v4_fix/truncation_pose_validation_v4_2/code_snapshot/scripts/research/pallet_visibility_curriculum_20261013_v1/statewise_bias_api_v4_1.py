"""Opt-in CAL mean-bias runtime; preserves the frozen v4 RGB observation API.

The six-coordinate prior mean is read only from the supplied CAL model. Neither
image observations nor the initial pose are shifted to apply that mean.
"""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .statewise_joint_api_v4 import StatewiseJointV4, finish_observations


class StatewiseBiasV4_1(StatewiseJointV4):
    def solve_observations(self, observation):
        from .statewise_uncertainty_calibration_v4 import (
            load_uncertainty_model, apply_calibrated_measurements,
            solver_config_from_model)
        from .statewise_uncertain_solver_v4_bias import solve_uncertain_joint_ot_pnp

        source = deepcopy(observation)
        geometry = source['initial_geometry']
        model = self.uncertainty_model
        if model is None:
            raise ValueError('An explicitly fitted CAL-only uncertainty model is required')
        if not isinstance(model, Mapping):
            model = load_uncertainty_model(model)
        if (model.get('schema') != 'STATEWISE_UNCERTAINTY_CAL_V4_1'
                or model.get('fit_partition') != 'CAL'):
            raise ValueError('Bias runtime requires a valid CAL-only uncertainty model')
        supplied = deepcopy(self.config.get('solver', {}))
        strength = float(supplied.get('bias_strength', 0.))
        mean = model.get('prior_fit', {}).get('median_bias6')
        if mean is None:
            if strength != 0.:
                raise ValueError('Nonzero bias requires the fitted CAL prior median')
            mean = [0.] * 6
        mean = np.asarray(mean, dtype=float)
        if mean.shape != (6,) or not np.isfinite(mean).all():
            raise ValueError('CAL prior median must be six finite coordinates')
        if 'prior_bias6' in supplied and not np.array_equal(
                np.asarray(supplied['prior_bias6'], dtype=float), mean):
            raise ValueError('Runtime prior bias must equal the supplied CAL model median')
        original = source['frozen_visible_measurements']
        fit = apply_calibrated_measurements(
            original, model, states=source['states'],
            nonself_scale_px=source['ot_domains']['nonself_prediction_scale_px'])
        config = solver_config_from_model(
            model, prior_weight=supplied.get('prior_weight', 1.))
        config.update(supplied)
        config.update(prior_bias6=mean.tolist(), bias_strength=strength)
        solution = solve_uncertain_joint_ot_pnp(
            geometry['native_xyz'], geometry['K'], source['states'], fit,
            source['ot_domains']['domains'], geometry['image_hw'],
            coarse_pose={'R': geometry['R_cf'], 't': geometry['centroid'],
                'physical_registration_Q': geometry['physical_registration_Q']},
            config=config)
        if original != source['frozen_visible_measurements']:
            raise RuntimeError('Frozen observation changed')
        result = finish_observations(source, solution, fit, model)
        result.update(schema='STATEWISE_CAL_BIAS_RGB_V4_1',
                      method='STATEWISE_SOFT_BI_CAL_MEAN_PRIOR_ALL_OT_V4_1')
        result['provenance'].update(
            prior_mean_from_supplied_CAL_model=True,
            prior_mean_estimated_at_runtime=False,
            bias_strength=strength,
            BI_observations_shifted_to_apply_prior_bias=False,
            initial_pose_shifted_to_apply_prior_bias=False,
            inherited_frozen_V4_RGB_observation_pipeline=True)
        return result
