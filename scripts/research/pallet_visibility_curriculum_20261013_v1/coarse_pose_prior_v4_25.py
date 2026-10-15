"""Pure opt-in CAL coarse-pose regularizer, using the exact frozen P1 API.

No file/model/reference/image/keypoint reads or fits. The original first
physical rotation and model-center camera position are fixed once, then
rebound into each trial registered native basis. FULL uses old six-dimensional
whitening; TRANSLATION uses the marginal covariance Sigma6[3:,3:] and the last
three *unwhitened* old residuals, never a conditional/full-precision block.

This is an unbounded Gaussian, correlated coarse-prediction regularizer.
It is not independent image evidence, a posterior, or a calibrated physical
coverage statement. Existing native observations and every hard guard remain.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from . import statewise_uncertain_solver_v4_bias as BIAS


REGULARIZER_KIND = 'CAL_COARSE_POSE_PRIOR'
MODES = ('FULL', 'TRANSLATION')
SCHEMA = 'FIXED_ORIGINAL_PHYSICAL_CAL_COARSE_POSE_PRIOR_V25'
_SOURCE_KEYS = ('schema', 'mode', 'factor_weight', 'bias_strength',
    'calibration_model_digest_sha256', 'prior_covariance6', 'prior_bias6',
    'original_native_xyz', 'original_R_cf', 'original_t', 'original_physical_Q',
    'original_first_R_physical', 'original_first_center_camera_xyz')


def _finite(value, shape, name):
    result = np.asarray(value, float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + ' must be finite with shape ' + str(shape))
    return result


def _rotation(value, name):
    result = _finite(value, (3, 3), name)
    if (not np.allclose(result.T @ result, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(result), 1., atol=1e-7, rtol=0)):
        raise ValueError(name + ' must be a proper rotation')
    return result


def _covariance(value):
    result = _finite(value, (6, 6), 'CAL covariance6')
    if not np.allclose(result, result.T, atol=1e-12, rtol=0):
        raise ValueError('CAL covariance6 must be symmetric')
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError('CAL covariance6 must be positive definite') from error
    return result


def _parameters(mode, weight, strength, digest):
    if mode not in MODES:
        raise ValueError('Explicit FULL or TRANSLATION mode required')
    if (isinstance(weight, (bool, np.bool_)) or not np.isfinite(weight) or weight < 0
            or isinstance(strength, (bool, np.bool_)) or not np.isfinite(strength)
            or not 0 <= strength <= 1):
        raise ValueError('Nonnegative factor weight and bias_strength in [0,1] required')
    if not (isinstance(digest, str) and len(digest) == 64
            and all(character in '0123456789abcdef' for character in digest)):
        raise ValueError('Explicit fixed CAL model SHA256 digest required')


def _source_digest(prepared):
    return hashlib.sha256(json.dumps({key: prepared[key] for key in _SOURCE_KEYS},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _validate(prepared):
    if (prepared.get('schema') != SCHEMA
            or prepared.get('source_binding_sha256') != _source_digest(prepared)):
        raise ValueError('Fixed original coarse/CAL parameter binding changed')
    _parameters(prepared['mode'], prepared['factor_weight'], prepared['bias_strength'],
        prepared['calibration_model_digest_sha256'])
    _covariance(prepared['prior_covariance6'])
    _finite(prepared['prior_bias6'], (6,), 'CAL bias6')
    _rotation(prepared['original_first_R_physical'], 'Original first physical R')
    center = _finite(prepared['original_first_center_camera_xyz'], (3,), 'Original model center')
    if center[2] <= 0:
        raise ValueError('Original fixed model center must have positive depth')


def prepare_coarse_pose_prior(original_xyz, original_coarse_pose, covariance6, bias6, *,
        calibration_model_digest_sha256, mode='FULL', factor_weight=1., bias_strength=1.):
    """Fix the original physical pose and CAL parameters before trial families."""
    _parameters(mode, factor_weight, bias_strength, calibration_model_digest_sha256)
    xyz = _finite(original_xyz, (8, 3), 'Original registered xyz')
    R = _rotation(original_coarse_pose['R'], 'Original coarse R_cf')
    t = _finite(original_coarse_pose['t'], (3,), 'Original coarse t')
    Q = _rotation(original_coarse_pose.get('physical_registration_Q', np.eye(3)), 'Original physical Q')
    physical = R @ Q
    if 'R_physical' in original_coarse_pose and not np.allclose(
            physical, _rotation(original_coarse_pose['R_physical'], 'Supplied physical R'), atol=1e-7, rtol=0):
        raise ValueError('Original R_physical must equal original R_cf @ Q')
    center = R @ xyz.mean(axis=0) + t
    if center[2] <= 0:
        raise ValueError('Original fixed model center must have positive depth')
    covariance = _covariance(covariance6)
    bias = _finite(bias6, (6,), 'CAL bias6')
    prepared = dict(schema=SCHEMA, mode=mode, factor_weight=float(factor_weight),
        bias_strength=float(bias_strength), calibration_model_digest_sha256=calibration_model_digest_sha256,
        prior_covariance6=covariance.tolist(), prior_bias6=bias.tolist(),
        original_native_xyz=xyz.tolist(), original_R_cf=R.tolist(), original_t=t.tolist(),
        original_physical_Q=Q.tolist(), original_first_R_physical=physical.tolist(),
        original_first_center_camera_xyz=center.tolist())
    prepared['source_binding_sha256'] = _source_digest(prepared)
    return prepared


def rebind_coarse_pose_prior(prepared, trial_xyz, trial_physical_registration_Q):
    """Preserve original physical R/center; use only trial basis and model mean."""
    _validate(prepared)
    xyz = _finite(trial_xyz, (8, 3), 'Trial registered xyz')
    Q = _rotation(trial_physical_registration_Q, 'Trial physical Q')
    same_basis = np.array_equal(Q, np.asarray(prepared['original_physical_Q']))
    # The algebraic shortcut preserves bitwise old-P1 residuals in the exact
    # original basis; it never changes which coarse prediction is used.
    coarse_R = (np.asarray(prepared['original_R_cf']) if same_basis else
        np.asarray(prepared['original_first_R_physical']) @ Q.T)
    result = {key: deepcopy(prepared[key]) for key in _SOURCE_KEYS}
    result.update(source_binding_sha256=prepared['source_binding_sha256'],
        trial_native_xyz=xyz.tolist(), trial_physical_registration_Q=Q.tolist(),
        trial_model_center_xyz=xyz.mean(axis=0).tolist(), trial_coarse_R_cf=coarse_R.tolist(),
        original_basis_exact_R_reused=same_basis)
    return result


def coarse_pose_prior_residual_and_jacobian(p, xyz, scale, bound_prior):
    """Use the exact frozen residual/J, with mean bias subtracted once."""
    _validate(bound_prior)
    xyz = _finite(xyz, (8, 3), 'Trial registered xyz')
    if not np.array_equal(xyz, np.asarray(bound_prior.get('trial_native_xyz'))):
        raise ValueError('Prior must be rebound to this exact trial native geometry')
    p = _finite(p, (6,), 'Pose parameter vector')
    if isinstance(scale, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite pose translation scale required')
    Q = _rotation(bound_prior['trial_physical_registration_Q'], 'Bound trial Q')
    R0 = _rotation(bound_prior['trial_coarse_R_cf'], 'Bound trial coarse R')
    if not np.allclose(R0 @ Q, bound_prior['original_first_R_physical'], atol=1e-7, rtol=0):
        raise ValueError('Bound coarse physical rotation must stay fixed')
    center_xyz = xyz.mean(axis=0)
    if not np.array_equal(center_xyz, np.asarray(bound_prior['trial_model_center_xyz'])):
        raise ValueError('Bound trial model center changed')
    center0 = np.asarray(bound_prior['original_first_center_camera_xyz'])
    covariance = np.asarray(bound_prior['prior_covariance6'])
    bias = np.asarray(bound_prior['prior_bias6']) * bound_prior['bias_strength']
    if bound_prior['mode'] == 'FULL':
        whitening = np.linalg.inv(np.linalg.cholesky(covariance))
        return BIAS._prior_residual_and_jacobian(p, float(scale), center_xyz,
            R0, center0, whitening, bias)
    # Obtain raw relative coordinates/J first. Do not use the last rows of
    # a full whitening matrix or precision matrix: those are conditional.
    raw, jacobian = BIAS._prior_residual_and_jacobian(p, float(scale), center_xyz,
        R0, center0, np.eye(6), None)
    whitening = np.linalg.inv(np.linalg.cholesky(covariance[3:, 3:]))
    return whitening @ (raw[3:] - bias[3:]), whitening @ jacobian[3:]


def coarse_pose_prior_value_and_gradient(p, xyz, scale, bound_prior):
    residual, jacobian = coarse_pose_prior_residual_and_jacobian(p, xyz, scale, bound_prior)
    weight = float(bound_prior['factor_weight'])
    value = float(weight * (residual @ residual))
    gradient = 2. * weight * residual @ jacobian
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError('Finite coarse prior value/gradient required')
    return value, gradient, dict(kind=REGULARIZER_KIND, mode=bound_prior['mode'],
        factor_weight=weight, bias_strength=bound_prior['bias_strength'],
        calibration_model_digest_sha256=bound_prior['calibration_model_digest_sha256'],
        original_source_binding_sha256=bound_prior['source_binding_sha256'],
        whitened_residual=residual.tolist(), weighted_squared_residual=value,
        original_first_physical_rotation_and_center_fixed=True,
        trial_model_center_uses_registered_xyz_mean=True, mean_bias_subtracted_once=True,
        translation_covariance_is_marginal_block=bound_prior['mode'] == 'TRANSLATION',
        full_precision_block_or_Schur_conditional_not_used=True,
        unbounded_Gaussian_correlated_coarse_regularizer=True,
        new_independent_image_evidence_or_calibrated_posterior_claim=False,
        S_U_raw_image_2D_or_regions=0, new_hard_constraints=0)


class CoarsePosePriorProblem:
    """Wrap bounded O/S before the optional mask wrapper; delegate all guards."""
    def __init__(self, base_bounded_OS, bound_prior):
        _validate(bound_prior)
        if not np.array_equal(base_bounded_OS.xyz, np.asarray(bound_prior.get('trial_native_xyz'))):
            raise ValueError('Prior and wrapped problem must have identical trial geometry')
        if not np.allclose(_rotation(base_bounded_OS.Q, 'Wrapped problem physical Q'),
                np.asarray(bound_prior.get('trial_physical_registration_Q')), atol=1e-12, rtol=0):
            raise ValueError('Prior and wrapped problem must have identical physical registration Q')
        self.base = base_bounded_OS
        self.coarse_prior = deepcopy(bound_prior)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _augment(self, base_value, parts, prior_value, prior_audit):
        costs = parts['regularizer_costs']
        if [record['kind'] for record in costs] != ['O_COARSE_LOCALIZATION_PRIOR', 'S_COARSE_VISIBILITY_PRIOR']:
            raise ValueError('Wrap exactly O/S once, before mask or another coarse-pose wrapper')
        native, old_prior = float(parts['native_observation_objective']), float(parts['regularizer_objective'])
        if not np.isclose(base_value, native + old_prior, atol=1e-9, rtol=1e-10):
            raise ValueError('Base must be native plus bounded O/S; place mask outside this wrapper')
        total = float(base_value + prior_value)
        return {**parts, 'regularizer_costs': deepcopy(costs) + [
                dict(kind=REGULARIZER_KIND, weighted_cost=prior_value)],
            'regularizer_objective': old_prior + prior_value,
            'total_regularized_objective': total, 'CAL_coarse_pose_prior_objective': prior_value,
            'coarse_pose_prior_loss': prior_value,
            'coarse_pose_prior_loss_source': 'CAL_COARSE_POSE_PRIOR only; bounded O/S remain separate',
            'CAL_coarse_pose_prior_audit': prior_audit,
            'coarse_pose_prior_count': 1, 'native_observation_objective': native,
            'native_costs_exclude_regularizers': True,
            'coarse_prior_does_not_create_observed_image_correspondences': True}

    def objective(self, p):
        value, gradient, parts = self.base.objective(p)
        extra, derivative, audit = coarse_pose_prior_value_and_gradient(
            p, self.xyz, self.scale, self.coarse_prior)
        return value + extra, gradient + derivative, self._augment(value, parts, extra, audit)

    def evaluate(self, R, t):
        checked = self.base.evaluate(R, t)
        if checked.get('objective') is None:
            return checked
        try:
            p = BIAS.V3._pose_vector(np.asarray(R), np.asarray(t), self.scale)
            extra, _, audit = coarse_pose_prior_value_and_gradient(p, self.xyz, self.scale, self.coarse_prior)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return {**checked, 'accepted': False, 'objective': None,
                'reason': str(error), 'coarse_prior_numeric_invalid_not_new_physical_guard': True}
        total = float(checked['objective'] + extra)
        return {**self._augment(checked['objective'], checked, extra, audit), 'objective': total}
