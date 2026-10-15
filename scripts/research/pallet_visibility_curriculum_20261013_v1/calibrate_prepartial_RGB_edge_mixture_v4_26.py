"""Separate CAL edge noise after restoring nonfallback prepartial B/I points.

The frozen SOURCE24 support-balanced NLL, floors, three starts, fixed native
CAL44 alignment and four LOGO folds are reused exactly. Only the authenticated
runtime-active population, schema and observation-generation provenance change.
No old model is relabelled. Preparation seals CAL inputs and label-file bytes;
reference decoding and real noise fitting occur only in the separate fit step.
The existing v24 tangent-gamma, original q/covariances, O/S and pose models are
not fitted or modified here. No image, backbone or pose optimization is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import types

from . import calibrate_source_preserving_bi_edge_mixture_v4_24 as SOURCE
from . import prepartial_RGB_point_preservation_v4_26 as AUTH


C = SOURCE.C
PARENT = SOURCE.PARENT
PREPARED = SOURCE.PREPARED.parent / 'prepartial_RGB_evidence_v4_26_checks'
OUTPUT = SOURCE.OUTPUT.parent / 'CAL_prepartial_RGB_edge_mixture_v4_26'
LABELS = SOURCE.LABELS
SCHEMA = 'CAL_PREPARTIAL_RGB_POINT_PRESERVING_SUPPORT_BALANCED_NMS_NOISE_V26'
PROTOCOL_SCHEMA = 'PREPARTIAL_RGB_POINT_PRESERVING_SUPPORT_BALANCED_PREFIT_PROTOCOL_V26'
RECIPE = {**deepcopy(SOURCE.RECIPE),
    'observation_generation': AUTH.NOISE_OBSERVATION_FAMILY,
    'applicability': 'ONLY v26 restored-prepartial source-policy runtime active groups; all CAL46 inventory retained',
    'complete_source_policy': AUTH.SOURCE_POLICY,
    'unknown_complete_source': 'preserve exact original complete q/covariance/reliability, including prepartial',
    'replacement': 'restore nonfallback prepartial 2D; remove own old-normal/tangent/oldpartial owner costs only; shared surviving owners remain once',
    'original_partial_hard_guards': 'unchanged measurements, full boundary intervals, OT/U/depth/geometry/canvas',
    'same_support_claim': 'cost ownership suppression, not a new proof of identical historical pixel supports',
    'tangent_gamma_model': 'reuse frozen v24 global model; not fitted here; no O/S/pose-prior calibration changes'}

conditional_covariance_floors = SOURCE.conditional_covariance_floors
mixture_noise_nll_and_gradient = SOURCE.mixture_noise_nll_and_gradient


fit_native_edge_mixture_model = types.FunctionType(SOURCE.fit_native_edge_mixture_model.__code__,
    {**SOURCE.fit_native_edge_mixture_model.__globals__, 'SCHEMA': SCHEMA, 'RECIPE': RECIPE},
    'fit_prepartial_RGB_preserving_native_edge_noise')
fit_native_edge_mixture_model.__kwdefaults__ = deepcopy(SOURCE.fit_native_edge_mixture_model.__kwdefaults__)
_EXPORT = types.FunctionType(SOURCE._EXPORT.__code__,
    {**SOURCE._EXPORT.__globals__, 'SCHEMA': SCHEMA, 'RECIPE': RECIPE,
     'MIX': types.SimpleNamespace(NOISE_OBSERVATION_FAMILY=AUTH.NOISE_OBSERVATION_FAMILY)},
    'export_prepartial_RGB_preserving_noise')


def mixture_noise_parameters(model):
    """Export variance components only; runtime adds each conditional floor once."""
    result = _EXPORT(model)
    result['noise_provenance'].update(applicability_profile=RECIPE['applicability'],
        source_policy=AUTH.SOURCE_POLICY, support_prior_policy=RECIPE['component_prior'],
        prepartial_RGB_preservation_generation='V26', archived_complete_tangent_calibrated=False,
        tangent_gamma_fitted_by_this_model=False, original_q_covariance_reliability_unchanged=True,
        original_partial_hard_guards_preserved=True)
    return result


def _verify_preparation(output):
    protocol = PARENT.OLD._verify_preparation(output)
    if protocol['recipe'] != RECIPE or protocol['schema'] != PROTOCOL_SCHEMA:
        raise ValueError('Separate v26 restored-prepartial active-population preparation required')
    return protocol


# SOURCE24's preparer already verifies all metadata seals, runtime active-group
# identity and hierarchical support/fit priors. It never decodes label values.
# A private globals dictionary changes only the declared generation/paths; no
# frozen parent function, module global or model is modified.
_PREPARE = types.FunctionType(SOURCE.prepare.__code__,
    {**SOURCE.prepare.__globals__, '__name__': __name__, 'PREPARED': PREPARED,
     'OUTPUT': OUTPUT, 'SCHEMA': SCHEMA, 'PROTOCOL_SCHEMA': PROTOCOL_SCHEMA,
     'RECIPE': RECIPE, 'AUTH': AUTH}, 'prepare_prepartial_RGB_CAL46_same_SOURCE24_recipe')


def prepare(output=OUTPUT):
    return _PREPARE(Path(output).resolve())


_FIT = types.FunctionType(SOURCE._FIT.__code__,
    {**SOURCE._FIT.__globals__, '__name__': __name__, '_verify_preparation': _verify_preparation,
     'fit_native_edge_mixture_model': fit_native_edge_mixture_model,
     'RECIPE': RECIPE, 'SCHEMA': SCHEMA, 'AUTH': AUTH, 'OUTPUT': OUTPUT},
    'fit_prepartial_RGB_CAL46_same_SOURCE24_recipe')


def fit(output=OUTPUT):
    return _FIT(Path(output).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'fit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else fit)(args.output)
