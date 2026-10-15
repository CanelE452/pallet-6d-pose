"""New V31 CAL46-only noise controller with explicit central SHA per call.

The frozen V30 controller supplies numerical preparation, reference alignment,
residuals, full/four-LOGO fitting and reporting code in a private namespace.
Only M31, paths and schema/provenance are rebound. Neither an old fitted model
nor parent module globals are changed. No data is read on import.

Every CLI stage requires --central-plan-sha256; the expected value is never a
source literal or default. Preparation and actual CAL-reference fitting have
separate root GO flags. Labels are byte-hashed before fitting, not decoded by
plan/prepare. All actual stages remain outside source/synthetic tests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType, SimpleNamespace
import sys

from . import calibrate_geometry_guided_edge_noise_v4_30 as PARENT
from . import geometry_guided_edge_noise_calibration_v4_31 as M

C, OLD = PARENT.C, PARENT.OLD
OUTPUT = PARENT.OUTPUT.parent / 'CAL_geometry_guided_edge_noise_v4_31'
PREPARED = PARENT.OUTPUT.parent / 'geometry_guided_native_evidence_v4_31_checks'
PLAN_OUTPUT = PARENT.OUTPUT.parent / 'CAL_geometry_guided_edge_noise_plan_v4_31'
LABELS = PARENT.LABELS
CENTRAL_PLAN = OUTPUT.parent / 'observed_anchor_native_ray_method_plan_v4_31/PREDECLARATION.json'
PROTOCOL_SCHEMA = 'SEALED_Q_ANCHORED_SIGNED_NATIVE_RAY_CAL46_EDGE_TANGENT_PROTOCOL_V31'
PLAN_SCHEMA = 'NEW_Q_ANCHORED_SIGNED_NATIVE_RAY_NOISE_CAL_PLAN_V31'

_verify, _union_maps, _snapshot = PARENT._verify, PARENT._union_maps, PARENT._snapshot


def _central_declaration(expected_sha256):
    """Fail before JSON decode unless caller pins this exact central file."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)
            or C.sha(CENTRAL_PLAN) != expected_sha256):
        raise ValueError('Explicit exact central V31 SHA required; no default or substitution')
    central = C.read(CENTRAL_PLAN)
    if (central.get('observation_family') != M.OBSERVATION_FAMILY
            or central.get('population') != M.POPULATION
            or central.get('selection_profile') != M.SELECTION_PROFILE
            or central.get('raw_observation_generation') != M.RAW_OBSERVATION_GENERATION
            or central.get('source_policies') != list(M.SOURCE_POLICIES)
            or central.get('partitions') != dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0)):
        raise ValueError('New V31 family/population/profile/raw generation and full inventory required')
    maps = [central[key] for key in ('source_sha256', 'input_sha256', 'protected_user_sha256')]
    for mapping in maps:
        _verify(mapping)
    _union_maps(*maps)  # Cross-class conflicts fail closed as well.
    return central


def _sources():
    sources = PARENT._sources()
    modules = (sys.modules[__name__], M, M.PARENT, M.CORE, M.CORE.ASSOCIATION)
    additions = {str(Path(module.__file__).resolve().relative_to(C.ROOT)): C.sha(module.__file__)
                 for module in modules}
    test = Path(__file__).with_name('test_calibrate_geometry_guided_edge_noise_v4_31.py')
    additions[str(test.resolve().relative_to(C.ROOT))] = C.sha(test)
    return _union_maps(sources, additions)


def _clone(function, namespace):
    copied = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return copied


def _facade(expected_sha256):
    """Per-operation private function globals; never mutate parent globals."""
    namespace = {**vars(PARENT), 'M': M, 'OUTPUT': OUTPUT, 'PREPARED': PREPARED,
        'PLAN_OUTPUT': PLAN_OUTPUT, 'LABELS': LABELS, 'CENTRAL_PLAN': CENTRAL_PLAN,
        # This is a per-call value, not a module constant or source default.
        'CENTRAL_PLAN_SHA256': expected_sha256, 'PROTOCOL_SCHEMA': PROTOCOL_SCHEMA,
        '_central_declaration': lambda: _central_declaration(expected_sha256), '_sources': _sources}
    for name in ('validate_prepared_row', 'prepare', '_verify_preparation',
                 '_fit_or_status', '_heldout_description', 'fit'):
        namespace[name] = _clone(getattr(PARENT, name), namespace)
    verify_preparation = namespace['_verify_preparation']

    def verify_bound_preparation(output):
        protocol = verify_preparation(output)
        central = _central_declaration(expected_sha256)
        if protocol['input_sha256'].get(str(CENTRAL_PLAN.relative_to(C.ROOT))) != expected_sha256:
            raise ValueError('Prepared input must bind the same caller-supplied central V31 SHA')
        _union_maps(protocol['source_sha256'], protocol['input_sha256'],
                    protocol['protected_user_sha256'], *(central[key] for key in
                    ('source_sha256', 'input_sha256', 'protected_user_sha256')))
        return protocol

    namespace['_verify_preparation'] = verify_bound_preparation
    return SimpleNamespace(**namespace)


def validate_prepared_row(row):
    """Same frozen metadata/selected-axis/floor checks, typed to M31 only."""
    return _facade(None).validate_prepared_row(row)


def plan(output=PLAN_OUTPUT, *, expected_central_plan_sha256=None):
    central = _central_declaration(expected_central_plan_sha256)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve all noise declarations')
    sources = _union_maps(central['source_sha256'], _sources())
    inputs = _union_maps(central['input_sha256'],
        {str(CENTRAL_PLAN.relative_to(C.ROOT)): expected_central_plan_sha256})
    _union_maps(sources, inputs, central['protected_user_sha256'])
    value = dict(schema=PLAN_SCHEMA, UTC=datetime.now(timezone.utc).isoformat(),
        observation_generation=M.OBSERVATION_FAMILY,
        raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
        noise_population=M.POPULATION, selection_profile=M.SELECTION_PROFILE,
        source_policies=M.SOURCE_POLICIES,
        line_recipes={policy: M.recipe_for(policy) for policy in M.SOURCE_POLICIES},
        tangent_recipes={policy: M.recipe_for(policy, tangent=True) for policy in M.SOURCE_POLICIES},
        metadata_contract='All46 ordered unique rows; both policies; one selected actual line per physical edge and exact selected measured-line tangent axes.',
        CAL_reference=str(LABELS.relative_to(C.ROOT)), CAL_reference_bytes_sha256=C.sha(LABELS),
        sources=sources, input_sha256=inputs,
        central_source_sha256=deepcopy(central['source_sha256']),
        central_input_sha256=deepcopy(central['input_sha256']),
        central_plan_path=str(CENTRAL_PLAN.relative_to(C.ROOT)),
        central_plan_sha256=expected_central_plan_sha256,
        protected_user_sha256=central['protected_user_sha256'],
        source_and_metadata_before_new_CAL_reference_decode=True,
        labels_historically_exposed_exploratory=True, old_model_relabelled=False,
        no_support_policy='Explicit MODEL_UNAVAILABLE; no old model substitution; single-session fit descriptive/weak.',
        unseen_fold_policy='Held session excluded before numerical errors; empty training fold explicit MODEL_UNAVAILABLE.',
        inventory46_compatible44_full_plus_four_LOGO_both_policies=True,
        expected_models_when_all_populations_supported=20,
        expected_optimizer_starts_when_all_populations_supported=60,
        frozen_numerical_controller_sha256=C.sha(PARENT.__file__),
        non_Cal_numeric_reference_decodes=0, image_NN_point_or_pose_fits=0, default_adoption=False)
    output.mkdir(parents=True)
    C.write(output / 'PLAN.json', value)
    _snapshot(output, sources)
    _verify(sources);_verify(inputs);_verify(central['protected_user_sha256'])
    C.write(output / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(output / 'PLAN.json'), sources=sources))
    return value


def prepare(output=OUTPUT, prepared=PREPARED, plan_folder=PLAN_OUTPUT, *,
            root_metadata_GO=False, expected_central_plan_sha256=None):
    if root_metadata_GO is not True:
        raise PermissionError('Separate root new-CAL46 metadata preparation GO before IO')
    _central_declaration(expected_central_plan_sha256)
    return _facade(expected_central_plan_sha256).prepare(output, prepared, plan_folder)


def _verify_preparation(output, *, expected_central_plan_sha256=None):
    _central_declaration(expected_central_plan_sha256)
    return _facade(expected_central_plan_sha256)._verify_preparation(Path(output).resolve())


def _fit_or_status(records, policy, kind, bindings, held=None):
    return _facade(None)._fit_or_status(records, policy, kind, bindings, held)


def _heldout_description(rows, session, line_result, tangent_result):
    return _facade(None)._heldout_description(rows, session, line_result, tangent_result)


def fit(output=OUTPUT, *, root_CAL_fit_GO=False, expected_central_plan_sha256=None):
    if root_CAL_fit_GO is not True:
        raise PermissionError('Separate explicit root CAL-reference/fit GO before IO')
    _central_declaration(expected_central_plan_sha256)
    return _facade(expected_central_plan_sha256).fit(output, root_CAL_fit_GO=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'prepare', 'fit'))
    parser.add_argument('--central-plan-sha256', required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--plan-folder', type=Path, default=PLAN_OUTPUT)
    parser.add_argument('--root-metadata-GO', action='store_true')
    parser.add_argument('--root-CAL-fit-GO', action='store_true')
    args = parser.parse_args()
    options = dict(expected_central_plan_sha256=args.central_plan_sha256)
    if args.stage == 'plan':
        plan(args.plan_folder, **options)
    elif args.stage == 'prepare':
        prepare(args.output, args.prepared, args.plan_folder,
                root_metadata_GO=args.root_metadata_GO, **options)
    else:
        fit(args.output, root_CAL_fit_GO=args.root_CAL_fit_GO, **options)
