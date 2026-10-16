"""Postscore-only V31 TEST49 e/selected objective and feasibility diagnostic.

The frozen V29b diagnostic numerical decomposition, native-WD validation,
same-unordered-box transport, classification and one-TEST-row streaming are
privately rebound to M31. No new method, optimization, e runtime prior/start/
candidate/fallback, image measurement or reference annotation is introduced.
Actual preparation and replay both require separate root postscore GO and an
explicit central SHA; source-only construction reads no observation or score.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace
import sys

from . import diagnose_frozen_e_pose_under_v29_v4_29b as PARENT
from . import evaluate_geometry_guided_edges_v4_31 as EXP

C, E = EXP.C, EXP.E
OUTPUT = EXP.OUTPUT.parent / 'frozen_e_geometry_guided_objective_diagnosis_v4_31b'
ARMS = tuple(EXP.ARMS[1:])
REPORTING_TOLERANCE = PARENT.REPORTING_TOLERANCE
PARENT_SOURCE_SHA256 = 'a001b4d7a47b2d011c733689d2a743a62c75002791b3b2acb7e65fbe8bcbe978'
SCHEMA = 'POSTSCORE_FROZEN_E_V31_OBJECTIVE_DIAGNOSIS_V31B'
PLAN_SCHEMA = 'POSTSCORE_FROZEN_E_V31_OBJECTIVE_DIAGNOSIS_PLAN_V31B'


def _runtime_facade():
    prep, cal = EXP._modules()
    return SimpleNamespace(**{**vars(EXP.PARENT), **vars(EXP), 'PREP':prep, 'CAL':cal})


M = _runtime_facade()


def _pins():
    if C.sha(PARENT.__file__) != PARENT_SOURCE_SHA256:
        raise ValueError('Exact frozen V29b numerical diagnostic source required')


def profiles():
    expected = ('ANCHOR_RAY_EDGE_ALL_BI_NATIVE', 'ANCHOR_RAY_EDGE_ALL_BI_CONTAINMENT',
                'ANCHOR_RAY_EDGE_SOURCE_NATIVE', 'ANCHOR_RAY_EDGE_SOURCE_CONTAINMENT')
    if ARMS != expected:
        raise ValueError('Exact V31 four-arm inventory required')
    return {arm:dict(source_policy=policy, containment=bool(index),
        no_full_TP_rotation_pose_prior=True, both_WD_families_allowed=True)
        for policy, arms in EXP.POLICY_ARMS.items() for index, arm in enumerate(arms)}


def _clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def _run_code():
    """The entire numerical replay body changes one output schema constant."""
    tree = ast.parse(inspect.getsource(PARENT.run)); changed = []
    class Version(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'POSTSCORE_FROZEN_E_V29_OBJECTIVE_DIAGNOSIS_V29B':
                changed.append(True)
                return ast.copy_location(ast.Constant(SCHEMA), node)
            return node
    tree = Version().visit(tree)
    if len(changed) != 1:
        raise ValueError('Exactly one diagnostic output schema may change')
    ast.fix_missing_locations(tree)
    code = compile(tree, __file__, 'exec')
    return next(value for value in code.co_consts if isinstance(value, type(PARENT.run.__code__)))


def _facade(expected_central_plan_sha256=None):
    _pins()
    namespace = {**vars(PARENT), 'M':M, 'C':C, 'E':E, 'OUTPUT':OUTPUT, 'ARMS':ARMS,
        'profiles':profiles, '__file__':__file__}
    for name in ('validate_saved_pose_basis', 'transport_diagnostic_pose', 'source_inventory',
                 'plan_summary', 'native_buckets', 'evaluate_parts', 'comparison',
                 'make_problem', '_public'):
        namespace[name] = _clone(getattr(PARENT, name), namespace)
    old_reader = _clone(PARENT._read_parent_small, namespace)

    def read_bound_parent():
        EXP._verify_plan(expected_central_plan_sha256)
        protocol, inputs, prediction = old_reader()
        if (protocol.get('central_plan_sha256') != expected_central_plan_sha256
                or protocol['declaration']['sha256'] != expected_central_plan_sha256
                or protocol.get('selection_profile') != EXP.G.SELECTION_PROFILE
                or protocol.get('noise_observation_family') != EXP.G.OBSERVATION_FAMILY
                or protocol.get('raw_observation_generation') != EXP.G.RAW_OBSERVATION_GENERATION):
            raise ValueError('Same explicitly central-bound V31 scored population required')
        return protocol, inputs, prediction

    namespace['_read_parent_small'] = read_bound_parent
    namespace['run'] = FunctionType(_run_code(), namespace, 'diagnostic_V31_run', PARENT.run.__defaults__)
    namespace['run'].__kwdefaults__ = deepcopy(PARENT.run.__kwdefaults__)
    return SimpleNamespace(**namespace)


# Pure numerical/formatting APIs preserve exact frozen function code objects.
_PURE = _facade()
validate_saved_pose_basis = _PURE.validate_saved_pose_basis
transport_diagnostic_pose = _PURE.transport_diagnostic_pose
source_inventory = _PURE.source_inventory
plan_summary = _PURE.plan_summary
native_buckets = _PURE.native_buckets
evaluate_parts = _PURE.evaluate_parts
comparison = _PURE.comparison
make_problem = _PURE.make_problem
_TEST_prefix = PARENT._TEST_prefix


def prepare(output=OUTPUT, *, root_postscore_GO=False, expected_central_plan_sha256=None):
    if root_postscore_GO is not True:
        raise ValueError('Separate root full31-postscore preparation GO before any data IO')
    _pins(); EXP._verify_plan(expected_central_plan_sha256)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every V31 diagnostic attempt')
    parent, inputs, prediction = _facade(expected_central_plan_sha256)._read_parent_small()
    sources = dict(parent['source_sha256'])
    for module in (sys.modules[__name__], PARENT, PARENT.PREVIOUS, EXP, EXP.G, EXP.AUTH,
                   M.PREP, M.CAL, M.CAL.M, M.BOUNDED, M.CONTAIN, M.CONTAIN.FACTOR):
        path = Path(module.__file__).resolve()
        sources = EXP.PARENT._merge_frozen(sources, {str(path.relative_to(C.ROOT)):C.sha(path)})
    test = Path(__file__).with_name('test_diagnose_frozen_e_pose_under_v31_v4_31b.py')
    sources = EXP.PARENT._merge_frozen(sources, {str(test.resolve().relative_to(C.ROOT)):C.sha(test)})
    names = ('PROTOCOL.json', 'INPUT_SEAL.json', 'PREDICTION_SEAL.json', 'SCORE_SEAL.json',
        'TEST_SCORE.json', 'independent_root_audit_v1/AUDIT.json', 'independent_root_audit_v1/AUDIT_SEAL.json')
    paths = [EXP.OUTPUT / name for name in names] + [EXP.PLAN]
    bindings = {str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}
    for relative,digest in parent['input_sha256'].items():
        if 'MODEL' in Path(relative).name:
            bindings = EXP.PARENT._merge_frozen(bindings, {relative:digest})
    for mapping in (sources, bindings, parent['protected_user_sha256']):
        E._verify_sources(mapping)
    output.mkdir(parents=True)
    plan = dict(schema=PLAN_SCHEMA, UTC=datetime.now(timezone.utc).isoformat(),
        TEST_ids=parent['ids'][46:95], profiles=profiles(), arms=list(ARMS),
        observation_family=EXP.G.OBSERVATION_FAMILY, noise_population=EXP.G.POPULATION,
        selection_profile=EXP.G.SELECTION_PROFILE, raw_observation_generation=EXP.G.RAW_OBSERVATION_GENERATION,
        central_plan_sha256=expected_central_plan_sha256,
        source_sha256=sources, input_sha256=bindings, protected_user_sha256=parent['protected_user_sha256'],
        deferred_journal_sha256={str((EXP.OUTPUT/'INPUTS.jsonl.gz').relative_to(C.ROOT)):inputs['input_sha256'],
            str((EXP.OUTPUT/'PREDICTIONS.jsonl.gz').relative_to(C.ROOT)):prediction['predictions_sha256']},
        population='All original TEST49 x four V31 uniform methods; no source/guard/fallback filtering.',
        source_and_plan_before_diagnostic_TEST_journal_or_numeric_score_decode=True,
        prior_campaign_accounting_scalar_score_exposure=True, postscore_exploratory_not_blinded=True,
        primary_comparison='Saved e in its own native WD and selected pose in its own native WD, both evaluated under the same frozen V31 arm objective; original correspondence retained.',
        alternate_transport='Separate same-unordered-body Rcf=Rphysical@Qtrial.T with centered native t unchanged; native channels can be permuted, so this is not the original saved-e channel hypothesis.',
        operation='Only frozen objective(parts) and evaluate; no optimization, selector, start search, parameter changes or new measurement.',
        cost_terms=['native complete point','unreplaced partial normal','fallback T4',
            'actual named projected edge','selected actual tangent of archived q','bounded O','bounded S','optional C'],
        full_pose_TP_rotation_prior=0., reporting_energy_tolerance=REPORTING_TOLERANCE,
        posthoc_translation_regression_rule='saved e translation<=5cm and V31 arm translation>5cm',
        original_e_added_to_runtime_start_candidate_fallback_or_prior=False,
        S_U_new_q_or_local_queries=0, new_RGB_NN_TLS_GT_decode_or_posefit=0,
        memory='One matched TEST pair at a time from gzip prefix95; CAL46 JSON skipped, REMAINDER143 JSON not parsed.',
        decision_scope='Separate good e excluded by guards, valid e with lower/higher/equal objective and own fallback. No global-optimum, finite-edge-identity or WD/image causal proof.',
        replay_requires_separate_root_postscore_GO=True, default_e_unchanged=True)
    C.write(output/'PLAN.json', plan)
    C.write(output/'PREPARATION_SEAL.json', dict(plan_sha256=C.sha(output/'PLAN.json'),
        source_sha256=sources, input_sha256=bindings, before_diagnostic_TEST_row_decode=True))
    return dict(prepared_TEST=49, arms=4, plan_sha256=C.sha(output/'PLAN.json'), diagnostic_TEST_decodes=0)


def run(output=OUTPUT, *, root_postscore_GO=False, expected_central_plan_sha256=None):
    if root_postscore_GO is not True:
        raise ValueError('Separate root postscore replay GO before any row/model/score IO')
    EXP._verify_plan(expected_central_plan_sha256)
    # The existing prepared source seal includes central bytes. Bind the same
    # caller value explicitly as well before the frozen replay body runs.
    plan = C.read(Path(output).resolve()/'PLAN.json')
    if (plan['central_plan_sha256'] != expected_central_plan_sha256
            or plan['selection_profile'] != EXP.G.SELECTION_PROFILE
            or plan['observation_family'] != EXP.G.OBSERVATION_FAMILY
            or plan['noise_population'] != EXP.G.POPULATION
            or plan['raw_observation_generation'] != EXP.G.RAW_OBSERVATION_GENERATION):
        raise ValueError('Exact newly typed V31 diagnostic source/population/central required')
    return _facade(expected_central_plan_sha256).run(output, root_postscore_GO=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','run'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--central-plan-sha256', required=True)
    parser.add_argument('--root-postscore-GO', action='store_true')
    args = parser.parse_args()
    print(dict(prepare=prepare,run=run)[args.stage](args.output,
        root_postscore_GO=args.root_postscore_GO, expected_central_plan_sha256=args.central_plan_sha256))
