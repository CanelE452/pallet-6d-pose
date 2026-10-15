"""Fixed two-case V30 report; privately reuse frozen V29 drawing arithmetic.

Prepare only pins source/central/noise seals. Actual rendering requires all238
five-arm predictions, fixed scores, independent zero-failure audit and separate
root memory GO. No new image measurement, model call, reference or pose fit.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace

from . import render_geometry_guided_process_v4_29 as PARENT
from . import evaluate_geometry_guided_edges_v4_30 as EXP
from . import prepare_geometry_guided_native_evidence_v4_30 as PREP
from . import calibrate_geometry_guided_edge_noise_v4_30 as CAL

C, E, G = EXP.C, EXP.E, EXP.G
CASES = PARENT.CASES
ARMS = EXP.ARMS
LABELS = ('기존 e (기본값)', 'V30 전체 B/I 선 · native', 'V30 전체 B/I 선 · C',
          'V30 RGB q 보존 · native', 'V30 RGB q 보존 · C')
OUTPUT = PARENT.B / 'geometry_guided_process_v4_30_v1'
PLAN = PARENT.B / 'geometry_guided_process_plan_v4_30/PREDECLARATION.json'
PARENT_SOURCE_SHA256 = 'ec75d1630f3dcfa7cfcbf740a2f70785c46fe86aa99278177e5a20fbb9633f5f'
CENTRAL_SHA256 = '1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'
MODEL_SEAL_SHA256 = '5681a7bf36bc6ce4d7146df48f94ca9791be44c49a77954b7ad2132129b13adc'


def _clone(function, updates):
    result = FunctionType(function.__code__, {**function.__globals__, **updates},
                          function.__name__, function.__defaults__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def _verify_map(mapping):
    E._verify_sources(mapping)


def _sources():
    sources = {}
    for module in (PARENT, PARENT.OBS, PARENT.OBS.V, PARENT.P, EXP, EXP.PARENT,
                   G, EXP.AUTH, PREP, PREP.RERANK, CAL, CAL.M):
        path = Path(module.__file__).resolve()
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    for path in (Path(__file__).resolve(), Path(__file__).with_name('test_render_geometry_guided_process_v4_30.py')):
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    return sources


def prepare():
    if PLAN.exists(): raise FileExistsError('Preserve every renderer declaration')
    if (EXP.OUTPUT / 'SCORE_SEAL.json').exists():
        raise ValueError('Fix cases/source before new V30 scores')
    if C.sha(PARENT.__file__) != PARENT_SOURCE_SHA256:
        raise ValueError('Exact immutable V29 renderer required')
    if C.sha(EXP.PLAN) != CENTRAL_SHA256 or C.sha(CAL.OUTPUT / 'MODEL_SEAL.json') != MODEL_SEAL_SHA256:
        raise ValueError('Exact central V30 method and separately fitted noise seals required')
    central = C.read(EXP.PLAN)
    if central['arms'] != list(ARMS) or central['selection_profile'] != G.SELECTION_PROFILE:
        raise ValueError('Exact uniform V30 arms/profile required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        _verify_map(central[key])
    sources = EXP.PARENT._merge_frozen(central['source_sha256'], _sources())
    inputs = EXP.PARENT._merge_frozen(central['input_sha256'], {
        str(EXP.PLAN.relative_to(C.ROOT)): CENTRAL_SHA256,
        str((CAL.OUTPUT / 'MODEL_SEAL.json').relative_to(C.ROOT)): MODEL_SEAL_SHA256})
    _verify_map(sources)
    PLAN.parent.mkdir(parents=True, exist_ok=False)
    C.write(PLAN, dict(schema='FIXED_TWO_USER_CASES_ASSOCIATED_EDGE_PROCESS_PLAN_V30',
        UTC=datetime.now(timezone.utc).isoformat(), cases=list(CASES), arms=list(ARMS),
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=central['protected_user_sha256'],
        frozen_parent_renderer_sha256=PARENT_SOURCE_SHA256, central_method_sha256=CENTRAL_SHA256,
        observation_output=str(PREP.OBS.relative_to(C.ROOT)), observation_protocol='FINAL_PROTOCOL.json',
        prediction_output=str(EXP.OUTPUT.relative_to(C.ROOT)), model_seal_sha256=MODEL_SEAL_SHA256,
        observation_family=G.OBSERVATION_FAMILY, population=G.POPULATION,
        raw_observation_generation=G.RAW_OBSERVATION_GENERATION, selection_profile=G.SELECTION_PROFILE,
        stage_order=['original RGB', 'frozen first N3 and gated BI q', 'actual target mask and O/U ranges',
            'V30 graph-selected actual chains before source-policy AUTH activation',
            'same five separately saved final methods; active or fallback-plan lines distinguished'],
        drawing='Private copy of frozen parent function; only labels/schema/page text/bindings change.',
        legacy_v29_cost_plan_key_is_wire_alias_only=True,
        inherited_top_selected_physical_edges_is_ignored=True,
        selected_chain_refs_graph_and_active_plan_are_authoritative=True,
        no_optimizer_trajectory_invented=True, S_U_point_markers=0, GT_wire=0,
        S_image_queries_and_point_costs=0, frozen_coarse_S_regularizer_remains=True,
        U_regions_only=True, target_mask_absence_is_unknown=True,
        new_RGB_NN_NMS_TLS_posefit_or_reference_decode=0,
        source_and_fixed_case_selection_before_new_V30_primary_scores=True,
        rendering_requires_full238_prediction_score_independent_audit_and_explicit_root_memory_GO=True,
        no_default_adoption=True))
    C.write(PLAN.with_name('PLAN_SEAL.json'), dict(plan_sha256=C.sha(PLAN),
        source_sha256=sources, input_sha256=inputs, new_primary_rows_or_scores_read=0))
    print(dict(plan=str(PLAN), sha256=C.sha(PLAN), new_primary_rows_or_scores_read=0), flush=True)


def validate_completed_metadata(protocol, prediction, audit, obsprotocol, obsseal):
    """Pure shape/profile gate; numeric scores are not consulted here."""
    ids = protocol['ids']
    if (len(ids) != 238 or len(set(ids)) != 238 or protocol['arms'] != list(ARMS)
            or prediction['rows'] != 238 or prediction['ordered_ids'] != ids
            or prediction['arms'] != list(ARMS) or prediction['arm_count'] != 5
            or not prediction['all_declared_arms_complete']
            or tuple(prediction.get(p) for p in ('CAL', 'TEST', 'REMAINDER', 'severe')) != (46, 49, 143, 0)
            or audit['rows'] != 238 or audit['failure_count'] or audit['failures']
            or obsprotocol['ids'] != ids or obsseal['rows'] != 238 or obsseal['ordered_ids'] != ids):
        raise ValueError('Complete exact all238/five-arm predictions, observations and zero-failure audit required')
    for item in (protocol, obsprotocol, obsseal):
        if (item.get('selection_profile') != G.SELECTION_PROFILE
                or item.get('raw_observation_generation') != G.RAW_OBSERVATION_GENERATION):
            raise ValueError('Exact V30 profile and unchanged raw V29 generation required')
    if protocol.get('noise_observation_family') != G.OBSERVATION_FAMILY:
        raise ValueError('Separately fitted V30 noise family required')


def _require_plan(plan):
    if (plan.get('observation_family') != G.OBSERVATION_FAMILY or plan.get('population') != G.POPULATION
            or plan.get('selection_profile') != G.SELECTION_PROFILE
            or plan.get('raw_observation_generation') != G.RAW_OBSERVATION_GENERATION):
        raise ValueError('Displayed active/fallback plan must be V30, not a legacy key version guess')
    binding = plan['CAL_noise_binding']
    if binding['available']:
        noise = binding['noise']['noise_provenance']; tangent = binding['tangent']
        for item in (noise, tangent):
            if (item.get('observation_family') != G.OBSERVATION_FAMILY
                    or item.get('selection_profile') != G.SELECTION_PROFILE
                    or item.get('raw_observation_generation') != G.RAW_OBSERVATION_GENERATION
                    or item.get('source_policy') != plan['source_policy']):
                raise ValueError('Displayed factors must retain policy-specific fitted V30 model provenance')


def _active_plan_lines(result, row, arm):
    if arm == ARMS[0]: return {}, []
    solution = result.get('solution', {})
    plan = solution.get('v30_cost_plan', solution.get('v29_cost_plan'))
    if plan is None: plan = result['new_geometry_edge_cost_plans'][result['WD_family']]
    _require_plan(plan)
    if 'v30_cost_plan' in solution and 'v29_cost_plan' in solution:
        if E._digest(solution['v30_cost_plan']) != E._digest(solution['v29_cost_plan']):
            raise ValueError('Internal wire aliases must describe the same V30 plan')
    # A private shallow view supplies the frozen drawing helper's legacy alias.
    # No source result, measured chain or numerical plan is mutated.
    view = {**result, 'solution': {**solution, 'v29_cost_plan': plan}}
    return PARENT._active_plan_lines(view, row, arm)


def _completed():
    plan = C.read(PLAN)
    if plan['cases'] != list(CASES) or plan['arms'] != list(ARMS):
        raise ValueError('Fixed same two cases and five methods required')
    if C.sha(PLAN) != C.read(PLAN.with_name('PLAN_SEAL.json'))['plan_sha256']:
        raise ValueError('Renderer source/plan declaration changed')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        _verify_map(plan[key])
    protocol = C.read(EXP.OUTPUT / 'PROTOCOL.json'); pred = C.read(EXP.OUTPUT / 'PREDICTION_SEAL.json')
    inputseal = C.read(EXP.OUTPUT / 'INPUT_SEAL.json')
    score = C.read(EXP.OUTPUT / 'SCORE_SEAL.json')
    auditpath = EXP.OUTPUT / 'independent_root_audit_v1/AUDIT.json'
    audit = C.read(auditpath); aseal = C.read(auditpath.with_name('AUDIT_SEAL.json'))
    op = C.read(PREP.OBS / 'FINAL_PROTOCOL.json'); os = C.read(PREP.OBS / 'OBSERVATION_SEAL.json')
    validate_completed_metadata(protocol, pred, audit, op, os)
    if (C.sha(EXP.OUTPUT / 'PROTOCOL.json') != pred['protocol_sha256']
            or C.sha(EXP.OUTPUT / 'INPUT_SEAL.json') != pred['input_seal_sha256']
            or C.sha(EXP.OUTPUT / 'INPUTS.jsonl.gz') != inputseal['input_sha256']
            or C.sha(EXP.OUTPUT / 'PREDICTIONS.jsonl.gz') != pred['predictions_sha256']
            or C.sha(EXP.OUTPUT / 'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']
            or C.sha(auditpath) != aseal['receipt_sha256']
            or audit['prediction_seal']['sha256'] != C.sha(EXP.OUTPUT / 'PREDICTION_SEAL.json')
            or C.sha(PREP.OBS / 'FINAL_PROTOCOL.json') != os['protocol_sha256']
            or C.sha(PREP.OBS / 'OBSERVATIONS.jsonl.gz') != os['observations_sha256']
            or C.sha(CAL.OUTPUT / 'MODEL_SEAL.json') != MODEL_SEAL_SHA256):
        raise ValueError('Current prediction/audit/score/FINAL_PROTOCOL/model seals must agree')
    for partition, count, ids in [('CAL', 46, protocol['ids'][:46]), ('REST', 192, protocol['ids'][46:])]:
        path = PREP.OBS / (partition + '_OBSERVATION_SEAL.json'); part = C.read(path)
        if part['rows'] != count or part['ordered_ids'] != ids or os['partition_seals'][partition] != C.sha(path):
            raise ValueError('Both exact complete CAL46/REST192 observation seals required')
    for ancestor in (protocol, op):
        for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'): _verify_map(ancestor[key])
    observations, od = PARENT._exact_rows(PREP.OBS / 'OBSERVATIONS.jsonl.gz', CASES)
    predictions, pd = PARENT._exact_rows(EXP.OUTPUT / 'PREDICTIONS.jsonl.gz', CASES)
    tasks, td = PARENT._exact_rows(EXP.OUTPUT / 'INPUTS.jsonl.gz', CASES)
    scores = {}
    for partition, ident in [('TEST', CASES[1]), ('REMAINDER', CASES[0])]:
        path = EXP.OUTPUT / (partition + '_SCORE.json')
        if C.sha(path) != score['score_sha256'][partition]: raise ValueError('Sealed fixed metrics changed')
        matching = [r for r in C.read(path)['rows'] if r['id'] == ident]
        if len(matching) != 1: raise ValueError('Exactly one fixed-case scalar score required')
        scores[ident] = matching[0]
    for ident in CASES:
        row, frame, task = observations[ident], predictions[ident], tasks[ident]
        PREP.validate_runtime_join(task, row)
        if (row.get('selection_profile') != G.SELECTION_PROFILE
                or row['observation'].get('selection_profile') != G.SELECTION_PROFILE
                or task.get('v30_selection_profile') != G.SELECTION_PROFILE
                or frame.get('selection_profile') != G.SELECTION_PROFILE
                or frame.get('noise_observation_family') != G.OBSERVATION_FAMILY
                or frame.get('raw_observation_generation') != G.RAW_OBSERVATION_GENERATION
                or row['observation'].get('observation_generation') != G.RAW_OBSERVATION_GENERATION
                or row['observation']['association_configuration'] != row['observation']['chain_observation']['configuration']
                or task['v29_observation'] != row['observation']
                or task['baseline'] != frame['arms'][ARMS[0]]
                or set(frame['arms']) != set(ARMS)):
            raise ValueError('Exact two-row V30 observation/task/prediction source join required')
    return observations, predictions, tasks, scores, dict(observation_rows=od, prediction_rows=pd, task_rows=td)


def _write_product(path, value):
    if Path(path).name in ('PROCESS_DATA.json', 'RENDER_SEAL.json'):
        value = {**value, 'selection_profile': G.SELECTION_PROFILE,
            'observation_family': G.OBSERVATION_FAMILY, 'population': G.POPULATION,
            'raw_observation_generation': G.RAW_OBSERVATION_GENERATION,
            'final_observation_protocol': E._binding(PREP.OBS / 'FINAL_PROTOCOL.json'),
            'policy_specific_CAL_model_seal_sha256': MODEL_SEAL_SHA256,
            'frozen_parent_renderer_sha256': PARENT_SOURCE_SHA256,
            'legacy_v29_cost_plan_is_internal_alias_only': True}
    C.write(path, value)


def _renderer_function():
    if C.sha(PARENT.__file__) != PARENT_SOURCE_SHA256:
        raise ValueError('Frozen V29 drawing implementation changed')
    tree = ast.parse(inspect.getsource(PARENT.render)); changed = []
    class TextOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if not isinstance(node.value, str): return node
            old = node.value; new = old
            if old in ('FIXED_TWO_CASES_GEOMETRY_GUIDED_PROCESS_DATA_V29', 'FIXED_TWO_CASES_GEOMETRY_GUIDED_RENDER_SEAL_V29'):
                new = old.replace('_V29', '_V30')
            elif old.startswith('<!doctype html>'):
                new = old.replace('v29 실제 선 관측과 최종 자세', 'v30 B/I 대응 선과 최종 자세')
                new = new.replace('기하 가이드로 찾은 실제 선과 저장된 최종 자세', 'B/I 대응 검사를 거친 실제 선과 저장된 V30 최종 자세')
                new = new.replace('S는 모델 wire/PnP 역할만, U는 넓은 범위만 사용합니다.',
                    'S의 영상 점·국소 질의·2D 점 비용은 없고 모델 wire/PnP와 기존 bounded coarse-S 항을 사용합니다. U는 넓은 범위만 사용합니다.')
            elif old.startswith('청록=일반 선 · 노랑=부분 선;'):
                new = 'V30 원 B/I–무한선 법선 43px 검사 통과 · ' + old
            if new != old:
                changed.append(old); return ast.copy_location(ast.Constant(new), node)
            return node
    tree = TextOnly().visit(tree); ast.fix_missing_locations(tree)
    if len(changed) != 4: raise ValueError('Only four declared schema/page/caption constants may change')
    observation = SimpleNamespace(OUTPUT=PREP.OBS, OLD=PARENT.OBS.OLD)
    storage = SimpleNamespace(**{**vars(C), 'write': _write_product})
    namespace = {**PARENT.render.__globals__, '__file__': __file__, 'C': storage,
        'OBS': observation, 'OUTPUT': OUTPUT, 'PLAN': PLAN, 'PREDICTIONS': EXP.OUTPUT,
        'CASES': CASES, 'ARMS': ARMS, 'LABELS': LABELS, '_completed': _completed,
        '_active_plan_lines': _active_plan_lines}
    exec(compile(tree, __file__, 'exec'), namespace)
    return namespace['render']


def render(*, root_postscore_audit_memory_GO=False):
    if root_postscore_audit_memory_GO is not True:
        raise ValueError('Wait for explicit root postscore/audit/memory GO before IO')
    return _renderer_function()(root_postscore_audit_memory_GO=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('prepare', 'render'))
    parser.add_argument('--root-postscore-audit-memory-GO', action='store_true'); args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    else: render(root_postscore_audit_memory_GO=args.root_postscore_audit_memory_GO)
