"""Postscore TEST49 e-vs-selected decomposition under frozen V29 objectives.

This is a diagnostic, not a new method. The saved e pose is never added to a
runtime start, candidate, fallback or prior. Both saved WD correspondences are
preserved. A separately labelled physical-body transport may change native
channel correspondence; it never replaces the primary saved-e-WD comparison.
Only one matched TEST row is decoded at a time; CAL/REMAINDER rows, images,
reference annotations, new measurement and optimization are not decoded/run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import evaluate_geometry_guided_edges_v4_29 as M
from . import diagnose_frozen_e_pose_under_v24_v4_25 as PREVIOUS

C, E = M.C, M.E
OUTPUT = M.OUTPUT.parent/'frozen_e_geometry_guided_objective_diagnosis_v4_29b'
ARMS = tuple(M.ARMS[1:])
REPORTING_TOLERANCE = 1e-8
_proper = PREVIOUS._proper
_TEST_prefix = PREVIOUS._TEST_prefix
transport_physical_pose_to_family = PREVIOUS.transport_physical_pose_to_family
_physical_body_gap = PREVIOUS._physical_body_gap


def profiles():
    expected = ('GEOMETRY_EDGE_ALL_BI_NATIVE', 'GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
                'GEOMETRY_EDGE_SOURCE_NATIVE', 'GEOMETRY_EDGE_SOURCE_CONTAINMENT')
    if ARMS != expected:
        raise ValueError('Exact frozen V29 four-arm inventory required')
    return {arm:dict(source_policy=policy, containment=bool(index),
        no_full_TP_rotation_pose_prior=True, both_WD_families_allowed=True)
        for policy, arms in M.POLICY_ARMS.items() for index, arm in enumerate(arms)}


def validate_saved_pose_basis(pose, family):
    """Legacy centroid is native t only for these centered cuboids."""
    xyz = np.asarray(family['native_xyz'], float)
    if (xyz.shape != (8, 3) or not np.isfinite(xyz).all()
            or not np.allclose(xyz.mean(0), 0., atol=1e-12, rtol=0)):
        raise ValueError('Centered frozen native XYZ required; do not equate arbitrary t and centroid')
    R, Q = _proper(pose['R_cf']), _proper(family['physical_registration_Q'])
    physical = _proper(pose['R_physical'])
    t = np.asarray(pose['centroid'], float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('Finite saved native t required')
    if not np.allclose(R@Q, physical, atol=1e-10, rtol=0):
        raise ValueError('Saved R_cf/Q/physical rotation correspondence changed')
    if 'native_xyz' in pose and not np.allclose(pose['native_xyz'], xyz, atol=1e-12, rtol=0):
        raise ValueError('Saved pose native geometry differs from its own WD')
    return R, t


def transport_diagnostic_pose(pose, original_family, trial_family):
    validate_saved_pose_basis(pose, original_family)
    gap = _physical_body_gap(original_family, trial_family)
    if gap > 1e-12:
        return None, None, dict(available=False, unordered_physical_body_gap_m=gap,
            reason='Registered cuboid sets differ; numeric pose insertion would change the body')
    # Also validates the trial centered basis, without using its channel map as
    # a substitute for the original saved-e native correspondence.
    xyz = np.asarray(trial_family['native_xyz'], float)
    if not np.allclose(xyz.mean(0), 0., atol=1e-12, rtol=0):
        return None, None, dict(available=False, reason='Trial geometry is not centered')
    R, t, audit = transport_physical_pose_to_family(pose, trial_family)
    return R, t, {**audit, 'available':True, 'unordered_physical_body_gap_m':gap,
        'same_unordered_physical_box_only':True,
        'alternate_WD_native_channels_may_be_permuted':True,
        'not_the_original_saved_e_channel_correspondence':original_family['name']!=trial_family['name']}


def source_inventory(task):
    sources = Counter(); partial_sources = Counter()
    partial = {b['native_corner'] for b in task['boundary_merged']}
    for measurement in task['spec']['visible']:
        k = measurement['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B', 'I'):
            raise ValueError('NonS BI gate before archived source fields')
        key = (task['states'][k], measurement['source'])
        sources[key] += 1
        if k in partial:
            partial_sources[key] += 1
    encode = lambda counts:[dict(state=state, source=source, count=count)
        for (state,source),count in sorted(counts.items())]
    return dict(archived_BI_sources=encode(sources), original_partial_BI_sources=encode(partial_sources),
        S_U_raw_coordinate_reads=0, actual_new_point_or_line_queries=0)


def plan_summary(plan):
    groups = plan['active_groups']; corners = plan['corner_plans']
    return dict(source_policy=plan['source_policy'], active_physical_edge_count=len(groups),
        new_owner_count=len(plan['new_owner_ids']), new_owner_ids=plan['new_owner_ids'],
        owner_modes=dict(Counter(c['mode'] for c in corners)),
        owner_sources=dict(Counter(c['source_archive']['original_measurement_source'] for c in corners)),
        active_edges=[dict(physical_native_edge=sorted([g['native_corner'],g['native_neighbor']]),
            component_id=g['component_id'], proposal_grade=g.get('proposal_grade'),
            owners=g['activation_owners']) for g in groups],
        owner_plans=[dict(native_corner=c['native_corner'], mode=c['mode'],
            source=c['source_archive']['original_measurement_source'],
            selected_component_ids=c['selected_component_ids']) for c in corners],
        no_new_precise_image_point=True)


def native_buckets(records):
    kinds = Counter(); factors = defaultdict(float)
    for r in records:
        value = float(r['weighted_robust_cost'])
        if not np.isfinite(value) or value < 0.:
            raise ValueError('Finite nonnegative frozen native cost required')
        kinds[r['kind']] += value
        factors[(r['kind'],r.get('native_corner'),r.get('native_neighbor'))] += value
    return dict(kinds), [dict(kind=k[0],native_corner=k[1],native_neighbor=k[2],cost=value)
        for k,value in sorted(factors.items(),key=lambda item:str(item[0]))]


def evaluate_parts(problem, R, t):
    """Frozen parts/evaluate only. No selector, start search or optimizer."""
    try:
        checked = problem.evaluate(R,t)
        p = M.BOUNDED.G._pose_vector(np.asarray(R),np.asarray(t),problem.scale)
        total, _, parts = problem.objective(p)
        records = parts['native_costs']
        native = float(parts['native_observation_objective'])
        regularizers = parts['regularizer_costs']
        if [r['kind'] for r in regularizers] != list(M.BOUNDED.REGULARIZER_KINDS):
            raise ValueError('Only frozen bounded O/S regularizers permitted')
        ov, sv = [float(r['weighted_cost']) for r in regularizers]
        shape = float(parts.get('global_containment_cost',0.))
        weight = float(parts.get('global_containment_weight',0.))
        values = [total,native,ov,sv,shape,weight]
        if not np.isfinite(values).all() or min(values) < 0.:
            raise ValueError('Finite nonnegative decomposition required')
        summed = native+ov+sv+weight*shape
        native_sum = sum(float(r['weighted_robust_cost']) for r in records)
        gap = max(abs(float(total)-summed),abs(native-native_sum))
        if checked.get('objective') is not None:
            gap = max(gap,abs(float(checked['objective'])-float(total)))
        if gap > 1e-8*max(1.,abs(float(total))):
            raise ValueError('Frozen objective/evaluate/native ledger mismatch')
        kinds, factors = native_buckets(records)
        audits = {key:checked.get(key) for key in ('visible_protection_audit','joint_constraint_audit',
            'U_region_constraint_audit','boundary_observation_audit','actual_edge_support_constraint_audit')}
        return dict(accepted_current_guards=bool(checked['accepted']), finite_objective=True,
            total=float(total),native=native,O=ov,S=sv,P=0.,C=shape,C_weight=weight,
            weighted_C=weight*shape,native_by_kind=kinds,native_by_factor=factors,
            native_factor_counts=dict(Counter(r['kind'] for r in records)),
            native_costs=records,minimum_joint_slack=checked.get('minimum_joint_slack'),
            guard_audits=audits,reason=checked.get('reason'),
            maximum_internal_ledger_difference=gap,
            current_noise_and_tangent_policy_used=True,old_symmetric_mask_used=False,
            S_U_new_image_queries=0,pose_optimization_or_runtime_candidate_insertions=0)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
        return dict(accepted_current_guards=False,finite_objective=False,total=None,
            reason=str(error),numeric_failure_not_global_infeasibility_proof=True)


def comparison(evaluation, selected, *, selected_accepted):
    if not selected_accepted:
        return 'selected_is_own_fallback_not_a_solved_pose'
    if not evaluation.get('finite_objective') or not selected.get('finite_objective'):
        return 'diagnostic_numeric_evaluation_unavailable'
    if not evaluation['accepted_current_guards']:
        return 'saved_e_excluded_by_current_hard_or_numeric_guards'
    return PREVIOUS.classify_comparison(True,True,evaluation['total'],selected['total'],REPORTING_TOLERANCE)


def make_problem(task, family, models, policy, prior_inputs, *, containment):
    core, _ = M._problem(task,family,models,policy)
    regions, oparameters, original_S = prior_inputs
    rebound = M.BOUNDED.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])
    bounded = M.BOUNDED.BoundedGeometricPriorsProblem(core,regions,oparameters,rebound)
    cfg = M.BASE.OUTLINE._outline_config(task['outline']['configuration'])
    problem = M.CONTAIN.ContinuousContainmentProblem(bounded,task['outline'],cfg) if containment else bounded
    return problem, core.plan


def _read_parent_small():
    protocol = C.read(M.OUTPUT/'PROTOCOL.json')
    inputs = C.read(M.OUTPUT/'INPUT_SEAL.json')
    prediction = C.read(M.OUTPUT/'PREDICTION_SEAL.json')
    score = C.read(M.OUTPUT/'SCORE_SEAL.json')
    auditpath = M.OUTPUT/'independent_root_audit_v1/AUDIT.json'
    audit, aseal = C.read(auditpath), C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if (protocol['arms']!=list(M.ARMS) or inputs['rows']!=238 or prediction['rows']!=238
            or prediction['TEST']!=49 or prediction['arms']!=list(M.ARMS)
            or not prediction['all_declared_arms_complete'] or prediction['arm_count']!=5
            or inputs['protocol_sha256']!=C.sha(M.OUTPUT/'PROTOCOL.json')
            or prediction['protocol_sha256']!=C.sha(M.OUTPUT/'PROTOCOL.json')
            or prediction['input_seal_sha256']!=C.sha(M.OUTPUT/'INPUT_SEAL.json')
            or score['prediction_seal']['sha256']!=C.sha(M.OUTPUT/'PREDICTION_SEAL.json')
            or score['score_sha256']['TEST']!=C.sha(M.OUTPUT/'TEST_SCORE.json')
            or audit['rows']!=238 or audit['failure_count'] or audit['failures']
            or C.sha(auditpath)!=aseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(M.OUTPUT/'PREDICTION_SEAL.json')):
        raise ValueError('Completed immutable V29 all238/five-arm score and zero-failure audit required')
    ids = protocol['ids']
    if len(ids)!=238 or len(set(ids))!=238 or prediction['ordered_ids']!=ids:
        raise ValueError('Exact complete original238 order required')
    return protocol, inputs, prediction


def prepare(output=OUTPUT):
    """Postscore plan, before this diagnostic decodes any TEST row."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve all previous diagnostic attempts')
    parent, inputs, prediction = _read_parent_small()
    sources = dict(parent['source_sha256'])
    for module in (M,M.G,M.AUTH,M.PREP,M.CAL,M.CAL.M,M.BOUNDED,M.CONTAIN,
                   M.CONTAIN.FACTOR,PREVIOUS):
        path = Path(module.__file__).resolve()
        sources = M._merge_frozen(sources,{str(path.relative_to(C.ROOT)):C.sha(path)})
    for path in (Path(__file__).resolve(),Path(__file__).with_name('test_diagnose_frozen_e_pose_under_v29_v4_29b.py')):
        sources = M._merge_frozen(sources,{str(path.relative_to(C.ROOT)):C.sha(path)})
    names = ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json',
        'TEST_SCORE.json','independent_root_audit_v1/AUDIT.json','independent_root_audit_v1/AUDIT_SEAL.json')
    paths = [M.OUTPUT/name for name in names]
    bindings = {str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}
    # Recheck exact model bytes already bound by the original runtime protocol,
    # without opening labels or the full observation ancestry again.
    for relative,digest in parent['input_sha256'].items():
        if 'MODEL' in Path(relative).name:
            bindings = M._merge_frozen(bindings,{relative:digest})
    for mapping in (bindings,):
        E._verify_sources(mapping)
    for mapping in (sources,parent['protected_user_sha256']):
        E._verify_sources(mapping)
    output.mkdir(parents=True)
    plan = dict(schema='POSTSCORE_FROZEN_E_V29_OBJECTIVE_DIAGNOSIS_PLAN_V29B',
        UTC=datetime.now(timezone.utc).isoformat(),population='All original TEST49; no guard/source/fallback exclusions',
        TEST_ids=parent['ids'][46:95],profiles=profiles(),arms=list(ARMS),
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=parent['protected_user_sha256'],
        deferred_journal_sha256={str((M.OUTPUT/'INPUTS.jsonl.gz').relative_to(C.ROOT)):inputs['input_sha256'],
            str((M.OUTPUT/'PREDICTIONS.jsonl.gz').relative_to(C.ROOT)):prediction['predictions_sha256']},
        source_and_plan_before_diagnostic_TEST_journal_or_numeric_score_decode=True,
        prior_campaign_accounting_scalar_score_exposure=True,postscore_exploratory_not_blinded=True,
        primary_comparison='Original saved e native WD versus selected original native WD, same frozen arm objective; both WD families are allowed by that arm.',
        alternate_transport='Separate same-unordered-box Rcf=Rphysical@Qtrial.T with centered t unchanged; may change native channel correspondence, never labelled original e identity.',
        operation='Frozen objective(parts)/evaluate only; no selector, optimizer, start search, factor tuning or new measurement.',
        cost_terms=['native complete point','unreplaced partial normal','old fallback T4',
            'new actual projected physical edge','selected measured-line archived-q tangent','bounded O','bounded S','optional C'],
        full_pose_TP_rotation_prior=0.,reporting_energy_tolerance=REPORTING_TOLERANCE,
        posthoc_translation_regression_rule='saved e translation<=5cm and current arm translation>5cm',
        original_e_added_to_runtime_start_candidate_fallback_or_prior=False,
        new_pixels_NN_TLS_GT_decode_or_pose_optimization=0,S_U_new_q_or_local_queries=0,
        memory='Stream gzip prefix95; skip CAL46 JSON; decode only matched TEST49 input/prediction pairs, release each pair. No remaining143 JSON decode.',
        decision_scope='Eligible lower-energy e means bounded runtime candidate/search coverage gap, not global optimum. Lower-energy selected does not identify image/WD causality or guarantee correctness.',
        replay_requires_separate_root_postscore_GO=True,default_e_unchanged=True)
    C.write(output/'PLAN.json',plan)
    C.write(output/'PREPARATION_SEAL.json',dict(plan_sha256=C.sha(output/'PLAN.json'),
        source_sha256=sources,input_sha256=bindings,before_diagnostic_TEST_row_decode=True))
    print(dict(prepared_TEST=49,arms=4,plan_sha256=C.sha(output/'PLAN.json'),diagnostic_TEST_decodes=0),flush=True)


def _public(evaluation):
    return {k:v for k,v in evaluation.items() if k!='native_costs'}


def run(output=OUTPUT, *, root_postscore_GO=False):
    if root_postscore_GO is not True:
        raise ValueError('Separate root postscore replay GO required before any row/model/score decode')
    output = Path(output).resolve()
    if (output/'POSTSEAL_INPUT_BINDINGS_V1.json').exists() or (output/'DIAGNOSIS.json').exists():
        raise FileExistsError('Preserve completed or partial diagnostic')
    plan, seal = C.read(output/'PLAN.json'), C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PLAN.json')!=seal['plan_sha256'] or plan['profiles']!=profiles() or plan['arms']!=list(ARMS):
        raise ValueError('Predeclared source/profile/TEST plan changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256','deferred_journal_sha256'):
        E._verify_sources(plan[key])
    parent, _, _ = _read_parent_small()
    if plan['TEST_ids']!=parent['ids'][46:95]:
        raise ValueError('No original TEST49 row exclusions')
    C.write(output/'POSTSEAL_INPUT_BINDINGS_V1.json',dict(plan=E._binding(output/'PLAN.json'),
        deferred_journal_sha256=plan['deferred_journal_sha256'],before_diagnostic_TEST_decode=True,
        original_runtime_or_results_changes=0))
    models = M._models()
    score = C.read(M.OUTPUT/'TEST_SCORE.json')
    if score['count']!=49 or [r['id'] for r in score['rows']]!=plan['TEST_ids']:
        raise ValueError('Exact all49 saved scalar score required')
    scored = {r['id']:r for r in score['rows']}
    rows = []; counts = {arm:Counter() for arm in ARMS}; regressions = {arm:Counter() for arm in ARMS}
    maximum = 0.
    for task,row in zip_longest(_TEST_prefix(M.OUTPUT/'INPUTS.jsonl.gz',plan['TEST_ids']),
            _TEST_prefix(M.OUTPUT/'PREDICTIONS.jsonl.gz',plan['TEST_ids'])):
        if task is None or row is None or task['id']!=row['id'] or set(row['arms'])!=set(M.ARMS):
            raise ValueError('Exact matched all49 input/prediction rows required')
        before = E._digest(task)
        if task['baseline']!=row['arms'][M.ARMS[0]]:
            raise ValueError('Frozen e alias differs')
        e_name = task['baseline']['WD_family']; e_family = task['families'][e_name]
        e_pose = task['baseline']['actual_pose']; e_R,e_t = validate_saved_pose_basis(e_pose,e_family)
        prior_inputs = M.OLD._prior_inputs(task,models)
        results = {}; problem_cache = {}; evaluation_cache = {}; plan_cache = {}
        for arm,profile in profiles().items():
            policy = profile['source_policy']; containment = profile['containment']
            selected = row['arms'][arm]; selected_name = selected['WD_family']
            selected_family = task['families'][selected_name]
            R,t = validate_saved_pose_basis(selected['actual_pose'],selected_family)
            trials = {}; transports = {}
            for family_name,family in task['families'].items():
                key = (policy,containment,family_name)
                if key not in problem_cache:
                    problem,current_plan = make_problem(task,family,models,policy,prior_inputs,containment=containment)
                    problem_cache[key] = problem; plan_cache[key] = current_plan
                if family_name==e_name:
                    transported_R,transported_t = e_R,e_t
                    transport = dict(available=True,original_saved_e_native_correspondence=True,
                        unordered_physical_body_gap_m=0.,native_channel_identity_preserved=True)
                else:
                    transported_R,transported_t,transport = transport_diagnostic_pose(e_pose,e_family,family)
                transports[family_name] = transport
                if transport['available']:
                    evaluation_cache[key] = evaluate_parts(problem_cache[key],transported_R,transported_t)
                    trials[family_name] = _public(evaluation_cache[key])
                else:
                    trials[family_name] = dict(unavailable=True,reason=transport['reason'])
            key = (policy,containment,selected_name)
            current = evaluate_parts(problem_cache[key],R,t)
            old = evaluation_cache[(policy,containment,e_name)]
            current_plan = plan_cache[key]
            if E._digest(current_plan)!=E._digest(selected['solution'].get('v29_cost_plan',current_plan)):
                raise ValueError('Fresh current objective source ownership differs from saved plan')
            if selected['accepted']:
                if not current['accepted_current_guards'] or not current['finite_objective']:
                    raise ValueError('Saved accepted selected pose fails current objective/guards')
                gap = abs(current['total']-float(selected['solution']['objective']))
                maximum = max(maximum,gap,current['maximum_internal_ledger_difference'])
                if gap>1e-8*max(1.,abs(current['total'])):
                    raise ValueError('Frozen selected objective differs from saved scalar')
            label = comparison(old,current,selected_accepted=selected['accepted'])
            counts[arm][label] += 1
            metrics = scored[task['id']]['metrics']
            regression = metrics[M.ARMS[0]]['translation_cm']<=5.<metrics[arm]['translation_cm']
            if regression:
                regressions[arm][label] += 1
            scalar_preference = bool(selected['accepted'] and old.get('finite_objective')
                and current.get('finite_objective') and old['total']>current['total']+REPORTING_TOLERANCE)
            results[arm] = dict(source_policy=policy,containment=containment,
                saved_e_family=e_name,selected_family=selected_name,
                both_saved_families_allowed_by_this_arm=True,
                primary_e_evaluation=_public(old),selected_evaluation=_public(current),
                e_transport_diagnostics=transports,e_physical_body_trials_by_WD=trials,
                classification=label,selected_accepted=selected['accepted'],
                selected_origin=selected.get('selected_endpoint_origin'),fallback_audit=selected.get('fallback_audit'),
                selected_candidate_name=selected['candidate_selection']['selected_name'],
                selected_active_plan=plan_summary(current_plan),
                e_original_WD_active_plan=plan_summary(plan_cache[(policy,containment,e_name)]),
                saved_scalar_metrics=dict(e=metrics[M.ARMS[0]],selected=metrics[arm]),
                posthoc_e5cm_to_selected_gt5cm_regression=bool(regression),
                finite_objective_prefers_selected_even_if_e_guard_excluded=scalar_preference,
                eligible_objective_prefers_worse_selected=bool(regression and label=='current_objective_prefers_selected_over_diagnostic_e'),
                native_and_model_term_e_minus_selected={k:old[k]-current[k] for k in
                    ('native','O','S','P','weighted_C','total')} if old.get('finite_objective') and current.get('finite_objective') else None,
                new_e_runtime_candidate_start_prior_or_fallback_insertions=0)
        if E._digest(task)!=before:
            raise ValueError('Diagnostic altered frozen current row')
        rows.append(dict(id=task['id'],partition='TEST',saved_e_family=e_name,
            source_inventory=source_inventory(task),arms=results))
        print(str(len(rows))+'/49 '+task['id'],flush=True)
        # No prior row's observation/problem cache survives to the next decode.
        del task,row,problem_cache,evaluation_cache,plan_cache,prior_inputs,problem,selected,current_plan
    if len(rows)!=49 or [r['id'] for r in rows]!=plan['TEST_ids']:
        raise ValueError('All49/fallback rows must remain')
    for mapping in (plan['source_sha256'],plan['input_sha256'],plan['protected_user_sha256'],plan['deferred_journal_sha256']):
        E._verify_sources(mapping)
    data = dict(schema='POSTSCORE_FROZEN_E_V29_OBJECTIVE_DIAGNOSIS_V29B',rows=rows,count=49,arm_count=4,
        summary=dict(classifications={a:dict(c) for a,c in counts.items()},
            regression_classifications={a:dict(c) for a,c in regressions.items()},
            eligible_objective_prefers_worse_selected_count={a:sum(r['arms'][a]['eligible_objective_prefers_worse_selected'] for r in rows) for a in ARMS},
            maximum_selected_saved_objective_difference=maximum),
        plan=E._binding(output/'PLAN.json'),preparation_seal=E._binding(output/'PREPARATION_SEAL.json'),
        original_e_native_correspondence_primary=True,alternate_transport_different_channel_hypothesis_explicit=True,
        runtime_or_parameter_changes=0,new_pixels_NN_TLS_GT_decode_or_posefit=0,S_U_new_q_or_local_queries=0,
        no_causal_error_or_WD_attribution=True,default_e_unchanged=True)
    C.write(output/'DIAGNOSIS.json',data)
    C.write(output/'DIAGNOSIS_SEAL.json',dict(diagnosis_sha256=C.sha(output/'DIAGNOSIS.json'),
        plan_sha256=C.sha(output/'PLAN.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        count=49,arm_count=4,new_fit_or_runtime_changes=0))
    print(dict(count=49,arms=4,summary=data['summary'],diagnosis_sha256=C.sha(output/'DIAGNOSIS.json')),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--root-postscore-GO',action='store_true')
    args = parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    else:run(args.output,root_postscore_GO=args.root_postscore_GO)
