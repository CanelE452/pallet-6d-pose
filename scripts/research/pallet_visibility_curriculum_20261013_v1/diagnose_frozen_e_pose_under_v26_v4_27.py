"""Predeclared TEST49 frozen-e cost/guard diagnosis, after v26 full seals.

The original e physical pose and its saved WD/native correspondence are
diagnostic only. They are never inserted into a runtime proposal, prior target,
fallback or optimizer. Six current v26 arms are assessed with their own cost,
including both unlocked and predicted-e-family selectors. Other transported
WD hypotheses are labelled separately, never mistaken for the original e
correspondence or for an admissible locked-family candidate.

Preparation reads only small protocol/input-seal/model/source bytes. Current
predictions, scores and large input journals are not read until a separate
postseal root GO. Execution decodes TEST49 only, no CAL or remaining143 rows,
images, S/U coordinates, reference poses/corners, or new optimization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
import shutil

import numpy as np

from . import audit_prepartial_RGB_preservation_v4_26 as A
from . import diagnose_frozen_e_pose_under_v24_v4_25 as PREVIOUS


V, C, E = A.V, A.V.C, A.V.E
OUTPUT = V.OUTPUT.parent/'frozen_e_restored_objective_diagnosis_v4_27'
ARMS = tuple(V.ARMS[1:])
REPORTING_TOLERANCE = 1e-8
_proper = PREVIOUS._proper
transport_physical_pose_to_family = PREVIOUS.transport_physical_pose_to_family
_physical_body_gap = PREVIOUS._physical_body_gap
_TEST_prefix = PREVIOUS._TEST_prefix
_costs = PREVIOUS._costs


def _profiles():
    expected=(('NATIVE_ZERO_COARSE','FULL',False,V.ARMS[1:3]),
        ('TRANSLATION_NATIVE','TRANSLATION',False,V.ARMS[3:5]),
        ('TRANSLATION_CONTAINMENT','TRANSLATION',True,V.ARMS[5:7]))
    if V.PROFILES!=expected or ARMS!=tuple(V.ARMS[1:]) or len(ARMS)!=6:
        raise ValueError('Exact six noP/TP/TP+C unlocked/locked arms required')
    return {arm:dict(profile=profile,mode=mode,containment=containment,locked=index==1)
        for profile,mode,containment,arms in expected for index,arm in enumerate(arms)}


def classify_comparison(*,family_allowed,e_guard_accepted,e_AUTH_eligible,
        selected_accepted,e_total,selected_total,tolerance=REPORTING_TOLERANCE):
    if not selected_accepted:
        return 'selected_is_own_fallback_not_a_solved_pose'
    if not family_allowed:
        return 'diagnostic_e_native_family_excluded_by_requested_family'
    if not e_guard_accepted:
        return 'diagnostic_e_excluded_by_current_hard_or_numeric_guard'
    if not e_AUTH_eligible:
        return 'diagnostic_e_excluded_by_current_native_AUTH'
    return PREVIOUS.classify_comparison(True,True,e_total,selected_total,tolerance)


def _source_inventory(task):
    partial={b['native_corner'] for b in task['boundary_merged']}
    all_sources,partial_sources,restored_sources=Counter(),Counter(),Counter()
    restored=[]
    for measurement in task['spec']['visible']:
        k=measurement['native_corner']
        # Source/xy/covariance fields are not read before the permanent gates.
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B','I'):
            raise ValueError('Permanent initial-nonS BI gate before archived source fields')
        source=measurement['source']
        all_sources[(task['states'][k],source)]+=1
        if k in partial:
            partial_sources[(task['states'][k],source)]+=1
            if source!=V.RESTORE.FALLBACK_SOURCE:
                restored.append(k);restored_sources[(task['states'][k],source)]+=1
    encode=lambda values:[dict(state=state,source=source,count=count)
        for (state,source),count in sorted(values.items())]
    return dict(all_original_BI_sources=encode(all_sources),original_partial_BI_sources=encode(partial_sources),
        restored_prepartial_sources=encode(restored_sources),restored_prepartial_ids=sorted(restored),
        no_original_raw_backbone_q_substituted_for_archived_RGB_q=True,S_U_coordinate_reads=0)


def _validate_saved_e_basis(pose,family):
    xyz=np.asarray(family['native_xyz'],float)
    if xyz.shape!=(8,3) or not np.isfinite(xyz).all() or not np.allclose(xyz.mean(0),0.,atol=1e-12,rtol=0):
        raise ValueError('Existing centered registered cuboid required for unchanged centroid transport')
    if not np.allclose(_proper(pose['R_cf'])@_proper(family['physical_registration_Q']),
            _proper(pose['R_physical']),atol=1e-10,rtol=0):
        raise ValueError('Saved e native and physical basis must match its saved WD family')
    if 'native_xyz' in pose and not np.allclose(pose['native_xyz'],xyz,atol=1e-12,rtol=0):
        raise ValueError('Saved e named native geometry changed')


def _evaluate(problem,task,R,t,regions,oparameters,sprepared,original_prior,*,containment):
    """Independent current N/O/S/P/C decomposition; fresh old guards retained."""
    try:
        fresh=problem.evaluate(R,t)
        p=A.NATIVE.V._pose_vector(R,t,problem.scale)
        native,_,parts=problem.objective(p)
        ov,_,oa=A.A.O.bounded_o_localization_value_and_gradient(p,problem.xyz,problem.K,
            problem.scale,regions,**oparameters)
        rebound=A.A.S.rebind_bounded_visibility_trial_geometry(sprepared,problem.xyz)
        sv,_,sa=A.A.S.coarse_visibility_prior_value_and_gradient(p,problem.xyz,problem.scale,rebound)
        pv,_,pa=A.direct_old_P1_prior(p,problem.xyz,problem.scale,original_prior,problem.Q)
        cfg=A.A.MASK_CORE._outline_config(task['outline']['configuration'])
        weight=float(cfg['outline_factor_weight']) if containment and task['outline'].get('supported') else 0.
        shape,ca=0.,dict(valid_projection=True,enabled=False,unsupported_or_profile_disabled=True)
        if weight>0.:
            shape,_,ca=A.CONTAINMENT.outline_value_and_gradient(p,problem.xyz,problem.K,
                problem.scale,task['outline'],cfg)
        finite=bool(np.isfinite([native,ov,sv,pv,shape]).all())
        valid_shape=bool(ca.get('valid_projection') and shape>=0.)
        total=float(native+ov+sv+pv+weight*shape) if finite and valid_shape else None
        kinds,factors=_costs(parts['native_costs'])
        regularizers=[dict(kind=kind,weighted_cost=float(value)) for kind,value in
            zip(V.P.REGULARIZER_KINDS,(ov,sv,pv))]
        return dict(accepted_current_hard_and_numeric_guards=bool(fresh['accepted'] and finite and valid_shape),
            original_hard_guard_accepted=bool(fresh['accepted']),native=float(native),O_prior=float(ov),
            S_prior=float(sv),P_prior=float(pv),containment=float(shape),containment_weight=weight,
            weighted_containment=float(weight*shape),total=total,native_by_kind=kinds,native_by_factor=factors,
            native_costs=parts['native_costs'],regularizer_costs=regularizers,
            minimum_joint_slack=fresh.get('minimum_joint_slack'),
            guard_audits={key:fresh.get(key) for key in ('visible_protection_audit','joint_constraint_audit',
                'U_region_constraint_audit','boundary_observation_audit')},
            current_noise_generation=V.RESTORE.NOISE_OBSERVATION_FAMILY,
            coarse_prior_audit=pa,containment_numeric_valid=valid_shape,
            O_point_terms=oa['O_point_terms'],S_image_or_2D_terms=sa['S_image_or_2D_terms'],
            old_symmetric_mask_cost_or_tie_break_used=False)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
        return dict(accepted_current_hard_and_numeric_guards=False,original_hard_guard_accepted=False,
            total=None,native_by_kind={},native_by_factor={},native_costs=[],regularizer_costs=[],
            reason=str(error),numeric_failure_not_a_global_infeasibility_proof=True)


def _authenticate_diagnostic(task,evaluation,*,family,requested_family,containment):
    """Isolated current AUTH check; never touches any production candidate bank."""
    allowed=requested_family is None or family==requested_family
    if not allowed or not evaluation['accepted_current_hard_and_numeric_guards']:
        return dict(family_allowed=allowed,AUTH_eligible=False,diagnostic_only=True)
    shape=evaluation['containment']
    candidate=dict(name='DIAGNOSTIC_E_ONLY_NOT_RUNTIME_INSERTION',family=family,accepted=True,
        all_joint_constraints_satisfied=True,native_costs=V.BASE._bridge_costs(evaluation['native_costs']),
        regularizer_costs=evaluation['regularizer_costs'],global_containment_cost=shape,global_mask_cost=shape)
    checked=V.P.select_candidates(task['states'],[candidate],
        containment_weight=evaluation['containment_weight'] if containment else 0.,locked_family=requested_family)
    eligible=checked['selected_name'] is not None
    if eligible and not np.isclose(checked['candidates'][0]['total_energy'],evaluation['total'],atol=1e-9,rtol=1e-10):
        raise ValueError('Same diagnostic AUTH cost and independent objective required')
    return dict(family_allowed=True,AUTH_eligible=eligible,diagnostic_only=True,
        requested_family=requested_family,source_e_numeric_pose_inserted_into_runtime=False)


def _verify_parent():
    protocol=C.read(V.OUTPUT/'PROTOCOL.json')
    pred=C.read(V.OUTPUT/'PREDICTION_SEAL.json');inputs=C.read(V.OUTPUT/'INPUT_SEAL.json')
    score=C.read(V.OUTPUT/'SCORE_SEAL.json')
    auditpath=V.OUTPUT/'independent_root_audit_v1/AUDIT.json'
    audit,aseal=C.read(auditpath),C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if (pred['rows']!=238 or pred['TEST']!=49 or pred['arms']!=list(V.ARMS)
            or not pred['all7_arms_complete'] or not pred['all_declared_arms_complete']
            or C.sha(V.OUTPUT/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(V.OUTPUT/'PROTOCOL.json')!=pred['protocol_sha256']
            or C.sha(V.OUTPUT/'INPUT_SEAL.json')!=pred['input_seal_sha256']
            or C.sha(V.OUTPUT/'INPUTS.jsonl.gz')!=inputs['input_sha256']
            or C.sha(V.OUTPUT/'PROTOCOL.json')!=inputs['protocol_sha256']
            or C.sha(V.OUTPUT/'PREDICTION_SEAL.json')!=score['prediction_seal']['sha256']
            or C.sha(V.OUTPUT/'TEST_SCORE.json')!=score['score_sha256']['TEST']
            or audit['rows']!=238 or audit['failure_count'] or audit['failures']
            or C.sha(auditpath)!=aseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(V.OUTPUT/'PREDICTION_SEAL.json')):
        raise ValueError('Complete v26 all238 seven-arm seals, fixed score and zero-failure audit required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    return protocol


def prepare(output=OUTPUT):
    """Freeze source/plan now; do not read live large inputs/predictions/scores."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all diagnostic generations')
    if (V.OUTPUT/'SCORE_SEAL.json').exists():raise ValueError('Predeclare this source before new v26 score')
    profiles=_profiles()
    parent=C.read(V.OUTPUT/'PROTOCOL.json');inputseal=C.read(V.OUTPUT/'INPUT_SEAL.json')
    if (parent['arms']!=list(V.ARMS) or inputseal['rows']!=238
            or inputseal['protocol_sha256']!=C.sha(V.OUTPUT/'PROTOCOL.json')
            or len(parent['ids'])!=238 or len(set(parent['ids']))!=238):
        raise ValueError('Exact original prepared238 and declared seven-arm profile required')
    sources=dict(parent['source_sha256'])
    for module in (A,A.PREVIOUS,A.NATIVE,A.SOURCE,A.RESTORE_CAL,A.A.TCAL,A.A.O,A.A.S,
            A.A.OFIT,A.A.SFIT,A.CONTAINMENT,PREVIOUS):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    paths=[V.OUTPUT/'PROTOCOL.json',V.OUTPUT/'INPUT_SEAL.json',V.PLAN]
    for module in (A.RESTORE_CAL,A.A.TCAL,A.A.OFIT,A.A.SFIT):
        paths.extend(module.OUTPUT/name for name in ('MODEL.json','MODEL_SEAL.json'))
    paths.append(C.ROOT/parent['coarse_prior_model_binding']['path'])
    bindings={str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}
    for mapping in (sources,bindings,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    C.write(output/'PLAN.json',dict(schema='FROZEN_E_RESTORED_V26_OBJECTIVE_DIAGNOSIS_PLAN_V27',
        created_UTC=datetime.now(timezone.utc).isoformat(),TEST_ids=parent['ids'][46:95],
        population='ALL original TEST49; every fallback/native-flag row retained',source_arms=list(ARMS),profiles=profiles,
        operation='diagnostic saved e physical pose in saved e native WD + separately labelled two-WD transport; fresh N/O/S/P/C and guards',
        primary_comparison='saved e WD/native correspondence, eligible only under actual requested family filter',
        other_WD_transport_is_different_native_correspondence_not_the_saved_e_semantics=True,
        isolated_AUTH_check_never_modifies_a_runtime_candidate_bank=True,
        original_e_pose_runtime_candidate_or_fallback_insertion=False,e_pose_new_prior_target=False,
        new_pose_optimization_or_model_fit=False,new_image_NN_or_reference_pose_corner_decode=False,
        S_U_raw_or_local_queries=0,posthoc_scalar_score_only_after_full_v26_seal_audit_score_and_root_GO=True,
        reporting_energy_tolerance=REPORTING_TOLERANCE,posthoc_translation_threshold_cm=5.,
        read_scope_after_GO='stream first95 compressed lines, JSON-decode only TEST49; CAL/REMAINDER JSON decode0',
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=parent['protected_user_sha256'],
        deferred_INPUTS_expected_sha256=inputseal['input_sha256'],
        inherited_input_sha256_deferred_until_postseal_GO=parent['input_sha256'],
        current_v26_prediction_or_score_read_during_preparation=False,
        large_input_or_prediction_journal_read_during_preparation=False,
        full_prediction_audit_score_bound_in_postseal_receipt_before_diagnostic_decode=True,
        current_default_e_unchanged=True,exploratory_diagnosis_not_parameter_or_runtime_selection=True))
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot changed')
    C.write(output/'PREPARATION_SEAL.json',dict(plan_sha256=C.sha(output/'PLAN.json'),source_sha256=sources,
        input_sha256=bindings,source_and_plan_before_current_prediction_or_score_decode=True))
    print(dict(predeclared_TEST=49,arms=6,plan_sha256=C.sha(output/'PLAN.json'),current_prediction_or_score_reads=0),flush=True)


def _public_evaluation(evaluation):
    return {k:v for k,v in evaluation.items() if k not in ('native_by_factor','native_costs','regularizer_costs')}


def run(output=OUTPUT,*,root_postscore_GO=False):
    if root_postscore_GO is not True:raise ValueError('Wait for separate root full-v26 seal/audit/score diagnosis GO')
    output=Path(output).resolve()
    if (output/'POSTSEAL_INPUT_BINDINGS_V1.json').exists() or (output/'DIAGNOSIS.json').exists():
        raise FileExistsError('Preserve completed or partial postseal diagnostics')
    plan,seal=C.read(output/'PLAN.json'),C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PLAN.json')!=seal['plan_sha256'] or plan['profiles']!=_profiles() or plan['source_arms']!=list(ARMS):
        raise ValueError('Exact predeclared diagnostic source/plan/profile required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(plan[key])
    parent=_verify_parent()
    if parent['ids'][46:95]!=plan['TEST_ids'] or C.sha(V.OUTPUT/'INPUTS.jsonl.gz')!=plan['deferred_INPUTS_expected_sha256']:
        raise ValueError('Original ordered TEST49/frozen input bytes changed')
    paths=[V.OUTPUT/name for name in ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json',
        'INPUTS.jsonl.gz','PREDICTIONS.jsonl.gz','TEST_SCORE.json','independent_root_audit_v1/AUDIT.json',
        'independent_root_audit_v1/AUDIT_SEAL.json')]
    bindings={str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}
    C.write(output/'POSTSEAL_INPUT_BINDINGS_V1.json',dict(before_diagnostic_TEST_or_scalar_score_decode=True,
        input_sha256=bindings,plan_sha256=C.sha(output/'PLAN.json'),no_runtime_changes=True))
    edge=C.read(A.RESTORE_CAL.OUTPUT/'MODEL.json');tangent=C.read(A.A.TCAL.OUTPUT/'MODEL.json')
    omodel=C.read(A.A.OFIT.OUTPUT/'MODEL.json');smodel=C.read(A.A.SFIT.OUTPUT/'MODEL.json')
    oparameters=A.A.OFIT.prior_parameters(omodel);A.A.S.validate_model(smodel)
    coarse_binding=parent['coarse_prior_model_binding'];coarsemodel=C.read(C.ROOT/coarse_binding['path'])
    scores=C.read(V.OUTPUT/'TEST_SCORE.json')
    if scores['count']!=49 or [r['id'] for r in scores['rows']]!=plan['TEST_ids']:
        raise ValueError('No posthoc TEST49 frame/fallback/native-flag exclusion')
    scored={r['id']:r for r in scores['rows']}
    rows=[];counts={arm:Counter() for arm in ARMS};regressions={arm:[] for arm in ARMS}
    maximum_difference=0.
    for task,row in zip_longest(_TEST_prefix(V.OUTPUT/'INPUTS.jsonl.gz',plan['TEST_ids']),
            _TEST_prefix(V.OUTPUT/'PREDICTIONS.jsonl.gz',plan['TEST_ids'])):
        if task is None or row is None or task['id']!=row['id'] or set(row['arms'])!=set(V.ARMS):
            raise ValueError('Exact complete ordered TEST49 inputs/predictions required')
        ident=task['id']
        if task['baseline']!=row['arms'][V.ARMS[0]]:raise ValueError('Frozen e baseline changed')
        e_family_name=task['baseline']['WD_family'];e_family=task['families'][e_family_name]
        _validate_saved_e_basis(task['baseline']['actual_pose'],e_family)
        source_inventory=_source_inventory(task)
        first=task['families']['ORIGINAL_WD'];coarse=first['coarse_pose']
        sprepared=A.A.S.prepare_bounded_coarse_visibility_prior(first['native_xyz'],coarse['R'],coarse['t'],
            np.asarray(task['initial_self_hidden'],bool),smodel)
        archives=[]
        for record in task['O_localization_abstention']['records']:
            domain=record['original_domain'];k=domain['native_corner']
            if task['initial_self_hidden'][k] or task['states'][k]!='O':raise ValueError('O state gate before region fields')
            if domain['binding']['calibration_model_sha256']!=omodel['source_O_model_digest']:raise ValueError('Fixed O model binding')
            archives.append(domain)
        regions=A.A.O.prepare_o_localization_regions(archives,task['states'],task['initial_self_hidden'])
        originals={mode:A._original_prior_fields(task,coarsemodel,coarse_binding['sha256'],mode)
            for mode in ('FULL','TRANSLATION')}
        problems={};e_trials={};transports={}
        for family_name,family in task['families'].items():
            if _physical_body_gap(e_family,family)>1e-12:raise ValueError('Same registered unordered physical body required')
            problem,_=A._reconstruct_native(task,family,edge,tangent)
            problems[family_name]=problem
            R,t,transport=transport_physical_pose_to_family(task['baseline']['actual_pose'],family)
            transports[family_name]=transport
            e_trials[family_name]={}
            for profile in {p['profile']:p for p in plan['profiles'].values()}.values():
                e_trials[family_name][profile['profile']]=_evaluate(problem,task,R,t,regions,oparameters,sprepared,
                    originals[profile['mode']],containment=profile['containment'])
        results={}
        for arm,profile in plan['profiles'].items():
            selected=row['arms'][arm];family_name=selected['WD_family'];family=task['families'][family_name]
            pose=selected['actual_pose'];R,t=np.asarray(pose['R_cf']),np.asarray(pose['centroid'])
            if not np.allclose(R@_proper(family['physical_registration_Q']),_proper(pose['R_physical']),atol=1e-10,rtol=0):
                raise ValueError('Selected physical Q correspondence changed')
            current=_evaluate(problems[family_name],task,R,t,regions,oparameters,sprepared,
                originals[profile['mode']],containment=profile['containment'])
            if selected['accepted']:
                if not current['accepted_current_hard_and_numeric_guards']:raise ValueError('Accepted selected pose fails fresh current guards')
                difference=abs(current['total']-selected['solution']['objective'])
                maximum_difference=max(maximum_difference,difference)
                if difference>1e-8*max(1.,abs(current['total'])):raise ValueError('Current saved selected objective/decomposition differs')
            requested=e_family_name if profile['locked'] else None
            early=selected['early_family_audit']
            if early['enabled']!=profile['locked'] or early['requested_predicted_family']!=requested or early['e_numeric_pose_used']:
                raise ValueError('Saved requested WD restriction differs from predeclared analysis')
            old=e_trials[e_family_name][profile['profile']]
            eligibility=_authenticate_diagnostic(task,old,family=e_family_name,requested_family=requested,containment=profile['containment'])
            classification=classify_comparison(family_allowed=eligibility['family_allowed'],
                e_guard_accepted=old['accepted_current_hard_and_numeric_guards'],e_AUTH_eligible=eligibility['AUTH_eligible'],
                selected_accepted=selected['accepted'],e_total=old['total'],selected_total=current['total'])
            counts[arm][classification]+=1
            keys=sorted(set(old['native_by_factor'])|set(current['native_by_factor']),key=str)
            factors=[dict(kind=k[0],native_corner=k[1],native_neighbor=k[2],e_pose_cost=old['native_by_factor'].get(k,0.),
                selected_cost=current['native_by_factor'].get(k,0.),e_minus_selected=old['native_by_factor'].get(k,0.)-current['native_by_factor'].get(k,0.)) for k in keys]
            metrics=scored[ident]['metrics'];regression=bool(metrics[V.ARMS[0]]['translation_cm']<=5.<metrics[arm]['translation_cm'])
            if regression:regressions[arm].append(ident)
            results[arm]=dict(profile=profile,saved_e_native_family=e_family_name,current_selected_family=family_name,
                requested_predicted_family=requested,original_e_semantics_family_allowed=eligibility['family_allowed'],
                diagnostic_e_AUTH=eligibility,diagnostic_e_evaluation=_public_evaluation(old),
                selected_current_evaluation=_public_evaluation(current),selected_was_accepted=selected['accepted'],
                classification=classification,native_factors=factors,
                e_minus_selected_model_terms={key:old.get(key,0.)-current.get(key,0.) for key in
                    ('O_prior','S_prior','P_prior','weighted_containment')},
                current_active_physical_edges=[g['physical_native_edge'] for g in problems[family_name].plan['active_groups']],
                sealed_scalar_metrics_posthoc_only=dict(e=metrics[V.ARMS[0]],current=metrics[arm],translation_threshold_regression=regression))
        rows.append(dict(id=ident,partition='TEST',states=task['states'],source_inventory=source_inventory,
            saved_e_native_family=e_family_name,saved_e_pose=task['baseline']['actual_pose'],
            diagnostic_e_transport_audits=transports,
            diagnostic_e_all_WD_correspondence_evaluations={name:{key:_public_evaluation(value) for key,value in profiles.items()}
                for name,profiles in e_trials.items()},arms=results,runtime_candidate_or_prior_or_fallback_insertions=0))
    if len(rows)!=49 or [r['id'] for r in rows]!=plan['TEST_ids']:raise ValueError('All49 including fallback/native flags must remain')
    for mapping in (plan['source_sha256'],plan['input_sha256'],plan['protected_user_sha256'],bindings):E._verify_sources(mapping)
    data=dict(schema='FROZEN_E_RESTORED_V26_OBJECTIVE_DIAGNOSIS_V27',count=49,arm_count=6,rows=rows,
        summary=dict(classifications={a:dict(v) for a,v in counts.items()},posthoc_translation_threshold_regressions=regressions,
            maximum_selected_objective_decomposition_difference=maximum_difference),
        plan=E._binding(output/'PLAN.json'),preparation_seal=E._binding(output/'PREPARATION_SEAL.json'),
        postseal_binding_receipt=E._binding(output/'POSTSEAL_INPUT_BINDINGS_V1.json'),
        original_e_pose_native_family_preserved_for_primary_comparison=True,
        alternate_WD_transport_is_conditional_correspondence_not_saved_e_identity=True,
        no_e_pose_runtime_candidate_prior_target_or_fallback_addition=True,no_GT_reference_pose_or_2D_decode=True,
        S_U_coordinate_or_local_queries=0,new_image_NN_model_fit_or_pose_optimizations=0,
        objective_preference_not_identified_causal_image_or_WD_error=True,
        bounded_search_gap_not_global_optimality_or_reference_accuracy_proof=True,
        inherited_TEST_derived_reference_reuse_exploratory_not_independent_validation=True,
        all_original_sources_predictions_models_and_default_unchanged=True)
    C.write(output/'DIAGNOSIS.json',C.finite(data))
    C.write(output/'DIAGNOSIS_SEAL.json',dict(diagnosis_sha256=C.sha(output/'DIAGNOSIS.json'),
        plan_sha256=C.sha(output/'PLAN.json'),source_sha256=plan['source_sha256'],input_sha256=bindings,
        protected_user_sha256=plan['protected_user_sha256'],no_new_fits_or_reference_decodes=True))
    print(data['summary'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--after-v26-score-audit-root-go',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:run(args.output,root_postscore_GO=args.after_v26_score_audit_root_go)
