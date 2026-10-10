"""Independent saved-row moments, paired uncertainty and posthoc arithmetic.

No statistics module is imported. No scorer, model, numerical pose solver or private GT loader is called.
Only recorded proxy errors, recorded native references and old session draws
are used; passing this audit is not an accuracy or physical-ownership claim.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

from . import common as C

KEYS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
UNITS = dict(translation_cm='cm', rotation_deg='degree', ADDsym_cm='cm')
SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')
BOOTSTRAP = C.REPO / '_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz'


def compact_saved_witnesses(value):
    """Independent reader pruning; never mutate serialized full input artifacts."""
    if isinstance(value, dict):
        return {key: compact_saved_witnesses(item) for key, item in value.items()
                if key not in ('all_candidate_solutions', 'alternatives')}
    if isinstance(value, list):
        return [compact_saved_witnesses(item) for item in value]
    return value


def compact_rows(path):
    for row in C.rows(path):
        yield compact_saved_witnesses(row)


class Audit:
    def __init__(self):
        self.exact = 0
        self.numeric = 0
        self.maximum_difference = 0.
        self.failures = []

    def eq(self, actual, expected, label):
        self.exact += 1
        if actual != expected:
            self.failures.append(dict(check=label, actual=actual, expected=expected))

    def near(self, actual, expected, label):
        self.numeric += 1
        if actual is None or expected is None:
            if actual != expected:
                self.failures.append(dict(check=label, actual=actual, expected=expected))
            return
        if not (math.isfinite(actual) and math.isfinite(expected)):
            self.failures.append(dict(check=label, actual=actual, expected=expected))
            return
        difference = abs(actual-expected)
        self.maximum_difference = max(self.maximum_difference, difference)
        if difference > 1e-9 + 1e-12 * abs(expected):
            self.failures.append(dict(check=label, actual=actual, expected=expected, absolute_difference=difference))


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def distribution(values, unit):
    n = len(values)
    mean = math.fsum(values)/n if n else None
    variance = math.fsum((value-mean)**2 for value in values)/(n-1) if n > 1 else None
    return dict(n=n, mean=mean, sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=quantile(values, .5), P90=quantile(values, .9),
                max=max(values) if values else None, unit=unit)


def check_distribution(audit, actual, values, unit, label):
    expected = distribution(values, unit)
    audit.eq(actual['n'], expected['n'], label+'.n')
    audit.eq(actual['unit'], unit, label+'.unit')
    for name in ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'max'):
        audit.near(actual[name], expected[name], label+'.'+name)


def value(row, key):
    return float(row['pose']['ADDsym_m'])*100 if key == 'ADDsym_cm' else float(row['pose'][key])


def point_error(point, reference):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point+reference) or point == [-1, -1]:
        return None
    return math.hypot(point[0]-reference[0], point[1]-reference[1])


def derive_posthoc(raw, n3, observation, audit):
    ref = raw['evaluation_reference']
    label = raw['method']+'/'+raw['id']
    audit.eq(ref['phase_control'], 'N3_SUBPIX', label+'.phase')
    audit.eq(ref['initial_N3_native_points'], n3['native_points'], label+'.N3')
    audit.eq(type(ref['matched']) is bool, True, label+'.matched_guard')
    states = ref['human_states_native']
    audit.eq(states, raw['mask_audit']['human_states_native'], label+'.states')
    known = set(ref['valid_native_ids']) if ref['matched'] else set()
    native = n3['native_points']; inputs = raw.get('input_points', native); output = raw['native_points']
    solver = raw.get('solver') or {}
    eligible = list(solver.get('eligible', range(8))); used = list(solver.get('used', []))
    fit = list(solver.get('fit_input_ids', [])) if raw['new_pose_estimated'] else []
    raw_final = list(solver.get('final_inliers', [])); final = raw_final if raw['new_pose_estimated'] else []
    reproj = list(raw.get('reprojected_ids', [])); H = set(raw['hidden_initial'])
    audit.eq(set(reproj), H if raw['hidden_reprojected'] else set(), label+'.reprojection_ids')
    audit.eq(bool(set(fit) & H), False, label+'.hidden_fit')
    for key in ('prior_used','initial_pose_used','initial_projection_used','initial_dimension_prior_used',
                'excluded_image_coordinates_used_for_scoring','excluded_image_coordinates_used_for_equivalence'):
        audit.eq(solver[key], False, label+'.independent_solver.'+key)
    audit.eq(solver['known_dimension_constraint_used'], False, label+'.no_external_dimension_prior')
    audit.eq(set(solver['excluded']), H, label+'.excluded')
    for key in ('used','fit_input_ids','final_inliers','generator_ids'):
        audit.eq(bool(H & set(solver.get(key, []))), False, label+'.hidden_absent.'+key)
    spec = C.METHOD_SPECS.get(raw['method'])
    arm, supply = (spec['arm'], spec['supply']) if spec else (None, 'NATIVE_N3')
    audit.eq(raw['head_arm'], arm, label+'.head')
    audit.eq(raw['observation_supply'], supply, label+'.supply')
    audit.eq(raw['selection_validation_fit_excludes_candidate_corner'], supply == 'CORNERWISE_HYBRID', label+'.selection_mode')
    corners = []
    for k in range(8):
        reference = ref['native_points_px'][k]
        reference_known = k in known and all(type(v) in (int,float) and math.isfinite(v) for v in reference)
        errors = {name: point_error(points[k], reference) if reference_known else None
                  for name, points in [('N3',native),('input',inputs),('output',output)]}
        quality = ('UNKNOWN_REFERENCE_OR_INPUT' if errors['input'] is None else
                   'CORRECT_WITHIN8PX' if errors['input'] <= 8 else 'INCORRECT_OVER8PX')
        corners.append(dict(id=k, human_state=states[k], reference_known=reference_known,
            N3_error_px=errors['N3'], input_error_px=errors['input'], output_error_px=errors['output'],
            input_quality=quality, solver_eligible=k in eligible, solver_used=k in used,
            accepted_new_pose_fit=k in fit, accepted_new_pose_final_inlier=k in final,
            raw_candidate_final_inlier=k in raw_final, actually_reprojected=k in reproj,
            output_source=raw.get('output_coordinate_sources',[None]*9)[k],
            directly_visible_damage_5_to_10=bool(states[k]=='DIRECT_VISIBLE' and errors['N3'] is not None and
                                               errors['output'] is not None and errors['N3']<=5 and errors['output']>10)))
    delta = {key:value(raw,key)-value(n3,key) for key in KEYS} if raw['pose']['available'] and n3['pose']['available'] else None
    outcome = ('POSE_UNAVAILABLE' if delta is None else
               'BOTH_BETTER' if delta['translation_cm']<0 and delta['rotation_deg']<0 else
               'BOTH_WORSE' if delta['translation_cm']>0 and delta['rotation_deg']>0 else 'MIXED_OR_UNCHANGED')
    mask = ('MASK_NOT_APPLIED' if not raw['mask_audit']['mask_applied'] else
            'NO_KNOWN_VISIBILITY' if not raw['mask_audit']['known_ids'] else
            'WRONG_ON_KNOWN' if raw['mask_audit']['mask_wrong_on_known'] else 'MATCHES_ON_KNOWN')
    contract = raw.get('observation_contract') or {}
    observed = {int(r['id']):r for r in observation.get('corners',[])} if observation else {}
    admission = {int(r['id']):r for r in contract.get('corner_admission',[])}
    records = {int(r['id']):r for r in contract.get('cornerwise_records',[])}
    selected = set(raw.get('selected_corner_ids',[]))
    audit.eq(set(admission),set(observed),label+'.actual_corner_admission_ids')
    audit.eq(len(admission),len(contract.get('corner_admission',[])),label+'.admission_unique')
    audit.eq(len(observed),len(observation.get('corners',[])) if observation else 0,label+'.corner_unique')
    if supply == 'BOUNDARY_ONLY':
        audit.eq(bool(records),False,label+'.no_sparse_LOO_records')
        audit.eq(contract['boundary_only_LOO_used_for_admission'],False,label+'.no_sparse_LOO')
        audit.eq(contract['boundary_only_native_distance_used_for_admission'],False,label+'.no_sparse_native_gate')
        audit.eq(contract['native_N3_used_for_numeric_final_fit'],False,label+'.no_sparse_native_numeric_observation')
        audit.eq(selected,{k for k,r in admission.items() if r['accepted']},label+'.sparse_selected')
        audit.eq(set(solver.get('eligible',[])) <= selected,True,label+'.sparse_premask_eligible_ids')
        for key in ('used','fit_input_ids','final_inliers','generator_ids'):
            audit.eq(set(solver.get(key,[])) <= selected-H,True,label+'.sparse_actual_ids.'+key)
        for k in range(8):
            if k not in selected:
                audit.eq(all(type(v) not in (int,float) or not math.isfinite(v) for v in inputs[k]),
                         True,label+'.unobserved_sparse_input.'+str(k))
            else:
                for j in range(2):audit.near(inputs[k][j],observed[k]['xy'][j],label+'.sparse_actual_xy.'+str(k)+'.'+str(j))
    elif supply == 'CORNERWISE_HYBRID':
        audit.eq(selected,{k for k,r in records.items() if r['accepted']},label+'.hybrid_selected')
    else:
        audit.eq(bool(observed or admission or records or selected),False,label+'.native_no_boundary')
    for k, record in records.items():
        rlabel=label+'.heldout.'+str(k)
        audit.eq(record['initial_prior_includes_heldout_influence'],False,rlabel+'.prior_influence')
        audit.eq(record['numeric_validation_pose_prior_used'],False,rlabel+'.numeric_prior')
        audit.eq(record['projected_coordinate_used_as_observation'],False,rlabel+'.no_generated_observation')
        audit.eq(set(record['heldout_fit_requested_excluded_ids']),H|{k},rlabel+'.requested')
        for j in range(2):audit.near(record['candidate_xy'][j],observed[k]['xy'][j],rlabel+'.actual_candidate_xy.'+str(j))
        if record['heldout_pose_calls']:
            ls=record['loo_solver'];blocked=H|{k}
            audit.eq(ls['prior_used'],False,rlabel+'.solver_prior')
            audit.eq(set(ls['excluded']),blocked,rlabel+'.solver_excluded')
            audit.eq(set(ls['hidden']),H,rlabel+'.actual_H')
            audit.eq(set(ls['temporary_excluded']),{k},rlabel+'.temporary_k')
            for key in ('used','fit_input_ids','final_inliers','generator_ids'):
                audit.eq(bool(blocked & set(ls.get(key,[]))),False,rlabel+'.'+key)
    boundary=[]
    for k, gate in sorted(admission.items()):
        item=observed[k];record=records.get(k)
        candidate_error=point_error(item['xy'],ref['native_points_px'][k]) if k in known else None
        prior_error=corners[k]['N3_error_px']
        reason=(gate['reason'] if not gate['accepted'] else 'INITIAL_SELF_HIDDEN' if k in H else
                'SPARSE_BOUNDARY_ADMITTED_WITHOUT_NATIVE_GATE' if supply=='BOUNDARY_ONLY' else record['reason'])
        boundary.append(dict(id=k,head_arm=arm,observation_supply=supply,
            candidate_xy=item['xy'],edges=list(item['edges']),radius_px=item['radius_px'],
            admission_accepted=gate['accepted'],supply_selected=k in selected,accepted=k in selected and k not in H,
            reason=reason,self_hidden_excluded=k in H,
            displayed_in_final_output=corners[k]['output_source']=='VALIDATED_BOUNDARY_INTERSECTION',
            accepted_new_pose_fit=k in selected and k not in H and k in fit,
            accepted_new_pose_final_inlier=k in selected and k not in H and k in final,
            candidate_error_px=candidate_error,N3_error_px=prior_error,
            candidate_proxy_quality='UNKNOWN_REFERENCE' if candidate_error is None else
            'CORRECT_WITHIN8PX' if candidate_error<=8 else 'INCORRECT_OVER8PX',
            delta_vs_N3_px=candidate_error-prior_error if candidate_error is not None and prior_error is not None else None,
            candidate_LOO_residual_px=record['candidate_LOO_residual_px'] if record else None,
            native_N3_LOO_residual_px=record['native_N3_LOO_residual_px'] if record else None,
            heldout_pose_calls=record['heldout_pose_calls'] if record else 0,
            validation_initial_prior_is_shared=bool(record['initial_prior_includes_heldout_influence']) if record else False,
            selection_validation='CORNERWISE_NATIVE_LOO' if record else 'NO_LOO_ADMISSION',
            physical_boundary_ownership_independently_verified=False))
    pools={name:dict(ids=sel,quality_counts=dict(Counter(corners[k]['input_quality'] for k in sel)),
                    human_direct_visible_ids=[k for k in sel if states[k]=='DIRECT_VISIBLE'])
           for name,sel in [('eligible',eligible),('solver_used',used),('accepted_fit',fit),('accepted_final_inliers',final)]}
    return dict(id=raw['id'],session=raw['session'],method=raw['method'],head_arm=arm,observation_supply=supply,
        selected_corner_ids=sorted(selected),stored_solver_geometry=solver.get('geometry'),output_status=raw['output_status'],
        new_pose_estimated=raw['new_pose_estimated'],fallback_used=raw['fallback_used'],
        reference_phase='fixed N3_SUBPIX; existing GEOMETRIC_PROXY',
        final_inlier_scope='accepted NEW_POSE only; fallback solver candidate IDs separately recorded',
        corners=corners,pools=pools,boundary_candidates=boundary,mask_state=mask,mask_audit=raw['mask_audit'],
        pose_outcome_vs_N3=outcome,delta_vs_N3=delta,hidden_initial=raw['hidden_initial'],
        actually_reprojected_ids=reproj,hidden_set_changed=raw['hidden_set_changed'])


def tree_check(audit, actual, expected, label):
    if isinstance(expected, dict):
        for key, value_ in expected.items():
            if key not in actual:
                audit.eq(False, True, label+'.missing.'+key)
            else:
                tree_check(audit, actual[key], value_, label+'.'+key)
    elif isinstance(expected, list):
        audit.eq(len(actual), len(expected), label+'.length')
        if len(actual) == len(expected):
            for i, v in enumerate(expected):tree_check(audit, actual[i], v, label+'.'+str(i))
    elif isinstance(expected, float):
        audit.near(actual, expected, label)
    else:
        audit.eq(actual, expected, label)


def verify_diagnostics(audit, actual, rows, label):
    audit.eq(actual['frames'], len(rows), label+'.frames')
    for mask in ('WRONG_ON_KNOWN','MATCHES_ON_KNOWN','MASK_NOT_APPLIED','NO_KNOWN_VISIBILITY'):
        for outcome in ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE'):
            selected = [r for r in rows if r['mask_state'] == mask and r['pose_outcome_vs_N3'] == outcome]
            tree_check(audit, actual['mask_and_pose_groups_vs_fixed_N3'][mask+'__'+outcome],
                       dict(n=len(selected), ids=[r['id'] for r in selected],
                            new_pose=sum(r['new_pose_estimated'] for r in selected),
                            fallback=sum(r['fallback_used'] for r in selected)), label+'.'+mask+'.'+outcome)
    corners = [c for r in rows for c in r['corners']]
    for category, selected in [('DIRECT_VISIBLE',[c for c in corners if c['human_state']=='DIRECT_VISIBLE']),
                               ('SELF_OCCLUDED',[c for c in corners if c['human_state']=='SELF_OCCLUDED']),
                               ('ACTUALLY_REPROJECTED',[c for c in corners if c['actually_reprojected']])]:
        block = actual[category]
        audit.eq(block['corners'], len(selected), label+'.'+category+'.count')
        for field in ('N3','input','output'):
            check_distribution(audit,block['errors'][field],
                [c[field+'_error_px'] for c in selected if c[field+'_error_px'] is not None], 'px',label+'.'+category+'.'+field)
        if category=='DIRECT_VISIBLE':
            audit.eq(block['damage_5px_to_over10px'],sum(c['directly_visible_damage_5_to_10'] for c in selected),label+'.damage')
        elif category=='ACTUALLY_REPROJECTED':
            audit.eq(block['unknown_output_error'],sum(c['output_error_px'] is None for c in selected),label+'.unknown_reprojection')
    boundary = [c for r in rows for c in r['boundary_candidates']]
    tree_check(audit,actual['boundary_supply_counts'],dict(computed=len(boundary),
        admission_accepted=sum(c['admission_accepted'] for c in boundary),
        selected_including_H=sum(c['supply_selected'] for c in boundary),
        selected_excluded_H=sum(c['supply_selected'] and c['self_hidden_excluded'] for c in boundary),
        selected_nonhidden=sum(c['accepted'] for c in boundary),
        accepted_NEW_fit=sum(c['accepted_new_pose_fit'] for c in boundary),
        accepted_NEW_final_inlier=sum(c['accepted_new_pose_final_inlier'] for c in boundary),
        actually_displayed=sum(c['displayed_in_final_output'] for c in boundary)),label+'.boundary_supply_counts')
    for field, data in [('all_candidate_boundary_quality',boundary),
                        ('adopted_boundary_quality',[c for c in boundary if c['accepted']]),
                        ('displayed_boundary_quality',[c for c in boundary if c['displayed_in_final_output']])]:
        valid = [c for c in data if c['delta_vs_N3_px'] is not None]
        block=actual[field]
        audit.eq(block['candidate_proxy_quality_counts'],dict(Counter(c['candidate_proxy_quality'] for c in data)),
                 label+'.'+field+'.candidate_proxy_quality_counts')
        check_distribution(audit,block['candidate_error_px'],
            [c['candidate_error_px'] for c in data if c['candidate_error_px'] is not None],'px',
            label+'.'+field+'.absolute_candidate_error')
        for key, expected in dict(candidates=len(data),known_N3_comparisons=len(valid),
            improved=sum(c['delta_vs_N3_px']<0 for c in valid),worsened=sum(c['delta_vs_N3_px']>0 for c in valid),
            unchanged=sum(c['delta_vs_N3_px']==0 for c in valid),reasons=dict(Counter(c['reason'] for c in data))).items():
            audit.eq(block[key],expected,label+'.'+field+'.'+key)
        check_distribution(audit,block['delta_error_px'],[c['delta_vs_N3_px'] for c in valid],'px',label+'.'+field)
    audit.eq(actual['hidden_set_changed_frames'],sum(r['hidden_set_changed'] for r in rows),label+'.hidden_changed')
    audit.eq(actual['causal_boundary_ownership_error_fraction_identified'],False,label+'.causal_limit')


def run(args):
    C.verify_protocol(args)
    C.protect(args)
    out=C.output_path(args,'VERIFICATION.json')
    folder=Path(args.input)
    result=dict(schema='independent_fixed_three_head_two_supply_saved_row_audit_v7',passed=False,complete=False,
                statistics_module_imported=False,new_model_score_PnP_ray_training_RGB_calls=0,
                independent_physical_GT_validated=False,existing_proxy_rows_only=True,
                actual_arithmetic_runs=1)
    audit=Audit()
    counters=Counter()
    try:
        recorded=C.read(folder/'METRICS.json')
        cohort=C.read(args.cohort)
        with gzip.open(BOOTSTRAP,'rt') as stream:bootstrap=json.load(stream)
        paths=dict(observations=folder/'OBSERVATIONS.jsonl.gz',new_scored_rows=folder/'PREDICTIONS.jsonl.gz',fixed_scored_rows=folder/'FIXED_PREDICTIONS.jsonl.gz',
            scoring_receipt=folder/'SCORING_RECEIPT.json',geometry_seal=folder/'GEOMETRY_SEAL.json',
            inference_receipt=folder/'INFERENCE_RECEIPT.json',
            old_v4_control_parity=folder/'V4_CONTROL_PARITY.json',
            cohort=args.cohort,bootstrap=BOOTSTRAP,
            statistics_code=Path(__file__).with_name('statistics.py'),protocol=args.protocol)
        before={k:C.binding(p) for k,p in paths.items()}
        audit.eq(recorded['bindings'],before,'recorded_input_bindings')
        C.require(recorded['bindings']==before,'full raw/input hash binding failed before row accumulation')
        posthoc_binding=C.binding(folder/'POSTHOC_ROWS.jsonl.gz')
        audit.eq(recorded['posthoc_rows'],posthoc_binding,'posthoc_binding')
        C.require(recorded['posthoc_rows']==posthoc_binding,'posthoc hash failed before row accumulation')
        observations=list(compact_rows(folder/'OBSERVATIONS.jsonl.gz'))
        raw=list(compact_rows(folder/'PREDICTIONS.jsonl.gz'))
        fixed=list(compact_rows(folder/'FIXED_PREDICTIONS.jsonl.gz'))
        posthoc=list(compact_rows(folder/'POSTHOC_ROWS.jsonl.gz'))
        result['saved_row_memory_policy']=dict(original_full_serialized_rows_preserved=True,
            complete_raw_SHA_checked_before_loading=True,
            unused_keys_recursively_discarded_in_RAM=['all_candidate_solutions','alternatives'],
            required_coordinate_pool_and_LOO_ID_witnesses_retained=True)
        inference=C.read(folder/'INFERENCE_RECEIPT.json')
        for key, expected in dict(complete=True,cleanup_error=None,actual_complete_frames=245,
                                 method_rows=1960,fixed_rows=490,observation_rows=735).items():
            audit.eq(inference[key],expected,'inference_completion.'+key)
        scoring_receipt=C.read(folder/'SCORING_RECEIPT.json')
        audit.eq(scoring_receipt['inference_receipt'],C.binding(folder/'INFERENCE_RECEIPT.json'),
                 'inference_completion.scoring_binding')
        parity=C.read(folder/'V4_CONTROL_PARITY.json')
        audit.eq(parity['passed'],True,'old_v4_control_parity.PASS')
        audit.eq(parity['checked_before_GT_reference_reads'],True,'old_v4_control_parity.pre_GT')
        audit.eq(parity['reference_used_for_pose_or_selection'],False,'old_v4_control_parity.not_method_input')
        ids=cohort['ids']; labels={r['id']:r['label'] for r in cohort['frames']}
        methods=('BASE','N3_SUBPIX')+tuple(C.METHODS)
        by={m:{} for m in methods}
        for row in fixed+raw:
            C.require(row['method'] in by and row['id'] not in by[row['method']],'duplicate/mismatched saved row')
            by[row['method']][row['id']]=row
        for m in methods:audit.eq(set(by[m]),set(ids),'cohort.'+m)
        audit.eq(len(raw),1960,'raw_rows');audit.eq(len(fixed),490,'fixed_rows')
        audit.eq(len(posthoc),1960,'posthoc_rows')
        counters.update(raw_scored_rows_loaded=len(raw)+len(fixed),posthoc_rows_loaded=len(posthoc))
        post_by={(r['method'],r['id']):r for r in posthoc}
        audit.eq(len(post_by),len(posthoc),'posthoc_uniqueness')
        obs={(r['head_arm'],r['id']):r for r in observations}
        audit.eq(len(obs),735,'three_head_observations_unique')
        audit.eq(len(observations),735,'three_head_observations_count')
        for arm in C.ARMS:audit.eq({fid for a,fid in obs if a==arm},set(ids),'observations.'+arm)
        for r in observations:audit.eq(r['GT_input'],False,'observations.no_GT')
        counters['source_observations_loaded']=len(observations)
        derived=[]
        for row in raw:
            observation=obs[(row['head_arm'],row['id'])] if row['head_arm'] else None
            expected=derive_posthoc(row,by['N3_SUBPIX'][row['id']],observation,audit)
            actual_posthoc=post_by[(row['method'],row['id'])]
            p_label='posthoc.'+row['method']+'.'+row['id']
            audit.eq(actual_posthoc['selected_corner_ids'],sorted(set(row.get('selected_corner_ids',[]))),
                     p_label+'.admitted_proposal_ids_not_final_inliers')
            audit.eq(actual_posthoc['pools']['accepted_final_inliers']['ids'],
                     list((row.get('solver') or {}).get('final_inliers',[])) if row['new_pose_estimated'] else [],
                     p_label+'.accepted_final_inlier_ids_separate_from_proposals')
            tree_check(audit,actual_posthoc,expected,p_label)
            derived.append(expected)
        sessions=bootstrap['sessions'];draws=bootstrap['counts'];si={s:i for i,s in enumerate(sessions)}
        C.require(len(sessions)==13 and len(si)==13 and len(draws)==10000 and
            all(len(d)==13 and sum(d)==13 and all(isinstance(v,int) and v>=0 for v in d) for d in draws),
            'frozen session multiplicities differ')
        audit.eq(recorded['bootstrap']['serialized_raw_sha256'],bootstrap['serialized_raw_sha256'],'bootstrap_raw_SHA')
        audit.eq(recorded['bootstrap']['new_draws_generated'],0,'new_bootstrap_draws')
        contrasts=tuple([(arm+'_'+supply,'N3_SUBPIX') for arm in C.ARMS
            for supply in ('BOUNDARY_ONLY','CORNERWISE_HYBRID')]+[
            (arm+'_CORNERWISE_HYBRID',arm+'_BOUNDARY_ONLY') for arm in C.ARMS]+[
            (a+'_'+supply,b+'_'+supply) for supply in ('BOUNDARY_ONLY','CORNERWISE_HYBRID')
            for a,b in [('IMAGE_NO_ROLE','GEOMETRY_ONLY'),('IMAGE_ROLE','IMAGE_NO_ROLE')]]+[
            ('N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK'),
            (C.PRIMARY,'N3_INDEPENDENT_ROBUST_H'),(C.PRIMARY,'N3_INDEPENDENT_ROBUST_NO_MASK')])
        audit.eq(contrasts,tuple(C.CONTRASTS),'fixed_sixteen_contrasts')
        for stratum,selected in [('combined',ids),('easy',[fid for fid in ids if labels[fid]=='clean']),
                                 ('medium',[fid for fid in ids if labels[fid]=='moderate'])]:
            data=recorded['strata'][stratum]
            audit.eq(data['ids'],selected,stratum+'.IDs');audit.eq(data['frames'],len(selected),stratum+'.N')
            audit.eq(set(data['methods']),set(methods),stratum+'.methods')
            audit.eq(set(data['contrasts']),{a+'_minus_'+b for a,b in contrasts},stratum+'.contrasts')
            for m in methods:
                rows=[by[m][fid] for fid in selected];block=data['methods'][m]
                operational=[r for r in rows if r['pose']['available']]
                new=[r for r in rows if r['pose']['available'] and r['new_pose_estimated']]
                fallback=[r for r in rows if r['pose']['available'] and r['fallback_used']]
                for k,v in dict(denominator=len(rows),operational=len(operational),new_pose=len(new),fallback=len(fallback),
                    no_pose=len(rows)-len(operational),fixed_control=sum(r['output_status']=='FRESH_FIXED_CONTROL' for r in rows),
                    statuses=dict(Counter(r['output_status'] for r in rows))).items():audit.eq(block[k],v,stratum+'.'+m+'.'+k)
                for scope,scoped in [('operational',operational),('new_pose',new),('fallback',fallback)]:
                    for k in KEYS:
                        check_distribution(audit,block['metrics'][scope][k],[value(r,k) for r in scoped],UNITS[k],stratum+'.'+m+'.'+scope+'.'+k)
            for a,b in contrasts:
                for scope in SCOPES:
                    paired=[fid for fid in selected if by[a][fid]['pose']['available'] and by[b][fid]['pose']['available'] and
                        (scope=='common_operational' or by[a][fid]['new_pose_estimated']) and
                        (scope!='both_new_pose' or by[b][fid]['new_pose_estimated'])]
                    block=data['contrasts'][a+'_minus_'+b][scope]
                    for k,v in dict(scope=scope,denominator=len(selected),common_frames=len(paired),pair_ids=paired,
                        candidate_new_pose_in_pairs=sum(by[a][fid]['new_pose_estimated'] for fid in paired),
                        comparator_new_pose_in_pairs=sum(by[b][fid]['new_pose_estimated'] for fid in paired)).items():
                        audit.eq(block[k],v,stratum+'.'+a+'.'+b+'.'+scope+'.'+k)
                    counts=[sum(by[a][fid]['session']==s for fid in paired) for s in sessions]
                    denominator=[sum(w*n for w,n in zip(draw,counts)) for draw in draws]
                    nonempty=[i for i,n in enumerate(denominator) if n]
                    audit.eq(block['bootstrap_nonempty_resamples'],len(nonempty),'nonempty.'+stratum+'.'+a+'.'+b+'.'+scope)
                    audit.eq(block['bootstrap_empty_resamples'],10000-len(nonempty),'empty.'+stratum+'.'+a+'.'+b+'.'+scope)
                    counters['pair_groups_checked']+=1;counters['resample_denominators_computed']+=10000
                    for k in KEYS:
                        delta=[value(by[a][fid],k)-value(by[b][fid],k) for fid in paired]
                        totals=[math.fsum(d for fid,d in zip(paired,delta) if by[a][fid]['session']==s) for s in sessions]
                        sampled=[math.fsum(draws[i][j]*totals[j] for j in range(13))/denominator[i] for i in nonempty]
                        mb=block['metrics'][k]
                        audit.eq(mb['n'],len(delta),'paired_n.'+k);audit.eq(mb['unit'],UNITS[k],'paired_unit.'+k)
                        audit.eq(mb['bootstrap_nonempty_resamples'],len(nonempty),'paired_nonempty.'+k)
                        audit.near(mb['mean_delta'],math.fsum(delta)/len(delta) if delta else None,'paired_mean.'+k)
                        expected_ci=[quantile(sampled,.025),quantile(sampled,.975)] if sampled else None
                        if expected_ci is None:audit.eq(mb['CI95'],None,'empty_CI.'+k)
                        else:
                            audit.near(mb['CI95'][0],expected_ci[0],'CI95_lower.'+k)
                            audit.near(mb['CI95'][1],expected_ci[1],'CI95_upper.'+k)
                        counters['metric_CI_slots_checked']+=1
                        counters['paired_delta_values_computed']+=len(delta)
                        counters['nonempty_resampled_means_computed']+=len(sampled)
            for m in C.METHODS:
                verify_diagnostics(audit,data['diagnostics'][m],
                    [r for r in derived if r['method']==m and r['id'] in set(selected)],stratum+'.'+m+'.diagnostics')
        audit.eq(recorded['methods'],recorded['strata']['combined']['methods'],'combined_method_alias')
        audit.eq(recorded['contrasts'],recorded['strata']['combined']['contrasts'],'combined_contrast_alias')
        p=recorded['contrasts'][C.PRIMARY+'_minus_N3_SUBPIX']['common_operational']
        full=p['common_frames']==245
        audit.eq(recorded['verdict']['all245_paired_outputs_available'],full,'verdict.complete')
        audit.eq(recorded['verdict']['primary_full_operational_T_and_R_improved'],full and
            p['metrics']['translation_cm']['mean_delta']<0 and p['metrics']['rotation_deg']['mean_delta']<0,'verdict.mean')
        with (folder/'METRICS.csv').open(newline='') as stream:csv_rows=list(csv.DictReader(stream))
        audit.eq(len(csv_rows),270,'CSV_rows')
        for row in csv_rows:
            block=recorded['strata'][row['stratum']]['methods'][row['method']]['metrics'][row['scope']][row['metric']]
            for k in ('n','mean','sample_variance','sample_std','median','P90','max'):
                audit.near(float(row[k]) if row[k] else None,block[k],'CSV.'+k)
            audit.eq(row['unit'],block['unit'],'CSV.unit')
        after={k:C.binding(p) for k,p in paths.items()}
        audit.eq(after,before,'unchanged_raw_input_bindings')
        C.verify_protocol(args);preserved=C.protect(args)
        result.update(complete=True,input_bindings=before,posthoc=C.binding(folder/'POSTHOC_ROWS.jsonl.gz'),
            metrics=C.binding(folder/'METRICS.json'),code=C.binding(__file__),
            protected_prior_state=preserved,actual_counts=dict(counters),
            exact_checks=audit.exact,numeric_checks=audit.numeric,max_numeric_difference=audit.maximum_difference,
            failures=audit.failures,failure_count=len(audit.failures),passed=not audit.failures)
    except Exception as exc:
        result.update(exception_type=type(exc).__name__,exception=str(exc),failure_count=len(audit.failures)+1,
                      failures=audit.failures,actual_counts=dict(counters))
    C.write_new(out,result)
    print(json.dumps(dict(passed=result['passed'],complete=result['complete'],
        failure_count=result.get('failure_count'),actual_counts=result.get('actual_counts'))),flush=True)
    if not result['passed']:raise SystemExit(1)


def main():
    parser=C.parser(__doc__)
    parser.add_argument('--input',default=str(C.PRIVATE/'accuracy'))
    run(parser.parse_args())


if __name__=='__main__':main()
