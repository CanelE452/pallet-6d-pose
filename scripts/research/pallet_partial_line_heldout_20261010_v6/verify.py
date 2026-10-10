"""Independent saved-row moments, paired uncertainty and posthoc arithmetic.

No statistics module, scorer, model, pose solver or private GT is imported.
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


def derive_posthoc(raw, n3, audit):
    ref = raw['evaluation_reference']
    audit.eq(ref['matched'],True,raw['method']+'/'+raw['id']+'.matched_proxy_reference')
    audit.eq(ref['phase_control'], 'N3_SUBPIX', raw['method']+'/'+raw['id']+'.phase')
    audit.eq(ref['initial_N3_native_points'], n3['native_points'], raw['method']+'/'+raw['id']+'.N3')
    states = ref['human_states_native']
    audit.eq(states, raw['mask_audit']['human_states_native'], raw['method']+'/'+raw['id']+'.states')
    known = set(ref['valid_native_ids'])
    native = n3['native_points']
    inputs = raw.get('input_points', native)
    output = raw['native_points']
    solver = raw.get('solver') or {}
    eligible = list(solver.get('eligible', range(8)))
    used = list(solver.get('used', []))
    fit = list(solver.get('fit_input_ids', [])) if raw['new_pose_estimated'] else []
    raw_final = list(solver.get('final_inliers', []))
    final = raw_final if raw['new_pose_estimated'] else []
    reproj = list(raw.get('reprojected_ids', []))
    audit.eq(set(reproj), set(raw['hidden_initial']) if raw['hidden_reprojected'] else set(),
             raw['method']+'/'+raw['id']+'.reprojection_ids')
    audit.eq(bool(set(fit) & set(raw['hidden_initial'])), False, raw['method']+'/'+raw['id']+'.hidden_fit')
    label = raw['method']+'/'+raw['id']+'.independent_solver'
    for key in ('prior_used','initial_pose_used','initial_projection_used','initial_dimension_prior_used',
                'excluded_image_coordinates_used_for_scoring','excluded_image_coordinates_used_for_equivalence'):
        audit.eq(solver[key],False,label+'.'+key)
    audit.eq(solver['known_dimension_constraint_used'],False,label+'.no_external_parity_available')
    blocked = set(raw['hidden_initial'])
    audit.eq(set(solver['excluded']),blocked,label+'.excluded')
    for key in ('used','fit_input_ids','final_inliers','generator_ids'):
        audit.eq(bool(blocked & set(solver.get(key,[]))),False,label+'.'+key)
    corners = []
    for k in range(8):
        reference = ref['native_points_px'][k]
        reference_known = k in known and all(v is not None and math.isfinite(v) for v in reference)
        errors = {name: point_error(points[k], reference) if reference_known else None
                  for name, points in [('N3', native), ('input', inputs), ('output', output)]}
        quality = ('UNKNOWN_REFERENCE_OR_INPUT' if errors['input'] is None else
                   'CORRECT_WITHIN8PX' if errors['input'] <= 8 else 'INCORRECT_OVER8PX')
        source = raw.get('output_coordinate_sources', [None]*9)[k]
        corners.append(dict(id=k, human_state=states[k], reference_known=reference_known,
            N3_error_px=errors['N3'], input_error_px=errors['input'], output_error_px=errors['output'],
            input_quality=quality, solver_eligible=k in eligible, solver_used=k in used,
            accepted_new_pose_fit=k in fit, accepted_new_pose_final_inlier=k in final,
            raw_candidate_final_inlier=k in raw_final, actually_reprojected=k in reproj,
            output_source=source, directly_visible_damage_5_to_10=bool(states[k] == 'DIRECT_VISIBLE' and
                errors['N3'] is not None and errors['output'] is not None and errors['N3'] <= 5 and errors['output'] > 10)))
    delta = {key: value(raw, key)-value(n3, key) for key in KEYS} if (
        raw['pose']['available'] and n3['pose']['available']) else None
    outcome = ('POSE_UNAVAILABLE' if delta is None else
               'BOTH_BETTER' if delta['translation_cm'] < 0 and delta['rotation_deg'] < 0 else
               'BOTH_WORSE' if delta['translation_cm'] > 0 and delta['rotation_deg'] > 0 else 'MIXED_OR_UNCHANGED')
    mask = ('MASK_NOT_APPLIED' if not raw['mask_audit']['mask_applied'] else
            'NO_KNOWN_VISIBILITY' if not raw['mask_audit']['known_ids'] else
            'WRONG_ON_KNOWN' if raw['mask_audit']['mask_wrong_on_known'] else 'MATCHES_ON_KNOWN')
    boundary = []
    for record in (raw.get('observation_contract') or {}).get('cornerwise_records', []):
        k = record['id']
        if raw['method'] in (C.PRIMARY, 'N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES'):
            label = raw['id']+'.heldout.'+str(k)
            audit.eq(record['initial_prior_includes_heldout_influence'],False,label+'.prior_influence')
            audit.eq(record['numeric_validation_pose_prior_used'],False,label+'.numeric_prior')
            audit.eq(record['projected_coordinate_used_as_observation'],False,label+'.no_generated_observation')
            audit.eq(set(record['heldout_fit_requested_excluded_ids']),set(raw['hidden_initial'])|{k},label+'.requested')
            if record['heldout_pose_calls']:
                ls = record['loo_solver']; blocked = set(raw['hidden_initial'])|{k}
                audit.eq(ls['prior_used'],False,label+'.solver_prior')
                audit.eq(set(ls['excluded']),blocked,label+'.solver_excluded')
                audit.eq(set(ls['hidden']),set(raw['hidden_initial']),label+'.actual_self_hidden_only')
                audit.eq(set(ls['temporary_excluded']),{k},label+'.temporary_heldout_only')
                for key in ('used','fit_input_ids','final_inliers','generator_ids'):
                    audit.eq(bool(blocked & set(ls.get(key,[]))),False,label+'.'+key)
        error = point_error(record['candidate_xy'], ref['native_points_px'][k]) if k in known else None
        prior_error = corners[k]['N3_error_px']
        boundary.append(dict(id=k, accepted=record['accepted'], reason=record['reason'],
            displayed_in_final_output=corners[k]['output_source'] == 'VALIDATED_BOUNDARY_INTERSECTION',
            candidate_error_px=error, N3_error_px=prior_error,
            delta_vs_N3_px=error-prior_error if error is not None and prior_error is not None else None,
            candidate_LOO_residual_px=record['candidate_LOO_residual_px'],
            native_N3_LOO_residual_px=record['native_N3_LOO_residual_px'],
            heldout_pose_calls=record['heldout_pose_calls'], validation_initial_prior_is_shared=bool(record['initial_prior_includes_heldout_influence']),
            physical_boundary_ownership_independently_verified=False))
    pools = {name: dict(ids=selected,
        quality_counts=dict(Counter(corners[k]['input_quality'] for k in selected)),
        human_direct_visible_ids=[k for k in selected if states[k] == 'DIRECT_VISIBLE'])
        for name, selected in [('eligible', eligible), ('solver_used', used), ('accepted_fit', fit),
                               ('accepted_final_inliers', final)]}
    return dict(id=raw['id'], session=raw['session'], method=raw['method'], output_status=raw['output_status'],
        new_pose_estimated=raw['new_pose_estimated'], fallback_used=raw['fallback_used'],
        corners=corners, pools=pools, boundary_candidates=boundary, mask_state=mask,
        pose_outcome_vs_N3=outcome, delta_vs_N3=delta, hidden_initial=raw['hidden_initial'],
        actually_reprojected_ids=reproj, hidden_set_changed=raw['hidden_set_changed'])


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
    for field, data in [('all_candidate_boundary_quality',boundary),
                        ('adopted_boundary_quality',[c for c in boundary if c['accepted']]),
                        ('displayed_boundary_quality',[c for c in boundary if c['displayed_in_final_output']])]:
        valid = [c for c in data if c['delta_vs_N3_px'] is not None]
        block=actual[field]
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
    result=dict(schema='same_observation_partial_line_saved_row_audit_v6',passed=False,complete=False,
                statistics_module_imported=False,new_model_score_PnP_ray_training_RGB_calls=0,
                independent_physical_GT_validated=False,existing_proxy_rows_only=True,
                actual_arithmetic_runs=1)
    audit=Audit()
    counters=Counter()
    try:
        raw=list(C.rows(folder/'PREDICTIONS.jsonl.gz'))
        fixed=list(C.rows(folder/'FIXED_PREDICTIONS.jsonl.gz'))
        recorded=C.read(folder/'METRICS.json')
        posthoc=list(C.rows(folder/'POSTHOC_ROWS.jsonl.gz'))
        cohort=C.read(args.cohort)
        with gzip.open(BOOTSTRAP,'rt') as stream:bootstrap=json.load(stream)
        paths=dict(new_scored_rows=folder/'PREDICTIONS.jsonl.gz',fixed_scored_rows=folder/'FIXED_PREDICTIONS.jsonl.gz',
            scoring_receipt=folder/'SCORING_RECEIPT.json',geometry_seal=folder/'GEOMETRY_SEAL.json',
            inference_receipt=folder/'INFERENCE_RECEIPT.json',
            old_v5_control_parity=folder/'V5_CONTROL_PARITY.json',
            cohort=args.cohort,bootstrap=BOOTSTRAP,
            statistics_code=Path(__file__).with_name('statistics.py'),protocol=args.protocol)
        before={k:C.binding(p) for k,p in paths.items()}
        audit.eq(recorded['bindings'],before,'recorded_input_bindings')
        audit.eq(recorded['posthoc_rows'],C.binding(folder/'POSTHOC_ROWS.jsonl.gz'),'posthoc_binding')
        inference=C.read(folder/'INFERENCE_RECEIPT.json')
        for key, expected in dict(complete=True,cleanup_error=None,actual_complete_frames=245,
                                 method_rows=980,fixed_rows=490).items():
            audit.eq(inference[key],expected,'inference_completion.'+key)
        scoring_receipt=C.read(folder/'SCORING_RECEIPT.json')
        audit.eq(scoring_receipt['inference_receipt'],C.binding(folder/'INFERENCE_RECEIPT.json'),
                 'inference_completion.scoring_binding')
        parity=C.read(folder/'V5_CONTROL_PARITY.json')
        audit.eq(parity['passed'],True,'old_v5_control_parity.PASS')
        audit.eq(parity['checked_before_GT_reference_reads'],True,'old_v5_control_parity.pre_GT')
        audit.eq(parity['reference_used_for_pose_or_selection'],False,'old_v5_control_parity.not_method_input')
        ids=cohort['ids']; labels={r['id']:r['label'] for r in cohort['frames']}
        methods=('BASE','N3_SUBPIX')+tuple(C.METHODS)
        by={m:{} for m in methods}
        for row in fixed+raw:
            C.require(row['method'] in by and row['id'] not in by[row['method']],'duplicate/mismatched saved row')
            by[row['method']][row['id']]=row
        for m in methods:audit.eq(set(by[m]),set(ids),'cohort.'+m)
        audit.eq(len(raw),980,'raw_rows');audit.eq(len(fixed),490,'fixed_rows')
        audit.eq(len(posthoc),980,'posthoc_rows')
        counters.update(raw_scored_rows_loaded=len(raw)+len(fixed),posthoc_rows_loaded=len(posthoc))
        post_by={(r['method'],r['id']):r for r in posthoc}
        audit.eq(len(post_by),len(posthoc),'posthoc_uniqueness')
        derived=[]
        for row in raw:
            expected=derive_posthoc(row,by['N3_SUBPIX'][row['id']],audit)
            tree_check(audit,post_by[(row['method'],row['id'])],expected,'posthoc.'+row['method']+'.'+row['id'])
            derived.append(expected)
        sessions=bootstrap['sessions'];draws=bootstrap['counts'];si={s:i for i,s in enumerate(sessions)}
        C.require(len(sessions)==13 and len(si)==13 and len(draws)==10000 and
            all(len(d)==13 and sum(d)==13 and all(isinstance(v,int) and v>=0 for v in d) for d in draws),
            'frozen session multiplicities differ')
        audit.eq(recorded['bootstrap']['serialized_raw_sha256'],bootstrap['serialized_raw_sha256'],'bootstrap_raw_SHA')
        audit.eq(recorded['bootstrap']['new_draws_generated'],0,'new_bootstrap_draws')
        contrasts=[(C.PRIMARY,m) for m in
            ('N3_SUBPIX','N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES','N3_INDEPENDENT_ROBUST_H',
             'N3_INDEPENDENT_ROBUST_NO_MASK','BASE')]+[
            ('N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK'),
            ('N3_INDEPENDENT_ROBUST_NO_MASK','N3_SUBPIX')]
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
        audit.eq(len(csv_rows),162,'CSV_rows')
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
