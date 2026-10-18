"""Independent post-seal audit of all-action-failure H1 diagnosis.

Only the declarative registry and the earlier independent NumPy arithmetic
are imported. Production action construction, metrics and table aggregation
are not imported. Reference decoding follows explicit GO and immutable seals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import resource
import time

from scripts.research.pallet_statewise_end_to_end_recovery_v1 import independent_check as I
from .registry import ALL_ARMS, L_ARMS, POPULATIONS, EXPECTED_N

ARMS = tuple(a for a in L_ARMS if '_H1_' in a)
COMPARATORS = ('N0', 'E0', 'E0_REGENERATED')
METRICS = (('T', 'translation_cm'), ('R', 'rotation_deg'))


def indexed(records, name):
    result = {}
    for row in records:
        key = row.get('id')
        if not isinstance(key, str) or key in result:
            raise ValueError('Invalid/duplicate ' + name + ' ID: ' + repr(key))
        result[key] = row
    return result


def exact_identities(values, keys, expected, name, problems):
    counts = Counter(tuple(r.get(k) for k in keys) for r in values)
    wanted = set(expected)
    duplicate = [dict(identity=list(k), n=n) for k, n in counts.items() if n != 1]
    missing = wanted - set(counts)
    unknown = set(counts) - wanted
    if duplicate or missing or unknown:
        problems.append(dict(path=name, kind='EXACT_IDENTITY_COVERAGE', duplicates=duplicate,
            missing=sorted(map(list, missing)), unexpected=sorted(map(list, unknown))))


def near(first, second, tolerance, kind='aggregate'):
    absolute = tolerance.get(kind + '_abs', tolerance['aggregate_abs'])
    return abs(first-second) <= absolute + tolerance['relative']*max(abs(first), abs(second))


def compare(actual, reported, path, tolerance, problems):
    if isinstance(actual, dict):
        if not isinstance(reported, dict):
            problems.append(dict(path=path, kind='DICT_TYPE')); return
        for key, value in actual.items():
            if key not in reported:
                problems.append(dict(path=path+'/'+key, kind='MISSING_FIELD'))
            else:
                compare(value, reported[key], path+'/'+key, tolerance, problems)
    elif isinstance(actual, list):
        if not isinstance(reported, list) or len(actual) != len(reported):
            problems.append(dict(path=path, kind='LIST_LENGTH')); return
        for i, (a, b) in enumerate(zip(actual, reported)):
            compare(a, b, path+'/'+str(i), tolerance, problems)
    elif isinstance(actual, bool) or actual is None or isinstance(actual, (str, int)):
        if type(actual) is not type(reported) or actual != reported:
            problems.append(dict(path=path, kind='EXACT', independent=actual, reported=reported))
    elif not isinstance(reported, (int, float)) or isinstance(reported, bool) or not near(actual, reported, tolerance):
        problems.append(dict(path=path, kind='NUMERIC', independent=actual, reported=reported))


def statuses(entry):
    raw = entry.get('prediction_status')
    value = dict(raw) if isinstance(raw, dict) else {}
    for key in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout', 'status', 'reason'):
        if key not in value:
            value[key] = entry.get(key)
    if isinstance(raw, str):
        value['status'] = raw
    return value


def same_pose(first, second, scale):
    if not isinstance(first, dict) or not isinstance(second, dict): return False
    if first.get('available') is not True or second.get('available') is not True: return False
    try:
        return (all(abs(float(x)-float(y)) <= 1e-7*scale for x, y in zip(first['centroid'], second['centroid']))
            and all(abs(float(x)-float(y)) <= 1e-7 for a, b in zip(first['R_physical'], second['R_physical']) for x, y in zip(a, b)))
    except (KeyError, TypeError, ValueError):
        return False


def source_identity(pose, baselines, scale):
    if not isinstance(pose, dict) or pose.get('available') is not True:
        return dict(role='NO_POSE', aliases=[])
    aliases = [a for a in ('N0', 'E0_REGENERATED', 'E0') if same_pose(pose, baselines[a], scale)]
    role = ('N0' if 'N0' in aliases else 'E0_REGENERATED' if 'E0_REGENERATED' in aliases
        else 'HISTORICAL_E0' if 'E0' in aliases else 'OTHER')
    return dict(role=role, aliases=aliases)


def resolve_catalog(entry, feature):
    """Resolve archived actual-action IDs, never manufacture a new pose."""
    candidates = {c['candidate_id']: c for c in feature['candidates']}
    if len(candidates) != len(feature['candidates']):
        raise ValueError('Duplicate feature candidate ID')
    result = []
    for item in entry.get('actions', []):
        cid = item['candidate_id']
        if cid == 'H0_NONE': raise ValueError('H1 catalog contains virtual none')
        if cid in candidates: pose = I.candidate_pose(candidates[cid])
        elif cid == 'N0': pose = feature['fallback']
        elif cid == 'E0_REGENERATED': pose = feature['e0_regenerated']
        else: raise ValueError('Unresolvable H1 action: ' + str(cid))
        if any(same_pose(pose, previous[1], feature.get('object_scale_m', 1.)) for previous in result):
            raise ValueError('Duplicate physical H1 actions')
        result.append((cid, pose))
    if len({item[0] for item in result}) != len(result):
        raise ValueError('Duplicate H1 catalog ID')
    return result


def moments(values):
    if not values:
        return dict(n=0, sum=None, mean=None, min=None, max=None, aggregate_nonfinite=False)
    # Ordinary sum deliberately reproduces archived raw-expression moments;
    # stable arithmetic is retained separately, never used to change signs.
    total = sum(values); finite = math.isfinite(total)
    return dict(n=len(values), sum=total if finite else None,
        mean=total/len(values) if finite else None, min=min(values), max=max(values), aggregate_nonfinite=not finite)


def one_detail(identity, arm, comparator, selected, baseline, stage, selected_source, baseline_source, selected_status, baseline_status):
    result = dict(**identity, arm=arm, comparator=comparator,
        selected_candidate_id=stage['selected_candidate_id'],
        eligible_candidate_count=stage['candidate_count'], eligible_available_n=stage['available_n'],
        eligible_has_success=False, selected_source=selected_source['role'],
        selected_aliases_json=json.dumps(selected_source['aliases'], ensure_ascii=False, sort_keys=True),
        comparator_source=baseline_source['role'], selected_fallback=selected_status.get('fallback'),
        selected_status=selected_status.get('status'), selected_technical_failure=selected_status.get('technical_failure'),
        comparator_fallback=baseline_status.get('fallback'), comparator_status=baseline_status.get('status'))
    for prefix, side in (('selected', selected), ('comparator', baseline)):
        result.update({prefix+'_'+key: value for key, value in side.items() if key not in ('translation_cm', 'rotation_deg')})
    for axis, key in METRICS:
        result['selected_'+key] = selected[key]; result['comparator_'+key] = baseline[key]
        reasons = []
        for side in (selected, baseline):
            missing = []
            if not side['actual_pose_present']: missing.append('missing_actual_pose')
            elif side['actual_pose_available'] is not True:
                missing.append('actual_pose_unavailable' if side['actual_pose_available'] is False else 'actual_pose_availability_unknown')
            if not side['metric_present']: missing.append('missing_metric')
            elif side['metric_pose_available'] is not True:
                missing.append('metric_pose_unavailable' if side['metric_pose_available'] is False else 'metric_pose_availability_unknown')
            number = side[key]
            if number is None: missing.append('null_error')
            elif isinstance(number, bool) or not isinstance(number, (int, float)): missing.append('invalid_error_type')
            elif not math.isfinite(number): missing.append('nonfinite_error')
            reasons.append(missing)
        result[axis+'_selected_exclusion_reasons'] = ';'.join(reasons[0])
        result[axis+'_comparator_exclusion_reasons'] = ';'.join(reasons[1])
        comparable = not any(reasons)
        delta = selected[key]-baseline[key] if comparable else None
        overflow = delta is not None and not math.isfinite(delta)
        if overflow: comparable, delta = False, None
        result[axis+'_delta_exclusion_reason'] = 'nonfinite_subtraction' if overflow else ''
        result[axis+'_comparable'] = comparable; result[axis+'_delta'] = delta
        result[axis+'_worse'] = delta > 0 if comparable else None
        result[axis+'_improved'] = delta < 0 if comparable else None
        result[axis+'_equal'] = delta == 0 if comparable else None
    result['both_axes_comparable'] = result['T_comparable'] and result['R_comparable']
    result['worse_either'] = result['T_worse'] or result['R_worse'] if result['both_axes_comparable'] else None
    result['improved_either'] = result['T_improved'] or result['R_improved'] if result['both_axes_comparable'] else None
    result['equal_both'] = result['T_equal'] and result['R_equal'] if result['both_axes_comparable'] else None
    return result


def summarize(records, population, arm, comparator, denominator):
    n = len(records)
    value = dict(population=population, arm=arm, comparator=comparator,
        population_denominator=denominator, all_failure_n=n, not_all_failure_n=denominator-n,
        both_axes_comparable_n=sum(r['both_axes_comparable'] for r in records),
        both_axes_excluded_n=sum(not r['both_axes_comparable'] for r in records),
        worseEither_n=sum(r['worse_either'] is True for r in records),
        improvedEither_n=sum(r['improved_either'] is True for r in records),
        equalBoth_n=sum(r['equal_both'] is True for r in records),
        selected_source_counts=dict(Counter(r['selected_source'] or 'UNKNOWN' for r in records)),
        selected_fallback_counts=dict(Counter('true' if r['selected_fallback'] is True else 'false' if r['selected_fallback'] is False else 'unknown' for r in records)),
        selected_status_counts=dict(Counter(str(r['selected_status']) if r['selected_status'] is not None else 'UNKNOWN' for r in records)))
    for prefix in ('selected', 'comparator'):
        for flag in ('actual_pose_present', 'actual_pose_available', 'metric_present', 'metric_pose_available'):
            values = [r[prefix+'_'+flag] for r in records]
            for label, count in (('true', sum(v is True for v in values)), ('false', sum(v is False for v in values)), ('unknown', sum(v is not True and v is not False for v in values))):
                value[prefix+'_'+flag+'_'+label+'_n'] = count
    for axis, _ in METRICS:
        deltas = [r[axis+'_delta'] for r in records if r[axis+'_comparable']]
        value[axis+'_comparable_n'] = len(deltas); value[axis+'_excluded_n'] = n-len(deltas)
        value[axis+'_delta'] = moments(deltas)
        for direction in ('worse', 'improved', 'equal'):
            subset = [r[axis+'_delta'] for r in records if r[axis+'_'+direction] is True]
            value[axis+'_'+direction+'_n'] = len(subset)
            value[axis+'_'+direction+'_delta'] = moments(subset)
        for suffix in ('selected_exclusion_reasons', 'comparator_exclusion_reasons', 'delta_exclusion_reason'):
            key = axis+'_'+suffix
            output_key = key+'s' if suffix == 'delta_exclusion_reason' else key
            value[output_key] = dict(Counter(reason for r in records for reason in r[key].split(';') if reason))
    return value


def metric_side(pose, reference):
    stable = I.pose_metric(pose, reference); raw = I.strict_direction_values(pose, reference)
    return dict(actual_pose_present=isinstance(pose, dict), actual_pose_available=pose.get('available') if isinstance(pose, dict) else None,
        metric_present=True, metric_pose_available=stable['pose_available'], success=stable['success'], metric_error=None,
        **raw), stable


def guard(run_dir, original_score, evaluation, analysis, scoring_GO):
    if not scoring_GO:
        raise PermissionError('Explicit root postseal scoring GO required before any input or reference decode')
    paths = dict(plan=run_dir/'PLAN_LOCK.json', gate=run_dir/'ALL_OUTPUTS_SCORING_GO.json',
        prediction_seal=run_dir/'PREDICTION_SEAL.json', predictions=run_dir/'PREDICTIONS.jsonl.gz',
        evaluation_seal=evaluation/'EVALUATION_SEAL.json', analysis_seal=analysis/'ANALYSIS_SEAL.json',
        l_seal=run_dir/'action_predictions_v1/PREDICTION_SEAL.json', l_predictions=run_dir/'action_predictions_v1/PREDICTIONS.jsonl.gz',
        feature_seal=run_dir/'d1_real_candidates_v1/FEATURE_SEAL.json', features=run_dir/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz',
        protocol=run_dir/'EVALUATION_PROTOCOL.json', old_score_seal=original_score/'SCORE_SEAL.json',
        references=original_score/'FIXED_REFERENCES.json')
    # Reference payload is hashed only; it is not decoded here.
    hashes = {k: I.sha(p) for k, p in paths.items()}
    pred = I.read(paths['prediction_seal']); gate = I.read(paths['gate']); ev = I.read(paths['evaluation_seal'])
    ls = I.read(paths['l_seal']); fs = I.read(paths['feature_seal']); a = I.read(paths['analysis_seal'])
    if pred.get('complete') is not True or pred.get('before_scoring') is not True or pred.get('reference_inputs') is not False:
        raise ValueError('Prediction-only pre-reference complete seal required')
    if gate.get('all_registered_predictions_sealed') is not True or gate.get('prediction_seal_sha256') != hashes['prediction_seal'] or gate.get('plan_sha256') != hashes['plan']:
        raise ValueError('Actual all-output scoring GO/plan binding required')
    if pred.get('plan_lock_sha256') != hashes['plan'] or pred.get('predictions_sha256') != hashes['predictions'] or pred.get('evaluation_protocol_sha256') != hashes['protocol']:
        raise ValueError('Prediction immutable inputs changed')
    if set(pred.get('arms', [])) != set(ALL_ARMS) or len(pred.get('arms', [])) != 37:
        raise ValueError('Exact predeclared37 arm inventory required')
    if len(pred.get('ordered_ids', [])) != 238 or len(set(pred['ordered_ids'])) != 238 or pred.get('rows') != 238:
        raise ValueError('Exact238 identity inventory required')
    if ev.get('schema') != 'BOUNDARY_ACTION_EVALUATION_SEAL_V1' or ev.get('complete') is not True or ev.get('all_failures_retained') is not True:
        raise ValueError('Actual complete evaluation seal required')
    if ev.get('prediction_seal_sha256') != hashes['prediction_seal'] or ev.get('input_predictions_sha256') != hashes['predictions']:
        raise ValueError('Evaluation prediction identity changed')
    if ev.get('registry_sha256') != I.sha(Path(__file__).with_name('registry.py')):
        raise ValueError('Registry changed since evaluation')
    for directory, document in ((evaluation, ev), (analysis, a)):
        if document.get('complete') is not True: raise ValueError('Incomplete analysis/evaluation')
        for filename, digest in document['artifacts'].items():
            if I.sha(directory/filename) != digest: raise ValueError('Changed artifact: '+filename)
    for name in ('SCORED_ROWS.jsonl.gz', 'STAGE_DIAGNOSIS.jsonl.gz'):
        if ev['artifacts'].get(name) is None: raise ValueError('Missing evaluated journal binding')
    if a.get('input_bindings', {}).get('evaluation_seal', {}).get('sha256') != hashes['evaluation_seal']:
        raise ValueError('Analysis used another evaluation')
    if ls.get('complete') is not True or ls.get('predictions_sha256') != hashes['l_predictions'] or ls.get('features_sha256') != hashes['features'] or ls.get('feature_seal_sha256') != hashes['feature_seal']:
        raise ValueError('Common L catalog/bank binding changed')
    if fs.get('complete') is not True or fs.get('head_features_sha256') != hashes['features']:
        raise ValueError('Common feature seal incomplete or changed')
    if pred.get('input_bindings', {}).get('L') != hashes['l_predictions'] or pred.get('input_bindings', {}).get('real') != hashes['features']:
        raise ValueError('Assembled predictions used another L catalog or real bank')
    oldseal = I.read(paths['old_score_seal'])
    if ev.get('reference_sha256') != hashes['references'] or oldseal.get('artifacts', {}).get('FIXED_REFERENCES.json') != hashes['references']:
        raise ValueError('Original immutable reference bytes changed')
    return paths, hashes, pred


def run(run_dir, original_score, evaluation, analysis, output, *, scoring_GO=False):
    started, cpu = time.monotonic(), time.process_time()
    run_dir, original_score, evaluation, analysis, output = map(Path, (run_dir, original_score, evaluation, analysis, output))
    if output.exists(): raise FileExistsError('Preserve previous audit: '+str(output))
    paths, hashes, seal = guard(run_dir, original_score, evaluation, analysis, scoring_GO)
    # This is the first reference payload decode, after every immutable gate.
    references = I.read(paths['references'])['references']
    ledger = indexed(I.rows(paths['predictions']), 'prediction')
    catalogs = indexed(I.rows(paths['l_predictions']), 'catalog')
    features = indexed(I.rows(paths['features']), 'feature')
    stages = indexed(I.rows(evaluation/'STAGE_DIAGNOSIS.jsonl.gz'), 'stage')
    scored = indexed(I.rows(evaluation/'SCORED_ROWS.jsonl.gz'), 'scored')
    for name, values in (('prediction', ledger), ('catalog', catalogs), ('feature', features), ('stage', stages), ('scored', scored)):
        if list(values) != seal['ordered_ids']: raise ValueError('Changed238 order/identity: '+name)
    tolerance = I.read(paths['protocol'])['independent_tolerance']
    problems, details, representation = [], [], []
    all_failure = {a: [] for a in ARMS}; stage_checks = 0; action_checks = 0
    observed = Counter(row['partition'] for row in ledger.values())
    if observed != Counter(TEST=49, REMAINDER=143, CAL=46): raise ValueError('Original partition denominator changed')
    for fid, row in ledger.items():
        identity = {k: row[k] for k in ('id', 'partition', 'session', 'cluster_id')}
        reference = references[row['partition']][fid]
        feature = features[fid]; selected_entries = row['predictions']
        baseline_poses = {a: selected_entries[a]['actual_pose'] for a in COMPARATORS}
        sides, stable = {}, {}
        for arm in ARMS+COMPARATORS:
            sides[arm], stable[arm] = metric_side(selected_entries[arm]['actual_pose'], reference)
        for arm in ARMS:
            archived = catalogs[fid]['predictions'][arm]
            if archived != selected_entries[arm]:
                problems.append(dict(path=fid+'/'+arm, kind='ASSEMBLED_CATALOG_CHANGED'))
            resolved = resolve_catalog(archived, feature)
            catalog_metrics = [(cid, I.pose_metric(pose, reference)) for cid, pose in resolved]
            action_checks += len(catalog_metrics)
            success_ids = [cid for cid, metric in catalog_metrics if metric['success']]
            actual_stage = dict(candidate_count=len(resolved), available_n=sum(m['pose_available'] for _, m in catalog_metrics),
                has_success=bool(success_ids), success_ids=success_ids)
            stage = stages[fid]['arms'][arm]
            compare(actual_stage, stage['after_constraints'], fid+'/'+arm+'/stage', tolerance, problems); stage_checks += 1
            compare(stable[arm]['success'], stage['final_success'], fid+'/'+arm+'/final_success', tolerance, problems)
            compare(archived.get('selected_candidate_id'), stage.get('selected_candidate_id'), fid+'/'+arm+'/selectedID', tolerance, problems)
            if resolved:
                selected = [pose for cid, pose in resolved if cid == archived.get('selected_candidate_id')]
                if len(selected) != 1 or not same_pose(selected[0], archived['actual_pose'], feature.get('object_scale_m', 1.)):
                    problems.append(dict(path=fid+'/'+arm, kind='SELECTED_POSE_NOT_CATALOG_ACTION'))
            for key in METRICS:
                name = key[1]; original, stable_value = sides[arm][name], stable[arm][name]
                if original is not None and stable_value is not None and original != stable_value:
                    representation.append(dict(id=fid, arm=arm, metric=name, original_expression=original, stable_expression=stable_value, difference=stable_value-original))
            if actual_stage['has_success']: continue
            all_failure[arm].append(fid)
            if stable[arm]['success']:
                problems.append(dict(path=fid+'/'+arm, kind='SELECTED_SUCCESS_DESPITE_ALL_ACTION_FAILURE'))
            own_source = source_identity(archived['actual_pose'], baseline_poses, row['object_scale_m'])
            for comparator in COMPARATORS:
                own = one_detail(identity, arm, comparator, sides[arm], sides[comparator],
                    dict(selected_candidate_id=archived.get('selected_candidate_id'), **actual_stage), own_source,
                    source_identity(baseline_poses[comparator], baseline_poses, row['object_scale_m']),
                    statuses(archived), statuses(selected_entries[comparator]))
                details.append(own)
                for axis, key in METRICS:
                    if all(stable[a][key] is not None for a in (arm, comparator)) and own[axis+'_comparable']:
                        stable_delta = stable[arm][key]-stable[comparator][key]; raw_delta = own[axis+'_delta']
                        sign = lambda x: (x > 0)-(x < 0)
                        if stable_delta != raw_delta:
                            representation.append(dict(id=fid, arm=arm, comparator=comparator, metric=key,
                                original_delta=raw_delta, stable_delta=stable_delta, direction_changed=sign(raw_delta)!=sign(stable_delta)))
    summaries = [summarize([r for r in details if r['arm']==arm and r['comparator']==comparator and r['partition'] in parts],
        pop, arm, comparator, EXPECTED_N[pop]) for pop, parts in POPULATIONS.items() for arm in ARMS for comparator in COMPARATORS]
    reported = I.read(analysis/'H1_ALL_FAILURE_SUMMARY.json')
    compare(all_failure, reported['all_failure_ids_by_arm'], 'all_failure_ids', tolerance, problems)
    for key, value in dict(complete=True, arms=list(ARMS), comparators=list(COMPARATORS), populations={k:list(v) for k,v in POPULATIONS.items()},
        population_denominators=EXPECTED_N, full_frame_n=238, detail_rows=len(details), summary_rows=180).items():
        compare(value, reported.get(key), 'summary/'+key, tolerance, problems)
    exact_identities(reported['rows'], ('population','arm','comparator'), ((p,a,c)for p in POPULATIONS for a in ARMS for c in COMPARATORS), 'summary.rows', problems)
    reported_summaries = {(r['population'],r['arm'],r['comparator']):r for r in reported['rows']}
    for value in summaries:
        key = tuple(value[k]for k in ('population','arm','comparator'))
        compare(value, reported_summaries.get(key), '/'.join(key), tolerance, problems)
    with (analysis/'H1_ALL_FAILURE_ROWS.csv').open(newline='') as stream:
        csv_values = list(csv.DictReader(stream))
    exact_identities(csv_values, ('id','arm','comparator'), ((r['id'],r['arm'],r['comparator'])for r in details), 'detail.csv', problems)
    csv_index = {(r['id'],r['arm'],r['comparator']):r for r in csv_values}
    scalar_checks = 0
    for value in details:
        key = tuple(value[k]for k in ('id','arm','comparator')); saved = csv_index.get(key, {})
        for field, expected in value.items():
            text = saved.get(field)
            if isinstance(expected, bool): actual = True if text=='True' else False if text=='False' else text
            elif expected is None: actual = None if text=='' else text
            elif isinstance(expected, int):
                try: actual = int(text)
                except (ValueError,TypeError): actual = text
            elif isinstance(expected, float):
                try: actual = float(text)
                except (ValueError,TypeError): actual = text
            else: actual = text
            compare(expected, actual, '/'.join(key)+'/'+field, tolerance, problems); scalar_checks += 1
    for name, path in paths.items():
        if I.sha(path) != hashes[name]: raise ValueError('Input changed during independent audit: '+name)
    output.mkdir(parents=True, exist_ok=False)
    I.write(output/'REPRESENTATION_DIFFERENCES.json', dict(schema='H1_FAILURE_ORIGINAL_VS_STABLE_REPRESENTATION_V1',
        count=len(representation), differences=representation, strict_count_basis='Independent original norm/acos/sin(pi) expression; stable differences never reclassified.'))
    I.write(output/'INDEPENDENT_SUMMARIES.json', dict(rows=summaries, all_failure_ids_by_arm=all_failure))
    receipt = dict(schema='INDEPENDENT_H1_ALL_FAILURE_TABLE_AUDIT_V1', complete=True,
        status='PASS_WITH_DOCUMENTED_REPRESENTATION_DIFFERENCES' if not problems and representation else 'PASS' if not problems else 'FAIL',
        failures=len(problems), problems=problems, frame_n=238, H1_arm_n=len(ARMS), comparator_n=3,
        summary_rows=len(summaries), detail_rows=len(details), stage_membership_checks=stage_checks,
        actual_catalog_pose_metric_evaluations=action_checks, detail_scalar_checks=scalar_checks,
        representation_differences=len(representation), input_bindings=hashes, source_sha256=I.sha(__file__),
        independent_metric_source_sha256=I.sha(I.__file__), registry_sha256=I.sha(Path(__file__).with_name('registry.py')),
        reference_decode_after_all_output_and_evaluation_gates=True, production_action_metric_aggregate_imports=False,
        optimizer_model_forward_or_selection_changes=0, previous_action_audit_modified=False,
        cost=dict(wall_seconds=time.monotonic()-started, CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        artifacts={p.name:I.sha(p)for p in output.iterdir()if p.is_file()})
    I.write(output/'INDEPENDENT_H1_FAILURE_CHECK.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','original-score','evaluation','analysis','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--scoring-GO', action='store_true')
    args = p.parse_args()
    result = run(args.run_dir,args.original_score,args.evaluation,args.analysis,args.output,scoring_GO=args.scoring_GO)
    print(json.dumps({k:v for k,v in result.items()if k!='problems'}, allow_nan=False))
    if result['failures']: raise SystemExit(1)


if __name__ == '__main__': main()
