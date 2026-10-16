"""Independent saved-score outlier recount; no inference imports or GT loading."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ['POSEFIX_RAW', 'R0', 'N0', 'N1', 'E0']
ERRORS = ('translation_cm', 'rotation_deg')
PAIRS = [('N1', 'N0'), ('E0', 'N0'), ('N0', 'POSEFIX_RAW'),
         ('N1', 'POSEFIX_RAW'), ('E0', 'POSEFIX_RAW')]
VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def distribution(values):
    return dict(n=len(values), mean=statistics.fmean(values),
        variance_sample=statistics.variance(values), SD_sample=statistics.stdev(values),
        median=statistics.median(values), max=max(values))


def success(row, arm):
    metric = row['metrics'][arm]
    return metric['translation_cm'] <= 5 and metric['rotation_deg'] <= 5


def summarize(rows):
    arms, paired = {}, {}
    for arm in ARMS:
        arms[arm] = dict(full_n=len(rows), success=sum(success(r, arm) for r in rows),
            errors={k: distribution([r['metrics'][arm][k] for r in rows]) for k in ERRORS})
    for arm, baseline in PAIRS:
        groups = {k: [] for k in ('recovery', 'damage', 'both_success', 'both_failure')}
        for row in rows:
            a, b = success(row, arm), success(row, baseline)
            key = 'both_success' if a and b else 'recovery' if a else 'damage' if b else 'both_failure'
            groups[key].append(row['id'])
        paired[arm+'-'+baseline] = dict(
            transition={k: dict(count=len(v), ids=v) for k, v in groups.items()},
            mean_difference={k: statistics.fmean(r['metrics'][arm][k] - r['metrics'][baseline][k]
                for r in rows) for k in ERRORS})
    return dict(full_n=len(rows), arms=arms, paired=paired)


def run(source_root, single_score, guarded_score, observed, prior, output):
    p95 = source_root / VIS / 'posefix_sub_comparison_v1/SCORED_ROWS.jsonl.gz'
    p143 = source_root / VIS / 'remainder_after_test49_v1/posefix_baseline_supplement_v4_14/POSTSEAL_SCORE.json'
    old_prior_hash = sha(prior)
    single, guarded = read(single_score)['rows'], read(guarded_score)['rows']
    assert [r['id'] for r in single] == [r['id'] for r in guarded]
    with gzip.open(p95, 'rt') as stream:
        old = [json.loads(line) for line in stream]
    pf = {r['id']: r['metric'] for r in old if r['method'] == 'POSEFIX_RAW'}
    pf.update({r['id']: r['metrics']['POSEFIX_RAW'] for r in read(p143)['rows']})
    rows = []
    for s, g in zip(single, guarded):
        if s['partition'] == 'CAL':
            continue
        metrics = {a: s['metrics'][a] for a in ('R0', 'N0', 'N1')}
        metrics.update(E0=g['metrics']['E0'], POSEFIX_RAW=pf[s['id']])
        assert all(m[k] is not None and math.isfinite(m[k]) for m in metrics.values() for k in ERRORS)
        rows.append(dict(id=s['id'], session=s['session'], metrics=metrics))
    assert len(rows) == len({r['id'] for r in rows}) == 192
    thresholds = {}
    for key in ERRORS:
        values = [r['metrics']['N0'][key] for r in rows]
        q1, q3 = quantile(values, .25), quantile(values, .75)
        thresholds[key] = dict(Q1=q1, Q3=q3, IQR=q3-q1, upper=q3+1.5*(q3-q1))
    top10 = {r['id'] for r in sorted(rows, key=lambda r: (-r['metrics']['N0']['translation_cm'], r['id']))[:10]}
    predicates = {
        'FULL192': lambda r: False,
        'N0_IQR_T_R': lambda r: any(r['metrics']['N0'][k] > thresholds[k]['upper'] for k in ERRORS),
        'ALL5_IQR_T_R': lambda r: any(r['metrics'][a][k] > thresholds[k]['upper'] for a in ARMS for k in ERRORS),
        'N0_IQR_T_ONLY': lambda r: r['metrics']['N0']['translation_cm'] > thresholds['translation_cm']['upper'],
        'N0_T_GT50': lambda r: r['metrics']['N0']['translation_cm'] > 50,
        'ALL5_T_GT50': lambda r: any(r['metrics'][a]['translation_cm'] > 50 for a in ARMS),
        'N0_TOP10_T': lambda r: r['id'] in top10,
    }
    cohorts = {}
    for name, predicate in predicates.items():
        excluded = [r for r in rows if predicate(r)]
        retained = [r for r in rows if not predicate(r)]
        cohorts[name] = dict(original_n=192, excluded_n=len(excluded),
            excluded_ids=[r['id'] for r in excluded], retained_ids=[r['id'] for r in retained],
            summary=summarize(retained))
    # Root results are first decoded after the independent calculations above.
    actual = read(observed)
    failures, numbers = [], []
    checks = 0
    def compare(a, b, path):
        nonlocal checks
        checks += 1
        if isinstance(a, dict):
            for k, value in a.items():
                if k not in b:
                    failures.append(dict(path=path+'.'+k, missing=True))
                else:
                    compare(value, b[k], path+'.'+k)
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            difference = abs(a-b)
            numbers.append(difference)
            if difference > 1e-9:
                failures.append(dict(path=path, independent=a, observed=b, difference=difference))
        elif a != b:
            failures.append(dict(path=path, independent=a, observed=b))
    compare(set(cohorts) == set(actual['cohorts']), True, 'all7cohorts')
    compare(thresholds, actual['thresholds'], 'thresholds')
    for name, cohort in cohorts.items():
        compare(cohort, actual['cohorts'][name], 'cohorts.'+name)
    for name, digest in read(observed.parent/'SEAL.json')['assets'].items():
        compare(sha(observed.parent/name), digest, 'root_seal.'+name)
    full = cohorts['FULL192']['summary']
    union = cohorts['ALL5_IQR_T_R']
    removed = {}
    excluded_set = set(union['excluded_ids'])
    for name, pair in full['paired'].items():
        removed[name] = {}
        for transition in ('recovery', 'damage'):
            ids = [i for i in pair['transition'][transition]['ids'] if i in excluded_set]
            removed[name][transition] = dict(count=len(ids), ids=ids)
    result = dict(schema='INDEPENDENT_COMMON_SET_OUTLIER_RECOUNT_V2',
        status='PASS' if not failures else 'FAIL', checks=checks,
        failure_count=len(failures), failures=failures, absolute_tolerance=1e-9,
        max_numeric_difference=max(numbers), independent_v1_sha256_preserved=old_prior_hash,
        input_bindings={'PRIVATE:single_score/SCORED_ROWS.json': sha(single_score),
            'PRIVATE:guarded_score/SCORED_ROWS.json': sha(guarded_score),
            str(p95.relative_to(source_root)): sha(p95), str(p143.relative_to(source_root)): sha(p143),
            'PRIVATE:posefix_gap_audit/outlier_v2/OUTLIER_RESULTS.json': sha(observed)},
        thresholds=thresholds, cohorts=cohorts,
        ALL5_IQR_selection_effect=dict(excluded_n=union['excluded_n'],
            fraction_of_original192_excluded=union['excluded_n']/192,
            same_retained_set_for_all5_methods=True,
            excluded_successes_per_method={a: full['arms'][a]['success']-union['summary']['arms'][a]['success'] for a in ARMS},
            removed_joint_transition_cases=removed,
            limitation='The71/192 exclusions remove newly damaged cases too. This outcome-dependent37percent trimming changes case mix, cannot establish safety, and does not replace the full192 evaluation.'),
        implementation_independence='Original saved single/guarded/PoseFix journals; no root-module or root-CSV import. Standard-library statistics and explicit linear quantiles independently check root NumPy summaries.',
        independent_source_sha256=sha(Path(__file__)),
        new_model_PnP_annotation_image_training_seed_or_GT_edit_calls=0)
    assert sha(prior) == old_prior_hash
    with output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'single-score', 'guarded-score', 'observed', 'prior', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root, args.single_score, args.guarded_score, args.observed, args.prior, args.output)
    print(json.dumps(dict(status=result['status'], checks=result['checks'], failures=result['failure_count'],
        max_difference=result['max_numeric_difference'], output_sha256=sha(args.output),
        cohorts={k: dict(excluded=v['excluded_n'], retained=v['summary']['full_n']) for k, v in result['cohorts'].items()})))
