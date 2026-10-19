"""Constructed metadata only: no real data, plots, poses, refs or scientific API."""
from copy import deepcopy
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
from . import export_results as E

SECRET = 'PRIVATE_POSE_REFERENCE_WEIGHT_SENTINEL'


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def rewrite_constructed(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def distribution(values, n):
    a = sorted(x for x in values if x is not None)
    def percentile(p):
        j = (len(a) - 1) * p
        lo, hi = math.floor(j), math.ceil(j)
        return a[lo] + (a[hi] - a[lo]) * (j - lo)
    return dict(n=len(a), missing=n-len(a), full_denominator=n, mean=statistics.mean(a),
        sample_variance=statistics.variance(a), sample_std=statistics.stdev(a), median=statistics.median(a),
        P90=percentile(.9), P95=percentile(.95), max=max(a), ddof=1, available_only=True, tail_trimming=False)


def fixture(root):
    rows = []
    for i in range(238):
        metric = dict(translation_cm=1., rotation_deg=2., success=True, pose_available=True, R=SECRET, t=SECRET)
        if i == 5: metric.update(translation_cm=10000., rotation_deg=179., success=False)
        if i == 6: metric.update(translation_cm=None, rotation_deg=None, success=False, pose_available=False)
        rows.append(dict(id=f'CONSTRUCTED_{i:03d}', partition='TEST' if i<49 else 'REMAINDER' if i<192 else 'CAL',
            session='SESSION', cluster_id='CLUSTER_'+str(i%6), metrics={'BASE_s20261018': deepcopy(metric),
            'NEW_s20261018': deepcopy(metric), 'NEW_s20261019': deepcopy(metric)}, poses=SECRET, reference=SECRET))
    sources = []
    for source_index, extra in enumerate(('NEW_s20261018', 'NEW_s20261019')):
        arms = ['BASE_s20261018', extra]
        actual = [dict(r, metrics={a: r['metrics'][a] for a in arms}) for r in rows]
        summary = {}
        for pop, parts in E.POPS.items():
            subset = [r for r in actual if r['partition'] in parts]; n = len(subset); summary[pop] = {}
            for arm in arms:
                ms = [r['metrics'][arm] for r in subset]
                flags = [(None, False, True)[int(r['id'][-3:])%3] for r in subset]
                summary[pop][arm] = dict(denominator=n, successes=sum(m['success'] for m in ms),
                    success_rate=sum(m['success'] for m in ms)/n, no_pose=sum(not m['pose_available'] for m in ms),
                    translation_cm=distribution([m['translation_cm'] for m in ms],n),
                    rotation_deg=distribution([m['rotation_deg'] for m in ms],n),
                    T_gt50cm=sum(m['translation_cm'] is not None and m['translation_cm']>50 for m in ms),
                    T_gt100cm=sum(m['translation_cm'] is not None and m['translation_cm']>100 for m in ms),
                    R_gt45deg=sum(m['rotation_deg'] is not None and m['rotation_deg']>45 for m in ms),
                    status_counts={'fallback': dict(true_n=sum(x is True for x in flags),
                        false_n=sum(x is False for x in flags), unknown_n=sum(x is None for x in flags)), 'raw_status': SECRET},
                    failures=[r['id'] for r in subset if not r['metrics'][arm]['success']], private_pose=SECRET)
        ev, ind, pred = (root/str(source_index)/k for k in ('evaluation','independent','prediction'))
        save(pred/'PREDICTIONS.jsonl.gz', [dict(id=r['id'], actual_pose=SECRET) for r in actual])
        E.M.write(pred/'PREDICTION_SEAL.json', dict(complete=True))
        for path in (ev/'SCORED_ROWS.jsonl.gz', ind/'INDEPENDENT_ROWS.jsonl.gz'): save(path, actual)
        for path in (ev/'SUMMARY.json', ind/'INDEPENDENT_SUMMARY.json'): E.M.write(path, summary)
        for path in (ev/'PAIRED.json', ind/'INDEPENDENT_PAIRED.json'): E.M.write(path, {})
        binding = dict(path=str(pred/'PREDICTIONS.jsonl.gz'), sha256=E.M.sha(pred/'PREDICTIONS.jsonl.gz'),
            seal_sha256=E.M.sha(pred/'PREDICTION_SEAL.json'))
        E.M.write(ev/'EVALUATION_SEAL.json', dict(complete=True, arms=arms, populations=E.EXPECTED,
            reference_sha256='CONSTRUCTED_HASH_ONLY_NO_REFERENCE_FILE', prediction_bindings=[binding],
            artifacts={name:E.M.sha(ev/name) for name in ('SCORED_ROWS.jsonl.gz','SUMMARY.json','PAIRED.json')}))
        E.M.write(ind/'INDEPENDENT_VERIFICATION.json', dict(PASS=True, errors=[], rows=238, arms=arms,
            populations=E.EXPECTED, reference_sha256='CONSTRUCTED_HASH_ONLY_NO_REFERENCE_FILE',
            prediction_bindings=[binding], success_classifications_exact=True, source_sha256='CONSTRUCTED_CHECKER_HASH',
            canonical_evaluation_seal_sha256=E.M.sha(ev/'EVALUATION_SEAL.json')))
        sources.append(dict(evaluation_dir=str(ev), independent_dir=str(ind),
            independent_artifacts={name:E.M.sha(ind/name) for name in E.C.INDEPENDENT}))
    manifest = root/'MANIFEST.json'; E.M.write(manifest, dict(sources=sources))
    return manifest


def rejected(call, label):
    try: call()
    except (ValueError, FileExistsError): return
    raise AssertionError('Accepted invalid metadata: '+label)


def tests():
    with TemporaryDirectory(prefix='export-results-constructed-') as directory:
        root=Path(directory);manifest=fixture(root);output=root/'public';seal=E.export(manifest,output)
        assert seal['complete'] and seal['summary_rows']==12 and seal['scalar_row_rows']==714
        with (output/'ROW_METRICS.csv').open(newline='') as stream:
            reader=csv.DictReader(stream);assert reader.fieldnames==list(E.ROW_FIELDS);rows=list(reader)
        assert len(rows)==238*3 and {r['arm'] for r in rows}==set(seal['arms'])
        assert all(r['translation_cm']=='10000.0' for r in rows if r['id']=='CONSTRUCTED_005')
        assert all(r['translation_cm']=='' and r['rotation_deg']=='' and r['success']=='False' and
            r['pose_available']=='False' for r in rows if r['id']=='CONSTRUCTED_006')
        with (output/'ALL_POPULATION_METRICS.csv').open(newline='') as stream:
            reader=csv.DictReader(stream);assert reader.fieldnames==list(E.ALL_FIELDS);summary=list(reader)
        assert len(summary)==12 and all(int(r['denominator'])==E.EXPECTED[r['population']] for r in summary)
        assert all(int(r['T_n'])+int(r['T_missing'])==int(r['denominator']) for r in summary)
        assert all(int(r['fallback_known_n'])+int(r['fallback_unknown_n'])==int(r['denominator']) for r in summary)
        assert float(next(r for r in summary if r['population']=='PRIMARY192')['T_sample_variance_cm2'])>0
        for path in output.iterdir():
            text=path.read_text();assert SECRET not in text and str(root) not in text
        rejected(lambda:E.export(manifest,output),'existing output')
        # A canonical artifact's bytes must still match its original seal.
        for name in ('SUMMARY.json','PAIRED.json'):
            case=root/('tampered_'+name);mp=fixture(case);ev=Path(E.M.read(mp)['sources'][0]['evaluation_dir'])
            rewrite_constructed(ev/name,dict(changed=True));rejected(lambda:E.authenticate_tables(mp),'canonical '+name)
        # A re-pinned independent bundle from another reference is not ancestry.
        case=root/'ancestry';mp=fixture(case);plan=E.M.read(mp);src=plan['sources'][0];ind=Path(src['independent_dir'])
        ck=E.M.read(ind/'INDEPENDENT_VERIFICATION.json');ck['reference_sha256']='OTHER_REFERENCE';rewrite_constructed(ind/'INDEPENDENT_VERIFICATION.json',ck)
        src['independent_artifacts']['INDEPENDENT_VERIFICATION.json']=E.M.sha(ind/'INDEPENDENT_VERIFICATION.json');rewrite_constructed(mp,plan)
        rejected(lambda:E.authenticate_tables(mp),'independent ancestry')
        # All individual bundles can be sealed but repeated baseline summaries
        # still must agree exactly, rather than choose the last source.
        case=root/'baseline';mp=fixture(case);plan=E.M.read(mp);src=plan['sources'][1];ev=Path(src['evaluation_dir']);ind=Path(src['independent_dir'])
        sm=E.M.read(ev/'SUMMARY.json');sm['PRIMARY192']['BASE_s20261018']['translation_cm']['mean']+=1e-9
        rewrite_constructed(ev/'SUMMARY.json',sm);es=E.M.read(ev/'EVALUATION_SEAL.json');es['artifacts']['SUMMARY.json']=E.M.sha(ev/'SUMMARY.json');rewrite_constructed(ev/'EVALUATION_SEAL.json',es)
        ck=E.M.read(ind/'INDEPENDENT_VERIFICATION.json');ck['canonical_evaluation_seal_sha256']=E.M.sha(ev/'EVALUATION_SEAL.json');rewrite_constructed(ind/'INDEPENDENT_VERIFICATION.json',ck)
        src['independent_artifacts']['INDEPENDENT_VERIFICATION.json']=E.M.sha(ind/'INDEPENDENT_VERIFICATION.json');rewrite_constructed(mp,plan)
        rejected(lambda:E.authenticate_tables(mp),'tiny repeated baseline summary change')
        case=root/'unique_summary';up=fixture(case);plan_unique=E.M.read(up);src_unique=plan_unique['sources'][0]
        ind_unique=Path(src_unique['independent_dir']);sm_unique=E.M.read(ind_unique/'INDEPENDENT_SUMMARY.json')
        sm_unique['PRIMARY192']['NEW_s20261018']['translation_cm']['mean']+=1
        rewrite_constructed(ind_unique/'INDEPENDENT_SUMMARY.json',sm_unique)
        src_unique['independent_artifacts']['INDEPENDENT_SUMMARY.json']=E.M.sha(ind_unique/'INDEPENDENT_SUMMARY.json')
        rewrite_constructed(up,plan_unique);rejected(lambda:E.authenticate_tables(up),'saved unique-arm summary disagreement')
        sample=E.M.read(Path(plan['sources'][0]['evaluation_dir'])/'SUMMARY.json')['PRIMARY192']['BASE_s20261018']
        bad=deepcopy(sample);bad['translation_cm']['missing']+=1
        rejected(lambda:E.public_summary('PRIMARY192','BASE',bad),'missing accounting')
        bad=deepcopy(sample);bad['denominator']=191
        rejected(lambda:E.public_summary('PRIMARY192','BASE',bad),'population denominator')
    return dict(PASS=True, constructed_metadata_only=True, actual_dataset_rows_read=0,
        raw_references_decoded=0, solve_fit_forward_metric_calls=0, all_four_populations_and_unique_seed_arms=True,
        privacy_whitelist_outliers_nulls_unknown_fallback_and_new_folder_checked=True,
        canonical_summary_paired_tampering_wrong_ancestry_duplicate_baseline_and_denominator_rejected=True,
        unique_arm_canonical_independent_saved_summary_disagreement_rejected=True,
        source_sha256=E.M.sha(E.__file__))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=tests();E.M.write(a.output,result);print(json.dumps(result,sort_keys=True))
