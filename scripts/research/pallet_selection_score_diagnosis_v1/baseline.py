"""Read-only cohort grouping and cross-tabs of already sealed pose metrics.

No pose/reference scoring or model code is imported.  New group aliases map
explicitly to the old oracle's actual earliest_observed_failure_stage field.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

GROUP_MAP = {
    'family_selection': 'FAMILY_SELECTION_LOSS',
    'within_family_final_selection': 'WITHIN_FAMILY_SELECTION_LOSS',
    'confirmed_original_consumer_observation_or_proposal_limit': 'NO_SUCCESS_CANDIDATE',
}
ARMS = ('N0', 'POSEFIX_RAW', 'E0')
REPRESENTATIVE_RULE = {
    'schema': 'SELECTION_SCORE_DIAG_REPRESENTATIVE_RULE_V1',
    'fixed_before_new_cost_results': True,
    'maximum_main_text_examples': 6,
    'quota': {'FAMILY_SELECTION_LOSS': 2, 'WITHIN_FAMILY_SELECTION_LOSS': 2,
              'FINAL_SUCCESS_CONTROL': 2},
    'ranking': 'descending original sealed E0 final translation_cm, then id',
    'recording_rule': 'Within each group take highest ranked unseen recording/session before repeating a session; if only one recording/session is available preserve the next-ranked case and flag unavoidable repetition.',
    'success_control_eligibility': 'A retained-feasible reference-failing comparison candidate exists and the same active additive family seen in selection-loss pairs protects the original reference-successful selected candidate; availability is decided from full pair ledger, never an invented pose.',
    'no_custom_T_R_mixture': True,
    'all_selection32_are_shown_in_gallery': True,
    'prior_no_success_examples_not_selection32_representatives': True,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Do not hide an empty required table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        yield from map(json.loads, stream)


def truth(value):
    if value == 'True' or value is True:
        return True
    if value == 'False' or value is False:
        return False
    if value in ('', None):
        return None
    raise ValueError('Unrecognized boolean '+repr(value))


def group(row):
    if row['final_joint_success'] is True:
        if row['earliest_observed_failure_stage'] is not None:
            raise ValueError('Success with a failure-stage label')
        return 'FINAL_SUCCESS_CONTROL'
    return GROUP_MAP[row['earliest_observed_failure_stage']]


def read_metrics(path):
    result = defaultdict(dict)
    with Path(path).open(newline='') as stream:
        for r in csv.DictReader(stream):
            if r['arm'] not in ARMS:
                continue
            if r['arm'] in result[r['id']]:
                raise ValueError('Duplicate metric row')
            result[r['id']][r['arm']] = {
                'success': truth(r['success_le5']),
                'available': truth(r['pose_available']),
                'T_cm': float(r['translation_cm']) if r['translation_cm'] else None,
                'R_deg': float(r['rotation_deg']) if r['rotation_deg'] else None,
                'partition': r['partition'], 'session': r['session'],
                'cluster_id': r['cluster_id'],
            }
    return dict(result)


def cohorts(row):
    return ['CAL46'] if row['partition'] == 'CAL' else [
        'PRIMARY192', 'TEST49' if row['partition'] == 'TEST' else 'REMAINDER143']


def stats(values):
    finite = [x for x in values if x is not None]
    return {'N': len(finite), 'mean': statistics.mean(finite) if finite else None,
            'sample_SD': statistics.stdev(finite) if len(finite)>1 else None,
            'sample_variance': statistics.variance(finite) if len(finite)>1 else None,
            'max': max(finite) if finite else None}


def aggregate(oracles, metrics):
    groups = defaultdict(list)
    case_rows = []
    for r in oracles:
        ident = r['id']; g = group(r); m = metrics[ident]
        if set(m) != set(ARMS):
            raise ValueError('Three baseline rows required '+ident)
        if m['E0']['success'] is not r['final_joint_success']:
            raise ValueError('Oracle/metric same-candidate success mismatch '+ident)
        for a in ARMS:
            v=m[a]
            if v['partition'] != r['partition'] or v['session'] != r['session']:
                raise ValueError('Baseline identity mismatch '+ident)
            expected = None if v['T_cm'] is None or v['R_deg'] is None else v['T_cm']<=5 and v['R_deg']<=5
            if v['success'] is not expected:
                raise ValueError('Stored same-candidate joint threshold mismatch '+ident)
        c={'id': ident,'partition': r['partition'],'session':r['session'],
           'cluster_id':m['E0']['cluster_id'],'diagnostic_group':g,
           'original_earliest_observed_failure_stage': r['earliest_observed_failure_stage'],
           'consumer_generated_bank_complete':r['original_consumer_generated_bank_complete']}
        for a in ARMS:
            c.update({a+'_success':m[a]['success'],a+'_translation_cm':m[a]['T_cm'],a+'_rotation_deg':m[a]['R_deg']})
        case_rows.append(c)
        for co in cohorts(r):
            groups[(co,g)].append(ident)
            groups[(co,'ALL')].append(ident)
    cross=[]; patterns=[]
    for (co,g), ids in sorted(groups.items()):
        r={'cohort':co,'diagnostic_group':g,'images':len(ids),
           'sessions':len({metrics[i]['E0']['session'] for i in ids}),
           'recordings':len({metrics[i]['E0']['cluster_id'] for i in ids})}
        for a in ARMS:
            r[a+'_success']=sum(metrics[i][a]['success'] is True for i in ids)
            r[a+'_failure']=sum(metrics[i][a]['success'] is False for i in ids)
            r[a+'_unresolved']=sum(metrics[i][a]['success'] is None for i in ids)
            for axis in ('T_cm','R_deg'):
                for k,v in stats([metrics[i][a][axis] for i in ids]).items():
                    r[a+'_'+axis+'_'+k]=v
        r['N0_or_PoseFix_success']=sum(any(metrics[i][a]['success'] is True for a in ARMS[:2]) for i in ids)
        r['N0_and_PoseFix_fail']=sum(all(metrics[i][a]['success'] is False for a in ARMS[:2]) for i in ids)
        r['all_three_fail']=sum(all(metrics[i][a]['success'] is False for a in ARMS) for i in ids)
        cross.append(r)
        cnt=Counter(tuple(metrics[i][a]['success'] for a in ARMS) for i in ids)
        for pattern,n in sorted(cnt.items(),key=lambda x:str(x[0])):
            patterns.append(dict(cohort=co,diagnostic_group=g,N0_success=pattern[0],
                                 POSEFIX_RAW_success=pattern[1],E0_success=pattern[2],images=n))
    return case_rows,cross,patterns


def build(old_private, output):
    start=time.monotonic(); cpu=time.process_time()
    old_private,output=Path(old_private),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    op=old_private/'oracle_v1/ORACLE_ROWS.jsonl.gz'
    mp=old_private/'score_v2/ROW_METRICS.csv'
    oracles=list(read_rows(op)); metrics=read_metrics(mp)
    ids=[r['id'] for r in oracles]
    if len(ids)!=238 or len(set(ids))!=238 or set(ids)!=set(metrics):
        raise ValueError('Exact 238 identity union required')
    cases,cross,patterns=aggregate(oracles,metrics)
    primary=Counter(r['diagnostic_group'] for r in cases if r['partition']!='CAL')
    expected={'FAMILY_SELECTION_LOSS':10,'WITHIN_FAMILY_SELECTION_LOSS':22,
              'NO_SUCCESS_CANDIDATE':46,'FINAL_SUCCESS_CONTROL':114}
    if dict(primary)!=expected:
        raise ValueError('Original cohort disagreement '+str(primary))
    selection=[r for r in cases if r['partition']!='CAL' and r['diagnostic_group'] in list(expected)[:2]]
    if any(r['partition']!='REMAINDER' for r in selection):
        raise ValueError('Original selection32 split disagreement')
    write_json(output/'REPRESENTATIVE_RULE.json',REPRESENTATIVE_RULE)
    write_csv(output/'GROUP_MEMBERSHIP_238.csv',cases)
    write_csv(output/'BASELINE_CROSS_TABLE.csv',cross)
    write_csv(output/'BASELINE_PATTERN_CROSS_TABLE.csv',patterns)
    receipt={'schema':'SELECTION_BASELINE_READONLY_V1','oracle_input':str(op),
             'oracle_sha256':sha(op),'metrics_input':str(mp),'metrics_sha256':sha(mp),
             'new_group_alias_source_field':'earliest_observed_failure_stage',
             'new_group_alias_mapping':GROUP_MAP,'success_alias':'final_joint_success=true -> FINAL_SUCCESS_CONTROL',
             'primary_groups':dict(primary),'CAL_groups':dict(Counter(r['diagnostic_group'] for r in cases if r['partition']=='CAL')),
             'selection32_all_REMAINDER':True,'all238_consumer_complete':all(r['original_consumer_generated_bank_complete'] for r in oracles),
             'new_metric_scoring_calls':0,'NN_forward_calls':0,'pose_solver_calls':0,
             'candidate_generation_calls':0,'training_updates':0,
             'elapsed_seconds':time.monotonic()-start,'process_CPU_seconds':time.process_time()-cpu,
             'artifacts':{p.name:sha(p) for p in output.iterdir()}}
    write_json(output/'BASELINE_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-private',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.old_private,a.output),ensure_ascii=False))
