"""Independent streaming count: stored eligible_success flags and exp sums.

Does not import the diagnosis implementation, a model or a pose metric.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import math
from pathlib import Path
import resource
import statistics
import time
from .common import read,rows,sha,write


def run(run_dir,*,synthetic_GO=False):
    if synthetic_GO is not True:raise PermissionError('Synthetic-only diagnostic GO required before IO')
    start,cpu=time.monotonic(),time.process_time();run=Path(run_dir);out=run/'positive_mass_v1'
    report_path=out/'POSITIVE_MASS_DIAGNOSIS.json';seal_path=out/'POSITIVE_MASS_SEAL.json'
    seal=read(seal_path)
    if sha(report_path)!=seal['artifacts'][report_path.name]:raise ValueError('Diagnosis changed')
    result=read(report_path);dp=run/'head_synthetic_distributions_v1/DISTRIBUTIONS.jsonl.gz';lp=run/'head_labels_v1/SYNTHETIC_HEAD_TARGETS.jsonl.gz'
    if sha(dp)!=result['input_bindings']['distributions'] or sha(lp)!=result['input_bindings']['labels']:raise ValueError('Source ledgers changed')
    # Unlike the diagnosis, use the already stored per-candidate eligible_success
    # flag, not positive_candidate_indices, and direct exp sums, not logsumexp.
    labels={r['id']:{m['candidate_id'] for m in r['candidate_metrics'] if m['eligible_success']}
            for r in rows(lp) if r['split']=='validation'}
    with (out/'POSITIVE_MASS_ROWS.csv').open()as f:csvrows={(r['id'],r['arm']):r for r in csv.DictReader(f)}
    counts=defaultdict(Counter);probs=defaultdict(lambda:defaultdict(list));maxgap=0.;frames=0
    for frame in rows(dp):
        if frame['split']!='validation':continue
        frames+=1;good=labels[frame['id']]
        for arm,d in frame['predictions'].items():
            cs=counts[arm];cs['frames']+=1;cs['good_available']+=bool(good)
            p=[math.exp(x) for x in d['log_probabilities']];none=d['selected_index']==d['none_index'];cs['abstentions']+=none
            positive=[p[i] for i,cid in enumerate(d['candidate_ids']) if cid in good]
            total=math.fsum(positive);pnone=p[d['none_index']];record=csvrows[(frame['id'],arm)]
            maxgap=max(maxgap,abs(total-float(record['P_sum_good'])),abs(pnone-float(record['P_none'])))
            if good and none:
                cs['good_abstained']+=1;cs['good_sum_gt_none']+=total>pnone;cs['good_sum_le_none']+=total<=pnone
                cs['k='+str(len(good))]+=1
                for key,value in [('P_none',pnone),('P_max_good',max(positive)),('P_sum_good',total)]:probs[arm][key].append(value)
            cs['no_good_abstained']+=not good and none
    if frames!=235 or len(labels)!=235 or len(csvrows)!=235*6:raise ValueError('Full validation denominator lost')
    for arm,actual in counts.items():
        a=result['arms'][arm];c=a['categories'];expected=dict(frames=235,good_available=c['good_available']['n'],
            abstentions=a['abstentions'],good_abstained=c['good_available_abstained']['n'],
            good_sum_gt_none=c['good_available_abstained_sum_gt_none']['n'],good_sum_le_none=c['good_available_abstained_sum_le_none']['n'],
            no_good_abstained=c['no_good_abstained']['n'])
        if any(actual[k]!=v for k,v in expected.items()):raise ValueError('Independent direct count disagrees: '+arm)
        if any(actual['k='+k]!=v for k,v in c['good_available_abstained']['positive_count_distribution'].items()):raise ValueError('Positive-count distribution disagrees')
        for key,values in probs[arm].items():
            if abs(statistics.fmean(values)-c['good_available_abstained']['summaries'][key]['mean'])>1e-12:raise ValueError('Probability mean disagrees')
    if maxgap>1e-12:raise ValueError('Direct probability sums disagree with row ledger')
    receipt=dict(schema='INDEPENDENT_SYNTHETIC_POSITIVE_MASS_COUNT_CHECK_V1',status='PASS',validation_frames=frames,
                 arms={a:dict(c) for a,c in counts.items()},maximum_direct_probability_row_difference=maxgap,
                 independent_algorithm='stream stored eligible_success flags; selected_index==none_index; direct fsum(exp(logp))',
                 input_bindings=dict(diagnosis=sha(report_path),diagnosis_seal=sha(seal_path),distributions=sha(dp),labels=sha(lp)),
                 source_sha256=sha(__file__),new_model_or_metric_calls=0,PRIMARY_results_read=False,
                 cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write(out/'INDEPENDENT_MASS_COUNT_CHECK.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--synthetic-GO',action='store_true');a=p.parse_args()
    r=run(a.run_dir,synthetic_GO=a.synthetic_GO);print(r['status'],r['maximum_direct_probability_row_difference'])


if __name__=='__main__':main()
