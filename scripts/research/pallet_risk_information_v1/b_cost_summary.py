"""Read-only unique-frame interpretation of archived B fixed-pose costs.

This is synthetic diagnostic accounting, never a selector validation. No new
cost/pose evaluation occurs. Different objectives are reported separately.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import resource
from .common import rows,read,write,sha


def summarize(records,tolerance):
    groups={}
    for split in ('train','validation','ALL1003'):
        subset=[r for r in records if split=='ALL1003' or r['split']==split];families={}
        for family in ('ORIGINAL_WD','SWAPPED_WD'):
            modes={}
            for mode in ('fixed','A','B','AB','edge'):
                count=Counter();ids=[]
                for row in subset:
                    count['frames']+=1
                    ff=[next(f for f in p['families'] if f['family']==family) for p in row['registered_phases']]
                    cc=[next(c for c in f['same_objective_comparisons'] if c['objective']=='G13_FIXED_POSE_'+mode) for f in ff]
                    good=[c for c in cc if c['reference_original_constraints_feasible'] and c['reference_cost'] is not None]
                    count['reference_original_feasible_any_registered_phase']+=bool(good)
                    if not row['selected_E0_available']:
                        count['selected_E0_unavailable']+=1;continue
                    failed=not row['selected_E0_metric_diagnostic_only']['success_5cm5deg']
                    count['selected_E0_failed']+=failed
                    stored_costs={c['selected_E0_cost'] for c in cc}
                    stored_slacks={tuple(c['selected_h_min_slack'] or []) for c in cc}
                    if len(stored_costs)!=1 or len(stored_slacks)!=1:raise ValueError('Same E0 package changed across reference phases')
                    value=cc[0]['selected_E0_cost'];slacks=cc[0]['selected_h_min_slack']
                    eligible=bool(slacks and max(slacks)>=-tolerance)
                    count['selected_E0_original_feasible']+=eligible
                    count['selected_failed_and_original_feasible']+=failed and eligible
                    count['reference_feasible_and_failed_selected_feasible']+=bool(good) and failed and eligible
                    valid=bool(good) and value is not None and failed and eligible
                    if valid:
                        best=min(c['reference_cost'] for c in good);gap=best-value
                        count['valid_same_package_failed_E0_comparisons']+=1
                        count['failed_E0_lower_than_best_feasible_registered_reference']+=gap>0
                        if gap>0:ids.append(row['id'])
                modes[mode]=dict(counts=dict(count),lower_cost_frame_ids=ids,
                    interpretation='same G13 family/mode only; original E0 need not have been selected by this diagnostic objective; no new policy validation')
            families[family]=modes
        groups[split]=dict(denominator=len(subset),families=families)
    return dict(schema='B_SAME_OBJECTIVE_UNIQUE_FRAME_SUMMARY_V1',groups=groups,
        unique_frame_counts_not_phase_opportunities=True,selected_E0_and_reference_original_feasibility_both_required=True,
        all_registered_reference_phases_considered=True,GT_never_added_to_bank=True,
        finite_support_is_separate_not_added_to_original_objective_feasibility=True,
        new_cost_evaluations=0,new_pose_solves=0,new_candidates=0,new_model_forwards=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','diagnostic-seal','tolerance-receipt','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();s=read(a.diagnostic_seal);t=read(a.tolerance_receipt)
    if sha(a.rows)!=s['rows_sha256'] or s.get('complete')is not True:raise ValueError('Complete sealed diagnostic required')
    if t.get('rows')!=1003 or t.get('missing_explicit_values')!=0 or len(t['unique_values_and_counts'])!=1:raise ValueError('Original tolerance ambiguity')
    value,n=next(iter(t['unique_values_and_counts'].items()))
    if n!=1003:raise ValueError('Original tolerance coverage mismatch')
    start,cpu=time.monotonic(),time.process_time();data=list(rows(a.rows))
    if len(data)!=1003 or len({r['id']for r in data})!=1003:raise ValueError('Exact1003 IDs required')
    result=summarize(data,float(value));result.update(source_sha256=sha(__file__),
        diagnostic_seal_sha256=sha(a.diagnostic_seal),diagnostic_rows_sha256=sha(a.rows),
        tolerance_receipt_sha256=sha(a.tolerance_receipt),original_tolerance=float(value),
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(a.output,result)


if __name__=='__main__':main()
