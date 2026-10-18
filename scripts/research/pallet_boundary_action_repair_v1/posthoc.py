"""Fixed historical strata and seed/factor descriptions after output sealing.

These reference-derived strata are never producer features or model selection
inputs. No new fitting, threshold search or optimizer is called here.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import statistics
import time
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read, rows, sha, write
from scripts.research.pallet_statewise_end_to_end_recovery_v1.mechanism_diagnosis import original_group, GROUPS, PRIMARY_COUNTS
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import aggregate
from .evaluate import csv_write
from .registry import ALL_ARMS, NEW_ARMS, B_ARMS, L_GROUPS, SEEDS, POPULATIONS, COMPARATORS


def describe(values):
    return dict(seed_count=len(values),mean=statistics.mean(values),sample_SD=statistics.stdev(values)if len(values)>1 else None,
        min=min(values),max=max(values),values=values,independent_dataset_count=1)


def run(run_dir,evaluation,oracle_root,output):
    start=time.monotonic();run_dir=Path(run_dir);evaluation=Path(evaluation);oracle_root=Path(oracle_root);output=Path(output)
    seal=read(run_dir/'PREDICTION_SEAL.json');ev=read(evaluation/'EVALUATION_SEAL.json')
    if sha(run_dir/'PREDICTION_SEAL.json')!=ev['prediction_seal_sha256']or not seal['before_scoring']:
        raise ValueError('Posthoc analysis requires already sealed evaluated outputs')
    op=oracle_root/'ORACLE_ROWS.jsonl.gz';os=read(oracle_root/'ORACLE_SEAL.json')
    digest=os.get('artifacts',os.get('files',{})).get(op.name)
    if isinstance(digest,dict):digest=digest.get('sha256')
    if sha(op)!=digest:raise ValueError('Original historical stratum bytes changed')
    scored=list(rows(evaluation/'SCORED_ROWS.jsonl.gz'));original={r['id']:r for r in rows(op)}
    if set(original)!={r['id']for r in scored}:raise ValueError('Original238 stratum identities changed')
    groups={};membership=[]
    for r in scored:
        old=original[r['id']]
        if any(r[k]!=old[k]for k in ('partition','session'))or r['metrics']['E0']['success']!=old['final_joint_success']:
            raise ValueError('Historical E0 pose/stratum mismatch')
        groups[r['id']]=original_group(old);membership.append(dict(id=r['id'],partition=r['partition'],stratum=groups[r['id']]))
    if Counter(groups[r['id']]for r in scored if r['partition']!='CAL')!=Counter(PRIMARY_COUNTS):
        raise ValueError('Original114/10/22/46 classes changed')
    strata=[];staged=list(rows(evaluation/'STAGE_DIAGNOSIS.jsonl.gz'));stage={r['id']:r for r in staged}
    for pop,parts in POPULATIONS.items():
        subset=[r for r in scored if r['partition']in parts]
        for group in GROUPS:
            selected=[r for r in subset if groups[r['id']]==group]
            if not selected:continue
            summaries=aggregate(selected,ALL_ARMS)
            for arm,summary in summaries.items():
                record=dict(population=pop,stratum=group,arm=arm,denominator=len(selected),successes=summary['successes'],
                    translation_mean_cm=summary['translation_cm']['mean'],translation_sample_SD=summary['translation_cm']['sample_std'],
                    rotation_mean_deg=summary['rotation_deg']['mean'],rotation_sample_SD=summary['rotation_deg']['sample_std'],
                    T_gt50cm=summary['T_gt50cm'],T_gt100cm=summary['T_gt100cm'],R_gt45deg=summary['R_gt45deg'])
                for comparator in COMPARATORS:
                    record['recovery_vs_'+comparator]=sum(not r['metrics'][comparator]['success']and r['metrics'][arm]['success']for r in selected)
                    record['damage_vs_'+comparator]=sum(r['metrics'][comparator]['success']and not r['metrics'][arm]['success']for r in selected)
                if arm in stage[selected[0]['id']]['arms']:
                    records=[stage[r['id']]['arms'][arm]for r in selected]
                    record['generated_has_success_n']=sum(s['generated_before_constraints']['has_success']for s in records)
                    record['eligible_has_success_n']=sum(s['after_constraints']['has_success']for s in records)
                    record['selection_loss_n']=sum(not s['final_success']and s['after_constraints']['has_success']for s in records)
                strata.append(record)
    results=read(evaluation/'RESULTS.json');seed_summaries=[];factors=[];source_gain=[]
    for pop in POPULATIONS:
        p=results['populations'][pop]
        for group in L_GROUPS:
            for metric in ('successes','translation_cm/mean','rotation_deg/mean','T_gt50cm','T_gt100cm','R_gt45deg'):
                values=[]
                for seed in SEEDS:
                    value=p[group+'_s'+str(seed)]
                    for key in metric.split('/'):value=value[key]
                    values.append(value)
                seed_summaries.append(dict(population=pop,group=group,metric=metric,**describe(values)))
        for seed in SEEDS:
            suffix='_s'+str(seed);table={g:p[g+suffix]['successes']for g in L_GROUPS}
            factors.append(dict(population=pop,seed=seed,denominator=p['E0']['denominator'],
                H1_minus_H0_at_D0=table['D0_H1_MLP']-table['D0_H0_MLP'],
                H1_minus_H0_at_D1=table['D1_H1_MLP']-table['D1_H0_MLP'],
                D1_minus_D0_at_H0=table['D1_H0_MLP']-table['D0_H0_MLP'],
                D1_minus_D0_at_H1=table['D1_H1_MLP']-table['D0_H1_MLP'],
                descriptive_interaction_count=(table['D1_H1_MLP']-table['D1_H0_MLP'])-(table['D0_H1_MLP']-table['D0_H0_MLP']),
                pair_minus_full=table['D1_H1_PAIR_MLP']-table['D1_H1_MLP'],
                role_affine_minus_MLP=table['D1_H1_ROLE_AFFINE']-table['D1_H1_MLP']))
        selected=[r for r in scored if r['partition']in POPULATIONS[pop]]
        for arm in NEW_ARMS:
            for transition,predicate in [('RECOVERY',lambda a,b:not a and b),('DAMAGE',lambda a,b:a and not b),
                                         ('BOTH_FAILURE',lambda a,b:not a and not b),('BOTH_SUCCESS',lambda a,b:a and b)]:
                records=[r for r in selected if predicate(r['metrics']['E0']['success'],r['metrics'][arm]['success'])]
                counts=Counter(r['action_sources'][arm]['role']for r in records)
                for source,n in sorted(counts.items()):source_gain.append(dict(population=pop,arm=arm,vs='E0',transition=transition,source=source,n=n,transition_denominator=len(records)))
    output.mkdir(parents=True,exist_ok=False);csv_write(output/'HISTORICAL_STRATA.csv',strata)
    csv_write(output/'FACTOR_AND_PAIR_COUNTS.csv',factors);csv_write(output/'GAIN_DAMAGE_ACTUAL_SOURCE.csv',source_gain)
    write(output/'SEED_VARIABILITY.json',dict(rows=seed_summaries,note='Three fixed seeds on one dataset; not576 independent images; representative18 fixed in advance'))
    write(output/'POSTHOC_MECHANISM.json',dict(schema='POSTSEAL_FIXED_STRATA_AND_FACTORS_V1',complete=True,
        historical_primary_counts=PRIMARY_COUNTS,membership=membership,
        runtime_or_training_uses_strata=False,no_new_forward_PnP_fitting_or_hyperparameter_search=True,
        source_sha256=sha(__file__),evaluation_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'),
        original_oracle_sha256=sha(op),original_oracle_seal_sha256=sha(oracle_root/'ORACLE_SEAL.json'),
        elapsed_seconds=time.monotonic()-start,artifacts={p.name:sha(p)for p in output.iterdir()if p.is_file()}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--oracle-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.run_dir,a.evaluation,a.oracle_root,a.output)


if __name__=='__main__':main()
