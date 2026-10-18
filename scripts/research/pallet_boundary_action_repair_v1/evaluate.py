"""Postseal, reference-only evaluation. Never imported by prediction producers."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
from pathlib import Path
import resource
import time
import numpy as np
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import canonical, read, rows, save_rows, sha, write
from scripts.research.pallet_statewise_end_to_end_recovery_v1.evaluator import verify_new_seal, _safe_metric, _status, _candidate_pose
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import aggregate, paired
from .registry import (ALL_ARMS, NEW_ARMS, B_ARMS, L_ARMS, POPULATIONS, EXPECTED_N,
                       contrasts, evaluation_protocol)
from .action_data import actions, same_pose as same_action_pose


def csv_write(path, records):
    records=list(records)
    if not records: raise ValueError('Nonempty CSV required')
    path=Path(path); opener=gzip.open if path.suffix=='.gz' else open
    keys=list(dict.fromkeys(k for r in records for k in r))
    with opener(path,'xt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)


def same_pose(a,b,scale=1.):
    if not a or not b or not a.get('available') or not b.get('available'): return False
    return bool(np.max(np.abs(np.asarray(a['R_physical'])-b['R_physical']))<=1e-7 and
                np.max(np.abs(np.asarray(a['centroid'])-b['centroid']))<=1e-7*scale)


def source(pose, poses,scale=1.):
    if not pose.get('available'):return dict(role='NO_POSE',aliases=[])
    aliases=[x for x in ('N0','E0_REGENERATED','E0') if same_pose(pose,poses[x],scale)]
    role='N0' if 'N0' in aliases else 'E0_REGENERATED' if 'E0_REGENERATED' in aliases else 'HISTORICAL_E0' if 'E0' in aliases else 'OTHER'
    return dict(role=role,aliases=aliases)


def continuous(rows_,before,after):
    subset=[r for r in rows_ if not r['metrics'][before]['success'] and not r['metrics'][after]['success']]
    out={}
    for key in ('translation_cm','rotation_deg'):
        values=[dict(id=r['id'],before=r['metrics'][before][key],after=r['metrics'][after][key],
            delta=r['metrics'][after][key]-r['metrics'][before][key]) for r in subset
            if r['metrics'][after][key] is not None and r['metrics'][before][key] is not None]
        out[key]=dict(both_failure_full_n=len(subset),common_available_n=len(values),
            improved_n=sum(v['delta']<0 for v in values),worsened_n=sum(v['delta']>0 for v in values),
            tied_n=sum(v['delta']==0 for v in values),improvement_sum=sum(v['delta'] for v in values if v['delta']<0),
            worsening_sum=sum(v['delta'] for v in values if v['delta']>0),
            largest_worsening=sorted(values,key=lambda v:(-v['delta'],v['id']))[:10],
            largest_improvement=sorted(values,key=lambda v:(v['delta'],v['id']))[:10],all_deltas=values)
    return out


def bank_metrics(bank,reference):
    values=[_safe_metric(_candidate_pose(c),reference) for c in bank]
    return dict(candidate_count=len(bank),available_n=sum(m['pose_available'] for m in values),
        has_success=any(m['success'] for m in values),
        success_ids=[c.get('candidate_id','UNRECORDED') for c,m in zip(bank,values) if m['success']])


def candidate_diagnosis(run_dir, ledger, references):
    features={r['id']:r for r in rows(run_dir/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz')}
    out=[]
    for i,row in enumerate(ledger):
        ref=references[row['partition']][row['id']]; feature=features[row['id']]
        original=feature['candidates']; eligible=[c for c in original if c['eligible']]
        n0=dict(candidate_id='N0',actual_pose=feature['fallback'])
        e0=dict(candidate_id='E0_REGENERATED',actual_pose=feature['e0_regenerated'])
        scale=feature.get('object_scale_m',1.)
        generated=[]
        for candidate in original+[n0,e0]:
            if not any(same_action_pose(_candidate_pose(candidate),_candidate_pose(c),scale)for c in generated):generated.append(candidate)
        actual_actions=actions(feature);pair_actions=actions(feature,pair=True)
        full=bank_metrics(generated,ref);admissible=bank_metrics(actual_actions,ref)
        pair=bank_metrics(pair_actions,ref)
        result=dict(id=row['id'],partition=row['partition'],common_bank=dict(
            raw_source_count=feature.get('candidate_sources_before_dedup'),unique_count=len(original),
            eligible_count=len(eligible),H0_logit_count=len(eligible)+1,
            unique_actual_H1_action_count=len(actual_actions),unique_actual_pair_action_count=len(pair_actions),
            generated=full,eligible_actions=admissible,pair_actions=pair,
            state_counts=feature.get('state_counts'),observation_coverage=feature.get('observation_coverage'),
            projection_rejection_count=len(feature.get('projection_rejections',[]))),arms={})
        for arm in L_ARMS:
            final=_safe_metric(row['predictions'][arm]['actual_pose'],ref)
            bank=pair if '_PAIR_' in arm else admissible
            result['arms'][arm]=dict(final_success=final['success'],generated_before_constraints=pair if '_PAIR_' in arm else full,
                after_constraints=bank,selected_candidate_id=row['predictions'][arm]['selected_candidate_id'],
                stage_loss_class='FINAL_SUCCESS' if final['success'] else 'NO_SUCCESS_IN_ELIGIBLE_ACTIONS' if not bank['has_success'] else 'ACTION_SELECTION_LOSS')
        folder=run_dir/'b_finite_v1/frames'/f'{i:04d}'
        frame_seal=read(folder/'FRAME_SEAL.json')
        if frame_seal['id']!=row['id'] or frame_seal['result_sha256']!=sha(folder/'RESULT.jsonl.gz'):
            raise ValueError('B result binding changed')
        b=next(rows(folder/'RESULT.jsonl.gz'));carried=b['banks']['B_FINITE_BANK']
        for arm in B_ARMS:
            bank=b['banks'][arm];scores=b['finite_scores'][arm]
            if len(bank)!=len(scores):raise ValueError('Finite B candidate/score mismatch')
            admitted=[c for c,s in zip(bank,scores) if any(v['feasible'] for v in s.values())]
            attempts=[c for c in b['attempts'] if c['arm']==arm]
            generated=[n0]+attempts if arm=='B_FINITE_LOCAL' else carried+attempts
            pre=bank_metrics(generated,ref);post=bank_metrics(admitted,ref)
            final=_safe_metric(row['predictions'][arm]['actual_pose'],ref)
            result['arms'][arm]=dict(final_success=final['success'],generated_before_constraints=pre,
                after_constraints=post,retained_before_constraints=bank_metrics(bank,ref),
                new_optimizer_attempt_count=len(attempts),
                optimizer_success_n=sum(c.get('solve',{}).get('optimizer_success') is True for c in attempts),
                numeric_pass_n=sum(c.get('solve',{}).get('numeric_pass') is True for c in attempts),
                finite_support_n=sum(c.get('solve',{}).get('finite_support') is True for c in attempts),
                feasible_n=sum(c.get('solve',{}).get('feasible') is True for c in attempts),
                technical_exception_n=sum(bool(c.get('solve',{}).get('exception')) for c in attempts),
                fallback=row['predictions'][arm]['fallback'],
                selected_candidate_id=row['predictions'][arm]['selected_candidate_id'],
                selected_source=row['predictions'][arm].get('selected_candidate_source'),
                selected_finite_support=row['predictions'][arm].get('finite_support'),
                selected_optimizer=row['predictions'][arm].get('selected_optimizer'),
                stage_loss_class='FINAL_SUCCESS' if final['success'] else 'NO_SUCCESS_GENERATED' if not pre['has_success']
                    else 'SUCCESS_REJECTED_OR_NOT_RETAINED_BY_FINITE_CONSTRAINTS' if not post['has_success'] else 'SELECTION_OR_FALLBACK_LOSS')
        out.append(result)
    return out


def run(run_dir, original_score, output, *, scoring_GO=False):
    if not scoring_GO:raise PermissionError('Root postseal scoring GO required')
    started,cpu=time.monotonic(),time.process_time();run_dir=Path(run_dir);output=Path(output)
    ledger,seal,prediction_path=verify_new_seal(run_dir)
    if set(ledger[0]['predictions'])!=set(ALL_ARMS):raise ValueError('Predeclared arm inventory changed')
    if canonical(read(run_dir/'EVALUATION_PROTOCOL.json'))!=canonical(evaluation_protocol()):raise ValueError('Frozen evaluation protocol changed')
    if sha(run_dir/'EVALUATION_PROTOCOL.json')!=seal['evaluation_protocol_sha256']:raise ValueError('Evaluation lock changed')
    gate=read(run_dir/'ALL_OUTPUTS_SCORING_GO.json')
    if not gate['all_registered_predictions_sealed'] or gate['prediction_seal_sha256']!=sha(run_dir/'PREDICTION_SEAL.json'):
        raise ValueError('Final all-output gate missing')
    original_score=Path(original_score);refpath=original_score/'FIXED_REFERENCES.json'
    if sha(refpath)!=read(original_score/'SCORE_SEAL.json')['artifacts']['FIXED_REFERENCES.json']:
        raise ValueError('Original reference bytes changed')
    # The first new real-reference decode is deliberately below all guards.
    refs=read(refpath)['references'];output.mkdir(parents=True,exist_ok=False)
    write(output/'REFERENCE_OPEN_RECEIPT.json',dict(epoch=time.time(),reference_sha256=sha(refpath),
        prediction_seal_sha256=sha(run_dir/'PREDICTION_SEAL.json'),plan_sha256=sha(run_dir/'PLAN_LOCK.json'),
        scoring_GO_sha256=sha(run_dir/'ALL_OUTPUTS_SCORING_GO.json')))
    scored=[]
    for row in ledger:
        identity={k:row[k] for k in ('id','partition','session','cluster_id')};ref=refs[row['partition']][row['id']]
        poses={a:row['predictions'][a]['actual_pose'] for a in ALL_ARMS}
        value=dict(**identity,poses=poses,metrics={a:_safe_metric(poses[a],ref) for a in ALL_ARMS},
            status={a:_status(row['predictions'][a]) for a in ALL_ARMS},
            object_scale_m=row['object_scale_m'],action_sources={a:source(poses[a],poses,row['object_scale_m']) for a in ALL_ARMS})
        scored.append(value)
    subsets={k:[r for r in scored if r['partition'] in v] for k,v in POPULATIONS.items()}
    if {k:len(v) for k,v in subsets.items()}!=EXPECTED_N:raise ValueError('Population denominators changed')
    summaries={k:aggregate(v,ALL_ARMS) for k,v in subsets.items()}
    for subset in summaries.values():
        for record in subset.values():
            for key in ('translation_cm','rotation_deg'):
                sd=record[key].get('SD_ddof1',record[key].get('std',record[key].get('SD')))
                if sd is None:
                    # The original distribution explicitly names sample_std.
                    sd=record[key].get('sample_std')
                record[key]['variance_ddof1']=None if sd is None else sd**2
    pairs={}
    for name,subset in subsets.items():
        pairs[name]={}
        for before,after in contrasts():
            value=paired(subset,before,after);value['both_failure_continuous']=continuous(subset,before,after)
            pairs[name][before+'__TO__'+after]=value
        print('PAIRED_POPULATION_COMPLETE',name,len(pairs[name]),flush=True)
    diagnoses=candidate_diagnosis(run_dir,ledger,refs)
    source_counts=[];coverage=[]
    for population,subset in subsets.items():
        ids={r['id'] for r in subset}
        for arm in ALL_ARMS:
            counts=Counter(r['action_sources'][arm]['role'] for r in subset)
            for role in ('N0','E0_REGENERATED','HISTORICAL_E0','OTHER','NO_POSE'):
                source_counts.append(dict(population=population,arm=arm,source=role,n=counts[role],denominator=len(subset)))
        for arm in B_ARMS+L_ARMS:
            values=[r['arms'][arm] for r in diagnoses if r['id'] in ids]
            coverage.append(dict(population=population,arm=arm,denominator=len(values),
                generated_has_success_n=sum(r['generated_before_constraints']['has_success'] for r in values),
                eligible_has_success_n=sum(r['after_constraints']['has_success'] for r in values),
                final_success_n=sum(r['final_success'] for r in values),
                selection_loss_n=sum(not r['final_success'] and r['after_constraints']['has_success'] for r in values),
                no_eligible_success_n=sum(not r['after_constraints']['has_success'] for r in values)))
    save_rows(output/'SCORED_ROWS.jsonl.gz',scored);save_rows(output/'STAGE_DIAGNOSIS.jsonl.gz',diagnoses)
    write(output/'RESULTS.json',dict(schema='BOUNDARY_ACTION_EVALUATION_V1',arms=ALL_ARMS,
        populations=summaries,paired=pairs,independent_real_test_n=0,protocol=evaluation_protocol()))
    table=[]
    for population,methods in summaries.items():
        for arm,v in methods.items():
            row=dict(population=population,arm=arm,denominator=v['denominator'],successes=v['successes'],
                success_rate=v['success_rate'],no_pose=v['no_pose'],T_gt50cm=v['T_gt50cm'],T_gt100cm=v['T_gt100cm'],R_gt45deg=v['R_gt45deg'])
            for name in ('translation_cm','rotation_deg'):
                row.update({name+'/'+k:x for k,x in v[name].items() if not isinstance(x,(dict,list))})
            table.append(row)
    csv_write(output/'COMPARISON_ALL_ARMS.csv',table)
    csv_write(output/'ROW_METRICS.csv.gz',(dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],arm=a,
        **r['metrics'][a],source=r['action_sources'][a]['role'],fallback=r['status'][a].get('fallback'),status=r['status'][a].get('status')) for r in scored for a in ALL_ARMS))
    transitions=[]
    for population,values in pairs.items():
        for value in values.values():
            before,after=value['before'],value['after'];subset={r['id']:r for r in subsets[population]}
            for group,ids in value['ids'].items():
                for fid in ids:
                    r=subset[fid];a,b=r['metrics'][before],r['metrics'][after]
                    transitions.append(dict(population=population,before=before,after=after,transition=group,id=fid,
                        before_translation_cm=a['translation_cm'],after_translation_cm=b['translation_cm'],
                        before_rotation_deg=a['rotation_deg'],after_rotation_deg=b['rotation_deg']))
    csv_write(output/'PAIRED_TRANSITIONS.csv.gz',transitions);csv_write(output/'ACTION_SOURCE_COUNTS.csv',source_counts)
    csv_write(output/'CANDIDATE_COVERAGE.csv',coverage)
    write(output/'EVALUATION_SEAL.json',dict(schema='BOUNDARY_ACTION_EVALUATION_SEAL_V1',complete=True,
        input_predictions_sha256=sha(prediction_path),prediction_seal_sha256=sha(run_dir/'PREDICTION_SEAL.json'),
        reference_sha256=sha(refpath),source_sha256=sha(__file__),registry_sha256=sha(Path(__file__).with_name('registry.py')),
        artifacts={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),all_failures_retained=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--original-score',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scoring-GO',action='store_true');a=p.parse_args()
    run(a.run_dir,a.original_score,a.output,scoring_GO=a.scoring_GO)


if __name__=='__main__':main()
