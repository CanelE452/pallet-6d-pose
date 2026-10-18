"""Independent saved-pose arithmetic; no producer/evaluator/optimizer imports.

The previous campaign's independent atan2/fsum implementation is reused as
independent arithmetic, not its production evaluator. A secondary, separately
written original norm/acos expression verifies strict signs at floating zero.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import math
from pathlib import Path
import resource
import time
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import independent_check as I
from .registry import ALL_ARMS, NEW_ARMS, B_ARMS, L_ARMS, L_GROUPS, SEEDS, POPULATIONS, EXPECTED_N, contrasts


def status(entry):
    raw=entry.get('prediction_status');result=raw.copy() if isinstance(raw,dict) else {}
    for key in ('accepted','fallback','technical_failure','solver_rejected','timeout','status','reason'):
        if key not in result:result[key]=entry.get(key)
    if isinstance(raw,str):result['status']=raw
    return result


def csv_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='') as f:return list(csv.DictReader(f))


def exact_csv_identity(values, fields, expected, path, problems):
    """Require every declared identity once, without reading metric values.

    A correct row duplicated in place of another correct row must fail even
    when the total row count and every visited numeric value still agree.
    """
    expected=set(expected);counts=Counter(tuple(row.get(k)for k in fields)for row in values)
    duplicate=[dict(identity=list(key),rows=n)for key,n in counts.items()if n!=1]
    missing=expected-set(counts);unexpected=set(counts)-expected
    if duplicate:problems.append(dict(path=path,kind='DUPLICATE_IDENTITIES',fields=list(fields),identities=duplicate))
    if missing or unexpected:
        order=lambda key:tuple(str(x)for x in key)
        problems.append(dict(path=path,kind='EXACT_IDENTITY_COVERAGE',fields=list(fields),
            missing=[list(k)for k in sorted(missing,key=order)],unexpected=[list(k)for k in sorted(unexpected,key=order)]))
    return expected


def near(a,b,tolerance):
    return abs(a-b)<=tolerance['aggregate_abs']+tolerance['relative']*max(abs(a),abs(b))


def compare_tree(actual, expected, path, tolerance, problems):
    """Compare every independently computed field; no evaluator totals reused."""
    if isinstance(actual,dict):
        if not isinstance(expected,dict):problems.append(dict(path=path,kind='DICT_TYPE'));return
        for key,value in actual.items():
            if key not in expected:
                # Implementation-specific CI notes are not numeric evidence.
                if key in ('reason','method','empty_resamples'):continue
                problems.append(dict(path=path+'/'+key,kind='MISSING_FIELD'));continue
            compare_tree(value,expected[key],path+'/'+key,tolerance,problems)
    elif isinstance(actual,list):
        if not isinstance(expected,list) or len(actual)!=len(expected):problems.append(dict(path=path,kind='LIST_LENGTH'));return
        for j,(a,b) in enumerate(zip(actual,expected)):compare_tree(a,b,path+'/'+str(j),tolerance,problems)
    elif isinstance(actual,bool) or actual is None or isinstance(actual,str) or isinstance(actual,int):
        if actual!=expected:problems.append(dict(path=path,kind='EXACT',independent=actual,production=expected))
    elif isinstance(actual,float):
        if not isinstance(expected,(int,float)) or not near(actual,float(expected),tolerance):
            problems.append(dict(path=path,kind='NUMERIC',independent=actual,production=expected))


def role(pose,poses,scale=1.):
    if not pose.get('available'):return dict(role='NO_POSE',aliases=[])
    def close(other):
        if not other.get('available'):return False
        return all(abs(float(x)-float(y))<=1e-7*scale for x,y in zip(pose['centroid'],other['centroid'])) and all(
            abs(float(x)-float(y))<=1e-7 for a,b in zip(pose['R_physical'],other['R_physical']) for x,y in zip(a,b))
    aliases=[a for a in ('N0','E0_REGENERATED','E0') if close(poses[a])]
    name='N0' if 'N0'in aliases else 'E0_REGENERATED' if 'E0_REGENERATED'in aliases else 'HISTORICAL_E0' if 'E0'in aliases else 'OTHER'
    return dict(role=name,aliases=aliases)


def run(run_dir,original_score,evaluation,output,*,scoring_GO=False):
    if not scoring_GO:raise PermissionError('Postseal root GO required')
    started,cpu=time.monotonic(),time.process_time();run_dir=Path(run_dir);evaluation=Path(evaluation);output=Path(output)
    plan=I.read(run_dir/'PLAN_LOCK.json');seal=I.read(run_dir/'PREDICTION_SEAL.json');ev=I.read(evaluation/'EVALUATION_SEAL.json')
    if not seal.get('complete') or not seal.get('before_scoring') or seal.get('reference_inputs') is not False:
        raise ValueError('Complete prediction-only pre-reference seal required')
    if I.sha(run_dir/'PLAN_LOCK.json')!=seal['plan_lock_sha256'] or I.sha(run_dir/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:
        raise ValueError('Prediction or plan bytes changed')
    if I.sha(run_dir/'PREDICTION_SEAL.json')!=ev['prediction_seal_sha256']:raise ValueError('Evaluation prediction seal changed')
    for name,digest in ev['artifacts'].items():
        if I.sha(evaluation/name)!=digest:raise ValueError('Production artifact changed: '+name)
    ledger=list(I.rows(run_dir/'PREDICTIONS.jsonl.gz'));ids=[r['id']for r in ledger]
    if ids!=seal['ordered_ids'] or len(ids)!=238 or len(set(ids))!=238:raise ValueError('Exact238 output IDs required')
    if any(set(r['predictions'])!=set(ALL_ARMS)for r in ledger):raise ValueError('Arm missing on failing frame')
    tolerance=I.read(run_dir/'EVALUATION_PROTOCOL.json')['independent_tolerance']
    refpath=Path(original_score)/'FIXED_REFERENCES.json'
    if I.sha(refpath)!=ev['reference_sha256']:raise ValueError('Reference bytes changed')
    references=I.read(refpath)['references']
    production={r['id']:r for r in I.rows(evaluation/'SCORED_ROWS.jsonl.gz')};results=I.read(evaluation/'RESULTS.json')
    scored=[];problems=[];maximum=dict(translation_cm=0.,rotation_deg=0.);representational_sign_changes=Counter()
    for row in ledger:
        ref=references[row['partition']][row['id']];poses={a:row['predictions'][a]['actual_pose']for a in ALL_ARMS}
        own={k:row[k]for k in ('id','partition','session','cluster_id')}
        own.update(metrics={a:I.pose_metric(poses[a],ref)for a in ALL_ARMS},
            strict_direction_metrics={a:I.strict_direction_values(poses[a],ref)for a in ALL_ARMS},
            status={a:status(row['predictions'][a])for a in ALL_ARMS},action_sources={a:role(poses[a],poses,row['object_scale_m'])for a in ALL_ARMS})
        p=production[row['id']]
        for k in ('partition','session','cluster_id'):
            if p[k]!=own[k]:problems.append(dict(path=row['id']+'/'+k,kind='METADATA'))
        for a in ALL_ARMS:
            m,pm=own['metrics'][a],p['metrics'][a]
            for k in ('success','pose_available'):
                if m[k]!=pm[k]:problems.append(dict(path=row['id']+'/'+a+'/'+k,kind='METRIC_EXACT'))
            for k in maximum:
                if m[k] is None or pm[k] is None:
                    if m[k]!=pm[k]:problems.append(dict(path=row['id']+'/'+a+'/'+k,kind='AVAILABLE'))
                else:
                    gap=abs(m[k]-pm[k]);maximum[k]=max(maximum[k],gap)
                    if gap>tolerance[k+'_abs']+tolerance['relative']*abs(pm[k]):
                        problems.append(dict(path=row['id']+'/'+a+'/'+k,kind='METRIC_NUMERIC',gap=gap))
            compare_tree(own['status'][a],p['status'][a],row['id']+'/'+a+'/status',tolerance,problems)
            compare_tree(own['action_sources'][a],p['action_sources'][a],row['id']+'/'+a+'/source',tolerance,problems)
        scored.append(own)
    subsets={name:[r for r in scored if r['partition']in parts]for name,parts in POPULATIONS.items()}
    if {k:len(v)for k,v in subsets.items()}!=EXPECTED_N:raise ValueError('Independent denominator mismatch')
    own_summaries={};own_pairs={}
    for name,subset in subsets.items():
        own_summaries[name]=I.aggregate(subset,ALL_ARMS)
        compare_tree(own_summaries[name],results['populations'][name],name+'/aggregate',tolerance,problems)
        own_pairs[name]={}
        for before,after in contrasts():
            key=before+'__TO__'+after
            value=I.paired(subset,before,after,direction_view='original_expression')
            compare_tree(value,results['paired'][name][key],name+'/paired/'+key,tolerance,problems)
            own_pairs[name][key]=value
            for r in subset:
                for metric in ('translation_cm','rotation_deg'):
                    a,b=r['metrics'][before][metric],r['metrics'][after][metric]
                    c,d=r['strict_direction_metrics'][before][metric],r['strict_direction_metrics'][after][metric]
                    if any(x is None for x in (a,b,c,d)):continue
                    sign=lambda x:(x>0)-(x<0)
                    if sign(b-a)!=sign(d-c):representational_sign_changes[name+'/'+metric]+=1
        print('INDEPENDENT_POPULATION_COMPLETE',name,'mismatches',len(problems),flush=True)
    # Independent scalar CSV joins guard the public table's row/arm/unit identity.
    csvvalues=csv_rows(evaluation/'ROW_METRICS.csv.gz')
    exact_csv_identity(csvvalues,('id','arm'),((r['id'],a)for r in scored for a in ALL_ARMS),'ROW_METRICS',problems)
    csvindex={(r['id'],r['arm']):r for r in csvvalues}
    for row in scored:
        for arm in ALL_ARMS:
            c=csvindex.get((row['id'],arm))
            if not c:continue
            if any(c[k]!=row[k]for k in ('partition','session','cluster_id')) or c['source']!=row['action_sources'][arm]['role']:
                problems.append(dict(path=row['id']+'/'+arm+'/csv',kind='CSV_IDENTITY'))
            for key in ('success','pose_available'):
                if c[key]!=str(row['metrics'][arm][key]):
                    problems.append(dict(path=row['id']+'/'+arm+'/csv/'+key,kind='CSV_METRIC_EXACT'))
            for k in ('translation_cm','rotation_deg'):
                x=row['metrics'][arm][k];y=None if c[k]==''else float(c[k])
                if (x is None)!=(y is None) or x is not None and not near(x,y,tolerance):
                    problems.append(dict(path=row['id']+'/'+arm+'/csv/'+k,kind='CSV_NUMERIC'))
    public_summary=csv_rows(evaluation/'COMPARISON_ALL_ARMS.csv')
    summary_keys=exact_csv_identity(public_summary,('population','arm'),((p,a)for p in POPULATIONS for a in ALL_ARMS),'COMPARISON_ALL_ARMS',problems)
    for entry in public_summary:
        if(entry.get('population'),entry.get('arm'))not in summary_keys:continue
        own=own_summaries[entry['population']][entry['arm']]
        for key in ('denominator','successes','no_pose','T_gt50cm','T_gt100cm','R_gt45deg'):
            if int(entry[key])!=own[key]:problems.append(dict(path='COMPARISON/'+entry['population']+'/'+entry['arm']+'/'+key,kind='CSV_COUNT'))
        for axis in ('translation_cm','rotation_deg'):
            for key,value in own[axis].items():
                field=axis+'/'+key
                if field not in entry or isinstance(value,bool)or isinstance(value,str):continue
                decoded=None if entry[field]==''else float(entry[field])
                if (decoded is None)!=(value is None)or value is not None and not near(float(value),decoded,tolerance):
                    problems.append(dict(path='COMPARISON/'+entry['population']+'/'+entry['arm']+'/'+field,kind='CSV_STATISTIC'))
    public_sources=csv_rows(evaluation/'ACTION_SOURCE_COUNTS.csv')
    source_keys=exact_csv_identity(public_sources,('population','arm','source'),
        ((p,a,s)for p in POPULATIONS for a in ALL_ARMS for s in ('N0','E0_REGENERATED','HISTORICAL_E0','OTHER','NO_POSE')),'ACTION_SOURCE_COUNTS',problems)
    for entry in public_sources:
        if(entry.get('population'),entry.get('arm'),entry.get('source'))not in source_keys:continue
        subset=subsets[entry['population']];n=sum(r['action_sources'][entry['arm']]['role']==entry['source']for r in subset)
        if n!=int(entry['n'])or len(subset)!=int(entry['denominator']):problems.append(dict(path='ACTION_SOURCE_COUNTS/'+str(entry),kind='CSV_COUNT'))
    public_pairs=csv_rows(evaluation/'PAIRED_TRANSITIONS.csv.gz')
    expected_pair_keys=exact_csv_identity(public_pairs,('population','before','after','id'),
        ((p,b,a,r['id'])for p,subset in subsets.items()for b,a in contrasts()for r in subset),'PAIRED_TRANSITIONS',problems)
    by_id={r['id']:r for r in scored}
    for entry in public_pairs:
        key=(entry['population'],entry['before'],entry['after'],entry['id'])
        if key not in expected_pair_keys:continue
        r=by_id[entry['id']];a=r['metrics'][entry['before']];b=r['metrics'][entry['after']]
        group='BOTH_SUCCESS'if a['success']and b['success']else'RECOVERY'if b['success']else'DAMAGE'if a['success']else'BOTH_FAILURE'
        if r['partition']not in POPULATIONS[entry['population']]or entry['transition']!=group:
            problems.append(dict(path='PAIRED_TRANSITIONS/'+str(key),kind='CSV_TRANSITION'))
        for prefix,value in [('before',a),('after',b)]:
            for axis in ('translation_cm','rotation_deg'):
                text=entry[prefix+'_'+axis];other=None if text==''else float(text);x=value[axis]
                if (x is None)!=(other is None)or x is not None and not near(x,other,tolerance):
                    problems.append(dict(path='PAIRED_TRANSITIONS/'+str(key)+'/'+prefix+'/'+axis,kind='CSV_NUMERIC'))
    # Oracle existence is recomputed directly from saved candidate poses, with
    # feasibility flags independently audited by the separate finite checker.
    reference_for=lambda r:references[r['partition']][r['id']]
    features={r['id']:r for r in I.rows(run_dir/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz')}
    coverage_by_id={}
    for index,row in enumerate(ledger):
        ref=reference_for(row);f=features[row['id']]
        n0=dict(actual_pose=f['fallback']);e0=dict(actual_pose=f['e0_regenerated'])
        any_good=lambda bank:any(I.pose_metric(I.candidate_pose(c),ref)['success']for c in bank)
        full=any_good(f['candidates']+[n0,e0]);eligible=any_good([c for c in f['candidates']if c['eligible']]+[n0,e0]);pair=any_good([n0,e0])
        arms={a:(pair,pair)if '_PAIR_'in a else(full,eligible)for a in L_ARMS}
        b=next(I.rows(run_dir/'b_finite_v1/frames'/f'{index:04d}'/'RESULT.jsonl.gz'));carried=b['banks']['B_FINITE_BANK']
        for arm in B_ARMS:
            generated=([n0]if arm=='B_FINITE_LOCAL'else carried)+[c for c in b['attempts']if c['arm']==arm]
            admitted=[c for c,s in zip(b['banks'][arm],b['finite_scores'][arm])if any(v['feasible']for v in s.values())]
            arms[arm]=(any_good(generated),any_good(admitted))
        coverage_by_id[row['id']]=arms
    candidate_csv=csv_rows(evaluation/'CANDIDATE_COVERAGE.csv')
    candidate_keys=exact_csv_identity(candidate_csv,('population','arm'),((p,a)for p in POPULATIONS for a in B_ARMS+L_ARMS),'CANDIDATE_COVERAGE',problems)
    for entry in candidate_csv:
        if(entry.get('population'),entry.get('arm'))not in candidate_keys:continue
        subset=subsets[entry['population']];arm=entry['arm']
        generated=[coverage_by_id[r['id']][arm][0]for r in subset];eligible=[coverage_by_id[r['id']][arm][1]for r in subset]
        final=[r['metrics'][arm]['success']for r in subset]
        counts=dict(denominator=len(subset),generated_has_success_n=sum(generated),eligible_has_success_n=sum(eligible),
            final_success_n=sum(final),selection_loss_n=sum(e and not f for e,f in zip(eligible,final)),no_eligible_success_n=sum(not e for e in eligible))
        for key,value in counts.items():
            if value!=int(entry[key]):problems.append(dict(path='CANDIDATE_COVERAGE/'+entry['population']+'/'+arm+'/'+key,kind='CSV_ORACLE'))
    strata_csv=run_dir/'posthoc_v1/HISTORICAL_STRATA.csv';strata_checked=0;factor_checked=0;gain_checked=0
    if strata_csv.exists():
        posthoc=I.read(run_dir/'posthoc_v1/POSTHOC_MECHANISM.json');membership=posthoc['membership']
        oracle_path=Path(original_score).parent/'oracle_v1/ORACLE_ROWS.jsonl.gz'
        if I.sha(oracle_path)!=posthoc['original_oracle_sha256']:raise ValueError('Original stratum source changed')
        aliases={'family_selection':'FAMILY_SELECTION_LOSS','within_family_final_selection':'WITHIN_FAMILY_SELECTION_LOSS',
            'confirmed_original_consumer_observation_or_proposal_limit':'NO_SUCCESS_CANDIDATE'}
        groups={}
        for old in I.rows(oracle_path):
            if old['final_joint_success']is True:group='FINAL_SUCCESS_CONTROL'
            elif old['final_joint_success']is False and old.get('original_consumer_generated_bank_complete')is True:
                group=aliases[old['earliest_observed_failure_stage']]
            else:raise ValueError('Original stratum is unresolved')
            groups[old['id']]=group
        if {r['id']:r['stratum']for r in membership}!=groups:problems.append(dict(path='HISTORICAL_STRATA',kind='ORIGINAL_MEMBERSHIP'))
        expected_counts={'FINAL_SUCCESS_CONTROL':114,'FAMILY_SELECTION_LOSS':10,'WITHIN_FAMILY_SELECTION_LOSS':22,'NO_SUCCESS_CANDIDATE':46}
        if Counter(groups[r['id']]for r in subsets['PRIMARY192'])!=Counter(expected_counts):problems.append(dict(path='HISTORICAL_STRATA',kind='ORIGINAL_CLASS_COUNTS'))
        strata_values=csv_rows(strata_csv)
        strata_keys=exact_csv_identity(strata_values,('population','stratum','arm'),
            ((p,g,a)for p,subset in subsets.items()for g in {groups[r['id']]for r in subset}for a in ALL_ARMS),'HISTORICAL_STRATA',problems)
        for entry in strata_values:
            if(entry.get('population'),entry.get('stratum'),entry.get('arm'))not in strata_keys:continue
            subset=[r for r in subsets[entry['population']]if groups[r['id']]==entry['stratum']];arm=entry['arm']
            n=sum(r['metrics'][arm]['success']for r in subset)
            if len(subset)!=int(entry['denominator'])or n!=int(entry['successes']):problems.append(dict(path='HISTORICAL_STRATA/'+str(entry),kind='COUNT'))
            for axis,prefix in [('translation_cm','translation'),('rotation_deg','rotation')]:
                dist=I.distribution([r['metrics'][arm][axis]for r in subset],len(subset))
                for field,key in [(prefix+'_mean_'+('cm'if prefix=='translation'else'deg'),'mean'),(prefix+'_sample_SD','sample_std')]:
                    text=entry[field];value=None if text==''else float(text);own=dist[key]
                    if (value is None)!=(own is None)or own is not None and not near(own,value,tolerance):
                        problems.append(dict(path='HISTORICAL_STRATA/'+entry['population']+'/'+arm+'/'+field,kind='STATISTIC'))
            for comparator in ('N0','E0','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR','LEGACY_L_HEAD_s20261018'):
                recovery=sum(not r['metrics'][comparator]['success']and r['metrics'][arm]['success']for r in subset)
                damage=sum(r['metrics'][comparator]['success']and not r['metrics'][arm]['success']for r in subset)
                if recovery!=int(entry['recovery_vs_'+comparator])or damage!=int(entry['damage_vs_'+comparator]):
                    problems.append(dict(path='HISTORICAL_STRATA/'+entry['population']+'/'+arm+'/'+comparator,kind='TRANSITION'))
            strata_checked+=1
        factor_path=strata_csv.parent/'FACTOR_AND_PAIR_COUNTS.csv'
        if factor_path.exists():
            factor_values=csv_rows(factor_path)
            factor_keys=exact_csv_identity(factor_values,('population','seed'),
                ((p,str(s))for p in POPULATIONS for s in SEEDS),'FACTOR_AND_PAIR_COUNTS',problems)
            for entry in factor_values:
                if(entry.get('population'),entry.get('seed'))not in factor_keys:continue
                population=entry['population'];suffix='_s'+entry['seed'];table={g:own_summaries[population][g+suffix]['successes']for g in L_GROUPS}
                expected=dict(denominator=len(subsets[population]),
                    H1_minus_H0_at_D0=table['D0_H1_MLP']-table['D0_H0_MLP'],
                    H1_minus_H0_at_D1=table['D1_H1_MLP']-table['D1_H0_MLP'],
                    D1_minus_D0_at_H0=table['D1_H0_MLP']-table['D0_H0_MLP'],
                    D1_minus_D0_at_H1=table['D1_H1_MLP']-table['D0_H1_MLP'],
                    descriptive_interaction_count=(table['D1_H1_MLP']-table['D1_H0_MLP'])-(table['D0_H1_MLP']-table['D0_H0_MLP']),
                    pair_minus_full=table['D1_H1_PAIR_MLP']-table['D1_H1_MLP'],
                    role_affine_minus_MLP=table['D1_H1_ROLE_AFFINE']-table['D1_H1_MLP'])
                for key,value in expected.items():
                    if int(entry[key])!=value:problems.append(dict(path='FACTOR_AND_PAIR_COUNTS/'+population+'/'+entry['seed']+'/'+key,kind='CSV_COUNT'))
                factor_checked+=1
        else:problems.append(dict(path='FACTOR_AND_PAIR_COUNTS',kind='REQUIRED_POSTHOC_NOT_RUN'))
        gain_path=strata_csv.parent/'GAIN_DAMAGE_ACTUAL_SOURCE.csv'
        if gain_path.exists():
            gain_values=csv_rows(gain_path);expected_gain={}
            for population,subset in subsets.items():
                for arm in NEW_ARMS:
                    source_groups=Counter();denominators=Counter()
                    for row in subset:
                        a,b=row['metrics']['E0']['success'],row['metrics'][arm]['success']
                        transition='BOTH_SUCCESS'if a and b else'RECOVERY'if b else'DAMAGE'if a else'BOTH_FAILURE'
                        source_groups[transition,row['action_sources'][arm]['role']]+=1;denominators[transition]+=1
                    for(transition,source),n in source_groups.items():
                        expected_gain[population,arm,'E0',transition,source]=(n,denominators[transition])
            gain_keys=exact_csv_identity(gain_values,('population','arm','vs','transition','source'),expected_gain,'GAIN_DAMAGE_ACTUAL_SOURCE',problems)
            for entry in gain_values:
                key=tuple(entry.get(k)for k in('population','arm','vs','transition','source'))
                if key not in gain_keys:continue
                n,denominator=expected_gain[key]
                if int(entry['n'])!=n or int(entry['transition_denominator'])!=denominator:
                    problems.append(dict(path='GAIN_DAMAGE_ACTUAL_SOURCE/'+str(key),kind='CSV_COUNT'))
                gain_checked+=1
        else:problems.append(dict(path='GAIN_DAMAGE_ACTUAL_SOURCE',kind='REQUIRED_POSTHOC_NOT_RUN'))
    else:problems.append(dict(path='HISTORICAL_STRATA',kind='REQUIRED_POSTHOC_NOT_RUN'))
    output.mkdir(parents=True,exist_ok=False)
    I.write(output/'INDEPENDENT_RESULTS.json',dict(populations=own_summaries,paired=own_pairs))
    I.write(output/'INDEPENDENT_CHECK.json',dict(schema='BOUNDARY_ACTION_INDEPENDENT_NUMERIC_V1',
        status='PASS'if not problems else 'FAILED',rows=238,arms=len(ALL_ARMS),row_arm_pairs=238*len(ALL_ARMS),
        populations={k:len(v)for k,v in subsets.items()},contrasts_per_population=len(contrasts()),
        tolerance_locked_before_primary=tolerance,maximum_metric_difference=maximum,
        stable_vs_original_expression_sign_changes=dict(representational_sign_changes),
        strict_direction_policy='secondary independent original floating expression, not a changed success criterion',
        all_mismatches=problems,mismatch_count=len(problems),
        reviewer_type='independent arithmetic code; model authored; not independent physical ground truth validation',
        production_metric_or_aggregate_imports=False,
        public_numeric_csv_checks=dict(comparison_rows=len(public_summary),row_metric_rows=len(csvvalues),
            source_count_rows=len(public_sources),paired_transition_rows=len(public_pairs),candidate_coverage_rows=len(candidate_csv),historical_stratum_rows=strata_checked,
            factor_count_rows=factor_checked,gain_damage_source_rows=gain_checked),
        independent_metric_source_sha256=I.sha(I.__file__),
        checker_source_sha256=I.sha(__file__),prediction_seal_sha256=I.sha(run_dir/'PREDICTION_SEAL.json'),
        evaluation_seal_sha256=I.sha(evaluation/'EVALUATION_SEAL.json'),
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    if problems:raise SystemExit('Independent arithmetic failed; original failed receipt retained: '+str(len(problems)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--original-score',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scoring-GO',action='store_true');a=p.parse_args()
    run(a.run_dir,a.original_score,a.evaluation,a.output,scoring_GO=a.scoring_GO)


if __name__=='__main__':main()
