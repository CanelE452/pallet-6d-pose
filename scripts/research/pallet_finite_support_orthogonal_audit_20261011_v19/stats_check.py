"""Own-frozen stdlib replay of recorded V19 error moments and paired CIs.

Only source and frozen artifacts are read; production modules are not imported.
No pose or GT error is recomputed, and no model or image is executed.
Saved proxy errors remain proxy errors; the historical V16 full audit remains FAIL.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

NEW=('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS','HYBRID_OBSERVED_FRAME_POINT_ONLY')
OLD='HYBRID_POINT_CONSENSUS_RESTORED'
METHODS=('BASE','N3_SUBPIX','HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP',OLD,*NEW)
CONTRASTS=((NEW[0],NEW[1]),(NEW[0],'N3_SUBPIX'),(NEW[0],METHODS[3]),(NEW[0],METHODS[2]),(NEW[0],OLD),(NEW[1],OLD))
METRICS=('translation_cm','rotation_deg','ADDsym_cm')
UNITS=dict(translation_cm='cm',rotation_deg='degree',ADDsym_cm='cm')
SCOPES=('common_operational','candidate_new_pose','both_new_pose')
FILES=('STATISTICS_CHECK_PROTOCOL.json','STATISTICS_CHECK_STARTED.json','STATISTICS_CHECKS.json')
SCALAR_SHA='da0a91d232914a15768f233a698ebe33e5c1604e2506fee143f4dffb6732afb7'
ABS_TOL=1e-9;REL_TOL=1e-12
LIMITS=dict(saved_proxy_arithmetic_only=True,physical_truth_or_unseen_accuracy_certified=False,
    geometry_recomputed=False,old_full_V16_564_FAIL_remains_FAIL=True,new_GT_model_fit_projection_SVD_seed_calls=0)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def paths(a):
    folder=Path(a.input);m=read(folder/'METRICS.json');aliases=('cohort','bootstrap','fixed_scores','ordinary_scores','point_scores','new_scores','unchanged_statistics','unchanged_score')
    f={k:Path(m['inputs'][k]['path']) for k in aliases}
    f.update({n:folder/n for n in ('METRICS.json','STATISTICS_STARTED.json','EVALUATION_PROTOCOL.json','SCORING_RECEIPT.json','SCORING_AUTHORITY.json','SCORING_STARTED.json')})
    for n in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):f[n]=Path(m['inputs'][n]['path'])
    f.update(adapter=Path(m['inputs']['adapter']['path']),geometry_receipt=Path(m['inputs']['GEOMETRY_RECEIPT.json']['path']),
        geometry_checker=Path(m['inputs']['checker']['path']),checker=Path(__file__),
        independent_scalar=Path(a.repo)/'scripts/research/pallet_sparse_local_line_20261010_v9/verify.py')
    return f
def metadata(f,b):
    m,e,s,score,authority,start=[read(f[n]) for n in ('METRICS.json','EVALUATION_PROTOCOL.json','STATISTICS_STARTED.json','SCORING_RECEIPT.json','SCORING_AUTHORITY.json','SCORING_STARTED.json')]
    require(m['schema']=='observed_frame_raw_row_metrics_v19' and m['complete'] is True and m['V16_full_scalar_validation_passed'] is False,'completed V19 statistics, old FAIL preserved')
    require(e['schema']=='observed_frame_postseal_evaluation_protocol_v19' and e['methods']==m['methods']==s['methods']==list(METHODS) and e['contrasts']==m['contrasts']==s['contrasts']==[list(p) for p in CONTRASTS],'locked seven methods/six contrasts')
    require(e['old_full564_scalar_FAIL_remains_FAIL'] is True and e['frozen_before_new_GT_access_and_statistical_arithmetic'] is True and s['new_seed_draws']==s['new_geometry_or_GT_score_calls']==0,'prior freeze/no new geometry/draws')
    extra=dict(new_scores=b['new_scores'],scoring_receipt=b['SCORING_RECEIPT.json'],scoring_authority=b['SCORING_AUTHORITY.json'],scoring_started=b['SCORING_STARTED.json'],evaluation_protocol=b['EVALUATION_PROTOCOL.json'])
    require(m['inputs']==s['inputs'] and set(m['inputs'])==set(e['inputs'])|set(extra),'exact statistics input aliases')
    require(all(m['inputs'][k]==v for k,v in e['inputs'].items()) and all(same(m['inputs'][k],v) for k,v in extra.items()),'whole declared evaluation closure retained')
    for k in ('cohort','bootstrap','fixed_scores','ordinary_scores','point_scores','new_scores','unchanged_statistics','unchanged_score','adapter'):require(same(b[k],m['inputs'][k]),'current arithmetic input '+k)
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['frames']==245 and score['scored_rows']==490 and score['methods']==list(NEW),'completed paired490 score')
    require(same(b['new_scores'],score['predictions']) and authority['schema']=='isolated_observed_frame_proxy_score_authority_v19' and authority['inputs']==e['inputs'] and same(b['EVALUATION_PROTOCOL.json'],authority['own_evaluation_protocol']),'actual isolated score closure')
    require(authority['score_dispatch']==e['score_dispatch'] and authority['reference_or_error_formula_changes']==authority['numerical_geometry_calls']==0 and start['configured_scored_rows']==490 and start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'unchanged score formula after geometry PASS')
    v,p,vs=[read(f[n]) for n in ('VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json')]
    require(p['schema']=='observed_frame_hybrid_scalar_protocol_v19' and v['schema']=='observed_frame_hybrid_scalar_checks_v19' and v['complete'] is True and v['passed'] is True and v['failure_count']==0 and v['exception'] is None,'actual new geometry scalar PASS')
    require(v['inputs']==p['inputs']==vs['inputs'] and v['old_full_V16_validation_passed'] is False and v['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True,'separate historical failure scope')
    for item in (v['own_protocol'],vs['own_protocol'],start['independent_validation']):require(same(b['VALIDATION_PROTOCOL.json'],item),'actual validation protocol')
    for item in (v['checker'],p['checker']):require(same(b['geometry_checker'],item),'current geometry checker')
    require(same(b['geometry_receipt'],p['inputs']['geometry_receipt']) and v['actual_counts']['geometry_rows']==490 and v['actual_counts']['ledger_rows']==v['actual_counts']['frames']==245,'full paired245 authority')
    g=read(f['geometry_receipt']);require(g['complete'] is True and g['error'] is None and not g['cleanup_errors'] and g['environment_restored'] is True,'completed restored geometry receipt')
    require(m['population_variance_ddof']==m['population_std_ddof']==0 and m['unchanged_sample_variance_ddof']==1 and m['all245_preserved_no_performance_filter'] is True,'explicit population/sample definitions')
    return m
def kernels(f):
    require(binding(f['independent_scalar'])['sha256']==SCALAR_SHA,'immutable independent arithmetic source')
    names={'Audit','quantile','distribution'};nodes=[n for n in ast.parse(f['independent_scalar'].read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==names,'exact independent AST whitelist')
    env=dict(math=math,require=require,ABS_TOL=ABS_TOL,REL_TOL=REL_TOL)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f['independent_scalar'])+':unchanged_scalar_AST','exec'),env)
    return env
def moments(v,h):
    result=h['distribution'](v);mean=result['mean'];pv=math.fsum((x-mean)**2 for x in v)/len(v) if v else None
    return dict(result,population_variance=pv,population_std=math.sqrt(pv) if pv is not None else None)
def compare(saved,expected,unit,audit,label,counts):
    audit.eq(saved['n'],expected['n'],label+'.n');audit.eq(saved['unit'],unit,label+'.unit')
    for k,v in expected.items():
        if k!='n':audit.close(saved[k],v,label+'.'+k);counts['moment_scalars_checked']+=1
    counts['moment_distributions_checked']+=1
def compact(f,ids,sessions,audit):
    groups={m:{} for m in METHODS};count=0
    for key,allowed in (('fixed_scores',METHODS[:2]),('ordinary_scores',METHODS[2:4]),('point_scores',('HYBRID_POINT_CONSENSUS_ORIGINAL',OLD)),('new_scores',NEW)):
        seen=set()
        with gzip.open(f[key],'rt') as stream:
            for line in stream:
                row=json.loads(line);identity=(row['method'],row['id']);require(row['method'] in allowed and row['id'] in ids and identity not in seen,'unique complete source method/ID');seen.add(identity);count+=1
                if row['method'] not in groups:continue
                p=row['pose'];available=p['available'];new=row['new_pose_estimated'];fallback=row['fallback_used']
                require(all(type(v) is bool for v in (available,new,fallback)) and not(new and fallback) and row['pose_available']==available and row['session']==sessions[row['id']],'recorded availability/NEW/fallback/session')
                values={k:float(p['ADDsym_m'])*100 if k=='ADDsym_cm' else float(p[k]) for k in METRICS} if available else {}
                require(all(math.isfinite(v) for v in values.values()),'finite recorded errors')
                groups[row['method']][row['id']]=dict(id=row['id'],session=row['session'],available=available,new=new,fallback=fallback,status=row['output_status'],values=values)
        audit.eq(seen,{(m,i) for m in allowed for i in ids},'full source490.'+key)
    audit.eq(count,1960,'four complete paired streams');audit.eq({m:len(g) for m,g in groups.items()},{m:245 for m in METHODS},'selected seven full245 methods')
    return groups
def calculate(f,m,h,audit):
    c=read(f['cohort']);ids=c['ids'];sessions={r['id']:r['session'] for r in c['frames']};labels={r['id']:r['label'] for r in c['frames']}
    require(len(ids)==len(set(ids))==245 and set(sessions)==set(labels)==set(ids),'fixed full245 cohort')
    groups=compact(f,ids,sessions,audit)
    with gzip.open(f['bootstrap'],'rt') as stream:boot=json.load(stream)
    draws=boot['counts'];order=boot['sessions'];require(len(order)==len(set(order))==13 and set(sessions.values())==set(order) and len(draws)==10000,'existing13session10000draws')
    require(all(len(d)==13 and all(type(v) is int and v>=0 for v in d) and sum(d)==13 for d in draws),'fixed valid session draw multiplicities')
    counts=Counter(saved_error_rows_streamed=1960,saved_error_rows_selected=1715,frames=245,methods=7);out={}
    for name,chosen in (('combined',ids),('easy',[i for i in ids if labels[i]=='clean']),('medium',[i for i in ids if labels[i]=='moderate'])):
        block=m['strata'][name];audit.eq(block['ids'],chosen,name+'.ids');audit.eq(block['frames'],len(chosen),name+'.frames');audit.eq(len(chosen),{'combined':245,'easy':153,'medium':92}[name],name+'.fixed population')
        restored=dict(methods={},contrasts={})
        for method,g in groups.items():
            r=[g[i] for i in chosen];saved=block['methods'][method];pops=dict(operational=[v for v in r if v['available']],new_pose=[v for v in r if v['available'] and v['new']],fallback=[v for v in r if v['available'] and v['fallback']])
            summary=dict(denominator=len(r),operational=len(pops['operational']),new_pose=len(pops['new_pose']),fallback=len(pops['fallback']),no_pose=sum(not v['available'] for v in r),fixed_control=sum(v['status']=='FRESH_FIXED_CONTROL' for v in r),statuses=dict(Counter(v['status'] for v in r)))
            for k,v in summary.items():audit.eq(saved[k],v,name+'.'+method+'.'+k)
            summary['metrics']={}
            for scope,rows in pops.items():
                summary['metrics'][scope]={}
                for metric in METRICS:
                    value=moments([v['values'][metric] for v in rows],h);compare(saved['metrics'][scope][metric],value,UNITS[metric],audit,name+'.'+method+'.'+scope+'.'+metric,counts);summary['metrics'][scope][metric]=value
            restored['methods'][method]=summary
        for candidate,comparator in CONTRASTS:
            key=candidate+'_minus_'+comparator;restored['contrasts'][key]={};cg,bg=groups[candidate],groups[comparator]
            for scope in SCOPES:
                fixed=comparator=='N3_SUBPIX' and scope=='both_new_pose';effective='candidate_new_pose' if fixed else scope
                pair=[i for i in chosen if cg[i]['available'] and bg[i]['available'] and (effective=='common_operational' or cg[i]['new']) and (effective!='both_new_pose' or bg[i]['new'])]
                multiplicity=[sum(cg[i]['session']==s for i in pair) for s in order];den=[sum(x*y for x,y in zip(d,multiplicity)) for d in draws];nonempty=sum(v>0 for v in den)
                value=dict(scope='candidate_NEW_and_fixed_available' if fixed else scope,denominator=len(chosen),common_frames=len(pair),pair_ids=pair,candidate_new_pose_in_pairs=sum(cg[i]['new'] for i in pair),comparator_new_pose_in_pairs=sum(bg[i]['new'] for i in pair),bootstrap_nonempty_resamples=nonempty,bootstrap_empty_resamples=10000-nonempty)
                saved=block['contrasts'][key][scope]
                for k,v in value.items():audit.eq(saved[k],v,name+'.'+key+'.'+scope+'.'+k)
                if fixed:audit.eq(saved['fixed_control_NOT_relabelled_NEW'],True,'fixed N3 remains control');audit.eq(saved['comparator_fixed_available'],True,'fixed N3 available scope')
                value['metrics']={};value['matched_recorded_error_and_delta_moments']={}
                for metric in METRICS:
                    av=[cg[i]['values'][metric] for i in pair];bv=[bg[i]['values'][metric] for i in pair];dv=[x-y for x,y in zip(av,bv)];ss=[math.fsum(v for i,v in zip(pair,dv) if cg[i]['session']==s) for s in order]
                    samples=[math.fsum(x*y for x,y in zip(d,ss))/n for d,n in zip(draws,den) if n];ci=[h['quantile'](samples,p) for p in (.025,.975)] if samples else None
                    pv=dict(n=len(pair),mean_delta=math.fsum(dv)/len(dv) if dv else None,CI95=ci,bootstrap_nonempty_resamples=nonempty);saved_metric=saved['metrics'][metric]
                    for k in ('n','bootstrap_nonempty_resamples'):audit.eq(saved_metric[k],pv[k],name+'.pair.'+k)
                    audit.eq(saved_metric['unit'],UNITS[metric],'paired unit');audit.close(saved_metric['mean_delta'],pv['mean_delta'],name+'.paired mean')
                    if ci is None:audit.eq(saved_metric['CI95'],None,'empty paired CI')
                    else:
                        audit.eq(len(saved_metric['CI95']),2,'paired CI length')
                        for j,v in enumerate(ci):audit.close(saved_metric['CI95'][j],v,name+'.paired CI endpoint')
                    counts['metric_CI_slots_checked']+=1;value['metrics'][metric]=pv;value['matched_recorded_error_and_delta_moments'][metric]={}
                    for population,raw in (('candidate',av),('comparator',bv),('delta',dv)):
                        result=moments(raw,h);compare(saved['matched_recorded_error_and_delta_moments'][metric][population],result,UNITS[metric],audit,name+'.matched.'+population+'.'+metric,counts);value['matched_recorded_error_and_delta_moments'][metric][population]=result
                counts['paired_groups_checked']+=1;restored['contrasts'][key][scope]=value
        out[name]=restored
    audit.eq(counts['moment_distributions_checked'],675,'189 marginal +486 matched distributions');audit.eq(counts['moment_scalars_checked'],5400,'eight fields per distribution');audit.eq(counts['paired_groups_checked'],54,'paired groups');audit.eq(counts['metric_CI_slots_checked'],162,'paired CI slots')
    for comparator,field in (('N3_SUBPIX','primary_both_T_R_means_better_than_N3_full245'),(NEW[1],'primary_both_T_R_means_better_than_point_full245')):
        p=out['combined']['contrasts'][NEW[0]+'_minus_'+comparator]['common_operational'];audit.eq(m[field],p['common_frames']==245 and all(p['metrics'][k]['mean_delta']<0 for k in METRICS[:2]),'saved full245 T/R conclusion')
    return dict(counts),out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.absolute();require(out!=a.input.absolute() and not any(q.is_symlink() for q in (out,*out.parents)),'distinct nonsymlink output');out.mkdir(parents=True,exist_ok=True)
    for n in FILES if a.stage=='freeze' else FILES[1:]:require(not (out/n).exists(),'preserve '+n)
    f=paths(a);b={k:binding(v) for k,v in f.items()};m=metadata(f,b);require(b['independent_scalar']['sha256']==SCALAR_SHA,'exact independent scalar source')
    require(not any(v.absolute().is_relative_to(out) for k,v in f.items() if k!='checker'),'output overlaps inputs')
    if a.stage=='freeze':
        write(out/FILES[0],dict(schema='independent_observed_frame_statistics_protocol_v19',inputs=b,checker=b['checker'],methods=list(METHODS),contrasts=[list(p) for p in CONTRASTS],scalar_absolute_tolerance=ABS_TOL,scalar_relative_tolerance=REL_TOL,independent_AST_whitelist=['Audit','quantile','distribution'],frozen_before_own_saved_error_arithmetic=True,limits=LIMITS));return
    own=read(out/FILES[0]);require(own['schema']=='independent_observed_frame_statistics_protocol_v19' and own['inputs']==b and own['methods']==list(METHODS) and own['contrasts']==[list(p) for p in CONTRASTS] and own['limits']==LIMITS and own['scalar_absolute_tolerance']==ABS_TOL and own['scalar_relative_tolerance']==REL_TOL,'own frozen code/inputs/policy changed')
    protocol=binding(out/FILES[0]);write(out/FILES[1],dict(inputs=b,own_protocol=protocol,checker=b['checker'],new_production_calls=0));h=kernels(f);audit=h['Audit']();error=None;counts={};reconstructed={}
    try:counts,reconstructed=calculate(f,m,h,audit);require(b=={k:binding(v) for k,v in f.items()},'inputs changed during arithmetic')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    write(out/FILES[2],dict(schema='independent_observed_frame_statistics_checks_v19',complete=error is None,passed=error is None,inputs=b,own_protocol=protocol,checker=b['checker'],error=error,exact_checks=audit.exact_checks,numeric_checks=audit.numeric_checks,max_absolute_difference=audit.max_difference,actual_counts=counts,reconstructed=reconstructed,limits=LIMITS,new_production_calls=0,old_full_V16_validation_passed=False))
    if error:raise SystemExit(1)
if __name__=='__main__':main()
