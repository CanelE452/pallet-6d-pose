"""Independent stdlib verification of saved V11 moments and paired CIs.

Own-freeze before arithmetic; reuse only exact AST scalar kernels from frozen
V9 verification. No production imports, model, GT, fitting, SVD or new draws.
Saved GEOMETRIC_PROXY DEV metric arithmetic is not independent physical truth.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
PRIVATE=Path('/tmp/pallet-endpoint-observability-repair-private-20261011-v11')
SOURCE=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/verify.py'
SOURCE_SHA='da0a91d232914a15768f233a698ebe33e5c1604e2506fee143f4dffb6732afb7'
FUNCS=('require','read','reject_symlink','digest','binding','bound','quantile','distribution','compact_rows')
METHODS=('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','ROLE_BOUNDARY_LOCAL_POINT_LINE','FILTERED_LOCAL','FILTERED_LOCAL_WEAK_SAFE')
CONTRASTS=(('FILTERED_LOCAL','ROLE_BOUNDARY_LOCAL_POINT_LINE'),('FILTERED_LOCAL_WEAK_SAFE','FILTERED_LOCAL'),
    ('FILTERED_LOCAL_WEAK_SAFE','ROLE_BOUNDARY_LOCAL_POINT_LINE'),('FILTERED_LOCAL_WEAK_SAFE','ROLE_BOUNDARY_H_ROBUST'),
    ('FILTERED_LOCAL_WEAK_SAFE','N3_SUBPIX'))
METRICS=('translation_cm','rotation_deg','ADDsym_cm')
SCOPES=('common_operational','candidate_new_pose','both_new_pose')
FILES=('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')
LIMITS=dict(standard_library_only=True,production_imports=False,source_GT_model_fit_SVD_new_draw_calls=0,
    saved_metric_arithmetic_is_not_independent_physical_truth=True,known_DEV_not_unseen_evaluation=True,
    rows_and_operating_policy_modified=False,automatic_retry=False)


def require(value,message):
    if not value:raise ValueError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def binding(path):
    p=Path(path).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input '+str(p))
    sha=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=sha.hexdigest())


def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def public_path(item):
    path=Path(item['path']);require(not path.is_absolute() and '..' not in path.parts,'public metric authority required')
    return REPO/path


def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'unchanged V9 scalar kernels required')
    selected=[n for n in ast.parse(SOURCE.read_text()).body if
        isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in (*FUNCS,'Audit')]
    require({n.name for n in selected}==set(FUNCS)|{'Audit'},'exact independent scalar AST whitelist required')
    module=ast.Module(body=selected,type_ignores=[])
    return module,dict(source=binding(SOURCE),names=list(FUNCS)+['Audit'],
        AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),absolute_tolerance=1e-9,relative_tolerance=1e-12)


def inputs(args):
    p=args.input;stats=read(p/'METRICS.json')
    paths={key:public_path(value) for key,value in stats['inputs'].items()}
    paths.update(METRICS=p/'METRICS.json',checker=Path(__file__).resolve(),scalar_source=SOURCE,
        statistics_started=p/'STATISTICS_STARTED.json',
        geometry_seal=p/'GEOMETRY_SEAL.json',repair_receipt=p/'CONTROL_RECEIPT.json',geometry=p/'GEOMETRY_SEALED.jsonl.gz',
        ledgers=p/'CONTROL_LEDGERS.jsonl.gz',population=p/'PARENT_POPULATION_CHECKS.json',
        geometry_validation=p/'VALIDATION_CHECKS.json',geometry_validation_protocol=p/'VALIDATION_PROTOCOL.json',
        geometry_checker=Path(__file__).with_name('validation_checks.py'),
        geometry_review=p/'REVIEW_VALIDATION_CHECKS.json',geometry_review_protocol=p/'REVIEW_VALIDATION_PROTOCOL.json',
        geometry_review_started=p/'REVIEW_VALIDATION_STARTED.json',geometry_review_checker=Path(__file__).with_name('validation_review.py'),
        scoring_authority=p/'SCORING_AUTHORITY_SUPPLEMENT.json',scoring_wrapper=Path(__file__).with_name('evaluation_review.py'),
        scoring_started=p/'SCORING_STARTED.json')
    require(Path(args.protocol).resolve()==paths['protocol'].resolve(),'exact frozen accuracy protocol required')
    paths['protected_originals']=public_path(read(paths['protocol'])['inputs']['parent:protection'])
    return paths


def metadata(paths,bindings):
    stats,core,score,seal,receipt,check,own=[read(paths[k]) for k in
        ('METRICS','protocol','scoring_receipt','geometry_seal','repair_receipt','geometry_validation','geometry_validation_protocol')]
    require(stats['schema']=='fixed_repair_raw_row_metrics_v11' and stats['complete'] is True and
        stats['methods']==list(METHODS) and stats['contrasts']==[list(p) for p in CONTRASTS] and stats['scopes']==list(SCOPES),'fixed V11 statistics scope differs')
    require(core['schema']=='fixed_endpoint_observability_repair_protocol_v11' and core['frames']==245 and
        core['methods']==list(METHODS[-2:]),'fixed repair geometry protocol differs')
    require(score['schema']=='repair_postseal_proxy_scoring_v11' and score['complete'] is True and score['frames']==245 and
        score['scored_rows']==490 and score['methods']==list(METHODS[-2:]) and score['GT_access_only_after_complete_geometry_and_cleanup'] is True,'complete postseal scoring required')
    require(same(score['predictions'],bindings['new_scores']) and same(score['protocol'],bindings['protocol']) and
        same(score['geometry_seal'],bindings['geometry_seal']),'scored new row authority differs')
    require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and
        seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and receipt['complete'] is True and
        receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,'full GT-free geometry and cleanup required')
    for key,alias in (('protocol','protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):
        require(same(seal[key],bindings[alias]),'sealed current byte join differs '+key)
    require(check['schema']=='supplemental_endpoint_observability_scalar_checks_v11' and check['complete'] is False and
        check['passed'] is False and check['checks']==95 and check['failure_count']==0 and check['exception']['type']=='KeyError' and
        check['exception']['message']=="'best_candidate'" and check['inputs']==own['inputs'] and
        same(check['own_protocol'],bindings['geometry_validation_protocol']) and same(check['checker'],bindings['geometry_checker']),
        'preserved specific original inspector failure required')
    for alias,key in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','protocol'),('geometry_seal','geometry_seal'),
        ('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','geometry_checker')):
        require(same(own['inputs'][alias],bindings[key]),'actual checked input bytes differ '+alias)
    review,review_own,review_started,authority=[read(paths[k]) for k in
        ('geometry_review','geometry_review_protocol','geometry_review_started','scoring_authority')]
    supplement=dict(original_checker_changed=False,numerical_core_or_saved_rows_changed=False,AST_optional_best_candidate_access_changes=1)
    require(review['schema']=='supplemental_endpoint_observability_scalar_checks_v11' and review['complete'] is True and
        review['passed'] is True and review['inputs']==review_own['inputs'] and review['review']==review_own['review']==supplement and
        same(review['own_protocol'],bindings['geometry_review_protocol']) and same(review['checker'],bindings['geometry_checker']) and
        same(review['review_checker'],bindings['geometry_review_checker']),'separate full geometry review PASS required')
    require(set(review_own['inputs'])==set(own['inputs'])|{'review_checker','original_failure','original_validation_protocol'} and
        all(review_own['inputs'][k]==v for k,v in own['inputs'].items()),'review changed original full checker inputs')
    for alias,key in (('review_checker','geometry_review_checker'),('original_failure','geometry_validation'),
        ('original_validation_protocol','geometry_validation_protocol')):require(same(review_own['inputs'][alias],bindings[key]),'review supplement bytes differ '+alias)
    require(review_started['inputs']==review_own['inputs'] and same(review_started['own_protocol'],bindings['geometry_review_protocol']) and
        review_started['actual_production_calls']==0,'review STARTED input authority differs')
    require(authority['schema']=='preserved_inspector_failure_joined_review_authority_v11' and
        authority['numerical_core_or_saved_rows_changed'] is False and
        authority['GT_access_only_after_complete_geometry_cleanup_and_full_review_PASS'] is True,'post-review scoring authority required')
    for key,alias in (('wrapper','scoring_wrapper'),('review_protocol','geometry_review_protocol'),('review_PASS','geometry_review'),
        ('original_failure','geometry_validation'),('original_score_kernel','own_code')):require(same(authority[key],bindings[alias]),'scoring authority byte join differs '+key)
    require(same(read(paths['scoring_started'])['independent_validation'],bindings['geometry_review_protocol']),
        'scorer must use actual review PASS rather than preserved original failure')
    for key,value in stats['inputs'].items():require(same(value,bindings[key]),'METRICS authority byte join differs '+key)
    started=read(paths['statistics_started'])
    require(started['inputs']==stats['inputs'] and started['configured_new_rows']==490 and started['new_seed_draws']==0,
        'statistics inputs were not fixed before the producing arithmetic')
    expected=dict(scored_rows=1470,new_rows=490,posthoc_rows=490,methods=6,strata=3,moment_distributions=162,
        moment_scalars=972,paired_groups=45,metric_CIs=135,new_bootstrap_draws=0)
    require(stats['actual_arithmetic']==expected,'fixed scalar and CI population differs')
    require(same(core['inputs']['parent:protection'],bindings['protected_originals']),
        'main-frozen existing tracked evidence protection bytes differ')
    protected=read(paths['protected_originals'])
    require(isinstance(protected.get('tracked',{}).get('files'),list),
        'existing protection uses wrapped tracked.files record list')
    tracked={value['path']:value for value in protected['tracked']['files']}
    require(len(tracked)==len(protected['tracked']['files']),'duplicate protected original path')
    for key in ('fixed_scores','old_local_scores','point_scores'):
        value=tracked.get(bindings[key]['path'])
        require(isinstance(value,dict) and same(value,bindings[key]),'unprotected original scored evidence '+key)
    for key in ('cohort','bootstrap','statistics_kernel','own_code'):
        require(any(same(value,bindings[key]) and value['path']==bindings[key]['path'] for value in core['inputs'].values()),'unfrozen original public input '+key)
    return stats


def guard(args,stage):
    for p in (args.input,args.output,args.protocol):require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink ancestry')
    out=args.output.resolve();require(out==DOC.resolve() or out.is_relative_to(PRIVATE.resolve()),'new exclusive V11 output required')
    require(out.is_dir(),'existing output directory required')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not (out/name).exists(),'preserve '+name)
    return out


def freeze(args):
    out=guard(args,'freeze');paths=inputs(args);b={k:binding(p) for k,p in paths.items()};metadata(paths,b);_module,info=extraction()
    write(out/FILES[0],dict(schema='independent_repair_statistics_protocol_v11',inputs=b,helpers=info,limits=LIMITS,
        frames=245,methods=list(METHODS),contrasts=[list(p) for p in CONTRASTS],scopes=list(SCOPES),
        moment_scalars=972,paired_groups=45,metric_CI_slots=135,existing_sessions=13,existing_draws=10000,new_draws=0,
        frozen_after_full_geometry_cleanup_PASS_scoring_statistics_before_own_arithmetic=True))
    print('REPAIR_STATISTICS_VERIFICATION_FROZEN',flush=True)


def run(args):
    out=guard(args,'run');lock=read(out/FILES[0]);lb=binding(out/FILES[0]);paths=inputs(args);b={k:binding(p) for k,p in paths.items()}
    module,info=extraction();require(lock['schema']=='independent_repair_statistics_protocol_v11' and lock['inputs']==b and lock['helpers']==info and lock['limits']==LIMITS,'own frozen scalar inputs differ')
    stats=metadata(paths,b);write(out/FILES[1],dict(protocol=lb,actual_saved_arithmetic_attempt=1,new_model_GT_fit_draw_calls=0))
    env=dict(REPO=REPO,Path=Path,math=math,json=json,gzip=gzip,hashlib=hashlib,ABS_TOL=1e-9,REL_TOL=1e-12)
    exec(compile(module,str(SOURCE)+':unchanged_independent_scalar_AST','exec'),env)
    audit=env['Audit']();counts=Counter();error=None;began=time.monotonic()
    try:
        cohort=read(paths['cohort']);ids=cohort['ids'];frames={f['id']:f for f in cohort['frames']}
        audit.eq(len(ids),245,'full245');audit.eq(len(set(ids)),245,'unique245');audit.eq([f['id'] for f in cohort['frames']],ids,'original_order')
        audit.eq(dict(Counter(f['label'] for f in frames.values())),dict(clean=153,moderate=92),'difficulty_scope')
        with gzip.open(paths['bootstrap'],'rt',encoding='utf-8') as f:draws_meta=json.load(f)
        sessions,draws=draws_meta['sessions'],draws_meta['counts'];audit.eq(len(set(sessions)),13,'sessions');audit.eq(len(sessions),13,'session_width');audit.eq(len(draws),10000,'old_draws')
        for d in draws:require(len(d)==13 and all(type(v) is int and v>=0 for v in d) and sum(d)==13,'unchanged draw multiplicities')
        index={s:i for i,s in enumerate(sessions)};groups={m:{} for m in METHODS}
        for key,n,allowed in (('new_scores',490,set(METHODS[-2:])),('old_local_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2]))):
            loaded=0
            for row in env['compact_rows'](paths[key]):
                require(row['method'] in allowed and row['id'] in frames and row['id'] not in groups[row['method']],'unknown/duplicate saved method/ID')
                audit.eq(row['session'],frames[row['id']]['session'],'saved_session');require(row['session'] in index,'unknown session')
                groups[row['method']][row['id']]=row;loaded+=1;counts['saved_scored_rows_streamed']+=1
            audit.eq(loaded,n,'population.'+key)
        for m,g in groups.items():audit.eq(set(g),set(ids),'same245.'+m)
        for name,selected in (('combined',ids),('easy',[i for i in ids if frames[i]['label']=='clean']),('medium',[i for i in ids if frames[i]['label']=='moderate'])):
            block=stats['strata'][name];audit.eq(block['frames'],len(selected),name+'.frames');audit.eq(block['ids'],selected,name+'.ids')
            audit.eq(set(block['methods']),set(METHODS),name+'.methods');audit.eq(set(block['contrasts']),{a+'_minus_'+c for a,c in CONTRASTS},name+'.contrasts')
            for m in METHODS:
                data=[groups[m][i] for i in selected];available=[r for r in data if r['available']];new=[r for r in available if r['new']];fallback=[r for r in available if r['fallback']];stored=block['methods'][m]
                values=dict(denominator=len(data),operational=len(available),new_pose=len(new),fallback=len(fallback),no_pose=len(data)-len(available),statuses=dict(Counter(r['status'] for r in data)),fixed_control=sum(r['status']=='FRESH_FIXED_CONTROL' for r in data))
                for key,value in values.items():audit.eq(stored[key],value,name+'.'+m+'.'+key)
                for scope,chosen in (('operational',available),('new_pose',new),('fallback',fallback)):
                    for metric in METRICS:
                        expected=env['distribution']([r['values'][metric] for r in chosen]);actual=stored['metrics'][scope][metric];audit.eq(actual['n'],expected['n'],'moment_n')
                        for key in ('mean','sample_variance','sample_std','median','P90','max'):audit.close(actual[key],expected[key],'moment.'+key);counts['moment_scalars_checked']+=1
                        counts['distributions_checked']+=1
            for a,c in CONTRASTS:
                candidate,control=groups[a],groups[c]
                for scope in SCOPES:
                    pair=[i for i in selected if candidate[i]['available'] and control[i]['available'] and (scope=='common_operational' or candidate[i]['new']) and (scope!='both_new_pose' or control[i]['new'])]
                    multiplicity=[sum(candidate[i]['session']==s for i in pair) for s in sessions];den=[sum(d[j]*multiplicity[j] for j in range(13)) for d in draws];nonempty=sum(v>0 for v in den)
                    actual=block['contrasts'][a+'_minus_'+c][scope]
                    expected=dict(scope=scope,denominator=len(selected),common_frames=len(pair),pair_ids=pair,candidate_new_pose_in_pairs=sum(candidate[i]['new'] for i in pair),comparator_new_pose_in_pairs=sum(control[i]['new'] for i in pair),bootstrap_nonempty_resamples=nonempty,bootstrap_empty_resamples=10000-nonempty)
                    for key,value in expected.items():audit.eq(actual[key],value,'paired.'+key)
                    for metric in METRICS:
                        delta=[candidate[i]['values'][metric]-control[i]['values'][metric] for i in pair];sums=[math.fsum(v for i,v in zip(pair,delta) if candidate[i]['session']==s) for s in sessions]
                        samples=[math.fsum(d[j]*sums[j] for j in range(13))/v for d,v in zip(draws,den) if v>0];interval=[env['quantile'](samples,.025),env['quantile'](samples,.975)] if samples else None;record=actual['metrics'][metric]
                        audit.eq(record['n'],len(pair),'paired_n');audit.eq(record['bootstrap_nonempty_resamples'],nonempty,'paired_nonempty');audit.close(record['mean_delta'],env['distribution'](delta)['mean'],'paired_mean')
                        if interval is None:audit.eq(record['CI95'],None,'empty_CI');counts['None_CI_slots']+=1
                        else:
                            require(isinstance(record['CI95'],list) and len(record['CI95'])==2,'CI shape')
                            for x,y in zip(record['CI95'],interval):audit.close(x,y,'CI_limit')
                            counts['nonempty_CI_slots']+=1
                        counts['metric_CI_slots_checked']+=1;counts['paired_delta_scalars_computed']+=len(delta);counts['nonempty_resampled_means_computed']+=len(samples)
                    counts['pair_groups_checked']+=1;counts['resample_denominators_computed']+=10000
        for key,value in dict(saved_scored_rows_streamed=1470,moment_scalars_checked=972,distributions_checked=162,pair_groups_checked=45,metric_CI_slots_checked=135).items():audit.eq(counts[key],value,'complete.'+key)
        primary=stats['strata']['combined']['contrasts']['FILTERED_LOCAL_WEAK_SAFE_minus_N3_SUBPIX']['common_operational']
        audit.eq(stats['primary_both_T_R_means_better_than_N3_on_full245'],primary['common_frames']==245 and all(primary['metrics'][m]['mean_delta']<0 for m in METRICS[:2]),'saved_verdict')
        audit.eq({k:binding(p) for k,p in paths.items()},b,'input_bytes_unchanged');audit.eq(binding(out/FILES[0]),lb,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None
    write(out/FILES[2],dict(schema='independent_repair_statistics_checks_v11',complete=passed,passed=passed,protocol=lb,
        checker=b['checker'],inputs=b,helpers=info,limits=LIMITS,actual_counts=dict(counts),exact_checks=audit.exact_checks,
        numeric_checks=audit.numeric_checks,max_numeric_difference=audit.max_difference,error=error,
        failure_count=int(error is not None),first_failure_preserved=True,new_draws_generated=0,wall_seconds=time.monotonic()-began))
    print('REPAIR_STATISTICS_VERIFICATION',passed,dict(counts),flush=True)
    if not passed:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name,default in (('input',DOC),('output',DOC),('protocol',DOC/'PROTOCOL.json')):p.add_argument('--'+name,type=Path,default=default)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
