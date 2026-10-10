"""Postseal counts and ID joins from existing endpoint/LOCAL witnesses.

Standard library only. Freeze this supplement before its scalar work. No new
fit, score, Jacobian, SVD, observation filter, outcome or uncertainty threshold.
Optional pose comparisons copy already saved V9 posthoc deltas and categories.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'
PRIVATE=Path('/tmp/pallet-unused-line-endpoint-audit-private-20261011-v10')
V9=REPO/'_docs/experiments/pallet_sparse_local_line_20261010_v9'
V1=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
CONTRACT=DOC/'POSTSEAL_DIAGNOSTICS_CONTRACT_KO.md'
NAMES=('DIAGNOSTIC_SUMMARY_PROTOCOL.json','DIAGNOSTIC_SUMMARY_STARTED.json',
    'DIAGNOSTIC_SUMMARY.json','DIAGNOSTIC_FRAME_ROWS.jsonl.gz',
    'PENDING_DIAGNOSTIC_FRAME_ROWS.jsonl.gz','INTERRUPTED_DIAGNOSTIC_FRAME_ROWS.jsonl.gz')
DECISIONS=('SUPPORTED_RETAIN','CONTRADICTED_REJECT','UNVERIFIED_RETAIN')
OUTCOMES=('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE')
GROUPS=('ALL','V9_NUMERIC_NEW','V9_BASELINE_FALLBACK','HAS_UNUSED_LINES','NO_UNUSED_LINES',
    'UNDER4_POINTS_WITH_LINES','ANY_CHECKABLE_ENDPOINT','ANY_CONTRADICTED_ENDPOINT',
    'ALL_ENDPOINTS_SUPPORTED','ANY_UNKNOWN_ENDPOINT','ALL_ENDPOINTS_UNKNOWN',
    'BEST_OBSERVED_WEAK_EXISTING_V1','BEST_MODELED_WEAK_EXISTING_V1',
    'BEST_OBSERVED_RANK6_WEAK_EXISTING_V1','NO_BEST_CANDIDATE')
POLICY=dict(schema='postseal_fixed_unused_line_diagnostic_counts_policy_v10',frames=245,
    group_names=list(GROUPS),strata=['combined','easy','medium'],
    weak_condition='unchanged V1 s_min < s_max * 1e-4, evaluated on copied stored singular values',
    weak_condition_is_diagnostic_only=True,new_uncertainty_or_performance_threshold=False,
    stored_rank_condition_softcost_gaps_start_status_copied=True,
    new_Jacobian_SVD_fit_GT_score_training_RGB_calls=0,
    no_original_NEW_fallback_pose_observation_or_mask_change=True,
    optional_pose_outcomes='copy immutable V9 posthoc deltas/category only; no recomputation',
    saved_scored_rows=980,saved_LOCAL_posthoc_rows=245,
    optional_outcome_population='V9 statistics producer writes LOCAL245 only; scored980 live in three separate files',
    no_new_moments_CI_draws_or_latency=True,physical_ownership_truth='UNKNOWN',
    known_DEV_evidence=True,no_causal_line_or_method_improvement_claim=True,automatic_retry=False)


def require(value,reason):
    if not value:raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()),'blank saved row')
            yield json.loads(line)


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def semantic(value):
    return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def binding(path):
    p=Path(path).absolute()
    require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,
        sha256=h.hexdigest(),bytes=p.stat().st_size)


def same(a,b):return all(a[k]==b[k] for k in ('sha256','bytes'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(safe(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def weak_origin():
    tree=ast.parse(V1.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HypothesisBank')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_jacobian')
    expected=ast.parse('bool(s[-1] < s[0] * 1e-4)',mode='eval').body
    expressions=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.keyword) and n.arg=='weak_condition':expressions.append(n.value)
    require(len(expressions)==1 and ast.dump(expressions[0],include_attributes=False)==ast.dump(expected,include_attributes=False),
        'existing V1 weak_condition source expression differs; no substitute threshold')
    return dict(code=binding(V1),function='HypothesisBank._jacobian',
        function_AST_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),
        expression_AST_sha256=hashlib.sha256(ast.dump(expected,include_attributes=False).encode()).hexdigest(),
        expression='bool(s[-1] < s[0] * 1e-4)',diagnostic_only=True)


def input_paths(args):
    paths={name:args.input/name for name in ('AUDIT_PROTOCOL.json','AUDIT_SEAL.json','AUDIT_RECEIPT.json',
        'AUDIT_ROWS.jsonl.gz','AUDIT_LEDGERS.jsonl.gz','PARENT_POPULATION_CHECKS.json',
        'VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json')}
    paths.update(code=Path(__file__).resolve(),contract=CONTRACT,cohort=args.cohort,weak_origin=V1,
        audit_code=Path(__file__).with_name('audit.py'),audit_checker=Path(__file__).with_name('validation_checks.py'),
        v9_geometry=args.v9_input/'GEOMETRY_SEALED.jsonl.gz',v9_seal=args.v9_input/'GEOMETRY_SEAL.json')
    if args.include_saved_outcomes:
        paths.update(saved_pose_outcomes=args.v9_input/'POSTHOC_ROWS.jsonl.gz',
            saved_pose_outcome_validation=args.v9_input/'VERIFICATION.json',
            saved_pose_outcome_statistics=args.v9_input/'METRICS.json',
            saved_pose_outcome_public_review=args.v9_input/'PUBLIC_REVIEW_CHECKS.json',
            saved_pose_outcome_producer=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/statistics.py')
    return paths


def completed(paths,bindings):
    seal,receipt,validation,own,pop=[read(paths[name]) for name in ('AUDIT_SEAL.json','AUDIT_RECEIPT.json',
        'VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','PARENT_POPULATION_CHECKS.json')]
    require(seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and
        seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,
        'complete full245 GT-free audit seal required before posthoc counts')
    require(receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_error'] is None and
        receipt['environment_and_monkeypatch_cleanup_completed'] is True,'successful audit cleanup required')
    require(validation['schema']=='supplemental_unused_line_endpoint_scalar_checks_v10' and
        validation['complete'] is True and validation['passed'] is True and validation['inputs']==own['inputs'],
        'own independent endpoint scalar PASS required')
    require(same(validation['own_protocol'],bindings['VALIDATION_PROTOCOL.json']) and
        same(validation['checker'],bindings['audit_checker']),'actual successful checker bytes differ')
    for name,key,alias in (('AUDIT_PROTOCOL.json','protocol','audit_protocol'),('AUDIT_ROWS.jsonl.gz','audit','audit_rows'),
        ('AUDIT_LEDGERS.jsonl.gz','ledgers','audit_ledgers'),('AUDIT_RECEIPT.json','receipt','audit_receipt'),
        ('PARENT_POPULATION_CHECKS.json','parent_population_checks','parent_population_checks')):
        require(same(seal[key],bindings[name]) and same(own['inputs'][alias],bindings[name]),'postseal checked byte join '+name)
    require(same(own['inputs']['audit_seal'],bindings['AUDIT_SEAL.json']) and
        same(own['inputs']['audit_code'],bindings['audit_code']),'checked audit source/seal differs')
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245,'complete unchanged parent population')
    original=read(paths['v9_seal']);require(original['complete'] is True and same(original['geometry'],bindings['v9_geometry']),
        'immutable V9 raw geometry byte join differs')
    require(same(own['inputs']['v9:GEOMETRY_SEALED.jsonl.gz'],bindings['v9_geometry']),'V9 copy belongs to another audit')
    if 'saved_pose_outcomes' in paths:
        check=read(paths['saved_pose_outcome_validation'])
        require(check['passed'] is True and check['complete'] is True,'existing V9 statistic verification PASS required')
        stats=read(paths['saved_pose_outcome_statistics']);public=read(paths['saved_pose_outcome_public_review'])
        require(stats['complete'] is True and same(stats['posthoc_rows'],bindings['saved_pose_outcomes']) and
            same(check['inputs']['METRICS.json'],bindings['saved_pose_outcome_statistics']),
            'saved posthoc/statistic arithmetic authority byte join differs')
        require(public['complete'] is True and public['passed'] is True and
            same(public['inputs']['POSTHOC_ROWS.jsonl.gz'],bindings['saved_pose_outcomes']),
            'saved full posthoc population public review byte join differs')
        arithmetic=stats['actual_arithmetic']
        require(all(arithmetic[key]==expected for key,expected in
            (('saved_scored_rows',980),('local_rows',245),('point_control_rows',245),
             ('fixed_rows',490),('posthoc_rows',245))),
            'V9 scored980 and LOCAL-only posthoc245 populations must remain distinct')
        require(same(stats['bindings']['statistics_code'],bindings['saved_pose_outcome_producer']),
            'saved LOCAL posthoc producer source bytes differ')
        # This supplemental source is saved posthoc evidence, not a fit input.
        # Its current bytes are frozen below before any copied-delta counting.
    return pop


def guard(args,stage):
    for p in (args.input,args.output,args.v9_input,args.cohort):
        p=p.absolute();require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink ancestry')
    root=args.output.resolve()
    require(root==DOC.resolve() or root.is_relative_to(PRIVATE.resolve()),'new exclusive V10 DOC/private output only')
    require(args.input.is_dir() and args.v9_input.is_dir() and args.output.is_dir(),'existing directories required')
    require(not root.is_relative_to(args.v9_input.resolve()) and not args.v9_input.resolve().is_relative_to(root),
        'new output overlaps immutable V9 evidence')
    for name in NAMES if stage=='freeze' else NAMES[1:]:require(not (root/name).exists(),'preserve '+name)
    return root


def freeze(args):
    output=guard(args,'freeze');paths=input_paths(args);bindings={k:binding(p) for k,p in paths.items()}
    completed(paths,bindings);origin=weak_origin()
    write(output/NAMES[0],dict(schema='postseal_unused_line_diagnostic_summary_protocol_v10',
        policy=POLICY,inputs=bindings,weak_condition_origin=origin,include_saved_outcomes=args.include_saved_outcomes,
        frozen_after_independent_audit_PASS_before_own_scalar_counts=True,
        saved_outcomes_decoded_only_after_new_audit_seal_and_own_freeze=True,new_model_fit_score_calls=0))
    print('UNUSED_LINE_DIAGNOSTICS_FROZEN',flush=True)


def geometry_copy(value):
    if value is None:return None
    require(isinstance(value,dict) and isinstance(value.get('singular_values'),list) and
        all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in value['singular_values']),
        'finite stored singular-value witness required')
    s=value['singular_values']
    return dict(stored_witness=copy.deepcopy(value),
        existing_V1_weak_condition=None if not s else s[-1]<s[0]*1e-4,
        rank_not_recomputed=True,condition_not_recomputed=True,diagnostic_only=True)


def candidate_copy(value):
    if value is None:return None
    geom=value['point_line_geometry']
    return dict(saved_fields={k:copy.deepcopy(value.get(k)) for k in ('candidate_index','start_index','source',
        'dimension_index','centroid','soft_l1_cost','accepted','rejected','final_inliers','final_line_inliers',
        'line_residual_RMS_px','diagnostic_inlier_support_rank6')},
        observed=geometry_copy(geom.get('observed_normal_joint')),
        modeled=geometry_copy(geom.get('modeled_normal_joint')))


def summarize(row,local,ledger,outcome,label):
    require(row['id']==local['id']==ledger['id'] and row['session']==local['session']==ledger['session'],
        'audit/V9/ledger frame join differs')
    require(row['local_geometry_row_semantic_sha256']==semantic(local),'immutable full V9 row copy differs')
    require(row['original_V9_pose_and_status_preserved'] is True and row['no_final_observation_filter_applied'] is True,
        'diagnostic cannot be a filtered final fit')
    s=local['solver'];best=candidate_copy(s.get('best_candidate'));records=row['endpoint_records']
    decisions=Counter(r['decision'] for r in records);require(set(decisions)<=set(DECISIONS),'unknown endpoint label')
    require([r['edge'] for r in records]==row['original_actual_unused_edges']==s['line_edges'],
        'not the same actual V9 unused line pool')
    require(ledger['endpoint_validation_requests']==len(records),'endpoint count vs ledger differs')
    observed=best['observed'] if best else None;modeled=best['modeled'] if best else None
    ow=bool(observed and observed['existing_V1_weak_condition']);mw=bool(modeled and modeled['existing_V1_weak_condition'])
    flags=dict(ALL=True,V9_NUMERIC_NEW=local['new_pose_estimated'],V9_BASELINE_FALLBACK=local['fallback_used'],
        HAS_UNUSED_LINES=bool(records),NO_UNUSED_LINES=not records,
        UNDER4_POINTS_WITH_LINES=len(s['used'])<4 and bool(records),
        ANY_CHECKABLE_ENDPOINT=bool(decisions['SUPPORTED_RETAIN']+decisions['CONTRADICTED_REJECT']),
        ANY_CONTRADICTED_ENDPOINT=bool(decisions['CONTRADICTED_REJECT']),
        ALL_ENDPOINTS_SUPPORTED=bool(records) and decisions['SUPPORTED_RETAIN']==len(records),
        ANY_UNKNOWN_ENDPOINT=bool(decisions['UNVERIFIED_RETAIN']),
        ALL_ENDPOINTS_UNKNOWN=bool(records) and decisions['UNVERIFIED_RETAIN']==len(records),
        BEST_OBSERVED_WEAK_EXISTING_V1=ow,BEST_MODELED_WEAK_EXISTING_V1=mw,
        BEST_OBSERVED_RANK6_WEAK_EXISTING_V1=ow and observed['stored_witness']['numerical_rank']==6,
        NO_BEST_CANDIDATE=best is None)
    require(set(flags)==set(GROUPS) and all(type(v) is bool for v in flags.values()),'fixed diagnostic-only bool groups')
    saved=None
    if outcome is not None:
        require(outcome['id']==row['id'] and outcome['session']==row['session'] and
            outcome['method']=='ROLE_BOUNDARY_LOCAL_POINT_LINE','stored posthoc outcome ID/method differs')
        d=outcome['local_point_line_diagnostics']
        require(d['actual_point_ids']==s['used'] and d['actual_unused_line_edges']==s['line_edges'] and
            d['numeric_LOCAL_NEW']==local['new_pose_estimated'],'saved outcome comes from another factor pool/status')
        saved={key:copy.deepcopy(d[key]) for key in ('actual_pose_vs_N3','actual_pose_vs_point')}
        require(all(saved[k]['category'] in OUTCOMES for k in saved),'unknown saved pose category')
    return dict(schema='postseal_unused_line_diagnostic_frame_v10',id=row['id'],session=row['session'],label=label,
        audit_row_semantic_sha256=semantic(row),original_V9_row_semantic_sha256=semantic(local),
        original_V9_status=local['output_status'],original_V9_state=s['state'],
        original_V9_NEW=local['new_pose_estimated'],original_V9_fallback=local['fallback_used'],
        original_H=copy.deepcopy(row['initial_H']),actual_sparse_used_ids=copy.deepcopy(s['used']),
        actual_unused_line_edges=copy.deepcopy(s['line_edges']),diagnostic_flags=flags,
        endpoint_diagnoses=[{key:copy.deepcopy(r[key]) for key in ('edge','decision','reason','endpoint_RMS_px',
            'validation_excluded','physical_ownership_certified','validation_pose_may_be_wrong_consensus')} for r in records],
        best_candidate=best,all_saved_candidates=[candidate_copy(c) for c in s['all_candidate_solutions']],
        stored_alternatives=[{key:copy.deepcopy(c.get(key)) for key in ('candidate_index','dimension_index',
            'numerically_equivalent_to_selected','numerical_objective_tie','objective_difference_from_selected_px2')} for c in s['alternatives']],
        original_local_start_cache=copy.deepcopy(s.get('local_start_cache')),
        copied_saved_pose_outcomes=saved,source_outcome_semantic_sha256=semantic(outcome) if outcome is not None else None,
        new_Jacobian_SVD_optimizer_GT_score_calls=0,original_pose_or_observation_modified=False,
        physical_ownership_truth='UNKNOWN',numeric_NEW_is_not_goal_success=True)


def run(args):
    output=guard(args,'run');own=read(output/NAMES[0]);own_binding=binding(output/NAMES[0])
    paths=input_paths(args);bindings={k:binding(p) for k,p in paths.items()}
    require(own['schema']=='postseal_unused_line_diagnostic_summary_protocol_v10' and own['policy']==POLICY and
        own['inputs']==bindings and own['weak_condition_origin']==weak_origin() and
        own['include_saved_outcomes']==args.include_saved_outcomes,'own frozen scalar diagnostic source/policy differs')
    population=completed(paths,bindings);cohort=read(paths['cohort']);ordered=cohort['ids']
    require(ordered==population['ordered_ids'] and len(ordered)==len(set(ordered))==245 and
        [f['id'] for f in cohort['frames']]==ordered and all(f['label'] in ('clean','moderate') for f in cohort['frames']),
        'full original easy-medium cohort required')
    write(output/NAMES[1],dict(schema='postseal_unused_line_diagnostic_summary_started_v10',
        protocol=binding(output/NAMES[0]),configured_frames=245,new_model_fit_score_calls=0,automatic_retry=False))
    began=time.monotonic();counts=Counter();groups={stratum:{name:dict(ids=[],original_status=Counter(),
        endpoint_decisions=Counter(),saved_N3_outcomes=Counter(),saved_point_outcomes=Counter()) for name in GROUPS}
        for stratum in POLICY['strata']};error=None;raw=None;zipped=None;cleanup_errors=[]
    streams=[iter(rows(paths[k])) for k in ('AUDIT_ROWS.jsonl.gz','v9_geometry','AUDIT_LEDGERS.jsonl.gz')]
    outcomes=iter(rows(paths['saved_pose_outcomes'])) if args.include_saved_outcomes else None
    try:
        raw=(output/NAMES[4]).open('xb');zipped=gzip.GzipFile(filename=NAMES[3],fileobj=raw,mode='wb',mtime=0)
        for i,fid in enumerate(ordered):
            joined=[next(stream,None) for stream in streams];require(all(isinstance(r,dict) for r in joined),'incomplete audit/V9 join')
            require(all(r['id']==fid for r in joined),'saved populations reordered')
            outcome=next(outcomes,None) if outcomes is not None else None
            require(outcomes is None or isinstance(outcome,dict),'missing optional saved outcome')
            if outcome is not None:counts['optional_saved_pose_outcomes_visited']+=1
            value=summarize(*joined,outcome,cohort['frames'][i]['label'])
            strata=('combined','easy' if value['label']=='clean' else 'medium')
            for stratum in strata:
                for name in GROUPS:
                    if not value['diagnostic_flags'][name]:continue
                    group=groups[stratum][name];group['ids'].append(fid);group['original_status'][value['original_V9_status']]+=1
                    group['endpoint_decisions'].update(r['decision'] for r in value['endpoint_diagnoses'])
                    if value['copied_saved_pose_outcomes'] is not None:
                        group['saved_N3_outcomes'][value['copied_saved_pose_outcomes']['actual_pose_vs_N3']['category']]+=1
                        group['saved_point_outcomes'][value['copied_saved_pose_outcomes']['actual_pose_vs_point']['category']]+=1
            zipped.write((json.dumps(safe(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
            counts['frames_complete']+=1;counts['endpoint_records']+=len(value['endpoint_diagnoses'])
            counts['stored_LOCAL_candidate_records_copied']+=len(value['all_saved_candidates'])
            counts['optional_saved_pose_outcomes_copied']+=int(outcome is not None)
        for stream in streams:require(next(stream,None) is None,'extra saved rows outside245')
        if outcomes is not None:require(next(outcomes,None) is None,'extra saved outcomes outside245')
        require(outcomes is None or counts['optional_saved_pose_outcomes_visited']==
            counts['optional_saved_pose_outcomes_copied']==245,'complete LOCAL-only245 outcome stream required')
        require({k:binding(p) for k,p in paths.items()}==bindings,'input bytes changed during scalar join')
        require(binding(output/NAMES[0])==own_binding,'own scalar protocol bytes changed during scalar join')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        actions=[]
        if zipped is not None:actions.append(('gzip_close',zipped.close))
        if raw is not None:actions.extend((('raw_flush',raw.flush),
            ('raw_fsync',lambda:os.fsync(raw.fileno())),('raw_close',raw.close)))
        for phase,action in actions:
            try:action()
            except BaseException as exc:cleanup_errors.append(dict(type=type(exc).__name__,message=str(exc),phase=phase))
    complete=error is None and not cleanup_errors and counts['frames_complete']==245
    target=output/(NAMES[3] if complete else NAMES[5])
    if (output/NAMES[4]).exists():
        try:
            os.link(output/NAMES[4],target)
            if complete:(output/NAMES[4]).unlink()
            descriptor=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        except BaseException as exc:complete=False;error=dict(type=type(exc).__name__,message=str(exc),phase='exclusive_preservation',original_error=error)
    for stratum in groups:
        for name in groups[stratum]:
            item=groups[stratum][name];item['frames']=len(item['ids'])
    result=dict(schema='postseal_unused_line_diagnostic_summary_v10',complete=complete,passed=complete,
        protocol=own_binding,inputs=bindings,policy=POLICY,weak_condition_origin=own['weak_condition_origin'],
        counts=dict(counts),groups=groups,include_saved_outcomes=args.include_saved_outcomes,error=error,
        cleanup_errors=cleanup_errors,
        frame_rows=binding(target) if target.is_file() else None,wall_seconds=time.monotonic()-began,
        new_MODEL_RGB_GTscore_PnP_LOCAL_Jacobian_SVD_moment_CI_calls=0,observation_or_NEW_changed=False,
        classification_weak_condition_diagnostic_only=True,causal_improvement_claim=False,
        physical_ownership_certified=False,automatic_retry=False)
    write(output/NAMES[2],result)
    print('UNUSED_LINE_DIAGNOSTICS',complete,dict(counts),flush=True)
    if not complete:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    p.add_argument('--v9-input',type=Path,default=V9);p.add_argument('--cohort',type=Path,default=COHORT)
    p.add_argument('--include-saved-outcomes',action='store_true');args=p.parse_args()
    (freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
