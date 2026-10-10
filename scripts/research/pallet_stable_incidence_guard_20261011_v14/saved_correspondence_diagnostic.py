"""Count saved correspondence/pose evidence; do not compute new pose metrics.

The 8px point labels are existing geometric-proxy labels, not physical wire
ownership certificates. Line covariance/radius values are copied source-CAL
surrogates. No reference file, model, fitting, Jacobian, SVD or draw is opened.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
METHOD='FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE'
FILES=('CORRESPONDENCE_DIAGNOSTIC_PROTOCOL.json','CORRESPONDENCE_DIAGNOSTIC_STARTED.json',
       'CORRESPONDENCE_DIAGNOSTIC_ROWS.jsonl.gz','CORRESPONDENCE_DIAGNOSTIC_CHECKS.json')
QUALITY=('CORRECT_WITHIN8PX','INCORRECT_OVER8PX','UNKNOWN_REFERENCE_OR_INPUT')

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(p.read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:yield json.loads(line)
def binding(p):
    require(p.is_file() and not any(v.is_symlink() for v in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))
def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def inputs():
    names=('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json',
        'GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','POSTHOC_ROWS.jsonl.gz','METRICS.json',
        'VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json','EVALUATION_PROTOCOL.json',
        'SCORING_RECEIPT.json','SCORING_STARTED.json','PREDICTIONS.jsonl.gz','STATISTICS_STARTED.json',
        'SCORING_DISPATCH_FIRST_FAILURE.json','SCORING_DISPATCH_REVIEW_PROTOCOL.json',
        'SCORING_DISPATCH_REVIEW_STARTED.json','SCORING_AUTHORITY_SUPPLEMENT.json','STATISTICS_AUTHORITY_SUPPLEMENT.json')
    paths={n:DOC/n for n in names};paths['code']=Path(__file__).resolve()
    paths['guard_code']=CODE/'pipeline.py';paths['posthoc_producer']=CODE/'evaluation.py';paths['checker']=CODE/'validation_checks.py'
    paths['scoring_review_code']=CODE/'evaluation_review.py'
    return {k:binding(p) for k,p in paths.items()}
def authority(b):
    seal,check,metrics,score=[read(DOC/n) for n in ('GEOMETRY_SEAL.json','VALIDATION_CHECKS.json','METRICS.json','SCORING_RECEIPT.json')]
    require(seal['complete'] is True and seal['rows']==seal['frames']==seal['ledger_rows']==245
        and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True
        and check['complete'] is True and check['passed'] is True and check['checks']==42715 and check['failure_count']==0
        and metrics['complete'] is True and metrics['schema']=='stable_incidence_raw_row_metrics_v14'
        and score['complete'] is True and score['scored_rows']==245 and score['methods']==[METHOD],'actual sealed/scored245 and scalar PASS')
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('protocol','PROTOCOL.json'),('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        require(same(seal[key],b[name]),'actual geometry authority '+key)
    require(same(metrics['posthoc_rows'],b['POSTHOC_ROWS.jsonl.gz']),'saved posthoc authority')
    require(check['inputs']==read(DOC/'VALIDATION_PROTOCOL.json')['inputs'] and same(check['own_protocol'],b['VALIDATION_PROTOCOL.json'])
        and same(check['checker'],b['checker']),
        'current independent guard protocol')
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json'),('checker','checker')):
        require(same(check['inputs'][key],b[name]),'current independently checked '+key)
    require(same(score['protocol'],b['PROTOCOL.json']) and same(score['geometry_seal'],b['GEOMETRY_SEAL.json'])
        and same(score['predictions'],b['PREDICTIONS.jsonl.gz']),'current saved pose-score lineage')
    require(metrics['inputs']==read(DOC/'STATISTICS_STARTED.json')['inputs']
        and same(metrics['inputs']['new_scores'],b['PREDICTIONS.jsonl.gz'])
        and same(metrics['inputs']['scoring_receipt'],b['SCORING_RECEIPT.json'])
        and same(metrics['inputs']['geometry_validation'],b['VALIDATION_CHECKS.json'])
        and same(metrics['inputs']['own_code'],b['posthoc_producer']),'current unchanged statistics/posthoc lineage')
    return dict(reference='saved N3-phase GEOMETRIC_PROXY known DEV; not independent physical pose truth',
        point_quality_threshold_is_existing_8px=True,line_covariance_is_source_CAL_surrogate=True,
        line_covariance_is_not_real_ownership_or_calibrated_pose_uncertainty=True,new_GT_metric_fit_J_SVD_model_draw_calls=0)

def evidence(g,p,ledger):
    require(g['id']==p['id']==ledger['id'] and g['method']==p['method']==METHOD,'same selected saved row')
    require(same_semantic(g)==ledger['guarded_row_semantic_sha256'],'checked guarded geometry semantic lineage')
    s=g['solver'];new=g['new_pose_estimated'];fit=list(s['fit_input_ids']) if new else [];final=list(s['final_inliers']) if new else []
    edges=list(s['fit_line_edges']) if new else [];corners={c['id']:c for c in p['corners']};H=g['hidden_initial']
    require(set(corners)==set(range(8)) and set(fit).isdisjoint(H) and p['pools']['accepted_fit']['ids']==fit
        and p['pools']['accepted_final_inliers']['ids']==final,'actual accepted-fit/final/H identity')
    def quality(ids):
        value={q:[] for q in QUALITY}
        for k in ids:
            q=corners[k]['input_quality'];require(q in value,'saved quality label');value[q].append(k)
        return value
    supplied=quality(s['used']);accepted=quality(fit);inliers=quality(final);correct=len(accepted[QUALITY[0]])
    wrong=len(inliers[QUALITY[1]]);allgood=bool(fit) and correct==len(fit)
    outcome=p['pose_outcome_vs_N3'];require(outcome in ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE'),'saved outcome')
    sources={l['edge']:l for l in s.get('actual_source_lines',[])};lines=[]
    for e in edges:
        require(e in sources and e not in s['consumed_edges'],'actual unused edge source')
        raw=sources[e]['raw'];cov=raw.get('parameter_covariance')
        available=isinstance(cov,list) and len(cov)==2 and all(isinstance(v,list) and len(v)==2 for v in cov)
        lines.append(dict(edge=e,actual_source_semantic_sha256=same_semantic(raw),query_ids=raw.get('queries'),
            query_radii_px=raw.get('query_radii_px'),parameter_covariance=cov if available else None,
            covariance_available=available,covariance_is_calibrated_surrogate=raw.get('covariance_is_calibrated_surrogate'),
            support_length_px=raw.get('support_length_px'),residual_rms_px=raw.get('residual_rms_px'),
            standardized_residual_rms=raw.get('standardized_residual_rms'),physical_ownership_truth='UNKNOWN'))
    flags=dict(all_frames=True,NEW=new,fallback=g['fallback_used'],
        supplied_correct_q_ge4=len(supplied[QUALITY[0]])>=4,
        supplied_correct_q_ge4_and_final_bad1=new and len(supplied[QUALITY[0]])>=4 and wrong==1,
        supplied_correct_q_ge4_and_final_bad2=new and len(supplied[QUALITY[0]])>=4 and wrong==2,
        accepted_correct_q_ge4=new and correct>=4,
        final_bad_q_eq1=new and wrong==1,final_bad_q_eq2=new and wrong==2,final_bad_q_gt2=new and wrong>2,
        accepted_correct_q_ge4_and_final_bad1=new and correct>=4 and wrong==1,
        accepted_correct_q_ge4_and_final_bad2=new and correct>=4 and wrong==2,
        accepted_correct_q_lt4_with_lines=new and correct<4 and bool(edges),
        accepted_point_count_lt4_with_lines=new and len(fit)<4 and bool(edges),
        line_only_NEW=new and not fit and bool(edges),point_only_NEW=new and bool(fit) and not edges,
        all_fit_q_known_correct=new and allgood,
        all_fit_q_known_correct_but_both_worse=new and allgood and outcome=='BOTH_WORSE',
        correct_q_ge4_but_both_worse=new and correct>=4 and outcome=='BOTH_WORSE',
        wrong_known_mask_both_better=p['mask_state']=='WRONG_ON_KNOWN' and outcome=='BOTH_BETTER',
        matches_known_mask_both_worse=p['mask_state']=='MATCHES_ON_KNOWN' and outcome=='BOTH_WORSE')
    w=g['stable_incidence_guard'];layout={k:w.get(k) for k in ('gate_applied','guard_state','accepted',
        'actual_fit_input_ids','actual_fit_line_edges','finite_scalar_rows','incidence_scalar_rows','removed_longitudinal_rows',
        'observed_weak','modeled_weak')};geometry=w.get('stable_incidence_geometry')
    if geometry:layout['stored_normal_geometry']=geometry
    return dict(id=g['id'],session=g['session'],output_status=g['output_status'],new_pose_estimated=new,
        fallback_used=g['fallback_used'],fit_point_ids=fit,final_inlier_ids=final,fit_line_edges=edges,H=H,
        supplied_point_quality=supplied,accepted_fit_point_quality=accepted,final_inlier_point_quality=inliers,
        boundary_candidates=p['boundary_candidates'],line_uncertainty_sourcecopies=lines,stable_normal_layout=layout,
        mask_state=p['mask_state'],mask_audit=p['mask_audit'],saved_pose_outcome_vs_N3=outcome,saved_delta_vs_N3=p['delta_vs_N3'],
        group_flags=flags,physical_ownership_certified=False,new_pose_metrics_or_covariance_recomputed=False)
def same_semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    for n in FILES if args.stage=='freeze' else FILES[1:]:require(not(DOC/n).exists(),'preserve '+n)
    b=inputs();limits=authority(b)
    if args.stage=='freeze':
        write(DOC/FILES[0],dict(schema='saved_correspondence_diagnostic_protocol_v14',inputs=b,limits=limits,
            frames=245,frozen_before_own_classification_and_counts=True,method_or_outputs_changed=False));return
    own=read(DOC/FILES[0]);require(own['inputs']==b and own['limits']==limits,'own frozen diagnostic changed')
    write(DOC/FILES[1],dict(protocol=binding(DOC/FILES[0]),inputs=b,new_production_calls=0))
    seen=[];groups={};count=Counter();error=None;cleanup_errors=[];pending=DOC/('PENDING_'+FILES[2]);stream=None;raw=None
    try:
        post=iter(rows(DOC/'POSTHOC_ROWS.jsonl.gz'));ledgers=iter(rows(DOC/'CONTROL_LEDGERS.jsonl.gz'))
        raw=pending.open('xb');stream=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
        for g in rows(DOC/'GEOMETRY_SEALED.jsonl.gz'):
            p,l=next(post),next(ledgers);v=evidence(g,p,l);seen.append(g['id']);count['frames']+=1
            for name,flag in v['group_flags'].items():
                group=groups.setdefault(name,[])
                if flag:group.append(g['id'])
            for line in v['line_uncertainty_sourcecopies']:count['actual_fit_lines']+=1;count['covariance_available']+=line['covariance_available']
            stream.write((json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode())
        require(next(post,None) is None and next(ledgers,None) is None and seen==read(DOC/'PARENT_POPULATION_CHECKS.json')['ordered_ids']
            and len(seen)==245 and inputs()==b,'full ordered245 unchanged saved population')
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:
        if stream:
            try:stream.close()
            except BaseException as e:cleanup_errors.append(dict(type=type(e).__name__,message=str(e),phase='gzip_close'))
        if raw:
            for phase,operation in (('raw_flush',raw.flush),('raw_fsync',lambda:os.fsync(raw.fileno())),('raw_close',raw.close)):
                try:operation()
                except BaseException as e:cleanup_errors.append(dict(type=type(e).__name__,message=str(e),phase=phase))
        error=error or (cleanup_errors[0] if cleanup_errors else None)
        if error is None:
            try:os.link(pending,DOC/FILES[2]);pending.unlink()
            except BaseException as e:error=dict(type=type(e).__name__,message=str(e),phase='exclusive_publish')
        result=dict(schema='saved_correspondence_diagnostic_checks_v14',complete=error is None,passed=error is None,
            own_protocol=binding(DOC/FILES[0]),inputs=b,limits=limits,counts=dict(count),groups={k:dict(count=len(v),ids=v) for k,v in groups.items()},
            error=error,cleanup_errors=cleanup_errors,preserved_prefix=binding(pending) if pending.is_file() else None,
            rows=binding(DOC/FILES[2]) if (DOC/FILES[2]).is_file() else None,
            stored_combined_pose_summary_copy=read(DOC/'METRICS.json')['strata']['combined']['methods'][METHOD],
            new_GT_metric_PnP_optimizer_J_SVD_model_draw_calls=0,all245_frames_preserved=error is None,
            point_proxy_labels_are_not_physical_ownership_certificates=True)
        write(DOC/FILES[3],result)
    require(error is None,'saved diagnostic incomplete; preserve first failure/prefix')
if __name__=='__main__':main()
