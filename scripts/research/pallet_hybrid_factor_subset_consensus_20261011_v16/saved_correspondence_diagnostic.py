"""Read saved paired correspondence evidence; do not produce new poses.

Existing 8px labels use the scored row's fixed N3-phase geometric proxy. Human
visibility is saved metadata. Neither label certifies physical pose or edge
ownership. No controlled point-removal/corruption experiment is performed.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
QUALITY=('CORRECT_WITHIN8PX','INCORRECT_OVER8PX','UNKNOWN_REFERENCE_OR_INPUT')
FACES=dict(near=(0,1,2,3),far=(4,5,6,7),top=(0,1,4,5),bottom=(2,3,6,7),left=(0,3,4,7),right=(1,2,5,6))
FILES=('CORRESPONDENCE_DIAGNOSTIC_PROTOCOL.json','CORRESPONDENCE_DIAGNOSTIC_STARTED.json','CORRESPONDENCE_DIAGNOSTIC_ROWS.jsonl.gz','CORRESPONDENCE_DIAGNOSTIC_CHECKS.json')
LIMITS=dict(new_pose_GT_file_model_forward_training_PnP_optimizer_J_SVD_draw_calls=0,
    physical_ownership_certified=False,proxy_correct_is_not_independent_physical_truth=True,
    controlled_remove_correct1_or2_or_leave_wrong1_or2_experiments_performed=False,
    natural_errors_do_not_establish_controlled_corruption_causality=True,
    known_cuboid_faces_only_not_general_coplanarity_or_physical_boundary=True,
    same_q_line_ablation_includes_changes_in_hypotheses_consensus_and_refits=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(p.read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    require(p.is_file() and not any(v.is_symlink() for v in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def inputs():
    names=('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json','GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz',
        'VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json','EVALUATION_PROTOCOL.json','SCORING_AUTHORITY.json','SCORING_STARTED.json',
        'SCORING_RECEIPT.json','PREDICTIONS.jsonl.gz','STATISTICS_STARTED.json','METRICS.json','POSTHOC_ROWS.jsonl.gz',
        'STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')
    paths={n:DOC/n for n in names};paths.update(code=Path(__file__),checker=CODE/'validation_checks.py',posthoc_producer=CODE/'evaluation.py',verifier=CODE/'verify.py',
        original_saved_semantics=REPO/'scripts/research/pallet_stable_incidence_guard_20261011_v14/saved_correspondence_diagnostic.py',
        existing_weak_policy=REPO/'scripts/research/pallet_stable_incidence_guard_20261011_v14/pipeline.py')
    return {k:binding(p) for k,p in paths.items()}
def authority(b):
    core,seal,receipt,check,own,score,metrics,verify=[read(DOC/n) for n in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','SCORING_RECEIPT.json','METRICS.json','VERIFICATION.json')]
    require(core['schema']=='fixed_hybrid_factor_subset_consensus_protocol_v16' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490,'fixed paired geometry')
    require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True
        and receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,'actual complete restored paired geometry')
    for k,n in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):require(same(seal[k],b[n]),'sealed bytes '+k)
    require(own['schema']=='hybrid_factor_subset_scalar_protocol_v16' and check['schema']=='hybrid_factor_subset_scalar_checks_v16' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['inputs']==own['inputs'] and same(check['own_protocol'],b['VALIDATION_PROTOCOL.json']) and same(check['checker'],b['checker']),'actual geometry scalar PASS')
    for k,n in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json'),('checker','checker')):require(same(own['inputs'][k],b[n]),'actual checked bytes '+k)
    require(score['complete'] is True and score['frames']==245 and score['scored_rows']==490 and score['methods']==list(METHODS) and same(score['predictions'],b['PREDICTIONS.jsonl.gz']) and same(score['protocol'],b['PROTOCOL.json']) and same(score['geometry_seal'],b['GEOMETRY_SEAL.json']),'actual paired score lineage')
    require(metrics['schema']=='hybrid_factor_subset_raw_row_metrics_v16' and metrics['complete'] is True and same(metrics['posthoc_rows'],b['POSTHOC_ROWS.jsonl.gz'])
        and metrics['inputs']==read(DOC/'STATISTICS_STARTED.json')['inputs'] and same(metrics['inputs']['new_scores'],b['PREDICTIONS.jsonl.gz']) and same(metrics['inputs']['own_code'],b['posthoc_producer']),'actual fixed-quality posthoc lineage')
    vp,vs=read(DOC/'STATISTICS_VERIFICATION_PROTOCOL.json'),read(DOC/'STATISTICS_VERIFICATION_STARTED.json')
    require(verify['schema']=='independent_hybrid_factor_subset_statistics_checks_v16' and verify['complete'] is True and verify['passed'] is True and verify['failure_count']==0 and verify['error'] is None
        and verify['inputs']==vp['inputs']==vs['inputs'] and same(verify['protocol'],b['STATISTICS_VERIFICATION_PROTOCOL.json']) and same(verify['checker'],b['verifier'])
        and same(verify['inputs']['METRICS'],b['METRICS.json']) and same(verify['inputs']['new_scores'],b['PREDICTIONS.jsonl.gz'])
        and verify['actual_counts']['saved_scored_rows_streamed']==1470 and verify['actual_counts']['moment_scalars_checked']==972 and verify['actual_counts']['metric_CI_slots_checked']==135,'actual independent saved-statistics PASS')
    return dict(reference='saved N3-phase GEOMETRIC_PROXY known DEV; not independent physical truth',quality_threshold_px=8,
        quality_labels_are_original_saved_posthoc_labels=True,line_uncertainty_is_source_CAL_surrogate=True,
        combinatorial_faces=FACES,limits=LIMITS)
def layout(ids):
    require(len(set(ids))==len(ids) and set(ids)<=set(range(8)),'corner IDs')
    faces=[name for name,v in FACES.items() if ids and set(ids)<=set(v)]
    return dict(ids=list(ids),count=len(ids),contained_in_known_cuboid_faces=faces,
        at_least4_on_one_known_face=len(ids)>=4 and bool(faces),general_coplanarity_not_computed=True,
        off_known_faces_does_not_prove_noncoplanarity=True)
def spectra(g):
    if not g:return None
    out={}
    for k in ('point_only','observed_normal_lines_only','modeled_normal_lines_only','observed_normal_joint','modeled_normal_joint'):
        if k in g:
            w=g[k];s=w['singular_values'];out[k]=dict(w,weak_by_existing_1e4_saved_spectrum=len(s)!=6 or s[-1]<s[0]*1e-4)
    return dict(reports=out,point_ids=g['point_ids'],line_edges=g['line_edges'],new_J_or_SVD_computed=False,local_only=True)
def evidence(g,p):
    require(g['id']==p['id'] and g['session']==p['session'] and g['method']==p['method'] and g['method'] in METHODS,'same scored/posthoc identity')
    s=g['solver'];new=g['new_pose_estimated'];H=list(g['hidden_initial']);fit=list(s['fit_input_ids']) if new else [];final=list(s['final_inliers']) if new else [];edges=list(s['fit_line_edges']) if new else []
    require(set(fit).isdisjoint(H) and p['pools']['accepted_fit']['ids']==fit and p['pools']['accepted_final_inliers']['ids']==final and g['reprojections_reused_as_observations'] is False,'accepted actual fit/H identity')
    ref=g['evaluation_reference'];require(ref['phase_control']=='N3_SUBPIX' and ref['initial_N3_native_points']==g['original_native_N3_points'] and ref['human_states_native']==g['mask_audit']['human_states_native'],'fixed original N3 phase and visibility')
    corners={c['id']:c for c in p['corners']};require(set(corners)==set(range(8)) and g['mask_audit']==p['mask_audit'],'saved corner/mask source identity')
    known=set(ref['valid_native_ids']) if ref['matched'] else set();scalar_labels={};disagreements=[]
    def valid(x):return len(x)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in x) and x!=[-1,-1]
    for k,c in corners.items():
        q,gt=g['input_points'][k],ref['native_points_px'][k];error=math.hypot(q[0]-gt[0],q[1]-gt[1]) if k in known and valid(q) and valid(gt) else None
        require((error is None)==(c['input_error_px'] is None) and (error is None or abs(error-c['input_error_px'])<=1e-9),'saved proxy distance identity')
        scalar_labels[k]='UNKNOWN_REFERENCE_OR_INPUT' if error is None else 'CORRECT_WITHIN8PX' if error<=8 else 'INCORRECT_OVER8PX'
        require(c['input_quality'] in QUALITY,'original saved quality label')
        if scalar_labels[k]!=c['input_quality']:disagreements.append(k)
    def quality(ids):
        result={q:[] for q in QUALITY}
        for k in ids:result[corners[k]['input_quality']].append(k)
        return result
    supplied,accepted,inliers=quality(s['used']),quality(fit),quality(final);correct=len(accepted[QUALITY[0]]);bad=len(inliers[QUALITY[1]]);allgood=bool(fit) and correct==len(fit)
    states=ref['human_states_native'];visibility_known={k for k,v in enumerate(states) if v not in ('UNKNOWN','UNANNOTATED')};humanH={k for k,v in enumerate(states) if v=='SELF_OCCLUDED'}
    mismatch=sorted((set(H)^humanH)&visibility_known);mask='NO_KNOWN_VISIBILITY' if not visibility_known else 'WRONG_ON_KNOWN' if mismatch else 'MATCHES_ON_KNOWN'
    require(mask==p['mask_state'],'natural known-state mask label')
    reproj=list(g['reprojected_ids']);require(reproj==(H if new else []) and (new or g['native_points']==g['original_native_N3_points']),'one H projection or full original fallback')
    if new:
        for k in H:require(g['native_points'][k]==s['projected'][k],'saved H projection assignment')
    outcome=p['pose_outcome_vs_N3'];require(outcome in ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE'),'saved outcome')
    flags=dict(all_frames=True,NEW=new,fallback=g['fallback_used'],no_pose=not g['pose']['available'],supplied_correct_q_ge4=len(supplied[QUALITY[0]])>=4,
        supplied_correct_q_ge4_and_final_bad1=new and len(supplied[QUALITY[0]])>=4 and bad==1,supplied_correct_q_ge4_and_final_bad2=new and len(supplied[QUALITY[0]])>=4 and bad==2,
        accepted_correct_q_ge4=new and correct>=4,final_bad_q_eq1=new and bad==1,final_bad_q_eq2=new and bad==2,final_bad_q_gt2=new and bad>2,
        accepted_correct_q_ge4_and_final_bad1=new and correct>=4 and bad==1,accepted_correct_q_ge4_and_final_bad2=new and correct>=4 and bad==2,
        accepted_correct_q_lt4_with_lines=new and correct<4 and bool(edges),accepted_point_count_lt4_with_lines=new and len(fit)<4 and bool(edges),line_only_NEW=new and not fit and bool(edges),point_only_NEW=new and bool(fit) and not edges,
        all_fit_q_known_correct=new and allgood,all_fit_q_known_correct_but_both_worse=new and allgood and outcome=='BOTH_WORSE',correct_q_ge4_but_both_worse=new and correct>=4 and outcome=='BOTH_WORSE',
        wrong_known_mask_both_better=mask=='WRONG_ON_KNOWN' and outcome=='BOTH_BETTER',matches_known_mask_both_worse=mask=='MATCHES_ON_KNOWN' and outcome=='BOTH_WORSE')
    best=s.get('best_candidate');guard=best.get('stable_incidence_guard') if best else None;sources={l['edge']:l for l in s.get('actual_source_lines',[])};lines=[]
    for e in edges:
        require(e in sources and e not in s['consumed_edges'],'actual unused source edge');raw=sources[e]['raw']
        lines.append(dict(edge=e,source_semantic_sha256=semantic(raw),query_ids=raw.get('queries'),query_radii_px=raw.get('query_radii_px'),parameter_covariance=raw.get('parameter_covariance'),covariance_is_calibrated_surrogate=raw.get('covariance_is_calibrated_surrogate'),physical_ownership_truth='UNKNOWN'))
    return dict(id=g['id'],session=g['session'],method=g['method'],input_points=g['input_points'],selected_boundary_ids=g['selected_corner_ids'],output_status=g['output_status'],new_pose_estimated=new,fallback_used=g['fallback_used'],pose=g['pose'],solver_state=s['state'],
        H=H,reprojected_ids=reproj,fit_point_ids=fit,final_inlier_ids=final,fit_line_edges=edges,supplied_point_quality=supplied,accepted_fit_point_quality=accepted,final_inlier_point_quality=inliers,
        correct_layouts={name:layout(q[QUALITY[0]]) for name,q in (('supplied',supplied),('accepted_fit',accepted),('final_inliers',inliers))},
        natural_mask=dict(state=mask,known_ids=sorted(visibility_known),human_self_hidden_ids=sorted(humanH),mismatch_ids=mismatch,false_self_excluded_known_ids=sorted(set(H)&(visibility_known-humanH)),
            false_self_excluded_direct_visible_ids=[k for k in H if states[k]=='DIRECT_VISIBLE'],missed_self_hidden_ids=sorted(humanH-set(H))),
        saved_pose_outcome_vs_N3=outcome,saved_delta_vs_N3=p['delta_vs_N3'],boundary_candidates=p['boundary_candidates'],line_uncertainty_sourcecopies=lines,
        accepted_finite_geometry=spectra(s.get('point_line_geometry')) if new else None,stored_best_finite_geometry=spectra(best.get('point_line_geometry')) if best else None,
        stored_best_normal_geometry=spectra(guard.get('stable_incidence_geometry')) if guard else None,stored_normal_guard_state=guard['guard_state'] if guard else None,
        selected_dimension_index=best['dimension_index'] if new else None,selected_extents=best['cf_extents'] if new else None,
        stored_best_dimension_index=best['dimension_index'] if best else None,stored_best_extents=best['cf_extents'] if best else None,proxy_reference_semantic_sha256=semantic(ref),
        independent_scalar_quality_labels=scalar_labels,scalar_quality_label_disagreement_ids=disagreements,original_saved_labels_used_for_counts=True,
        group_flags=flags,physical_ownership_certified=False,new_pose_metrics_recomputed=False)
def paired(line,point):
    require(line['id']==point['id'] and line['session']==point['session'] and line['input_points']==point['input_points'] and line['H']==point['H'] and line['selected_boundary_ids']==point['selected_boundary_ids']
        and line['proxy_reference_semantic_sha256']==point['proxy_reference_semantic_sha256'] and line['supplied_point_quality']==point['supplied_point_quality'],'exact same selected-q/proxy line ablation')
    a,b=line['pose'],point['pose'];delta=None
    if a['available'] and b['available']:delta=dict(translation_cm=a['translation_cm']-b['translation_cm'],rotation_deg=a['rotation_deg']-b['rotation_deg'],ADDsym_cm=100*(a['ADDsym_m']-b['ADDsym_m']))
    outcome='POSE_UNAVAILABLE' if delta is None else 'BOTH_BETTER' if delta['translation_cm']<0 and delta['rotation_deg']<0 else 'BOTH_WORSE' if delta['translation_cm']>0 and delta['rotation_deg']>0 else 'MIXED_OR_UNCHANGED'
    return dict(id=line['id'],point_line_status=line['output_status'],point_only_status=point['output_status'],point_line_solver_state=line['solver_state'],point_only_solver_state=point['solver_state'],
        point_line_NEW=line['new_pose_estimated'],point_only_NEW=point['new_pose_estimated'],saved_operational_point_line_minus_point_only=delta,operational_pose_outcome=outcome,
        numeric_line_rescue=line['new_pose_estimated'] and not point['new_pose_estimated'],numeric_point_only_rescue=point['new_pose_estimated'] and not line['new_pose_estimated'],
        no_new_pose_metrics_computed=True,line_ablation_changes_hypotheses_consensus_and_refits=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    for n in FILES if args.stage=='freeze' else FILES[1:]:require(not(DOC/n).exists(),'preserve '+n)
    b=inputs();limits=authority(b)
    if args.stage=='freeze':write(DOC/FILES[0],dict(schema='saved_hybrid_correspondence_diagnostic_protocol_v16',inputs=b,limits=limits,frames=245,rows=490,methods=list(METHODS),frozen_before_own_classification_and_counts=True,method_or_outputs_changed=False));return
    own=read(DOC/FILES[0]);require(own['inputs']==b and own['limits']==limits,'own frozen diagnostic changed');write(DOC/FILES[1],dict(protocol=binding(DOC/FILES[0]),inputs=b,new_production_calls=0))
    groups={m:{} for m in METHODS};count=Counter();pairs=[];error=None;cleanup=[];pending=DOC/('PENDING_'+FILES[2]);stream=raw=None
    try:
        post={}
        for p in rows(DOC/'POSTHOC_ROWS.jsonl.gz'):
            key=(p['method'],p['id']);require(p['method'] in METHODS and key not in post,'unique paired posthoc');post[key]=p
        require(len(post)==490,'full490 posthoc');scored=iter(rows(DOC/'PREDICTIONS.jsonl.gz'));ids=read(DOC/'PARENT_POPULATION_CHECKS.json')['ordered_ids'];require(len(ids)==len(set(ids))==245,'full ordered245')
        raw=pending.open('xb');stream=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
        for fid in ids:
            pair=[]
            for method in METHODS:
                g=next(scored);require((g['id'],g['method'])==(fid,method),'ordered original paired490');v=evidence(g,post.pop((method,fid)));del g;pair.append(v);count[method+':rows']+=1
                for name,flag in v['group_flags'].items():
                    bucket=groups[method].setdefault(name,[])
                    if flag:bucket.append(fid)
                count[method+':scalar_quality_disagreements']+=len(v['scalar_quality_label_disagreement_ids'])
                count[method+':actual_fit_lines']+=len(v['line_uncertainty_sourcecopies'])
            comparison=paired(*pair);pairs.append(comparison)
            for v in pair:stream.write((json.dumps(dict(v,paired_same_q_comparison=comparison),ensure_ascii=False,allow_nan=False)+'\n').encode())
            count['frames']+=1
        require(not post and next(scored,None) is None and count['frames']==245 and all(count[m+':rows']==245 for m in METHODS) and inputs()==b,'complete unchanged245/490 saved population')
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:
        if stream:
            try:stream.close()
            except BaseException as e:cleanup.append(dict(type=type(e).__name__,message=str(e),phase='gzip_close'))
        if raw:
            for phase,op in (('raw_flush',raw.flush),('raw_fsync',lambda:os.fsync(raw.fileno())),('raw_close',raw.close)):
                try:op()
                except BaseException as e:cleanup.append(dict(type=type(e).__name__,message=str(e),phase=phase))
        error=error or (cleanup[0] if cleanup else None)
        if error is None:
            try:os.link(pending,DOC/FILES[2]);pending.unlink()
            except BaseException as e:error=dict(type=type(e).__name__,message=str(e),phase='exclusive_publish')
        result=dict(schema='saved_hybrid_correspondence_diagnostic_checks_v16',complete=error is None,passed=error is None,own_protocol=binding(DOC/FILES[0]),inputs=b,limits=limits,counts=dict(count),
            groups={m:{k:dict(count=len(v),ids=v) for k,v in group.items()} for m,group in groups.items()},paired_same_q_results=pairs,
            paired_operating_status_groups={state:dict(count=sum(v['operational_pose_outcome']==state for v in pairs),ids=[v['id'] for v in pairs if v['operational_pose_outcome']==state]) for state in ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE')},
            error=error,cleanup_errors=cleanup,preserved_prefix=binding(pending) if pending.is_file() else None,rows=binding(DOC/FILES[2]) if (DOC/FILES[2]).is_file() else None,
            stored_combined_pose_summary_copies={m:read(DOC/'METRICS.json')['strata']['combined']['methods'][m] for m in METHODS},
            new_GT_metric_PnP_optimizer_J_SVD_model_draw_calls=0,all245_frames_and490_method_rows_preserved=error is None)
        write(DOC/FILES[3],result)
    require(error is None,'saved paired diagnostic incomplete; preserve first failure/prefix')
if __name__=='__main__':main()
