"""Join saved solver support and proxy correspondence quality; produce no poses."""
import argparse
import ast
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
ORDINARY=REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'
PRIVATE=Path('/tmp/pallet-consensus-decision-repair-private-20261011-v18')
METHODS=('HYBRID_STANDARD_PNP','HYBRID_POINT_CONSENSUS_RESTORED')
QUALITY=('CORRECT_WITHIN8PX','INCORRECT_OVER8PX','UNKNOWN_REFERENCE_OR_INPUT')
FACES=dict(z_negative=(0,1,2,3),z_positive=(4,5,6,7),y_negative=(0,1,4,5),y_positive=(2,3,6,7),x_negative=(0,3,4,7),x_positive=(1,2,5,6))
FILES=('CORRESPONDENCE_AUDIT_PROTOCOL.json','CORRESPONDENCE_AUDIT_STARTED.json','CORRESPONDENCE_AUDIT_CHECKS.json','CORRESPONDENCE_AUDIT_ROWS.jsonl.gz','CORRESPONDENCE_AUDIT_SUMMARY.json','CORRESPONDENCE_AUDIT_SUMMARY.csv')
LIMITS=dict(new_GT_annotation_RGB_model_PnP_optimizer_projection_J_SVD_training_seed_timing_calls=0,
    correspondence_quality_is_existing_N3_phase_GEOMETRIC_PROXY_not_physical_truth=True,
    solver_8px_inliers_are_pose_fit_consistency_not_proxy_correctness=True,physical_boundary_ownership='UNKNOWN',
    controlled_corruption_experiments_performed=False,causal_solver_or_line_effect_not_proven=True,
    face_ID_containment_is_not_general_coplanarity_or_observability=True,new_error_norms_moments_or_CIs=0)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(p.read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def public(v):
    p=Path(v['path']);require(not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs') and v.get('origin','public_repository')=='public_repository','public saved artifact required');return REPO/p
def paths(a):
    f=dict(checker=Path(__file__).resolve(),posthoc_source=CODE.parent/'pallet_three_head_observation_20261010_v7/statistics.py',cuboid_source=CODE.parent/'pallet_observation_refiner_20261009_v1/solver.py',ordinary_backend=CODE.parent/'pallet_cornerwise_independent_20261010_v4/pose.py',runtime_checker=CODE/'runtime_checks.py')
    names=('METRICS.json','STATISTICS_STARTED.json','VERIFICATION.json','STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','SCORING_RECEIPT.json','PREDICTIONS.jsonl.gz','POSTHOC_ROWS.jsonl.gz','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json','GEOMETRY_SEAL.json')
    for prefix,folder in (('restored',a.input),('ordinary',a.ordinary_input)):
        f.update({prefix+':'+n:folder/n for n in names});m=read(folder/'METRICS.json')
        f[prefix+':cohort']=public(m['inputs']['cohort']);f[prefix+':producer']=public(m['inputs']['own_code'])
        f[prefix+':verifier']=public(read(folder/'VERIFICATION.json')['checker']);f[prefix+':geometry_checker']=public(read(folder/'VALIDATION_CHECKS.json')['checker'])
    f['fixed_scores']=public(read(a.input/'METRICS.json')['inputs']['fixed_scores'])
    f.update({n:a.input/n for n in ('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')});return f
def authority(f,b):
    expected=dict(restored=('point_consensus_decision_raw_row_metrics_v18','independent_point_consensus_decision_statistics_checks_v18',1470,972,135),ordinary=('native_hybrid_standard_raw_row_metrics_v17','independent_native_hybrid_standard_statistics_checks_v17',980,648,81))
    for prefix,(ms,vs,n,m,c) in expected.items():
        x=lambda name:read(f[prefix+':'+name]);metric=x('METRICS.json');v=x('VERIFICATION.json');vp=x('STATISTICS_VERIFICATION_PROTOCOL.json');start=x('STATISTICS_VERIFICATION_STARTED.json');score=x('SCORING_RECEIPT.json')
        require(metric['schema']==ms and metric['complete'] is True and metric['inputs']==x('STATISTICS_STARTED.json')['inputs'],'actual statistics '+prefix)
        for key,name in (('new_scores','PREDICTIONS.jsonl.gz'),('own_code','producer'),('cohort','cohort')):require(same(metric['inputs'][key],b[prefix+':'+name]),'produced posthoc input '+prefix+key)
        require(same(metric['posthoc_rows'],b[prefix+':POSTHOC_ROWS.jsonl.gz']),'actual posthoc bytes')
        require(v['schema']==vs and v['complete'] is True and v['passed'] is True and v['failure_count']==0 and v['error'] is None and v['inputs']==vp['inputs']==start['inputs'] and same(v['protocol'],b[prefix+':STATISTICS_VERIFICATION_PROTOCOL.json']) and same(v['checker'],b[prefix+':verifier']) and same(v['inputs']['checker'],b[prefix+':verifier']),'actual independent statistics PASS')
        for key,name in (('METRICS','METRICS.json'),('new_scores','PREDICTIONS.jsonl.gz')):require(same(v['inputs'][key],b[prefix+':'+name]),'independently checked bytes')
        require(v['actual_counts']['saved_scored_rows_streamed']==n and v['actual_counts']['moment_scalars_checked']==m and v['actual_counts']['metric_CI_slots_checked']==c,'actual independent topology')
        require(score['complete'] is True and score['frames']==245 and score['scored_rows']==490 and same(score['predictions'],b[prefix+':PREDICTIONS.jsonl.gz']),'actual490 scores')
        seal=x('GEOMETRY_SEAL.json');check=x('VALIDATION_CHECKS.json');own=x('VALIDATION_PROTOCOL.json');gs=x('VALIDATION_STARTED.json')
        require(seal['complete'] is True and seal['frames']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and same(score['geometry_seal'],b[prefix+':GEOMETRY_SEAL.json']),'actual paired seal')
        require(check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['inputs']==own['inputs']==gs['inputs'] and same(check['own_protocol'],b[prefix+':VALIDATION_PROTOCOL.json']) and same(check['checker'],b[prefix+':geometry_checker']) and same(own['checker'],b[prefix+':geometry_checker']),'actual geometry PASS')
    require(same(b['restored:cohort'],b['ordinary:cohort']) and same(read(f['restored:METRICS.json'])['inputs']['fixed_scores'],b['fixed_scores']),'same245 and immutable fixedN3')
    rc,rp,rs=[read(f[n]) for n in ('RUNTIME_CHECKS.json','RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json')]
    require(rc['schema']=='independent_restored_point_consensus_runtime_checks_v18' and rc['complete'] is True and rc['passed'] is True and rc['failure_count']==0 and rc['exception'] is None and rc['rows']==150 and rc['measured']==130 and rc['summary_slots']==56 and rc['inputs']==rp['inputs']==rs['inputs'] and same(rc['own_protocol'],b['RUNTIME_CHECK_PROTOCOL.json']) and same(rc['checker'],b['runtime_checker']),'actual fresh runtime independent PASS')
    require(same(rc['inputs']['predictions'],b['restored:PREDICTIONS.jsonl.gz']) and same(rc['inputs']['validation'],b['restored:VALIDATION_CHECKS.json']),'runtime checked same accuracy authority')
    scope=read(f['restored:VALIDATION_CHECKS.json'])['scope'];require(scope['original_full_scalar_validation_passed'] is False and scope['point_only_original_failed_predicates']==0 and scope['independent_parent_point_check_count_not_claimed'] is True,'preserve fullV16FAIL and narrowpoint scope')
    tree=ast.parse(f['cuboid_source'].read_text());fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='cuboid');ret=next(v for v in fn.body if isinstance(v,ast.Return));signs=[]
    for row in ret.value.args[0].elts:signs.append(tuple(-1 if isinstance(v,ast.UnaryOp) and isinstance(v.op,ast.USub) else 1 for v in row.elts))
    require(signs==[(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)],'fixed cuboid ID signs')
    models=ast.parse('self.models = [cuboid(*d) for d in self.dims]').body[0];require(any(ast.dump(n,include_attributes=False)==ast.dump(models,include_attributes=False) for n in ast.walk(tree)),'HypothesisBank model ID source')
    bank=next(n for n in ast.parse(f['ordinary_backend'].read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PoseBank');require(len(bank.bases)==1 and isinstance(bank.bases[0],ast.Name) and bank.bases[0].id=='HypothesisBank','ordinary inherits same corner model source')
    require(all(all(signs[k][axis]==sign for k in FACES[name]) for name,axis,sign in (('x_negative',0,-1),('x_positive',0,1),('y_negative',1,-1),('y_positive',1,1),('z_negative',2,-1),('z_positive',2,1))),'face-ID source mapping')
def layout(ids):return dict(ids=ids,count=len(ids),contained_in_known_cuboid_faces=[name for name,v in FACES.items() if ids and set(ids)<=set(v)],general_coplanarity_or_rank_not_computed=True)
def geometry(g):
    if not isinstance(g,dict):return g
    return {k:geometry(v) if isinstance(v,dict) else v for k,v in g.items() if k not in ('matrix','observed_normal_J','modeled_normal_J','selected_candidate','normalized_matrix','column_normalized','J')}
def thin(g):
    s=g['solver'];best=s.get('best_candidate') or {};geom=s.get('point_line_geometry') or s.get('geometry')
    return {**{k:g[k] for k in ('id','session','method','input_points','original_native_N3_points','hidden_initial','native_points','selected_corner_ids','new_pose_estimated','fallback_used','output_status','hidden_reprojected','reprojected_ids','reprojections_reused_as_observations','pose','evaluation_reference')},
        'solver_state':s['state'],'used':s['used'],'fit':s['fit_input_ids'] if g['new_pose_estimated'] else [],'final':s['final_inliers'] if g['new_pose_estimated'] else [],'raw_candidate_final':s['final_inliers'],
        'self_residuals_used_px':s.get('residuals_used_px'),'hidden_projection_source':{k:s['projected'][k] for k in g['hidden_initial']} if g['new_pose_estimated'] else {},'geometry':geometry(geom),
        'approved_final_pose_metadata':{k:s.get(k) if g['new_pose_estimated'] else None for k in ('selected_hypothesis','dimension_index','cf_extents')},
        'stored_best_is_approved_final_pose':g['new_pose_estimated'],'selected_candidate_metadata':{k:geometry(best[k]) for k in ('selected_hypothesis','dimension_index','cf_extents','fit_input_ids','fit_line_edges','score_count','truncated_sse_px2','soft_l1_cost','stable_incidence_guard') if k in best}}
def compact_scores(path,method):
    out={};seen=Counter()
    for g in rows(path):
        seen[g['method']]+=1
        if g['method']==method:require(g['id'] not in out,'duplicate scored ID');out[g['id']]=thin(g)
    require(sum(seen.values())==490 and len(seen)==2 and all(v==245 for v in seen.values()) and len(out)==245,'full paired490 scored population');return out
def compact_post(path,method):
    out={};seen=Counter()
    for p in rows(path):
        seen[p['method']]+=1
        if p['method']==method:require(p['id'] not in out,'duplicate posthoc ID');out[p['id']]=p
    require(sum(seen.values())==490 and len(seen)==2 and all(v==245 for v in seen.values()) and len(out)==245,'full paired490 posthoc population');return out
def evidence(g,p):
    require(all(g[k]==p[k] for k in ('id','session','method','output_status','new_pose_estimated','fallback_used','hidden_initial')),'scored/posthoc identity');corners={v['id']:v for v in p['corners']};require(set(corners)==set(range(8)),'all8 corners')
    pools={};H=g['hidden_initial'];require(g['reprojections_reused_as_observations'] is False and set(H).isdisjoint(g['used']) and set(H).isdisjoint(g['fit']) and set(H).isdisjoint(g['final']),'H excluded/no reproj refit')
    require(g['hidden_reprojected']==bool(g['new_pose_estimated'] and H) and g['reprojected_ids']==(H if g['new_pose_estimated'] else []) and g['native_points'][8]==g['original_native_N3_points'][8],'once H display/center preservation')
    require(all(g['native_points'][k]==xy for k,xy in g['hidden_projection_source'].items()) and (not g['fallback_used'] or g['native_points']==g['original_native_N3_points']),'saved H projection/full native fallback')
    for name,ids in (('solver_used',g['used']),('accepted_fit',g['fit']),('accepted_final_inliers',g['final'])):
        require(ids==p['pools'][name]['ids'] and len(set(ids))==len(ids) and set(ids)<=set(range(8)),'actual pool IDs '+name);q={label:[k for k in ids if corners[k]['input_quality']==label] for label in QUALITY}
        require(set(ids)==set(k for values in q.values() for k in values) and dict(Counter(corners[k]['input_quality'] for k in ids))==p['pools'][name]['quality_counts'],'saved quality counts '+name)
        pools[name]=dict(ids=ids,quality_ids=q,quality_counts={k:len(v) for k,v in q.items()},correct_proxy_layout=layout(q[QUALITY[0]]))
    for k,c in corners.items():
        error=c['input_error_px'];require(c['input_quality']==(QUALITY[2] if error is None else QUALITY[0] if error<=8 else QUALITY[1]),'fixed saved8px proxy classification');require(c['accepted_new_pose_fit']==(k in g['fit']) and c['accepted_new_pose_final_inlier']==(k in g['final']),'corner/pool membership')
    ref=g['evaluation_reference'];require(ref['phase_control']=='N3_SUBPIX' and ref['initial_N3_native_points']==g['original_native_N3_points'] and ref['human_states_native']==p['mask_audit']['human_states_native'],'saved reference phase')
    supplied=[k for k in range(8) if k not in H and g['input_points'][k]!=[-1,-1]];bad=[k for k in supplied if corners[k]['input_quality']==QUALITY[1]]
    residual=g['self_residuals_used_px'];self_errors=dict(zip(g['used'],residual)) if residual is not None else None
    if g['new_pose_estimated']:require(residual is not None and len(residual)==len(g['used']) and g['final']==[k for k in g['used'] if self_errors[k]<=8],'stored solver self8px predicate')
    boundary=[{k:c[k] for k in ('id','accepted','candidate_proxy_quality','candidate_error_px','N3_error_px','delta_vs_N3_px','candidate_LOO_residual_px','native_N3_LOO_residual_px','reason','accepted_new_pose_fit','accepted_new_pose_final_inlier')} for c in p['boundary_candidates']]
    return dict(id=g['id'],session=g['session'],method=g['method'],output_status=g['output_status'],new_pose_estimated=g['new_pose_estimated'],fallback_used=g['fallback_used'],solver_state=g['solver_state'],pools=pools,
        supplied_ids=supplied,supplied_quality_ids={q:[k for k in supplied if corners[k]['input_quality']==q] for q in QUALITY},supplied_correct_proxy_layout=layout([k for k in supplied if corners[k]['input_quality']==QUALITY[0]]),solver_self_residual_px_by_used_id=self_errors,all_bad_supplied_are_accepted_final_inliers=bool(g['new_pose_estimated'] and bad and set(bad)<=set(g['final'])),
        raw_candidate_final_inlier_ids=g['raw_candidate_final'],raw_candidate_ids_are_not_accepted_fallback_support=not g['new_pose_estimated'],corners=p['corners'],boundary_candidates=boundary,
        mask_state=p['mask_state'],mask_audit=p['mask_audit'],mask_definition_provenance=p.get('mask_definition_provenance','unchanged V17 saved legacy posthoc; UNKNOWN handling can differ'),pose_outcome_vs_N3=p['pose_outcome_vs_N3'],delta_vs_N3=p['delta_vs_N3'],H=H,reprojected_ids=g['reprojected_ids'],stored_geometry=g['geometry'],approved_final_pose_metadata=g['approved_final_pose_metadata'],stored_best_is_approved_final_pose=g['stored_best_is_approved_final_pose'],selected_candidate_metadata=g['selected_candidate_metadata'],physical_boundary_ownership='UNKNOWN')
def guard(a):
    require(a.input.resolve()==DOC and a.ordinary_input.resolve()==ORDINARY,'fixed public two-family inputs');out=a.output.absolute();require(out.is_dir() and (out==DOC or out.is_relative_to(PRIVATE)) and not any(p.is_symlink() for p in (out,*out.parents)),'exclusive existing DOC/private output')
    for n in FILES[1:] if a.stage=='run' else FILES:require(not (out/n).exists(),'output exists '+n)
    require(not (out/('INTERRUPTED_'+FILES[3])).exists(),'interrupted prefix exists');return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--ordinary-input',type=Path,default=ORDINARY);p.add_argument('--output',type=Path,default=DOC);a=p.parse_args();out=guard(a);f=paths(a);b={k:binding(v) for k,v in f.items()};authority(f,b)
    if a.stage=='freeze':write(out/FILES[0],dict(schema='saved_proxy_correspondence_audit_protocol_v18',inputs=b,checker=b['checker'],methods=list(METHODS),frames=245,limits=LIMITS,face_IDs=FACES,frozen_before_own_saved_row_decode_and_counts=True));return
    own=read(out/FILES[0]);ob=binding(out/FILES[0]);require(own['schema']=='saved_proxy_correspondence_audit_protocol_v18' and own['inputs']==b and own['limits']==LIMITS and own['methods']==list(METHODS) and own['face_IDs']=={k:list(v) for k,v in FACES.items()},'own fixed protocol')
    write(out/FILES[1],dict(inputs=b,own_protocol=ob,checker=b['checker'],new_production_calls=0));error=None;count=0;groups={};boundary_counts=Counter();prefix=out/('INTERRUPTED_'+FILES[3]);raw=None;stream=None
    try:
        cohort=read(f['restored:cohort']);ids=cohort['ids'];labels={r['id']:r['label'] for r in cohort['frames']};require(len(ids)==len(set(ids))==245 and Counter(labels.values())==Counter(clean=153,moderate=92),'exacteasy/medium245')
        gs=[compact_scores(f[x+':PREDICTIONS.jsonl.gz'],m) for x,m in zip(('ordinary','restored'),METHODS)];ps=[compact_post(f[x+':POSTHOC_ROWS.jsonl.gz'],m) for x,m in zip(('ordinary','restored'),METHODS)]
        require(all(set(v)==set(ids) for v in (*gs,*ps)),'all245 saved identities');raw=prefix.open('xb');stream=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
        for fid in ids:
            ordinary,restored=[g[fid] for g in gs]
            for key in ('session','input_points','original_native_N3_points','hidden_initial','selected_corner_ids','evaluation_reference'):require(ordinary[key]==restored[key],'same frozen q/native/H/reference '+key)
            e=[evidence(g[fid],v[fid]) for g,v in zip(gs,ps)];require(e[0]['corners']==e[1]['corners'] or all(all(a[k]==c[k] for k in ('id','human_state','reference_known','N3_error_px','input_error_px','input_quality')) for a,c in zip(e[0]['corners'],e[1]['corners'])),'same supplied proxy labels')
            delta={k:restored['pose'][k]-ordinary['pose'][k] for k in ('translation_cm','rotation_deg','ADDsym_m')} if restored['pose']['available'] and ordinary['pose']['available'] else None
            pair_outcome='POSE_UNAVAILABLE' if delta is None else 'BOTH_BETTER' if delta['translation_cm']<0 and delta['rotation_deg']<0 else 'BOTH_WORSE' if delta['translation_cm']>0 and delta['rotation_deg']>0 else 'MIXED_OR_UNCHANGED'
            packet=dict(id=fid,session=ordinary['session'],stratum=labels[fid],input_points=ordinary['input_points'],original_native_N3_points=ordinary['original_native_N3_points'],saved_evaluation_reference=ordinary['evaluation_reference'],methods=e,restored_minus_ordinary_saved_pose_error=delta,saved_pose_error_units=dict(translation_cm='cm',rotation_deg='degree',ADDsym_m='m'),paired_pose_outcome=pair_outcome,same_q_H_reference_verified=True,limits=LIMITS)
            stream.write((json.dumps(packet,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode());count+=1
            for item in e:
                for stratum in ('combined',labels[fid]):
                    key=(stratum,item['method'],item['output_status'],item['mask_state'],item['pose_outcome_vs_N3'],item['pools']['accepted_final_inliers']['quality_counts'][QUALITY[1]],item['pools']['accepted_fit']['quality_counts'][QUALITY[0]],item['pools']['accepted_final_inliers']['quality_counts'][QUALITY[0]],tuple(item['pools']['accepted_final_inliers']['correct_proxy_layout']['contained_in_known_cuboid_faces']))
                    groups.setdefault(key,[]).append(fid)
                for c in item['boundary_candidates']:
                    if c['accepted']:boundary_counts[(item['method'],'adopted')]+=1;boundary_counts[(item['method'],'improved' if c['delta_vs_N3_px'] is not None and c['delta_vs_N3_px']<0 else 'worsened' if c['delta_vs_N3_px'] is not None and c['delta_vs_N3_px']>0 else 'unchanged_or_unknown')]+=1
        stream.close();stream=None;raw.flush();os.fsync(raw.fileno());raw.close();raw=None;require(count==245 and {k:binding(v) for k,v in f.items()}==b and binding(out/FILES[0])==ob,'all245/input preservation')
        table=[dict(stratum=k[0],method=k[1],status=k[2],mask_state=k[3],pose_outcome_vs_N3=k[4],bad_proxy_final_inliers=k[5],correct_proxy_fit_points=k[6],correct_proxy_final_inliers=k[7],correct_final_inlier_known_face_containment=list(k[8]),n=len(v),ids=v) for k,v in sorted(groups.items())]
        require(not (out/FILES[3]).exists(),'rows publication exists');os.rename(prefix,out/FILES[3]);write(out/FILES[4],dict(schema='saved_proxy_correspondence_audit_summary_v18',complete=True,frames=count,method_rows=490,groups=table,adopted_boundary_counts=[dict(method=k[0],category=k[1],count=v) for k,v in sorted(boundary_counts.items())],limits=LIMITS,rows=binding(out/FILES[3])))
        with (out/FILES[5]).open('x',newline='') as csvfile:
            writer=csv.DictWriter(csvfile,fieldnames=tuple(k for k in table[0] if k!='ids'));writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='ids'} for r in table);csvfile.flush();os.fsync(csvfile.fileno())
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        for handle in (stream,raw):
            if handle:
                try:handle.close()
                except BaseException as exc:
                    if error is None:error=dict(type=type(exc).__name__,message=str(exc),cleanup=True)
    write(out/FILES[2],dict(schema='saved_proxy_correspondence_audit_checks_v18',complete=error is None and count==245,passed=error is None and count==245,inputs=b,own_protocol=ob,checker=b['checker'],frames=count,method_rows=2*count,exception=error,new_production_calls=0,limits=LIMITS,outputs={n:binding(out/n) for n in FILES[3:] if (out/n).is_file()},preserved_prefix=binding(prefix) if prefix.is_file() else None,first_failure_preserved=True))
    if error is not None:raise SystemExit(1)

if __name__=='__main__':main()
