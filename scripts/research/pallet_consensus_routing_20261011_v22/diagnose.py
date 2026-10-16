"""One frozen saved-candidate routing audit; ROOT alone runs arithmetic.

All245 frames remain. No pose/GT/model is recomputed or selected by accuracy.
The V21 point closure and same-q V17 ordinary outputs are immutable evidence;
native ordinary is a different-coordinate reference, not a backend-only pair.
"""
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
import traceback

PRIVATE=Path('/tmp/pallet-consensus-routing-private-20261011-v22')
V21=Path('/tmp/pallet-whole-u-closure-private-20261011-v21')
V20=Path('/tmp/pallet-pnp-refit-closure-private-20261011-v20/diagnose.py')
V20_SHA='0f7e766084c7f0c40289bd92190f3d943f6a19b9d720aecad4bee59e7c2a8067'
STATS_SHA='9ea0be750a7f921bb572801b48953b9530f6a34799a368dc0507c0e34f9cf0d2'
SCALAR_SHA='d9f8079d42515d6c72eac4c75676493766b95395c63e049414fb67fe48631121'
PRIMARY='HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
ORDINARY='HYBRID_STANDARD_PNP'
NATIVE='NATIVE_STANDARD_PNP'
METHODS=('BASE','N3_SUBPIX',ORDINARY,NATIVE,'HYBRID_POINT_CONSENSUS_RESTORED',
    'HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS','HYBRID_OBSERVED_FRAME_POINT_ONLY',PRIMARY)
HYBRID=tuple(m for m in METHODS if m not in ('BASE','N3_SUBPIX',NATIVE))
FILES=('PROTOCOL.json','STARTED.json','ROWS.jsonl.gz','SUMMARY.json','CHECKS.json')
GOOD,BAD,UNKNOWN='CORRECT_WITHIN8PX','INCORRECT_OVER8PX','UNKNOWN'
NAMES={'require','safe','Audit','finite','ids','eligible','dimensions','dot','model','project','determinant','physical_rotation','equivalent'}
CONSTANTS={'ATOL','RTOL','RADIUS','TIE'}
LIMITS=dict(new_production_PnP_optimizer_Rodrigues_J_SVD_model_RGB_training_GT_calls=0,
    independent_saved_pose_projection_and_point_distance_only=True,
    geometry_and_errors_are_saved_independently_checked_proxy_evidence=True,
    numerical_NEW_is_not_accuracy_success=True,physical_pose_or_ownership_certified=False,
    source_axis_ID_order_not_independent_physical_axis_tag=True,
    ordinary_native_coordinate_supply_is_separate=True,
    stored_rank_and_incidence_only_no_new_J_or_rank=True,
    same_q_pixel_nonworse_invariant_only_for_admitted_ordinary_candidates=True,
    audit_PASS_not_accuracy_goal_or_solver_bug_PASS=True,
    historical_full_V16_564_FAIL_remains_FAIL=True,
    no_new_threshold_seed_settings_learning_or_performance_retry=True,
    candidate_decision_and_all_parent_bytes_unchanged=True,automatic_retry=False)

def require(value,message):
    if not value:raise ValueError(message)

def read(path):return json.loads(Path(path).read_bytes())

def binding(path):
    path=Path(path).absolute();require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'missing/symlink '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            require(bool(line.strip()),'blank saved row');yield json.loads(line)

def extract(path,expected,names,assignments=False):
    require(binding(path)['sha256']==expected,'exact immutable source '+str(path));tree=ast.parse(Path(path).read_bytes())
    selected=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in selected}==set(names) and len(selected)==len(names),'complete unique source whitelist')
    if assignments:selected=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    return module,dict(source=binding(path),names=sorted(names),whole_reused_function_AST_changes=0,
        AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest())

def prepare(a):
    snames={'require','read','binding','same','paths','metadata'}
    st,sp=extract(a.stats_checker,STATS_SHA,snames,assignments=True)
    se=dict(Path=Path,Counter=Counter,hashlib=hashlib,json=json,math=math,os=os,__file__=str(a.stats_checker))
    exec(compile(st,str(a.stats_checker)+':exact metadata only','exec'),se)
    args=argparse.Namespace(repo=a.repo,input=a.input)
    f=se['paths'](args);b={k:binding(p) for k,p in f.items()};m=se['metadata'](f,b)
    sf={n:a.statistics/n for n in ('STATISTICS_CHECK_PROTOCOL.json','STATISTICS_CHECK_STARTED.json','STATISTICS_CHECKS.json')}
    own,start,check=[read(sf[n]) for n in sf]
    require(own['schema']=='independent_whole_U_closure_statistics_protocol_v21'
        and check['schema']=='independent_whole_U_closure_statistics_checks_v21'
        and check['complete'] is check['passed'] is True and check['error'] is None
        and check['new_production_calls']==start['new_production_calls']==0
        and check['old_full_V16_validation_passed'] is False
        and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True
        and own['inputs']==start['inputs']==check['inputs']==b,'actual independent V21 statistics PASS/current closure')
    for key,value in dict(saved_error_rows_streamed=2205,saved_error_rows_selected=1960,frames=245,methods=8,
        moment_distributions_checked=783,moment_scalars_checked=6264,paired_groups_checked=63,metric_CI_slots_checked=189).items():
        require(check['actual_counts'][key]==value,'complete independent eight-method arithmetic '+key)
    require(same(check['own_protocol'],binding(sf['STATISTICS_CHECK_PROTOCOL.json']))
        and same(start['own_protocol'],check['own_protocol']) and same(check['checker'],binding(a.stats_checker)),'statistics source/protocol envelope')
    f.update({'statistics:'+k:v for k,v in sf.items()})
    f.update(code=Path(__file__).resolve(),audit_source=V20,scalar_source=a.repo/'scripts/research/pallet_same_observation_controls_20261010_v8/validation_checks.py')
    f['posthoc']=a.input/'POSTHOC_ROWS.jsonl.gz'
    for alias,name in (('ordinary_geometry','GEOMETRY_SEALED.jsonl.gz'),('ordinary_seal','GEOMETRY_SEAL.json'),
        ('ordinary_validation','VALIDATION_CHECKS.json'),('ordinary_own','VALIDATION_PROTOCOL.json'),('ordinary_started','VALIDATION_STARTED.json')):
        f[alias]=Path(m['inputs']['ordinary:'+name]['path'])
    f['new_geometry']=Path(m['inputs']['GEOMETRY.jsonl.gz']['path'])
    seal=read(f['ordinary_seal']);oc,op,os_=map(read,(f['ordinary_validation'],f['ordinary_own'],f['ordinary_started']))
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True
        and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490
        and same(seal['geometry'],binding(f['ordinary_geometry'])),'same-q ordinary sealed490')
    require(oc['schema']=='native_hybrid_standard_scalar_checks_v17' and oc['complete'] is oc['passed'] is True
        and oc['failure_count']==0 and oc['exception'] is None and oc['new_production_calls']==0
        and oc['counts']['frames']==245 and oc['counts']['rows']==490
        and oc['inputs']==op['inputs']==os_['inputs'] and same(oc['own_protocol'],binding(f['ordinary_own']))
        and same(os_['own_protocol'],oc['own_protocol']) and same(oc['inputs']['geometry'],binding(f['ordinary_geometry'])),'same-q ordinary full independent PASS')
    reused={'source_tree','semantic','compact_common','compact_pose','compact_rank','compact_ordinary','score_pose'}
    at,ap=extract(V20,V20_SHA,reused)
    audit_tree=ast.parse(V20.read_bytes());truth_fields=next(n.value for n in audit_tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='TRUTH_FIELDS' for t in n.targets))
    ae=dict(Path=Path,ast=ast,hashlib=hashlib,copy=copy,json=json,math=math,require=require,
        SCALAR_SHA=SCALAR_SHA,NAMES=NAMES,CONSTANTS=CONSTANTS,TRUTH_FIELDS=ast.literal_eval(truth_fields))
    exec(compile(at,str(V20)+':unchanged saved-pose source functions','exec'),ae)
    scalar,scalar_proof=ae['source_tree'](f['scalar_source'],binding(f['scalar_source']))
    return f,m,dict(metadata=sp,saved_pose=ap,independent_scalar=scalar_proof),ae,scalar

def compact_score(row):
    fields=('id','session','method','output_status','new_pose_estimated','fallback_used','pose_available',
        'input_points','original_native_N3_points','K','xyz','raw_hw','hidden_initial','head_checkpoint_sha256',
        'calibration_binding','selected_corner_ids','actual_pose','native_points')
    result={k:copy.deepcopy(row.get(k)) for k in fields}
    p=row['pose'];result['pose']=dict(available=p['available'],**{k:copy.deepcopy(p.get(k)) for k in ('translation_cm','rotation_deg','ADDsym_m')})
    require(all(type(v) is bool for v in (p['available'],row['pose_available'],row['new_pose_estimated'],row['fallback_used']))
        and p['available']==row['pose_available'] and not(row['new_pose_estimated'] and row['fallback_used']), 'typed saved pose/status')
    if p['available']:require(all(type(result['pose'][k]) in (int,float) and math.isfinite(result['pose'][k])
        for k in ('translation_cm','rotation_deg','ADDsym_m')),'finite recorded proxy errors')
    return result

def index_scores(f,ids,counts):
    groups={m:{} for m in METHODS}
    streams=(('fixed_scores',METHODS[:2]),('ordinary_scores',METHODS[2:4]),
        ('point_scores',('HYBRID_POINT_CONSENSUS_ORIGINAL',METHODS[4])),('history_scores',METHODS[5:7]),('new_scores',(PRIMARY,)))
    for key,allowed in streams:
        seen=set()
        for raw in rows(f[key]):
            k=(raw['method'],raw['id']);require(k not in seen and k[0] in allowed and k[1] in ids,'unique allowed saved score');seen.add(k);counts['score_rows_streamed']+=1
            if k[0] in groups:groups[k[0]][k[1]]=compact_score(raw)
        require(seen=={(m,i) for m in allowed for i in ids},'full source scored stream '+key)
    require(all(set(g)==set(ids) for g in groups.values()) and counts['score_rows_streamed']==2205,'eight full245 saved error populations')
    return groups

def outcome(delta):
    if delta is None:return 'UNAVAILABLE'
    t,r=delta['translation_cm'],delta['rotation_deg']
    return 'BOTH_BETTER' if t<0 and r<0 else 'BOTH_WORSE' if t>0 and r>0 else 'MIXED_OR_EQUAL'

def delta(candidate,reference):
    if not(candidate['pose']['available'] and reference['pose']['available']):return None
    return {k:candidate['pose'][k]-reference['pose'][k] for k in ('translation_cm','rotation_deg','ADDsym_m')}

def geometry_copy(value):
    if not isinstance(value,dict):return None
    return {k:copy.deepcopy(v) for k,v in value.items() if k in ('observed_normal_joint','modeled_normal_joint',
        'normal_only_joint','point_normal_joint','line_normal_joint','observed_weak','modeled_weak','state','local_only')}

def candidate_copy(c):
    names=('candidate_index','logical_start_index','proposal_kind','dimension_index','point_generator','point_generator_ids',
        'point_solution_index','fit_input_ids','final_inliers','accepted','rejected','score_count','truncated_sse_px2',
        'fit_pool_all_final8px_inliers','actual_fit_objective','actual_refit_performed','actual_SciPy_least_squares_called',
        'actual_iterative_OpenCV_optimizer_called','optimizer_success','scipy_status','global_uniqueness_proven')
    r={k:copy.deepcopy(c.get(k)) for k in names};r['stored_actual_fit_geometry']=geometry_copy(c.get('point_line_geometry'))
    guard=c.get('stable_incidence_guard');r['stored_incidence']={k:copy.deepcopy(guard.get(k)) for k in ('accepted','guard_state','stable_incidence_geometry')} if isinstance(guard,dict) else None
    return r

def routing(row,ordinary,scores,post,h,ae,audit,counts):
    fid=row['id'];common=ae['compact_common'](row);s=row['solver'];q,K,xyz,H=[row[k] for k in ('input_points','K','xyz','hidden_initial')]
    audit.eq(common,{k:ordinary[k] for k in common},fid+'.same_hybrid_coordinates_provenance')
    require(common=={k:ordinary[k] for k in common},'not exactly same hybrid observations')
    require(h['finite'](K,(3,3)) and K[0][0]>0 and K[1][1]>0 and K[2]==[0.,0.,1.] and K[0][1]==K[1][0]==0
        and h['finite'](xyz,(3,)) and min(xyz)>0 and len(q)==9 and h['ids'](H),'fixed pinhole/metric registry/H')
    U=[k for k in h['eligible'](q,row['raw_hw']) if k not in H]
    require(s['used']==ordinary['used']==U and s['hidden']==ordinary['hidden']==H and 8 not in U
        and s['line_edges']==s['fit_line_edges']==s['final_line_inliers']==[],'same actual U/H; no line factors or center')
    for m in HYBRID:
        for key in ('input_points','original_native_N3_points','K','xyz','raw_hw','hidden_initial','head_checkpoint_sha256','calibration_binding','selected_corner_ids'):
            require(scores[m][key]==row[key],'same fixed hybrid score supply '+m+'/'+key)
    require(scores[NATIVE]['input_points']==row['original_native_N3_points'],'native control uses original N3 coordinate supply')
    require(all(scores[m]['id']==fid and scores[m]['session']==row['session'] for m in METHODS),'score frame/session joins')
    for key in ('output_status','new_pose_estimated','fallback_used','pose_available','actual_pose','native_points'):
        require(scores[PRIMARY][key]==row[key],'new score/geometry immutable output '+key)
    require(post['id']==fid and post['session']==row['session'] and post['method']==PRIMARY
        and post['new_pose_estimated']==row['new_pose_estimated'] and post['fallback_used']==row['fallback_used'],'posthoc/geometry identity')
    new=row['new_pose_estimated'];require(new is s['available'] and not(new and row['fallback_used']),'approved output and stored best distinct')
    best=s.get('best_candidate');best_pose=ae['compact_pose'](best);opose=ordinary['best']
    bs=ae['score_pose'](best_pose,q,K,xyz,U,h,audit,counts,fid+'.stored_best')
    oscore=ae['score_pose'](opose,q,K,xyz,U,h,audit,counts,fid+'.ordinary_stored_best')
    if best is not None:
        audit.eq(best['score_count'],bs['score_count'],fid+'.best_count');audit.near(best['truncated_sse_px2'],bs['factor_truncated_sse_px2'],fid+'.best_same_pool_cost')
    closure=s['closure_ledger'];actual=s['all_candidate_solutions'];entries=[];ordinary_matches=[];accepted=[]
    viable=sorted((c for c in actual if c['accepted']),key=lambda c:(-c['score_count'],c['truncated_sse_px2'],c['dimension_index'],c['logical_start_index']))
    require((best is None)==(not viable),'stored best iff accepted candidate exists')
    if best is not None:
        audit.eq(best['candidate_index'],viable[0]['candidate_index'],fid+'.unchanged_count_SSE_dim_start_winner')
    non_equivalent_true_ties=[]
    for c in viable[1:]:
        tied=c['score_count']==best['score_count'] and abs(c['truncated_sse_px2']-best['truncated_sse_px2'])<=h['TIE']
        equivalent=h['equivalent'](c,best,xyz);counts['stored_viable_equivalence_checks']+=1
        if tied and not equivalent:non_equivalent_true_ties.append(c['candidate_index'])
    if best is not None:
        audit.eq(s['unresolved_ambiguity'],bool(non_equivalent_true_ties),fid+'.true_numerical_tie_only')
        audit.eq(new,not bool(non_equivalent_true_ties),fid+'.best_and_true_tie_operational_decision')
    for e in closure['candidates']:
        counts['live_standard_entries_scanned']+=1;record=copy.deepcopy(e);index=e['candidate_index']
        if index is None:
            require(e['accepted'] is False and bool(e['rejected']),'typed no-geometry standard rejection')
            record.update(same_saved_ordinary_branch=None,branch_match_scope='UNKNOWN_NO_STORED_CANDIDATE_POSE')
        else:
            require(type(index) is int and 0<=index<len(actual),'candidate ledger index')
            c=actual[index];require(c['candidate_index']==index and c['accepted']==e['accepted'] and c['rejected']==e['rejected']
                and c['proposal_kind']==e['proposal_kind'] and c['fit_input_ids']==U,'actual standard branch ledger')
            record['candidate']=candidate_copy(c)
            equivalent=opose is not None and h['equivalent'](c,opose,xyz)
            counts['stored_ordinary_branch_equivalence_checks']+=int(opose is not None)
            record['same_saved_ordinary_branch']=bool(equivalent)
            if equivalent:ordinary_matches.append(record)
            if c['accepted']:
                require(c['final_inliers']==U and c['score_count']==len(U),'admitted whole-U has maximum count')
                cs=ae['score_pose'](ae['compact_pose'](c),q,K,xyz,U,h,audit,counts,fid+'.standard.'+str(index))
                audit.eq(cs['final_inliers'],U,fid+'.standard_all_U');audit.near(c['truncated_sse_px2'],cs['factor_truncated_sse_px2'],fid+'.standard_cost')
                require(best is not None,'admitted standard must have a ranked best')
                audit.eq(best['score_count'],len(U),fid+'.union_maximum_count')
                audit.ok(best['truncated_sse_px2']<=c['truncated_sse_px2']+h['ATOL']+h['RTOL']*abs(c['truncated_sse_px2']),fid+'.union_same_q_pixel_nonworse')
                accepted.append(record)
        entries.append(record)
    if closure.get('skipped'):route='NO_STANDARD_CLOSURE_ATTEMPT'
    elif best is not None and str(best.get('proposal_kind','')).startswith('WHOLE_U_STANDARD_'):route='CLOSURE_SELECTED_STORED_BEST'
    elif accepted:route='STANDARD_RETAINED_OTHER_WINNER'
    else:route='ALL_STANDARD_CANDIDATES_REJECTED'
    matched_admitted=[e for e in ordinary_matches if e['accepted']]
    ordinary_route='NO_ORDINARY_STORED_BEST' if opose is None else 'ORDINARY_BRANCH_NOT_MATCHED_IN_STORED_POSE_PACKETS' if not ordinary_matches else 'ORDINARY_BRANCH_ADMITTED' if matched_admitted else 'ORDINARY_BRANCH_REJECTED'
    corners=post['corners'];require([c['id'] for c in corners]==list(range(8)) and all(c['input_quality'] in (GOOD,BAD,UNKNOWN) for c in corners),'eight saved proxy labels')
    require(post['mask_definition_provenance']['score_known_excludes_UNKNOWN_UNANNOTATED'] is True,'known mask definition retained')
    fit=s['fit_input_ids'] if new else [];final=s['final_inliers'] if new else []
    require(set(H).isdisjoint(fit) and 8 not in fit and post['finite_geometry_diagnostics']['actual_fit_ids']==fit
        and post['finite_geometry_diagnostics']['final8px_inliers']==final,'approved fit quality scope')
    pools={name:dict(ids=ids_,proxy_correct_ids=[k for k in ids_ if corners[k]['input_quality']==GOOD],
        proxy_bad_ids=[k for k in ids_ if corners[k]['input_quality']==BAD],proxy_unknown_ids=[k for k in ids_ if corners[k]['input_quality']==UNKNOWN],
        known_visibility_proxy_correct_ids=[k for k in ids_ if corners[k]['input_quality']==GOOD and corners[k]['human_state'] not in ('UNKNOWN','UNANNOTATED')],
        directly_visible_proxy_correct_ids=[k for k in ids_ if corners[k]['input_quality']==GOOD and corners[k]['human_state']=='DIRECT_VISIBLE'])
        for name,ids_ in (('used',U),('approved_fit',fit),('approved_final_inliers',final))}
    audit.eq(row['reprojected_ids'],H if new else [],fid+'.approved_H_once')
    audit.eq(s['operation_counts']['hidden_reprojection_assignments'],len(H) if new else 0,fid+'.H_assignment_count')
    require(not row['reprojections_reused_as_observations'] and not s['reprojected_points_reused_as_observations'],'H no-refit')
    require(row['native_points'][8]==row['original_native_N3_points'][8],'detector center unchanged')
    for k in range(8):
        expected=s['projected'][k] if new and k in H else q[k] if new else row['original_native_N3_points'][k]
        require(row['native_points'][k]==expected,'NEW H-once or full native fallback coordinate '+str(k))
    deltas={m:delta(scores[PRIMARY],scores[m]) for m in METHODS[:-1]}
    tie_flags=[{k:copy.deepcopy(c.get(k)) for k in ('candidate_index','proposal_kind','dimension_index','logical_start_index','score_count',
        'truncated_sse_px2','numerically_equivalent_to_selected','numerical_objective_tie','objective_difference_from_selected_px2')}
        for c in s['alternatives']]
    return dict(id=fid,session=row['session'],new_pose_estimated=new,fallback_used=row['fallback_used'],output_status=row['output_status'],
        solver_state=s['state'],routing=route,ordinary_branch_routing=ordinary_route,ordinary_approved_NEW=ordinary['new_pose_estimated'],
        stored_best_is_approved_NEW=new,best=candidate_copy(best) if best else None,ordinary_stored_best=opose,
        independently_scored_ordinary=oscore,independently_scored_best=bs,live_standard_candidates=entries,
        live_standard_accepted_count=len(accepted),live_standard_rejected_count=len(entries)-len(accepted),
        ordinary_matching_returned_count=len(ordinary_matches),ordinary_matching_admitted_count=len(matched_admitted),
        stored_multiple_solutions=s['multiple_solutions'],stored_unresolved_ambiguity=s['unresolved_ambiguity'],
        independently_recounted_non_equivalent_true_tie_candidate_indices=non_equivalent_true_ties,alternatives=tie_flags,
        saved_errors={m:scores[m]['pose'] for m in METHODS},saved_error_units=dict(translation_cm='cm',rotation_deg='degree',ADDsym_m='m'),
        saved_deltas_vs_controls=deltas,saved_T_R_outcome_vs_controls={m:outcome(d) for m,d in deltas.items()},
        known_mask_state=post['mask_state'],saved_mask_audit=copy.deepcopy(post['mask_audit']),pools=pools,
        q_H_and_native_center_preserved=True,partial_geometry_rank_is_not_physical_axis_certificate=True,physical_ownership_truth='UNKNOWN')

def summarize(records,cohort):
    result={};labels={r['id']:r['label'] for r in cohort['frames']}
    for name,chosen in (('combined',cohort['ids']),('easy',[i for i in cohort['ids'] if labels[i]=='clean']),('medium',[i for i in cohort['ids'] if labels[i]=='moderate'])):
        selected=[r for r in records if r['id'] in chosen];groups=Counter();by_route={};paired={}
        for r in selected:
            for control in METHODS[:-1]:
                key=(r['routing'],r['ordinary_branch_routing'],r['new_pose_estimated'],r['known_mask_state'],
                    len(r['pools']['approved_fit']['known_visibility_proxy_correct_ids']),len(r['pools']['approved_final_inliers']['proxy_bad_ids']),control,r['saved_T_R_outcome_vs_controls'][control]);groups[key]+=1
            by_route.setdefault(r['routing'],[]).append(r['id'])
        for route,route_ids in by_route.items():
            paired[route]={}
            for control in METHODS[:-1]:
                rr=[r for r in selected if r['id'] in route_ids and r['saved_deltas_vs_controls'][control] is not None]
                paired[route][control]=dict(n=len(rr),ordered_pair_ids=[r['id'] for r in rr],
                    saved_scalar_mean_delta={metric:math.fsum(r['saved_deltas_vs_controls'][control][metric] for r in rr)/len(rr) if rr else None
                        for metric in ('translation_cm','rotation_deg','ADDsym_m')},
                    derived_only_from_saved_errors=True,CI_or_pose_truth_not_recomputed=True)
        result[name]=dict(frames=len(selected),ordered_ids=chosen,routing_counts=dict(Counter(r['routing'] for r in selected)),
            ordinary_branch_routing_counts=dict(Counter(r['ordinary_branch_routing'] for r in selected)),routing_frame_ids=by_route,
            routing_saved_scalar_delta_readout=paired,
            cross_groups=[dict(routing=k[0],ordinary_branch_routing=k[1],NEW=k[2],known_mask_state=k[3],
                known_visibility_proxy_correct_fit_count=k[4],bad_proxy_final_count=k[5],control=k[6],saved_T_R_outcome=k[7],n=n)
                for k,n in sorted(groups.items(),key=lambda p:repr(p[0]))],
            accuracy_means_or_CIs_recomputed=False,new_physical_ground_truth=False)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('freeze','run'))
    for n in ('repo','input','statistics'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--stats-checker',type=Path,default=V21/'stats_check.py');p.add_argument('--output',type=Path,default=PRIVATE/'audit')
    a=p.parse_args()
    for n in ('repo','input','statistics','stats_checker','output'):setattr(a,n,getattr(a,n).absolute())
    require(a.output==PRIVATE/'audit' and not any(x.is_symlink() for x in (a.output,*a.output.parents)),'exclusive fixed private22 output')
    f,m,proof,ae,scalar=prepare(a);b={k:binding(v) for k,v in f.items()}
    require(not any(Path(v).is_relative_to(a.output) or a.output.is_relative_to(Path(v)) for v in f.values()),'output overlaps frozen input')
    proto=a.output/FILES[0];cohort=read(f['cohort']);ids=cohort['ids'];require(len(ids)==len(set(ids))==245,'whole245 before decode')
    authority=dict(inputs=b,source_proof=proof,methods=list(METHODS),whole245_ids=ids,limits=LIMITS,
        scientific_row_decode_only_after_own_protocol=True,new_production_calls=0)
    if a.phase=='freeze':
        require(not a.output.exists(),'absent audit output');a.output.mkdir(parents=True)
        write(proto,dict(schema='saved_point_closure_routing_protocol_v22',**authority));return
    own=read(proto);require(own['schema']=='saved_point_closure_routing_protocol_v22' and all(own[k]==v for k,v in authority.items()),'exact own frozen source/parents')
    for n in FILES[1:]:require(not(a.output/n).exists(),'first failure/output preserved '+n)
    write(a.output/FILES[1],dict(protocol=binding(proto),inputs=b,source_proof=proof,new_production_calls=0))
    h=dict(math=math,json=json);exec(compile(scalar,str(f['scalar_source'])+':unchanged independent stdlib','exec'),h)
    audit=h['Audit']();counts=Counter();records=[];error=None
    try:
        groups=index_scores(f,ids,counts);ordinary={};seen=set()
        for raw in rows(f['ordinary_geometry']):
            k=(raw['method'],raw['id']);require(k not in seen and k[0] in (ORDINARY,NATIVE) and k[1] in ids,'full ordinary geometry identity');seen.add(k);counts['ordinary_geometry_rows_streamed']+=1
            if k[0]==ORDINARY:ordinary[k[1]]=ae['compact_ordinary'](raw)
        require(seen=={(m,i) for m in (ORDINARY,NATIVE) for i in ids},'full paired ordinary geometry490')
        posthoc={}
        for raw in rows(f['posthoc']):
            require(raw['method']==PRIMARY and raw['id'] in ids and raw['id'] not in posthoc,'new full245 posthoc')
            posthoc[raw['id']]={k:copy.deepcopy(raw[k]) for k in ('id','session','method','new_pose_estimated','fallback_used','corners','mask_state','mask_audit','mask_definition_provenance')}
            # Only approved-ID fields are needed; never retain large J/guard packets.
            posthoc[raw['id']]['finite_geometry_diagnostics']={k:copy.deepcopy(raw['finite_geometry_diagnostics'][k]) for k in ('actual_fit_ids','final8px_inliers')}
            counts['posthoc_rows_streamed']+=1
        require(set(posthoc)==set(ids),'full245 posthoc retained')
        with gzip.open(a.output/FILES[2],'xt') as stream:
            for raw in rows(f['new_geometry']):
                index=counts['new_geometry_rows_streamed'];require(index<245 and raw['id']==ids[index] and raw['method']==PRIMARY,'ordered full245 new geometry')
                row=routing(raw,ordinary[raw['id']],{m:groups[m][raw['id']] for m in METHODS},posthoc[raw['id']],h,ae,audit,counts)
                counts['new_geometry_rows_streamed']+=1;json.dump(row,stream,ensure_ascii=False,allow_nan=False);stream.write('\n')
                # Summary retains only categorical IDs/counts; full candidates exist in output rows.
                records.append({k:copy.deepcopy(row[k]) for k in ('id','routing','ordinary_branch_routing','new_pose_estimated','known_mask_state','pools','saved_T_R_outcome_vs_controls','saved_deltas_vs_controls')})
        require(counts['new_geometry_rows_streamed']==counts['posthoc_rows_streamed']==245 and counts['ordinary_geometry_rows_streamed']==490,'complete fixed geometry counts')
        require(counts['independent_project_entries']==counts['independent_project_returns'] and counts['independent_project_exceptions']==0,'closed independent projection dispatch')
        require(b=={k:binding(v) for k,v in f.items()},'all frozen inputs unchanged')
        write(a.output/FILES[3],dict(schema='saved_point_closure_routing_summary_v22',complete=True,frames=245,
            strata=summarize(records,cohort),actual_counts=dict(counts),rows=binding(a.output/FILES[2]),limits=LIMITS,
            saved_error_units=dict(translation_cm='cm',rotation_deg='degree',ADDsym_m='m')))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and audit.failure_count==0
    write(a.output/FILES[4],dict(schema='saved_point_closure_routing_checks_v22',complete=error is None,passed=passed,
        exception=error,checks=audit.checks,failure_count=audit.failure_count,failures=audit.failures,max_absolute_difference=audit.max_difference,
        actual_counts=dict(counts),inputs=b,own_protocol=binding(proto),code=b['code'],source_proof=proof,
        new_production_calls=0,old_full_V16_validation_passed=False,limits=LIMITS,
        rows=binding(a.output/FILES[2]) if (a.output/FILES[2]).is_file() else None))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
