"""Summarize all saved V16 failed scalar witnesses without new geometry.

Freeze byte inputs before reading/comparing saved numbers. Count failed
predicates and distinct stored candidate contexts, not independent poses.
The underlying complete geometry scalar audit remains FAIL with 564 failures.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
FILES=('FAILURE_SUMMARY_PROTOCOL.json','FAILURE_SUMMARY_STARTED.json','FAILURE_WITNESS_SUMMARY.json','FAILURE_SUMMARY_CHECKS.json')
METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
LIMITS=dict(new_GT_model_PnP_optimizer_projection_J_SVD_metric_timing_calls=0,
    residual_derivative_or_tolerance_recalculated=False,original_failed_values_and_contexts_unchanged=True,
    geometry_validation_passed=False,threshold_changes=0,physical_cause_or_pose_accuracy_certified=False,
    unique_candidates_mean_stored_method_phase_index_contexts_not_unique_physical_poses=True,
    failed_predicates_can_share_one_numerical_difference=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def paths(folder):
    f={n:folder/n for n in ('FAILURE_WITNESSES.json','FAILURE_WITNESS_CHECKS.json','FAILURE_WITNESS_PROTOCOL.json','FAILURE_WITNESS_STARTED.json',
        'TRACE_VALIDATION_CHECKS.json','TRACE_VALIDATION_PROTOCOL.json','TRACE_VALIDATION_STARTED.json','VALIDATION_READY_CHECKS.json')}
    f.update(checker=Path(__file__),witness_source=CODE/'failure_witness.py',trace_source=CODE/'validation_failure_trace.py');return f
def authority(f,b):
    saved,check,own,start,trace,tp,ts,ready=[read(f[k]) for k in ('FAILURE_WITNESSES.json','FAILURE_WITNESS_CHECKS.json','FAILURE_WITNESS_PROTOCOL.json','FAILURE_WITNESS_STARTED.json',
        'TRACE_VALIDATION_CHECKS.json','TRACE_VALIDATION_PROTOCOL.json','TRACE_VALIDATION_STARTED.json','VALIDATION_READY_CHECKS.json')]
    require(saved['schema']=='hybrid_stored_failure_witnesses_v16' and saved['complete'] is True and saved['geometry_validation_passed'] is False,'complete saved witnesses do not mean geometry PASS')
    require(check['schema']=='hybrid_stored_failure_witness_checks_v16' and check['complete'] is True and check['passed'] is True and check['error'] is None and same(check['witnesses'],b['FAILURE_WITNESSES.json']),'actual stored-witness join PASS')
    require(check['inputs']==own['inputs']==start['inputs'] and same(check['own_protocol'],b['FAILURE_WITNESS_PROTOCOL.json']) and same(start['own_protocol'],b['FAILURE_WITNESS_PROTOCOL.json']) and same(own['inputs']['script'],b['witness_source']),'own witness source/freeze/STARTED')
    require(same(own['inputs']['chosen_failure_checks'],b['TRACE_VALIDATION_CHECKS.json']) and same(own['inputs']['trace_checker'],b['trace_source']) and same(own['inputs']['chosen_failure_protocol'],b['TRACE_VALIDATION_PROTOCOL.json']),'same complete trace selected by witness producer')
    require(trace['schema']=='hybrid_factor_subset_scalar_failure_trace_checks_v16' and trace['complete'] is True and trace['passed'] is False and trace['trace_complete'] is True and trace['trace_passed'] is True and trace['scalar_validation_passed'] is False and trace['exception'] is None and trace['checks']==198134007 and trace['failure_count']==564 and trace['all_failed_predicates_retained'] is True,'original full scalar result stays FAIL564')
    require(trace['inputs']==tp['inputs']==ts['inputs'] and same(trace['own_protocol'],b['TRACE_VALIDATION_PROTOCOL.json']) and same(ts['own_protocol'],b['TRACE_VALIDATION_PROTOCOL.json']) and same(trace['trace_checker'],b['trace_source']),'actual logging-only trace authority')
    require(ready['complete'] is True and ready['passed'] is False and ready['checks']==trace['checks'] and ready['failure_count']==564 and ready['exception'] is None and trace['failures'][:len(ready['failures'])]==ready['failures'] and len(ready['failures'])==100,'original capped100 failure log preserved')
    marker=trace['failure_trace'];require(marker==tp['failure_trace']==ts['failure_trace'] and marker['logging_guard_AST_changes']==1 and marker['scalar_formula_threshold_or_acceptance_AST_changes']==0 and marker['whole_original_scalar_source_restore_proven'] is True and marker['geometry_solver_or_operating_rows_modified'] is False,'only logging cap changed')
    require(saved['scope']==check['scope'] and saved['scope']['total_failed_predicates']==saved['scope']['retained_failed_predicates']==564 and saved['scope']['retained_prefix_not_complete'] is False and check['retained_failures_joined']==564 and check['rows_streamed']==490,'all564 joined from all490 saved rows')
    require([v['stored_failure'] for v in saved['witnesses']]==trace['failures'] and len(saved['witnesses'])==564 and check['unique_candidate_contexts']==len(saved['contexts']),'original failed actual/expected values and context population')
    return saved,trace
def compact_context(v,key):
    c=v['candidate'];decision=v['candidate_acceptance_decision_stored'];require(type(decision) is bool,'stored candidate acceptance presence')
    phase=key.rsplit('.',2)[1];require(phase in ('proposal','refit'),'saved checker proposal/refit context phase')
    if decision:require(type(c['accepted']) is bool,'stored candidate acceptance boolean')
    status='ACCEPTED_STORED' if decision and c['accepted'] else 'REJECTED_STORED' if decision else 'NO_STORED_ACCEPTANCE_DECISION'
    return dict(id=v['id'],session=v['session'],method=v['method'],phase=phase,stored_candidate_phase=c['phase'],candidate_index=c.get('candidate_index'),proposal_index=c.get('proposal_index'),
        candidate_acceptance_status=status,stored_rejection_reason=c.get('rejected'),row_output_status=v['row_output_status'],solver_state=v['solver_state'],solver_available=v['solver_available'],
        best_representative=v['candidate_is_solver_best_representative'],approved_NEW=v['candidate_is_approved_NEW_output'],proposal_is_best_source=v['proposal_is_best_representative_source'],
        stored_fit_point_ids=c['fit_input_ids'],stored_fit_line_edges=c['fit_line_edges'],stored_final_inlier_ids=c.get('final_inliers'),stored_final_line_inliers=c.get('final_line_inliers'),
        stored_dimension_index=c['dimension_index'],stored_cf_extents=c['cf_extents'],stored_centroid=c['centroid'],
        stored_signed_model_camera_depths_m=c.get('model_camera_depths_m'),stored_projected_finite_segments=v['stored_projected_finite_segments'],
        full_original_context_semantic_sha256=semantic(v),no_projection_or_depth_regeneration=True)
def row_type(location):
    partition=location['row_partition']
    if partition is not None:
        require(partition['kind'] in ('POINT_XY','LINE_NORMAL','LONGITUDINAL_HINGE'),'known source finite row type');return partition['kind']
    if 'independent_column_norms' in location['label_suffix'].split('.'):return 'UNMAPPED_COLUMN_NORM'
    return 'MAPPED_EDGE_METADATA' if location['scalar_path_mapped'] else 'UNMAPPED_OTHER'
def summarize(saved):
    contexts={k:compact_context(v,k) for k,v in saved['contexts'].items()};require(all(v['method'] in METHODS for v in contexts.values()),'known paired method')
    groups={};method_counts={m:Counter() for m in METHODS};out=[];seen_contexts=set()
    def add(dimension,key,w,c):
        group=groups.setdefault(dimension,{}).setdefault(str(key),dict(failures=0,context_keys=set(),frame_ids=set(),frame_method_keys=set(),retained_log_positions=[]))
        group['failures']+=1;group['context_keys'].add(w['context_key']);group['frame_ids'].add(c['id']);group['frame_method_keys'].add((c['method'],c['id']));group['retained_log_positions'].append(w['retained_log_position'])
    for position,w in enumerate(saved['witnesses']):
        require(w['retained_log_position']==position and w['context_key'] in contexts,'ordered original failure/context join');c=contexts[w['context_key']];kind=row_type(w['location']);failure=w['stored_failure'];seen_contexts.add(w['context_key'])
        expected=c['best_representative'] and c['row_output_status']=='NEW_POSE';require(c['approved_NEW'] is expected,'stored approved NEW vs best representative')
        method_counts[c['method']]['failed_predicates']+=1;method_counts[c['method']]['approved_NEW_failed_predicates']+=int(c['approved_NEW']);method_counts[c['method']]['accepted_failed_predicates']+=int(c['candidate_acceptance_status']=='ACCEPTED_STORED')
        categories=dict(method=c['method'],phase=c['phase'],candidate_acceptance=c['candidate_acceptance_status'],rejection=c['stored_rejection_reason'] or 'NONE_RECORDED',
            solver_state=c['solver_state'],best_representative=c['best_representative'],approved_NEW=c['approved_NEW'],proposal_is_best_source=c['proposal_is_best_source'],
            finite_row_type=kind,scalar_mapping=w['location']['scalar_path_mapped'],normal_incidence_compact_rows=w['location']['normal_incidence_compact_rows'])
        for dimension,key in categories.items():add(dimension,key,w,c)
        out.append(dict(retained_log_position=position,context_key=w['context_key'],id=c['id'],method=c['method'],phase=c['phase'],
            failing_finite_row_type=kind,stored_failure=failure,stored_location=w['location'],
            stored_compared_scalar_equals_recorded_actual=w['stored_compared_scalar_equals_recorded_actual'],
            candidate_acceptance_status=c['candidate_acceptance_status'],stored_rejection_reason=c['stored_rejection_reason'],
            best_representative=c['best_representative'],approved_NEW=c['approved_NEW'],proposal_is_best_source=c['proposal_is_best_source']))
    require(seen_contexts==set(contexts),'every saved candidate context represented')
    result_groups={d:{k:dict(failed_predicates=g['failures'],unique_candidate_contexts=len(g['context_keys']),context_keys=sorted(g['context_keys']),
        unique_frames=len(g['frame_ids']),frame_ids=sorted(g['frame_ids']),unique_frame_method_pairs=len(g['frame_method_keys']),
        frame_method_pairs=[list(v) for v in sorted(g['frame_method_keys'])],retained_log_positions=g['retained_log_positions']) for k,g in values.items()} for d,values in groups.items()}
    for m in METHODS:
        selected={k:v for k,v in contexts.items() if v['method']==m};method_counts[m].update(unique_candidate_contexts=len(selected),unique_frames=len({v['id'] for v in selected.values()}),
            accepted_contexts=sum(v['candidate_acceptance_status']=='ACCEPTED_STORED' for v in selected.values()),rejected_contexts=sum(v['candidate_acceptance_status']=='REJECTED_STORED' for v in selected.values()),
            best_representative_contexts=sum(v['best_representative'] for v in selected.values()),approved_NEW_contexts=sum(v['approved_NEW'] for v in selected.values()),proposal_best_source_contexts=sum(v['proposal_is_best_source'] for v in selected.values()))
    return dict(contexts=contexts,failed_predicates=out,groups=result_groups,methods={m:dict(v) for m,v in method_counts.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args()
    require(args.output.is_dir(),'explicit existing output directory')
    for name in FILES if args.stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)
    f=paths(args.input);b={k:binding(v) for k,v in f.items()}
    if args.stage=='freeze':write(args.output/FILES[0],dict(schema='saved_failure_witness_summary_protocol_v16',inputs=b,limits=LIMITS,frozen_before_saved_number_comparisons=True));return
    own=read(args.output/FILES[0]);require(own['inputs']==b and own['limits']==LIMITS and own['frozen_before_saved_number_comparisons'] is True,'unchanged own freeze');ob=binding(args.output/FILES[0])
    write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_numeric_geometry_calls=0));error=None;summary=None
    try:
        saved,trace=authority(f,b);summary=summarize(saved);require({k:binding(v) for k,v in f.items()}==b,'all saved inputs unchanged')
        write(args.output/FILES[2],dict(schema='saved_failure_witness_summary_v16',complete=True,own_protocol=ob,inputs=b,limits=LIMITS,
            original_scalar_validation_passed=False,original_scalar_checks=trace['checks'],original_scalar_failures=trace['failure_count'],
            geometry_or_operating_outputs_changed=False,**summary))
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    write(args.output/FILES[3],dict(schema='saved_failure_witness_summary_checks_v16',complete=error is None,passed=error is None,own_protocol=ob,inputs=b,limits=LIMITS,
        original_geometry_scalar_passed=False,original_geometry_scalar_failure_count=564,error=error,first_failure_preserved=True,
        summary=binding(args.output/FILES[2]) if (args.output/FILES[2]).is_file() else None,retained_partial_summary=summary if error else None))
    require(error is None,'saved summary incomplete; preserve original data and first failure')
if __name__=='__main__':main()
