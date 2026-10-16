"""Own-frozen GT-free joins from stored scalar failures to stored candidates.

No projection, derivative, SVD, residual, score or tolerance is recomputed.
The default failed READY receipt retained its first 100 failures, not all 564.
For a later complete logging-only trace use --checks and a new output folder
for both freeze and run. This diagnostic PASS never means geometry PASS.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
PRIVATE=Path('/tmp/pallet-hybrid-factor-subset-consensus-private-20261011-v16')
METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
FILES=('FAILURE_WITNESS_PROTOCOL.json','FAILURE_WITNESS_STARTED.json','FAILURE_WITNESSES.json','FAILURE_WITNESS_CHECKS.json')
PATTERN=re.compile(r'^frame\.(\d+)\.('+ '|'.join(METHODS)+r')\.(proposal|refit)\.(\d+)\.(.+)$')
LIMITS=dict(new_model_GT_PnP_optimizer_projection_J_SVD_metric_timing_calls=0,
    numeric_geometry_recomputed=False,stored_actual_expected_unchanged=True,
    physical_ownership_or_correct_pose_certified=False,geometry_PASS_claimed=False,
    original_receipts_rows_or_policy_changed=False,automatic_retry=False)

def require(value,message):
    if not value:raise ValueError(message)

def safe(path):
    p=Path(path).absolute();require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink '+str(p));return p

def read(path):return json.loads(safe(path).read_text())

def bind(path):
    p=safe(path);require(p.is_file(),'missing '+str(p));h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))

def write(path,value):
    with safe(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def declared_path(value,parent):
    p=Path(value['path'])
    if p.is_absolute():return safe(p)
    require('..' not in p.parts,'unsafe declared path')
    return safe(REPO/p if p.parts[0] in ('scripts','_docs') else parent/p)

def inputs(args):
    p=args.input;chosen=read(args.checks)
    f=dict(script=Path(__file__).resolve(),geometry=p/'GEOMETRY_SEALED.jsonl.gz',
        geometry_seal=p/'GEOMETRY_SEAL.json',accuracy_protocol=p/'PROTOCOL.json',execution_receipt=p/'CONTROL_RECEIPT.json',
        ready_protocol=p/'VALIDATION_READY_PROTOCOL.json',ready_started=p/'VALIDATION_READY_STARTED.json',
        ready_checks=p/'VALIDATION_READY_CHECKS.json',chosen_failure_checks=args.checks,
        chosen_failure_protocol=declared_path(chosen['own_protocol'],args.checks.parent),
        immutable_checker=CODE/'validation_checks.py',ready_checker=CODE/'validation_environment_ready.py',
        ready_authority=CODE/'validation_authority_ready.py',candidate_label_source=REPO/'scripts/research/pallet_factor_subset_consensus_20261011_v15/validation_checks.py',
        partition_source=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py')
    ready=read(f['ready_protocol']);f['cohort']=declared_path(ready['inputs']['cohort'],p)
    if chosen['schema']=='hybrid_factor_subset_scalar_failure_trace_checks_v16':
        f.update(chosen_failure_started=args.checks.parent/'TRACE_VALIDATION_STARTED.json',
            trace_checker=CODE/'validation_failure_trace.py')
    return f,{k:bind(v) for k,v in f.items()}

def authority(f,b):
    ready,own,start,chosen,seal,receipt=[read(f[k]) for k in ('ready_checks','ready_protocol','ready_started','chosen_failure_checks','geometry_seal','execution_receipt')]
    require(ready['schema']=='hybrid_factor_subset_scalar_checks_v16' and ready['complete'] is True
        and ready['passed'] is False and ready['exception'] is None and ready['failure_count']>0,
        'complete failed READY receipt required, never a geometry PASS')
    require(ready['inputs']==own['inputs']==start['inputs'] and same(ready['own_protocol'],b['ready_protocol'])
        and same(start['own_protocol'],b['ready_protocol']) and same(ready['checker'],b['immutable_checker'])
        and same(own['checker'],b['immutable_checker'])
        and same(ready['ready_checker'],b['ready_checker']) and same(ready['authority_code'],b['ready_authority']),
        'current failed READY source and own protocol/STARTED bytes')
    require(ready['environment_ready']==own['environment_ready']==start['environment_ready']
        and ready['environment_ready']['scalar_formula_or_threshold_changes']==0
        and ready['new_production_calls']==0,'failed READY is environment-only, no new production calls')
    for key,alias in (('geometry','geometry'),('repair_protocol','accuracy_protocol'),('geometry_seal','geometry_seal'),('repair_receipt','execution_receipt'),('cohort','cohort')):
        require(same(own['inputs'][key],b[alias]) and same(chosen['inputs'][key],b[alias]),'same sealed failure input '+key)
    require(seal['schema']=='hybrid_factor_subset_geometry_seal_v16' and seal['complete'] is True
        and seal['frames']==245 and seal['rows']==490 and same(seal['geometry'],b['geometry'])
        and same(seal['protocol'],b['accuracy_protocol']) and same(seal['control_receipt'],b['execution_receipt'])
        and seal['cleanup_completed_before_seal'] is True and seal['GT_read_allowed'] is False
        and receipt['complete'] is True and receipt['cleanup_completed'] is True
        and receipt['error'] is None and receipt['cleanup_error'] is None,'complete sealed GT-free geometry and cleanup')
    require(chosen['complete'] is True and chosen['passed'] is False and chosen['exception'] is None
        and chosen['failure_count']==ready['failure_count'] and chosen['checks']==ready['checks']
        and same(chosen['own_protocol'],b['chosen_failure_protocol']),'chosen log preserves completed scalar failure')
    if chosen['schema']=='hybrid_factor_subset_scalar_failure_trace_checks_v16':
        require(chosen['trace_complete'] is True and chosen['trace_passed'] is True
            and chosen['scalar_validation_passed'] is False and chosen['all_failed_predicates_retained'] is True,
            'actual complete logging-only trace required')
        trace_own,trace_start=read(f['chosen_failure_protocol']),read(f['chosen_failure_started'])
        require(chosen['inputs']==trace_own['inputs']==trace_start['inputs']
            and same(trace_start['own_protocol'],b['chosen_failure_protocol'])
            and same(chosen['trace_checker'],b['trace_checker'])
            and same(trace_own['inputs']['trace_checker'],b['trace_checker'])
            and all(same(trace_own['inputs'][key],b[alias]) for key,alias in (
                ('completed_failed_ready_protocol','ready_protocol'),('completed_failed_ready_started','ready_started'),
                ('completed_failed_ready_checks','ready_checks'),('ready_environment_source','ready_checker'))),
            'trace own protocol/STARTED, source and preserved failed READY bytes')
        marker=chosen['failure_trace'];require(marker==trace_own['failure_trace']==trace_start['failure_trace']
            and marker['logging_guard_AST_changes']==1 and marker['scalar_formula_threshold_or_acceptance_AST_changes']==0
            and marker['whole_original_scalar_source_restore_proven'] is True
            and marker['geometry_solver_or_operating_rows_modified'] is False
            and chosen['failures'][:len(ready['failures'])]==ready['failures']
            and len(chosen['failures'])==chosen['failure_count'],'unchanged scalar predicates and original retained failure prefix')
    else:require(chosen['schema']==ready['schema'] and same(b['chosen_failure_checks'],b['ready_checks']),'original READY or complete trace only')
    failures=chosen['failures'];require(isinstance(failures,list) and len(failures)<=chosen['failure_count'],'retained failure log')
    return chosen,dict(total_failed_predicates=chosen['failure_count'],retained_failed_predicates=len(failures),
        retained_prefix_not_complete=len(failures)<chosen['failure_count'],original_READY_retained_predicates=len(ready['failures']),
        scalar_checks=chosen['checks'],scalar_validation_passed=False,limits=LIMITS)

def summary(candidate):
    keys=('phase','source','proposal_kind','proposal_index','candidate_index','logical_start_index','dimension_index',
        'selected_hypothesis','accepted','rejected','optimizer_success','fit_input_ids','fit_line_edges','final_inliers',
        'final_line_inliers','final_z','centroid','cf_extents','R_cf','R_physical','projected','model_camera_depths_m',
        'soft_l1_cost','reported_soft_l1_cost','sse_px2','truncated_sse_px2','score_count','fit_pool_all_final8px_inliers')
    return {k:candidate[k] for k in keys if k in candidate}

def scalar_partition(candidate,index):
    g=candidate['point_line_geometry'];points=g['point_ids']
    if index<2*len(points):return dict(kind='POINT_XY',point_id=points[index//2],coordinate=('x','y')[index%2],finite_scalar_row=index)
    for line_index,block in enumerate(g['edge_blocks']):
        if block['scalar_start']<=index<block['scalar_end']:
            relative=index-block['scalar_start'];part=dict(kind='LINE_NORMAL' if relative<2 else 'LONGITUDINAL_HINGE',
                finite_scalar_row=index,line_index=line_index,edge=block['edge'],endpoints=block['endpoints'],
                projected_segment_length_px=block['projected_segment_length_px'])
            if relative<2:part['endpoint_id']=block['endpoints'][relative]
            else:
                query=relative-2;part.update(support_query_position=query,support_query_id=block['support_query_ids'][query],
                    stored_branch=block['hinge_branches'][query],stored_longitudinal_support_px=block['longitudinal_support_px'][query],
                    stored_hinge_px=block['outside_hinges_px'][query],stored_support_xy=block['actual_support_xy'][query])
            return part
    raise ValueError('finite scalar row outside stored partition')

def locate(candidate,suffix):
    tokens=suffix.split('.');normal=tokens[0]=='normal_incidence'
    if normal:tokens=tokens[1:]
    field=tokens[0];container=candidate.get('stable_incidence_guard',{}) if normal else candidate['point_line_geometry']
    row=None;column=None;stored=None;association=None
    if field in ('independent_observed_normal_J','independent_modeled_normal_J') and len(tokens)==3:
        matrix_row,column=int(tokens[1]),int(tokens[2]);key=field.removeprefix('independent_');stored=container[key][matrix_row][column]
        row=container['incidence_source_row_indices'][matrix_row] if normal else matrix_row
    elif field in ('independent_actual_scalar_residuals_px','scalar_residuals_px') and len(tokens)==2:
        row=int(tokens[1]);stored=candidate['point_line_geometry']['actual_scalar_residuals_px'][row]
    elif field=='independent_edge_blocks' and len(tokens)>=3:
        line_index=int(tokens[1]);block=candidate['point_line_geometry']['edge_blocks'][line_index];key=tokens[2]
        association=dict(line_index=line_index,edge=block['edge'],projected_segment_length_px=block['projected_segment_length_px'])
        stored=block[key]
        if len(tokens)==4:
            query=int(tokens[3]);stored=stored[query]
            if key=='normal_signed_px':row=block['scalar_start']+query
            elif key in ('outside_hinges_px','longitudinal_support_px'):row=block['scalar_start']+2+query
    return dict(label_suffix=suffix,normal_incidence_compact_rows=normal,matrix_column=column,
        row_partition=scalar_partition(candidate,row) if row is not None else None,
        associated_line=association,stored_compared_scalar=stored,scalar_path_mapped=stored is not None)

def context(row,candidate,phase,index):
    solver=row['solver'];best=solver.get('best_candidate');representative=bool(best and phase=='refit' and best.get('candidate_index')==index)
    g=candidate['point_line_geometry'];blocks=g['edge_blocks'];projected=candidate['projected']
    return dict(id=row['id'],session=row['session'],method=row['method'],row_output_status=row['output_status'],
        solver_state=solver['state'],solver_available=solver['available'],candidate=summary(candidate),
        candidate_acceptance_decision_stored='accepted' in candidate,candidate_is_solver_best_representative=representative,
        candidate_is_approved_NEW_output=representative and row['new_pose_estimated'] is True,
        proposal_is_best_representative_source=bool(best and phase=='proposal' and best.get('proposal_index')==index),
        winner=summary(best) if best else None,hidden_initial=row['hidden_initial'],used_point_ids=solver['used'],
        stored_K=row['K'],stored_model_xyz=row['xyz'],
        stored_fit_point_xy=[dict(point_id=k,xy=row['input_points'][k]) for k in g['point_ids']],
        stored_fit_source_lines=[v for v in solver['actual_source_lines'] if v['edge'] in g['line_edges']],
        point_line_geometry=g,stable_incidence_guard={k:v for k,v in candidate.get('stable_incidence_guard',{}).items()
            if k not in ('selected_candidate','original_all_candidates')},
        stored_projected_finite_segments=[dict(edge=b['edge'],endpoints=b['endpoints'],
            projected_endpoint_xy=[projected[k] for k in b['endpoints']],
            projected_segment_length_px=b['projected_segment_length_px']) for b in blocks],
        integer_partition_only_no_geometry_recalculation=True,physical_ownership_truth='UNKNOWN')

def guard(args,stage):
    out=safe(args.output);require(out.is_dir() and (out==DOC or out.is_relative_to(PRIVATE)),'existing exclusive DOC/private output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(out/name).exists(),'preserve '+name)
    return out

def freeze(args):
    out=guard(args,'freeze');f,b=inputs(args);_,scope=authority(f,b)
    write(out/FILES[0],dict(schema='hybrid_stored_failure_witness_protocol_v16',inputs=b,scope=scope,
        limits=LIMITS,frozen_before_any_raw_decode=True,label_mapping='zero-based frame; fixed paired METHODS; proposal/refit array index'))
    print('FAILURE_WITNESS_FROZEN',scope['retained_failed_predicates'],flush=True)

def run(args):
    out=guard(args,'run');f,b=inputs(args);chosen,scope=authority(f,b);own=read(out/FILES[0]);require(own['inputs']==b and own['scope']==scope,'own failure witness freeze differs')
    write(out/FILES[1],dict(own_protocol=bind(out/FILES[0]),inputs=b,new_numeric_calls=0))
    result=dict(schema='hybrid_stored_failure_witness_checks_v16',complete=False,passed=False,own_protocol=bind(out/FILES[0]),
        inputs=b,scope=scope,limits=LIMITS,error=None);contexts={};witnesses=[]
    try:
        wanted={};cohort=read(f['cohort']);require(len(cohort['ids'])==245,'fixed cohort')
        for position,failure in enumerate(chosen['failures']):
            match=PATTERN.fullmatch(failure['check']);require(match is not None,'unmapped candidate failure label '+failure['check'])
            frame,method,phase,index,suffix=match.groups();frame,index=int(frame),int(index)
            require(0<=frame<245,'frame index outside cohort');wanted.setdefault((frame,method),[]).append((position,failure,phase,index,suffix))
        streamed=0
        with gzip.open(f['geometry'],'rt',encoding='utf-8') as source:
            for number,line in enumerate(source):
                row=json.loads(line);frame=number//2;require(frame<245 and row['id']==cohort['ids'][frame]
                    and row['method']==METHODS[number%2],'fixed complete paired row order');streamed+=1
                for position,failure,phase,index,suffix in wanted.get((frame,row['method']),[]):
                    array=row['solver']['proposal_solutions' if phase=='proposal' else 'all_candidate_solutions'];candidate=array[index]
                    require(candidate['proposal_index' if phase=='proposal' else 'candidate_index']==index,'candidate array index')
                    key=str(frame)+'.'+row['method']+'.'+phase+'.'+str(index)
                    if key not in contexts:contexts[key]=context(row,candidate,phase,index)
                    location=locate(candidate,suffix)
                    require(not location['scalar_path_mapped'] or location['stored_compared_scalar']==failure['actual'],
                        'mapped candidate scalar differs from retained actual '+failure['check'])
                    witnesses.append(dict(retained_log_position=position,stored_failure=failure,
                        context_key=key,location=location,stored_compared_scalar_equals_recorded_actual=(location['stored_compared_scalar']==failure['actual'])
                        if location['scalar_path_mapped'] else None))
        witnesses.sort(key=lambda v:v['retained_log_position']);require(streamed==490 and len(witnesses)==len(chosen['failures'])
            and [v['stored_failure'] for v in witnesses]==chosen['failures'],'all retained failures joined unchanged')
        require(inputs(args)[1]==b,'failure inputs changed during extraction')
        payload=dict(schema='hybrid_stored_failure_witnesses_v16',complete=True,scope=scope,contexts=contexts,witnesses=witnesses,
            limits=LIMITS,geometry_validation_passed=False,new_pose_outputs=0)
        write(out/FILES[2],payload);result.update(complete=True,passed=True,rows_streamed=streamed,
            retained_failures_joined=len(witnesses),unique_candidate_contexts=len(contexts),witnesses=bind(out/FILES[2]))
    except Exception as error:
        result['error']=dict(type=type(error).__name__,message=str(error));result['retained_joined_prefix']=witnesses
    write(out/FILES[3],result);print('FAILURE_WITNESS',result['passed'],len(witnesses),flush=True)
    if not result['passed']:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--input',type=Path,default=DOC);p.add_argument('--checks',type=Path,default=DOC/'VALIDATION_READY_CHECKS.json')
    p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
