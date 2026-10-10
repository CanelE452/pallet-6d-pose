"""Own-frozen stdlib audit of the fresh safe PRIMARY runtime records.

Reconstruct recorded clock deltas, count joins and measured timing summaries.
No inference, image, GT, solver, optimizer, Jacobian, SVD or new timing calls.
Stored parity/projection witnesses do not prove hardware or physical ownership.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
PRIVATE=Path('/tmp/pallet-endpoint-observability-repair-private-20261011-v11')
STAGES=('detector','N3_correction','initial_pose','boundary_observation','endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
METHOD='FILTERED_LOCAL_WEAK_SAFE';TIMES=('full',*STAGES,'return');POSE=('R_cf','R_physical','centroid','cf_extents')
OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
LIMITS=dict(new_inference_RGB_GT_PnP_optimizer_Jacobian_SVD_timing_calls=0,recorded_clock_and_counter_consistency_only=True,
    candidate_pose_and_SVD_not_recomputed=True,physical_ownership_certified=False,global_uniqueness_proven=False,
    hardware_authenticity_proven=False,quiet_snapshots_can_miss_transient_competition=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p);require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(d):return {k:v for k,v in d.items() if v}
def describe(values):
    x=sorted(values);n=len(x);mean=math.fsum(x)/n;var=math.fsum((v-mean)**2 for v in x)/(n-1)
    def q(p):a=(n-1)*p;i=math.floor(a);return x[i]+(x[min(i+1,n-1)]-x[i])*(a-i)
    return dict(n=n,mean=mean,sample_variance=var,sample_std=math.sqrt(var),median=q(.5),P90=q(.9),maximum=x[-1],unit='ms',ddof=1)
class Audit:
    def __init__(self):self.checks=0;self.failure_count=0;self.failures=[];self.max_difference=0.
    def eq(self,a,b,label):
        self.checks+=1
        if a!=b:
            self.failure_count+=1
            if len(self.failures)<100:self.failures.append(dict(check=label,actual=a,expected=b))
    def near(self,a,b,label,tol=1e-7,relative=1e-12):
        if isinstance(b,dict):
            self.eq(set(a)==set(b),True,label+'.keys')
            for k in b:self.near(a[k],b[k],label+'.'+k,tol,relative)
        elif isinstance(b,list):
            self.eq(len(a),len(b),label+'.length')
            for i,(v,w) in enumerate(zip(a,b)):self.near(v,w,label+'.'+str(i),tol,relative)
        elif type(b) in (int,float):
            require(type(a) in (int,float) and math.isfinite(a) and math.isfinite(b),'nonfinite scalar');d=abs(a-b)
            self.max_difference=max(self.max_difference,d);self.eq(d<=tol+relative*abs(b),True,label)
        else:self.eq(a,b,label)
def paths(p,t):
    runtime={'runtime_protocol':'RUNTIME_PROTOCOL.json','receipt':'RUNTIME_RECEIPT.json','resources':'RUNTIME_RESOURCES.json','rows':'RUNTIME_ROWS.jsonl.gz','preflight':'RUNTIME_PREFLIGHT.json','journal':'RUNTIME_STARTED.json'}
    names={'geometry':'GEOMETRY_SEALED.jsonl.gz','core_protocol':'PROTOCOL.json','review':'REVIEW_VALIDATION_CHECKS.json','review_protocol':'REVIEW_VALIDATION_PROTOCOL.json','original_failure':'VALIDATION_CHECKS.json'}
    result={k:p/v for k,v in names.items()};result.update({k:t/v for k,v in runtime.items()});result['checker']=Path(__file__).resolve();result['fixed_geometry']=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz'
    for name in ('runtime.py','deployment.py','pipeline.py','evaluation_review.py','validation_review.py','runtime_review.py','deployment_review.py'):result['code:'+name]=Path(__file__).with_name(name)
    for name in ('RUNTIME_PROTOCOL.json','RUNTIME_PREFLIGHT.json','RUNTIME_STARTED.json','RUNTIME_RECEIPT.json','RUNTIME_RESOURCES.json','INTERRUPTED_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_PACKET.json'):result['preserved:'+name]=p/name
    result['review_contract']=p/'RUNTIME_REVIEW_CONTRACT_KO.md'
    return result
def metadata(files,b):
    proto,r,res=read(files['runtime_protocol']),read(files['receipt']),read(files['resources'])
    require(proto['schema']=='fresh_safe_primary_runtime_protocol_v11' and proto['method']==METHOD and proto['configured_calls']==150 and
        proto['warmup']==20 and proto['measured']==130 and proto['stages']==list(STAGES),'fixed runtime protocol differs')
    require(r['schema']=='fresh_safe_primary_runtime_receipt_v11' and r['complete'] is True and r['status']=='COMPLETE' and
        r['statistics_official'] is True and r['serialized_rows']==150 and not r.get('error') and not r['cleanup_errors'],'full successful runtime required')
    for k,a in (('protocol','runtime_protocol'),('accuracy_protocol','core_protocol'),('raw_rows','rows'),('resources','resources')):require(same(r[k],b[a]),'runtime byte authority '+k)
    require(same(proto['inputs']['runtime:code'],b['code:runtime.py']) and same(proto['inputs']['runtime:deployment'],b['code:deployment.py']) and
        same(proto['inputs']['runtime:GEOMETRY_SEALED.jsonl.gz'],b['geometry']),'frozen code/geometry differs')
    old,failed_runtime=read(files['preserved:RUNTIME_PROTOCOL.json']),read(files['preserved:RUNTIME_RECEIPT.json'])
    require(failed_runtime['complete'] is False and failed_runtime['status']=='FAILED_RUNTIME' and failed_runtime['serialized_rows']==0 and failed_runtime['summaries'] is None and failed_runtime['execution']['calls_started']==1 and failed_runtime['execution'].get('calls_complete',0)==0 and not failed_runtime['cleanup_errors'] and failed_runtime['error']['type']=='TypeError' and failed_runtime['error']['message']=="dict() got multiple values for keyword argument 'id'" and same(failed_runtime['protocol'],b['preserved:RUNTIME_PROTOCOL.json']),'preserved original runtime failure differs')
    extra={'review_code':'code:runtime_review.py','packet_adapter':'code:deployment_review.py','review_contract':'review_contract',**{k:k for k in b if k.startswith('preserved:')}}
    require(set(proto['inputs'])==set(old['inputs'])|{'metadata_review:'+k for k in extra} and all(proto['inputs'][k]==v for k,v in old['inputs'].items()) and all(same(proto['inputs']['metadata_review:'+k],b[v]) for k,v in extra.items()),'unchanged runtime closure or metadata supplement differs')
    for key in ('fixed_geometry','code:pipeline.py','code:evaluation_review.py','code:validation_review.py'):require(any(same(v,b[key]) and v['path']==b[key]['path'] for v in proto['inputs'].values()),'unbound original '+key)
    require(res['snapshots']==r['resource_snapshots'] and same(res['protocol'],b['runtime_protocol']) and res['all_recorded_snapshots_quiet'] is True,'resource identity differs')
    pre,journal=read(files['preflight']),read(files['journal']);require(pre['passed'] is True and same(pre['protocol'],b['runtime_protocol']) and
        journal['status']=='COMPLETE' and same(journal['receipt'],b['receipt']) and journal['execution']==r['execution'],'preflight/journal differs')
    review,own,failed=[read(files[k]) for k in ('review','review_protocol','original_failure')]
    require(review['complete'] is True and review['passed'] is True and review['inputs']==own['inputs'] and same(review['own_protocol'],b['review_protocol']) and
        failed['passed'] is False and failed['checks']==95 and failed['exception']['message']=="'best_candidate'" and
        same(proto['inputs']['runtime:REVIEW_VALIDATION_CHECKS.json'],b['review']),'preserved failure/joined review differs')
    return proto,r,res
def guard(args,stage):
    for p in (args.input,args.runtime_input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directory required')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'new V11 outputs only')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)
def freeze(args):
    guard(args,'freeze');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};metadata(f,b)
    write(args.output/FILES[0],dict(schema='independent_safe_primary_runtime_check_protocol_v11',inputs=b,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0])
    require(own['inputs']==b and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs');proto,r,res=metadata(f,b)
    write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0));a=Audit();totals={};timings={k:[] for k in TIMES};count=0;error=None
    try:
        panel=proto['panel'];a.eq(len(panel),26,'panel26');a.eq(len({v['session_id'] for v in panel}),13,'sessions13');a.eq(Counter(v['session_id'] for v in panel),{s:2 for s in {v['session_id'] for v in panel}},'two_per_session')
        authority={v['id']:v for v in proto['normalization']['joined_frame_identities']};a.eq(len(authority),245,'normalization245')
        schedule=[dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[dict(phase='measured',repeat=j,index=i,image_index=i) for j in range(5) for i in range(26)];a.eq(proto['schedule'],schedule,'actual_fixed_schedule')
        refs={};seen=set();fixed={};panelids={p['frame_id'] for p in panel}
        for v in rows(f['geometry']):
            key=(v['method'],v['id']);require(key not in seen and v['id'] in authority,'unknown/duplicate geometry');seen.add(key)
            if v['method']==METHOD and v['id'] in {p['frame_id'] for p in panel}:refs[v['id']]={k:v[k] for k in ('native_points','actual_pose','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata','calibration_binding','head_checkpoint_sha256')}
        a.eq(seen,{(m,i) for m in ('FILTERED_LOCAL',METHOD) for i in authority},'geometry490_full_pairs');a.eq(set(refs),{p['frame_id'] for p in panel},'panel_reference26')
        for v in rows(f['fixed_geometry']):
            if v['id'] in panelids:fixed[v['method'],v['id']]=v['native_points']
        a.eq(set(fixed),{(m,p) for m in ('BASE','N3_SUBPIX') for p in panelids},'original_Base_N3_panel')
        for i,v in enumerate(rows(f['rows'])):
            require(i<150,'extra runtime row');job=schedule[i];label='row.'+str(i)
            for k,w in job.items():a.eq(v[k],w,label+'.schedule.'+k)
            p=panel[job['image_index']];a.eq((v['id'],v['session'],v['method']),(p['frame_id'],p['session_id'],METHOD),label+'.identity')
            ticks=v['stage_ticks_ns'];a.eq([t[0] for t in ticks],list(STAGES),label+'.six_stages');ns=[v['t0_ns'],*[t[1] for t in ticks],v['t1_ns']]
            require(all(type(t) is int for t in ns) and all(y>x for x,y in zip(ns,ns[1:])),'invalid monotonic ticks')
            for k,x,y in zip((*STAGES,'return'),ns,ns[1:]):a.near(v[k+'_ms'],(y-x)/1e6,label+'.clock.'+k)
            a.near(v['full_ms'],(ns[-1]-ns[0])/1e6,label+'.full_clock');a.near(v['full_ms'],math.fsum(v[k+'_ms'] for k in (*STAGES,'return')),label+'.full_stage_sum')
            for k in TIMES:
                require(type(v[k+'_ms']) in (int,float) and math.isfinite(v[k+'_ms']) and v[k+'_ms']>0,'nonpositive duration')
                if v['phase']=='measured':timings[k].append(v[k+'_ms'])
            o=v['full_fresh_output'];native=o['native_N3_points'];s=o['solver'];H=o['hidden_initial'];ledger=v['fresh_repair_ledger'];proposal=ledger['full_rank_only_solver']
            reference=refs[v['id']]
            for k,w in reference.items():
                if k!='actual_pose':a.near(o[k],w,label+'.saved_parity.'+k,relative=0)
            actual,expected_pose=o['actual_pose'],reference['actual_pose'];a.eq(actual['available'],expected_pose['available'],label+'.pose_available');a.eq(actual.get('selected_hypothesis'),expected_pose.get('selected_hypothesis'),label+'.pose_dimension')
            if actual['available']:
                for k in OPTIONAL:a.eq(k in actual,k in expected_pose,label+'.optional_pose_presence.'+k)
                for k in (*POSE,*(k for k in OPTIONAL if k in expected_pose)):a.near(actual[k],expected_pose[k],label+'.saved_Rt.'+k,relative=0)
            a.near(o['original_base_points'],fixed['BASE',v['id']],label+'.original_BASE',relative=0);a.near(native,fixed['N3_SUBPIX',v['id']],label+'.original_N3',relative=0)
            a.eq(o['K'],authority[v['id']]['K'],label+'.K');a.eq(o['xyz'],authority[v['id']]['xyz'],label+'.dims')
            for k,w in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True)):a.eq(v[k],w,label+'.'+k)
            a.eq(o['saved_accuracy_coordinates_or_pose_as_input'],False,label+'.no_replay');a.eq(o['fresh_capture_input_only'],True,label+'.fresh_input');a.eq(o['reprojections_reused_as_observations'],False,label+'.no_refit')
            a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_LOCAL');a.eq(ledger['reported_output_views'],2,label+'.two_views');a.eq(ledger['endpoint_decision_source'],'FRESH_NATIVE_ENDPOINT_LOO',label+'.fresh_endpoint')
            for c in proposal['all_candidate_solutions']:a.eq(bool(set(H)&set(c['fit_input_ids'])),False,label+'.candidate_H_exclusion')
            if o['new_pose_estimated']:
                a.eq(bool(set(H)&set(s['fit_input_ids'])),False,label+'.H_exclusion');a.eq(o['actual_pose'],s,label+'.actual_Rt')
                for k in H:a.near(o['native_points'][k],s['projected'][k],label+'.H_projection.'+str(k),1e-12)
            else:a.eq(o['actual_pose'],o['initial_pose'],label+'.full_fallback')
            a.eq(o['native_points'][8],native[8],label+'.center');expected=copy.deepcopy(v['original_fresh_detector_prediction']);selected=expected['selected_index']
            if selected is not None:a.eq(expected['candidates'][selected]['keypoints_xy'][8],native[8],label+'.raw_center');expected['candidates'][selected]['keypoints_xy']=o['native_points']
            a.eq(o['prediction']['selected_index'],selected,label+'.selected_index');a.eq(o['prediction']['candidates'],expected['candidates'],label+'.all_candidate_conf_box_preserved')
            d=v['counter_deltas'];a.eq(set(d),{'cv','optimizer','head','base','fresh','local','native','model'},label+'.counter_groups');a.eq(d['head'],dict(entries=1,returns=1),label+'.head');a.eq(d['model'],dict(detector=1,N3=1),label+'.models')
            a.eq(sparse(d['local']),sparse(ledger['local_operation_counts']),label+'.LOCAL_counts');a.eq(sparse(d['native']),sparse(ledger['endpoint_native_bank_delta']),label+'.native_counts')
            require(set(d['cv'])<={p+':'+k for p in CV for k in ('entries','returns','exceptions')},'unknown CV primitive');op=ledger['local_operation_counts'];a.eq(d['optimizer'].get('entries',0),op['local_optimizer_calls'],label+'.actual_LOCAL_optimizer')
            for key,source in (('returns','local_optimizer_completions'),('residual_entries','optimizer_residual_calls'),('jacobian_entries','optimizer_jacobian_calls')):a.eq(d['optimizer'].get(key,0),op[source],label+'.optimizer_ledger.'+key)
            for p in CV:a.eq(d['cv'].get(p+':entries',0),d['cv'].get(p+':returns',0)+d['cv'].get(p+':exceptions',0),label+'.CV_complete.'+p)
            a.eq(d['optimizer'].get('entries',0),d['optimizer'].get('returns',0)+d['optimizer'].get('exceptions',0),label+'.optimizer_complete');a.eq(v['fresh_point_bank_ledger']['coordinate_bank_count'],1,label+'.fresh_sparse_bank')
            for key,values in d.items():
                require(all(type(n) is int and n>=0 for n in values.values()),'invalid counter delta');totals.setdefault(key,Counter()).update(values)
            count+=1
        a.eq(count,150,'all150');a.eq(len(timings['full']),130,'measured130')
        for k in TIMES:a.near(describe(timings[k]),r['summaries'][k],'summary.'+k)
        for key,field in (('cv','live_cv_counts'),('optimizer','live_optimizer_counts'),('head','live_head_counts'),('base','actual_base_counts'),('fresh','actual_fresh_counts'),('local','local_operation_counts'),('native','native_operation_counts')):a.eq(sparse(totals[key]),sparse(r[field]),'actual.sum.'+key)
        for key in ('detector','N3'):a.eq(r['model_forwards'][key],r['model_forwards_after_loading'][key]+totals['model'][key],'actual.model.'+key)
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('image_decode_returns',26),('model_initializations_started',1),('model_initializations_complete',1)):a.eq(r['execution'][key],value,'actual.execution.'+key)
        for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored'):a.eq(r['cleanup'][k],True,'recorded.cleanup.'+k)
        a.eq(r['protected_before'],r['protected_after'],'recorded.protection');require(bool(res['snapshots']),'resource snapshots missing')
        for s in res['snapshots']:a.eq(s['quiet'] and s['temperature_under_80'],True,'recorded.quiet')
        a.eq({k:binding(p) for k,p in f.items()},b,'inputs_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and count==150 and a.failure_count==0
    write(args.output/FILES[2],dict(schema='independent_safe_primary_runtime_checks_v11',complete=error is None and count==150,passed=passed,inputs=b,own_protocol=ob,limits=LIMITS,
        checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,max_absolute_difference=a.max_difference,first_failure_preserved=True))
    print('SAFE_RUNTIME_CHECK','PASS' if passed else 'FAIL',a.checks,flush=True)
    if not passed:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--runtime-input',type=Path,default=DOC/'runtime_review');p.add_argument('--output',type=Path,default=DOC)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
