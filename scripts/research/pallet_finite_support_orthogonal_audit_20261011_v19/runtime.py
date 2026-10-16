"""Own frozen V19 full fresh path; saved geometry is post-return parity only.

ROOT alone executes this file. The original V16 deployment class is compiled
whole in an isolated namespace, dispatching its single filtered solve to V19.
No paired point ablation or saved pose is used by a prediction.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib
import io
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PACKAGE='scripts.research.pallet_finite_support_orthogonal_audit_20261011_v19'
PRIMITIVES=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
STAGES=('detector','N3_correction','initial_pose','boundary_observation','endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
SOURCE_SHA=dict(deployment='3d03d1073fd16629514665c5fe990e6f6b8826b8cd8ac9989705aac978485fe7',runtime='0465082112b0ceda8d00b204c7585d5c12218ada1118086b90abc061f28fa6dd')

def require(value,message):
    if not value:raise RuntimeError('V19_RUNTIME: '+message)

def binding(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=h.hexdigest())

def read(path):return json.loads(Path(path).read_bytes())
def same(left,right):return all(left[k]==right[k] for k in ('bytes','sha256'))

def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def source_paths(a):
    root=Path(a.repo)/'scripts/research'
    return dict(deployment=root/'pallet_hybrid_factor_subset_consensus_20261011_v16/deployment.py',
                runtime=root/'pallet_consensus_decision_repair_20261011_v18/runtime.py')

def source_nodes(a):
    paths=source_paths(a);result={}
    for key,names in (('deployment',('assemble_result','Pipeline')),('runtime',('schedule','instruments','delta'))):
        require(binding(paths[key])['sha256']==SOURCE_SHA[key],'immutable source '+key)
        tree=ast.parse(paths[key].read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        require({n.name for n in nodes}==set(names),'complete reused source '+key);result[key]=nodes
    return paths,result

def bookkeeping(a):
    paths,nodes=source_nodes(a)
    env=dict(__name__='isolated_v19_bookkeeping',__package__='scripts.research.pallet_consensus_decision_repair_20261011_v18',
             contextmanager=contextmanager,R=SimpleNamespace(PRIMITIVES=PRIMITIVES))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes['runtime'],type_ignores=[])),str(paths['runtime']),'exec'),env)
    return env,{k:{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in v} for k,v in nodes.items()}

def pipeline_class(a):
    parent=importlib.import_module('scripts.research.pallet_hybrid_factor_subset_consensus_20261011_v16.deployment')
    current=importlib.import_module(PACKAGE+'.pipeline');paths,nodes=source_nodes(a)
    dispatch=SimpleNamespace(PRIMARY=current.PRIMARY,repair_observations=current._isolated_repair)
    facade=SimpleNamespace(PRIMARY=current.PRIMARY,PARENT_METHOD=current.PARENT_METHOD,POLICY=current.POLICY,
        P15=dispatch,S4=current.P16.S4,PoseBank=current.P16.PoseBank,require=current.require)
    env=dict(vars(parent));env.update(P=facade,__name__='isolated_v19_fresh_deployment')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes['deployment'],type_ignores=[])),str(paths['deployment']),'exec'),env)
    cls=env['Pipeline'];original=cls.predict
    def predict(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        value.update(observed_frame_pipeline_source_proof=copy.deepcopy(current.SOURCE_PROOF),
            finite_observation_policy=copy.deepcopy(current.POLICY),pose_estimation_kind='HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS',
            new_finite_objective_is_not_historical_scalar_failure_repair=True,paired_point_accuracy_arm_executed_inside_primary_call=False)
        return value
    cls.predict=predict
    return cls,current.PRIMARY

def protect(a):
    result={}
    for key,root in (('research',a.repo),('original',a.source_root)):
        def git(*v):return subprocess.check_output(['git','-C',str(root),*v])
        result[key]=dict(HEAD=git('rev-parse','HEAD').decode().strip(),index=hashlib.sha256(git('ls-files','--stage','-z')).hexdigest(),
            diff=hashlib.sha256(git('diff','--binary')).hexdigest(),status=hashlib.sha256(git('status','--porcelain=v1','-uall')).hexdigest())
    return result

def gate(a):
    folder=Path(a.geometry);receipt=read(folder/'GEOMETRY_RECEIPT.json');check=read(Path(a.validation)/'VALIDATION_CHECKS.json')
    own=read(Path(a.validation)/'VALIDATION_PROTOCOL.json');started=read(Path(a.validation)/'VALIDATION_STARTED.json')
    require(receipt['complete'] is True and receipt['error'] is None and not receipt['cleanup_errors'] and receipt['inputs_unchanged']
        and receipt['actual_execution']==dict(frames_started=245,geometry_rows=490,ledger_rows=245,frames_complete=245),'completed paired geometry')
    for key,name in (('geometry','GEOMETRY.jsonl.gz'),('ledgers','LEDGERS.jsonl.gz'),('protocol','GEOMETRY_PROTOCOL.json')):
        require(same(receipt[key],binding(folder/name)),'geometry receipt '+key)
    require(own['schema']=='observed_frame_hybrid_scalar_protocol_v19' and check['schema']=='observed_frame_hybrid_scalar_checks_v19'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None
        and check['new_production_calls']==0 and all(check['actual_counts'][k]==v for k,v in dict(geometry_rows=490,ledger_rows=245,frames=245).items()),'actual independent full geometry PASS')
    require(check['old_full_V16_validation_passed'] is False and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True,'original full V16 FAIL remains distinct')
    require(check['inputs']==own['inputs']==started['inputs'] and same(check['own_protocol'],binding(Path(a.validation)/'VALIDATION_PROTOCOL.json'))
        and same(started['own_protocol'],check['own_protocol']) and started['new_production_calls']==0,'independent envelope')
    for key,name in (('geometry','GEOMETRY.jsonl.gz'),('ledgers','LEDGERS.jsonl.gz'),('geometry_protocol','GEOMETRY_PROTOCOL.json'),
                     ('geometry_started','GEOMETRY_STARTED.json'),('geometry_receipt','GEOMETRY_RECEIPT.json')):
        require(same(check['inputs'][key],binding(folder/name)),'independent input '+key)
    require(same(check['checker'],binding(a.validator)),'current independent checker')
    return receipt,check

def setup(a):
    sys.path.insert(0,str(a.repo));sys.dont_write_bytecode=True
    C=importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.common')
    V7=importlib.import_module('scripts.research.pallet_three_head_observation_20261010_v7.common')
    helpers=importlib.import_module('scripts.research.pallet_sparse_local_line_20261010_v9.runtime_review')
    previous=importlib.import_module('scripts.research.pallet_three_head_observation_20261010_v7.runtime')
    book,witness=bookkeeping(a);return C,V7,helpers,previous,book,witness

def prerequisites(a,C,V7,helpers):
    gate(a);eligible,normalization=helpers.normalized_frames(a);panel=read(C.CORRECTED/'RUNTIME_PANEL.json')['frames'];ids={r['frame_id'] for r in panel}
    require(len(panel)==len(ids)==26 and len({r['session_id'] for r in panel})==13 and Counter(r['session_id'] for r in panel)=={s:2 for s in {r['session_id'] for r in panel}},'fixed panel')
    methods=list(ast.literal_eval(next(n.value for n in ast.parse((Path(a.repo)/('scripts/research/'+PACKAGE.split('.')[-1]+'/pipeline.py')).read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='METHODS' for t in n.targets))))
    references={};counts=Counter();seen=set();fields=('native_points','actual_pose','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')
    for row in C.rows(Path(a.geometry)/'GEOMETRY.jsonl.gz'):
        key=(row['method'],row['id']);require(key not in seen and row['method'] in methods and row['id'] in eligible,'full geometry identity');seen.add(key);counts[row['method']]+=1
        if row['method']==methods[0] and row['id'] in ids:references[row['id']]={k:copy.deepcopy(row[k]) for k in fields}
    require(counts=={m:245 for m in methods} and set(references)==ids and all(eligible[r['frame_id']]['session']==r['session_id'] for r in panel),'full245/26 reference')
    return panel,eligible,references,normalization,methods[0]

def inputs(a,V7):
    paths=dict(code=Path(__file__),validator=Path(a.validator),**source_paths(a))
    for name in ('GEOMETRY_PROTOCOL.json','GEOMETRY_STARTED.json','GEOMETRY_RECEIPT.json','GEOMETRY.jsonl.gz','LEDGERS.jsonl.gz'):paths['geometry:'+name]=Path(a.geometry)/name
    for name in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):paths['validation:'+name]=Path(a.validation)/name
    for p in sorted((Path(a.repo)/('scripts/research/'+PACKAGE.split('.')[-1])).glob('*.py')):paths['new_source:'+p.name]=p
    for key,expected in read(Path(a.geometry)/'GEOMETRY_PROTOCOL.json')['inputs'].items():
        if key.startswith('source:'):
            path=Path(a.repo)/key[len('source:'):];require(same(binding(path),expected),'CPU/geometry-tested source closure '+key);paths['tested:'+key]=path
    result={key:binding(p) for key,p in paths.items()};result.update({'fresh_v7:'+key:value for key,value in V7.inputs(a).items()})
    return result

@contextmanager
def no_geometry_reads(a):
    old=[(builtins,'open',builtins.open),(io,'open',io.open)];roots=(Path(a.geometry).resolve(),Path(a.validation).resolve())
    def wrap(fn):
        def call(path,*args,**kw):
            if isinstance(path,(str,Path)):require(not any(Path(path).resolve().is_relative_to(root) for root in roots),'saved geometry entered inference')
            return fn(path,*args,**kw)
        return call
    try:
        for owner,key,fn in old:setattr(owner,key,wrap(fn))
        yield
    finally:
        for owner,key,fn in old:setattr(owner,key,fn)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','preflight','measure'))
    for key in ('repo','private','geometry','validation','validator','source-root','baseline-root','fits','cohort','calibration','calibration-root','prior-bindings'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--base-weights');parser.add_argument('--N3-weights');parser.add_argument('--remaining-seconds',type=float,default=3600.)
    a=parser.parse_args()
    for key in ('repo','private','geometry','validation','validator','source_root','baseline_root','fits','cohort','calibration','calibration_root','prior_bindings'):setattr(a,key,getattr(a,key).resolve())
    a.output=str(a.private/'runtime');out=Path(a.output);proto=out/'RUNTIME_PROTOCOL.json'
    require(not any(p.is_symlink() for p in (out,*out.parents)) and not any(out.is_relative_to(p) or p.is_relative_to(out) for p in (a.repo,a.source_root,a.baseline_root,a.fits,a.geometry,a.validation)),'new disjoint runtime output')
    C,V7,H,old,book,witness=setup(a);panel,eligible,refs,normalization,primary=prerequisites(a,C,V7,H);bound=inputs(a,V7)
    if a.phase=='freeze':
        require(not out.exists(),'fresh absent runtime folder');out.mkdir()
        write(proto,dict(schema='fresh_observed_frame_primary_runtime_protocol_v19',inputs=bound,method=primary,stages=list(STAGES),schedule=book['schedule'](),
            warmup=20,measured=130,configured_calls=150,panel=panel,normalization=normalization,bookkeeping_AST=witness,protection=protect(a),
            geometry_used_only_after_return=True,paired_point_only_accuracy_arm_not_timed=True,automatic_retry=False,new_RGB=0,new_training=0,
            first_predict_detector_forwards=2,other_predict_detector_forwards=1,N3_forwards_per_prediction=1,no_artificial_pre_forward=True,
            included='fresh detector/N3/SubPix/initial and Base-feature poses/ROLE+CAL/H-union-k/pointPnP/native endpoint LOO/one observed-frame consensus solve/H projection/metadata',
            excluded=['model loading','file decode','saved parity','resource and protection probes','logging','GT scoring']))
        print('V19_RUNTIME_FROZEN',flush=True);return
    protocol=read(proto);require(protocol['inputs']==bound and protocol['schedule']==book['schedule']() and protocol['method']==primary and protocol['bookkeeping_AST']==witness and protocol['protection']==protect(a),'frozen runtime inputs/protection')
    if a.phase=='preflight':
        resource=C.quiet_snapshot();write(out/'RUNTIME_PREFLIGHT.json',dict(protocol=binding(proto),passed=bool(resource['quiet'] and resource['temperature_under_80']),resource=resource,actual_model_fit_decode_calls=0));require(resource['quiet'] and resource['temperature_under_80'],'quiet preflight');return
    pre=read(out/'RUNTIME_PREFLIGHT.json');require(pre['passed'] is True and same(pre['protocol'],binding(proto)),'own preflight PASS')
    for name in ('RUNTIME_STARTED.json','RUNTIME_RECEIPT.json','RUNTIME_ROWS.jsonl.gz','PENDING_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_PACKET.json','RUNTIME_RESOURCES.json'):require(not(out/name).exists(),'preserve '+name)
    import cv2
    import numpy as np
    from scripts.research.pallet_three_head_observation_20261010_v7.run import RowWriter
    result=dict(schema='fresh_observed_frame_primary_runtime_receipt_v19',complete=False,status='FAILED_RUNTIME',protocol=binding(proto),method=primary,configured_calls=150,normalization=normalization)
    write(out/'RUNTIME_STARTED.json',dict(protocol=binding(proto),inputs=bound,status='STARTED'));before=protect(a);state=Counter();cv=Counter();opt=Counter();svd=Counter();head=Counter();resources=[];cleanup={};errors=[];rows=[];pipeline=writer=active=None;written=False;hooks=[];began=time.monotonic();environment={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')}
    def probe(phase):
        r=C.quiet_snapshot();r['phase']=phase;resources.append(r);require(r['quiet'] and r['temperature_under_80'],'quiet/thermal resource guard')
    try:
        probe('before_decode');images=[]
        for p in panel:
            f=eligible[p['frame_id']];path=a.source_root/f['image'];require(V7.sha(path)==f['image_sha256'],'original RGB SHA');image=cv2.imread(str(path));require(image is not None and list(image.shape[:2])==f['raw_hw'],'original RGB shape');images.append(image);state['image_decode_calls']+=1
        writer=RowWriter(out/'RUNTIME_ROWS.jsonl.gz',interrupted_path=out/'INTERRUPTED_RUNTIME_ROWS.jsonl.gz')
        with C.legacy_context(a):
            cls,label=pipeline_class(a);require(label==primary,'single new primary')
            with V7.inference_canary():pipeline=cls(a)
            torch=pipeline.torch;require(torch.cuda.is_available(),'fixed GPU path');state['model_initializations']=1;loaded=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards);require(loaded==dict(detector=0,N3=0),'no loading forwards')
            module=pipeline.heads['IMAGE_ROLE'];hooks=[module.register_forward_pre_hook(lambda *v:head.update(entries=1)),module.register_forward_hook(lambda *v:head.update(returns=1))];probe('after_models');torch.cuda.reset_peak_memory_stats();last_probe=time.monotonic()
            with book['instruments'](cv,opt,svd,cleanup),V7.inference_canary(),no_geometry_reads(a),pipeline.old.no_truth_reads(),torch.no_grad():
                for job in book['schedule']():
                    require(time.monotonic()-began<a.remaining_seconds,'time budget, preserve prefix');active=None;written=False
                    if time.monotonic()-last_probe>=5 or job['phase']=='measured' and job['image_index']==0:probe('during_paths');last_probe=time.monotonic()
                    frame=eligible[panel[job['image_index']]['frame_id']];state['calls_started']+=1;old.durable_owned(out/'RUNTIME_STARTED.json',dict(protocol=binding(proto),inputs=bound,status='RUNNING',active_job=job,execution=dict(state)))
                    groups=dict(cv=cv,optimizer=opt,svd=svd,head=head,base=pipeline.base.counts,fresh=pipeline.counts,local=pipeline.local_operation_counts,native=pipeline.endpoint_operation_counts,direct=pipeline.direct_point_operation_counts,bank_capture=pipeline.point_bank_capture_counts,native_reuse=pipeline.native_bank_reuse_counts);counters={k:v.copy() for k,v in groups.items()};model_before=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards);ticks=[]
                    def mark(stage):torch.cuda.synchronize();ticks.append((stage,time.perf_counter_ns()))
                    torch.cuda.synchronize();t0=time.perf_counter_ns();output=pipeline.predict(images[job['image_index']],frame['K'],frame['xyz'],{k:frame[k] for k in ('id','session','object_type')},method=primary,stage_callback=mark);torch.cuda.synchronize();t1=time.perf_counter_ns()
                    times={};previous=t0
                    for stage,tick in ticks:times[stage+'_ms']=(tick-previous)/1e6;previous=tick
                    times['return_ms']=(t1-previous)/1e6;d={k:book['delta'](v,counters[k]) for k,v in groups.items()};d['model']=book['delta'](Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards),model_before);require([s for s,_ in ticks]==list(STAGES) and d['head']==dict(entries=1,returns=1),'stage/head count');require(d['model']==dict(detector=2 if state['calls_started']==1 else 1,N3=1),'first2/rest1 detector, N3=1')
                    active=dict(**job,id=frame['id'],session=frame['session'],method=primary,t0_ns=t0,t1_ns=t1,stage_ticks_ns=ticks,full_ms=(t1-t0)/1e6,**times,full_fresh_output=output,original_fresh_detector_prediction=copy.deepcopy(pipeline.last_capture['raw']),fresh_point_bank_ledger=copy.deepcopy(pipeline.last_point_bank_ledger),fresh_point_bank_reuse=copy.deepcopy(pipeline.last_point_bank_reuse),fresh_repair_ledger=copy.deepcopy(pipeline.last_repair_ledger),execution_events=copy.deepcopy(pipeline.last_execution_events),counter_deltas=d,parity_status='PENDING')
                    reference=refs[frame['id']];active['points_parity']=pipeline.runtime._point_parity(output['native_points'],reference['native_points'],1e-7,(primary,frame['id']));active['pose_parity']=H.pose_parity(output['actual_pose'],reference['actual_pose'])
                    require(all(C.finite(output[k])==C.finite(reference[k]) for k in ('hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')),'outcome parity')
                    for arm,key in (('BASE','original_base_points'),('N3_SUBPIX','native_N3_points')):active[arm+'_parity']=pipeline.runtime._point_parity(output[key],frame['points'][arm],1e-7,(arm,frame['id']))
                    ledger=pipeline.last_repair_ledger;reuse=pipeline.last_point_bank_reuse;operations=ledger['local_operation_counts'];require(ledger['actual_filtered_local_solve_calls']==ledger['reported_output_views']==1 and ledger['endpoint_decision_source']=='FRESH_NATIVE_ENDPOINT_LOO','single primary actual solve')
                    require(d['local']==operations and d['direct']==reuse['direct_bank_delta'] and d['native']==reuse['endpoint_bank_delta'] and reuse['constructor_restored'] and reuse['endpoint_constructor_restored'] and reuse['same_coordinate_bank_reused'] and reuse['additional_selected_Generic_calls']==0,'bank phase/candidate ledger')
                    require(d['optimizer'].get('entries',0)==operations['proposal_optimizer_calls']+operations['refit_optimizer_calls']==operations['local_optimizer_calls'] and d['optimizer'].get('returns',0)==operations['local_optimizer_completions'] and d['optimizer'].get('exceptions',0)==operations['proposal_optimizer_exceptions']+operations['refit_optimizer_exceptions'],'optimizer dispatch')
                    require(d['optimizer'].get('returns',0)==operations['proposal_optimizer_returns']+operations['refit_optimizer_returns'] and d['optimizer'].get('residual_entries',0)==operations['optimizer_residual_calls'] and d['optimizer'].get('jacobian_entries',0)==operations['optimizer_jacobian_calls'],'proposal/refit callback ledger')
                    if output['new_pose_estimated']:require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']),'H fit leak');np.testing.assert_allclose(np.asarray(output['native_points'])[output['hidden_initial']],np.asarray(output['solver']['projected'])[output['hidden_initial']],rtol=0,atol=1e-12)
                    active.update(parity_status='PASS',saved_geometry_used_only_after_return=True);written=True;writer.write(active);rows.append({k:active[k] for k in ('phase','full_ms',*(s+'_ms' for s in STAGES),'return_ms')});active=None;state['calls_complete']+=1
                    if state['calls_complete']%25==0:print('V19_FRESH_RUNTIME',state['calls_complete'],150,flush=True)
            probe('after_fresh_paths');require(state['calls_complete']==150 and head==dict(entries=150,returns=150),'actual150 complete')
            for k in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):require(pipeline.base.counts[k]==150,'base path '+k)
            for k in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_repair_paths','fresh_filtered_local_paths'):require(pipeline.counts[k+'_started']==pipeline.counts[k+'_complete']==150,'fresh path '+k)
            require(pipeline.models.detector_forwards==151 and pipeline.models.n3_forwards==150 and pipeline.counts['reported_primary_views_from_single_filtered_local']==150,'actual model/output totals')
            require(opt['entries']==opt['returns']+opt['exceptions'] and svd['entries']==svd['returns']+svd['exceptions'] and all(cv[n+':entries']==cv[n+':returns']+cv[n+':exceptions'] for n in PRIMITIVES),'actual entry/return/exception totals')
            require(all(opt[k+'_entries']==opt[k+'_returns']+opt[k+'_exceptions'] for k in ('residual','jacobian')),'actual callback accounting')
            result.update(complete=True,status='COMPLETE',peak_GPU_bytes=torch.cuda.max_memory_allocated(),model_forwards_after_loading=dict(loaded))
    except BaseException as exc:result.update(complete=False,status='FAILED_RUNTIME',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        for hook in hooks:
            try:hook.remove()
            except BaseException as exc:errors.append(dict(phase='hook_remove',type=type(exc).__name__,message=str(exc)))
        cleanup['head_hooks_removed']=not any(e['phase']=='hook_remove' for e in errors)
        if pipeline is not None:
            result.update(model_bindings=pipeline.bindings,model_forwards=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards),actual_base_counts=dict(pipeline.base.counts),actual_fresh_counts=dict(pipeline.counts),local_operation_counts=dict(pipeline.local_operation_counts),native_operation_counts=dict(pipeline.endpoint_operation_counts),direct_point_operation_counts=dict(pipeline.direct_point_operation_counts),point_bank_capture_counts=dict(pipeline.point_bank_capture_counts),native_bank_reuse_counts=dict(pipeline.native_bank_reuse_counts),packet_compatibility_counts=dict(pipeline.packet_compatibility_counts))
            if not result['complete']:
                try:write(out/'INTERRUPTED_RUNTIME_PACKET.json',C.finite(dict(capture=pipeline.last_capture,point=pipeline.last_point_result,point_ledger=pipeline.last_point_bank_ledger,reuse=pipeline.last_point_bank_reuse,repair=pipeline.last_repair_ledger,active_row=active)))
                except BaseException as exc:errors.append(dict(phase='interrupted_packet',type=type(exc).__name__,message=str(exc)))
            try:pipeline.close();cleanup['pipeline_closed']=pipeline.closed
            except BaseException as exc:errors.append(dict(phase='close',type=type(exc).__name__,message=str(exc)))
        try:
            cleanup['environment_restored']=environment=={k:os.environ.get(k) for k in environment};require(cleanup['environment_restored'],'environment restoration');require(inputs(a,V7)==bound and protect(a)==before,'protected source/input bytes');require(not cleanup.get('instrument_restoration_errors') and (not cleanup.get('instruments_entered') or cleanup.get('instruments_restored') and cleanup.get('opencv_threads_restored')) and (pipeline is None or cleanup.get('pipeline_closed') and cleanup['head_hooks_removed']),'instrument/pipeline restoration');probe('after_cleanup')
        except BaseException as exc:errors.append(dict(phase='restoration',type=type(exc).__name__,message=str(exc)))
        if writer is not None:
            try:
                if active is not None and not written:writer.write(active)
                writer.close();require(not writer.close_errors,'stream close');require(not result['complete'] or writer.count==150,'serialized150')
                if result['complete'] and not errors:writer.promote()
                else:writer.preserve_interrupted()
            except BaseException as exc:errors.append(dict(phase='stream',type=type(exc).__name__,message=str(exc)));writer.preserve_interrupted()
            errors.extend(writer.close_errors+writer.preservation_errors)
        if errors:result.update(complete=False,status='FAILED_RUNTIME_CLEANUP')
        result.update(cleanup=cleanup,cleanup_errors=errors,execution=dict(state),live_cv_counts=dict(cv),live_optimizer_counts=dict(opt),live_SVD_counts=dict(svd),live_head_counts=dict(head),resource_snapshots=resources,all_recorded_resource_snapshots_quiet=bool(resources and all(r['quiet'] and r['temperature_under_80'] for r in resources)),serialized_rows=0 if writer is None else writer.count,statistics_official=result['complete'],protected_before=before,protected_after=protect(a))
        raw=out/('RUNTIME_ROWS.jsonl.gz' if result['complete'] else 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz');result['raw_rows']=binding(raw) if raw.is_file() else None
        result['summaries']={s:old.describe([r[s+'_ms'] for r in rows if r['phase']=='measured']) for s in ('full',*STAGES,'return')} if result['complete'] else None
        write(out/'RUNTIME_RESOURCES.json',dict(protocol=binding(proto),snapshots=resources,observed_snapshots_only=True));result['resources']=binding(out/'RUNTIME_RESOURCES.json');write(out/'RUNTIME_RECEIPT.json',C.finite(result));old.durable_owned(out/'RUNTIME_STARTED.json',dict(protocol=binding(proto),inputs=bound,status=result['status'],execution=dict(state),receipt=binding(out/'RUNTIME_RECEIPT.json')))
    require(result['complete'],'fresh runtime incomplete, first prefix/failure preserved')

if __name__=='__main__':main()
