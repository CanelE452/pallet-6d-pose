"""Own-frozen complete fresh PRIMARY timing; saved geometry is parity only."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys
import time
import traceback
from . import runner as R
from . import evaluation_review as E
from . import pipeline as P
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_sparse_local_line_20261010_v9.runtime_review import normalized_frames, pose_parity
from ..pallet_three_head_observation_20261010_v7 import common as V7
from ..pallet_three_head_observation_20261010_v7.runtime import describe, durable_owned

STAGES=('detector','N3_correction','initial_pose','boundary_observation',
        'endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
NAMES=('RUNTIME_STARTED.json','RUNTIME_RECEIPT.json','RUNTIME_RESOURCES.json','RUNTIME_ROWS.jsonl.gz',
       'PENDING_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_PACKET.json')

def schedule():
    return [dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[
        dict(phase='measured',repeat=r,index=i,image_index=i) for r in range(5) for i in range(26)]

def protect(args): return R.A.protect(R.audit_args(args))

def prerequisites(args):
    E.geometry_gate(args);score=C.read(R.DOC/'SCORING_RECEIPT.json')
    C.require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and
        score['scored_rows']==245 and score['frames']==245 and score['methods']==list(P.METHODS), 'complete repair scoring required')
    for key,name in (('protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('predictions','PREDICTIONS.jsonl.gz')):
        C.bound(R.DOC/name,score[key],'runtime completed scoring '+key)
    eligible,normalization=normalized_frames(args);panel=C.read(C.CORRECTED/'RUNTIME_PANEL.json')['frames']
    ids={f['frame_id'] for f in panel}
    C.require(len(panel)==len(ids)==26 and len({f['session_id'] for f in panel})==13 and
        Counter(f['session_id'] for f in panel)=={s:2 for s in {f['session_id'] for f in panel}} and
        ids<=set(eligible) and all(eligible[f['frame_id']]['session']==f['session_id'] for f in panel), 'fixed26/13 panel differs')
    fields=('native_points','actual_pose','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')
    references={};seen=set();counts=Counter()
    for row in C.rows(R.DOC/'GEOMETRY_SEALED.jsonl.gz'):
        key=(row['method'],row['id'])
        C.require(row['method'] in P.METHODS and row['id'] in eligible and key not in seen,'duplicate/foreign geometry reference')
        seen.add(key);counts[row['method']]+=1
        if row['method']==P.PRIMARY and row['id'] in ids: references[row['id']]={k:copy.deepcopy(row[k]) for k in fields}
    C.require(counts=={m:245 for m in P.METHODS} and set(references)==ids,'full245/PRIMARY26 reference population differs')
    return panel,eligible,references,normalization

def inputs(args):
    bindings={'repair:'+k:v for k,v in R.inputs(args).items()}
    bindings.update({'fresh_v7:'+k:v for k,v in V7.inputs(args).items()})
    paths=dict(code=Path(__file__),contract=R.DOC/'RUNTIME_CONTRACT_KO.md',panel=C.CORRECTED/'RUNTIME_PANEL.json',
        normalized_helper=C.CODE/'runtime_review.py',deployment=R.CODE/'deployment.py',accuracy_protocol=Path(args.protocol),
        geometry_gate=R.CODE/'evaluation_review.py',original_score_kernel=R.CODE/'evaluation.py',
        geometry_checker=R.CODE/'validation_checks.py',review_checker=R.CODE/'validation_review.py',
        cusp_review_contract=R.DOC/'CUSP_REVIEW_CONTRACT_KO.md')
    paths.update({n:R.DOC/n for n in ('GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','GEOMETRY_SEAL.json',
        'CONTROL_RECEIPT.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json','VALIDATION_STARTED.json',
        'REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json',
        'SCORING_AUTHORITY_SUPPLEMENT.json',
        'SCORING_RECEIPT.json','SCORING_STARTED.json','PREDICTIONS.jsonl.gz')})
    bindings.update({'runtime:'+k:C.binding(v) for k,v in paths.items()});return bindings

def frozen(args):
    protocol=C.read(args.runtime_protocol)
    C.require(protocol['schema']=='fresh_finite_edge_primary_runtime_protocol_v13' and protocol['method']==P.PRIMARY and
        protocol['configured_calls']==150 and protocol['schedule']==schedule() and protocol['stages']==list(STAGES) and
        protocol['inputs']==inputs(args) and protocol['source_root']==str(Path(args.source_root).resolve()) and
        protocol['baseline_root']==str(Path(args.baseline_root).resolve()),'frozen runtime policy/source differs')
    return protocol

def freeze(args):
    C.require(Path(args.runtime_protocol).resolve()==R.DOC/'RUNTIME_PROTOCOL.json','new own runtime protocol required')
    R.output(args,'RUNTIME_PROTOCOL.json');panel,_,_,normalization=prerequisites(args);bindings=inputs(args)
    C.write_new(R.output(args,'RUNTIME_PROTOCOL.json'),dict(schema='fresh_finite_edge_primary_runtime_protocol_v13',inputs=bindings,
        method=P.PRIMARY,stages=list(STAGES),configured_calls=150,warmup=20,repeats=5,measured=130,schedule=schedule(),
        panel=panel,normalization=normalization,protected_before=protect(args),automatic_retry=False,
        source_root=str(Path(args.source_root).resolve()),baseline_root=str(Path(args.baseline_root).resolve()),
        saved_geometry_used_only_after_return=True,new_RGB=0,new_training=0,new_seed=0,
        timing='RAM RGB/K/dimensions -> complete fresh single finite-edge PRIMARY prediction, synchronized CUDA/perf_counter_ns',
        excluded=['model loading','file decode','resource/protection probes','post-return parity','logging','GT scoring'],
        one_finite_bank_call_supplies_single_output=True,actual_proposal_and_inlier_refit_dispatches_recorded=True,
        first_prediction_detector_forwards=2,other_prediction_detector_forwards=1,N3_forwards_per_prediction=1,
        no_artificial_pre_forward=True))
    frozen(args);print('FINITE_EDGE_PRIMARY_RUNTIME_FROZEN',C.binding(args.runtime_protocol)['sha256'],flush=True)

def preflight(args):
    frozen(args);prerequisites(args)
    for name in NAMES:R.output(args,name)
    resource=C.quiet_snapshot();passed=bool(resource['quiet'] and resource['temperature_under_80'])
    C.write_new(R.output(args,'RUNTIME_PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.runtime_protocol),
        resource=resource,protection=protect(args),actual_decode_model_fit_timed_calls=0))
    C.require(passed,'resource guard; preserve first preflight');print('FINITE_EDGE_PRIMARY_RUNTIME_PREFLIGHT_PASS',flush=True)

@contextmanager
def instruments(cv,optimizer,cleanup):
    import cv2
    from . import solver as S
    saved=[];threads=cv2.getNumThreads();original=S.least_squares
    def fit(fun,x0,*a,**kw):
        optimizer['entries']+=1
        def residual(*v,**options):optimizer['residual_entries']+=1;return fun(*v,**options)
        jac=kw.get('jac')
        if callable(jac):
            def counted_jac(*v,**options):optimizer['jacobian_entries']+=1;return jac(*v,**options)
            kw['jac']=counted_jac
        try:value=original(residual,x0,*a,**kw)
        except BaseException:optimizer['exceptions']+=1;raise
        optimizer['returns']+=1;optimizer['success_reports']+=int(bool(value.success));return value
    cleanup['instruments_entered']=True
    try:
        saved.append((S,'least_squares',original));S.least_squares=fit
        for name in R.PRIMITIVES:
            fn=getattr(cv2,name);saved.append((cv2,name,fn))
            def call(*a,_fn=fn,_name=name,**kw):
                cv[_name+':entries']+=1
                try:value=_fn(*a,**kw)
                except BaseException:cv[_name+':exceptions']+=1;raise
                cv[_name+':returns']+=1;return value
            setattr(cv2,name,call)
        cv2.setNumThreads(1);yield
    finally:
        failures=[]
        for owner,name,value in reversed(saved):
            try:setattr(owner,name,value)
            except BaseException as exc:failures.append(dict(type=type(exc).__name__,message=str(exc)))
        try:cv2.setNumThreads(threads)
        except BaseException as exc:failures.append(dict(type=type(exc).__name__,message=str(exc)))
        cleanup.update(instrument_restoration_errors=failures,instruments_restored=all(getattr(o,n,None) is v for o,n,v in saved),
            opencv_threads_restored=cv2.getNumThreads()==threads)

def delta(after,before):return {k:after.get(k,0)-before.get(k,0) for k in set(after)|set(before)}

def measure(args):
    import cv2
    import numpy as np
    import scipy
    from .deployment import Pipeline
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    own=frozen(args);panel,eligible,references,normalization=prerequisites(args)
    for name in NAMES:R.output(args,name)
    pre=C.read(R.DOC/'RUNTIME_PREFLIGHT.json')
    C.require(pre['passed'] is True and C.same_binding(pre['protocol'],C.binding(args.runtime_protocol)), 'own preflight PASS required')
    before=protect(args);journal=R.output(args,'RUNTIME_STARTED.json');environment={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')}
    result=dict(schema='fresh_finite_edge_primary_runtime_receipt_v13',complete=False,status='FAILED_RUNTIME',protocol=C.binding(args.runtime_protocol),
        accuracy_protocol=C.binding(args.protocol),configured_calls=150,method=P.PRIMARY,normalization=normalization)
    C.write_new(journal,dict(status='STARTED',configured_calls=150,protocol=result['protocol']))
    writer=None;pipeline=None;active=None;active_write_attempted=False;hooks=[];rows=[];probes=[];cleanup={};errors=[];cv=Counter();opt=Counter();head=Counter();state=Counter();began=time.monotonic()
    def probe(phase,job=None):
        snapshot=C.quiet_snapshot();snapshot.update(phase=phase,job=job);probes.append(snapshot)
        C.require(snapshot['quiet'] and snapshot['temperature_under_80'],'resource competition/thermal guard')
    try:
        probe('before_decode_and_models');images=[]
        for entry in panel:
            frame=eligible[entry['frame_id']];path=Path(args.source_root)/frame['image']
            C.require(V7.sha(path)==frame['image_sha256'],'original RGB hash differs')
            state['image_decode_calls']+=1;image=cv2.imread(str(path),cv2.IMREAD_COLOR);state['image_decode_returns']+=1
            C.require(image is not None and list(image.shape[:2])==frame['raw_hw'],'original RGB decode/shape differs');images.append(image)
        writer=RowWriter(R.output(args,'RUNTIME_ROWS.jsonl.gz'),interrupted_path=R.output(args,'INTERRUPTED_RUNTIME_ROWS.jsonl.gz'))
        with C.legacy_context(args):
            try:
                state['model_initializations_started']+=1
                with V7.inference_canary():pipeline=Pipeline(args)
                state['model_initializations_complete']+=1;torch=pipeline.torch;C.require(torch.cuda.is_available(),'fixed GPU path required')
                module=pipeline.heads['IMAGE_ROLE']
                def entered(*unused):head['entries']+=1
                def returned(*unused):head['returns']+=1
                hooks=[module.register_forward_pre_hook(entered),module.register_forward_hook(returned)]
                result['environment']=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                    opencv=cv2.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),torch_threads=torch.get_num_threads(),
                    cv_threads=1,batch=1,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
                result['model_forwards_after_loading']=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards)
                probe('after_model_loading');torch.cuda.reset_peak_memory_stats();last_probe=time.monotonic()
                with instruments(cv,opt,cleanup),V7.inference_canary(),pipeline.old.no_truth_reads(),torch.no_grad():
                    for job in schedule():
                        active=None;active_write_attempted=False;C.require(time.monotonic()-began<=args.remaining_seconds,'operating time limit; preserve prefix')
                        if time.monotonic()-last_probe>=5 or (job['phase']=='measured' and job['image_index']==0):
                            probe('during_fresh_paths',job);last_probe=time.monotonic()
                        frame=eligible[panel[job['image_index']]['frame_id']];state['calls_started']+=1
                        durable_owned(journal,dict(status='RUNNING',active_job=job,execution=dict(state)))
                        counters=dict(cv=cv.copy(),optimizer=opt.copy(),head=head.copy(),base=pipeline.base.counts.copy(),fresh=pipeline.counts.copy(),
                            local=pipeline.local_operation_counts.copy(),native=pipeline.endpoint_operation_counts.copy(),
                            model=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards))
                        ticks=[]
                        def mark(stage):torch.cuda.synchronize();ticks.append((stage,time.perf_counter_ns()))
                        torch.cuda.synchronize();t0=time.perf_counter_ns()
                        output=pipeline.predict(images[job['image_index']],frame['K'],frame['xyz'],{k:frame[k] for k in ('id','session','object_type')},method=P.PRIMARY,stage_callback=mark)
                        torch.cuda.synchronize();t1=time.perf_counter_ns();times={};previous=t0
                        for stage,tick in ticks:times[stage+'_ms']=(tick-previous)/1e6;previous=tick
                        times['return_ms']=(t1-previous)/1e6
                        after=dict(cv=cv,optimizer=opt,head=head,base=pipeline.base.counts,fresh=pipeline.counts,local=pipeline.local_operation_counts,
                            native=pipeline.endpoint_operation_counts,model=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards))
                        active=dict(**job,id=frame['id'],session=frame['session'],method=P.PRIMARY,t0_ns=t0,t1_ns=t1,stage_ticks_ns=ticks,full_ms=(t1-t0)/1e6,**times,
                            full_fresh_output=output,original_fresh_detector_prediction=copy.deepcopy(pipeline.last_capture['raw']),
                            fresh_point_bank_ledger=pipeline.last_point_bank_ledger,fresh_repair_ledger=pipeline.last_repair_ledger,
                            execution_events=pipeline.last_execution_events,counter_deltas={k:delta(after[k],v) for k,v in counters.items()},parity_status='PENDING')
                        C.require([s for s,_ in ticks]==list(STAGES) and active['counter_deltas']['head']==dict(entries=1,returns=1),'fresh stage/head execution differs')
                        C.require(active['counter_deltas']['model']==dict(detector=2 if state['calls_started']==1 else 1,N3=1),'actual fresh detector/N3 forward count differs')
                        reference=references[frame['id']]
                        active['points_parity']=pipeline.runtime._point_parity(output['native_points'],reference['native_points'],1e-7,(P.PRIMARY,frame['id']))
                        active['pose_parity']=pose_parity(output['actual_pose'],reference['actual_pose'])
                        C.require(all(C.finite(output[k])==C.finite(reference[k]) for k in ('hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')),'post-return outcome parity differs')
                        for arm,key in (('BASE','original_base_points'),('N3_SUBPIX','native_N3_points')):
                            active[arm+'_parity']=pipeline.runtime._point_parity(output[key],frame['points'][arm],1e-7,(arm,frame['id']))
                        ledger=pipeline.last_repair_ledger
                        C.require(ledger['actual_filtered_local_solve_calls']==1 and ledger['endpoint_decision_source']=='FRESH_NATIVE_ENDPOINT_LOO','fresh local/endpoint work missing')
                        if output['new_pose_estimated']:
                            C.require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']),'self-hidden fit leakage')
                            np.testing.assert_allclose(np.asarray(output['native_points'])[output['hidden_initial']],np.asarray(output['solver']['projected'])[output['hidden_initial']],rtol=0,atol=1e-12)
                        active.update(parity_status='PASS',saved_geometry_used_only_after_return=True,original_candidate_center_preservation_checked_inside_predict=True)
                        active_write_attempted=True;writer.write(active);rows.append({k:active[k] for k in ('phase','full_ms',*(s+'_ms' for s in STAGES),'return_ms')});active=None;state['calls_complete']+=1
                        if state['calls_complete']%25==0:print('FINITE_EDGE_PRIMARY_RUNTIME',state['calls_complete'],150,flush=True)
                probe('after_fresh_paths')
                C.require(state['calls_complete']==150 and head==dict(entries=150,returns=150),'full150/head execution required')
                for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):
                    C.require(pipeline.base.counts[key]==150,'actual base route count '+key)
                for key in ('fresh_predict_calls','fresh_sparse_point_paths','fresh_repair_paths','fresh_filtered_local_paths'):
                    C.require(pipeline.counts[key+'_started']==pipeline.counts[key+'_complete']==150,'actual fresh route count '+key)
                C.require(pipeline.counts['reported_views_from_single_filtered_local']==150 and
                    pipeline.counts['fresh_native_endpoint_requests_started']==pipeline.counts['fresh_native_endpoint_requests_complete'],'fresh native requests/views differ')
                operations=pipeline.local_operation_counts
                C.require(opt['entries']==operations['proposal_optimizer_calls']+operations['refit_optimizer_calls'] and
                    opt['returns']==operations['proposal_optimizer_returns']+operations['refit_optimizer_returns'] and
                    opt['exceptions']==operations['proposal_optimizer_exceptions']+operations['refit_optimizer_exceptions'] and
                    opt['entries']==operations['local_optimizer_calls'] and opt['returns']==operations['local_optimizer_completions'] and
                    opt['residual_entries']==operations['optimizer_residual_calls'] and opt['jacobian_entries']==operations['optimizer_jacobian_calls'] and
                    opt['entries']==opt['returns']+opt['exceptions'],'live optimizer ledger differs')
                C.require(all(cv[n+':entries']==cv[n+':returns']+cv[n+':exceptions'] for n in R.PRIMITIVES),'live CV entry/return/exception mismatch')
                C.require(pipeline.models.detector_forwards==151 and pipeline.models.n3_forwards==150 and
                    result['model_forwards_after_loading']==dict(detector=0,N3=0),'actual first-predict initialization accounting differs')
                result.update(complete=True,status='COMPLETE',peak_GPU_bytes=torch.cuda.max_memory_allocated())
            except BaseException as exc:
                result.setdefault('error',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
            finally:
                for hook in hooks:
                    try:hook.remove()
                    except BaseException as exc:errors.append(dict(phase='head_hook_remove',type=type(exc).__name__,message=str(exc)))
                cleanup['head_hooks_removed']=not any(e['phase']=='head_hook_remove' for e in errors)
                if pipeline is not None:
                    result.update(model_bindings=pipeline.bindings,actual_base_counts=dict(pipeline.base.counts),actual_fresh_counts=dict(pipeline.counts),
                        local_operation_counts=dict(pipeline.local_operation_counts),native_operation_counts=dict(pipeline.endpoint_operation_counts),
                        packet_compatibility_counts=dict(pipeline.packet_compatibility_counts),
                        model_forwards=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,
                            detector_internal_initialization_calls=pipeline.models.detector_forwards-pipeline.base.counts['detector_calls']),live_head_counts=dict(head))
                    if not result['complete']:
                        try:C.write_new(R.output(args,'INTERRUPTED_RUNTIME_PACKET.json'),dict(capture=pipeline.last_capture,point=pipeline.last_point_result,
                            point_ledger=pipeline.last_point_bank_ledger,repair_ledger=pipeline.last_repair_ledger,events=pipeline.last_execution_events,active_row=active))
                        except BaseException as exc:errors.append(dict(phase='interrupted_packet',type=type(exc).__name__,message=str(exc)))
                    try:pipeline.close();cleanup['pipeline_closed']=pipeline.closed
                    except BaseException as exc:errors.append(dict(phase='pipeline_close',type=type(exc).__name__,message=str(exc)))
    except BaseException as exc:
        result.update(complete=False,status='FAILED_RUNTIME');result.setdefault('error',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        try:
            cleanup['environment_restored']=environment=={k:os.environ.get(k) for k in environment}
            C.require((not cleanup.get('instruments_entered') or (not cleanup.get('instrument_restoration_errors') and cleanup.get('instruments_restored') and cleanup.get('opencv_threads_restored'))) and
                (pipeline is None or (cleanup.get('pipeline_closed') and cleanup.get('head_hooks_removed'))) and cleanup['environment_restored'],'lifecycle cleanup incomplete')
            frozen(args);after_protection=protect(args);C.require(after_protection==before,'protected bytes changed');result['protected_after']=after_protection;probe('after_cleanup')
        except BaseException as exc:errors.append(dict(phase='cleanup_protection_resource',type=type(exc).__name__,message=str(exc)))
        if writer is not None:
            try:
                if active is not None and not active_write_attempted:writer.write(active)
                writer.close();C.require(not writer.close_errors,'row stream close/fsync failed')
                if result['complete'] and not errors:
                    C.require(writer.count==150,'full150 serialized rows required');writer.promote()
                else:writer.preserve_interrupted()
            except BaseException as exc:errors.append(dict(phase='stream',type=type(exc).__name__,message=str(exc)));writer.preserve_interrupted()
            errors.extend(writer.close_errors+writer.preservation_errors)
        if errors:result.update(complete=False,status='FAILED_RUNTIME_CLEANUP')
        result.update(protected_before=before,cleanup=cleanup,cleanup_errors=errors,execution=dict(state),live_cv_counts=dict(cv),live_optimizer_counts=dict(opt),
            resource_snapshots=probes,all_recorded_resource_snapshots_quiet=bool(probes and all(p['quiet'] and p['temperature_under_80'] for p in probes)),
            serialized_rows=0 if writer is None else writer.count,elapsed_seconds=time.monotonic()-began,statistics_official=result['complete'])
        path=R.DOC/('RUNTIME_ROWS.jsonl.gz' if result['complete'] else 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz');result['raw_rows']=C.binding(path) if path.is_file() else None
        result['summaries']={s:describe([r[s+'_ms'] for r in rows if r['phase']=='measured']) for s in ('full',*STAGES,'return')} if result['complete'] else None
        C.write_new(R.output(args,'RUNTIME_RESOURCES.json'),dict(protocol=result['protocol'],snapshots=probes,
            all_recorded_snapshots_quiet=result['all_recorded_resource_snapshots_quiet'],observed_snapshots_only=True))
        result['resources']=C.binding(R.DOC/'RUNTIME_RESOURCES.json')
        C.write_new(R.output(args,'RUNTIME_RECEIPT.json'),result);durable_owned(journal,dict(status=result['status'],execution=dict(state),receipt=C.binding(R.DOC/'RUNTIME_RECEIPT.json')))
    C.require(result['complete'],'fresh finite-edge PRIMARY timing incomplete; preserve first failure/prefix');return result

def main():
    parser=R.parser()
    for action in parser._actions:
        if action.dest=='stage':action.choices=('freeze','preflight','measure')
    existing={a.dest for a in parser._actions}
    for name,default in (('base-weights',None),('N3-weights',None),('calibration',str(V7.V.DOC/'CALIBRATION.json')),('calibration-root',str(V7.CAL_ROOT)),('runtime-protocol',str(R.DOC/'RUNTIME_PROTOCOL.json'))):
        if name.replace('-','_') not in existing:parser.add_argument('--'+name,default=default)
    args=parser.parse_args();{'freeze':freeze,'preflight':preflight,'measure':measure}[args.stage](args)

if __name__=='__main__':main()
