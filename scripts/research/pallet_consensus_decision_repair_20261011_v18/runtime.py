"""Fresh restored point-only consensus timing with original V4 selection.

Every timed call includes detector/N3, initial/feature poses, ROLE head,
H-union-corner LOO, point control, all four-ID roots, actual-inlier refits,
normal-incidence checks and the restored count/SSE decision. Saved geometry
is used only after return. No endpoint or finite-line factor is executed.
"""
from __future__ import annotations
import ast
import hashlib
from collections import Counter
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys
import time
import traceback
from . import runner as R
from . import evaluation as E
from . import pipeline as P
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_sparse_local_line_20261010_v9.runtime_review import normalized_frames, pose_parity
from ..pallet_three_head_observation_20261010_v7 import common as V7
from ..pallet_three_head_observation_20261010_v7.runtime import describe, durable_owned

STAGES=('detector','N3_correction','initial_pose','boundary_observation',
        'hybrid_selection_and_point_consensus','final_pose_reprojection_and_metadata')
NAMES=('RUNTIME_STARTED.json','RUNTIME_RECEIPT.json','RUNTIME_RESOURCES.json','RUNTIME_ROWS.jsonl.gz',
       'PENDING_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_ROWS.jsonl.gz','INTERRUPTED_RUNTIME_PACKET.json')

BOOKKEEPING_SOURCE=C.REPO/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/runtime.py'
BOOKKEEPING_SHA='8ff4ae77da4b73e732c0b45625f5255c05750cccfab0c1034ce2add0f007359f'

def bookkeeping_witness():
    C.require(C.binding(BOOKKEEPING_SOURCE)['sha256']==BOOKKEEPING_SHA,'immutable V17 bookkeeping source')
    names=('schedule','protect','instruments','delta')
    old={n.name:n for n in ast.parse(BOOKKEEPING_SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names}
    current={n.name:n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names}
    C.require(set(old)==set(current)==set(names) and all(ast.dump(old[k],include_attributes=False)==ast.dump(current[k],include_attributes=False) for k in names),
        'whole fixed schedule/instrument/lifecycle function bodies changed')
    return dict(source=C.binding(BOOKKEEPING_SOURCE),unchanged_functions=list(names),
        whole_function_AST_sha256={k:hashlib.sha256(ast.dump(old[k],include_attributes=False).encode()).hexdigest() for k in names},
        timing_or_numeric_formula_changes_in_reused_functions=0,point_consensus_route_dispatch_is_new_and_separately_bound=True)

def schedule():
    return [dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[
        dict(phase='measured',repeat=r,index=i,image_index=i) for r in range(5) for i in range(26)]

def protect(args): return R.protect(args)

def prerequisites(args):
    _,score=E.score_gate(args)
    C.require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and
        score['scored_rows']==490 and score['frames']==245 and score['methods']==list(P.METHODS), 'complete paired decision scoring required')
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
    bindings.update({'postseal_evaluation:'+k:v for k,v in E.evaluation_inputs(args).items()})
    paths=dict(code=Path(__file__),contract=R.DOC/'RUNTIME_CONTRACT_KO.md',panel=C.CORRECTED/'RUNTIME_PANEL.json',
        normalized_helper=C.CODE/'runtime_review.py',deployment=R.CODE/'deployment.py',accuracy_protocol=Path(args.protocol),
        geometry_gate=R.CODE/'evaluation.py',geometry_checker=R.CODE/'validation_checks.py',
        numerical_optimizer_source=R.REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py')
    paths.update(bookkeeping_source=BOOKKEEPING_SOURCE,frontend=R.REPO/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/deployment.py',
        numerical_point_solver=R.REPO/'scripts/research/pallet_factor_subset_consensus_20261011_v15/solver.py',
        incidence_guard=R.REPO/'scripts/research/pallet_stable_incidence_guard_20261011_v14/pipeline.py')
    paths.update({n:R.DOC/n for n in ('GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','GEOMETRY_SEAL.json',
        'CONTROL_RECEIPT.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json','VALIDATION_STARTED.json',
        'EVALUATION_PROTOCOL.json','SCORING_AUTHORITY.json','SCORING_RECEIPT.json','SCORING_STARTED.json','PREDICTIONS.jsonl.gz')})
    bindings.update({'runtime:'+k:C.binding(v) for k,v in paths.items()});return bindings

def frozen(args):
    protocol=C.read(args.runtime_protocol)
    C.require(protocol['schema']=='fresh_restored_point_consensus_primary_runtime_protocol_v18' and protocol['method']==P.PRIMARY and
        protocol['configured_calls']==150 and protocol['schedule']==schedule() and protocol['stages']==list(STAGES) and
        protocol['inputs']==inputs(args) and protocol['bookkeeping_source']==bookkeeping_witness() and protocol['source_root']==str(Path(args.source_root).resolve()) and
        protocol['baseline_root']==str(Path(args.baseline_root).resolve()),'frozen runtime policy/source differs')
    return protocol

def freeze(args):
    C.require(Path(args.runtime_protocol).resolve()==R.DOC/'RUNTIME_PROTOCOL.json','new own runtime protocol required')
    R.output(args,'RUNTIME_PROTOCOL.json');panel,_,_,normalization=prerequisites(args);bindings=inputs(args)
    C.write_new(R.output(args,'RUNTIME_PROTOCOL.json'),dict(schema='fresh_restored_point_consensus_primary_runtime_protocol_v18',inputs=bindings,
        method=P.PRIMARY,stages=list(STAGES),configured_calls=150,warmup=20,repeats=5,measured=130,schedule=schedule(),
        panel=panel,normalization=normalization,protected_before=protect(args),automatic_retry=False,
        source_root=str(Path(args.source_root).resolve()),baseline_root=str(Path(args.baseline_root).resolve()),
        saved_geometry_used_only_after_return=True,new_RGB=0,new_training=0,new_seed=0,
        timing='RAM RGB/K/dimensions -> fresh V4 hybrid selection plus one actual point-only consensus solve and pure count/SSE output decision, synchronized CUDA/perf_counter_ns',
        excluded=['model loading','file decode','resource/protection probes','post-return parity','logging','GT scoring'],
        one_original_point_consensus_solve_supplies_single_output=True,original_paired_accuracy_arm_not_timed=True,
        fresh_original_V4_cornerwise_selection_reexecuted=True,same_actual_selected_point_bank_reused=True,
        existing_raw_four_ID_numeric_problems_reused=True,additional_identical_Generic_calls=0,
        actual_inlier_min4_and_normal_incidence_gates_unchanged=True,true_numerical_tie_veto_preserved=True,
        whole_U_vote_then_truncated_SSE_decision=True,actual_optimizer_J_and_SVD_are_timed_not_zero=True,
        actual_endpoint_line_calls=0,bookkeeping_source=bookkeeping_witness(),
        first_prediction_detector_forwards=2,other_prediction_detector_forwards=1,N3_forwards_per_prediction=1,
        no_artificial_pre_forward=True))
    frozen(args);print('RESTORED_POINT_CONSENSUS_PRIMARY_RUNTIME_FROZEN',C.binding(args.runtime_protocol)['sha256'],flush=True)

def preflight(args):
    frozen(args);prerequisites(args)
    for name in NAMES:R.output(args,name)
    resource=C.quiet_snapshot();passed=bool(resource['quiet'] and resource['temperature_under_80'])
    C.write_new(R.output(args,'RUNTIME_PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.runtime_protocol),
        resource=resource,protection=protect(args),actual_decode_model_fit_timed_calls=0))
    C.require(passed,'resource guard; preserve first preflight');print('RESTORED_POINT_CONSENSUS_PRIMARY_RUNTIME_PREFLIGHT_PASS',flush=True)

@contextmanager
def instruments(cv,optimizer,svd,cleanup):
    import cv2
    import numpy as np
    from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as S
    saved=[];threads=cv2.getNumThreads();original=S.least_squares;original_svd=np.linalg.svd
    def fit(fun,x0,*a,**kw):
        optimizer['entries']+=1
        def residual(*v,**options):
            optimizer['residual_entries']+=1
            try:value=fun(*v,**options)
            except BaseException:optimizer['residual_exceptions']+=1;raise
            optimizer['residual_returns']+=1;return value
        jac=kw.get('jac')
        if callable(jac):
            def counted_jac(*v,**options):
                optimizer['jacobian_entries']+=1
                try:value=jac(*v,**options)
                except BaseException:optimizer['jacobian_exceptions']+=1;raise
                optimizer['jacobian_returns']+=1;return value
            kw['jac']=counted_jac
        try:value=original(residual,x0,*a,**kw)
        except BaseException:optimizer['exceptions']+=1;raise
        optimizer['returns']+=1;optimizer['success_reports']+=int(bool(value.success));return value
    def singular(*args,**kwargs):
        svd['entries']+=1
        try:value=original_svd(*args,**kwargs)
        except BaseException:svd['exceptions']+=1;raise
        svd['returns']+=1;return value
    cleanup['instruments_entered']=True
    try:
        saved.append((S,'least_squares',original));S.least_squares=fit
        saved.append((np.linalg,'svd',original_svd));np.linalg.svd=singular
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
    result=dict(schema='fresh_restored_point_consensus_primary_runtime_receipt_v18',complete=False,status='FAILED_RUNTIME',protocol=C.binding(args.runtime_protocol),
        accuracy_protocol=C.binding(args.protocol),configured_calls=150,method=P.PRIMARY,normalization=normalization)
    C.write_new(journal,dict(status='STARTED',configured_calls=150,protocol=result['protocol']))
    writer=None;pipeline=None;active=None;active_write_attempted=False;hooks=[];rows=[];probes=[];cleanup={};errors=[];cv=Counter();opt=Counter();svd=Counter();head=Counter();state=Counter();began=time.monotonic()
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
                with instruments(cv,opt,svd,cleanup),V7.inference_canary(),pipeline.old.no_truth_reads(),torch.no_grad():
                    for job in schedule():
                        active=None;active_write_attempted=False;C.require(time.monotonic()-began<=args.remaining_seconds,'operating time limit; preserve prefix')
                        if time.monotonic()-last_probe>=5 or (job['phase']=='measured' and job['image_index']==0):
                            probe('during_fresh_paths',job);last_probe=time.monotonic()
                        frame=eligible[panel[job['image_index']]['frame_id']];state['calls_started']+=1
                        durable_owned(journal,dict(status='RUNNING',active_job=job,execution=dict(state)))
                        counters=dict(cv=cv.copy(),optimizer=opt.copy(),head=head.copy(),base=pipeline.base.counts.copy(),fresh=pipeline.counts.copy(),
                            consensus=pipeline.consensus_operation_counts.copy(),direct=pipeline.direct_point_operation_counts.copy(),
                            guard=pipeline.guard_operation_counts.copy(),decision=pipeline.decision_counts.copy(),
                            svd=svd.copy(),bank_capture=pipeline.point_bank_capture_counts.copy(),
                            model=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards))
                        ticks=[]
                        def mark(stage):torch.cuda.synchronize();ticks.append((stage,time.perf_counter_ns()))
                        torch.cuda.synchronize();t0=time.perf_counter_ns()
                        output=pipeline.predict(images[job['image_index']],frame['K'],frame['xyz'],{k:frame[k] for k in ('id','session','object_type')},method=P.PRIMARY,stage_callback=mark)
                        torch.cuda.synchronize();t1=time.perf_counter_ns();times={};previous=t0
                        for stage,tick in ticks:times[stage+'_ms']=(tick-previous)/1e6;previous=tick
                        times['return_ms']=(t1-previous)/1e6
                        after=dict(cv=cv,optimizer=opt,head=head,base=pipeline.base.counts,fresh=pipeline.counts,
                            consensus=pipeline.consensus_operation_counts,direct=pipeline.direct_point_operation_counts,
                            guard=pipeline.guard_operation_counts,decision=pipeline.decision_counts,
                            svd=svd,bank_capture=pipeline.point_bank_capture_counts,
                            model=Counter(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards))
                        active=dict(**job,id=frame['id'],session=frame['session'],method=P.PRIMARY,t0_ns=t0,t1_ns=t1,stage_ticks_ns=ticks,full_ms=(t1-t0)/1e6,**times,
                            full_fresh_output=output,original_fresh_detector_prediction=copy.deepcopy(pipeline.last_capture['raw']),
                            fresh_point_bank_ledger=pipeline.last_point_bank_ledger,fresh_point_bank_reuse=pipeline.last_point_bank_reuse,
                            fresh_consensus_ledger=pipeline.last_consensus_ledger,
                            execution_events=pipeline.last_execution_events,counter_deltas={k:delta(after[k],v) for k,v in counters.items()},parity_status='PENDING')
                        C.require([s for s,_ in ticks]==list(STAGES) and active['counter_deltas']['head']==dict(entries=1,returns=1),'fresh stage/head execution differs')
                        C.require(active['counter_deltas']['model']==dict(detector=2 if state['calls_started']==1 else 1,N3=1),'actual fresh detector/N3 forward count differs')
                        reference=references[frame['id']]
                        active['points_parity']=pipeline.runtime._point_parity(output['native_points'],reference['native_points'],1e-7,(P.PRIMARY,frame['id']))
                        active['pose_parity']=pose_parity(output['actual_pose'],reference['actual_pose'])
                        C.require(all(C.finite(output[k])==C.finite(reference[k]) for k in ('hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')),'post-return outcome parity differs')
                        for arm,key in (('BASE','original_base_points'),('N3_SUBPIX','native_N3_points')):
                            active[arm+'_parity']=pipeline.runtime._point_parity(output[key],frame['points'][arm],1e-7,(arm,frame['id']))
                        ledger=pipeline.last_consensus_ledger;reuse=pipeline.last_point_bank_reuse
                        C.require(ledger['original_point_only_solver_calls']==ledger['pure_output_decisions']==ledger['reported_primary_views']==1
                            and ledger['line_factors']==ledger['endpoint_LOO_calls']==ledger['second_solver_calls']==ledger['native_paired_control_calls']==0,
                            'fresh primary one point-consensus route differs')
                        C.require(reuse['same_coordinate_bank_reused'] is True and reuse['constructor_restored'] is True
                            and 1<=reuse['actual_constructor_count']<=2 and reuse['numeric_cache_before']==reuse['numeric_cache_after']
                            and reuse['additional_Generic_calls']==reuse['additional_Generic_cache_misses']==0
                            and active['counter_deltas']['direct']==reuse['direct_bank_delta']==ledger['direct_point_bank_operation_counts']
                            and active['counter_deltas']['consensus']==ledger['consensus_operation_counts']
                            and active['counter_deltas']['guard']==ledger['stable_incidence_operation_counts'],
                            'live point-consensus shared selected bank differs')
                        captured=reuse['actual_constructor_count']
                        C.require(active['counter_deltas']['bank_capture']==dict(observer_scopes_started=1,constructor_entries=captured,
                            constructor_returns=captured,observer_scopes_restored=1),'live hybrid bank constructor lifecycle differs')
                        solved=output['solver'];d=active['counter_deltas'];operations=d['consensus'];optimizer=d['optimizer']
                        C.require(optimizer.get('entries',0)==operations['local_optimizer_calls']
                            and optimizer.get('returns',0)==operations['local_optimizer_completions']
                            and optimizer.get('exceptions',0)==operations['local_optimizer_exceptions']
                            and optimizer.get('residual_entries',0)==operations['optimizer_residual_calls']
                            and optimizer.get('jacobian_entries',0)==operations['optimizer_jacobian_calls'],
                            'actual optimizer dispatch/return/exception and callbacks differ')
                        C.require(solved['solver']=='FACTOR_SUBSET_FINITE_EDGE_CONSENSUS' and solved['prior_used'] is False
                            and solved['known_dimension_constraint_used'] is False and solved['line_edges']==solved['fit_line_edges']==[]
                            and (not solved['available'] or len(solved['fit_input_ids'])>=4)
                            and solved['decision_restoration']['execution_kind']=='FRESH_ORIGINAL_POINT_ONLY_SOLVE',
                            'actual point-only refit and restored decision contract differs')
                        C.require(ledger['decision']==solved['decision_restoration']
                            and d['decision']==dict(fresh_original_solver_calls=1,pure_output_decisions=1,
                                restored=int(ledger['decision']['restored']),true_tie_retained=int(ledger['decision']['true_non_equivalent_numeric_tie_retained']),
                                new_display_H_assignments=ledger['decision']['new_display_H_assignments']),
                            'fresh pure decision and real H assignment ledger differs')
                        if output['new_pose_estimated']:
                            C.require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']),'self-hidden fit leakage')
                            np.testing.assert_allclose(np.asarray(output['native_points'])[output['hidden_initial']],np.asarray(output['solver']['projected'])[output['hidden_initial']],rtol=0,atol=1e-12)
                        active.update(parity_status='PASS',saved_geometry_used_only_after_return=True,original_candidate_center_preservation_checked_inside_predict=True)
                        active_write_attempted=True;writer.write(active);rows.append({k:active[k] for k in ('phase','full_ms',*(s+'_ms' for s in STAGES),'return_ms')});active=None;state['calls_complete']+=1
                        if state['calls_complete']%25==0:print('RESTORED_POINT_CONSENSUS_PRIMARY_RUNTIME',state['calls_complete'],150,flush=True)
                probe('after_fresh_paths')
                C.require(state['calls_complete']==150 and head==dict(entries=150,returns=150),'full150/head execution required')
                C.require(pipeline.base.counts['LOO_pose_paths']==pipeline.base.counts['IMAGE_ROLE_LOO_pose_paths'],
                    'fresh original hybrid H-union-corner LOO counts differ')
                for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):
                    C.require(pipeline.base.counts[key]==150,'actual base route count '+key)
                for key in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_consensus_paths'):
                    C.require(pipeline.counts[key+'_started']==pipeline.counts[key+'_complete']==150,'actual fresh route count '+key)
                C.require(pipeline.counts['reported_primary_views_from_single_consensus_solve']==150,'fresh point-consensus primary view count differs')
                captures=pipeline.point_bank_capture_counts
                C.require(captures['observer_scopes_started']==captures['observer_scopes_restored']==150 and
                    150<=captures['constructor_entries']==captures['constructor_returns']<=300 and
                    captures.get('constructor_exceptions',0)==0,'actual150 native/selected constructor lifecycles differ')
                C.require(opt['entries']==opt['returns']+opt['exceptions']
                    and opt['entries']==pipeline.consensus_operation_counts['local_optimizer_calls']
                    and opt['returns']==pipeline.consensus_operation_counts['local_optimizer_completions']
                    and opt['exceptions']==pipeline.consensus_operation_counts['local_optimizer_exceptions']
                    and opt['residual_entries']==opt['residual_returns']+opt['residual_exceptions']
                    and opt['jacobian_entries']==opt['jacobian_returns']+opt['jacobian_exceptions'],
                    'live consensus optimizer/callback accounting differs')
                C.require(all(cv[n+':entries']==cv[n+':returns']+cv[n+':exceptions'] for n in R.PRIMITIVES),'live CV entry/return/exception mismatch')
                C.require(svd['entries']==svd['returns']+svd['exceptions'],'live SVD entry/return/exception mismatch')
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
                        consensus_operation_counts=dict(pipeline.consensus_operation_counts),
                        direct_point_operation_counts=dict(pipeline.direct_point_operation_counts),
                        stable_incidence_operation_counts=dict(pipeline.guard_operation_counts),
                        decision_counts=dict(pipeline.decision_counts),
                        point_bank_capture_counts=dict(pipeline.point_bank_capture_counts),
                        packet_compatibility_counts=dict(pipeline.packet_compatibility_counts),
                        model_forwards=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,
                            detector_internal_initialization_calls=pipeline.models.detector_forwards-pipeline.base.counts['detector_calls']),live_head_counts=dict(head))
                    if not result['complete']:
                        try:C.write_new(R.output(args,'INTERRUPTED_RUNTIME_PACKET.json'),dict(capture=pipeline.last_capture,point=pipeline.last_point_result,
                            point_ledger=pipeline.last_point_bank_ledger,point_bank_reuse=pipeline.last_point_bank_reuse,
                            consensus_ledger=pipeline.last_consensus_ledger,events=pipeline.last_execution_events,active_row=active))
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
        result.update(protected_before=before,cleanup=cleanup,cleanup_errors=errors,execution=dict(state),live_cv_counts=dict(cv),live_optimizer_counts=dict(opt),live_SVD_counts=dict(svd),
            resource_snapshots=probes,all_recorded_resource_snapshots_quiet=bool(probes and all(p['quiet'] and p['temperature_under_80'] for p in probes)),
            serialized_rows=0 if writer is None else writer.count,elapsed_seconds=time.monotonic()-began,statistics_official=result['complete'])
        path=R.DOC/('RUNTIME_ROWS.jsonl.gz' if result['complete'] else 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz');result['raw_rows']=C.binding(path) if path.is_file() else None
        result['summaries']={s:describe([r[s+'_ms'] for r in rows if r['phase']=='measured']) for s in ('full',*STAGES,'return')} if result['complete'] else None
        C.write_new(R.output(args,'RUNTIME_RESOURCES.json'),dict(protocol=result['protocol'],snapshots=probes,
            all_recorded_snapshots_quiet=result['all_recorded_resource_snapshots_quiet'],observed_snapshots_only=True))
        result['resources']=C.binding(R.DOC/'RUNTIME_RESOURCES.json')
        C.write_new(R.output(args,'RUNTIME_RECEIPT.json'),result);durable_owned(journal,dict(status=result['status'],execution=dict(state),receipt=C.binding(R.DOC/'RUNTIME_RECEIPT.json')))
    C.require(result['complete'],'fresh restored point-consensus HYBRID timing incomplete; preserve first failure/prefix');return result

def main():
    parser=R.parser()
    for action in parser._actions:
        if action.dest=='stage':action.choices=('freeze','preflight','measure')
    existing={a.dest for a in parser._actions}
    for name,default in (('ordinary-parent',str(E.ORDINARY_DOC)),('base-weights',None),('N3-weights',None),('calibration',str(V7.V.DOC/'CALIBRATION.json')),('calibration-root',str(V7.CAL_ROOT)),('runtime-protocol',str(R.DOC/'RUNTIME_PROTOCOL.json'))):
        if name.replace('-','_') not in existing:parser.add_argument('--'+name,default=default)
    args=parser.parse_args();{'freeze':freeze,'preflight':preflight,'measure':measure}[args.stage](args)

if __name__=='__main__':main()
