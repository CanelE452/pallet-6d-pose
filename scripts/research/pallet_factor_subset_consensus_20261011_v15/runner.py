"""Run one frozen factor-subset consensus path on all245 GT-free packets.

The original sparse coordinates and endpoint evidence are immutable. This
accuracy replay creates new hypotheses/refits; replay wall time is not latency.
Scoring is permitted only after the complete seal and independent geometry PASS.
"""
from collections import Counter
from contextlib import contextmanager
import builtins
import copy
import io
import os
from pathlib import Path
import subprocess
import time
import traceback

from ..pallet_finite_edge_consensus_repair_20261011_v13 import runner as P
from ..pallet_stable_incidence_guard_20261011_v14 import runner as G
from ..pallet_sparse_local_line_20261010_v9 import common as C

REPO=C.REPO;CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
PRIMITIVES=P.PRIMITIVES
NAMES=P.NAMES

def parser():
    p=P.parser();p.set_defaults(output=str(DOC),protocol=str(DOC/'PROTOCOL.json'),
        prior_bindings=str(DOC/'PROTECTION_BEFORE.json'));return p

def parent_args(args):
    a=copy.copy(args);a.output=str(G.DOC);a.protocol=str(G.DOC/'PROTOCOL.json')
    a.prior_bindings=str(G.DOC/'PROTECTION_BEFORE.json');return a

def output(args,name):
    root=Path(args.output).absolute()
    C.require(root.resolve()==DOC.resolve() and not any(p.is_symlink() for p in (root,*root.parents)),
        'exclusive new V15 output required')
    C.require(name==Path(name).name and not(root/name).exists(),'preserve '+name);return root/name

def inputs(args):
    result={'parent:'+k:v for k,v in G.inputs(parent_args(args)).items()}
    for name in ('PROTOCOL.json','PARENT_POPULATION_CHECKS.json','GEOMETRY_SEALED.jsonl.gz',
        'CONTROL_LEDGERS.jsonl.gz','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json',
        'VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):
        result['v14:'+name]=C.binding(G.DOC/name)
    for name in ('__init__.py','solver.py','pipeline.py','runner.py','test_solver.py','cpu_checks.py','validation_checks.py'):
        result['code:'+name]=C.binding(CODE/name)
    for name in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json','METHOD_CONTRACT_KO.md','PROTECTION_BEFORE.json'):
        result['own:'+name]=C.binding(DOC/name)
    return result

def completed_parent(args):
    G.frozen(parent_args(args))
    seal,receipt,check,own=[C.read(G.DOC/name) for name in
        ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json')]
    C.require(seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245
        and seal['cleanup_completed_before_seal'] is True and seal['GT_read_allowed'] is False,
        'full245 parent GT-free seal required')
    C.require(receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_error'] is None
        and receipt['cleanup_completed'] is True,'complete parent numerical cleanup required')
    C.require(check['schema']=='stable_incidence_scalar_checks_v14' and check['complete'] is True
        and check['passed'] is True and check['inputs']==own['inputs'],'independent parent incidence PASS required')
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('CONTROL_LEDGERS.jsonl.gz','ledgers'),('CONTROL_RECEIPT.json','control_receipt'),
        ('PARENT_POPULATION_CHECKS.json','parent_population_checks')):
        C.bound(G.DOC/name,seal[key],'sealed parent '+name)
    C.bound(G.DOC/'VALIDATION_PROTOCOL.json',check['own_protocol'],'parent checker protocol')
    C.bound(G.CODE/'validation_checks.py',check['checker'],'parent independent checker')
    C.bound(G.DOC/'VALIDATION_PROTOCOL.json',C.read(G.DOC/'VALIDATION_STARTED.json')['own_protocol'],'parent checker STARTED')
    C.bound(G.DOC/'GEOMETRY_SEALED.jsonl.gz',own['inputs']['geometry'],'checked parent geometry')
    C.bound(G.DOC/'GEOMETRY_SEAL.json',own['inputs']['geometry_seal'],'checked parent seal')
    return P.completed_parent(args)

def cpu_gate(args):
    from .cpu_checks import inputs as cpu_inputs
    own=C.read(DOC/'CPU_PROTOCOL.json');receipt=C.read(DOC/'CPU_CHECKS.json');start=C.read(DOC/'CPU_STARTED.json')
    configured=dict(PALLET_SOURCE_ROOT=str(Path(args.source_root).resolve()),PALLET_BASELINE_ROOT=str(Path(args.baseline_root).resolve()))
    C.require(own['schema']=='factor_subset_CPU_protocol_v15' and receipt['schema']=='factor_subset_CPU_checks_v15'
        and own['inputs']==receipt['inputs']==cpu_inputs() and own['configured_environment']==receipt['configured_environment']==configured,
        'fixed CPU source/split environment differs')
    C.require(receipt['complete'] is True and receipt['passed'] is True and receipt['tests_run']==12
        and receipt['failures']==receipt['errors']==[] and receipt['error'] is None
        and receipt['source_unchanged'] is True and receipt['opencv_threads_restored'] is True
        and receipt['environment_restored'] is True and receipt['instruments_restored'] is True
        and receipt['asset_canary_restored'] is True and receipt['cleanup_errors']==[]
        and receipt['forbidden_real_asset_reads']==[] and receipt['normal_package_import'] is True,
        'actual unchanged twelve-group synthetic CPU PASS required')
    C.bound(DOC/'CPU_PROTOCOL.json',receipt['own_protocol'],'actual CPU protocol')
    C.require(C.same_binding(start['protocol'],receipt['own_protocol']) and start['configured_environment']==configured
        and start['configured_test_groups']==own['configured_test_groups']==12,'CPU STARTED differs')

def protect(args):
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    C.require(Path(args.prior_bindings).resolve()==DOC/'PROTECTION_BEFORE.json','new V15 protection snapshot required')
    snapshot=C.read(args.prior_bindings);tracked=snapshot['tracked']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    C.require(tracked['original_research_head']==head,'protection must bind current published parent HEAD')
    check=verify(tracked,args.source_root);C.require(check['passed'],'protected published/source bytes changed')
    for b in snapshot.get('additional_completed_audits',[]):C.bound(REPO/b['path'],b,'preserved additional '+b['path'])
    return dict(published_parent_HEAD=head,unchanged_original_verifier=check,own_latest_HEAD_snapshot=True)

def frozen(args):
    from .pipeline import POLICY,METHODS
    own=C.read(args.protocol)
    C.require(own['schema']=='fixed_factor_subset_consensus_protocol_v15' and own['policy']==POLICY
        and own['methods']==list(METHODS) and own['inputs']==inputs(args),'frozen subset policy/source differs')
    C.require(own['source_root']==str(Path(args.source_root).resolve()) and own['baseline_root']==str(Path(args.baseline_root).resolve()),
        'split roots differ')
    C.bound(DOC/'PARENT_POPULATION_CHECKS.json',own['parent_population_checks'],'complete immutable parent population')
    completed_parent(args);cpu_gate(args);return own

def packets(args,completion):
    yield from P.packets(args,completion)

def freeze(args):
    from .pipeline import POLICY,METHODS
    output(args,'PROTOCOL.json');output(args,'PARENT_POPULATION_CHECKS.json')
    C.require(Path(args.protocol).resolve()==DOC/'PROTOCOL.json','formal own protocol required')
    bindings=inputs(args);completed_parent(args);cpu_gate(args);protected=protect(args)
    parent=P.A.population(P.audit_args(args));ids=[p['parent_row']['id'] for p,_ in packets(args,parent['completion'])]
    C.require(ids==parent['ordered_ids'] and len(ids)==245,'full GT-free245 source proof')
    proof=dict(schema='factor_subset_parent_population_v15',complete=True,passed=True,frames=245,ordered_ids=ids,
        completion=parent['completion'],unchanged_parent_population=parent,
        full_V7_population_checked=True,source_observations_not_posthoc_scores=True)
    C.require(inputs(args)==bindings,'source bytes changed during freeze')
    C.write_new(output(args,'PARENT_POPULATION_CHECKS.json'),proof)
    C.write_new(output(args,'PROTOCOL.json'),dict(schema='fixed_factor_subset_consensus_protocol_v15',policy=POLICY,
        methods=list(METHODS),inputs=bindings,frames=245,rows=245,ledger_rows=245,
        parent_population_checks=C.binding(DOC/'PARENT_POPULATION_CHECKS.json'),protected_before=protected,
        source_root=str(Path(args.source_root).resolve()),baseline_root=str(Path(args.baseline_root).resolve()),
        numerical_work_before_freeze=0,existing_DEV_results_known=True,unseen_evaluation=False,
        repaired_candidate_generation_not_performance_threshold_search=True,
        maximum_four_ID_subsets_per_dimension=70,maximum_registered_dimensions=2,
        maximum_mixed_three_block_subsets=1140,maximum_mixed_LOCAL_proposal_calls=2280,
        maximum_LOCAL_optimizer_calls='4560 + R valid raw direct point roots',
        partial_line_path_is_separate_same_IMAGE_ROLE_ablation=True,
        mixed_LOCAL_root_completeness_proven=False,global_uniqueness_proven=False,
        new_training_updates=0,new_RGB=0,new_random_starts=0,no_automatic_retry=True,
        scoring_only_after_complete_geometry_cleanup_and_independent_validation=True))
    frozen(args);print('FACTOR_SUBSET_CONSENSUS_FROZEN',C.binding(args.protocol)['sha256'],flush=True)

def preflight(args):
    frozen(args)
    for name in NAMES:output(args,name)
    resource=C.quiet_snapshot();passed=bool(resource['quiet'] and resource['temperature_under_80'])
    C.write_new(output(args,'PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.protocol),
        resource_snapshot=resource,protection=protect(args),actual_solver_entries=0))
    C.require(passed,'competing workload/thermal guard; preserve preflight')
    print('FACTOR_SUBSET_CONSENSUS_PREFLIGHT_PASS',flush=True)

@contextmanager
def numeric_scope(execution,optimizer,ce,cr,cx,svd,forbidden,cleanup):
    import cv2
    import numpy as np
    from . import solver as S
    from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as F
    from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline
    from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
    from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank
    saved=[];threads=cv2.getNumThreads()
    original_optimizer=F.least_squares
    def counted_optimizer(fun,x0,*a,**kw):
        optimizer['entries']+=1
        def residual(*v,**options):optimizer['residual_entries']+=1;return fun(*v,**options)
        jac=kw.get('jac')
        if callable(jac):
            def counted_jac(*v,**options):optimizer['jacobian_entries']+=1;return jac(*v,**options)
            kw['jac']=counted_jac
        try:value=original_optimizer(residual,x0,*a,**kw)
        except BaseException:optimizer['exceptions']+=1;raise
        optimizer['returns']+=1;optimizer['success_reports']+=int(bool(value.success));return value
    original_solve=S.FactorSubsetConsensusBank.solve
    def counted_solve(self,*a,**kw):
        execution['filtered_solver_entries']+=1
        try:value=original_solve(self,*a,**kw)
        except BaseException:execution['filtered_solver_exceptions']+=1;raise
        execution['filtered_solver_returns']+=1;return value
    original_svd=np.linalg.svd
    def counted_svd(*a,**kw):
        svd['entries']+=1
        try:value=original_svd(*a,**kw)
        except BaseException:svd['exceptions']+=1;raise
        svd['returns']+=1;return value
    def blocked(*a,**kw):forbidden['model_or_unexpected_pose_path_entries']+=1;raise RuntimeError('SUBSET_GT_FREE_REPLAY_GUARD')
    targets=((Pipeline,'__init__'),(Pipeline,'capture'),(Pipeline,'outcomes'),(PoseBank,'solve'),
        (HypothesisBank,'_lm'),(F.FiniteEdgeConsensusBank,'solve'))
    try:
        for name in PRIMITIVES:
            fn=getattr(cv2,name);saved.append((cv2,name,fn))
            def counted(*a,_fn=fn,_name=name,**kw):
                ce[_name]+=1
                try:value=_fn(*a,**kw)
                except BaseException:cx[_name]+=1;raise
                cr[_name]+=1;return value
            setattr(cv2,name,counted)
        for owner,name,value in ((F,'least_squares',counted_optimizer),(np.linalg,'svd',counted_svd),
            (S.FactorSubsetConsensusBank,'solve',counted_solve)):
            saved.append((owner,name,getattr(owner,name)));setattr(owner,name,value)
        for owner,name in targets:saved.append((owner,name,getattr(owner,name)));setattr(owner,name,blocked)
        cv2.setNumThreads(1);yield
    finally:
        errors=[]
        for owner,name,fn in reversed(saved):
            try:setattr(owner,name,fn)
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        try:cv2.setNumThreads(threads)
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        cleanup.update(cleanup_errors=errors,primitive_and_optimizer_restored=all(getattr(o,n) is f for o,n,f in saved),
            opencv_threads_restored=cv2.getNumThreads()==threads)

def run(args):
    from .pipeline import solve_repair_packet,METHODS
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_same_observation_controls_20261010_v8.runner import diagnostic_canary
    own=frozen(args)
    for name in NAMES:output(args,name)
    pre=C.read(DOC/'PREFLIGHT.json')
    C.require(pre['passed'] is True and C.same_binding(pre['protocol'],C.binding(args.protocol)),'quiet preflight required')
    before=protect(args);proof=C.read(DOC/'PARENT_POPULATION_CHECKS.json')
    C.write_new(output(args,'CONTROL_STARTED.json'),dict(protocol=C.binding(args.protocol),frames=245,
        configured_filtered_solver_entries=245,configured_geometry_rows=245,new_model_GT_calls=0,
        deterministic_all_four_ID_roots=True,exhaustive_mixed_three_block_subsets=True))
    counts=Counter({k:0 for k in ('frames_started','frames_complete','filtered_solver_entries','filtered_solver_returns',
        'filtered_solver_exceptions','geometry_rows','ledger_rows')})
    opt=Counter({k:0 for k in ('entries','returns','exceptions','success_reports','residual_entries','jacobian_entries')})
    svd=Counter(dict(entries=0,returns=0,exceptions=0))
    ce,cr,cx=(Counter({n:0 for n in PRIMITIVES}) for _ in range(3))
    operations,point_operations,forbidden=Counter(),Counter(),Counter()
    statuses={m:Counter() for m in METHODS};typed_states=Counter();aggregate_bounds=Counter();unavailable_bounds=Counter()
    attempted,probes,streams=[],[],{};cleanup={};active=None;error=None;after=None;began=time.monotonic()
    env={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')};opens=(builtins.open,io.open,Path.open)
    def quiet(phase):
        value=dict(phase=phase,frames_complete=counts['frames_complete'],**C.quiet_snapshot())
        C.write_new(output(args,'RESOURCE_%03d.json'%len(probes)),value);probes.append(value)
        C.require(value['quiet'] and value['temperature_under_80'],'competing workload/thermal guard')
    try:
        quiet('before_factor_subset_solver')
        for alias,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz')):
            streams[alias]=RowWriter(output(args,name),interrupted_path=output(args,'INTERRUPTED_'+name))
        with C.original_environment(args),diagnostic_canary(args,attempted),numeric_scope(counts,opt,ce,cr,cx,svd,forbidden,cleanup):
            for packet,audit in packets(args,proof['completion']):
                C.require(time.monotonic()-began<=args.remaining_seconds,'operating wall limit')
                active=dict(id=packet['parent_row']['id'],local_packet=packet,audit_row=audit)
                counts['frames_started']+=1;live_before=[v.copy() for v in (ce,cr,cx,svd,opt)]
                result,ledger=solve_repair_packet(packet,audit)
                for key,now,old in zip(('primitive_entry_calls','primitive_return_calls','primitive_exception_calls',
                    'svd_dispatch','optimizer_dispatch'),(ce,cr,cx,svd,opt),live_before):
                    ledger[key]={k:now[k]-old[k] for k in now}
                C.require(len(result)==1 and [row['method'] for row in result]==list(METHODS),'one subset consensus output required')
                solved=result[0]['solver'];bounds=solved.get('call_bounds') or {}
                for key in ('mixed_subsets','registered_dimensions','direct_valid_roots',
                    'LOCAL_optimizer_calls_at_most','maximum_direct_Generic_calls'):
                    if type(bounds.get(key)) is int:aggregate_bounds[key]+=bounds[key]
                    else:unavailable_bounds[key]+=1
                ledger['actual_finite_Generic_dispatch_matches_point_bank']=ledger['primitive_entry_calls']['solvePnPGeneric']==solved['hypothesis_bank_operation_counts']['generic_calls']
                C.require(ledger['actual_finite_Generic_dispatch_matches_point_bank'],'actual same-bank finite Generic dispatch differs')
                for row in result:
                    streams['geometry'].write(row);counts['geometry_rows']+=1;statuses[row['method']][row['output_status']]+=1
                streams['ledgers'].write(ledger);counts['ledger_rows']+=1
                operations.update(solved['operation_counts']);point_operations.update(solved['hypothesis_bank_operation_counts'])
                typed_states[solved['state']]+=1;counts['frames_complete']+=1;active=None
                if counts['frames_complete']%32==0:
                    quiet('during_factor_subset_solver');print('FACTOR_SUBSET_CONSENSUS',counts['frames_complete'],245,flush=True)
        C.require(counts['frames_complete']==counts['ledger_rows']==counts['filtered_solver_returns']==245
            and counts['filtered_solver_entries']==245 and counts['filtered_solver_exceptions']==0
            and operations['local_solve_calls']==245 and counts['geometry_rows']==245,'full245/245 completion required')
        C.require(opt['entries']==operations['local_optimizer_calls'] and opt['returns']==operations['local_optimizer_completions']
            and opt['exceptions']==operations['local_optimizer_exceptions']
            and opt['residual_entries']==operations['optimizer_residual_calls'] and opt['jacobian_entries']==operations['optimizer_jacobian_calls'],
            'actual optimizer dispatch/callback join differs')
        C.require(ce['solvePnPGeneric']==point_operations['generic_calls'] and not forbidden and not attempted
            and all(ce[name]==0 for name in ('solvePnP','solvePnPRefineLM','cornerSubPix')),
            'unexpected model/assets/additional pose generator')
        C.require(all(ce[name]==cr[name]+cx[name] for name in PRIMITIVES)
            and opt['entries']==opt['returns']+opt['exceptions'] and svd['entries']==svd['returns']+svd['exceptions'],
            'actual numerical entries/returns/exceptions differ')
        C.require(cleanup['primitive_and_optimizer_restored'] and cleanup['opencv_threads_restored']
            and not cleanup['cleanup_errors'],'numerical cleanup incomplete')
        cleanup['split_environment_restored']=env=={k:os.environ.get(k) for k in env}
        cleanup['asset_canary_open_functions_restored']=opens==(builtins.open,io.open,Path.open)
        C.require(cleanup['split_environment_restored'] and cleanup['asset_canary_open_functions_restored'],'environment/canary cleanup incomplete')
        quiet('after_factor_subset_solver_cleanup');frozen(args);after=protect(args);C.require(before==after,'protected bytes changed')
        for stream in streams.values():stream.close()
        C.require(not any(stream.close_errors for stream in streams.values()),'raw stream close/fsync failure')
        for stream in streams.values():stream.promote()
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and counts['frames_complete']==245
    if not complete:
        for stream in streams.values():stream.preserve_interrupted()
        if active is not None:C.write_new(output(args,'INTERRUPTED_CONTROL_PACKET.json'),active)
    raw={name:C.binding(stream.final_path if stream.published else stream.interrupted_path) for name,stream in streams.items()
        if (stream.final_path if stream.published else stream.interrupted_path).is_file()}
    receipt=dict(schema='factor_subset_consensus_execution_v15',complete=complete,protocol=C.binding(args.protocol),
        actual_execution=dict(counts),actual_optimizer=dict(opt),actual_OpenCV_entry_calls=dict(ce),actual_OpenCV_return_calls=dict(cr),
        actual_OpenCV_exception_calls=dict(cx),actual_SVD_dispatch=dict(svd),local_operation_counts=dict(operations),
        direct_point_bank_operation_counts=dict(point_operations),aggregate_declared_call_bounds=dict(aggregate_bounds),
        aggregate_bound_unavailable_frame_counts=dict(unavailable_bounds),
        output_status_counts={m:dict(v) for m,v in statuses.items()},solver_state_counts=dict(typed_states),
        cleanup_completed=complete and bool(cleanup.get('primitive_and_optimizer_restored')),
        cleanup_error=cleanup.get('cleanup_errors') or None,cleanup=cleanup,error=error,resource_snapshots=probes,
        protection_before=before,protection_after=after,forbidden_entries=dict(forbidden),forbidden_asset_reads=attempted,
        raw_rows=raw,elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,
        one_filtered_numerical_solver_per_frame=True,one_output_view=True,
        endpoint_decisions_replayed_without_native_numeric_recompute=True,
        new_training_updates=0,new_RGB=0,new_models_GT_score_calls=0,automatic_retry=False)
    C.write_new(output(args,'CONTROL_RECEIPT.json'),receipt)
    if complete:
        C.write_new(output(args,'GEOMETRY_SEAL.json'),dict(schema='factor_subset_consensus_geometry_seal_v15',complete=True,
            frames=245,rows=245,ledger_rows=245,protocol=C.binding(args.protocol),geometry=raw['geometry'],ledgers=raw['ledgers'],
            control_receipt=C.binding(DOC/'CONTROL_RECEIPT.json'),parent_population_checks=own['parent_population_checks'],
            cleanup_completed_before_seal=True,GT_read_allowed=False,new_models_GT_calls=0))
    print('FACTOR_SUBSET_CONSENSUS_COMPLETE',complete,dict(counts),flush=True)
    if not complete:raise SystemExit(1)

def main():
    args=parser().parse_args();{'freeze':freeze,'preflight':preflight,'run':run}[args.stage](args)

if __name__=='__main__':main()
