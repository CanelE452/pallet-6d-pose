"""One frozen endpoint/observability repair, using existing GT-free observations.

One filtered LOCAL numerical solve supplies one finite-edge consensus output. This
replay is accuracy geometry work; its wall time is not deployment latency.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import builtins
import io
import os
from pathlib import Path
import time
import traceback

from ..pallet_unused_line_endpoint_audit_20261011_v10 import runner as A
from ..pallet_sparse_local_line_20261010_v9 import common as C

REPO=C.REPO
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
NAMES=('CONTROL_STARTED.json','CONTROL_RECEIPT.json','GEOMETRY_SEAL.json',
       'GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz',
       'INTERRUPTED_GEOMETRY.jsonl.gz','INTERRUPTED_CONTROL_LEDGERS.jsonl.gz',
       'INTERRUPTED_CONTROL_PACKET.json')
PRIMITIVES=A.PRIMITIVES


def parser():
    p=A.parser();p.set_defaults(output=str(DOC),protocol=str(DOC/'PROTOCOL.json'))
    return p


def audit_args(args):
    a=copy.copy(args);a.output=str(A.DOC);a.protocol=str(A.DOC/'AUDIT_PROTOCOL.json')
    a.prior_bindings=str(A.DOC/'PROTECTION_BEFORE.json')
    return a


def output(args,name):
    root=Path(args.output).absolute()
    C.require(root.resolve()==DOC.resolve() and not any(p.is_symlink() for p in (root,*root.parents)),
              'new exclusive V13 output required')
    C.require(name==Path(name).name and not (root/name).exists(),'preserve '+name)
    return root/name


def inputs(args):
    paths={'parent:'+k:p for k,p in A.input_paths(audit_args(args)).items()}
    for name in ('AUDIT_PROTOCOL.json','AUDIT_ROWS.jsonl.gz','AUDIT_LEDGERS.jsonl.gz',
                 'AUDIT_SEAL.json','AUDIT_RECEIPT.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json'):
        paths['v10:'+name]=A.DOC/name
    for name in ('__init__.py','solver.py','pipeline.py','runner.py','validation_checks.py','evaluation.py','test_solver.py','verify.py','cpu_checks.py'):
        paths['code:'+name]=CODE/name
    paths['method_contract']=DOC/'METHOD_CONTRACT_KO.md'
    paths['checker_contract']=DOC/'CHECKER_CONTRACT_KO.md'
    for version,name in (('pallet_endpoint_observability_repair_20261011_v11','pipeline.py'),):
        paths['inherited:'+name]=REPO/'scripts/research'/version/name
    for name in ('DIAGNOSTIC_PROTOCOL.json','DIAGNOSTIC_RECEIPT.json','SEGMENT_ROWS.jsonl.gz','SEGMENT_CHECK_PROTOCOL.json','SEGMENT_CHECKS.json'):
        paths['v12:'+name]=REPO/'_docs/experiments/pallet_finite_segment_support_20261011_v12'/name
    paths['cpu_checks']=DOC/'CPU_CHECKS.json'
    paths['cpu_protocol']=DOC/'CPU_PROTOCOL.json'
    paths['cpu_review_checks']=DOC/'CPU_REVIEW_CHECKS.json'
    paths['cpu_review_protocol']=DOC/'CPU_REVIEW_PROTOCOL.json'
    paths['cpu_review_source']=CODE/'cpu_review.py'
    return {k:C.binding(p) for k,p in paths.items()}


def completed_parent(args):
    a=audit_args(args);core=A.frozen(a)
    seal,receipt,check,own=[C.read(A.DOC/n) for n in
                          ('AUDIT_SEAL.json','AUDIT_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json')]
    C.require(seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245
              and seal['cleanup_completed_before_seal'] is True and seal['GT_read_allowed'] is False,
              'complete GT-free endpoint seal required')
    C.require(receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_error'] is None
              and receipt['environment_and_monkeypatch_cleanup_completed'] is True,'endpoint cleanup required')
    C.require(check['complete'] is True and check['passed'] is True and check['inputs']==own['inputs'],
              'independent endpoint PASS required')
    for name,key in (('AUDIT_PROTOCOL.json','protocol'),('AUDIT_ROWS.jsonl.gz','audit'),
                     ('AUDIT_LEDGERS.jsonl.gz','ledgers'),('AUDIT_RECEIPT.json','receipt')):
        C.bound(A.DOC/name,seal[key],'sealed endpoint '+name)
    C.bound(A.DOC/'VALIDATION_PROTOCOL.json',check['own_protocol'],'endpoint checker protocol')
    C.bound(A.CODE/'validation_checks.py',check['checker'],'actual endpoint checker bytes')
    C.bound(A.DOC/'AUDIT_ROWS.jsonl.gz',own['inputs']['audit_rows'],'checked endpoint raw rows')
    C.bound(A.DOC/'AUDIT_SEAL.json',own['inputs']['audit_seal'],'checked endpoint seal')
    return core


def packets(args,completion):
    audit=iter(C.rows(A.DOC/'AUDIT_ROWS.jsonl.gz'))
    for original,_ledger in A.packets(audit_args(args),completion):
        row=next(audit,None);p=original['local_packet'];local=original['local_row']
        C.require(row is not None and row['id']==local['id'] and row['session']==local['session'],
                  'complete ordered same-frame endpoint rows required')
        C.require(row['local_geometry_row_semantic_sha256']==C.semantic_sha256(local)
                  and row['parent_geometry_row_semantic_sha256']==C.semantic_sha256(p['parent_row'])
                  and row['original_ROLE_observation_semantic_sha256']==C.semantic_sha256(p['role_observation']),
                  'full immutable source provenance join differs')
        yield p,row
    C.require(next(audit,None) is None,'extra endpoint row outside245')


def frozen(args):
    own=C.read(args.protocol)
    from .pipeline import POLICY,METHODS
    C.require(own['schema']=='fixed_endpoint_observability_repair_protocol_v13'
              and own['policy']==POLICY and own['methods']==list(METHODS)
              and own['inputs']==inputs(args),'frozen repair source/policy/inputs differ')
    C.require(own['source_root']==str(Path(args.source_root).resolve())
              and own['baseline_root']==str(Path(args.baseline_root).resolve()),'split roots differ')
    C.bound(DOC/'PARENT_POPULATION_CHECKS.json',own['parent_population_checks'],'parent population proof')
    completed_parent(args)
    return own


def freeze(args):
    from .pipeline import POLICY,METHODS
    output(args,'PROTOCOL.json');output(args,'PARENT_POPULATION_CHECKS.json')
    C.require(Path(args.protocol).resolve()==DOC/'PROTOCOL.json','own formal protocol required')
    bindings=inputs(args);completed_parent(args);protected=A.protect(audit_args(args))
    C.require(C.read(DOC/'CPU_CHECKS.json')['passed'] is False and C.read(DOC/'CPU_REVIEW_CHECKS.json')['passed'] is True and C.read(REPO/'_docs/experiments/pallet_finite_segment_support_20261011_v12/SEGMENT_CHECKS.json')['passed'] is True,'preserved strict-count test failure, corrected sufficient-correspondence CPU PASS and independent segment diagnosis required')
    parent=A.population(audit_args(args));ids=[p['parent_row']['id'] for p,_ in packets(args,parent['completion'])]
    C.require(ids==parent['ordered_ids'] and len(ids)==245,'full GT-free245 source proof')
    proof=dict(schema='repair_parent_population_v13',complete=True,passed=True,frames=245,
               ordered_ids=ids,completion=parent['completion'],unchanged_parent_population=parent)
    C.require(inputs(args)==bindings,'source bytes changed during freeze')
    C.write_new(output(args,'PARENT_POPULATION_CHECKS.json'),proof)
    C.write_new(output(args,'PROTOCOL.json'),dict(schema='fixed_endpoint_observability_repair_protocol_v13',
        policy=POLICY,methods=list(METHODS),inputs=bindings,frames=245,rows=245,ledger_rows=245,
        parent_population_checks=C.binding(DOC/'PARENT_POPULATION_CHECKS.json'),protected_before=protected,
        source_root=str(Path(args.source_root).resolve()),baseline_root=str(Path(args.baseline_root).resolve()),
        numerical_work_before_freeze=0,existing_DEV_results_known=True,unseen_evaluation=False,
        selected_for_known_failure_mechanism=True,performance_threshold_search=False,
        new_training_updates=0,new_RGB=0,no_automatic_retry=True,
        scoring_only_after_complete_geometry_cleanup_and_independent_validation=True))
    frozen(args);print('FINITE_EDGE_CONSENSUS_FROZEN',C.binding(args.protocol)['sha256'],flush=True)


def preflight(args):
    frozen(args)
    for name in NAMES:output(args,name)
    resource=C.quiet_snapshot();protected=A.protect(audit_args(args))
    passed=bool(resource['quiet'] and resource['temperature_under_80'])
    C.write_new(output(args,'PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.protocol),
                 resource_snapshot=resource,protection=protected,actual_solver_entries=0))
    C.require(passed,'competing workload/thermal guard; preserve preflight')
    print('FINITE_EDGE_CONSENSUS_PREFLIGHT_PASS',flush=True)


@contextmanager
def numeric_scope(execution,optimizer,cv_entries,cv_returns,cv_exceptions,svd,forbidden,cleanup):
    import cv2
    import numpy as np
    from . import solver as S
    from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline
    from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank
    original_optimizer=S.least_squares
    def counted_optimizer(fun,x0,*a,**kw):
        optimizer['entries']+=1
        def residual(*v,**options):optimizer['residual_entries']+=1;return fun(*v,**options)
        jac=kw.get('jac')
        if callable(jac):
            def counted_jac(*v,**options):optimizer['jacobian_entries']+=1;return jac(*v,**options)
            kw['jac']=counted_jac
        try:fit=original_optimizer(residual,x0,*a,**kw)
        except BaseException:optimizer['exceptions']+=1;raise
        optimizer['returns']+=1;optimizer['success_reports']+=int(bool(fit.success));return fit
    original_local_solve=S.FiniteEdgeConsensusBank.solve
    def counted_local_solve(self,*a,**kw):
        execution['filtered_solver_entries']+=1
        try:value=original_local_solve(self,*a,**kw)
        except BaseException:execution['filtered_solver_exceptions']+=1;raise
        execution['filtered_solver_returns']+=1;return value
    saved=[(S,'least_squares',original_optimizer),(np.linalg,'svd',np.linalg.svd),
           (S.FiniteEdgeConsensusBank,'solve',original_local_solve)]
    original_svd=np.linalg.svd
    def counted_svd(*a,**kw):
        svd['entries']+=1
        try:value=original_svd(*a,**kw)
        except BaseException:svd['exceptions']+=1;raise
        svd['returns']+=1;return value
    targets=((Pipeline,'__init__'),(Pipeline,'capture'),(Pipeline,'outcomes'),
             (HypothesisBank,'_generic'),(HypothesisBank,'_lm'))
    def blocked(*a,**kw):forbidden['model_or_unexpected_point_fit_entries']+=1;raise RuntimeError('REPAIR_GT_FREE_REPLAY_GUARD')
    for owner,name in targets:saved.append((owner,name,getattr(owner,name)))
    original_threads=cv2.getNumThreads()
    for name in PRIMITIVES:
        fn=getattr(cv2,name);saved.append((cv2,name,fn))
        def counted(*a,_fn=fn,_name=name,**kw):
            cv_entries[_name]+=1
            try:value=_fn(*a,**kw)
            except BaseException:cv_exceptions[_name]+=1;raise
            cv_returns[_name]+=1;return value
        setattr(cv2,name,counted)
    S.least_squares=counted_optimizer;np.linalg.svd=counted_svd
    S.FiniteEdgeConsensusBank.solve=counted_local_solve
    for owner,name in targets:setattr(owner,name,blocked)
    cv2.setNumThreads(1)
    try:yield
    finally:
        errors=[]
        for owner,name,fn in reversed(saved):
            try:setattr(owner,name,fn)
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        try:cv2.setNumThreads(original_threads)
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        cleanup.update(cleanup_errors=errors,primitive_and_optimizer_restored=all(getattr(o,n) is f for o,n,f in saved),
                       opencv_threads_restored=cv2.getNumThreads()==original_threads)


def run(args):
    from .pipeline import solve_repair_packet,METHODS
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_same_observation_controls_20261010_v8.runner import diagnostic_canary
    own=frozen(args)
    for name in NAMES:output(args,name)
    pre=C.read(DOC/'PREFLIGHT.json')
    C.require(pre['passed'] is True and C.same_binding(pre['protocol'],C.binding(args.protocol)),'quiet preflight required')
    before=A.protect(audit_args(args));proof=C.read(DOC/'PARENT_POPULATION_CHECKS.json')
    C.write_new(output(args,'CONTROL_STARTED.json'),dict(protocol=C.binding(args.protocol),frames=245,
                   configured_filtered_solver_entries=245,configured_geometry_rows=245,new_model_GT_calls=0))
    counts=Counter({k:0 for k in ('frames_started','frames_complete','filtered_solver_entries','filtered_solver_returns',
                                'filtered_solver_exceptions','geometry_rows','ledger_rows')})
    opt=Counter({k:0 for k in ('entries','returns','exceptions','success_reports','residual_entries','jacobian_entries')})
    svd=Counter({k:0 for k in ('entries','returns','exceptions')})
    ce,cr,cx=(Counter({n:0 for n in PRIMITIVES}) for _ in range(3))
    operations,forbidden,statuses=Counter(),Counter(),{m:Counter() for m in METHODS}
    attempted,probes,streams=[],[],{};cleanup={};active=None;error=None;after=None;began=time.monotonic()
    environment_before={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')}
    open_before=(builtins.open,io.open,Path.open)
    def quiet(phase):
        value=dict(phase=phase,frames_complete=counts['frames_complete'],**C.quiet_snapshot())
        C.write_new(output(args,'RESOURCE_%03d.json'%len(probes)),value);probes.append(value)
        C.require(value['quiet'] and value['temperature_under_80'],'competing work/thermal guard')
    try:
        quiet('before_filtered_solver')
        streams['geometry']=RowWriter(output(args,'GEOMETRY_SEALED.jsonl.gz'),interrupted_path=output(args,'INTERRUPTED_GEOMETRY.jsonl.gz'))
        streams['ledgers']=RowWriter(output(args,'CONTROL_LEDGERS.jsonl.gz'),interrupted_path=output(args,'INTERRUPTED_CONTROL_LEDGERS.jsonl.gz'))
        with C.original_environment(args),diagnostic_canary(args,attempted),numeric_scope(counts,opt,ce,cr,cx,svd,forbidden,cleanup):
            for packet,audit in packets(args,proof['completion']):
                C.require(time.monotonic()-began<=args.remaining_seconds,'operating wall limit')
                active=dict(id=packet['parent_row']['id'],local_packet=packet,audit_row=audit)
                counts['frames_started']+=1
                live_before=[v.copy() for v in (ce,cr,cx,svd,opt)]
                result,ledger=solve_repair_packet(packet,audit)
                for key,now,old in zip(('primitive_entry_calls','primitive_return_calls','primitive_exception_calls',
                                        'svd_dispatch','optimizer_dispatch'),(ce,cr,cx,svd,opt),live_before):
                    ledger[key]={k:now[k]-old[k] for k in now}
                C.require(len(result)==1 and [r['method'] for r in result]==list(METHODS),'one finite-edge consensus output required')
                for row in result:streams['geometry'].write(row);counts['geometry_rows']+=1;statuses[row['method']][row['output_status']]+=1
                streams['ledgers'].write(ledger);counts['ledger_rows']+=1
                operations.update(result[0]['solver']['operation_counts']);counts['frames_complete']+=1;active=None
                if counts['frames_complete']%32==0:quiet('during_filtered_solver');print('FINITE_EDGE_CONSENSUS',counts['frames_complete'],245,flush=True)
        C.require(counts['frames_complete']==counts['ledger_rows']==counts['filtered_solver_returns']==245
                  and counts['filtered_solver_entries']==245 and counts['filtered_solver_exceptions']==0
                  and operations['local_solve_calls']==245 and counts['geometry_rows']==245,'full245/245 completion required')
        C.require(opt['entries']==operations['local_optimizer_calls'] and opt['returns']==operations['local_optimizer_completions']
                  and opt['residual_entries']==operations['optimizer_residual_calls']
                  and opt['jacobian_entries']==operations['optimizer_jacobian_calls'],'actual optimizer dispatch join differs')
        C.require(not forbidden and not attempted and all(ce[n]==0 for n in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','cornerSubPix')),
                  'unexpected model/assets/point generators')
        C.require(cleanup['primitive_and_optimizer_restored'] and cleanup['opencv_threads_restored']
                  and not cleanup['cleanup_errors'],'numerical cleanup incomplete')
        cleanup['split_environment_restored']=environment_before=={k:os.environ.get(k) for k in environment_before}
        cleanup['asset_canary_open_functions_restored']=open_before==(builtins.open,io.open,Path.open)
        C.require(cleanup['split_environment_restored'] and cleanup['asset_canary_open_functions_restored'],
                  'split environment/asset canary cleanup incomplete')
        quiet('after_filtered_solver_cleanup');frozen(args);after=A.protect(audit_args(args));C.require(before==after,'protected bytes changed')
        for w in streams.values():w.close()
        C.require(not any(w.close_errors for w in streams.values()),'raw stream close/fsync failure')
        for w in streams.values():w.promote()
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and counts['frames_complete']==245
    if not complete:
        for w in streams.values():w.preserve_interrupted()
        if active is not None:C.write_new(output(args,'INTERRUPTED_CONTROL_PACKET.json'),active)
    raw={name:C.binding(w.final_path if w.published else w.interrupted_path) for name,w in streams.items()
         if (w.final_path if w.published else w.interrupted_path).is_file()}
    receipt=dict(schema='endpoint_observability_repair_execution_v13',complete=complete,protocol=C.binding(args.protocol),
        actual_execution=dict(counts),actual_optimizer=dict(opt),actual_OpenCV_entry_calls=dict(ce),actual_OpenCV_return_calls=dict(cr),
        actual_OpenCV_exception_calls=dict(cx),actual_SVD_dispatch=dict(svd),local_operation_counts=dict(operations),
        output_status_counts={m:dict(v) for m,v in statuses.items()},cleanup_completed=complete and bool(cleanup.get('primitive_and_optimizer_restored')),
        cleanup_error=cleanup.get('cleanup_errors') or None,cleanup=cleanup,error=error,resource_snapshots=probes,
        protection_before=before,protection_after=after,forbidden_entries=dict(forbidden),forbidden_asset_reads=attempted,
        raw_rows=raw,elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,
        one_filtered_numerical_solver_per_frame=True,one_output_view=True,
        new_training_updates=0,new_RGB=0,new_models_GT_score_calls=0,automatic_retry=False)
    C.write_new(output(args,'CONTROL_RECEIPT.json'),receipt)
    if complete:
        C.write_new(output(args,'GEOMETRY_SEAL.json'),dict(schema='endpoint_observability_geometry_seal_v13',complete=True,
            frames=245,rows=245,ledger_rows=245,protocol=C.binding(args.protocol),geometry=raw['geometry'],ledgers=raw['ledgers'],
            control_receipt=C.binding(DOC/'CONTROL_RECEIPT.json'),parent_population_checks=own['parent_population_checks'],
            cleanup_completed_before_seal=True,GT_read_allowed=False,new_models_GT_calls=0))
    print('FINITE_EDGE_CONSENSUS_COMPLETE',complete,dict(counts),flush=True)
    if not complete:raise SystemExit(1)


def main():
    args=parser().parse_args();{'freeze':freeze,'preflight':preflight,'run':run}[args.stage](args)


if __name__=='__main__':main()
