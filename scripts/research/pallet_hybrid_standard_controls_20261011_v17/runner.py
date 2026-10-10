"""Freeze and run one fixed native/hybrid ordinary control on all245 frames.

No models, selection, lines, LOCAL optimization or truth are read/executed.
V7's independent point-coordinate PASS is required. V16's completed but failed
line-scalar audit is preserved as history and does not gate this point-only
comparison. Replay wall time is not deployment latency.
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

from ..pallet_hybrid_factor_subset_consensus_20261011_v16 import runner as P
from ..pallet_same_observation_controls_20261010_v8 import common as B8
from ..pallet_same_observation_controls_20261010_v8 import runner as R8
from ..pallet_sparse_local_line_20261010_v9 import common as C

REPO=C.REPO;CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'
PRIVATE=Path('/tmp/pallet-hybrid-standard-controls-private-20261011-v17')
PRIMITIVES=P.PRIMITIVES;NAMES=P.NAMES


def parser():
    p=P.parser();p.set_defaults(output=str(DOC),protocol=str(DOC/'PROTOCOL.json'),
        prior_bindings=str(DOC/'PROTECTION_BEFORE.json'));return p

def parent_args(args):
    a=copy.copy(args);a.output=str(P.DOC);a.protocol=str(P.DOC/'PROTOCOL.json')
    a.prior_bindings=str(P.DOC/'PROTECTION_BEFORE.json');return a

def output(args,name):
    root=Path(args.output).absolute()
    C.require((root.resolve()==DOC.resolve() or root.resolve().is_relative_to(PRIVATE.resolve()))
        and not any(v.is_symlink() for v in (root,*root.parents)),'exclusive new V17 output required')
    for dep in (args.source_root,args.baseline_root,args.parent,args.point_parent,args.v9_input,args.fits):
        other=Path(dep).resolve();C.require(not root.resolve().is_relative_to(other)
            and not other.is_relative_to(root.resolve()),'output overlaps prior dependency')
    C.require(name==Path(name).name and not(root/name).exists(),'preserve '+name);return root/name

def cpu_bindings():
    paths={name:B8.DOC/name for name in ('CPU_TEST_PROTOCOL.json','CONTROL_CHECKS.json','CONTROL_CHECKS_STARTED.json',
        'CONTROL_CHECKS_ATTEMPT_A_REASON.json','CONTROL_CHECKS_ATTEMPT_B.json','CONTROL_CHECKS_ATTEMPT_B_STARTED.json')}
    paths.update({name:B8.CODE/name for name in ('controls.py','test_controls.py')})
    return {k:C.binding(v) for k,v in paths.items()}

def v16_scalar_bindings():
    paths=dict(original_checker=P.CODE/'validation_checks.py',
        original_protocol=P.DOC/'VALIDATION_PROTOCOL.json',
        original_started=P.DOC/'VALIDATION_STARTED.json',
        original_failure=P.DOC/'FIRST_SCALAR_ENVIRONMENT_FAILURE.json',
        ready_checker=P.CODE/'validation_environment_ready.py',
        ready_protocol=P.DOC/'VALIDATION_READY_PROTOCOL.json',
        ready_started=P.DOC/'VALIDATION_READY_STARTED.json',
        ready_checks=P.DOC/'VALIDATION_READY_CHECKS.json',
        authority_code=P.CODE/'validation_authority_ready.py')
    return {k:C.binding(v) for k,v in paths.items()}

def v16_failed_scalar_audit():
    """Validate the failed lineage without calling the V16 PASS authority."""
    from ..pallet_hybrid_factor_subset_consensus_20261011_v16.validation_authority_ready import failure_proof,source_proof
    original=failure_proof(P.DOC);_,marker=source_proof();bindings=v16_scalar_bindings()
    own,start,check=[C.read(P.DOC/n) for n in ('VALIDATION_READY_PROTOCOL.json',
        'VALIDATION_READY_STARTED.json','VALIDATION_READY_CHECKS.json')]
    C.require(own['schema']=='hybrid_factor_subset_scalar_protocol_v16'
        and check['schema']=='hybrid_factor_subset_scalar_checks_v16'
        and check['complete'] is True and check['passed'] is False
        and check['checks']==198134007 and check['failure_count']==564
        and check['exception'] is None and check['new_production_calls']==0,
        'preserve the actual complete-but-FAIL V16 line-scalar audit')
    C.require(own['inputs']==start['inputs']==check['inputs']
        and own['environment_ready']==start['environment_ready']==check['environment_ready']==marker,
        'same source-only environment correction and actual failed inputs')
    for value in (own,start,check):
        C.require(C.same_binding(value['ready_checker'],bindings['ready_checker'])
            and C.same_binding(value['authority_code'],bindings['authority_code']),
            'current V16 correction sources differ')
    C.require(C.same_binding(check['own_protocol'],bindings['ready_protocol'])
        and C.same_binding(start['own_protocol'],bindings['ready_protocol'])
        and C.same_binding(own['checker'],bindings['original_checker'])
        and C.same_binding(check['checker'],bindings['original_checker']),
        'original scalar source and failed READY protocol differ')
    extras=dict(ready_checker='ready_checker',authority_code='authority_code',
        original_validation_protocol='original_protocol',original_validation_started='original_started',
        original_failure='original_failure')
    C.require(set(own['inputs'])==set(original['inputs'])|set(extras)
        and all(own['inputs'][k]==v for k,v in original['inputs'].items())
        and all(C.same_binding(own['inputs'][k],bindings[v]) for k,v in extras.items()),
        'original failed source/input closure changed')
    for name,key in (('PROTOCOL.json','repair_protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('CONTROL_LEDGERS.jsonl.gz','ledgers'),('GEOMETRY_SEAL.json','geometry_seal'),
        ('CONTROL_RECEIPT.json','repair_receipt'),('PARENT_POPULATION_CHECKS.json','parent_population_checks')):
        C.bound(P.DOC/name,own['inputs'][key],'actual failed checked V16 '+name)
    return dict(schema='preserved_V16_line_scalar_failure_for_independent_point_control_v17',
        complete=True,passed=False,checks=check['checks'],failure_count=check['failure_count'],
        exception=None,bindings=bindings,environment_ready=marker,
        V16_line_scalar_PASS_required_for_V17=False,V16_estimated_pose_or_line_J_used_by_V17=False)

def v7_coordinate_bindings(args):
    folder=Path(args.parent)
    paths=dict(v7_validation=folder/'VALIDATION_CHECKS.json',
        v7_validation_protocol=folder/'VALIDATION_PROTOCOL.json',
        v7_validation_started=folder/'VALIDATION_STARTED.json',
        v7_checker=REPO/'scripts/research/pallet_three_head_observation_20261010_v7/validation_checks.py',
        v7_seal=folder/'GEOMETRY_SEAL.json')
    return {k:C.binding(v) for k,v in paths.items()}

def v7_coordinate_authority(args):
    folder=Path(args.parent);bindings=v7_coordinate_bindings(args)
    own,start,check,seal,receipt=[C.read(folder/n) for n in ('VALIDATION_PROTOCOL.json',
        'VALIDATION_STARTED.json','VALIDATION_CHECKS.json','GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json')]
    C.require(own['schema']=='supplemental_v7_point_scalar_validation_protocol_v1'
        and check['schema']=='supplemental_v7_all_point_candidate_scalar_checks_v1'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0
        and check['inputs']==own['inputs']
        and check['new_model_GT_PnP_optimizer_image_ray_training_calls']==0,
        'actual independent V7 point-coordinate validation PASS required')
    C.require(C.same_binding(start['protocol'],bindings['v7_validation_protocol'])
        and C.same_binding(check['protocol'],bindings['v7_validation_protocol'])
        and C.same_binding(own['checker'],bindings['v7_checker'])
        and C.same_binding(own['inputs']['checker_code'],bindings['v7_checker']),
        'V7 point-coordinate checker/STARTED source authority differs')
    C.require(seal['complete'] is True and seal['GT_read_allowed'] is False
        and seal['frames']==245 and seal['rows']==1960 and seal['fixed_rows']==490
        and receipt['complete'] is True and receipt['actual_complete_frames']==245
        and receipt['method_rows']==1960 and receipt['observation_rows']==735 and receipt['fixed_rows']==490
        and receipt['cleanup_error'] is None and receipt['inference_error'] is None
        and receipt['GT_access_during_inference'] is False,
        'complete original GT-free V7 point-coordinate source and cleanup required')
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('OBSERVATIONS.jsonl.gz','observations'),('FIXED_GEOMETRY_SEALED.jsonl.gz','fixed_geometry')):
        C.bound(folder/name,seal[key],'sealed V7 source '+name)
        C.bound(folder/name,own['inputs']['accuracy_protocol' if name=='PROTOCOL.json' else name],
            'actual validated V7 source '+name)
    C.bound(folder/'GEOMETRY_SEAL.json',own['inputs']['GEOMETRY_SEAL.json'],'validated V7 seal')
    C.bound(folder/'INFERENCE_RECEIPT.json',own['inputs']['INFERENCE_RECEIPT.json'],'validated V7 cleanup receipt')
    return dict(complete=True,passed=True,source_population_frames=245,bindings=bindings,
        V7_selected_hybrid_and_native_coordinates_are_actual_method_inputs=True,
        V16_estimated_pose_line_J_or_LOCAL_outputs_are_method_inputs=False)

def inputs(args):
    result={'parent:'+k:v for k,v in P.inputs(parent_args(args)).items()}
    for name in ('PROTOCOL.json','PARENT_POPULATION_CHECKS.json','GEOMETRY_SEALED.jsonl.gz',
        'CONTROL_LEDGERS.jsonl.gz','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json'):
        result['v16:'+name]=C.binding(P.DOC/name)
    result.update({'v16_ready:'+k:v for k,v in v16_scalar_bindings().items()})
    result.update(v7_coordinate_bindings(args))
    result.update({'ordinary_CPU:'+k:v for k,v in cpu_bindings().items()})
    result['ordinary_scalar']=C.binding(B8.CODE/'validation_checks.py')
    for name in ('__init__.py','pipeline.py','runner.py','validation_checks.py'):
        result['code:'+name]=C.binding(CODE/name)
    for name in ('METHOD_CONTRACT_KO.md','PROTECTION_BEFORE.json'):
        result['own:'+name]=C.binding(DOC/name)
    return result

def completed_parent(args):
    P.frozen(parent_args(args));v16_failed_scalar_audit();v7_coordinate_authority(args)
    seal,receipt,pop=[C.read(P.DOC/name) for name in ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json')]
    C.require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490
        and seal['cleanup_completed_before_seal'] is True and seal['GT_read_allowed'] is False
        and receipt['complete'] is True and receipt['cleanup_completed'] is True
        and receipt['error'] is None and receipt['cleanup_error'] is None,'complete paired parent and cleanup required')
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('CONTROL_LEDGERS.jsonl.gz','ledgers'),('CONTROL_RECEIPT.json','control_receipt'),
        ('PARENT_POPULATION_CHECKS.json','parent_population_checks')):
        C.bound(P.DOC/name,seal[key],'sealed V16 parent '+name)
    C.require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245
        and pop['geometry_rows_visited']==1960 and pop['observation_rows_visited']==735
        and pop['fixed_rows_visited']==490,'complete immutable V7 original populations required')
    R8.CPU_authority(cpu_bindings());return pop

def protect(args):
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    C.require(Path(args.prior_bindings).resolve()==DOC/'PROTECTION_BEFORE.json','new V17 snapshot required')
    snapshot=C.read(args.prior_bindings);tracked=snapshot['tracked']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    C.require(tracked['original_research_head']==head,'snapshot must bind current published parent HEAD')
    check=verify(tracked,args.source_root);C.require(check['passed'],'preserved published/source bytes changed')
    for item in snapshot.get('additional_completed_audits',[]):C.bound(REPO/item['path'],item,'immutable additional '+item['path'])
    return dict(published_parent_HEAD=head,unchanged_original_verifier=check,own_latest_HEAD_snapshot=True)

def frozen(args):
    from .pipeline import METHODS,POLICY
    own=C.read(args.protocol)
    C.require(own['schema']=='fixed_native_hybrid_standard_protocol_v17' and own['methods']==list(METHODS)
        and own['policy']==POLICY and own['inputs']==inputs(args),'fixed ordinary source/policy differs')
    C.require(own['source_root']==str(Path(args.source_root).resolve())
        and own['baseline_root']==str(Path(args.baseline_root).resolve()),'split roots differ')
    C.require(own['independent_point_coordinate_authority']==v7_coordinate_authority(args)
        and own['preserved_V16_failed_line_scalar_audit']==v16_failed_scalar_audit(),
        'actual point PASS and preserved independent line FAIL declarations differ')
    C.bound(DOC/'PARENT_POPULATION_CHECKS.json',own['parent_population_checks'],'full immutable parent population')
    completed_parent(args);return own

def freeze(args):
    from .pipeline import METHODS,POLICY
    output(args,'PROTOCOL.json');output(args,'PARENT_POPULATION_CHECKS.json')
    C.require(Path(args.protocol).resolve()==DOC/'PROTOCOL.json' and Path(args.output).resolve()==DOC,
        'formal protocol/population must be frozen in new DOC')
    bindings=inputs(args);parent=completed_parent(args);protected=protect(args)
    ids=[packet['parent_row']['id'] for packet in P.packets(args,parent['completion'])]
    C.require(ids==parent['ordered_ids'] and len(ids)==245,'exact full GT-free parent population')
    proof=dict(schema='native_hybrid_standard_parent_population_v17',complete=True,passed=True,frames=245,
        ordered_ids=ids,completion=parent['completion'],parent_population=C.binding(P.DOC/'PARENT_POPULATION_CHECKS.json'),
        full_V7_population_checked=True,geometry_rows_visited=1960,observation_rows_visited=735,fixed_rows_visited=490,
        selected_parent_method='IMAGE_ROLE_CORNERWISE_HYBRID',selected_hybrid_rows=245,
        full_parent_population_rows_retained=0,one_frame_hybrid_and_ROLE_row_kept=True,scored_or_human_input=False)
    C.require(inputs(args)==bindings,'source changed during freeze')
    C.write_new(output(args,'PARENT_POPULATION_CHECKS.json'),proof)
    C.write_new(output(args,'PROTOCOL.json'),dict(schema='fixed_native_hybrid_standard_protocol_v17',policy=POLICY,
        methods=list(METHODS),inputs=bindings,frames=245,rows=490,ledger_rows=245,
        parent_population_checks=C.binding(DOC/'PARENT_POPULATION_CHECKS.json'),protected_before=protected,
        source_root=str(Path(args.source_root).resolve()),baseline_root=str(Path(args.baseline_root).resolve()),
        ordinary_CPU_authority=R8.CPU_authority(cpu_bindings()),new_PnP_J_SVD_or_model_calls_before_freeze=0,
        existing_DEV_results_known=True,unseen_evaluation=False,actual_standard_paths_per_frame=2,
        maximum_whole_U_Generic_calls_per_frame=4,maximum_LM_calls_per_frame=2,
        maximum_unique_coordinate_banks_per_frame=2,partial_line_factors=0,
        independent_point_coordinate_authority=v7_coordinate_authority(args),
        preserved_V16_failed_line_scalar_audit=v16_failed_scalar_audit(),
        scoring_only_after_complete_geometry_cleanup_and_independent_validation=True,no_automatic_retry=True))
    frozen(args);print('NATIVE_HYBRID_STANDARD_FROZEN',C.binding(args.protocol)['sha256'],flush=True)

def preflight(args):
    frozen(args)
    for name in NAMES:output(args,name)
    value=C.quiet_snapshot();passed=bool(value['quiet'] and value['temperature_under_80'])
    C.write_new(output(args,'PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.protocol),
        resource_snapshot=value,protection=protect(args),actual_standard_solver_entries=0))
    C.require(passed,'competing workload/thermal guard; preserve preflight')

@contextmanager
def numeric_scope(execution,ce,cr,cx,svd,jacobian,forbidden,cleanup):
    import cv2
    import numpy as np
    from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
    from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank
    from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline
    from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as F
    original_solve=PoseBank.solve;saved=[];threads=cv2.getNumThreads()
    def blocked(*a,**kw):forbidden['model_or_LOCAL_or_initial_entries']+=1;raise RuntimeError('STANDARD_GT_FREE_REPLAY_GUARD')
    def solver(self,*a,**kw):
        C.require(kw.get('robust') is False,'only fixed ordinary solve permitted')
        execution['standard_solver_entries']+=1
        try:value=original_solve(self,*a,**kw)
        except BaseException:execution['standard_solver_exceptions']+=1;raise
        execution['standard_solver_returns']+=1;return value
    def wrap(fn,counter):
        def call(*a,**kw):
            counter['entries']+=1
            try:value=fn(*a,**kw)
            except BaseException:counter['exceptions']+=1;raise
            counter['returns']+=1;return value
        return call
    try:
        for name in PRIMITIVES:
            fn=getattr(cv2,name);saved.append((cv2,name,fn))
            def counted(*a,_fn=fn,_name=name,**kw):
                ce[_name]+=1
                try:value=_fn(*a,**kw)
                except BaseException:cx[_name]+=1;raise
                cr[_name]+=1;return value
            setattr(cv2,name,counted)
        for owner,name,value in ((PoseBank,'solve',solver),(np.linalg,'svd',wrap(np.linalg.svd,svd)),
            (HypothesisBank,'_jacobian',wrap(HypothesisBank._jacobian,jacobian)),
            (Pipeline,'__init__',blocked),(Pipeline,'capture',blocked),(Pipeline,'outcomes',blocked),
            (F,'least_squares',blocked)):
            saved.append((owner,name,getattr(owner,name)));setattr(owner,name,value)
        cv2.setNumThreads(1);yield
    finally:
        errors=[]
        for owner,name,fn in reversed(saved):
            try:setattr(owner,name,fn)
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        try:cv2.setNumThreads(threads)
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        cleanup.update(cleanup_errors=errors,primitive_and_pose_guards_restored=all(getattr(o,n) is f for o,n,f in saved),
            opencv_threads_restored=cv2.getNumThreads()==threads)

def run(args):
    from .pipeline import METHODS,solve_standard_packet
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_same_observation_controls_20261010_v8.runner import diagnostic_canary
    own=frozen(args)
    for name in NAMES:output(args,name)
    folder=Path(args.output);pre=C.read(folder/'PREFLIGHT.json')
    C.require(pre['passed'] is True and C.same_binding(pre['protocol'],C.binding(args.protocol)),'own quiet preflight required')
    before=protect(args);proof=C.read(DOC/'PARENT_POPULATION_CHECKS.json')
    C.write_new(output(args,'CONTROL_STARTED.json'),dict(protocol=C.binding(args.protocol),frames=245,
        configured_standard_solver_entries=490,configured_geometry_rows=490,configured_output_views_per_frame=2,
        new_model_GT_LOCAL_selection_calls=0))
    counts=Counter({k:0 for k in ('frames_started','frames_complete','standard_solver_entries','standard_solver_returns',
        'standard_solver_exceptions','geometry_rows','ledger_rows')})
    ce,cr,cx=(Counter({n:0 for n in PRIMITIVES}) for _ in range(3))
    svd,jac=(Counter(dict(entries=0,returns=0,exceptions=0)) for _ in range(2))
    operations,forbidden=Counter(),Counter();statuses={m:Counter() for m in METHODS};states={m:Counter() for m in METHODS}
    attempted,probes,streams=[],[],{};cleanup={};active=None;error=None;after=None;began=time.monotonic()
    env={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')};opens=(builtins.open,io.open,Path.open)
    def quiet(phase):
        v=dict(phase=phase,frames_complete=counts['frames_complete'],**C.quiet_snapshot())
        C.write_new(output(args,'RESOURCE_%03d.json'%len(probes)),v);probes.append(v)
        C.require(v['quiet'] and v['temperature_under_80'],'competing workload/thermal guard')
    try:
        quiet('before_standard_pair')
        for alias,name,interrupted in (('geometry','GEOMETRY_SEALED.jsonl.gz','INTERRUPTED_GEOMETRY.jsonl.gz'),
            ('ledgers','CONTROL_LEDGERS.jsonl.gz','INTERRUPTED_CONTROL_LEDGERS.jsonl.gz')):
            streams[alias]=RowWriter(output(args,name),interrupted_path=output(args,interrupted))
        with C.original_environment(args),diagnostic_canary(args,attempted),numeric_scope(counts,ce,cr,cx,svd,jac,forbidden,cleanup):
            for packet in P.packets(args,proof['completion']):
                C.require(time.monotonic()-began<=args.remaining_seconds,'operating wall limit')
                active=dict(id=packet['parent_row']['id'],hybrid_packet=packet);counts['frames_started']+=1
                live=[v.copy() for v in (ce,cr,cx,svd,jac,counts)]
                results,ledger=solve_standard_packet(packet)
                for key,now,old in zip(('primitive_entry_calls','primitive_return_calls','primitive_exception_calls',
                    'svd_dispatch','jacobian_dispatch','actual_solver_dispatch'),(ce,cr,cx,svd,jac,counts),live):
                    ledger[key]={k:now[k]-old[k] for k in now}
                C.require(len(results)==2 and [r['method'] for r in results]==list(METHODS),'fixed two real ordinary outputs')
                C.require(ledger['primitive_entry_calls']['solvePnPGeneric']==ledger['bank_operation_counts']['generic_calls']
                    and ledger['primitive_entry_calls']['solvePnPRefineLM']==ledger['bank_operation_counts']['lm_calls'],
                    'actual whole-U Generic/LM bank join differs')
                operations.update(ledger['bank_operation_counts'])
                for row in results:
                    streams['geometry'].write(row);counts['geometry_rows']+=1
                    statuses[row['method']][row['output_status']]+=1;states[row['method']][row['solver']['state']]+=1
                streams['ledgers'].write(ledger);counts['ledger_rows']+=1;counts['frames_complete']+=1;active=None
                if counts['frames_complete']%32==0:quiet('during_standard_pair');print('NATIVE_HYBRID_STANDARD',counts['frames_complete'],245,flush=True)
        C.require(counts['frames_complete']==counts['ledger_rows']==245 and counts['geometry_rows']==490
            and counts['standard_solver_entries']==counts['standard_solver_returns']==490
            and counts['standard_solver_exceptions']==0,'complete245/490 ordinary outputs required')
        C.require(ce['solvePnPGeneric']==operations['generic_calls'] and ce['solvePnPRefineLM']==operations['lm_calls']
            and not attempted and not forbidden and ce['solvePnP']==ce['cornerSubPix']==0,'unexpected asset/model/initial/LOCAL path')
        C.require(all(ce[n]==cr[n]+cx[n] for n in PRIMITIVES)
            and svd['entries']==svd['returns']+svd['exceptions'] and jac['entries']==jac['returns']+jac['exceptions'],
            'actual entry/return/exception accounting differs')
        C.require(cleanup['primitive_and_pose_guards_restored'] and cleanup['opencv_threads_restored']
            and not cleanup['cleanup_errors'],'numeric cleanup incomplete')
        cleanup['split_environment_restored']=env=={k:os.environ.get(k) for k in env}
        cleanup['asset_canary_open_functions_restored']=opens==(builtins.open,io.open,Path.open)
        C.require(cleanup['split_environment_restored'] and cleanup['asset_canary_open_functions_restored'],'env/canary cleanup incomplete')
        quiet('after_standard_cleanup');frozen(args);after=protect(args);C.require(before==after,'protected bytes changed')
        for stream in streams.values():stream.close()
        C.require(not any(s.close_errors for s in streams.values()),'stream close/fsync failed')
        for stream in streams.values():stream.promote()
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and counts['frames_complete']==245
    if not complete:
        for stream in streams.values():stream.preserve_interrupted()
        if active is not None:C.write_new(output(args,'INTERRUPTED_CONTROL_PACKET.json'),active)
    raw={k:C.binding(s.final_path if s.published else s.interrupted_path) for k,s in streams.items()
        if (s.final_path if s.published else s.interrupted_path).is_file()}
    C.write_new(output(args,'CONTROL_RECEIPT.json'),dict(schema='native_hybrid_standard_execution_v17',complete=complete,
        protocol=C.binding(args.protocol),actual_execution=dict(counts),actual_OpenCV_entry_calls=dict(ce),
        actual_OpenCV_return_calls=dict(cr),actual_OpenCV_exception_calls=dict(cx),actual_SVD_dispatch=dict(svd),
        actual_Jacobian_dispatch=dict(jac),ordinary_bank_operation_counts=dict(operations),
        output_status_counts={m:dict(v) for m,v in statuses.items()},solver_state_counts={m:dict(v) for m,v in states.items()},
        cleanup_completed=complete and bool(cleanup.get('primitive_and_pose_guards_restored')),cleanup=cleanup,
        cleanup_error=cleanup.get('cleanup_errors') or None,error=error,resource_snapshots=probes,
        protection_before=before,protection_after=after,forbidden_entries=dict(forbidden),forbidden_asset_reads=attempted,
        raw_rows=raw,elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,
        stream_preservation={k:dict(rows=s.count,published=s.published,close_errors=s.close_errors,
            interruption_preservation_errors=s.preservation_errors) for k,s in streams.items()},
        actual_standard_paths_per_frame=2,reported_views_per_frame=2,new_training_updates=0,new_RGB=0,
        new_model_GT_LOCAL_selection_calls=0,automatic_retry=False))
    if complete:
        C.write_new(output(args,'GEOMETRY_SEAL.json'),dict(schema='native_hybrid_standard_geometry_seal_v17',complete=True,
            frames=245,rows=490,ledger_rows=245,methods=list(METHODS),protocol=C.binding(args.protocol),
            geometry=raw['geometry'],ledgers=raw['ledgers'],control_receipt=C.binding(folder/'CONTROL_RECEIPT.json'),
            parent_population_checks=own['parent_population_checks'],cleanup_completed_before_seal=True,
            GT_read_allowed=False,new_models_GT_calls=0))
    print('NATIVE_HYBRID_STANDARD_COMPLETE',complete,dict(counts),flush=True)
    if not complete:raise SystemExit(1)

def main():
    args=parser().parse_args();{'freeze':freeze,'preflight':preflight,'run':run}[args.stage](args)

if __name__=='__main__':main()
