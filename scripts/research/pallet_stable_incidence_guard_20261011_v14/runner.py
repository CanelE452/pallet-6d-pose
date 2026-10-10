"""Guard all245 immutable V13 proposals; replay wall time is not latency."""
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

from . import pipeline as G
from ..pallet_finite_edge_consensus_repair_20261011_v13 import runner as P
from ..pallet_finite_edge_consensus_repair_20261011_v13 import evaluation_review as E
from ..pallet_sparse_local_line_20261010_v9 import common as C

REPO=C.REPO;CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
PRIMITIVES=P.PRIMITIVES
NAMES=P.NAMES

def parser():
    p=P.parser();p.set_defaults(output=str(DOC),protocol=str(DOC/'PROTOCOL.json'),prior_bindings=str(DOC/'PROTECTION_BEFORE.json'));return p

def parent_args(args):
    a=copy.copy(args);a.output=str(P.DOC);a.protocol=str(P.DOC/'PROTOCOL.json');return a

def output(args,name):
    root=Path(args.output).absolute()
    C.require(root.resolve()==DOC.resolve() and not any(p.is_symlink() for p in (root,*root.parents)),'exclusive V14 output required')
    C.require(name==Path(name).name and not(root/name).exists(),'preserve '+name);return root/name

def inputs(args):
    result={'parent:'+k:v for k,v in P.inputs(parent_args(args)).items()}
    for n in ('PROTOCOL.json','PARENT_POPULATION_CHECKS.json','GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz',
        'GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','CPU_PROTOCOL.json','CPU_CHECKS.json','CPU_REVIEW_PROTOCOL.json','CPU_REVIEW_CHECKS.json',
        'VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json',
        'REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json'):
        result['parent:'+n]=C.binding(P.DOC/n)
    for n in ('__init__.py','pipeline.py','test_guard.py','cpu_checks.py','cpu_environment_review.py','cpu_package_review.py','runner.py','validation_checks.py'):
        result['code:'+n]=C.binding(CODE/n)
    for n in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json','REVIEW_CPU_PROTOCOL.json','REVIEW_CPU_STARTED.json','REVIEW_CPU_CHECKS.json',
        'PACKAGE_CPU_PROTOCOL.json','PACKAGE_CPU_STARTED.json','PACKAGE_CPU_CHECKS.json','METHOD_CONTRACT_KO.md','PROTECTION_BEFORE.json'):
        result['own:'+n]=C.binding(DOC/n)
    for alias,p in (('parent_authority',Path(E.__file__)),('parent_review_checker',P.CODE/'validation_review.py'),
        ('parent_cusp_contract',P.DOC/'CUSP_REVIEW_CONTRACT_KO.md'),('pre_GT_cause',P.DOC/'CUSP_OBSERVABILITY_DIAGNOSIS_KO.md'),
        ('normal_kernel',REPO/'scripts/research/pallet_partial_line_independent_20261010_v5/solver.py'),
        ('scalar_source',REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py')):
        result[alias]=C.binding(p)
    return result

def parent_gate(args):
    authority=E.geometry_gate(parent_args(args));cpu=C.read(DOC/'CPU_CHECKS.json');proto=C.read(DOC/'CPU_PROTOCOL.json')
    C.require(cpu['schema']=='stable_incidence_CPU_checks_v14' and cpu['passed'] is False and cpu['complete'] is True
        and cpu['tests_run']==7 and cpu['failures']==[] and len(cpu['errors'])==3
        and all("KeyError: 'PALLET_SOURCE_ROOT'\n" in e['traceback'] for e in cpu['errors']) and cpu['error'] is None
        and cpu['source_unchanged'] is True and cpu['opencv_threads_restored'] is True
        and not cpu['cleanup_errors'] and cpu['inputs']==proto['inputs'],'preserve original four-PASS/three-missing-environment-errors CPU result')
    C.bound(DOC/'CPU_PROTOCOL.json',cpu['own_protocol'],'original failed guard CPU protocol')
    C.require(C.same_binding(C.read(DOC/'CPU_STARTED.json')['protocol'],cpu['own_protocol']),'original CPU STARTED differs')
    from .cpu_checks import inputs as cpu_inputs
    C.require(cpu_inputs()==proto['inputs'],'original CPU source bytes differ')
    review=C.read(DOC/'REVIEW_CPU_CHECKS.json');own=C.read(DOC/'REVIEW_CPU_PROTOCOL.json');started=C.read(DOC/'REVIEW_CPU_STARTED.json')
    C.require(review['schema']==cpu['schema'] and own['schema']==proto['schema'] and review['complete'] is True
        and review['passed'] is False and review['tests_run']==0 and review['failures']==review['errors']==[]
        and review['error']['type']=='ImportError' and review['error']['message']=='attempted relative import with no known parent package'
        and review['source_unchanged'] is True and review['cleanup_errors']==[] and review['inputs']==own['inputs'],
        'preserve the failed zero-test relative-import environment review')
    C.bound(DOC/'REVIEW_CPU_PROTOCOL.json',review['own_protocol'],'failed environment-review CPU protocol')
    C.require(C.same_binding(started['protocol'],review['own_protocol']),'review CPU STARTED differs')
    marker=review['environment_review'];C.require(marker==own['environment_review']==started['environment_review']
        and marker['original_failed_CPU_attempt_preserved'] is True and marker['original_tests_and_guard_changed'] is False
        and marker['original_tests_run']==7 and marker['original_errors']==3 and marker['environment_only_correction'] is True
        and marker['same_seven_groups_reexecuted'] is True and marker['original_CPU_AST_edits']==0
        and marker['configured_environment']==dict(PALLET_SOURCE_ROOT=str(Path(args.source_root).resolve()),PALLET_BASELINE_ROOT=str(Path(args.baseline_root).resolve())),
        'CPU review changed numerical core or declared split environment')
    expected=dict(proto['inputs'])
    for p in (CODE/'cpu_environment_review.py',DOC/'CPU_PROTOCOL.json',DOC/'CPU_STARTED.json',DOC/'CPU_CHECKS.json'):
        key=str(p.relative_to(REPO));C.bound(p,own['inputs'][key],'review CPU bound '+key);expected[key]=own['inputs'][key]
    C.require(own['inputs']==expected,'failed CPU review changed original input closure')
    success=C.read(DOC/'PACKAGE_CPU_CHECKS.json');fixed=C.read(DOC/'PACKAGE_CPU_PROTOCOL.json');start=C.read(DOC/'PACKAGE_CPU_STARTED.json')
    C.require(success['schema']==cpu['schema'] and fixed['schema']==proto['schema'] and success['complete'] is True
        and success['passed'] is True and success['tests_run']==7 and success['failures']==success['errors']==[]
        and success['error'] is None and success['source_unchanged'] is True and success['opencv_threads_restored'] is True
        and success['cleanup_errors']==[] and success['inputs']==fixed['inputs'],'actual normal-package seven-group CPU PASS required')
    C.bound(DOC/'PACKAGE_CPU_PROTOCOL.json',success['own_protocol'],'actual successful package CPU protocol')
    C.require(C.same_binding(start['protocol'],success['own_protocol']),'actual package CPU STARTED differs')
    marker=success['package_review'];C.require(marker==fixed['package_review']==start['package_review']
        and marker['original_failed_CPU_attempt_preserved'] is True and marker['failed_environment_review_preserved'] is True
        and marker['original_tests_and_guard_changed'] is False and marker['original_tests_run']==7 and marker['original_errors']==3
        and marker['prior_environment_review_tests_run']==0 and marker['Python_package_and_environment_setup_only'] is True
        and marker['normal_original_package_import'] is True and marker['configured_actual_groups']==7
        and marker['configured_environment']==dict(PALLET_SOURCE_ROOT=str(Path(args.source_root).resolve()),PALLET_BASELINE_ROOT=str(Path(args.baseline_root).resolve())),
        'successful CPU review changed tests/guard or declared package/environment')
    expected=dict(proto['inputs'])
    for p in (CODE/'cpu_package_review.py',CODE/'cpu_environment_review.py',*[DOC/n for n in
        ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json','REVIEW_CPU_PROTOCOL.json','REVIEW_CPU_STARTED.json','REVIEW_CPU_CHECKS.json')]):
        key=str(p.relative_to(REPO));C.bound(p,fixed['inputs'][key],'actual package CPU bound '+key);expected[key]=fixed['inputs'][key]
    C.require(fixed['inputs']==expected,'successful CPU package review changed original source closure');return authority

def protect(args):
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    C.require(Path(args.prior_bindings).resolve()==DOC/'PROTECTION_BEFORE.json','new V14 protection snapshot required')
    snapshot=C.read(args.prior_bindings);tracked=snapshot['tracked'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    C.require(tracked['original_research_head']==head,'protection must bind current parent HEAD')
    check=verify(tracked,args.source_root);C.require(check['passed'],'protected published/source bytes changed')
    for b in snapshot.get('additional_completed_audits',[]):C.bound(REPO/b['path'],b,'preserved additional '+b['path'])
    return dict(published_parent_HEAD=head,unchanged_original_verifier=check,own_latest_HEAD_snapshot=True)

def frozen(args):
    own=C.read(args.protocol)
    C.require(own['schema']=='fixed_selected_stable_incidence_protocol_v14' and own['policy']==G.POLICY
        and own['methods']==list(G.METHODS) and own['inputs']==inputs(args),'frozen guard policy/source differs')
    C.require(own['source_root']==str(Path(args.source_root).resolve()) and own['baseline_root']==str(Path(args.baseline_root).resolve()),'split roots differ')
    C.bound(DOC/'PARENT_POPULATION_CHECKS.json',own['parent_population_checks'],'complete immutable parent population')
    parent_gate(args);return own

def parent_rows(args):
    expected=C.read(P.DOC/'PARENT_POPULATION_CHECKS.json')['ordered_ids'];seen=[]
    for row in C.rows(P.DOC/'GEOMETRY_SEALED.jsonl.gz'):
        C.require(len(seen)<245 and row['id']==expected[len(seen)] and row['method']==G.PARENT_METHOD and not G.TRUTH.intersection(row),'ordered GT-free parent row differs')
        seen.append(row['id']);yield row
    C.require(seen==expected and len(seen)==245,'full immutable parent245 required')

def freeze(args):
    output(args,'PROTOCOL.json');output(args,'PARENT_POPULATION_CHECKS.json');C.require(Path(args.protocol).resolve()==DOC/'PROTOCOL.json','formal own protocol required')
    bindings=inputs(args);parent_gate(args);protected=protect(args);ids=[r['id'] for r in parent_rows(args)]
    C.require(inputs(args)==bindings,'freeze source bytes changed')
    C.write_new(output(args,'PARENT_POPULATION_CHECKS.json'),dict(schema='stable_incidence_parent_population_v14',complete=True,passed=True,
        ordered_ids=ids,frames=245,rows=245,parent_geometry=C.binding(P.DOC/'GEOMETRY_SEALED.jsonl.gz'),new_model_GT_PnP_optimizer_calls=0))
    C.write_new(output(args,'PROTOCOL.json'),dict(schema='fixed_selected_stable_incidence_protocol_v14',policy=G.POLICY,
        methods=list(G.METHODS),inputs=bindings,frames=245,rows=245,ledger_rows=245,parent_population_checks=C.binding(DOC/'PARENT_POPULATION_CHECKS.json'),
        source_root=str(Path(args.source_root).resolve()),baseline_root=str(Path(args.baseline_root).resolve()),protected_before=protected,
        numerical_work_before_freeze=0,existing_DEV_results_known=True,unseen_evaluation=False,new_training_RGB_seed_optimizer_PnP_calls=0,
        cause_declared_before_parent_GT_scoring=True,scoring_only_after_full_cleanup_and_independent_PASS=True,no_automatic_retry=True))
    frozen(args);print('STABLE_INCIDENCE_FROZEN',flush=True)

def preflight(args):
    frozen(args)
    for n in NAMES:output(args,n)
    q=C.quiet_snapshot();passed=bool(q['quiet'] and q['temperature_under_80'])
    C.write_new(output(args,'PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.protocol),resource_snapshot=q,protection=protect(args),actual_guard_entries=0))
    C.require(passed,'competing work/thermal guard; preserve preflight')

@contextmanager
def numerical_scope(ce,cr,cx,svd,forbidden,cleanup):
    import cv2
    import numpy as np
    from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as S
    from ..pallet_sparse_local_line_20261010_v9.evaluator import forbid_models_fits_and_local_optimizers
    saved=[];threads=cv2.getNumThreads()
    def wrapper(fn,count_returns,count_exceptions,name):
        def call(*a,**kw):
            (svd if name=='svd' else ce)['entries' if name=='svd' else name]+=1
            try:value=fn(*a,**kw)
            except BaseException:count_exceptions['exceptions' if name=='svd' else name]+=1;raise
            count_returns['returns' if name=='svd' else name]+=1;return value
        return call
    def blocked(*a,**kw):forbidden['unexpected_finite_optimizer_or_solve']+=1;raise RuntimeError('STABLE_INCIDENCE_NO_NEW_FIT')
    targets=[(S,'least_squares'),(S.FiniteEdgeConsensusBank,'solve'),(S.FiniteEdgeConsensusBank,'__init__')]
    try:
        for owner,name in [(cv2,n) for n in PRIMITIVES]+[(np.linalg,'svd')]:
            fn=getattr(owner,name);saved.append((owner,name,fn));setattr(owner,name,wrapper(fn,svd if name=='svd' else cr,svd if name=='svd' else cx,name))
        for owner,name in targets:saved.append((owner,name,getattr(owner,name)));setattr(owner,name,blocked)
        cv2.setNumThreads(1)
        with forbid_models_fits_and_local_optimizers() as (local,models):
            try:yield
            finally:forbidden.update(local);forbidden.update(models)
    finally:
        errors=[]
        for owner,name,fn in reversed(saved):
            try:setattr(owner,name,fn)
            except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
        try:cv2.setNumThreads(threads)
        except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
        cleanup.update(cleanup_errors=errors,primitive_optimizer_restored=all(getattr(o,n) is f for o,n,f in saved),opencv_threads_restored=cv2.getNumThreads()==threads)

def run(args):
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_same_observation_controls_20261010_v8.runner import diagnostic_canary
    own=frozen(args)
    for n in NAMES:output(args,n)
    pre=C.read(DOC/'PREFLIGHT.json');C.require(pre['passed'] is True and C.same_binding(pre['protocol'],C.binding(args.protocol)),'quiet preflight required')
    before=protect(args);C.write_new(output(args,'CONTROL_STARTED.json'),dict(protocol=C.binding(args.protocol),configured_rows=245,new_models_GT_fits=0))
    ce,cr,cx=(Counter({n:0 for n in PRIMITIVES}) for _ in range(3));svd=Counter(dict(entries=0,returns=0,exceptions=0))
    counts=Counter();operations=Counter();statuses=Counter();forbidden=Counter();attempted=[];probes=[];streams={};cleanup={};error=None;active=None;after=None;began=time.monotonic()
    env={k:os.environ.get(k) for k in ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')};opens=(builtins.open,io.open,Path.open)
    def quiet(phase):
        value=dict(phase=phase,frames_complete=counts['frames_complete'],**C.quiet_snapshot());C.write_new(output(args,'RESOURCE_%03d.json'%len(probes)),value);probes.append(value)
        C.require(value['quiet'] and value['temperature_under_80'],'competing work/thermal guard')
    try:
        quiet('before_incidence_guard')
        for alias,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz')):
            streams[alias]=RowWriter(output(args,name),interrupted_path=output(args,'INTERRUPTED_'+name))
        with C.original_environment(args),diagnostic_canary(args,attempted),numerical_scope(ce,cr,cx,svd,forbidden,cleanup):
            for parent in parent_rows(args):
                C.require(time.monotonic()-began<=args.remaining_seconds,'operating wall limit');active=parent;counts['frames_started']+=1
                earlier=[v.copy() for v in (ce,cr,cx,svd)];row,w=G.guard_row(parent)
                ledger=dict(id=row['id'],session=row['session'],parent_V13_row_semantic_sha256=C.semantic_sha256(parent),
                    guarded_row_semantic_sha256=C.semantic_sha256(row),stable_incidence_guard=w,guard_counts=w['counts'])
                for key,now,old in zip(('primitive_entry_calls','primitive_return_calls','primitive_exception_calls','svd_dispatch'),(ce,cr,cx,svd),earlier):ledger[key]={k:now[k]-old[k] for k in now}
                streams['geometry'].write(row);streams['ledgers'].write(ledger);counts['frames_complete']+=1;counts['geometry_rows']+=1;counts['ledger_rows']+=1
                counts['parent_NEW']+=bool(parent['new_pose_estimated']);counts['guard_applied']+=w['gate_applied'];counts['guard_accepted']+=w['accepted'];counts['parent_unavailable_noop']+=not w['gate_applied']
                operations.update({k:v for k,v in w['counts'].items() if type(v) is int});statuses[row['output_status']]+=1;active=None
                if counts['frames_complete']%32==0:quiet('during_incidence_guard')
        cleanup.update(split_environment_restored=env=={k:os.environ.get(k) for k in env},asset_canary_open_functions_restored=opens==(builtins.open,io.open,Path.open))
        C.require(counts['frames_complete']==counts['geometry_rows']==counts['ledger_rows']==245 and operations['guard_calls']==245,'full245 guard completion required')
        C.require(counts['parent_NEW']==counts['guard_applied']==operations['geometry_info_calls'] and ce['projectPoints']==operations['geometry_projection_entries'],'actual incidence dispatch join differs')
        C.require(not forbidden and not attempted and all(ce[n]==0 for n in PRIMITIVES if n!='projectPoints'),'unexpected model/GT/PnP/fit work')
        C.require(all(cleanup[k] for k in ('primitive_optimizer_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored')) and not cleanup['cleanup_errors'],'cleanup incomplete')
        quiet('after_incidence_cleanup');frozen(args);after=protect(args);C.require(before==after,'protected parent bytes changed')
        for writer in streams.values():writer.close()
        C.require(not any(w.close_errors for w in streams.values()),'raw close/fsync failure')
        for writer in streams.values():writer.promote()
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    complete=error is None and counts['frames_complete']==245
    if not complete:
        for writer in streams.values():writer.preserve_interrupted()
        if active is not None:C.write_new(output(args,'INTERRUPTED_CONTROL_PACKET.json'),active)
    raw={k:C.binding(w.final_path if w.published else w.interrupted_path) for k,w in streams.items() if (w.final_path if w.published else w.interrupted_path).is_file()}
    C.write_new(output(args,'CONTROL_RECEIPT.json'),dict(schema='selected_stable_incidence_execution_v14',complete=complete,protocol=C.binding(args.protocol),
        actual_execution=dict(counts),guard_operation_counts=dict(operations),actual_OpenCV_entry_calls=dict(ce),actual_OpenCV_return_calls=dict(cr),actual_OpenCV_exception_calls=dict(cx),
        actual_SVD_dispatch=dict(svd),output_status_counts={G.METHOD:dict(statuses)},cleanup=cleanup,cleanup_error=cleanup.get('cleanup_errors') or None,
        cleanup_completed=complete,error=error,forbidden_entries=dict(forbidden),forbidden_asset_reads=attempted,resource_snapshots=probes,raw_rows=raw,
        protection_before=before,protection_after=after,new_training_RGB_models_GT_optimizer_PnP_calls=0,original_finite_fit_repeated=False,
        elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,automatic_retry=False))
    if complete:C.write_new(output(args,'GEOMETRY_SEAL.json'),dict(schema='selected_stable_incidence_geometry_seal_v14',complete=True,frames=245,rows=245,ledger_rows=245,
        protocol=C.binding(args.protocol),geometry=raw['geometry'],ledgers=raw['ledgers'],control_receipt=C.binding(DOC/'CONTROL_RECEIPT.json'),
        parent_population_checks=own['parent_population_checks'],cleanup_completed_before_seal=True,GT_read_allowed=False,new_models_GT_optimizer_PnP_calls=0))
    print('STABLE_INCIDENCE_COMPLETE',complete,dict(counts),flush=True)
    if not complete:raise SystemExit(1)

def main():
    args=parser().parse_args();{'freeze':freeze,'preflight':preflight,'run':run}[args.stage](args)
if __name__=='__main__':main()
