"""Freeze, guard and execute only the unused-line correspondence diagnosis.

Full parent streams are validated without models/GT. The only new numerical
path is the already fixed native point-only endpoint-heldout PnP audit. No
downstream filtered LOCAL fit, accuracy scoring or runtime measurement occurs.
"""
from __future__ import annotations

import builtins
from collections import Counter
from contextlib import contextmanager
import copy
import io
import os
from pathlib import Path
import subprocess
import time
import traceback

from .audit import METHOD, POLICY
from ..pallet_sparse_local_line_20261010_v9 import common as C9

REPO = C9.REPO
CODE = Path(__file__).resolve().parent
DOC = REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'
PRIVATE = Path('/tmp/pallet-unused-line-endpoint-audit-private-20261011-v10')
OUTPUTS = ('AUDIT_STARTED.json', 'AUDIT_RECEIPT.json', 'AUDIT_SEAL.json',
    'AUDIT_ROWS.jsonl.gz', 'AUDIT_LEDGERS.jsonl.gz',
    'INTERRUPTED_AUDIT_ROWS.jsonl.gz', 'INTERRUPTED_AUDIT_LEDGERS.jsonl.gz',
    'INTERRUPTED_AUDIT_PACKET.json')
PRIMITIVES = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'projectPoints', 'Rodrigues', 'cornerSubPix')
read, rows, binding, bound, require = C9.read, C9.rows, C9.binding, C9.bound, C9.require
write_new, same_binding, semantic = C9.write_new, C9.same_binding, C9.semantic_sha256


def parser():
    p = C9.parser(__doc__, ('freeze', 'preflight', 'run'))
    p.set_defaults(output=str(DOC), protocol=str(DOC/'AUDIT_PROTOCOL.json'),
        prior_bindings=str(DOC/'PROTECTION_BEFORE.json'))
    p.add_argument('--v9-input', default=str(C9.DOC))
    p.add_argument('--v9-protocol', default=str(C9.DOC/'PROTOCOL.json'))
    return p


def v9_args(args):
    a = copy.copy(args)
    a.output, a.protocol = args.v9_input, args.v9_protocol
    # This is an immutable V9 input binding, not the V10 protection policy.
    # Its verifier requiring the old HEAD is never called by this driver.
    a.prior_bindings = str(C9.DOC/'PROTECTION_BEFORE.json')
    return a


def output_path(args, name, existing=False):
    require(name == Path(name).name and name not in ('', '.', '..'), 'new V10 basename required')
    target_root = Path(args.output).absolute()
    require(not any(p.is_symlink() for p in (target_root, *target_root.parents)), 'symlink output ancestry')
    root = target_root.resolve()
    require(root == DOC.resolve() or root.is_relative_to(PRIVATE.resolve()), 'exact new V10 DOC/private output required')
    for dep in (args.source_root, args.baseline_root, args.parent, args.point_parent, args.v9_input, args.fits):
        other = Path(dep).resolve()
        require(not root.is_relative_to(other) and not other.is_relative_to(root), 'output overlaps parent/source/model inputs')
    path = root/name
    require(not path.is_symlink() and (existing or not path.exists()), 'preserve '+name)
    return path


def input_paths(args):
    a = v9_args(args)
    paths = {'reused_v9:'+key: path for key, path in C9.input_paths(a).items()}
    paths.update({name:CODE/name for name in ('__init__.py', 'audit.py', 'runner.py', 'validation_checks.py')})
    paths.update(audit_contract=DOC/'AUDIT_CONTRACT_KO.md', reproduce=DOC/'REPRODUCE.md',
        protection=Path(args.prior_bindings), cohort=Path(args.cohort),
        v9_protocol=Path(args.v9_protocol), v9_geometry=Path(args.v9_input)/'GEOMETRY_SEALED.jsonl.gz',
        v9_ledgers=Path(args.v9_input)/'CONTROL_LEDGERS.jsonl.gz', v9_seal=Path(args.v9_input)/'GEOMETRY_SEAL.json',
        v9_receipt=Path(args.v9_input)/'CONTROL_RECEIPT.json',
        v9_population=Path(args.v9_input)/'PARENT_POPULATION_CHECKS.json',
        v9_validation_protocol=Path(args.v9_input)/'VALIDATION_PROTOCOL.json',
        v9_validation=Path(args.v9_input)/'VALIDATION_CHECKS.json')
    for name in ('PROTOCOL.json', 'ENDPOINT_VALIDATION_CHECKS.json'):
        paths['original_v6:'+name] = REPO/'_docs/experiments/pallet_partial_line_heldout_20261010_v6'/name
    for version, name in (('pallet_partial_line_heldout_20261010_v6','solver.py'),
        ('pallet_partial_line_independent_20261010_v5','solver.py'),
        ('pallet_cornerwise_independent_20261010_v4','pose.py'),
        ('pallet_three_head_observation_20261010_v7','validation_checks.py'),
        ('pallet_same_observation_controls_20261010_v8','runner.py'),
        ('pallet_three_head_observation_20261010_v7','run.py')):
        paths['original_code:'+version+'/'+name] = REPO/'scripts/research'/version/name
    return paths


def input_bindings(args):
    return {key:binding(path) for key,path in input_paths(args).items()}


def metadata(args):
    a = v9_args(args)
    core = C9.verify_protocol(a)
    folder = Path(args.v9_input)
    seal, receipt, validation, own = (read(folder/n) for n in
        ('GEOMETRY_SEAL.json', 'CONTROL_RECEIPT.json', 'VALIDATION_CHECKS.json', 'VALIDATION_PROTOCOL.json'))
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and
        seal['cleanup_completed_before_seal'] is True and seal['no_scored_or_human_inputs'] is True and
        seal['frames'] == seal['rows'] == seal['ledger_rows'] == 245, 'complete GT-free V9 seal required')
    require(receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_error'] is None and
        receipt['environment_and_monkeypatch_cleanup_completed'] is True, 'V9 cleanup incomplete')
    require(validation['complete'] is True and validation['passed'] is True and
        validation['inputs'] == own['inputs'], 'independent V9 geometry PASS required')
    for name,key in (('PROTOCOL.json','protocol'), ('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('CONTROL_LEDGERS.jsonl.gz','ledgers'), ('CONTROL_RECEIPT.json','control_receipt'),
        ('PARENT_POPULATION_CHECKS.json','parent_population_checks')):
        bound(folder/name,seal[key],'V9 seal '+key)
    bound(folder/'VALIDATION_PROTOCOL.json',validation['own_protocol'],'V9 geometry own protocol')
    bound(C9.CODE/'validation_checks.py',validation['checker'],'V9 actual independent checker')
    original = read(REPO/'_docs/experiments/pallet_partial_line_heldout_20261010_v6/PROTOCOL.json')
    endpoint = original['endpoint_validation_policy']
    require(endpoint['endpoint_RMS_threshold_px'] == 8. and endpoint['validation_robust'] is True and
        endpoint['unavailable_is_no_match'] is False and endpoint['unavailable_is_frame_failure'] is False and
        endpoint['observed_line_used_to_choose_validation_pose'] is False and
        endpoint['validation_projection_becomes_observation'] is False, 'original frozen V6 endpoint policy differs')
    cpu = read(REPO/'_docs/experiments/pallet_partial_line_heldout_20261010_v6/ENDPOINT_VALIDATION_CHECKS.json')
    require(cpu['passed'] is True, 'original endpoint CPU contract PASS required')
    return core, seal


def protect(args):
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    require(Path(args.prior_bindings).resolve() == (DOC/'PROTECTION_BEFORE.json').resolve(),
        'V10 requires its own new post-V9 protection snapshot')
    snapshot = read(args.prior_bindings)
    data = snapshot['tracked']
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(data['original_research_head'] == head, 'snapshot must bind current published parent HEAD')
    names = {row['path'] for row in data['files']}
    require({'scripts/research/pallet_sparse_local_line_20261010_v9/solver.py',
        '_docs/experiments/pallet_sparse_local_line_20261010_v9/PROTOCOL.json'} <= names,
        'snapshot predates V9 publication; stale V8 HEAD protection rejected')
    result = verify(data,args.source_root)
    require(result['passed'], 'published/source bytes changed')
    for b in snapshot.get('additional_completed_audits', []):
        bound(REPO/b['path'],b,'additional preserved audit '+b['path'])
    return dict(published_parent_HEAD=head, unchanged_original_verifier=result,
        own_latest_HEAD_snapshot=True, published_V7_V8_V9_protected=True)


def packets(args, completion):
    a = v9_args(args)
    local_rows, local_ledgers = iter(rows(Path(args.v9_input)/'GEOMETRY_SEALED.jsonl.gz')), iter(rows(Path(args.v9_input)/'CONTROL_LEDGERS.jsonl.gz'))
    for parent, obs, _point in C9.parent_frames(a):
        local, old_ledger = next(local_rows,None), next(local_ledgers,None)
        require(local is not None and old_ledger is not None and local['id'] == parent['id'] == old_ledger['id'] and
            local['session'] == parent['session'] == old_ledger['session'] and local['method'] == C9.PRIMARY,
            'V9 local rows/ledgers missing, reordered or mismatched')
        packet = dict(local_packet=dict(parent_row=parent, role_observation=obs,
            native_N3_points=obs['native_N3_points'], parent_completion=completion), local_row=local)
        require(local['parent_geometry_row_semantic_sha256'] == semantic(parent) and
            local['parent_observation_row_semantic_sha256'] == semantic(obs) and
            old_ledger['parent_packet_semantic_sha256'] == semantic(packet['local_packet']), 'original sealed source join differs')
        require(not {'pose','corner','mask_audit','GT','ground_truth','human_states_native'}.intersection(local),
            'scored/human data entered geometry stream')
        yield packet, old_ledger
    require(next(local_rows,None) is None and next(local_ledgers,None) is None, 'extra V9 rows outside cohort')


def population(args):
    a = v9_args(args)
    reused = C9.population_proof(a,C9.input_bindings(a))
    ids, sessions = [], Counter()
    for packet,_ledger in packets(args,reused['completion']):
        ids.append(packet['local_row']['id']); sessions[packet['local_row']['session']] += 1
    cohort = read(args.cohort)
    require(ids == cohort['ids'] and len(ids) == len(set(ids)) == 245 and
        cohort['counts'] == dict(clean=153,moderate=92,severe_excluded=74,original=319), 'complete ordered easy-medium245 required')
    return dict(schema='full_GT_free_unused_line_audit_parent_population_checks_v10', complete=True,passed=True,
        frames=245,v9_geometry_rows=245,v9_ledger_rows=245,ordered_ids=ids,sessions=dict(sessions),
        completion=reused['completion'],reused_v9_population_proof=reused,
        reused_original_schema_retained=True, full_parent_population_rows_retained=0,
        one_frame_parent_ROLE_point_control_local_candidate_rows_kept=True,
        GT_scored_or_human_inputs_decoded=False,new_model_image_PnP_LOCAL_GT_calls=0)


def frozen(args):
    own = read(args.protocol)
    require(own['schema'] == 'fixed_frozen_unused_line_endpoint_audit_protocol_v10' and
        own['policy'] == POLICY and own['inputs'] == input_bindings(args) and own['frames'] == 245 and
        own['new_final_pose_paths'] == own['new_LOCAL_optimizer_calls'] == own['new_GT_scores'] == 0,
        'own diagnosis code/input/scope differs')
    bound(Path(args.output)/'PARENT_POPULATION_CHECKS.json',own['parent_population_checks'],'new parent population proof')
    require(own['source_root'] == str(Path(args.source_root).resolve()) and
        own['baseline_root'] == str(Path(args.baseline_root).resolve()), 'explicit split roots differ')
    metadata(args)
    return own


def freeze(args):
    require(Path(args.protocol).resolve() == (DOC/'AUDIT_PROTOCOL.json').resolve(), 'formal own V10 protocol required')
    destination=copy.copy(args);destination.output=str(DOC)
    output_path(destination,'AUDIT_PROTOCOL.json');output_path(args,'PARENT_POPULATION_CHECKS.json')
    inputs=input_bindings(args);metadata(args);before=protect(args)
    proof=population(args)
    require(input_bindings(args)==inputs,'inputs changed during parent population verification')
    write_new(output_path(args,'PARENT_POPULATION_CHECKS.json'),proof)
    write_new(Path(args.protocol),dict(schema='fixed_frozen_unused_line_endpoint_audit_protocol_v10',policy=POLICY,
        inputs=inputs,frames=245,method=METHOD,parent_completion=proof['completion'],
        parent_population_checks=binding(Path(args.output)/'PARENT_POPULATION_CHECKS.json'),
        protected_before=before,source_root=str(Path(args.source_root).resolve()),
        baseline_root=str(Path(args.baseline_root).resolve()),
        own_freeze_after_V9_publication_before_own_validation_PnP=True,
        new_final_pose_paths=0,new_LOCAL_optimizer_calls=0,new_GT_scores=0,new_training_RGB=0,
        existing_V9_accuracy_known=True,evaluation_is_unseen=False,
        no_accuracy_or_runtime_claim=True,no_automatic_retry=True))
    frozen(args);print('UNUSED_LINE_AUDIT_FROZEN',binding(args.protocol)['sha256'],flush=True)


def preflight(args):
    frozen(args);protect(args)
    for name in OUTPUTS:output_path(args,name)
    snapshot=C9.quiet_snapshot()
    passed=bool(snapshot['quiet'] and snapshot['temperature_under_80'])
    write_new(output_path(args,'AUDIT_PREFLIGHT.json'),dict(schema='independent_unused_line_audit_preflight_v10',
        passed=passed,protocol=binding(args.protocol),resource_snapshot=snapshot,actual_validation_PnP_requests=0,
        protection=protect(args),automatic_retry=False))
    require(passed,'AUDIT_PENDING: competing workload/thermal guard; preflight preserved')
    print('UNUSED_LINE_AUDIT_PREFLIGHT_PASS',flush=True)


@contextmanager
def numerical_scope(calls, dispatch, math_entries, cleanup):
    from . import audit as A
    from ..pallet_sparse_local_line_20261010_v9 import pipeline as P9, solver as S9
    from ..pallet_partial_line_independent_20261010_v5 import solver as S5
    from ..pallet_partial_line_heldout_20261010_v6 import solver as S6
    from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline
    from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
    from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank
    import numpy as np
    targets=((P9,'solve_local_packet'),(S9.LocalPointLineBank,'__init__'),(S9.LocalPointLineBank,'solve'),
        (S9,'least_squares'),(S5,'least_squares'),(S5.PointLineBank,'solve'),
        (S6.EndpointValidatedPointLineBank,'solve'),(Pipeline,'__init__'),(Pipeline,'capture'),(Pipeline,'outcomes'))
    saved=[(owner,name,getattr(owner,name)) for owner,name in targets]
    native_solve = PoseBank.solve
    saved.append((PoseBank, 'solve', native_solve))
    native_math = [(np.linalg,'svd','numpy_svd'),(HypothesisBank,'_jacobian','native_pose_jacobian')]
    math_saved = [(owner,name,getattr(owner,name),key) for owner,name,key in native_math]
    openings=[(builtins,'open',builtins.open),(io,'open',io.open)]
    forbidden=('PREDICTIONS.jsonl','POSTHOC','METRICS.json','TARGETS.json','GROUND_TRUTH',
        'STATIC_VISIBILITY','annotation','COCO','READY_PREPARED_TARGETS','target_arrays',
        '.pt','.pth','/rgb/','/depth/')
    def blocked(*a,**kw):
        calls['forbidden_model_or_LOCAL_entries']+=1
        raise RuntimeError('UNUSED_LINE_AUDIT_CANARY: model/LOCAL/final fitting forbidden')
    def guarded(fn):
        def call(path,*a,**kw):
            if isinstance(path,(str,Path)) and any(s.lower() in str(path).lower() for s in forbidden):
                calls['forbidden_asset_reads']+=1
                raise RuntimeError('UNUSED_LINE_AUDIT_CANARY: GT/image/model asset read forbidden')
            return fn(path,*a,**kw)
        return call
    def validation_call(*a, **kw):
        dispatch['entries'] += 1
        try:
            value = native_solve(*a, **kw)
        except BaseException:
            dispatch['exceptions'] += 1
            raise
        dispatch['returns'] += 1
        return value
    def math_wrapper(fn,key):
        def call(*a, **kw):
            math_entries[key+':entries'] += 1
            try:
                value = fn(*a, **kw)
            except BaseException:
                math_entries[key+':exceptions'] += 1
                raise
            math_entries[key+':returns'] += 1
            return value
        return call
    try:
        cleanup['numerical_scope_started'] = True
        for owner,name,_ in saved[:-1]:setattr(owner,name,blocked)
        PoseBank.solve = validation_call
        for owner,name,fn,key in math_saved:setattr(owner,name,math_wrapper(fn,key))
        for owner,name,fn in openings:setattr(owner,name,guarded(fn))
        yield A
    finally:
        for owner,name,fn,key in reversed(math_saved):setattr(owner,name,fn)
        for owner,name,fn in reversed(openings+saved):setattr(owner,name,fn)
        cleanup['numerical_scope_restored'] = all(getattr(owner,name) is fn for owner,name,fn in openings+saved)
        cleanup['native_math_dispatch_restored'] = all(getattr(owner,name) is fn for owner,name,fn,key in math_saved)
        require(cleanup['numerical_scope_restored'] and cleanup['native_math_dispatch_restored'],
            'numerical canary/open/native solve/Jacobian/SVD restoration failed')


@contextmanager
def primitive_scope(entries, returns, exceptions, cleanup):
    import cv2
    saved = {name:getattr(cv2,name) for name in PRIMITIVES}
    threads = cv2.getNumThreads()
    cleanup['primitive_scope_started'] = True
    try:
        cv2.setNumThreads(1)
        for name,fn in saved.items():
            def counted(*a, _name=name, _fn=fn, **kw):
                entries[_name] += 1
                try:
                    value = _fn(*a, **kw)
                except BaseException:
                    exceptions[_name] += 1
                    raise
                returns[_name] += 1
                return value
            setattr(cv2,name,counted)
        yield dict(opencv_version=cv2.__version__, opencv_threads=1,
            actual_primitive_dispatch_instrumentation=True)
    finally:
        for name,fn in saved.items():setattr(cv2,name,fn)
        cv2.setNumThreads(threads)
        cleanup['primitive_scope_restored'] = all(getattr(cv2,name) is fn for name,fn in saved.items())
        cleanup['opencv_thread_count_restored'] = cv2.getNumThreads() == threads
        require(cleanup['primitive_scope_restored'] and cleanup['opencv_thread_count_restored'],
            'OpenCV functions/thread count restoration failed')


def run(args):
    own=frozen(args);before=protect(args)
    pre=read(Path(args.output)/'AUDIT_PREFLIGHT.json')
    require(pre['passed'] is True and same_binding(pre['protocol'],binding(args.protocol)), 'independent quiet preflight required')
    for name in OUTPUTS:output_path(args,name)
    proof=read(Path(args.output)/'PARENT_POPULATION_CHECKS.json')
    counts=Counter({name:0 for name in ('frames_started','frames_complete','audit_rows','ledger_rows',
        'endpoint_validation_requests','endpoint_validation_records_complete')})
    operations,canary=Counter(),Counter()
    dispatch=Counter({name:0 for name in ('entries','returns','exceptions')})
    math_entries = Counter({name+':'+status:0 for name in ('numpy_svd','native_pose_jacobian')
        for status in ('entries','returns','exceptions')})
    primitive, primitive_returns, primitive_exceptions = (Counter({name:0 for name in PRIMITIVES}) for _ in range(3))
    cleanup, environment, library_environment = {}, None, None
    environment_names = ('PALLET_SOURCE_ROOT','PALLET_BASELINE_ROOT')
    environment_before = {name:os.environ.get(name) for name in environment_names}
    statuses={name:0 for name in ('SUPPORTED_RETAIN','CONTRADICTED_REJECT','UNVERIFIED_RETAIN')}
    resources,streams,cleanup_errors=[],{},[]
    error,cleanup_error,active=None,None,None
    began=time.monotonic();last_probe=began
    def probe(phase):
        snap=dict(phase=phase,frames_complete=counts['frames_complete'],**C9.quiet_snapshot())
        write_new(output_path(args,'AUDIT_RESOURCE_%03d.json'%len(resources)),snap);resources.append(snap)
        require(snap['quiet'] and snap['temperature_under_80'],'AUDIT_PENDING: competing workload/thermal guard')
    write_new(output_path(args,'AUDIT_STARTED.json'),dict(status='STARTED',protocol=binding(args.protocol),
        configured_frames=245,new_model_LOCAL_GT_calls=0,automatic_retry=False))
    try:
        probe('before_validation')
        with C9.original_environment(args) as environment:
            from ..pallet_three_head_observation_20261010_v7.run import RowWriter
            streams['audit']=RowWriter(output_path(args,'AUDIT_ROWS.jsonl.gz'),
                interrupted_path=output_path(args,'INTERRUPTED_AUDIT_ROWS.jsonl.gz'))
            streams['ledgers']=RowWriter(output_path(args,'AUDIT_LEDGERS.jsonl.gz'),
                interrupted_path=output_path(args,'INTERRUPTED_AUDIT_LEDGERS.jsonl.gz'))
            with numerical_scope(canary,dispatch,math_entries,cleanup) as A,primitive_scope(primitive,primitive_returns,primitive_exceptions,cleanup) as library_environment:
                for packet,old_ledger in packets(args,proof['completion']):
                    require(time.monotonic()-began<=args.remaining_seconds,'AUDIT_PENDING: operating limit')
                    if time.monotonic()-last_probe>=5:probe('during_validation');last_probe=time.monotonic()
                    counts['frames_started']+=1;active=dict(id=packet['local_row']['id'],input_packet=packet,execution_events=[])
                    p0=primitive.copy()
                    r0,e0,d0,m0 = primitive_returns.copy(),primitive_exceptions.copy(),dispatch.copy(),math_entries.copy()
                    def observe(event):
                        active['execution_events'].append(copy.deepcopy(event))
                        if event['phase']=='endpoint_request_started':counts['endpoint_validation_requests']+=1
                    row,ledger=A.audit_packet(packet,observer=observe)
                    active['audit_return']=row;active['ledger_return']=ledger
                    ledger.update(primitive_entry_calls={name:primitive[name]-p0[name] for name in primitive},
                        primitive_return_calls={name:primitive_returns[name]-r0[name] for name in PRIMITIVES},
                        primitive_exception_calls={name:primitive_exceptions[name]-e0[name] for name in PRIMITIVES},
                        native_validation_dispatch={name:dispatch[name]-d0[name] for name in ('entries','returns','exceptions')},
                        actual_native_math_dispatch={name:math_entries[name]-m0[name] for name in math_entries},
                        original_V9_ledger_semantic_sha256=semantic(old_ledger))
                    require(ledger['native_validation_dispatch']['entries']==ledger['endpoint_validation_requests'],
                        'actual native request dispatch differs from completed edge records')
                    operations.update(ledger['native_bank_delta'])
                    for record in row['endpoint_records']:
                        counts['endpoint_validation_records_complete']+=1
                        statuses[record['decision']]=statuses.get(record['decision'],0)+1
                    streams['audit'].write(row);streams['ledgers'].write(ledger)
                    counts.update(frames_complete=1,audit_rows=1,ledger_rows=1);active=None
                    if counts['frames_complete']%32==0:print('UNUSED_LINE_AUDIT',counts['frames_complete'],245,flush=True)
            require(counts['frames_complete']==counts['audit_rows']==counts['ledger_rows']==245,
                'diagnosis did not retain all245')
            require(not canary,'forbidden entry/read occurred')
            probe('after_validation')
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        def cleanup_failure(exc,phase):
            nonlocal cleanup_error
            value=dict(type=type(exc).__name__,message=str(exc),phase=phase)
            cleanup_errors.append(value)
            if cleanup_error is None:cleanup_error=value
        cleanup['split_environment_restored'] = {name:os.environ.get(name) for name in environment_names} == environment_before
        if not cleanup['split_environment_restored']:
            cleanup_failure(RuntimeError('original split environment not restored'),'environment_restore')
        if active is not None:
            active.update(actual_validation_dispatch=dict(dispatch),actual_OpenCV_entry_calls=dict(primitive),
                actual_OpenCV_return_calls=dict(primitive_returns),actual_OpenCV_exception_calls=dict(primitive_exceptions),
                actual_native_math_dispatch=dict(math_entries))
            try:write_new(output_path(args,'INTERRUPTED_AUDIT_PACKET.json'),active)
            except BaseException as exc:cleanup_failure(exc,'active_packet')
        for stream in streams.values():
            try:stream.close();require(not stream.close_errors,'stream close/fsync failed')
            except BaseException as exc:cleanup_failure(exc,'close')
        try:
            after=protect(args);require(after==before,'protected bytes changed');frozen(args)
        except BaseException as exc:
            after=None;cleanup_failure(exc,'protection_or_bindings')
        restored=all(cleanup.get(key) is True for key in ('split_environment_restored',
            'numerical_scope_restored','native_math_dispatch_restored','primitive_scope_restored','opencv_thread_count_restored'))
        complete=error is None and cleanup_error is None and restored and counts['frames_complete']==245
        for stream in streams.values():
            try:
                if complete:stream.promote()
                else:stream.preserve_interrupted()
                require(not stream.preservation_errors,'stream preservation failed')
            except BaseException as exc:
                complete=False;cleanup_failure(exc,'publish_or_preserve')
        result=dict(schema='frozen_unused_line_endpoint_audit_receipt_v10',complete=complete,error=error,
            cleanup_error=cleanup_error,cleanup_errors=cleanup_errors,protocol=binding(args.protocol),parent_population_checks=own['parent_population_checks'],
            execution=dict(counts),decision_counts=statuses,native_bank_operation_counts=dict(operations),
            actual_OpenCV_entry_calls=dict(primitive),actual_OpenCV_return_calls=dict(primitive_returns),
            actual_OpenCV_exception_calls=dict(primitive_exceptions),actual_OpenCV_primitive_names=list(PRIMITIVES),
            actual_native_validation_dispatch=dict(dispatch),primitive_restoration_checks=cleanup,
            actual_native_math_dispatch=dict(math_entries),
            forbidden_entries_or_asset_reads=dict(canary),environment_and_monkeypatch_cleanup_completed=restored and cleanup_error is None,
            explicit_split_environment=environment,library_environment=library_environment,
            protection_before=before,protection_after=after,resource_snapshots=resources,
            GT_read_allowed=False,new_model_RGB_LOCAL_final_pose_GT_score_calls=0,
            new_validation_point_PnP_requests=dispatch['entries'],
            elapsed_seconds=time.monotonic()-began,replay_wall_is_deployment_latency=False,
            raw_rows={name:binding(stream.final_path) if stream.final_path.exists() else binding(stream.interrupted_path)
                if stream.interrupted_path.exists() else None for name,stream in streams.items()},
            stream_counts={name:stream.count for name,stream in streams.items()},automatic_retry=False)
        result['stream_diagnostics']={name:dict(rows=stream.count,published=stream.published,
            pending_basename=stream.pending_path.name,final_basename=stream.final_path.name,
            interrupted_basename=stream.interrupted_path.name,close_errors=stream.close_errors,
            preservation_errors=stream.preservation_errors) for name,stream in streams.items()}
        write_new(output_path(args,'AUDIT_RECEIPT.json'),result)
        if complete:
            write_new(output_path(args,'AUDIT_SEAL.json'),dict(schema='frozen_unused_line_endpoint_audit_seal_v10',
                complete=True,frames=245,rows=245,ledger_rows=245,GT_read_allowed=False,
                cleanup_completed_before_seal=True,protocol=binding(args.protocol),
                audit=result['raw_rows']['audit'],ledgers=result['raw_rows']['ledgers'],
                receipt=binding(Path(args.output)/'AUDIT_RECEIPT.json'),
                parent_population_checks=own['parent_population_checks'],new_final_pose_paths=0,new_GT_scores=0))
    print('UNUSED_LINE_AUDIT',bool(result['complete']),dict(counts),flush=True)
    require(result['complete'],'diagnosis incomplete; original receipt/prefix preserved')


def main():
    args=parser().parse_args()
    require(args.remaining_seconds > 0, 'positive operating limit required')
    {'freeze':freeze,'preflight':preflight,'run':run}[args.stage](args)


if __name__=='__main__':
    main()
