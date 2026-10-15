"""Complete canonical29b CAL46 after a verified host restart, without rewrites.

Source-only preparation is separate from actual observation GO. The original
PROTOCOL/PREPARATION/CAL_EXECUTION and all32 completed spools stay unchanged.
Only missing14 CAL inputs call the identical frozen v29.observe_one. New
resume metadata supplements, rather than replaces, the original source seal.
Bounded pending tasks and row spooling retain no completed observation dicts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
from pathlib import Path
import multiprocessing
import time
import json

from . import observe_geometry_guided_edges_v4_29b as B

C,V,OLD=B.C,B.V,B.OLD
OUTPUT=B.OUTPUT
PLAN=OUTPUT.parent/'geometry_guided_observation_resume_plan_v4_29c/PREDECLARATION.json'
RESTART=C.DOC/'publication_20261015_v1/HOST_RESTART_AND_PARTIAL_CAL.json'
PREFLIGHT='CAL_RESUME_PREFLIGHT_SEAL_V1.json'
EXECUTION='CAL_RESUME_EXECUTION_PROTOCOL_V1.json'
LIVE='CAL_RESUME_LIVE_PROGRESS_V1.json'
EXECUTION_SEAL='CAL_RESUME_EXECUTION_SEAL_V1.json'
SUPPLEMENT='CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json'
FAILURE='CAL_RESUME_EXECUTION_FAILURE_V1.json'


def _source_bindings():
    paths=[Path(__file__),Path(__file__).with_name('test_resume_geometry_guided_edges_v4_29c.py'),
        Path(B.__file__),Path(V.__file__),Path(B.P.__file__)]
    return {str(p.resolve().relative_to(C.ROOT)):C.sha(p)for p in paths}


def _check_source_map(mapping):
    for relative,expected in mapping.items():
        if C.sha(C.ROOT/relative)!=expected:raise ValueError('Resume source changed: '+relative)


def declare():
    if PLAN.exists():raise FileExistsError('Preserve resume declaration')
    restart=C.read(RESTART)
    if (not restart['verification']['all_passed']or restart['verification']['failure_count']
            or restart['partial_observation']['complete_spool_rows']!=32):
        raise ValueError('Verified exact32 restart receipt required')
    PLAN.parent.mkdir(parents=True,exist_ok=False)
    C.write(PLAN,dict(schema='EXACT32_CANONICAL29B_CAL_RESUME_PREDECLARATION_V29C',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256=_source_bindings(),
        restart_receipt=OLD._binding(RESTART),canonical_output=str(OUTPUT.relative_to(C.ROOT)),
        central_predeclaration_sha256=B.P.CENTRAL_PREDECLARATION_SHA256,
        expected_existing_CAL_rows=32,expected_missing_CAL_rows=14,final_CAL_rows=46,total_population=238,
        fixed_worker='observe_geometry_guided_edges_v4_29.observe_one',ranking_adapter_applied=False,
        original_files_immutable=['PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz',
            'CAL_EXECUTION_PROTOCOL.json','CAL_LIVE_PROGRESS.json','REUSE_SEAL.json','all32 original row_spool files'],
        new_metadata_names=[PREFLIGHT,EXECUTION,LIVE,EXECUTION_SEAL,SUPPLEMENT,FAILURE],
        completed_rows_never_remeasured_or_overwritten=True,missing14_selection='exact original ordered CAL set difference',
        pixel_NMS_TLS_guides_graph_states_q_weights_regions_or_noise_changes=0,
        no_new_observation_started_during_declaration=True,actual_observation_requires_explicit_root_GO=True,
        ROOT_publication_completion_required_before_actual_resume=True,
        final_journal='original29b _emit_partition, original-order46 with extra resume seal reference',
        REST192='unchanged29b observe only after fullCAL and metadata success; separate root GO',
        new_RGB_NN_posefit_labels_refs_or_scores_read=0))
    print(json.dumps(dict(plan=str(PLAN),sha256=C.sha(PLAN))),flush=True)


def _verify_spool_set(output,metadata,expected):
    """Single-row/SHA/canonical/input-math proof; no image/model/label IO."""
    output=Path(output);paths=set((output/'row_spool').glob('*.jsonl.gz'))
    wanted={B._row_path(output,ident)for ident in expected}
    if paths!=wanted:raise ValueError('Exact complete spool set required; no extra partial/new row')
    result={}
    for ident,entry in expected.items():
        path=B._row_path(output,ident);row=B._read_single(path,entry);m=metadata.get(ident,{})
        if (row['partition']!='CAL'or m.get('partition')!='CAL'or row['schema']!=V.ROW_SCHEMA
                or row['observation_generation']!=B.P.OBSERVATION_GENERATION
                or row['unchanged_observation_math_sha256']!=m.get('math_sha256')
                or row['unchanged_observation_math_sha256']!=entry['unchanged_observation_math_sha256']):
            raise ValueError('Exact original CAL schema/state/query math binding required')
        result[ident]=B._index_row(row,path)
    return result


def preflight(output=OUTPUT):
    output=Path(output).resolve()
    if(output/PREFLIGHT).exists():raise FileExistsError('Preserve completed resume preflight')
    plan=C.read(PLAN);_check_source_map(plan['source_sha256'])
    if C.sha(RESTART)!=plan['restart_receipt']['sha256']:raise ValueError('Sealed host-restart receipt changed')
    protocol,prep=B._load(output)
    if output!=C.ROOT/plan['canonical_output']:raise ValueError('Keep canonical29b path')
    if(output/'CAL_OBSERVATION_SEAL.json').exists()or(output/'CAL_OBSERVATIONS.jsonl.gz').exists():
        raise FileExistsError('Never replace completed/partial CAL journal')
    receipt=C.read(RESTART);partial=receipt['partial_observation'];expected={r['id']:r for r in partial['spools']}
    metadata={}
    for task in C.rows(output/'INPUTS.jsonl.gz'):
        metadata[task['id']]=dict(partition=task['partition'],math_sha256=task['unchanged_observation_math_sha256'])
    if list(metadata)!=prep['ordered_ids']or len(metadata)!=238:raise ValueError('Exact prepared238 order required')
    if len(expected)!=32 or not receipt['verification']['all_passed']:raise ValueError('Verified exact32 restart rows required')
    entries=_verify_spool_set(output,metadata,expected)
    missing=[ident for ident in prep['ordered_ids']if metadata[ident]['partition']=='CAL'and ident not in entries]
    if len(missing)!=14 or len(entries)+len(missing)!=46:raise ValueError('Only exact missing14 CAL may be queried')
    immutable={str((output/name).relative_to(C.ROOT)):C.sha(output/name)for name in
        ('PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz','CAL_EXECUTION_PROTOCOL.json','CAL_LIVE_PROGRESS.json','REUSE_SEAL.json')}
    immutable.update({str(B._row_path(output,ident).relative_to(C.ROOT)):entry['sha256']for ident,entry in entries.items()})
    C.write(output/PREFLIGHT,dict(schema='EXACT32_CANONICAL29B_CAL_RESUME_PREFLIGHT_SEAL_V29C',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256=plan['source_sha256'],
        plan=OLD._binding(PLAN),restart_receipt=OLD._binding(RESTART),
        original_protocol=OLD._binding(output/'PROTOCOL.json'),original_preparation=OLD._binding(output/'PREPARATION_SEAL.json'),
        original_execution=OLD._binding(output/'CAL_EXECUTION_PROTOCOL.json'),immutable_file_sha256=immutable,
        exact_existing_entries=entries,existing_rows=32,missing_CAL_ids=missing,missing_rows=14,final_CAL_rows=46,
        all238_inputs_source_protected_maps_verified=True,new_RGB_pixel_queries=0,new_NN_posefit_ref_label_decodes=0))
    print(json.dumps(dict(preflight_rows=32,missing=14,sha256=C.sha(output/PREFLIGHT))),flush=True)


def _collect_missing(output,missing_tasks,entries,math_by_id,workers):
    """Run the exact old worker, bounded queue; persist metadata only."""
    output=Path(output);iterator=iter(missing_tasks);pending={};started=time.monotonic();new_entries={}
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=V._init_worker)
    def fill():
        while len(pending)<workers:
            task=next(iterator,None)
            if task is None:return
            if task['id']in entries:raise ValueError('Existing row must never reach image worker')
            pending[pool.submit(V.observe_one,task)]=task['id']
    try:
        fill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ident=pending.pop(future);row=future.result();path=B._row_path(output,ident)
                if (row['id']!=ident or row['partition']!='CAL'or row['schema']!=V.ROW_SCHEMA
                        or row['observation_generation']!=B.P.OBSERVATION_GENERATION
                        or row['unchanged_observation_math_sha256']!=math_by_id[ident]):
                    raise ValueError('New row must bind its original CAL input')
                C.save_rows(path,[row]);entry=B._index_row(row,path);entries[ident]=entry;new_entries[ident]=entry
                progress=dict(partition='CAL',original_rows_reused=32,new_rows_completed=len(new_entries),
                    completed=len(entries),total=46,id=ident,summary=row['summary'],
                    elapsed_seconds=round(time.monotonic()-started,3),retained_completed_rows_in_RAM=0)
                C.write(output/LIVE,progress);print(json.dumps(progress),flush=True);del row
            done.clear();del future;fill()
    except BaseException as error:
        C.write(output/FAILURE,dict(error=repr(error),new_completed_ids=list(new_entries),
            original_rows_preserved=True,no_final_CAL_seal=True))
        OLD._stop_pool(pool);raise
    else:pool.shutdown(wait=True)
    return new_entries


def resume(output=OUTPUT,workers=3,*,root_actual_observation_GO=False):
    # This gate precedes *all* IO, including loading publication/reuse state.
    if not root_actual_observation_GO:raise ValueError('Wait for explicit root actual-observation/publication-complete GO')
    if isinstance(workers,bool)or not isinstance(workers,int)or not 1<=workers<=8:raise ValueError('Use1..8 root-approved workers')
    output=Path(output).resolve();protocol,prep=B._load(output);pre=C.read(output/PREFLIGHT)
    _check_source_map(pre['source_sha256']);_check_source_map(pre['immutable_file_sha256'])
    if C.sha(PLAN)!=pre['plan']['sha256']or C.sha(RESTART)!=pre['restart_receipt']['sha256']:raise ValueError('Resume preflight bindings changed')
    for name in(EXECUTION,EXECUTION_SEAL,SUPPLEMENT,'CAL_OBSERVATIONS.jsonl.gz','CAL_OBSERVATION_SEAL.json'):
        if(output/name).exists():raise FileExistsError('Never overwrite a prior execution/final archive: '+name)
    metadata={}
    for task in C.rows(output/'INPUTS.jsonl.gz'):
        metadata[task['id']]=dict(partition=task['partition'],math_sha256=task['unchanged_observation_math_sha256'])
    entries=_verify_spool_set(output,metadata,pre['exact_existing_entries'])
    missing=set(pre['missing_CAL_ids'])
    if len(entries)!=32 or len(missing)!=14 or missing&set(entries):raise ValueError('Exact32+14 disjoint CAL set required')
    C.write(output/EXECUTION,dict(schema='CANONICAL29B_CAL_RESUME_EXECUTION_PROTOCOL_V29C',
        UTC=datetime.now(timezone.utc).isoformat(),workers=workers,maximum_pending_tasks=workers,
        same_observation_function='observe_geometry_guided_edges_v4_29.observe_one',
        exact_unchanged_worker=OLD._binding(Path(V.__file__)),producer=OLD._binding(Path(B.P.__file__)),
        source_sha256=pre['source_sha256'],preflight_seal=OLD._binding(output/PREFLIGHT),
        plan=OLD._binding(PLAN),reused_rows=32,new_CAL_rows=14,missing_CAL_ids=pre['missing_CAL_ids'],
        original_execution_protocol_unmodified=True,root_actual_observation_GO_received=True,
        completed_rows_never_remeasured=True,pixel_NMS_TLS_guides_graph_or_guard_changes=0))
    def tasks():
        for task in C.rows(output/'INPUTS.jsonl.gz'):
            if task['id']in missing:yield task
    new=_collect_missing(output,tasks(),entries,{k:m['math_sha256']for k,m in metadata.items()},workers)
    if set(new)!=missing or len(entries)!=46:raise ValueError('Every exact missing14 and original32 required')
    _check_source_map(pre['source_sha256']);_check_source_map(pre['immutable_file_sha256'])
    C.write(output/EXECUTION_SEAL,dict(schema='CANONICAL29B_CAL_RESUME_EXECUTION_SEAL_V29C',
        rows=14,original_rows_reused=32,final_CAL_rows=46,new_row_entries=new,
        source_sha256=pre['source_sha256'],execution_protocol=OLD._binding(output/EXECUTION),
        preflight=OLD._binding(output/PREFLIGHT),plan=OLD._binding(PLAN),
        all_original_source_input_protected_maps_verified=True,original32_immutable=True,
        same_observe_one_NMS_TLS_guides_graph=True,new_ref_labels_NN_posefit_calls=0))
    # The original emitter makes each canonical CAL output for the FIRST time.
    # Its original execution reference remains; the initial seal is augmented
    # before publishing the completion supplement or notifying consumers.
    B._emit_partition(output,'CAL',protocol,prep,entries)
    seal=C.read(output/'CAL_OBSERVATION_SEAL.json')
    seal.update(resume_execution_seal=OLD._binding(output/EXECUTION_SEAL),
        resume_preflight_seal=OLD._binding(output/PREFLIGHT),resume_source_sha256=pre['source_sha256'],
        original_execution_protocol_refers_to_preboot_start=True,complete_CAL46_includes_exact32_preboot_plus14_resumed=True)
    C.write(output/'CAL_OBSERVATION_SEAL.json',seal)
    C.write(output/SUPPLEMENT,dict(schema='CANONICAL29B_CAL46_RESUME_COMPLETION_SUPPLEMENT_V29C',
        rows=46,exact_preboot_rows=32,new_resumed_rows=14,source_sha256=pre['source_sha256'],
        resume_execution_seal=OLD._binding(output/EXECUTION_SEAL),
        CAL_observation_seal=OLD._binding(output/'CAL_OBSERVATION_SEAL.json'),
        original_protocol=OLD._binding(output/'PROTOCOL.json'),original_preparation=OLD._binding(output/'PREPARATION_SEAL.json'),
        original_execution=OLD._binding(output/'CAL_EXECUTION_PROTOCOL.json'),
        canonical_29b_path_unchanged=True,all_old_rows_and_original_frozen_files_unchanged=True,
        no_old_observation_remeasurement=True,producer_math_changes=0,new_ref_label_NN_posefit_calls=0))
    print(json.dumps(dict(sealed_CAL=46,exact_preboot_rows=32,resumed_rows=14,
        observation_seal_sha256=C.sha(output/'CAL_OBSERVATION_SEAL.json'),supplement_sha256=C.sha(output/SUPPLEMENT))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','preflight','resume'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--root-actual-observation-GO',action='store_true');args=parser.parse_args()
    if args.stage=='declare':declare()
    elif args.stage=='preflight':preflight(args.output)
    else:resume(args.output,args.workers,root_actual_observation_GO=args.root_actual_observation_GO)
