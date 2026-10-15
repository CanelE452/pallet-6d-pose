"""Bounded-memory orchestration of the immutable v29 image observer.

Only scheduling/storage changes: same byte-identical238 inputs, same worker,
guide policy and pixel/math producer. Exact completed checkpoint rows can be
reused without another image query. Pending tasks <= workers, completed rows
are immediately spooled and futures released. Final journals stream in the
original order, including every zero-edge row. No fitting or labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

from . import observe_geometry_guided_edges_v4_29 as V

C,OLD,P=V.C,V.OLD,V.P
OUTPUT=V.OUTPUT.parent/'geometry_guided_edge_observations_v4_29b'


def _row_path(output,ident):
    return Path(output)/'row_spool'/(hashlib.sha256(ident.encode()).hexdigest()+'.jsonl.gz')


def _index_row(row,path):
    return dict(id=row['id'],partition=row['partition'],summary=row['summary'],
        path=str(path.name),sha256=C.sha(path),canonical_row_sha256=OLD._digest(row),
        unchanged_observation_math_sha256=row['unchanged_observation_math_sha256'])


def _read_single(path,entry):
    if C.sha(path)!=entry['sha256']:raise ValueError('Durable row spool changed')
    stream=iter(C.rows(path));row=next(stream,None)
    if row is None or next(stream,None)is not None or row['id']!=entry['id']or OLD._digest(row)!=entry['canonical_row_sha256']:
        raise ValueError('Exact single canonical row required')
    return row


def _summary(entries):
    return dict(rows=len(entries),partition_counts=dict(Counter(e['partition']for e in entries)),
        selected_physical_edges=sum(e['summary']['selected_physical_edges']for e in entries),
        selected_partial_edges=sum(e['summary']['selected_partial_edges']for e in entries),
        actual_chain_hypotheses=sum(e['summary']['actual_chain_hypotheses']for e in entries),
        zero_selected_edge_frames=sum(e['summary']['zero_selected_edges']for e in entries),
        all_frames_and_abstentions_retained=True,S_U_local_queries=0,new_NN_posefit_reference_decodes=0)


def _load(output):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if (seal['rows']!=238 or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']
            or C.sha(output/'REUSE_SEAL.json')!=seal['reuse_seal_sha256']):raise ValueError('Complete frozen238 source/input/reuse preparation required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(protocol[key])
    return protocol,seal


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all old/partial generations')
    parent=C.read(V.OUTPUT/'PROTOCOL.json');prep=C.read(V.OUTPUT/'PREPARATION_SEAL.json')
    if (prep['rows']!=238 or C.sha(V.OUTPUT/'PROTOCOL.json')!=prep['protocol_sha256']
            or C.sha(V.OUTPUT/'INPUTS.jsonl.gz')!=prep['input_sha256']
            or C.sha(V.PLAN)!=P.CENTRAL_PREDECLARATION_SHA256):raise ValueError('Exact parent frozen source/math/input required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(parent[key])
    if(V.OUTPUT/'CAL_OBSERVATION_SEAL.json').exists():raise ValueError('This operational generation preserves interrupted CAL, not a replacement of a complete archive')
    source=dict(parent['source_sha256']);source[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(Path(__file__))
    inputs=dict(parent['input_sha256'])
    for name in('PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz','CAL_EXECUTION_PROTOCOL.json','CAL_OBSERVATIONS_checkpoint.jsonl.gz','CAL_LIVE_PROGRESS.json'):
        path=V.OUTPUT/name
        if path.exists():inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    output.mkdir(parents=True);(output/'row_spool').mkdir()
    shutil.copyfile(V.OUTPUT/'INPUTS.jsonl.gz',output/'INPUTS.jsonl.gz')
    if C.sha(output/'INPUTS.jsonl.gz')!=prep['input_sha256']:raise ValueError('Byte-identical input copy required')
    metadata={}
    for row in C.rows(output/'INPUTS.jsonl.gz'):
        metadata[row['id']]=dict(partition=row['partition'],math_sha256=row['unchanged_observation_math_sha256'])
    if list(metadata)!=prep['ordered_ids']or len(metadata)!=238:raise ValueError('No dropout or order change')
    checkpoint=V.OUTPUT/'CAL_OBSERVATIONS_checkpoint.jsonl.gz';reused={}
    with gzip.open(checkpoint,'rt')as stream:
        for raw in stream:
            row=json.loads(raw);ident=row['id'];expected=metadata.get(ident)
            if (ident in reused or expected is None or row['partition']!='CAL'or expected['partition']!='CAL'
                    or row['schema']!=V.ROW_SCHEMA or row['observation_generation']!=P.OBSERVATION_GENERATION
                    or row['unchanged_observation_math_sha256']!=expected['math_sha256']):raise ValueError('Exact completed original CAL row required')
            path=_row_path(output,ident)
            with gzip.open(path,'wt',compresslevel=5)as saved:saved.write(raw)
            entry=_index_row(row,path);entry.update(reused_without_new_RGB_query=True,
                source_json_line_sha256=hashlib.sha256(raw.encode()).hexdigest())
            reused[ident]=entry
    for relative,digest in inputs.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Original interruption archive changed')
    C.write(output/'REUSE_SEAL.json',dict(schema='EXACT_COMPLETED_V29_ROW_REUSE_SEAL_V29B',rows=len(reused),
        entries=reused,source_checkpoint=OLD._binding(checkpoint),new_RGB_queries=0,
        no_row_numeric_or_measurement_changes=True))
    protocol={**parent,'schema':'BOUNDED_MEMORY_SAME_FRESH_GEOMETRY_EDGE_PROTOCOL_V29B',
        'source_sha256':source,'input_sha256':inputs,'orchestration_only_change':True,
        'same_observe_one_source':OLD._binding(Path(V.__file__)),
        'same_producer_source':OLD._binding(Path(P.__file__)),
        'input_bytes_identical_to_original_preparation':True,'maximum_pending_tasks':'workers',
        'completed_retention':'ID/summary/digest only; exact per-row gzip spool; completed future pop',
        'finalization':'original ordered streaming one row at a time',
        'exact_old_completed_rows_reused':len(reused),'new_pixel_queries_during_this_preparation':0,
        'created_UTC':datetime.now(timezone.utc).isoformat()}
    C.write(output/'PROTOCOL.json',protocol)
    C.write(output/'ORCHESTRATION_CHANGE_RECEIPT.json',dict(schema='MEMORY_SAFETY_ORCHESTRATION_CHANGE_V29B',
        original_run=OLD._binding(V.OUTPUT/'CAL_EXECUTION_PROTOCOL.json'),
        original_checkpoint=OLD._binding(checkpoint),old_completed_rows=len(reused),
        reason='parent completed rows and completed futures retained large chain archives; stop before OOM',
        stopped_after_rows=10,producer_elapsed_seconds_last_completed=164.837,
        observed_parent_RSS_MiB_before_shutdown=1631.3,
        original_pool_children_stopped_after_SIGINT=True,parent_interpreter_feeder_shutdown_required_SIGTERM=True,
        original_final_observation_seal_exists=False,producer_guide_NMS_TLS_graph_or_guard_changes=0))
    for relative,digest in source.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Exact new source snapshot required')
    C.write(output/'PREPARATION_SEAL.json',dict(schema='BOUNDED_MEMORY_GEOMETRY_EDGE_PREPARATION_SEAL_V29B',rows=238,
        ordered_ids=prep['ordered_ids'],source_sha256=source,protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),reuse_seal_sha256=C.sha(output/'REUSE_SEAL.json'),
        operational_receipt_sha256=C.sha(output/'ORCHESTRATION_CHANGE_RECEIPT.json'),
        producer_math_unchanged=True,no_numeric_reference_decode=True,new_RGB_queries=0))
    print(json.dumps(dict(prepared=238,reused_exact_CAL_rows=len(reused),output=str(output))),flush=True)


def _emit_partition(output,partition,protocol,prep,entries):
    ids=[ident for ident in prep['ordered_ids']if entries.get(ident,{}).get('partition')==partition]if partition=='CAL'else[
        ident for ident in prep['ordered_ids']if entries.get(ident,{}).get('partition')in('TEST','REMAINDER')]
    expected=46 if partition=='CAL' else 192
    if len(ids)!=expected:raise ValueError('Every original partition row including abstentions required')
    ordered=[entries[ident]for ident in ids]
    C.save_rows(output/(partition+'_OBSERVATIONS.jsonl.gz'),(_read_single(_row_path(output,e['id']),e)for e in ordered))
    C.write(output/(partition+'_SUMMARY.json'),_summary(ordered));C.write(output/(partition+'_ROW_INDEX.json'),dict(entries=ordered))
    for key in('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(protocol[key])
    C.write(output/(partition+'_OBSERVATION_SEAL.json'),dict(schema='SAME_FRESH_GEOMETRY_EDGE_SPOOLED_PARTITION_SEAL_V29B',
        rows=expected,ordered_ids=ids,observations_sha256=C.sha(output/(partition+'_OBSERVATIONS.jsonl.gz')),
        summary_sha256=C.sha(output/(partition+'_SUMMARY.json')),row_index_sha256=C.sha(output/(partition+'_ROW_INDEX.json')),
        execution_protocol_sha256=C.sha(output/(partition+'_EXECUTION_PROTOCOL.json')),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=protocol['source_sha256'],no_numeric_reference_decodes=True,no_NN_or_pose_run=True))


def observe(output=OUTPUT,partition='CAL',workers=3):
    output=Path(output).resolve();protocol,prep=_load(output)
    if partition not in('CAL','REST'):raise ValueError('Use CAL46 or REST192')
    if partition=='REST':OLD._verify_observation_seal(output,'CAL')
    if(output/(partition+'_OBSERVATION_SEAL.json')).exists()or(output/(partition+'_EXECUTION_PROTOCOL.json')).exists():raise FileExistsError('Preserve every execution')
    if isinstance(workers,bool)or not isinstance(workers,int)or not 1<=workers<=8:raise ValueError('Use1..8 workers')
    entries=dict(C.read(output/'REUSE_SEAL.json')['entries'])if partition=='CAL'else{}
    expected=46 if partition=='CAL' else 192
    def tasks():
        for task in C.rows(output/'INPUTS.jsonl.gz'):
            if(task['partition']=='CAL')==(partition=='CAL')and task['id']not in entries:yield task
    iterator=iter(tasks());pending={};started=time.monotonic()
    C.write(output/(partition+'_EXECUTION_PROTOCOL.json'),dict(workers=workers,maximum_pending_tasks=workers,
        UTC=datetime.now(timezone.utc).isoformat(),exact_unchanged_worker=OLD._binding(Path(V.__file__)),
        reused_rows=len(entries),same_observation_function='observe_geometry_guided_edges_v4_29.observe_one',
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json')))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=V._init_worker)
    def fill():
        while len(pending)<workers:
            task=next(iterator,None)
            if task is None:return
            pending[pool.submit(V.observe_one,task)]=task['id']
    try:
        fill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ident=pending.pop(future);row=future.result()
                if row['id']!=ident:raise ValueError('Worker identity changed')
                path=_row_path(output,ident);C.save_rows(path,[row]);entries[ident]=_index_row(row,path)
                progress=dict(partition=partition,completed=len(entries),total=expected,id=ident,
                    elapsed_seconds=round(time.monotonic()-started,3),summary=row['summary'],retained_completed_rows_in_RAM=0)
                C.write(output/(partition+'_LIVE_PROGRESS.json'),progress);print(json.dumps(progress),flush=True)
                del row
            done.clear();del future;fill()
    except BaseException as error:
        C.write(output/(partition+'_EXECUTION_FAILURE.json'),dict(error=repr(error),completed_ids=list(entries),no_observation_seal=True))
        OLD._stop_pool(pool);raise
    else:pool.shutdown(wait=True)
    _emit_partition(output,partition,protocol,prep,entries)
    print(json.dumps(dict(sealed=partition,summary=_summary(list(entries.values())))),flush=True)


def finalize(output=OUTPUT):
    output=Path(output).resolve();protocol,prep=_load(output)
    if(output/'OBSERVATION_SEAL.json').exists():raise FileExistsError('Preserve complete observation archive')
    entries={}
    for partition in('CAL','REST'):
        OLD._verify_observation_seal(output,partition)
        for entry in C.read(output/(partition+'_ROW_INDEX.json'))['entries']:
            if entry['id']in entries:raise ValueError('Duplicate original ID')
            entries[entry['id']]=entry
    if set(entries)!=set(prep['ordered_ids'])or len(entries)!=238:raise ValueError('All238 must remain')
    ordered=[entries[ident]for ident in prep['ordered_ids']]
    C.save_rows(output/'OBSERVATIONS.jsonl.gz',(_read_single(_row_path(output,e['id']),e)for e in ordered))
    C.write(output/'SUMMARY.json',_summary(ordered))
    for key in('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(protocol[key])
    C.write(output/'OBSERVATION_SEAL.json',dict(schema='SAME_FRESH_GEOMETRY_EDGE_SPOOLED_ALL238_SEAL_V29B',
        rows=238,ordered_ids=prep['ordered_ids'],observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'),
        summary_sha256=C.sha(output/'SUMMARY.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=protocol['source_sha256'],
        partition_seals={p:C.sha(output/(p+'_OBSERVATION_SEAL.json'))for p in('CAL','REST')},
        all_frames_including_abstentions_preserved=True,producer_math_unchanged=True,no_numeric_reference_decodes=True,no_NN_or_pose_run=True))
    print(json.dumps(dict(sealed=238,summary=_summary(ordered))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','observe','finalize'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--partition',choices=('CAL','REST'),default='CAL')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='observe':observe(args.output,args.partition,args.workers)
    else:finalize(args.output)
