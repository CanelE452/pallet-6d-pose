"""Append cached REST192 to frozen V30 CAL46; no original file is edited.

The exact CAL worker is reused. Only new row spools and REST/final metadata
are written. FINAL_PROTOCOL is the all238 ancestry union; PROTOCOL remains
the immutable CAL preparation. Actual execution needs a separate root GO.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path

from . import rerank_observed_bi_edge_population_v4_30 as R
from . import bind_operational_geometry_guided_eval_v4_29d as OP

C, OLD = R.C, R.OLD
OUTPUT = R.OUTPUT
PLAN = R.D.B / 'observed_bi_association_REST_plan_v4_30b/PREDECLARATION.json'
CENTRAL_SHA256 = '1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'
PROFILE = R.SELECTION_PROFILE
RAW = R.RAW_OBSERVATION_GENERATION
REST_PROTOCOL = 'REST_PROTOCOL_V30B.json'
REST_PREP = 'REST_PREPARATION_SEAL_V30B.json'
REST_EXEC = 'REST_EXECUTION_PROTOCOL_V30B.json'


def _union(*mappings):
    result = {}
    for mapping in mappings:
        R._union(result, mapping)
    return result


def _binding(path):
    path = Path(path)
    return dict(path=str(path.relative_to(C.ROOT)), sha256=C.sha(path))


def _verify(maps):
    for mapping in maps:
        R.D.R._verify(mapping)


def declare():
    if PLAN.exists():
        raise FileExistsError('Preserve all REST declarations')
    files = [Path(__file__), Path(R.__file__), Path(OP.__file__),
             Path(__file__).with_name('test_rerank_observed_bi_edge_REST_v4_30b.py')]
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN, dict(schema='V30_CACHED_REST192_APPEND_PREDECLARATION_V30B',
        UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.resolve().relative_to(C.ROOT)):C.sha(p) for p in files},
        central_plan_sha256=CENTRAL_SHA256, CAL_rows=46, REST_rows=192,
        populations=dict(TEST=49,REMAINDER=143), all_rows_and_zero_edge_cases_retained=True,
        exact_worker='rerank_observed_bi_edge_population_v4_30._worker',
        maximum_workers=3,maximum_pending=3,parent_completed_payloads_retained=0,
        original_CAL_protocol_seals_spools_and_sources_immutable=True,
        raw_generation=RAW,selection_profile=PROFILE,new_RGB_NMS_TLS_NN_labels_posefit=0,
        final_protocol='FINAL_PROTOCOL.json; original PROTOCOL.json remains CAL-only ancestry',
        authoritative_selection='observation.selected_chain_refs, new joint_graph and row.summary',
        inherited_selected_physical_edges='parent V29 legacy count; not the V30 selected count',
        raw_partial_line_ledger='all raw partial alternatives, preserved unchanged; not a selected-only ledger'))


def _cal_files(output):
    names = ['PROTOCOL.json','PREPARATION_SEAL.json','CAL_EXECUTION_PROTOCOL.json',
             'CAL_ROW_INDEX.json','CAL_OBSERVATIONS.jsonl.gz','CAL_SUMMARY.json',
             'CAL_OBSERVATION_SEAL.json']
    paths = [output/name for name in names]
    index = C.read(output/'CAL_ROW_INDEX.json')['entries']
    for entry in index:
        path = OLD._row_path(output,entry['id'])
        if C.sha(path)!=entry['sha256']:
            raise ValueError('Frozen CAL spool changed')
        paths.append(path)
    return {str(p.relative_to(C.ROOT)):C.sha(p) for p in paths}


def prepare(*,root_metadata_GO=False,output=OUTPUT):
    if root_metadata_GO is not True:
        raise PermissionError('Root REST metadata preparation GO before IO')
    output = Path(output)
    for name in (REST_PROTOCOL,REST_PREP,REST_EXEC,'FINAL_PROTOCOL.json','OBSERVATION_SEAL.json'):
        if (output/name).exists():
            raise FileExistsError('Preserve every REST/final preparation attempt')
    plan = C.read(PLAN)
    if plan['central_plan_sha256']!=CENTRAL_SHA256 or C.sha(R.DECLARATION)!=CENTRAL_SHA256:
        raise ValueError('Frozen central and REST declarations required')
    _verify([plan['source_sha256']])
    central=C.read(R.DECLARATION);cal=C.read(output/'PROTOCOL.json')
    cseal=C.read(output/'CAL_OBSERVATION_SEAL.json');cp=C.read(output/'PREPARATION_SEAL.json')
    cindex=C.read(output/'CAL_ROW_INDEX.json')['entries']
    ids=cal['ids'];cal_ids=ids[:46];rest_ids=ids[46:]
    if (len(ids)!=238 or len(set(ids))!=238 or len(rest_ids)!=192
            or cal['selection_profile']!=PROFILE or cal['raw_observation_generation']!=RAW
            or cseal['rows']!=46 or cseal['ordered_ids']!=cal_ids
            or [e['id'] for e in cindex]!=cal_ids or cseal['selection_profile']!=PROFILE
            or C.sha(output/'PROTOCOL.json')!=cseal['protocol_sha256']
            or C.sha(output/'PROTOCOL.json')!=cp['protocol_sha256']
            or C.sha(output/'PREPARATION_SEAL.json')!=cseal['preparation_seal_sha256']
            or C.sha(output/'CAL_ROW_INDEX.json')!=cseal['row_index_sha256']
            or C.sha(output/'CAL_OBSERVATIONS.jsonl.gz')!=cseal['observations_sha256']):
        raise ValueError('Complete unchanged V30 CAL46 required before REST preparation')
    closure=OP.verify_operational_closure()
    if closure['ordered_ids']!=ids:
        raise ValueError('Same original ordered all238 source population required')
    parent_protocol=C.read(OLD.OUTPUT/'PROTOCOL.json')
    parent_seal=C.read(OLD.OUTPUT/'REST_OBSERVATION_SEAL.json')
    parent_index=C.read(OLD.OUTPUT/'REST_ROW_INDEX.json')['entries']
    if (parent_seal['rows']!=192 or parent_seal['ordered_ids']!=rest_ids
            or [e['id'] for e in parent_index]!=rest_ids
            or Counter(e['partition'] for e in parent_index)!=Counter(TEST=49,REMAINDER=143)
            or C.sha(OLD.OUTPUT/'REST_ROW_INDEX.json')!=parent_seal['row_index_sha256']
            or parent_protocol['configuration']!=cal['configuration']):
        raise ValueError('Exact original ordered REST192 index/configuration required')
    immutable=_cal_files(output)
    sources=_union(cal['source_sha256'],closure['source_sha256'],plan['source_sha256'])
    inputs=_union(cal['input_sha256'],closure['input_sha256'],immutable,
        {str(p.relative_to(C.ROOT)):C.sha(p) for p in (PLAN,R.DECLARATION,
            OLD.OUTPUT/'REST_ROW_INDEX.json',OLD.OUTPUT/'REST_OBSERVATION_SEAL.json')})
    protected=_union(cal['protected_user_sha256'],closure['protected_user_sha256'])
    R.bind_central_maps(sources,inputs,protected,central)
    for entry in parent_index:
        path=OLD._row_path(OLD.OUTPUT,entry['id'])
        if C.sha(path)!=entry['sha256']:
            raise ValueError('Original REST spool changed')
        inputs[str(path.relative_to(C.ROOT))]=entry['sha256']
        if OLD._row_path(output,entry['id']).exists():
            raise FileExistsError('Never overwrite an existing REST row')
    _verify([sources,inputs,protected,immutable])
    protocol=dict(schema='V30_CACHED_REST192_APPEND_PROTOCOL_V30B',ids=ids,REST_ids=rest_ids,
        CAL_ids=cal_ids,selection_profile=PROFILE,raw_observation_generation=RAW,
        configuration=cal['configuration'],source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,original_CAL_file_sha256=immutable,
        parent_REST_entries=parent_index,operational_closure=closure,
        plan=_binding(PLAN),same_frozen_worker=True,workers_limit=3,pending_limit=3,
        inherited_selected_physical_edges_is_parent_legacy_count=True,
        selected_chain_refs_graph_and_row_summary_are_authoritative=True,
        raw_partial_alternative_ledger_preserved=True,new_RGB_NMS_TLS_NN_labels_posefit=0)
    C.write(output/REST_PROTOCOL,protocol)
    C.write(output/REST_PREP,dict(rows=192,ordered_ids=rest_ids,
        protocol_sha256=C.sha(output/REST_PROTOCOL),CAL_observation_seal=_binding(output/'CAL_OBSERVATION_SEAL.json'),
        original_CAL_file_sha256=immutable,before_any_REST_row_mask_decode_or_rank=True))
    _verify([sources,inputs,protected,immutable])
    return _binding(output/REST_PREP)


def _load(output):
    protocol=C.read(output/REST_PROTOCOL);prep=C.read(output/REST_PREP)
    if C.sha(output/REST_PROTOCOL)!=prep['protocol_sha256'] or prep['ordered_ids']!=protocol['REST_ids']:
        raise ValueError('Sealed REST metadata changed')
    _verify([protocol[k] for k in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256')])
    return protocol


def rerank(*,root_REST_rerank_GO=False,workers=3,output=OUTPUT):
    if root_REST_rerank_GO is not True:
        raise PermissionError('Root actual cached REST192 rerank GO before IO')
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=3:
        raise ValueError('Use bounded1..3 workers')
    output=Path(output);protocol=_load(output)
    if (output/REST_EXEC).exists():
        raise FileExistsError('Preserve every actual REST attempt')
    C.write(output/REST_EXEC,dict(workers=workers,maximum_pending=workers,
        UTC=datetime.now(timezone.utc).isoformat(),preparation_seal=_binding(output/REST_PREP),
        worker_source_sha256=C.sha(R.__file__),same_worker='R30._worker',new_RGB_NMS_TLS_NN_labels_posefit=0))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    tasks=iter(protocol['parent_REST_entries']);pending={};entries={};records={}
    def fill():
        while len(pending)<workers:
            case=next(tasks,None)
            if case is None:return
            if OLD._row_path(output,case['id']).exists():
                raise FileExistsError('Never overwrite a REST spool')
            pending[pool.submit(R._worker,case,protocol['configuration'],output)]=case['id']
    try:
        fill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ident=pending.pop(future);entry,record=future.result()
                if entry['id']!=ident or ident in entries:
                    raise ValueError('Exact unique pending identity required')
                entries[ident]=entry;records[ident]=record
                C.write(output/'REST_LIVE_PROGRESS_V30B.json',dict(completed=len(entries),total=192,id=ident))
                print(dict(completed=len(entries),total=192,id=ident),flush=True)
            fill()
    except BaseException as error:
        C.write(output/'REST_FAILURE_RECEIPT_V30B.json',dict(exception=type(error).__name__,reason=str(error),
            completed_ids=list(entries),pending_ids=list(pending)))
        for future in pending:future.cancel()
        for process in list((getattr(pool,'_processes',{}) or {}).values()):
            if process.is_alive():process.terminate()
        pool.shutdown(wait=False,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    ids=protocol['REST_ids']
    if set(entries)!=set(ids):
        raise ValueError('Every TEST49/REMAINDER143 row including zero-selected cases required')
    ordered=[entries[i] for i in ids]
    C.write(output/'REST_ROW_INDEX.json',dict(entries=ordered))
    R.save_rows_fast(output/'REST_OBSERVATIONS.jsonl.gz',
        (OLD._read_single(OLD._row_path(output,i),entries[i]) for i in ids))
    if [r['id'] for r in C.rows(output/'REST_OBSERVATIONS.jsonl.gz')]!=ids:
        raise ValueError('Exact ordered REST journal roundtrip required')
    C.write(output/'REST_SUMMARY.json',dict(rows=192,records=[records[i] for i in ids],
        zero_selected=sum(records[i]['zero_selected'] for i in ids),all_rows_kept=True,
        selection_profile=PROFILE,new_RGB_NMS_TLS_NN_labels_posefit=0))
    _load(output)
    C.write(output/'REST_OBSERVATION_SEAL.json',dict(rows=192,ordered_ids=ids,
        observations_sha256=C.sha(output/'REST_OBSERVATIONS.jsonl.gz'),
        row_index_sha256=C.sha(output/'REST_ROW_INDEX.json'),summary_sha256=C.sha(output/'REST_SUMMARY.json'),
        protocol_sha256=C.sha(output/REST_PROTOCOL),preparation_seal_sha256=C.sha(output/REST_PREP),
        execution_protocol_sha256=C.sha(output/REST_EXEC),selection_profile=PROFILE,
        raw_observation_generation=RAW,source_sha256=protocol['source_sha256']))
    return _binding(output/'REST_OBSERVATION_SEAL.json')


def finalize(*,root_finalize_GO=False,output=OUTPUT):
    if root_finalize_GO is not True:
        raise PermissionError('Root all238 finalize GO before IO')
    output=Path(output);protocol=_load(output)
    for name in ('FINAL_PROTOCOL.json','OBSERVATIONS.jsonl.gz','OBSERVATION_SEAL.json'):
        if (output/name).exists():raise FileExistsError('Preserve every final generation')
    cal=C.read(output/'PROTOCOL.json');restseal=C.read(output/'REST_OBSERVATION_SEAL.json')
    calseal=C.read(output/'CAL_OBSERVATION_SEAL.json')
    if (restseal['rows']!=192 or restseal['ordered_ids']!=protocol['REST_ids']
            or calseal['ordered_ids']!=protocol['CAL_ids']
            or restseal['protocol_sha256']!=C.sha(output/REST_PROTOCOL)
            or restseal['observations_sha256']!=C.sha(output/'REST_OBSERVATIONS.jsonl.gz')
            or restseal['row_index_sha256']!=C.sha(output/'REST_ROW_INDEX.json')):
        raise ValueError('Both complete ordered CAL/REST partitions required')
    entries=C.read(output/'CAL_ROW_INDEX.json')['entries']+C.read(output/'REST_ROW_INDEX.json')['entries']
    ids=protocol['ids']
    if len(entries)!=238 or [e['id'] for e in entries]!=ids:
        raise ValueError('Exact original ordered238 index required')
    sources=_union(cal['source_sha256'],protocol['source_sha256'])
    inputs=_union(cal['input_sha256'],protocol['input_sha256'],
        {str((output/n).relative_to(C.ROOT)):C.sha(output/n) for n in
         (REST_PROTOCOL,REST_PREP,REST_EXEC,'REST_ROW_INDEX.json','REST_OBSERVATION_SEAL.json',
          'CAL_OBSERVATION_SEAL.json')})
    protected=_union(cal['protected_user_sha256'],protocol['protected_user_sha256'])
    final={**cal,'schema':'V30_ALL238_FINAL_ASSOCIATION_PROTOCOL_V30B','ids':ids,
        'selection_profile':PROFILE,'raw_observation_generation':RAW,
        'source_sha256':sources,'input_sha256':inputs,'protected_user_sha256':protected,
        'CAL_protocol_immutable':_binding(output/'PROTOCOL.json'),
        'REST_protocol':_binding(output/REST_PROTOCOL),
        'inherited_selected_physical_edges_is_parent_legacy_count':True,
        'selected_chain_refs_graph_and_row_summary_are_authoritative':True,
        'raw_partial_alternative_ledger_preserved':True}
    _verify([sources,inputs,protected,protocol['original_CAL_file_sha256']])
    C.write(output/'FINAL_PROTOCOL.json',final)
    R.save_rows_fast(output/'OBSERVATIONS.jsonl.gz',
        (OLD._read_single(OLD._row_path(output,e['id']),e) for e in entries))
    if [r['id'] for r in C.rows(output/'OBSERVATIONS.jsonl.gz')]!=ids:
        raise ValueError('Ordered all238 journal roundtrip required')
    _verify([sources,inputs,protected,protocol['original_CAL_file_sha256']])
    C.write(output/'OBSERVATION_SEAL.json',dict(rows=238,ordered_ids=ids,
        observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'FINAL_PROTOCOL.json'),
        partition_seals={p:C.sha(output/(p+'_OBSERVATION_SEAL.json')) for p in ('CAL','REST')},
        selection_profile=PROFILE,raw_observation_generation=RAW,
        source_sha256=sources,all_rows_and_zero_support_cases_retained=True,
        new_RGB_NMS_TLS_NN_labels_posefit=0))
    C.write(output/'FINAL_COMPLETION_RECEIPT_V30B.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,
        final_protocol=_binding(output/'FINAL_PROTOCOL.json'),observation_seal=_binding(output/'OBSERVATION_SEAL.json'),
        original_CAL_file_sha256=protocol['original_CAL_file_sha256'],
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        new_RGB_NMS_TLS_NN_labels_posefit=0))
    return _binding(output/'OBSERVATION_SEAL.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('declare','prepare','rerank','finalize'))
    p.add_argument('--root-metadata-GO',action='store_true');p.add_argument('--root-REST-rerank-GO',action='store_true')
    p.add_argument('--root-finalize-GO',action='store_true');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.stage=='declare':declare()
    elif a.stage=='prepare':print(prepare(root_metadata_GO=a.root_metadata_GO))
    elif a.stage=='rerank':print(rerank(root_REST_rerank_GO=a.root_REST_rerank_GO,workers=a.workers))
    else:print(finalize(root_finalize_GO=a.root_finalize_GO))
