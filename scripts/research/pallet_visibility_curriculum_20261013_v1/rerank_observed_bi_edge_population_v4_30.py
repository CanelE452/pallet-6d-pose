"""NEW V30 uniform cached selection population; original V29 pixels unchanged.

Prepare seals metadata/source/parent spool bytes before any cached-row rerank.
Workers stream one parent row and save one new spool; the parent retains only
small entries. CAL46 completes before any later REST192 generation. No RGB
image is decoded and no measurement, NN, label or pose fit is executed.
"""
from __future__ import annotations
import argparse
import gzip
import json
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing
import time
import numpy as np
from . import run_observed_bi_line_association_CAL_v4_30b as D

C,F,OLD=D.C,D.F,D.OBS
OUTPUT=D.B/'geometry_guided_edge_association_observations_v4_30'
DECLARATION=D.B/'observed_bi_associated_edges_plan_v4_30/PREDECLARATION.json'
ANCESTRY=D.B/'geometry_guided_native_evidence_resume_bound_v4_29c'
RAW_OBSERVATION_GENERATION=F.P.OBSERVATION_GENERATION
SELECTION_PROFILE=F.SELECTION_PROFILE


def save_rows_fast(path,values):
    """Same JSON row values as C.save_rows, gzip1 instead of original gzip5."""
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve all prior completed/partial rows')
    encoder=json.JSONEncoder(ensure_ascii=False,separators=(',',':'),allow_nan=False)
    with gzip.open(path,'wt',compresslevel=1)as stream:
        for row in values:
            for text in encoder.iterencode(C.finite(row)):stream.write(text)
            stream.write('\n')


def rebuild_selected_references(graph,chain_observation,states,hidden):
    """Same producer owner semantics, joining exact original ray/chain indices."""
    selected=[];seen=set()
    for node in graph['line_candidates']:
        if node['name']not in graph['selected_line_names']:continue
        ri,ci=map(int,node['provenance'][0].split(':'));ray=chain_observation['edge_records'][ri]
        k,mate=ray['native_corner'],ray['native_neighbor']
        if hidden[k]or hidden[mate]or states[k]not in('B','I')or states[mate]=='S':
            raise ValueError('Initial S/eligible BI gates before actual chain payload')
        edge=tuple(sorted([k,mate]))
        if tuple(sorted(node['native_edge']))!=edge:raise ValueError('Selected node native edge must match its original ray IDs')
        if edge in seen:raise ValueError('One graph-winner line per physical edge required')
        seen.add(edge);chain=ray['chains'][ci]
        actual=np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'],bool)]
        normal=np.asarray(chain['normal']);norm=np.linalg.norm(normal)
        if(not np.array_equal(actual,node['actual_inliers'])or not np.array_equal(normal/norm,node['normal'])
                or float(chain['offset'])/norm!=node['offset']):raise ValueError('First provenance must match actual TLS/inliers exactly')
        selected.append(dict(ray_index=ri,chain_index=ci,native_corner=k,native_neighbor=mate,
            guide_reference=deepcopy(chain['guide_reference']),guide_reference_id=chain['guide_reference_id'],chain_id=chain['chain_id'],
            line_name=node['name'],native_edge=list(edge),proposal_grade=node['proposal_grade'],
            eligible_BI_owner_ids=[a for a in edge if not hidden[a]and states[a]in('B','I')],
            endpoint_provenance=[dict(native_corner=a,state=states[a],actual_self_anchor=a==k,
                model_only_other_BI_endpoint=a!=k and states[a]in('B','I'),
                image_coordinate_used_as_search_guide=a==k or ray.get('neighbor_image_coordinate_used',False),
                full_2D_point_observation_created=False)for a in edge]))
    if {r['line_name']for r in selected}!=set(graph['selected_line_names']):raise ValueError('Exact selected graph/reference membership')
    return selected


def rerank_cached_row(row,configuration,mask):
    """Same cached algorithm for every CAL/TEST/REMAINDER row, including zeros."""
    if row['partition']not in('CAL','TEST','REMAINDER'):raise ValueError('Declared inventory partition required')
    observation=row['observation'];geometry=row['initial_geometry'];states=row['states']
    if observation['observation_generation']!=RAW_OBSERVATION_GENERATION:raise ValueError('Original actual V29 raw generation required')
    if F.P.VIS.mask_pixel_sha256(mask)!=observation['target_mask_pixel_sha256']:raise ValueError('Original mask pixels required')
    binding=dict(sha256=row['image_sha256'],image_hw=geometry['image_hw']);guides=D.R.restore_pose_guides(row)
    regenerated=F.G.prepare_joint_geometry_guides(geometry,states,row['visible_measurements'],image_binding=binding,
        configuration=configuration,pose_guides=guides)
    if D.R._canonical_sha(regenerated)!=D.R._canonical_sha(observation['guide_bank']):raise ValueError('Exact frozen guide bank before rank')
    context=F.P.VIS.prepare_edge_visibility(mask,geometry,states,row['visible_measurements'],image_binding=binding,
        occluder_mask=None,occluder_binding=None)
    before=D.R._canonical_sha(observation['chain_observation']);started=time.monotonic()
    result=F.rank_observed_BI_associated_graph(observation['chain_observation'],geometry,states,row['visible_measurements'],
        image_binding=binding,configuration=configuration,pose_guides=guides,pallet_mask=mask,visibility_context=context)
    seconds=time.monotonic()-started;graph=result['graph'];cfg,_=F.G._config(configuration)
    D.R.decorate_graph(graph,observation['chain_observation'],cfg)
    refs=rebuild_selected_references(graph,observation['chain_observation'],states,geometry['initial_self_hidden'])
    visibility=[]
    for ref in refs:
        ray=observation['chain_observation']['edge_records'][ref['ray_index']];chain=ray['chains'][ref['chain_index']]
        actual=np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'],bool)]
        visibility.append(context.classify_expected_edge(ref['native_corner'],ref['native_neighbor'],observed_chain_points=actual,
            mask_context_kind=ray.get('mask_context_kind')))
    if D.R._canonical_sha(observation['chain_observation'])!=before:raise ValueError('All raw alternatives/coefficients/indices immutable')
    # New containers; large immutable raw arrays are shared within this one row.
    new_observation={**observation,'joint_graph':graph,'selected_chain_refs':refs,'edge_visibility':visibility,
        'raw_observation_generation':RAW_OBSERVATION_GENERATION,'selection_profile':SELECTION_PROFILE,
        'association_recipe':deepcopy(F.RECIPE),'association_configuration':deepcopy(observation['chain_observation']['configuration']),
        'association_ledger':result['node_association_ledger'],'association_raw_archive_sha256':before,
        'parent_graph_sha256':D.R._canonical_sha(observation['joint_graph']),
        'old_V29_noise_validated_for_this_selection_population':False}
    summary={**row['summary'],'selected_physical_edges':len(refs),
        'selected_partial_edges':sum(r['proposal_grade']==F.P.PARTIAL_GRADE for r in refs),'zero_selected_edges':not refs}
    new_row={**row,'observation':new_observation,'summary':summary,'selection_profile':SELECTION_PROFILE,
        'cached_actual_measurements_unchanged':True,'new_RGB_NMS_TLS_labels_posefit':0}
    return new_row,dict(id=row['id'],partition=row['partition'],rank_seconds=seconds,
        old_unique_candidates=observation['joint_graph']['actual_unique_line_candidates'],new_unique_candidates=graph['actual_unique_line_candidates'],
        checked_raw_nodes=result['candidate_nodes_checked'],rejected_raw_nodes=result['rejected_node_count'],
        old_selected=len(observation['selected_chain_refs']),new_selected=len(refs),zero_selected=not refs)


def _union(target,values):
    for path,digest in values.items():
        if path in target and target[path]!=digest:raise ValueError('Source/input ancestry conflict: '+path)
        target[path]=digest


def bind_central_maps(sources,inputs,protected,central):
    """Central frozen source/input/protection maps constrain downstream bytes."""
    for key,target in(('source_sha256',sources),('input_sha256',inputs),('protected_user_sha256',protected)):
        D.R._verify(central[key]);_union(target,central[key])


def prepare(*,root_metadata_GO=False,expected_central_plan_sha256=None):
    if root_metadata_GO is not True:raise PermissionError('Separate root metadata preparation GO before IO')
    if(not isinstance(expected_central_plan_sha256,str)or len(expected_central_plan_sha256)!=64
            or C.sha(DECLARATION)!=expected_central_plan_sha256):raise ValueError('Exact root central plan required')
    if OUTPUT.exists():raise FileExistsError('Preserve every preparation')
    central=C.read(DECLARATION)
    parent_protocol=C.read(OLD.OUTPUT/'PROTOCOL.json');seal=C.read(OLD.OUTPUT/'CAL_OBSERVATION_SEAL.json')
    index=C.read(OLD.OUTPUT/'CAL_ROW_INDEX.json')['entries'];ids=parent_protocol['ids'][:46]
    if len(ids)!=46 or len(set(ids))!=46 or seal['ordered_ids']!=ids or[e['id']for e in index]!=ids:
        raise ValueError('Exact original ordered CAL46, no dropout')
    if seal['row_index_sha256']!=C.sha(OLD.OUTPUT/'CAL_ROW_INDEX.json'):raise ValueError('Original CAL index binding')
    ancestry=C.read(ANCESTRY/'PREPARATION_SEAL.json');sources=dict(ancestry['source_sha256']);inputs=dict(ancestry['input_sha256'])
    protected=dict(parent_protocol['protected_user_sha256']);bind_central_maps(sources,inputs,protected,central)
    for m in(D,F,F.P,F.G,F.P.ORDER,F.P.VIS,F.G.NMS,F.B,F.M,D.R,OLD,OLD.V,C):
        p=Path(m.__file__).resolve();_union(sources,{str(p.relative_to(C.ROOT)):C.sha(p)})
    for p in(Path(__file__),Path(__file__).with_name('test_rerank_observed_bi_edge_population_v4_30.py')):
        _union(sources,{str(p.resolve().relative_to(C.ROOT)):C.sha(p)})
    for p in(DECLARATION,ANCESTRY/'PREPARATION_SEAL.json',ANCESTRY/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json',
            OLD.OUTPUT/'PROTOCOL.json',OLD.OUTPUT/'PREPARATION_SEAL.json',OLD.OUTPUT/'CAL_OBSERVATION_SEAL.json',OLD.OUTPUT/'CAL_ROW_INDEX.json'):
        _union(inputs,{str(p.relative_to(C.ROOT)):C.sha(p)})
    for e in index:
        p=OLD._row_path(OLD.OUTPUT,e['id'])
        if C.sha(p)!=e['sha256']:raise ValueError('Original CAL spool bytes')
        _union(inputs,{str(p.relative_to(C.ROOT)):e['sha256']})
    for mapping in(sources,inputs,protected):D.R._verify(mapping)
    OUTPUT.mkdir(parents=True);(OUTPUT/'row_spool').mkdir()
    protocol=dict(schema='V30_CACHED_ACTUAL_EDGE_ASSOCIATION_SELECTION_PROTOCOL',ids=parent_protocol['ids'],
        CAL_ids=ids,configuration=parent_protocol['configuration'],selection_profile=SELECTION_PROFILE,
        raw_observation_generation=RAW_OBSERVATION_GENERATION,source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,central_plan_sha256=expected_central_plan_sha256,
        measurement_parent_folder=str(OLD.OUTPUT.relative_to(C.ROOT)),new_images_NN_labels_posefit=0,
        storage=dict(parent_gzip_level=5,new_gzip_level=1,JSON_row_semantics_unchanged=True,C_rows_roundtrip_required=True),
        operational_design=dict(maximum_workers=3,maximum_pending=3,large_parent_rows_retained=0,
            worker_live_data='one shared parent/new raw archive + graph/contexts; transient serializer canonical copy; roundtrip only after source rows released',
            parent_retains='at most46 small index/summary entries',
            memory='rank smoke RSS about0.85GiB; engineering reserve2GiB perworker/6GiB for3workers, not a peak-RAM guarantee; root memory coordination required'),
        old_V29_noise_model_validation=False,CAL_parent_entries=index)
    C.write(OUTPUT/'PROTOCOL.json',protocol)
    C.write(OUTPUT/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json'),ordered_CAL_ids=ids,
        rows=46,raw_generation=RAW_OBSERVATION_GENERATION,selection_profile=SELECTION_PROFILE,
        before_cached_row_or_mask_decode=True,new_RGB_label_or_fit_calls=0))
    for relative,digest in sources.items():
        p=OUTPUT/'code_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(p)!=digest:raise ValueError('Exact source snapshot')
    print(dict(CAL46=46,source_bindings=len(sources),input_bindings=len(inputs),prepared_seal_sha256=C.sha(OUTPUT/'PREPARATION_SEAL.json')),flush=True)


def _worker(case,configuration,output):
    row=OLD._read_single(OLD._row_path(OLD.OUTPUT,case['id']),case)
    path=C.ROOT/row['mask_binding']['path']
    if C.sha(path)!=row['mask_binding']['sha256']:raise ValueError('Exact parent mask bytes')
    with np.load(path,allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
    new,record=rerank_cached_row(row,configuration,mask);new['parent_observation_row']=dict(case)
    destination=OLD._row_path(output,case['id']);save_rows_fast(destination,[new])
    entry=OLD._index_row(new,destination)
    # Release raw arrays/graphs before the independent decompressed roundtrip.
    del row,new,mask
    # Existing reader and canonical digest independently roundtrip this spool.
    roundtrip=OLD._read_single(destination,entry)
    if roundtrip['selection_profile']!=SELECTION_PROFILE:raise ValueError('New profile roundtrip')
    del roundtrip
    return entry,record


def rerank(*,root_CAL_rerank_GO=False,workers=3):
    if root_CAL_rerank_GO is not True:raise PermissionError('Separate root actual CAL46 cached-rerank GO before IO')
    if isinstance(workers,bool)or not isinstance(workers,int)or not 1<=workers<=3:raise ValueError('Bounded1..3 independent workers')
    protocol=C.read(OUTPUT/'PROTOCOL.json');prep=C.read(OUTPUT/'PREPARATION_SEAL.json')
    if C.sha(OUTPUT/'PROTOCOL.json')!=prep['protocol_sha256']:raise ValueError('Frozen new preparation')
    if(OUTPUT/'CAL_EXECUTION_PROTOCOL.json').exists():raise FileExistsError('Preserve every actual attempt')
    for key in('source_sha256','input_sha256','protected_user_sha256'):D.R._verify(protocol[key])
    C.write(OUTPUT/'CAL_EXECUTION_PROTOCOL.json',dict(workers=workers,maximum_pending=workers,
        UTC=datetime.now(timezone.utc).isoformat(),prepared_sha256=C.sha(OUTPUT/'PREPARATION_SEAL.json'),
        cached_actual_rank_only=True,new_RGB_NMS_labels_or_posefit=0))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    tasks=iter(protocol['CAL_parent_entries']);pending={};entries={};records={}
    def fill():
        while len(pending)<workers:
            case=next(tasks,None)
            if case is None:return
            pending[pool.submit(_worker,case,protocol['configuration'],OUTPUT)]=case['id']
    try:
        fill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ident=pending.pop(future);entry,record=future.result()
                if entry['id']!=ident:raise ValueError('Exact pending identity')
                entries[ident]=entry;records[ident]=record
                C.write(OUTPUT/'CAL_LIVE_PROGRESS.json',dict(completed=len(entries),total=46,id=ident))
                print(dict(completed=len(entries),total=46,id=ident,selected=record['new_selected']),flush=True)
            fill()
    except BaseException as error:
        C.write(OUTPUT/'CAL_FAILURE_RECEIPT.json',dict(exception=type(error).__name__,reason=str(error),completed_ids=list(entries),pending_ids=list(pending)))
        for future in pending:future.cancel()
        for process in list((getattr(pool,'_processes',{})or{}).values()):
            if process.is_alive():process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    ids=protocol['CAL_ids']
    if set(entries)!=set(ids):raise ValueError('All46 including zero-selection rows required')
    ordered=[entries[i]for i in ids];C.write(OUTPUT/'CAL_ROW_INDEX.json',dict(entries=ordered))
    save_rows_fast(OUTPUT/'CAL_OBSERVATIONS.jsonl.gz',(OLD._read_single(OLD._row_path(OUTPUT,i),entries[i])for i in ids))
    if [r['id']for r in C.rows(OUTPUT/'CAL_OBSERVATIONS.jsonl.gz')]!=ids:raise ValueError('C.rows bulk journal ordered roundtrip')
    C.write(OUTPUT/'CAL_SUMMARY.json',dict(rows=46,records=[records[i]for i in ids],zero_selected=sum(records[i]['zero_selected']for i in ids),
        all_rows_and_abstentions_retained=True,selection_profile=SELECTION_PROFILE,new_RGB_NMS_TLS_labels_posefit=0))
    for key in('source_sha256','input_sha256','protected_user_sha256'):D.R._verify(protocol[key])
    C.write(OUTPUT/'CAL_OBSERVATION_SEAL.json',dict(rows=46,ordered_ids=ids,observations_sha256=C.sha(OUTPUT/'CAL_OBSERVATIONS.jsonl.gz'),
        row_index_sha256=C.sha(OUTPUT/'CAL_ROW_INDEX.json'),summary_sha256=C.sha(OUTPUT/'CAL_SUMMARY.json'),
        protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json'),preparation_seal_sha256=C.sha(OUTPUT/'PREPARATION_SEAL.json'),
        execution_protocol_sha256=C.sha(OUTPUT/'CAL_EXECUTION_PROTOCOL.json'),source_sha256=protocol['source_sha256'],
        selection_profile=SELECTION_PROFILE,raw_observation_generation=RAW_OBSERVATION_GENERATION,new_RGB_NMS_labels_posefit=0))
    print(dict(CAL46=46,seal_sha256=C.sha(OUTPUT/'CAL_OBSERVATION_SEAL.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','rerank'))
    p.add_argument('--root-metadata-GO',action='store_true');p.add_argument('--root-CAL-rerank-GO',action='store_true')
    p.add_argument('--central-plan-sha256');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    prepare(root_metadata_GO=a.root_metadata_GO,expected_central_plan_sha256=a.central_plan_sha256)if a.stage=='prepare'else rerank(root_CAL_rerank_GO=a.root_CAL_rerank_GO,workers=a.workers)
