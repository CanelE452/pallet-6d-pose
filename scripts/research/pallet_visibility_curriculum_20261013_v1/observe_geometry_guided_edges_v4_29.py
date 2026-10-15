"""Freeze238 inputs, then fresh CAL46/REST192 image-line observations.

Preparation uses only sealed prediction inputs and existing CAL model bytes.
Actual images are touched only by the explicitly invoked observe stage. The
same producer and original native start policy apply to every partition.
No labels, reference poses, new NN forward, pose fit or scored pose routing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
import gzip
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from . import image_edge_observations_v4_29 as P
from . import observe_grazing_native_edge_chains_v4_18b as OBS
from . import evaluate_coarse_prior_containment_v4_25 as TASKS
from . import refine_grazing_image_chain_mixtures_v4_18c as START_SOURCE

OLD=OBS.OLD
OUTPUT=OLD.BOUNDARY/'geometry_guided_edge_observations_v4_29'
PLAN=OLD.BOUNDARY/'geometry_guided_edges_plan_v4_29/PREDECLARATION.json'
ROW_SCHEMA='UNIFORM_FRESH_GEOMETRY_GUIDED_EDGE_OBSERVATION_ROW_V4_29'
_MODEL=None


def _join(task,observed):
    for key in ('id','partition','session','image','image_sha256'):
        if task[key]!=observed[key]:raise ValueError('Exact sealed frame identity required: '+key)
    spec=task['spec'];geometry=observed['initial_geometry'];family=task['families']['ORIGINAL_WD']
    if (task['states']!=spec['states']or task['states']!=observed['states']
            or spec['visible']!=observed['visible_measurements']or task['u_regions']!=observed['u_regions']
            or geometry['initial_self_hidden']!=task['initial_self_hidden']):
        raise ValueError('Sealed state/BI/U/initial-S observations must be unchanged')
    for record in spec['visible']:
        k=P.G.NMS.V3._id(record)
        if task['initial_self_hidden'][k]or task['states'][k]not in('B','I'):
            raise ValueError('Only gated B/I evidence is permitted')
    center=np.asarray(family['coarse_pose']['R'])@np.asarray(family['native_xyz']).mean(0)+np.asarray(family['coarse_pose']['t'])
    for actual,expected in ((geometry['native_xyz'],family['native_xyz']),
            (geometry['physical_registration_Q'],family['physical_registration_Q']),
            (geometry['R_cf'],family['coarse_pose']['R']),(geometry['centroid'],center),
            (geometry['K'],spec['K'])):
        if not np.allclose(actual,expected,atol=1e-12,rtol=0):raise ValueError('Original registered geometry differs')
    if geometry['image_hw']!=spec['image_hw']:raise ValueError('Original image canvas differs')
    old_domains={r['native_corner']:r['original_domain']for r in task['O_localization_abstention']['records']}
    if [old_domains.get(d['native_corner'],d)for d in spec['domains']]!=observed['domains']:
        raise ValueError('Only predeclared O ellipse-to-canvas adaptation may differ')
    if task['new_chain_observation']!=observed['chain_observation']:
        raise ValueError('Unregularized start problem requires the exact original fresh17c archive')
    guide_task={key:task[key]for key in('id','states','families','proposals','spec','boundary_merged',
        'u_regions','initial_self_hidden','image_sha256','new_chain_observation')}
    result=dict(id=task['id'],partition=task['partition'],session=task['session'],image=task['image'],
        image_sha256=task['image_sha256'],mask_binding=observed['mask_binding'],
        initial_geometry=geometry,states=task['states'],visible_measurements=spec['visible'],
        domains=spec['domains'],original_observer_domains=observed['domains'],u_regions=task['u_regions'],
        nonself_prediction_scale_px=observed['nonself_prediction_scale_px'],guide_task=guide_task,
        source_task_sha256=OLD._digest(task),source_observation_sha256=OLD._digest(observed))
    result['unchanged_observation_math_sha256']=OLD._digest(_math(result))
    return result


def _math(row):
    return {key:row[key]for key in('id','partition','session','image_sha256','mask_binding',
        'initial_geometry','states','visible_measurements','domains','original_observer_domains',
        'u_regions','nonself_prediction_scale_px','guide_task')}


def _seal(folder,sealname,journal,hashkey):
    protocol=C.read(folder/'PROTOCOL.json');seal=C.read(folder/sealname)
    if (seal['rows']!=238 or seal[hashkey]!=C.sha(folder/journal)
            or seal['protocol_sha256']!=C.sha(folder/'PROTOCOL.json')):
        raise ValueError('Complete parent input/observation seal required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(protocol[key])
    return protocol,seal


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all completed or partial generations')
    if C.sha(PLAN)!=P.CENTRAL_PREDECLARATION_SHA256:raise ValueError('Exact pre-image central plan required')
    declaration=C.read(PLAN)
    task_protocol,task_seal=_seal(TASKS.OUTPUT,'INPUT_SEAL.json','INPUTS.jsonl.gz','input_sha256')
    obs_protocol,obs_seal=_seal(OBS.OUTPUT,'OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz','observations_sha256')
    if task_seal['ordered_ids']!=obs_seal['ordered_ids']:raise ValueError('Exact ordered238 join required')
    if task_protocol['protected_user_sha256']!=obs_protocol['protected_user_sha256']:raise ValueError('Protected inventories differ')
    # Same old unregularized CAL model as v19, solely to rank original starts.
    START_SOURCE._verified_model()
    sources=OBS._merge_maps(task_protocol['source_sha256'],obs_protocol['source_sha256'])
    for module in (P,P.G,P.ORDER,P.VIS,P.STARTS,OBS,OLD,TASKS,START_SOURCE):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(Path(__file__))
    inputs=OBS._merge_maps(task_protocol['input_sha256'],obs_protocol['input_sha256'])
    for folder,names in ((TASKS.OUTPUT,('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz')),
            (OBS.OUTPUT,('PROTOCOL.json','OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz')),
            (START_SOURCE.CAL.OUTPUT,('MODEL.json','MODEL_SEAL.json','PROTOCOL.json','PREPARATION_SEAL.json'))):
        for name in names:
            path=folder/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    inputs[str(PLAN.relative_to(C.ROOT))]=C.sha(PLAN)
    protected=task_protocol['protected_user_sha256'];output.mkdir(parents=True)
    protocol=dict(schema='UNIFORM_FRESH_GEOMETRY_GUIDED_EDGE_OBSERVATION_PROTOCOL_V29',
        ids=task_seal['ordered_ids'],ordered_ids=task_seal['ordered_ids'],population=dict(CAL=46,TEST=49,REMAINDER=143,total=238,severe=0),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        prepared_before_new_RGB_observations_UTC=datetime.now(timezone.utc).isoformat(),
        central_declaration=OLD._binding(PLAN),central_declaration_sha256=P.CENTRAL_PREDECLARATION_SHA256,
        arms=declaration['arms'],configuration=deepcopy(P.DEFAULT_CONFIGURATION),recipe=deepcopy(P.RECIPE),
        observation_generation=P.OBSERVATION_GENERATION,producer_schema=P.SCHEMA,
        guide_selection='v19 unregularized choose_native_starts; six diverse + coarse per family; ORIGINAL fixed XYZ/Q rays',
        old_numeric_e_pose_or_GT_error_guides=False,actual_image_queries_during_preparation=0,
        CAL_then_REST_same_worker=True,all_alternatives_and_abstentions_preserved=True,
        selected_graph_winner_per_physical_edge=True,new_noise_and_tangent_models_for_each_source_policy_required=True,
        independent_occluder_mask_available=False,targetmask_absence_is_not_external_occlusion=True,
        S_U_local_coordinate_or_corner_queries=0,new_NN_posefit_reference_decodes=0,default_pipeline_replaced=False)
    C.write(output/'PROTOCOL.json',protocol)
    counts=Counter();ids=[]
    def rows():
        for task,observed in zip_longest(C.rows(TASKS.OUTPUT/'INPUTS.jsonl.gz'),C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')):
            if task is None or observed is None:raise ValueError('Parent row count differs')
            row=_join(task,observed);counts[row['partition']]+=1;ids.append(row['id']);yield row
    C.save_rows(output/'INPUTS.jsonl.gz',rows())
    if ids!=task_seal['ordered_ids']or counts!=Counter(CAL=46,TEST=49,REMAINDER=143):raise ValueError('No population dropout')
    for mapping in (sources,inputs,protected):OLD._verify_map(mapping)
    for relative,digest in sources.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'PREPARATION_SEAL.json',dict(schema='FRESH_GEOMETRY_EDGE_PREPARATION_SEAL_V29',rows=238,
        ordered_ids=ids,source_sha256=sources,protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),central_predeclaration_sha256=C.sha(PLAN),
        prepared_before_new_RGB_observation=True,no_reference_numeric_decode=True))
    print(json.dumps(dict(prepared=238,partition_counts=counts,output=str(output))),flush=True)


def _init_worker():
    global _MODEL
    cv2.setNumThreads(1);_MODEL=START_SOURCE._verified_model()


def observe_one(task):
    started=time.monotonic();before=OLD._digest(_math(task))
    if before!=task['unchanged_observation_math_sha256']:raise ValueError('Prepared observer quantities changed')
    model=_MODEL if _MODEL is not None else START_SOURCE._verified_model()
    guide_task=task['guide_task']
    problems={name:START_SOURCE._problem(guide_task,family,model)[0]for name,family in guide_task['families'].items()}
    guides,selection=P.prepare_task_pose_guides(guide_task,problems)
    image_path=C.ROOT/task['image'];mask_path=C.ROOT/task['mask_binding']['path']
    if C.sha(image_path)!=task['image_sha256']or C.sha(mask_path)!=task['mask_binding']['sha256']:
        raise ValueError('Frozen actual RGB/mask bytes changed')
    image=cv2.imread(str(image_path))
    if image is None:raise ValueError('Actual image decode failed')
    with np.load(mask_path,allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
    observation=P.observe_joint_geometry_edge_evidence(image,mask,task['initial_geometry'],task['states'],
        task['visible_measurements'],image_binding=dict(sha256=task['image_sha256'],image_hw=list(image.shape[:2])),
        pose_guides=guides,configuration=P.DEFAULT_CONFIGURATION,occluder_mask=None,occluder_binding=None)
    if OLD._digest(_math(task))!=before:raise ValueError('Fresh observation changed original inputs')
    rays=observation['chain_observation']['edge_records'];refs=observation['selected_chain_refs']
    summary=dict(actual_chain_hypotheses=sum(len(r.get('chains',[]))for r in rays),selected_physical_edges=len(refs),
        selected_partial_edges=sum(r['proposal_grade']==P.PARTIAL_GRADE for r in refs),
        fresh_partial_alternatives=len(observation['partial_line_ledger']),zero_selected_edges=not bool(refs),
        guide_count=len(guides),new_NN_posefit_reference_decodes=0)
    result={key:task[key]for key in('id','partition','session','image','image_sha256','mask_binding',
        'initial_geometry','states','visible_measurements','domains','original_observer_domains','u_regions',
        'nonself_prediction_scale_px','unchanged_observation_math_sha256','source_task_sha256','source_observation_sha256')}
    result.update(schema=ROW_SCHEMA,observation_generation=P.OBSERVATION_GENERATION,observation=observation,
        guide_selection_audit=selection,summary=summary,elapsed_seconds=time.monotonic()-started,
        contract=dict(S_U_local_coordinate_or_corner_queries=0,new_NN_posefit_reference_decodes=0,
            old_q_noise_weights_states_regions_unchanged=True,all_alternatives_preserved=True,
            graph_selected_one_per_physical_edge=True,precise_corner_adoptions=0,
            fresh_generation_CAL_noise_required=True,independent_occluder_available=False))
    return result


def _summary(rows):
    return dict(rows=len(rows),partition_counts=dict(Counter(r['partition']for r in rows)),
        selected_physical_edges=sum(r['summary']['selected_physical_edges']for r in rows),
        selected_partial_edges=sum(r['summary']['selected_partial_edges']for r in rows),
        actual_chain_hypotheses=sum(r['summary']['actual_chain_hypotheses']for r in rows),
        zero_selected_edge_frames=sum(r['summary']['zero_selected_edges']for r in rows),
        all_frames_and_abstentions_retained=True,S_U_local_queries=0,new_NN_posefit_reference_decodes=0)


def observe(output=OUTPUT,partition='CAL',workers=3):
    output=Path(output).resolve();tasks,protocol,prep=OLD._load(output)
    if partition not in('CAL','REST'):raise ValueError('Use CAL46 or later REST192')
    if partition=='REST':OLD._verify_observation_seal(output,'CAL')
    chosen=[t for t in tasks if(t['partition']=='CAL')==(partition=='CAL')];expected=46 if partition=='CAL' else 192
    if len(chosen)!=expected:raise ValueError('Complete uniform partition required')
    if isinstance(workers,bool)or not isinstance(workers,int)or not 1<=workers<=8:raise ValueError('Use1..8 workers')
    checkpoint=output/(partition+'_OBSERVATIONS_checkpoint.jsonl.gz');final=output/(partition+'_OBSERVATIONS.jsonl.gz')
    if final.exists()or checkpoint.exists()or(output/(partition+'_OBSERVATION_SEAL.json')).exists():raise FileExistsError('Preserve partial runs')
    C.write(output/(partition+'_EXECUTION_PROTOCOL.json'),dict(workers=workers,OpenCV_threads_per_worker=1,
        UTC=datetime.now(timezone.utc).isoformat(),ordered_ids=[r['id']for r in chosen],
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),same_observation_function='observe_one'))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_init_worker)
    futures={pool.submit(observe_one,t):t['id']for t in chosen};completed={};started=time.monotonic()
    try:
        with gzip.open(checkpoint,'wt',compresslevel=5)as durable:
            for future in as_completed(futures):
                identifier=futures[future]
                try:row=future.result()
                except BaseException as error:
                    C.write(output/(partition+'_EXECUTION_FAILURE.json'),dict(id=identifier,error=repr(error),
                        completed_ids=list(completed),no_observation_seal=True));raise
                if row['id']!=identifier:raise ValueError('Worker frame identity differs')
                completed[identifier]=row;durable.write(json.dumps(C.finite(row),separators=(',',':'),allow_nan=False)+'\n');durable.flush()
                progress=dict(partition=partition,completed=len(completed),total=expected,id=identifier,
                    elapsed_seconds=round(time.monotonic()-started,3),summary=row['summary'])
                C.write(output/(partition+'_LIVE_PROGRESS.json'),progress);print(json.dumps(progress),flush=True)
    except BaseException:OLD._stop_pool(pool);raise
    else:pool.shutdown(wait=True)
    ordered=[completed[t['id']]for t in chosen]
    for key in('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(protocol[key])
    C.save_rows(final,ordered);C.write(output/(partition+'_SUMMARY.json'),_summary(ordered))
    C.write(output/(partition+'_OBSERVATION_SEAL.json'),dict(schema='FRESH_GEOMETRY_EDGE_PARTITION_SEAL_V29',
        rows=expected,ordered_ids=[t['id']for t in chosen],observations_sha256=C.sha(final),
        summary_sha256=C.sha(output/(partition+'_SUMMARY.json')),execution_protocol_sha256=C.sha(output/(partition+'_EXECUTION_PROTOCOL.json')),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=protocol['source_sha256'],no_numeric_reference_decodes=True,no_NN_or_pose_run=True))
    print(json.dumps(dict(sealed=partition,summary=_summary(ordered))),flush=True)


def finalize(output=OUTPUT):
    output=Path(output).resolve();tasks,protocol,prep=OLD._load(output)
    if(output/'OBSERVATION_SEAL.json').exists():raise FileExistsError('Preserve complete238 seal')
    for partition in('CAL','REST'):OLD._verify_observation_seal(output,partition)
    rows={r['id']:r for partition in('CAL','REST')for r in C.rows(output/(partition+'_OBSERVATIONS.jsonl.gz'))}
    if len(rows)!=238 or set(rows)!={t['id']for t in tasks}:raise ValueError('All238 including abstentions required')
    ordered=[]
    for task in tasks:
        row=rows[task['id']]
        for key in('partition','states','initial_geometry','visible_measurements','domains','u_regions','unchanged_observation_math_sha256'):
            if row[key]!=task[key]:raise ValueError('Original prediction inputs changed: '+key)
        ordered.append(row)
    C.save_rows(output/'OBSERVATIONS.jsonl.gz',ordered);C.write(output/'SUMMARY.json',_summary(ordered))
    for key in('source_sha256','input_sha256','protected_user_sha256'):OLD._verify_map(protocol[key])
    C.write(output/'OBSERVATION_SEAL.json',dict(schema='FRESH_GEOMETRY_EDGE_ALL238_OBSERVATION_SEAL_V29',
        rows=238,ordered_ids=[t['id']for t in tasks],observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'),
        summary_sha256=C.sha(output/'SUMMARY.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=protocol['source_sha256'],
        partition_seals={p:C.sha(output/(p+'_OBSERVATION_SEAL.json'))for p in('CAL','REST')},
        all_frames_including_abstentions_preserved=True,no_numeric_reference_decodes=True,no_NN_or_pose_run=True))
    print(json.dumps(dict(sealed=238,summary=_summary(ordered))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','observe','finalize'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--partition',choices=('CAL','REST'),default='CAL')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='observe':observe(args.output,args.partition,args.workers)
    else:finalize(args.output)
