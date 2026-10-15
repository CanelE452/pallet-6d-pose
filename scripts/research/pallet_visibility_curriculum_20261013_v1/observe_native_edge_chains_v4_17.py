"""Uniform238 cached-RGB NMS/topology observations; no NN/pose/GT scoring.

Preparation freezes the original v15 image/mask/geometry/visible input
population and sources. Exactly the same observation function handles CAL46
and the later TEST49+remaining143. All alternatives and zero-edge frames stay
in the archive. No old q, noise, state, O/T domain or U region is replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np
import scipy

from . import common as C
from . import image_supported_edge_chains_v4_17b as NMS
from . import native_edge_topology_junctions_v4_17 as TOPOLOGY


BOUNDARY=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
PARENT=BOUNDARY/'fixed_wd_archived_tangent_eval_v4_15'
OUTPUT=BOUNDARY/'native_edge_observations_v4_17'


def _digest(value):
    return hashlib.sha256(json.dumps(C.finite(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _binding(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(C.ROOT)),sha256=C.sha(path),bytes=path.stat().st_size)


def _verify_map(bindings):
    for relative,digest in bindings.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Bound source/input changed: '+relative)


def _math_fields(task):
    return {key:task[key] for key in ('states','initial_geometry','visible_measurements','domains','u_regions',
                                     'old_boundary_all_sha256','nonself_prediction_scale_px')}


def _thin_task(row):
    """Whitelist production quantities; never copy raw/S/reference containers."""
    geometry=row['finish_input']['_original_source']['initial_geometry']
    visible=row['spec']['visible'];states=row['spec']['states']
    if len(states)!=8 or len(geometry['initial_self_hidden'])!=8:raise ValueError('Eight authoritative state gates required')
    for record in visible:
        k=NMS.V3._id(record)
        if geometry['initial_self_hidden'][k] or states[k] not in ('B','I'):
            raise ValueError('Only permanent nonS B/I records belong in the frozen visible archive')
    if visible!=row['finish_input']['frozen_BI_fit']:raise ValueError('Archived visible math differs')
    if states!=row['finish_input']['_original_source']['states']:raise ValueError('Archived authoritative states differ')
    original=row['stage1_family_seed_bank']['ORIGINAL_WD']['family']
    if (not np.allclose(geometry['native_xyz'],original['native_xyz'],atol=1e-12,rtol=0)
            or not np.allclose(geometry['physical_registration_Q'],original['physical_registration_Q'],atol=1e-12,rtol=0)):
        raise ValueError('Observer must use the initial ORIGINAL registered geometry, not a reselected WD phase')
    partial=row['finish_input']['partial_boundary_observation']
    scale=float(partial['unchanged_nonself_prediction_scale_px'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('Immutable nonS prediction scale required')
    task=dict(id=row['id'],partition=row['partition'],session=row['session'],
        image=row['image'],image_sha256=row['image_sha256'],mask_binding=deepcopy(row['mask_binding']),
        states=deepcopy(states),initial_geometry=deepcopy(geometry),visible_measurements=deepcopy(visible),
        domains=deepcopy(row['spec']['domains']),u_regions=deepcopy(row['u_regions']),
        old_boundary_all_sha256=_digest(row['boundary_all']),nonself_prediction_scale_px=scale,
        initial_geometry_is_model_only=True,source_archive_row_id=row['id'])
    task['unchanged_observation_math_sha256']=_digest(_math_fields(task))
    return task


def _sources(parent):
    result=dict(parent['source_sha256'])
    modules=(sys.modules[__name__],C,NMS,TOPOLOGY,NMS.D,NMS.G,NMS.F,NMS.V3)
    for module in modules:
        path=Path(module.__file__).resolve();result[str(path.relative_to(C.ROOT))]=C.sha(path)
    return result


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every earlier observation preparation')
    parent=C.read(PARENT/'PROTOCOL.json');seal=C.read(PARENT/'INPUT_SEAL.json');journal=PARENT/'INPUTS.jsonl.gz'
    if seal['rows']!=238 or C.sha(journal)!=seal['input_sha256'] or C.sha(PARENT/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Exact complete238 frozen production-input archive required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify_map(parent[key])
    tasks=[_thin_task(row) for row in C.rows(journal)]
    if len(tasks)!=238 or [t['id'] for t in tasks]!=seal['ordered_ids'] or len(set(t['id'] for t in tasks))!=238:
        raise ValueError('No duplicate, dropped or reordered original238 frame allowed')
    if Counter(t['partition'] for t in tasks)!=Counter(CAL=46,TEST=49,REMAINDER=143):
        raise ValueError('Exact nonsevere CAL46/TEST49/remaining143 inventory required')
    sources=_sources(parent);inputs={str(path.relative_to(C.ROOT)):C.sha(path)
        for path in (journal,PARENT/'PROTOCOL.json',PARENT/'INPUT_SEAL.json')}
    for task in tasks:
        image=C.ROOT/task['image'];mask=C.ROOT/task['mask_binding']['path']
        if C.sha(image)!=task['image_sha256'] or C.sha(mask)!=task['mask_binding']['sha256']:
            raise ValueError('Frozen original RGB/mask file changed: '+task['id'])
        inputs[task['image']]=task['image_sha256'];inputs[task['mask_binding']['path']]=task['mask_binding']['sha256']
    output.mkdir(parents=True)
    protocol=dict(schema='UNIFORM_NATIVE_EDGE_OBSERVATION_PROTOCOL_V4_17',ordered_ids=seal['ordered_ids'],
        population=dict(CAL=46,TEST=49,REMAINDER=143,total=238,severe_excluded_by_inherited_frozen_inventory=True),
        prepared_before_new_RGB_observations_UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=parent['protected_user_sha256'],
        inherited_parent_population=parent['population'],configuration=NMS.DEFAULT_CONFIGURATION,
        source_binding_scope='conservative immutable v15 source union plus new observer/topology/driver',
        geometry='initial ORIGINAL native geometry and permanent initial-S gate; no pose/family re-selection',
        image_evidence='decode exact cached original RGB and frozen full clean SAM mask; actual image NMS/chain TLS only',
        all_chain_and_topology_alternatives_preserved=True,selected_chain_indices_display_only_and_removed_from_measurement_archive=True,
        unchanged_production_quantities=['states','visible q/covariance/fit covariance/reliability','geometry/K/Q','OT domains','U regions','nonS scale'],
        S_local_queries=0,U_corner_queries=0,O_T_coordinate_queries=0,
        new_NN_forwards=0,pose_calls=0,reference_coordinate_decodes=0,model_fitting=0,
        engineering_angle_floor_is_not_CAL_total_error_model=True,
        existing_cached_v17_CAL_angular_model_not_used=True,
        full_KP_adoptions=0,default_pipeline_replacement=False,
        same_worker_function_and_config_for_CAL_and_later192=True,
        exploratory_reused_data_not_an_independent_TEST=True,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,scipy=scipy.__version__))
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Source snapshot mismatch')
    C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for value in (sources,inputs,parent['protected_user_sha256']):_verify_map(value)
    C.write(output/'PREPARATION_SEAL.json',dict(schema='UNIFORM_NATIVE_EDGE_OBSERVATION_PREPARATION_SEAL_V4_17',
        protocol_sha256=C.sha(output/'PROTOCOL.json'),input_sha256=C.sha(output/'INPUTS.jsonl.gz'),rows=238,
        ordered_ids=seal['ordered_ids'],source_sha256=sources,prepared_before_new_RGB_observation=True,
        no_reference_numeric_decode=True))
    print(json.dumps(dict(prepared=238,output=str(output),before_new_RGB_observations=True)),flush=True)


def _load(output):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256'] or C.sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']:
        raise ValueError('Prepared source/input seal changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify_map(protocol[key])
    tasks=list(C.rows(output/'INPUTS.jsonl.gz'))
    if [t['id'] for t in tasks]!=seal['ordered_ids'] or len(tasks)!=238:raise ValueError('Incomplete prepared238 inventory')
    return tasks,protocol,seal


def _init_worker():
    cv2.setNumThreads(1)


def observe_one(task):
    """The identical production observation operation for every partition."""
    started=time.monotonic();before=_digest(_math_fields(task))
    if before!=task['unchanged_observation_math_sha256']:raise ValueError('Input observation math changed')
    image_path=C.ROOT/task['image'];mask_path=C.ROOT/task['mask_binding']['path']
    if C.sha(image_path)!=task['image_sha256'] or C.sha(mask_path)!=task['mask_binding']['sha256']:
        raise ValueError('Actual cached RGB/mask changed before observation')
    image=cv2.imread(str(image_path))
    if image is None:raise ValueError('Original actual RGB decode failed')
    with np.load(mask_path,allow_pickle=False) as data:mask=data['pallet_mask_clean'].copy()
    binding=dict(sha256=task['image_sha256'],image_hw=list(image.shape[:2]))
    observation=NMS.observe_image_supported_edge_chains(image,mask,task['initial_geometry'],task['states'],
        task['visible_measurements'],image_binding=binding)
    producer_digest=_digest(observation)
    # A producer display reference cannot become an implicit chosen measurement.
    # Every chain, pixel and hypothesis remains untouched and archived.
    for ray in observation['edge_records']:
        if 'selected_chain_index' in ray:
            ray['producer_display_reference_chain_index']=ray.pop('selected_chain_index')
            ray['no_chain_selected_as_measurement']=True
    proposal=TOPOLOGY.propose_native_edge_junctions(observation,mask,task['initial_geometry'],task['states'],
        task['visible_measurements'],image_binding=observation['image_binding'],mask_binding=dict(
            image_sha256=task['image_sha256'],mask_file_binding=task['mask_binding'],
            mask_pixel_sha256=TOPOLOGY.mask_pixel_sha256(mask)))
    if _digest(_math_fields(task))!=before:raise RuntimeError('Observer mutated fixed q/cov/state/geometry/domains')
    rays=observation['edge_records'];supported=[r for r in rays if r.get('supported')]
    chains=sum(len(r.get('chains',[])) for r in rays)
    candidates=sum(len(r['candidates']) for r in proposal['records'])
    summary=dict(supported_directed_edges=len(supported),
        supported_unique_native_edges=len({tuple(sorted(r['native_edge'])) for r in supported}),
        actual_chain_hypotheses=chains,B_junction_candidates=candidates,
        B_corners_with_junction_candidate=sum(bool(r['candidates']) for r in proposal['records']),
        external_chain_hypotheses=sum(sum(c['mask_context_kind']=='TARGET_BACKGROUND_OUTLINE' for c in r.get('chains',[])) for r in rays),
        internal_chain_hypotheses=sum(sum(c['mask_context_kind']=='TARGET_TARGET_INTERNAL' for c in r.get('chains',[])) for r in rays),
        linking_overflow_abstained_edges=sum(r.get('component_linking_audit',{}).get('combinatorial_overflow_abstained',False) for r in rays),
        zero_supported_edges=len(supported)==0)
    return dict(schema='UNIFORM_NATIVE_EDGE_OBSERVATION_ROW_V4_17',id=task['id'],partition=task['partition'],session=task['session'],
        image=task['image'],image_sha256=task['image_sha256'],mask_binding=task['mask_binding'],
        states=task['states'],initial_geometry=task['initial_geometry'],visible_measurements=task['visible_measurements'],
        domains=task['domains'],u_regions=task['u_regions'],nonself_prediction_scale_px=task['nonself_prediction_scale_px'],
        unchanged_observation_math_sha256=before,producer_output_sha256_before_display_metadata_only_removal=producer_digest,
        chain_observation=observation,junction_proposals=proposal,summary=summary,
        elapsed_seconds=time.monotonic()-started,contract=dict(old_q_covariance_state_changes=0,
            precise_KP_adoptions=0,pose_calls=0,NN_calls=0,reference_coordinate_decodes=0,
            S_local_queries=0,U_corner_queries=0,O_T_coordinate_queries=0,
            all_alternatives_preserved=True,engineering_uncertainty_not_CAL_model=True))


def _stop_pool(pool):
    processes=list(getattr(pool,'_processes',{}).values())
    pool.shutdown(wait=False,cancel_futures=True)
    for process in processes:
        if process.is_alive():process.terminate()
    for process in processes:process.join(timeout=2.)


def _summary(rows):
    return dict(rows=len(rows),partition_counts=dict(Counter(r['partition'] for r in rows)),
        frames_with_supported_edges=sum(not r['summary']['zero_supported_edges'] for r in rows),
        frames_with_zero_supported_edges=sum(r['summary']['zero_supported_edges'] for r in rows),
        supported_directed_edges=sum(r['summary']['supported_directed_edges'] for r in rows),
        supported_unique_native_edges=sum(r['summary']['supported_unique_native_edges'] for r in rows),
        actual_chain_hypotheses=sum(r['summary']['actual_chain_hypotheses'] for r in rows),
        B_junction_candidates=sum(r['summary']['B_junction_candidates'] for r in rows),
        frames_with_B_junction_candidate=sum(r['summary']['B_junction_candidates']>0 for r in rows),
        linking_overflow_abstained_edges=sum(r['summary']['linking_overflow_abstained_edges'] for r in rows),
        old_q_covariance_states_changed=0,precise_KP_adoptions=0,NN_calls=0,pose_calls=0,reference_coordinate_decodes=0)


def observe(output=OUTPUT,partition='CAL',workers=3):
    output=Path(output).resolve();tasks,protocol,prep=_load(output)
    if partition not in ('CAL','REST'):raise ValueError('Use exact CAL46 or the later192')
    if partition=='REST':_verify_observation_seal(output,'CAL')
    chosen=[t for t in tasks if (t['partition']=='CAL')==(partition=='CAL')]
    expected=46 if partition=='CAL' else 192
    if len(chosen)!=expected:raise ValueError('Incomplete uniform subset')
    name=partition+'_OBSERVATIONS';final=output/(name+'.jsonl.gz');checkpoint=output/(name+'_checkpoint.jsonl.gz')
    if final.exists() or checkpoint.exists() or (output/(partition+'_OBSERVATION_SEAL.json')).exists():
        raise FileExistsError('Preserve every completed or partial execution')
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=8:raise ValueError('Use1..8 CPU workers')
    C.write(output/(partition+'_EXECUTION_PROTOCOL.json'),dict(workers=workers,OpenCV_threads_per_worker=1,
        BLAS_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        UTC=datetime.now(timezone.utc).isoformat(),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        ordered_ids=[t['id'] for t in chosen],same_observation_function='observe_one'))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_init_worker)
    futures={pool.submit(observe_one,t):t['id'] for t in chosen};completed={};started=time.monotonic()
    try:
        with gzip.open(checkpoint,'wt',compresslevel=5) as durable:
            for future in as_completed(futures):
                identifier=futures[future]
                try:row=future.result()
                except BaseException as error:
                    C.write(output/(partition+'_EXECUTION_FAILURE.json'),dict(id=identifier,error=repr(error),
                        completed_ids=list(completed),no_observation_seal=True,no_GT_or_pose_or_NN_run=True))
                    raise
                if row['id']!=identifier:raise ValueError('Worker frame identity differs')
                completed[identifier]=row
                durable.write(json.dumps(C.finite(row),separators=(',',':'),allow_nan=False)+'\n');durable.flush()
                print(json.dumps(dict(partition=partition,completed=len(completed),total=expected,id=identifier,
                    elapsed_seconds=round(time.monotonic()-started,3),summary=row['summary'])),flush=True)
    except BaseException:
        _stop_pool(pool);raise
    else:pool.shutdown(wait=True)
    ordered=[completed[t['id']] for t in chosen]
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify_map(protocol[key])
    C.save_rows(final,ordered);summary=_summary(ordered)
    C.write(output/(partition+'_SUMMARY.json'),summary)
    C.write(output/(partition+'_OBSERVATION_SEAL.json'),dict(schema='UNIFORM_NATIVE_EDGE_OBSERVATION_PARTITION_SEAL_V4_17',
        rows=expected,ordered_ids=[t['id'] for t in chosen],observations_sha256=C.sha(final),
        summary_sha256=C.sha(output/(partition+'_SUMMARY.json')),
        execution_protocol_sha256=C.sha(output/(partition+'_EXECUTION_PROTOCOL.json')),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=protocol['source_sha256'],no_numeric_reference_decodes=True,no_NN_or_pose_run=True))
    print(json.dumps(dict(sealed=partition,summary=summary)),flush=True)


def _verify_observation_seal(output,partition):
    seal=C.read(output/(partition+'_OBSERVATION_SEAL.json'))
    for path,key in ((output/(partition+'_OBSERVATIONS.jsonl.gz'),'observations_sha256'),
                     (output/(partition+'_SUMMARY.json'),'summary_sha256'),
                     (output/(partition+'_EXECUTION_PROTOCOL.json'),'execution_protocol_sha256'),
                     (output/'PREPARATION_SEAL.json','preparation_seal_sha256'),(output/'PROTOCOL.json','protocol_sha256')):
        if C.sha(path)!=seal[key]:raise ValueError('Uniform observation partition seal changed')
    return seal


def finalize(output=OUTPUT):
    output=Path(output).resolve();tasks,protocol,prep=_load(output)
    if (output/'OBSERVATION_SEAL.json').exists():raise FileExistsError('Preserve complete238 seal')
    pieces=[_verify_observation_seal(output,name) for name in ('CAL','REST')]
    rows={r['id']:r for name in ('CAL','REST') for r in C.rows(output/(name+'_OBSERVATIONS.jsonl.gz'))}
    if len(rows)!=238 or set(rows)!={t['id'] for t in tasks}:raise ValueError('Every original238 including zero-edge/fallback frames required')
    ordered=[]
    for task in tasks:
        row=rows[task['id']]
        if (row['partition']!=task['partition'] or row['states']!=task['states']
                or row['visible_measurements']!=task['visible_measurements']
                or row['unchanged_observation_math_sha256']!=task['unchanged_observation_math_sha256']):
            raise ValueError('Output frame/state/q/noise binding changed')
        ordered.append(row)
    C.save_rows(output/'OBSERVATIONS.jsonl.gz',ordered);summary=_summary(ordered)
    C.write(output/'SUMMARY.json',summary)
    C.write(output/'OBSERVATION_SEAL.json',dict(schema='UNIFORM_NATIVE_EDGE_OBSERVATION_SEAL_V4_17',rows=238,
        ordered_ids=[t['id'] for t in tasks],observations_sha256=C.sha(output/'OBSERVATIONS.jsonl.gz'),
        summary_sha256=C.sha(output/'SUMMARY.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),partition_seals={name:C.sha(output/(name+'_OBSERVATION_SEAL.json')) for name in ('CAL','REST')},
        source_sha256=protocol['source_sha256'],all_frames_including_abstentions_preserved=True,
        no_old_q_covariance_state_changes=True,no_numeric_reference_decodes=True,no_NN_or_pose_run=True,
        default_pipeline_replaced=False))
    print(json.dumps(dict(sealed=238,summary=summary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','observe','finalize'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--partition',choices=('CAL','REST'),default='CAL')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='observe':observe(args.output,args.partition,args.workers)
    else:finalize(args.output)
