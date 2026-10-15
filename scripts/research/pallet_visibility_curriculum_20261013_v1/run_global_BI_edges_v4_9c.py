"""Sealed95 global B/I corrections and same-weight prior1/prior0 pose controls.

Corrections use actual saved masks/RGB, never labels. All correction and pose
predictions finish and seal before CAL/TEST annotations are opened for score.
V47/V48 cached controls remain exact; only changed mathematical inputs refit.
"""
from __future__ import annotations
import os
for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_thread_key]='1'
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_visibility_pose_v4_8 as V48
from . import evaluate_shared_edge_partial_b_v4_7 as V47
from . import visible_bi_global_edge_correction_v4_9c as G
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .scoring import pose_metric,distribution

OUTPUT=V47.PARENT.BOUNDARY/'global_BI_edge_eval_v4_9c'
ARMS=('SAME_V47_PRIOR1','SAME_V48_PRIOR0','ALL_VISIBLE_GLOBAL_EDGE_PRIOR1','ALL_VISIBLE_GLOBAL_EDGE_PRIOR0')


def _load():
    rows,model,config,bindings,parent=V48._load()
    old=C.read(V48.OUTPUT/'PROTOCOL.json');complete=C.read(V48.OUTPUT/'PREDICTION_COMPLETE.json')
    if complete['rows']!=95 or not complete['before_any_new_pose_scoring']:raise ValueError('Complete sealed V48 required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(old[key])
    caches={}
    for partition,count in (('CAL',46),('TEST',49)):
        journal=V48.OUTPUT/f'{partition}_PREDICTIONS.jsonl.gz';sp=V48.OUTPUT/f'{partition}_PREDICTION_SEAL.json';seal=C.read(sp)
        if (seal['rows']!=count or C.sha(journal)!=seal['predictions_sha256']
                or C.sha(V48.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']
                or C.sha(sp)!=complete['ordered_partition_prediction_seal_sha256'][partition]):raise ValueError('V48 cache seal changed')
        values=list(C.rows(journal));expected=[r['id'] for r in rows if r['partition']==partition]
        if [r['id'] for r in values]!=expected:raise ValueError('Exact ordered95 cache required')
        caches.update({r['id']:r for r in values})
        for p in (journal,sp):bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    for p in (V48.OUTPUT/'PROTOCOL.json',V48.OUTPUT/'PREDICTION_COMPLETE.json'):bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    masks=V47.PARENT.BOUNDARY/'image_evidence_v4_6';ms=C.read(masks/'SEAL.json')
    if ms['records']!=95 or ms['GT_or_pose_labels_read']:raise ValueError('Frozen95 image masks required')
    E._verify_sources(ms['source_sha256']);E._verify_sources(ms['input_sha256'])
    if C.sha(masks/'RECORDS.json')!=ms['records_sha256']:raise ValueError('Mask records changed')
    metas={r['id']:r for r in C.read(masks/'RECORDS.json')}
    if set(metas)!=set(caches):raise ValueError('Same95 RGB/mask population required')
    for p in (masks/'SEAL.json',masks/'RECORDS.json',masks/'PROTOCOL.json'):bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    for row in rows:
        cache=caches[row['id']];meta=metas[row['id']];path=masks/meta['mask_npz'];image=C.ROOT/row['image']
        if (cache['states']!=row['states'] or cache['frozen_BI_fit']!=row['frozen_BI_fit']
                or cache['partial_boundary_observation']!=row['partial_boundary_observation']
                or cache['u_region_observation']!=row['u_region_observation']
                or C.sha(path)!=meta['mask_file_sha256']
                or C.sha(image)!=row['image_sha256'] or meta['image_sha256']!=row['image_sha256']):raise ValueError('Frozen image/observation binding changed')
        row['_V48_cache']=cache;row['_mask_path']=str(path)
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path);bindings[str(image.relative_to(C.ROOT))]=C.sha(image)
    return rows,model,config,bindings,parent,old


def _correction(task):
    row=task;image=cv2.imread(str(C.ROOT/row['image']))
    with np.load(row['_mask_path'],allow_pickle=False) as z:mask=z['pallet_mask_clean'].copy()
    frozen=deepcopy(row['frozen_BI_fit']);before=E._digest(frozen)
    observation=G.correct_visible_bi_global_edges(image,mask,row['runtime']['initial_geometry'],row['states'],frozen)
    if E._digest(frozen)!=before:raise ValueError('Producer modified frozen input')
    fit=observation['visible_measurements'];old={m['native_corner']:m for m in frozen};new={m['native_corner']:m for m in fit}
    if set(new)!=set(old):raise ValueError('Production states fixed: same B/I measurement IDs required')
    changed=set(observation['changed_native_ids'])
    for k in old:
        if (old[k]['fit_covariance_px2']!=new[k]['fit_covariance_px2']
                or old[k]['fit_reliability_weight']!=new[k]['fit_reliability_weight']):raise ValueError('Original CAL total fit weights changed')
        if k not in changed and old[k]!=new[k]:raise ValueError('Unchanged visible measurement changed')
        if k in changed and row['states'][k] not in ('B','I'):raise ValueError('Excluded state remeasured')
    partial=deepcopy(row['partial_boundary_observation'])
    partial['original_boundary_observations_reference']=deepcopy(partial['boundary_observations'])
    partial['boundary_observations']=[b for b in partial['boundary_observations'] if b['native_corner'] not in changed]
    partial['removed_partial_ids_replaced_by_full_RGB']=[b['native_corner'] for b in partial['original_boundary_observations_reference'] if b['native_corner'] in changed]
    for record in partial['records']:
        if record['native_corner'] in changed:record.update(accepted_partial=False,reason='new_full_RGB_intersection_replaces_partial_normal_cost')
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],cohort=row['cohort'],image=row['image'],image_sha256=row['image_sha256'],
        states=row['states'],runtime=row['runtime'],frozen_BI_fit=fit,original_BI_fit=frozen,
        observation=observation,partial_boundary_observation=partial,u_region_observation=row['u_region_observation'],
        contract=dict(states_geometry_K_Q_OT_U_unchanged=True,original_CAL_fit_covariance_and_reliability_unchanged=True,
            unsupported_or_ambiguous_BI_unchanged=True,all_visible_BI_remeasurement=True,S_local_queries=0,U_local_queries=0,new_mask_forwards=0,GT_inputs=False)))


def _spec(source_row,correction,config):
    spec=E._physics_spec(source_row['_original_source'],correction['frozen_BI_fit'],config)
    return {**spec,'boundary_observations':correction['partial_boundary_observation']['boundary_observations'],
            'u_regions':correction['u_region_observation']['u_regions'],'boundary_factor_weight':1.}


def _pose(task):
    row,correction,model,config=task;started=time.monotonic();arms={
        ARMS[0]:deepcopy(row['arms'][V47.ARMS[2]]),ARMS[1]:deepcopy(row['_V48_cache']['arms'][V48.ARMS[1]])}
    audits={};new_calls=0
    for weight,arm,cache_arm in ((1.,ARMS[2],ARMS[0]),(0.,ARMS[3],ARMS[1])):
        cfg=deepcopy(config);cfg['prior_weight']=weight
        spec=_spec(row,correction,cfg);baseline=V48._spec(row,cfg)
        unchanged=E._digest(spec)==E._digest(baseline)
        if unchanged:arms[arm]=deepcopy(arms[cache_arm])
        else:
            solution=solve_boundary_u_region_joint_ot_pnp(**spec);new_calls+=1
            arms[arm]=V47._finish(row['_original_source'],correction['frozen_BI_fit'],model,solution,
                correction['u_region_observation']['u_regions'],arm,correction['partial_boundary_observation'])
        audits[arm]=dict(exact_original_spec_cache_used=unchanged,prior_weight=weight,new_input_sha256=E._digest(spec),old_input_sha256=E._digest(baseline))
    return C.finite({**correction,'arms':arms,'cache_audit':audits,'pose_solver_calls':new_calls,'elapsed_seconds':time.monotonic()-started})


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing experiment')
    rows,model,config,bindings,parent,old=_load()
    sources=deepcopy(old['source_sha256'])
    for path in (Path(__file__),Path(G.__file__)):sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256'];E._verify_sources(protected)
    output.mkdir(parents=True);protocol=dict(schema='ALL_VISIBLE_GLOBAL_BI_EDGE_PAIRED_V4_9C',ids=[r['id'] for r in rows],arms=list(ARMS),
        primary_population={'CAL':46,'TEST':49,'severe':0},source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        configuration=G.DEFAULT_CONFIGURATION,pose_prior_weights=[1.,0.],original_solver_configuration=config,
        unchanged_effective_CAL_fit_covariance_and_reliability=True,full_measured_point_replaces_same_corner_partial_terms=True,
        same_states_K_Q_model_OT_U=True,S_local_queries=0,U_image_or_line_or_precise_point_terms=0,
        no_model_or_mask_forward=True,all95_predictions_before_annotation_score=True,
        TEST_status='REUSED_ITERATIVE_EXPLORATORY',no_runtime_GT_or_ID_routing=True,workers=workers)
    C.write(output/'PROTOCOL.json',protocol)
    for rel,digest in sources.items():
        dest=output/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/rel,dest)
        if C.sha(dest)!=digest:raise ValueError('Source snapshot changed')
    started=time.monotonic();corrections={};poses={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        for partition,count in (('CAL',46),('TEST',49)):
            population=[r for r in rows if r['partition']==partition]
            futures={pool.submit(_correction,row):row['id'] for row in population}
            for future in as_completed(futures):
                value=future.result()
                if value['id']!=futures[future] or value['id'] in corrections:raise ValueError('Correction worker identity mismatch')
                corrections[value['id']]=value;C.write(output/f'{partition}_correction_frames'/(value['id'].replace(':','__')+'.json'),value)
                print(f'CORRECTION {len(corrections)}/95 {value["id"]} changed={value["observation"]["changed_native_ids"]}',flush=True)
            journal=output/f'{partition}_CORRECTIONS.jsonl.gz';C.save_rows(journal,[corrections[r['id']] for r in population])
            C.write(output/f'{partition}_CORRECTION_SEAL.json',dict(rows=count,predictions_sha256=C.sha(journal),protocol_sha256=C.sha(output/'PROTOCOL.json'),before_annotations=True))
        for v in (sources,bindings,protected):E._verify_sources(v)
        C.write(output/'CORRECTION_COMPLETE.json',dict(rows=95,CAL=46,TEST=49,severe=0,GT_labels_read=False))
        for partition,count in (('CAL',46),('TEST',49)):
            population=[r for r in rows if r['partition']==partition]
            futures={pool.submit(_pose,(row,corrections[row['id']],model,config)):row['id'] for row in population}
            for future in as_completed(futures):
                value=future.result()
                if value['id']!=futures[future] or value['id'] in poses:raise ValueError('Pose worker identity mismatch')
                poses[value['id']]=value;C.write(output/f'{partition}_pose_frames'/(value['id'].replace(':','__')+'.json'),value)
                C.write(output/'LIVE_PROGRESS.json',dict(phase='pose',completed=len(poses),total=95,latest_id=value['id'],elapsed_seconds=time.monotonic()-started))
                print(f'POSE {len(poses)}/95 {value["id"]} newfits={value["pose_solver_calls"]}',flush=True)
            journal=output/f'{partition}_POSE_PREDICTIONS.jsonl.gz';C.save_rows(journal,[poses[r['id']] for r in population])
            for v in (sources,bindings,protected):E._verify_sources(v)
            C.write(output/f'{partition}_POSE_PREDICTION_SEAL.json',dict(rows=count,predictions_sha256=C.sha(journal),protocol_sha256=C.sha(output/'PROTOCOL.json'),before_annotations=True))
    C.write(output/'PREDICTION_COMPLETE.json',dict(rows=95,CAL=46,TEST=49,severe=0,all_corrections_and_poses_before_labels=True,new_pose_fits=sum(r['pose_solver_calls'] for r in poses.values()),elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');complete=C.read(output/'PREDICTION_COMPLETE.json')
    if complete['rows']!=95 or not complete['all_corrections_and_poses_before_labels']:raise ValueError('Seal all95 before scoring')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    parts={}
    for partition,count in (('CAL',46),('TEST',49)):
        for stem,sealname in (('CORRECTIONS','CORRECTION_SEAL'),('POSE_PREDICTIONS','POSE_PREDICTION_SEAL')):
            path=output/f'{partition}_{stem}.jsonl.gz';seal=C.read(output/f'{partition}_{sealname}.json')
            if seal['rows']!=count or C.sha(path)!=seal['predictions_sha256'] or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256'] or not seal['before_annotations']:raise ValueError('Prediction binding changed')
        parts[partition]=list(C.rows(output/f'{partition}_POSE_PREDICTIONS.jsonl.gz'))
        if len(parts[partition])!=count or len({r['id'] for r in parts[partition]})!=count:raise ValueError('All-frame score population required')
    for partition in ('CAL','TEST'):
        labelpath=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if partition=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(labelpath)['rows'];values=[];inventory=[];before=[];after=[]
        for row in parts[partition]:
            label=labels[row['id']];alignment=_reference_alignment(row,label);mapping=np.asarray(label['metadata']['native_to_reference_corner_ids']);gt=np.asarray(label['target']['gt'])[mapping];valid=np.asarray(label['target']['valid'])[mapping];human=label['metadata']['human_states_native']
            old={m['native_corner']:m for m in row['original_BI_fit']};new={m['native_corner']:m for m in row['frozen_BI_fit']}
            for k in row['observation']['changed_native_ids']:
                inventory.append(dict(id=row['id'],native_corner=k,state=row['states'][k],human_state=human[k],valid=bool(valid[k]),native_compatible=alignment['compatible'],old_xy=old[k]['xy'],new_xy=new[k]['xy'],shift_px=float(np.linalg.norm(np.asarray(new[k]['xy'])-old[k]['xy'])),old_annotation_error_px=float(np.linalg.norm(np.asarray(old[k]['xy'])-gt[k])) if valid[k] else None,new_annotation_error_px=float(np.linalg.norm(np.asarray(new[k]['xy'])-gt[k])) if valid[k] else None))
            for k in old:
                if valid[k] and human[k]=='DIRECT_VISIBLE' and alignment['compatible']:
                    before.append(float(np.linalg.norm(np.asarray(old[k]['xy'])-gt[k])));after.append(float(np.linalg.norm(np.asarray(new[k]['xy'])-gt[k])))
            values.append(dict(id=row['id'],image=row['image'],states=row['states'],metrics={a:pose_metric(row['arms'][a]['actual_pose'],label['target']['truth']) for a in ARMS},
                accepted={a:row['arms'][a]['accepted'] for a in ARMS},ambiguous={a:row['arms'][a]['ambiguity_audit']['material_ambiguity'] for a in ARMS},physical_native_compatible=alignment['compatible'],changed_native_ids=row['observation']['changed_native_ids']))
        C.write(output/f'{partition}_SCORE.json',dict(partition=partition,severe=0,all_fallbacks_kept=True,rows=values,
            pose_summary_all={a:V48._summary(values,a) for a in ARMS},annotation_DIRECT_native_compatible=dict(points=len(before),before=distribution(before),after=distribution(after)),
            all_changed_inventory_before_valid_filter=inventory,changed_points=len(inventory),target_binding=E._binding(labelpath),TEST_reused_iterative=True,no_GT_or_label_routing_in_prediction=True))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.write(output/'SCORE_SEAL.json',dict(CAL_sha256=C.sha(output/'CAL_SCORE.json'),TEST_sha256=C.sha(output/'TEST_SCORE.json'),prediction_complete_sha256=C.sha(output/'PREDICTION_COMPLETE.json'),runtime_GT_inputs=False))
    print('SCORED_ALL95_GT_FREE_GLOBAL_OBSERVATIONS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--stage',choices=('predict','score','all'),default='all');args=parser.parse_args()
    if args.stage in ('predict','all'):predict(args.output,args.workers)
    if args.stage in ('score','all'):score(args.output)
