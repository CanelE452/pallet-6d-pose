"""Fresh143 reproduction of the frozen G10-WD then D-normal/e rule.

Consume sealed actual fresh RGB observations. Generate new143 prior1 and
two-WD prior0 search seeds, run immutable numeric G10 from those seeds, then
the immutable D/G13 partial refinement and same-family selector. No prior95
pose journal substitutes for a remaining-frame execution. Prediction only;
the separately owned derived-reference scorer opens after the complete seal.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import shutil
import time
import types

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import statewise_partial_boundary_u_solver_v4_6 as P
from . import statewise_two_wd_family_v4_9 as FAMILY
from . import evaluate_two_wd_families_v4_9 as W
from . import evaluate_global_outline_pose_v4_10 as G10
from . import evaluate_face_partial_BI_outline_v4_13d as D
from . import select_frozen_wd_partial_refinement_v4_13e as SELECTOR
from .statewise_uncertainty_calibration_v4 import load_uncertainty_model

INPUTS=E.COHORT/'remainder_after_test49_v1/fresh_observations_v4_14'
OUTPUT=INPUTS.parent/'pose_evaluation_v4_14'


def _load():
    protocol=C.read(INPUTS/'PROTOCOL.json');seal=C.read(INPUTS/'OBSERVATION_SEAL.json');journal=INPUTS/'OBSERVATIONS.jsonl.gz'
    if (seal['rows']!=143 or not seal['before_remaining_pose_prediction_and_scoring'] or seal['pose_solver_calls']!=0
            or seal['annotations_read'] or C.sha(journal)!=seal['observations_sha256']
            or C.sha(INPUTS/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Complete GT-free actual143 observation seal required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    mask_seal=C.read(INPUTS/'masks/SEAL.json');records=C.read(INPUTS/'masks/RECORDS.json')
    if mask_seal['records']!=143 or C.sha(INPUTS/'masks/RECORDS.json')!=mask_seal['records_sha256']:
        raise ValueError('Complete actual143 mask seal required')
    by_mask={m['id']:m for m in records};rows=list(C.rows(journal))
    if [r['id'] for r in rows]!=seal['ordered_ids'] or len(rows)!=143:raise ValueError('Exact ordered143 required')
    for r in rows:
        r['_MASK']=by_mask[r['id']]
        if r['_MASK']['image_sha256']!=r['image_sha256']:raise ValueError('Same actual143 RGB/mask required')
        maskpath=INPUTS/'masks'/r['_MASK']['mask_npz']
        if C.sha(maskpath)!=r['_MASK']['mask_file_sha256']:raise ValueError('Actual143 mask bytes changed')
    model_path=E.FIX/'calibration/final_adapter_CAL46/MODEL.json';model=load_uncertainty_model(model_path)
    config=deepcopy(protocol['solver_configuration'])
    if config['prior_weight']!=1. or config['bias_strength']!=1.:raise ValueError('Frozen original P1/bias1 seed configuration required')
    sources=deepcopy(protocol['source_sha256']);inputs=deepcopy(protocol['input_sha256'])
    for p in (INPUTS/'PROTOCOL.json',INPUTS/'OBSERVATION_SEAL.json',journal,INPUTS/'masks/SEAL.json',INPUTS/'masks/RECORDS.json',model_path):
        inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    return rows,model,config,sources,inputs,protocol['protected_user_sha256']


def _run(task):
    row,model,config=task;started=time.monotonic();before=E._digest(row)
    spec=W.PREVIOUS._spec(row,config)
    prior=P.solve_boundary_u_region_joint_ot_pnp(**spec)
    prior.pop('parent_baseline_solution',None)
    prior_arm=W.V47._finish(row['_original_source'],row['frozen_BI_fit'],model,prior,
        row['u_region_observation']['u_regions'],'FRESH143_PRIOR1_SEARCH_SEED',row['partial_boundary_observation'])
    fresh=deepcopy(row)
    # Compatibility field for immutable G10 caller, containing newly executed
    # remaining-frame evidence only. No previous V48 journal is consulted.
    fresh['_cached_V48']={'arms':{W.PREVIOUS.ARMS[0]:prior_arm},'origin':'fresh_remaining_frame_P1_search'}
    cfg0=deepcopy(config);cfg0['prior_weight']=0.
    family_spec=W.PREVIOUS._spec(fresh,cfg0)
    families=FAMILY.solve_two_wd_families(family_spec['xyz'],family_spec['K'],family_spec['states'],
        family_spec['visible'],family_spec['domains'],family_spec['image_hw'],family_spec['coarse_pose'],
        initializations=family_spec['initializations'],config=cfg0,
        boundary_observations=family_spec['boundary_observations'],u_regions=family_spec['u_regions'],
        solver_callable=P.solve_boundary_u_region_joint_ot_pnp)
    fresh['_WD']=families
    operation=types.FunctionType(G10._run.__code__,{**G10._run.__globals__,'MASKS':INPUTS/'masks'},name='_fresh_remaining_numeric_G10')
    old=operation((fresh,model,cfg0))
    # D's actual face RGB producer, typed likelihood revocation and G13
    # refinement are called unchanged. Its prepass binds this fresh input.
    # D._run's nested old caller uses G10.MASKS from its global module. Supply
    # a private G10 module facade instead of mutating the immutable module.
    facade=types.SimpleNamespace(**{**vars(G10),'MASKS':INPUTS/'masks'})
    d_observe=types.FunctionType(D._observe.__code__,{**D._observe.__globals__,'OLD10':facade},name='_fresh_remaining_D_observation')
    observed=d_observe((fresh,C.read(D.LOCALIZATION_PATH)))
    # Its cloned OLD10._run inherits that function's globals, so the function
    # facade carries a separately rebound numeric caller as well.
    facade._run=operation
    face_operation=types.FunctionType(D._run.__code__,{**D._run.__globals__,'OLD10':facade},name='_fresh_remaining_D_refinement')
    partial=face_operation((fresh,model,cfg0,old,observed))
    selected=SELECTOR.select_same_family(old,partial)
    if E._digest(row)!=before:raise ValueError('Fresh remaining solver mutated sealed observations')
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],image=row['image'],
        image_sha256=row['image_sha256'],states=row['states'],frozen_family=old['selected_family'],
        d_reselected_family=partial['arms'][D.ARMS[1]]['selected_family'],
        selected_model_geometry={key:deepcopy(next(f for f in old['families'] if f['name']==old['selected_family'])[key])
            for key in ('native_xyz','cf_extents','physical_registration_Q')},
        arms={'OLD_G10':old['arms']['SELECTED_JOINT_OUTLINE'],
            'D_JOINT_RESELECT':partial['arms'][D.ARMS[1]]['arms']['SELECTED_JOINT_OUTLINE'],
            SELECTOR.ARMS[2]:selected},
        full_prior_seed=E._compact_solution(prior),full_WD_search=families,
        old_global_outline=old,partial_global_outline=partial,
        fresh_input_sha256=before,elapsed_seconds=time.monotonic()-started,
        contract=dict(GT_inputs=False,old95_pose_cache_used_for_remaining_frames=False,
            new_remaining_P1_and_full_WD0_search_executed=True,old_geometry_family_fixed_before_precision_refinement=True,
            fixed_CAL_model_states_BI_covariance_OT_U_Q_preserved=True,S_local_queries=0,U_local_queries=0,
            derived_reference_scoring_is_separate=True)))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every prior remaining143 execution')
    rows,model,config,sources,inputs,protected=_load()
    for module in (__file__,P.__file__,FAMILY.__file__,W.__file__,G10.__file__,D.__file__,D.G.__file__,D.OBS.__file__,SELECTOR.__file__):
        p=Path(module);sources[str(p.relative_to(C.ROOT))]=C.sha(p)
    for hashes in (sources,inputs,protected):E._verify_sources(hashes)
    output.mkdir(parents=True)
    protocol=dict(schema='FRESH_REMAINDER143_FROZEN_WD_D_V4_14',rows=143,severe=0,ids=[r['id'] for r in rows],
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        solver_configuration=config,global_outline_configuration=G10.G.DEFAULT_OUTLINE_CONFIGURATION,
        partial_producer_configuration=D.OBS.DEFAULT_CONFIGURATION,rule=SELECTOR.RULE,
        new_remaining_frames_not_old95_pose_cache=True,GT_labels_or_metrics_read=False,
        same_frozen_models_and_parameters_as_original49=True,workers=workers,CPU_threads_per_worker=1,
        actual143_observation_seal_before_new_pose_prediction=True,complete143_before_scoring=True)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        p=output/'code_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,p);assert C.sha(p)==digest
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(_run,(r,model,config)):r['id'] for r in rows}
        for future in as_completed(futures):
            r=future.result();done[r['id']]=r
            C.write(output/'frames'/(r['id'].replace(':','__')+'.json'),r)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=143,latest_id=r['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/143 {r["id"]} fixed={r["frozen_family"]}',flush=True)
    for hashes in (sources,inputs,protected):E._verify_sources(hashes)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[r['id']] for r in rows])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=143,severe=0,ordered_ids=[r['id'] for r in rows],
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        before_remaining_GT_or_metric_read=True,elapsed_seconds=time.monotonic()-started,
        rule=SELECTOR.RULE,all_frames_and_fallbacks_preserved=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();predict(args.output,args.workers)
