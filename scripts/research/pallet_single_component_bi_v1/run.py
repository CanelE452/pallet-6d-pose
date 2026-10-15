"""Predictor-only component ablation; no references or method selection.

Four existing RAW/combined outputs remain byte-identical. The two isolated
initializers receive their own frozen state/mask/B/I computation and solver.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from multiprocessing import get_context
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1.instrumentation import instrument

ARMS=('R0','R1','N3_F','N3_BI','SUBPIX_F','SUBPIX_BI','N0','N1')
BRANCHES=('N3_ONLY','SUBPIX_ONLY')
PAIR=dict(N3_ONLY=('N3_F','N3_BI'),SUBPIX_ONLY=('SUBPIX_F','SUBPIX_BI'))

def freeze(folder,prior):
    folder,prior=Path(folder),Path(prior);code=Path(__file__).resolve().parents[3]
    frames=B.read(folder/'inputs_v1/INPUTS.json')['frames'];seal=B.read(folder/'inputs_v1/INPUT_SEAL.json')
    if len(frames)!=238 or len({f['id'] for f in frames})!=238:raise ValueError('Fixed238 required')
    original=B.read(prior/'inputs_v2/INPUTS.json')['frames']
    if any({k:v for k,v in a.items() if k!='arms'}!={k:v for k,v in b.items() if k!='arms'} for a,b in zip(frames,original)):
        raise ValueError('Same original population, image and camera metadata required')
    for a,b in zip(frames,original):
        if a['id']!=b['id'] or any(a['arms'][x]!=v for x,v in b['arms'].items()):
            raise ValueError('Original RAW/N3_SUBPIX initialization changed')
    old=B.read(prior/'run_v2/PROTOCOL.json');sources={**old['source_sha256'],**old['legacy_source_sha256']}
    for name in ('__init__.py','inputs.py','run.py','score.py'):
        p=Path(__file__).with_name(name);sources[str(p.relative_to(code))]=RT.sha256(p)
    for name,h in sources.items():
        if RT.sha256(code/name)!=h:raise ValueError('Source changed: '+name)
    data=B.read(folder/'inputs_v1/INPUTS.json')
    if 'inputs_sha256' in seal and seal['inputs_sha256']!=RT.sha256(folder/'inputs_v1/INPUTS.json'):
        raise ValueError('New input seal mismatch')
    protocol=dict(schema='FIXED_SINGLE_COMPONENT_BI_ABLATION_V1',UTC=datetime.now(timezone.utc).isoformat(),
        arms=list(ARMS),new_branches=list(BRANCHES),rows=238,arm_rows=1904,
        ordered_ids=[f['id'] for f in frames],populations=dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192),
        original_main_publication='dea11c25132bfb87a433073853bb632fb5170658',
        new_learning_seeds=0,new_training=0,new_data=0,new_GT_or_visibility_edits=0,
        new_YOLO_N3_or_SubPix_forwards=0,initial_coordinates='exact original same-seed four-method 20261008 cache',
        B_I='exact minimal runtime; each isolated initializer recomputes own geometry/state/prompts/masks/measurements',
        S='no local state/mask prompt/image correction/PnP image observation; conditional geometry and final model projection only',
        O_T_U='excluded from exact 2D pose correspondences; no region cost',
        reused_arms=['R0','R1','N0','N1'],new_arms=['N3_F','N3_BI','SUBPIX_F','SUBPIX_BI'],
        solver_configuration=RT.SOLVER_CONFIG,source_sha256=sources,
        immutable_inputs_sha256={str(p):RT.sha256(p) for p in (
            folder/'inputs_v1/INPUTS.json',folder/'inputs_v1/INPUT_SEAL.json',
            prior/'inputs_v2/INPUTS.json',prior/'run_v2/PROTOCOL.json',
            prior/'run_v2/PREDICTIONS.jsonl.gz',prior/'run_v2/PREDICTION_SEAL.json',prior/'run_v2/OBSERVATION_SEAL.json')},
        no_method_selection_or_threshold_changes=True,publish='user explicitly requested main; no new branch',
        operating_default_or_paper_changed=False,previous_development_exposure=True,
        success='translation_cm<=5 AND rotation_deg<=5',historical='strict <5 and <5',
        contrasts=[['R1','R0'],['N1','N0'],['N3_F','R0'],['SUBPIX_F','R0'],['N0','N3_F'],['N0','SUBPIX_F'],
            ['N3_BI','N3_F'],['SUBPIX_BI','SUBPIX_F'],['N1','N3_BI'],['N1','SUBPIX_BI'],
            ['N3_BI','R1'],['SUBPIX_BI','R1']],
        model_descriptors=old['model_descriptors'],caps=dict(gpu_process_seconds=3600,cpu_seconds=7200,pose_workers=3,threads=1))
    B.write(folder/'PROTOCOL.json',protocol);print('FROZEN',RT.sha256(folder/'PROTOCOL.json'),flush=True)

def verify(folder,source_root=None):
    folder=Path(folder);p=B.read(folder/'PROTOCOL.json');code=Path(__file__).resolve().parents[3]
    for name,h in p['source_sha256'].items():
        if RT.sha256(code/name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in p['immutable_inputs_sha256'].items():
        if RT.sha256(name)!=h:raise ValueError('Frozen input changed: '+name)
    if source_root:
        for info in p['model_descriptors'].values():
            if RT.sha256(Path(source_root)/info['path'])!=info['sha256']:raise ValueError('Frozen model changed')
    return p

def observe(folder,source_root):
    folder=Path(folder);verify(folder,source_root);frames=B.read(folder/'inputs_v1/INPUTS.json')['frames']
    budget=B.Budget(folder);budget.check('gpu_process');begin=time.monotonic();done=0;status='incomplete'
    try:
        model=B.components(source_root,'cuda')
        for i,f in enumerate(frames):
            for branch in BRANCHES:
                path=folder/'run/observations'/f'{i:03d}_{branch}.json'
                key=B.digest(dict(frame=f,branch=branch,protocol=RT.sha256(folder/'PROTOCOL.json')))
                if path.exists():
                    cached=B.read(path)
                    if cached['binding']!=key or cached['result_sha256']!=B.digest(cached['result']):raise ValueError('Own observation cache changed')
                    m=cached['result']['mask_binding']
                    if RT.sha256(folder/'run'/m['path'])!=m['sha256']:raise ValueError('Own mask changed')
                    continue
                tick=time.monotonic()
                with instrument(model) as meter:
                    result=B._observed(source_root,f,branch,model);calls=meter.snapshot()
                masks=result.pop('masks');mp=folder/'run/masks'/f'{i:03d}_{branch}.npz'
                mp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mp,**masks)
                result.update(status='observed',actual_calls=calls,elapsed_seconds=time.monotonic()-tick,
                    mask_binding=dict(path=str(mp.relative_to(folder/'run')),sha256=RT.sha256(mp)))
                B.write(path,dict(id=f['id'],branch=branch,binding=key,result=result,result_sha256=B.digest(result)))
                done+=1
                if done%20==0 or done==476:print('OBS',done,476,flush=True)
                budget.check('gpu_process',time.monotonic()-begin)
        paths=sorted((folder/'run/observations').glob('*.json'))
        if len(paths)!=476:raise ValueError('All476 observations required')
        B.write(folder/'run/OBSERVATION_SEAL.json',dict(rows=238,branches=476,
            own_initializer_geometry_states_masks_BI=True,GT_inputs=False,
            protocol_sha256=RT.sha256(folder/'PROTOCOL.json'),
            observations_sha256={p.name:RT.sha256(p) for p in paths}),same_ok=True)
        status='complete'
    except Exception:
        (folder/'OBSERVE_TECHNICAL_FAILURE.txt').write_text(traceback.format_exc());raise
    finally:budget.record('single_component_own_observations','gpu_process',time.monotonic()-begin,status=status,new_observations=done)

def solve(folder,prior):
    folder,prior=Path(folder),Path(prior);protocol=verify(folder);frames=B.read(folder/'inputs_v1/INPUTS.json')['frames']
    budget=B.Budget(folder);budget.check('cpu_solver_analysis');begin=time.monotonic();done=0;cpu=0.;status='incomplete'
    try:
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
            tasks={}
            for i,f in enumerate(frames):
                for branch in BRANCHES:
                    o=B.read(folder/'run/observations'/f'{i:03d}_{branch}.json');target=folder/'run/poses'/f'{i:03d}_{branch}.json'
                    if o['result_sha256']!=B.digest(o['result']):raise ValueError('Observation changed')
                    if target.exists():
                        old=B.read(target)
                        if old['observation_binding']!=o['binding'] or old['result_sha256']!=B.digest(old['result']):raise ValueError('Own pose changed')
                        continue
                    tasks[pool.submit(B._task,(o['result']['observation'],f['arms'][branch],f['K'],f['raw_hw']))]=(target,f,branch,o['binding'])
            for future in as_completed(tasks):
                target,f,branch,key=tasks[future];result,seconds,calls=future.result();cpu+=seconds;done+=1
                B.write(target,dict(id=f['id'],branch=branch,observation_binding=key,result=result,
                    result_sha256=B.digest(result),worker_cpu_seconds=seconds,actual_calls=calls,status='solved'))
                if done%20==0 or done==476:print('PNP',done,476,flush=True)
                budget.check('cpu_solver_analysis',time.monotonic()-begin)
        old=list(B.rows(prior/'run_v2/PREDICTIONS.jsonl.gz'));output=[];binding_rows=[]
        for i,(f,row) in enumerate(zip(frames,old)):
            if f['id']!=row['id']:raise ValueError('Original order changed')
            row=deepcopy(row);bindings={}
            for branch in BRANCHES:
                base,after=PAIR[branch];r=B.read(folder/'run/poses'/f'{i:03d}_{branch}.json')['result']
                row['predictions'][base]=B._baseline(f['arms'][branch])
                row['predictions'][after]=dict(actual_pose=r['actual_pose'],accepted=r['accepted'],fallback=r['fallback'],
                    status='solved',reason=r['solution'].get('fallback_reason'))
                bindings[branch]={k:dict(path=str(p),sha256=RT.sha256(p)) for k,p in (
                    ('observation',folder/'run/observations'/f'{i:03d}_{branch}.json'),
                    ('solution',folder/'run/poses'/f'{i:03d}_{branch}.json'))}
            row['predictions']={a:row['predictions'][a] for a in ARMS};output.append(row)
            binding_rows.append(dict(id=f['id'],branches=bindings))
        B.save_rows(folder/'PREDICTIONS.jsonl.gz',output)
        B.write(folder/'INDEPENDENT_BINDINGS.json',dict(records=binding_rows,ordered_ids=protocol['ordered_ids']))
        B.write(folder/'PREDICTION_SEAL.json',dict(rows=238,arm_rows=1904,arms=list(ARMS),ordered_ids=protocol['ordered_ids'],
            predictions_sha256=RT.sha256(folder/'PREDICTIONS.jsonl.gz'),protocol_sha256=RT.sha256(folder/'PROTOCOL.json'),
            observation_seal_sha256=RT.sha256(folder/'run/OBSERVATION_SEAL.json'),
            independent_bindings_sha256=RT.sha256(folder/'INDEPENDENT_BINDINGS.json'),
            original_four_arms_exactly_preserved=True,GT_inputs=False,before_new_scoring=True,all_rows_kept=True))
        status='complete'
    except Exception:
        (folder/'SOLVE_TECHNICAL_FAILURE.txt').write_text(traceback.format_exc());raise
    finally:budget.record('single_component_BI_solve','cpu_solver_analysis',time.monotonic()-begin,
        status=status,new_pose_solves=done,worker_seconds=cpu,workers=3)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','observe','solve'))
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    if a.stage=='freeze':freeze(a.folder,a.prior)
    elif a.stage=='observe':observe(a.folder,a.source_root)
    else:solve(a.folder,a.prior)
