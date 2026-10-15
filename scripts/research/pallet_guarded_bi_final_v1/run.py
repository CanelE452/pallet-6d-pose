"""Bounded follow-up: own E observations and two frozen acceptance rules.

No labels are imported. RAW/N3 observations are reused from their own sealed
branches. E has its own registered geometry, states, mask prompts and B/I.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1.instrumentation import instrument

ARMS=('R0','R1','RA','RB','N0','N1','NA','NB','E0','E1','EA','EB')

def verify(folder):
    p=B.read(Path(folder)/'METHOD_PROTOCOL.json')
    code=Path(__file__).resolve().parents[3]
    for name,h in p['inference_source_sha256'].items():
        if RT.sha256(code/name)!=h: raise ValueError('Inference source changed: '+name)
    for name,h in p['immutable_inputs_sha256'].items():
        if RT.sha256(name)!=h: raise ValueError('Input changed: '+name)
    return p

def observe(source_root,folder):
    folder=Path(folder); p=verify(folder); data=B.read(folder/'e_inputs/INPUTS.json')
    budget=B.Budget(folder); budget.check('gpu_process');begin=time.monotonic();done=0;status='incomplete'
    try:
        model=B.components(source_root,'cuda')
        for i,f in enumerate(data['frames']):
            target=folder/'e_run/observations'/f'{i:03d}_E_FROZEN.json'
            key=B.digest(dict(frame=f,protocol=RT.sha256(folder/'METHOD_PROTOCOL.json')))
            if target.exists():
                old=B.read(target)
                if old['binding']!=key or old['result_sha256']!=B.digest(old['result']):
                    raise ValueError('Own E observation cache changed')
                m=old['result']['mask_binding']
                if RT.sha256(folder/'e_run'/m['path'])!=m['sha256']: raise ValueError('Own E masks changed')
                continue
            tick=time.monotonic()
            with instrument(model) as meter:
                result=B._observed(source_root,f,'E_FROZEN',model)
                calls=meter.snapshot()
            masks=result.pop('masks');mpath=folder/'e_run/masks'/f'{i:03d}_E_FROZEN.npz'
            mpath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mpath,**masks)
            result.update(status='observed',actual_calls=calls,elapsed_seconds=time.monotonic()-tick,
                mask_binding=dict(path=str(mpath.relative_to(folder/'e_run')),sha256=RT.sha256(mpath)))
            B.write(target,dict(id=f['id'],branch='E_FROZEN',binding=key,result=result,result_sha256=B.digest(result)))
            done+=1
            if done%10==0 or done==238: print('E_OBSERVE',done,238,flush=True)
            budget.check('gpu_process',time.monotonic()-begin)
        paths=sorted((folder/'e_run/observations').glob('*.json'))
        if len(paths)!=238: raise ValueError('All238 own E observations required')
        B.write(folder/'e_run/OBSERVATION_SEAL.json',dict(rows=238,GT_inputs=False,
            protocol_sha256=RT.sha256(folder/'METHOD_PROTOCOL.json'),
            observation_sha256={x.name:RT.sha256(x) for x in paths},
            own_E_states_masks_BI=True,RAW_N3_observations_borrowed=False),same_ok=True)
        status='complete'
    except Exception:
        (folder/'E_OBSERVE_TECHNICAL_FAILURE.txt').write_text(traceback.format_exc());raise
    finally:
        budget.record('E_own_observations','gpu_process',time.monotonic()-begin,status=status,new_observations=done)

def _task(task):
    result,seconds,calls=B._task(task)
    result.update(own_input_F_coarse_start=False,own_E_coarse_start=True,
        own_initializer='E_FROZEN',additional_cached_pose_bank_starts=0)
    # This is one explicitly named E initializer, never an unreported E bank.
    result['E_or_cached_pose_bank_starts']=1
    return result,seconds,calls

def solve(folder):
    folder=Path(folder);verify(folder);data=B.read(folder/'e_inputs/INPUTS.json')
    budget=B.Budget(folder);begin=time.monotonic();cpu=0.;done=0;status='incomplete'
    try:
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
            tasks={}
            for i,f in enumerate(data['frames']):
                obs=B.read(folder/'e_run/observations'/f'{i:03d}_E_FROZEN.json')
                target=folder/'e_run/poses'/f'{i:03d}_E_FROZEN.json'
                if obs['result_sha256']!=B.digest(obs['result']): raise ValueError('E observation changed')
                if target.exists():
                    old=B.read(target)
                    if old['observation_binding']!=obs['binding'] or old['result_sha256']!=B.digest(old['result']):
                        raise ValueError('E pose cache changed')
                    continue
                task=(obs['result']['observation'],f['arms']['E_FROZEN'],f['K'],f['raw_hw'])
                tasks[pool.submit(_task,task)]=(target,f,obs['binding'])
            for future in as_completed(tasks):
                target,f,key=tasks[future];result,seconds,calls=future.result();cpu+=seconds;done+=1
                B.write(target,dict(id=f['id'],branch='E_FROZEN',observation_binding=key,result=result,
                    result_sha256=B.digest(result),actual_calls=calls,worker_cpu_seconds=seconds,status='solved'))
                if done%10==0 or done==238: print('E_PNP',done,238,flush=True)
                budget.check('cpu_solver_analysis',time.monotonic()-begin)
        status='complete'
    except Exception:
        (folder/'E_SOLVE_TECHNICAL_FAILURE.txt').write_text(traceback.format_exc());raise
    finally:
        budget.record('E_own_BI_solve','cpu_solver_analysis',time.monotonic()-begin,
            worker_seconds=cpu,status=status,new_pose_solves=done,workers=3)

def guard(folder,prior):
    from .guard import apply_guard
    folder=Path(folder);prior=Path(prior);verify(folder)
    frames=B.read(prior/'inputs_v2/INPUTS.json')['frames']
    eframes=B.read(folder/'e_inputs/INPUTS.json')['frames']
    old=list(B.rows(prior/'run_v2/PREDICTIONS.jsonl.gz'));out=[];audit=[];bindings=[];begin=time.monotonic()
    for i,(f,ef,row) in enumerate(zip(frames,eframes,old)):
        if f['id']!=ef['id'] or f['id']!=row['id']: raise ValueError('Population order changed')
        row=deepcopy(row); bound=dict(id=f['id'],branches={})
        for prefix,branch,run,frame in (('R','RAW_YOLO',prior/'run_v2',f),
                ('N','N3_SUBPIX',prior/'run_v2',f),('E','E_FROZEN',folder/'e_run',ef)):
            o=B.read(run/'observations'/f'{i:03d}_{branch}.json')
            s=B.read(run/'poses'/f'{i:03d}_{branch}.json')
            if o['id']!=f['id'] or s['id']!=f['id'] or o['result_sha256']!=B.digest(o['result']) or s['result_sha256']!=B.digest(s['result']):
                raise ValueError('Own branch cache changed')
            coarse=frame['arms'][branch]['coarse_pose'];solved=s['result'];obs=o['result']['observation']
            bound['branches'][prefix]=dict(
                observation=dict(path=str(run/'observations'/f'{i:03d}_{branch}.json'),sha256=RT.sha256(run/'observations'/f'{i:03d}_{branch}.json')),
                solution=dict(path=str(run/'poses'/f'{i:03d}_{branch}.json'),sha256=RT.sha256(run/'poses'/f'{i:03d}_{branch}.json')))
            if prefix=='E':
                pose=deepcopy(coarse);pose.setdefault('available',True)
                row['predictions']['E0']=dict(actual_pose=pose,accepted=None,fallback=None,
                    status='unchanged_frozen_E',reason='Exact independently registered frozen E pose')
                row['predictions']['E1']=dict(actual_pose=solved['actual_pose'],accepted=solved['accepted'],
                    fallback=solved['fallback'],status='solved',reason=solved['solution'].get('fallback_reason'))
            for mode in ('A','B'):
                tick=time.perf_counter()
                result=apply_guard(obs,solved,coarse,frame['K'],mode)
                elapsed=time.perf_counter()-tick
                row['predictions'][prefix+mode]={k:result[k] for k in ('actual_pose','accepted','fallback','status','reason')}
                audit.append(dict(id=f['id'],branch=branch,arm=prefix+mode,
                    elapsed_seconds=elapsed,result=result,
                    observation_sha256=RT.sha256(run/'observations'/f'{i:03d}_{branch}.json'),
                    solve_sha256=RT.sha256(run/'poses'/f'{i:03d}_{branch}.json')))
        row['predictions']={a:row['predictions'][a] for a in ARMS};out.append(row);bindings.append(bound)
    if len(out)!=238 or len(audit)!=1428: raise ValueError('All original rows and rules required')
    B.save_rows(folder/'PREDICTIONS.jsonl.gz',out);B.save_rows(folder/'GUARD_AUDITS.jsonl.gz',audit)
    B.write(folder/'INDEPENDENT_BINDINGS.json',dict(ordered_ids=[f['id'] for f in frames],records=bindings))
    B.write(folder/'PREDICTION_SEAL.json',dict(rows=238,arm_rows=2856,arms=list(ARMS),
        ordered_ids=[f['id'] for f in frames],protocol_sha256=RT.sha256(folder/'METHOD_PROTOCOL.json'),
        predictions_sha256=RT.sha256(folder/'PREDICTIONS.jsonl.gz'),
        guard_audits_sha256=RT.sha256(folder/'GUARD_AUDITS.jsonl.gz'),
        independent_bindings_sha256=RT.sha256(folder/'INDEPENDENT_BINDINGS.json'),before_new_CAL_scoring=True,
        GT_inputs=False,all_rows_kept=True,prior_results_development_exposed=True))
    B.Budget(folder).record('two_guards_three_initializers','cpu_solver_analysis',time.monotonic()-begin,
        guard_calls=1428,new_pose_solves=0,status='complete')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('observe','solve','guard'))
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    if a.stage=='observe': observe(a.source_root,a.folder)
    elif a.stage=='solve': solve(a.folder)
    else: guard(a.folder,a.prior)
if __name__=='__main__':main()
