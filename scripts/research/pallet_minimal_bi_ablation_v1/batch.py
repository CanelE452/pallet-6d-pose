"""Frozen predictor-only batch, parity gate and coordinate-only diagnostics.

Numeric references are deliberately not imported. Existing coarse inputs are
explicit; every image-dependent observation belongs to its own initialization.
Outputs are append-only per stage and may be resumed only at the same binding.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import time
import traceback

import cv2
import numpy as np
from . import runtime as RT
from .instrumentation import instrument

ARMS = ('R0', 'R1', 'N0', 'N1')
BRANCHES = ('RAW_YOLO', 'N3_SUBPIX')
DIAGNOSTIC_NAMES = ('R1_PRECORRECTION_XY_FIXED_OBSERVATION',
                    'N1_PRECORRECTION_XY_FIXED_OBSERVATION')
LEGACY = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
BOUNDARY = LEGACY/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
REFERENCE = BOUNDARY/'new_gain10_paired_BI_v2'
PARITY_INPUTS = BOUNDARY/'raw_ns_BI_gain10_input_v1/INPUTS.json'
MODEL_PATHS = dict(
    state_descriptor=LEGACY/'nonsevere_v1/statewise_joint_ot_20261014_v2/state_visibility/bounded_ownership_spatial_v3/MODEL.json',
    mask_descriptor=LEGACY/'nonsevere_v1/statewise_joint_ot_20261014_v2/state_visibility/MODEL.json',
    boundary_config=LEGACY/'nonsevere_v1/BOUNDARY_OBSERVATION_CONFIG.json')


def read(path): return json.loads(Path(path).read_text())
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def write(path, value, *, same_ok=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(RT._json(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if same_ok and path.read_text() == text: return
        raise FileExistsError('Preserve existing output: '+str(path))
    path.write_text(text)
def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream: yield json.loads(line)
def save_rows(path, values):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    with gzip.open(path, 'wt') as stream:
        for value in values:
            stream.write(json.dumps(RT._json(value), ensure_ascii=False, allow_nan=False)+'\n')


class Budget:
    """Process stage elapsed and separate summed worker CPU-time evidence."""
    def __init__(self, folder): self.path = Path(folder)/'RESOURCE_LEDGER.jsonl'
    def entries(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
    def check(self, kind, active_elapsed=0.):
        elapsed = sum(r['elapsed_seconds'] for r in self.entries() if r['kind'] == kind) + active_elapsed
        cap = {'gpu_process': 3600., 'cpu_solver_analysis': 7200.}[kind]
        if elapsed >= cap: raise RuntimeError('RESOURCE_CAP '+kind)
    def record(self, stage, kind, elapsed, *, worker_seconds=None, status='complete', **extra):
        entry = dict(stage=stage, kind=kind, elapsed_seconds=float(elapsed),
            summed_worker_cpu_seconds=worker_seconds, status=status,
            UTC=datetime.now(timezone.utc).isoformat(), **extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream: stream.write(json.dumps(entry, allow_nan=False)+'\n')


def components(root, device):
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return RT.load_components(asset_root=root, device=device,
        **{k:Path(root)/p for k,p in MODEL_PATHS.items()})


def pilot_ids(frames):
    """Round robin sorted sessions, within each session SHA256(ID) order."""
    grouped = {}
    for f in frames:
        if f['partition'] == 'CAL': grouped.setdefault(f['session'], []).append(f['id'])
    for values in grouped.values(): values.sort(key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    chosen=[]
    while len(chosen)<6:
        progressed=False
        for name in sorted(grouped):
            if grouped[name] and len(chosen)<6:
                chosen.append(grouped[name].pop(0)); progressed=True
        if not progressed: break
    if len(chosen)!=6: raise ValueError('Six fixed CAL images required')
    return chosen


def compare(a,b,path='',*,atol=1e-9,rtol=0.,subset=False):
    """Recursive audit: strings/booleans/IDs exact; declared floats tolerant."""
    if isinstance(a,dict):
        if not isinstance(b,dict) or (not subset and set(a)!=set(b)) or not set(a)<=set(b):
            raise AssertionError('Different keys: '+path+' '+str(set(a)^set(b)))
        for k,v in a.items(): compare(v,b[k],path+'/'+k,atol=atol,rtol=rtol,subset=subset)
    elif isinstance(a,(list,tuple)):
        if not isinstance(b,(list,tuple)) or len(a)!=len(b): raise AssertionError('Different lengths: '+path)
        for i,(u,v) in enumerate(zip(a,b)): compare(u,v,path+'/'+str(i),atol=atol,rtol=rtol,subset=subset)
    elif type(a) is float:
        if not isinstance(b,(float,int)) or not np.isclose(a,b,atol=atol,rtol=rtol,equal_nan=False):
            raise AssertionError(f'Float mismatch {path}: {a!r} != {b!r}')
    elif type(a) != type(b) or a != b:
        raise AssertionError(f'Exact mismatch {path}: {a!r} != {b!r}')


def _observed(root, frame, branch, model):
    image=cv2.imread(str(Path(root)/frame['image']))
    if image is None or list(image.shape[:2])!=frame['raw_hw']:
        raise ValueError('Original image unreadable/canvas mismatch')
    if RT.sha256(Path(root)/frame['image'])!=frame['image_sha256']:
        raise ValueError('Original RGB byte binding changed')
    source=frame['arms'][branch]
    return RT.observe(image,source['native_points'],frame['K'],source['coarse_pose'],
        components=model,prediction_source=branch,record_id=frame['id'])


def parity(root, folder, *, device='cuda'):
    """Run all20 actual observations and poses against immutable source10."""
    folder=Path(folder); budget=Budget(folder.parent); budget.check('gpu_process')
    if folder.exists(): raise FileExistsError('A parity attempt is immutable')
    folder.mkdir(parents=True)
    original=read(Path(root)/PARITY_INPUTS)
    protocol=read(Path(root)/REFERENCE/'PROTOCOL.json')
    if RT.sha256(Path(root)/PARITY_INPUTS)!=protocol['inputs_sha256']:
        raise ValueError('Original fixed gain10 input changed')
    start=time.monotonic(); status='technical_incomplete'; completed=[]
    try:
        model=components(root,device)
        for i,f in enumerate(original['frames']):
            saved=read(Path(root)/REFERENCE/'frames'/f'{i:02d}_PREDICTION.json')
            if saved['id']!=f['id']: raise ValueError('Gain10 order changed')
            for branch in BRANCHES:
                begin=time.monotonic(); result=_observed(root,f,branch,model)
                old=saved['branches'][branch]
                # O/T domains and U display are the only excluded payloads.
                for key in ('schema','states','initial_geometry','state_observation','mask_audit',
                            'visible_correction','frozen_visible_measurements',
                            'original_raw_points','original_coarse_pose'):
                    compare(old['observation'][key], result['observation'][key],key,
                        atol=1e-8,rtol=1e-8)
                compare(old['base_visible_correction'],result['base_visible_correction'],'base',atol=1e-8,rtol=1e-8)
                compare(old['masks_provenance'],result['masks_provenance'],'masks_provenance',atol=1e-8,rtol=1e-8)
                with np.load(Path(root)/old['mask_binding']['path'],allow_pickle=False) as masks:
                    for key in ('pallet_mask','state_pallet_mask'):
                        if not np.array_equal(masks[key],result['masks'][key]):
                            raise AssertionError('Mask not identical: '+key)
                observed_elapsed=time.monotonic()-begin
                pose_start=time.monotonic()
                pose=RT.solve(result['observation'],f['arms'][branch]['coarse_pose'],f['K'],f['raw_hw'])
                compare(old['pose_result'],pose,'pose_result',atol=1e-9,rtol=0,subset=True)
                pose_elapsed=time.monotonic()-pose_start
                masks=result.pop('masks')
                np.savez_compressed(folder/f'{i:02d}_{branch}_MASKS.npz',**masks)
                write(folder/f'{i:02d}_{branch}.json',dict(id=f['id'],branch=branch,
                    observation_result=result,pose_result=pose))
                completed.append(dict(id=f['id'],branch=branch,observation_exact_contract=True,
                    mask_arrays_bitwise_equal=True,pose_within_locked_tolerance=True,
                    observation_seconds=observed_elapsed,pose_seconds=pose_elapsed,
                    accepted=pose['accepted'],fallback=pose['fallback']))
                print('PARITY',len(completed),'/20',branch,f['id'],flush=True)
        status='complete'
        write(folder/'PARITY_CHECK.json',dict(schema='MINIMAL_BI_GAIN10_PARITY_V1',status=status,
            rows=10,observations=20,pose_solves=20,policy=RT.PARITY_POLICY,
            audit_scalar_tolerance=dict(atol=1e-8,rtol=1e-8),
            omitted_production_payloads=['ot_domains','u_region_display'],
            measurements_and_final_projections_separate=True,records=completed,GT_inputs=False,
            reference_protocol_sha256=RT.sha256(Path(root)/REFERENCE/'PROTOCOL.json')))
    except Exception:
        (folder/'TECHNICAL_FAILURE.txt').write_text(traceback.format_exc()); raise
    finally:
        budget.record('gain10_parity','gpu_process',time.monotonic()-start,status=status,
            observations_done=len(completed),pose_seconds_included=sum(x['pose_seconds'] for x in completed))


def _task(task):
    begin=time.process_time()
    observation,source,K,hw=task
    with instrument(None) as meter:
        result=RT.solve(observation,source['coarse_pose'],K,hw)
        actual_calls=meter.snapshot()
    return result,time.process_time()-begin,actual_calls


def _input(folder, config=None):
    folder=Path(folder); seal=read(folder/'INPUT_SEAL.json'); data=read(folder/'INPUTS.json')
    if RT.sha256(folder/'INPUTS.json')!=seal['inputs_sha256']:
        raise ValueError('Input seal changed')
    if config is not None and (config['inputs_sha256']!=seal['inputs_sha256'] or
            config['input_seal_sha256']!=RT.sha256(folder/'INPUT_SEAL.json')):
        raise ValueError('Inputs differ from the frozen protocol')
    if len(data['frames'])!=238 or len(set(f['id'] for f in data['frames']))!=238:
        raise ValueError('Fixed238 population required')
    return data,seal


def freeze(root,input_dir,folder,parity_dir):
    folder=Path(folder); data,seal=_input(input_dir)
    parity_record=read(Path(parity_dir)/'PARITY_CHECK.json')
    if parity_record['status']!='complete' or len(parity_record['records'])!=20:
        raise ValueError('Full20 parity is required before main freeze')
    regression_path=folder.parent/'REGRESSION_SUCCESS_CHECK.json'
    regression=read(regression_path)
    if regression['status']!='PASS' or regression['failure_count']!=0 or regression['inclusive_counts']!=dict(R0=0,R1=7,N0=10,N1=10):
        raise ValueError('Unchanged gain10 final-success regression gate required')
    source={str(p.relative_to(root)):RT.sha256(p) for p in sorted((Path(root)/'scripts/research/pallet_minimal_bi_ablation_v1').glob('*.py'))}
    old=read(Path(root)/REFERENCE/'PROTOCOL.json')['source_sha256']
    for path,h in old.items():
        if RT.sha256(Path(root)/path)!=h: raise ValueError('Bound existing source changed: '+path)
    config=dict(schema='MINIMAL_BI_SINGLE_CONFIG_PROTOCOL_V1',
        UTC=datetime.now(timezone.utc).isoformat(),arms=list(ARMS),ordered_ids=seal['ordered_ids'],
        rows=238,arm_rows=952,populations=dict(PRIMARY192=192,TEST=49,REMAINDER=143,CAL=46),
        source_sha256=source,legacy_source_sha256=old,input_seal_sha256=RT.sha256(Path(input_dir)/'INPUT_SEAL.json'),
        inputs_sha256=seal['inputs_sha256'],parity_sha256=RT.sha256(Path(parity_dir)/'PARITY_CHECK.json'),
        regression_success_sha256=RT.sha256(regression_path),
        model_descriptors={k:dict(path=str(p),sha256=RT.sha256(Path(root)/p)) for k,p in MODEL_PATHS.items()},
        solver_configuration=RT.SOLVER_CONFIG,GT_inputs=False,training_updates=0,new_learning_seeds=0,
        new_initial_detector_forwards=0,mask_and_state_own_initialization=True,
        success='translation_cm<=5 AND rotation_deg<=5',historical_success='translation_cm<5 AND rotation_deg<5',
        rotation='unchanged physical registration and C2 symmetry',translation='same registered model center cm',
        bootstrap=dict(replicates=10000,seed=20260917,unit='original recording cluster',estimand='pooled frame success difference',percentiles=[2.5,97.5]),
        primary_contrasts=[['R1','R0'],['N1','N0'],['R1','N1'],['R1','N0']],
        p_values='NA: no verified existing success-test implementation; no CI-inverted p',
        pilot_ids=pilot_ids(data['frames']),pilot_selection='CAL session round-robin, SHA256 ID order, no scores/GT',
        coordinate_diagnostics=list(DIAGNOSTIC_NAMES),coordinate_diagnostic_primary_only=True,
        coordinate_diagnostic_frozen='own post-observation states/IDs/support/covariance/solver; replace only xy with own native input',
        timing=dict(panel='same six CAL pilot IDs',batch=1,warmups_per_path=2,repeats_per_image=5,
            balanced_order_seed=20260917,boundary='in-memory RGB + own native points + own F to final pose',
            file_decode=False,model_load=False,full_inference_time='NA unless actual complete forward is wired and measured'),
        caps=dict(gpu_process_elapsed_seconds=3600,cpu_solver_analysis_elapsed_seconds=7200,pose_workers=3,BLAS_threads=1),
        no_pose_rows='retained as failure; no fake algorithm fallback for technical exception',
        defaults_or_e_changed=False,paper_changed=False,development_reuse=True)
    write(folder/'PROTOCOL.json',config)
    return config


def check_binding(root,folder):
    config=read(Path(folder)/'PROTOCOL.json')
    for path,h in {**config['source_sha256'],**config['legacy_source_sha256']}.items():
        if RT.sha256(Path(root)/path)!=h: raise ValueError('Source changed after freeze: '+path)
    for value in config['model_descriptors'].values():
        if RT.sha256(Path(root)/value['path'])!=value['sha256']: raise ValueError('Model descriptor changed')
    return config


def observe(root,input_dir,folder,*,device='cuda',pilot=False):
    folder=Path(folder); config=check_binding(root,folder); data,_=_input(input_dir,config)
    frames=[(i,f) for i,f in enumerate(data['frames']) if not pilot or f['id'] in config['pilot_ids']]
    budget=Budget(folder.parent);budget.check('gpu_process');start=time.monotonic();done=0;status='technical_incomplete'
    try:
        model=components(root,device)
        for i,f in frames:
            for branch in BRANCHES:
                target=folder/'observations'/f'{i:03d}_{branch}.json'
                source=f['arms'][branch]
                key=digest(dict(frame=f,branch=branch,protocol=RT.sha256(folder/'PROTOCOL.json')))
                if target.exists():
                    cached=read(target)
                    if cached['binding']!=key or cached['result_sha256']!=digest(cached['result']):
                        raise ValueError('Observation cache binding/content changed')
                    mask=cached['result'].get('mask_binding')
                    if mask and RT.sha256(folder/mask['path'])!=mask['sha256']:
                        raise ValueError('Own cached mask bytes changed')
                    continue
                tick=time.monotonic()
                actual_calls=None
                if not source['coarse_pose'].get('available',True):
                    result=dict(status='no_initial_pose',reason='Own initial F did not produce a pose')
                else:
                    with instrument(model) as meter:
                        result=_observed(root,f,branch,model)
                        actual_calls=meter.snapshot()
                    masks=result.pop('masks')
                    mpath=folder/'masks'/f'{i:03d}_{branch}.npz';mpath.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(mpath,**masks)
                    result['mask_binding']=dict(path=str(mpath.relative_to(folder)),sha256=RT.sha256(mpath))
                    result['status']='observed'
                result['elapsed_seconds']=time.monotonic()-tick
                result['actual_calls']=actual_calls
                write(target,dict(id=f['id'],branch=branch,binding=key,result=result,result_sha256=digest(result)))
                done+=1
                print('OBS',done,'/',len(frames)*2,branch,f['id'],result['status'],flush=True)
                budget.check('gpu_process',time.monotonic()-start)
        status='complete'
        if not pilot:
            records=[dict(id=f['id'],branches={b:read(folder/'observations'/f'{i:03d}_{b}.json')['result'] for b in BRANCHES}) for i,f in enumerate(data['frames'])]
            save_rows(folder/'OBSERVATIONS.jsonl.gz',records)
            write(folder/'OBSERVATION_SEAL.json',dict(rows=238,branches=476,
                observations_sha256=RT.sha256(folder/'OBSERVATIONS.jsonl.gz'),protocol_sha256=RT.sha256(folder/'PROTOCOL.json'),
                all_own_image_observations=True,GT_inputs=False))
    except Exception:
        (folder/('PILOT_OBSERVE_TECHNICAL_FAILURE.txt' if pilot else 'OBSERVE_TECHNICAL_FAILURE.txt')).write_text(traceback.format_exc()); raise
    finally:
        budget.record('CAL_pilot_observe' if pilot else 'full_observe','gpu_process',time.monotonic()-start,
            status=status,new_observations=done)


def _baseline(source):
    pose=deepcopy(source['coarse_pose']);pose.setdefault('available',True)
    return dict(actual_pose=pose,accepted=None,fallback=None,
        status='cached_initial_F' if pose['available'] else 'no_initial_pose',
        reason='Unchanged own common F; no original numerical-acceptance flag available')


def solve(root,input_dir,folder,*,workers=3,pilot=False,coordinate_diagnostic=False):
    if workers not in (1,2,3): raise ValueError('Maximum three CPU workers')
    folder=Path(folder);config=check_binding(root,folder);data,_=_input(input_dir,config)
    frames=[(i,f) for i,f in enumerate(data['frames']) if (not pilot or f['id'] in config['pilot_ids']) and (not coordinate_diagnostic or f['partition']!='CAL')]
    sub='coordinate_diagnostics' if coordinate_diagnostic else 'poses'
    budget=Budget(folder.parent);budget.check('cpu_solver_analysis');start=time.monotonic();worker_seconds=0.;done=0;status='technical_incomplete'
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            tasks={}
            for i,f in frames:
                for b in BRANCHES:
                    target=folder/sub/f'{i:03d}_{b}.json'
                    record=read(folder/'observations'/f'{i:03d}_{b}.json');result=record['result']
                    if record['result_sha256']!=digest(result):raise ValueError('Own observation content changed')
                    if target.exists():
                        cached=read(target)
                        if cached.get('observation_binding')!=record['binding'] or cached.get('coordinate_replacement')!=coordinate_diagnostic:
                            raise ValueError('Pose cache binding changed')
                        if 'result' in cached and cached['result_sha256']!=digest(cached['result']):
                            raise ValueError('Pose cache content changed')
                        continue
                    if result['status']=='no_initial_pose':
                        write(target,dict(id=f['id'],branch=b,observation_binding=record['binding'],
                            coordinate_replacement=coordinate_diagnostic,status='no_initial_pose',
                            actual_pose=dict(available=False),accepted=False,fallback=False))
                        continue
                    obs=deepcopy(result['observation'])
                    if coordinate_diagnostic:
                        for m in obs['frozen_visible_measurements']:
                            k=m['native_corner'];xy=np.asarray(f['arms'][b]['native_points'][k],float)
                            cov=np.asarray(m['covariance_px2'],float)
                            if m['state'] not in ('B','I') or xy.shape!=(2,) or not np.isfinite(xy).all() or cov.shape!=(2,2) or np.linalg.eigvalsh(cov).min()<=0:
                                raise ValueError('Coordinate diagnostic cannot replace incompatible observation')
                            m['xy']=xy.tolist()
                    future=pool.submit(_task,(obs,f['arms'][b],f['K'],f['raw_hw']))
                    tasks[future]=(target,i,f,b,record['binding'])
            for future in as_completed(tasks):
                target,i,f,b,key=tasks[future];result,cpu_seconds,actual_calls=future.result();worker_seconds+=cpu_seconds
                write(target,dict(id=f['id'],branch=b,observation_binding=key,status='solved',
                    result=result,result_sha256=digest(result),actual_calls=actual_calls,
                    worker_cpu_seconds=cpu_seconds,coordinate_replacement=coordinate_diagnostic))
                done+=1
                print('PNP_DIAG' if coordinate_diagnostic else 'PNP',done,'/',len(tasks),b,f['id'],result['accepted'],flush=True)
                budget.check('cpu_solver_analysis',time.monotonic()-start)
        status='complete'
        if not pilot:
            output=[]
            for i,f in frames:
                row={k:deepcopy(f[k]) for k in ('id','partition','session','cluster_id')};row['predictions']={}
                row['predictions']['R0']=_baseline(f['arms']['RAW_YOLO']);row['predictions']['N0']=_baseline(f['arms']['N3_SUBPIX'])
                for branch,arm in (('RAW_YOLO','R1'),('N3_SUBPIX','N1')):
                    p=read(folder/sub/f'{i:03d}_{branch}.json')
                    r=p.get('result',p)
                    row['predictions'][arm]=dict(actual_pose=r['actual_pose'],accepted=r['accepted'],fallback=r['fallback'],
                        status=p['status'],reason=r.get('solution',{}).get('fallback_reason'))
                row['predictions']={a:row['predictions'][a] for a in ARMS};output.append(row)
            target=folder/('COORDINATE_DIAGNOSTIC_PREDICTIONS.jsonl.gz' if coordinate_diagnostic else 'PREDICTIONS.jsonl.gz')
            save_rows(target,output)
            if not coordinate_diagnostic:
                write(folder/'PREDICTION_SEAL.json',dict(rows=238,arm_rows=952,arms=list(ARMS),ordered_ids=config['ordered_ids'],
                    protocol_sha256=RT.sha256(folder/'PROTOCOL.json'),predictions_sha256=RT.sha256(target),
                    input_seal_sha256=config['input_seal_sha256'],inputs_sha256=config['inputs_sha256'],
                    observation_seal_sha256=RT.sha256(folder/'OBSERVATION_SEAL.json'),before_scoring=True,GT_inputs=False,
                    no_population_drop=True,all_failures_preserved=True))
    except Exception:
        (folder/(sub.upper()+'_TECHNICAL_FAILURE.txt')).write_text(traceback.format_exc()); raise
    finally:
        budget.record(('coordinate_diagnostics' if coordinate_diagnostic else 'CAL_pilot_solve' if pilot else 'full_solve'),
            'cpu_solver_analysis',time.monotonic()-start,worker_seconds=worker_seconds,status=status,new_pose_solves=done,workers=workers)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('parity','freeze','observe','solve','coordinate'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-dir',type=Path);p.add_argument('--parity-dir',type=Path)
    p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.stage=='parity':parity(a.root,a.output,device=a.device)
    elif a.stage=='freeze':freeze(a.root,a.input_dir,a.output,a.parity_dir)
    elif a.stage=='observe':observe(a.root,a.input_dir,a.output,device=a.device,pilot=a.pilot)
    else:solve(a.root,a.input_dir,a.output,workers=a.workers,pilot=a.pilot,coordinate_diagnostic=a.stage=='coordinate')

if __name__=='__main__':main()
