"""Sealed resumable D1 observation and original-e/feature batches.

The observation command uses one model job. CPU candidate construction is a
separate command so the controller can enforce the shared four-worker budget.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
from copy import deepcopy
import hashlib,os,sys,time,traceback
from pathlib import Path
import numpy as np

from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,rows,sha,write,save_rows,canonical,PredictionFirewall
from . import d1_producer as P


def source_bindings():
    here=Path(__file__).parent;root=here.parents[2]
    paths=[here/'d1_batch.py',here/'d1_producer.py',here/'pipeline_contract.py']
    # The original inference helpers are immutable but explicitly bound too;
    # this is source identity, not a claim of material-owner certification.
    paths+=list((here.parent/'pallet_visibility_curriculum_20261013_v1').glob('*.py'))
    for package,names in [('pallet_statewise_end_to_end_recovery_v1',('solver.py','owner_model.py','boundary_model.py','common.py')),
        ('pallet_statewise_correspondence_diagnosis_v1',('typed_replay.py',)),
        ('pallet_selection_score_diagnosis_v1',('fixed_cost.py',)),
        ('pallet_minimal_bi_ablation_v1',('runtime.py','instrumentation.py'))]:
        paths.extend(here.parent/package/name for name in names)
    return {str(p.relative_to(root)):sha(p)for p in sorted(set(paths))}


def firewall(old):
    old=Path(old)
    return PredictionFirewall([old/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz',
        old/'synthetic_v1/GEOMETRY_SIDETABLE.jsonl.gz',
        old.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json',
        old.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/oracle_v1/ORACLE_ROWS.jsonl.gz']).install()


def select_frames(inputs,domain,limit=None,technical=False):
    data=list(rows(inputs))
    if technical:
        # Before model results: first input-order CAL or synthetic rows only.
        if domain=='real':data=[f for f in data if f['partition']=='CAL']
        if limit is None or not 0<limit<=2:raise ValueError('Technical allowance is at most two/domain')
    if limit is not None:data=data[:limit]
    if not data:raise ValueError('No input frames')
    return data


class NeuralMeter:
    """Actual unchanged forwards; CUDA events around each top-level network."""
    def __init__(self):self.events=[];self.handles=[]
    def __enter__(self):
        import torch
        from scripts.research.pallet_minimal_bi_ablation_v1.instrumentation import CallMeter
        outer=self
        class Meter(CallMeter):
            def _forward_hook(self,model,key):
                identity=(id(model),key)
                if identity in self._hooked:return
                self._hooked.add(identity);starts=[]
                def pre(module,args):
                    self.counts[key]+=1
                    if torch.cuda.is_available():
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();starts.append((a,b))
                def post(module,args,result):
                    if starts:
                        a,b=starts.pop();b.record();outer.events.append((key,a,b))
                self._hooks.extend([model.register_forward_pre_hook(pre),model.register_forward_hook(post)])
        self.meter=Meter(block_unused=False).__enter__();return self
    def receipt(self):
        import torch
        if torch.cuda.is_available():torch.cuda.synchronize()
        timing=Counter()
        for key,a,b in self.events:timing[key]+=a.elapsed_time(b)/1000
        return dict(actual_calls=self.meter.snapshot(),CUDA_forward_seconds_by_module=dict(timing),
            CUDA_forward_seconds=sum(timing.values()),CUDA_event_scope='top-level Tiny/state-backbone/SAM-encoder/SAM-decoder actual forwards; CPU gaps excluded',
            peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated()if torch.cuda.is_available()else 0)
    def __exit__(self,*exc):return self.meter.__exit__(*exc)


@contextmanager
def numerical_meter():
    """Wrap actual original CPU calls without changing arguments/results."""
    import cv2,scipy.optimize
    counts=Counter();patches=[]
    def patch(obj,name,key):
        old=getattr(obj,name)
        def call(*args,**kwargs):counts[key]+=1;return old(*args,**kwargs)
        patches.append((obj,name,old));setattr(obj,name,call)
    for name in ('solvePnPGeneric','solvePnP','solvePnPRefineLM','solveP3P','cornerSubPix'):
        if hasattr(cv2,name):patch(cv2,name,name)
    original=scipy.optimize.minimize
    # All e modules are explicitly imported before installing this context.
    for name,module in list(sys.modules.items()):
        if module is not None and name.startswith('scripts.research.pallet_visibility_curriculum_20261013_v1'):
            if getattr(module,'minimize',None)is original:patch(module,'minimize','scipy_minimize_calls')
    try:yield counts
    finally:
        for obj,name,old in reversed(patches):setattr(obj,name,old)


def observe_batch(inputs,output,old_run,domain,*,device='cuda',limit=None,technical=False,GO=False):
    if not GO:raise ValueError('Explicit root execution GO required')
    started,cpu=time.monotonic(),time.process_time();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    guard=firewall(old_run);frames=select_frames(inputs,domain,limit,technical)
    lock=dict(schema='D1_OBSERVATION_LOCK_V1',inputs_sha256=sha(inputs),source_sha256=source_bindings(),
        ordered_ids=[f['id']for f in frames],domain=domain,device=device,technical=technical,GT_inputs=False)
    if (out/'OBSERVATION_LOCK.json').exists():
        if read(out/'OBSERVATION_LOCK.json')!=lock:raise ValueError('Observation source/input lock changed')
    else:write(out/'OBSERVATION_LOCK.json',lock)
    pending=[];done={};new=0
    for i,f in enumerate(frames):
        folder=out/'frames'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True)
        seal=folder/'OBSERVATION_SEAL.json'
        if seal.exists():
            s=read(seal)
            if s['id']!=f['id']or s['source_sha256']!=source_bindings()or sha(folder/'OBSERVATION.jsonl.gz')!=s['observation_sha256']or sha(folder/'MASK.npz')!=s['mask_sha256']:
                raise ValueError('Completed observation cache changed')
            done[f['id']]=s
        else:pending.append((i,f,folder))
    bundle=P.load_models(device)if pending else None
    for i,f,folder in pending:
        tick,tcpu=time.monotonic(),time.process_time()
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            with NeuralMeter()as meter:
                row,arrays=P.observe(f,bundle);resource=meter.receipt()
            np.savez_compressed(folder/'MASK.npz',**arrays);save_rows(folder/'OBSERVATION.jsonl.gz',[row])
            s=dict(id=f['id'],complete=True,observation_sha256=sha(folder/'OBSERVATION.jsonl.gz'),
                mask_sha256=sha(folder/'MASK.npz'),source_sha256=source_bindings(),input_row_sha256=hashlib.sha256(canonical(f).encode()).hexdigest(),
                actual_state_counts=dict(Counter(row['states'])),
                actual_state_classifier_executed=resource['actual_calls']['state_classifier_head_calls']>0,
                production_state_classifier_stage_invoked=True,
                resource=dict(**resource,wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-tcpu),
                reference_reads=0,e0_executed=False)
            write(folder/'OBSERVATION_SEAL.json',s);done[f['id']]=s;new+=1
        except Exception as exc:
            write(folder/'OBSERVATION_FAILURE.json',dict(id=f['id'],status='FAILED_TECHNICAL',error=repr(exc),traceback=traceback.format_exc()))
            raise
        print(canonical(dict(stage='D1_OBSERVE',done=len(done),total=len(frames),id=f['id'],wall_seconds=time.monotonic()-started)),flush=True)
    seal=dict(schema='D1_ALL_OBSERVATIONS_SEAL_V1',complete=True,rows=len(frames),new_observations=new,
        ordered_ids=[f['id']for f in frames],input_sha256=sha(inputs),source_sha256=source_bindings(),
        model_bindings=bundle['bindings']if bundle else read(out/'OBSERVATION_SEAL.json')['model_bindings'],
        frame_seals_sha256={f'{i:04d}':sha(out/'frames'/f'{i:04d}'/'OBSERVATION_SEAL.json')for i in range(len(frames))},
        state_counts=dict(sum((Counter(s['actual_state_counts'])for s in done.values()),Counter())),
        GPU_forward_seconds=sum(s['resource']['CUDA_forward_seconds']for s in done.values()),
        actual_calls=dict(sum((Counter({k:v for k,v in s['resource']['actual_calls'].items()if isinstance(v,int)})for s in done.values()),Counter())),
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,reference_firewall=guard.receipt(),all_failures_retained=True)
    write(out/'OBSERVATION_SEAL.json',seal,replace=(out/'OBSERVATION_SEAL.json').exists());return seal


def _preload_solver():
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_boundary_u_solver_v4_6
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_global_outline_solver_v4_10
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_outline_solver_v4_13
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluate_visibility_pose_v4_8


def _cpu_models():
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import observe_remainder_rgb_v4_14 as O
    from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_uncertainty_calibration_v4 import load_uncertainty_model
    from scripts.research.pallet_visibility_curriculum_20261013_v1.truncation_interval_v4_2 import load_truncation_interval_model
    return read(O.LEGACY_PROTOCOL)['solver_configuration'],load_uncertainty_model(O.UNCERTAINTY_MODEL),load_truncation_interval_model(O.INTERVAL_MODEL)


def feature_frame(frame,result,mask,old_config):
    from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S,boundary_model as B
    import cv2
    inp=dict(frame,packet=result['packet'],families=result['families'],fallback=frame['initial_pose'])
    scale=max(float(np.linalg.norm(np.ptp(f['native_xyz'],axis=0)))for f in result['families'].values())
    allc=result['candidates'];admitted=[];rejected=[]
    for c in allc:
        R=np.asarray(c['R_physical']);t=np.asarray(c['centroid']);proper=bool(np.isfinite(R).all()and np.isfinite(t).all()and np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)and abs(np.linalg.det(R)-1)<1e-6)
        f=result['families']['ORIGINAL_WD'];native=R@np.asarray(f['physical_registration_Q']).T
        depth=((np.asarray(f['native_xyz'])-np.asarray(f['native_xyz']).mean(0))@native.T+t)[:,2]
        if proper and depth.min()>1e-5:admitted.append(c)
        else:rejected.append(dict(candidate_id=c['candidate_id'],reason='INVALID_OR_NONPOSITIVE_DEPTH'))
    bank=S.unique_poses(admitted,scale)
    bundle=B.build_boundary_bundle(cv2.imread(frame['image']),mask,result['packet'],old_config.get('boundary_config'))
    models=S.compile_frame(inp,old_config,bundle=bundle)
    scores=[S.candidate_scores(c,models)for c in bank];features=S.features_for_bank(inp,models,bank,scores)
    for feature,candidate in zip(features,bank):
        aliases=[];all_sources=[]
        for source in admitted:
            if np.max(np.abs(np.asarray(source['R_physical'])-candidate['R_physical']))<=1e-7 and np.max(np.abs(np.asarray(source['centroid'])-candidate['centroid']))<=1e-7*scale:
                all_sources.append(source['candidate_id'])
                if source['source']in('N0','E0'):aliases.append(source['source'])
        feature['source_baseline_aliases']=sorted(set(aliases));feature['source_aliases']=sorted(set(all_sources));feature['object_scale_m']=scale
        if aliases:feature['eligible']=True
    spec=result['packet']['solver_input'];sigmas=[float(np.sqrt(np.trace(m['fit_covariance_px2'])/2))for m in spec['visible']]
    return dict(id=frame['id'],split=frame['split'],partition=frame['partition'],session=frame['session'],cluster_id=frame['cluster_id'],
        candidates=features,fallback=frame['initial_pose'],e0_regenerated=result['e0_regenerated'],object_scale_m=scale,
        feature_names=list(S.FEATURE_NAMES),transformation='signed_log1p31_plus31_explicit_validity_bits',
        state_counts=dict(Counter(spec['states'])),
        observation_coverage=dict(full_or_archived_BI=len(spec['visible']),partial=len(spec['boundary_observations']),
            O_T_domains=len(spec['domains']),U_regions=len(spec['u_regions']),
            B_region_profiles=sum(g['supported_query_count']for g in bundle['groups'])),
        covariance_summary=dict(effective_sigma_px_min=min(sigmas)if sigmas else None,
            effective_sigma_px_median=float(np.median(sigmas))if sigmas else None,
            effective_sigma_px_max=max(sigmas)if sigmas else None,
            fallback_measurements=sum(m.get('source')=='VISIBLE_BACKBONE_FALLBACK'for m in spec['visible'])),
        candidate_sources_before_dedup=len(allc),projection_rejections=rejected,
        proof=result['proof']),bundle


def solver_job(frame,observation_folder,output_folder,old_run):
    import cv2
    cv2.setNumThreads(1);guard=firewall(old_run);_preload_solver()
    started,cpu=time.monotonic(),time.process_time();folder=Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
    obs=Path(observation_folder);s=read(obs/'OBSERVATION_SEAL.json')
    if s['id']!=frame['id']or sha(obs/'OBSERVATION.jsonl.gz')!=s['observation_sha256']or sha(obs/'MASK.npz')!=s['mask_sha256']:raise ValueError('Observation bytes changed')
    row=next(rows(obs/'OBSERVATION.jsonl.gz'))
    with np.load(obs/'MASK.npz')as a:mask=a['pallet_mask_clean'].copy()
    config,model,interval=_cpu_models()
    with numerical_meter()as calls:result=P.solve_e(row,frame,mask,config,model,interval)
    old_config=read(Path(old_run)/'CONFIG_FROZEN.json')
    features,bundle=feature_frame(frame,result,mask,old_config)
    save_rows(folder/'E_RUN.jsonl.gz',[result]);save_rows(folder/'FEATURES.jsonl.gz',[features]);save_rows(folder/'BOUNDARY_BUNDLE.jsonl.gz',[bundle])
    s=dict(schema='D1_FRAME_EXECUTION_SEAL_V1',id=frame['id'],complete=True,source_sha256=source_bindings(),
        observation_seal_sha256=sha(obs/'OBSERVATION_SEAL.json'),e_run_sha256=sha(folder/'E_RUN.jsonl.gz'),
        features_sha256=sha(folder/'FEATURES.jsonl.gz'),boundary_bundle_sha256=sha(folder/'BOUNDARY_BUNDLE.jsonl.gz'),
        e0_independently_executed=True,new_B_or_A_candidates=False,history_only_union=False,
        operations=dict(calls),resource=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu),
        candidate_counts=dict(raw=features['candidate_sources_before_dedup'],unique=len(features['candidates']),eligible=sum(c['eligible']for c in features['candidates'])),
        reference_firewall=guard.receipt())
    write(folder/'FRAME_SEAL.json',s);return s


def solve_batch(inputs,observation_dir,output,old_run,domain,*,workers=1,limit=None,technical=False,GO=False):
    if not GO:raise ValueError('Explicit root execution GO required')
    if not 1<=workers<=4:raise ValueError('At most four workers')
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frames=select_frames(inputs,domain,limit,technical);obs=Path(observation_dir);oseal=read(obs/'OBSERVATION_SEAL.json')
    if [f['id']for f in frames]!=oseal['ordered_ids']:raise ValueError('Same exact observations required')
    lock=dict(schema='D1_SOLVER_LOCK_V1',inputs_sha256=sha(inputs),observation_seal_sha256=sha(obs/'OBSERVATION_SEAL.json'),
        source_sha256=source_bindings(),ordered_ids=[f['id']for f in frames],old_config_sha256=sha(Path(old_run)/'CONFIG_FROZEN.json'),workers=workers)
    if (out/'BATCH_LOCK.json').exists():
        if read(out/'BATCH_LOCK.json')!=lock:raise ValueError('Solver lock changed')
    else:write(out/'BATCH_LOCK.json',lock)
    done={};pending=[]
    for i,f in enumerate(frames):
        folder=out/'frames'/f'{i:04d}';p=folder/'FRAME_SEAL.json'
        if p.exists():
            s=read(p)
            if s['id']!=f['id']or s['source_sha256']!=source_bindings()or sha(folder/'FEATURES.jsonl.gz')!=s['features_sha256']or sha(folder/'E_RUN.jsonl.gz')!=s['e_run_sha256']:raise ValueError('Completed solver cache changed')
            done[f['id']]=s
        else:pending.append((f,obs/'frames'/f'{i:04d}',folder,old_run))
    with ProcessPoolExecutor(max_workers=workers)as pool:
        futures={pool.submit(solver_job,*job):job for job in pending}
        for future in as_completed(futures):
            job=futures[future]
            try:s=future.result();done[s['id']]=s
            except Exception as exc:
                write(Path(job[2])/'FAILURE.json',dict(id=job[0]['id'],error=repr(exc),traceback=traceback.format_exc(),status='FAILED_TECHNICAL'));raise
            print(canonical(dict(stage='D1_E_SOLVE',done=len(done),total=len(frames),id=s['id'],wall_seconds=time.monotonic()-started)),flush=True)
    save_rows(out/'HEAD_FEATURES.jsonl.gz',[next(rows(out/'frames'/f'{i:04d}'/'FEATURES.jsonl.gz'))for i in range(len(frames))])
    seal=dict(schema='D1_FEATURE_SEAL_V1',complete=True,rows=len(frames),head_features_sha256=sha(out/'HEAD_FEATURES.jsonl.gz'),
        input_sha256=sha(inputs),observation_seal_sha256=sha(obs/'OBSERVATION_SEAL.json'),source_sha256=source_bindings(),
        ordered_ids=[f['id']for f in frames],e0_independently_executed=True,e0_name='E0_REGENERATED',
        same_automatic_procedure_both_domains=True,history_only_union=False,new_B_or_A_candidates=False,
        feature_meaning_unchanged=True,feature_scores_use_original_unrepaired_A_B_energies=True,
        operations=dict(sum((Counter(s['operations'])for s in done.values()),Counter())),
        wall_seconds=time.monotonic()-started,sum_frame_CPU_seconds=sum(s['resource']['CPU_seconds']for s in done.values()),
        failures=[],all_rows_kept=True,before_supervision_and_PRIMARY_scoring=True)
    write(out/'FEATURE_SEAL.json',seal);return seal


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['observe','solve']);p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--old-run',required=True);p.add_argument('--domain',choices=['real','synthetic'],required=True)
    p.add_argument('--observation-dir');p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--technical',action='store_true');p.add_argument('--root-GO',action='store_true');a=p.parse_args()
    if a.stage=='observe':r=observe_batch(a.inputs,a.output,a.old_run,a.domain,device=a.device,limit=a.limit,technical=a.technical,GO=a.root_GO)
    else:r=solve_batch(a.inputs,a.observation_dir,a.output,a.old_run,a.domain,workers=a.workers,limit=a.limit,technical=a.technical,GO=a.root_GO)
    print(canonical(r),flush=True)


if __name__=='__main__':main()
