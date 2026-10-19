"""Sealed real238/synthetic1003 F cue pools and actual policy selections.

Only prediction inputs/cached automatic states and masks are read. The later
CAL-only fit is a separate module and process. No old score condition is used.
"""
from pathlib import Path
import argparse
import json
import time
import inspect
from collections import Counter
import numpy as np
import cv2
from . import f_observer as F
from .s1_union import rows,read,sha,write,save,payload_sha,finite

def source_binding():
    from scripts.research.pallet_boundary_action_repair_v1.d1_batch import source_bindings
    result=source_bindings()
    for path in (Path(__file__),Path(F.__file__),Path(F.O.__file__),
                 Path(inspect.getsourcefile(F.O.intersection_uncertainty))):
        result[str(path.relative_to(Path(__file__).resolve().parents[3]))]=sha(path)
    from scripts.research.pallet_bi_conditioned_observation_v1 import solver as L
    result[str(Path(L.__file__).relative_to(Path(__file__).resolve().parents[3]))]=sha(L.__file__)
    return result

def run(d1,campaign,output,GO=False,precompute_core=False):
    if GO is not True:raise PermissionError('Explicit F stage GO required')
    d1,campaign,output=map(Path,(d1,campaign,output));output.mkdir(parents=True,exist_ok=True)
    choice_path=campaign/'F_SYSTEM_CHOICE.json';core_path=campaign/'F_CORE_PRECOMPUTE_LOCK.json'
    if precompute_core:
        core=read(core_path)
        if core.get('final_configuration_selected')is not False or core.get('H1_fits_authorized_by_this_lock')!=0:raise ValueError('Honest core-only dependency lock required')
        if core['policy_lock_sha256']!=sha(campaign/'F_OBSERVER_POLICY_LOCK_V2.json'):raise ValueError('Fixed unconditional F policy changed')
        choice=None
    else:
        choice=read(choice_path)
        if choice.get('locked_before_F_production')is not True:raise ValueError('F system selection must be frozen')
    from scripts.research.pallet_boundary_action_repair_v1.d1_batch import firewall
    guard=firewall(d1.parent/'pallet-statewise-end-to-end-recovery-v1')
    cv2.setNumThreads(1);start=time.monotonic();cpu=time.process_time();binding=source_binding()
    lock=dict(schema='FULL_F_OBSERVATION_PRODUCER_LOCK_V1',source_binding=binding,policy=F.POLICY,
              campaign_plan_sha256=sha(campaign/'CAMPAIGN_PLAN.json'),system_choice_sha256=sha(choice_path)if choice is not None else None,
              stage='UNCONDITIONAL_F_CORE_PRECOMPUTATION'if precompute_core else'FINAL_CONFIGURATION_F_OBSERVATIONS',
              deferred_final_configuration=precompute_core,core_dependency_proof_sha256=sha(core_path)if precompute_core else None,
              environment_sha256=sha(campaign/'ENVIRONMENT_INITIAL.json'),
              policy_lock_sha256=sha(campaign/'F_OBSERVER_POLICY_LOCK_V2.json'),numeric_reference_inputs=False,
              invocation='independent F observation policy production, not frozen-scorer bit change')
    if read(campaign/'F_OBSERVER_POLICY_LOCK_V2.json')['source_sha256']!=sha(F.__file__):raise ValueError('F observation policy code changed')
    write(output/'PRODUCER_LOCK.json',lock);manifest=[];counts=Counter()
    for domain,upname in [('real','REAL238'),('synthetic','SYNTHETIC1003')]:
        upstream=d1/f'd1_upstream_v1/{upname}_UPSTREAM.jsonl.gz'
        frames=list(rows(upstream));expected=238 if domain=='real'else 1003
        if len(frames)!=expected:raise ValueError('Whole canonical population required')
        osroot=d1/f'd1_{domain}_observations_v1';oseal=read(osroot/'OBSERVATION_SEAL.json')
        if sha(upstream)!=oseal['input_sha256']:raise ValueError('Original input seal changed')
        for i,frame in enumerate(frames):
            obsroot=osroot/'frames'/f'{i:04d}';out=output/domain/f'{i:04d}'
            os=read(obsroot/'OBSERVATION_SEAL.json')
            if sha(obsroot/'OBSERVATION_SEAL.json')!=oseal['frame_seals_sha256'][f'{i:04d}']:raise ValueError('Frozen observation seal ancestry changed')
            if os['input_row_sha256']!=payload_sha(frame):raise ValueError('Changed frozen frame')
            mask_path=obsroot/'MASK.npz';op=obsroot/'OBSERVATION.jsonl.gz'
            if sha(op)!=os['observation_sha256']or sha(mask_path)!=os['mask_sha256']:raise ValueError('Changed frozen observation/mask')
            dependencies=dict(frame_sha256=payload_sha(frame),observation_sha256=sha(op),mask_sha256=sha(mask_path),producer_lock_sha256=sha(output/'PRODUCER_LOCK.json'))
            if (out/'FRAME_SEAL.json').exists():
                receipt=read(out/'FRAME_SEAL.json')
                if receipt['dependencies']!=dependencies:raise ValueError('F cache dependency changed')
                for name,digest in receipt['artifacts'].items():
                    if sha(out/name)!=digest:raise ValueError('Completed F row changed')
                counts['exact_completed_reuses']+=1
            else:
                tick=time.monotonic();tcpu=time.process_time()
                if sha(frame['image'])!=frame['image_sha256']:raise ValueError('Changed RGB')
                obs=next(rows(op))
                if obs['id']!=frame['id']:raise ValueError('Observation identity mismatch')
                with np.load(mask_path,allow_pickle=False)as mask:pool=F.cue_pool(frame,obs,mask)
                selections={arm:F.choose(pool,frame,arm)for arm in ('COMMON','BI')}
                # Original immutable typed parser authenticates new revocations;
                # every source dedup is applied by the shared backend later.
                from scripts.research.pallet_bi_conditioned_observation_v1 import solver as L
                for selection in selections.values():
                    L._prepare(frame['native_xyz'],frame['K'],selection['states'],selection['full_points'],frame['raw_hw'],selection['partial_lines'],L.SOLVER_CONFIG)
                save(out/'POOL.jsonl.gz',[pool]);save(out/'SELECTIONS.jsonl.gz',[dict(id=frame['id'],partition=frame['partition'],split=frame['split'],selections=selections)])
                receipt=dict(complete=True,id=frame['id'],dependencies=dependencies,
                    artifacts={n:sha(out/n)for n in ('POOL.jsonl.gz','SELECTIONS.jsonl.gz')},
                    wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-tcpu,
                    reference_reads=0,model_forwards=0,pose_optimizer_calls=0,
                    actual_image_line_queries=sum(len(c.get('rays',[]))+len(c.get('aliases_and_unsupported',[]))for c in pool['corners']),
                    accepted_unique_intersection_candidates=len([c for corner in pool['corners']for c in corner.get('candidates',[]) ]),
                    COMMON_BI_identical_raw_selection_bytes=selections['COMMON']==selections['BI'],
                    raw_selection_identity_caveat='includes unused policy provenance; effective solver packet equality is checked separately')
                write(out/'FRAME_SEAL.json',receipt);counts['new_frames']+=1
            manifest.append(dict(domain=domain,index=i,id=frame['id'],partition=frame['partition'],split=frame['split'],seal_path=str(out/'FRAME_SEAL.json'),seal_sha256=sha(out/'FRAME_SEAL.json')))
            if i%50==0:print(json.dumps(dict(stage='F_ACTUAL_POLICY_OBSERVATION',domain=domain,done=i+1,total=expected,wall_seconds=time.monotonic()-start)),flush=True)
    if len(manifest)!=1241:raise ValueError('Incomplete F observations')
    result=dict(complete=True,schema='FULL_F_OBSERVATION_MANIFEST_V1',rows=1241,frames=manifest,counts=dict(counts),
        producer_lock_sha256=sha(output/'PRODUCER_LOCK.json'),reference_firewall=guard.receipt(),
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,real_reference_inputs=False,
        new_state_mask_PF_forwards=0,new_observation_network_fits=0,pose_optimizer_calls=0,H1_fits=0,
        component_complete=True,final_F_complete=False,unconditional_core_precomputation=precompute_core)
    write(output/'OBSERVATION_MANIFEST.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('d1','campaign','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--GO',action='store_true');p.add_argument('--precompute-core',action='store_true')
    a=p.parse_args();r=run(a.d1,a.campaign,a.output,a.GO,a.precompute_core)
    print({k:r[k]for k in ('complete','rows','counts','wall_seconds')})
