"""Fixed existing MobileSAM synthetic B support; never renderer GT masks.

The original full-geometry prompt and component cleaner are reused unchanged.
Tiny/state classification is not executed: synthetic states remain explicitly
conditional on the frozen coarse geometry. No labels enter this process.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from .synthetic_data import rows,read,write,save,sha,cost,no_answers,VIS


def gpu_state():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    foreign=[x for x in result.splitlines()if '/usr/share/rustdesk/rustdesk'not in x]
    import os
    foreign=[x for x in foreign if x.split(',')[0].strip()!=str(os.getpid())]
    if foreign:raise RuntimeError('Preserve foreign GPU jobs; unavailable: '+str(foreign))
    status=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip()
    return dict(status=status,compute_apps=result.strip(),rustdesk_preserved=True)


def run(repo,private,pilot_only=True):
    import torch
    from scripts.research.pallet_visibility_curriculum_20261013_v1.capture_boundary_recovery_masks_v4_6 import GeometryPromptSAMCapture,build_geometry_prompt,DEFAULT_CONFIGURATION
    from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v2 import registered_geometry
    from scripts.research.pallet_statewise_end_to_end_recovery_v1 import synthetic_data as D
    start,cpu=time.monotonic(),time.process_time();out=private/'synthetic_v1';maskdir=out/'masks';maskdir.mkdir(exist_ok=True)
    name='PILOT512_ROOT_FRAMES_NO_GT.jsonl.gz'if pilot_only else 'EXPANSION1003_ROOT_FRAMES_NO_GT.jsonl.gz';path=out/name;inputseal=read(out/name.replace('.jsonl.gz','_SEAL.json'));assert sha(path)==inputseal['predictions_sha256']
    frames=list(rows(path));study=repo/VIS/'statewise_joint_ot_20261014_v2';runtime_path=study/'RUNTIME_CONFIG_V3.json';runtime=read(runtime_path);boundary=(study/runtime['boundary_mask_config']).resolve();descriptor=(study/runtime['mask_descriptor']).resolve();samconfig=read(boundary);tiny=read(descriptor)['pallet_segmenter'];weight=(boundary.parent/samconfig['mobile_sam_checkpoint']).resolve();vendor=(boundary.parent/samconfig['mobile_sam_vendor']).resolve()
    protocol_path=out/'SYNTHETIC_MASK_PROTOCOL_LOCK.json'
    protocol=dict(schema='P6_EXISTING_FROZEN_MOBILESAM_AUTOMATIC_MASK_PROTOCOL_V1',provider='GeometryPromptSAMCapture from original capture_boundary_recovery_masks_v4_6',
        source_sha256=sha(__file__),runtime_config_sha256=sha(runtime_path),boundary_config_sha256=sha(boundary),mask_descriptor_sha256=sha(descriptor),
        MobileSAM_checkpoint_sha256=sha(weight),expected_checkpoint_sha256=samconfig['model_sha256']['mobile_sam_checkpoint'],tiny_checkpoint_unchanged_not_forwarded=tiny,
        provider_sources={str(repo/'scripts/research/pallet_visibility_curriculum_20261013_v1'/n):sha(repo/'scripts/research/pallet_visibility_curriculum_20261013_v1'/n)for n in ['capture_boundary_recovery_masks_v4_6.py','statewise_masks_v2.py']},
        vendor_source_sha256={str(p.relative_to(vendor)):sha(p)for p in sorted(vendor.rglob('*.py'))},configuration=DEFAULT_CONFIGURATION,
        GT_masks_or_labels_or_coordinates_inputs=False,prompts='same full registered coarse projection + finite nonS N3/SubPix query union, padding19; box-only, no pointprompts',
        synthetic_states='unchanged coarsegeometry/canvas only; external classifier/Tiny not executed',
        state_mask_placeholder='zeros passed only to unused carry-through argument. NEVER used for visibility, B histogram, or headfeature; original provenance claim EXACT_FROZEN_FULLFRAME_TINY_CACHE explicitly replaced',
        target_mask='actual fixed MobileSAM+unchanged relative-component cleaner; no target semantic guarantee',
        pilot512_then_reuse_existing_masks_in_expansion1003=True,new_training=0,new_model_introduction=False,GPU_hours_max=8.)
    if protocol_path.exists():
        if read(protocol_path)!=protocol:raise ValueError('Frozen mask protocol changed')
    else:write(protocol_path,protocol)
    if protocol['MobileSAM_checkpoint_sha256']!=protocol['expected_checkpoint_sha256']:raise ValueError('Checkpoint changed')
    state=gpu_state();torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    generated=reused=0;GPU_ms=0.;records=[];newframes=[]
    with no_answers():
        provider=GeometryPromptSAMCapture.from_configs(read(descriptor),boundary,'cuda')
        for n,frame in enumerate(frames):
            fid=frame['id'];stem=fid.replace(':','__');npz=maskdir/(stem+'.npz');recordpath=maskdir/(stem+'.json');r=frame['packet'];pose=frame['fallback'];spec=r['solver_input']
            if npz.exists()or recordpath.exists():
                if not npz.exists()or not recordpath.exists():raise ValueError('Incomplete earlier mask artifact; preserve for diagnosis')
                receipt=read(recordpath)
                if receipt['image_sha256']!=frame['image_sha256']or receipt['mask_sha256']!=sha(npz)or receipt['mask_protocol_sha256']!=sha(protocol_path):raise ValueError('Reused mask binding mismatch')
                reused+=1
            else:
                if sha(frame['image'])!=frame['image_sha256']:raise ValueError('Bound RGB changed')
                image=cv2.imread(frame['image']);h,w=image.shape[:2]
                geometry=registered_geometry([h,w],spec['K'],spec['xyz'],pose)
                prompt=build_geometry_prompt(geometry,frame['trace']['initial_points'],spec['states'])
                before=time.monotonic();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                result=provider(image,prompt=prompt,state_pallet_mask=np.zeros((h,w),np.float32))
                end.record();torch.cuda.synchronize();elapsed=begin.elapsed_time(end);GPU_ms+=elapsed
                inherited=deepcopy(result['provenance']);inherited['inherited_state_features_source_string']=inherited['state_features_source'];inherited['state_features_source']='NOT_RUN_ZERO_PLACEHOLDER_CARRY_ONLY_GEOMETRIC_SYNTHETIC_STATES';inherited['state_mask_placeholder_used_for_features']=False
                np.savez_compressed(npz,pallet_mask_clean=result['pallet_mask_clean'].astype(np.uint8),pallet_mask_raw=result['pallet_mask_raw'].astype(np.uint8),sam_candidate_quality=result['sam_candidate_quality'],sam_candidate_masks=result['sam_candidate_masks'])
                receipt=dict(id=fid,image_sha256=frame['image_sha256'],mask_sha256=sha(npz),mask_protocol_sha256=sha(protocol_path),provider_provenance=inherited,
                    GT_masks_or_labels_read=False,S_raw_coordinates_used_as_prompt=False,new_mask_forward_calls=1,new_Tiny_or_visibility_forwards=0,new_model_training=0,
                    elapsed_wall_seconds=time.monotonic()-before,CUDA_elapsed_ms=elapsed)
                write(recordpath,receipt);generated+=1
            new=deepcopy(frame);new['packet']['provenance']['original_mask_binding']=dict(path=str(npz),sha256=receipt['mask_sha256'],kind='NEW_AUTOMATIC_FIXED_MOBILESAM_SYNTHETIC_MASK; not rendererGT')
            new['trace']['mask_binding']=new['packet']['provenance']['original_mask_binding'];new['automatic_mask_production']=dict(receipt=str(recordpath),new_GT_mask_inputs=False,source='same existing frozen GeometryPromptSAMCapture; synthetic states geometry only')
            records.append(receipt);newframes.append(new)
            if n%32==0:print('FIXED_SAM_SYNTHETIC',n+1,len(frames),'new',generated,'reuse',reused,'GPU_ms',round(GPU_ms,1),flush=True)
            if n==3 and generated and GPU_ms/4000*len(frames)>8*3600:raise RuntimeError('Pilot throughput exceeds approved GPU budget; preserved first4 masks')
            if GPU_ms>8*3600*1000:raise RuntimeError('Approved cumulative mask GPU budget exceeded')
            if n%64==0:gpu_state()
        output='PILOT512_ROOT_FRAMES_AUTO_MASK.jsonl.gz'if pilot_only else 'EXPANSION1003_ROOT_FRAMES_AUTO_MASK.jsonl.gz';save(out/output,newframes)
    receipt=dict(schema='P6_SYNTHETIC_AUTOMATIC_MASK_PRODUCTION_SEAL_V1',rows=len(newframes),predictions_sha256=sha(out/output),input_sha256=sha(path),mask_protocol_sha256=sha(protocol_path),mask_npz_count=len(records),new_model_forward_calls=generated,reused_masks=reused,
        new_Tiny_or_visibility_forwards=0,GT_masks_or_labels_read=False,GPU_elapsed_seconds=GPU_ms/1000,GPU_hours_counted=GPU_ms/3600000,initial_GPU_state=state,
        pytorch_version=torch.__version__,cuda_version=torch.version.cuda,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),mask_records=[dict(id=r['id'],mask_sha256=r['mask_sha256'])for r in records],cost=cost(start,cpu))
    receipt['cost']['new_model_forwards']=generated
    write(out/output.replace('.jsonl.gz','_SEAL.json'),receipt)
    print(json.dumps(dict(status='AUTOMATIC_MASK_FRAMES_SEALED',output=output,rows=len(newframes),generated=generated,reused=reused,GPU_seconds=GPU_ms/1000,cost=cost(start,cpu))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--private',required=True,type=Path);p.add_argument('--expansion',action='store_true');a=p.parse_args();run(a.repo,a.private,not a.expansion)


if __name__=='__main__':main()
