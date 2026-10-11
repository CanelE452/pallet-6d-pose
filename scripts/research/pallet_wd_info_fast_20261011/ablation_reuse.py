"""Reuse only whitelisted per-ID saved N0/N1/N2/N3 outputs: zero model runs."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C
from scripts.research.pallet_wd_info_fast_20261011.classifier import modules

ARMS=('N0_BASE_REPLAY','N1_SYM_ONLY','N2_DIM_ONLY','N3_DIM_SYM')


def moments(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),SD=float(a.std(ddof=1)),variance=float(a.var(ddof=1)),
                median=float(np.median(a)),P90=float(np.quantile(a,.9)),P95=float(np.quantile(a,.95)))


def contrast(a,b,ma,mb):
    # Axes are fixed seed, native frame, original corner index. No GT alignment.
    residual=np.linalg.norm(a-b,axis=-1)
    valid=ma&mb&np.isfinite(residual)
    collapsed=np.divide(np.where(valid,residual,0).sum(0),valid.sum(0),
                        out=np.full(residual.shape[1:],np.nan),where=valid.sum(0)>0)
    return dict(frames=a.shape[1],paired_seed_rows=a.shape[0]*a.shape[1],
                matched_native_corners=int(valid.sum()),
                corner_displacement_px_frame_mean_fixed_seeds=moments(collapsed[np.isfinite(collapsed)]),
                pooled_seed_corner_displacement_px_descriptive=moments(residual[valid]),
                per_seed=[dict(seed=i+1,corner_displacement_px=moments(residual[i][valid[i]])) for i in range(a.shape[0])],
                coordinate_frame='native image px',correspondence='fixed current corner index0..7, no GT remapping')


def run():
    started=time.monotonic();C.lock_ready()
    contract=C.read(C.DOC/'INPUT_CONTRACT.json');ids=sorted(contract['allowed_ids']);allowed=set(ids)
    protected=set(contract['excluded_protected_ids']);assert len(ids)==173 and not allowed&protected
    calib_path=C.SOURCE/'_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json'
    calibration=C.read(calib_path)
    assert calibration['rule']=={'lam':1.0,'max_move_image_diagonal_fraction':.01}
    _,_,_,_=modules()
    import refiner
    extractor=refiner.E.old('features')
    cache_manifest_path=C.SOURCE/'_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json'
    cache={r['id']:r for r in C.read(cache_manifest_path)['records'] if r['id'] in allowed}
    assert set(cache)==allowed
    q0=[]; net=[];gain=[];index=[];cache_hashes=[]
    for fid in ids:
        binding=cache[fid];path=C.SOURCE/binding['path'];assert C.sha(path)==binding['sha256']
        packet=torch.load(path,map_location='cpu',weights_only=False);captured=packet['captured']
        inputs=extractor.branch_inputs(captured);assert inputs is not None
        selected=int(captured['selected_index']);assert selected==int(np.argmax([r['score'] for r in captured['candidates']]))
        q0.append(np.asarray(captured['candidates'][selected]['keypoints_xy'],float));net.append(np.asarray(inputs['points'],float));gain.append(float(inputs['gain']));index.append(selected)
        cache_hashes.append(dict(id=fid,sha256=binding['sha256']))
        del packet,captured,inputs
    q0=np.asarray(q0);net=np.asarray(net);gain=np.asarray(gain);points={};masks={};top={};source_hash=hashlib.sha256();cp=[];read_count=0
    for arm in ARMS:
        for seed in (1,2,3):
            name=f'{arm}_seed{seed}';record=calibration['temperatures'][name]
            assert record['temperature']==1.0
            binding=record['checkpoint'];assert C.sha(C.SOURCE/binding['path'])==binding['sha256']
            cp.append(dict(arm=arm,seed=seed,**binding));a=[];m=[];t=[]
            for j,fid in enumerate(ids):
                assert fid in allowed and fid not in protected
                path=C.SOURCE/'data/pallet/results/pallet_dim_conditioned_p_v1/DEV_logits'/name/(fid.replace(':','__')+'.npz')
                with np.load(path,allow_pickle=False) as packet:
                    saved=packet['points'];support=packet['support'];logits=packet['logits']
                assert saved.shape==(9,2) and support.shape==(8,) and logits.shape==(8,222)
                native=q0[j]+(saved.astype(float)-net[j])/gain[j];native[8]=q0[j,8]
                a.append(native[:8]);m.append(support);t.append(logits.argmax(-1));read_count+=1
                source_hash.update((name+'|'+fid+'|'+C.sha(path)+'\n').encode())
            points[name]=np.asarray(a);masks[name]=np.asarray(m);top[name]=np.asarray(t)
    expected={r['id']:r for r in C.read(C.PRIVATE/'B1_PREDICTIONS_SEALED.json') if r['method']=='CORRECT'}
    parity=max(float(np.max(np.linalg.norm(points['N3_DIM_SYM_seed1'][j]-np.asarray(expected[fid]['q'])[:8],axis=-1))) for j,fid in enumerate(ids))
    assert parity<=1e-4
    contrasts={}
    for a,b,label in [('N1_SYM_ONLY','N0_BASE_REPLAY','symmetry_with_dimensions_absent'),
                      ('N3_DIM_SYM','N2_DIM_ONLY','symmetry_with_dimensions_present'),
                      ('N2_DIM_ONLY','N0_BASE_REPLAY','dimensions_with_symmetry_absent'),
                      ('N3_DIM_SYM','N1_SYM_ONLY','dimensions_with_symmetry_present'),
                      ('N3_DIM_SYM','N0_BASE_REPLAY','joint_ablation'),
                      ('N2_DIM_ONLY','N1_SYM_ONLY','dimension_only_vs_symmetry_only')]:
        pa=np.stack([points[f'{a}_seed{s}'] for s in (1,2,3)]);pb=np.stack([points[f'{b}_seed{s}'] for s in (1,2,3)])
        ma=np.stack([masks[f'{a}_seed{s}'] for s in (1,2,3)]);mb=np.stack([masks[f'{b}_seed{s}'] for s in (1,2,3)])
        ta=np.stack([top[f'{a}_seed{s}'] for s in (1,2,3)]);tb=np.stack([top[f'{b}_seed{s}'] for s in (1,2,3)])
        v=ma&mb; contrasts[a+'__minus__'+b]=dict(kind='different_learned_checkpoints_paired_seed',meaning=label,
            **contrast(pa,pb,ma,mb),candidate_argmax_changes=int(((ta!=tb)&v).sum()),candidate_argmax_supported=int(v.sum()))
    seed_results={};move={}
    for arm in ARMS:
        pairs={}
        for first,second in ((1,2),(1,3),(2,3)):
            key=f'seed{first}_vs_seed{second}'
            record=contrast(points[f'{arm}_seed{first}'][None],points[f'{arm}_seed{second}'][None],
                            masks[f'{arm}_seed{first}'][None],masks[f'{arm}_seed{second}'][None])
            record['source_seed_pair']=[first,second];record.pop('per_seed')
            pairs[key]=record
        seed_results[arm]=pairs
        p=np.stack([points[f'{arm}_seed{s}'] for s in (1,2,3)]);m=np.stack([masks[f'{arm}_seed{s}'] for s in (1,2,3)])
        move[arm]=contrast(p,np.repeat(q0[None,:,:8],3,axis=0),m,m)
    b1=C.read(C.DOC/'B1_RESULTS.json');table=[]
    for name,r in contrasts.items():
        table.append(dict(contrast=name,kind=r['kind'],seeds=[1,2,3],frames=173,
                          **r['corner_displacement_px_frame_mean_fixed_seeds']))
    for name,r in b1['summaries'].items():
        table.append(dict(contrast='N3_seed1:'+name+'__minus__CORRECT',kind='same_checkpoint_dimension_counterfactual',
                          seeds=[1],frames=173,**r['coordinate_delta_px']))
    result=dict(status='REUSED',code=C.binding(Path(__file__)),frames=173,sessions=9,seeds=[1,2,3],
                checkpoint_pairs=contrasts,seed_variation=seed_results,Base_to_method_movement=move,
                different_checkpoint_vs_same_checkpoint_table=table,
                native_coordinate_parity_with_B1_CORRECT_max_error_px=parity,
                original_calibration=C.binding(calib_path),original_cache_manifest=C.binding(cache_manifest_path),
                source_checkpoints=cp,allowed_cache_input_bindings=cache_hashes,
                saved_logits_decoded_point_files_read=read_count,saved_output_binding_digest_sha256=source_hash.hexdigest(),
                allowed_ID_sha256=contract['allowed_id_sha256'],protected_ID_cache_reads=0,
                protected_rows_new_evaluation=0,new_fits=0,new_model_forwards=0,new_OpenCV_calls=0,
                existing_full319_GT_or_prediction_files_read=False,
                reference_pose_fields_read=False,coordinates_only=True,real_6D_ablation_status='NOT_RECOMPUTED: source candidate poses for N0/N1/N2 not evaluated in this 0-solver reuse',
                interpretation='Different checkpoint changes do not isolate dimension input causality; smaller than seed changes does not imply zero dimension effect. The173whitelist differs from historical319, so claims cannot be validated by treating these as the same population.',
                elapsed_seconds=time.monotonic()-started)
    C.write(C.DOC/'B1_ABLATION_REUSE.json',result)
    print('B1_ABLATION_REUSE_DONE',read_count,round(time.monotonic()-started,2),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
