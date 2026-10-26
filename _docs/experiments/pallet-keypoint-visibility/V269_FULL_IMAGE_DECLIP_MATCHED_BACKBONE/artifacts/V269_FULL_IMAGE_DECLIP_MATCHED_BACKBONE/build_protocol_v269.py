"""Seal the prospective three-arm protocol after SYN features exist; no fit.

Root executes this once after reviewing feature/source receipts and code. This
program verifies bindings and array headers; it does not run image encoders,
read current RGB/annotations, score DEV, instantiate optimizers or train heads.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE=Path(__file__).absolute().parent
PREV=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
ARMS=('EVA_STANDARD','EVA_CSA','EVA_DECLIP')


def safe(value):
    p=Path(value)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def binding(value):
    p=safe(value);h=hashlib.sha256()
    with p.open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def checked(bound):
    actual=binding(bound['path'])
    assert actual=={k:bound[k]for k in actual}
    return safe(bound['path'])


def read(path):return json.loads(safe(path).read_bytes())


def main():
    path=safe(HERE/'PROTOCOL_BEFORE_FIT.json');assert not path.exists()
    d=read(HERE/'DESIGN_BEFORE_DATA_FORWARDS.json')
    assert d['completed'] and tuple(d['planned_new_arms'])==ARMS and d['new_fit_budget']==3
    assert d['seed']==17 and d['optimizer_updates_each']==1000 and d['batch_size']==8
    tcpath=safe(HERE/'syn_mask_inputs_v1/COMPLETE.json');tc=read(tcpath)
    assert tc['PASS'] and tc['completed'] and tc['all1003_original_M4_SHA_verified']
    assert len(tc['ten_original_M4_exact_replays'])==10 and all(x['target_FP16_exact']for x in tc['ten_original_M4_exact_replays'])
    fcpath=safe(HERE/'syn_fullframe_features_v1/COMPLETE.json');fc=read(fcpath)
    assert fc['PASS'] and fc['completed']
    assert fc['frames']==1003 and fc['train_val']==[768,235]
    assert fc['source_RGB_SHA_verified_frames']==1003 and fc['new_optimizer_updates']==0
    assert fc['original_M4_query_native_pose_or_current_reads']==0
    assert fc['encoder_code']==binding(HERE/'fullframe_eva_declip_v269.py')
    checked(fc['code']);checked(fc['frame_order']);checked(fc['source_RGB_manifest'])
    assert fc['source_RGB_manifest']['sha256']=='22b4ea7f28ed8de8b13e83d6a83aa5774a26b6e41da135b3920dccea022e3a75'
    assert fc['frame_order']['sha256']==fc['source_RGB_manifest']['sha256']
    frames=[json.loads(x)for x in safe(fc['frame_order']['path']).read_bytes().splitlines()]
    assert [x['frame_index']for x in frames]==list(range(1003))
    assert [x['split']for x in frames]==['train']*768+['validation']*235
    m4=read(checked(tc['outputs']['M4_SOURCE_BINDINGS.json']))['M4']
    assert all(Path(f['RGB']['path']).stem.removesuffix('_rgb')==Path(m['path']).stem for f,m in zip(frames,m4))
    proofs=fc['encoder_proofs'];assert set(proofs)==set(ARMS)
    assert proofs['EVA_STANDARD']['state_before']==proofs['EVA_CSA']['state_before']!=proofs['EVA_DECLIP']['state_before']
    for a in ARMS:
        assert proofs[a]['strict_visual_state_load'] and proofs[a]['trainable_parameters']==0
        assert proofs[a]['state_before']==proofs[a]['state_after']
        assert proofs[a]['named_buffers_before']==proofs[a]['named_buffers_after']
        assert proofs[a]['all_persistent_and_nonpersistent_buffers_unchanged']
        assert proofs[a]['actual_full_frame_forwards']==1003
    source={k:fc['outputs'][k]for k in ('tokens'+a for a in ARMS)}
    source.update(targets=tc['outputs']['MASK_TARGETS.npz'],draws=tc['outputs']['FIT_DRAWS1000x8.npy'])
    for b in source.values():checked(b)
    for a in ARMS:
        z=np.load(safe(source['tokens'+a]['path']),mmap_mode='r',allow_pickle=False)
        assert z.shape==(1003,1024,512) and z.dtype==np.float16
    with np.load(safe(source['targets']['path']),allow_pickle=False)as z:
        mask=z['mask'];fit=z['fit'];ids=z['frame_ids']
        assert mask.shape==(1003,1,32,32) and mask.dtype==np.float16
        assert np.array_equal(fit,np.arange(1003)<768) and np.array_equal(ids,np.arange(1003))
        frequency=float(mask[fit].astype(np.float64).mean())
    draws=np.load(safe(source['draws']['path']),allow_pickle=False)
    assert np.array_equal(draws,np.random.Generator(np.random.PCG64(17)).choice(np.flatnonzero(fit),size=(1000,8),replace=True))
    assert fit[draws].all() and frequency==tc['FIT_foreground_frequency']
    aq=read(HERE/'ENCODER_ARTIFICIAL_SMOKE.json');assert aq['PASS']
    hq=read(HERE/'HEAD_TRAINER_PRE_FIT_QA.json');assert hq['PASS'] and hq['model_parameters']==26721
    assert hq['optimizer_updates']==0 and hq['current_data_reads']==0
    assets=read(HERE/'OFFICIAL_ASSETS_COMPLETE.json');assert assets['PASS'] and assets['completed']
    for b in assets['checkpoints']:checked(b)
    model=binding(HERE/'model_mask_head_v269.py')
    assert model['sha256']=='4fcb2593fc07595d44fa43ed4eb7c042e4297d7fee7f93a0737b863850f74590'
    assert hq['model_code']==model and hq['training_code']==binding(HERE/'train_fullmask_v269.py')
    current_complete=safe(PREV/'current_fullframe_inputs_v2/COMPLETE.json')
    assert binding(current_complete)['sha256']=='83913adc35316e6dbf226d7c593f1307c06ca0d33399264fb0bb0f7109705e10'
    current_helper=safe(PREV/'probe_current_fullframes_v267.py')
    assert binding(current_helper)['sha256']=='c3463ef12ef245fdf156d77c8e3f6c911e3d343d64db1ca0244b67739726b84a'
    policy=dict(schema='V269_MATCHED_EVA_THREE_ARM_SYN_MASK_HEAD_V1',completed=True,PASS=True,pre_fit=True,
        created_UTC=datetime.now(timezone.utc).isoformat(),design=binding(HERE/'DESIGN_BEFORE_DATA_FORWARDS.json'),
        protocol_builder=binding(Path(__file__).absolute()),source_mask_complete=binding(tcpath),source_feature_complete=binding(fcpath),
        source_feature_extractor=fc['code'],frozen_encoder_states={a:proofs[a]['state_before']for a in ARMS},
        frozen_encoder_all_buffer_states={a:proofs[a]['named_buffers_before']for a in ARMS},
        independent_design_source_review=binding(HERE/'INDEPENDENT_DESIGN_SOURCE_REVIEW.json'),
        model_code=model,training_code=binding(HERE/'train_fullmask_v269.py'),encoder_code=binding(HERE/'fullframe_eva_declip_v269.py'),
        inference_code=binding(HERE/'infer_fullframe_v269.py'),evaluation_code=binding(HERE/'evaluate_fullframe_v269.py'),
        encoder_artificial_QA=binding(HERE/'ENCODER_ARTIFICIAL_SMOKE.json'),head_training_QA=binding(HERE/'HEAD_TRAINER_PRE_FIT_QA.json'),
        official_assets_complete=binding(HERE/'OFFICIAL_ASSETS_COMPLETE.json'),official_remote_metadata=binding(HERE/'OFFICIAL_REMOTE_METADATA.json'),
        checkpoints=[{k:b[k]for k in ('path','sha256','size_bytes')}for b in assets['checkpoints']],
        official_source_branches=[{k:b[k]for k in ('official_repository','branch','commit')}for b in assets['source_branches']],
        source_bindings=source,train_arms=list(ARMS),reported_arms=list(ARMS),new_fits_max=3,seed=17,
        optimizer_updates_each=1000,batch_size=8,optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        threshold=.5,FIT_foreground_frequency=frequency,positive_weight=.5/frequency,negative_weight=.5/(1-frequency),
        loss='Unchanged V267 class-balanced soft pixel BCE per-frame weighted mass normalization + soft Dice1 eps1e-6',
        mask_geometry='Original whole M4 uint8>127 -> torch bilinear512 align_cornersFalse antialiasTrue -> avgpool16 -> float16 32x32; no GT crop',
        encoder_geometry='Whole original RGB -> PIL bicubic512 square -> /255 -> same CLIP normalization; no KP crop or sliding window; official projected512 patch tokens, CLS removed, L2 pertoken',
        arm_interpretation=dict(EVA_STANDARD='Original EVA02-CLIP-B-16 weights, official vanilla QK+residual+FFN dense readout',
            EVA_CSA='Same original EVA weights, official csa QQ/KK attention readout',
            EVA_DECLIP='Released DeCLIP EVA weights, identical official csa readout to EVA_CSA'),
        contrasts=d['contrasts'],public_real_pretraining_disclosure=d['DeCLIP_pretraining_disclosure'],
        checkpoint_selection='Exactly3fixedfinal1000 fits; all3finalCPs sealed before DEV scoring; no checkpoint/threshold/recipe selection on DEV/current; no automatic extra fits',
        original_dataset=dict(frames=1003,FIT=768,DEV=235),real_fit=0,new_PnP=0,q_native_pose_M0_inputs=0,
        current_input_helper=binding(current_helper),current_input_complete=binding(current_complete),
        query_readout=dict(sample='Original raw32logit map bilinear, normalized(q+.5)/rawWH, sigmoid aftersample',
            align_corners=False,padding_mode='border',D_if='logit>=0',E_if='logit<0',U='45 unchanged unavailable original inputs',
            coordinate_and_native_order='unchanged',scope='MASK_PRESENCE_PROXY_E_NOT_CAUSE'),
        evaluation='Same current223 repeated development evaluation only; all new maps/predictions sealed before human GT join; PRIMARY1237 D1147/E90, wood/square and SELF separately; no real pixel IoU',
        semantic_limit='M4 learns target-visible pixel presence; non-target complement is not an external-occlusion cause label and not equivalent to native keypoint visibility',
        historical_baselines='Existing V267 OpenAI FULL_CLIP and FULL_CLIP_SURGERY_ARCH saved outputs; no re-fit of historical baselines',
        source_publication='Do not vendor downloaded third-party official_source or pretrained assets; publish pinned URLs/commits/SHA and fetch instructions',
        further_automatic_fits=False)
    with path.open('x')as f:json.dump(policy,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,protocol=binding(path))),flush=True)


if __name__=='__main__':main()
