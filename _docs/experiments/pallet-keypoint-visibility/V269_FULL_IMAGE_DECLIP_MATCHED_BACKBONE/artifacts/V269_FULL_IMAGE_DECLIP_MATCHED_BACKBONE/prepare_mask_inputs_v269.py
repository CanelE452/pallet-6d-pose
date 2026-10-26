"""Bind unchanged original-SYN M4 targets/draws; no RGB/current/model reads."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

HERE = Path(__file__).absolute().parent
PREV = HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
REPLAY_FRAMES = (0,1,56,112,168,222,500,767,768,1002)


def safe(value):
    p = Path(value)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112','cache','.pyc','..')), str(p)
    for a in reversed((p,*p.parents)): assert not a.is_symlink(),str(a)
    return p


def binding(path):
    p=safe(path);raw=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def write(path,obj):
    p=safe(path);assert not p.exists()
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(1)
    cp=safe(PREV/'fullmask_targets_v1/COMPLETE.json')
    assert binding(cp)['sha256']=='245629eaaeb66db0439f36be5155d36d8efda28ffdbaf0907d9ad03e8256d32a'
    c=json.loads(cp.read_bytes());assert c['PASS'] and c['completed']
    out=safe(HERE/'syn_mask_inputs_v1');assert not out.exists();out.mkdir()
    write(out/'PLAN_BEFORE_TARGET_REPLAY.json',dict(completed=True,source_complete=binding(cp),
        reuse_exact_targets_and_draws=True,replay_frames=list(REPLAY_FRAMES),all1003_source_M4_SHA_check=True,
        fixed_transform='M4 uint8>127 -> torch bilinear512 align_cornersFalse antialiasTrue -> avgpool16 -> float16',
        current_or_real_reads=0,RGB_reads=0,M0_reads=0,new_fits=0,new_model_forwards=0))
    for n in ('MASK_TARGETS.npz','FIT_DRAWS1000x8.npy','M4_SOURCE_BINDINGS.json'):
        p=checked(c['outputs'][n]);dest=safe(out/n)
        with dest.open('xb')as f:f.write(p.read_bytes())
        assert binding(dest)['sha256']==c['outputs'][n]['sha256']
    with np.load(safe(out/'MASK_TARGETS.npz'),allow_pickle=False)as z:
        target=np.array(z['mask'],copy=True);fit=np.array(z['fit'],copy=True);ids=np.array(z['frame_ids'],copy=True)
    assert target.shape==(1003,1,32,32)and target.dtype==np.float16
    assert fit.dtype==bool and np.array_equal(fit,np.arange(1003)<768)and np.array_equal(ids,np.arange(1003))
    draws=np.load(safe(out/'FIT_DRAWS1000x8.npy'),allow_pickle=False)
    assert np.array_equal(draws,np.random.Generator(np.random.PCG64(17)).choice(np.flatnonzero(fit),size=(1000,8),replace=True))
    assert fit[draws].all()
    sources=json.loads(safe(out/'M4_SOURCE_BINDINGS.json').read_bytes())['M4'];assert len(sources)==1003
    replay=[];h=hashlib.sha256()
    for j,b in enumerate(sources):
        p=checked(b);h.update(b['sha256'].encode())
        if j not in REPLAY_FRAMES:continue
        y=np.asarray(Image.open(io.BytesIO(p.read_bytes())).convert('L'))>127
        x=torch.from_numpy(y.astype(np.float32))[None,None]
        expected=F.avg_pool2d(F.interpolate(x,(512,512),mode='bilinear',align_corners=False,antialias=True),16,16)[0].numpy().astype(np.float16)
        assert np.array_equal(expected,target[j])
        replay.append(dict(frame_index=j,original_hw=list(y.shape),target_FP16_exact=True,source_M4=b))
    frequency=float(target[fit].astype(np.float64).mean());assert frequency==c['FIT_foreground_frequency']
    write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_complete=binding(cp),
        code=binding(Path(__file__).absolute()),outputs={n:binding(out/n)for n in ('MASK_TARGETS.npz','FIT_DRAWS1000x8.npy','M4_SOURCE_BINDINGS.json','PLAN_BEFORE_TARGET_REPLAY.json')},
        frames=1003,FIT=768,DEV=235,arrays=dict(mask=dict(shape=list(target.shape),dtype=str(target.dtype)),fit=dict(shape=list(fit.shape),dtype=str(fit.dtype)),frame_ids=dict(shape=list(ids.shape),dtype=str(ids.dtype))),
        all1003_original_M4_SHA_verified=True,M4_SHA_sequence_digest=h.hexdigest(),ten_original_M4_exact_replays=replay,
        FIT_foreground_frequency=frequency,positive_weight=.5/frequency,negative_weight=.5/(1-frequency),
        draw_PCG64_17_exact=True,draws_FIT_only=True,whole_original_M4_geometry=True,
        new_fits=0,new_model_forwards=0,current_or_real_reads=0,RGB_reads=0,M0_reads=0,query_native_pose_reads=0))
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'))),flush=True)


if __name__=='__main__':main()
