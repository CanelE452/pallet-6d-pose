"""Sealed full-image RGB/CLIP refinement; query lookup after all maps saved.

No backbone optimization, query-centred image crop, GT read, coordinate change,
mask dilation, threshold search or optimizer is present in this program.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import types
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
ARMS = ('HIGHRES_REFINE', 'BOUNDARY_REFINE')


def safe(p):
    p = Path(p)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink(), str(a)
    return p


def binding(p):
    p = safe(p); h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def read(b):
    p = safe(b['path']); raw = p.read_bytes()
    assert len(raw) == b['size_bytes'] and hashlib.sha256(raw).hexdigest() == b['sha256'], str(p)
    return raw


def module(b, name):
    m = types.ModuleType(name); m.__file__ = b['path']; sys.modules[name] = m
    exec(compile(read(b), m.__file__, 'exec'), m.__dict__)
    return m


def write(p, obj):
    with safe(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def state_sha(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        x = t.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def full_rgb256(rgb):
    """Full RGB512 bicubic then2x2 average, /255 and frozen fp16 storage."""
    raw = np.asarray(rgb)
    assert raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 3
    square = np.array(Image.fromarray(raw).resize((512,512), Image.Resampling.BICUBIC), copy=True)
    x = torch.from_numpy(square).permute(2,0,1).float()[None]
    return (F.avg_pool2d(x,2,2)/255)[0].numpy().astype(np.float16)


def sample_queries(maps, queries):
    result = np.zeros(len(queries), np.float32)
    for i, q in enumerate(queries):
        if not q['available']:
            continue
        xy = np.asarray(q['original_q0'], np.float64); h,w = q['raw_hw']
        assert xy.shape == (2,) and np.isfinite(xy).all()
        grid = torch.tensor((xy+.5)/np.array([w,h])*2-1,dtype=torch.float32).reshape(1,1,1,2)
        field = torch.from_numpy(maps[int(q['frame_index'])]).reshape(1,1,*maps.shape[-2:])
        result[i] = float(F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])
    return result


def main():
    torch.set_num_threads(1); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    freeze = json.loads(safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_bytes())
    assert freeze['PASS'] and freeze['threshold'] == .5
    policy = json.loads(read(freeze['protocol']))
    assert binding(Path(__file__).absolute()) == policy['inference_code']
    api = module(freeze['model_code'], 'v268_frozen_refiner')
    ioapi = module(policy['current_input_helper'], 'v268_existing_original_rgb_io')
    inputs = json.loads(read(policy['current_input_complete']))
    assert inputs['PASS'] and inputs['accessible_verified_frames'] == 223
    frames = [json.loads(x) for x in read(inputs['frames']).splitlines()]
    assert [r['frame_index'] for r in frames] == list(range(223))
    # These arrays were frozen in V267 before its GT join; no current GT input.
    tokb = policy['current_CS_tokens']; assert binding(tokb['path']) == tokb
    tokens = np.load(safe(tokb['path']),mmap_mode='r',allow_pickle=False)
    with np.load(io.BytesIO(read(policy['V267_current_frame_masks'])),allow_pickle=False) as z:
        coarse = np.array(z['FULL_CLIP_SURGERY_ARCH_logit'],copy=True)[:,None]
    assert tokens.shape == (223,1024,512) and coarse.shape == (223,1,32,32)
    out = safe(HERE/'current_inference_v1'); assert not out.exists(); out.mkdir()
    rgb = np.empty((223,3,256,256),np.float16)
    for i,frame in enumerate(frames):
        decoded=ioapi.read_rgb(frame)
        assert tuple(decoded.shape[:2]) == tuple(frame['raw_hw'])
        rgb[i]=full_rgb256(decoded)
    fixture_ix=np.array([0,56,112,168,222],np.int64)
    fixture=dict(frame_index=fixture_ix,rgb256=rgb[fixture_ix],tokens=np.array(tokens[fixture_ix],copy=True),coarse_logits=coarse[fixture_ix])
    maps={}; proofs={}; bindings={}
    device='cuda:0'
    for arm in ARMS:
        model=api.FullFrameRefiner().to(device).eval().requires_grad_(False)
        cp=torch.load(io.BytesIO(read(freeze['checkpoints'][arm])),map_location='cpu')
        model.load_state_dict(cp['state_dict'],strict=True)
        initial=state_sha(model)
        values=[]; delta_max=0.; protected_violations=0; finite=True
        with torch.inference_mode():
            for start in range(0,223,8):
                ix=np.arange(start,min(start+8,223))
                x=torch.as_tensor(np.array(rgb[ix],copy=True),device=device,dtype=torch.float32)
                t=torch.as_tensor(np.array(tokens[ix],copy=True),device=device,dtype=torch.float32)
                c=torch.as_tensor(np.array(coarse[ix],copy=True),device=device,dtype=torch.float32)
                packet=model(x,t,c)
                raw=packet['mask_logits']; delta=packet['bounded_delta']
                assert raw.shape == delta.shape == (len(ix),1,256,256)
                assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(delta).all())
                base=F.interpolate(c,(256,256),mode='bilinear',align_corners=False)
                assert bool(torch.allclose(raw,base+delta,atol=1e-6,rtol=1e-6))
                dm=float(delta.abs().max()); assert dm <= 6.000001; delta_max=max(delta_max,dm)
                protected=(base.abs()>6)
                protected_violations+=int(((raw>=0)!=(base>=0))[protected].sum())
                values.append(raw[:,0].cpu().numpy())
                if start%40 == 0: print(json.dumps(dict(arm=arm,full_frames=start+len(ix),total=223)),flush=True)
        raw=np.concatenate(values);assert raw.shape==(223,256,256)
        assert state_sha(model)==initial and protected_violations==0
        path=safe(out/(arm+'_FULL_MASKS.npz'))
        np.savez_compressed(path,frame_index=np.arange(223,dtype=np.int64),logits=raw)
        assert path.stat().st_size < 100*1024*1024
        bindings[arm]=binding(path);maps[arm]=raw;fixture[arm+'_logit']=raw[fixture_ix]
        proofs[arm]=dict(checkpoint=freeze['checkpoints'][arm],head_state_sha256=initial,
            full_frame_forwards=223,largest_absolute_residual=delta_max,
            sign_changes_where_same256grid_coarse_abs_gt6=protected_violations,
            old_direct32query_bound_claim=False,source_RGB_is_full_frame=True,new_optimizer_updates=0)
        del model
    np.savez_compressed(safe(out/'FULL_FRAME_FORWARD_REPLAY5.npz'),**fixture)
    write(HERE/'CURRENT_FRAME_MASKS.json',dict(completed=True,PASS=True,arms=bindings,resolution=256,
        storage='Perarmfull256rawfloat32logits; probability=sigmoid(logits).Noquantization/postdilation.',frames=223))
    mapseal=binding(HERE/'CURRENT_FRAME_MASKS.json')
    write(out/'FULL_FRAME_MAPS_SAVED_BEFORE_QUERY_SAMPLING.json',dict(completed=True,PASS=True,maps=mapseal,
        per_arm_maps=bindings,current_GT_reads=0,q_or_native_inputs_to_model=0,new_optimizer_updates=0,new_backbone_forwards=0))
    # Query sidecar is read only after both sets of full-frame maps are sealed.
    queries=[json.loads(x)for x in read(inputs['queries_manifest']).splitlines()]
    assert len(queries)==1784 and sum(r['available']for r in queries)==1739
    sampled={a:sample_queries(maps[a],queries)for a in ARMS}
    base256=F.interpolate(torch.from_numpy(coarse),(256,256),mode='bilinear',align_corners=False)[:,0].numpy()
    coarse32score=sample_queries(coarse[:,0],queries);coarse256score=sample_queries(base256,queries)
    valid=np.array([q['available']for q in queries],bool)
    drift=np.abs(coarse32score-coarse256score)
    write(out/'NOOP_256_RESAMPLING_DIAGNOSTIC_BEFORE_GT.json',dict(completed=True,PASS=True,
        valid_queries=int(valid.sum()),label_changes_from_32_to_256_then_query=int(((coarse32score>=0)!=(coarse256score>=0))[valid].sum()),
        max_absolute_logit_difference=float(drift[valid].max()),mean_absolute_logit_difference=float(drift[valid].mean()),
        current_GT_reads=0,new_fits_or_threshold_choice=0,scope='Numericalresamplingcontrolonly;no-opnotlearnedmodelorselectablepolicy.'))
    predictions=[]
    for i,row in enumerate(queries):
        r={k:row[k]for k in('global_index','frame_index','frame_id','row_id','native_slot','available')}
        for arm in ARMS:
            x=float(sampled[arm][i]);p=float(1/(1+np.exp(-np.clip(x,-80,80))))
            r[arm]=dict(p_target=p,E_logit=-x,p_E=1-p,pred='U'if not row['available']else'D'if x>=0 else'E',interpretation='MASK_PRESENCE_PROXY_E_NOT_CAUSE')
        predictions.append(r)
    predpath=safe(HERE/'CURRENT_ALL1784_PREDICTIONS.jsonl')
    with predpath.open('x')as f:
        for row in predictions:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    write(HERE/'CURRENT_PREDICTIONS_FROZEN.json',dict(completed=True,PASS=True,
        configuration=binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'),predictions=binding(predpath),frame_masks=mapseal,
        current_frame_inputs=inputs['frames'],current_query_inputs=inputs['queries_manifest'],current_input_complete=policy['current_input_complete'],
        code=binding(Path(__file__).absolute()),model_proofs=proofs,feature_bindings=dict(CS_tokens=tokb,coarse_maps=policy['V267_current_frame_masks']),
        forward_replay_inputs=binding(out/'FULL_FRAME_FORWARD_REPLAY5.npz'),rows=1784,frames=223,supported=1739,
        fixed_threshold=.5,sample_padding_mode='border',sample_align_corners=False,raw_probability_maps='sigmoid(raw256logits); qlookuprawbefore sigmoid',
        original_query_native_order_unchanged=True,current_GT_annotation_or_real_fit_reads=0,new_optimizer_updates=0,new_PnP=0,
        mask_presence_is_not_external_occlusion_cause=True,whole_image_scope='Frozenfull512CLIPfeatures +entireRGB512bicubic→256avgpool;fullmaskssavedbeforeqlookup.'))
    print(json.dumps(dict(completed=True,PASS=True,freeze=binding(HERE/'CURRENT_PREDICTIONS_FROZEN.json'))),flush=True)


if __name__=='__main__':
    main()
