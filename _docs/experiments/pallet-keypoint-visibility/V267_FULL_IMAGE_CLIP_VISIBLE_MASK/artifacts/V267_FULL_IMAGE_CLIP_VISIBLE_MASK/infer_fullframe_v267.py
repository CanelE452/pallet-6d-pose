"""Full-image CLIP masks; native q is sampled only after maps are saved.

M4 supervises visible target presence, not native semantic external occlusion.
The D/E mapping below is explicitly a mask-presence proxy for evaluation.
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
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
ARMS={'FULL_CLIP':'PLAIN_CLIP','FULL_CLIP_SURGERY_ARCH':'CLIP_SURGERY_ARCH'}


def safe(value):
    p=Path(value)
    assert p.is_absolute() and not any(s in str(p).lower()for s in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def bind(path):
    p=safe(path);h=hashlib.sha256()
    with p.open('rb')as f:
        for raw in iter(lambda:f.read(1048576),b''):h.update(raw)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def read(b):
    raw=safe(b['path']).read_bytes()
    assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256']
    return raw


def module(b,name):
    m=types.ModuleType(name);m.__file__=b['path'];sys.modules[name]=m
    exec(compile(read(b),m.__file__,'exec'),m.__dict__);return m


def write(path,obj):
    with safe(path).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def sample_queries(logit_maps,queries):
    result=np.zeros(len(queries),np.float32)
    valid=np.array([r['available']for r in queries],bool)
    for i,row in enumerate(queries):
        if not valid[i]:continue
        q=np.asarray(row['original_q0'],np.float64);h,w=row['raw_hw']
        assert np.isfinite(q).all()and q.shape==(2,)
        # Raw pixel centres -> full-frame normalized coordinates. No crop.
        uv=(q+.5)/np.array([w,h],np.float64)
        grid=torch.as_tensor(uv*2-1,dtype=torch.float32).reshape(1,1,1,2)
        field=torch.as_tensor(logit_maps[int(row['frame_index'])]).reshape(1,1,32,32)
        result[i]=float(F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])
    return result


def main():
    freeze=json.loads(safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_bytes())
    assert freeze['completed']and freeze['PASS']and freeze['threshold']==.5
    protocol=json.loads(read(freeze['protocol']))
    assert bind(Path(__file__).absolute())==protocol['inference_code']
    encoder=module(protocol['encoder_code'],'v267_current_fullimage_encoder')
    head_api=module(freeze['model_code'],'v267_current_fullmask_head')
    ioapi=module(protocol['current_input_helper'],'v267_current_input_io')
    inputs=json.loads(read(protocol['current_input_complete']))
    assert inputs['PASS']and inputs['accessible_verified_frames']==223
    frames=[json.loads(x)for x in read(inputs['frames']).splitlines()]
    assert [r['frame_index']for r in frames]==list(range(223))
    out=safe(HERE/'current_inference_v1');assert not out.exists();out.mkdir()
    encoder.configure_runtime();device='cuda:0'
    maps={};proofs={};feature_bindings={};replay={}
    for arm,sourcearm in ARMS.items():
        visual,proof=encoder.load_visual(protocol['generic_CLIP_weights']['path'],sourcearm,device)
        assert proof['frozen_ready_state_sha256']==protocol['frozen_encoder_states'][sourcearm]
        path=safe(out/(arm+'_TOKENS_FP16.npy'))
        tokens=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(223,1024,512))
        for start in range(0,223,2):
            rows=frames[start:start+2]
            rgb=[ioapi.read_rgb(row)for row in rows]
            features=encoder.extract_fullframe_tokens(visual,rgb,device=device,batch_size=2)
            assert np.array_equal(features['raw_hw'],np.asarray([r['raw_hw']for r in rows]))
            tokens[start:start+len(rows)]=features['dense_tokens'].astype(np.float16)
            if start%40==0:print(json.dumps(dict(arm=arm,full_frames_encoded=start+len(rows),total=223)),flush=True)
        tokens.flush()
        assert encoder.state_sha(visual)==proof['frozen_ready_state_sha256']
        del visual;torch.cuda.empty_cache()
        head=head_api.FullMaskHead().to(device).eval().requires_grad_(False)
        state=torch.load(io.BytesIO(read(freeze['checkpoints'][sourcearm])),map_location='cpu')
        head.load_state_dict(state['state_dict'],strict=True)
        before=head_api.state_sha(head.state_dict())
        packet=head_api.infer_mask(head,tokens,device=device,batch_size=32)
        assert head_api.state_sha(head.state_dict())==before
        raw=packet['mask_logits'][:,0];prob=packet['probability'][:,0]
        assert raw.shape==prob.shape==(223,32,32)and np.isfinite(raw).all()
        maps[arm]=prob;maps[arm+'_logit']=raw
        # Fixed full-frame fixture selection from input indices, not annotations.
        selected=np.array([0,56,112,168,222],np.int64)
        replay[arm+'_tokens']=np.array(tokens[selected],copy=True)
        replay[arm+'_logit']=raw[selected]
        proofs[arm]=dict(encoder=proof,head_state_sha256=before,checkpoint=freeze['checkpoints'][sourcearm])
        del tokens,head;feature_bindings[arm]=bind(path)
    replay['frame_index']=selected
    np.savez_compressed(safe(out/'FULL_FRAME_FORWARD_REPLAY5.npz'),**replay)
    np.savez_compressed(safe(HERE/'CURRENT_FRAME_MASKS.npz'),frame_index=np.arange(223,dtype=np.int64),**maps)
    # All full-frame maps are sealed before the code reads the query sidecar.
    mapseal=bind(HERE/'CURRENT_FRAME_MASKS.npz')
    write(out/'FULL_FRAME_MAPS_SAVED_BEFORE_QUERY_SAMPLING.json',dict(completed=True,PASS=True,
        maps=mapseal,frames=223,per_arm_new_full_image_forwards=223,query_or_label_inputs_to_encoder=0,
        current_human_GT_or_annotation_reads=0,new_optimizer_updates=0))
    queries=[json.loads(x)for x in read(inputs['queries_manifest']).splitlines()]
    assert len(queries)==1784 and [r['global_index']for r in queries]==list(range(1784))
    assert sum(r['available']for r in queries)==1739
    sampled={arm:sample_queries(maps[arm+'_logit'],queries)for arm in ARMS}
    predictions=[]
    for i,row in enumerate(queries):
        r={k:row[k]for k in ('global_index','frame_index','frame_id','row_id','native_slot','available')}
        for arm in ARMS:
            x=float(sampled[arm][i]);p=float(1/(1+np.exp(-np.clip(x,-80,80))))
            r[arm]=dict(p_target=p,E_logit=-x,p_E=1-p,pred='U'if not row['available']else'D'if x>=0 else'E',
                interpretation='MASK_PRESENCE_PROXY_E_NOT_CAUSE')
        predictions.append(r)
    pred_path=safe(HERE/'CURRENT_ALL1784_PREDICTIONS.jsonl')
    with pred_path.open('x')as f:
        for row in predictions:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    receipt=dict(completed=True,PASS=True,configuration=bind(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'),
        predictions=bind(pred_path),frame_masks=mapseal,current_frame_inputs=inputs['frames'],
        current_query_inputs=inputs['queries_manifest'],current_input_complete=protocol['current_input_complete'],
        code=bind(Path(__file__).absolute()),feature_bindings=feature_bindings,model_proofs=proofs,
        forward_replay_inputs=bind(out/'FULL_FRAME_FORWARD_REPLAY5.npz'),rows=1784,frames=223,supported=1739,
        raw_probability_maps='sigmoid(raw32logits); pointreadout samplesRAWlogits before sigmoid',
        sample_padding_mode='border',sample_align_corners=False,fixed_threshold=.5,
        current_GT_annotation_or_real_fit_reads=0,new_optimizer_updates=0,new_PnP=0,
        model_input_scope='Every backbone forward sees entire RGB512square; no query input. KP map sampling occurs after maps saved. Frame metadata includes unused query fields.',
        mask_presence_is_not_external_occlusion_cause=True,original_query_native_order_unchanged=True)
    write(HERE/'CURRENT_PREDICTIONS_FROZEN.json',receipt)
    print(json.dumps(dict(completed=True,PASS=True,freeze=bind(HERE/'CURRENT_PREDICTIONS_FROZEN.json'))),flush=True)


if __name__=='__main__':main()
