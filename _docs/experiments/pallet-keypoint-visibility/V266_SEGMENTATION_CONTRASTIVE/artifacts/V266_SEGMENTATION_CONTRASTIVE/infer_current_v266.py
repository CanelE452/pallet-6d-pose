"""Sealed SYN finals -> unchanged current223 tokens; annotations are never read."""
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

HERE = Path(__file__).absolute().parent
PREV = HERE.parent/'V264_HEATMAP_LOCAL_VISIBILITY'
ARMS = ('SEGMENTATION', 'SEGMENTATION_CONTRAST')


def safe(value):
    p = Path(value)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):
        assert not a.is_symlink(),str(a)
    return p


def bind(path):
    p=safe(path); raw=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def read(b):
    raw=safe(b['path']).read_bytes()
    assert len(raw)==b['size_bytes'] and hashlib.sha256(raw).hexdigest()==b['sha256']
    return raw


def write(path,value):
    with safe(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')


def load_module(binding,name):
    m=types.ModuleType(name);m.__file__=binding['path'];sys.modules[name]=m
    exec(compile(read(binding),binding['path'],'exec'),m.__dict__)
    return m


def main():
    freeze=json.loads(safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_bytes())
    assert freeze['completed'] and freeze['PASS'] and freeze['threshold_E_logit']==0
    protocol=json.loads(read(freeze['protocol']))
    assert bind(Path(__file__).absolute())==protocol['inference_code']
    m=load_module(freeze['model_code'],'v266_current_model')
    oldfreeze=json.loads(safe(PREV/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    old=[json.loads(x) for x in read(oldfreeze['predictions']).splitlines()]
    features=np.load(io.BytesIO(read(oldfreeze['feature_packet'])),allow_pickle=False)
    complete=json.loads(read(oldfreeze['current_source_complete']))
    tok=complete['outputs']['SPATIAL_TOKENS_FP16.npy'];sup=complete['outputs']['TOKEN_SUPPORT.npy']
    assert bind(tok['path'])==tok and bind(sup['path'])==sup
    tokens=np.load(safe(tok['path']),mmap_mode='r',allow_pickle=False)
    support=np.load(safe(sup['path']),allow_pickle=False)
    auxiliary=features['auxiliary'];uv=features['uv']
    base=np.array([r['HEATMAP_LOCAL']['E_logit'] for r in old],np.float32)
    available=np.array([r['available'] for r in old],bool)
    assert len(tokens)==len(old)==1784 and available.sum()==1739
    out=safe(HERE/'current_inference_v1');assert not out.exists();out.mkdir()
    torch.set_num_threads(1);torch.manual_seed(17)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    device='cuda:0'
    probabilities={};descriptors={};predictions={}
    for arm in ARMS:
        model=m.OwnerVisibilityResidual().to(device).eval()
        cp=torch.load(io.BytesIO(read(freeze['checkpoints'][arm])),map_location='cpu')
        model.load_state_dict(cp['state_dict'],strict=True)
        arrays={'E_logit':[],'dense_probabilities':[],'descriptor':[]}
        with torch.inference_mode():
            for start in range(0,1784,64):
                sl=slice(start,start+64)
                result=model(torch.as_tensor(np.array(tokens[sl],copy=True),device=device).float(),
                    torch.as_tensor(support[sl],device=device).bool(),
                    torch.as_tensor(auxiliary[sl],device=device).float(),
                    torch.as_tensor(uv[sl],device=device).float(),
                    torch.as_tensor(base[sl],device=device).float())
                for k in arrays:arrays[k].append(result[k].cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in arrays.items()}
        logits=arrays['E_logit']; assert np.isfinite(logits).all()
        probabilities[arm]=arrays['dense_probabilities']
        descriptors[arm]=arrays['descriptor']
        predictions[arm]=(logits,np.where(available,np.where(logits>0,1,0),-1).astype(np.int8))
    np.savez_compressed(out/'DENSE_PROBABILITIES.npz',**probabilities)
    np.savez_compressed(out/'READOUT_DESCRIPTORS.npz',**descriptors,base_E_logit=base,available=available,
        auxiliary=auxiliary,uv=uv)
    # Bounded full-dense forward replay fixture; chosen from input indices before GT join.
    selected=np.arange(0,1784,56,dtype=np.int64)
    np.savez_compressed(out/'FORWARD_REPLAY_INPUTS32.npz',source_index=selected,
        tokens=np.array(tokens[selected],copy=True),support=support[selected],auxiliary=auxiliary[selected],
        uv=uv[selected],base_E_logit=base[selected],
        **{arm+'_E_logit':predictions[arm][0][selected] for arm in ARMS})
    records=[]
    for i,row in enumerate(old):
        r={k:row[k] for k in ('global_index','row_id','frame_id','native_slot','available')}
        r['CURRENT_V264']=row['HEATMAP_LOCAL']
        for arm in ARMS:
            logit=float(predictions[arm][0][i]);p=int(predictions[arm][1][i])
            r[arm]=dict(E_logit=logit,p_E=float(1/(1+np.exp(-np.clip(logit,-700,700)))),
                pred={0:'D',1:'E',-1:'U'}[p])
        records.append(r)
    predpath=safe(HERE/'CURRENT_ALL1784_PREDICTIONS.jsonl')
    with predpath.open('x') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
    receipt=dict(completed=True,PASS=True,configuration=bind(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'),
        predictions=bind(predpath),dense_predictions=bind(out/'DENSE_PROBABILITIES.npz'),
        descriptor_predictions=bind(out/'READOUT_DESCRIPTORS.npz'),
        forward_replay_inputs=bind(out/'FORWARD_REPLAY_INPUTS32.npz'),
        code=bind(Path(__file__).absolute()),current_source_complete=oldfreeze['current_source_complete'],
        V264_predictions=oldfreeze['predictions'],rows=1784,supported=1739,current_GT_or_annotation_reads=0,
        optimizer_updates=0,new_backbone_forwards=0,new_spatial_localization_forwards=0,new_PnP=0,
        new_segmentation_forward_rows_per_arm=1784,location_coordinates_unchanged=True,
        dense_class_order=['TARGET_PALLET','FOREIGN_OBJECT','BACKGROUND'])
    write(HERE/'CURRENT_PREDICTIONS_FROZEN.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
