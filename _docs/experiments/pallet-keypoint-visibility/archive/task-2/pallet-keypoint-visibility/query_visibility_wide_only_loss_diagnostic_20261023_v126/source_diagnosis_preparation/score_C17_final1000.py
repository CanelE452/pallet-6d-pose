"""C17 immutable fixed-final1000 source scorer. No fit, RGB, encoder or current."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

V=Path(__file__).resolve().parents[1]
OUT=V/'source_diagnosis/C17_FIXED_FINAL1000_NOT_SELECTOR'
DESIGN=V/'ROOT_C18_DESIGN_BEFORE_SOURCE_FIT.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def state_sha(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main(entry):
    e=read(entry)
    assert e['approved_source_only_fixed_checkpoint_forward']
    assert all(not e[k]for k in ('fit_allowed','current_allowed','V112_allowed','PnP_allowed'))
    assert e['code']==bind(__file__) and e['design']==bind(DESIGN)
    d=read(check(e['design']))
    for k in ('parent_C17_final_checkpoint','parent_C17_fit','head_class_code','old_features','old_flat_features','wide_features'):
        check(d[k])
    assert not OUT.exists(),'Preserve any previous attempt; no automatic restart'
    import torch
    assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
    assert str(Path(sys.executable).resolve())=='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True)
    write(OUT/'STARTED.json',{'entry':bind(entry),'fixed_checkpoint_update':1000,'no_restart':True})
    start=time.monotonic()
    try:
        packs=[];offset=0;units=[];available=[];factual=[];context=[];features=[]
        for key,n,u in [('old',46108,15004),('old_flat',4896,4428),('wide',4896,4428)]:
            with np.load(d[key+'_features']['path'],allow_pickle=False)as f:
                x=f['features'].copy();um=f['occurrence_unit'].copy();a=f['query_available'].copy()
                fa=f['factual_query_available'].copy();ca=f['context_available'].copy();indices=f['global_occurrence_index']
            assert x.shape==(u,1928)and x.dtype==np.float32 and np.isfinite(x).all()
            assert um.shape==(n,)and a.shape==fa.shape==ca.shape==(n,)
            assert np.array_equal(indices,np.arange(n)) and np.array_equal(a,fa&ca)
            assert um.min()>=0 and um.max()<u
            features.append(torch.as_tensor(x,device='cuda'))
            units.append(offset+um);available.append(a);factual.append(fa);context.append(ca);offset+=u
        unit=np.concatenate(units);a=np.concatenate(available);fa=np.concatenate(factual);ca=np.concatenate(context)
        assert len(unit)==len(a)==55900 and offset==23860
        tree=ast.parse(Path(d['head_class_code']['path']).read_text())
        cls=[z for z in tree.body if isinstance(z,ast.ClassDef)and z.name=='SpatialTargetQueryHead']
        assert len(cls)==1
        ns={'torch':torch};exec(compile(ast.Module(body=cls,type_ignores=[]),d['head_class_code']['path']+'::class_only','exec'),ns)
        model=ns['SpatialTargetQueryHead']().cuda().float().eval().requires_grad_(False)
        payload=torch.load(d['parent_C17_final_checkpoint']['path'],map_location='cpu',weights_only=True)
        assert payload['update']==1000 and payload['head_class_code']==d['head_class_code']
        model.load_state_dict(payload['state_dict'],strict=True);del payload
        assert sum(p.numel()for p in model.parameters())==221825
        before=state_sha(model)
        with torch.inference_mode():
            raw=torch.cat([model(x)for x in features])
            repeat=torch.cat([model(x)for x in features])
            assert torch.equal(raw,repeat)
        assert state_sha(model)==before
        uraw=raw.cpu().numpy().astype(np.float32)
        z=np.where(a,uraw[unit],0).astype(np.float32)
        pred=np.where(a,(z>=0).astype(np.int8),-1).astype(np.int8)
        path=OUT/'RAW_C17_FINAL1000_SOURCE_SCORES_INPUT_ONLY.npz'
        np.savez_compressed(path,query_logit=z,prediction=pred,query_available=a,
                            factual_query_available=fa,context_available=ca,
                            input_unit_index=unit,unit_raw_logit=uraw,global_occurrence_index=np.arange(55900))
        write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),
                design=bind(DESIGN),checkpoint=d['parent_C17_final_checkpoint'],fixed_update=1000,
                selected_update_not_used=900,output=bind(path),head_class_code=d['head_class_code'],
                separate_evaluation_units=[15004,4428,4428],head_full_passes=2,head_matrix_calls=6,
                repeat_exact=True,state_before=before,state_after=state_sha(model),threshold_logit=0,
                GT_or_source_labels_read_before_predictions=False,fit_encoder_RGB_current_V112_PnP_calls=0,
                checkpoint_SHA_after=sha(d['parent_C17_final_checkpoint']['path']),seconds=time.monotonic()-start))
        assert sha(d['parent_C17_final_checkpoint']['path'])==d['parent_C17_final_checkpoint']['sha256']
        print(json.dumps(read(OUT/'PREDICTION_SEAL.json')),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'no_automatic_restart':True})
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True,type=Path)
    main(ap.parse_args().entry)
