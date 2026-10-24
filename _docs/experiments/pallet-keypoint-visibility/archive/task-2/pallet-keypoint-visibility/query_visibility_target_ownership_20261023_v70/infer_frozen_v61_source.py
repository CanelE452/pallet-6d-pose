"""One frozen V61 head inference on the same source VAL cached P3 crops.

No labels, RGB, feature extraction, model fitting, or current evaluation reads.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
TASK = P.parent
V66 = P/'query_visibility_controlled_real_20261022_v66'
CHECKPOINT_SHA = 'c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def ref(path):
    return {'path':str(Path(path).resolve()),'sha256':sha(path)}

def checked(binding, expected=None):
    path=Path(binding['path']).resolve()
    if expected is not None:
        assert path==Path(expected).resolve()
    assert sha(path)==binding['sha256']
    return path

def state_hash(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()

def run(entry_path):
    entry_path=Path(entry_path).resolve()
    lock=json.loads(entry_path.read_text())
    assert lock['schema']=='V70_FROZEN_V61_SOURCE_ENTRYLOCK_V1'
    assert lock['approved'] and lock['source_only'] and not lock['current223_allowed']
    assert lock['new_training_allowed'] is False
    checked(lock['code'],__file__)
    checkpoint=checked(lock['checkpoint'],TASK/'target_presence_classifier_20261022_v61/BEST_PRESENCE_HEAD.pt')
    assert lock['checkpoint']['sha256']==CHECKPOINT_SHA
    modulefile=checked(lock['model_code'],TASK/'external_ray_proxy_head_20261022_v60/head_and_roi.py')
    cache=checked(lock['cache'],V66/'joined_source_cache/CONTROLLED_PLUS_PALLOX_VAL.npz')
    checked(lock['cache_prior_seal'],V66/'joined_source_cache/SEAL.json')
    checked(lock['input_seal'],P/'query_visibility_sam21_frozen_20261023_v68/source_inputs/SEAL.json')
    deny=checked(lock['denylist'],P/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json')
    out=Path(lock['out']).resolve()
    assert out.is_relative_to(ROOT/'source_evaluation') and not out.exists()
    prior=json.loads(Path(lock['cache_prior_seal']['path']).read_text())
    registered=next(x for x in prior['outputs'] if x['candidate']=='CONTROLLED_PLUS_PALLOX' and x['split']=='VAL')
    assert registered['sha256']==lock['cache']['sha256']
    # These fields are exclusively frozen input features and identity metadata.
    keys=('x','sample_id','source','group_id','image_sha256','global_id','target_index')
    with np.load(cache,allow_pickle=False) as z:
        data={k:z[k] for k in keys}
    x=data['x']; n=len(x)
    assert n==8590 and x.shape==(8590,65,9,9) and x.dtype==np.float16
    assert len(data['global_id'])==1091 and len(set(data['sample_id']))==n
    assert not(set(data['image_sha256']) & set(json.loads(deny.read_text())['image_sha256']))
    assert np.isfinite(x).all() and np.isin(x[:,64],(0,1)).all()
    native_valid=x[:,64,4,4]>.5
    assert int(native_valid.sum())==8584
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    spec=importlib.util.spec_from_file_location('preserved_v61_head',modulefile)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.ProxyHead().cpu().eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    assert sum(p.numel() for p in model.parameters())==28033
    before=state_hash(model)
    indices=np.flatnonzero(native_valid)
    logits=np.full(n,np.nan,np.float64)
    start=time.monotonic();batches=0
    with torch.inference_mode():
        for i in range(0,len(indices),256):
            ix=indices[i:i+256]
            logits[ix]=model(torch.from_numpy(x[ix]).float()).cpu().numpy().astype(np.float64)
            batches+=1
    probability=np.full(n,np.nan,np.float64)
    probability[native_valid]=1/(1+np.exp(-np.clip(logits[native_valid],-80,80)))
    assert np.isfinite(probability[native_valid]).all() and np.isnan(probability[~native_valid]).all()
    assert state_hash(model)==before and sha(checkpoint)==CHECKPOINT_SHA
    assert sha(__file__)==lock['code']['sha256']
    out.mkdir(parents=True)
    npz=out/'SOURCE_OUTPUTS.npz'
    np.savez_compressed(npz,sample_id=data['sample_id'],source=data['source'],group_id=data['group_id'],
        image_sha256=data['image_sha256'],global_id=data['global_id'][data['target_index']],
        native_valid=native_valid,logit=logits,pV=probability)
    result={'schema':'V70_FROZEN_V61_SAME_SOURCE_VAL_OUTPUT_SEAL','completed':True,
        'source_only':True,'queries':n,'targets':1091,'native_valid':8584,'native_invalid_U':6,
        'scope':'Frozen original V61 head, same source native65x9x9 P3 crops. Exact output only; radius2 is not manufactured.',
        'bindings':{'entry':ref(entry_path),'code':ref(__file__),'checkpoint':ref(checkpoint),
            'model_code':ref(modulefile),'cache':lock['cache'],'cache_prior_seal':lock['cache_prior_seal'],
            'input_seal':lock['input_seal'],'denylist':lock['denylist']},
        'outputs':[ref(npz)],'model_state_sha256':before,'final_model_state_sha256':state_hash(model),
        'model_unchanged':True,'parameters':28033,'threshold':.5,'comparison':'>=',
        'batch_size':256,'forward_batches':batches,'seconds':time.monotonic()-start,
        'current_RGB_GT_scores_inputs_read':False,'source_RGB_or_GT_read':False,
        'cached_input_keys':list(keys),'feature_extractor_forward_calls':0,
        'optimizer_updates':0,'PnP_calls':0,'checkpoint_or_threshold_selected':False}
    seal=out/'SEAL.json';seal.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'seal':ref(seal),'seconds':result['seconds']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--entry-lock',required=True)
    run(parser.parse_args().entry_lock)
