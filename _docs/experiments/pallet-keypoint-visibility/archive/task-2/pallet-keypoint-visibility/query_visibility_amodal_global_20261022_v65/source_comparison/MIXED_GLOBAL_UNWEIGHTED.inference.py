"""Frozen source-cache inference, with held-out evaluation RGB SHA exclusion."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb")as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','cache','out','eval-denylist'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();run=Path(a.run).resolve();out=Path(a.out)
    if any(out.with_suffix(suffix).exists()for suffix in ('.npz','.json','.inference.py')):raise FileExistsError(out)
    freeze=json.loads((run/'MODEL_FREEZE.json').read_text())
    if not freeze.get('completed')or freeze.get('eval_data_used')or freeze.get('PnP_used'):
        raise ValueError('a completed source-only frozen run is required')
    model_path=run/'code'/'query_global_amodal_head.py';trainer_path=run/'code'/'fit_global_amodal_head.py'
    for path in (model_path,trainer_path):
        if sha256(path)!=freeze['code_sha256'][path.name]:raise ValueError('frozen code hash mismatch before import')
    checkpoint=run/'BEST_GLOBAL_AMODAL_HEAD.pt'
    if sha256(checkpoint)!=freeze['checkpoint_sha256']:raise ValueError('checkpoint changed after freeze')
    if sha256(a.eval_denylist)!=freeze['eval_denylist_sha256']:
        raise ValueError('held-out evaluation image denylist changed since fit')
    bindings=freeze.get('source_provenance',{}).get('content',{}).get('bindings',{})
    allowed={value for key,value in bindings.items()if str(key).startswith('source_cache/')and str(key).endswith('_GLOBAL.npz')}
    if not allowed:raise ValueError('frozen source-global cache allowlist missing')
    if sha256(a.cache)not in allowed:raise ValueError('cache content is not in source allowlist; refused before NPZ read')
    head=module(model_path,'query_global_amodal_head');trainer=module(trainer_path,'v65_frozen_source_trainer')
    data=trainer.load_cache(a.cache)
    denied=set(json.loads(Path(a.eval_denylist).read_text())['image_sha256'])
    if not denied:raise ValueError('nonempty held-out evaluation image denylist is required')
    if (set(data['image_sha256'].astype(str))|set(data['global_image_sha256'].astype(str)))&denied:
        raise ValueError('current evaluation image SHA found in purported source cache')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model,payload=head.load_model(checkpoint,a.device)
    indices=np.flatnonzero(data['dense_valid'][:,4,4]);outputs=[]
    def tensor(key,ix,dtype=torch.float32):return torch.as_tensor(np.asarray(data[key][ix]),device=a.device,dtype=dtype)
    with torch.inference_mode():
        for start in range(0,len(indices),a.batch_size):
            ix=indices[start:start+a.batch_size];targets,inverse=np.unique(data['target_index'][ix],return_inverse=True)
            args=dict(x=tensor('x',ix),rgb_native=tensor('rgb_native',ix)/255,sam=tensor('sam',ix),context=tensor('context',ix),
                box_grid=tensor('box_grid',ix),global_input=torch.cat((tensor('global_rgb',targets)/255,tensor('global_sam',targets),tensor('global_prompt',targets)),1),
                target_inverse=torch.as_tensor(inverse,device=a.device),global_index64=tensor('global_index64',ix,torch.long),
                global_index8=tensor('global_index8',ix,torch.long),global_query_valid=tensor('global_query_valid',ix,torch.bool))
            outputs.append(model(**args).cpu().numpy())
    logits=np.concatenate(outputs).astype(np.float64);probs=np.exp(logits-np.logaddexp.reduce(logits,axis=1)[:,None])
    truth=data['joint'][indices,4,4]
    out.parent.mkdir(parents=True,exist_ok=True)
    inference_snapshot=out.with_suffix('.inference.py');shutil.copy2(__file__,inference_snapshot)
    np.savez_compressed(out.with_suffix('.npz'),index=indices,joint_logits=logits,truth_joint=truth,
        truth_classes=head.joint_to_classes(truth),source=data['source'][indices],sample_id=data['sample_id'][indices],
        image_sha256=data['image_sha256'][indices],target_index=data['target_index'][indices],**head.marginals(probs))
    metadata={'format':'V65_FROZEN_SOURCE_OUTPUTS','run':str(run),'checkpoint_sha256':trainer.sha256(checkpoint),
        'cache':str(Path(a.cache).resolve()),'cache_sha256':trainer.sha256(a.cache),'queries':len(indices),
        'source_counts':{str(s):int((data['source'][indices]==s).sum())for s in np.unique(data['source'][indices])},
        'model_code_sha256':trainer.sha256(model_path),'inference_snapshot':str(inference_snapshot.resolve()),'inference_code_sha256':trainer.sha256(inference_snapshot),'eval_denylist_sha256':trainer.sha256(a.eval_denylist),
        'outputs_sha256':trainer.sha256(out.with_suffix('.npz')),'output_sha256':trainer.sha256(out.with_suffix('.npz')),'device':a.device,'torch':torch.__version__,
        'optimizer_updates':0,'model_or_threshold_selection':False,'private_data_read':False,'PnP_calls':0}
    trainer.write_json(out.with_suffix('.json'),metadata);print(json.dumps(metadata))

if __name__=='__main__':main()
