"""Frozen source-cache scoring. Does not access current evaluation data."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'))
from query_head import load_model
from fit_query_head import load_cache,binary_metrics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_hash(m):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()for _,v in sorted(m.state_dict().items()))).hexdigest()


ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);ap.add_argument('--v61',action='store_true');ap.add_argument('--device',default='cpu');args=ap.parse_args()
out=Path(args.out);out.mkdir(exist_ok=True,parents=True);assert not(out/'SOURCE_OUTPUTS.npz').exists()
torch.set_num_threads(1)
d=load_cache(args.cache)
deny=set(json.load(open(ROOT/'contract/EVALUATION_SHA256_DENYLIST.json'))['image_sha256'])
assert not(set(d['image_sha256'])&deny)
if args.v61:
    modulefile=ROOT.parent.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
    spec=importlib.util.spec_from_file_location('preserved_v61_head',modulefile);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=module.ProxyHead().to(args.device);m.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=True));m.eval();config=None
else:m,payload=load_model(args.checkpoint,args.device);config=m.config
m.requires_grad_(False);before=state_hash(m);indices=np.flatnonzero(d['valid']);logits=[];start=time.monotonic()
with torch.no_grad():
    for i in range(0,len(indices),256):
        ix=indices[i:i+256];x=torch.from_numpy(d['x'][ix]).float().to(args.device)
        if config is None:z=m(x)
        else:
            rgb=torch.from_numpy(d['rgb'][ix]).float().to(args.device)if config.rgb_branch else None
            context=torch.from_numpy(d['context'][ix]).float().to(args.device)if config.target_context else None
            sam=torch.from_numpy(d['sam'][ix]).float().to(args.device)if config.sam_branch else None
            z=m(x,rgb,context=context,sam=sam)
        logits.append(z.cpu().numpy())
assert state_hash(m)==before
z=np.concatenate(logits).astype(np.float64);p=1/(1+np.exp(-np.clip(z,-80,80)))
np.savez_compressed(out/'SOURCE_OUTPUTS.npz',index=indices,logits=z,probability=p,y=d['y'][indices],source=d['source'][indices])
reports=[]
for source in sorted(set(d['source'][indices])):
    mask=d['source'][indices]==source
    reports.append({'source':source,**binary_metrics(d['y'][indices][mask],p[mask])})
result={'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_sha256':sha(args.checkpoint),'source_cache_sha256':sha(args.cache),'source_metrics':reports,'macro_source_BCE':float(np.mean([x['BCE']for x in reports])),'threshold':.5,'model_parameters_unchanged':True,'eval_images_accessed':False,'fit_updates':0,'PnP_calls':0,'seconds':time.monotonic()-start}
(out/'SOURCE_INFERENCE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
