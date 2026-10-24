"""Frozen source-cache cross-domain scoring, no optimizer/private/PnP imports."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'))
from fit_multitarget_head import load_cache

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(m):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()for _,v in sorted(m.state_dict().items()))).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not(out/'SOURCE_OUTPUTS.npz').exists()
freeze=json.loads((run/'MODEL_FREEZE.json').read_text());checkpoint=run/'BEST_MULTITARGET_HEAD.pt';assert sha(checkpoint)==freeze['checkpoint_sha256']
code=run/'code/query_multitarget_head.py';spec=importlib.util.spec_from_file_location('_v64_frozen_source_head',code);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
d=load_cache(a.cache);deny=set(json.loads((ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256']);assert not(set(d['image_sha256'])&deny)
torch.set_num_threads(1);model,payload=module.load_model(checkpoint,'cpu');model.eval().requires_grad_(False);before=state_sha(model);ix=np.flatnonzero(d['dense_valid'][:,4,4]);values=[];start=time.monotonic()
with torch.inference_mode():
 for offset in range(0,len(ix),256):
    ii=ix[offset:offset+256];inputs={k:torch.from_numpy(np.asarray(d[k][ii])).float()for k in ['x','rgb_native','sam','context','box_grid']};inputs['rgb_native']/=255
    values.append(model(**inputs).numpy())
assert state_sha(model)==before
z=np.concatenate(values).astype(np.float64);p=np.exp(z-np.logaddexp.reduce(z,axis=1)[:,None]);mm=module.marginals(p);y=d['joint'][ix,4,4]
np.savez_compressed(out/'SOURCE_OUTPUTS.npz',index=ix,joint_logits=z,truth_joint=y,truth_classes=module.joint_to_classes(y),source=d['source'][ix],**mm)
reports=[]
for source in sorted(set(d['source'][ix])):
 mask=d['source'][ix]==source;reports.append(dict(source=str(source),N=int(mask.sum()),joint_NLL=float(-np.log(np.clip(p[mask][np.arange(mask.sum()),y[mask]],1e-12,1)).mean())))
report=dict(checkpoint_sha256=sha(checkpoint),cache_sha256=sha(a.cache),code_sha256=sha(code),N=len(ix),source_reports=reports,macro_source_joint_NLL=float(np.mean([r['joint_NLL']for r in reports])),model_state_unchanged=True,current_eval_accessed=False,new_updates=0,PnP_calls=0,seconds=time.monotonic()-start)
(out/'SOURCE_INFERENCE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
