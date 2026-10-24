from pathlib import Path
import sys,time,json,csv
import numpy as np,torch
R=Path(__file__).resolve().parent;OLD=R.parent/'external_ray_proxy_head_20261022_v60';sys.path.insert(0,str(OLD));from common import read,sha,utc
from head_and_roi import ProxyHead
assert read(OLD/'REAL_VERIFIED_PARITY_V2.json')['PASS'];assert sha(R/'BEST_PRESENCE_HEAD.pt')==read(R/'MODEL_FREEZE.json')['checkpoint_sha256'];torch.set_num_threads(1);m=ProxyHead().cuda();m.load_state_dict(torch.load(R/'BEST_PRESENCE_HEAD.pt',map_location='cuda'));m.eval();m.requires_grad_(False);before={k:v.clone()for k,v in m.state_dict().items()};x=np.load(OLD/'REAL_ROI_CACHE.npz');u=read(OLD/'PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz');out=[];start=time.monotonic()
with torch.no_grad():
 for i in range(0,len(u),32):out.append(m(torch.from_numpy(x['x'][i:i+32]).reshape(-1,65,9,9).cuda().float()).reshape(-1,8).cpu().numpy())
z=np.concatenate(out);p=1/(1+np.exp(-z));assert all(torch.equal(v,before[k])for k,v in m.state_dict().items());np.savez_compressed(R/'REAL_PRESENCE_OUTPUTS.npz',logits=z,probability=p,pixel_valid=x['pixel_valid']);rr=[]
for i,f in enumerate(u):
 H,W=f['raw_hw']
 for k,q in enumerate(f['initial_points'][:8]):rr.append(dict(id=f['id'],group=f['group'],native_corner=k,query_x=q[0],query_y=q[1],presence_probability=float(p[i,k]),predicted_present=bool(p[i,k]>=.5),valid=bool(0<=q[0]<W and 0<=q[1]<H and x['pixel_valid'][i,k]),target_mask_truth='UNAVAILABLE; no quantitative real presence accuracy'))
with(R/'REAL_QUERY_PREDICTIONS.csv').open('x')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
with(R/'REAL_INFERENCE_COMPLETE.json').open('x')as f:json.dump(dict(UTC=utc(),frames=len(u),query_outputs=len(rr),seconds=time.monotonic()-start,frozen_head_unchanged=True,real_target_masks_available=False,real_accuracy_computed=False,real_fit_updates=0,new_backbone_forward=0,head_sha256=sha(R/'BEST_PRESENCE_HEAD.pt')),f,indent=2)
print('REAL_PRESENCE_INFERENCE_ONLY_COMPLETE',len(u),flush=True)
