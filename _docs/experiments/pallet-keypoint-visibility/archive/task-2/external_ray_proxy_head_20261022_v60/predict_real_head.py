"""Inference only. No real targets, fit, calibration, normalization update."""
from common import *
import time
import numpy as np,torch
from head_and_roi import ProxyHead
torch.set_num_threads(1);assert read(ROOT/'REAL_VERIFIED_PARITY_V2.json')['PASS'];freeze=read(ROOT/'MODEL_FREEZE.json');assert sha(ROOT/'BEST_HEAD.pt')==freeze['checkpoint_sha256'];cache=np.load(ROOT/'REAL_ROI_CACHE.npz');x=cache['x'];net=ProxyHead().cuda();net.load_state_dict(torch.load(ROOT/'BEST_HEAD.pt',map_location='cuda'));net.eval();net.requires_grad_(False);before={k:v.clone()for k,v in net.state_dict().items()};out=[];start=time.monotonic()
with torch.no_grad():
 for j in range(0,len(x),32):out.append(net(torch.from_numpy(x[j:j+32]).reshape(-1,65,9,9).cuda().float()).reshape(-1,8).cpu().numpy())
 # Repeat fixed first batch to verify unchanged deterministic inference.
 repeat=net(torch.from_numpy(x[:32]).reshape(-1,65,9,9).cuda().float()).reshape(-1,8).cpu().numpy()
z=np.concatenate(out);assert np.array_equal(repeat,z[:32]);assert all(torch.equal(v,before[k])for k,v in net.state_dict().items());np.savez_compressed(ROOT/'REAL_HEAD_OUTPUTS.npz',logits=z,probability=1/(1+np.exp(-z)),pixel_valid=cache['pixel_valid']);save('REAL_HEAD_INFERENCE_COMPLETE.json',dict(UTC=utc(),frames=len(x),seconds=time.monotonic()-start,repeat_first32_max_logit_difference=float(np.max(abs(repeat-z[:32]))),frozen_weights_unchanged=True,optimizer_updates=0,reference_reads=0,checkpoint_sha256=sha(ROOT/'BEST_HEAD.pt')));print('REAL_FROZEN_HEAD_COMPLETE',len(x),flush=True)
