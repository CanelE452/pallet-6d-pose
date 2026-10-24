"""Source-cache schema and deterministic CUDA backward without optimizer updates."""
import argparse
import json
from pathlib import Path
import numpy as np
from fit_global_amodal_head import (load_cache,source_balanced_loss,source_balanced_global_loss,sha256)
import torch
from query_global_amodal_head import QueryGlobalAmodalHead,parameter_count


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
 p.add_argument('--batch-size',type=int,default=256);p.add_argument('--device',default='cuda');args=p.parse_args()
 out=Path(args.out)
 if out.exists():raise FileExistsError(out)
 data=load_cache(args.cache);n=min(args.batch_size,len(data['x']));ix=np.arange(n)
 targets,inverse=np.unique(data['target_index'][ix],return_inverse=True)
 torch.set_num_threads(1);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 def tensor(key,ix,dtype=torch.float32):return torch.as_tensor(np.asarray(data[key][ix]),device=args.device,dtype=dtype)
 if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
 model=QueryGlobalAmodalHead().to(args.device)
 inputs=dict(x=tensor('x',ix),rgb_native=tensor('rgb_native',ix)/255,sam=tensor('sam',ix),context=tensor('context',ix),
  box_grid=tensor('box_grid',ix),global_input=torch.cat((tensor('global_rgb',targets)/255,tensor('global_sam',targets),tensor('global_prompt',targets)),1),
  target_inverse=torch.as_tensor(inverse,device=args.device),global_index64=tensor('global_index64',ix,torch.long),
  global_index8=tensor('global_index8',ix,torch.long),global_query_valid=tensor('global_query_valid',ix,torch.bool))
 center,dense,global_logits=model(**inputs,return_dense=True)
 y=tensor('joint',ix,torch.long);v=tensor('dense_valid',ix,torch.bool);codes=torch.zeros(n,device=args.device,dtype=torch.long)
 loss=source_balanced_loss(center,y[:,4,4],v[:,4,4],torch.ones(n,device=args.device),codes)
 loss=loss+.25*source_balanced_loss(dense,y,v,torch.ones_like(y,dtype=torch.float32),codes)
 loss=loss+.25*source_balanced_global_loss(global_logits,tensor('global_joint',targets,torch.long),tensor('global_valid',targets,torch.bool),torch.zeros(len(targets),device=args.device,dtype=torch.long))
 loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters())
 result={'PASS':True,'cache':args.cache,'cache_sha256':sha256(args.cache),'batch_queries':n,'unique_targets':len(targets),
  'parameters':parameter_count(model),'deterministic_backward':True,'loss':float(loss.detach().cpu()),'optimizer_updates':0,
  'peak_cuda_memory_allocated':torch.cuda.max_memory_allocated()if args.device.startswith('cuda')else None,
  'torch':torch.__version__,'code_sha256':{name:sha256(Path(__file__).with_name(name))for name in ('query_global_amodal_head.py','fit_global_amodal_head.py',Path(__file__).name)},
  'private_data_read':False,'PnP_calls':0}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
