#!/usr/bin/env python3
"""Independent source-only expected pure inference features, CPU logit fixed."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V63=ROOT.parent/'query_visibility_public_real_20261022_v63';V60=ROOT.parent.parent/'external_ray_proxy_head_20261022_v60'
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys=None):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in(z.files if keys is None else keys)}
def independent(rgb,q,L,box,logit):
 h,w=rgb.shape[:2];q=np.asarray(q,float);axis=(np.arange(9)-4)*L/9;xx,yy=np.meshgrid(axis,axis);xy=q[:,None,None]+np.stack([xx,yy],-1)[None];pix=np.floor(xy+.5).astype(int)
 valid=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(pix[...,0]>=0)&(pix[...,0]<w)&(pix[...,1]>=0)&(pix[...,1]<h)
 off=np.array([(dx,dy)for dy in range(-2,3)for dx in range(-2,3)]);loc=pix[...,None,:]+off;inside=(loc[...,0]>=0)&(loc[...,0]<w)&(loc[...,1]>=0)&(loc[...,1]<h);values=rgb[np.clip(loc[...,1],0,h-1),np.clip(loc[...,0],0,w-1)].copy();values[~inside]=0;native=values.transpose(0,4,3,1,2).reshape(8,75,9,9)
 b=np.asarray(box,float);bs=b[2:]-b[:2];mid=(b[:2]+b[2:])/2
 bg=np.stack([2*(xy[...,0]-b[0])/bs[0]-1,2*(xy[...,1]-b[1])/bs[1]-1,np.full(xy.shape[:-1],np.log2(L/bs[0])),np.full(xy.shape[:-1],np.log2(L/bs[1]))],1);bg=np.clip(bg,-4,4).astype(np.float32)
 near=cv2.dilate(logit,np.ones((5,5),np.uint8));sam=np.stack([np.where(valid,np.clip(a[np.clip(pix[...,1],0,h-1),np.clip(pix[...,0],0,w-1)]/8,-4,4),0).astype(np.float16)for a in[logit,near]],1)
 step=1.25*bs/64;a=np.arange(64)+.5-32;gx,gy=np.meshgrid(a,a);gxy=mid+np.stack([gx,gy],-1)*step;gp=np.floor(gxy+.5).astype(int);gv=(gxy[...,0]>=0)&(gxy[...,0]<w)&(gxy[...,1]>=0)&(gxy[...,1]<h)&(gp[...,0]>=0)&(gp[...,0]<w)&(gp[...,1]>=0)&(gp[...,1]<h)
 grgb=cv2.remap(rgb,gxy[...,0].astype(np.float32),gxy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).transpose(2,0,1)
 gsam=np.where(gv,np.clip(logit[np.clip(gp[...,1],0,h-1),np.clip(gp[...,0],0,w-1)]/8,-4,4),0).astype(np.float16)[None]
 prompt=((gxy[...,0]>=b[0])&(gxy[...,0]<=b[2])&(gxy[...,1]>=b[1])&(gxy[...,1]<=b[3])&gv)[None]
 cropxy=(pix-mid)/step+31.5;idx=np.floor(cropxy+.5).astype(int);qv=valid&(idx>=0).all(-1)&(idx<64).all(-1);i64=np.where(qv,idx[...,1]*64+idx[...,0],0);i8=np.where(qv,(idx[...,1]//8)*8+idx[...,0]//8,0)
 qpix=np.floor(q+.5).astype(int);qgood=np.isfinite(q).all(1)&~(q==-1).all(1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)&(qpix[:,0]>=0)&(qpix[:,0]<w)&(qpix[:,1]>=0)&(qpix[:,1]<h)
 return dict(rgb_native=native,box_grid=bg,sam=sam,global_rgb=grgb,global_sam=gsam,global_prompt=prompt,global_valid=gv,global_index64=i64,global_index8=i8,global_query_valid=qv,dense_valid=valid,query_validity=qgood,global_input=np.concatenate([grgb.astype(np.float32)/255,gsam.astype(np.float32),prompt.astype(np.float32)],axis=0))
def drift(current,stored):
 d=current.astype(float)-stored.astype(float);return dict(values=int(d.size),unequal_values=int(np.count_nonzero(d)),max_abs=float(np.abs(d).max()),mean_abs=float(np.abs(d).mean()),foreground_sign_flips=int(((current>0)!=(stored>0)).sum()))
if __name__=='__main__':
 assert not(OUT/'EXPECTED_FEATURES.npz').exists()
 manifest=json.loads((OUT/'FIXTURE_INPUTS.json').read_text());fixtures=manifest['inputs'];logits=load(OUT/'CPU_LOGIT_FIXTURES.npz');expected={};results=[]
 synx=load(V60/'SYNTH_ROI_CACHE.npz',['x'])['x'];synmeta=json.loads((V60/'SYNTH_FEATURE_METADATA.json').read_text());synlookup={r['id']:i for i,r in enumerate(synmeta)}
 caches={}
 for r in fixtures:
  source,split=r['source'],r['split'].upper();key=(source,split)
  if key not in caches:caches[key]=load(ROOT/f'source_cache/{source}_{split}_GLOBAL.npz',['id','native_corner','sample_id','x','context','rgb_native','sam','box_grid','global_id','global_rgb','global_sam','global_prompt','global_valid','global_index64','global_index8','global_query_valid'])
  a=caches[key];i=int(r['fixture_index']);prefix=f'f{i:03d}_';rgb=np.asarray(Image.open(r['image']).convert('RGB'));assert sha(r['image'])==r['image_sha256'];f=independent(rgb,r['query'],r['side'],r['box'],logits[f'logit_{i:03d}']);ix=np.flatnonzero(a['id'].astype(str)==r['id']);corners=a['native_corner'][ix].astype(int);ti=int(np.flatnonzero(a['global_id'].astype(str)==r['id'])[0])
  if source=='SYNTH':x8=synx[synlookup[r['id']]]
  else:
   assert len(ix)==8 and set(corners)==set(range(8));x8=a['x'][ix[np.argsort(corners)]]
  weight=x8[:,64].astype(np.float32);context=(x8[:,:64].astype(np.float32)*weight[:,None]).sum((0,2,3))/max(weight.sum(),1)
  f['context']=context;f['x8']=x8
  same={k:int(np.count_nonzero(f[k][corners]!=a[k][ix]))for k in['rgb_native','box_grid','global_index64','global_index8','global_query_valid']}
  same.update({k:int(np.count_nonzero(f[k]!=a[k][ti]))for k in['global_rgb','global_prompt','global_valid']})
  same['context']=int(np.count_nonzero(context[None]!=a['context'][ix]))
  for k,v in f.items():expected[prefix+k]=v
  results.append(dict(fixture_index=i,id=r['id'],source=source,split=split,source_rows=len(ix),exact_nonSAM_cache_mismatches=same,local_SAM_CPU_vs_stored_GPU=drift(f['sam'][corners],a['sam'][ix]),global_SAM_CPU_vs_stored_GPU=drift(f['global_sam'],a['global_sam'][ti])))
 np.savez_compressed(OUT/'EXPECTED_FEATURES.npz',**expected)
 report=dict(source_only=True,private_data_read=False,GPU_used=False,model_updates=0,code_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'FIXTURE_INPUTS.json'),CPU_logit_sha256=sha(OUT/'CPU_LOGIT_FIXTURES.npz'),expected_features_sha256=sha(OUT/'EXPECTED_FEATURES.npz'),same_logit_recipe_compare_is_separate_from_CPU_GPU_drift=True,all_nonSAM_source_cache_checks_pass=all(not any(r['exact_nonSAM_cache_mismatches'].values())for r in results),fixtures=results)
 (OUT/'EXPECTED_FEATURES_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
