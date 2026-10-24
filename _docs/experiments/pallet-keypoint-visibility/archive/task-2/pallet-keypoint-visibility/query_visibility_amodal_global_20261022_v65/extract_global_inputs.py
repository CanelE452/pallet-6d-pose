"""Replay frozen source SAM into whole target crops without reading GT masks."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import cv2
import numpy as np
import torch
from PIL import Image
from global_recipe import grid,native_valid

ROOT=Path(__file__).resolve().parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
OUT=ROOT/'sam_source'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

if __name__=='__main__':
    assert not (OUT/'GLOBAL_INPUTS.npz').exists()
    inputs=V63/'sam_source/INPUTS.jsonl'
    lock=json.loads((V63/'sam_source/LOCK.json').read_text())
    assert sha(inputs)==lock['input_sha256']
    rows=[json.loads(s) for s in inputs.read_text().splitlines()]
    denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    assert not ({r['image_sha256'] for r in rows}&denied)
    with np.load(V64/'sam_source/OVERLAY.npz') as z:
        old=z['sam'];lookup={s:i for i,s in enumerate(z['id'].astype(str))}
    loader=V63/'sam/standalone_mobile_sam.py'
    spec=importlib.util.spec_from_file_location('v65_frozen_sam_loader',loader)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    model,predictor=m.load_model('cuda:0')
    write(OUT/'STARTED.json',dict(input_sha256=sha(inputs),code_sha256=sha(__file__),recipe_sha256=sha(ROOT/'global_recipe.py'),sam_checkpoint_sha256=m.CHECKPOINT_SHA,GT_masks_read=0,current_eval_accessed=False,training_updates=0,PnP_calls=0))
    values={k:[] for k in ['global_rgb','global_sam','global_prompt','global_valid','global_id','global_image_sha256','global_box','source','split']}
    previous=None;start=time.monotonic();frames=0;checks=0
    with torch.inference_mode():
      for i,r in enumerate(rows):
        assert not any('mask' in k for k in r)
        if previous!=r['image']:
            assert sha(r['image'])==r['image_sha256']
            rgb=np.asarray(Image.open(r['image']).convert('RGB'))
            assert list(rgb.shape[:2])==r['raw_hw']
            predictor.set_image(rgb);previous=r['image'];frames+=1
        logits,quality,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True)
        assert np.isfinite(quality).all()
        logit=logits[int(np.argmax(quality))]
        h,w=r['raw_hw'];q=np.asarray(r['query'],float);a=(np.arange(9)-4)*r['side']/9
        xx,yy=np.meshgrid(a,a);localxy=q[:,None,None]+np.stack([xx,yy],-1)[None]
        p=np.floor(localxy+.5).astype(np.int64);good=native_valid(localxy,p,[h,w])
        for ch,full in enumerate([logit,cv2.dilate(logit,np.ones((5,5),np.uint8))]):
            out=np.where(good,np.clip(full[np.clip(p[...,1],0,h-1),np.clip(p[...,0],0,w-1)]/8,-4,4),0).astype(np.float16)
            assert np.array_equal(out,old[lookup[r['id']],:,ch]),r['id'];checks+=out.size
        xy,p,_,_=grid(r['box']);valid=native_valid(xy,p,[h,w])
        rgb_grid=cv2.remap(rgb,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        sampled=np.clip(logit[np.clip(p[...,1],0,h-1),np.clip(p[...,0],0,w-1)]/8,-4,4)
        sampled=np.where(valid,sampled,0).astype(np.float16)
        x0,y0,x1,y1=r['box']
        prompt=(xy[...,0]>=x0)&(xy[...,0]<=x1)&(xy[...,1]>=y0)&(xy[...,1]<=y1)&valid
        values['global_rgb'].append(rgb_grid.transpose(2,0,1));values['global_sam'].append(sampled[None])
        values['global_prompt'].append(prompt[None]);values['global_valid'].append(valid)
        for key,field in [('global_id','id'),('global_image_sha256','image_sha256'),('global_box','box'),('source','source'),('split','split')]:values[key].append(r[field])
        if(i+1)%200==0:print('GLOBAL_SOURCE_SAM',i+1,len(rows),round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(OUT/'GLOBAL_INPUTS.npz',**{k:np.asarray(v)for k,v in values.items()})
    write(OUT/'SEAL.json',dict(targets=len(rows),frames=frames,seconds=time.monotonic()-start,input_sha256=sha(inputs),global_inputs_sha256=sha(OUT/'GLOBAL_INPUTS.npz'),local_exact_and_r2_replay_cells=checks,local_replay_mismatches=0,GT_masks_read=0,current_eval_accessed=False,training_updates=0,PnP_calls=0,recipe_sha256=sha(ROOT/'global_recipe.py')))
    print(json.dumps(json.loads((OUT/'SEAL.json').read_text())),flush=True)
