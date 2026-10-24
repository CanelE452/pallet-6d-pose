"""Replay frozen v63 source SAM; add actual radius2 logit readouts.

Queries, image/box/alternative selection and original exact logit feature grid
must replay the preserved source overlay. No labels/GT mask paths enter here.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import cv2
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
OUT=ROOT/'sam_source';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

assert not(OUT/'OVERLAY.npz').exists()
inputs=V63/'sam_source/INPUTS.jsonl';source_lock=json.loads((V63/'sam_source/LOCK.json').read_text())
assert sha(inputs)==source_lock['input_sha256']
rows=[json.loads(s)for s in inputs.read_text().splitlines()]
denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
assert not({r['image_sha256']for r in rows}&denied)
with np.load(V63/'sam_source/OVERLAY.npz')as z:old_ids=z['id'].astype(str);old=z['sam']
lookup={s:i for i,s in enumerate(old_ids)}
loader=V63/'sam/standalone_mobile_sam.py'
spec=importlib.util.spec_from_file_location('v64_frozen_sam_loader',loader);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
model,predictor=module.load_model('cuda:0')
write(OUT/'STARTED.json',dict(input_sha256=sha(inputs),original_overlay_sha256=sha(V63/'sam_source/OVERLAY.npz'),sam_checkpoint_sha256=module.CHECKPOINT_SHA,
                             loader_sha256=sha(loader),code_sha256=sha(__file__),targets=len(rows),private_input_accessed=False,PnP_calls=0))
values=[];ids=[];hashes=[];splits=[];sources=[];previous=None;frames=0;start=time.monotonic();replay_errors=0
with torch.inference_mode():
 for i,r in enumerate(rows):
    assert not any('mask'in k for k in r)
    if previous!=r['image']:
        assert sha(r['image'])==r['image_sha256']
        image=np.asarray(Image.open(r['image']).convert('RGB'));assert list(image.shape[:2])==r['raw_hw']
        predictor.set_image(image);previous=r['image'];frames+=1
    alternatives,quality,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True)
    assert np.isfinite(quality).all();logit=alternatives[int(np.argmax(quality))]
    near=cv2.dilate(logit,np.ones((5,5),np.uint8))
    h,w=r['raw_hw'];q=np.asarray(r['query'],float);a=(np.arange(9)-4)*r['side']/9;dx,dy=np.meshgrid(a,a)
    xy=q[:,None,None]+np.stack([dx,dy],-1)[None];pxy=np.floor(xy+.5).astype(np.int64)
    valid=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(pxy[...,0]>=0)&(pxy[...,0]<w)&(pxy[...,1]>=0)&(pxy[...,1]<h)
    grid=[]
    for full in [logit,near]:
        sampled=full[np.clip(pxy[...,1],0,h-1),np.clip(pxy[...,0],0,w-1)]
        grid.append(np.where(valid,np.clip(sampled/8,-4,4),0).astype(np.float16))
    grid=np.stack(grid,axis=1)
    replay_errors+=int(np.count_nonzero(grid[:,0:1]!=old[lookup[r['id']]]))
    assert np.array_equal(grid[:,0:1],old[lookup[r['id']]]),('source exact replay changed',r['id'])
    assert(grid[:,1]>=grid[:,0]).all()
    values.append(grid);ids.append(r['id']);hashes.append(r['image_sha256']);splits.append(r['split']);sources.append(r['source'])
    if(i+1)%200==0:print('TWO_SCALE_SAM',i+1,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
np.savez_compressed(OUT/'OVERLAY.npz',id=ids,sam=np.asarray(values),image_sha256=hashes,split=splits,source=sources)
write(OUT/'SEAL.json',dict(schema='SOURCE_SAM_EXACT_RADIUS2_OVERLAY_V1',targets=len(ids),frames=frames,seconds=time.monotonic()-start,
                        overlay_sha256=sha(OUT/'OVERLAY.npz'),input_sha256=sha(inputs),sam_checkpoint_sha256=module.CHECKPOINT_SHA,
                        original_exact_grid_cells_checked=int(old.size),original_exact_grid_mismatches=replay_errors,GT_masks_read=0,current_evaluation_accessed=False,new_training_updates=0,new_PnP_calls=0,
                        radius2='clipped5x5 max of the SAME selected full-resolution logit before native9 nearest sampling; not neighboring native9 cells'))
print(json.dumps(json.loads((OUT/'SEAL.json').read_text())),flush=True)
