"""Source-only target-instance query supervision from official YCB-V masks."""
from pathlib import Path
import argparse,csv,hashlib,json,time
import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'source_cache/v2';OUT.mkdir(parents=True,exist_ok=True)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_for(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)


def random_xy(mask,rng,fallback):
    yy,xx=np.where(mask)
    if not len(xx):return fallback.copy()
    i=int(rng.integers(len(xx)))
    # Subpixel jitter preserves the chosen nearest integer pixel.
    return np.array([xx[i],yy[i]],float)+rng.uniform(-.45,.45,2)


def sample_queries(vis,full,key,bbox):
    rng=np.random.default_rng(seed_for(key))
    height,width=vis.shape
    inside=cv2.distanceTransform(vis.astype(np.uint8),cv2.DIST_L2,5)
    outside=cv2.distanceTransform((~vis).astype(np.uint8),cv2.DIST_L2,5)
    fulloutside=cv2.distanceTransform((~full).astype(np.uint8),cv2.DIST_L2,5)
    posboundary=vis&(inside<=2)
    negboundary=(~vis)&(outside<=2)
    fullring=(~full)&(fulloutside<=3)
    external=full&~vis
    yy,xx=np.where(vis);fallback=np.array([xx[len(xx)//2],yy[len(yy)//2]],float)
    masks=[posboundary,posboundary,vis&(inside>=3),external,negboundary,fullring,vis,full]
    names=['visible_boundary','visible_boundary','visible_interior','external_if_available','visible_boundary_negative','outside_full_boundary','rough_bbox_corner_jitter','full_support_any']
    queries=[]
    for k,m in enumerate(masks):
        if k==6:
            bx,by,bw,bh=bbox
            corner=np.array([[bx,by],[bx+bw-1,by],[bx+bw-1,by+bh-1],[bx,by+bh-1]])[int(rng.integers(4))]
            q=corner+rng.normal(0,.05*max(bw,bh),2)
            queries.append(np.clip(q,[0,0],[width-1,height-1]))
            continue
        if not m.any():m=negboundary if k==3 else vis
        q=random_xy(m,rng,fallback)
        q=np.clip(q,[0,0],[width-1,height-1])
        queries.append(q)
    # Annotation-derived source object scale is randomly perturbed. No evaluation
    # annotation is used and deployment retains the original N0 span-based scale.
    L=float(np.clip(.25*max(bbox[2],bbox[3])*rng.uniform(.8,1.2),16,max(vis.shape)))
    return np.array(queries),names,L


def patches(features,queries,L,hw,gain,offset,pad):
    height,width=hw
    a=((np.arange(9)+.5)/9-.5)*L
    xx,yy=np.meshgrid(a,a)
    grid=queries[:,None,None,:]+np.stack([xx,yy],-1)[None]
    continuous=(grid[...,0]>=0)&(grid[...,0]<width)&(grid[...,1]>=0)&(grid[...,1]<height)
    network=(grid+pad)*gain+offset
    index=network/8
    norm=2*(index+.5)/80-1
    with torch.no_grad():
        sampled=F.grid_sample(torch.from_numpy(features).float().expand(8,-1,-1,-1),torch.from_numpy(norm).float(),align_corners=False,padding_mode='zeros')
    x=np.concatenate([sampled.numpy()*continuous[:,None],continuous[:,None].astype(np.float32)],axis=1).astype(np.float16)
    pixel=np.floor(grid+.5).astype(int)
    nearest=(pixel[...,0]>=0)&(pixel[...,0]<width)&(pixel[...,1]>=0)&(pixel[...,1]<height)
    return x,grid,pixel,continuous&nearest


parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--split',choices=['train','validation'],required=True);parser.add_argument('--max-instances-per-frame',type=int,default=2);parser.add_argument('--cache-tag',default='');args=parser.parse_args()
assert not args.cache_tag or args.cache_tag.replace('_','').isalnum()
prefix='YCB_'+(args.cache_tag+'_'if args.cache_tag else'')+args.split.upper()
torch.set_num_threads(1);cv2.setNumThreads(1)
manifest=Path(args.manifest);raw=manifest.read_bytes();rows=[json.loads(s)for s in raw.decode().splitlines()if s.strip()]
frames={}
for r in rows:frames.setdefault((r['scene_id'],r['frame_id']),[]).append(r)
deny=set(json.load(open(ROOT/'contract/EVALUATION_SHA256_DENYLIST.json'))['image_sha256'])
keys=['x','y','valid','dense_y','dense_valid','boundary','source','id','sample_id','native_corner','context','group_id','image_sha256']
data={k:[]for k in keys};metadata=[];skipped=[];start=time.monotonic()
for fi,((scene,frame),instances)in enumerate(sorted(frames.items())):
    feature_path=ROOT/'source_cache/YCB_P3_FRAMES'/f'{scene:06d}_{frame:06d}.npz'
    if not feature_path.exists():raise FileNotFoundError(feature_path)
    with np.load(feature_path)as f:
        features=f['p3'];gain=float(f['gain']);offset=f['offset'];pad=int(f['added_border']);hw=tuple(f['raw_hw']);image_sha=str(f['image_sha256'])
    assert image_sha not in deny
    # Stable source-only subsampling bounds compute while representing instances.
    candidates=[r for r in instances if r['full_pixels']>=128 and r['vis_pixels']>=16]
    candidates.sort(key=lambda r:seed_for(f"20261022:{scene}:{frame}:{r['instance_id']}"))
    for r in candidates[:args.max_instances_per_frame]:
        assert r['image_sha256']==image_sha
        vis=cv2.imread(r['mask_vis'],cv2.IMREAD_UNCHANGED)>0
        full=cv2.imread(r['mask_full'],cv2.IMREAD_UNCHANGED)>0
        assert vis.shape==hw and full.shape==hw and not(vis&~full).any()
        fid=f"YCB:{scene:06d}:{frame:06d}:{r['instance_id']:06d}"
        queries,sampling,L=sample_queries(vis,full,fid,r['bbox_obj'])
        x,grid,pixel,dense_valid=patches(features,queries,L,hw,gain,offset,pad)
        weight=x[:,64].astype(np.float32)
        context=(x[:,:64].astype(np.float32)*weight[:,None]).sum((0,2,3))/max(weight.sum(),1)
        height,width=hw
        dense=vis[np.clip(pixel[...,1],0,height-1),np.clip(pixel[...,0],0,width-1)].astype(np.uint8)
        for k,q in enumerate(queries):
            px,py=np.floor(q+.5).astype(int)
            assert dense_valid[k,4,4]
            y=int(vis[py,px]);assert dense[k,4,4]==y
            stencil=vis[max(0,py-1):min(height,py+2),max(0,px-1):min(width,px+2)]
            boundary=stencil.size!=9 or int(stencil.sum())not in (0,9)
            sample=f'{fid}:{k}'
            values=dict(x=x[k],y=y,valid=True,dense_y=dense[k],dense_valid=dense_valid[k],boundary=boundary,source='YCB',id=fid,sample_id=sample,native_corner=k,context=context,group_id=f'YCB:{scene:06d}:{frame:06d}',image_sha256=image_sha)
            for key,value in values.items():data[key].append(value)
            external=bool(full[py,px]and not vis[py,px])
            metadata.append(dict(sample_id=sample,id=fid,scene_id=scene,frame_id=frame,instance_id=r['instance_id'],object_id=r['object_id'],image=r['image'],image_sha256=image_sha,mask_vis=r['mask_vis'],mask_full=r['mask_full'],query_index=k,query_x=float(q[0]),query_y=float(q[1]),roi_side=L,presence_target=y,full_member=bool(full[py,px]),external_proxy=external,outside_full=not bool(full[py,px]),semantic_self='UNKNOWN',boundary=boundary,sampling=sampling[k],split=args.split))
    if (fi+1)%100==0:print('YCB_QUERY',args.split,fi+1,len(frames),'queries',len(metadata),'seconds',round(time.monotonic()-start,2),flush=True)
arrays={key:np.array(value)for key,value in data.items()}
output=OUT/f'{prefix}.npz'
assert not output.exists(),'refusing to overwrite source cache'
np.savez_compressed(output,**arrays)
with (OUT/f'{prefix}_QUERIES.csv').open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
record={'source_manifest':str(manifest),'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'split':args.split,'frames':len(frames),'queries':len(metadata),'positive':int(arrays['y'].sum()),'external_proxy':sum(r['external_proxy']for r in metadata),'outside_full':sum(r['outside_full']for r in metadata),'target_conditioning':'input-only prototype from8queryROI; no source mask or pose used as neural input','sampling':'source masks choose supervised queries; annotation-derived object scale jittered. Deployment preserves unchanged N0 query/crop.','semantic_self_truth':'UNAVAILABLE; never label mask-zero as self','current_eval_overlap':0,'new_optimizer_updates':0,'output':str(output),'output_sha256':sha(output),'script_sha256':sha(__file__),'seconds':time.monotonic()-start}
(OUT/f'{prefix}_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
