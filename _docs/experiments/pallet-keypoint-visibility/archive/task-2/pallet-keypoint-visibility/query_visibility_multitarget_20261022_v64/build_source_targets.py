"""Source-only exact/nearby visible and full-support query supervision.

No current evaluation RGB, reference coordinate, human state or score is read.
All original P3/context fields and query positions are retained. RGB features
are raw 5x5 native-pixel patches centered at each *labelled integer pixel*.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
V61=ROOT.parent.parent/'target_presence_classifier_20261022_v61'
OUT=ROOT/'source_cache';OUT.mkdir(exist_ok=True,parents=True)
JOINT=np.array([[0,-1,-1],[1,2,-1],[3,4,5]],np.int8)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvrows(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def integer_grid(q,side,hw):
    q=np.asarray(q,float);h,w=hw
    offset=(np.arange(9)-4)*side/9;dx,dy=np.meshgrid(offset,offset)
    xy=q[:,None,None]+np.stack((dx,dy),-1)[None]
    pixel=np.floor(xy+.5).astype(np.int64)
    valid=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)
    valid&=(pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h)
    return xy,pixel,valid

def sample_mask(mask,pixel):
    h,w=mask.shape
    return mask[np.clip(pixel[...,1],0,h-1),np.clip(pixel[...,0],0,w-1)]

def local_rgb(rgb,pixel):
    """Output N,75,9,9 ordered by RGB channel, dy(-2..2), dx(-2..2)."""
    h,w=rgb.shape[:2];patches=[]
    for dy in range(-2,3):
        for dx in range(-2,3):
            px=pixel[...,0]+dx;py=pixel[...,1]+dy
            valid=(px>=0)&(px<w)&(py>=0)&(py<h)
            value=rgb[np.clip(py,0,h-1),np.clip(px,0,w-1)].copy();value[~valid]=0
            patches.append(value.transpose(0,3,1,2))
    # N,C,25,H,W -> N,(C*25),H,W.
    return np.stack(patches,axis=2).reshape(len(pixel),75,9,9)

def target_grid(xy,side,box):
    x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
    assert bw>=2 and bh>=2
    a=np.stack((2*(xy[...,0]-x0)/bw-1,2*(xy[...,1]-y0)/bh-1,
                np.full(xy.shape[:-1],np.log2(side/bw)),np.full(xy.shape[:-1],np.log2(side/bh))),axis=1)
    return np.clip(a,-4,4).astype(np.float32)

def make(source,split):
    output=OUT/f'{source}_{split}.npz';assert not output.exists()
    inp=V63/'source_cache/v2'/f'{source}_{split}.npz'
    with np.load(inp,allow_pickle=False)as z:old={k:z[k]for k in z.files}
    assert old['x'].shape[1:]==(65,9,9)
    denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    assert not(set(old['image_sha256'].astype(str))&denied)
    inputs={r['id']:r for r in[json.loads(s)for s in(V63/'sam_source/INPUTS.jsonl').read_text().splitlines()]}
    if source=='SYNTH':
        masks={r['id']:(r['visible_path'],r['amodal_path'])for r in csvrows(V61/'MASK_CONTRACT_AUDIT.csv')};threshold=127
    else:
        manifest=V63/'ycb'/('train_instances.jsonl'if split=='TRAIN'else'SOURCE_VAL_LOCKED_200.jsonl')
        raw=[json.loads(s)for s in manifest.read_text().splitlines()]
        masks={f"YCB:{r['scene_id']:06d}:{r['frame_id']:06d}:{r['instance_id']:06d}":(r['mask_vis'],r['mask_full'])for r in raw};threshold=0
    n=len(old['x']);classes=np.empty((n,2,9,9),np.int8);valid=np.empty((n,9,9),bool);rgb_native=np.empty((n,75,9,9),np.uint8);box_grid=np.empty((n,4,9,9),np.float32)
    class_boundary=np.empty((n,9,9),bool);query_rows=[];previous=None;checks=0;start=time.monotonic()
    groups={}
    for i,s in enumerate(old['id'].astype(str)):groups.setdefault(s,[]).append(i)
    for gi,(ident,ix)in enumerate(groups.items()):
        r=inputs[ident];assert r['source']==source and r['image_sha256']not in denied
        assert all(old['image_sha256'][j]==r['image_sha256']for j in ix)
        if previous!=r['image']:
            assert sha(r['image'])==r['image_sha256']
            rgb=np.array(Image.open(r['image']).convert('RGB'));previous=r['image']
        assert list(rgb.shape[:2])==r['raw_hw'];h,w=r['raw_hw']
        vis=np.array(Image.open(masks[ident][0]))>threshold;full=np.array(Image.open(masks[ident][1]))>threshold
        assert vis.shape==full.shape==(h,w)and not(vis&~full).any()
        v2=cv2.dilate(vis.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        f2=cv2.dilate(full.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        # One-pixel boundary diagnostic, never a GT relabel or hidden loss mask.
        ext=full&~vis
        state=np.where(vis,0,np.where(full,1,2)).astype(np.uint8)
        boundary=cv2.dilate(state,np.ones((3,3),np.uint8))!=cv2.erode(state,np.ones((3,3),np.uint8))
        q=np.asarray(r['query'],float);xy,pixel,good=integer_grid(q,r['side'],[h,w])
        target=np.stack([np.where(sample_mask(v,pixel),0,np.where(sample_mask(f,pixel),1,2))for v,f in [(vis,full),(v2,f2)]],axis=1).astype(np.int8)
        joint=JOINT[target[:,0],target[:,1]];assert (joint[good]>=0).all()
        patches=local_rgb(rgb,pixel);bg=target_grid(xy,r['side'],r['box']);bd=sample_mask(boundary,pixel)
        for j in ix:
            k=int(old['native_corner'][j]);classes[j]=target[k];valid[j]=good[k];rgb_native[j]=patches[k];box_grid[j]=bg[k];class_boundary[j]=bd[k]
            assert good[k,4,4]and (target[k,0,4,4]==0)==bool(old['y'][j])
            assert np.array_equal(good[k],old['dense_valid'][j])
            assert np.array_equal((target[k,0]==0).astype(np.uint8),old['dense_y'][j])
            px,py=pixel[k,4,4]
            assert np.array_equal(patches[k,[12,37,62],4,4],rgb[py,px]);checks+=1
            query_rows.append(dict(cache_row_index=j,sample_id=str(old['sample_id'][j]),id=ident,source=source,split=split,query_index=k,query_x=float(q[k,0]),query_y=float(q[k,1]),pixel_x=int(px),pixel_y=int(py),side=r['side'],exact_state=int(target[k,0,4,4]),radius2_state=int(target[k,1,4,4]),joint_state=int(joint[k,4,4]),native_boundary=bool(boundary[py,px]),semantic_self='UNKNOWN',image_sha256=r['image_sha256']))
        if(gi+1)%200==0:print('SOURCE_TARGETS',source,split,gi+1,len(groups),round(time.monotonic()-start,1),flush=True)
    assert checks==n
    # Preserve original IDs/P3/context; redundant old exact labels remain for audits.
    arrays={k:v for k,v in old.items()if k not in ['dense_y','boundary','dense_valid']}
    arrays.update(classes=classes,joint=JOINT[classes[:,0],classes[:,1]],dense_valid=valid,rgb_native=rgb_native,box_grid=box_grid,class_boundary=class_boundary)
    np.savez_compressed(output,**arrays)
    query_rows.sort(key=lambda r:r['cache_row_index'])
    with output.with_name(output.stem+'_QUERIES.csv').open('w',newline='')as f:wrt=csv.DictWriter(f,fieldnames=list(query_rows[0]));wrt.writeheader();wrt.writerows(query_rows)
    center=arrays['joint'][:,4,4];counts={str(c):int((center==c).sum())for c in range(6)}
    audit=dict(source=source,split=split,queries=n,groups=len(groups),center_joint_counts=counts,exact_RGB_center_checks=checks,original_P3_context_and_query_labels_preserved=True,
               valid_dense_cells=int(valid.sum()),source_input=dict(path=str(inp),sha256=sha(inp)),output_sha256=sha(output),code_sha256=sha(__file__),
               target='exact and clipped5x5 any target support, each visible/full; support difference is external proxy, not semantic self',
               rgb='5x5 native RGB patch centered at each same nearest pixel as the dense target; channel order color*25+dy*5+dx',
               box='input-only same rough box passed to frozen SAM; relativeXY and log2(side/boxsize), no target mask used',
               eval_data_read=False,training_updates=0,new_PnP_calls=0,seconds=time.monotonic()-start)
    write(output.with_name(output.stem+'_MANIFEST.json'),audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['SYNTH','YCB'],required=True);ap.add_argument('--split',choices=['TRAIN','VALIDATION'],required=True);a=ap.parse_args();cv2.setNumThreads(1);make(a.source,a.split)
