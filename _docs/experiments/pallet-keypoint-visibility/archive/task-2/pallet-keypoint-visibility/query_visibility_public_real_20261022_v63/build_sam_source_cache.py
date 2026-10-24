"""Frozen, box-only MobileSAM feature grids for source training only.

GT masks/labels never enter prompts, mask selection, or this inference script.
The YCB rough source bbox and initial synthetic keypoint bounds match the
already registered box baseline. Output is an immutable source-only overlay.
"""
import argparse,csv,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sam'))
from standalone_mobile_sam import load_model,CHECKPOINT_SHA,seed

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt')as f:return json.load(f)
    return json.loads(Path(p).read_text())
def csvrows(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def prepare():
    out=ROOT/'sam_source';out.mkdir(exist_ok=True)
    assert not(out/'INPUTS.jsonl').exists()
    denied=set(read(ROOT/'contract/EVALUATION_SHA256_DENYLIST.json')['image_sha256'])
    v60=ROOT.parent.parent/'external_ray_proxy_head_20261022_v60'
    syn=read(v60/'PREPARED_INPUTS_SYNTH.json.gz')
    sides={r['id']:float(r['side'])for r in read(v60/'SYNTH_FEATURE_METADATA.json')}
    inputs=[];bindings=[]
    for r in syn:
        q=np.asarray(r['initial_points'][:8],float);h,w=r['raw_hw']
        finite=np.isfinite(q).all(1)&~(q==-1).all(1)
        lo=np.maximum(q[finite].min(0),[0.,0.]);hi=np.minimum(q[finite].max(0),[w-1.,h-1.])
        assert finite.sum()>=2 and (hi-lo>=2).all(),r['id']
        inputs.append(dict(id=r['id'],source='SYNTH',split=r['split'],image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],box=np.r_[lo,hi].tolist(),query=q.tolist(),side=sides[r['id']]))
    for split in ['TRAIN','VALIDATION']:
        manifest=ROOT/'ycb'/('train_instances.jsonl'if split=='TRAIN'else'SOURCE_VAL_LOCKED_200.jsonl')
        instances=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip()]
        lookup={(r['scene_id'],r['frame_id'],r['instance_id']):r for r in instances}
        qp=ROOT/'source_cache/v2'/f'YCB_{split}_QUERIES.csv';groups={}
        for r in csvrows(qp):groups.setdefault(r['id'],[]).append(r)
        for ident,qq in sorted(groups.items()):
            qq.sort(key=lambda x:int(x['query_index']));assert len(qq)==8
            a=qq[0];r=lookup[int(a['scene_id']),int(a['frame_id']),int(a['instance_id'])]
            h,w=r['height'],r['width'];bx,by,bw,bh=r['bbox_obj'];center=np.array([bx+(bw-1)/2,by+(bh-1)/2])
            factor=float(np.random.default_rng(seed('SAM_SOURCE_BOX:'+ident)).uniform(.9,1.1));half=np.array([max(1,bw-1),max(1,bh-1)])*factor/2
            box=np.r_[np.maximum(center-half,[0.,0.]),np.minimum(center+half,[w-1.,h-1.])]
            inputs.append(dict(id=ident,source='YCB',split=split.lower(),image=r['image'],image_sha256=r['image_sha256'],raw_hw=[h,w],box=box.tolist(),query=[[float(q['query_x']),float(q['query_y'])]for q in qq],side=float(a['roi_side'])))
        bindings.extend([dict(path=str(p),sha256=sha(p))for p in [manifest,qp]])
    for r in inputs:
        assert r['image_sha256']not in denied
        assert not any('mask'in k for k in r)
    inputs.sort(key=lambda r:(r['source'],r['image'],r['id']))
    with(out/'INPUTS.jsonl').open('w')as f:
        for r in inputs:f.write(json.dumps(r)+'\n')
    lock=dict(schema='FROZEN_SAM_SOURCE_OVERLAY_V1',targets=len(inputs),frames=len(set(r['image_sha256']for r in inputs)),checkpoint_sha256=CHECKPOINT_SHA,
              prompt='box only, no positive query or GT mask prompts',box='unchanged synthetic N0 q8 minmax; YCB source bbox with fixed .9..1.1 scale jitter',
              selected_mask='maximum predicted quality, tie first',feature='selected full-resolution logit nearest at floor(q+offset+.5), offsets(i-4)*side/9; /8 clipped[-4,4], invalid zero',
              labels_accessed_for_inference=False,current_eval_accessed=False,training=False,new_PnP_calls=0,input_sha256=sha(out/'INPUTS.jsonl'),code_sha256=sha(__file__),source_bindings=bindings)
    write(out/'LOCK.json',lock);print(json.dumps(lock),flush=True)

def infer(device):
    import torch
    from PIL import Image
    out=ROOT/'sam_source';lock=read(out/'LOCK.json');assert lock['code_sha256']==sha(__file__)
    assert lock['input_sha256']==sha(out/'INPUTS.jsonl')
    assert not(out/'SEAL.json').exists()
    denied=set(read(ROOT/'contract/EVALUATION_SHA256_DENYLIST.json')['image_sha256'])
    inputs=[json.loads(s)for s in(out/'INPUTS.jsonl').read_text().splitlines()]
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model,predictor=load_model(device);previous=None;ids=[];grids=[];qualities=[];hashes=[];splits=[];sources=[];frames=0;start=time.monotonic()
    with torch.inference_mode():
        for i,r in enumerate(inputs):
            assert r['image_sha256']not in denied
            if previous!=r['image']:
                assert sha(r['image'])==r['image_sha256']
                image=np.asarray(Image.open(r['image']).convert('RGB'));assert list(image.shape[:2])==r['raw_hw']
                predictor.set_image(image);previous=r['image'];frames+=1
            logits,quality,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True)
            assert np.isfinite(quality).all();choice=int(np.argmax(quality));logit=logits[choice];h,w=r['raw_hw']
            q=np.asarray(r['query'],float);offset=(np.arange(9)-4)*r['side']/9;dx,dy=np.meshgrid(offset,offset)
            xy=q[:,None,None,:]+np.stack([dx,dy],-1)[None];pixel=np.floor(xy+.5).astype(int)
            valid=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h)
            grid=logit[np.clip(pixel[...,1],0,h-1),np.clip(pixel[...,0],0,w-1)]
            grid=np.where(valid,np.clip(grid/8,-4,4),0).astype(np.float16)[:,None]
            assert np.isfinite(grid).all()
            ids.append(r['id']);grids.append(grid);qualities.append(quality);hashes.append(r['image_sha256']);splits.append(r['split']);sources.append(r['source'])
            if(i+1)%200==0:print('SAM_SOURCE_FEATURE',i+1,len(inputs),'frames',frames,'seconds',round(time.monotonic()-start,1),flush=True)
    target=out/'OVERLAY.npz';assert not target.exists()
    np.savez_compressed(target,id=ids,sam=np.asarray(grids),quality=np.asarray(qualities),image_sha256=hashes,split=splits,source=sources)
    write(out/'SEAL.json',dict(targets=len(ids),frames=frames,seconds=time.monotonic()-start,checkpoint_sha256=CHECKPOINT_SHA,code_sha256=sha(__file__),overlay_sha256=sha(target),input_sha256=lock['input_sha256'],GT_masks_read=False,current_eval_accessed=False,fit_updates=0,new_PnP_calls=0))
    print(json.dumps(read(out/'SEAL.json')),flush=True)

def merge():
    out=ROOT/'source_cache/v4';out.mkdir(exist_ok=True);overlay=ROOT/'sam_source/OVERLAY.npz';seal=read(ROOT/'sam_source/SEAL.json');assert seal['overlay_sha256']==sha(overlay)
    with np.load(overlay)as z:ids=z['id'].astype(str);grids=z['sam'];hh=z['image_sha256'].astype(str)
    lookup={s:i for i,s in enumerate(ids)};assert len(lookup)==len(ids)
    for split in ['TRAIN','VALIDATION']:
        pieces=[];records=[]
        for source in ['SYNTH','YCB']:
            inp=ROOT/'source_cache/v2'/f'{source}_{split}_RGB.npz'
            with np.load(inp,allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
            ix=np.array([lookup[s]for s in arrays['id'].astype(str)]);assert np.array_equal(hh[ix],arrays['image_sha256'].astype(str))
            arrays['sam']=grids[ix,arrays['native_corner'].astype(int)]
            assert arrays['sam'].shape==(len(ix),1,9,9)
            dest=out/f'{source}_{split}_RGB_SAM.npz';assert not dest.exists();np.savez_compressed(dest,**arrays)
            pieces.append(arrays);records.append(dict(path=str(dest),sha256=sha(dest),source_queries=len(ix),source_input_sha256=sha(inp)))
        merged={k:np.concatenate([a[k]for a in pieces])for k in pieces[0]}
        dest=out/f'MIXED_{split}_RGB_SAM.npz';assert not dest.exists();np.savez_compressed(dest,**merged)
        records.append(dict(path=str(dest),sha256=sha(dest),source_queries=len(merged['y'])))
        write(out/f'{split}_PROVENANCE.json',dict(outputs=records,overlay_sha256=sha(overlay),script_sha256=sha(__file__),all_existing_source_fields_preserved=True,current_eval_used=False))
        print(json.dumps(records),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','infer','merge']);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    {'prepare':prepare,'infer':lambda:infer(args.device),'merge':merge}[args.command]()
