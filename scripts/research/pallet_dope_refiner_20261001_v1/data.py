"""Compact DOPE metadata cache and online, frozen source feature batches."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import time
import cv2
import numpy as np
import torch
from common import *

ARRAY_SPECS = {
 'points':((9,2),'float32'), 'point_valid':((9,),'bool'),
 'boxes':((4,),'float32'), 'input_shape':((2,),'int32'),
 'scale_xy':((2,),'float64'), 'shift_xy':((2,),'float64'),
 'gt_points':((9,2),'float32'), 'gt_valid':((9,),'bool'),
 'matched':((),'bool'), 'iou':((),'float64'), 'detected':((),'bool'),
 'score':((),'float64'), 'done':((),'bool')}

def load_image(record):
    path=Path(record['image']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['image_sha256'],record['id']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    assert image is not None and list(image.shape[:2])==record['prepared_shape_hw'],record['id']
    return image

def iou(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():return 0.
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    area=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
    return float(inter/area) if area>0 else 0.

def target(record,scale,shift):
    assert len(record['targets'])==1
    t=record['targets'][0];wh=np.array(record['prepared_shape_hw'][::-1])
    k=np.array(t['keypoints_normalized'],float)
    valid=(k[:,2]>0)&np.isfinite(k[:,:2]).all(-1)&~(k[:,:2]==-1).all(-1)
    points=k[:,:2]*wh*scale+shift
    points[~valid]=np.nan
    bb=np.array(t['box_xywh_normalized'])*np.tile(wh,2)
    box=np.r_[bb[:2]-bb[2:]/2,bb[:2]+bb[2:]/2]*np.tile(scale,2)+np.tile(shift,2)
    return points,valid,box

def load_arrays(mode='r'):
    return {k:np.load(RAW/'cache'/f'{k}.npy',mmap_mode=mode) for k in ARRAY_SPECS}

def cache_source(adapter):
    protocol=verify_lock();records=read(SOURCE)['records'];n=len(records)
    directory=RAW/'cache';directory.mkdir(parents=True,exist_ok=True)
    completion=DOC/'SOURCE_CACHE_COMPLETE.json'
    if completion.exists():
        c=read(completion)
        assert c['protocol_sha256']==sha(DOC/'PROTOCOL.json')
        for entry in c['arrays']:assert sha(ROOT/entry['path'])==entry['sha256']
        return c
    manifest=directory/'CACHE_MANIFEST.json'
    spec=dict(n=n,protocol_sha256=sha(DOC/'PROTOCOL.json'),source=bound(SOURCE),
              arrays={k:dict(shape=[n,*shape],dtype=dtype) for k,(shape,dtype) in ARRAY_SPECS.items()})
    write(manifest,spec)
    for key,(shape,dtype) in ARRAY_SPECS.items():
        path=directory/f'{key}.npy'
        if not path.exists():
            a=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=(n,*shape))
            a[:]=False if dtype=='bool' else 0;a.flush();del a
    arrays=load_arrays('r+');todo=np.flatnonzero(~arrays['done']);start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(0,len(todo),16):
            rows=todo[begin:begin+16];selected=[records[int(j)] for j in rows]
            images=list(pool.map(load_image,selected))
            for r in selected:assert sha(r['label'])==r['label_sha256'],r['id']
            groups=defaultdict(list)
            for j,image in enumerate(images):groups[tuple(image.shape)].append(j)
            predictions=[None]*len(rows)
            for shape,indices in sorted(groups.items()):
                for off in range(0,len(indices),4):
                    js=indices[off:off+4]
                    out=adapter.infer_batch([images[j] for j in js],source_pre_padded=True,return_features=False)
                    for j,p in zip(js,out):predictions[j]=p
            assert len(predictions)==len(rows)
            for idx,record,pred in zip(rows,selected,predictions):
                # Exact common adapter schema; no GT entered the model call.
                affine=np.asarray(pred['affine_input_to_net'],float)
                scale=np.diag(affine)[:2];shift=affine[:2,2]
                assert np.all(scale>0) and np.allclose(affine[:2,:2],np.diag(scale))
                points=np.asarray(pred['points_net'],np.float32)
                valid=np.asarray(pred['valid'],bool)
                box=np.full(4,np.nan,np.float32) if pred['bbox_net'] is None else np.asarray(pred['bbox_net'],np.float32)
                gt,gv,gtbox=target(record,scale,shift)
                overlap=iou(box,gtbox);matched=overlap>=.5
                values=dict(points=points,point_valid=valid,boxes=box,input_shape=pred['input_shape'][-2:],
                    scale_xy=scale,shift_xy=shift,gt_points=gt,gt_valid=gv,matched=matched,
                    iou=overlap,detected=bool(valid[:8].any()),score=pred['score'])
                for key,value in values.items():arrays[key][idx]=value
            # Commit data before the bitmap; an interrupted chunk is safely recomputed.
            for key,a in arrays.items():
                if key!='done':a.flush()
            arrays['done'][rows]=True;arrays['done'].flush()
            if begin==0 or (begin+16)%256==0 or begin+16>=len(todo):
                status=gpu();progress=dict(complete_rows=int(arrays['done'].sum()),total=n,
                    elapsed_seconds=time.perf_counter()-start,gpu=status)
                write(RAW/'CACHE_PROGRESS.json',progress,freeze=False)
                print('DOPE_SOURCE_CACHE',progress['complete_rows'],n,round(progress['elapsed_seconds'],1),flush=True)
    assert arrays['done'].all()
    parts=np.array([r['partition'] for r in records])
    usable=arrays['matched'] & (arrays['gt_valid'][:,:8]&arrays['point_valid'][:,:8]).any(-1)
    c=dict(complete=True,PASS=True,created_at=now(),protocol_sha256=sha(DOC/'PROTOCOL.json'),
       source=bound(SOURCE),cache_manifest=bound(manifest),arrays=[bound(directory/f'{k}.npy') for k in ARRAY_SPECS],
       rows=n,partition_counts={p:dict(total=int((parts==p).sum()),matched=int((arrays['matched']&(parts==p)).sum()),
            usable=int((usable&(parts==p)).sum()),missing_all_corners=int((~arrays['detected']&(parts==p)).sum()))
            for p in ('train','calibration','selection','heldout')},
       full_feature_cache=False,all_image_and_label_hashes_verified=True,
       source_original_coordinate_system='prepared source pixels; subtract reflect100 for native origin only; residual pixel units unchanged')
    write(completion,c);return c

class Dataset:
    def __init__(self):
        self.records=read(SOURCE)['records'];self.arrays=load_arrays()
        receipt=read(DOC/'SOURCE_CACHE_COMPLETE.json')
        assert receipt['complete'] and self.arrays['done'].all()
        assert receipt['protocol_sha256']==sha(DOC/'PROTOCOL.json') and receipt['source']['sha256']==sha(SOURCE)
        for entry in receipt['arrays']:assert sha(ROOT/entry['path'])==entry['sha256']
        for name,(shape,dtype) in ARRAY_SPECS.items():
            assert self.arrays[name].shape==(len(self.records),*shape) and str(self.arrays[name].dtype)==dtype
        self.partitions=np.array([r['partition'] for r in self.records])
        usable=self.arrays['matched'] & (self.arrays['gt_valid'][:,:8]&self.arrays['point_valid'][:,:8]).any(-1)
        self.train_rows=np.flatnonzero((self.partitions=='train')&usable)
        assert len(self.train_rows)>0,'No usable DOPE source training rows'
        self.validation_rows=np.flatnonzero(self.partitions!='train')
        self.pool=ThreadPoolExecutor(max_workers=4)
    def batch_meta(self,rows,device='cuda'):
        return {k:torch.from_numpy(np.array(self.arrays[k][rows],copy=True)).to(device)
                for k in ('points','point_valid','boxes','input_shape','gt_points','gt_valid')}
    def prepare(self,rows,adapter):
        images=list(self.pool.map(load_image,[self.records[int(i)] for i in rows]))
        return [adapter.prepare(image,source_pre_padded=True) for image in images]
    def batch_features(self,rows,adapter,prepared=None):
        prepared=self.prepare(rows,adapter) if prepared is None else prepared
        groups=defaultdict(list)
        for j,p in enumerate(prepared):groups[tuple(p['tensor'].shape)].append(j)
        p3=[None]*len(rows);p4=[None]*len(rows)
        for shape,indices in sorted(groups.items()):
            x=torch.stack([prepared[j]['tensor'] for j in indices]).to(adapter.device)
            a,b=adapter.features_only(x)
            for local,j in enumerate(indices):
                p3[j]=a[local].detach().to(torch.float16)
                p4[j]=b[local].detach().to(torch.float16)
            del x,a,b
        # VGG ran at the native individual shape; only completed frozen features
        # are zero padded for a mixed-shape head batch, as in historical YOLO P.
        result=self.batch_meta(rows,str(adapter.device))
        for key,features in (('p3',p3),('p4',p4)):
            h=max(a.shape[-2] for a in features);w=max(a.shape[-1] for a in features)
            out=torch.zeros((len(rows),features[0].shape[0],h,w),dtype=torch.float16,device=adapter.device)
            for j,a in enumerate(features):out[j,:,:a.shape[-2],:a.shape[-1]]=a
            result[key]=out
        for j,p in enumerate(prepared):
            affine=np.array(p['affine_input_to_net']);i=int(rows[j])
            assert np.array_equal(affine[:2,2],self.arrays['shift_xy'][i])
            assert np.array_equal(np.diag(affine)[:2],self.arrays['scale_xy'][i])
            assert list(p['tensor'].shape[-2:])==self.arrays['input_shape'][i].tolist()
        return result
    def close(self):self.pool.shutdown()

def orders(data):
    result={}
    for seed in SEEDS:
        p=RAW/f'order_seed{seed}.npy'
        rng=np.random.default_rng(seed);parts=[];n=6000*16
        while n:
            part=rng.permutation(data.train_rows)[:n];parts.append(part);n-=len(part)
        order=np.concatenate(parts).reshape(6000,16)
        if p.exists():assert np.array_equal(np.load(p),order)
        else:
            with p.open('xb') as f:np.save(f,order)
        result[str(seed)]=bound(p)
    write(DOC/'BATCH_ORDER.json',dict(seeds=result,train_rows=len(data.train_rows),
        train_rows_sha256=hashlib.sha256(data.train_rows.astype('<i8').tobytes()).hexdigest(),
        D_P_same_order=True,per_seed_exposures=96000))
    return result
