"""Lock source-only sampling, audit GT geometry, cache pretrained output features."""
import argparse, collections, hashlib, time
import cv2
import numpy as np
import torch
from core import *

def freeze():
    if (OUT/'PROTOCOL.json').exists():raise RuntimeError('Protocol already exists; do not overwrite')
    allrows=[json.loads(l) for l in META.open()]
    specs=read(CONTRACT)
    def rank(r):return hashlib.sha256(('dimsym1:'+r['merged_stem']).encode()).hexdigest()
    valgroups={r['pair_group_id'] for r in allrows if r['split']=='val'}
    splits={}
    for split,count in [('train',2048),('val',512)]:
        rows=sorted([r for r in allrows if r['split']==split and
                     (split=='val' or r['pair_group_id'] not in valgroups)],key=rank)
        selected=[];seen=set()
        for r in rows:
            if r['pair_group_id'] in seen:continue
            seen.add(r['pair_group_id'])
            asset=r.get('source_asset') or specs['legacy_stream_asset_resolution'][r['source']]
            r=dict(r,asset=asset,order=specs['assets'][asset]['max_valid_order'])
            assert r['order'] in (1,2)
            selected.append(r)
            if len(selected)==count:break
        assert len(selected)==count
        splits[split]=selected
        write(OUT/f'{split}_manifest.json',dict(records=selected,counts=dict(collections.Counter(r['asset'] for r in selected))))
    assert not ({r['pair_group_id'] for r in splits['train']}&{r['pair_group_id'] for r in splits['val']})
    write(OUT/'PROTOCOL.json',dict(schema='dimension_symmetry_2x2_v1',locked=True,
        arms=ARMS,seeds=[1,2,3],steps=1000,batch=32,lr=.0001,weight_decay=.0001,
        imgsize=640,confidence=.25,match_iou=.5,train_requested=2048,val_requested=512,
        train_scope='Pretrained one2one 1x1 keypoint and sigma outputs plus identity-initialized FiLM. Backbone, detector, spatial pose features, BN and flow frozen. CUDA float32 through verified process-local NVIDIA runtime.',
        control='All four arms have identical parameters; no-dim arms receive zero conditioning vector. Constant FiLM can be absorbed into the last affine layer.',
        loss='Stock per-object OKS + visibility + RLE, gains 12/1/1; one assigned anchor per training object. C2 chooses one joint minimum; centre remains fixed. No line or pose loss in this first point-only 2x2.',
        evaluation='Same C1/C2 evaluation for all arms. GT-free highest-confidence detection; IoU matching only for scoring. All frames included; unmatched frame receives raw image diagonal penalty. PnP uses correct fixed dimensions for all arms.',
        primary='Mean frame corner error / raw image diagonal, capped at 1; same valid mask and one physical symmetry per frame.',
        gate=dict(required_seeds=3,min_relative_gain_over_A=.01,mean_better_than_stock=True,
                  p90_nonworse_than_A=True,bootstrap_delta_upper_below_zero=True,
                  new_bad_frames_from_A_le10_to_gt20=0),
        contrasts={'dims_without_sym':'C-A','dims_with_sym':'D-B','sym_without_dims':'B-A','sym_with_dims':'D-C','interaction':'D-C-B+A'},
        limits=['Reused synthetic validation, seen asset families; not independent real or new-pallet generalization.',
                'No confirmed C4 examples; no universal impossibility conclusion from this output-layer budget.',
                'Point-only architecture; no Hough benefit tested here.',
                'Camera-dynamic labels retained. This does not establish fixed-object semantic pose recovery.'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in [CKPT,META,GEOM,CONTRACT]},
        manifest_sha256={s:sha(OUT/f'{s}_manifest.json') for s in splits}))
    print('Protocol and 2048/512 group-disjoint manifests locked',flush=True)

def pack(rows):
    return {k:torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}

def prepare(limit=None,device='cuda:0'):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.loss import PoseLoss26
    m=base_model().to(device);head=m.model[-1];criterion=PoseLoss26(m,tal_topk=7,tal_topk2=1)
    captured={}
    hooks=[mod.register_forward_hook(lambda module,args,out,i=i:captured.__setitem__(i,out.detach()))
           for i,mod in enumerate(head.one2one_cv4)]
    table=np.load(GEOM);lookup={s:i for i,s in enumerate(table['stems'])}
    start=time.time();audit={};all_hashes={}
    for split in ['train','val']:
        target=OUT/f'{split}.pt'
        if target.exists():raise RuntimeError(f'Cache already exists: {target}')
        rows=read(OUT/f'{split}_manifest.json')['records']
        if limit:rows=rows[:limit]
        cached=[];records=[];geometry=[];parity=[];skipped=[];perms=collections.Counter()
        for n,row in enumerate(rows):
            # Observation pass reads only image and known dimensions.
            im=cv2.imread(str(ROOT/row['image_path']));assert im is not None
            ih,iw=im.shape[:2];pad=float(row['prepared_pad_px']);rh,rw=row['raw_shape_height_width']
            assert (ih,iw)==(rh+2*pad,rw+2*pad)
            gain=min(640/ih,640/iw)
            neww,newh=round(iw*gain),round(ih*gain)
            offset=np.array([round((640-neww)/2-.1),round((640-newh)/2-.1)])
            img=LetterBox((640,640),auto=False)(image=im)
            x=(torch.from_numpy(img[:,:,::-1].transpose(2,0,1).copy()).float()[None]/255).to(device)
            with torch.no_grad():
                out=m(x);raw=out[1]['one2one']
                decoded=head._inference(raw)
            confidence,j=raw['scores'][0,0].sigmoid().max(0);j=int(j)
            observed_box=decoded[0,:4,j].cpu().numpy()
            selected=j
            # Supervision phase is allowed to choose TAL anchors for training only.
            label=np.loadtxt(ROOT/row['yolo_label_path'],ndmin=2)
            assert label.shape==(1,32), (row['merged_stem'],label.shape)
            box=label[0,1:5].copy()
            box[:2]=(box[:2]*[iw,ih]*gain+offset)/640
            box[2:]*=np.array([iw,ih])*gain/640
            gt=label[0,5:].reshape(9,3).copy()
            gt[:,:2]=(gt[:,:2]*[iw,ih]*gain+offset)/640
            gt_raw=(gt[:,:2]*640-offset)/gain-pad
            idx=lookup[row['merged_stem']];X=table['Xcf'][idx];R=table['R'][idx];t=table['t'][idx]
            K4=table['K'][idx];K=np.array([[K4[0],0,K4[2]-pad],[0,K4[1],K4[3]-pad],[0,0,1]])
            Q=X@R.T+t;uv=Q[:,:2]/Q[:,2:]*np.array([K[0,0],K[1,1]])+K[:2,2]
            mask=gt[:8,2]>0
            er=np.linalg.norm(uv[mask]-gt_raw[:8][mask],axis=-1)
            geometry.extend(er.tolist())
            perm=c2_permutation(X);perms[str(perm.tolist())]+=1
            batch=dict(img=x,batch_idx=torch.zeros(1),cls=torch.zeros(1,1),bboxes=torch.tensor(box[None],dtype=torch.float32),keypoints=torch.tensor(gt[None],dtype=torch.float32))
            batch={k:v.to(device) for k,v in batch.items()}
            if split=='train':
                with torch.no_grad():assignment,_,_=criterion.get_assigned_targets_and_loss(raw,batch)
                fg=assignment[0][0].nonzero().flatten()
                if len(fg)==0:
                    skipped.append(row['merged_stem']);continue
                assert len(fg)==1,('one TAL anchor per object required',len(fg))
                selected=int(fg[0])
            sizes=[a.shape[-2]*a.shape[-1] for a in captured.values()]
            level=int(np.searchsorted(np.cumsum(sizes),selected,side='right'))
            local=selected-sum(sizes[:level]);fw=captured[level].shape[-1]
            feature=captured[level][0].reshape(45,-1)[:,local].cpu().numpy().copy()
            stride=float(head.stride[level]);anchor=np.array([local%fw+.5,local//fw+.5],dtype=np.float32)
            target_gt=gt.copy();target_gt[:,:2]*=640/stride
            xywh=box*640;targetbox=np.r_[xywh[:2]-xywh[2:]/2,xywh[:2]+xywh[2:]/2]
            stock=(raw['kpts'][0,:,selected].reshape(9,3).cpu().numpy()[:,:2]+anchor)*stride
            # Recover cached output directly from actual pretrained convolutions.
            with torch.no_grad():
                f=torch.from_numpy(feature)[None,:,None,None].to(device)
                recovered=head.one2one_cv4_kpts[level](f).reshape(9,3).cpu().numpy()
                sigma=head.one2one_cv4_sigma[level](f).reshape(9,2).cpu().numpy()
            parity.append(float(np.max(np.abs((recovered[:,:2]+anchor)*stride-stock))))
            dims=table['dims'][idx]
            cached.append(dict(features=feature,level=np.array(level,dtype=np.int64),
                anchor=anchor,stride=np.array(stride,dtype=np.float32),
                dims_whd=dims[[0,2,1]].astype(np.float32),target=target_gt.astype(np.float32),
                area=np.array(xywh[2]*xywh[3]/stride**2,dtype=np.float32),
                order=np.array(row['order'],dtype=np.int64),perm=perm.astype(np.int64),
                letterbox_offset=offset.astype(np.float32),letterbox_gain=np.array(gain,dtype=np.float32),
                pad=np.array(pad,dtype=np.float32),stock_grid=np.c_[stock/stride,recovered[:,2],sigma].astype(np.float32)))
            records.append(dict(id=row['merged_stem'],group=row['pair_group_id'],asset=row['asset'],order=row['order'],
                image=row['image_path'],raw_hw=[rh,rw],confidence=float(confidence),
                box_raw=((observed_box-np.tile(offset,2))/gain-pad).tolist(),
                gt_box_raw=((targetbox-np.tile(offset,2))/gain-pad).tolist(),
                gt_raw=gt_raw.tolist(),valid=(gt[:,2]>0).tolist(),K=K.tolist(),dims_xyz=dims.tolist(),
                Xcf=X.tolist(),R=R.tolist(),t=t.tolist(),perm=perm.tolist(),
                observation_anchor=j,training_anchor=selected if split=='train' else None))
            all_hashes[row['image_path']]=sha(ROOT/row['image_path'])
            all_hashes[row['yolo_label_path']]=sha(ROOT/row['yolo_label_path'])
            if (n+1)%128==0:print(f'cache {split} {n+1}/{len(rows)} elapsed {time.time()-start:.0f}s',flush=True)
        assert geometry and max(geometry)<.1,('GT projection error',max(geometry))
        assert max(parity)<.002,('cache does not preserve original outputs',max(parity))
        assert len(cached)>0
        name=f'{split}_smoke' if limit else split
        torch.save(pack(cached),OUT/f'{name}.pt')
        write(OUT/f'{name}_records.json',records)
        audit[split]=dict(requested=len(rows),cached=len(cached),no_assigned_anchor=skipped,
                          reprojection_max_px=max(geometry),cache_parity_max_px=max(parity),
                          corner_permutations=dict(perms),orders=dict(collections.Counter(r['order'] for r in records)))
    for h in hooks:h.remove()
    write(OUT/('CACHE_SMOKE_AUDIT.json' if limit else 'CACHE_AUDIT.json'),dict(PASS=True,splits=audit,
          image_label_sha256=all_hashes,elapsed_seconds=time.time()-start,
          cache_sha256={s:sha(OUT/f'{s}{"_smoke" if limit else ""}.pt') for s in ['train','val']}))
    print('Cache geometry/parity audit passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','cache']);p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.phase=='freeze':freeze()
    else:prepare(a.limit,a.device)
