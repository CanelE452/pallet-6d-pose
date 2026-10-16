"""Render stress at image level, rerun YOLO, keep targets outside inference."""
import argparse
import sys
import time
import numpy as np
import torch
import cv2
from common import *
sys.path.insert(0, str(KIT.parent/'pallet_point_line_v5_local'))
from cli_export import Capture, roi_sample, R0, R0_SHA
from scripts.research.pallet_line_pose_v1.source_data import transform


def populations():
    out = {}
    for split in ('train', 'synth_val'):
        records = read(EXPORT/f'{split}.json')['records']
        if split == 'train':
            records = sorted(records, key=lambda r:digest('v11-train:'+r['id']))[:384]
        rows = []
        for rec in records:
            old = torch.load(EXPORT/rec['observation'], map_location='cpu', weights_only=True)
            rows.append(dict(id=rec['id'], image=rec['original_image'], padded=True,
                K=old['K'][0].tolist(), dims=old['dims'][0].tolist(), session=rec['session']))
        out[split] = rows
    rows = []
    for rec in read(ROOT/'data/pallet/results/next_accuracy_v2/GT_PARTITION.json'):
        if rec['split_role'] != 'HELD_OUT': continue
        label = ROOT/'challenge/data/01_real/live_capture_gt'/(rec['session']+'_manual_gt')/(rec['frame']+'.json')
        doc = read(label); intr = doc['camera_data']['intrinsics']
        rows.append(dict(id=rec['frame_id'], image=str(label.with_suffix('.png').relative_to(ROOT)),
            label=str(label.relative_to(ROOT)), label_sha256=sha(label), padded=False, session=rec['session'],
            K=[[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],
            dims=[1.1,1.1,.15], occlusion_label=doc['objects'][0].get('occlusion_level'),
            natural_truncation=doc['objects'][0].get('truncation',{}).get('is_truncated',False)))
    assert len(rows)==297
    out['real'] = rows
    return out


def raw_image(rec):
    im = cv2.imread(str(ROOT/rec['image']))
    assert im is not None
    return im[100:-100,100:-100].copy() if rec['padded'] else im


def resolve_real_images(pop):
    repairs=[]
    for rec in pop['real']:
        if (ROOT/rec['image']).exists():continue
        frame=Path(rec['label']).stem
        candidates=list((ROOT/'challenge/data/01_real/_live_captures').glob(f'*/sessions/{rec["session"]}/rgb/{frame}.png'))
        assert len(candidates)==1,(rec['id'],candidates)
        new=str(candidates[0].relative_to(ROOT))
        repairs.append(dict(id=rec['id'],old=rec['image'],resolved=new));rec['image']=new
    return repairs


def stress(im, box, condition, fid):
    h,w = im.shape[:2]; meta=dict(origin=[0,0], mask=None, original_hw=[h,w])
    if condition=='clean':return im,meta
    q=int(condition[-2:])/100
    x1,y1,x2,y2 = np.asarray(box,float); bw=max(x2-x1,8.);bh=max(y2-y1,8.)
    rng=np.random.default_rng(int(digest('v11-stress:'+fid)[:8],16))
    if condition.startswith('occlude'):
        center=x1+float(rng.uniform(.3,.7))*bw
        a=max(0,min(w-1,int(round(center-q*bw/2))))
        b=max(a+1,min(w,int(round(center+q*bw/2))))
        c=max(0,min(h-1,int(np.floor(y1-.1*bh))));d=max(c+1,min(h,int(np.ceil(y2+.1*bh))))
        canvas=im.copy();color=rng.integers(40,216,size=3).astype(np.uint8)
        canvas[c:d,a:b]=color
        meta.update(mask=[a,c,b,d],fill_bgr=color.tolist())
        return canvas,meta
    left=bool(rng.integers(2))
    if left:
        cut=max(1,min(w-32,int(round(x1+q*bw))))
        meta['origin']=[cut,0];canvas=im[:,cut:].copy()
    else:
        cut=max(32,min(w-1,int(round(x2-q*bw))))
        canvas=im[:,:cut].copy()
    meta.update(crop_side='left' if left else 'right',cut=cut)
    return canvas,meta


class Extractor:
    def __init__(self):
        from ultralytics import YOLO
        assert sha(R0)==R0_SHA
        self.model=YOLO(str(R0),task='pose');self.cap=Capture(self.model)

    def __call__(self, im, K, dims):
        h,w=im.shape[:2]
        canvas=cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101)
        result=self.model.predict(canvas,imgsz=640,conf=.001,iou=.7,max_det=300,
            device='cuda:0',half=False,verbose=False)[0]
        if not len(result.boxes):return None,dict(detected=False,confidence=None,box=None)
        best=int(result.boxes.conf.argmax());confidence=float(result.boxes.conf[best])
        box=result.boxes.xyxy[best].detach()-100
        info=dict(detected=confidence>=.25,confidence=confidence,box=box.cpu().tolist(),candidate_index=best)
        if not info['detected']:return None,info
        fh,fw=self.cap.features[1].shape[-2:]
        tf=transform((h+200,w+200),(fh*16,fw*16))
        aff=np.array([[tf.gain,0,100*tf.gain+tf.pad_ltrb[0]],
                      [0,tf.gain,100*tf.gain+tf.pad_ltrb[1]]])
        planes=[];masks=[]
        for stride,plane in zip((8,16),self.cap.features):
            a=aff/stride;a[:,2]-=.5;v,inside=roi_sample(plane,box,a);planes.append(v);masks.append(inside)
        obs=dict(features=torch.cat(planes)[None].cpu(),content=(masks[0]&masks[1])[None,None].cpu(),
            box=box[None].cpu(),K=torch.tensor(K,dtype=torch.float32)[None],
            dims=torch.tensor(dims,dtype=torch.float32)[None],image_hw=torch.tensor([[h,w]],dtype=torch.float32),
            points=(result.keypoints.xy[best,:8].detach()-100)[None].cpu())
        assert obs['features'].shape==(1,192,24,24)
        return obs,info


def export(pop):
    extractor=Extractor();start=time.monotonic();parity=[]
    for split,recs in pop.items():
        cleanpath=RUN/'cache'/split/'clean'/'manifest.json'
        for condition in CONDITIONS:
            if split=='train' and condition=='clean':continue
            directory=RUN/'cache'/split/condition;out=directory/'manifest.json'
            if out.exists():continue
            directory.mkdir(parents=True,exist_ok=True)
            clean={r['id']:r for r in read(cleanpath)['rows']} if condition!='clean' and split!='train' else {}
            if split=='train':
                # These observed training boxes were already frozen before V11.
                src=read(EXPORT/'train.json')['records']
                lookup={r['id']:r for r in src}
                clean={r['id']:dict(box=torch.load(EXPORT/lookup[r['id']]['observation'],weights_only=True)['box'][0].tolist()) for r in recs}
            rows=[];pending=[];chunks=[];obsindex=0
            for i,rec in enumerate(recs):
                im=raw_image(rec)
                box=clean.get(rec['id'],{}).get('box')
                if box is None:box=[im.shape[1]*.2,im.shape[0]*.2,im.shape[1]*.8,im.shape[0]*.8]
                canvas,meta=stress(im,box,condition,rec['id'])
                K=np.array(rec['K']);K[:2,2]-=meta['origin']
                O,info=extractor(canvas,K.tolist(),rec['dims'])
                row=dict(**rec,condition=condition,**meta,**info,image_hw=list(canvas.shape[:2]),
                    image_sha256=sha(ROOT/rec['image']),observation_index=obsindex if O is not None else None)
                rows.append(row)
                if O is not None:
                    pending.append(O);obsindex+=1
                if len(pending)>=64 or (i+1==len(recs) and pending):
                    name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([o[k] for o in pending]) for k in pending[0]},directory/name)
                    chunks.append(name);pending=[]
                if (i+1)%100==0:print('EXTRACT',split,condition,i+1,len(recs),round(time.monotonic()-start),flush=True)
            write(out,dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(directory/p) for p in chunks},
                checkpoint_sha256=R0_SHA,policy='Top confidence >=0.25, no GT matching. All requested frames retained.',
                stress='Image-level controlled corruption, anchored to frozen clean detector box; GT not used. Crop changes canvas and camera principal point. Reflect100 canonical wrapper retained.'))
            print('EXPORTED',split,condition,'detected',obsindex,'/',len(rows),flush=True)
    extractor.cap.handle.remove()


def targets(pop):
    audit={}
    for split,recs in pop.items():
        source={r['id']:r for r in read(V10/f'targets_{split}.json')['rows']} if split!='real' else {}
        for condition in CONDITIONS:
            if split=='train' and condition=='clean':continue
            manifest=read(RUN/'cache'/split/condition/'manifest.json');rows=[]
            for rec in manifest['rows']:
                if split=='real':
                    label=ROOT/rec['label'];assert sha(label)==rec['label_sha256']
                    ann=read(label)['objects'][0]['keypoint_annotations'][:8]
                    gt=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann],float)
                    valid=np.array([a.get('source')=='manual_click' and a.get('xy') is not None for a in ann])
                else:gt=np.array(source[rec['id']]['points']);valid=np.array(source[rec['id']]['valid'])
                assert np.isfinite(gt).all() and valid.any()
                # Original annotation remains the target even when newly hidden or outside the crop.
                shifted=gt-np.array(rec['origin']);h,w=rec['image_hw']
                outside=(shifted[:,0]<0)|(shifted[:,0]>=w)|(shifted[:,1]<0)|(shifted[:,1]>=h)
                hidden=np.zeros(8,bool)
                if rec['mask'] is not None:
                    x1,y1,x2,y2=rec['mask'];hidden=(gt[:,0]>=x1)&(gt[:,0]<x2)&(gt[:,1]>=y1)&(gt[:,1]<y2)
                rows.append(dict(id=rec['id'],points=shifted.tolist(),valid=valid.tolist(),hidden=hidden.tolist(),outside=outside.tolist(),
                    session=rec['session'],original_points=gt.tolist()))
            write(RUN/'targets'/f'{split}_{condition}.json',dict(rows=rows,scope='Original ID, no correspondence reassignment. Real primary includes manual clicks only; original clicks retained under controlled stress.'))
            audit[f'{split}_{condition}']=dict(frames=len(rows),valid_points=sum(sum(r['valid']) for r in rows),
                newly_hidden=int(sum(sum(np.array(r['hidden'])&np.array(r['valid'])) for r in rows)),
                outside=int(sum(sum(np.array(r['outside'])&np.array(r['valid'])) for r in rows)))
    write(RUN/'GT_AUDIT.json',dict(status='PASS',counts=audit,real_label_hashes_verified=True,
        natural_real_occlusion=dict(unknown=sum(r['occlusion_label']=='unknown' for r in pop['real']),
             truncated=sum(r['natural_truncation'] for r in pop['real'])),
        limitations='No independent reclick audit. Natural occlusion labels unknown; generated stress is not naturally occluded deployment data. No full-6D reference claim.'))


def linecache(pop):
    for seed in SEEDS:
        model=dht(seed);model.requires_grad_(False)
        for split in pop:
            for condition in CONDITIONS:
                if split=='train' and condition=='clean':continue
                directory=RUN/'cache'/split/condition;out=directory/f'lines_seed{seed}.pt'
                if out.exists():continue
                manifest,O=load_observations(split,condition,'cuda:0');ls=[];ms=[]
                with torch.inference_mode():
                    for first in range(0,len(O['points']),32):
                        obs=subset(O,slice(first,first+32));p=model(obs,point_maps(obs['points'],obs['box']),decode=True)
                        ls.append(p['lines'].flatten(1,2).cpu());ms.append(p['mass'].flatten(1,2).cpu())
                torch.save(dict(lines=torch.cat(ls),mass=torch.cat(ms)),out)
                print('LINES',split,condition,seed,len(O['points']),flush=True)
                del O
        del model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','lines'],default='prepare');args=ap.parse_args()
    setup();cv2.setNumThreads(1);RUN.mkdir(parents=True,exist_ok=True)
    poppath=RUN/'POPULATIONS.json'
    pop=read(poppath) if poppath.exists() else populations()
    repairs=resolve_real_images(pop)
    if repairs:
        write(RUN/'IMAGE_LOCATOR_REPAIR.json',dict(reason='Some annotation folders contain JSON only; use original capture RGB for the identical session/frame.',
            previous_population_sha256=sha(poppath) if poppath.exists() else None,repairs=repairs))
        write(poppath,pop)
    if not poppath.exists():write(poppath,pop)
    if args.phase=='lines':linecache(pop);return
    protocol=RUN/'PROTOCOL.json'
    if not protocol.exists():
        write(protocol,dict(question='Can optional additive lines beat preserved local refinement, especially under occlusion and truncation?',
            arms=ARMS,seeds=SEEDS,steps=2000,batch=16,lr_refiner=3e-4,lr_dht=1e-5,
            train='Original1775 clean synthetic + four image-stress variants of SHA-selected384 train frames. No real training.',
            evaluation='Reused synthetic505 and reused real297/9sessions; five conditions each, paired within each original image.',
            populations_sha256=sha(poppath),conditions=CONDITIONS,
            stress='Occlusion strips cover20/40% clean detected box width, 120% box height; deterministic location/color. Left/right crop removes20/40% box width from chosen side. Same variant inputs for every model. No GT for placement.',
            architecture='Frozen V10 local baseline + zero-initialized extra residual with learned per-point gate. Local control unrestricted; line residual normal-guided. Frozen vs jointly updated DHT; original YOLO always frozen.',
            primary='Original camera-facing point IDs. Real manual_click corners only. Mean per-frame pixel error and all-frame normalized error (missing detection=1); paired session cluster intervals. Report hidden/outside/visible corners separately.',
            acceptance='Incremental line claim needs lower full-population normalized error than same-budget local_extra in all3seeds, paired cluster95%CI below0, >=1% conditional point improvement; real clean relative degradation<=1%. Separate verdict by condition. No checkpoint selection or real tuning.',
            controls='V10 local and directional unmodified; same-model line-off exact local parity; wrong-image lines intervention; local_extra shares refiner/gate update budget, joint DHT adds capacity explicitly.',
            limitations='Artificial image-level stress, reused DEV real with unknown natural occlusion; no independent final holdout or full6D claim. Native reflect100 kept, including at crop boundaries.',
            source_sha256={p.name:sha(p) for p in KIT.glob('*.py')}))
    export(pop)
    # Label-derived point coordinates are prepared after image inference finishes.
    targets(pop)
    write(RUN/'PREPARATION_COMPLETE.json',dict(complete=True,protocol_sha256=sha(protocol)))


if __name__=='__main__':main()
