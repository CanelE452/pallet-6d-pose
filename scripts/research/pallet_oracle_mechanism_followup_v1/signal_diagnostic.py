"""Frozen TRAIN augmentation/gradient probe. No optimizer step or DEV access."""
import argparse,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from . import common as C
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer,pose_parameter

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def batch_subset(batch,indices):
    indices=torch.tensor(indices,device=batch['img'].device)
    old=batch['batch_idx'].long();mask=(old[:,None]==indices[None,:]).any(1)
    mapping=torch.full((len(batch['img']),),-1,device=indices.device,dtype=torch.long);mapping[indices]=torch.arange(len(indices),device=indices.device)
    out={}
    for k,v in batch.items():
        if k=='img':out[k]=v[indices]
        elif k=='batch_idx':out[k]=mapping[old[mask]].to(v.dtype)
        elif k in ('cls','bboxes','keypoints'):out[k]=v[mask]
        elif isinstance(v,(list,tuple)) and len(v)==len(batch['img']):out[k]=[v[i] for i in indices.cpu().tolist()]
        else:out[k]=v
    return out

def gradient(model,batch,parameters):
    loss,items=model(batch)
    # coordinate+visibility+RLE; detector cannot update pose-only parameters.
    terms=[1,2]+([5] if len(loss)>5 else [])
    objective=loss[terms].sum()/len(batch['img'])
    grads=torch.autograd.grad(objective,parameters,allow_unused=True)
    flat=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,grads)])
    k=batch['keypoints'];v=k[...,2]
    return flat.detach(),dict(loss_items=items.detach().cpu().tolist(),coordinate_objective_per_image=float(objective.detach()),
        images=len(batch['img']),instances=len(k),supervised=int((v==2).sum()),ignore=int((v==1).sum()),invisible=int((v==0).sum()))

def main(material):
    import cv2
    from ultralytics import YOLO
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    dst=C.DOC/f'SIGNAL_DIAGNOSTIC_{material}.json'
    if dst.exists():print('SIGNAL_REUSED',material);return
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available();thermal_guard()
    protocol=C.read(C.M.DOC/'WOOD_TRAIN_PROTOCOL.json' if material=='WOOD' else C.P.REC/'pose_only/PROTOCOL.json')
    args=dict(protocol['args'],lr0=1e-5,model=str(C.ROOT/C.checkpoint(material,'R0')['path']),
        data=str(C.ROOT/protocol['datasets']['REF']['data']['path']),project=str(C.RAW/'signal_probe'),
        name=material,exist_ok=False,workers=0)
    t=PoseOnlyTrainer(overrides=args);t._setup_train()
    counter=[];t.optimizer.register_step_post_hook(lambda *a:counter.append(1))
    dataset=t.train_loader.dataset
    real=[i for i,p in enumerate(dataset.im_files) if not Path(p).name.startswith('syn__')]
    source=[i for i,p in enumerate(dataset.im_files) if Path(p).name.startswith('syn__')]
    # Fixed midpoint TRAIN selection, no prediction/reference ranking. Four16-image mixed batches.
    choose=lambda ids:[ids[int((j+.5)*len(ids)/32)] for j in range(32)]
    rr,ss=choose(real),choose(source)
    samples={};fingerprints={}
    for mode in ('EXISTING_AFFINE','NO_RANDOM_AFFINE'):
        hyp=t.args
        hyp.translate=.1 if mode=='EXISTING_AFFINE' else 0.
        hyp.scale=.25 if mode=='EXISTING_AFFINE' else 0.
        dataset.transforms=dataset.build_transforms(hyp)
        samples[mode]=[]
        for j in range(4):
            entries=[]
            for index in rr[j*8:j*8+8]+ss[j*8:j*8+8]:
                set_seed(20260928+index);entries.append(dataset[index])
            b=dataset.collate_fn(entries);samples[mode].append(b)
        fingerprints[mode]=[dict(images=hashlib.sha256(b['img'].numpy().tobytes()).hexdigest(),
            targets=hashlib.sha256(b['keypoints'].numpy().tobytes()).hexdigest(),files=b['im_file']) for b in samples[mode]]
    C.save(C.RAW/f'{material}_PROBE_INPUT_FINGERPRINTS.json',fingerprints,True)
    out={};checkpoints={}
    try:
        for arm in ('R0','REF_LR5'):
            binding=C.checkpoint(material,arm);C.verify(binding);checkpoints[arm]=binding
            stored=torch.load(C.ROOT/binding['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
            t.model.load_state_dict(stored,strict=True);t._model_train()
            # Original branch weighting at initial versus final epoch; do not fake resumable optimizer.
            t.model.criterion=t.model.init_criterion()
            if arm!='R0':
                for _ in range(5):t.model.criterion.update()
            params=[p for p in t.model.parameters() if p.requires_grad]
            predictor=YOLO(str(C.ROOT/binding['path']),task='pose');out[arm]={}
            for mode,bb in samples.items():
                records=[];follow=[]
                for j,b0 in enumerate(bb):
                    b=t.preprocess_batch({k:v.clone() if torch.is_tensor(v) else v for k,v in b0.items()})
                    r=batch_subset(b,list(range(8)));s=batch_subset(b,list(range(8,16)))
                    gr,rl=gradient(t.model,r,params);gs,sl=gradient(t.model,s,params)
                    nr,ns=float(gr.norm()),float(gs.norm());dot=float(gr@gs)
                    records.append(dict(batch=j,real=rl,source=sl,real_norm=nr,source_norm=ns,dot=dot,
                        cosine=dot/nr/ns if nr*ns>1e-12 else None,source_over_real_norm=ns/nr if nr>1e-12 else None,
                        source_projection_onto_real=dot/nr**2 if nr>1e-12 else None))
                    for i in range(8):
                        image=b0['img'][i].permute(1,2,0).numpy()[:,:,::-1].copy()
                        p=predictor.predict(image,conf=.001,imgsz=640,rect=True,augment=False,half=False,device=0,verbose=False)[0]
                        index=(b0['batch_idx']==i).nonzero().reshape(-1);assert len(index)==1
                        target=b0['keypoints'][index[0]].numpy();mask=target[:8,2]==2
                        hh,ww=image.shape[:2];gt=target[:8,:2]*[ww,hh]
                        if p.boxes is None or not len(p.boxes):err=np.full(mask.sum(),np.hypot(hh,ww));det=False
                        else:
                            ii=int(p.boxes.conf.argmax());q=p.keypoints.xy[ii,:8].cpu().numpy();err=np.linalg.norm(q[mask]-gt[mask],axis=1);det=True
                        box=b0['bboxes'][index[0]].numpy();diag=float(np.hypot(box[2]*ww,box[3]*hh))
                        follow.append(dict(image_id=Path(b0['im_file'][i]).stem,supervised=int(mask.sum()),
                            input640_errors_px=err.tolist(),bbox_diagonal_px=diag,normalized_errors=(err/diag).tolist(),detected=det))
                    thermal_guard()
                out[arm][mode]=dict(gradient_batches=records,target_following=follow)
                print('SIGNAL',material,arm,mode,'cos',[round(r['cosine'],3) for r in records],flush=True)
            assert all(torch.equal(stored[k],v.detach().cpu()) for k,v in t.model.state_dict().items()),'Frozen diagnostic changed weights/buffers'
            del predictor;torch.cuda.empty_cache()
        assert not counter
        C.save(C.DOC/f'SIGNAL_DIAGNOSTIC_{material}.json',dict(material=material,checkpoints=checkpoints,results=out,
            optimizer_steps=0,fits=0,training_only=True,evaluation_reference_read=False,AMP=False,
            trainable='Original pose+flow, frozen buffers; gradients of coordinate+visibility+RLE per image in same parameter space',
            sample='32real+32source exposures midpoint of TRAIN slots; fixed seeded augmentation, same images both modes; limited diagnostic not exhaustive',
            augmentation='translate0.1/scale0.25 versus both0, HSV retained; input640px AND bbox-normalized residual, not nativepx',
            inference='Highest predicted confidence only, no target-based candidate choice; TRAIN labels score after augmentation',
            caveat='Few batches, frozen EMA checkpoint rather than original online state; negative cosine alone not proof of causal conflict',
            seconds=time.monotonic()-start),True)
    finally:C.resource('TRAIN_signal_diagnostic_'+material,time.monotonic()-start,gpu=True,details='zerooptimizerstep frozen gradient and affine on/off probe')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('material',choices=['PLASTIC','WOOD']);main(p.parse_args().material)
