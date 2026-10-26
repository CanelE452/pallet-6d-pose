"""Once-only whole-frame SYN R0 neck/one2one-assignment packet. No optimizer."""
import os,sys,json,time,hashlib,types
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from PIL import Image
W=Path(__file__).parent
def module(p):
    for a in [p,*p.parents]:assert not a.is_symlink()
    b=p.read_bytes();m=types.ModuleType('v270_model');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(b,str(p),'exec'),m.__dict__);return m
m=module(W/'model_initial_joint_v270.py')
RGB=W.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/data_audit/FULL_FRAME_RGB_INPUTS.jsonl'
LABEL=W/'raw_model_targets9_v2/COMPLETE.json'
def main():
    out=m.safe(W/'syn_initial_features_v1');assert not out.exists();out.mkdir()
    assert m.binding(W/'DESIGN_BEFORE_DATA_FORWARDS.json')['sha256']=='5050f76ff299689afa034c596e6d65724e8cab8febccb8c2e37ef41ac474a7f3'
    lc=json.loads(m.safe(LABEL).read_text());lp=m.verify(lc['outputs']['RAW_MODEL_TARGETS9.npz']);labels=np.load(lp,allow_pickle=False)
    assert m.binding(RGB)['sha256']=='22b4ea7f28ed8de8b13e83d6a83aa5774a26b6e41da135b3920dccea022e3a75'
    rows=[json.loads(x) for x in m.safe(RGB).read_text().splitlines()];assert len(rows)==1003
    assert json.loads(m.safe(W/'ARTIFICIAL_API_QA.json').read_text())['PASS']
    model=m.InitialKPJoint(device='cuda');model.eval();before=model.frozen_sha();initial=model.head_state()
    torch.save(initial,m.safe(out/'INITIAL_HEAD_STATE.pt'))
    receipt=dict(PASS=True,code=m.binding(Path(__file__)),model_code=m.binding(W/'model_initial_joint_v270.py'),design=m.binding(W/'DESIGN_BEFORE_DATA_FORWARDS.json'),RGB=m.binding(RGB),labels=m.binding(LABEL),R0=m.binding(m.R0),initial=m.binding(out/'INITIAL_HEAD_STATE.pt'),scope='Only whole-frame SYN1003, no optimizer or real inputs; code sealed before first SYN forward')
    m.dump(W/'BEFORE_FIRST_SYN_DATA_FORWARD.json',receipt)
    fit=np.asarray(labels['fit']);assert fit.sum()==768
    draws=np.random.Generator(np.random.PCG64(17)).choice(np.flatnonzero(fit),size=(300,8),replace=True);np.save(out/'FIT_DRAWS300x8.npy',draws)
    data={k:np.array(labels[k]) for k in ('E9','Emask9','frame_index','fit','model_channel_to_native')}
    data['xy_px']=np.zeros((1003,9,2),np.float32);data['loc_valid9']=np.array(labels['coordvalid9']);data['canvas_obj9']=np.zeros((1003,9),bool);data['raw_to_input_affine']=np.zeros((1003,2,3),np.float32)
    bbox=np.zeros((1003,4),np.float32);assigned=np.full(1003,-1,np.int64);detanchor=assigned.copy();detconf=np.zeros(1003,np.float32)
    pyramids=None;criterion=m.PoseLoss26(model.base,tal_topk=7,tal_topk2=1);start=time.time()
    for lo in range(0,1003,4):
        inds=list(range(lo,min(lo+4,1003)));images=[];boxes=[]
        for i in inds:
            row=rows[i];assert row['frame_index']==i;m.verify(row['RGB'])
            rgb=np.asarray(Image.open(m.safe(row['RGB']['path'])).convert('RGB'));assert list(rgb.shape[:2])==row['raw_hw']
            image,A=m.prepare_rgb(rgb);images.append(image);data['raw_to_input_affine'][i]=A
            xy=np.array(labels['gt_xy9_raw'][i]);h,w=rgb.shape[:2]
            known=np.isfinite(xy).all(-1);canvas=known&(xy[:,0]>=-100)&(xy[:,0]<w+100)&(xy[:,1]>=-100)&(xy[:,1]<h+100)
            data['canvas_obj9'][i]=canvas;xy640=xy@A[:,:2].T+A[:,2];data['xy_px'][i]=xy640
            validxy=xy640[canvas];assert len(validxy)>0
            mn=validxy.min(0);mx=validxy.max(0);wh=np.maximum(mx-mn,1);cen=(mn+mx)/2
            bbox[i]=np.r_[cen-wh/2,cen+wh/2];boxes.append(np.r_[cen,wh]/640)
        with torch.no_grad():
            feats=model.extract_pyramid(torch.stack(images).cuda());packet=model.forward_pyramid(feats)
            batch=dict(batch_idx=torch.arange(len(inds),device='cuda'),cls=torch.zeros(len(inds),1,device='cuda'),bboxes=torch.tensor(np.array(boxes),device='cuda'))
            assignment,_,_=criterion.get_assigned_targets_and_loss(packet['det'],batch)
            fg,gtidx,tb,anchors,strides=assignment
            for j,i in enumerate(inds):
                ai=torch.where(fg[j])[0];assert len(ai)<=1,(i,ai)
                if len(ai):assigned[i]=int(ai[0]);assert int(gtidx[j,ai[0]])==0
            scores=packet['det']['scores'].sigmoid().amax(1);detanchor[inds]=scores.argmax(1).cpu().numpy();detconf[inds]=scores.max(1).values.cpu().numpy()
        if pyramids is None:
            pyramids=[np.lib.format.open_memmap(out/f'pyramid_P{k+3}.npy',mode='w+',dtype=np.float32,shape=(1003,*f.shape[1:])) for k,f in enumerate(feats)]
        for store,f in zip(pyramids,feats):store[inds]=f.cpu().numpy()
        if lo%100==0:print(f'SYN {lo+len(inds)}/1003 elapsed {time.time()-start:.1f}s',flush=True)
    for x in pyramids:x.flush()
    assert model.frozen_sha()==before
    assert np.array_equal(fit,np.array([r['split']=='train' for r in rows]))
    np.savez_compressed(out/'TRAIN_INPUTS.npz',**data)
    np.savez_compressed(out/'ASSIGNMENTS.npz',assigned_anchor=assigned,detection_anchor=detanchor,detection_confidence=detconf,bbox_xyxy_px=bbox,anchor_points=anchors.cpu().numpy(),stride_tensor=strides.cpu().numpy())
    (out/'FRAME_ORDER.jsonl').write_bytes(m.safe(RGB).read_bytes())
    e=data['E9'][fit&(assigned>=0)];em=data['Emask9'][fit&(assigned>=0)];counts=[int(((e==v)&em).sum()) for v in (0,1)];weights=[sum(counts)/(2*c) for c in counts]
    files=['INITIAL_HEAD_STATE.pt','FIT_DRAWS300x8.npy','TRAIN_INPUTS.npz','ASSIGNMENTS.npz','FRAME_ORDER.jsonl']+[f'pyramid_P{i}.npy' for i in (3,4,5)]
    complete=dict(PASS=True,completed=True,outputs={f:m.binding(out/f) for f in files},code=m.binding(Path(__file__)),model_code=m.binding(W/'model_initial_joint_v270.py'),source_labels=m.binding(LABEL),before_forward=m.binding(W/'BEFORE_FIRST_SYN_DATA_FORWARD.json'),frame_count=1003,FIT=768,DEV=235,assigned_count=int((assigned>=0).sum()),unassigned_ids=np.flatnonzero(assigned<0).tolist(),E_counts_FIT_assigned=counts,E_class_weights=weights,frozen_before_sha256=before,frozen_after_sha256=model.frozen_sha(),neck_dtype='float32',elapsed_seconds=time.time()-start,optimizer_steps=0,current223_reads=0,canvas_definition='Known finite point inside raw image plus reflect100 padded canvas; loc_valid uses independent source positive-depth/original-raster validity',assignment='Official R0 PoseLoss26 TAL topk7/topk2=1 once; identical frozen index each fit')
    m.dump(out/'COMPLETE.json',complete);print(json.dumps({k:complete[k] for k in ('assigned_count','E_counts_FIT_assigned','elapsed_seconds')}),flush=True)
if __name__=='__main__':main()
