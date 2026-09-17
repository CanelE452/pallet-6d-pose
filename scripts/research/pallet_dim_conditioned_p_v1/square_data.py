"""Separate approved square dataset; original labels untouched, R0 cache frozen."""
import argparse
import numpy as np
import torch,cv2
import dcp_env as E
from data import PaperData
from refiner import context,specification
from inference import serial,registry_input
from dev_evaluate import iou

def membership():
    m=E.read(E.SYM_RAW/'A/square_membership.json');side={r['id']:r for r in E.read(E.SYM_RAW/'A/square_annotation_target_view.json')['records']}
    assert len(m['train']['records'])==696 and len(m['val']['records'])==155
    tr=m['train']['records'];va=m['val']['records']
    for field in ['id','image_sha256','raw_pixel_sha256']:assert not ({r[field] for r in tr}&{r[field] for r in va})
    return [dict(r,split=split,target=side[r['id']]['yolo_target'],source_annotation_sha256=side[r['id']]['source_annotation_sha256']) for split in ['train','val'] for r in m[split]['records']]

@torch.no_grad()
def cache():
    # Phase 4 begins only after paper interpretation/runtime; never overlap GPUs.
    assert E.read(E.DOC/'RUNTIME_AND_MEMORY.json')['complete'];E.gpu();rows=membership();fx=E.old('features');extractor=fx.FrozenYoloFeatures(E.R0);root=E.RAW/'square_cache';root.mkdir(parents=True,exist_ok=True)
    specs=dict(p3=((851,64,80,80),np.float16),p4=((851,128,40,40),np.float16),points=((851,9,2),np.float32),boxes=((851,4),np.float32),
      point_valid=((851,9),bool),input_shape=((851,2),np.int16),gt_points=((851,9,2),np.float32),gt_valid=((851,9),bool),matched=((851,),bool),gain=((851,),np.float64))
    arrays={k:np.load(root/f'{k}.npy',mmap_mode='r+') if (root/f'{k}.npy').exists() else np.lib.format.open_memmap(root/f'{k}.npy',mode='w+',dtype=dtype,shape=shape) for k,(shape,dtype) in specs.items()}
    files=[]
    for i,r in enumerate(rows):
        dst=root/f'{i:04d}.json'
        if dst.exists():files.append(E.bound(dst));continue
        assert E.sha(E.ROOT/r['image'])==r['image_sha256'] and E.sha(E.ROOT/r['label'])==r['label_sha256']
        group,frame=r['id'].split('__',1)
        assert E.sha(E.ROOT/'challenge/data/01_real/live_capture_gt'/group/(frame+'.json'))==r['source_annotation_sha256']
        im=cv2.imread(str(E.ROOT/r['image']));assert im.shape[:2]==tuple(np.array(r['raw_hw'])+200)
        cap=extractor.predict(im,already_padded=True);inp=fx.branch_inputs(cap);gain,offset=fx.canvas_affine(cap['canvas_shape'],cap['input_shape']);a=np.array(r['target'],float);assert len(a)==1;a=a[0];size=np.array(im.shape[:2])[::-1]
        kp=a[5:].reshape(9,3);gt=kp[:,:2]*size*gain+offset;center=a[1:3]*size*gain+offset;half=a[3:5]*size*gain/2;gtbox=np.r_[center-half,center+half]
        for k in ['p3','p4']:arrays[k][i]=cap[k][0].cpu().numpy()
        arrays['points'][i]=inp['points'] if inp is not None else np.full((9,2),np.nan)
        arrays['boxes'][i]=inp['boxes'] if inp is not None else np.full(4,np.nan)
        arrays['point_valid'][i]=inp['point_valid'] if inp is not None else False;arrays['input_shape'][i]=cap['input_shape'];arrays['gt_points'][i]=gt;arrays['gt_valid'][i]=kp[:,2]>0
        arrays['matched'][i]=inp is not None and iou(inp['boxes'],gtbox)>=.5;arrays['gain'][i]=gain
        candidates=serial(cap['candidates'])
        for c in candidates:
            c['keypoints_xy']=(np.array(c['keypoints_xy'])-100).tolist();c['box_xyxy']=(np.array(c['box_xyxy'])-100).tolist()
        for value in arrays.values():value.flush()
        E.write(dst,dict(id=r['id'],candidates=candidates,selected_index=cap['selected_index'],GT_for_inference=False,raw_hw=r['raw_hw'],image_sha256=r['image_sha256']));files.append(E.bound(dst))
        if i%100==0:print('SQUARE_CACHE',i,851,flush=True)
    extractor.close()
    E.write(E.DOC/'SQUARE_CACHE_COMPLETE.json',dict(complete=True,rows=851,train=696,DEV=155,usable_train=int((arrays['matched'][:696]&arrays['gt_valid'][:696,:8].any(-1)).sum()),
      files=files,original_labels_unchanged=True,corrected_target_view=E.bound(E.SYM_RAW/'A/square_annotation_target_view.json'),
      source_membership=E.bound(E.SYM_RAW/'A/square_membership.json'),R0_sha256=E.sha(E.R0),overlap=0,conf=.001,reflect='original prepared100 border, already_padded=True; no double pad',
      cache_bindings=[E.bound(root/f'{k}.npy') for k in arrays]))
    E.freeze(E.DOC/'SECONDARY_TRAINING_CODE_LOCK.json',dict(files=[E.bound(E.HERE/k) for k in ['square_data.py','data.py','refiner.py','train.py']],
      square_train=696,square_DEV=155,mixed_batch_paper=8,mixed_batch_square=8,real_DEV_selection=False))
    audit_square_range()

def audit_square_range():
    from refiner import local_phase
    from collections import Counter
    data=SquareData();b={k:torch.from_numpy(np.array(data.arrays[k][data.train_rows])) for k in ['points','boxes','point_valid','gt_points','gt_valid']}
    b.update(permutations=torch.from_numpy(np.tile(data.perms,(len(data.train_rows),1,1))),group_valid=torch.ones(len(data.train_rows),4,dtype=torch.bool))
    diag=(b['boxes'][:,2:]-b['boxes'][:,:2]).norm(dim=-1).clamp_min(1)
    gt,v,g,_=local_phase(b['points'],b['point_valid'],b['gt_points'],b['gt_valid'],b['permutations'],b['group_valid'],diag)
    stats={}
    for name,target,valid in [('before',b['gt_points'],b['gt_valid']),('after',gt,v)]:
        mask=valid[:,:8]&b['point_valid'][:,:8];res=(torch.linalg.vector_norm(target[:,:8]-b['points'][:,:8],dim=-1)/diag[:,None])[mask].numpy()
        stats[name]=dict(corners=len(res),median=float(np.median(res)),P90=float(np.quantile(res,.9)),max=float(res.max()),fraction_in_radius=float((res<=.08).mean()),fraction_outside_radius=float((res>.08).mean()))
    E.write(E.DOC/'SQUARE_TARGET_RANGE_AUDIT.json',dict(complete=True,C4_frames=len(data.train_rows),branch_counts=dict(Counter(g.tolist())),residuals=stats,radius_unchanged=True,DEV_used=False))

class SquareData:
    def __init__(self):
        assert E.read(E.DOC/'SQUARE_CACHE_COMPLETE.json')['complete'];E.verify(E.read(E.DOC/'SECONDARY_TRAINING_CODE_LOCK.json')['files']);self.rows=membership();root=E.RAW/'square_cache'
        self.arrays={p.stem:np.load(p,mmap_mode='r') for p in root.glob('*.npy')};self.norm=E.read(E.DOC/'DIM_NORMALIZATION_LOCK.json')
        self.train_rows=np.flatnonzero(self.arrays['matched'][:696]&self.arrays['gt_valid'][:696,:8].any(-1));self.validation_rows=np.arange(696,851)
        self.perms=np.array(E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations'])
    def order(self,seed):
        sampler=E.old('train').ShuffledRows(self.train_rows,seed);a=np.stack([sampler.take(16) for _ in range(6000)]);dst=E.RAW/f'orders/SQUARE_seed{seed}.npy';dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert np.array_equal(np.load(dst),a)
        else:np.save(dst,a)
        return a
    def batch(self,rows,arm,device='cuda',supervision=True):
        keys=['p3','p4','points','boxes','point_valid','input_shape']+(['gt_points','gt_valid'] if supervision else [])
        b={k:torch.from_numpy(np.array(self.arrays[k][rows])).to(device) for k in keys};n=len(rows);dim,_=specification(arm)
        if dim:b['context']=torch.from_numpy(context(np.tile([1.1,1.1,.15],(n,1)),np.full(n,4),self.norm,dim==8)).to(device)
        if supervision:b.update(permutations=torch.from_numpy(np.tile(self.perms,(n,1,1))).to(device),group_valid=torch.ones(n,4,dtype=torch.bool,device=device))
        return b

class MixedData:
    def __init__(self):
        self.paper=PaperData();self.square=SquareData();self.train_rows=np.r_[self.paper.train_rows,self.square.train_rows+60000]
    def order(self,seed):
        paper=E.old('train').ShuffledRows(self.paper.train_rows,seed);square=E.old('train').ShuffledRows(self.square.train_rows,seed)
        a=np.stack([np.r_[paper.take(8),square.take(8)+60000] for _ in range(6000)]);dst=E.RAW/f'orders/MIXED_seed{seed}.npy';dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert np.array_equal(np.load(dst),a)
        else:np.save(dst,a)
        return a
    def batch(self,rows,arm,device='cuda',supervision=True):
        assert (rows[:8]<60000).all() and (rows[8:]>=60000).all()
        a=self.paper.batch(rows[:8],arm,device,supervision);b=self.square.batch(rows[8:]-60000,arm,device,supervision)
        assert set(a)==set(b);return {k:torch.cat([a[k],b[k]],0) for k in a}

def square_metadata():
    """GT pose only for evaluation; intrinsics from authoritative annotated image."""
    from pose import cuboid,solve
    meta={};truth={};xyz=np.array([1.1,.15,1.1])
    for r in membership()[696:]:
        group,frame=r['id'].split('__',1);ann=E.ROOT/'challenge/data/01_real/live_capture_gt'/group/(frame+'.json')
        a=E.read(ann)['camera_data']['intrinsics'];K=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float)
        target=np.array(r['target'])[0,5:].reshape(9,3);points=target[:,:2]*(np.array(r['raw_hw'])[::-1]+200)-100;usable=target[:8,2]>0
        assert usable.sum()>=6;solved=solve(cuboid(*xyz),points[:8],K,usable);assert solved is not None
        R,t,residual=solved;meta[r['id']]=(K,xyz,False);truth[r['id']]=dict(R=R,t=t,xyz=xyz,body_R=R,body_xyz=xyz,order=4)
    return meta,truth

if __name__=='__main__':torch.set_num_threads(4);cv2.setNumThreads(1);cache()
