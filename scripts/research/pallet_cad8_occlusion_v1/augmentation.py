"""RGB-only occlusion after spatial augmentation. Target coordinates stay intact."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer

SEED=902106


def sha(tensor):
    a=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def plan(batch,rng,probability=.75):
    h,w=batch['img'].shape[-2:];idx=batch['batch_idx'].detach().cpu().numpy().reshape(-1)
    points=batch['keypoints'].detach().cpu().numpy();boxes=batch['bboxes'].detach().cpu().numpy()
    planned=[]
    for i,path in enumerate(batch['im_file']):
        if not Path(path).name.startswith('eval_cad__'):continue
        hits=np.flatnonzero(idx==i)
        if not len(hits) or rng.random()>=probability:continue
        n=int(hits[0]);q=points[n,:8];valid=(q[:,2]==2)&np.isfinite(q[:,:2]).all(1)&(q[:,:2]>=0).all(1)&(q[:,:2]<=1).all(1)
        usable=np.flatnonzero(valid)
        if len(usable)<4:continue
        bw,bh=boxes[n,2:]*[w,h];xy=q[:,:2]*[w,h]
        if bw<=4 or bh<=4:continue
        rects=[];covered=np.zeros(8,bool)
        for _ in range(int(rng.integers(1,3))):
            upper=[j for j in [0,1,4,5] if valid[j]]
            j=int(rng.choice(upper if upper else usable));center=xy[j].copy()
            edges=[(a,b) for a,b in [(0,1),(1,5),(5,4),(4,0)] if valid[a] and valid[b]]
            if edges and rng.random()<.5:
                a,b=edges[int(rng.integers(len(edges)))];center=(xy[a]+xy[b])/2
            size=rng.uniform(.18,.35,2)*[bw,bh]
            center+=rng.uniform(-.025,.025,2)*[bw,bh]
            x0,y0=np.floor(center-size/2).astype(int);x1,y1=np.ceil(center+size/2).astype(int)
            x0,x1=int(np.clip(x0,0,w)),int(np.clip(x1,0,w));y0,y1=int(np.clip(y0,0,h)),int(np.clip(y1,0,h))
            hit=valid&(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
            if x1<=x0 or y1<=y0 or (valid&~(covered|hit)).sum()<4:continue
            covered|=hit
            rects.append(dict(x0=x0,y0=y0,x1=x1,y1=y1,color=rng.uniform(.08,.92,3).tolist(),noise_seed=int(rng.integers(2**31)),covered=np.flatnonzero(hit).tolist()))
        if rects:planned.append(dict(image_index=i,file=str(path),rectangles=rects,covered_corners=np.flatnonzero(covered).tolist()))
    return planned


def apply(images,plans):
    for item in plans:
        for r in item['rectangles']:
            rng=np.random.default_rng(r['noise_seed'])
            texture=np.clip(np.array(r['color'])[:,None,None]+rng.normal(0,.06,(3,16,16)),0,1).astype(np.float32)
            patch=torch.nn.functional.interpolate(torch.from_numpy(texture).to(images.device)[None],size=(r['y1']-r['y0'],r['x1']-r['x0']),mode='bilinear',align_corners=False)[0]
            images[item['image_index'],:,r['y0']:r['y1'],r['x0']:r['x1']]=patch


class OcclusionPoseTrainer(PoseOnlyTrainer):
    def preprocess_batch(self,batch):
        out=super().preprocess_batch(batch)
        if not hasattr(self,'occlusion_rng'):
            self.occlusion_rng=np.random.default_rng(SEED);self.occlusion_trace=[];self.occlusion_previews=[]
        step=len(self.occlusion_trace)
        targets={k:sha(out[k]) for k in ['keypoints','bboxes','batch_idx','cls']}
        before_image_sha=sha(out['img']) if step%64==0 else None
        plans=plan(out,self.occlusion_rng)
        preview=None
        if self.args.name=='OCCLUDED' and len(self.occlusion_previews)<8 and plans:
            first=plans[0];i=first['image_index']
            preview=dict(step=step,file=first['file'],plan=first,before=out['img'][i].detach().cpu().numpy().copy())
        if self.args.name=='OCCLUDED':apply(out['img'],plans)
        assert {k:sha(out[k]) for k in targets}==targets,'Image occlusion changed labels'
        if preview is not None:
            preview['after']=out['img'][preview['plan']['image_index']].detach().cpu().numpy().copy();self.occlusion_previews.append(preview)
        self.occlusion_trace.append(dict(step=step,files=list(out['im_file']),targets=targets,pre_image_sha=before_image_sha,plan=plans))
        return out


def tests():
    q=torch.tensor([[[.3,.3,2.],[.7,.3,2.],[.7,.7,2.],[.3,.7,2.],[.4,.4,2.],[.6,.4,2.],[.6,.6,2.],[.4,.6,2.],[.5,.5,2.]]]*2)
    batch=dict(img=torch.zeros((2,3,200,200)),keypoints=q,bboxes=torch.tensor([[.5,.5,.6,.6]]*2),batch_idx=torch.tensor([0,1]),im_file=['eval_cad__test.png','syn__test.png'])
    before=q.clone();plans=plan(batch,np.random.default_rng(SEED),1.)
    assert plans and all(p['image_index']==0 for p in plans)
    original=batch['img'].clone();apply(batch['img'],plans)
    assert torch.equal(q,before) and torch.equal(batch['img'][1],original[1]) and not torch.equal(batch['img'][0],original[0])
    for p in plans:assert len(p['covered_corners'])<=4
    assert plans==plan(batch,np.random.default_rng(SEED),1.)
    rng=np.random.default_rng(333);state=rng.bit_generator.state
    _=plan(batch,np.random.default_rng(SEED),1.);assert rng.bit_generator.state==state
    return dict(real_only=True,coordinates_and_masks_unchanged=True,synthetic_image_unchanged=True,deterministic_plan=True,minimum_four_uncovered=True)
