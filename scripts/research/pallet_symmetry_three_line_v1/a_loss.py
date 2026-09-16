"""Stock loss with user-approved joint, two-head-shared target orbit selection."""
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics.utils.loss import E2ELoss,PoseLoss26
from ultralytics.utils.ops import xyxy2xywh
from a_joint import joint_minimum

def stable_target_order(batch):
    """Stock's offset mapping assumes sorted batch_idx; preserve within-image GT order."""
    out=dict(batch);order=torch.argsort(batch['batch_idx'].flatten(),stable=True)
    for key in ['batch_idx','cls','bboxes','keypoints','permutations','group_valid']:
        if key in batch:out[key]=batch[key][order]
    return out

class GenericSymmetryPoseLoss:
    def __init__(self,model,equivalent=False):
        self.stock=E2ELoss(model,PoseLoss26);self.equivalent=equivalent
        self.last_audit={}

    @torch.no_grad()
    def costs(self,predictions,batch):
        """Exact stock reductions decomposed BEFORE each head's batch-global clamp."""
        preds=self.stock.one2many.parse_output(predictions)
        n=len(batch['batch_idx']);perms=batch['permutations'];ng=perms.shape[1]
        costs=torch.zeros(n,ng,3,device=batch['keypoints'].device,dtype=torch.float64)
        details=[]
        for hi,(name,fn,weight) in enumerate([('one2many',self.stock.one2many,self.stock.o2m),('one2one',self.stock.one2one,self.stock.o2o)]):
            p=preds[name]
            (mask,idx,boxes,anchors,stride),det,_=fn.get_assigned_targets_and_loss(p,batch)
            detail=dict(positive_anchors=int(mask.sum()),det_loss=det.cpu().tolist())
            if not mask.any():details.append(detail);continue
            bs=p['kpts'].shape[0]
            coords=p['kpts'].permute(0,2,1).contiguous().view(bs,-1,*fn.kpt_shape)
            if fn.rle_loss is not None and p.get('kpts_sigma') is not None:
                sig=p['kpts_sigma'].permute(0,2,1).contiguous().view(bs,-1,fn.kpt_shape[0],2)
                coords=torch.cat([coords,sig],-1)
            pred=fn.kpts_decode(anchors,coords)[mask]
            if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite positive head outputs')
            counts=torch.bincount(batch['batch_idx'].long(),minlength=bs)
            offsets=counts.cumsum(0)-counts
            image,anchor=mask.nonzero(as_tuple=True)
            row=offsets[image]+idx[mask]
            area=xyxy2xywh((boxes/stride)[mask])[:,2:].prod(1,keepdim=True)
            imgsz=torch.tensor(p['feats'][0].shape[2:],device=pred.device,dtype=pred.dtype)*fn.stride[0]
            kp=batch['keypoints'].float().clone();kp[...,:2]*=imgsz[[1,0]]
            detail['positive_global_GT_rows']=row.cpu().tolist()
            for g in range(ng):
                gt=kp.gather(1,perms[:,g,:,None].expand(-1,-1,kp.shape[-1]))[row].clone()
                gt[...,:2]/=stride[anchor,:,None]
                valid=gt[...,2]!=0
                d=(pred[...,0]-gt[...,0]).pow(2)+(pred[...,1]-gt[...,1]).pow(2)
                factor=valid.shape[1]/(torch.sum(valid!=0,dim=1)+1e-9)
                exponent=d/((2*fn.keypoint_loss.sigmas).pow(2)*(area+1e-9)*2)
                pos=(factor[:,None]*((1-torch.exp(-exponent))*valid)).mean(1)/len(pred)
                vis=F.binary_cross_entropy_with_logits(pred[...,2],valid.float(),reduction='none').mean(1)/len(pred)
                linear=(pos*fn.hyp.pose+vis*fn.hyp.kobj)*weight
                costs[:,g,0].scatter_add_(0,row,linear.double())
                if fn.rle_loss is not None and pred.shape[-1] in (4,5) and valid.any():
                    sigma=pred[valid][:,-2:].sigmoid()
                    err=(pred[valid][:,:2]-gt[valid][:,:2])/(sigma+1e-9)
                    if not torch.isfinite(err).all():raise FloatingPointError('Nonfinite RLE input')
                    err=err.clamp(-100,100)
                    phi=fn.flow_model.log_prob(err)
                    rle=torch.log(sigma)-phi[:,None]
                    if fn.rle_loss.residual:rle+=torch.log(sigma*2)+torch.abs(err)
                    weights=fn.target_weights[None].expand(len(pred),-1)[valid]
                    if fn.rle_loss.use_target_weight:rle*=weights[:,None]
                    if fn.rle_loss.size_average:rle/=len(rle)
                    raw=rle.sum(-1)*fn.hyp.rle*weight
                    costs[:,g,hi+1].scatter_add_(0,row[:,None].expand_as(valid)[valid],raw.double())
            details.append(detail)
        if not torch.isfinite(costs).all():raise FloatingPointError('Nonfinite joint costs')
        return costs.cpu().numpy(),details

    def __call__(self,predictions,batch):
        batch=stable_target_order(batch)
        if not self.equivalent or not len(batch['batch_idx']) or not batch['group_valid'][:,1:].any():
            self.last_audit=dict(branches=[0]*len(batch['batch_idx']),method='identity_stock')
            return self.stock(predictions,batch)
        costs,heads=self.costs(predictions,batch)
        branch,certificate=joint_minimum(costs,batch['group_valid'].cpu().numpy())
        choice=torch.tensor(branch,device=batch['keypoints'].device)
        perms=batch['permutations'][torch.arange(len(choice),device=choice.device),choice]
        selected=dict(batch);selected['keypoints']=batch['keypoints'].gather(1,perms[:,:,None].expand_as(batch['keypoints']))
        self.last_audit=dict(branches=branch.tolist(),**certificate,head_weights=[self.stock.o2m,self.stock.o2o],heads=heads)
        return self.stock(predictions,selected)
