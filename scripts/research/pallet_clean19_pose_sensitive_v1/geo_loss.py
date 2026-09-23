"""Synthetic-only relative sensitivity quadratic, after existing pose gain."""
import torch
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26
from ultralytics.utils.loss import E2ELoss

def quadratic(pred_px,gt_px,supervised,H,enabled):
    mask=supervised.repeat_interleave(2,dim=-1).to(pred_px.dtype)
    residual=(pred_px-gt_px).reshape(-1,16)*mask
    n=mask.sum(-1);valid=enabled&(n>0)
    if not valid.any():return pred_px.sum()*0.
    per=torch.einsum('bi,bij,bj->b',residual,H,residual)/n.clamp_min(1)
    assert torch.isfinite(per).all()
    return per[valid].mean()

class GeoLoss(TrueIgnorePoseLoss26):
    def __init__(self,model,tal_topk=10,tal_topk2=None):
        super().__init__(model,tal_topk,tal_topk2)
        self.lambda_geo=float(getattr(model,'lambda_geo',0.));self.collect_geo=bool(getattr(model,'collect_geo',False));self.latest_geo=None

    def loss(self,preds,batch):
        self.geo_batch=batch;self.latest_geo=None
        return super().loss(preds,batch)

    def calculate_keypoints_loss(self,masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts):
        values=super().calculate_keypoints_loss(masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts)
        if self.lambda_geo==0 and not self.collect_geo:return values
        zero=pred_kpts.sum()*0.;self.latest_geo=zero
        if not masks.any():return values
        batch=self.geo_batch
        if 'geo_H' not in batch:raise AssertionError('missing geo metadata')
        H=batch['geo_H'].to(pred_kpts.device);enabled=batch['geo_enabled'].to(pred_kpts.device)
        assert H.shape==(masks.shape[0],16,16) and enabled.shape==(masks.shape[0],)
        imgidx=torch.nonzero(masks,as_tuple=False)[:,0]
        target=self._select_target_keypoints(keypoints,batch_idx,target_gt_idx,masks)[masks]
        st=stride_tensor.view(1,-1,1).expand(masks.shape[0],-1,1)[masks]
        pred=pred_kpts[masks][:,:8,:2]*st[:,None,:]
        gt=target[:,:8,:2];mask=target[:,:8,2]==2
        # One object per source occurrence; target indices must be native instance0.
        assert (target_gt_idx[masks]==0).all()
        self.latest_geo=quadratic(pred,gt,mask,H[imgidx],enabled[imgidx])
        kpt,obj,rle=values
        return kpt+self.lambda_geo/self.hyp.pose*self.latest_geo,obj,rle

def criterion(model):return E2ELoss(model,GeoLoss)

def geo_total(loss,batch_size):
    terms=[]
    for part,weight in [(loss.one2many,loss.o2m),(loss.one2one,loss.o2o)]:
        if part.latest_geo is not None:terms.append(part.latest_geo*weight*batch_size)
    return sum(terms)
