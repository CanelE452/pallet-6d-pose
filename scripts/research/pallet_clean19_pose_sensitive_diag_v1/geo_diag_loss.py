"""Coordinate-separable proxy; does not imply output preservation through shared weights."""
import torch
from ultralytics.utils.loss import E2ELoss
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26

def diagonal(pred,gt,mask,weight,enabled):
    scalar_mask=mask.repeat_interleave(2,-1).to(pred.dtype)
    r=(pred-gt).reshape(-1,16)*scalar_mask
    n=scalar_mask.sum(-1);valid=enabled&(n>0)
    if not valid.any():return pred.sum()*0.
    values=(weight*r.square()*scalar_mask).sum(-1)/n.clamp_min(1)
    assert torch.isfinite(values).all()
    return values[valid].mean()

class DiagLoss(TrueIgnorePoseLoss26):
    def __init__(self,model,tal_topk=10,tal_topk2=None):
        super().__init__(model,tal_topk,tal_topk2)
        self.lambda_diag=float(getattr(model,'lambda_diag',0.));self.collect_diag=bool(getattr(model,'collect_diag',False));self.latest_diag=None
    def loss(self,preds,batch):
        self.diag_batch=batch;self.latest_diag=None
        return super().loss(preds,batch)
    def calculate_keypoints_loss(self,masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts):
        base=super().calculate_keypoints_loss(masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts)
        self.latest_diag=pred_kpts.sum()*0.
        if self.lambda_diag==0 and not self.collect_diag:return base
        if not masks.any():return base
        batch=self.diag_batch;weight=batch['diag_weight'].to(pred_kpts.device);enabled=batch['diag_enabled'].to(pred_kpts.device)
        assert weight.shape==(masks.shape[0],16) and enabled.shape==(masks.shape[0],)
        imgidx=torch.nonzero(masks,as_tuple=False)[:,0]
        assert (target_gt_idx[masks]==0).all() # all cached examples preflight verified single-object
        assert (torch.bincount(batch_idx.flatten().long(),minlength=masks.shape[0])==1).all()
        target=self._select_target_keypoints(keypoints,batch_idx,target_gt_idx,masks)[masks]
        stride=stride_tensor.view(1,-1,1).expand(masks.shape[0],-1,1)[masks]
        pred=pred_kpts[masks][:,:8,:2]*stride[:,None,:]
        gt=target[:,:8,:2];supervised=target[:,:8,2]==2
        self.latest_diag=diagonal(pred,gt,supervised,weight[imgidx],enabled[imgidx])
        if self.lambda_diag==0:return base
        location,obj,rle=base
        return location+self.lambda_diag/self.hyp.pose*self.latest_diag,obj,rle

def criterion(model):return E2ELoss(model,DiagLoss)
def diag_total(loss,batch_size):return batch_size*(loss.o2m*loss.one2many.latest_diag+loss.o2o*loss.one2one.latest_diag)
