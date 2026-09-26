"""Original true-ignore pose loss with hard-slot detection/visibility gradients removed."""
import torch
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xyxy2xywh
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26


class HardXYLoss(TrueIgnorePoseLoss26):
    def loss(self,preds,batch):
        self.hard=torch.as_tensor(batch.get('hard',[False]*preds['scores'].shape[0]),device=self.device,dtype=torch.bool)
        return super().loss(preds,batch)

    def get_assigned_targets_and_loss(self,preds,batch):
        if not self.hard.any():return super().get_assigned_targets_and_loss(preds,batch)
        loss=torch.zeros(3,device=self.device)
        distr=preds['boxes'].permute(0,2,1).contiguous();scores=preds['scores'].permute(0,2,1).contiguous()
        anchors,strides=make_anchors(preds['feats'],self.stride,.5)
        hw=torch.tensor(preds['feats'][0].shape[2:],device=self.device,dtype=scores.dtype)*self.stride[0]
        targets=torch.cat((batch['batch_idx'].view(-1,1),batch['cls'].view(-1,1),batch['bboxes']),1)
        targets=self.preprocess(targets.to(self.device),scores.shape[0],scale_tensor=hw[[1,0,1,0]])
        labels,boxes=targets.split((1,4),2);predboxes=self.bbox_decode(anchors,distr)
        _,tb,ts,fg,idx=self.assigner(scores.detach().sigmoid(),(predboxes.detach()*strides).type(boxes.dtype),anchors*strides,labels,boxes,boxes.sum(2,keepdim=True).gt_(0.))
        # Retain original full-batch denominator; zero hard contributions only.
        denom=max(ts.sum(),1);keep=~self.hard
        bce=self.bce(scores,ts.to(scores.dtype))
        if self.class_weights is not None:bce*=self.class_weights
        loss[1]=(bce*keep[:,None,None]).sum()/denom
        detfg=fg&keep[:,None]
        if detfg.any():
            loss[0],loss[2]=self.bbox_loss(distr,predboxes,anchors,tb/strides,ts,denom,detfg,hw,strides)
        loss[0]*=self.hyp.box;loss[1]*=self.hyp.cls;loss[2]*=self.hyp.dfl
        return (fg,idx,tb,anchors,strides),loss,loss.detach()

    def calculate_keypoints_loss(self,masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts):
        if not self.hard.any():
            return super().calculate_keypoints_loss(masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts)
        selected=self._select_target_keypoints(keypoints,batch_idx,target_gt_idx,masks)
        selected[...,:2]/=stride_tensor.view(1,-1,1,1)
        xy=pred_kpts.sum()*0;obj=xy;rle=xy
        if masks.any():
            gt=selected[masks];pred=pred_kpts[masks]
            area=xyxy2xywh((target_bboxes/stride_tensor)[masks])[:,2:].prod(1,keepdim=True)
            supervise=gt[...,2]==2;ignore=gt[...,2]==1
            xy=self.keypoint_loss(pred,gt,supervise,area)
            if self.rle_loss is not None and pred.shape[-1] in (4,5):rle=self.calculate_rle_loss(pred,gt,supervise).clamp(min=0)
            if pred.shape[-1] in (3,5):
                normal=(~self.hard[:,None].expand_as(masks))[masks]
                keep=(~ignore)&normal[:,None]
                if keep.any():
                    terms=torch.nn.functional.binary_cross_entropy_with_logits(pred[...,2],supervise.to(pred.dtype),reduction='none')
                    obj=(terms*keep).sum()/keep.sum()
        return xy,obj,rle


def criterion(model):
    from ultralytics.utils.loss import E2ELoss
    assert model.end2end
    return E2ELoss(model,HardXYLoss)
