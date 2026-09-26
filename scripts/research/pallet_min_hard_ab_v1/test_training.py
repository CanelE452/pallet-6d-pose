"""Executable loss/gradient and 5,120-occurrence pair integrity gates."""
import copy
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from scripts.self_training_yolo.v3.true_ignore_pose_loss import make_criterion
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as S
from . import common as C
from .loss import criterion
from .train import FrozenDataset


def main():
    S.setup();gpu=S.guard();lock=C.read(C.DOC/'TRAIN_OCCURRENCE_LOCK.json')
    model=YOLO(str(C.ROOT/lock['original_init']['path']),task='pose').model.cuda().float().train()
    for p in model.parameters():p.requires_grad_(True)
    model.args=get_cfg(overrides=lock['args']);stock=make_criterion(model);new=criterion(model)
    image=torch.randn(2,3,128,128,device='cuda');pred=model(image)
    kp=torch.zeros(2,9,3,device='cuda');kp[:,:,:2]=torch.rand(2,9,2,device='cuda')*.4+.3;kp[:,:,2]=2;kp[:,4:,2]=1
    batch=dict(img=image,batch_idx=torch.tensor([0.,1.],device='cuda'),cls=torch.zeros(2,1,device='cuda'),
               bboxes=torch.tensor([[.5,.5,.8,.8]]*2,device='cuda'),keypoints=kp,hard=(False,False))
    a=stock(pred,batch)[0];b=new(pred,batch)[0];torch.testing.assert_close(a,b,rtol=0,atol=0)
    for hard in [(True,True),(False,True)]:
        pp=model(image)
        for branch in pp.values():
            if isinstance(branch,dict):
                for k in ('boxes','scores','kpts'):
                    branch[k].retain_grad()
        batch['hard']=hard;loss=new(pp,batch)[0]
        if all(hard):assert torch.equal(loss[[0,2,3,4]],torch.zeros_like(loss[[0,2,3,4]])),loss
        loss.sum().backward()
        for branch in (pp['one2many'],pp['one2one']):
            for k in ('boxes','scores'):
                grad=branch[k].grad
                if grad is not None:assert torch.count_nonzero(grad[torch.tensor(hard,device='cuda')])==0,(hard,k)
            g=branch['kpts'].grad.reshape(2,9,3,-1)
            assert torch.count_nonzero(g[torch.tensor(hard,device='cuda'),:,2])==0
            assert torch.count_nonzero(g[:,4:])==0
            assert torch.count_nonzero(g[torch.tensor(hard,device='cuda'),:4,:2])>0
        model.zero_grad(set_to_none=True)
    ds={a:FrozenDataset(a) for a in ('H_MANUAL','H_PSEUDO')};count=0;changed=0
    for epoch in range(5):
        for d in ds.values():d.epoch=epoch
        for i in range(1024):
            a,b=[d[i] for d in ds.values()]
            for k in ('img','bboxes','cls','batch_idx'):assert torch.equal(a[k],b[k]),k
            assert torch.equal(a['keypoints'][...,2],b['keypoints'][...,2])
            if a['hard']:
                count+=1;changed+=int(not torch.equal(a['keypoints'],b['keypoints']))
                assert torch.all(a['keypoints'][:,8,2]==1)
            else:assert torch.equal(a['keypoints'],b['keypoints'])
        print('PAIR_TEST',epoch+1,flush=True)
    assert count==changed==320
    C.save(C.DOC/'PRETRAIN_TESTS.json',dict(passed=True,original_loss_exact_no_hard=True,
           hard_detection_visibility_gradient_zero=True,hidden_gradient_zero=True,visible_xy_gradient_nonzero=True,
           paired_occurrences=5120,hard_occurrences=count,coordinate_different_occurrences=changed,
           same_RGB_box_support=True,gpu=gpu,created_at=C.now()),immutable=True)
    print('PRETRAIN_TESTS_PASS',flush=True)


if __name__=='__main__':main()
