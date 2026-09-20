import numpy as np
import torch
from . import dino_rgb_detail_model as M


def test_parent_initialization_and_zero_branch_exact():
    torch.set_num_threads(2);torch.manual_seed(7);a=M.V.Head();torch.manual_seed(7);b=M.Head()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    torch.nn.init.normal_(a.out.weight);torch.nn.init.normal_(a.out.bias)
    b.out.load_state_dict(a.out.state_dict())
    f=torch.randn(1,384,56,42);r=torch.randn(1,3,192,144);p=torch.zeros(1,9,2);v=torch.ones(1,9,dtype=torch.bool)
    torch.testing.assert_close(a(f,p,v),b((f,r),p,v),atol=0,rtol=0)


def test_learned_rgb_and_dino_paths_receive_gradient_without_point_hint():
    m=M.Head();torch.nn.init.normal_(m.out.weight,std=.01);torch.nn.init.normal_(m.detail[-1].weight,std=.01)
    f=torch.randn(1,384,56,42,requires_grad=True);r=torch.randn(1,3,192,144,requires_grad=True)
    p=torch.ones(1,9,2)*200;v=torch.ones(1,9,dtype=torch.bool);z=m((f,r),p,v)
    torch.testing.assert_close(z,m((f,r),torch.full_like(p,float('nan')),~v),atol=0,rtol=0)
    M.loss(z,p,v).backward()
    assert f.grad.abs().sum()>0 and r.grad.abs().sum()>0 and m.detail[0].weight.grad.abs().sum()>0


def test_rgb_grid_alignment_and_normalization():
    y,x=np.mgrid[:768,:576];rgb=np.stack([x/4,y/4,x*0+128]).astype(np.float32);mean=np.array([123.68,116.78,103.94],np.float32)
    out=M.detail_rgb(rgb-mean[:,None,None],mean)
    assert out.shape==(3,192,144) and out.dtype==np.float16
    np.testing.assert_allclose(out[:,10,20],np.array([20,10,128])/255-.5,atol=3e-4,rtol=0)


def test_center_and_invalid_targets_do_not_supply_supervision():
    z=torch.zeros(1,9,192,144,requires_grad=True);q=torch.ones(1,9,2)*-20;q[:,8]=100
    v=torch.ones(1,9,dtype=torch.bool);l=M.loss(z,q,v);l.backward()
    assert l==0 and z.grad.abs().sum()==0
