import torch
from torch import nn
from . import dino_mid_feature_model as M


def test_parent_initialization_and_zero_mid_path_exact():
    torch.set_num_threads(2);torch.manual_seed(9);a=M.V.Head();torch.manual_seed(9);b=M.Head()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    torch.nn.init.normal_(a.out.weight);torch.nn.init.normal_(a.out.bias);b.out.load_state_dict(a.out.state_dict())
    f=torch.randn(1,384,56,42);g=torch.randn_like(f);p=torch.zeros(1,9,2);v=torch.ones(1,9,dtype=torch.bool)
    torch.testing.assert_close(a(f,p,v),b((f,g),p,v),atol=0,rtol=0)


def test_both_feature_paths_learn_without_point_hints():
    m=M.Head();torch.nn.init.normal_(m.out.weight,std=.01);torch.nn.init.normal_(m.mid[-1].weight,std=.01)
    f=torch.randn(1,384,56,42,requires_grad=True);g=torch.randn_like(f,requires_grad=True)
    p=torch.ones(1,9,2)*200;v=torch.ones(1,9,dtype=torch.bool);z=m((f,g),p,v)
    torch.testing.assert_close(z,m((f,g),torch.full_like(p,float('nan')),~v),atol=0,rtol=0)
    M.loss(z,p,v).backward();assert f.grad.abs().sum()>0 and g.grad.abs().sum()>0 and m.mid[0].weight.grad.abs().sum()>0


def test_sixth_block_stops_exactly_and_removes_class_and_register_tokens():
    class Add(nn.Module):
        def __init__(self,value):super().__init__();self.value=value;self.calls=0
        def forward(self,x):self.calls+=1;return x+self.value
    class Fake(nn.Module):
        def __init__(self):
            super().__init__();self.blocks=nn.ModuleList([Add(i+1) for i in range(12)]);self.norm=nn.Identity();self.chunked_blocks=False;self.num_register_tokens=2
        def prepare_tokens_with_masks(self,x):return torch.zeros(1,7,4)
    b=Fake();mid=M.block_features(b,torch.zeros(1))[0]
    assert mid.shape==(1,4,4) and (mid==21).all()
    assert [z.calls for z in b.blocks]==[1]*6+[0]*6
    b=Fake();a,c=M.block_features(b,torch.zeros(1),True)
    assert (a==21).all() and (c==78).all() and [z.calls for z in b.blocks]==[1]*12


def test_center_and_invalid_targets_excluded():
    z=torch.zeros(1,9,192,144,requires_grad=True);p=torch.ones(1,9,2)*-20;p[:,8]=100
    v=torch.ones(1,9,dtype=torch.bool);loss=M.loss(z,p,v);loss.backward()
    assert loss==0 and z.grad.abs().sum()==0
