import pytest
import torch
from scripts.research.pallet_dht_coupling_v2.gradient_surgery import parameter_dependencies, project_from_sum, mask_digest


@pytest.mark.parametrize('a,b,conflict', [([1.,0.],[1.,1.],False),([1.,0.],[-1.,1.],True),
    ([0.,0.],[1.,2.],False),([1.,2.],[0.,0.],False),([1.,2.],[-1.,-2.],True)])
def test_symmetric_reference_and_private_coordinates(a,b,conflict):
    a,b=torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)
    # Extremely large task-private gradients must neither affect the projection
    # denominator nor receive invented gradients from the other task.
    total=[a+b,torch.tensor([1e8],dtype=torch.float64),torch.tensor([-2e8],dtype=torch.float64),None]
    aux=[b,None,total[2],None]
    output,stats,mask,_=project_from_sum(total,aux,[True,True,False,False])
    expected=a+b
    dot=(a*b).sum()
    if conflict:
        expected=(a-dot/b.square().sum()*b)+(b-dot/a.square().sum()*a)
    torch.testing.assert_close(output[0],expected,atol=1e-14,rtol=1e-14)
    assert stats['conflict']==conflict and mask==[True,False,False,False]
    assert output[1] is total[1] and output[2] is total[2] and output[3] is None
    if not conflict:assert output[0] is total[0]


def test_actual_graph_mask_and_combined_backward_equivalence():
    shared=torch.nn.Parameter(torch.tensor([.4,-.8],dtype=torch.float64))
    main_private=torch.nn.Parameter(torch.tensor([3.],dtype=torch.float64))
    aux_private=torch.nn.Parameter(torch.tensor([4.],dtype=torch.float64))
    connected_zero=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    unused=torch.nn.Parameter(torch.tensor([5.],dtype=torch.float64))
    params=[shared,main_private,aux_private,connected_zero,unused]
    main=(shared*torch.tensor([2.,-1.])).sum()+main_private.square().sum()+connected_zero.sum()*0
    line=(shared*torch.tensor([-3.,1.])).sum()+aux_private.square().sum()+connected_zero.sum()*0
    deps=parameter_dependencies(main)
    a=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(line,params,retain_graph=True,allow_unused=True)
    assert [id(p) in deps for p in params]==[g is not None for g in a]
    (main+line).backward()
    total=[p.grad for p in params]
    output,stats,mask,reconstructed=project_from_sum(total,b,[id(p) in deps for p in params])
    assert mask==[True,False,False,True,False]
    for x,y,keep in zip(reconstructed,a,mask):
        if keep:torch.testing.assert_close(x,y,atol=0,rtol=0)
    dot=sum((x*y).sum() for x,y,keep in zip(a,b,mask) if keep)
    aa=sum(x.square().sum() for x,keep in zip(a,mask) if keep)
    bb=sum(x.square().sum() for x,keep in zip(b,mask) if keep)
    for i,keep in enumerate(mask):
        if keep:torch.testing.assert_close(output[i],a[i]+b[i]-dot/bb*b[i]-dot/aa*a[i])
        else:assert output[i] is total[i]
    assert stats['shared_tensors']==2


def test_disabled_preserves_original_sum_despite_conflict():
    total=[torch.tensor([1.,1.])];aux=[torch.tensor([-1.,2.])]
    out,stats,mask,_=project_from_sum(total,aux,[True],enabled=False)
    assert out[0] is total[0] and stats['conflict'] and not stats['applied']
    assert mask_digest(['shared'],mask)[0]==['shared']


def test_nonfinite_is_fatal():
    with pytest.raises(FloatingPointError):
        project_from_sum([torch.tensor([float('nan')])],[torch.ones(1)],[True])
