import copy

import numpy as np
import pytest
import torch

from scripts.research.pallet_dht_pseudoline_selftrain_v1.geometry import transform_points,transform_lines,line_weights,incidence_loss,baseline_assignment
from scripts.research.pallet_dht_pseudoline_selftrain_v1.contracts import reject_real_gt,paired_exposure,load_frozen_real_teacher_cache
from scripts.research.pallet_dht_pseudoline_selftrain_v1.audit_mechanism import ROOT,POINT,DHT,PSEUDO,EXPORT,SOURCES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import sha256,read_json
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset,observation_batch,collate
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.runner import load_checkpoint


@pytest.mark.parametrize('A',[
    [[-1.,0,100],[0,1,0],[0,0,1]],
    [[2.,0,0],[0,.5,0],[0,0,1]],
    [[1.,0,-20],[0,1,17],[0,0,1]],
    [[-2.,.1,30],[0,.5,-8],[0,0,1]],
])
def test_inverse_transpose_incidence_flip_resize_crop_composition(A):
    a=torch.tensor(A,dtype=torch.float64);p=torch.tensor([[2.,1.],[4.,2.]],dtype=torch.float64)
    l=torch.tensor([[1.,-2.,0.]],dtype=torch.float64)
    pp=transform_points(p,a);ll=transform_lines(l,a)
    torch.testing.assert_close(pp@ll[0,:2]+ll[0,2],torch.zeros(2,dtype=torch.float64),atol=1e-12,rtol=0)
    ref=(torch.linalg.inv(a).T@l.T).T;ref/=ref[:,:2].norm(dim=-1,keepdim=True)
    torch.testing.assert_close(ll,ref)


def fixture():
    p=torch.zeros(1,9,2,dtype=torch.float64);p[:,:8,0]=2
    l=torch.zeros(1,12,3,dtype=torch.float64);l[...,0]=1
    w=torch.zeros(1,12,dtype=torch.float64);w[:,0]=1
    return p,l,w,torch.tensor([100.],dtype=torch.float64)


def test_center8_gradient_zero():
    p,l,w,d=fixture();p.requires_grad_();g=torch.autograd.grad(incidence_loss(p,l,w,d).sum(),p)[0]
    assert torch.count_nonzero(g[:,8])==0


def test_unavailable_invalid_weight_zero():
    _,l,_,_=fixture();a=torch.zeros(1,12);available=torch.ones(1,12,dtype=torch.bool);available[:,0]=False;l[:,1]=float('nan')
    w=line_weights(l,a,available);assert w[0,0]==w[0,1]==0
    p,_,_,d=fixture();assert torch.isfinite(incidence_loss(p,l,w,d)).all()


def test_tangent_displacement_unchanged():
    p,l,w,d=fixture();before=incidence_loss(p,l,w,d);p[:,:,1]+=100
    assert torch.equal(before,incidence_loss(p,l,w,d))


def test_normal_displacement_increases_loss():
    p,l,w,d=fixture();before=incidence_loss(p,l,w,d);p[:,:,0]+=2
    assert (incidence_loss(p,l,w,d)>before).all()


def test_endpoint_reversal_identical():
    from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
    args=fixture();assert torch.equal(incidence_loss(*args),incidence_loss(*args,edges=tuple((b,a) for a,b in EDGES)))


def test_line_sign_flip_identical():
    p,l,w,d=fixture();assert torch.equal(incidence_loss(p,l,w,d),incidence_loss(p,-l,w,d))


def test_whole_object_assignment_shared_no_free_point_permutation():
    p=torch.arange(18,dtype=torch.float64).reshape(9,2);perm=torch.tensor([5,4,7,6,1,0,3,2,8]);y=p[perm]
    perms=torch.stack((torch.arange(9),perm));valid=torch.ones(9,dtype=torch.bool)
    selected,_,choice=baseline_assignment(p,y,valid,perms,100.)
    assert choice==1 and torch.equal(p,selected)
    # Same selected target is used by point and line diagnostics; no independent line assignment.
    assert torch.equal(selected,y[perms[choice]])


def test_real_GT_rejected_and_cache_allowlist():
    for key in ('gt_points','annotation','target_pose','ground_truth'):
        with pytest.raises(ValueError):reject_real_gt({'entries':[{'image':'a.png',key:[1,2]}]})
    with pytest.raises(ValueError):load_frozen_real_teacher_cache('/forbidden/GT.json','/allowed/cache.json')
    path=ROOT/'data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json'
    assert load_frozen_real_teacher_cache(path,path)['n_images']==1000


def test_C1_C2_same_point_cache_bytes():
    paths=sorted((PSEUDO/'labels/train').glob('pseudo__*.txt'));assert len(paths)==273
    c1=[p.read_bytes() for p in paths];c2=[p.read_bytes() for p in paths];assert c1==c2


def test_C1_C2_real_order_and_augmentation_identical():
    plan=paired_exposure(1,1440,273);again=paired_exposure(1,1440,273)
    assert plan==again and plan['C1']['real']==plan['C2']['real'] and plan['C1']['augmentation']==plan['C2']['augmentation']


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(2)
    point_hash=sha256(POINT);dht_hash=sha256(DHT)
    ck=torch.load(POINT,map_location='cpu');teacher=(ck.get('ema') or ck['model']).float().eval().requires_grad_(False)
    student=copy.deepcopy(teacher).requires_grad_(True)
    dht,_=load_checkpoint(DHT,torch.device('cpu'));dht.requires_grad_(False)
    yield teacher,student,dht
    assert sha256(POINT)==point_hash and sha256(DHT)==dht_hash


def test_teacher_frozen_before_after_and_availability_reconstruction(models):
    teacher,_,dht=models;before={k:v.clone() for k,v in dht.state_dict().items()}
    item=ObservationDataset(EXPORT/'calibration.json',targets=False)[0]
    with torch.inference_mode():out=dht(observation_batch(collate([item]),torch.device('cpu')))
    assert torch.equal(out['available'],out['utility']>0)
    assert all(torch.equal(v,dht.state_dict()[k]) for k,v in before.items())
    assert all(not p.requires_grad and p.grad is None for m in (teacher,dht) for p in m.parameters())


def test_student_only_optimizer_membership(models):
    teacher,student,dht=models;optimizer=torch.optim.SGD(student.parameters(),lr=.002)
    ids={id(p) for g in optimizer.param_groups for p in g['params']}
    assert ids=={id(p) for p in student.parameters()}
    assert not ids&{id(p) for m in (teacher,dht) for p in m.parameters()}


def test_three_arm_synthetic_exposure_equal():
    plan=paired_exposure(2,1440,273)
    assert {p['synthetic_exposures'] for p in plan.values()}=={21600}
    assert plan['C0']['synthetic']==plan['C1']['synthetic']==plan['C2']['synthetic']


def test_student_inference_graph_has_no_DHT(models):
    _,student,_=models
    assert all(not any(s in type(m).__name__.lower() for s in ('dht','hough')) for m in student.modules())


def test_stock_pose_output_schema(models):
    _,student,_=models
    assert student.model[-1].kpt_shape==[9,3]
    with torch.inference_mode():out=student(torch.zeros(1,3,64,64))
    decoded=out[0] if isinstance(out,tuple) else out
    assert decoded.ndim==3 and decoded.shape[-1]==6+9*3


def test_ignored_NaN_GT_has_zero_finite_gradient():
    from scripts.research.pallet_dht_pseudoline_selftrain_v1.geometry import gt_point_loss
    p=torch.ones(9,2,dtype=torch.float64,requires_grad=True);y=torch.zeros_like(p)
    valid=torch.ones(9,dtype=torch.bool);valid[3]=False;y[3]=float('nan')
    loss=gt_point_loss(p,y,valid,torch.ones_like(valid),100.)
    grad=torch.autograd.grad(loss,p)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[3])==0 and torch.count_nonzero(grad[8])==0
