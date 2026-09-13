import inspect
import hashlib
import numpy as np
import pytest
import torch
from common import *
from generic_point_refiner import GenericPointRefiner,decode,targets,loss,sample
from point_inference import replace_selected,PointInference

@pytest.fixture(scope='module')
def fixture():
    torch.manual_seed(7);torch.set_num_threads(4)
    m=GenericPointRefiner(c3=2,c4=3,stencil_fraction=.13)
    p=torch.tensor([[[24.,24.]]*9]);box=torch.tensor([[8.,8.,48.,48.]])
    args=[torch.randn(1,2,8,8),torch.randn(1,3,4,4),p,box,torch.ones(1,9,dtype=torch.bool),torch.tensor([[64.,64.]])]
    out=m(*args)
    return m,args,out

def test_r0_sha():assert sha(R0)==R0_SHA
@pytest.mark.parametrize('seed',[1,2,3])
def test_old_checkpoint(seed):
    v=read(B/'LINE_SOURCE_BINDING.json')['seeds'][str(seed)];assert sha(ROOT/v['checkpoint'])==v['checkpoint_sha256']
def test_source_manifest():
    p=str((LINE/'SOURCE_MANIFEST.json').relative_to(ROOT));assert sha(ROOT/p)==read(B/'LINE_SOURCE_BINDING.json')['paths'][p]
def test_zero_correction(fixture):assert torch.equal(decode(fixture[2],lam=0),fixture[1][2])
def test_center(fixture):assert torch.equal(fixture[2]['points'][:,8],fixture[1][2][:,8])
@pytest.mark.parametrize('field',['box_xyxy','score','candidate_index'])
def test_candidate_preservation(field):
    c=[dict(candidate_index=i,score=.8-i*.1,box_xyxy=np.arange(4.),keypoints_xy=np.arange(18.).reshape(9,2)) for i in range(3)]
    r=replace_selected(c,1,np.zeros((9,2)),np.ones((9,2)),.5,1)
    assert all(np.array_equal(a[field],b[field]) for a,b in zip(c,r))
    assert np.array_equal(c[0]['keypoints_xy'],r[0]['keypoints_xy'])
    assert np.array_equal(c[2]['keypoints_xy'],r[2]['keypoints_xy'])
def test_feature_coordinate():
    p=torch.tensor([4.,4.]);v=old('model').pixel_to_grid(p,80,80,8)
    assert torch.allclose(v,torch.tensor([-0.9875,-0.9875]))
@pytest.mark.parametrize('stride',[8,16])
def test_bilinear(stride):
    a=torch.arange(16.).reshape(1,1,4,4)
    p=torch.tensor([[[[[(1.5)*stride,(1.5)*stride], [2*stride,2*stride]]]]])
    r=sample(a,p,torch.tensor([[4*stride,4*stride]]),stride)
    assert torch.allclose(r.flatten(),torch.tensor([5.,7.5]))
def test_cardinality(fixture):assert fixture[2]['logits'].shape==(1,8,222)
def test_single_null(fixture):assert int((fixture[0].displacements.norm(dim=-1)==0).sum())==1
def test_distinct_nonnull(fixture):assert len(torch.unique(fixture[0].displacements[:-1],dim=0))==221
def test_radius(fixture):assert torch.allclose(fixture[0].displacements.norm(dim=-1).max(),torch.tensor(.08))
def test_target_normalized(fixture):
    q=targets(fixture[2],fixture[1][2],fixture[1][4]);assert torch.allclose(q['distribution'].sum(-1),torch.ones(1,8))
def test_target_nearest(fixture):
    o=fixture[2];gt=fixture[1][2].clone();gt[:,:8]+=o['candidate_displacements'][:,14,None]
    assert (targets(o,gt,fixture[1][4])['distribution'].argmax(-1)==14).all()
@pytest.mark.parametrize('perm',[list(range(9)),[2,3,0,1,6,7,4,5,8]],ids=['C1','C2'])
def test_canonical_assignment_roundtrip(fixture,perm):
    # Existing L uses canonical fixed assignments, not a C2 invariant loss.
    # Reindexing predictions and labels together commutes with point targets.
    o=fixture[2];gt=fixture[1][2]+torch.arange(9)[None,:,None]*.1
    q=targets(o,gt,fixture[1][4])['distribution']
    remapped={**o,'points_raw':o['points_raw'][:,perm], 'point_support':o['point_support'][:,perm[:8]]}
    r=targets(remapped,gt[:,perm],fixture[1][4][:,perm])['distribution']
    assert torch.equal(r,q[:,perm[:8]]) and np.array_equal(np.array(perm)[perm],np.arange(9))
def test_no_topology():
    source=inspect.getsource(GenericPointRefiner);assert all(s not in source for s in ['SIDE_EDGES','WLS','Hough','neighbor','EDGES'])
def test_frozen_yolo_source():
    source=inspect.getsource(old('features').FrozenYoloFeatures)
    assert 'requires_grad_(False)' in source and '.detach()' in source
    assert 'R0' not in inspect.getsource(GenericPointRefiner.forward)
def test_optimizer_only_point(fixture):
    m=fixture[0];opt=torch.optim.AdamW(m.parameters());assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters()}
def test_finite_gradient(fixture):
    m,args,_=fixture;m.zero_grad(set_to_none=True);o=m(*args);l=loss(o,args[2]+1,args[4]);l.backward()
    assert torch.isfinite(l) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.adapt3[0].weight.grad.norm()>0 and m.adapt4[0].weight.grad.norm()>0 and m.patch_body[0].weight.grad.norm()>0
@pytest.mark.parametrize('seed',[1,2,3])
def test_order(seed):
    o=np.load(BRAW/f'order_seed{seed}.npy');h=hashlib.sha256(o.astype('<i8').tobytes()).hexdigest()
    assert h==read(B/'INIT_ORDER_PARITY.json')['seeds'][str(seed)]['batch_order_sha256']
def test_parameters():
    x=read(B/'PARAMETER_BUDGET_LOCK.json');assert sum(p.numel() for p in GenericPointRefiner(**x['config']).parameters())==x['P_params'] and abs(x['relative_difference'])<=.1
def test_no_gt_forward():
    assert all('gt' not in n.lower() and 'target' not in n.lower() for n in inspect.signature(GenericPointRefiner.forward).parameters)
    assert 'gt' not in inspect.signature(PointInference.predict).parameters
def test_invalid(fixture):
    m,args,_=fixture;args=list(args);args[2]=args[2].clone();args[2][0,0]=-1;args[4]=args[4].clone();args[4][0,1]=False
    out=m(*args);assert torch.equal(out['points'][0,:2],args[2][0,:2])
def test_coordinate_units():
    c=[dict(keypoints_xy=np.ones((9,2))*100)];q=np.ones((9,2))*10
    r=replace_selected(c,0,q,q+2,.5,1)
    assert np.array_equal(r[0]['keypoints_xy'][:8],np.ones((8,2))*104) and np.array_equal(r[0]['keypoints_xy'][8],c[0]['keypoints_xy'][8])
def test_selection_freeze_gate():
    assert "selection['complete'] and selection['no_real_selection']" in inspect.getsource(PointInference.__init__)
    assert read(DOC/'MASTER_PROTOCOL_LOCK.json')['B_fits']==3
def test_empty_supervision(fixture):
    _,a,o=fixture;v=torch.zeros_like(a[4]);assert loss(o,a[2],v)==0
def test_cap(fixture):
    o=fixture[2];q=decode(o,lam=4,cap=.1);assert (q[:,:8]-o['points_raw'][:,:8]).norm(dim=-1).max()<=.10001
