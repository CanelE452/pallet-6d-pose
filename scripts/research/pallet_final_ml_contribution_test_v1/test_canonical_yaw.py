"""Correct repository C2 yaw semantics; no change to fixed training targets."""
import numpy as np
import torch
from common import ROOT,module
from generic_point_refiner import GenericPointRefiner,targets

REFERENCE=ROOT/'scripts/research/pallet_symmetry_dht_local_v1/symdht_local/constants.py'

def test_repository_canonical_yaw_assignment():
    reference=module('final_ml_canonical_symmetry_reference',REFERENCE)
    perm=list(reference.C2_YAW)
    assert perm==[5,4,7,6,1,0,3,2,8] and np.array_equal(np.array(perm)[perm],np.arange(9))
    p=torch.arange(18,dtype=torch.float32).reshape(1,9,2)
    head=GenericPointRefiner(c3=1,c4=1)
    o=dict(points_raw=p,point_support=torch.ones(1,8,dtype=torch.bool),candidate_displacements=head.displacements[None]*100,box_diagonal=torch.tensor([100.]))
    gt=p+torch.arange(9)[None,:,None]*.1;valid=torch.ones(1,9,dtype=torch.bool)
    q=targets(o,gt,valid)['distribution']
    shifted={**o,'points_raw':p[:,perm],'point_support':o['point_support'][:,perm[:8]]}
    actual=targets(shifted,gt[:,perm],valid[:,perm])['distribution']
    assert torch.equal(actual,q[:,perm[:8]])
