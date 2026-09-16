"""Regression coverage for real caller errors found before final reporting."""
import json,unittest
import numpy as np
import torch
from metrics import measure,contrast
from b_model import raw_lattice
from preflight import geometry,rotations
from audit_math import derive_permutations
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES

class AdapterTests(unittest.TestCase):
    def test_no_annotation_is_not_zero(self):
        p=np.array([np.arange(9)])
        r=measure(np.zeros((9,2)),np.full((9,2),np.nan),np.zeros(9,bool),p,(480,640))
        self.assertEqual(r,{'evaluable':False})
    def test_partial_nan_target_strict_JSON(self):
        y=np.zeros((9,2));y[1]=np.nan;v=np.ones(9,bool);v[1]=False
        r=measure(np.zeros((9,2)),y,v,np.array([np.arange(9)]),(480,640))
        json.dumps(r,allow_nan=False);self.assertEqual(r['n_corners'],7)
        self.assertIsNone(r['target'][1][0])
    def test_missed_detection_full_penalty(self):
        r=measure(np.full((9,2),np.nan),np.zeros((9,2)),np.ones(9,bool),np.array([np.arange(9)]),(480,640))
        self.assertEqual(r['E_sym'],1);self.assertEqual(r['n_corners'],8)
    def test_no_metric_specific_branch(self):
        perms=derive_permutations(geometry(1,.1,1),rotations(4),np.array(EDGES))
        y=np.arange(18).reshape(9,2)*10;r=measure(y[perms[1]],y,np.ones(9,bool),perms,(480,640))
        self.assertEqual(r['E_sym'],0);self.assertEqual(r['branch'],1)
        self.assertTrue(np.array_equal(np.array(r['target']),y[perms[1]]))
    def test_cluster_pairing_not_seed_dataset_triplication(self):
        left=np.array([[1,2,3,4]]*3,float);right=left+1
        r=contrast(left,right,['s1','s1','s2','s2'])
        self.assertEqual(r['frames'],4);self.assertEqual(r['units'],2)
        self.assertEqual(r['CI95'],[-1,-1]);self.assertEqual(r['improved_seeds'],3)
    def test_raw_lattice_rho_bin_floor(self):
        lines=torch.tensor([[1.,0,0],[0,1,1]],dtype=torch.float64)
        l,s=raw_lattice(lines,torch.tensor([10.,20,12,24],dtype=torch.float64))
        self.assertTrue(torch.equal(s,torch.ones_like(s)))
        self.assertTrue(torch.allclose(l[:,:2].norm(dim=-1),torch.ones_like(s)))
if __name__=='__main__':unittest.main()
