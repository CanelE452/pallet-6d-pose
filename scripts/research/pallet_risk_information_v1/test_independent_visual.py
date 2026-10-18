from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from . import independent_visual as I,visual_patches as V
from scripts.research.pallet_boundary_action_repair_v1.gallery import EDGES


class IndependentVisualTests(unittest.TestCase):
    def test_original_queries_midpoints_degree_three(self):
        xyz=np.array([[-1,-2,-3],[1,-2,-3],[1,2,-3],[-1,2,-3],[-1,-2,3],[1,-2,3],[1,2,3],[-1,2,3]],float)
        self.assertTrue(np.array_equal(I.queries(xyz,EDGES),V.original_model_queries(xyz)));self.assertEqual(I.queries(xyz,EDGES).shape,(20,3))
    def test_scalar_projection_native_Q_and_center_origin(self):
        xyz=np.array([[-1,-2,-3],[1,-2,-3],[1,2,-3],[-1,2,-3],[-1,-2,3],[1,-2,3],[1,2,3],[-1,2,3]],float)+[.2,.1,.4]
        geometry=dict(native_xyz=xyz,K=[[500,0,100],[0,510,90],[0,0,1]],physical_registration_Q=[[0,0,1],[0,1,0],[-1,0,0]])
        pose=dict(R_physical=np.eye(3),centroid=[.1,.2,20.]);a,z=I.project(pose,geometry,EDGES);b,q=V.project_queries(pose,geometry)
        self.assertTrue(np.allclose(a,b,rtol=0,atol=1e-12));self.assertTrue(np.array_equal(z,q))
    def compare_sample(self,pixels,depths,side):
        rgb=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
        first=I.bilinear(rgb,pixels,depths,side);second=V.sample(rgb,pixels,depths,side)
        for a,b in zip(first,second):self.assertTrue(np.array_equal(a,b))
        return first
    def test_subpixel_bilinear(self):self.compare_sample(np.tile([3.25,2.75],(20,1)),np.ones(20),2.5)
    def test_offcanvas_never_clipped(self):
        p,v,q=self.compare_sample(np.tile([-100.,-100.],(20,1)),np.ones(20),5.);self.assertEqual(float(p.sum()),0.);self.assertTrue(q.all())
    def test_zero_negative_depth_nonfinite_center(self):
        xy=np.tile([3.,3.],(20,1));xy[2]=np.nan;z=np.ones(20);z[:2]=[0.,-1.]
        p,v,q=self.compare_sample(xy,z,3.);self.assertFalse(q[:3].any());self.assertEqual(float(p[:3].sum()),0.)
    def test_exact_last_pixel_and_missing_box(self):
        self.compare_sample(np.tile([8.,6.],(20,1)),np.ones(20),.1)
        p,v,q=self.compare_sample(np.tile([3.,3.],(20,1)),np.ones(20),None);self.assertEqual(float(p.sum()),0.);self.assertFalse(q.any())
    def test_FP16_cached_provider_returns_FP32_and_binary_mask(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fid='fixture';inputs=root/'inputs.jsonl.gz'
            import gzip
            with gzip.open(inputs,'wt')as f:f.write(json.dumps(dict(id=fid))+'\n')
            prefix=root/hashlib.sha256(fid.encode()).hexdigest();array=np.zeros((1,20,4,16,16),np.float16);array[:,:,:3]=.12345;array[:,:,3]=1.
            np.save(str(prefix)+'.npy',array);Path(str(prefix)+'.json').write_text(json.dumps(dict(candidate_ids=['N0'])))
            provider=V.PatchProvider(inputs,root);p,v=provider(dict(id=fid),[dict(candidate_id='N0')],[0])
            self.assertEqual(p.dtype,np.float32);self.assertTrue(np.array_equal(p,array.astype(np.float32)));self.assertTrue(np.array_equal(v,np.ones((1,20))))
    def test_NULL_conserves_masks_and_validity_without_model(self):
        p=np.arange(20*4*16*16,dtype=np.float32).reshape(20,4,16,16);null=np.concatenate((np.zeros_like(p[:,:3]),p[:,3:]),axis=1)
        self.assertFalse(null[:,:3].any());self.assertTrue(np.array_equal(p[:,3:],null[:,3:]))
    def test_prediction_API_static_whitelist(self):
        repo=Path(__file__).resolve().parents[3];checks,_=I.source_contracts(repo);self.assertTrue(all(checks.values()),checks)


if __name__=='__main__':unittest.main()
