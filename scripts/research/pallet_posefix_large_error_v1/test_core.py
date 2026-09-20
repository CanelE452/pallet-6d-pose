import copy
import unittest
import numpy as np
from scripts.research.pallet_posefix_large_error_v1 import core as C
from scripts.research.pallet_posefix_large_error_v1.train import corrupted


class Contracts(unittest.TestCase):
    def prediction(self):
        return dict(candidates=[dict(box_xyxy=[10,20,210,120],score=.9,
            keypoints_xy=np.arange(18).reshape(9,2).tolist(),keypoints_conf=[.8]*9)],selected_index=0)

    def test_cap_and_preservation(self):
        base=self.prediction(); raw=copy.deepcopy(base)
        raw['candidates'][0]['keypoints_xy']=(np.array(raw['candidates'][0]['keypoints_xy'])+100).tolist()
        out=C.cap_prediction(base,raw,.01,(480,640))
        p=np.array(C.selected(base)['keypoints_xy']); q=np.array(C.selected(out)['keypoints_xy'])
        np.testing.assert_allclose(np.linalg.norm((q-p)[:8],axis=-1),8,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(q[8],p[8])
        for k in ('score','box_xyxy','keypoints_conf'): self.assertEqual(C.selected(base)[k],C.selected(out)[k])
        self.assertEqual(C.cap_prediction(base,raw,0,(480,640)),base)

    def test_manual_corruption_only(self):
        valid=np.array([1,0,1,0,0,0,0,0,0],bool)
        matrix=C.axis_aligned_crop_matrix(np.array([10,20,210,120]))
        x=dict(points=np.arange(18,dtype=np.float32).reshape(9,2),valid=np.ones(9,bool),
               target=np.ones((9,2),np.float32)*100,target_valid=valid,matrix=matrix,bbox_diagonal=200.)
        orig=x['points'].copy(); out=corrupted(x,np.random.default_rng(64),True)
        np.testing.assert_array_equal(x['points'],orig)
        np.testing.assert_array_equal(out['points'][~valid],orig[~valid])
        radii=np.linalg.norm(out['points'][valid]-x['target'][valid],axis=-1)/matrix[0,0]
        self.assertTrue(((radii>=10)&(radii<=30)).all())

    def test_gt_free_preprocess(self):
        p=self.prediction(); before=copy.deepcopy(p)
        x=C.prepare_input(np.zeros((240,320,3),np.uint8),p)
        self.assertEqual(p,before); self.assertEqual(x['rgb'].shape,(3,384,288))
        back=C.transform_points(x['points'],np.linalg.inv(x['matrix']))
        np.testing.assert_allclose(back,x['original_points'],atol=1e-5,rtol=0)


if __name__=='__main__': unittest.main()
