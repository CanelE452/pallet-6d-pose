import unittest
import numpy as np

from .foundation import build_prompts,choose_mask,fixed_support_appearance
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C


class FoundationTests(unittest.TestCase):
    def test_quality_ties_keep_original_index_and_input_is_preserved(self):
        masks=np.array([[[1,0],[0,1]],[[0,1],[1,0]]],bool)
        preserved=masks.copy();selected,index=choose_mask(masks,[.85,.85])
        self.assertEqual(index,0);selected[:]=False
        np.testing.assert_array_equal(masks,preserved)
        with self.assertRaises(ValueError):choose_mask(masks,[np.nan,.9])

    def test_prompts_use_observed_box_and_no_anchor_when_RGB_is_unsupported(self):
        q=np.array([[30,30],[120,30],[120,40],[30,40],[40,100],[110,100],[110,110],[40,110]],float)
        app=dict(visibility_score=[.99]*8,diagnostics=dict(patch_radius_px=6))
        xyz=C.cuboid(1.,.1,1.)
        prompts=build_prompts((150,170),q,np.zeros(8,bool),app,xyz,np.eye(3),np.array([0.,0.,3.]),np.zeros((150,170),np.uint8))
        self.assertEqual(prompts[0]['box'],[30.,30.,120.,110.])
        self.assertEqual(prompts[1]['anchors'],[])
        self.assertTrue(all(x==0 for x in prompts[1]['labels']))
        np.testing.assert_array_equal(q[0],[30,30])
        hidden=np.ones(8,bool)
        prompts=build_prompts((150,170),q,hidden,app,xyz,np.eye(3),np.array([0.,0.,3.]),np.ones((150,170),np.uint8))
        self.assertEqual(prompts[1]['anchors'],[])

    def test_mask_support_cannot_override_original_geometry_and_is_not_a_new_head(self):
        mask=dict(states=['VISIBLE_INTERIOR']*6+['SELF_OCCLUDED','OUT_OF_FRAME'],
            visibility_score=[1.]*8,boundary_score=[.8]*8,local_accuracy_score=[0.]*8,
            diagnostics=dict(corners=[dict(reason='frozen_source_CNN_mask_and_raw_RGB_features',inward_material_fraction=v)
                                      for v in [.9,.4,.8,.9,.9,.9]]+
                            [dict(reason='supplied_initial_pose_self_occlusion'),dict(reason='predicted_coordinate_outside_frame')]))
        raw=dict(visibility_score=[.9,.9,.2,.9,.9,.9,0.,0.])
        result=fixed_support_appearance(mask,raw)
        self.assertEqual(result['states'][1:3],['UNKNOWN','UNKNOWN'])
        self.assertEqual(result['states'][6:],['SELF_OCCLUDED','OUT_OF_FRAME'])
        self.assertEqual(result['visibility_score'][0],.9)
        self.assertEqual(mask['states'][1],'VISIBLE_INTERIOR')


if __name__=='__main__':unittest.main()
