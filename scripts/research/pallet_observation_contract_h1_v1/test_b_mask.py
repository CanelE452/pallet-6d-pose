import unittest
import numpy as np
from .b_mask_diagnostic import line_labels,transition_summary


class MaskOwnershipTests(unittest.TestCase):
    def test_anti_aliased_values_remain_ambiguous(self):
        mask=np.array([[0,1,2,254,255]],np.uint8);xy=np.array([[i,0]for i in range(5)])
        labels=line_labels(mask,xy,encoding='renderer_uint8')
        self.assertEqual(labels.tolist(),[0,-1,-1,-1,1]);self.assertIsNone(transition_summary(labels)['full_transition_count'])

    def test_target_runs_are_not_brightness_changes(self):
        self.assertEqual(transition_summary([0,0,1,1,0,0])['transition_class'],'MULTIPLE_TRANSITIONS')
        self.assertEqual(transition_summary([0,0,1,1,1])['transition_class'],'SINGLE_TRANSITION')

    def test_nearest_pixel_original_rounding_and_outside_are_explicit(self):
        mask=np.array([[255,0]],np.uint8)
        self.assertEqual(line_labels(mask,[[-.4,0],[1.49,0],[1.5,0]],encoding='renderer_uint8').tolist(),[1,0,-2])

    def test_sampler_binary_encoding_does_not_promote_renderer_gray(self):
        self.assertEqual(line_labels(np.array([[0,1]]),[[0,0],[1,0]],encoding='sam_binary').tolist(),[0,1])
        self.assertEqual(line_labels(np.array([[0,1]]),[[0,0],[1,0]],encoding='renderer_uint8').tolist(),[0,-1])


if __name__=='__main__':unittest.main()
