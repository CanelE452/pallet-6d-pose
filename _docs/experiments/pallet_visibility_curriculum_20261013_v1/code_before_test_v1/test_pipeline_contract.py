"""Guard calibration compatibility and keep annotation access outside inference."""
import json
from pathlib import Path
import unittest
import numpy as np

from .calibration import features, predict
from .run import no_annotations
from .scoring import pose_metric


class IntegrationTests(unittest.TestCase):
    def test_rgb_head_applies_to_source_cnn_evidence_and_preserves_geometry_states(self):
        corners=[{'reason':'frozen_source_CNN_mask_and_raw_RGB_features'} for _ in range(8)]
        appearance=dict(states=['UNKNOWN']*6+['SELF_OCCLUDED','OUT_OF_FRAME'],visibility_score=[0.]*8,
                        boundary_score=[.8]*8,local_accuracy_score=[0.]*8,
                        diagnostics={'corners':corners,'patch_radius_px':10})
        model={'schema':'test','mean':[0.]*features(appearance).shape[1],
               'scale':[1.]*features(appearance).shape[1],
               'coefficients':[0.]*features(appearance).shape[1],
               'intercept':4.,'visible_threshold':.5,'boundary_threshold':.35}
        original=json.dumps(appearance)
        result=predict(appearance,model)
        self.assertEqual(json.dumps(appearance),original)
        self.assertTrue(all(state=='VISIBLE_BOUNDARY' for state in result['states'][:6]))
        self.assertEqual(result['states'][6:],['SELF_OCCLUDED','OUT_OF_FRAME'])
        self.assertEqual(result['visibility_score'][6:],[0.,0.])
        self.assertGreater(result['visibility_score'][0],.9)

    def test_all_evaluation_annotation_paths_are_blocked_before_open(self):
        paths=['/missing/TARGETS.json','/missing/SCORING_METADATA.json',
               '/missing/ORACLE_CAL_INPUTS.json','/missing/annotations/frame.json',
               '/missing/GEOMETRY_RESOLVED_POSE_GT.json']
        with no_annotations():
            for path in paths:
                with self.subTest(path=path):
                    with self.assertRaises(AssertionError):
                        Path(path).read_text()
                    with self.assertRaises(AssertionError):
                        open(path)

    def test_proper_symmetry_does_not_treat_width_depth_swap_as_rectangular_symmetry(self):
        truth={'R':np.eye(3).tolist(),'t':[0,0,3.],'xyz':[1.1,.11,1.3],'order':2}
        from .scoring import rotation_y
        p={'R_physical':rotation_y(np.pi).tolist(),'centroid':[0,0,3.]}
        metric=pose_metric(p,truth)
        self.assertLess(metric['rotation_deg'],1e-5)
        self.assertLess(metric['ADDsym_cm'],1e-8)
        p['R_physical']=rotation_y(np.pi/2).tolist()
        metric=pose_metric(p,truth)
        self.assertAlmostEqual(metric['rotation_deg'],90.,places=5)
        self.assertGreater(metric['ADDsym_cm'],50.)


if __name__=='__main__':
    unittest.main()
