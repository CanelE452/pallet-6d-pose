import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.research.pallet_large_error_refiner_v1 import run as R
from scripts.research.pallet_large_error_refiner_v1.evaluate import recovery_damage, gate


class Contract(unittest.TestCase):
    def test_manual_source_not_global_gt_source(self):
        entries = [dict(xy=[10,20],visibility=2,in_frame=True,source=s)
                   for s in ['manual_click','pnp_projected','unknown','extrapolated','manual_click','manual_click','manual_click','manual_click','manual_click']]
        entries[4]['in_frame'] = False; entries[5]['visibility'] = 0; entries[6]['xy'] = [-2,3]
        doc = dict(camera_data=dict(width=640,height=480),objects=[dict(gt_source='manual',keypoint_annotations=entries)])
        _,v = R.manual_target(doc)
        self.assertEqual(v.tolist(),[True,False,False,False,False,False,False,True,False])

    def test_fixed_denominator_inclusive_recovery(self):
        base = [dict(id='a',evaluable=True,canonical_valid=[True]*4,canonical_errors=[25,30,2,4])]
        new = [dict(id='a',evaluable=True,canonical_valid=[True]*4,canonical_errors=[10,11,11,10])]
        m = recovery_damage(base,new)
        self.assertEqual(m,dict(hard=2,recovered=1,recovery_rate=.5,good=2,damaged=1,damage_rate=.5))

    def test_empty_denominator_not_zero(self):
        rows = [dict(id='a',evaluable=True,canonical_valid=[True],canonical_errors=[7])]
        m = recovery_damage(rows,rows)
        self.assertIsNone(m['recovery_rate']); self.assertIsNone(m['damage_rate'])

    def test_low_confidence_stops_before_pnp(self):
        p = dict(selected_index=0,candidates=[dict(score=.1,keypoints_xy=np.ones((9,2)),keypoints_conf=np.ones(9))])
        lock = R.E.read(R.FILTER)
        result = gate(p,p,640,None,None,lock)
        self.assertFalse(result['accepted']); self.assertEqual(result['reason'],'confidence/valid corners')

    def test_missing_detection_fails(self):
        p = dict(selected_index=None,candidates=[])
        self.assertFalse(gate(p,p,640,None,None,None)['accepted'])


if __name__ == '__main__': unittest.main()
