import json
import unittest

import numpy as np

from . import cycle_manual as M


class ManualCapabilityContracts(unittest.TestCase):
    def test_padded_raw_outside_native_is_retained_without_confidence(self):
        q=np.zeros((9,2),float);q[5]=[670.,300.]
        mask=np.zeros(9,bool);mask[5]=True
        values=np.asarray(M.fixed_label([120,80,520,420],q,mask,[480,640]).split(),float)
        points=values[5:].reshape(9,3)
        self.assertEqual(points[5,2],2)
        np.testing.assert_allclose(points[5,:2]*[840,680]-100,q[5],atol=1e-8,rtol=0)
        self.assertTrue(np.all(points[np.arange(9)!=5,2]==1))
        self.assertTrue(np.all(points[np.arange(9)!=5,:2]==.5))

    def test_center_and_canvas_fail_closed(self):
        q=np.full((9,2),10.);mask=np.zeros(9,bool);mask[8]=True
        with self.assertRaises(AssertionError):M.fixed_label([0,0,100,100],q,mask,[480,640])
        mask[8]=False;mask[0]=True;q[0,0]=741
        with self.assertRaises(AssertionError):M.fixed_label([0,0,100,100],q,mask,[480,640])

    def test_no_completed_point_becomes_manual(self):
        annotation={'camera_data':{'height':480,'width':640},'objects':[{'keypoint_annotations':[
            dict(xy=[20,30],source='manual_click' if i in (0,4) else 'pnp_projected',
                 visibility=2 if i in (0,4) else 1,reason='visible' if i in (0,4) else 'unknown',in_frame=True)
            for i in range(9)]}]}
        mask=np.zeros(9,bool);mask[[0,4]]=True
        _,actual=M.manual_points(annotation,mask);np.testing.assert_array_equal(actual,mask)
        annotation['objects'][0]['keypoint_annotations'][0]['source']='unknown'
        with self.assertRaises(AssertionError):M.manual_points(annotation,mask)

    def test_same_box_and_support_only_xy_changes(self):
        mask=np.zeros(9,bool);mask[:4]=True
        q=np.full((9,2),50.);r=q+2
        a,b=[np.asarray(M.fixed_label([0,0,100,100],p,mask,[480,640]).split(),float) for p in (q,r)]
        np.testing.assert_array_equal(a[:5],b[:5])
        np.testing.assert_array_equal(a[5:].reshape(9,3)[:,2],b[5:].reshape(9,3)[:,2])
        self.assertFalse(np.array_equal(a,b))


if __name__=='__main__':unittest.main()
