"""Implementation parity, original-frame coordinate states and no CNN recrop."""
import importlib.util
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

_s=importlib.util.spec_from_file_location("roi_candidate",Path(__file__).with_name("roi_appearance.py"))
R=importlib.util.module_from_spec(_s);sys.modules[_s.name]=R;_s.loader.exec_module(R)


def differences(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):return [(path,"keys",sorted(set(a)^set(b)))]
        return [d for k in a for d in differences(a[k],b[k],path+"/"+k)]
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [(path,"length",[len(a),len(b)])]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+f"/{i}")]
    if isinstance(a,(int,float,np.number)) and not isinstance(a,bool) and isinstance(b,(int,float,np.number)):
        return [] if abs(a-b)<=1e-10 else [(path,"number",[a,b])]
    return [] if a==b else [(path,"value",[a,b])]


def scene():
    image=np.full((1000,1200,3),235,np.uint8)
    points=np.array([[490.5,530.5],[650.5,480.5],[655.5,505.5],[495.5,555.5],
                     [450.5,390.5],[630.5,390.5],[635.5,415.5],[455.5,415.5]])
    hull=cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillConvexPoly(image,hull,(60,145,75))
    for offset in range(6):
        cv2.line(image,(460+offset*25,410),(490+offset*25,500),(30,100,45),2)
    probability=np.full(image.shape[:2],.03,np.float32)
    mask=np.zeros(image.shape[:2],np.uint8);cv2.fillConvexPoly(mask,hull,255)
    probability[mask>0]=.95
    return image,points,np.zeros(8,bool),probability


class RoiTests(unittest.TestCase):
    def test_rgb_and_segmented_parity_for_compact_half_integer_points(self):
        image,q,h,p=scene();descriptor=R.crop_descriptor(image,q,h)
        self.assertLess(descriptor["area_fraction"],.2)
        self.assertEqual(descriptor["bounds"][0]%2,0);self.assertEqual(descriptor["bounds"][1]%2,0)
        self.assertEqual(differences(R.A.infer_appearance(image,q,h),R.infer_appearance_roi(image,q,h)),[])
        self.assertEqual(differences(R.S.infer_segmented_appearance(image,q,h,p),R.infer_segmented_appearance_roi(image,q,h,p)),[])

    def test_missing_sentinel_and_true_frame_out_of_bounds_priorities_are_preserved(self):
        image,q,h,p=scene();q[0]=[-1,-1];q[1]=[-20,480];h[2]=True
        full=R.A.infer_appearance(image,q,h);roi=R.infer_appearance_roi(image,q,h)
        self.assertEqual(full["states"][0],"UNKNOWN")
        self.assertEqual(full["states"][1],"OUT_OF_FRAME")
        self.assertEqual(full["states"][2],"SELF_OCCLUDED")
        self.assertEqual(differences(full,roi),[])

    def test_full_original_cnn_mask_is_validated_before_slicing(self):
        image,q,h,p=scene();p[0,0]=np.nan
        with self.assertRaises(ValueError):R.infer_segmented_appearance_roi(image,q,h,p)

    def test_images_points_mask_and_hidden_flags_are_not_mutated(self):
        image,q,h,p=scene();copies=(image.copy(),q.copy(),h.copy(),p.copy())
        R.infer_appearance_roi(image,q,h);R.infer_segmented_appearance_roi(image,q,h,p)
        for original,preserved in zip((image,q,h,p),copies):np.testing.assert_array_equal(original,preserved)

    def test_true_image_border_nan_one_point_and_huge_hull_parity(self):
        image,q,h,p=scene()
        cases=[]
        border=q-q.min(0)+[.5,.5];border[1]=[-3.,45.];cases.append(border)
        missing=q.copy();missing[2]=[np.nan,np.nan];missing[3]=[-1.,-1.];cases.append(missing)
        one=np.full((8,2),np.nan);one[0]=[500.5,450.5];cases.append(one)
        huge=q.copy();huge[4]=[1e6,1e6];cases.append(huge)
        for points in cases:
            with self.subTest(points=points.tolist()):
                self.assertEqual(differences(R.A.infer_appearance(image,points,h),R.infer_appearance_roi(image,points,h)),[])
                self.assertEqual(differences(R.S.infer_segmented_appearance(image,points,h,p),R.infer_segmented_appearance_roi(image,points,h,p)),[])
        self.assertTrue(R.crop_descriptor(image,huge,h)["full_frame"])


if __name__=="__main__":unittest.main()
