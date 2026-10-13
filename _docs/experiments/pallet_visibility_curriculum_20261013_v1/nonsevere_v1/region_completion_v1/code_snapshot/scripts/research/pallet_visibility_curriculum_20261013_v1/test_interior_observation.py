from copy import deepcopy
import unittest

import cv2
import numpy as np

from . import common as C
from . import interior_observation as I


def toy_scene(image_shift=(0.,0.)):
    """Image vertex can move independently of the fixed model/query."""
    K=np.array([[180.,0.,112.],[0.,180.,112.],[0.,0.,1.]])
    R=cv2.Rodrigues(np.array([-.25,.4,.07]))[0];t=np.array([0.,0.,3.])
    d=np.array([1.2,.6,1.]);X=C.cuboid(*d);P=C.project(X,R,t,K);H=C.self_hidden(X,R,t)
    hull=np.zeros(8,bool);hull[cv2.convexHull(P.astype(np.float32),returnPoints=False).ravel()]=True
    k=int(np.flatnonzero(~hull&~H)[0]);q=np.vstack([P,P.mean(0)]);q[k]+=[2.,-1.]
    actual=P.copy();actual[k]+=image_shift
    image=np.full((224,224,3),230,np.uint8)
    faces=((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
    normals=np.array([[0,0,-1],[0,0,1],[0,-1,0],[1,0,0],[0,1,0],[-1,0,0]])
    camera=-R.T@t
    for index,(face,n) in enumerate(zip(faces,normals)):
        if n@(camera-X[list(face)].mean(0))>0:
            color=(45,75,105,145,185,210)[index]
            cv2.fillConvexPoly(image,np.rint(actual[list(face)]).astype(np.int32),(color,)*3)
    pose=dict(available=True,R_cf=R.tolist(),centroid=t.tolist(),cf_extents=d.tolist())
    observation=dict(raw_points=q.tolist(),fused_points=q.tolist(),self_hidden=H.tolist(),fixed_H=H.tolist(),
        boundary_visibility=dict(expected_outline=hull.tolist(),conditional_truncation=[False]*8),
        visibility_appearance=dict(visibility_score=np.where(H,0.,.95).tolist(),states=['SELF_OCCLUDED'if value else'VISIBLE_INTERIOR'for value in H],
            local_accuracy_score=[.12]*8,diagnostics={}))
    return image,q,K,pose,observation,k


class InteriorObservationTests(unittest.TestCase):
    def test_actual_three_face_junction_produces_bounded_independent_measurement(self):
        image,q,K,pose,o,k=toy_scene()
        before=(image.copy(),q.copy(),K.copy(),deepcopy(o))
        result=I.infer_interior_observations(image,q,K,pose,o)
        self.assertIn(str(k),result['supplemental_observations'])
        r=result['supplemental_observations'][str(k)]
        self.assertEqual(len(r['lines']),2)
        self.assertGreaterEqual(r['successful_pair_resamples'],16)
        self.assertLessEqual(r['applied_shift_px'],6.)
        self.assertFalse(r['semantic_visibility_certified'])
        self.assertFalse(r['coordinate_confidence_is_calibrated'])
        self.assertGreater(np.linalg.eigvalsh(r['covariance_px2']).min(),0.)
        self.assertTrue(np.array_equal(image,before[0]) and np.array_equal(q,before[1]) and np.array_equal(K,before[2]))
        self.assertEqual(o,before[3]);self.assertEqual(q[8].tolist(),before[1][8].tolist())

    def test_image_change_moves_measurement_without_moving_model_projection(self):
        first,q,K,pose,o,k=toy_scene();second,_,_,_,_,_=toy_scene((3.,-2.))
        a=I.infer_interior_observations(first,q,K,pose,o)['supplemental_observations'][str(k)]
        b=I.infer_interior_observations(second,q,K,pose,o)['supplemental_observations'][str(k)]
        self.assertGreater(np.linalg.norm(np.asarray(a['candidate_xy'])-b['candidate_xy']),1.)
        self.assertNotEqual(a['image_pixel_sha256'],b['image_pixel_sha256'])

    def test_flat_image_is_absent_evidence_not_a_projected_point(self):
        image,q,K,pose,o,k=toy_scene();image[:]=110
        result=I.infer_interior_observations(image,q,K,pose,o)
        self.assertEqual(result['supplemental_observations'],{})
        self.assertIn(k,result['eligible_interior_ids'])
        self.assertFalse(result['provenance']['projected_coordinates_used_as_measurements'])

    def test_one_parallel_texture_family_cannot_localize_junction(self):
        image,q,K,pose,o,k=toy_scene();image[:]=80
        direction=C.project(C.cuboid(*pose['cf_extents']),pose['R_cf'],pose['centroid'],K)
        vector=direction[I.NEIGHBORS[k][0]]-direction[k];vector/=np.linalg.norm(vector)
        cv2.line(image,tuple(np.rint(q[k]-vector*60).astype(int)),tuple(np.rint(q[k]+vector*60).astype(int)),(210,)*3,2)
        result=I.infer_interior_observations(image,q,K,pose,o)
        self.assertEqual(result['supplemental_observations'],{})

    def test_wrong_actual_image_query_covariance_and_projection_claim_rejected(self):
        image,q,K,pose,o,k=toy_scene();result=I.infer_interior_observations(image,q,K,pose,o)
        for kind in ('hash','query','covariance','blend','projection','along_range'):
            bad=deepcopy(result);r=bad['supplemental_observations'][str(k)]
            if kind=='hash': bad['image_binding']['image_pixel_sha256']='0'*64
            if kind=='query': r['raw_query_xy'][0]+=1
            if kind=='covariance': r['covariance_px2'][0][0]+=.5
            if kind=='blend': r['measured_xy'][0]+=1
            if kind=='projection': bad['provenance']['projected_coordinates_used_as_measurements']=True
            if kind=='along_range':
                line=r['lines'][0];line['along_range_px'][0]-=2
                line['inlier_span_fraction']*= (line['along_range_px'][1]-line['along_range_px'][0]-2)/(line['along_range_px'][1]-line['along_range_px'][0])
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                I.validate_interior_observations(bad,q,K,pose,o,image_hw=image.shape[:2],image_pixel_sha256=I.image_pixel_sha256(image))

    def test_missing_outside_and_independent_occluder_never_added(self):
        image,q,K,pose,o,k=toy_scene()
        for mode in ('missing','outside','occluder'):
            qq=q.copy();oo=deepcopy(o)
            if mode=='missing':qq[k]=[-1.,-1.]
            if mode=='outside':qq[k]=[-2.,100.]
            if mode=='occluder':oo['occluding_evidence']={str(k):dict(schema='independent_occluding_region_v1',native_corner=k,
                query_xy=q[k].tolist(),independent_of_target_palette=True,different_instance_from_target=True,
                region_contains_query=True,region_observed_rgb=True,instance_id='other',source_reference='synthetic',source_type='instance_segmentation')}
            oo['raw_points']=qq.tolist()
            with self.subTest(mode=mode):
                result=I.infer_interior_observations(image,qq,K,pose,oo)
                self.assertNotIn(str(k),result['supplemental_observations'])

    def test_initial_h_or_truncation_binding_is_not_a_label_override(self):
        image,q,K,pose,o,k=toy_scene();bad=deepcopy(o);bad['self_hidden'][k]=True
        with self.assertRaises(ValueError):I.infer_interior_observations(image,q,K,pose,bad)
        bad=deepcopy(o);bad['boundary_visibility']['conditional_truncation'][k]=True
        with self.assertRaises(ValueError):I.infer_interior_observations(image,q,K,pose,bad)
        result=I.infer_interior_observations(image,q,K,pose,o)
        self.assertTrue(all(not o['self_hidden'][int(key)]for key in result['supplemental_observations']))


if __name__=='__main__': unittest.main()
