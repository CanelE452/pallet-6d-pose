"""Constructed contract fixtures; no dataset, references, models or fits."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import observer as O
from . import features as F
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C


def fixture(blank=False):
    mask = np.zeros((144, 144), np.float32)
    mask[30:110, 30:110] = 1.
    image = np.repeat((20 + mask * 210).astype(np.uint8)[..., None], 3, axis=2)
    if blank:
        image[:] = 80; mask[:] = 0.
    projection = np.asarray([[34,34],[104,34],[104,104],[34,104],[52,52],[122,52],[122,122],[52,122]], float)
    raw = np.full((9, 2), np.nan); raw[0] = projection[0]
    geometry = dict(image_hw=[144,144], initial_self_hidden=[False]*8,
        initial_projection_xy=projection, native_xyz=C.cuboid(1.,1.,1.), cf_extents=[1.,1.,1.])
    states = ['B'] + ['O']*7
    binding = dict(sha256='a'*64, image_hw=[144,144])
    return image, mask, raw, geometry, states, binding


def observe_fixture(blank=False):
    image, mask, raw, geometry, states, binding = fixture(blank)
    return O.observe(image, mask, raw, geometry, states, frame_id='fixture', image_binding=binding)


def exact_partial():
    original = dict(original_measurement_source='VISIBLE_BACKBONE_FALLBACK', original_observed_xy=[34.,34.],
        original_observed_covariance_px2=[[4.,0.],[0.,4.]], original_effective_fit_covariance_px2=[[5.,0.],[0.,5.]],
        original_fit_reliability_weight=.8, original_raw_xy=[34.,34.])
    return dict(native_corner=0, state='B', image_binding=dict(sha256='a'*64,image_hw=[144,144]),
        fit_reliability_weight=.8, replaces_complete_BI_likelihood=True,
        explicit_complete_BI_likelihood_revocation=True, association_provenance=original,
        lines=[dict(normal_unit_xy=[1.,0.], offset_px=-30., normal_variance_px2=5.,
            support_tangent_interval_px=[-100.,100.], support_extrapolation_px=11.,
            provenance=dict(actual_RGB_line_id=1,native_neighbor=1))])


class CommonObserverContract(unittest.TestCase):
    def test_actual_cues_use_same_grid_and_original_line_operator(self):
        bank = observe_fixture(); corner = bank['corners'][0]
        self.assertEqual(len(corner['rays']), 6)
        self.assertEqual({r['native_neighbor'] for r in corner['rays']}, {1,3,4})
        self.assertGreater(corner['full_alternative_count'], 0)
        self.assertLessEqual(len(corner['candidates']), 13)
        self.assertEqual(bank['queried_native_ids'], [0])
        image, mask, raw, geometry, states, binding = fixture()
        lab = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
        gradients = cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
        projection = geometry['initial_projection_xy']
        for sampling in corner['samplings']:
            rays = [r for r in corner['rays'] if r['native_neighbor'] == sampling['native_neighbor']]
            self.assertEqual([r['query_grid_sha256'] for r in rays], [sampling['query_grid_sha256']]*2)
            self.assertEqual(sampling['grid_shape'], [15,49,2])
            for ray in rays:
                expected = O.OLD._line(image,mask,lab,gradients,raw,0,ray['native_neighbor'],
                    projection[ray['native_neighbor']]-projection[0],projection.mean(0),ray['cue'],O.CONFIGURATION)
                for key, value in expected.items():
                    self.assertEqual(ray[key], value)
                if ray['cue_summary']['selected_sample_count']:
                    self.assertTrue(all(s['valid'] for s in ray['cue_summary']['values'].values()))

    def test_intersection_applied_coordinate_and_own_covariance(self):
        bank = observe_fixture(); corner = bank['corners'][0]
        for candidate in corner['candidates'][1:]:
            q0 = np.asarray(candidate['q0']); raw = np.asarray(candidate['raw_intersection_xy'])
            alpha = candidate['blend_alpha']; cov = np.asarray(candidate['conditional_covariance']['covariance_px2'])
            np.testing.assert_array_equal(candidate['applied_xy'], q0+alpha*(raw-q0))
            np.testing.assert_allclose(candidate['covariance_px2'], (1-alpha)**2*4*np.eye(2)+alpha**2*cov+np.eye(2),rtol=0,atol=1e-12)
            self.assertEqual(len(set(candidate['source_neighbors'])), 2)
            self.assertEqual(len(candidate['physical_source_ids']), 2)
            self.assertLessEqual(alpha,.5)
        json.dumps(bank, allow_nan=False)

    def test_empty_unobserved_S_and_center_never_become_measurements(self):
        image, mask, raw, geometry, states, binding = fixture(True)
        states[4] = 'S'; geometry['initial_self_hidden'][4] = True
        raw[8] = [1e20,-1e20]
        bank = O.observe(image,mask,raw,geometry,states,frame_id='no-evidence',image_binding=binding)
        self.assertEqual([r['native_neighbor'] for r in bank['corners'][0]['samplings']], [1,3])
        self.assertEqual(len(bank['corners'][0]['candidates']),1)
        for corner in bank['corners'][1:]:
            self.assertFalse(corner['pixel_queried']); self.assertEqual(corner['candidates'],[])
        self.assertEqual(bank['corners'][0]['candidates'][0]['applied_xy'], [34.,34.])

    def test_short_original_model_guides_keep_valid_absence_metadata(self):
        image, mask, raw, geometry, states, binding = fixture(True)
        geometry['initial_projection_xy'][[1,3,4]] = [[35.,34.],[34.,35.],[35.,35.]]
        bank=O.observe(image,mask,raw,geometry,states,frame_id='shortguides',image_binding=binding)
        corner=bank['corners'][0]
        self.assertEqual(len(corner['rays']),6)
        self.assertTrue(all(r['reason']=='short_model_guide' for r in corner['rays']))
        self.assertEqual({r['native_neighbor'] for r in corner['rays']},{1,3,4})
        rows=F.feature_rows(bank)
        self.assertEqual(len(rows[0]['x_values']),1)
        self.assertTrue(np.isfinite(rows[0]['x_values']).all())

    def test_typed_partial_preserved_only_without_full_alternative(self):
        partial = exact_partial(); image, mask, raw, geometry, states, binding = fixture(True)
        bank = O.observe(image,mask,raw,geometry,states,frame_id='partial',image_binding=binding,shared_partial=[partial])
        self.assertEqual(bank['corners'][0]['partial_line'],partial)
        self.assertEqual(bank['corners'][0]['partial_original_visible']['xy'],[34.,34.])
        self.assertEqual(bank['corners'][0]['candidates'][0]['applied_xy'],[34.,34.])
        image, mask, raw, geometry, states, binding = fixture()
        bank = O.observe(image,mask,raw,geometry,states,frame_id='full',image_binding=binding,shared_partial=[partial])
        self.assertIsNone(bank['corners'][0]['partial_line'])
        partial['association_provenance']['original_raw_xy'] = [35.,34.]
        image, mask, raw, geometry, states, binding = fixture(True)
        with self.assertRaisesRegex(ValueError,'original N0'):
            O.observe(image,mask,raw,geometry,states,frame_id='badbinding',image_binding=binding,shared_partial=[partial])

    def test_raw_covariance_must_belong_to_same_anchor(self):
        image, mask, raw, geometry, states, binding = fixture(True)
        row = dict(native_corner=0,state='B',xy=[40.,40.],covariance_px2=[[99.,0.],[0.,99.]],source='VISIBLE_BACKBONE_FALLBACK')
        bank = O.observe(image,mask,raw,geometry,states,frame_id='wrongwinner',image_binding=binding,raw_measurements=[row])
        np.testing.assert_array_equal(bank['corners'][0]['candidates'][0]['covariance_px2'],np.eye(2)*4)
        row['xy']=[34.,34.]; row['covariance_px2']=[[4.,0.],[0.,4.]]
        row['fit_covariance_px2']=[[7.,1.],[1.,8.]]; row['fit_reliability_weight']=.8
        bank = O.observe(image,mask,raw,geometry,states,frame_id='matchingraw',image_binding=binding,raw_measurements=[row])
        self.assertEqual(bank['corners'][0]['candidates'][0]['covariance_px2'],row['covariance_px2'])
        self.assertEqual(bank['corners'][0]['candidates'][0]['fit_covariance_px2'],row['fit_covariance_px2'])
        self.assertEqual(bank['corners'][0]['candidates'][0]['fit_reliability_weight'],.8)
        row['covariance_px2']=[[1.,2.],[2.,1.]]
        with self.assertRaises(np.linalg.LinAlgError):
            O.observe(image,mask,raw,geometry,states,frame_id='badcov',image_binding=binding,raw_measurements=[row])

    def test_condition_changes_c_only_and_allowlist_excludes_identifiers(self):
        bank = observe_fixture(); corner=bank['corners'][0]; candidate=corner['candidates'][1]
        x,v=F.candidate_features(bank,corner,candidate)
        self.assertEqual(x.shape,(181,)); self.assertEqual(v.shape,(181,))
        self.assertTrue(np.isfinite(x).all()); self.assertTrue(np.isin(v,[0,1]).all())
        hostile=deepcopy(bank); hostile['frame_id']='GT_success_1'; hostile['states'][0]='I'
        hostile['reference_error_px']=0.; hostile['GT_state']='B'; hostile['image_binding']['sha256']='b'*64
        hostile['corners'][0]['candidates'][1]['candidate_id']='GT_winner'
        y,w=F.candidate_features(hostile,hostile['corners'][0],hostile['corners'][0]['candidates'][1])
        np.testing.assert_array_equal(x,y); np.testing.assert_array_equal(v,w)
        np.testing.assert_array_equal(F.condition('B','COMMON'),F.condition('I','COMMON'))
        self.assertFalse(np.array_equal(F.condition('B','BI'),F.condition('I','BI')))
        self.assertGreater(v.sum(),100)
        self.assertTrue(all(name not in F.FEATURE_NAMES for name in ('frame_id','state','reference_error_px')))
        rows=F.feature_rows(bank); self.assertEqual(rows[0]['native_corner'],0)
        self.assertEqual(len(rows[0]['x_values']),len(corner['candidates']))

    def test_parallel_out_of_frame_and_maximum_twelve_pair_contract(self):
        rays=[]
        for slot, mate in enumerate((1,3,4)):
            theta=slot*np.pi/3; n=np.array([np.cos(theta),np.sin(theta)]); t=np.array([-n[1],n[0]])
            for cue_i,cue in enumerate(('B','I')):
                offset=-float(n@np.array([50.,50.]))-(cue_i+1)*.5
                points=(-offset*n)[None,:]+np.linspace(-10.,10.,15)[:,None]*t
                rays.append(dict(edge=mate,supported=True,native_neighbor=mate,normal=n.tolist(),offset=offset,
                    observed_points=points.tolist(),observation_confidence=[1.]*15,inlier_mask=[True]*15,
                    source_line_id=f'm{mate}{cue}',physical_source_id=f'pixels{mate}{cue}',cue=cue))
        alternatives,_=O._full_candidates(0,np.array([50.,50.]),rays,[100,100])
        self.assertEqual(len(alternatives),12)
        for ray in rays: ray['normal']=[1.,0.]
        alternatives,rejected=O._full_candidates(0,np.array([50.,50.]),rays,[100,100])
        self.assertEqual(alternatives,[]); self.assertEqual(len(rejected),12)
        self.assertTrue(all(r['reason']=='parallel_or_ill_conditioned_lines' for r in rejected))
        rays[0]['normal']=[0.,1.]; rays[0]['offset']=-1000.
        alternatives,rejected=O._full_candidates(0,np.array([50.,50.]),rays,[100,100])
        self.assertEqual(alternatives,[])
        self.assertTrue(any(r['reason']=='intersection_outside_image_or_v2_trust_radius' for r in rejected))


if __name__=='__main__':
    unittest.main()
