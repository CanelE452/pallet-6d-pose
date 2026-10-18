"""Meaningful contract checks; no network, reference, or optimizer invocation."""
import inspect
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import numpy as np

from . import pipeline_contract as C,d1_producer as P,d1_batch as B


class D1ContractTests(unittest.TestCase):
    def test_CPU_reporting_facade_keeps_keyword_defaults(self):
        from . import d1_cpu_logging_v3 as V3
        def original_batch(inputs,*,limit=None,technical=False,workers=1,GO=False):
            return inputs,limit,technical,workers,GO
        with patch.object(B,'solve_batch',original_batch):
            self.assertEqual(V3.solve_batch('input',workers=2,GO=True),('input',None,False,2,True))

    def test_producer_signatures_have_no_supervision(self):
        self.assertEqual(list(inspect.signature(P.observe).parameters),['frame','bundle'])
        self.assertEqual(list(inspect.signature(P.solve_e).parameters),['row','frame','mask','config','model','interval'])
        self.assertFalse({'target','reference','truth','visibility','bbox'}&C.ALLOWED)

    def test_unknown_reference_payload_rejected_before_RGB_read(self):
        for name in ('GT_pose','truth','reference','bbox','human_states'):
            with self.assertRaisesRegex(ValueError,'Unrecognized'):
                C.validate_input(dict(schema=C.SCHEMA,**{name:object()}))

    def test_technical_selection_is_preexisting_CAL_input_order(self):
        items=[dict(id='bad',partition='TEST'),dict(id='cal2',partition='CAL'),dict(id='cal1',partition='CAL')]
        with patch.object(B,'rows',return_value=iter(items)):
            self.assertEqual([r['id']for r in B.select_frames('unused','real',1,True)],['cal2'])
        with patch.object(B,'rows',return_value=iter(items)),self.assertRaises(ValueError):
            B.select_frames('unused','real',3,True)

    def test_numerical_meter_counts_real_call_and_restores(self):
        import cv2
        previous=cv2.cornerSubPix
        with B.numerical_meter()as counts:
            with self.assertRaises(cv2.error):cv2.cornerSubPix(np.zeros((2,2),np.uint8),None,(1,1),(-1,-1),(1,1,.01))
            self.assertEqual(counts['cornerSubPix'],1)
        self.assertIs(cv2.cornerSubPix,previous)

    def test_hierarchical_e_keeps_g10_family_after_g13_reselection(self):
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as LC
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_boundary_u_solver_v4_6 as PS
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_two_wd_family_v4_9 as W
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_global_outline_solver_v4_10 as G
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import face_aware_partial_bi_observations_v4_13d as F
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluate_visibility_pose_v4_8 as S
        import cv2
        xyz=LC.cuboid(1.,.1,2.).tolist();R=np.eye(3).tolist();Q=R;t=[0.,0.,4.]
        def family(name,z):
            solution=dict(accepted=True,R=R,t=[0.,0.,z],R_physical=R,feasible_pose_samples=[])
            return dict(name=name,native_xyz=xyz,cf_extents=[1.,.1,2.],physical_registration_Q=Q,
                coarse_pose=dict(R=R,t=t,physical_registration_Q=Q),initializations=None,solution=solution)
        fams=[family('ORIGINAL_WD',4.1),family('SWAPPED_WD',4.2)]
        wd=dict(families=deepcopy(fams),selected_family='ORIGINAL_WD')
        g10=dict(families=deepcopy(fams),selected_family='ORIGINAL_WD')
        g13=dict(families=[family('ORIGINAL_WD',4.3),family('SWAPPED_WD',4.4)],selected_family='SWAPPED_WD')
        spec=dict(xyz=xyz,K=R,states=['B']*8,visible=[],domains=[],image_hw=[3,3],
            coarse_pose=dict(R=R,t=t,physical_registration_Q=Q),initializations=None,
            config=dict(prior_weight=1.),boundary_observations=[],u_regions=[])
        row=dict(runtime=dict(initial_geometry={}),states=['B']*8,frozen_BI_fit=[],correction=dict(raw_points=[]),
            partial_boundary_observation=dict(unchanged_nonself_prediction_scale_px=10.))
        frame=dict(id='toy',partition='CAL',image='unused',image_sha256='hash',raw_hw=[3,3],
            initial_pose=dict(available=True,R_physical=R,centroid=t))
        with patch.object(S,'_spec',return_value=deepcopy(spec)),patch.object(PS,'solve_boundary_u_region_joint_ot_pnp',return_value=deepcopy(fams[0]['solution'])),\
             patch.object(W,'solve_two_wd_families',side_effect=[wd,g10,g13]),patch.object(G,'observe_global_outline',return_value={}),\
             patch.object(F,'observe_face_aware_partial_bi',return_value=dict(boundary_observations=[dict(native_corner=0)])),\
             patch.object(cv2,'imread',return_value=np.zeros((3,3,3),np.uint8)):
            result=P.solve_e(row,frame,np.ones((3,3)),dict(prior_weight=1.),{}, {})
        self.assertEqual(result['selected_family'],'ORIGINAL_WD')
        self.assertEqual(result['e0_regenerated']['centroid'],[0.,0.,4.3])
        self.assertTrue(result['proof']['e0_independently_executed'])
        self.assertNotEqual(result['e0_regenerated']['centroid'],frame['initial_pose']['centroid'])
        self.assertFalse(result['proof']['history_only_union'])


if __name__=='__main__':unittest.main()
