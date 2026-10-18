"""Constructed geometry/contract tests; no reference or real result reads."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from . import feature_support as F


def pose(x=0.):
    return dict(available=True,R_physical=np.eye(3).tolist(),centroid=[x,0.,3.])


def frame():
    return dict(id='toy',split='train',object_scale_m=1.,fallback=pose(),e0_regenerated=pose(.1),
        candidates=[dict(candidate_id='raw',actual_pose=pose(.2),eligible=True,features=[0.]*62,
            source_baseline_aliases=[])])


def empty_values(): return [0.]*F.M


def extra(fr):
    aa=F.AD.actions(fr)
    return dict(id=fr['id'],action_set_sha256=F.canonical_hash([F.action_binding(a) for a in aa]),
        actions=[dict(candidate_id=a['candidate_id'],extra_values=empty_values(),extra_validity=[1.]*F.M,
            action_binding_sha256=F.canonical_hash(F.action_binding(a))) for a in aa])


class Contracts(unittest.TestCase):
    def test_schema_dimension(self):
        self.assertEqual(F.M,144);self.assertEqual(len(F.FEATURE_NAMES),len(set(F.FEATURE_NAMES)))
        self.assertEqual(F.FEATURE_SPEC['full_input_dimension'],476)
        self.assertEqual(F.FEATURE_SPEC['MLP_parameters'],34753)

    def test_both_family_physical_roundtrip(self):
        xyz=np.array([[x,y,z] for x in [-.7,.7] for y in [-.1,.1] for z in [-.4,.4]])+np.array([.12,-.13,.17])
        target=pose();target['R_physical']=Rotation.from_rotvec([.2,-.3,.1]).as_matrix().tolist()
        for angle in [0.,np.pi/2]:
            c=SimpleNamespace(Q=Rotation.from_rotvec([0.,angle,0.]).as_matrix(),xyz=xyz,scale=2.)
            p=F.OLD._native_pose(target,c);actual=F.OLD._actual_pose(p,c)
            np.testing.assert_allclose(actual['R_physical'],target['R_physical'],atol=1e-14)
            np.testing.assert_allclose(actual['centroid'],target['centroid'],atol=1e-14)

    def test_append_preserves_actions_and_pose(self):
        fr=frame();before=deepcopy(fr);er=extra(fr)
        n=F.fit_extra_normalizer([er],['toy']);x=F.normalized_extra(fr,er,n)
        self.assertEqual(x.shape,(3,288));self.assertEqual(fr,before)

    def test_alias_tamper_rejected(self):
        fr=frame();er=extra(fr);n=F.fit_extra_normalizer([er],['toy'])
        er['actions'][0]['action_binding_sha256']='bad'
        with self.assertRaises(ValueError):F.normalized_extra(fr,er,n)

    def test_order_tamper_rejected(self):
        fr=frame();er=extra(fr);n=F.fit_extra_normalizer([er],['toy'])
        er['actions'].reverse()
        with self.assertRaises(ValueError):F.normalized_extra(fr,er,n)

    def test_train_only_normalizer(self):
        a=extra(frame());b=deepcopy(a);b['id']='validation'
        a['actions'][0]['extra_values'][0]=3.;b['actions'][0]['extra_values'][0]=1e14
        n1=F.fit_extra_normalizer([a,b],['toy']);n2=F.fit_extra_normalizer([a],['toy'])
        self.assertEqual(n1,n2)

    def test_missing_and_unseen_not_false_zero_error(self):
        fr=frame();er=extra(fr)
        for a in er['actions']:a['extra_validity'][0]=0.
        n=F.fit_extra_normalizer([er],['toy']);er['actions'][0]['extra_validity'][0]=1.
        er['actions'][0]['extra_values'][0]=123.
        x=F.normalized_extra(fr,er,n)
        self.assertEqual(x[0,0],0.);self.assertEqual(x[0,F.M],1.)

    def test_no_clip_signed_log(self):
        fr=frame();er=extra(fr);n=F.fit_extra_normalizer([er],['toy'])
        er['actions'][0]['extra_values'][0]=-1e14
        x=F.normalized_extra(fr,er,n)
        self.assertLess(x[0,0],-30.)

    def test_runtime_minimize_alias_forbidden(self):
        F.preload_backends()
        with F.fixed_pose_only() as meter:
            with self.assertRaisesRegex(RuntimeError,'FIXED_POSE_ONLY'):F.OLD.minimize(lambda x:sum(x*x),[0.])
            self.assertEqual(len(meter['forbidden_invocation_attempts']),1)

    def test_runtime_pnp_forbidden(self):
        F.preload_backends();import cv2
        with F.fixed_pose_only():
            with self.assertRaisesRegex(RuntimeError,'FIXED_POSE_ONLY'):cv2.solvePnP(None,None,None,None)

    def test_zero_actions_shape(self):
        fr=dict(id='empty',candidates=[],fallback=dict(available=False))
        er=dict(id='empty',actions=[],action_set_sha256=F.canonical_hash([]))
        n=dict(mean=[0.]*F.M,scale=[1.]*F.M,valid_train_counts=[0]*F.M)
        self.assertEqual(F.normalized_extra(fr,er,n).shape,(0,288))

    def test_full_measurement_source_flags(self):
        flags=F._source_flags(dict(source='VISIBLE_BACKBONE_FALLBACK',image_measurement=True,measurement_is_pose_projection=False))
        self.assertTrue(flags['image_measurement']);self.assertTrue(flags['backbone_fallback'])
        self.assertFalse(flags['pose_projection'])

    def test_partial_missing_flags_not_inferred(self):
        flags=F._source_flags(dict(association_provenance={'original_measurement_source':'SOME_SOURCE',
            'normal_projected_display_xy':[10,10]}),True)
        self.assertIsNone(flags['image_measurement']);self.assertIsNone(flags['pose_projection'])

    def test_profile_zero_is_missing_confidence(self):
        # Test value construction against an empty original bundle, not the
        # vacuous bool returned by the original finite helper.
        c=SimpleNamespace(xyz=np.zeros((8,3)),K=np.eye(3),scale=1.,family_name=F.FAMILIES[0],
            measurements=[],boundary=[],domains=[],regions=[],use_outline=False,states=['U']*8)
        model=SimpleNamespace(ctx=c,bundle=dict(groups=[],consumed_native_ids=[],image_hw=[100,100],pixel_scale=1.))
        s=dict(scores={m:1. for m in F.MODES},audits={'fixed':{'parts':{'full':0.,'partial':0.,'outline':0.}}},
            h_min_slack={m:[1.]*(4 if m in ('A','AB') else 1) for m in F.MODES},feasible={m:True for m in F.MODES})
        with patch.object(F.OLD,'_native_pose',return_value=np.zeros(6)),patch.object(F.OLD.J,'projection_and_jacobian',return_value=(np.zeros((8,2)),np.ones(8),None,None)):
            v,d=F.family_values(model,pose(),s,F.FAMILIES[0])
        byname=dict(zip([x['name'] for x in F.PER_FAMILY_SPEC],v))
        self.assertEqual(byname['B_no_fixed_observation'],1.)
        self.assertIsNone(byname['B_supported_profile_fraction'])
        self.assertIsNone(byname['B_all_profiles_supported'])
        self.assertIsNone(byname['full_residual_mean'])


if __name__=='__main__':unittest.main()
