"""Temporary synthetic fixtures only: never creates production review records."""
import copy
import unittest

import numpy as np

from scripts.research.pallet_static_registry_review_20261003_v1.geometry_visibility_proposals import proposal_for_object


def object_fixture():
    pose = np.eye(4)
    pose[:3, 3] = [.6, -.4, 4.]
    dimensions = dict(width=1., height=.2, depth=1.2)
    return dict(keypoint_frame='camera_dynamic_0123_v4',
        physical_dimensions_m=dict(x=1., y=.2, z=1.2),
        camera_facing_pnp=dict(dimensions_m=dimensions, pose_transform=pose.tolist(),
                               axis_assignment_confirmed=False),
        canonical_pose_candidates=[dict(pose_transform=pose.tolist(), axis_assignment='YAW_0',
            canonical_to_camera_facing_keypoint_permutation=list(range(9)))],
        keypoint_annotations=[dict(xy=[200.+i, 100.], source='pnp_projected',
                                  visibility=1, reason='unknown', in_frame=True) for i in range(8)])


class GeometryProposalTests(unittest.TestCase):
    def test_bounds_are_actual_image_and_do_not_claim_human_review(self):
        obj = object_fixture()
        obj['keypoint_annotations'][0]['xy'] = [-.01, 10.]
        obj['keypoint_annotations'][1]['xy'] = [640., 10.]
        obj['keypoint_annotations'][2]['xy'] = [639.999, 479.999]
        obj['keypoint_annotations'][3]['xy'] = [float('nan'), 10.]
        result = proposal_for_object(obj, dict(width=320, height=240), (640, 480))
        for i in (0, 1):
            self.assertEqual(result[i]['status'], 'OUT_OF_FRAME')
            self.assertTrue(result[i]['human_confirmation_required'])
            self.assertTrue(result[i]['machine_proposal_only'])
            self.assertNotIn('human_reviewed', result[i])
        self.assertNotEqual(result.get(2, {}).get('status'), 'OUT_OF_FRAME')
        self.assertNotIn(3, result)

    def test_signed_axis_mapping_uses_physical_dimensions_and_agreement(self):
        obj = object_fixture()
        expected = proposal_for_object(obj, dict(width=640, height=480))
        self.assertEqual([i for i, v in expected.items() if v['status'] == 'SELF_OCCLUDED'], [5])
        candidate = copy.deepcopy(obj['canonical_pose_candidates'][0])
        rotation = np.diag([-1., 1., -1.])
        pose = np.asarray(candidate['pose_transform'])
        pose[:3, :3] = pose[:3, :3] @ rotation
        candidate['pose_transform'] = pose.tolist()
        candidate['axis_assignment'] = 'YAW_180'
        candidate['canonical_to_camera_facing_keypoint_permutation'] = [5,4,7,6,1,0,3,2,8]
        obj['canonical_pose_candidates'].append(candidate)
        result = proposal_for_object(obj, dict(width=640, height=480))
        self.assertEqual(list(result), [5])
        self.assertEqual(result[5]['evidence']['hypotheses_compared'], 3)
        self.assertNotIn('DIRECT_VISIBLE', [v['status'] for v in result.values()])
        obj['canonical_pose_candidates'][-1]['canonical_to_camera_facing_keypoint_permutation'] = list(range(9))
        self.assertNotIn(5, proposal_for_object(obj, dict(width=640, height=480)))
        # A rectangular quarter turn must use canonical physical W/D rather
        # than applying the swapped camera-facing dimensions a second time.
        quarter = object_fixture()
        pose = np.eye(4)
        pose[:3, 3] = [.55, -.4, 4.]
        quarter['camera_facing_pnp']['pose_transform'] = pose.tolist()
        quarter['camera_facing_pnp']['dimensions_m'] = dict(width=1.2, height=.2, depth=1.)
        pose[:3, :3] = [[0.,0.,1.], [0.,1.,0.], [-1.,0.,0.]]
        quarter['canonical_pose_candidates'] = [dict(pose_transform=pose.tolist(), axis_assignment='YAW_90',
            canonical_to_camera_facing_keypoint_permutation=[1,5,6,2,0,4,7,3,8])]
        self.assertEqual(list(proposal_for_object(quarter, dict(width=640, height=480))), [4,5])

    def test_invalid_pose_missing_axis_and_prior_visible_do_not_override(self):
        obj = object_fixture()
        original = copy.deepcopy(obj)
        obj['keypoint_annotations'][5]['visibility'] = 2
        obj['keypoint_annotations'][5]['reason'] = 'visible'
        self.assertEqual(proposal_for_object(obj, dict(width=640, height=480)), {})
        obj = copy.deepcopy(original)
        obj['canonical_pose_candidates'] = []
        self.assertEqual(proposal_for_object(obj, dict(width=640, height=480)), {})
        obj = copy.deepcopy(original)
        obj['camera_facing_pnp']['pose_transform'][0][0] = 2.
        self.assertEqual(proposal_for_object(obj, dict(width=640, height=480)), {})
        self.assertEqual(original, object_fixture())


if __name__ == '__main__':
    unittest.main()
