"""Preservation tests against frozen observations; no pose fitting or GT."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

import numpy as np

from . import reobs
from .common import canonical, digest

ROOT = Path(os.environ.get('PALLET_GUARDED_DIAG_FIXTURES',
                          Path.home()/'Documents/pallet-research/pallet-guarded-bi-final-20261016-v1'))


class ReobservationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ROOT.exists():
            raise unittest.SkipTest('Frozen local evidence is required')
        cls.frame = json.loads((ROOT/'e_inputs/INPUTS.json').read_text())['frames'][0]
        obs = json.loads((ROOT/'e_run/observations/000_E_FROZEN.json').read_text())['result']['observation']
        cls.p = reobs.packet(obs, cls.frame['arms']['E_FROZEN']['coarse_pose'],
                            cls.frame['K'], cls.frame['raw_hw'])

    def test_serialization_active_equivalence(self):
        p2 = json.loads(canonical(self.p))
        self.assertEqual(reobs.active_values(self.p), reobs.active_values(p2))
        a, b = reobs._problem(self.p), reobs._problem(p2)
        for offset in (np.zeros(6), np.full(6, .0001)):
            v = a['initial_vector'] + offset
            np.testing.assert_array_equal(a['objective_pair'](v)[1], b['objective_pair'](v)[1])
            np.testing.assert_array_equal(a['constraint_pair'](v)[1], b['constraint_pair'](v)[1])

    def test_unused_display_does_not_enter_likelihood(self):
        changed = deepcopy(self.p)
        changed['observation']['unused_display'] = {'S_readout': [[1e12, -1e12]]*8,
                                                    'internal_audit_ID': 'POISON'}
        self.assertEqual(reobs.active_values(self.p), reobs.active_values(changed))

    def test_seed_reference_inactive_in_actual_configuration(self):
        p = reobs._problem(self.p)
        self.assertFalse(p['cfg']['protect_seed_projection'])
        v = p['initial_vector']
        args = (v, p['xyz'], p['K'], p['measurements'], [], ())
        after = (p['image_hw'], p['scale'], p['cfg'])
        np.testing.assert_array_equal(
            reobs.A.constraint_values_and_jacobian(*args, p['reference'], *after)[0],
            reobs.A.constraint_values_and_jacobian(*args, p['reference']+100000, *after)[0])

    def test_valid_channel_permutation_and_semantic_change_differ(self):
        p = reobs._problem(self.p)
        permutation = np.array([4, 5, 6, 7, 0, 1, 2, 3])
        inverse = np.argsort(permutation)
        moved = deepcopy(p['measurements'])
        for m in moved:
            m['native_corner'] = int(inverse[m['native_corner']])
        v = p['initial_vector']
        before = reobs.A.visible_residual_and_jacobian(v, p['xyz'], p['K'],
                                                        p['measurements'], p['scale'])
        after = reobs.A.visible_residual_and_jacobian(v, p['xyz'][permutation], p['K'],
                                                       moved, p['scale'])
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        q1 = reobs.J.project_native(p['xyz'], *reobs.J._pose(v, p['scale']), p['K'])[0]
        q2 = reobs.J.project_native(p['xyz'][permutation], *reobs.J._pose(v, p['scale']), p['K'])[0]
        np.testing.assert_array_equal(q1, q2[inverse])
        bad = reobs.A.visible_residual_and_jacobian(v, p['xyz'], p['K'], moved, p['scale'])[0]
        self.assertFalse(np.allclose(before[0], bad))

    def test_packet_not_mutated_by_active_evaluation(self):
        frozen = digest(self.p)
        reobs.active_values(self.p)
        self.assertEqual(digest(self.p), frozen)

    def test_registered_basis_exchange_preserves_physical_pose_and_projection(self):
        old=self.p['coarse_pose'];extents=old['cf_extents']
        P=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        xyz=reobs.RT.C.cuboid(*extents)
        new_extents=[extents[2],extents[1],extents[0]]
        new_xyz=reobs.RT.C.cuboid(*new_extents)
        perm=np.argmin(np.linalg.norm((new_xyz@P.T)[:,None,:]-xyz[None,:,:],axis=2),axis=1)
        np.testing.assert_array_equal(new_xyz@P.T,xyz[perm])
        new=deepcopy(old)
        new['cf_extents']=new_extents
        new['R_cf']=(np.asarray(old['R_cf'])@P).tolist()
        Q=np.asarray(old['R_cf']).T@np.asarray(old['R_physical'])
        new['physical_registration_Q']=(P.T@Q).tolist()
        a=reobs.RT.registered_geometry(self.p['image_hw'],self.p['K'],xyz,old)
        b=reobs.RT.registered_geometry(self.p['image_hw'],self.p['K'],new_xyz,new)
        np.testing.assert_allclose(a['R_physical'],b['R_physical'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(a['initial_projection_xy'][perm],b['initial_projection_xy'],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(a['initial_self_hidden'][perm],b['initial_self_hidden'])
        self.assertEqual([a['initial_roles'][i] for i in perm],b['initial_roles'])


if __name__ == '__main__':
    unittest.main()
