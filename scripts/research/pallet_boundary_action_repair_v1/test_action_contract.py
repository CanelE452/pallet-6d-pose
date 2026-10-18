"""Expected action/learning behavior is constructed before real scoring."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from torch import nn
from .action_data import actions, fit_normalizer, inputs, label_frame, pose_error
from .action_models import RoleAffine, collate, loss_value, choose, make_model, parameter_count


def pose(tx=0, angle=0):
    a = np.deg2rad(angle)
    return dict(available=True, R_physical=[[float(np.cos(a)), 0., float(np.sin(a))], [0., 1., 0.], [-float(np.sin(a)), 0., float(np.cos(a))]], centroid=[tx, 0., 2.])


def candidate(name, p, value=0., eligible=True, aliases=()):
    return dict(candidate_id=name, actual_pose=p, features=[value] * 31 + [1.] * 31, eligible=eligible, source_baseline_aliases=list(aliases))


def frame(n0, others=()):
    cs = ([candidate('N0', n0, aliases=['N0'])] if n0.get('available') else []) + list(others)
    return dict(id='toy', split='train', fallback=n0, candidates=cs, object_scale_m=1.)


def target(): return dict(R_fixed=np.eye(3).tolist(), t=[0., 0., 2.], symmetry_order=1, split='train')


class ActionContract(unittest.TestCase):
    def test_n0_good_other_bad(self):
        f = frame(pose(), [candidate('bad', pose(.2))]); lab = label_frame(f, target())
        self.assertEqual([m['success_5cm5deg'] for m in lab['action_metrics']], [True, False])

    def test_n0_bad_other_good(self):
        f = frame(pose(.2), [candidate('good', pose())]); lab = label_frame(f, target())
        self.assertEqual([m['success_5cm5deg'] for m in lab['action_metrics']], [False, True])

    def test_both_good_multilabel(self):
        lab = label_frame(frame(pose(), [candidate('also', pose(.01))]), target())
        self.assertEqual([m['success_5cm5deg'] for m in lab['action_metrics']], [True, True]); self.assertFalse(lab['all_zero_H1'])

    def test_all_failure_kept(self):
        lab = label_frame(frame(pose(.1), [candidate('also', pose(.2))]), target())
        self.assertTrue(lab['all_zero_H1']); self.assertFalse(lab['zero_actions_H1']); self.assertEqual(len(lab['action_metrics']), 2)

    def test_cross_candidate_error_not_combined(self):
        lab = label_frame(frame(pose(0, 20), [candidate('goodrotation', pose(.2))]), target())
        self.assertTrue(lab['all_zero_H1'])

    def test_duplicate_roles_unique_no_none(self):
        f = frame(pose(), [candidate('none-alias', pose(), aliases=['E0'])]); aa = actions(f)
        self.assertEqual(len(aa), 1); self.assertEqual(set(aa[0]['source_baseline_aliases']), {'N0', 'E0'})
        norm = fit_normalizer([f]); x, entries = inputs(f, norm, 'H1'); self.assertEqual(len(entries), 1); self.assertEqual(x[0, -1], 1.)

    def test_n0_missing_and_zero_actions(self):
        f = frame(dict(available=False), [candidate('only', pose())]); self.assertEqual(len(actions(f)), 1)
        f = frame(dict(available=False)); lab = label_frame(f, target()); self.assertTrue(lab['zero_actions_H1']); self.assertIsNone(choose([], [], 'H1'))

    def test_order_and_name_invariant(self):
        f = frame(pose(.1), [candidate('z', pose(.2), 2), candidate('a', pose(.3), 3)])
        g = deepcopy(f); g['candidates'] = list(reversed(g['candidates']))
        for i, c in enumerate(g['candidates']): c['candidate_id'] = 'renamed' + str(i)
        norm = fit_normalizer([f]); a, aa = inputs(f, norm, 'H1'); b, bb = inputs(g, norm, 'H1'); np.testing.assert_array_equal(a, b)
        self.assertEqual(choose([1., 1., 1.], aa, 'H1'), 0); self.assertTrue(aa[0]['is_N0']); self.assertTrue(bb[0]['is_N0'])

    def test_ineligible_padding_and_missing_validity(self):
        f = frame(pose(), [candidate('good', pose(.2), 2)]); norm = fit_normalizer([f])
        g = deepcopy(f); g['candidates'].append(candidate('ineligible', pose(.3), 1e6, False))
        x, _ = inputs(f, norm, 'H1'); xx, _ = inputs(g, norm, 'H1'); np.testing.assert_array_equal(x, xx)
        h = deepcopy(f); h['candidates'][1]['features'][31] = 0.
        hx, _ = inputs(h, norm, 'H1'); self.assertEqual(hx[1, 0], 0.); self.assertEqual(hx[1, 31], 0.); self.assertNotEqual(x[1, 31], hx[1, 31])
        model = nn.Linear(188, 1); batch = (torch.zeros(1, 2, 188), torch.tensor([[True, False]]), torch.tensor([[1., 0.]]))
        baseline = loss_value(model, batch, 'H1'); batch[0][0, 1] = 1e6
        torch.testing.assert_close(baseline, loss_value(model, batch, 'H1'))

    def test_common_affine_cancels_role_affine_can_use_context(self):
        # Same own-feature inputs and common g at coordinate62. The old affine
        # score difference stays0 as g changes. Role affine N0 coefficient2
        # and OTHER coefficient-1 makes its role difference change by3*g.
        old = nn.Linear(188, 1); role = RoleAffine()
        with torch.no_grad():
            old.weight.zero_(); old.bias.zero_(); old.weight[0, 62] = 4.
            role.n0.weight.zero_(); role.n0.bias.zero_(); role.other.weight.zero_(); role.other.bias.zero_()
            role.n0.weight[0, 62] = 2.; role.other.weight[0, 62] = -1.
        x = torch.zeros(2, 188); x[0, -1] = 1.
        d0 = (old(x)[0] - old(x)[1]).item(); r0 = (role(x)[0] - role(x)[1]).item(); x[:, 62] = 3.
        self.assertEqual(d0, (old(x)[0] - old(x)[1]).item()); self.assertEqual((role(x)[0] - role(x)[1]).item() - r0, 9.)

    def test_frame_balance_multiple_positive_allzero_and_empty(self):
        class IdentityModel(nn.Module):
            def forward(self, x): return x[..., :1]
        xx = torch.zeros((3, 3, 188)); mask = torch.tensor([[True, False, False], [True, True, True], [False, False, False]])
        yy = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        self.assertAlmostEqual(float(loss_value(IdentityModel(), (xx, mask, yy), 'H1')), float(np.log(2)), places=6)

    def test_h0_legacy_inputs_and_parameters_exact(self):
        from ..pallet_statewise_end_to_end_recovery_v1.learned_head import model_inputs
        f = frame(pose(), [candidate('other', pose(.2), 2)]); norm = fit_normalizer([f])
        x, _ = inputs(f, norm, 'H0'); old, _ = model_inputs(f, norm); np.testing.assert_array_equal(x, old.numpy())
        self.assertEqual(parameter_count('MLP'), 16321); self.assertEqual(parameter_count('ROLE_AFFINE'), 376)

    def test_pair_has_only_actual_roles_and_dedup(self):
        f = frame(pose(), [candidate('e', pose(.2), aliases=['E0']), candidate('other', pose(.3))]); pair = actions(f, True)
        self.assertEqual(len(pair), 2); self.assertEqual({x['action_role'] for x in pair}, {'N0', 'E0'})
        f['candidates'][1]['actual_pose'] = pose(); pair = actions(f, True); self.assertEqual(len(pair), 1)


if __name__ == '__main__': unittest.main()
