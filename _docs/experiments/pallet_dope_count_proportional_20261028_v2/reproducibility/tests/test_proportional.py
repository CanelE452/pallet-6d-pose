"""Pure schedule checks and tiny CPU scalar-gradient checks; no data/model I/O."""
from __future__ import annotations

import importlib.util
import ast
from contextlib import nullcontext
import hashlib
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

TRAINER = Path(__file__).resolve().parents[1]/'code/train_proportional.py'
spec = importlib.util.spec_from_file_location('proportional_trainer_tests', TRAINER)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


class ScheduleTests(unittest.TestCase):
    def test_module_has_no_neural_import(self):
        self.assertNotIn('torch', T.__dict__)
        self.assertNotIn('numpy', T.__dict__)

    def test_exact_uniform_rng_contract(self):
        batches, metadata = T.joint_schedule(110, 550, 16, 828, 2000042)
        rng = random.Random(2000042)
        flat = []
        while len(flat) < 13248:
            order = list(range(660))
            rng.shuffle(order)
            flat.extend(order)
        expected = [flat[i:i + 16] for i in range(0, 13248, 16)]
        self.assertEqual(batches, expected)
        self.assertEqual(metadata['total_draws'], 13248)
        self.assertEqual(metadata['synthetic_draws'] + metadata['real_draws'], 13248)
        self.assertEqual(metadata['per_image_probability_at_cycle_start'], 1/660)
        self.assertEqual(metadata['synthetic_probability'], 1/6)
        self.assertEqual(metadata['real_probability'], 5/6)

    def test_private_rng_and_repeatability(self):
        before = random.getstate()
        one = T.joint_schedule(20, 20, 16, 828, 42)
        self.assertEqual(before, random.getstate())
        random.random()
        self.assertEqual(one, T.joint_schedule(20, 20, 16, 828, 42))
        self.assertNotEqual(one, T.joint_schedule(20, 20, 16, 828, 43))

    def test_empty_branch_and_duplicate_draws_are_allowed(self):
        batches, meta = T.joint_schedule(4, 20, 16, 828, 42)
        self.assertGreater(meta['synthetic_empty_batches'], 0)
        self.assertGreater(meta['within_batch_duplicate_draws'], 0)
        self.assertTrue(any(len(set(batch)) < 16 for batch in batches))
        self.assertTrue(any(all(i >= 4 for i in batch) for batch in batches))

    def test_short_schedule_does_not_redraw_to_force_coverage(self):
        batches, meta = T.joint_schedule(2200, 550, 16, 1, 42)
        self.assertLess(meta['unique_exposed'], meta['joint_unique_pool'])
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 16)

    def test_cycles_cover_exact_pool_and_global_exposure_differs_at_most_one(self):
        for ns, nr in [(4, 20), (20, 20), (2200, 550)]:
            batches, meta = T.joint_schedule(ns, nr, 16, 828, 2000042)
            flat = [i for batch in batches for i in batch]
            n = ns + nr
            for start in range(0, len(flat) - n + 1, n):
                self.assertEqual(set(flat[start:start + n]), set(range(n)))
            self.assertEqual(meta['unique_exposed'], n)
            self.assertLessEqual(meta['maximum_per_image_exposure'] - meta['minimum_per_image_exposure'], 1)

    def test_split_preserves_joint_and_source_orders(self):
        syn = [dict(id=f's{i}') for i in range(3)]
        real = [dict(id=f'r{i}') for i in range(2)]
        si, ri, joint = T.split_batch([4, 0, 3, 0, 2], syn, real)
        self.assertEqual(si, [0, 0, 2])
        self.assertEqual(ri, [1, 0])
        self.assertEqual([d['id'] for d in joint], ['r1', 's0', 'r0', 's0', 's2'])

    def test_source_weights_match_actual_counts(self):
        self.assertEqual(T.branch_weights(8, 8), (1.0, 1.0))
        self.assertEqual(T.branch_weights(3, 13), (.375, 1.625))
        self.assertEqual(T.branch_weights(16, 0), (2.0, 0.0))
        self.assertEqual(T.branch_weights(0, 16), (0.0, 2.0))
        with self.assertRaises(RuntimeError):
            T.branch_weights(8, 7)

    def test_source_zero_skips_original_helper(self):
        calls = []
        def backward(*args, **kwargs):
            calls.append((args, kwargs))
            return 7.0 * args[6]
        H = SimpleNamespace(_backward_branch=backward)
        out = T.backward_count_weighted(H, object(), [], [], [], list(range(16)),
                                        'cpu', 8, object(), False, None)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0][0][5])
        self.assertEqual(calls[0][0][6], 2.0)
        self.assertEqual(out['loss_total'], 14.0)
        self.assertIsNone(out['loss_syn_branch_mean'])
        self.assertEqual(out['loss_real_branch_mean'], 7.0)


class ScalarGradientTests(unittest.TestCase):
    """Exercise actual source coefficients with variable trusted-channel counts."""
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.torch.set_num_threads(1)

    def helper(self, parameter, calls):
        torch = self.torch
        def backward(network, rows, indices, device, microbatch, synthetic, scale,
                     augmentation=None, amp=False, scaler=None):
            calls.append((synthetic, len(indices), scale))
            # Exact original branch denominator: all present valid channels
            # times pixels, not an equal average of per-image masked means.
            masks = torch.tensor([rows[i]['mask'] for i in indices], dtype=torch.bool)
            targets = torch.tensor([rows[i]['target'] for i in indices], dtype=torch.float64)
            predictions = parameter.expand_as(targets)
            error = torch.where(masks, predictions - targets, torch.zeros_like(targets))
            loss = error.square().sum() / masks.sum() * scale
            loss.backward()
            return float(loss.detach())
        return SimpleNamespace(_backward_branch=backward)

    def records(self, n, offset):
        return [dict(mask=[True, True, i % 3 != 0],
                     target=[offset + i/10, offset - .4, offset + .8]) for i in range(n)]

    def original_loss(self, parameter, rows):
        torch = self.torch
        mask = torch.tensor([r['mask'] for r in rows], dtype=torch.bool)
        target = torch.tensor([r['target'] for r in rows], dtype=torch.float64)
        return torch.where(mask, parameter - target, torch.zeros_like(target)).square().sum()/mask.sum()

    def test_c_eight_plus_eight_value_and_gradient_exact_with_variable_masks(self):
        torch = self.torch
        syn, real = self.records(8, 0.1), self.records(8, 1.3)
        p_old = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        old = self.original_loss(p_old, syn) + self.original_loss(p_old, real)
        old.backward()
        p_new = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        calls = []
        out = T.backward_count_weighted(self.helper(p_new, calls), None, syn, real,
            list(range(8)), list(range(8)), 'cpu', 8, None, False, None)
        self.assertEqual(out['loss_total'], float(old.detach()))
        self.assertEqual(float(p_new.grad), float(p_old.grad))
        self.assertEqual(calls, [(True, 8, 1.0), (False, 8, 1.0)])

    def test_unequal_counts_preserve_original_masks_and_use_count_weights(self):
        torch = self.torch
        syn, real = self.records(3, 0.1), self.records(13, 1.3)
        p_expected = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        expected = .375*self.original_loss(p_expected, syn) + 1.625*self.original_loss(p_expected, real)
        expected.backward()
        p_actual = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        calls = []
        out = T.backward_count_weighted(self.helper(p_actual, calls), None, syn, real,
            list(range(3)), list(range(13)), 'cpu', 8, None, False, None)
        self.assertAlmostEqual(out['loss_total'], float(expected.detach()), places=14)
        self.assertAlmostEqual(float(p_actual.grad), float(p_expected.grad), places=14)
        self.assertEqual(calls, [(True, 3, .375), (False, 13, 1.625)])

    def test_every_ratio_and_seed_has_exact_fixed_budget(self):
        for nr in (20, 550):
            for percent in (20, 50, 100, 200, 400):
                for seed in (42, 43, 44):
                    ns = nr*percent//100
                    draws = 0
                    batches, meta = T.joint_schedule(ns, nr, 16, 828, seed + 2000000)
                    for batch in batches:
                        n_syn = sum(i < ns for i in batch)
                        ws, wr = T.branch_weights(n_syn, 16 - n_syn)
                        self.assertEqual(ws + wr, 2.0)
                        draws += len(batch)
                    self.assertEqual(draws, 13248)


class OriginalHelperGradientTests(unittest.TestCase):
    """Execute unchanged AST nodes from the hash-bound original loss helper.

    Tensor inputs are manufactured here; no image, label, checkpoint or GPU is
    read. Only the three original function definitions are compiled, avoiding
    unrelated adapter/augmentation module imports during these unit checks.
    """
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        path = Path('/home/minjae/Documents/pallet-research/dope-loo-flip-per-type-20261026-v1/training.py')
        source_bytes = path.read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == '0004f7848077164cfd92d4804e86eaf136c054919fb175249ca09c64d8a155da'
        tree = ast.parse(source_bytes.decode())
        names = {'masked_mse', '_collate', '_backward_branch'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == 3
        namespace = dict(torch=torch, nullcontext=nullcontext)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        cls.H = SimpleNamespace(**{name: namespace[name] for name in names})

    def rows(self, n, real=False):
        torch = self.torch
        rows = []
        for i in range(n):
            mask = torch.ones(9, dtype=torch.bool)
            if real:
                mask[8] = False
                mask[6:8] = i % 3 != 0
            rows.append(dict(image=torch.full((1, 2, 3), .1 + i/20),
                beliefs=torch.full((9, 2, 3), .2 + i/40), belief_mask=mask,
                affinities=torch.full((16, 2, 3), -.1 + i/50),
                affinity_mask=torch.ones(16, dtype=torch.bool)))
        return rows

    def network(self, parameter):
        def forward(image):
            b, _, h, w = image.shape
            return ([parameter*image.expand(b, 9, h, w)*(i + 1)/6 for i in range(6)],
                    [parameter*image.expand(b, 16, h, w)*(i + 1)/6 for i in range(6)])
        return forward

    def compare_actual_original(self, ns, nr):
        torch, H = self.torch, self.H
        syn, real = self.rows(ns), self.rows(nr, real=True)
        ws, wr = T.branch_weights(ns, nr)
        expected_parameter = torch.tensor(.8, requires_grad=True)
        expected_network = self.network(expected_parameter)
        expected_syn = H._backward_branch(expected_network, syn, list(range(ns)), 'cpu', 8,
                                          True, ws) if ns else 0.0
        expected_real = H._backward_branch(expected_network, real, list(range(nr)), 'cpu', 8,
                                           False, wr) if nr else 0.0
        actual_parameter = torch.tensor(.8, requires_grad=True)
        actual = T.backward_count_weighted(H, self.network(actual_parameter), syn, real,
            list(range(ns)), list(range(nr)), 'cpu', 8, None, False, None)
        self.assertEqual(actual['loss_total'], expected_syn + expected_real)
        self.assertTrue(torch.equal(actual_parameter.grad, expected_parameter.grad))
        self.assertEqual(actual['synthetic_loss_coefficient'], ws)
        self.assertEqual(actual['real_loss_coefficient'], wr)

    def test_original_helper_c_exact_value_and_gradient(self):
        self.compare_actual_original(8, 8)

    def test_original_helper_unequal_source_counts_microbatch_invariance(self):
        self.compare_actual_original(3, 13)

    def test_original_helper_all_real_branch(self):
        self.compare_actual_original(0, 16)

    def test_original_helper_all_synthetic_branch(self):
        self.compare_actual_original(16, 0)


class RecoveryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_exact_completed_final_state_is_accepted(self):
        torch = self.torch
        state = {'weight': torch.tensor([1., 2.]), 'bias': torch.tensor([.3])}
        T.verify_final_recovery(torch, {k: v.clone() for k, v in state.items()}, state, 828,
                                {'weight': 1656, 'bias': 1614}, {'weight': 828, 'bias': 786})

    def test_partial_or_incorrect_adam_state_is_refused(self):
        torch = self.torch
        state = {'weight': torch.tensor([1., 2.])}
        for updates, steps in [(827, {'weight': 1655}), (828, {'weight': 1655}),
                               (828, {'other': 1656})]:
            with self.assertRaises(RuntimeError):
                T.verify_final_recovery(torch, state, state, updates, steps, {'weight': 828})

    def test_different_final_value_dtype_shape_and_keyset_are_refused(self):
        torch = self.torch
        state = {'weight': torch.tensor([1., 2.])}
        for final in [{'weight': torch.tensor([1., 2.1])},
                      {'weight': torch.tensor([1., 2.], dtype=torch.float64)},
                      {'weight': torch.tensor([[1., 2.]])},
                      {'other': torch.tensor([1., 2.])}]:
            with self.assertRaises(RuntimeError):
                T.verify_final_recovery(torch, final, state, 828, {'weight': 1656}, {'weight': 828})

    def history_fixture(self):
        syn, real = [{'id': 's0'}], [{'id': 'r0'}]
        si, ri, joint = T.split_batch([0]*3 + [1]*13, syn, real)
        row = dict(update=1, epoch=1, epoch_step=1, joint_draws=joint,
            synthetic_ids=['s0']*3, real_ids=['r0']*13, n_synthetic=3, n_real=13,
            synthetic_loss_coefficient=.375, real_loss_coefficient=1.625,
            loss_syn=.75, loss_real=4.875, loss_total=5.625,
            loss_syn_branch_mean=2., loss_real_branch_mean=3.,
            elapsed_seconds=1., scaler_scale=1024.)
        return [row], [(si, ri, joint)], syn, real

    def test_committed_history_is_bound_to_schedule_and_weights(self):
        history, decoded, syn, real = self.history_fixture()
        T.validate_history(history, decoded, syn, real)
        for key, bad in [('n_real', 12), ('epoch_step', 2), ('loss_total', 4.),
                         ('real_loss_coefficient', 1.), ('loss_syn_branch_mean', 4.),
                         ('synthetic_ids', ['s0']*2), ('loss_syn', float('nan'))]:
            invalid = copy.deepcopy(history)
            invalid[0][key] = bad
            with self.assertRaises(RuntimeError):
                T.validate_history(invalid, decoded, syn, real)

    def test_history_longer_than_schedule_is_refused(self):
        history, decoded, syn, real = self.history_fixture()
        with self.assertRaises(RuntimeError):
            T.validate_history(history*2, decoded, syn, real)


class OriginalArchitectureAdamLedgerTests(OriginalHelperGradientTests):
    """Exact frozen six-stage forward graph, tiny CPU heads, ordinary Adam."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch = cls.torch
        path = Path('/home/minjae/Documents/github/pallet-pose/Deep_Object_Pose/common/models.py')
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == 'd692ea3b0cabc2c657e0428801d236bba2a1a3564ec6d75ffb26376714a51727'
        tree = ast.parse(raw.decode())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DopeNetwork')
        forward = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        ns = {'torch': torch}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), 'exec'), ns)
        class TinyOriginalGraph(torch.nn.Module):
            forward = ns['forward']
            def __init__(self):
                super().__init__()
                self.vgg = torch.nn.Conv2d(1, 2, 1)
                self.stop_at_stage = 6
                self.numVec = self.numSeg = self.numEdgeVec = 0
                self.maskBeliefFusion = self.refinement_enabled = False
                for stage in range(1, 7):
                    setattr(self, f'm{stage}_2', torch.nn.Conv2d(2 if stage == 1 else 27, 9, 1))
                    setattr(self, f'm{stage}_1', torch.nn.Conv2d(2 if stage == 1 else 27, 16, 1))
        cls.TinyOriginalGraph = TinyOriginalGraph

    def prepared_optimizer(self):
        torch = self.torch
        network = self.TinyOriginalGraph()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-5, weight_decay=1e-4)
        self.H._backward_branch(network, self.rows(8), list(range(8)), 'cpu', 8, True, 1.)
        optimizer.step()
        # Manufacture an already initialized R1 counter without a checkpoint.
        for state in optimizer.state.values():
            state['step'].fill_(828)
        optimizer.zero_grad(set_to_none=True)
        return network, optimizer

    def test_real_only_uses_natural_adam_and_leaves_final_affinity_untouched(self):
        torch = self.torch
        network, optimizer = self.prepared_optimizer()
        named = list(network.named_parameters())
        names = [name for name, _ in named]
        ledger = {name: 0 for name in names}
        original_final = {name: parameter.detach().clone() for name, parameter in named
                          if name.startswith('m6_1.')}
        T.validate_parameter_steps(named, optimizer, ledger, 0)
        T.backward_count_weighted(self.H, network, [], self.rows(16, real=True), [],
                                  list(range(16)), 'cpu', 8, None, False, None)
        missing = T.inspect_gradient_presence(named, 0)
        self.assertEqual(missing, ['m6_1.bias', 'm6_1.weight'])
        for name, parameter in named:
            if not name.startswith('m6_1.'):
                self.assertIsNotNone(parameter.grad)
        optimizer.step()
        history = [dict(n_synthetic=0, no_gradient_parameter_names=missing)]
        ledger = T.gradient_ledger_from_history(names, history)
        steps = T.validate_parameter_steps(named, optimizer, ledger, 1)
        self.assertEqual(set(steps.values()), {828, 829})
        for name, parameter in named:
            if name.startswith('m6_1.'):
                self.assertTrue(torch.equal(parameter.detach(), original_final[name]))
                self.assertEqual(steps[name], 828)
            else:
                self.assertEqual(steps[name], 829)

    def test_real_only_then_mixed_history_recovers_exact_parameter_ledger(self):
        network, optimizer = self.prepared_optimizer()
        named = list(network.named_parameters())
        names = [name for name, _ in named]
        history = []
        for ns, nr in [(0, 16), (3, 13), (16, 0), (0, 16)]:
            optimizer.zero_grad(set_to_none=True)
            before = T.gradient_ledger_from_history(names, history)
            T.validate_parameter_steps(named, optimizer, before, len(history))
            T.backward_count_weighted(self.H, network, self.rows(ns), self.rows(nr, real=True),
                list(range(ns)), list(range(nr)), 'cpu', 8, None, False, None)
            missing = T.inspect_gradient_presence(named, ns)
            optimizer.step()
            history.append(dict(n_synthetic=ns, no_gradient_parameter_names=missing))
            ledger = T.gradient_ledger_from_history(names, history)
            T.validate_parameter_steps(named, optimizer, ledger, len(history))
        self.assertEqual({v for k, v in ledger.items() if k.startswith('m6_1.')}, {2})
        self.assertEqual({v for k, v in ledger.items() if not k.startswith('m6_1.')}, {4})
        corrupted = dict(ledger)
        corrupted['m6_1.weight'] += 1
        with self.assertRaises(RuntimeError):
            T.validate_parameter_steps(named, optimizer, corrupted, 4)

    def test_skipped_optimizer_update_is_detected(self):
        network, optimizer = self.prepared_optimizer()
        named = list(network.named_parameters())
        T.backward_count_weighted(self.H, network, self.rows(8), self.rows(8, real=True),
            list(range(8)), list(range(8)), 'cpu', 8, None, False, None)
        T.inspect_gradient_presence(named, 8)
        # Deliberately omit optimizer.step(), as an AMP skipped update would.
        with self.assertRaises(RuntimeError):
            T.validate_parameter_steps(named, optimizer, {name: 1 for name, _ in named}, 1)

    def test_unexpected_missing_gradients_and_history_are_rejected(self):
        network, optimizer = self.prepared_optimizer()
        named = list(network.named_parameters())
        with self.assertRaises(RuntimeError):
            T.inspect_gradient_presence(named, 1)
        with self.assertRaises(RuntimeError):
            T.gradient_ledger_from_history([name for name, _ in named],
                [dict(n_synthetic=0, no_gradient_parameter_names=[])])


if __name__ == '__main__':
    unittest.main(verbosity=2)
