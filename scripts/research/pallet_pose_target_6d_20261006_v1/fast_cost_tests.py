"""Small independent tests for operational fast-cost guards and durable WAL.

Only temporary toy caches are written. No actual F/PnP, network, optimizer,
scientific protocol or frozen reporting/contract-test code is called or changed.
"""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
from . import cost_cache as C
from . import fast_cost_cache as F

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_pose_target_6d_20261006_v1'
MOCK_COST_CALLS = 0


def forbidden_actual_cost(*args, **kwargs):
    global MOCK_COST_CALLS
    MOCK_COST_CALLS += 1
    raise AssertionError('Unexpected cost call in a recovery/reuse fixture')


def toy_result(index):
    cost = [.3, .1, .1][index]
    return dict(available=True, ADDsym_m=cost, translation_cm=index + .25,
                rotation_deg=index + .5, WD=2, final_hypothesis='long-face-front')


class FastIOContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name)
        self.binding = 'toy-bound-native-bank'
        self.writer = hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()
        self.arrays = C.open_arrays(self.cache, n=1, mode='w+')
        for name, (_, _, initial) in C.SPECS.items():
            self.arrays[name][:] = initial
        self.arrays['action_counts'][0] = 3
        self.arrays['row_state'][0] = 1
        self.arrays['usable_train'][0] = True
        (self.cache / 'journals').mkdir()
        (self.cache / 'receipts').mkdir()
        self.path = self.cache / 'journals/chunk_0000.bin'
        self.guard_path = self.cache / 'fast_guards/chunk_0000.json'
        self.pnp = {k: index + 1 for index, k in enumerate(C.PNP)}

    def tearDown(self):
        self.temp.cleanup()

    def reserve(self, indices=(0, 1, 2), prefix_bytes=0):
        F.reserve_row(self.guard_path, self.binding, self.writer, 0, list(indices), prefix_bytes)
        return json.loads(self.guard_path.read_text())

    def complete_wal(self, indices=(0, 1, 2)):
        C.write(self.path.with_suffix('.binding.json'), dict(binding=self.binding, chunk=0))
        journal = F.BufferedJournal(self.path)
        try:
            for index in indices:
                journal.attempt(0, index)
                journal.complete(0, index, toy_result(index), self.pnp)
            journal.sync()
        finally:
            journal.close()
        return C.replay_journal(self.path, self.arrays, [0])

    def worker(self, cost_function=None, expected_calls=0):
        source, bank_cache, cache = 'toy-source', 'toy-bank', str(self.cache)
        points = np.arange(3 * 9 * 2, dtype=np.float64).reshape(3, 9, 2)
        points[1:, 8] = points[0, 8]
        data = SimpleNamespace(indices=[0], source_frame=lambda row: dict(id='toy-frame', q=points[0]))
        bank = SimpleNamespace(get=lambda row, frame: dict(points=points))
        key = (source, bank_cache, cache, self.binding, self.writer)
        def counted_cost(*args, **kwargs):
            global MOCK_COST_CALLS
            MOCK_COST_CALLS += 1
            return cost_function(*args, **kwargs)
        with patch.dict(F.WORKER, dict(key=key, data=data, bank=bank, arrays=self.arrays), clear=True):
            with patch.object(C, 'candidate_cost', side_effect=counted_cost if cost_function else forbidden_actual_cost) as cost:
                receipt = F._worker((source, bank_cache, cache, 0, [0], self.binding, self.writer, 'toy-io-binding'))
            self.assertEqual(cost.call_count, expected_calls)
        return receipt

    def test_reservation_is_durable_and_keeps_native_indices(self):
        guard = self.reserve(indices=(0, 2), prefix_bytes=37)
        self.assertEqual(guard['row'], 0)
        self.assertEqual(guard['binding'], self.binding)
        self.assertEqual(guard['writer_sha256'], self.writer)
        self.assertEqual(guard['remaining_indices'], [0, 2])
        self.assertEqual(guard['prefix_bytes'], 37)

    def test_interrupted_before_first_completion_blocks_without_retry(self):
        guard = self.reserve()
        with patch.object(C, 'candidate_cost', side_effect=forbidden_actual_cost) as cost:
            recovery = F.guard_recovery(guard, set())
        self.assertEqual(recovery['status'], 'BLOCKED')
        self.assertEqual(recovery['reserved_count'], 3)
        self.assertEqual(recovery['durable_complete'], 0)
        self.assertEqual(recovery['ambiguous_remaining'], [0, 1, 2])
        self.assertIsNone(recovery['actual_F_started_exact'])
        self.assertEqual(recovery['actual_F_started_bounds'], [0, 3])
        receipt = self.worker()
        self.assertEqual(receipt['status'], 'BLOCKED_INTERRUPTED_RESERVATION')
        self.assertEqual(receipt['new_F_calls'], 0)
        cost.assert_not_called()

    def test_interrupted_partial_row_blocks_even_with_one_completion(self):
        guard = self.reserve()
        replay = self.complete_wal(indices=(0,))
        with patch.object(C, 'candidate_cost', side_effect=forbidden_actual_cost) as cost:
            recovery = F.guard_recovery(guard, replay['completed'])
        self.assertEqual(recovery['status'], 'BLOCKED')
        self.assertEqual(recovery['durable_complete'], 1)
        self.assertEqual(recovery['ambiguous_remaining'], [1, 2])
        self.assertFalse(self.arrays['done'][0])
        receipt = self.worker()
        self.assertEqual(receipt['status'], 'BLOCKED_INTERRUPTED_RESERVATION')
        self.assertEqual(receipt['durable_completed_F'], 1)
        self.assertEqual(receipt['fast_invocation_new_F_calls'], 0)
        cost.assert_not_called()

    def test_complete_WAL_pending_guard_recovers_exact_cache_zero_F(self):
        guard = self.reserve()
        replay = self.complete_wal()
        with patch.object(C, 'candidate_cost', side_effect=forbidden_actual_cost) as cost:
            recovery = F.guard_recovery(guard, replay['completed'])
        self.assertEqual(recovery['status'], 'COMPLETE_PROVEN')
        self.assertEqual(recovery['durable_complete'], 3)
        self.assertEqual(recovery['ambiguous_remaining'], [])
        self.assertEqual(replay['attempted'], {(0, 0), (0, 1), (0, 2)})
        self.assertEqual(replay['completed'], replay['attempted'])
        self.assertEqual(replay['PnP_counts'], {k: 3 * value for k, value in self.pnp.items()})
        self.assertEqual(C.hard_target(self.arrays['cost_ADDsym_m'][0, :3], self.arrays['F_available'][0, :3]), 1)
        self.assertTrue(np.array_equal(self.arrays['candidate_state'][0, :3], [2, 2, 2]))
        self.assertTrue(np.array_equal(self.arrays['cost_ADDsym_m'][0, :3], [.3, .1, .1]))
        receipt = self.worker()
        self.assertEqual(receipt['status'], 'PASS')
        self.assertTrue(self.arrays['done'][0])
        self.assertEqual(self.arrays['row_state'][0], 2)
        self.assertEqual(self.arrays['oracle_index'][0], 1)
        self.assertEqual(receipt['fast_invocation_new_F_calls'], 0)
        self.assertEqual(receipt['actual_F_started_this_invocation'], 0)
        self.assertEqual(receipt['actual_F_completed_this_invocation'], 0)
        self.assertEqual(receipt['actual_F_completed_cumulative'], 3)
        promoted = json.loads(self.guard_path.read_text())
        self.assertEqual(promoted['status'], 'ROW_COMPLETE')
        self.assertTrue(promoted['recovered_from_full_WAL'])
        self.assertEqual(promoted['completed_F'], 3)
        cost.assert_not_called()

    def test_torn_tail_blocks_even_if_all_reservations_completed(self):
        guard = self.reserve()
        replay = self.complete_wal()
        with self.path.open('ab') as stream:
            stream.write(b'\x01\x02')
        replay = C.replay_journal(self.path, self.arrays, [0])
        self.assertEqual(replay['tail_bytes'], 2)
        recovery = F.guard_recovery(guard, replay['completed'], replay['tail_bytes'])
        self.assertEqual(recovery['status'], 'BLOCKED')
        receipt = self.worker()
        self.assertEqual(receipt['status'], 'BLOCKED_INTERRUPTED_RESERVATION')
        self.assertEqual(receipt['guard']['tail_bytes'], 2)
        self.assertEqual(receipt['new_F_calls'], 0)

    def test_wrong_row_completions_do_not_clear_reservation(self):
        guard = self.reserve()
        recovery = F.guard_recovery(guard, {(1, 0), (1, 1), (1, 2)})
        self.assertEqual(recovery['status'], 'BLOCKED')
        self.assertEqual(recovery['durable_complete'], 0)
        self.assertEqual(recovery['ambiguous_remaining'], [0, 1, 2])

    def test_reservation_count_is_not_actual_completion_count(self):
        guard = self.reserve()
        before = F.guard_recovery(guard, set())
        replay = self.complete_wal(indices=(1,))
        after = F.guard_recovery(guard, replay['completed'])
        self.assertEqual(before['reserved_count'], after['reserved_count'])
        self.assertEqual(before['durable_complete'], 0)
        self.assertEqual(after['durable_complete'], 1)
        self.assertEqual(after['reserved_count'], after['durable_complete'] + len(after['ambiguous_remaining']))
        self.assertIsNone(after['actual_F_started_exact'])
        self.assertEqual(after['actual_F_started_bounds'], [1, 3])

    def test_legacy_PASS_reuse_is_readonly_zero_mock_F(self):
        replay = self.complete_wal()
        self.arrays['done'][0] = True
        self.arrays['row_state'][0] = 2
        self.arrays['oracle_index'][0] = 1
        for value in self.arrays.values():
            value.flush()
        path = self.cache / 'receipts/chunk_0000.json'
        receipt = dict(status='PASS', chunk=0, rows=[0], binding=self.binding,
            attempted=3, completed=3, incomplete_rows=0, new_F_calls=2,
            journal_bytes=self.path.stat().st_size, journal_sha256=C.sha(self.path),
            PnP_counts=replay['PnP_counts'])
        C.write(path, receipt)
        before = {p.relative_to(self.cache): p.read_bytes() for p in self.cache.rglob('*') if p.is_file()}
        reused = self.worker()
        after = {p.relative_to(self.cache): p.read_bytes() for p in self.cache.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(reused['readonly_reuse'])
        self.assertEqual(reused['new_F_calls'], 0)
        self.assertEqual(reused['fast_invocation_new_F_calls'], 0)
        self.assertEqual(reused['original_receipt_new_F_calls'], 2)

    def test_unreserved_legacy_attempt_is_blocked_without_new_F(self):
        C.write(self.path.with_suffix('.binding.json'), dict(binding=self.binding, chunk=0))
        journal = F.BufferedJournal(self.path)
        try:
            journal.attempt(0, 1)
            journal.sync()
        finally:
            journal.close()
        before = self.path.read_bytes()
        receipt = self.worker()
        self.assertEqual(receipt['status'], 'BLOCKED_OLD_INCOMPLETE_ATTEMPT')
        self.assertEqual(receipt['WAL_attempt_records'], 1)
        self.assertEqual(receipt['durable_completed_F'], 0)
        self.assertEqual(receipt['new_F_calls'], 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_duplicate_reservation_native_index_rejected_before_file(self):
        with self.assertRaises(AssertionError):
            self.reserve(indices=(0, 0))
        self.assertFalse(self.guard_path.exists())

    def test_fresh_row_writes_three_mock_costs_then_reuses_zero(self):
        observed = []
        def cost(frame, points):
            index = int(points[0, 0] / 18)
            self.assertEqual(frame['id'], 'toy-frame')
            self.assertTrue(np.array_equal(points[8], frame['q'][8]))
            observed.append(index)
            return toy_result(index)
        receipt = self.worker(cost_function=cost, expected_calls=3)
        self.assertEqual(observed, [0, 1, 2])
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['actual_F_started_this_invocation'], 3)
        self.assertEqual(receipt['actual_F_completed_this_invocation'], 3)
        self.assertEqual(receipt['reserved_candidates_this_invocation'], 3)
        self.assertEqual(receipt['journal_bytes'], 111)
        raw = self.path.read_bytes()
        order = []
        for offset in range(0, len(raw), 37):
            row, index, tag = C.ATTEMPT.unpack_from(raw, offset)
            completed_row, completed_index, completed_tag, *_ = C.COMPLETE.unpack_from(raw, offset + 7)
            order.append((row, index, tag, completed_row, completed_index, completed_tag))
        self.assertEqual(order, [(0, index, 1, 0, index, 2) for index in range(3)])
        self.assertEqual(self.arrays['oracle_index'][0], 1)
        guard = json.loads(self.guard_path.read_text())
        self.assertEqual(guard['status'], 'ROW_COMPLETE')
        self.assertEqual(guard['actual_F_started'], 3)
        self.assertEqual(guard['completed_F'], 3)
        self.assertTrue(self.arrays['done'][0])
        before = {p.relative_to(self.cache): p.read_bytes() for p in self.cache.rglob('*') if p.is_file()}
        reused = self.worker()
        self.assertTrue(reused['readonly_reuse'])
        self.assertEqual(reused['fast_invocation_new_F_calls'], 0)
        self.assertEqual(before, {p.relative_to(self.cache): p.read_bytes() for p in self.cache.rglob('*') if p.is_file()})

    def test_cost_exception_is_durable_then_rerun_blocks_zero_mock_F(self):
        observed = []
        def cost(frame, points):
            index = int(points[0, 0] / 18)
            observed.append(index)
            if index == 1:
                raise RuntimeError('Injected toy cost failure')
            return toy_result(index)
        receipt = self.worker(cost_function=cost, expected_calls=2)
        self.assertEqual(observed, [0, 1])
        self.assertEqual(receipt['status'], 'BLOCKED_RESERVED_ROW_EXCEPTION')
        self.assertEqual(receipt['new_F_calls'], 2)
        self.assertEqual(receipt['actual_F_completed_this_invocation'], 1)
        self.assertEqual(self.path.stat().st_size, 44)
        replay = C.replay_journal(self.path, self.arrays, [0])
        self.assertEqual(replay['attempted'], {(0, 0), (0, 1)})
        self.assertEqual(replay['completed'], {(0, 0)})
        guard = json.loads(self.guard_path.read_text())
        self.assertEqual(guard['status'], 'FAILED_RESERVED_ROW')
        self.assertEqual(guard['actual_F_started_this_invocation'], 2)
        self.assertEqual(guard['actual_F_completed_this_invocation'], 1)
        before = self.path.read_bytes()
        blocked = self.worker()
        self.assertEqual(blocked['status'], 'BLOCKED_INTERRUPTED_RESERVATION')
        self.assertEqual(blocked['new_F_calls'], 0)
        self.assertEqual(blocked['durable_completed_F'], 1)
        self.assertEqual(blocked['guard']['ambiguous_remaining'], [1, 2])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.arrays['done'][0])


def main(argv=None):
    if argv:
        raise ValueError('No real-cache or experiment options are supported')
    began = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FastIOContracts))
    def binding(path):
        return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    receipt = dict(schema='pose_target_fast_IO_test_results_v1', status='PASS' if result.wasSuccessful() else 'FAIL',
        command=' '.join([sys.executable, '-B', '-m', 'scripts.research.pallet_pose_target_6d_20261006_v1.fast_cost_tests']),
        tests_run=result.testsRun, passed=result.testsRun-len(result.failures)-len(result.errors),
        failed=len(result.failures)+len(result.errors), failures=[str(test) for test, _ in result.failures + result.errors],
        seconds_wall=time.monotonic()-began, source_bindings=[binding(Path(__file__)), binding(Path(F.__file__)), binding(Path(C.__file__))],
        toy_temp_caches_only=True, mock_F_calls=MOCK_COST_CALLS, actual_F_calls=0, actual_PnP_calls=0,
        mocked_cost_calls_only=True,
        new_network_forwards=0, new_optimizer_updates=0, protocol_changes=0,
        frozen_reporting_or_contract_test_changes=0, manuscript_writes=0)
    C.write(DOC / 'FAST_IO_TEST_RESULTS.json', receipt)
    if not result.wasSuccessful():
        raise SystemExit(1)
    return receipt


if __name__ == '__main__':
    main(sys.argv[1:])
