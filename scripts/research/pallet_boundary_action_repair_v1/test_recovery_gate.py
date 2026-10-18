"""Small fake receipts test provenance failures, never real data or solvers."""
from __future__ import annotations

from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import recovery_gate as G


TOY = dict(source_count=3, reused=dict(real=2, synthetic=1), final=dict(real=2, synthetic=3),
           pending=dict(real=0, synthetic=2), interrupted_journal_rows=2, unfinished_attempts=1, workers=4)


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


class RecoveryGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root, self.repo = base / 'private', base / 'repo'
        self.root.mkdir(); self.repo.mkdir()
        self.runtime = self.root / 'recovery'
        self.runtime.mkdir()
        self.contract = patch.object(G, 'CONTRACT', deepcopy(TOY))
        self.contract.start()
        self.addCleanup(self.contract.stop)
        self.addCleanup(self.tmp.cleanup)
        self.make_fixture()

    def path(self, name):
        return self.root / name

    def change(self, name, operation):
        path = self.path(name)
        value = G._read(path)
        operation(value)
        put(path, value)

    def rebind_interruption(self):
        self.change('D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json', lambda x: x.update(
            interrupted_execution_receipt_sha256=G._sha(self.path('D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json'))))
        digest = G._sha(self.path('D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json'))
        for name in ('D1_RECOVERY_LAUNCH_LOCK.json', 'D1_RECOVERY_FINALIZATION_RECEIPT.json'):
            self.change(name, lambda x: x.update(recovery_technical_amendment_lock_sha256=digest))

    def make_fixture(self):
        sources = {}
        for i in range(TOY['source_count']):
            relative = f'scripts/fake_{i}.py'
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'# immutable fake source {i}\n')
            sources[relative] = G._sha(path)
        launch_sources = {}
        for name in G.LAUNCH_SOURCES:
            path = self.repo / G.SCIENCE_DIR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable fake launcher, never executed\n')
            launch_sources[name] = G._sha(path)
        verifier = self.repo / G.SCIENCE_DIR / 'd1_parity_audit.py'
        verifier.write_text('# immutable fake independent verifier, never executed\n')
        verifier_sha = G._sha(verifier)
        reused = {}
        inputs = {}
        for domain, n in TOY['final'].items():
            folder = self.root / f'd1_{domain}_candidates_v1'
            put(folder / 'BATCH_LOCK.json', dict(fake_domain=domain))
            inputs[domain] = dict(inputs_sha256='a'*64, observation_seal_sha256='b'*64,
                preserved_batch_lock_sha256=G._sha(folder / 'BATCH_LOCK.json'))
            reused[domain] = {}
            for i in range(n):
                index = f'{i:04d}'
                path = folder / 'frames' / index / 'FRAME_SEAL.json'
                put(path, dict(schema='FAKE_FRAME_SEAL', domain=domain, index=i))
                if i < TOY['reused'][domain]:
                    reused[domain][index] = G._sha(path)
        for name in G.NAMESPACE_LINKS:
            target = self.root / name
            if not target.exists():
                if name.endswith('.json'):
                    put(target, dict(fake_namespace_link=name))
                else:
                    target.mkdir()
            (self.runtime / name).symlink_to(target, target_is_directory=target.is_dir())
        old_pending = [dict(domain='real', index=1)] + [dict(domain='synthetic', index=i) for i in range(3)]
        new_pending = [dict(domain='synthetic', index=i) for i in (1, 2)]
        old_lock = dict(schema='D1_SHARED_FIXED_E_RUNTIME_LOCK_V1', execution_batch_id='ORIGINAL', source_sha256=sources, workers=4,
                        ordered_pending=old_pending, domain_inputs=inputs)
        new_inputs = {d: dict(v, preserved_completed_frames=TOY['reused'][d]) for d, v in inputs.items()}
        new_lock = dict(schema='D1_SHARED_FIXED_E_RUNTIME_LOCK_V1', execution_batch_id='RECOVERY', source_sha256=sources, workers=4,
                        ordered_pending=new_pending, domain_inputs=new_inputs)
        put(self.path('D1_SHARED_EXECUTION_BATCH_LOCK.json'), old_lock)
        put(self.runtime / 'D1_SHARED_EXECUTION_BATCH_LOCK.json', new_lock)
        self.path('D1_SHARED_RUNTIME_JOURNAL.jsonl').write_text('{}\n{}\n')
        (self.runtime / 'D1_SHARED_RUNTIME_JOURNAL.jsonl').write_text('{}\n{}\n')
        self.path('D1_SHARED_FIXED_E_LIVE.log').write_text('preserved original log\n')
        archive = self.root / 'interrupted_archive' / 'synthetic' / '0001'
        archive.mkdir(parents=True)
        attempt = dict(domain='synthetic', index=1, archived_path=str(archive),
            FRAME_SEAL_present=False, file_sha256={}, wall_seconds=None, CPU_seconds=None,
            optimizer_calls=None, PnP_calls=None, NN_calls=0)
        interrupted = dict(schema='D1_INTERRUPTED_SHARED_EXECUTION_V1', status='INTERRUPTED_EXTERNAL_CAUSE_UNKNOWN',
            execution_batch_id='ORIGINAL', original_execution_lock_sha256=G._sha(self.path('D1_SHARED_EXECUTION_BATCH_LOCK.json')),
            original_journal_sha256=G._sha(self.path('D1_SHARED_RUNTIME_JOURNAL.jsonl')),
            original_log_sha256=G._sha(self.path('D1_SHARED_FIXED_E_LIVE.log')),
            worker_exit_codes=None, termination_signal_sender='UNKNOWN', concrete_termination_cause='UNKNOWN',
            agent_issued_termination=False, observed_tool_session_exit_code=143, parent_wall_seconds=None,
            interrupted_partial_operations=None, interrupted_partial_CPU_seconds=None,
            parent_process_lifetime_upper_bound_seconds=10.5, workers=4, journal_rows=2,
            completed_frame_seal_sha256=reused, domain_completed_frames=TOY['reused'],
            unsealed_attempts=[attempt], reference_or_accuracy_reads=0)
        put(self.path('D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json'), interrupted)
        amendment = dict(schema='D1_INTERRUPTION_RECOVERY_TECHNICAL_LOCK_V1', status='LOCKED',
            original_private_root=str(self.root), recovery_private_root=str(self.runtime),
            original_execution_batch_id='ORIGINAL', original_lock_sha256=interrupted['original_execution_lock_sha256'],
            original_journal_sha256=interrupted['original_journal_sha256'],
            interrupted_execution_receipt_sha256=G._sha(self.path('D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json')),
            exact_same_input_observation_solver_feature_procedure=True, model_or_scientific_setting_changes=False,
            new_observation_or_NN_forwards=0, new_PRIMARY_reference_reads=0,
            original_paths_preserved=['D1_SHARED_EXECUTION_BATCH_LOCK.json', 'D1_SHARED_RUNTIME_JOURNAL.jsonl', 'D1_SHARED_FIXED_E_LIVE.log'],
            new_runtime_namespace_links=sorted(G.NAMESPACE_LINKS), frozen_original_source_sha256=sources,
            recovery_launcher_source_sha256=launch_sources['d1_recovery_batch.py'],
            completed_frames_reused=reused, completed_frame_counts=TOY['reused'],
            pending_expected_rows=2, pending_expected_domain_rows=TOY['pending'], unfinished_attempts_preserved=[attempt])
        put(self.path('D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json'), amendment)
        argv = ['/fake/python', '-B', '-m', 'scripts.research.pallet_boundary_action_repair_v1.d1_detached_supervisor',
            '--private', str(self.root), '--old-run', '/fake/old', '--original-run', '/fake/original',
            '--typed-run', '/fake/typed', '--workers', '4']
        launch = dict(schema='D1_RECOVERY_DETACHED_LAUNCH_LOCK_V1', argv=argv, cwd=str(self.repo),
            new_session=True, thread_limits={k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            source_sha256=launch_sources, supervisor_pid=314, reference_or_accuracy_reads=0,
            recovery_technical_amendment_lock_sha256=G._sha(self.path('D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json')))
        put(self.path('D1_RECOVERY_LAUNCH_LOCK.json'), launch)
        child_argv = list(argv)
        child_argv[3] = 'scripts.research.pallet_boundary_action_repair_v1.d1_recovery_batch'
        child_argv.insert(4, 'run')
        child_argv.append('--root-GO')
        put(self.path('D1_RECOVERY_SUPERVISOR_START.json'), dict(schema='D1_DETACHED_SUPERVISOR_START_V1',
            child_argv=child_argv, supervisor_pid=314, child_pid=315,
            supervisor_source_sha256=launch_sources['d1_detached_supervisor.py'], reference_or_accuracy_reads=0))
        put(self.path('D1_RECOVERY_SUPERVISOR_RECEIPT.json'), dict(schema='D1_DETACHED_SUPERVISOR_EXIT_V1',
            status='COMPLETE', supervisor_pid=314, child_pid=315, child_actual_exit_code=0,
            source_sha256=launch_sources['d1_detached_supervisor.py'], reference_or_accuracy_reads=0))
        receipt = dict(schema='D1_SHARED_FIXED_E_RUNTIME_RECEIPT_V1', status='COMPLETE', failures=[],
            execution_batch_id='RECOVERY', execution_lock_sha256=G._sha(self.runtime / 'D1_SHARED_EXECUTION_BATCH_LOCK.json'),
            source_sha256=sources, workers=4, rows=2, domain_frame_counts=TOY['final'],
            same_shared_parentwall_count_once=True, reference_or_accuracy_reads=0)
        put(self.runtime / 'D1_SHARED_EXECUTION_BATCH_RECEIPT.json', receipt)
        put(self.path('D1_SHARED_EXECUTION_BATCH_RECEIPT.json'), receipt)
        runtime_sha = G._sha(self.runtime / 'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')
        checks = {name: True for name in G.REQUIRED_PARITY_CHECKS}
        feature_hashes, domains = {}, {}
        for domain, count in TOY['final'].items():
            folder = self.root / f'd1_{domain}_candidates_v1'
            seal = dict(schema='D1_FEATURE_SEAL_V1', complete=True, rows=count,
                ordered_ids=[f'{domain}/{i}' for i in range(count)], source_sha256=sources,
                execution_batch_id='RECOVERY', shared_execution_receipt_sha256=runtime_sha,
                execution_proof=dict(shared_runtime_lock_sha256=receipt['execution_lock_sha256']),
                input_sha256=inputs[domain]['inputs_sha256'], observation_seal_sha256=inputs[domain]['observation_seal_sha256'],
                e0_independently_executed=True, same_automatic_procedure_both_domains=True,
                feature_meaning_unchanged=True, feature_scores_use_original_unrepaired_A_B_energies=True,
                history_only_union=False, new_B_or_A_candidates=False)
            put(folder / 'FEATURE_SEAL.json', seal)
            feature_hashes[domain] = G._sha(folder / 'FEATURE_SEAL.json')
            execute = dict(schema='D1_ACTUAL_E_EXECUTION_RECEIPT_V1', rows=count,
                independent_execution_checks_passed=True, domain_rows_across_batches_same_frozen_source_inputs_observations=True,
                technical_interruption_recovery_checks=checks, feature_seal_sha256=feature_hashes[domain],
                shared_execution_batch_lock_sha256=receipt['execution_lock_sha256'],
                shared_execution_batch_receipt_sha256=runtime_sha, verifier_source_sha256=verifier_sha,
                reference_or_accuracy_reads=0)
            put(folder / 'D1_EXECUTION_RECEIPT.json', execute)
            domains[domain] = dict(D1_EXECUTION_RECEIPT_sha256=G._sha(folder / 'D1_EXECUTION_RECEIPT.json'))
        final = dict(schema='D1_INTERRUPTION_RECOVERY_FINALIZATION_V1', status='COMPLETE',
            recovery_technical_amendment_lock_sha256=launch['recovery_technical_amendment_lock_sha256'],
            all_original_and_restarted_rows_same_frozen_scientific_source=True, no_new_observation_or_model_forward=True,
            interrupted_lock_and_journal_unchanged=True, source_sha256=launch_sources['d1_recovery_batch.py'],
            canonical_receipt_exact_copy=True, recovery_runtime_receipt_sha256=runtime_sha,
            recovery_execution_batch_id='RECOVERY', recovery_runtime_lock_sha256=receipt['execution_lock_sha256'],
            recovery_runtime_journal_sha256=G._sha(self.runtime / 'D1_SHARED_RUNTIME_JOURNAL.jsonl'),
            final_feature_seals=feature_hashes, reference_or_accuracy_reads=0)
        put(self.path('D1_RECOVERY_FINALIZATION_RECEIPT.json'), final)
        parity = dict(schema='D1_REFERENCE_FREE_PARITY_AUDIT_V1', status='PASS', failures=[],
            source_sha256=verifier_sha,
            all238_and1003_across_batches_same_frozen_source_inputs_observations=True,
            technical_interruption_recovery_checks=checks, actual_producer_source_files_SHA_verified=True,
            actual_producer_source_file_count=3, domains=domains,
            execution_journal_coverage=dict(attempt_journal_count=2, journal_rows=4,
                distinct_domain_index_rows=4, original_planned_pending_rows=4, duplicate_rows=0))
        put(self.path('DATA_PIPELINE_PARITY.json'), parity)

    def verify(self):
        return G.verify_recovery_for_scoring(self.root, self.repo)

    def test_valid_chain_and_compact_output(self):
        result = self.verify()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['source_file_count'], 3)
        self.assertNotIn('frozen_original_source_sha256', result)
        self.assertNotIn('completed_frames_reused', result)

    def test_child_prefix_matches_frozen_supervisor_source(self):
        # Independent of both the helper and fixture builders: read the actual
        # immutable producer argv expression without importing or executing it.
        source = Path(G.__file__).with_name('d1_detached_supervisor.py').read_text()
        tree = ast.parse(source)
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'argv' for t in n.targets))
        prefix = [ast.literal_eval(n) for n in assignment.value.elts[1:5]]
        actual_fixture = G._read(self.path('D1_RECOVERY_SUPERVISOR_START.json'))['child_argv']
        self.assertEqual(prefix, ['-B', '-m',
            'scripts.research.pallet_boundary_action_repair_v1.d1_recovery_batch', 'run'])
        self.assertEqual(actual_fixture[1:5], prefix)
        self.assertEqual(self.verify()['status'], 'PASS')

    def test_missing_run_positional_rejected(self):
        self.change('D1_RECOVERY_SUPERVISOR_START.json', lambda x: x['child_argv'].pop(4))
        with self.assertRaisesRegex(ValueError, 'actual child argv'):
            self.verify()

    def test_wrong_recovery_lock_schema_rejected(self):
        path = self.runtime / 'D1_SHARED_EXECUTION_BATCH_LOCK.json'
        value = G._read(path); value['schema'] = 'UNRELATED_LOCK'
        put(path, value)
        with self.assertRaisesRegex(ValueError, 'runtime lock schemas'):
            self.verify()

    def test_changed_independent_verifier_rejected(self):
        (self.repo / G.SCIENCE_DIR / 'd1_parity_audit.py').write_text('# replaced verifier\n')
        with self.assertRaisesRegex(ValueError, 'parity verifier source bytes'):
            self.verify()

    def test_changed_scientific_source_rejected(self):
        (self.repo / 'scripts/fake_1.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'source bytes'):
            self.verify()

    def test_changed_reused_frame_rejected(self):
        put(self.path('d1_real_candidates_v1/frames/0000/FRAME_SEAL.json'), dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'reused frame bytes'):
            self.verify()

    def test_changed_original_journal_rejected(self):
        self.path('D1_SHARED_RUNTIME_JOURNAL.jsonl').write_text('replaced\n')
        with self.assertRaisesRegex(ValueError, 'original journal bytes'):
            self.verify()

    def test_nonidentical_canonical_copy_rejected(self):
        self.change('D1_SHARED_EXECUTION_BATCH_RECEIPT.json', lambda x: x.update(rows=1))
        with self.assertRaisesRegex(ValueError, 'canonical receipt exact'):
            self.verify()

    def test_missing_independent_parity_rejected(self):
        self.path('DATA_PIPELINE_PARITY.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_vacuous_or_failed_parity_rejected(self):
        self.change('DATA_PIPELINE_PARITY.json', lambda x: x.update(technical_interruption_recovery_checks={}))
        with self.assertRaisesRegex(ValueError, 'substantive parity'):
            self.verify()

    def test_bad_union_coverage_rejected(self):
        self.change('DATA_PIPELINE_PARITY.json', lambda x: x['execution_journal_coverage'].update(duplicate_rows=1))
        with self.assertRaisesRegex(ValueError, 'journal union/disjoint'):
            self.verify()

    def test_nonzero_supervised_exit_rejected(self):
        self.change('D1_RECOVERY_SUPERVISOR_RECEIPT.json', lambda x: x.update(child_actual_exit_code=143))
        with self.assertRaisesRegex(ValueError, 'actual waited child exit'):
            self.verify()

    def test_launcher_source_tamper_rejected(self):
        (self.repo / G.SCIENCE_DIR / 'd1_detached_supervisor.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'launcher bytes'):
            self.verify()

    def test_unknown_interrupted_cost_must_not_be_zero(self):
        self.change('D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json', lambda x: x.update(interrupted_partial_operations=0))
        self.rebind_interruption()
        with self.assertRaisesRegex(ValueError, 'remains null'):
            self.verify()

    def test_archived_attempt_not_discarded(self):
        self.path('interrupted_archive/synthetic/0001/extra.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'archive bytes'):
            self.verify()

    def test_boolean_false_is_not_a_numeric_zero_receipt(self):
        self.change('D1_RECOVERY_SUPERVISOR_RECEIPT.json', lambda x: x.update(reference_or_accuracy_reads=False))
        with self.assertRaisesRegex(ValueError, 'reference/accuracy assertion'):
            self.verify()

    def test_fifth_worker_rejected(self):
        self.change('D1_RECOVERY_LAUNCH_LOCK.json', lambda x: x['argv'].__setitem__(-1, '5'))
        with self.assertRaisesRegex(ValueError, 'launch path/session/workers'):
            self.verify()

    def test_runtime_symlink_redirection_rejected(self):
        link = self.runtime / 'd1_upstream_v1'
        link.unlink()
        wrong = self.root / 'wrong_upstream'; wrong.mkdir()
        link.symlink_to(wrong, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'namespace link target'):
            self.verify()

    def test_production_contract_requires_703_sources(self):
        with patch.object(G, 'CONTRACT', dict(TOY, source_count=703)):
            with self.assertRaisesRegex(ValueError, 'closure/cardinality'):
                self.verify()


if __name__ == '__main__':
    unittest.main()
