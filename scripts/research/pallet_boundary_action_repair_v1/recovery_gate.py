"""Reference-free byte/provenance gate for the one recorded D1 interruption.

This does not import a producer, decode pose/features/labels, or claim accuracy.
The independent D1 audit owns substantive journal-union and row-content checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


CONTRACT = dict(source_count=703, reused=dict(real=238, synthetic=454),
                final=dict(real=238, synthetic=1003), pending=dict(real=0, synthetic=549),
                interrupted_journal_rows=652, unfinished_attempts=4, workers=4)
NAMESPACE_LINKS = frozenset(('d1_upstream_v1', 'd1_real_observations_v1',
    'd1_synthetic_observations_v1', 'd1_real_candidates_v1', 'd1_synthetic_candidates_v1',
    'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json', 'D1_REAL_CONTROLLED_QUIESCE.json',
    'D1_CACHE_PARITY_FIXED_SELECTION.json', 'd1_cache_parity_actual'))
LAUNCH_SOURCES = frozenset(('d1_recovery_batch.py', 'd1_detached_supervisor.py'))
SCIENCE_DIR = Path('scripts/research/pallet_boundary_action_repair_v1')
REQUIRED_PARITY_CHECKS = frozenset(('original_lock_unchanged', 'original_journal_unchanged',
    'original_log_unchanged', 'original_interruption_receipt_bound',
    'both_attempt_locks_same_frozen_source', 'restart_unique_batch_id',
    'preserved_692_exact_cardinality', 'actual_recovery_exit_zero',
    'canonical_runtime_receipt_exact_copy', 'partial_interrupted_numeric_counts_not_fabricated',
    'journal_union_disjoint', 'journal_union_exact_original_pending'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _require(condition, label):
    if not condition:
        raise ValueError('D1 recovery gate: ' + label)


def _inside(path, base):
    resolved = Path(path).resolve()
    _require(resolved != base and resolved.is_relative_to(base), 'path outside expected root')
    return resolved


def _zero(value, label):
    _require(type(value) is int and value == 0, label)


def _options(argv, module):
    _require(isinstance(argv, list) and argv[1:4] == ['-B', '-m', module], 'launch module')
    rest = argv[4:]
    _require(len(rest) == 10 and len(set(rest[::2])) == 5, 'launch argument shape')
    values = dict(zip(rest[::2], rest[1::2]))
    _require(set(values) == {'--private', '--old-run', '--original-run', '--typed-run', '--workers'},
             'launch argument keys')
    return values


def verify_recovery_for_scoring(root, repo):
    """Return compact verified receipt bindings, or raise on absent/changed proof.

    Run only after D1 completion and its reference-free parity audit. This reads
    technical JSON and hashes source/frame-seal bytes; no pose bank is decoded.
    """
    root, repo = Path(root).resolve(), Path(repo).resolve()
    c = CONTRACT
    names = ('D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json',
        'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json', 'D1_RECOVERY_LAUNCH_LOCK.json',
        'D1_RECOVERY_FINALIZATION_RECEIPT.json', 'D1_RECOVERY_SUPERVISOR_START.json',
        'D1_RECOVERY_SUPERVISOR_RECEIPT.json', 'DATA_PIPELINE_PARITY.json')
    amendment, interrupted, launch, final, start, supervisor, parity = [_read(root / n) for n in names]
    bindings = {n: _sha(root / n) for n in names}
    _require(amendment['schema'] == 'D1_INTERRUPTION_RECOVERY_TECHNICAL_LOCK_V1'
             and amendment['status'] == 'LOCKED', 'amendment schema/status')
    _require(interrupted['schema'] == 'D1_INTERRUPTED_SHARED_EXECUTION_V1'
             and interrupted['status'] == 'INTERRUPTED_EXTERNAL_CAUSE_UNKNOWN', 'interruption schema/status')
    _require(launch['schema'] == 'D1_RECOVERY_DETACHED_LAUNCH_LOCK_V1', 'launch schema')
    _require(final['schema'] == 'D1_INTERRUPTION_RECOVERY_FINALIZATION_V1'
             and final['status'] == 'COMPLETE', 'finalization schema/status')
    _require(Path(amendment['original_private_root']).resolve() == root, 'original root binding')
    runtime = _inside(amendment['recovery_private_root'], root)
    old_lock_path, old_journal = root / 'D1_SHARED_EXECUTION_BATCH_LOCK.json', root / 'D1_SHARED_RUNTIME_JOURNAL.jsonl'
    new_lock_path, new_journal = runtime / old_lock_path.name, runtime / old_journal.name
    old_lock, new_lock = _read(old_lock_path), _read(new_lock_path)
    _require(all(doc['schema'] == 'D1_SHARED_FIXED_E_RUNTIME_LOCK_V1' for doc in (old_lock, new_lock)),
             'original/recovery runtime lock schemas')
    original_id, recovery_id = old_lock['execution_batch_id'], new_lock['execution_batch_id']
    _require(original_id != recovery_id and original_id == amendment['original_execution_batch_id']
             == interrupted['execution_batch_id'], 'distinct original/recovery batch IDs')
    _require(_sha(old_lock_path) == amendment['original_lock_sha256']
             == interrupted['original_execution_lock_sha256'], 'original lock bytes')
    _require(_sha(old_journal) == amendment['original_journal_sha256']
             == interrupted['original_journal_sha256'], 'original journal bytes')
    _require(_sha(root / 'D1_SHARED_FIXED_E_LIVE.log') == interrupted['original_log_sha256'], 'original log bytes')
    _require(amendment['interrupted_execution_receipt_sha256'] == bindings[names[1]], 'interruption receipt binding')
    _require(launch['recovery_technical_amendment_lock_sha256'] == bindings[names[0]]
             == final['recovery_technical_amendment_lock_sha256'], 'amendment launch/final binding')
    for doc in (amendment,):
        _require(doc['exact_same_input_observation_solver_feature_procedure'] is True
                 and doc['model_or_scientific_setting_changes'] is False, 'unchanged procedure assertions')
        _zero(doc['new_observation_or_NN_forwards'], 'new NN/observation assertion')
        _zero(doc['new_PRIMARY_reference_reads'], 'new PRIMARY reference assertion')
    for doc in (interrupted, launch, final, start, supervisor):
        _zero(doc['reference_or_accuracy_reads'], 'reference/accuracy assertion')
    _require(final['all_original_and_restarted_rows_same_frozen_scientific_source'] is True
             and final['no_new_observation_or_model_forward'] is True
             and final['interrupted_lock_and_journal_unchanged'] is True, 'final unchanged assertions')
    _require(interrupted['worker_exit_codes'] is None and interrupted['termination_signal_sender'] == 'UNKNOWN'
             and interrupted['concrete_termination_cause'] == 'UNKNOWN'
             and interrupted['agent_issued_termination'] is False, 'unknown termination preserved')
    _require(interrupted['observed_tool_session_exit_code'] == 143, 'observed interrupted exit code')
    _require(interrupted['parent_wall_seconds'] is None
             and interrupted['interrupted_partial_operations'] is None
             and interrupted['interrupted_partial_CPU_seconds'] is None, 'unmeasured interruption cost remains null')
    upper = interrupted['parent_process_lifetime_upper_bound_seconds']
    _require(type(upper) in (int, float) and math.isfinite(upper) and upper > 0, 'interrupted wall upper bound')
    _require(set(amendment['original_paths_preserved']) == {old_lock_path.name, old_journal.name,
             'D1_SHARED_FIXED_E_LIVE.log'}, 'preserved original paths')
    _require(set(amendment['new_runtime_namespace_links']) == NAMESPACE_LINKS
             and len(amendment['new_runtime_namespace_links']) == len(NAMESPACE_LINKS), 'namespace link inventory')
    for name in NAMESPACE_LINKS:
        _require((runtime / name).is_symlink() and (runtime / name).resolve() == (root / name).resolve()
                 and (root / name).exists(), 'namespace link target: ' + name)

    sources = amendment['frozen_original_source_sha256']
    _require(len(sources) == c['source_count'] and sources == old_lock['source_sha256']
             == new_lock['source_sha256'], 'exact frozen source closure/cardinality')
    for relative, digest in sources.items():
        _require(not Path(relative).is_absolute(), 'absolute source path')
        _require(_sha(_inside(repo / relative, repo)) == digest, 'source bytes: ' + relative)
    _require(set(launch['source_sha256']) == LAUNCH_SOURCES, 'two launcher source inventory')
    for name, digest in launch['source_sha256'].items():
        _require(_sha(repo / SCIENCE_DIR / name) == digest, 'launcher bytes: ' + name)
    _require(amendment['recovery_launcher_source_sha256'] == launch['source_sha256']['d1_recovery_batch.py']
             == final['source_sha256'], 'adapter source chain')
    supervisor_sha = launch['source_sha256']['d1_detached_supervisor.py']
    _require(start['supervisor_source_sha256'] == supervisor_sha == supervisor['source_sha256'], 'supervisor source chain')
    opts = _options(launch['argv'], 'scripts.research.pallet_boundary_action_repair_v1.d1_detached_supervisor')
    _require(Path(opts['--private']).resolve() == root and opts['--workers'] == str(c['workers'])
             and Path(launch['cwd']).resolve() == repo and launch['new_session'] is True, 'launch path/session/workers')
    _require(launch['thread_limits'] == {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
             'single-thread library budgets')
    expected_child = [launch['argv'][0], '-B', '-m',
        'scripts.research.pallet_boundary_action_repair_v1.d1_recovery_batch', 'run']
    for key in ('--private', '--old-run', '--original-run', '--typed-run', '--workers'):
        expected_child += [key, opts[key]]
    expected_child += ['--root-GO']
    _require(start['schema'] == 'D1_DETACHED_SUPERVISOR_START_V1' and start['child_argv'] == expected_child,
             'actual child argv')
    _require(start['supervisor_pid'] == launch['supervisor_pid'] == supervisor['supervisor_pid']
             and start['child_pid'] == supervisor['child_pid'], 'supervisor/child identity')
    _require(supervisor['schema'] == 'D1_DETACHED_SUPERVISOR_EXIT_V1'
             and supervisor['status'] == 'COMPLETE' and type(supervisor['child_actual_exit_code']) is int
             and supervisor['child_actual_exit_code'] == 0, 'actual waited child exit')
    _require(all(type(doc['workers']) is int and doc['workers'] == c['workers']
                 for doc in (old_lock, new_lock, interrupted)), 'four-worker batch budgets')

    reused = amendment['completed_frames_reused']
    _require(reused == interrupted['completed_frame_seal_sha256'] and set(reused) == set(c['reused'])
             and amendment['completed_frame_counts'] == interrupted['domain_completed_frames'] == c['reused'],
             'reused frame map/counts')
    for domain, count in c['reused'].items():
        _require(len(reused[domain]) == count, 'reused frame cardinality: ' + domain)
        for index, digest in reused[domain].items():
            _require(index.isdigit() and len(index) == 4, 'reused frame index')
            _require(_sha(root / ('d1_' + domain + '_candidates_v1') / 'frames' / index / 'FRAME_SEAL.json') == digest,
                     'reused frame bytes: ' + domain + '/' + index)
        old_inputs, new_inputs = old_lock['domain_inputs'][domain], new_lock['domain_inputs'][domain]
        _require(new_inputs['preserved_completed_frames'] == count and all(
            new_inputs[k] == old_inputs[k] for k in ('inputs_sha256', 'observation_seal_sha256', 'preserved_batch_lock_sha256')),
            'unchanged domain input binding: ' + domain)
        _require(_sha(root / ('d1_' + domain + '_candidates_v1') / 'BATCH_LOCK.json')
                 == new_inputs['preserved_batch_lock_sha256'], 'preserved domain batch lock bytes: ' + domain)
    _require(amendment['pending_expected_domain_rows'] == c['pending']
             and amendment['pending_expected_rows'] == sum(c['pending'].values())
             and len(new_lock['ordered_pending']) == sum(c['pending'].values()), 'pending row counts')
    _require({d: sum(r['domain'] == d for r in new_lock['ordered_pending']) for d in c['pending']} == c['pending'],
             'pending domain counts')
    _require(interrupted['journal_rows'] == c['interrupted_journal_rows'], 'original journal row count assertion')
    _require(len(old_lock['ordered_pending']) == c['interrupted_journal_rows'] + sum(c['pending'].values()),
             'original pending cardinality')
    attempts = amendment['unfinished_attempts_preserved']
    _require(attempts == interrupted['unsealed_attempts'] and len(attempts) == c['unfinished_attempts'], 'unfinished attempt inventory')
    for item in attempts:
        _require(item['FRAME_SEAL_present'] is False and all(item[k] is None for k in
                 ('wall_seconds', 'CPU_seconds', 'optimizer_calls', 'PnP_calls')), 'partial attempt counts are unknown')
        _zero(item['NN_calls'], 'CPU attempt no NN assertion')
        directory = _inside(item['archived_path'], root)
        _require(directory.is_dir() and {str(p.relative_to(directory)): _sha(p)
            for p in directory.rglob('*') if p.is_file()} == item['file_sha256'], 'unfinished attempt archive bytes')

    canonical_path = root / 'D1_SHARED_EXECUTION_BATCH_RECEIPT.json'
    runtime_path = runtime / canonical_path.name
    runtime_sha = _sha(runtime_path)
    _require(runtime_sha == _sha(canonical_path) == final['recovery_runtime_receipt_sha256']
             and final['canonical_receipt_exact_copy'] is True, 'canonical receipt exact runtime copy')
    receipt = _read(runtime_path)
    _require(receipt['schema'] == 'D1_SHARED_FIXED_E_RUNTIME_RECEIPT_V1' and receipt['status'] == 'COMPLETE'
             and receipt['failures'] == [], 'new runtime completion/failures')
    _require(receipt['execution_batch_id'] == recovery_id == final['recovery_execution_batch_id']
             and receipt['execution_lock_sha256'] == _sha(new_lock_path) == final['recovery_runtime_lock_sha256']
             and final['recovery_runtime_journal_sha256'] == _sha(new_journal), 'recovery final runtime chain')
    _require(receipt['source_sha256'] == sources and receipt['workers'] == c['workers']
             and receipt['rows'] == sum(c['pending'].values()) and receipt['domain_frame_counts'] == c['final']
             and receipt['same_shared_parentwall_count_once'] is True, 'new runtime source/count/budget')
    _zero(receipt['reference_or_accuracy_reads'], 'runtime reference assertion')
    _require(set(final['final_feature_seals']) == set(c['final']), 'final feature seal inventory')
    feature_bindings = {}
    for domain, count in c['final'].items():
        path = root / ('d1_' + domain + '_candidates_v1') / 'FEATURE_SEAL.json'
        digest, seal = _sha(path), _read(path)
        feature_bindings[domain] = digest
        _require(digest == final['final_feature_seals'][domain], 'final feature seal bytes: ' + domain)
        _require(seal['schema'] == 'D1_FEATURE_SEAL_V1' and seal['complete'] is True
                 and seal['rows'] == count and len(seal['ordered_ids']) == count
                 and len(set(seal['ordered_ids'])) == count, 'final feature row coverage: ' + domain)
        _require(seal['source_sha256'] == sources and seal['execution_batch_id'] == recovery_id
                 and seal['shared_execution_receipt_sha256'] == runtime_sha
                 and seal['execution_proof']['shared_runtime_lock_sha256'] == _sha(new_lock_path)
                 and seal['input_sha256'] == new_lock['domain_inputs'][domain]['inputs_sha256']
                 and seal['observation_seal_sha256'] == new_lock['domain_inputs'][domain]['observation_seal_sha256'],
                 'feature runtime/source/input binding: ' + domain)
        _require(all(seal[k] is True for k in ('e0_independently_executed', 'same_automatic_procedure_both_domains',
                 'feature_meaning_unchanged', 'feature_scores_use_original_unrepaired_A_B_energies'))
                 and seal['history_only_union'] is False and seal['new_B_or_A_candidates'] is False,
                 'feature unchanged scientific assertions: ' + domain)

    _require(parity['schema'] == 'D1_REFERENCE_FREE_PARITY_AUDIT_V1' and parity['status'] == 'PASS'
             and parity['failures'] == [] and parity['all238_and1003_across_batches_same_frozen_source_inputs_observations'] is True,
             'independent recovery-aware parity PASS required')
    _require(parity['source_sha256'] == _sha(repo / SCIENCE_DIR / 'd1_parity_audit.py'),
             'independent parity verifier source bytes')
    checks = parity['technical_interruption_recovery_checks']
    _require(REQUIRED_PARITY_CHECKS <= set(checks) and all(v is True for v in checks.values()), 'substantive parity recovery checks')
    coverage = parity['execution_journal_coverage']
    _require(coverage == dict(attempt_journal_count=2, journal_rows=len(old_lock['ordered_pending']),
        distinct_domain_index_rows=len(old_lock['ordered_pending']), original_planned_pending_rows=len(old_lock['ordered_pending']),
        duplicate_rows=0), 'independent journal union/disjoint coverage')
    _require(parity['actual_producer_source_files_SHA_verified'] is True
             and parity['actual_producer_source_file_count'] == c['source_count'], 'independent source checks')
    for domain in c['final']:
        path = root / ('d1_' + domain + '_candidates_v1') / 'D1_EXECUTION_RECEIPT.json'
        execution = _read(path)
        _require(_sha(path) == parity['domains'][domain]['D1_EXECUTION_RECEIPT_sha256']
                 and execution['schema'] == 'D1_ACTUAL_E_EXECUTION_RECEIPT_V1'
                 and execution['rows'] == c['final'][domain]
                 and execution['independent_execution_checks_passed'] is True
                 and execution['domain_rows_across_batches_same_frozen_source_inputs_observations'] is True
                 and execution['technical_interruption_recovery_checks'] == checks
                 and execution['feature_seal_sha256'] == feature_bindings[domain]
                 and execution['shared_execution_batch_lock_sha256'] == _sha(new_lock_path)
                 and execution['shared_execution_batch_receipt_sha256'] == runtime_sha
                 and execution['verifier_source_sha256'] == parity['source_sha256'], 'domain independent execution chain: ' + domain)
        _zero(execution['reference_or_accuracy_reads'], 'domain execution reference assertion')
    bindings.update(original_runtime_lock=_sha(old_lock_path), original_runtime_journal=_sha(old_journal),
        recovery_runtime_lock=_sha(new_lock_path), recovery_runtime_journal=_sha(new_journal),
        canonical_runtime_receipt=runtime_sha)
    return dict(schema='D1_RECOVERY_SCORING_GATE_BINDINGS_V1', status='PASS',
        original_execution_batch_id=original_id, recovery_execution_batch_id=recovery_id,
        source_file_count=len(sources), reused_frame_seal_counts=dict(c['reused']),
        final_feature_seal_sha256=feature_bindings, receipt_sha256=bindings,
        launch_source_sha256=dict(launch['source_sha256']), reference_or_accuracy_reads=0,
        interrupted_partial_costs_preserved_as_unknown=True, interrupted_wall_is_upper_bound=True,
        journal_union_owner='DATA_PIPELINE_PARITY.json: independent stored-row checks, not recomputed here',
        helper_source_sha256=_sha(__file__))
