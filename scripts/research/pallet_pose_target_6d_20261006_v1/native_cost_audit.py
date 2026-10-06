"""Verify every completed native TRAIN cost before publishing the fit gate.

Only saved bytes are read. No model, pose solver, geometry generator, feature
loader, auxiliary target pass or fit/evaluation result is required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
from .preflight import DOC, read, sha, inherited_files, original_state
from .baseline import BASELINE_ROOT

PNP = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')
RECORD = np.dtype(dict(
    names=['row_a', 'index_a', 'tag_a', 'row_c', 'index_c', 'tag_c', 'ADD', 'T', 'R', 'WD', 'PnP', 'Generic', 'LM'],
    formats=['<u4', '<u2', 'u1', '<u4', '<u2', 'u1', '<f8', '<f4', '<f4', 'u1', '<u2', '<u2', '<u2'],
    offsets=[0, 4, 6, 7, 11, 13, 14, 22, 26, 30, 31, 33, 35], itemsize=37))


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def durable_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_guard(guard, records, group, binding, writer_sha, journal_bytes):
    """A reservation is never counted as an actual call without completion."""
    assert guard['schema'] == 'durable_row_reservation_v1'
    assert guard['status'] == 'ROW_COMPLETE'
    assert guard['binding'] == binding and guard['writer_sha256'] == writer_sha
    row = guard['row']
    assert isinstance(row, int) and row in set(map(int, group))
    indices = guard['remaining_indices']
    assert indices and indices == sorted(set(indices))
    assert all(isinstance(i, int) and 0 <= i < 201 for i in indices)
    count = len(indices)
    assert guard['reserved_count'] == guard['actual_F_started'] == guard['completed_F'] == count
    assert guard['wal_bytes'] == journal_bytes
    prefix = guard['prefix_bytes']
    assert isinstance(prefix, int) and prefix >= 0 and prefix % 37 == 0
    assert prefix + count * 37 == journal_bytes
    reserved_records = records[prefix // 37:]
    assert len(reserved_records) == count
    assert (reserved_records['row_a'] == row).all() and (reserved_records['row_c'] == row).all()
    assert np.array_equal(reserved_records['index_a'], indices)
    assert np.array_equal(reserved_records['index_c'], indices)
    assert (reserved_records['tag_a'] == 1).all() and (reserved_records['tag_c'] == 2).all()
    return count


def _verify(cache, manifest):
    cache = Path(cache).resolve()
    binding_path = DOC / 'INPUT_BINDINGS.json'
    inputs = read(binding_path)
    header_path = cache / 'cache_header.json'
    header = read(header_path)
    io_path = cache / 'FAST_IO_BINDING.json'
    io = read(io_path)
    code_sha = sha(Path(__file__))
    memory_manifest_path = cache / 'PRETRAIN_COST_MEMORY_MANIFEST.json'
    assert read(memory_manifest_path) == manifest
    assert cache == Path(inputs['diagnostic_cache']).resolve()
    assert inherited_files() == inputs['protected_baseline_files']
    assert original_state(Path(inputs['source_ROOT'])) == inputs['original_user_checkout']
    for entry in inputs['external_inputs']:
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns'], entry['path']
    assert manifest['status'] == 'PASS' and not manifest['blocked_chunks']
    assert manifest['completed_rows'] == manifest['full_TRAIN_rows'] == 55915
    assert manifest['binding'] == header['binding']
    assert manifest['header_sha256'] == sha(header_path)
    assert header['input_bindings_sha256'] == sha(binding_path)
    assert header['protocol_sha256'] == inputs['protocol_sha256'] == sha(DOC / 'PROTOCOL.json')
    assert header['cost_cache_code_sha256'] == sha(Path(__file__).with_name('cost_cache.py'))
    assert header['source_rows'] == 60000 and header['usable_TRAIN_rows'] == 55915
    assert Path(header['source_root']).resolve() == Path(inputs['source_ROOT']).resolve()
    assert Path(header['bank_cache']).resolve() == Path(inputs['baseline_cache']).resolve()
    assert Path(header['cache_dir']).resolve() == cache
    assert header['journal_record_bytes'] == dict(attempt=7, complete=30)
    assert io['binding'] == manifest['IO_binding'] == hash_value(io['contract'])
    contract = io['contract']
    assert contract['science_binding'] == header['binding']
    assert contract['cost_cache_code_sha256'] == header['cost_cache_code_sha256']
    assert contract['fast_writer_sha256'] == manifest['fast_writer_sha256'] == sha(Path(__file__).with_name('fast_cost_cache.py'))
    assert contract['native_auditor_sha256'] == code_sha
    assert contract['transition_sha256'] == manifest['transition_sha256'] == sha(Path(contract['transition_path']))
    transition = read(Path(contract['transition_path']))
    assert manifest['slow_full_elapsed_seconds'] == transition['slow_full_elapsed_seconds']
    assert manifest['seconds_wall'] == manifest['slow_full_elapsed_seconds'] + manifest['fast_elapsed_seconds']
    old_doc = BASELINE_ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
    old_manifest = read(old_doc / 'A_manifest.json')
    assert header['bank_binding'] == old_manifest['bank_binding']
    assert header['bank_sha256'] == next(v['sha256'] for v in old_manifest['cache_files'] if v['name'] == 'source_banks.npy')
    expected_files = {entry['file'] for entry in header['files'].values()} | {'train_rows.npy'}
    assert len(manifest['files']) == len(expected_files)
    assert {entry['file'] for entry in manifest['files']} == expected_files
    for entry in manifest['files']:
        path = cache / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], path
    arrays = {name: np.load(cache / entry['file'], mmap_mode='r') for name, entry in header['files'].items()}
    for name, array in arrays.items():
        assert list(array.shape) == header['files'][name]['shape']
        assert array.dtype == np.dtype(header['files'][name]['dtype']) and not array.flags.writeable
    rows = np.load(cache / 'train_rows.npy', mmap_mode='r')
    assert rows.dtype == np.dtype('int64') and rows.shape == (55915,)
    assert len(np.unique(rows)) == 55915 and np.array_equal(np.flatnonzero(arrays['usable_train']), rows)
    row_hash = hashlib.sha256(str(rows.dtype).encode() + str(rows.shape).encode())
    row_hash.update(np.ascontiguousarray(rows).tobytes())
    assert row_hash.hexdigest() == header['train_rows_sha256']
    ids = read(DOC / 'ID_MANIFEST.json')['IDs']
    assert ids == read(old_doc / 'results/A_ID_MANIFEST.json')['IDs']
    assert len(ids['train']) == 55915 and hash_value(ids['train']) == header['train_ids_sha256']
    assert arrays['done'][rows].all() and (arrays['row_state'][rows] == 2).all()
    assert (arrays['action_counts'][rows] == 201).all()
    cells = chunks = available_cells = failed_cells = scalar_checks = fast_chunks = legacy_chunks = last_guard_cells = 0
    pnp = {name: 0 for name in PNP}
    seconds = 0.; receipt_invocation_F = 0
    journal_bindings = []
    for start in range(0, len(rows), 64):
        group = rows[start:start + 64]
        chunk = start // 64
        journal = cache / 'journals' / f'chunk_{chunk:04d}.bin'
        receipt_path = cache / 'receipts' / f'chunk_{chunk:04d}.json'
        receipt = read(receipt_path)
        assert read(journal.with_suffix('.binding.json')) == dict(binding=header['binding'], chunk=chunk)
        assert receipt['chunk'] == chunk and receipt['rows'] == group.tolist()
        assert receipt['status'] == 'PASS' and receipt['binding'] == header['binding']
        raw = journal.read_bytes()
        count = len(group) * 201
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == count * 37 == receipt['journal_bytes']
        assert digest == receipt['journal_sha256']
        records = np.frombuffer(raw, dtype=RECORD)
        expected_rows = np.repeat(group, 201)
        expected_indices = np.tile(np.arange(201), len(group))
        for key, expected in [('row_a', expected_rows), ('row_c', expected_rows),
                              ('index_a', expected_indices), ('index_c', expected_indices)]:
            assert np.array_equal(records[key], expected), (chunk, key)
            scalar_checks += count
        assert (records['tag_a'] == 1).all() and (records['tag_c'] == 2).all()
        scalar_checks += 2 * count
        r, i = records['row_a'], records['index_a']
        for field, name in [('ADD', 'cost_ADDsym_m'), ('T', 'translation_cm'),
                            ('R', 'rotation_deg'), ('WD', 'WD_hypothesis')]:
            assert np.array_equal(records[field], arrays[name][r, i], equal_nan=True), (chunk, name)
            scalar_checks += count
        available = np.isfinite(records['ADD'])
        assert np.array_equal(available, arrays['F_available'][r, i])
        assert (arrays['candidate_state'][r, i] == 2).all()
        scalar_checks += 2 * count
        assert np.isposinf(records['ADD'][~available]).all()
        assert np.isposinf(records['T'][~available]).all() and np.isposinf(records['R'][~available]).all()
        assert (records['WD'][~available] == 0).all()
        assert np.isfinite(records['T'][available]).all() and np.isfinite(records['R'][available]).all()
        assert ((records['WD'][available] >= 1) & (records['WD'][available] <= 3)).all()
        costs = np.array(arrays['cost_ADDsym_m'][group])
        has_target = np.isfinite(costs).any(-1)
        target = np.where(has_target, costs.argmin(-1), -1)
        assert np.array_equal(target, arrays['oracle_index'][group])
        scalar_checks += len(group)
        assert receipt['attempted'] == receipt['completed'] == count
        assert receipt['incomplete_rows'] == 0 and receipt['all_F_invalid_rows'] == int((~has_target).sum())
        assert 0 <= receipt['new_F_calls'] <= count
        for field, name in [('PnP', 'solvePnP'), ('Generic', 'solvePnPGeneric'), ('LM', 'solvePnPRefineLM')]:
            actual = int(records[field].astype(np.uint64).sum())
            assert actual == receipt['PnP_counts'][name]
            pnp[name] += actual
        guard_path = cache / 'fast_guards' / f'chunk_{chunk:04d}.json'
        if 'fast_writer_sha256' in receipt:
            assert receipt['fast_writer_sha256'] == contract['fast_writer_sha256']
            assert receipt['IO_binding'] == io['binding']
            assert receipt['actual_F_completed_cumulative'] == count
            assert receipt['actual_F_started_this_invocation'] == receipt['actual_F_completed_this_invocation'] == receipt['fast_invocation_new_F_calls']
            assert receipt['reserved_candidates_this_invocation'] == receipt['fast_invocation_new_F_calls']
            assert 0 <= receipt['fast_invocation_new_F_calls'] <= count
            guard = read(guard_path)
            last_guard_cells += verify_guard(guard, records, group, header['binding'], contract['fast_writer_sha256'], len(raw))
            fast_chunks += 1
        else:
            assert not guard_path.exists(), 'An old PASS receipt cannot hide a fast reservation'
            legacy_chunks += 1
        journal_bindings.append(dict(chunk=chunk, journal_sha256=digest, receipt_sha256=sha(receipt_path)))
        cells += count; chunks += 1
        available_cells += int(available.sum()); failed_cells += int((~available).sum())
        seconds += receipt['seconds']; receipt_invocation_F += receipt['new_F_calls']
    assert cells == 11238915 and chunks == 874 and fast_chunks + legacy_chunks == chunks
    assert manifest['candidate_F_attempted'] == manifest['candidate_F_completed'] == manifest['registered_F_calls_total'] == cells
    assert manifest['incomplete_attempted_F'] == manifest['pending_rows'] == manifest['incomplete_attempt_rows'] == 0
    assert manifest['candidate_F_available'] == available_cells and manifest['candidate_F_failures'] == failed_cells
    assert available_cells + failed_cells == cells
    assert manifest['available_targets'] == int((arrays['oracle_index'][rows] >= 0).sum())
    assert manifest['excluded_all_F_invalid'] == int((arrays['oracle_index'][rows] < 0).sum())
    assert manifest['available_targets'] + manifest['excluded_all_F_invalid'] == len(rows)
    assert manifest['oracle_NoOp'] == int((arrays['oracle_index'][rows] == 0).sum())
    assert manifest['PnP_counts'] == pnp
    assert manifest['chunk_seconds_sum'] == seconds
    assert manifest['invocation_new_F_calls'] == receipt_invocation_F
    expected_guard_files = {f'chunk_{entry["chunk"]:04d}.json' for entry in journal_bindings
                            if 'fast_writer_sha256' in read(cache / 'receipts' / f'chunk_{entry["chunk"]:04d}.json')}
    assert {p.name for p in (cache / 'fast_guards').glob('*.json')} == expected_guard_files
    return dict(schema='pretrain_native_cost_verification_v1', status='PASS',
        raw_native_cost_records=cells, completed_chunks=chunks, full_usable_TRAIN=len(rows),
        WAL_record_bytes=37, independent_scalar_checks=scalar_checks,
        check_scope='12 saved scalar fields per action and exact first-argmin per TRAIN row; additional metadata and failure-state assertions excluded from count',
        F_available=available_cells, F_failures=failed_cells, cost_PnP_counts=pnp,
        legacy_readonly_chunks=legacy_chunks, fast_chunks=fast_chunks,
        final_row_guard_completed_records=last_guard_cells,
        reservation_scope='all final fast guards ROW_COMPLETE with exact durable WAL suffix; reservations never substituted for actual F completions',
        science_binding=header['binding'], IO_binding=io['binding'], IO_contract_sha256=sha(io_path),
        code_sha256=code_sha, original_cost_code_sha256=header['cost_cache_code_sha256'],
        fast_writer_sha256=contract['fast_writer_sha256'], header_sha256=sha(header_path),
        input_bindings_sha256=sha(binding_path), memory_manifest_sha256=hash_value(manifest),
        memory_manifest_artifact_path=str(memory_manifest_path),
        memory_manifest_artifact_sha256=sha(memory_manifest_path),
        memory_manifest_scope='constructed before adding audit receipt SHA and post-audit elapsed metadata; intentionally differs from final publication JSON SHA',
        journal_receipt_set_sha256=hash_value(journal_bindings),
        protected_baseline_files=len(inputs['protected_baseline_files']),
        external_input_size_mtime_checks=len(inputs['external_inputs']),
        input_check_scope='unchanged size/mtime against sealed full-byte beginning SHA audit; protected baseline content and original user checkout rechecked before any NN',
        original_user_checkout='PRESERVED BEFORE AUTHORIZED MAIN PUBLICATION',
        slow_full_elapsed_seconds=manifest['slow_full_elapsed_seconds'], fast_elapsed_seconds=manifest['fast_elapsed_seconds'],
        additional_model_calls=0, additional_F_calls=0, additional_PnP_calls=0, additional_optimizer_updates=0,
        fit_evaluation_aux_dependencies=0, cache_writes=0)


def verify_native_cost(cache, manifest):
    began = time.monotonic()
    path = DOC / 'PRETRAIN_COST_VERIFICATION.json'
    try:
        result = _verify(cache, manifest)
        result['seconds_wall'] = time.monotonic() - began
        durable_write(path, result)
        return result
    except Exception as error:
        durable_write(path, dict(schema='pretrain_native_cost_verification_v1', status='FAILED_COST_INTEGRITY',
            exception=type(error).__name__, reason=str(error), code_sha256=sha(Path(__file__)),
            memory_manifest_sha256=hash_value(manifest), seconds_wall=time.monotonic() - began,
            additional_model_calls=0, additional_F_calls=0, additional_PnP_calls=0,
            additional_optimizer_updates=0, final_PASS_publication_allowed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cost-cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True,
        help='Fully constructed complete manifest; this CLI is never valid on a partial cache')
    args = parser.parse_args(argv)
    print(json.dumps(verify_native_cost(args.cost_cache, read(args.manifest)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
