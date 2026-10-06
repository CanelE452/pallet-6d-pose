"""Link final fast-I/O receipts to their frozen pre-training evidence.

This checks saved metadata only. The native-cost and final output verifiers
independently check all WAL records; this command adds their provenance links.
It never imports scientific modules, runs a solver/model, or writes the cache.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_pose_target_6d_20261006_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verify(cache, doc=DOC, code=Path(__file__).parent):
    started = time.monotonic()
    cache, doc, code = map(Path, (cache, doc, code))
    manifest = read(doc / 'POSE_COST_CACHE_MANIFEST.json')
    audit_path = doc / 'PRETRAIN_COST_VERIFICATION.json'
    audit = read(audit_path)
    io_path = cache / 'FAST_IO_BINDING.json'
    io = read(io_path)
    contract = io['contract']
    memory_path = cache / 'PRETRAIN_COST_MEMORY_MANIFEST.json'
    memory = read(memory_path)
    execution = read(doc / 'FAST_IO_EXECUTION.json')
    seal = read(doc / 'audit/FAST_IO_CODE_SEAL.json')
    tests = read(doc / 'FAST_IO_TEST_RESULTS.json')
    assert manifest['status'] == audit['status'] == execution['status'] == 'PASS'
    assert manifest['pretrain_cost_verification_sha256'] == sha(audit_path)
    assert audit['raw_native_cost_records'] == manifest['candidate_F_completed'] == 11238915
    assert audit['completed_chunks'] == 874 and audit['full_usable_TRAIN'] == 55915
    assert audit['F_available'] == manifest['candidate_F_available']
    assert audit['F_failures'] == manifest['candidate_F_failures']
    assert audit['cost_PnP_counts'] == manifest['PnP_counts']
    assert not manifest['blocked_chunks'] and manifest['incomplete_attempted_F'] == 0
    assert io['binding'] == canonical(contract)
    assert manifest['IO_binding'] == audit['IO_binding'] == execution['IO_binding'] == io['binding']
    assert audit['IO_contract_sha256'] == sha(io_path)
    header_path = cache / 'cache_header.json'
    header = read(header_path)
    assert manifest['header_sha256'] == audit['header_sha256'] == sha(header_path)
    assert contract['science_binding'] == audit['science_binding'] == header['binding']
    expected_codes = {'fast_cost_cache.py': 'fast_writer_sha256',
                      'native_cost_audit.py': 'native_auditor_sha256',
                      'cost_cache.py': 'cost_cache_code_sha256'}
    for name, field in expected_codes.items():
        assert contract[field] == seal['code_sha256'][name] == sha(code / name)
    assert audit['code_sha256'] == contract['native_auditor_sha256']
    assert manifest['fast_writer_sha256'] == audit['fast_writer_sha256'] == execution['fast_writer_sha256'] == contract['fast_writer_sha256']
    assert audit['original_cost_code_sha256'] == execution['original_cost_code_sha256'] == header['cost_cache_code_sha256'] == contract['cost_cache_code_sha256']
    transition_path = Path(contract['transition_path'])
    assert transition_path.resolve() == (cache / 'IO_TRANSITION.json').resolve()
    assert manifest['transition_sha256'] == seal['transition_sha256'] == contract['transition_sha256'] == sha(transition_path)
    assert read(transition_path)['status'] == 'OLD_QUIESCED_READY_FAST'
    assert memory_path.resolve() == Path(audit['memory_manifest_artifact_path']).resolve()
    assert audit['memory_manifest_artifact_sha256'] == sha(memory_path)
    assert audit['memory_manifest_sha256'] == canonical(memory)
    omitted = {'pretrain_cost_verification_sha256', 'native_audit_wall_seconds',
               'fast_elapsed_seconds', 'seconds_wall'}
    assert {k: v for k, v in manifest.items() if k not in omitted} == {k: v for k, v in memory.items() if k not in omitted}
    assert manifest['seconds_wall'] == manifest['slow_full_elapsed_seconds'] + manifest['fast_elapsed_seconds']
    assert manifest['fast_elapsed_seconds'] >= memory['fast_elapsed_seconds']
    assert manifest['native_audit_wall_seconds'] >= audit['seconds_wall'] >= 0
    assert execution['slow_full_elapsed_seconds'] == manifest['slow_full_elapsed_seconds']
    assert execution['fast_elapsed_seconds'] == manifest['fast_elapsed_seconds']
    assert execution['fast_actual_F_started_this_invocation'] == manifest['fast_invocation_new_F_calls']
    assert execution['readonly_reused_chunks'] == audit['legacy_readonly_chunks'] == 161
    assert execution['new_chunks'] == audit['fast_chunks'] == 713
    assert execution['original_header_bytes_unchanged'] and execution['original_source_bank_math_unchanged']
    assert not execution['blocked_chunks']
    assert audit['input_bindings_sha256'] == seal['input_bindings_sha256'] == sha(doc / 'INPUT_BINDINGS.json')
    assert seal['protocol_sha256'] == sha(doc / 'PROTOCOL.json')
    assert audit['protected_baseline_files'] == 351 and audit['external_input_size_mtime_checks'] == 710
    for key in ('additional_model_calls', 'additional_F_calls', 'additional_PnP_calls', 'additional_optimizer_updates'):
        assert audit[key] == 0
    assert tests['status'] == 'PASS' and tests['tests_run'] == tests['passed'] == 12 and tests['failed'] == 0
    assert tests['actual_F_calls'] == tests['actual_PnP_calls'] == tests['new_network_forwards'] == tests['new_optimizer_updates'] == 0
    for entry in tests['source_bindings']:
        assert sha(code / Path(entry['path']).name) == entry['sha256']
    for name, digest in seal['code_sha256'].items():
        assert sha(code / name) == digest
    return dict(schema='final_fast_IO_provenance_verification_v1', status='PASS',
                native_cost_records=11238915, chunks=874, IO_binding=io['binding'],
                pretrain_receipt_sha256=sha(audit_path), final_manifest_sha256=sha(doc / 'POSE_COST_CACHE_MANIFEST.json'),
                memory_manifest_artifact_sha256=sha(memory_path), memory_manifest_canonical_sha256=canonical(memory),
                FAST_IO_test_receipt_sha256=sha(doc / 'FAST_IO_TEST_RESULTS.json'),
                frozen_scientific_and_IO_code_unchanged=True,
                verification_scope='saved receipt, code, transition, memory and final manifest links; no new WAL/solver/model calls',
                new_F_calls=0, new_PnP_calls=0, new_model_calls=0, new_optimizer_updates=0,
                cache_writes=0, seconds=time.monotonic() - started, code_sha256=sha(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cost-cache', type=Path, default=Path('/tmp/pallet-pose-target-6d-cache'))
    args = parser.parse_args(argv)
    result = verify(args.cost_cache)
    output = DOC / 'audit/FAST_IO_FINAL_BINDING_VERIFICATION.json'
    pending = output.with_suffix('.pending')
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    os.replace(pending, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
