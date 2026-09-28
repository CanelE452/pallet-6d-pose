"""CPU-only final verification/accounting; no fitting, old writes, or Git mutation."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import time

from . import common as C


def tests():
    old_path = C.DOC / 'TEST_RESULTS.json'
    ledger_value = None
    if old_path.exists():
        previous = C.bind(old_path)
        archived = C.RAW / f'verification_history/TEST_RESULTS_{previous["sha256"]}.json'
        C.save(archived, C.read(old_path), freeze=True)
        assert C.sha(archived) == previous['sha256']
        ledger_value = C.read(C.DOC / 'RESOURCE_LEDGER.json')
        for event in ledger_value['events']:
            details = event.get('details') or {}
            if details.get('artifact') == previous:
                details['artifact'] = C.bind(archived)
                details['artifact_relocated_for_retention'] = True
    targets = ['scripts/research/pallet_pose_objective_followup_v2',
               'scripts/research/pallet_material_selftrain_closure_v1/test_pair.py',
               'scripts/research/pallet_type_selftrain_v1/test_contract.py']
    command = [sys.executable, '-m', 'pytest', '-q', *targets]
    begin = time.perf_counter()
    run = subprocess.run(command, cwd=C.ROOT, capture_output=True, text=True)
    print(run.stdout, end='')
    print(run.stderr, end='', file=sys.stderr)
    match = re.search(r'(\d+) passed in ([\d.]+)s', run.stdout)
    sources = list((C.ROOT / targets[0]).glob('*.py')) + [C.ROOT / p for p in targets[1:]]
    result = dict(created_at=C.now(), status='PASS' if run.returncode == 0 and match else 'FAIL',
                  tests=int(match[1]) if match else 0, seconds=float(match[2]) if match else None,
                  wall_seconds=time.perf_counter()-begin, command=' '.join(command),
                  source_files=[C.bind(p) for p in sorted(sources)], pytest_output=run.stdout,
                  returncode=run.returncode, new_fits=0, GPU_seconds=0)
    # Tests consume the current report's source hashes. Relocate the previous
    # receipt only after they have finished; the next report rebinds this ledger.
    if ledger_value is not None:
        C.save(C.DOC / 'RESOURCE_LEDGER.json', ledger_value)
    C.save(C.DOC / 'TEST_RESULTS.json', result)
    assert result['status'] == 'PASS'


def ledger():
    audit_path = C.DOC / 'AUDIT.json'
    if audit_path.exists() and C.read(audit_path)['status'] == 'FAIL':
        old_audit = C.bind(audit_path)
        C.save(C.RAW / f'verification_history/AUDIT_{old_audit["sha256"]}.json',
               C.read(audit_path), freeze=True)
    for name, key in [('BRANCH_DIAGNOSTIC.json', 'wall_seconds'),
                      ('TRAINABLE_PARAMETER_INVENTORY.json', 'CPU_wall_seconds'),
                      ('TEST_RESULTS.json', 'wall_seconds')]:
        path = C.DOC / name
        item = C.read(path)
        event = 'FINAL_CPU_' + name + ('_' + C.sha(path)[:12] if name == 'TEST_RESULTS.json' else '')
        if event not in {e['event'] for e in C.read(C.DOC / 'RESOURCE_LEDGER.json')['events']}:
            C.resource(event, seconds=item.get(key, 0), details=dict(artifact=C.bind(path), GPU_seconds=0))
    candidate_lock = C.RAW / 'branch_diagnostic/CANDIDATES_LOCK.json'
    if 'FINAL_CPU_BRANCH_CANDIDATE_FREEZE' not in {e['event'] for e in C.read(C.DOC / 'RESOURCE_LEDGER.json')['events']}:
        C.resource('FINAL_CPU_BRANCH_CANDIDATE_FREEZE', seconds=C.read(candidate_lock)['wall_seconds'],
                   details=dict(artifact=C.bind(candidate_lock), GPU_seconds=0))
    value = C.read(C.DOC / 'RESOURCE_LEDGER.json')
    assert value['totals']['fits'] == 12 and value['totals']['optimizer_updates'] == 3840
    value['totals']['elapsed_wall_seconds'] = time.time()-value['start_unix']
    assert value['totals']['elapsed_wall_seconds'] <= value['caps']['wall_seconds']
    value['closure_accounting'] = dict(all_declared_fits_complete=True, remaining_authorized_fits=0,
        method_search_closed=True, actual_main_cycles=3, effective_stochastic_replication='NOT_RUN',
        attempted_replication_fits_counted=4, attempted_replication_updates_counted=1280,
        execution='PARTIAL_BUDGET', accounting_at=C.now(),
        timing_scope='GPU seconds: active measured fit+inference/probe wall time, not GPU kernel profiler; CPU/report/publication wall may continue after this accounting snapshot')
    C.save(C.DOC / 'RESOURCE_LEDGER.json', value)
    C.state('ANALYSIS_COMPLETE_NO_MORE_FITS',
            ['12 actual fits', 'paired evaluation', 'D9 branch diagnostics', 'ineffective seed-repeat correction'],
            'Finalize reports/audit and publish scoped changes; no training continuation')


def audit_report():
    result = C.read(C.DOC / 'AUDIT.json')
    assert result['status'] in ('PASS', 'PASS_WITH_LIMITATIONS'), result['status']
    test = C.read(C.DOC / 'TEST_RESULTS.json')
    lines = ['# 최종 검증', '', f"검증 상태: {result['status']}. 외부 통제 tests 포함 {test['tests']} tests PASS.", '',
             '| 검사 | 결과 |', '|---|---|']
    lines += [f"| {name} | {value['status']} |" for name, value in result['checks'].items()]
    lines += ['', '검증 통과는 성능 개선이나 독립 재현 성공을 의미하지 않는다. 유효한 난수 반복은 미완료이며 실행 판정은 PARTIAL_BUDGET이다. 원고·기존 결과는 변경하지 않았다.', '',
              '[기계 검증](AUDIT.json), [test 실행](TEST_RESULTS.json), [최종 판정](FINAL_DECISION.json), [반복 정정](REPLICATION_VALIDITY_CORRECTION.md).', '']
    C.save(C.DOC / 'AUDIT_KO.md', '\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['tests', 'ledger', 'audit-report'])
    action = parser.parse_args().action
    {'tests': tests, 'ledger': ledger, 'audit-report': audit_report}[action]()
