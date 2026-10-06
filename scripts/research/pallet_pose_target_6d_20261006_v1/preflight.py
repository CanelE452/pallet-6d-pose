"""Read-only baseline and input audit for the matched final-pose target experiment.

No legacy driver is called. Large arrays are hashed in place, never copied.
Beginning and ending audits use the same independently inherited expectations.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_target_6d_20261006_v1'
DOC = ROOT / '_docs/experiments' / NAME
OLD_CODE = 'scripts/research/pallet_joint_action_handoff_20261006_v1'
OLD_DOC = '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
from .baseline import BASELINE_ROOT
BASE = 'a22fb14beb5e8df08076385000e0d53503c1ae29'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def original_state(source_root):
    status = subprocess.check_output(['git', '-C', str(source_root), 'status', '--porcelain=v1'], text=True)
    diff = subprocess.check_output(['git', '-C', str(source_root), 'diff', '--binary'], text=True)
    changed = git(source_root, 'diff', '--name-only').splitlines()
    state = dict(path=str(source_root), head=git(source_root, 'rev-parse', 'HEAD'),
                 branch=git(source_root, 'branch', '--show-current'),
                 status_sha256=hashlib.sha256(status.encode()).hexdigest(),
                 diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                 status_entries=len(status.splitlines()),
                 tracked_changes={p: {'sha256': sha(source_root / p)}
                                  for p in changed if (source_root / p).is_file()})
    return state


def inherited_files():
    names = git(BASELINE_ROOT, 'ls-files', OLD_CODE, OLD_DOC).splitlines()
    entries = []
    for name in names:
        path = BASELINE_ROOT / name
        actual_blob = git(BASELINE_ROOT, 'hash-object', str(path))
        expected_blob = git(ROOT, 'rev-parse', BASE + ':' + name)
        if actual_blob != expected_blob:
            raise ValueError('Protected baseline artifact changed: ' + name)
        entries.append(dict(path=name, sha256=sha(path), bytes=path.stat().st_size,
                            baseline_git_blob=expected_blob))
    return entries


def expected_inputs(source_root, baseline_cache):
    old = BASELINE_ROOT / OLD_DOC
    protocol = read(old / 'A_protocol.json')
    entries = {}

    def add(path, expected_sha, role, expected_bytes=None):
        p = Path(path).resolve()
        key = str(p)
        if key in entries:
            if entries[key]['expected_sha256'] != expected_sha:
                raise ValueError('Inherited SHA expectations disagree: ' + key)
            entries[key]['roles'].append(role)
        else:
            entries[key] = dict(path=key, expected_sha256=expected_sha,
                                expected_bytes=expected_bytes, roles=[role])

    for entry in protocol['input_bindings']:
        add(source_root / entry['path'], entry['sha256'], 'original A input contract', entry.get('bytes'))
    for entry in read(old / 'INPUT_DEPENDENCIES.json')['inputs']:
        path = source_root / entry['relative_to_source'] if entry['relative_to_source'] else Path(entry['path'])
        add(path, entry['sha256'], 'inherited input dependency: ' + '; '.join(entry['roles']), entry.get('bytes'))
    for entry in read(old / 'SOURCE_CACHE_HASHES.json')['arrays']:
        add(source_root / entry['path'], entry['sha256'], 'source array; current full-byte equality', entry['bytes'])
    manifest = read(old / 'A_manifest.json')
    for entry in manifest['cache_files']:
        add(baseline_cache / entry['name'], entry['sha256'], 'frozen original native GEO bank', entry['bytes'])
    for entry in manifest['fits']:
        path = baseline_cache / Path(entry['checkpoint_path']).relative_to(Path(manifest['cache_path']))
        add(path, entry['checkpoint_sha256'], 'old fit checkpoint; immutable and never trained')
    guard = read(baseline_cache / 'GENERATION_CODE_BINDINGS.json')
    for path, digest in guard['code_bindings'].items():
        add(BASELINE_ROOT / path, digest, 'bank generation/F code unchanged')
    # These additional baseline files have Git/receipt-bound content but no
    # separate inherited byte digest. Freeze their current bytes before use.
    for name in ('BANK_BINDING.json', 'BANK_NAMES.json', 'GENERATION_CODE_BINDINGS.json'):
        p = baseline_cache / name
        add(p, sha(p), 'baseline cache metadata locked at preflight')
    add(source_root / 'data/pallet/results/pallet_line_pose_v1/cache/done.npy',
        sha(source_root / 'data/pallet/results/pallet_line_pose_v1/cache/done.npy'),
        'completion bitmap locked at preflight')
    return list(entries.values())


def check_entry(entry):
    p = Path(entry['path'])
    started = time.monotonic()
    digest = sha(p)
    stat = p.stat()
    if digest != entry['expected_sha256']:
        raise ValueError('Input content mismatch: ' + str(p))
    if entry['expected_bytes'] is not None and stat.st_size != entry['expected_bytes']:
        raise ValueError('Input byte length mismatch: ' + str(p))
    return dict(entry, sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                check='current full-byte SHA256', elapsed_seconds=time.monotonic() - started)


def audit(source_root, baseline_cache, diagnostic_cache, ending=False):
    source_root, baseline_cache, diagnostic_cache = map(Path, (source_root, baseline_cache, diagnostic_cache))
    # A repeated preflight must never replace a sealed binding with different
    # elapsed-time metadata and thereby invalidate the completed diagnostic.
    if not ending and (DOC / 'INPUT_BINDINGS.json').exists():
        frozen = read(DOC / 'PREFLIGHT.json')
        if frozen['status'] != 'PASS' or frozen['input_bindings_sha256'] != sha(DOC / 'INPUT_BINDINGS.json'):
            raise ValueError('Existing preflight/binding is not a valid sealed beginning audit')
        return audit(source_root, baseline_cache, diagnostic_cache, ending=True)
    start = time.monotonic()
    receipt = DOC / ('audit/ENDING_INPUT_AUDIT.json' if ending else 'PREFLIGHT.json')
    if receipt.exists() and read(receipt).get('status') != 'PASS':
        previous = read(receipt)
        attempts_path = DOC / 'audit/PREFLIGHT_ATTEMPTS.json'
        attempts = read(attempts_path) if attempts_path.exists() else {'attempts': []}
        attempts['attempts'].append(previous)
        write(attempts_path, attempts)
    try:
        if git(BASELINE_ROOT, 'merge-base', 'HEAD', BASE) != BASE:
            raise ValueError('Worktree is not descended from required baseline')
        protected = inherited_files()
        original = original_state(source_root)
        handoff = ROOT.parent / 'pallet-pose-handoff-20261006'
        prior = dict(path=str(handoff), head=git(handoff, 'rev-parse', 'HEAD'),
                     status=git(handoff, 'status', '--porcelain=v1'))
        if prior['head'] != BASE or prior['status']:
            raise ValueError('Prior completed handoff worktree changed')
        if ending:
            initial = read(DOC / 'INPUT_BINDINGS.json')
            if (source_root.resolve() != Path(initial['source_ROOT']).resolve() or
                baseline_cache.resolve() != Path(initial['baseline_cache']).resolve() or
                diagnostic_cache.resolve() != Path(initial['diagnostic_cache']).resolve()):
                raise ValueError('Ending audit paths differ from sealed input bindings')
            expectations = initial['external_inputs']
        else:
            expectations = expected_inputs(source_root, baseline_cache)
        # Independent files: parallel streaming is bounded to two readers.
        with ThreadPoolExecutor(max_workers=2) as pool:
            inputs = list(pool.map(check_entry, expectations))
        protocol = DOC / 'PROTOCOL.json'
        if not ending:
            locked = read(protocol)
            if locked['baseline_commit'] != BASE:
                raise ValueError('New protocol baseline mismatch')
            IDs = read(BASELINE_ROOT / OLD_DOC / 'results/A_ID_MANIFEST.json')['IDs']['synthetic_evaluation']
            if len(IDs) != 1985 or len(set(IDs)) != 1985:
                raise ValueError('Inherited whole evaluation ID registry mismatch')
            write(DOC / 'SYNTH_ID_AUDIT.json', dict(schema='whole_synthetic_pose_target_ID_v1', IDs=IDs,
                source_ID_manifest_sha256=sha(BASELINE_ROOT / OLD_DOC / 'results/A_ID_MANIFEST.json'),
                role='all prior heldout1985 repeated-use development; no TRAIN eligibility filter for evaluation'))
            instruction = Path('/home/minjae/.codex/attachments/71ba7103-71b7-4b0f-8c45-886200fa8b67/Pasted text.txt')
            write(DOC / 'INPUT_BINDINGS.json', dict(schema='pose_target_input_bindings_v1',
                baseline_commit=BASE, protocol_sha256=sha(protocol),
                instructions=dict(path=str(instruction), sha256=sha(instruction), lines=len(instruction.read_text().splitlines()), read_scope='entire document'),
                protected_baseline_files=protected, external_inputs=inputs,
                original_user_checkout=original, prior_completed_worktree=prior,
                source_ROOT=str(source_root), baseline_cache=str(baseline_cache), diagnostic_cache=str(diagnostic_cache),
                source_cache_hash_scope='17 source arrays rehashed in full for this run; inherited expectations unmodified',
                bank_access='read-only np.load mmap_mode=r; no BankStore constructor or old stage driver'))
        else:
            if protected != initial['protected_baseline_files'] or original != initial['original_user_checkout'] or prior != initial['prior_completed_worktree']:
                raise ValueError('Beginning/ending protected evidence or user checkout differs')
            if sha(protocol) != initial['protocol_sha256']:
                raise ValueError('Locked new protocol changed after metric execution')
        environment = dict(python_executable=sys.executable, python_version=sys.version,
            packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'opencv-python')},
            GPU=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu', '--format=csv,noheader'], text=True).strip(),
            GPU_compute=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip())
        result = dict(schema='pose_target_readonly_input_audit_v1', status='PASS', phase='ending' if ending else 'beginning',
            observed_utc=datetime.now(timezone.utc).isoformat(), baseline_commit=BASE,
            branch=git(ROOT, 'branch', '--show-current'), worktree_head=git(ROOT, 'rev-parse', 'HEAD'),
            origin=git(ROOT, 'remote', 'get-url', 'origin'),
            remote_baseline_branch=git(ROOT, 'rev-parse', 'origin/research/joint-action-handoff-20261006'),
            remote_main=git(ROOT, 'rev-parse', 'origin/main'),
            remote_diff_scope='required baseline equals fetched remote handoff; fetched main88ee557 is baseline ancestor; no upstream changes since initial fetch',
            protected_file_count=len(protected), external_input_count=len(inputs),
            input_bytes_hashed=sum(x['bytes'] for x in inputs),
            input_bindings_sha256=sha(DOC / 'INPUT_BINDINGS.json'),
            protocol_sha256=sha(protocol),
            cost_cache_code_sha256=sha(Path(__file__).parent / 'cost_cache.py') if (Path(__file__).parent / 'cost_cache.py').exists() else None,
            environment=environment,
            copies=0, fit_updates=0, model_forward_calls=0, F_calls=0,
            elapsed_seconds=time.monotonic() - start)
        write(receipt, result)
        return result
    except Exception as error:
        write(receipt, dict(schema='pose_target_readonly_input_audit_v1', status='FAILED_CONTRACT',
            phase='ending' if ending else 'beginning', exception_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - start, fit_updates=0, model_forward_calls=0, F_calls=0))
        raise
