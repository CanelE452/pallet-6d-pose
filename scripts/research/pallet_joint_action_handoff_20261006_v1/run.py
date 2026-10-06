"""Handoff inspection and verified reuse, without writing historical evidence.

New schema: status.json records individual scientific and execution states;
run_manifest.json binds small deliverables and the explicit component receipts.
No missing data are synthesized. Git closure is stored outside the committed
files, avoiding a self-referential final commit hash.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
CODE = ROOT / 'scripts/research/pallet_joint_action_handoff_20261006_v1'
BUILD_TEMP_SUFFIXES = {'.aux', '.log', '.out', '.bbl', '.blg', '.toc', '.fls', '.xdv', '.fdb_latexmk'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def preservation():
    state = read(DOC / 'INITIAL_STATE.json')['original_checkout']
    root = Path(state['path'])
    current_status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain=v1'], text=True)
    current_diff = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary'], text=True)
    checks = {'original_head': git(root, 'rev-parse', 'HEAD') == state['head'],
              'original_branch': git(root, 'branch', '--show-current') == state['branch'],
              'original_status': hashlib.sha256(current_status.encode()).hexdigest() == state['status_sha256'],
              'original_diff': hashlib.sha256(current_diff.encode()).hexdigest() == state['diff_sha256']}
    for path, record in state['tracked_changes'].items():
        checks['tracked:' + path] = sha(root / path) == record['sha256']
    if not all(checks.values()):
        raise ValueError('Original checkout changed; preserve and inspect: ' + repr(checks))
    return checks


def reuse():
    # The existing stdlib verifier checks the published raw 8910-row evidence,
    # exact old 12/96 denominators and source hashes without fitting/inference.
    from scripts.research.pallet_github_publication_20261006_v1 import verify_published_evidence as V
    path = DOC / 'REUSED_EVIDENCE_VERIFICATION.json'
    if path.exists():
        old = read(path)
        if old.get('status') != 'PASS' or old.get('failed_check_count') != 0:
            raise ValueError('Existing reuse verification failed')
        if old.get('input_bindings'):
            for entry in old['input_bindings']:
                if sha(ROOT / entry['path']) != entry['sha256']:
                    raise ValueError('Reused evidence input drift: ' + entry['path'])
            return old
    seen = {ROOT / V.MANIFEST, Path(V.__file__)}
    original = V.Evidence.open
    def observed_open(evidence, source):
        rel = evidence.relative(source)
        entry = evidence.mapping.get(rel)
        seen.add(evidence.root / (entry['artifact_path'] if entry else rel))
        return original(evidence, source)
    V.Evidence.open = observed_open
    try:
        result = V.verify(ROOT)
    finally:
        V.Evidence.open = original
    if result['status'] != 'PASS':
        raise ValueError('Published evidence verification failed')
    result['input_bindings'] = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(seen)]
    result['input_binding_schema'] = 'New handoff metadata, all actual verifier-opened artifacts plus manifest/verifier code; resume checks bytes before reuse'
    write(path, result)
    return result


def file_inventory():
    entries = []
    for base in (CODE, DOC):
        for path in sorted(base.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:
                continue
            if path.name in ('run_manifest.json',) or path.suffix in ('.pending', '.pyc'):
                continue
            if path.suffix in BUILD_TEMP_SUFFIXES or path.name.endswith('.synctex.gz'):
                continue
            # Build temporary files/logs and copyrighted primary PDFs live
            # outside this tree; our own final manuscript PDFs are deliverables.
            entries.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size})
    for rel in ('readme.md', '_docs/notes/joint-action-handoff.md', '_docs/history/2026-10-06.md'):
        path = ROOT / rel
        entries.append({'path': rel, 'sha256': sha(path), 'bytes': path.stat().st_size})
    return entries


def assemble():
    initial = read(DOC / 'INITIAL_STATE.json')
    components = {}
    for name in ('A', 'B', 'C'):
        path = DOC / f'{name}_status.json'
        components[name] = read(path) if path.exists() else {
            'status': 'NOT_RUN_DEPENDENCY', 'dependency': f'{name} component receipt is not yet available'}
    runtime = read(DOC / 'RUNTIME_MATCHED.json') if (DOC / 'RUNTIME_MATCHED.json').exists() else None
    runtime_check = read(DOC / 'RUNTIME_VERIFICATION.json') if (DOC / 'RUNTIME_VERIFICATION.json').exists() else None
    runtime_a = read(DOC / 'RUNTIME_A.json') if (DOC / 'RUNTIME_A.json').exists() else None
    runtime_a_check = read(DOC / 'RUNTIME_A_VERIFICATION.json') if (DOC / 'RUNTIME_A_VERIFICATION.json').exists() else None
    validation_path = DOC / 'VALIDATION_RESULTS.json'
    validation = read(validation_path) if validation_path.exists() else None
    fits = []
    for path in sorted((DOC / 'A_fits').glob('*seed*.json')):
        fit = read(path)
        fits.append({'arm': fit['arm'], 'seed': fit['seed'],
                     'status': 'DONE' if fit.get('complete') and fit['updates'] == 6000 else 'NOT_RUN_DEPENDENCY',
                     'updates': fit['updates'], 'exposures': fit['exposures'],
                     'receipt': str(path.relative_to(DOC))})
    checks = preservation()
    status = {'schema': 'joint_action_handoff_status_v1', 'components': components,
              'formal_fit_states': fits,
              'validation': {'status': 'DONE' if validation and validation['status'] == 'PASS' else 'NOT_RUN_DEPENDENCY',
                             'receipt': str(validation_path.relative_to(DOC)),
                             'dependency': None if validation and validation['status'] == 'PASS' else 'final tests, bindings, independent review and compiled-paper verification'},
              'runtime': {'status': 'DONE' if runtime and runtime_check and runtime_check['status'] == 'PASS' else 'NOT_RUN_DEPENDENCY',
                          'dependency': None if runtime_check else 'matched final prediction parity verification'},
              'A_deployment_runtime': {'status': 'DONE' if runtime_a and runtime_a_check and runtime_a_check['status'] == 'PASS' else 'NOT_RUN_DEPENDENCY',
                                       'dependency': None if runtime_a_check else 'final locked A checkpoints, idle GPU, same 26-frame measurements and deployed-versus-cached J verification'},
              'reuse': {'status': 'REUSED_VALIDATED' if (DOC / 'REUSED_EVIDENCE_VERIFICATION.json').exists() else 'NOT_RUN_DEPENDENCY'},
              'original_checkout_preservation': {'status': 'DONE', 'checks': checks},
              'git': {'status': 'NOT_RUN_DEPENDENCY', 'dependency': 'final validation, explicit-path commit and normal push; closure receipt outside repository',
                      'branch': git(ROOT, 'branch', '--show-current'), 'remote': git(ROOT, 'remote', 'get-url', 'origin'),
                      'receipt': str(ROOT.parent / 'pallet-joint-action-push-20261006.json')},
              'scientific_completion': 'Independent physical reference and unseen independent confirmation require actual data; executable closure does not imply those claims are complete'}
    write(DOC / 'status.json', status)
    protocol = {'schema': 'joint_action_handoff_protocol_v1', 'base_commit': initial['worktree']['head'],
                'plan': 'EXECUTION_PLAN_KO.md', 'A': read(DOC / 'A_protocol.json') if (DOC / 'A_protocol.json').exists() else None,
                'B': 'measurement_packet/SCHEMA_KO.md', 'C': 'C_report_KO.md',
                'maximum_new_formal_fits': 6, 'maximum_formal_updates': 36000, 'maximum_smoke_updates': 4,
                'real_training_images': 0, 'new_hyperparameter_search': False,
                'new_schemas': 'Defined in component modules/documents; they are not assumed historical columns'}
    write(DOC / 'protocol.json', protocol)
    manifest = {'schema': 'joint_action_handoff_manifest_v1', 'initial_state': initial,
                'component_manifests': {name: read(DOC / f'{name}_manifest.json') if (DOC / f'{name}_manifest.json').exists() else None for name in ('A', 'B', 'C')},
                'matched_runtime': runtime, 'runtime_verification': runtime_check,
                'A_deployment_runtime': runtime_a, 'A_runtime_verification': runtime_a_check,
                'additional_input_dependencies': read(DOC / 'INPUT_DEPENDENCIES.json') if (DOC / 'INPUT_DEPENDENCIES.json').exists() else None,
                'source_cache_full_content_hashes': read(DOC / 'SOURCE_CACHE_HASHES.json') if (DOC / 'SOURCE_CACHE_HASHES.json').exists() else None,
                'commands': {'status': 'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage status',
                             'reuse': 'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage reuse',
                             'runtime': 'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.runtime --source-root PATH',
                             'runtime_A': 'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.runtime_a --source-root PATH --cache-dir PATH',
                             'verify_runtime_A': 'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.verify_runtime_a --source-root PATH',
                             'verify': 'python -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage verify'},
                'receipt_scope': 'Original raw/cache/checkpoints remain external/ignored; published small rows and hashes support offline verification',
                'files': file_inventory()}
    write(DOC / 'run_manifest.json', manifest)
    return status


def verify():
    manifest = read(DOC / 'run_manifest.json')
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file() or sha(path) != entry['sha256']:
            failures.append(entry['path'])
    if failures:
        raise ValueError('Deliverable binding drift: ' + repr(failures))
    return {'status': 'PASS', 'file_bindings_verified': len(manifest['files']),
            'original_checkout_checks': preservation()}


def status():
    value = read(DOC / 'status.json')
    receipt_path = Path(value['git']['receipt'])
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get('local_commit_sha') == git(ROOT, 'rev-parse', 'HEAD') and receipt.get('remote_commit_sha') == receipt.get('local_commit_sha'):
            value['git'] = {**receipt, 'status': 'DONE', 'receipt': str(receipt_path)}
    return value


def resume(args):
    if args.source_root is None or args.cache_dir is None:
        raise ValueError('resume requires --source-root and --cache-dir')
    reuse()
    from .source_hashes import generate
    generate(args.source_root, DOC / 'SOURCE_CACHE_HASHES.json')
    from .input_dependencies import generate as bind_inputs
    bind_inputs(args.source_root)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.a_experiment',
                    '--source-root', str(args.source_root), '--cache-dir', str(args.cache_dir),
                    '--stage', 'all', '--workers', str(args.workers)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.a_reporting'],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.mask_receipt',
                    '--source-root', str(args.source_root), '--cache-dir', str(args.cache_dir)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.runtime',
                    '--source-root', str(args.source_root)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.verify_runtime',
                    '--source-root', str(args.source_root)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.runtime_a',
                    '--source-root', str(args.source_root), '--cache-dir', str(args.cache_dir)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.verify_runtime_a',
                    '--source-root', str(args.source_root)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.subgroups',
                    '--source-root', str(args.source_root)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.research.pallet_joint_action_handoff_20261006_v1.execution_counts'],
                   cwd=ROOT, check=True)
    # Acquisition/evaluation needs real paired measurements and is deliberately
    # a separate explicit measurement CLI, never an implicit fixture substitute.
    if args.a_snippet:
        subprocess.run([sys.executable, str(CODE / 'paper.py'), 'integrate-a', '--a-snippet', str(args.a_snippet)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(CODE / 'paper.py'), 'validate'], cwd=ROOT, check=True)
    if args.tex_engine:
        subprocess.run([sys.executable, str(CODE / 'paper.py'), 'build', '--engine', args.tex_engine], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(CODE / 'paper.py'), 'finalize'], cwd=ROOT, check=True)
    value = assemble()
    verify()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('reuse', 'assemble', 'verify', 'status', 'resume'), default='status')
    parser.add_argument('--source-root', type=Path, help='Read-only original data/cache/checkpoint checkout; required for resume')
    parser.add_argument('--cache-dir', type=Path, help='External ignored A bank/checkpoint directory; required for resume')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--tex-engine', help='Optional actual Tectonic executable for a fresh manuscript build')
    parser.add_argument('--a-snippet', type=Path, help='Optional evidence-reviewed final A LaTeX summary; omitted on unchanged resume')
    args = parser.parse_args()
    result = resume(args) if args.stage == 'resume' else {'reuse': reuse, 'assemble': assemble, 'verify': verify, 'status': status}[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
