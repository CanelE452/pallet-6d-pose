"""Bind pilot constants and implemented training/geometry sources before real results."""
import argparse
from pathlib import Path

from scripts.research.pallet_dht_coupling_v2.prepare import read, write, sha, check


def verify(run_dir):
    root = Path(run_dir)
    binding = read(root / 'SOURCE_FREEZE.json')
    check(binding['protocol_sha256'] == sha(root / 'TRAIN_PROTOCOL.json'), 'Pilot protocol changed')
    for path, digest in binding['source_sha256'].items():
        check(sha(path) == digest, f'Bound source changed: {path}')
    for path, digest in binding['input_sha256'].items():
        check(sha(path) == digest, f'Bound source input changed: {path}')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    check((root / 'PURPOSE.md').is_file(), 'Purpose missing')
    check(not (root / 'SOURCE_FREEZE.json').exists(), 'Existing freeze must not be overwritten')
    check(not (root / 'CACHE_COMPLETION.json').exists() and not (root / 'DIAGNOSTIC.json').exists(),
          'Freeze must precede main results')
    protocol = read(root / 'TRAIN_PROTOCOL.json')
    preflight = read(root / 'PREFLIGHT.json')
    check(preflight['complete'] and preflight['PASS'], 'Preflight incomplete')
    src = Path(__file__).resolve().parent
    files = [src / f'{name}.py' for name in ('geometry', 'model', 'cache', 'train', 'diagnostic')]
    sources = {str(p): sha(p) for p in files}
    old = root.parent / 'pallet_dht_joint_v1'
    old_protocol = read(old / 'TRAIN_PROTOCOL.json')
    for path, digest in old_protocol['source_code_sha256'].items():
        check(sha(path) == digest, f'Existing backbone implementation changed: {path}')
        sources[path] = digest
    inputs = {
        protocol['backbone']['checkpoint']: protocol['backbone']['sha256'],
        protocol['data']['source_manifest']: protocol['data']['source_manifest_sha256'],
        protocol['data']['real_manifest']: protocol['data']['real_manifest_sha256'],
    }
    for path, digest in inputs.items():
        check(sha(path) == digest, f'Input differs: {path}')
    write(root / 'SOURCE_FREEZE.json', dict(complete=True, PASS=True,
        protocol_sha256=sha(root / 'TRAIN_PROTOCOL.json'),
        preflight_sha256=sha(root / 'PREFLIGHT.json'), source_sha256=sources, input_sha256=inputs,
        note='Real candidate diagnosis and main cache/training have not run at this binding.'))
    verify(root)
    print('Pilot protocol and implemented geometry/training sources frozen', flush=True)


if __name__ == '__main__':
    main()
