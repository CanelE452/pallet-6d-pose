"""Check public runtime-readiness file bindings; no benchmark or model imports.

This checks publication integrity, not successful future timing or private
checkpoint provenance. Actual startup guards are recorded by the separate
readiness controls and independent code review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

DOC = '_docs/experiments/pallet_kp_repair_runtime_20261010_v1'


def verify(root):
    read_files = set()
    def read(name):
        p = (root / name).resolve()
        assert not Path(name).is_absolute() and p.is_relative_to(root)
        read_files.add(name)
        return json.loads(p.read_text())
    def bindings(items):
        assert len(items) == len({v['path'] for v in items})
        for v in items:
            p = (root / v['path']).resolve()
            assert not Path(v['path']).is_absolute() and p.is_relative_to(root)
            data = p.read_bytes()
            assert len(data) == v['bytes'] and hashlib.sha256(data).hexdigest() == v['sha256'], v['path']
            read_files.add(v['path'])
    prior = read(DOC + '/PRIOR_PUBLICATION_BINDINGS.json')
    assert len(prior['files']) == prior['protected_prior_files'] == 311
    bindings(prior['files'])
    manifest = read(DOC + '/REVIEW_MANIFEST.json')
    assert all(Path(v['path']).name not in ('REVIEW_MANIFEST.json', 'PUBLIC_REVIEW.json', 'PUBLICATION.json')
               for v in manifest['files'])
    bindings(manifest['files'])
    return dict(schema='runtime_readiness_public_binding_review_v1', passed=True,
                protected_prior_files=311, current_manifest_files=len(manifest['files']),
                manifest_sha256=hashlib.sha256((root / DOC / 'REVIEW_MANIFEST.json').read_bytes()).hexdigest(),
                public_files_read=sorted(read_files),
                execution=dict(model_imports=0, detector_forwards=0, head_forwards=0,
                               training_updates=0, PnP_calls=0, ray_calls=0, timed_pipeline_calls=0),
                limits=['Checks public file integrity and prior publication preservation.',
                        'Does not reproduce startup controls, private checkpoints, model execution or measured latency.',
                        'Corrected-supervision learning and real pose improvement remain unexecuted.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    own = root / DOC / 'PUBLIC_REVIEW.json'
    if output.exists() or output.suffix != '.json' or (output.is_relative_to(root) and output != own):
        p.error('Use a fresh JSON output; protected files cannot be overwritten')
    try:
        result = verify(root)
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e), file=sys.stderr)
        return 1
    result['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'protected_prior_files', 'current_manifest_files')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
