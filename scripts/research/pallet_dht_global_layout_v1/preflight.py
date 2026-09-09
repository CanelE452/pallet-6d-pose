"""Verify immutable cache/data lineage before the global-layout pilot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def main(run_dir):
    run_dir = run_dir.resolve()
    assert (run_dir / 'PURPOSE.md').is_file()
    assert not (run_dir / 'CALIBRATION_SELECTION.json').exists(), 'Preflight must precede selection'
    old = REPO / 'data/pallet/results/pallet_dht_decoder_probe_v1'
    audit = REPO / 'data/pallet/results/pallet_dht_gt_audit_v1'
    expected = {}
    def register(path, digest):
        path = str(Path(path).resolve())
        assert path not in expected or expected[path] == digest, f'Conflicting digest {path}'
        expected[path] = digest
    for name, mapping in [('CACHE_COMPLETION.json', 'array_sha256'),
                          ('CACHE_COMPLETION.json', 'candidate_evidence_sha256'),
                          ('EXTRACTION_COMPLETE.json', 'frame_sha256')]:
        for path, digest in read(old / name)[mapping].items():
            register(path, digest)
    for path, digest in read(audit / 'INPUT_SNAPSHOT.json')['sha256'].items():
        register(path, digest)
    for folder, names in [(old, ['COMPLETION.json', 'MANIFEST.json', 'CACHE_MANIFEST.json',
                                'CACHE_RECORDS.json', 'CACHE_COMPLETION.json',
                                'EXTRACTION_COMPLETE.json', 'PREDICTIONS.json', 'RESULTS.json']),
                          (audit, ['COMPLETION.json', 'INPUT_SNAPSHOT.json',
                                   'ROOT_GT_VISUAL_REVIEW.json', 'specialist_gt_review.json',
                                   'GT_REVIEW_QUEUE.json', 'NUMERIC_AUDIT.json'])]:
        for name in names:
            register(folder / name, sha(folder / name))
    for folder in [old, audit]:
        completion = read(folder / 'COMPLETION.json')
        for name, digest in completion.get('artifact_sha256', {}).items():
            register(folder / name, digest)
    checkpoint = REPO / 'data/pallet/results/pallet_dht_joint_v1/runs/hough_joint_seed1/weights/final.pt'
    register(checkpoint, '0960fb32fd99fd0837a588792e07727298fc6f88f1e4ec3464efd69bd5574d37')
    manifest = REPO / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
    register(manifest, 'f87a2d1f4bfc7883bbd0a5562f43ad13711e5ae9aa6586c9a894ec64106437c8')
    start = time.monotonic()
    total_bytes = 0
    for index, (path, digest) in enumerate(sorted(expected.items())):
        assert sha(path) == digest, f'Immutable input changed: {path}'
        total_bytes += Path(path).stat().st_size
        if index % 1000 == 0:
            print(f'verified {index + 1}/{len(expected)} inputs', flush=True)
    write(run_dir / 'INPUT_SNAPSHOT.json', dict(
        complete=True, PASS=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(run_dir / 'PROTOCOL.json'),
        sha256=expected, n_inputs=len(expected), bytes_verified=total_bytes,
        elapsed_seconds=time.monotonic() - start,
        old_cache_read_only=True, new_cnn_forwards=0, new_training_updates=0,
        real_gt_used_for_coefficient_selection=False))
    print(f'PASS: {len(expected)} immutable inputs verified', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=REPO / 'data/pallet/results/pallet_dht_global_layout_v1')
    main(p.parse_args().run_dir)
