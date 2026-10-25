"""Build a sealed, explicit V239–V240 SOURCE feature/head completion and failed admission checkpoint."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')


def safe(p):
    p = Path(p); text = str(p).lower()
    assert not any(s in text for s in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation'))
    assert not p.is_symlink()
    return p


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read_bound(b):
    data = safe(b['path']).read_bytes()
    assert sha(data) == b['sha256']
    if 'size_bytes' in b:
        assert len(data) == b['size_bytes']
    return data


def write_json(p, d):
    with safe(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')


def inventory(p, exclusions):
    p = safe(p)
    for name in sorted(os.listdir(p)):
        if 'v112' in name.lower() or name == '__pycache__' or name.lower().endswith('.pyc'):
            continue
        candidate = p / name
        rel = candidate.relative_to(R).as_posix()
        # Large cache exclusions are applied before any per-entry stat/read.
        if rel in exclusions:
            continue
        candidate = safe(candidate)
        if candidate.is_dir():
            yield from inventory(candidate, exclusions)
        else:
            yield candidate


def package_inventory(p):
    for name in sorted(os.listdir(safe(p))):
        q = safe(p / name)
        if q.is_dir():
            yield from package_inventory(q)
        else:
            yield q


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_bytes = safe(args.plan).read_bytes(); plan = json.loads(plan_bytes)
    assert plan['approved_publication'] is True and plan['source_only'] is True
    assert plan['versions'] == ['V239', 'V240']
    # Actual completion, not a planned training result, is required before copying.
    for b in plan['required_completed_receipts']:
        d = json.loads(read_bound(b))
        assert d['completed'] is True and d['PASS'] is True
    out = safe(R / 'V240_PUBLICATION_PACKAGE_V2'); assert not out.exists(); out.mkdir()
    exclusions = set(plan['exclude_relative_paths'])
    sources = []
    for name in plan['top_directories']:
        assert len(Path(name).parts) == 1
        sources.extend(inventory(R / name, exclusions))
    source_map = []
    for p in sources:
        rel = p.relative_to(R).as_posix(); data = p.read_bytes()
        assert len(data) < 95_000_000, 'Unapproved large cache: ' + rel
        compressed = p.suffix == '.jsonl' and len(data) > 900_000
        storage = 'artifacts/' + rel + ('.gz' if compressed else '')
        target = safe(out / storage); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if compressed else data)
        source_map.append({'path': storage, 'source_relative_to_preparation': rel,
                           'source_sha256': sha(data), 'source_bytes': len(data),
                           'encoding': 'gzip-lossless' if compressed else 'identity'})
    for item in plan['extra_files']:
        rel = item['storage']; assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        data = read_bound(item['source']); target = safe(out / rel)
        target.parent.mkdir(parents=True, exist_ok=True); assert not target.exists(); target.write_bytes(data)
        source_map.append({'path': rel, 'source_binding': item['source'], 'source_sha256': sha(data),
                           'source_bytes': len(data), 'encoding': 'identity'})
    verifier = R / 'V233_PUBLICATION_VERIFY_A/verify_package.py'
    vb = verifier.read_bytes(); assert sha(vb) == 'bd34cada69b56591171a3ee0686bb4a6eb8c16c3bdf8ebfe994db13b7dac30bc'
    (out/'verify_package.py').write_bytes(vb)
    (out/'build_v240_publication_v2.py').write_bytes(safe(__file__).read_bytes())
    (out/'PUBLICATION_PLAN.json').write_bytes(plan_bytes)
    (out/'README_KO.md').write_bytes(read_bound(plan['readme']))
    (out/'THIRD_PARTY_NOTICES.md').write_bytes(read_bound(plan['third_party_notice']))
    write_json(out/'PUBLICATION_SOURCE_MAP.json', {'source_root': str(R), 'files': source_map,
        'large_cache_exclusions': sorted(exclusions), 'note': 'Only explicit SOURCE experiment folders and files. Large raw RGB/valid and failed intermediate feature cache excluded; original SHA and builders retained. New current/held data zero.'})
    files = []
    for p in package_inventory(out):
        b = p.read_bytes(); files.append({'path': p.relative_to(out).as_posix(), 'sha256': sha(b), 'bytes': len(b)})
    manifest = {'schema': 'V240_SOURCE_CONTEXT_TRAINED_PUBLICATION_V1', 'files': files, 'file_count': len(files),
                'total_bytes': sum(f['bytes'] for f in files), 'scope': plan['scientific_scope'],
                'source_gate_pass': plan['source_gate_pass'], 'current_admission': False,
                'new_current_or_held_payload': 0, 'publication_is_not_model_success': True}
    write_json(out/'MANIFEST.json', manifest)
    print(json.dumps({'completed': True, 'package': str(out), 'files': len(files),
                      'bytes': manifest['total_bytes'], 'manifest_sha256': sha((out/'MANIFEST.json').read_bytes())}))


if __name__ == '__main__':
    main()
