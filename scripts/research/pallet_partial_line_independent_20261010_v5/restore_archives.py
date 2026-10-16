"""Restore byte-identical large gzip evidence from ordered, hash-bound parts.

No models, ground truth, numerical solving, or dependencies outside Python.
Completed files are verified and left untouched. New files are created once.
"""
import argparse
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[3] / '_docs/experiments/pallet_partial_line_independent_20261010_v5'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(folder, name):
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts and name not in ('', '.'), name
    path = folder / relative
    assert not any(p.is_symlink() for p in (path, *path.parents)), name
    assert path.resolve().is_relative_to(folder.resolve()), name
    return path


def restore(folder, output):
    manifest = json.loads((folder / 'ARCHIVE_MANIFEST.json').read_text())
    assert manifest['encoding'] == 'ordered byte slices of original gzip; concatenate without recompression'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for item in manifest['files']:
        name = item['original']['path']
        assert Path(name).name == name, 'original basename required'
        target = safe(output, name)
        chunks = [safe(folder, p['path']) for p in item['parts']]
        total = 0
        for path, part in zip(chunks, item['parts']):
            assert path.is_file() and path.stat().st_size == part['bytes'] and digest(path) == part['sha256'], str(path)
            total += part['bytes']
        assert total == item['original']['bytes'], name
        existing = target.exists()
        if not existing:
            with target.open('xb') as stream:
                for path in chunks:
                    with path.open('rb') as part:
                        for block in iter(lambda: part.read(1024 * 1024), b''):
                            stream.write(block)
        assert target.stat().st_size == total and digest(target) == item['original']['sha256'], name
        results.append(dict(path=name, bytes=total, sha256=item['original']['sha256'], verified=True, existing_unchanged=existing))
    return dict(passed=True, files=results, model_PnP_training_RGB_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=DOC)
    p.add_argument('--output', type=Path, help='default: restore into input evidence directory')
    p.add_argument('--receipt', type=Path, help='optional new JSON verification receipt')
    args = p.parse_args()
    result = restore(args.input, args.output or args.input)
    if args.receipt:
        assert not args.receipt.is_symlink()
        with args.receipt.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
