"""Content-bind external frozen feature arrays without copying them.

The manifest binds all arrays, including the two 73.7GB feature files. A file
whose size/mtime changes while read is rejected. Resume reuses the receipt if
the explicit source path and stat bindings are unchanged; --rehash requests
full byte verification rather than claiming metadata proves byte equality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from .runtime import DOC, write


def generate(source_root, output, rehash=False):
    source = Path(source_root).resolve()
    directory = source / 'data/pallet/results/pallet_line_pose_v1/cache'
    manifest = directory / 'CACHE_MANIFEST.json'
    entries = json.loads(manifest.read_text())['arrays']
    old = json.loads(output.read_text()) if output.exists() else None
    if old and not rehash:
        if old['source_root'] != str(source):
            raise ValueError('Source path changed; --rehash is required')
        for binding in old['arrays']:
            stat = (source / binding['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (binding['bytes'], binding['mtime_ns']):
                raise ValueError('Source array stat changed; inspect then explicitly --rehash')
        return old
    started = time.perf_counter()
    arrays = []
    for key, spec in entries.items():
        path = directory / spec['file']
        before = path.stat()
        digest = hashlib.sha256()
        tick = time.perf_counter()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during hashing: ' + str(path))
        arrays.append({'name': key, 'path': str(path.relative_to(source)), 'sha256': digest.hexdigest(),
                       'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
                       'seconds': time.perf_counter() - tick})
        print('SOURCE_HASH', key, after.st_size, round(arrays[-1]['seconds'], 3), flush=True)
    result = {'schema': 'joint_action_source_array_content_bindings_v1', 'status': 'DONE',
              'source_root': str(source), 'arrays': arrays,
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'total_bytes_hashed': sum(r['bytes'] for r in arrays), 'elapsed_seconds': time.perf_counter() - started,
              'resume_scope': 'stored full byte SHA plus unchanged file stat; --rehash for current byte equality',
              'copies': 0, 'optimizer_updates': 0}
    if old and {r['name']: r['sha256'] for r in old['arrays']} != {r['name']: r['sha256'] for r in arrays}:
        raise ValueError('Original frozen array content changed; do not replace completed binding')
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DOC / 'SOURCE_CACHE_HASHES.json')
    parser.add_argument('--rehash', action='store_true')
    args = parser.parse_args()
    result = generate(args.source_root, args.output, args.rehash)
    print('CONTENT_BOUND', len(result['arrays']), result['total_bytes_hashed'])


if __name__ == '__main__':
    main()
