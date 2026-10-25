"""Verify a portable V256 package and restore exact original artifact bytes.

Only package-relative manifest bindings are read. No original_path fallback,
NumPy, Torch, array parsing, model loading, network, or Git operations.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

CHUNK = 1024*1024


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def lexical(s):
    need(not any(t in str(s).casefold() for t in ('v112', 'cache', '.pyc', 'current223',
         'publicheld', 'oldval', 'holdout', 'private_evaluation')), 'Forbidden path before FS')


def relative(s):
    lexical(s)
    need(isinstance(s, str) and '\\' not in s, 'POSIX relative path')
    p = PurePosixPath(s)
    need(not p.is_absolute() and p.parts and '..' not in p.parts and p.as_posix() == s,
         'Canonical relative path without traversal')
    return Path(*p.parts)


def safe(path, exists=True):
    lexical(path)
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute artifact root')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Missing artifact')
            continue
        need(not stat.S_ISLNK(mode), 'No symlinks')
    return p


def digest_file(p):
    h, size = hashlib.sha256(), 0
    with safe(p).open('rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class Parts(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.i, self.current = list(paths), 0, None

    def readable(self):
        return True

    def readinto(self, buffer):
        view, total = memoryview(buffer), 0
        while total < len(view):
            if self.current is None:
                if self.i == len(self.paths):
                    break
                self.current = safe(self.paths[self.i]).open('rb')
                self.i += 1
            n = self.current.readinto(view[total:])
            if n:
                total += n
            else:
                self.current.close()
                self.current = None
        return total

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def restore(package, output=None, verify_only=False):
    package = safe(package)
    raw = safe(package/'MANIFEST.json').read_bytes()
    m = json.loads(raw)
    need(m.get('completed') is True and m.get('PASS') is True and
         m['schema'] == 'V256_PORTABLE_ALL_ARTIFACT_BYTES_V1', 'Sealed package manifest')
    files = {}
    for row in m['files']:
        rel = row['relative_path']
        p = safe(package/relative(rel))
        need(rel not in files and rel != 'MANIFEST.json', 'Unique physical file, manifest excluded')
        sha, size = digest_file(p)
        need(sha == row['sha256'] and size == row['size_bytes'], 'Physical file SHA/size')
        files[rel] = p
    result = {'completed': True, 'PASS': True,
              'package_manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'physical_files_verified': len(files), 'original_absolute_path_reads': 0,
              'array_or_model_decode': 0, 'restored_artifacts': []}
    logical, used = set(), set()
    for artifact in m['artifacts']:
        rel = artifact['relative_path']
        relative(rel)
        need(rel not in logical, 'Unique logical restored path')
        logical.add(rel)
        need(artifact['encoding'] in ('raw', 'gzip'), 'Raw or lossless gzip only')
        parts = artifact['transport_files']
        need(parts and len(set(parts)) == len(parts) and all(p in files for p in parts), 'Closed ordered transport parts')
        need(not used.intersection(parts), 'No transport alias between artifacts')
        used.update(parts)
    need(used == set(files), 'Every declared physical file belongs to one restored artifact')
    if verify_only:
        return result
    output = safe(output, False)
    need(not output.exists(), 'Fresh restore directory, no overwrite or retry')
    output.mkdir()
    try:
        for artifact in m['artifacts']:
            target = safe(output/relative(artifact['relative_path']), False)
            target.parent.mkdir(parents=True, exist_ok=True)
            h, count = hashlib.sha256(), 0
            with Parts([files[p] for p in artifact['transport_files']]) as parts:
                stream = gzip.GzipFile(fileobj=io.BufferedReader(parts), mode='rb') if artifact['encoding'] == 'gzip' else parts
                with target.open('xb') as f:
                    for chunk in iter(lambda: stream.read(CHUNK), b''):
                        h.update(chunk)
                        count += len(chunk)
                        f.write(chunk)
                if stream is not parts:
                    stream.close()
            need(h.hexdigest() == artifact['original_sha256'] and count == artifact['original_size_bytes'],
                 'Exact restored full artifact bytes including original NPY header')
            result['restored_artifacts'].append({'relative_path': artifact['relative_path'],
                                                  'sha256': h.hexdigest(), 'size_bytes': count})
        receipt = safe(output/'RESTORE_QA.json', False)
        need(not receipt.exists(), 'Restore receipt name reserved')
        receipt.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
        return result
    except BaseException as exc:
        safe(output/'RESTORE_FAILURE.json', False).write_text(json.dumps({
            'completed': False, 'PASS': False, 'error': str(exc),
            'partial_restored_artifacts': result['restored_artifacts'], 'automatic_retry': False},
            ensure_ascii=False, indent=2)+'\n')
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    need(a.verify_only != (a.output_dir is not None), 'Choose --verify-only or fresh --output-dir')
    print(json.dumps(restore(a.package, a.output_dir, a.verify_only), ensure_ascii=False, indent=2))
