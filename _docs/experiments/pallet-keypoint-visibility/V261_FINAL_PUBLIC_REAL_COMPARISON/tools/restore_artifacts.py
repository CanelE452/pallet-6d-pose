"""Exact allowlist byte packaging and portable restoration; no payload decoding.

Derived from the frozen V256 build_portable_package.py / restore_artifacts.py.
Small leaves remain byte-identical. Only originals above90MiB use deterministic
gzip, streamed into ordered parts of at most64MiB. Absolute provenance is inert.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

CHUNK = 1024*1024
LARGE_BYTES = 90*1024*1024
PART_BYTES = 64*1024*1024
SCHEMA = 'V261_FINAL_PUBLIC_REAL_COMPARISON_OPAQUE_BYTES_V1'
R = Path(__file__).absolute().parent.parent
PACKAGE = R/'V261_FINAL_PUBLICATION_PACKAGE'
RESTORE_TOOL = 'tools/restore_artifacts.py'
RESERVED = {'MANIFEST.json', 'FAILURE.json', 'FULL_RESTORE_QA.json',
            'RESTORE_FAILURE.json', RESTORE_TOOL}


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def lexical(s):
    need(not any(t in str(s).casefold() for t in ('v112', 'cache', '.pyc')),
         'Forbidden path before FS')


def relative(s, reserved=False):
    lexical(s)
    need(type(s) is str and s and '\\' not in s and '\0' not in s,
         'Canonical POSIX relative path')
    p = PurePosixPath(s)
    need(not p.is_absolute() and p.parts and '..' not in p.parts and
         p.as_posix() == s and s != '.', 'No relative traversal or aliases')
    if not reserved:
        need(s not in RESERVED and p.parts[0] != 'transport_gzip', 'Reserved package namespace')
    return Path(*p.parts)


def safe(path, exists=True):
    lexical(path)
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute artifact path')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact artifact absent')
            continue
        need(not stat.S_ISLNK(mode), 'No artifact ancestor or leaf symlink')
    return p


def open_read(path):
    return os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb')


def measure(path):
    h, n = hashlib.sha256(), 0
    with open_read(path) as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
            n += len(chunk)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': n}


def validate_binding(b):
    need(isinstance(b, dict) and {'path', 'sha256', 'size_bytes'} <= set(b), 'Explicit bound leaf')
    need(type(b['size_bytes']) is int and b['size_bytes'] >= 0 and
         type(b['sha256']) is str and len(b['sha256']) == 64 and
         all(c in '0123456789abcdef' for c in b['sha256']), 'SHA256/size binding')
    return safe(b['path'])


def read_bound(b):
    validate_binding(b)
    with open_read(b['path']) as f:
        data = f.read()
    need(len(data) == b['size_bytes'] and hashlib.sha256(data).hexdigest() == b['sha256'],
         'Exact bound metadata or code bytes')
    return data


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False)+'\n').encode('utf-8')
    p = safe(path, False)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)
    return measure(p)


def physical(path, package):
    b = measure(path)
    return {'relative_path': path.relative_to(package).as_posix(),
            'sha256': b['sha256'], 'size_bytes': b['size_bytes']}


class Parts(io.RawIOBase):
    """Ordered declared byte parts only; no original-source fallback."""
    def __init__(self, paths):
        self.paths, self.index, self.current = list(paths), 0, None

    def readable(self):
        return True

    def readinto(self, buffer):
        view, total = memoryview(buffer), 0
        while total < len(view):
            if self.current is None:
                if self.index == len(self.paths):
                    break
                self.current = open_read(self.paths[self.index])
                self.index += 1
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


class Shards:
    """Streaming gzip sink; production chunk maximum is fixed64MiB."""
    def __init__(self, stem, package, part_bytes):
        self.stem, self.package, self.part_bytes = stem, package, part_bytes
        self.current, self.current_bytes, self.rows = None, 0, []
        self.sha, self.total = hashlib.sha256(), 0
        self.closed = False

    def finish_part(self):
        if self.current is not None:
            path = Path(self.current.name)
            self.current.close()
            self.rows.append(physical(path, self.package))
            self.current = None

    def write(self, data):
        need(not self.closed, 'Compressed sink already closed')
        view = memoryview(data)
        n = len(view)
        while view:
            if self.current is None:
                path = safe(Path(str(self.stem)+'.part%04d' % len(self.rows)), False)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.current, self.current_bytes = path.open('xb'), 0
            take = min(len(view), self.part_bytes-self.current_bytes)
            block = view[:take]
            self.current.write(block)
            self.sha.update(block)
            self.total += take
            self.current_bytes += take
            view = view[take:]
            if self.current_bytes == self.part_bytes:
                self.finish_part()
        return n

    def flush(self):
        if self.current is not None:
            self.current.flush()

    def close(self):
        self.finish_part()
        self.closed = True


def stream_original(package, artifact, sink=None):
    h, n = hashlib.sha256(), 0
    paths = [safe(package/relative(p, True)) for p in artifact['transport_files']]
    with Parts(paths) as parts:
        stream = (gzip.GzipFile(fileobj=io.BufferedReader(parts), mode='rb')
                  if artifact['encoding'] == 'gzip_parts' else parts)
        try:
            for chunk in iter(lambda: stream.read(CHUNK), b''):
                n += len(chunk)
                need(n <= artifact['original_size_bytes'], 'Restored bytes exceed exact original size')
                h.update(chunk)
                if sink is not None:
                    sink.write(chunk)
        finally:
            if stream is not parts:
                stream.close()
    need(n == artifact['original_size_bytes'] and h.hexdigest() == artifact['original_sha256'],
         'Original full bytes SHA256/size exact after transport')
    return h.hexdigest(), n


def transfer(source, rel, out, large_bytes=LARGE_BYTES, part_bytes=PART_BYTES):
    """Opaque copy/gzip only. Nonproduction limits are used only by invented tests."""
    validate_binding(source)
    relative(rel)
    large = source['size_bytes'] > large_bytes
    h, n = hashlib.sha256(), 0
    sink = stream = None
    try:
        if large:
            stem = safe(out/'transport_gzip'/relative(rel), False)
            stem = Path(str(stem)+'.gz')
            sink = Shards(stem, out, part_bytes)
            stream = gzip.GzipFile(filename='', fileobj=sink, mode='wb', mtime=0, compresslevel=6)
        else:
            target = safe(out/relative(rel), False)
            target.parent.mkdir(parents=True, exist_ok=True)
            sink = stream = target.open('xb')
        with open_read(source['path']) as src:
            for chunk in iter(lambda: src.read(CHUNK), b''):
                n += len(chunk)
                need(n <= source['size_bytes'], 'Source exceeds declared size')
                h.update(chunk)
                stream.write(chunk)
        stream.close()
        if large:
            sink.close()
        need(n == source['size_bytes'] and h.hexdigest() == source['sha256'], 'Exact source bytes')
        files = sink.rows if large else [physical(target, out)]
        need(bool(files), 'At least one declared transport leaf')
        need(all(r['size_bytes'] <= part_bytes for r in files) if large
             else files[0]['size_bytes'] <= large_bytes, 'Fixed transport leaf limit')
        artifact = {'relative_path': rel, 'group': None,
                    'encoding': 'gzip_parts' if large else 'identity',
                    'original_sha256': source['sha256'], 'original_size_bytes': n,
                    'transport_files': [r['relative_path'] for r in files],
                    'compressed_stream_sha256': sink.sha.hexdigest() if large else None,
                    'compressed_stream_bytes': sink.total if large else None,
                    'original_source_binding_inert': source}
        if large:
            stream_original(out, artifact)
        return artifact, files
    finally:
        try:
            if stream is not None:
                stream.close()
        finally:
            if sink is not None:
                sink.close()


def build(entry_path):
    entry_raw = open_read(entry_path)
    with entry_raw as f:
        raw = f.read()
    e = json.loads(raw)
    need(e.get('approved_actual_publication_package') is True and
         e.get('array_or_model_decode') is False and e.get('automatic_retry') is False and
         e.get('final_one_comparison') is True, 'Root explicit opaque final-publication entry')
    need(e['code'] == measure(Path(__file__).absolute()), 'Exact publication byte code')
    scope = json.loads(read_bound(e['root_scope']))
    need(scope.get('approved_actual_publication_package') is True and
         scope.get('entry_contract') == {k: v for k, v in e.items() if k != 'root_scope'},
         'Root exact entry contract')
    need(e['output_directory'] == str(PACKAGE), 'Fixed V261 package destination')
    rows = e['files']
    need(type(rows) is list and bool(rows), 'Root exact nonempty source allowlist')
    names = [r['relative_path'] for r in rows]
    need(len(names) == len(set(names)) and 'README_KO.md' in names, 'Unique logical paths and public readme')
    for row in rows:
        relative(row['relative_path'])
        validate_binding(row['source'])
        need(not Path(row['source']['path']).is_relative_to(PACKAGE), 'No package-to-itself source')
    groups = Counter(r['group'] for r in rows)
    need(set(groups) == set(e['coverage_contract']), 'All Root groups covered')
    canonical = lambda values: sorted(json.dumps(v, sort_keys=True) for v in values)
    for group, expected in e['coverage_contract'].items():
        need(canonical([r['source'] for r in rows if r['group'] == group]) == canonical(expected),
             'Exact declared group members; no outcome or basename filtering')
    need(e['large_original_threshold_bytes'] == LARGE_BYTES and e['gzip_part_bytes'] == PART_BYTES,
         'Fixed90MiB threshold and64MiB maximum parts')
    out = safe(e['output_directory'], False)
    need(not out.exists(), 'Fresh package, preserve old failures')
    out.mkdir()
    artifacts, files = [], []
    try:
        for index, row in enumerate(rows, 1):
            a, physical_rows = transfer(row['source'], row['relative_path'], out)
            a['group'] = row['group']
            artifacts.append(a)
            files.extend(physical_rows)
            if index % 100 == 0:
                print(json.dumps({'packaged': index, 'logical_files': len(rows)}), flush=True)
        # A self-contained byte restorer, with no old-path or helper dependency.
        source = measure(Path(__file__).absolute())
        target = safe(out/relative(RESTORE_TOOL, True), False)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open_read(source['path']) as src, target.open('xb') as dst:
            dst.write(src.read())
        b = measure(target)
        need(b['sha256'] == source['sha256'] and b['size_bytes'] == source['size_bytes'], 'Portable code identity')
        artifacts.append({'relative_path': RESTORE_TOOL, 'group': 'portable_restorer', 'encoding': 'identity',
                          'original_sha256': b['sha256'], 'original_size_bytes': b['size_bytes'],
                          'transport_files': [RESTORE_TOOL], 'compressed_stream_sha256': None,
                          'compressed_stream_bytes': None, 'original_source_binding_inert': source})
        files.append(physical(target, out))
        need(len({r['relative_path'] for r in files}) == len(files), 'No physical transport collision')
        m = {'schema': SCHEMA, 'completed': True, 'PASS': True,
             'code': e['code'], 'entry_sha256': hashlib.sha256(raw).hexdigest(),
             'files': sorted(files, key=lambda r: r['relative_path']), 'artifacts': artifacts,
             'file_count': len(files), 'total_bytes': sum(r['size_bytes'] for r in files),
             'logical_artifact_count': len(artifacts), 'group_counts': dict(groups),
             'coverage_contract': e['coverage_contract'], 'scientific_status': e['scientific_status'],
             'compression': {'large_original_threshold_bytes': LARGE_BYTES, 'gzip_part_bytes': PART_BYTES,
                             'gzip_mtime': 0, 'gzip_filename_header': '', 'gzip_level': 6},
             'original_absolute_paths_provenance_only': True, 'array_or_model_decode': 0,
             'Git_or_network_operations': 0, 'full_restore_required_before_stage': True}
        manifest = write_json(out/'MANIFEST.json', m)
        result = {'completed': True, 'PASS': True, 'manifest': manifest,
                  'file_count': len(files), 'total_bytes': m['total_bytes'],
                  'logical_artifacts': len(artifacts),
                  'large_encoded_originals': sum(a['encoding'] == 'gzip_parts' for a in artifacts),
                  'Git_or_network_operations': 0, 'array_or_model_decode': 0}
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        write_json(out/'FAILURE.json', {'completed': False, 'PASS': False,
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'completed_artifacts': artifacts, 'physical_files': files,
                   'partial_output_preserved': True, 'automatic_retry': False})
        raise


def verified_manifest(package):
    package = safe(package)
    with open_read(package/'MANIFEST.json') as f:
        raw = f.read()
    m = json.loads(raw)
    need(m.get('completed') is True and m.get('PASS') is True and m['schema'] == SCHEMA,
         'Sealed V261 package manifest')
    files = {}
    for row in m['files']:
        rel = row['relative_path']
        need(rel != 'MANIFEST.json' and rel not in files, 'Manifest excluded; unique physical leaves')
        p = safe(package/relative(rel, True))
        b = measure(p)
        need(b['sha256'] == row['sha256'] and b['size_bytes'] == row['size_bytes'], 'Physical SHA/size exact')
        need(b['size_bytes'] <= LARGE_BYTES, 'No physical leaf above90MiB')
        files[rel] = p
    used, logical = set(), set()
    for a in m['artifacts']:
        rel = a['relative_path']
        relative(rel, rel == RESTORE_TOOL)
        need(rel not in logical and a['encoding'] in ('identity', 'gzip_parts'), 'Unique logical artifact')
        logical.add(rel)
        parts = a['transport_files']
        need(parts and len(set(parts)) == len(parts) and set(parts) <= set(files) and
             not used.intersection(parts), 'Closed one-owner ordered transport parts')
        used.update(parts)
        if a['encoding'] == 'identity':
            need(a['original_size_bytes'] <= LARGE_BYTES and parts == [rel], 'Small original remains identity')
        else:
            need(a['original_size_bytes'] > LARGE_BYTES, 'Only oversized originals are compressed')
            h, count = hashlib.sha256(), 0
            for index, p in enumerate(parts):
                need(p == 'transport_gzip/'+rel+'.gz.part%04d' % index, 'Exact ordered gzip shard name')
                b = measure(files[p])
                need(0 < b['size_bytes'] <= PART_BYTES and
                     (index == len(parts)-1 or b['size_bytes'] == PART_BYTES), 'Fixed64MiB chunks except final')
                with open_read(files[p]) as f:
                    for chunk in iter(lambda: f.read(CHUNK), b''):
                        h.update(chunk)
                        count += len(chunk)
            need(count == a['compressed_stream_bytes'] and h.hexdigest() == a['compressed_stream_sha256'],
                 'Complete compressed stream binding')
    need(used == set(files) and len(files) == m['file_count'] and
         sum(r['size_bytes'] for r in m['files']) == m['total_bytes'] and
         len(logical) == m['logical_artifact_count'], 'Complete manifest accounting')
    return m, raw


def restore(package, output):
    package = safe(package)
    m, raw = verified_manifest(package)
    out = safe(output, False)
    need(not out.exists() and not out.is_relative_to(package), 'Fresh output outside immutable package')
    out.mkdir()
    results = []
    try:
        for a in m['artifacts']:
            target = safe(out/relative(a['relative_path'], a['relative_path'] == RESTORE_TOOL), False)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as dst:
                digest, size = stream_original(package, a, dst)
            results.append({'relative_path': a['relative_path'], 'sha256': digest, 'size_bytes': size})
        result = {'completed': True, 'PASS': True,
                  'package_manifest_sha256': hashlib.sha256(raw).hexdigest(),
                  'physical_files_verified': len(m['files']), 'logical_artifacts_restored': len(results),
                  'restored_artifacts': results, 'restored_original_bytes_exact': True,
                  'all_large_outputs_fully_restored': True,
                  'large_restored': sum(a['encoding'] == 'gzip_parts' for a in m['artifacts']),
                  'original_source_path_reads': 0, 'array_or_model_decode': 0,
                  'Git_or_network_operations': 0}
        receipt = write_json(out/'FULL_RESTORE_QA.json', result)
        print(json.dumps({'completed': True, 'PASS': True, 'receipt': receipt}, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        write_json(out/'RESTORE_FAILURE.json', {'completed': False, 'PASS': False,
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'partial_restored_artifacts': results, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--build-entry')
    group.add_argument('--package')
    ap.add_argument('--output-dir')
    args = ap.parse_args()
    if args.build_entry:
        need(args.output_dir is None, 'Build output comes only from Root entry')
        build(args.build_entry)
    else:
        need(args.output_dir is not None, 'Full restore requires fresh --output-dir')
        restore(Path(args.package), Path(args.output_dir))
