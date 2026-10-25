"""Root-entry-only opaque byte packaging with deterministic lossless gzip.

No treewalk, original-source pointer fallback, NumPy/Torch, Git or network.
Root supplies the complete file allowlist and expected coverage before execution.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import traceback

CHUNK = 1024*1024
PART_BYTES = 64*1024*1024
MINIMUM_GROUPS = {'raw_views': 4, 'canonical_features': 6, 'labels_samples_allrows': 2,
                  'attempts_code_protocols_QA': 1, 'paired_checkpoints': 2,
                  'paired_train_logs': 2, 'paired_draws': 2,
                  'paired_results_metrics': 2, 'reports': 1, 'visuals': 1}


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
    need(s not in ('MANIFEST.json', 'FAILURE.json', 'RESTORE_QA.json', 'RESTORE_FAILURE.json'), 'Reserved logical basename')
    return Path(*p.parts)


def safe(path, exists=True):
    lexical(path)
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute artifact path')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Missing artifact')
            continue
        need(not stat.S_ISLNK(mode), 'No artifact symlink')
    return p


def sha_size(path):
    h, size = hashlib.sha256(), 0
    with safe(path).open('rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def binding(path):
    sha, size = sha_size(path)
    return {'path': str(path), 'sha256': sha, 'size_bytes': size}


def read_bound(b):
    p = safe(b['path'])
    raw = p.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == b['sha256'] and len(raw) == b['size_bytes'], 'Bound metadata/code')
    return raw


def physical_binding(path, out):
    sha, size = sha_size(path)
    return {'relative_path': path.relative_to(out).as_posix(), 'sha256': sha, 'size_bytes': size}


def verify_gzip(path, expected):
    h, size = hashlib.sha256(), 0
    with safe(path).open('rb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='rb') as f:
            for chunk in iter(lambda: f.read(CHUNK), b''):
                h.update(chunk)
                size += len(chunk)
    need(h.hexdigest() == expected['sha256'] and size == expected['size_bytes'], 'Decompressed original bytes exact')


def transfer(source, rel, encoding, out, single_limit, part_bytes=PART_BYTES):
    """Only opaque 1MiB reads. part_bytes is fixed64MiB in production run."""
    src = safe(source['path'])
    dest = safe((out/'transport'/relative(rel)) if encoding == 'gzip' else (out/relative(rel)), False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    need(encoding in ('raw', 'gzip'), 'Explicit raw/gzip encoding')
    target = Path(str(dest)+'.gz') if encoding == 'gzip' else dest
    safe(target, False)
    h, size = hashlib.sha256(), 0
    with target.open('xb') as output:
        stream = gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0, compresslevel=6) if encoding == 'gzip' else output
        with src.open('rb') as f:
            for chunk in iter(lambda: f.read(CHUNK), b''):
                h.update(chunk)
                size += len(chunk)
                stream.write(chunk)
        if stream is not output:
            stream.close()
    need(h.hexdigest() == source['sha256'] and size == source['size_bytes'], 'Source SHA/size exact')
    compressed_size = os.lstat(safe(target)).st_size
    if encoding == 'raw':
        need(compressed_size <= single_limit, 'Raw artifact exceeds Root single-file limit; declare gzip')
    else:
        verify_gzip(target, source)
    outputs = []
    if encoding == 'gzip' and compressed_size > single_limit:
        need(0 < part_bytes < single_limit, 'Parts remain below Root single-file limit')
        joined_sha, joined_size = hashlib.sha256(), 0
        expected_compressed_sha, _ = sha_size(target)
        with target.open('rb') as f:
            index = 0
            while joined_size < compressed_size:
                part = safe(Path(str(target)+'.part%04d' % index), False)
                with part.open('xb') as g:
                    remaining = part_bytes
                    while remaining and joined_size < compressed_size:
                        chunk = f.read(min(CHUNK, remaining))
                        need(bool(chunk), 'Compressed stream unexpectedly truncated')
                        g.write(chunk)
                        joined_sha.update(chunk)
                        joined_size += len(chunk)
                        remaining -= len(chunk)
                outputs.append(physical_binding(part, out))
                index += 1
        need(joined_size == compressed_size and joined_sha.hexdigest() == expected_compressed_sha,
             'Split concatenation preserves exact compressed stream')
        target.unlink()
    else:
        outputs.append(physical_binding(target, out))
    return {'relative_path': rel, 'group': None, 'encoding': encoding,
            'original_sha256': source['sha256'], 'original_size_bytes': size,
            'transport_files': [x['relative_path'] for x in outputs],
            'compressed_stream_bytes': compressed_size if encoding == 'gzip' else None,
            'split_only_because_above_single_file_limit': len(outputs) > 1,
            'original_source_path_inert': source['path']}, outputs


def run(entry_path):
    raw = safe(entry_path).read_bytes()
    e = json.loads(raw)
    need(e.get('approved_actual_publication_package') is True, 'Root actual publication entry required')
    need(e['code'] == binding(Path(__file__).absolute()), 'Exact builder code')
    scope = json.loads(read_bound(e['root_scope']))
    need(scope.get('approved_actual_publication_package') is True and
         scope.get('entry_contract') == {k: v for k, v in e.items() if k != 'root_scope'}, 'Exact Root scope')
    need(e.get('source_payloads_only') is True and e.get('array_or_model_decode') is False and
         e.get('automatic_retry') is False, 'Opaque SOURCE only/no decode/no retry')
    restore_code = read_bound(e['restore_code'])
    need(e['restore_code']['path'] == str(Path(__file__).with_name('restore_artifacts.py').absolute()), 'Sibling closed restorer')
    rows = e['files']
    counts = Counter(row['group'] for row in rows)
    need(all(counts[g] >= n for g, n in MINIMUM_GROUPS.items()), 'Required full raw4/features6/labels/attempts/pairedfit/results/visual groups')
    need(counts['raw_views'] == 4 and counts['canonical_features'] == 6 and
         all(counts[k] == 2 for k in ('paired_checkpoints', 'paired_train_logs', 'paired_draws')), 'Exact matrix and pair file counts')
    need(set(e['coverage_contract']) == set(counts), 'Exact group coverage contract')
    for group, expected in e['coverage_contract'].items():
        actual = [r['source'] for r in rows if r['group'] == group]
        norm = lambda values: sorted(json.dumps(v, sort_keys=True) for v in values)
        need(norm(actual) == norm(expected), 'All declared source-group leaves included, no outcome filtering')
    logical = [r['relative_path'] for r in rows]
    need(len(set(logical)) == len(logical) and 'tools/restore_artifacts.py' not in logical, 'Unique logical paths')
    for rel in logical:
        relative(rel)
    limit = int(e['single_file_limit_bytes'])
    need(limit > PART_BYTES, 'Root single-file limit must exceed fixed64MiB parts')
    out = safe(e['output_directory'], False)
    need(not out.exists(), 'Fresh package, no overwrite/resume')
    out.mkdir()
    artifacts, files = [], []
    try:
        for index, row in enumerate(rows, 1):
            artifact, physical = transfer(row['source'], row['relative_path'], row['encoding'], out, limit)
            artifact['group'] = row['group']
            artifacts.append(artifact)
            files.extend(physical)
            print(json.dumps({'packaged': index, 'total': len(rows), 'group': row['group'],
                              'physical_parts': len(physical)}, sort_keys=True), flush=True)
        source = e['restore_code']
        artifact, physical = transfer(source, 'tools/restore_artifacts.py', 'raw', out, limit)
        artifact['group'] = 'portable_restore_code'
        artifacts.append(artifact)
        files.extend(physical)
        need(len({f['relative_path'] for f in files}) == len(files), 'No physical transport collisions')
        manifest = {'schema': 'V256_PORTABLE_ALL_ARTIFACT_BYTES_V1', 'completed': True, 'PASS': True,
                    'code': e['code'], 'entry_sha256': hashlib.sha256(raw).hexdigest(),
                    'files': files, 'artifacts': artifacts, 'group_counts': dict(counts),
                    'compression': {'gzip_mtime': 0, 'filename_header': '', 'level': 6,
                                    'stream_chunk_bytes': CHUNK, 'split_part_bytes': PART_BYTES,
                                    'single_file_limit_bytes': limit,
                                    'decompressed_original_SHA_and_size_exact': True},
                    'coverage_contract': e['coverage_contract'], 'scope': e['root_scope'],
                    'original_absolute_paths_inert_no_fallback': True, 'array_or_model_decode': 0,
                    'no_model_selection_or_outcome_filtering': True}
        safe(out/'MANIFEST.json', False).write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
        result = {'completed': True, 'PASS': True, 'manifest': binding(out/'MANIFEST.json'),
                  'logical_artifacts': len(artifacts), 'physical_files': len(files),
                  'original_total_bytes': sum(a['original_size_bytes'] for a in artifacts),
                  'transport_total_bytes': sum(f['size_bytes'] for f in files),
                  'array_or_model_decode': 0, 'Git_or_network_operations': 0}
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        safe(out/'FAILURE.json', False).write_text(json.dumps({'completed': False, 'PASS': False,
            'error': str(exc), 'traceback': traceback.format_exc(), 'completed_artifacts': artifacts,
            'physical_files': files, 'automatic_retry': False}, ensure_ascii=False, indent=2)+'\n')
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', type=Path, required=True)
    print(json.dumps(run(ap.parse_args().entry), indent=2))
