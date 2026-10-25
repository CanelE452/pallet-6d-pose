"""Verify published bytes and replay the frozen independent V247 saved chosen-mask QA.

All old absolute paths are identities only. Every scientific read must map to
a MANIFEST-declared package leaf. No original-source filesystem fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile

CHECKER_SHA = '7951046dfb1e1f8236bce031968de91b1b374bcbf106592a90871ad714734ce9'
ENTRY_SUFFIX = '/V247_ROOT_PUBLIC_TRAIN_SAM/ACTUAL_INDEPENDENT_SAVED_SAM_QA_ENTRY.json'
CAP = 95_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_absolute(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute path required')
    require(not any(s in str(path).casefold() for s in ('v112', '__pycache__', '.pyc')), 'Forbidden path')
    for parent in reversed((path, *path.parents)):
        try:
            mode = os.lstat(parent).st_mode
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(mode), 'Symlink forbidden')
    return path


def local_read(path):
    path = safe_absolute(path)
    require(stat.S_ISREG(os.lstat(path).st_mode), 'Regular file required')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        data = stream.read(CAP)
        require(not stream.read(1) and len(data) < CAP, 'Publication size cap')
    return data


def package_path(root, relative):
    require(isinstance(relative, str), 'Relative name required')
    part = PurePosixPath(relative)
    require(not part.is_absolute() and '..' not in part.parts and '\\' not in relative, 'Package traversal')
    require(relative not in ('', '.'), 'Empty leaf')
    return safe_absolute(root / relative)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(root, out):
    root = safe_absolute(root)
    manifest = json.loads(local_read(root / 'MANIFEST.json'))
    declared = {}
    for row in manifest['files']:
        relative = row['path']
        require(relative not in declared and row['bytes'] < CAP, 'Duplicate/oversize leaf')
        data = local_read(package_path(root, relative))
        require((len(data), sha(data)) == (row['bytes'], row['sha256']), 'Manifest bytes: ' + relative)
        declared[relative] = row
    source_map = json.loads(local_read(package_path(root, 'PUBLICATION_SOURCE_MAP.json')))
    aliases = {}
    for row in source_map['files']:
        require(row['encoding'] == 'identity', 'V247 identity encoding required')
        relative = row['path']
        require(relative in declared, 'Undeclared source-map leaf')
        leaf = declared[relative]
        require((row['source_sha256'], row['source_bytes']) == (leaf['sha256'], leaf['bytes']), 'Source-map SHA/size')
        original = row['source_path_provenance_only']
        require(isinstance(original, str) and original.startswith('/'), 'Original identity')
        if original in aliases:
            other = declared[aliases[original]]
            require((other['sha256'], other['bytes']) == (leaf['sha256'], leaf['bytes']), 'Conflicting alias')
        else:
            aliases[original] = relative
    entries = [key for key in aliases if key.endswith(ENTRY_SUFFIX)]
    require(len(entries) == 1, 'Exactly one frozen actual QA entry')
    translated_reads = []

    def mapped_read(original):
        # The original string is never passed to a filesystem operation.
        identity = str(original)
        require(identity in aliases, 'No original-source fallback: ' + identity)
        relative = aliases[identity]
        data = local_read(package_path(root, relative))
        leaf = declared[relative]
        require((len(data), sha(data)) == (leaf['bytes'], leaf['sha256']), 'Mapped bytes changed')
        translated_reads.append(relative)
        return data

    entry = json.loads(mapped_read(entries[0]))
    require(entry['code']['sha256'] == CHECKER_SHA, 'Frozen independent checker')
    checker = mapped_read(entry['code']['path'])
    require(sha(checker) == CHECKER_SHA, 'Checker bytes')
    namespace = {'__name__': 'v247_published_independent_QA', '__file__': entry['code']['path']}
    exec(compile(checker, entry['code']['path'], 'exec'), namespace)
    namespace['raw'] = mapped_read
    namespace['H'] = out
    namespace['run'](entries[0])
    review = json.loads(local_read(out / 'REVIEW.json'))
    require(review['completed'] is True and review['PASS'] is True, 'Independent result QA')
    require(review['original_queries'] == 352 and len(review['NPZ_checks']) == 42, 'Exact SOURCE saved cohort')
    table_keys = [k for k in aliases if k.endswith('/V247_SAVED_SAM_QA_C/CONDITIONAL_SUPPORT_TABLES.json')]
    require(len(table_keys) == 1, 'Published independent table')
    require(local_read(out / 'CONDITIONAL_SUPPORT_TABLES.json') == mapped_read(table_keys[0]), 'Replayed table bytes equal actual independent QA')
    print(json.dumps({
        'completed': True, 'PASS': True,
        'declared_package_files_verified': len(declared),
        'TRAIN_queries_independently_verified': 352,
        'TRAIN_saved_chosen_masks_independently_verified': 42,
        'mapped_reads': len(translated_reads),
        'distinct_mapped_leaves': len(set(translated_reads)),
        'original_source_filesystem_fallback': False,
        'review': str(out / 'REVIEW.json'),
        'scope': 'Frozen chosen-mask readout and conditional-support table arithmetic; no physical/native gold or model success certification',
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).absolute().parent)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out is None:
        output = Path(tempfile.mkdtemp(prefix='pallet-v247-verify-'))
    else:
        output = safe_absolute(args.out.absolute())
        require(not output.exists(), 'Fresh output directory required')
        output.mkdir()
    run(args.package.absolute(), output)
