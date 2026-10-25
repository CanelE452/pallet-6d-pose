#!/usr/bin/env python3
"""Independent, allowlist-only V236 package byte and portable-link audit."""
import gzip
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import PurePosixPath
import re
import stat
from urllib.parse import unquote, urlsplit

ROOT = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V236_PUBLICATION_PACKAGE'
OUT = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V236_PUBLICATION_VERIFY_A'
MANIFEST_SHA = '1209aa19c7ec860289f6134e1f42629dce871d2560c2f1045151f6ccb43f58c2'
DOCUMENTS = [
    'README_KO.md', 'DIAGNOSIS_V234_KO.md',
    'artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md',
    'artifacts/V235_SOURCE_GALLERY_A/gallery_v1/PORTABLE_GALLERY_KO.html',
    'artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/PORTABLE_CONTEXT_GALLERY_KO.html',
]


def lexical(path, relative=False):
    value = str(path)
    lower = value.lower()
    if 'v112' in lower or '__pycache__' in lower or lower.endswith('.pyc'):
        raise ValueError('Forbidden lexical path')
    if '\\' in value or '\x00' in value:
        raise ValueError('Invalid path text')
    parts = PurePosixPath(value).parts
    if '..' in parts or (relative and (value.startswith('/') or not value or '.' in parts)):
        raise ValueError('Non-canonical relative path')
    return value


def open_regular(path):
    path = lexical(path)
    parts = PurePosixPath(path).parts
    if not path.startswith('/'):
        raise ValueError('Expected absolute internal path')
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError('Not a regular file')
        return os.fdopen(descriptor, 'rb')
    finally:
        os.close(directory)


def read_owned(relative):
    relative = lexical(relative, relative=True)
    with open_regular(ROOT + '/' + relative) as handle:
        return handle.read()


def byte_digest(handle):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []

    def handle_starttag(self, tag, attributes):
        for key, value in attributes:
            if key in ('href', 'src') and value:
                self.links.append(value)


def links_in(name, text):
    if name.endswith('.html'):
        parser = HTMLLinks()
        parser.feed(text)
        return parser.links
    # Only rendered Markdown links/images; fenced examples are not navigation.
    text = re.sub(r'(?ms)^```.*?^```[^\n]*$', '', text)
    return [a or b for a, b in re.findall(
        r'!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s\)]+))(?:\s+"[^"\n]*")?\s*\)', text)]


def resolve_local(source, target, allowed):
    target = html.unescape(target)
    # Lexical reject is before any target file operation, including encoded forms.
    decoded = unquote(target)
    if 'v112' in decoded.lower() or '__pycache__' in decoded.lower():
        raise ValueError('Forbidden link text')
    split = urlsplit(decoded)
    if split.scheme or split.netloc or split.path.startswith('/'):
        return None  # Record as inert; never follow.
    path = split.path
    if not path:
        return source
    if '\\' in path or '\x00' in path or path.lower().endswith('.pyc'):
        raise ValueError('Forbidden link path')
    normalized = os.path.normpath(str(PurePosixPath(source).parent / path))
    if (source, target) == ('README_KO.md', '../V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md'):
        return None  # Explicit root-approved historical publication link; no target FS.
    lexical(normalized, relative=True)
    if normalized not in allowed:
        raise ValueError('Local link outside declared package: ' + normalized)
    return normalized


def main():
    raw_manifest = read_owned('MANIFEST.json')
    assert hashlib.sha256(raw_manifest).hexdigest() == MANIFEST_SHA
    manifest = json.loads(raw_manifest)
    records = manifest['files']
    assert manifest['file_count'] == len(records) == 222
    assert manifest['total_bytes'] == 47182119
    by_path = {}
    checked = []
    for item in records:
        name = lexical(item['path'], relative=True)
        assert name not in by_path and name != 'MANIFEST.json'
        assert re.fullmatch('[0-9a-f]{64}', item['sha256'])
        assert type(item['bytes']) is int and item['bytes'] >= 0
        by_path[name] = item
    # Validate every declared path before the first payload file is opened.
    for name, item in by_path.items():
        with open_regular(ROOT + '/' + name) as handle:
            size, digest = byte_digest(handle)
        assert (size, digest) == (item['bytes'], item['sha256']), name
        checked.append({'path': name, 'sha256': digest, 'bytes': size})
    assert sum(item['bytes'] for item in checked) == 47182119

    source_map_raw = read_owned('PUBLICATION_SOURCE_MAP.json')
    source_map = json.loads(source_map_raw)
    mapped = set()
    gzip_checks = []
    identity_count = 0
    for item in source_map['files']:
        name = lexical(item['path'], relative=True)
        assert name in by_path and name not in mapped
        mapped.add(name)
        if item['encoding'] == 'identity':
            assert (item['source_sha256'], item['source_bytes']) == (
                by_path[name]['sha256'], by_path[name]['bytes']), name
            identity_count += 1
        elif item['encoding'] == 'gzip-lossless':
            with open_regular(ROOT + '/' + name) as packed:
                with gzip.GzipFile(fileobj=packed, mode='rb') as expanded:
                    size, digest = byte_digest(expanded)
            assert (digest, size) == (item['source_sha256'], item['source_bytes']), name
            gzip_checks.append({'path': name, 'restored_sha256': digest,
                                'restored_bytes': size})
        else:
            raise ValueError('Unknown encoding')
    assert len(mapped) == 217 and identity_count == 213 and len(gzip_checks) == 4

    allowed_links = set(by_path) | {'MANIFEST.json'}
    documents = []
    all_targets = set()
    for name in DOCUMENTS:
        text = read_owned(name).decode('utf-8')
        links = links_in(name, text)
        local = []
        inert = []
        for link in links:
            target = resolve_local(name, link, allowed_links)
            if target is None:
                inert.append(link)
            else:
                local.append({'link': link, 'target': target})
                all_targets.add(target)
        documents.append({'path': name, 'sha256': by_path[name]['sha256'],
                          'local_link_count': len(local), 'local_links': local,
                          'nonlocal_inert_not_followed': inert})

    readme = read_owned('README_KO.md').decode('utf-8')
    report = read_owned(DOCUMENTS[2]).decode('utf-8')
    gallery = read_owned(DOCUMENTS[3]).decode('utf-8')
    context_gallery = read_owned(DOCUMENTS[4]).decode('utf-8')
    # Bounded claim checks only, not a new numerical/scientific result audit.
    assert '모델을 채택하지 않았다' in readme
    assert 'current admission=false' in report and 'selected arm=null' in report
    assert '12 bucket 중 11개 고유 crop과 빈 bucket 1개' in gallery
    assert 'SOURCE 내부 DEV는 재사용·노출된 자료' in gallery
    assert 'native mapping은 UNKNOWN' in context_gallery
    assert manifest['current_admission'] is False and manifest['source_gate_pass'] is False

    code_bytes = open_regular(__file__).read()
    result = {
        'completed': True, 'PASS': True,
        'package': ROOT,
        'manifest': {'path': ROOT + '/MANIFEST.json', 'sha256': MANIFEST_SHA,
                     'size_bytes': len(raw_manifest)},
        'verifier': {'path': __file__, 'sha256': hashlib.sha256(code_bytes).hexdigest()},
        'declared_file_count': len(checked), 'declared_total_bytes': 47182119,
        'all_declared_SHA_and_size_exact': True,
        'source_map_count': 217, 'identity_source_bindings_matched': identity_count,
        'gzip_lossless_checks': gzip_checks,
        'portable_documents': documents,
        'intentional_external_relative_historical_link': {
            'document': 'README_KO.md',
            'target': '../V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md',
            'independently_stat_or_read': False,
            'package_standalone_target_available': 'UNVERIFIED_OUTSIDE_SCOPE',
            'root_reported_main_repository_is_file_check': {
                'terminal': '8b78a9 exit0',
                'receipt_path': 'V236_COPIED_BYTES_AND_ENTRYPOINT_QA.json',
                'receipt_sha256': '5a9c8f15ee96c96f594e45aa8e71b9fe222f26470f3d6de0e9edc2291a73e2ca',
                'independent_checker_followed_receipt': False
            }
        },
        'all_package_internal_portable_links_PASS': True,
        'fully_self_contained_all_navigation_links': False,
        'local_link_count': sum(x['local_link_count'] for x in documents),
        'unique_local_link_targets': len(all_targets),
        'bounded_text_claim_checks_PASS': True,
        'no_admission_and_no_model_success_claim_preserved': True,
        'checks': checked,
        'limits': [
            'Declared manifest allowlist only; no treewalk or unlisted-file census.',
            'PT/PNG/NPY and other payloads opaque SHA/size only; no numerical or image decoding.',
            'Gzip streams restored to bytes and hashed only; restored JSONL not semantically parsed.',
            'Source-root and original path strings in source map and documents were inert; no original pointer followed.',
            'Only the five named portable documents were parsed for links and bounded text claims.',
            'Local file links checked against verified manifest; URL/fragment content and HTML browser rendering not audited.',
            'No protected evaluation, model execution, new case selection, package mutation, or scientific recomputation.'
        ],
        'V112_or_pycache_path_FS_access': 0,
        'original_source_pointer_follow_count': 0,
        'package_writes': 0,
    }
    output = lexical(OUT + '/ACTUAL_PACKAGE_QA.json')
    with open(output, 'x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'file_count': len(checked),
                      'bytes': 47182119, 'gzip_count': len(gzip_checks),
                      'local_links': result['local_link_count'],
                      'unique_link_targets': len(all_targets), 'receipt': output}, ensure_ascii=False))


if __name__ == '__main__':
    main()
