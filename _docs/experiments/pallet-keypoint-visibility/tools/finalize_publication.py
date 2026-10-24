"""Merge immutable archive and separately authored portable publication records.

No Git operations, model execution, dataset decoding, or source tree discovery.
Only exact administrative source paths and the completed archive manifest are read.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, posixpath, re, zlib
from urllib.parse import unquote, urlsplit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(value):
    if 'v112' in str(value).lower():
        raise ValueError('Forbidden name before access')
    return Path(value)


def relative(value):
    p = PurePosixPath(value)
    if 'v112' in value.lower() or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe publication relative path')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--target', required=True)
    args = ap.parse_args()
    source, target = checked(args.source), checked(args.target)
    admin = source/'publication_root_preparation'
    records = [json.loads(line) for line in (target/'publication/SOURCE_ARCHIVE_MANIFEST.jsonl').read_text().splitlines()]
    supplemental = json.loads((admin/'SUPPLEMENTAL_FILE_MANIFEST_V2.json').read_text())['entries']
    records.extend(supplemental)

    def add_raw(path, kind, source_path=None, source_sha=None):
        relative(path)
        f = target/path
        data = f.read_bytes()
        h = digest(data)
        records.append(dict(source_absolute=str(source_path) if source_path else None,
                            source_sha256=source_sha or h, published_relative=path,
                            published_sha256=h, size_bytes=len(data), stored_size_bytes=len(data),
                            status='PUBLISHED_RAW', gzipchunks=None, artifact_kind=kind))

    def copy_admin(origin, destination):
        checked(str(origin)); relative(destination)
        data = origin.read_bytes(); h = digest(data)
        f = target/destination
        if f.exists():
            raise ValueError('Administrative destination exists')
        f.parent.mkdir(parents=True, exist_ok=True)
        if len(data) < 1024**2:
            f.write_bytes(data); add_raw(destination, 'PUBLICATION_PROVENANCE', origin, h)
            return
        parts = []
        for i, start in enumerate(range(0, len(data), 32*1024**2)):
            block = data[start:start+32*1024**2]
            c = zlib.compressobj(6, zlib.DEFLATED, 31)
            packed = c.compress(block)+c.flush()
            part = destination+'.parts/part%05d.gz'%i
            pf = target/part; pf.parent.mkdir(parents=True, exist_ok=True)
            with pf.open('xb') as stream: stream.write(packed)
            parts.append(dict(path=part, sha256=digest(packed), compression='gzip',
                              stored_size_bytes=len(packed), decoded_size_bytes=len(block)))
        spec = dict(format='concatenated-bytes-v1', parts=parts,
                    content_sha256=h, content_size_bytes=len(data))
        descriptor = destination+'.parts.json'
        with (target/descriptor).open('x') as stream:
            json.dump(spec, stream, ensure_ascii=False, indent=2); stream.write('\n')
        records.append(dict(source_absolute=str(origin), source_sha256=h,
                            published_relative=destination, published_sha256=None,
                            size_bytes=len(data), status='PUBLISHED_GZIP_CHUNKS', gzipchunks=spec,
                            gzip_descriptor_path=descriptor,
                            gzip_descriptor_sha256=digest((target/descriptor).read_bytes()),
                            gzip_descriptor_size_bytes=(target/descriptor).stat().st_size,
                            artifact_kind='PUBLICATION_PROVENANCE'))
        add_raw(descriptor, 'LOSSLESS_STORAGE_DESCRIPTOR')

    c = source/'publication_preparation_c'
    for name in ['package_publication_snapshot.py', 'refine_inventory_and_hash.py',
                 'finalize_publication_candidates.py', 'PACKAGER_INVENTED_CHECKS.json',
                 'FINAL_PUBLICATION_PREPARATION_COMPLETE.json', 'PUBLICATION_WORKFLOW_KO.md',
                 'ROOT_COPY_PLAN_TEMPLATE_APPROVED_FALSE.json',
                 'PACKAGER_PREP_NO_OVERWRITE_FAILURE_PRESERVED.json']:
        copy_admin(c/name, 'tools/snapshot/'+name)
    for name in ['PUBLICATION_FILE_MANIFEST_CANDIDATE.jsonl',
                 'FINAL_CANDIDATE_SIZE_AND_COMPRESSION_SUMMARY.json',
                 'SMALL_OWN_HEAD_EXPLICIT_CANDIDATES.jsonl',
                 'SAVED_NUMERICAL_OUTPUT_EXPLICIT_CANDIDATES.jsonl',
                 'OTHER_SMALL_NUMERIC_EXPLICIT_CANDIDATES.jsonl',
                 'NPZ_HEADER_SCHEMA_RECORDS.jsonl', 'LOSSLESS_TEXT_COMPRESSION_STATS.jsonl']:
        copy_admin(c/'v5_final_candidates'/name, 'publication/source_inventory/'+name)
    for name in ['DIRECTORY_PRUNE_RECORDS.jsonl']:
        copy_admin(c/'v2_refined'/name, 'publication/source_inventory/'+name)
    copy_admin(admin/'ROOT_ACTUAL_ARCHIVE_COPY_PLAN.json', 'publication/ROOT_ACTUAL_ARCHIVE_COPY_PLAN.json')
    copy_admin(Path(__file__), 'tools/finalize_publication.py')

    # Descriptors are physical publication files too, distinct from original bytes.
    have = {r['published_relative'] for r in records if r.get('published_relative')}
    for r in list(records):
        descriptor = r.get('gzip_descriptor_path')
        if descriptor and descriptor not in have:
            add_raw(descriptor, 'LOSSLESS_STORAGE_DESCRIPTOR'); have.add(descriptor)

    compressed = {r['published_relative']:r['gzip_descriptor_path'] for r in records
                  if r.get('gzip_descriptor_path')}
    rewrites = []
    # Published derivatives may point to real descriptors. Original documents never change.
    for r in supplemental:
        path = r['published_relative']
        if r.get('artifact_kind') != 'PORTABLE_DERIVATIVE':
            continue
        f = target/path; old = f.read_text()
        changed = []
        def replace(match):
            url = match.group(1)
            parsed = urlsplit(url)
            if parsed.scheme or url.startswith(('/', '#')):
                return match.group(0)
            logical = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(parsed.path)))
            if logical not in compressed:
                return match.group(0)
            new = posixpath.relpath(compressed[logical], posixpath.dirname(path) or '.')
            changed.append(dict(original_url=url, published_url=new, logical=logical))
            return ']('+new+')'
        new = re.sub(r'\]\(([^\s)]+)\)', replace, old)
        if changed:
            new += '\n큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.\n'
            f.write_text(new)
            rewrites.append(dict(path=path, previous_derivative_sha256=digest(old.encode()), changes=changed))
            h = digest(new.encode()); r.update(published_sha256=h, size_bytes=len(new.encode()),
                                              stored_size_bytes=len(new.encode()), derivative_source_sha256=r['source_sha256'])
            # This is a separately derived current document, not an unchanged original.
            r['source_sha256'] = h
            r['derived_from_source_absolute'] = r['source_absolute']
            r['source_absolute'] = None
            r['artifact_kind'] = 'PORTABLE_DERIVATIVE_WITH_LOSSLESS_STORAGE_LINKS'
    (target/'publication/PORTABLE_COMPRESSION_LINK_REWRITES.json').write_text(json.dumps(rewrites, ensure_ascii=False, indent=2)+'\n')
    add_raw('publication/PORTABLE_COMPRESSION_LINK_REWRITES.json', 'PUBLICATION_PROVENANCE')
    for name in ['README.md', 'PUBLICATION_GUIDE_KO.md', 'publication/SOURCE_ARCHIVE_COPY_COMPLETE.json',
                 'publication/SOURCE_ARCHIVE_MANIFEST.jsonl']:
        add_raw(name, 'PUBLICATION_GUIDE_OR_PROVENANCE')
    manifest = dict(format='pallet-keypoint-visibility-publication-v2',
                    scope='Immutable original experiments and separate portable derivatives; V112 is excluded before access.',
                    entries=records)
    f = target/'PUBLICATION_MANIFEST.json'
    with f.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, separators=(',', ':')); stream.write('\n')
    if f.stat().st_size >= 100*1024**2:
        raise ValueError('Final manifest exceeds GitHub file bound')
    print(json.dumps(dict(completed=True, entries=len(records), manifest_bytes=f.stat().st_size,
                          portable_documents_relinked=len(rewrites))))


if __name__ == '__main__':
    main()
