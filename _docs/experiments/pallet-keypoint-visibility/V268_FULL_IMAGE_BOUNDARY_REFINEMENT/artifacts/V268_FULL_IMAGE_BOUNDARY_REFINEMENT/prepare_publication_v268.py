"""Copy frozen V268 results and exact small V267 dependencies into a fresh package.

This program performs no inference, training, Git/index mutation or artifact-code
imports. Run only after the source experiment has been declared final.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse
import hashlib
import json
import os
import posixpath
import time


NAME = 'V268_FULL_IMAGE_BOUNDARY_REFINEMENT'
OLD_NAME = 'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
SOURCE = Path(__file__).absolute().parent
OLD_SOURCE = SOURCE.parent / OLD_NAME
TARGET = Path('/home/minjae/Documents/pallet-visibility-publication-20261024/_docs/experiments/pallet-keypoint-visibility') / NAME
EXCLUDED_RGB = 'syn_refine_inputs_v1/RGB256_FP16.npy'
LIMIT = 100 * 1024 * 1024
DEPENDENCIES = (
    'current_fullframe_inputs_v2/COMPLETE.json',
    'current_fullframe_inputs_v2/FRAMES223_INPUT_ONLY.jsonl',
    'current_fullframe_inputs_v2/QUERIES1784_INPUT_ONLY.jsonl',
    'CURRENT_PREDICTIONS_FROZEN.json',
    'CURRENT_ALL1784_PREDICTIONS.jsonl',
    'CURRENT_FRAME_MASKS.npz',
    'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json',
    'PROTOCOL_BEFORE_FIT.json',
    'model_mask_head_v267.py',
    'fullframe_clip_v267.py',
    'OFFICIAL_SOURCE_PROVENANCE.json',
    'pretrained/DOWNLOAD_PROVENANCE.json',
    'syn_training_v1/CLIP_SURGERY_ARCH_FINAL1000.pt',
    'data_audit/COMPLETE.json',
    'data_audit/FULL_FRAME_RGB_INPUTS.jsonl',
    'data_audit/VISIBLE_MASK_TARGETS.jsonl',
    'syn_fullframe_features_v1/COMPLETE.json',
)


def safe(p):
    p = Path(p)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112', 'cache', '.pyc', '..')), str(p)
    for ancestor in reversed((p, *p.parents)):
        assert not ancestor.is_symlink(), str(ancestor)
    return p


def digest(p):
    p = safe(p)
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def owned_files(directory):
    """Walk only the owned V268 tree; never traverse old experiment trees."""
    directory = safe(directory)
    result = []
    with os.scandir(directory) as entries:
        children = sorted((Path(x.path) for x in entries), key=lambda x: x.name)
    for child in children:
        child = safe(child)
        if child.is_dir():
            result.extend(owned_files(child))
        else:
            assert child.is_file(), str(child)
            result.append(child)
    return result


def write_json(path, obj):
    with safe(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def copy_exact(source, destination, expected):
    source, destination = safe(source), safe(destination)
    assert source.stat().st_size < LIMIT, ('Included file must be <100MiB', str(source))
    safe(destination.parent).mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), str(destination)
    with source.open('rb') as src, destination.open('xb') as dst:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
            dst.write(chunk)
    actual = digest(destination)
    assert (actual['sha256'], actual['size_bytes']) == (expected['sha256'], expected['size_bytes']), str(destination)


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value is not None:
                self.links.append((tag, key, value))


def verify_links(package, html_paths):
    """Normalize URL paths lexically first; then enforce package confinement."""
    package = safe(package)
    checked, skipped, failures = [], [], []
    for html_path in html_paths:
        html_path = safe(html_path)
        relative = html_path.relative_to(package).as_posix()
        parser = HTMLLinks()
        parser.feed(html_path.read_text(encoding='utf-8'))
        for tag, attr, raw in parser.links:
            record = dict(html_path=relative, tag=tag, attribute=attr, value=raw)
            url = urlsplit(raw)
            if url.scheme.lower() == 'file':
                failures.append(dict(record, reason='file_URL_not_portable_or_confined'))
                continue
            if url.scheme and url.scheme.lower() not in ('http', 'https', 'mailto', 'tel', 'data'):
                failures.append(dict(record, reason='unsupported_URL_scheme'))
                continue
            if url.scheme or url.netloc:
                skipped.append(dict(record, reason='external_or_embedded_URL'))
                continue
            decoded = unquote(url.path)
            if not decoded:
                skipped.append(dict(record, reason='same_document_fragment_or_query'))
                continue
            if decoded.startswith('/') or '\\' in decoded or '\x00' in decoded:
                failures.append(dict(record, reason='absolute_or_nonportable_local_path'))
                continue
            normalized = posixpath.normpath(posixpath.join(posixpath.dirname(relative), decoded))
            if normalized in ('.', '..') or normalized.startswith('../'):
                failures.append(dict(record, normalized=normalized, reason='escapes_package'))
                continue
            try:
                target = safe(package / normalized)
                target.relative_to(package)
                assert target.exists(), str(target)
            except (AssertionError, ValueError) as error:
                failures.append(dict(record, normalized=normalized, reason=str(error)))
                continue
            checked.append(dict(record, package_path=normalized))
    return dict(completed=True, PASS=not failures, html_files=len(html_paths),
                checked_local_links=len(checked), skipped_URL_or_fragment_links=len(skipped),
                policy='URL decode, strip query/fragment, lexical normalization, package confinement, symlink guard, existence.',
                links=checked, skipped=skipped, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Execute only after root declares the final source freeze.')
    args = parser.parse_args()
    assert args.execute, 'Source inspection only by default; explicit --execute is required.'
    started = time.monotonic()
    assert SOURCE.name == NAME and safe(SOURCE).is_dir()
    safe(TARGET)
    assert not TARGET.exists(), ('Fresh publication directory required', str(TARGET))
    assert safe(TARGET.parent).is_dir()
    protocol = json.loads(safe(SOURCE / 'PROTOCOL_BEFORE_FIT.json').read_bytes())
    assert protocol['PASS'] and protocol['train_arms'] == ['HIGHRES_REFINE', 'BOUNDARY_REFINE']
    complete = json.loads(safe(SOURCE / 'syn_training_v1/COMPLETE.json').read_bytes())
    assert complete['PASS'] and complete['new_optimizer_updates'] == 2000
    assert complete['current_or_real_fit_reads'] == 0
    source_files = owned_files(SOURCE)
    relative_source = [p.relative_to(SOURCE).as_posix() for p in source_files]
    assert EXCLUDED_RGB in relative_source
    assert 'prepare_publication_v268.py' in relative_source
    assert 'PUBLICATION_README_KO.md' in relative_source
    selected = json.loads(safe(SOURCE / 'syn_selected_raw_inputs_v1/SOURCE_RGB_M4_PAIRS.json').read_bytes())
    assert selected['PASS'] and len(selected['files']) == 12
    assert {r['frame_index'] for r in selected['files']} == {776, 768, 770, 769, 792, 900}
    for row in selected['files']:
        b = digest(SOURCE / 'syn_selected_raw_inputs_v1' / row['output'])
        assert (b['sha256'], b['size_bytes']) == (row['sha256'], row['size_bytes'])
        assert (b['sha256'], b['size_bytes']) == (row['source']['sha256'], row['source']['size_bytes'])
    payload = TARGET / 'artifacts' / NAME
    planned = []
    for src in source_files:
        rel = src.relative_to(SOURCE).as_posix()
        if rel == EXCLUDED_RGB:
            continue
        b = digest(src)
        assert b['size_bytes'] < LIMIT, (rel, b['size_bytes'])
        planned.append(dict(source_path=str(src), package_path=(Path('artifacts') / NAME / rel).as_posix(),
                            sha256=b['sha256'], size_bytes=b['size_bytes'], origin='V268_owned_source'))
    # The only old-tree reads are this explicit dependency whitelist.
    for rel in DEPENDENCIES:
        b = digest(OLD_SOURCE / rel)
        assert b['size_bytes'] < LIMIT
        planned.append(dict(source_path=b['path'], package_path=(Path('artifacts') / NAME / 'dependencies' / OLD_NAME / rel).as_posix(),
                            sha256=b['sha256'], size_bytes=b['size_bytes'], origin='V267_exact_dependency'))
    rgb_excluded = digest(SOURCE / EXCLUDED_RGB)
    assert rgb_excluded == protocol['source_bindings']['RGB256']
    assert len({r['package_path'] for r in planned}) == len(planned)
    safe(TARGET).mkdir()
    for record in planned:
        copy_exact(record['source_path'], TARGET / record['package_path'], record)
    readme = digest(SOURCE / 'PUBLICATION_README_KO.md')
    copy_exact(readme['path'], TARGET / 'README_KO.md', readme)
    readme_record = dict(source_path=readme['path'], package_path='README_KO.md', sha256=readme['sha256'],
                         size_bytes=readme['size_bytes'], origin='wrapper_readme_from_V268_source')
    # Detect any concurrent source mutation, including a newly added owned file.
    assert [p.relative_to(SOURCE).as_posix() for p in owned_files(SOURCE)] == relative_source
    for record in planned + [readme_record]:
        current = digest(record['source_path'])
        assert (current['sha256'], current['size_bytes']) == (record['sha256'], record['size_bytes']), record['source_path']
    html = [TARGET / r['package_path'] for r in planned if Path(r['package_path']).suffix.lower() in ('.html', '.htm')]
    links = verify_links(TARGET, html)
    write_json(TARGET / 'LOCAL_LINK_VERIFICATION.json', links)
    assert links['PASS'], links['failures']
    link_binding = digest(TARGET / 'LOCAL_LINK_VERIFICATION.json')
    manifest = dict(completed=True, PASS=True, schema='V268_FROZEN_PUBLICATION_BYTE_COPY_V1',
        source=str(SOURCE), package=str(TARGET), payload_root=payload.relative_to(TARGET).as_posix(),
        included=planned + [readme_record], included_file_count=len(planned) + 1,
        payload_file_count=len(planned), payload_total_bytes=sum(r['size_bytes'] for r in planned),
        total_included_bytes=sum(r['size_bytes'] for r in planned) + readme_record['size_bytes'],
        largest_payload_file_bytes=max(r['size_bytes'] for r in planned), exact_V267_dependencies=len(DEPENDENCIES),
        source_and_copy_all_SHA_and_size_verified=True, source_snapshot_unchanged_after_copy=True,
        excluded=[dict(source_binding=rgb_excluded, exclusion='394MB whole-SYN RGB256 array not duplicated; stored source SHA/size verified.')],
        external_large_inputs=[dict(source_binding=protocol['source_bindings']['tokensCS'], copied=False,
                                    binding_scope='Frozen protocol metadata only; 1.05GB SYN encoder features not read or duplicated.'),
                               dict(source_binding=protocol['current_CS_tokens'], copied=False,
                                    binding_scope='Frozen protocol metadata only; 234MB current encoder features not read or duplicated.')],
        full_original_1003_RGB_M4_corpus_not_duplicated=True,
        six_exact_selected_SYN_RGB_M4_pairs_included=True,
        generic_CLIP_weights_and_official_source_repository_not_duplicated=True,
        generic_CLIP_provenance_included='artifacts/' + NAME + '/dependencies/' + OLD_NAME + '/pretrained/DOWNLOAD_PROVENANCE.json',
        original_corpus_bindings_included='artifacts/' + NAME + '/dependencies/' + OLD_NAME + '/data_audit',
        generated_wrapper_files=[dict(package_path='LOCAL_LINK_VERIFICATION.json', sha256=link_binding['sha256'], size_bytes=link_binding['size_bytes'])],
        new_fits_or_model_forwards_or_Git_mutations=0, elapsed_seconds=time.monotonic() - started)
    write_json(TARGET / 'RESULTS_PACKAGE_MANIFEST.json', manifest)
    print(json.dumps(dict(completed=True, PASS=True, package=str(TARGET), payload_files=len(planned),
                         payload_bytes=manifest['payload_total_bytes'], local_links=links['checked_local_links'],
                         manifest=digest(TARGET / 'RESULTS_PACKAGE_MANIFEST.json')), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
