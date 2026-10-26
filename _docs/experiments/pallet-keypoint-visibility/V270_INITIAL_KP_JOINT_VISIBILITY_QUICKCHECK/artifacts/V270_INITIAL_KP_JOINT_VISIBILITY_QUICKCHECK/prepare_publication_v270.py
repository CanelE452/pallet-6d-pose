"""Publish exact V270 pilot artifacts and preserve scientific failures; no model fitting."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,time
NAME='V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK'
SOURCE=Path(__file__).absolute().parent
TARGET=Path('/home/minjae/Documents/pallet-visibility-publication-20261024/_docs/experiments/pallet-keypoint-visibility')/NAME
LIMIT=100*1024*1024
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

def write_json(path, obj):
    with safe(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')

def copy_exact(source, destination, expected):
    source, destination = safe(source), safe(destination)
    assert source.stat().st_size < LIMIT, ('Included file must be <100MiB', str(source))
    safe(destination.parent).mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = digest(destination)
        assert (actual['sha256'], actual['size_bytes']) == (expected['sha256'], expected['size_bytes']), str(destination)
        return
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

def selected_files(directory, excluded):
    selected=[]
    for entry in sorted(os.scandir(safe(directory)),key=lambda x:x.name):
        p=safe(Path(entry.path));rel=p.relative_to(SOURCE).as_posix()
        if rel.split('/')[0]=='ultralytics_runtime':
            excluded.append(dict(relative_path=rel,reason='Runtime-only Ultralytics preferences; no scientific inputs.'))
            continue
        if rel in ('syn_initial_features_v1/pyramid_P3.npy','syn_initial_features_v1/pyramid_P4.npy','syn_initial_features_v1/pyramid_P5.npy'):
            excluded.append(dict(relative_path=rel,size_bytes=p.stat().st_size,reason='Full1003 FP32 frozen neck corpus; source SHA in syn_initial_features_v1/COMPLETE.json; five-frame full pyramid fixtures included.'))
            continue
        if p.is_dir():selected.extend(selected_files(p,excluded))
        else:
            assert p.is_file() and p.stat().st_size<LIMIT,str(p)
            selected.append((rel,digest(p)))
    return selected

def main():
    safe(TARGET).mkdir(exist_ok=True)
    payload=safe(TARGET/'artifacts'/NAME);payload.mkdir(parents=True,exist_ok=True)
    excluded=[];chosen=selected_files(SOURCE,excluded)
    def published_rel(rel):
        return rel+'.txt' if rel=='slot_prior_diagnosis_v1/PRE_SLOT_DIAGNOSIS_SYNDEV_READOUT_KO.html' else rel
    for rel,expected in chosen:copy_exact(SOURCE/rel,payload/published_rel(rel),expected)
    links=verify_links(payload,sorted(payload.rglob('*.html')))
    write_json(TARGET/'HTML_LOCAL_LINK_CHECK.json',links)
    assert links['PASS'],links['failures']
    md=(payload/'PUBLICATION_README_KO.md').read_text()
    import re
    md=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:'](artifacts/'+NAME+'/'+m.group(1)+')',md)
    with safe(TARGET/'README_KO.md').open('x',encoding='utf-8') as f:f.write(md)
    manifest=dict(PASS=True,completed=True,experiment=NAME,source_root=str(SOURCE),payload='artifacts/'+NAME,
      exact_source_files=[dict(relative_path=published_rel(rel),source_relative_path=rel,sha256=b['sha256'],size_bytes=b['size_bytes']) for rel,b in chosen],
      exact_source_copies=len(chosen),exact_source_payload_bytes=sum(b['size_bytes'] for _,b in chosen),excluded=excluded,
      included_original_checkpoint=True,excluded_corpus_pyramids_only=True,
      original_scientific_files_modified_for_publication=False,
      original_failures_preserved=True,head_replay_contract='v2 explicit float32epsilon; original failure preserved',
      canonical_evaluation='evaluation_v4',canonical_visual='visual_v3')
    write_json(TARGET/'PUBLICATION_MANIFEST.json',manifest)
    write_json(SOURCE/'PUBLICATION_PREPARATION_RECEIPT.json',dict(PASS=True,target=str(TARGET),manifest_sha256=digest(TARGET/'PUBLICATION_MANIFEST.json')['sha256'],source_files=len(chosen),payload_bytes=manifest['exact_source_payload_bytes'],local_links=links['checked_local_links']))
    print(json.dumps(dict(PASS=True,files=len(chosen),MB=manifest['exact_source_payload_bytes']/1e6,links=links['checked_local_links'])))
if __name__=='__main__':main()
