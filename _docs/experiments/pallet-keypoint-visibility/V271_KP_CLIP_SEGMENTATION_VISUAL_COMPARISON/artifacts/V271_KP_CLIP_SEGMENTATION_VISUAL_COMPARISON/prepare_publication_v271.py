"""Publish exact V271 saved-output visualization; no model execution."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,json,os,posixpath,re
NAME='V271_KP_CLIP_SEGMENTATION_VISUAL_COMPARISON'
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

def main():
    assert not TARGET.exists(),str(TARGET)
    safe(TARGET).mkdir();payload=safe(TARGET/'artifacts'/NAME);payload.mkdir(parents=True)
    chosen=[]
    for directory,dirs,names in os.walk(safe(SOURCE)):
        for name in dirs:safe(Path(directory)/name)
        for name in sorted(names):
            p=safe(Path(directory)/name);assert p.is_file() and p.stat().st_size<LIMIT
            rel=p.relative_to(SOURCE).as_posix();chosen.append((rel,digest(p)))
    for rel,info in chosen:copy_exact(SOURCE/rel,payload/rel,info)
    links=verify_links(payload,sorted(payload.rglob('*.html')))
    write_json(TARGET/'HTML_LOCAL_LINK_CHECK.json',links);assert links['PASS'],links['failures']
    text=(payload/'PUBLICATION_README_KO.md').read_text()
    text=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:'](artifacts/'+NAME+'/'+m.group(1)+')',text)
    (TARGET/'README_KO.md').write_text(text)
    manifest=dict(PASS=True,completed=True,experiment=NAME,source_root=str(SOURCE),payload='artifacts/'+NAME,
       exact_source_files=[dict(relative_path=rel,sha256=x['sha256'],size_bytes=x['size_bytes']) for rel,x in chosen],
       exact_source_copies=len(chosen),payload_bytes=sum(x['size_bytes'] for _,x in chosen),new_training=0,new_model_forward=0,new_PnP=0,new_threshold_search=0,
       visualization_only=True,real_pixel_segmentation_GT=False,scientific_masks='Frozen 32x32 logits, same original pixel-center bilinear, sigmoid, threshold0.5')
    write_json(TARGET/'PUBLICATION_MANIFEST.json',manifest)
    print(json.dumps(dict(PASS=True,files=len(chosen),bytes=manifest['payload_bytes'],HTML_links=links['checked_local_links'])))
if __name__=='__main__':main()
