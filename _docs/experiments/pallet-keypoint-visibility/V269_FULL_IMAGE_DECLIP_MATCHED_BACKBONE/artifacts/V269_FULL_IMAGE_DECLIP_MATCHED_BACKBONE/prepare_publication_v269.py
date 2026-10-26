"""Publish exact frozen V269 outputs; excludes official third-party trees and large features/weights."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,time
NAME='V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE'
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

def selected_files(directory,excluded):
    result=[]
    for entry in sorted(os.scandir(safe(directory)),key=lambda x:x.name):
        p=safe(Path(entry.path));rel=p.relative_to(SOURCE).as_posix()
        if rel.split('/')[0] in ('official_source','pretrained'):
            excluded.append({'relative_path':rel,'exclusion':'Official third-party source/large generic checkpoints excluded. Exact official URLs, commits and SHA are OFFICIAL_ASSETS_COMPLETE.json.'})
            continue
        if p.is_dir():result.extend(selected_files(p,excluded))
        elif p.suffix=='.npy' and ('feature' in rel or 'TOKENS' in rel.upper()):
            excluded.append({'relative_path':rel,'size_bytes':p.stat().st_size,'exclusion':'Large whole-corpus frozen feature array excluded; SHA is feature COMPLETE.json. Five-frame fixtures are included.'})
        else:
            assert p.is_file() and p.stat().st_size<LIMIT,(rel,p.stat().st_size)
            result.append(p)
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args();assert args.execute
    for rel in ('PROTOCOL_BEFORE_FIT.json','syn_training_v1/COMPLETE.json','CURRENT_PREDICTIONS_FROZEN.json','evaluation_v1/COMPLETE.json','ROOT_INDEPENDENT_METRICS_CHECK.json','fullframe_replay_v2/COMPLETE.json','FINAL_RESULT_SUMMARY.json'):
        j=json.loads(safe(SOURCE/rel).read_bytes());assert j.get('completed') and j.get('PASS',True),rel
    depseal=json.loads(safe(SOURCE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    db=depseal['current_query_inputs'];dp=safe(Path(db['path']));actual=digest(dp)
    assert (actual['sha256'],actual['size_bytes'])==(db['sha256'],db['size_bytes'])
    assert dp.name=='QUERIES1784_INPUT_ONLY.jsonl' and 'V267_FULL_IMAGE_CLIP_VISIBLE_MASK' in dp.parts
    assert not safe(TARGET).exists()
    excluded=[];files=selected_files(SOURCE,excluded)
    assert (SOURCE/'PUBLICATION_README_KO.md') in files
    planned=[]
    for p in files:
        b=digest(p);rel=p.relative_to(SOURCE).as_posix()
        planned.append(dict(source_path=str(p),package_path=(Path('artifacts')/NAME/rel).as_posix(),sha256=b['sha256'],size_bytes=b['size_bytes'],origin='V269_frozen_source'))
    planned.append(dict(source_path=str(dp),package_path=(Path('artifacts')/NAME/'dependencies/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/current_fullframe_inputs_v2/QUERIES1784_INPUT_ONLY.jsonl').as_posix(),sha256=db['sha256'],size_bytes=db['size_bytes'],origin='Exact_V267_query_replay_dependency'))
    safe(TARGET).mkdir()
    for b in planned:copy_exact(b['source_path'],TARGET/b['package_path'],b)
    b=digest(SOURCE/'PUBLICATION_README_KO.md');copy_exact(b['path'],TARGET/'README_KO.md',b)
    planned.append(dict(source_path=b['path'],package_path='README_KO.md',sha256=b['sha256'],size_bytes=b['size_bytes'],origin='wrapper_readme'))
    ex2=[];assert [str(x) for x in files]==[str(x) for x in selected_files(SOURCE,ex2)] and excluded==ex2
    for b in planned:
        now=digest(b['source_path']);assert (now['sha256'],now['size_bytes'])==(b['sha256'],b['size_bytes'])
    links=verify_links(TARGET,[TARGET/b['package_path']for b in planned if Path(b['package_path']).suffix=='.html'])
    write_json(TARGET/'LOCAL_LINK_VERIFICATION.json',links);assert links['PASS'],links['failures']
    write_json(TARGET/'RESULTS_PACKAGE_MANIFEST.json',dict(completed=True,PASS=True,source=str(SOURCE),package=str(TARGET),included=planned,included_file_count=len(planned),payload_total_bytes=sum(b['size_bytes']for b in planned),largest_file_bytes=max(b['size_bytes']for b in planned),excluded=excluded,all_SHA_and_sizes_verified=True,source_snapshot_unchanged=True,official_source_and_generic_weights_not_duplicated=True,metadata_bindings_with_original_absolute_paths_are_provenance_only=True,portable_replay_uses_relative_paths=True,new_fits_or_forwards_or_Git_mutations=0))
    print(json.dumps(dict(completed=True,PASS=True,files=len(planned),bytes=sum(b['size_bytes']for b in planned),package=str(TARGET),local_links=links['checked_local_links'])))

if __name__=='__main__':main()
