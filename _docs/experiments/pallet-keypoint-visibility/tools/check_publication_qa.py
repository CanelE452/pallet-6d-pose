#!/usr/bin/env python3
"""Manifest-only publication QA. Standard library; matched secrets never printed.

  python3 check_publication_qa.py --self-test
  python3 check_publication_qa.py --root PUBLICATION_ROOT --manifest MANIFEST.json

Run only after the staging trigger. --portable-prefix defaults to portable/;
--portable-path can name additional root-relative portable Markdown documents.
The sibling verify_published_results.py supplies only the manifest format adapter.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
from html.parser import HTMLParser
import argparse
import codecs
import gzip
import json
import posixpath
import re
import sys
import tempfile
from urllib.parse import unquote, urlsplit

from verify_published_results import manifest_records, record_storage

LIMIT = 100 * 1024 * 1024
TEXT_SUFFIXES = {'.md','.markdown','.rst','.txt','.json','.jsonl','.csv','.tsv','.html','.htm','.py','.js','.ts','.tsx','.jsx','.css','.yaml','.yml','.toml','.sh','.log','.xml','.svg','.ipynb','.ini','.cfg'}
PATTERNS = {
    'PRIVATE_KEY_HEADER': rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
    'AWS_ACCESS_KEY_ID': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'AWS_SECRET_KEY_ASSIGNMENT': rb'(?i:aws[_ -]?secret[_ -]?access[_ -]?key)[\s\x22\x27:=]{1,32}[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])',
    'GITHUB_TOKEN': rb'\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{20,255})\b',
    'OPENAI_API_KEY': rb'\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,255}\b',
    'SLACK_TOKEN': rb'\bxox[baprs]-[A-Za-z0-9-]{20,255}\b',
    'SLACK_WEBHOOK': rb'https://hooks\.slack\.com/services/T[A-Z0-9]{8,16}/B[A-Z0-9]{8,16}/[A-Za-z0-9]{20,64}',
}
COMPILED = {key: re.compile(value) for key,value in PATTERNS.items()}
WINDOW = 1024  # larger than the maximum complete supported credential pattern


def require(value, message):
    if not value: raise ValueError(message)


def forbidden_name(path):
    return not isinstance(path,str) or 'v112' in path.lower()


def safe_path(root, value):
    # This name check precedes resolve, stat, is_file, open, or exists.
    require(not forbidden_name(value), 'FORBIDDEN_PATH_PRECHECK')
    p=PurePosixPath(value)
    require(bool(value) and not p.is_absolute() and '..' not in p.parts and '\\' not in value, 'UNSAFE_STORAGE_PATH')
    result=(root/value).resolve()
    require(result.is_relative_to(root), 'OUTSIDE_PUBLICATION_ROOT')
    return result


def issue(issues, path, rule, count=1):
    issues.append(dict(path=path if isinstance(path,str) else '<invalid-path>',rule=rule,count=int(count)))


def selected(manifest):
    result=[]
    for record in manifest_records(manifest):
        status=record.get('status','')
        require(not status.startswith('CANDIDATE_'), 'INVENTORY_NOT_PUBLISHED')
        logical=record.get('published_relative',record.get('logical_path',record.get('path')))
        if logical is None:
            require(not status.startswith('PUBLISHED_'), 'PUBLISHED_RECORD_WITHOUT_PATH')
            continue
        result.append((logical,record_storage(record)))
    return result


def physical_specs(spec):
    if isinstance(spec,str): return [dict(path=spec)]
    require(isinstance(spec,dict), 'INVALID_STORAGE_SPEC')
    if 'parts' in spec:
        require(spec.get('format')=='concatenated-bytes-v1' and bool(spec['parts']), 'INVALID_PARTS_DESCRIPTOR')
        return spec['parts']
    return [spec]


def stored_name(spec):
    return spec.get('published_path',spec.get('path'))


def iter_decoded(root, spec):
    """Yield bytes from selected text without assembling a large logical file."""
    if isinstance(spec,str): spec=dict(path=spec)
    if 'parts' in spec:
        require(spec.get('format')=='concatenated-bytes-v1','INVALID_PARTS_DESCRIPTOR')
        for part in spec['parts']: yield from iter_decoded(root,part)
        return
    name=stored_name(spec); path=safe_path(root,name)
    if name.endswith(('.parts','.parts.json')):
        descriptor=json.loads(path.read_text(encoding='utf-8'))
        yield from iter_decoded(root,descriptor)
        return
    compression=spec.get('compression','gzip' if name.endswith('.gz') else 'none')
    require(compression in ('none','gzip'),'INVALID_COMPRESSION')
    opener=gzip.open if compression=='gzip' else open
    with opener(path,'rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''): yield block


def scan_credentials(chunks):
    counts=Counter(); pending=b''
    for chunk in chunks:
        data=pending+chunk
        cutoff=max(0,len(data)-WINDOW)
        # Delay all matches whose start lies in the overlap. Each match counted once.
        for rule,pattern in COMPILED.items():
            counts[rule]+=sum(match.start()<cutoff for match in pattern.finditer(data))
        pending=data[cutoff:]
    for rule,pattern in COMPILED.items(): counts[rule]+=sum(1 for _ in pattern.finditer(pending))
    return {key:value for key,value in counts.items() if value}


class HTMLTargets(HTMLParser):
    def __init__(self): super().__init__(); self.targets=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if value and ((tag=='img' and key=='src') or (tag=='a' and key=='href')):self.targets.append(value)


def markdown_targets(text):
    # Ignore fenced and inline code. This validates links, not Markdown rendering.
    lines=[]; fence=None
    for line in text.splitlines():
        match=re.match(r'^\s*(`{3,}|~{3,})',line)
        if match:
            mark=match.group(1)[0]
            if fence is None:fence=mark
            elif fence==mark:fence=None
            continue
        if fence is None:lines.append(line)
    clean=re.sub(r'`+[^`\n]*`+','', '\n'.join(lines))
    targets=[]; definitions={}
    for match in re.finditer(r'^\s{0,3}\[([^\]\n]+)\]:\s*(<[^>]+>|\S+)',clean,re.M):
        definitions[match.group(1).strip().lower()]=match.group(2).strip('<>')
    for match in re.finditer(r'!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[\x22\x27][^\n]*?[\x22\x27])?\s*\)',clean):
        targets.append(match.group(1).strip('<>'))
    missing=[]
    for match in re.finditer(r'!?\[([^\]\n]+)\]\[([^\]\n]*)\]',clean):
        name=(match.group(2) or match.group(1)).strip().lower()
        if name not in definitions:missing.append(name)
    targets.extend(definitions.values())
    parser=HTMLTargets();parser.feed(clean);targets.extend(parser.targets)
    return targets,len(missing)


def check_markdown(root, logical, text, issues):
    targets,undefined=markdown_targets(text)
    if undefined:issue(issues,logical,'UNDEFINED_MARKDOWN_REFERENCE',undefined)
    checked=0
    for target in targets:
        parsed=urlsplit(target)
        if parsed.scheme in ('http','https','mailto','tel','data') or target.startswith('//') or target.startswith('#'):
            continue  # no external request or anchor certification
        if parsed.scheme or parsed.path.startswith('/') or '\\' in parsed.path:
            issue(issues,logical,'NONPORTABLE_ABSOLUTE_LOCAL_LINK');continue
        if not parsed.path:continue
        local=posixpath.normpath(str(PurePosixPath(logical).parent/unquote(parsed.path)))
        if forbidden_name(local):
            issue(issues,logical,'FORBIDDEN_LINK_PATH_PRECHECK');continue
        try:path=safe_path(root,local)
        except ValueError:issue(issues,logical,'RELATIVE_LINK_OUTSIDE_PUBLICATION_ROOT');continue
        checked+=1
        if not path.exists():issue(issues,logical,'MISSING_RELATIVE_LINK_OR_IMAGE')
    return checked


def run_qa(root, manifest, portable_prefixes=('portable/',), portable_paths=()):
    root=Path(root).resolve();issues=[];entries=selected(manifest);stored={};text_count=0;link_count=0
    for logical,spec in entries:
        if forbidden_name(logical):issue(issues,logical,'FORBIDDEN_PATH_PRECHECK');continue
        safe=True
        for part in physical_specs(spec):
            name=stored_name(part)
            if forbidden_name(name):issue(issues,name,'FORBIDDEN_PATH_PRECHECK');safe=False;continue
            try:
                path=safe_path(root,name)
                if name not in stored:
                    size=path.stat().st_size;stored[name]=size
                    if size>=LIMIT:issue(issues,name,'STORED_FILE_NOT_BELOW_100_MIB')
            except (OSError,ValueError):issue(issues,name,'STAGED_FILE_PATH_OR_STAT_FAILED');safe=False
        if not safe:continue
        suffix=PurePosixPath(logical).suffix.lower()
        if suffix not in TEXT_SUFFIXES:continue
        text_count+=1
        try:
            matches=scan_credentials(iter_decoded(root,spec))
            for rule,count in matches.items():issue(issues,logical,rule,count)
            portable=(logical in portable_paths or any(logical.startswith(prefix) for prefix in portable_prefixes))
            if portable and suffix in ('.md','.markdown'):
                # Portable documentation is small; large original archives stay streamed.
                text=b''.join(iter_decoded(root,spec)).decode('utf-8')
                link_count+=check_markdown(root,logical,text,issues)
        except (OSError,ValueError,UnicodeError):issue(issues,logical,'TEXT_DECODE_OR_SCAN_FAILED')
    return dict(completed=True,PASS=not issues,selected_logical_files=len(entries),unique_stored_files_checked=len(stored),
        text_logical_files_scanned=text_count,portable_local_targets_checked=link_count,stored_file_limit_exclusive_bytes=LIMIT,
        credential_value_or_context_output=False,issues=issues,
        scope='Staged manifest-only QA. Credential-shaped matches are possible secrets, not active validation. Archived local absolute provenance is not a portability error. No original source discovery, model/data decoding, or V112 access.')


def self_tests():
    tests=[]
    # Synthetic values are assembled so the QA source does not contain fake credential literals.
    examples=[b'-----BEGIN '+b'PRIVATE KEY-----',b'AK' + b'IA' + b'A'*16,
              b'aws_secret_access_key='+b'Q'*40,b'gh' + b'p_'+b'a'*36,
              b'sk-'+b'proj-'+b'Z'*24,b'xo' + b'xb-'+b'1'*25,
              b'https://hooks.'+b'slack.com/services/T'+b'A'*8+b'/B'+b'B'*8+b'/'+b'C'*24]
    found=scan_credentials([b'\n'.join(examples)])
    require(set(found)==set(PATTERNS) and all(v==1 for v in found.values()),'ALL_RULES_SYNTHETIC_COUNTS')
    tests.append('All_seven_possible_credential_formats_counts_only')
    payload=b'prefix '+examples[3]+b' suffix'
    require(scan_credentials([payload[:18],payload[18:]])=={'GITHUB_TOKEN':1},'CROSS_CHUNK_ONCE')
    tests.append('Credential_cross_chunk_boundary_exact_once')
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'portable').mkdir();(root/'assets').mkdir()
        (root/'assets/p.png').write_bytes(b'invented PNG placeholder')
        doc='# Demo\n![raw](../assets/p.png)\n[web](https://example.test)\n[part](#demo)\n[ref][ok]\n[ok]: ../assets/p.png\n```md\n[ignored](/home/old)\n```\nOriginal provenance /home/old/file is plain text.\n'
        (root/'portable/index.md').write_text(doc)
        (root/'archive.txt').write_text('[original](/home/old/file)\n')
        manifest=dict(entries=[dict(published_relative=p,status='PUBLISHED_RAW',published_sha256='unused',gzipchunks=None) for p in ('portable/index.md','archive.txt','assets/p.png')])
        result=run_qa(root,manifest)
        require(result['PASS'] and result['portable_local_targets_checked']==2,'PORTABLE_AND_PROVENANCE')
        tests.append('Relative_images_references_external_code_and_archived_absolute_provenance')
        bad=[];check_markdown(root,'portable/index.md','![missing](none.png)\n[x][missing]\n[x](/home/notportable)',bad)
        require({i['rule'] for i in bad}=={'MISSING_RELATIVE_LINK_OR_IMAGE','UNDEFINED_MARKDOWN_REFERENCE','NONPORTABLE_ABSOLUTE_LOCAL_LINK'},'PORTABLE_FAILURES')
        tests.append('Missing_image_undefined_reference_nonportable_link')
        z=root/'too-big.bin'
        with z.open('wb') as handle:handle.truncate(LIMIT)
        size_manifest=dict(entries=[dict(published_relative='too-big.bin',status='PUBLISHED_RAW',published_sha256='unused',gzipchunks=None)])
        require(run_qa(root,size_manifest)['issues']==[dict(path='too-big.bin',rule='STORED_FILE_NOT_BELOW_100_MIB',count=1)],'EXCLUSIVE_SIZE_LIMIT')
        tests.append('Exactly100MiB_rejected_without_reading_binary')
        forbidden=dict(entries=[dict(published_relative='V112/nonexistent.txt',status='PUBLISHED_RAW',published_sha256='unused',gzipchunks=None)])
        result=run_qa(root,forbidden)
        require(result['unique_stored_files_checked']==0 and result['issues'][0]['rule']=='FORBIDDEN_PATH_PRECHECK','V112_PRECHECK_NO_STAT')
        tests.append('Forbidden_name_rejected_before_stat_open_or_resolve')
        (root/'part0.gz').write_bytes(gzip.compress(examples[0][:10],mtime=0));(root/'part1.gz').write_bytes(gzip.compress(examples[0][10:],mtime=0))
        parts=dict(format='concatenated-bytes-v1',parts=[dict(path='part0.gz',compression='gzip'),dict(path='part1.gz',compression='gzip')],content_sha256='fixture')
        cm=dict(entries=[dict(published_relative='chunked.txt',status='PUBLISHED_GZIP_CHUNKS',source_sha256='fixture',gzipchunks=parts)])
        result=run_qa(root,cm)
        require(result['issues']==[dict(path='chunked.txt',rule='PRIVATE_KEY_HEADER',count=1)],'CHUNKED_CREDENTIAL')
        serialized=json.dumps(result)
        require(examples[0].decode() not in serialized and 'PRIVATE KEY' not in serialized,'SECRET_NOT_EXPOSED')
        tests.append('Gzip_parts_streaming_and_no_match_value_or_context_in_report')
    return dict(completed=True,PASS=True,invented_only=True,checks=tests,count=len(tests),actual_staged_package_reads=0)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path);p.add_argument('--manifest')
    p.add_argument('--portable-prefix',action='append');p.add_argument('--portable-path',action='append',default=[])
    p.add_argument('--report',type=Path);a=p.parse_args()
    if a.self_test:
        require(a.root is None and a.manifest is None,'SELF_TEST_ONLY');answer=self_tests()
    else:
        require(a.root is not None and a.manifest is not None,'STAGED_ROOT_AND_MANIFEST_REQUIRED')
        root=a.root.resolve();manifest=json.loads(safe_path(root,a.manifest).read_text(encoding='utf-8'))
        answer=run_qa(root,manifest,tuple(a.portable_prefix or ['portable/']),tuple(a.portable_path))
    text=json.dumps(answer,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.report:
        with a.report.open('x',encoding='utf-8') as handle:handle.write(text)
    print(text,end='');return 0 if answer['PASS'] else 1


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError,UnicodeError):
        # Deliberately do not print exception messages or matched source text.
        print(json.dumps(dict(completed=False,PASS=False,issues=[dict(path='<input>',rule='INPUT_OR_SCHEMA_FAILURE',count=1)],credential_value_or_context_output=False)))
        sys.exit(1)
