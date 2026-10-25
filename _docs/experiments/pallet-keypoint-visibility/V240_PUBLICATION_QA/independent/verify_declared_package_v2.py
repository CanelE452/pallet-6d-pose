"""선언 package byte/gzip/link/status 검산. SOURCE pointer/배열 숫자/model load 없음."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import posixpath
import stat
import traceback
from urllib.parse import unquote, urlsplit

INERT_HISTORY_PREFIXES = ('../V239_', '../V236_')

def need(ok, message):
    if not ok: raise ValueError(message)

def safe(path):
    p=Path(path);text=str(p)
    need(p.is_absolute() and '..' not in p.parts and not any(s in text.casefold()
         for s in ('v112','__pycache__','.pyc')), 'lexical path rejection before FS')
    for q in reversed(p.parents):
        need(stat.S_ISDIR(os.lstat(q).st_mode), 'parent symlink/non-directory')
    need(stat.S_ISREG(os.lstat(p).st_mode), 'regular no-symlink leaf')
    return p

def read(path):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def digest(stream):
    h=hashlib.sha256();n=0
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);n+=len(block)
    return h.hexdigest(),n

def binding(path):
    b=read(path);return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def write_json(path,obj):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def relative(name):
    p=PurePosixPath(name)
    need(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name
         and not any(s in name.casefold() for s in ('v112','__pycache__','.pyc')), 'unsafe package leaf')
    return p

def get_path(value,keys):
    for k in keys:value=value[k]
    return value

def verify_package(e):
    package=Path(e['package']);manifest_name=e.get('manifest_name','MANIFEST.json')
    def leaf(name):return safe(package/relative(name))
    mb=read(leaf(manifest_name));sha=hashlib.sha256(mb).hexdigest()
    need(sha==e['manifest_sha256'],'manifest exact pin')
    m=json.loads(mb);files=m['files'];declared={}
    for item in files:
        name=item['path'];relative(name);need(name not in declared,'duplicate manifest leaf');declared[name]=item
    need(len(files)==m['file_count']==e['expected_file_count'],'declared count')
    total=0;verified=[]
    for name,item in declared.items():
        with os.fdopen(os.open(leaf(name),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:actual=digest(f)
        need(actual==(item['sha256'],item['bytes']),'declared bytes '+name)
        total+=actual[1];verified.append(dict(path=name,sha256=actual[0],size_bytes=actual[1]))
    need(total==m['total_bytes']==e['expected_total_bytes'],'declared byte total')
    def data(name):
        need(name in declared,'undeclared content access')
        return read(leaf(name))
    def meta(name):return json.loads(data(name))
    sm=meta(e.get('source_map','PUBLICATION_SOURCE_MAP.json'));mapped=set();gz=[]
    for item in sm['files']:
        name=item['path'];need(name in declared and name not in mapped,'source map closure');mapped.add(name)
        if item['encoding']=='identity':actual=declared[name]['sha256'],declared[name]['bytes']
        else:
            need(item['encoding']=='gzip-lossless' and name.endswith('.gz'),'known lossless encoding')
            with os.fdopen(os.open(leaf(name),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f, gzip.GzipFile(fileobj=f) as z:actual=digest(z)
            gz.append(dict(path=name,decoded_sha256=actual[0],decoded_bytes=actual[1]))
        need(actual==(item['source_sha256'],item['source_bytes']),'source byte provenance')
    need({x['path'] for x in gz}=={x for x in declared if x.endswith('.gz')},'all gzip closure')
    readme_name=e.get('readme','README_KO.md')
    readme=data(readme_name).decode('utf-8');local=[];inert=[];document_links=[]
    historical=e['inert_historical_links']
    need(len(historical)==len(set(historical)) and all(x.startswith(INERT_HISTORY_PREFIXES)
         for x in historical),'only explicit V239/V236 inert history')
    documents=[readme_name]+e.get('additional_documents',[])
    need(len(documents)==len(set(documents)),'unique declared documents')
    for doc in documents:
        text=data(doc).decode('utf-8')
        links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        if doc.lower().endswith(('.html','.htm')):
            links += re.findall(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']",text,flags=re.I)
        count=0;ignored=[]
        for link in links:
            if link in historical:inert.append(link);ignored.append(link);continue
            parts=urlsplit(link)
            if parts.scheme or parts.netloc:inert.append(link);ignored.append(link);continue
            if not parts.path:continue
            target=posixpath.normpath(posixpath.join(str(PurePosixPath(doc).parent),unquote(parts.path)))
            relative(target)
            need(target in declared or target==manifest_name,'document local target not declared '+doc+': '+target)
            local.append(target);count+=1
        document_links.append(dict(path=doc,local_references=count,inert_links=ignored))
    need(set(historical)<={x for x in inert},'declared historical links present')
    checks=e['status_checks'];need(len(checks)>=6,'Root must bind bounded status evidence')
    actual_status=[]
    for check in checks:
        value=get_path(meta(check['path']),check['keys'])
        need(type(value) is type(check['expected']) and value==check['expected'],'bounded status '+check['meaning'])
        actual_status.append(dict(meaning=check['meaning'],path=check['path'],keys=check['keys'],value=value))
    meanings={x['meaning'] for x in checks}
    need({'canonical_features_completed','both1000_completed','independent_QA_PASS',
          'SOURCE_gate_FAIL','not_adopted','no_new_current','no_PnP'}<=meanings,'required scientific status axes')
    for text in e.get('readme_required_strings',[]):need(text in readme,'README status wording')
    omitted=e['omitted_duplicate_feature']
    need(omitted['basename']=='NODE_PIXEL_FEATURES.npy' and omitted['sha256'] and omitted['size_bytes']>0,
         'duplicate feature omission binding metadata')
    need(not any(PurePosixPath(x).name=='NODE_PIXEL_FEATURES.npy' for x in declared),'duplicate cache omitted')
    plan_omitted=meta('PUBLICATION_PLAN.json')['excluded_large_cache_bindings'][0]
    status_omitted=meta('STATUS.json')['duplicate_NODE_PIXEL_FEATURES_excluded']
    for key in ('path','sha256','size_bytes'):
        need(plan_omitted[key]==status_omitted[key]==omitted[key],'omitted binding metadata closure')
    return dict(completed=True,PASS=True,manifest_sha256=sha,manifest_size_bytes=len(mb),
        package=str(package),declared_files_checked=len(files),total_bytes_excluding_manifest=total,
        source_map_files_checked=len(mapped),gzip_decoded_verified=gz,README_local_references=len(local),
        README_unique_local_targets=len(set(local)),document_link_QA=document_links,inert_links_not_followed=inert,bounded_status_QA=actual_status,
        omitted_duplicate_feature=omitted,verified_files=verified,
        limits=['Arrays/checkpoints/images opaque bytes only; no numeric decode, model load, forward or fit.',
                'gzip decoded bytes hashed only; no source JSONL semantic interpretation.',
                'SOURCE exposed DEV failure is not current success; no model/data/architecture choice.',
                'Source-map absolute pointers, omitted cache binding, external links and history were not followed.',
                'No original SOURCE/GT/current/held reads, Git or network.'])

def main(entry_path):
    e=json.loads(read(entry_path))
    need(e.get('approved_declared_publication_verification') is True and e.get('automatic_retry') is False,
         'separate exact package entry required before package access')
    need(binding(__file__)['sha256']==e['verifier_sha256'],'verifier code pin')
    output=Path(e['receipt_path']);need(output.parent==Path(__file__).absolute().parent,'own receipt folder')
    try:
        result=verify_package(e);result.update(entry=binding(entry_path),verifier=binding(__file__))
        write_json(output,result);print(json.dumps(dict(completed=True,PASS=True,receipt=binding(output))))
    except BaseException:
        write_json(output.with_name('FAILED_EXECUTION.json'),dict(completed=False,PASS=False,entry=binding(entry_path),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);main(p.parse_args().entry)
