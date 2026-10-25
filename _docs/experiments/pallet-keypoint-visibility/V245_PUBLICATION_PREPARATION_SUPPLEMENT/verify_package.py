"""V245 declared-package QA; opaque payloads, metadata, portable links and saved CSV math."""
import argparse,gzip,hashlib,json,os,posixpath,re,stat,subprocess,sys,traceback
from html.parser import HTMLParser
from pathlib import Path,PurePosixPath
from urllib.parse import unquote,urlsplit

H=Path(__file__).absolute().parent; R=H.parent
PACKAGE=R/'V245_PUBLICATION_PACKAGE'
REPLAY=R/'V245_SOURCE_REPORT_C/report_saved_source.py'
REPLAY_SHA='6437c623dba13a2330b0d49b5320e31595c4a2346f656f80c25a773c377b248e'
STATUS={'final_outputs_ready':True,'pipeline_QA_PASS':True,'SOURCE_gate_PASS':False,
        'new_training_updates':1000,'current_or_held_payloads':0,'model_adopted':False}
FORBIDDEN=('v112','__pycache__','.pyc','current223','publicheld','public_held','private_evaluation','source_holdout','oldval')

def need(ok,message):
    if not ok:raise ValueError(message)

def lex(value):
    s=str(value);need(not any(x in s.casefold() for x in FORBIDDEN),'forbidden lexical path');return Path(s)

def safe(value):
    p=lex(value);need(p.is_absolute() and '..' not in p.parts,'absolute path without traversal')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'symlink/non-directory ancestor')
    need(stat.S_ISREG(os.lstat(p).st_mode),'symlink/nonregular leaf');return p

def raw(p):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def digest(p,decoded=False):
    h=hashlib.sha256();n=0
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        stream=gzip.GzipFile(fileobj=f) if decoded else f
        try:
            for b in iter(lambda:stream.read(1048576),b''):h.update(b);n+=len(b)
        finally:
            if decoded:stream.close()
    return h.hexdigest(),n

def rel(s):
    p=PurePosixPath(s);lex(s)
    need(s and not p.is_absolute() and '..' not in p.parts and str(p)==s,'canonical relative leaf');return s

def bind(p):
    h,n=digest(p);return {'path':str(p),'sha256':h,'size_bytes':n}

def write(name,obj):
    p=H/name;lex(p)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'output ancestor')
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return bind(p)

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ('href','src') and v)

def closure(files):
    # Inspect only names under manifest-derived directories; unknown names fail before child stat/read.
    tree={'.':set()}
    for name in [*files,'MANIFEST.json']:
        parts=PurePosixPath(name).parts
        for i,part in enumerate(parts):
            parent='/'.join(parts[:i]) or '.';tree.setdefault(parent,set()).add(part)
            if i<len(parts)-1:tree.setdefault('/'.join(parts[:i+1]),set())
    for directory,names in tree.items():
        p=PACKAGE if directory=='.' else PACKAGE/directory;lex(p)
        for q in [*reversed(p.parents),p]:need(stat.S_ISDIR(os.lstat(q).st_mode),'package directory symlink')
        actual=os.listdir(p)
        for name in actual:lex(name)
        need(set(actual)==names,'undeclared/missing package member in '+directory)

def run(entry):
    e=json.loads(raw(entry))
    need(e['approved_declared_V245_package_QA'] is True,'Root actual package QA not approved')
    need(e['package']==str(PACKAGE) and e['parent_main']=='49560f404ae9d1b0d11ebcfe42a04a2b3df5c241','fixed package/parent')
    need(e['automatic_retry'] is False and e['leaf_limit_bytes']==95000000,'fixed operation limits')
    mraw=raw(PACKAGE/'MANIFEST.json');pin=hashlib.sha256(mraw).hexdigest()
    need(pin==e['manifest_sha256'],'exact manifest SHA');m=json.loads(mraw)
    need(m['schema']=='V245_SCENE_CF_SOURCE_TRIAL_PUBLICATION','manifest schema')
    files={rel(b['path']):b for b in m['files']}
    need(len(files)==len(m['files'])==m['file_count']==e['expected_file_count'],'exact file census')
    need(sum(b['bytes'] for b in files.values())==m['total_bytes']==e['expected_total_bytes'],'exact byte census')
    need('MANIFEST.json' not in files,'manifest self excluded');closure(files)
    for name,b in files.items():
        need(type(b['bytes']) is int and 0<=b['bytes']<95000000,'leaf strictly below 95MB '+name)
        need(digest(PACKAGE/name)==(b['sha256'],b['bytes']),'opaque SHA/size '+name)
    sm=json.loads(raw(PACKAGE/'PUBLICATION_SOURCE_MAP.json'))
    need(sm['absolute_pointers_are_provenance_only'] is True and sm['excluded_cache_leaf_access']==0 and
         sm['private_current_held_payloads']==0 and sm['old_frozen_sources_changed'] is False,'source-map scope flags')
    exclusions=json.loads(raw(PACKAGE/'LARGE_CACHE_EXCLUSIONS.json'))
    need(exclusions['saved_bindings_only_no_leaf_access'] is True,'inert exclusions')
    excluded={str(b['binding']['path']) for b in exclusions['files']};mapped=set();gz=[]
    for b in sm['files']:
        name=rel(b['path']);need(name in files and name not in mapped,'source-map leaf closure');mapped.add(name)
        source=str(b['source_path_provenance_only']);lex(source)
        need(source not in excluded,'excluded original binding is copied')
        expected=(b['source_sha256'],b['source_bytes'])
        if b['encoding']=='identity':need(expected==(files[name]['sha256'],files[name]['bytes']),'saved source-map identity '+name)
        elif b['encoding']=='gzip':
            need(digest(PACKAGE/name,True)==expected,'gzip decoded original binding '+name);gz.append(name)
        else:raise ValueError('unsupported source-map encoding')
    status=json.loads(raw(PACKAGE/'STATUS.json'))
    for k,v in STATUS.items():need(type(status[k]) is type(v) and status[k]==v,'bounded trial status '+k)
    need(m['scientific_status']==status and m['publication_is_not_model_success'] is True,'manifest/status authority')
    links=[]
    for doc in e['documents']:
        rel(doc);need(doc in files and doc.endswith(('.md','.html')),'declared document');text=raw(PACKAGE/doc).decode()
        if doc.endswith('.md'):refs=re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',text)
        else:p=Links();p.feed(text);refs=p.refs
        for ref in refs:
            u=urlsplit(ref)
            if u.scheme or u.netloc or not u.path:links.append({'document':doc,'reference':ref,'kind':'external_or_anchor_inert'});continue
            need(not u.path.startswith('/'),'absolute local user link')
            target=posixpath.normpath(posixpath.join(posixpath.dirname(doc),unquote(u.path)));lex(target)
            if target.startswith('../'):
                need(doc=='README_KO.md' and target in e['external_relative_links'],'unapproved historical link');kind='published_history_inert_NO_FS'
            else:need(target in files or target=='MANIFEST.json','broken declared local link '+target);kind='declared_internal'
            links.append({'document':doc,'reference':ref,'target':target,'kind':kind})
    replay=None
    if e['run_saved_arithmetic']:
        need(digest(REPLAY)[0]==REPLAY_SHA,'frozen arithmetic CODE pin')
        saved='artifacts/V245_SOURCE_REPORT_C/results_v1'
        for name in ('MANIFEST.json','SUMMARY.json','PAIRED_NATURAL7617.csv','CF_DIAGNOSTIC10812.csv'):need(saved+'/'+name in files,'saved arithmetic local input')
        cp=subprocess.run([sys.executable,'-B',str(REPLAY),'--verify-saved',str(PACKAGE/saved)],capture_output=True,text=True)
        need(cp.returncode==0,'saved arithmetic failed '+cp.stdout+cp.stderr);replay=json.loads(cp.stdout)
        need(replay=={'completed':True,'PASS':True,'stdlib_verify_saved':True,'natural_rows':7617,'CF_rows':10812,'current_admission':False},'saved arithmetic scope/results')
    result={'completed':True,'PASS':True,'manifest_sha256':pin,'package':str(PACKAGE),'declared_files':len(files),
            'total_bytes':m['total_bytes'],'all_leaf_bytes_below95000000':True,'manifest_path_closure_PASS':True,
            'source_map_identity_bindings_checked':len(mapped),'gzip_decoded_bindings':gz,'original_source_pointer_follow':0,
            'excluded_cache_leaf_stat_or_read':0,'scientific_status':status,'links':links,'saved_arithmetic':replay,
            'entry':bind(entry),'verifier_code':bind(Path(__file__)),'package_or_scientific_files_modified':False,
            'arrays_numeric_model_image_decode_Git_network_training_or_selection':0,
            'limits':'Package bytes and saved metadata identity only; original source leaf bytes were not reread. No scientific revalidation, model forward, threshold or candidate selection.'}
    print(json.dumps(write('REVIEW.json',result)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args()
    try:run(Path(a.entry))
    except BaseException:
        write('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'entry':str(a.entry),'traceback':traceback.format_exc(),'automatic_retry':False});raise
