"""V246 declared-package QA; opaque payloads, metadata, conditional-label status and portable links."""
import argparse,gzip,hashlib,json,os,posixpath,re,stat,subprocess,sys,traceback
from html.parser import HTMLParser
from pathlib import Path,PurePosixPath
from urllib.parse import unquote,urlsplit

H=Path(__file__).absolute().parent; R=H.parent
PACKAGE=R/'V246_PUBLICATION_PACKAGE'
STATUS={'final_outputs_ready':True,'pipeline_QA_PASS':True,'same_q_probe_complete':True,
        'new_training_updates':0,'current_or_held_payloads':0,'model_adopted':False,
        'native_labels_created':0,'training_admission':False,'SOURCE_VAL_input_metadata_read_disclosed':True}
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
    need(e['approved_declared_V246_package_QA'] is True,'Root actual package QA not approved')
    need(e['package']==str(PACKAGE) and len(e['parent_main'])==40 and e['parent_main'].startswith('bff8cc846'),'fixed package/parent')
    need(e['automatic_retry'] is False and e['leaf_limit_bytes']==95000000,'fixed operation limits')
    mraw=raw(PACKAGE/'MANIFEST.json');pin=hashlib.sha256(mraw).hexdigest()
    need(pin==e['manifest_sha256'],'exact manifest SHA');m=json.loads(mraw)
    need(m['schema']=='V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT_PUBLICATION','manifest schema')
    files={rel(b['path']):b for b in m['files']}
    need(len(files)==len(m['files'])==m['file_count']==e['expected_file_count'],'exact file census')
    need(sum(b['bytes'] for b in files.values())==m['total_bytes']==e['expected_total_bytes'],'exact byte census')
    need('MANIFEST.json' not in files,'manifest self excluded');closure(files)
    for name,b in files.items():
        need(type(b['bytes']) is int and 0<=b['bytes']<95000000,'leaf strictly below 95MB '+name)
        need(digest(PACKAGE/name)==(b['sha256'],b['bytes']),'opaque SHA/size '+name)
    sm=json.loads(raw(PACKAGE/'PUBLICATION_SOURCE_MAP.json'))
    need(sm['absolute_pointers_are_provenance_only'] is True,'source-map paths are inert')
    prohibited_originals={
        str(R.parent/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl'),
        str(R.parent/'query_visibility_matched_query_scalar_20261023_v94/pallox_actual_initializer/FIXED100/ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl'),
        str(R.parent/'query_visibility_controlled_real_20261022_v66/public_pallet/SYMMETRIC_ALL50_V2/PAIR_EVIDENCE_MANIFEST.jsonl')}
    mapped=set();by_original={}
    for b in sm['files']:
        name=rel(b['path']);need(name in files and name not in mapped,'source-map leaf closure');mapped.add(name)
        source=str(b['source_path_provenance_only']);lex(source)
        need(source not in prohibited_originals,'original mixed metadata must not be copied')
        need(b['encoding']=='identity' and (b['source_sha256'],b['source_bytes'])==(files[name]['sha256'],files[name]['bytes']),'saved source-map identity '+name)
        by_original.setdefault(source,[]).append(name)
    for original,name in sm.get('original_path_to_package_relative_path',{}).items():
        need(original in by_original and rel(name) in by_original[original],'supplemental source translator matches declared rows')
    certs=[n for n in files if n.endswith('.npz')]
    need(len(certs)==22 and all(n.startswith('certificates/SYMMETRIC_ALL50_V2/') and len(PurePosixPath(n).parts)==3 for n in certs),'only22 explicit certificate arrays')
    need(not any(n.endswith(('.npy','.pt','.pth','.ckpt','.safetensors')) for n in files),'no model/other array payload')
    cert_sources=[b for b in sm['files'] if b['path'] in certs]
    need(len(cert_sources)==22 and all(b['source_path_provenance_only']==str(R.parent/'query_visibility_controlled_real_20261022_v66/public_pallet/SYMMETRIC_ALL50_V2'/PurePosixPath(b['path']).name) for b in cert_sources),'certificate source identity prefix/basename')
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
    qpath=rel(e['independent_sameq_QA_relative_path']);need(qpath in files and files[qpath]['sha256']=='777bf71441ca679c1db1ad98531701ae98e50ba2532facf26035e8b2ddd85b4d','Root-run independent352 QA pin')
    qa=json.loads(raw(PACKAGE/qpath));need(qa['completed'] is True and qa['PASS'] is True and qa['physical_gold_certified'] is False and qa['human_gold'] is False and qa['training_admission'] is False,'conditional/not native gold')
    counts=qa['counts']
    for k,v in {'rows':352,'frames':44,'pairs':22,'original_continuous_available':297,'original_input_U_retained':55,'exact1pixel_counts':{'-1':318,'0':25,'1':9},'radius2_5x5_counts':{'-1':316,'0':29,'1':7},'helper_valid':296,'radius2_strict_E_core':7}.items():need(counts[k]==v,'saved independent count '+k)
    need(qa['original_available_but_rounded_pixel_unsupported_U']==1 and qa['source_native_or_SELF_labels_created']==0,'original55 plus rounded1U/no new native labels')
    expected_certs={(x['binding']['path'],x['binding']['sha256'],x['bytes_verified']) for x in qa['certificate_closed_SHA_checks']}
    copied_certs={(x['source_path_provenance_only'],x['source_sha256'],x['source_bytes']) for x in cert_sources}
    need(len(expected_certs)==22 and copied_certs==expected_certs,'exact22 saved independent certificate SHA/bytes closure')
    exclusions=json.loads(raw(PACKAGE/'EXCLUSIONS.json'))
    for key in ('original_mixed_metadata_not_copied','raw_RGB_GT_teacher_pointers_not_followed','exact22certificate_only','no_weights_or_other_arrays'):need(exclusions[key] is True,'precise exclusion '+key)
    cli=rel(e['portable_cli_relative_path']);need(cli in files and files[cli]['sha256']=='f82110a83bcdf85625b1dec4c475406c3e00cbf7d1150bc59728b8f909ff4a48','portable CLI CODE pin')
    result={'completed':True,'PASS':True,'manifest_sha256':pin,'package':str(PACKAGE),'declared_files':len(files),
            'total_bytes':m['total_bytes'],'all_leaf_bytes_below95000000':True,'manifest_path_closure_PASS':True,
            'source_map_identity_bindings_checked':len(mapped),'identity_encoding_only':True,'certificate_array_files':len(certs),'original_source_pointer_follow':0,
            'excluded_original_payload_stat_or_read':0,'scientific_status':status,'links':links,'independent_sameq_counts':counts,'portable_CLI_SHA':files[cli]['sha256'],'portable_CLI_executed_by_C':False,
            'entry':bind(entry),'verifier_code':bind(Path(__file__)),'package_or_scientific_files_modified':False,
            'arrays_numeric_model_image_decode_Git_network_training_or_selection':0,
            'limits':'Package bytes and saved metadata identity only; original source leaf bytes were not reread. Independent arithmetic QA is a pinned saved result; Root executes the reviewed portable CLI separately. No scientific revalidation, model forward, threshold or candidate selection.'}
    print(json.dumps(write('REVIEW.json',result)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args()
    try:run(Path(a.entry))
    except BaseException:
        write('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'entry':str(a.entry),'traceback':traceback.format_exc(),'automatic_retry':False});raise
