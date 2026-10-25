"""Declared-only V244 publication QA. Opaque bytes; no array/image/model decoding."""
import gzip,hashlib,json,os,posixpath,re,stat,traceback
from html.parser import HTMLParser
from pathlib import Path,PurePosixPath
from urllib.parse import unquote,urlsplit

H=Path(__file__).absolute().parent;R=H.parent;PKG=R/'V244_PUBLICATION_PACKAGE'
PIN='d5eba02892004b4637b0c4508eec1431284af102cb5048462cc7e11685b25c07'
HISTORY={'../V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/README_KO.md','../V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/README_KO.md','../V241_SCENE_CONSISTENT_CF_PREPARATION/README_KO.md','../EXPERIMENT_INDEX_KO.md'}
SOURCE_TOPS={'V242_CF_FEATURE_CODE_REVIEW_C','V242_CF_FEATURE_PACKET_QA_C','V242_CF_FEATURE_PRODUCER_B','V242_QUERY_REFERENCE_DIAGNOSTIC_C','V242_ROOT_SCENE_CF','V242_SCENE_CF_DRIVER_B','V242_SCENE_DATA_QA_C','V242_SCENE_DRIVER_CODE_REVIEW_A','V242_SCENE_GALLERY_A','V242_SCENE_MATH_REVIEW_A','V243_SCENE_CF_TRAINER_A','V243_TRAINER_CODE_REVIEW_B','assemble_v242_actual_qa_entry.py','assemble_v242_gallery_entry.py','V242_PUBLICATION_INVENTORY_C','V244_ROOT_FROZEN_BEHAVIOR','V244_OWN_BRANCH_DIAGNOSTIC_A','V244_OWN_BRANCH_CODE_REVIEW_C','V244_V235_PAIR_DIAGNOSTIC_B','V244_V235_PAIR_CODE_REVIEW_A','V244_SAVED_OUTPUT_QA_C','V244_PUBLICATION_CODE_REVIEW_B','build_v244_publication.py','publish_v244_main.py','verify_v244_saved_outputs_portable.py','V233_PUBLICATION_VERIFY_A'}

def need(ok,msg):
    if not ok:raise ValueError(msg)

def lex(value):
    s=str(value);need(not any(x in s.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','source_holdout','oldval','v177','v178','v182')),'lexical before filesystem');return Path(s)

def safe(p):
    p=lex(p);need(p.is_absolute() and '..' not in p.parts,'absolute no-traversal path')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'ancestor symlink/non-directory')
    try:need(stat.S_ISREG(os.lstat(p).st_mode),'leaf symlink/nonregular')
    except FileNotFoundError:pass
    return p

def raw(p):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def sha(p,decoded=False):
    d=hashlib.sha256();n=0
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        stream=gzip.GzipFile(fileobj=f,mode='rb') if decoded else f
        try:
            for b in iter(lambda:stream.read(1024*1024),b''):d.update(b);n+=len(b)
        finally:
            if decoded:stream.close()
    return d.hexdigest(),n

def rel(s):
    p=PurePosixPath(s);lex(s);need(not p.is_absolute() and '..' not in p.parts and str(p)==s,'canonical relative manifest leaf');return s

def write(name,o):
    p=H/name
    with os.fdopen(os.open(safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    digest,n=sha(p);return dict(path=str(p),sha256=digest,size_bytes=n)

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ('href','src') and v)

def run():
    mraw=raw(PKG/'MANIFEST.json');need(hashlib.sha256(mraw).hexdigest()==PIN,'exact final manifest pin')
    m=json.loads(mraw);files={rel(x['path']):x for x in m['files']}
    need(len(files)==len(m['files'])==m['file_count']==239 and sum(x['bytes'] for x in files.values())==m['total_bytes']==92459921,'239 exact leaves and byte total')
    checked=[]
    for name,b in files.items():
        d,n=sha(PKG/name);need((d,n)==(b['sha256'],b['bytes']),'package SHA/size '+name);checked.append(name)
    # Exclusions are read only as inert metadata, before any original-source leaf operation.
    exc=json.loads(raw(PKG/'LARGE_CACHE_EXCLUSIONS.json'));blocked={str(lex(b['path'])) for b in exc['bindings']}
    need(len(blocked)==5 and sum(b['size_bytes'] for b in exc['bindings'])==9348844160,'five huge cache inert bindings')
    sm=json.loads(raw(PKG/'PUBLICATION_SOURCE_MAP.json'));need(sm['absolute_pointers_are_provenance_only'] is True and sm['huge_excluded_cache_leaf_access']==0 and sm['private_or_current_or_held_payloads']==0 and sm['old_frozen_sources_changed'] is False,'source-map scope flags')
    source_checks=[];gzip_checks=[];mapped=set()
    for b in sm['files']:
        name=rel(b['path']);need(name in files and name not in mapped,'source-map declared leaf');mapped.add(name)
        source=lex(b['source_path_provenance_only']);need(str(source) not in blocked,'excluded cache rejected before stat/read')
        need(source.is_absolute() and '..' not in source.parts and source.is_relative_to(R) and source.relative_to(R).parts[0] in SOURCE_TOPS,'only explicitly admitted Root-owned source scopes')
        if source.relative_to(R).parts[0]=='V233_PUBLICATION_VERIFY_A':need(source.suffix=='.md','old scope notice only')
        need(source.suffix not in ('.pt','.pth','.jpg','.jpeg'),'no original weight/photo source follow')
        d,n=sha(source);need((d,n)==(b['source_sha256'],b['source_bytes']),'opaque original source binding '+name)
        if b['encoding']=='identity':need((d,n)==(files[name]['sha256'],files[name]['bytes']),'identity copy equality '+name)
        elif b['encoding']=='gzip':
            dd,nn=sha(PKG/name,True);need((dd,nn)==(d,n),'gzip decoded original equality '+name);gzip_checks.append(dict(path=name,decoded_sha256=dd,decoded_bytes=nn))
        else:raise ValueError('unknown source-map encoding')
        source_checks.append(dict(path=name,source_sha256=d,source_bytes=n,encoding=b['encoding']))
    need(len(source_checks)==234 and len(gzip_checks)==2,'complete copied-source/gzip census')
    inv=json.loads(raw(PKG/'artifacts/V242_PUBLICATION_INVENTORY_C/CANDIDATE_FILES.json'))
    bysource={b['source_path_provenance_only']:(b['source_sha256'],b['source_bytes']) for b in sm['files']}
    need(len(inv['candidates'])==146 and all(bysource.get(b['path'])==(b['sha256'],b['size_bytes']) for b in inv['candidates']),'all frozen inventory146 candidates preserved')
    need(len(inv['preserved_failure_fix_and_prior_version_artifacts'])==33 and all(b['path'] in bysource for b in inv['preserved_failure_fix_and_prior_version_artifacts']),'all33 failure/fix/prior records preserved')
    s=json.loads(raw(PKG/'STATUS.json'))
    for k,v in {'completed':True,'goal_status':'active_unsolved','fully_solved':False,'model_adopted':False,'current_candidate_admission':False,'V242_data_QA_PASS':True,'V242_all_original_parents':5406,'V242_new_proof_PASS':5367,'V242_unconstructed_retained':39,'V242_fixed_prefit_gallery_cases':4,'V242_new_CF_features':0,'V243_training':'CODE_READY_UNEXECUTED','V243_new_optimizer_updates':0,'V244_both_frozen_diagnostics_complete':True,'V244_own_branch_CPU_rows':7617,'new_training_updates':0,'new_current_or_held_payloads':0,'new_PnP':0,'publication_is_not_model_success':True}.items():need(s[k]==v,'status '+k)
    need(m['scientific_status']==s and m['publication_is_not_model_success'] is True,'manifest/status same authority')
    need(s['V244_own_branch_YCB_DEV_E']=={'N':57,'own_base_correct':0,'own_total_correct':1,'residual_wrong_direction':56,'residual_right_direction':1},'own saved QA headline counts')
    need(s['V244_old_V235_YCB_FIT_pool']=={'N':830,'COVER_E':789,'SHAM_D':785,'both_correct':751},'pair saved QA headline counts')
    docs=('README_KO.md','artifacts/V242_SCENE_GALLERY_A/gallery_v2/PORTABLE_GALLERY_KO.html');links=[]
    for doc in docs:
        text=raw(PKG/doc).decode()
        if doc.endswith('.md'):refs=re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',text)
        else:p=Links();p.feed(text);refs=p.refs
        for ref in refs:
            u=urlsplit(ref)
            if u.scheme or u.netloc or not u.path:links.append(dict(document=doc,reference=ref,kind='external_or_anchor_inert'));continue
            target=posixpath.normpath(posixpath.join(posixpath.dirname(doc),unquote(u.path)));lex(target)
            if target.startswith('../'):
                need(doc=='README_KO.md' and target in HISTORY,'undeclared external historical link');kind='known_published_historical_provenance_inert_NO_FS'
            else:need(target in files or target=='MANIFEST.json','broken package local link '+target);kind='manifest_declared_internal'
            links.append(dict(document=doc,reference=ref,target=target,kind=kind))
    readme=raw(PKG/'README_KO.md').decode();need(all(t in readme for t in ('새 모델 성능이 좋아졌다는 결과는 아직 없습니다','V243는 코드 준비 상태','별도 CONTROL','새 blind 검증이 아닙니다','native는 UNKNOWN','코드 준비를 실제 학습으로 세지 않습니다')),'bounded status/interpretation caveats')
    need('THIRD_PARTY_NOTICES.md' in files,'third-party notice preserved')
    cr=json.loads(raw(PKG/'artifacts/V244_PUBLICATION_CODE_REVIEW_B/REVIEW.json'))
    need(cr['completed'] is True and cr['PASS'] is True and cr['code']['builder']['sha256']=='c610c42e4445a61cf3e42339e9c6923f515777ac7bc5f0775c7af816199d2290' and cr['code']['publisher']['sha256']=='679d0e8112366e666a3e0c563d608d513b0f1e32f25b4bd30827ca8f260d80c7','reviewed publication code pins')
    result=dict(completed=True,PASS=True,manifest_sha256=PIN,package=str(PKG),declared_files=239,total_bytes=92459921,opaque_declared_leaf_SHA_size_PASS=True,source_map_original_opaque_identity_checks=source_checks,gzip_decoded_identity_checks=gzip_checks,inventory146_and_failure_fix33_preserved=True,links=links,scientific_status=s,publication_code_review_binding=files['artifacts/V244_PUBLICATION_CODE_REVIEW_B/REVIEW.json'],huge5_excluded_leaf_stat_or_read=0,arrays_numeric_headers_models_images_decode=0,new_Git_network_training_or_model_selection=0,historical_links_inert=True,package_or_source_files_modified=False,limits='Declared bytes/provenance/links/status QA. No experiment rerun, model inference, full raw-source RGB access, classifier selection or independent reconstruction of joined scientific fields.',verifier_code={'path':str(Path(__file__)),'sha256':sha(Path(__file__))[0],'size_bytes':sha(Path(__file__))[1]})
    receipt=write('REVIEW.json',result);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        write('FAILED_EXECUTION.json',dict(completed=False,PASS=False,manifest_sha256=PIN,traceback=traceback.format_exc(),automatic_retry=False));raise
