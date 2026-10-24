"""Independent, root-entry-only V187 publication byte/link audit. No Git calls.
Preparation uses invented bytes and code/protocol text, never phase inventories.
"""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse, codecs, gzip, hashlib, io, json, os, posixpath, re, stat, tempfile, traceback

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PUB=REPO/PREFIX
DELTA='publication/deltas/V187_20261024_v1/'
PHASE=V.name
ARCHIVE='archive/task-2/pallet-keypoint-visibility/'+PHASE+'/'
BASE_SHA='f5f4c2d819c7533492e24d2238baa8a6f638438c4458492d01c0321a34130fc7'
OUT=P/'publication_root_preparation/V187_DELTA_INDEPENDENT_REVIEW_A_V2_DIRECTORY_LINKS'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_DELTA_BYTES_V2_DIRECTORY_LINKS.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS_V2_DIRECTORY_LINKS.json'
C=V/'publication_delta_preparation_c'
C_CODE={'path':str(C/'build_completed_phase_delta.py'),'sha256':'1340a77b8f21376f8a9b9bcac2e4e29673b840863635b8fc0ba51cc5c9454833'}
C_PROTO={'path':str(C/'BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json'),'sha256':'ae9424e3efaf0f2d37572b076b1a0c092e6f0e59213eb8205e21d418983359e7'}
PORTABLE=[ARCHIVE+'PORTABLE_REPORT_KO.md',ARCHIVE+'fixed_source_nominal_visual_v1/PORTABLE_GALLERY_KO.md','EXPERIMENT_INDEX_THROUGH_V187_KO.md']
TEXT={'.py','.sh','.md','.txt','.json','.jsonl','.csv','.tsv','.html','.htm','.yaml','.yml','.toml','.log','.patch','.diff','.svg'}
LIMIT=100*1024**2

def require(v,s):
    if not bool(v):raise AssertionError(s)
def safe_name(path):
    p=Path(path)
    require('..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Forbidden path before access')
    return p
def relative(path):
    require(isinstance(path,str)and path and '\\'not in path,'POSIX relative publication path')
    p=safe_name(path);require(not p.is_absolute()and not urlsplit(path).scheme,'Relative publication path')
    require(str(PurePosixPath(path))==path,'Canonical path');return path
def opened(path):
    p=safe_name(path);require(p.is_absolute(),'Explicit absolute path')
    require(p==REPO or REPO in p.parents or P in p.parents,'Explicit publication/project roots only')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        final=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Regular file, no symlink');return os.fdopen(final,'rb')
def file_chunks(path):
    with opened(path)as f:
        for b in iter(lambda:f.read(1<<20),b''):yield b
def digest(chunks):
    h=hashlib.sha256();size=0
    for b in chunks:h.update(b);size+=len(b)
    return h.hexdigest(),size
def bind(path):return {'path':str(path),'sha256':digest(file_chunks(path))[0]}
def checked(b):
    require(isinstance(b.get('sha256'),str)and len(b['sha256'])==64,'Exact binding required')
    require(digest(file_chunks(b['path']))[0]==b['sha256'],'Binding drift '+b['path']);return Path(b['path'])
def read(path):
    with opened(path)as f:return json.load(f)
def readlines(path):
    with opened(path)as f:return [json.loads(s)for s in f if s.strip()]
def bytes_of(path):
    with opened(path)as f:return f.read()
def write(path,obj):
    safe_name(path)
    with Path(path).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def covered(entry):
    if entry['status']=='EXCLUDED':return set()
    paths={relative(entry['published_relative'])}
    if entry.get('gzipchunks')is not None:
        paths.update(relative(r['path'])for r in entry['gzipchunks']['parts'])
    if entry.get('gzip_descriptor_path'):paths.add(relative(entry['gzip_descriptor_path']))
    return paths
def physical(entry):
    if entry['status']=='EXCLUDED':return set()
    if entry.get('gzipchunks')is None:return {relative(entry['published_relative'])}
    return {relative(r['path'])for r in entry['gzipchunks']['parts']}
def assert_disjoint(base,entries):
    old=set().union(*(covered(e)for e in base))if base else set()
    for e in entries:require(not(old&covered(e)),'Old logical/physical/descriptor path collision')
    return old
def gunzip_parts(parts):
    for part in parts:
        require(part['compression']=='gzip','Declared gzip only')
        n=0
        with opened(PUB/relative(part['path']))as f:
            with gzip.GzipFile(fileobj=f,mode='rb')as g:
                for b in iter(lambda:g.read(1<<20),b''):n+=len(b);yield b
        require(n==part['decoded_size_bytes'],'Decoded chunk length')

PATTERNS=[('PEM_PRIVATE_KEY',re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----')),
          ('GITHUB_CLASSIC_TOKEN',re.compile(r'(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36}(?![A-Za-z0-9_])')),
          ('GITHUB_FINE_GRAINED_TOKEN',re.compile(r'(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{40,}(?![A-Za-z0-9_])')),
          ('OPENAI_TOKEN_FORMAT',re.compile(r'(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])')),
          ('AWS_ACCESS_ID_FORMAT',re.compile(r'(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])')),
          ('SLACK_TOKEN_FORMAT',re.compile(r'(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{20,}(?![A-Za-z0-9-])'))]
def scan_text(chunks):
    decoder=codecs.getincrementaldecoder('utf-8')('replace');carry='';line=0;hits=[]
    def inspect(s):
        nonlocal line
        line+=1
        for kind,pattern in PATTERNS:
            if pattern.search(s):hits.append({'kind':kind,'line':line})
    for b in chunks:
        parts=(carry+decoder.decode(b)).split('\n');carry=parts.pop()
        for s in parts:inspect(s)
    carry+=decoder.decode(b'',final=True)
    if carry:inspect(carry)
    return {'scanned_text_lines':line,'format_hits_redacted':hits}

class HrefParser(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        self.targets.extend(v for k,v in attrs if k in {'href','src'}and v is not None)
def link_targets(text):
    out=[]
    for m in re.finditer(r'!?\[[^\]\n]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+[^)]*)?\)',text):out.append(m.group(1).strip('<>'))
    h=HrefParser();h.feed(text);out.extend(h.targets);return out
def resolve_target(document,target):
    u=urlsplit(target)
    require(u.scheme.lower()!='file','file link forbidden')
    if u.scheme:
        require(u.scheme.lower()in {'http','https','mailto','data'},'Unexpected portable URI scheme');return None
    path=unquote(u.path)
    if not path:return None
    require(not path.startswith('/')and not re.match(r'^[A-Za-z]:',path)and '\\'not in path,'Absolute portable link forbidden')
    require('/home/'not in path,'Home path link forbidden')
    resolved=posixpath.normpath(posixpath.join(posixpath.dirname(document),path))
    require(resolved!='..'and not resolved.startswith('../'),'Link escapes publication root')
    return relative(resolved)
def portable_audit(document,text,exists):
    require('/home/'not in text and 'file://'not in text,'Home/file provenance is not portable entry text')
    local=[];external=0
    for target in link_targets(text):
        resolved=resolve_target(document,target)
        if resolved is None:external+=1
        else:require(exists(resolved),'Missing portable link: '+resolved);local.append(resolved)
    return {'document':document,'local_links':local,'local_link_count':len(local),'external_or_anchor_or_inline_count':external}
def ordinary_link_target(path):
    p=safe_name(path);require(p.is_absolute(),'Explicit absolute target')
    require(REPO in p.parents or P in p.parents,'Explicit publication/project target only')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        final=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    finally:os.close(fd)
    try:
        mode=os.fstat(final).st_mode
        return stat.S_ISREG(mode) or stat.S_ISDIR(mode)
    finally:os.close(final)
def exists_file_or_directory(relative_path):
    try:
        return ordinary_link_target(PUB/relative(relative_path))
    except FileNotFoundError:return False

def invented():
    tests=[]
    def add(n,v):tests.append({'name':n,'PASS':bool(v)})
    data=b'preserved source\n\x00bytes\xff';z=[gzip.compress(data[:8],mtime=0),gzip.compress(data[8:],mtime=0)]
    restored=b''.join(gzip.decompress(p)for p in z)
    add('two_independent_gzip_members_restore_original_bytes',digest([restored])==digest([data]))
    add('single_byte_corruption_changes_original_SHA',digest([data+b'x'])!=digest([data]))
    raw={'status':'PUBLISHED_RAW','published_relative':'old/a','gzipchunks':None}
    gz={'status':'PUBLISHED_GZIP_CHUNKS','published_relative':'old/b','gzip_descriptor_path':'old/b.parts.json','gzipchunks':{'parts':[{'path':'old/b.parts/part00000.gz'}]}}
    add('old_logical_part_descriptor_sets',covered(gz)=={'old/b','old/b.parts.json','old/b.parts/part00000.gz'})
    for name,path in [('raw','old/a'),('logical','old/b'),('part','old/b.parts/part00000.gz'),('descriptor','old/b.parts.json')]:
        try:assert_disjoint([raw,gz],[dict(raw,published_relative=path)]);ok=False
        except AssertionError:ok=True
        add('old_'+name+'_collision_rejected',ok)
    add('new_disjoint_path_accepted',bool(assert_disjoint([raw,gz],[dict(raw,published_relative='new/a')])))
    add('excluded_null_path_never_opened_or_normalized',covered({'status':'EXCLUDED','published_relative':None,'gzipchunks':None})==set())
    try:relative('archive/blocked_v112/file');ok=False
    except AssertionError:ok=True
    add('forbidden_name_rejected_without_stat',ok)
    s=scan_text([(b'plain\n'+('gh'+'p_'+'A'*36).encode())[:15],(b'plain\n'+('gh'+'p_'+'A'*36).encode())[15:]])
    add('credential_match_cross_chunk_redacted',s['format_hits_redacted']==[{'kind':'GITHUB_CLASSIC_TOKEN','line':2}])
    add('SHA_not_credential',not scan_text([('a'*64+'\n').encode()])['format_hits_redacted'])
    pages={'archive/p/img.png','index.md'}
    q=portable_audit('archive/p/report.md','![x](img.png)\n[up](../../index.md)\n[web](https://example.org)',lambda p:p in pages)
    add('relative_image_parent_index_and_remote',q['local_links']==['archive/p/img.png','index.md'])
    for name,target in [('home','/home/user/a'),('file','file:///tmp/a'),('escape','../../../a')]:
        try:portable_audit('archive/p/report.md','[x]('+target+')',lambda p:True);ok=False
        except AssertionError:ok=True
        add('portable_'+name+'_rejected',ok)
    add('strict_Git_100MiB_boundary',LIMIT-1<LIMIT and not LIMIT<LIMIT)
    add('previous_index_suffix_keeps_relative_parent',resolve_target('EXPERIMENT_INDEX_THROUGH_V187_KO.md','reports/old.md')=='reports/old.md')
    with tempfile.TemporaryDirectory(prefix='invented_directory_only_',dir=H)as t:
        root=Path(t);directory=root/'ordinary';directory.mkdir();link=root/'symlink';link.symlink_to(directory,target_is_directory=True)
        add('ordinary_directory_link_target_accepted_without_enumeration',ordinary_link_target(directory))
        try:ordinary_link_target(link);ok=False
        except OSError:ok=True
        add('symlink_directory_target_refused_NOFOLLOW',ok)
    return {'completed':True,'PASS':all(t['PASS']for t in tests),'invented_only':True,'checks':tests,'actual_phase_inventory_publication_or_source_hash_copy_reads':0,'Git_calls':0}

def prepare():
    tests=invented();tests['code']=bind(__file__);write(CHECKS,tests);require(tests['PASS'],'Invented checks')
    proto={'completed':True,'prepared_only':True,'code':bind(__file__),'checks':bind(CHECKS),'publisher_builder':C_CODE,'publisher_protocol':C_PROTO,
           'immutable_base_manifest':{'path':str(PUB/'PUBLICATION_MANIFEST.json'),'sha256':BASE_SHA},'delta_manifest_path':str(PUB/DELTA/'DELTA_MANIFEST.json'),'source_archive_manifest_path':str(PUB/DELTA/'SOURCE_ARCHIVE_MANIFEST.jsonl'),
           'source_phase':str(V),'repository':str(REPO),'output':str(OUT),'portable_pages':PORTABLE,
           'audit':['All new included original source files streamhash versus frozen sourceSHA; stored RAW or concatenated gzip roundtrip versus same SHA/size, every stored part SHA/size/CRC, every physical Git-stage file strictly<100MiB.','Old manifest metadata only; reject all delta logical/raw/part/descriptor collisions before opening new bodies. Rehash unchanged base manifest after audit; no old raw dataset/model/evaluation file bodies.','Exact SOURCE_ARCHIVE records occur in DELTA; separate physical descriptor records and BUILD physical_files close stored paths, delta/chain/readme.','Scan new logical text plus additional new physical text for credential formats; output only redacted type/path/line, never token content. This format scan is not a general secret detector.','Only three named portable entry pages receive relative-link checks. Copied provenance Markdown is scanned for credentials but not treated as a portable entry. Verify new index ends with old index bytes in same parent, global readme ends with preserved prior bytes.'],
           'link_target_policy':'V2 only: explicit ordinary regular files OR ordinary directories exist; every component opened with NOFOLLOW, no directory enumeration/content read. Symlinks refused, no logical gzip fallback. All other byte/credential/collision checks unchanged.',
           'V1_static_finding':bind(H/'ROOT_STATIC_DIRECTORY_LINK_FINDING_V1_PRESERVED.json'),
           'body_interpretation':'No image/model/NPZ/evaluation decode or semantic data parse; manifests/chain/descriptors are administrative metadata, other text is regex/link audit only. No network or Git commands.','actual_before_root_entry':0,'automatic_retry':False}
    write(PROTOCOL,proto)
    template={'approved_actual_V187_immutable_delta_audit':False,'completed_phase_frozen':False,'code':proto['code'],'protocol':bind(PROTOCOL),'checks':proto['checks'],'repository':str(REPO),'build_complete':{'path':None,'sha256':None},'base_manifest':proto['immutable_base_manifest'],'delta_manifest':{'path':proto['delta_manifest_path'],'sha256':None},'output':str(OUT),'Git_mutation_or_push':False,'image_model_eval_decode_or_parse':False,'automatic_retry':False}
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE_V2_DIRECTORY_LINKS.json',template)
    ready={'completed':True,'prepared_only':True,'code':proto['code'],'protocol':bind(PROTOCOL),'checks':proto['checks'],'template':bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE_V2_DIRECTORY_LINKS.json'),'actual_phase_enumeration_hash_copy_or_repo_access':0,'output_after_entry':str(OUT),'command':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).absolute())+' --entry <root-entry>'}
    write(H/'PREPARATION_COMPLETE_V2_DIRECTORY_LINKS.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=read(entry);require(e['approved_actual_V187_immutable_delta_audit']is True and e['completed_phase_frozen']is True,'Root completed-phase audit entry')
    for k in ('Git_mutation_or_push','image_model_eval_decode_or_parse','automatic_retry'):require(e[k]is False,'Forbidden action '+k)
    p=read(checked(e['protocol']));require(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL),'Frozen auditor')
    require(e['checks']==p['checks'],'Frozen invented checks');checked(p['checks'])
    require(e['repository']==str(REPO)and e['base_manifest']==p['immutable_base_manifest']and e['base_manifest']['sha256']==BASE_SHA,'Pinned immutable base/root')
    require(e['output']==p['output']==str(OUT)and V not in OUT.parents,'Output outside frozen phase')
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'STARTED.json',{'completed':False,'entry':bind(entry),'automatic_retry':False})
    try:
        base=read(checked(e['base_manifest']));build=read(checked(e['build_complete']))
        require(build['completed']is True and build['PASS']is True and build['repository']==str(REPO),'Completed staged build')
        require(build['base_manifest']==e['base_manifest']and build['delta_manifest']==e['delta_manifest'],'Build binding closure')
        require(e['delta_manifest']['path']==p['delta_manifest_path'],'Exact delta path')
        delta=read(checked(e['delta_manifest']));require(delta['format']=='immutable-completed-phase-delta-v1'and delta['phase']==PHASE and delta['previous_base_manifest']==e['base_manifest'],'Delta identity')
        entries=delta['entries'];old=assert_disjoint(base['entries'],entries)
        logical={};by_source={}
        for r in entries:
            key=(r['source_root_id'],r['source_relative_path']);require(key not in by_source,'Unique source record');by_source[key]=r
            safe_name(r['source_relative_path'])
            if r['status']=='EXCLUDED':continue
            key=relative(r['published_relative']);require(key not in logical,'Unique new logical record');logical[key]=r
        archive_path=PUB/DELTA/'SOURCE_ARCHIVE_MANIFEST.jsonl';archive=readlines(archive_path)
        require(DELTA+'SOURCE_ARCHIVE_MANIFEST.jsonl'in logical,'Source archive itself sealed')
        require(all(by_source.get((r['source_root_id'],r['source_relative_path']))==r for r in archive),'Every source archive record exactly preserved in delta')
        require(sum(r['status']!='EXCLUDED'for r in archive)==delta['source_phase_original_records'],'Source original logical count')
        require(all(r['source_root_id']=='P'and r['source_absolute']==str(P/r['source_relative_path'])and r['source_relative_path'].startswith(PHASE+'/')and (r['status']=='EXCLUDED'or r['published_relative'].startswith(ARCHIVE))for r in archive),'Only frozen new source phase')
        file_results=[];credential_hits=[];scanned=set();physical_expected={}
        for r in entries:
            if r['status']=='EXCLUDED':continue
            rel=relative(r['published_relative']);source=safe_name(r['source_absolute'])
            if r['source_root_id']=='P':require(V in source.parents,'New original source phase only')
            else:require(r['source_root_id']=='DELTA_DERIVED'and source==PUB/rel,'Derived metadata must be its new stored file')
            source_hash,source_size=digest(file_chunks(source))
            require(source_hash==r['source_sha256']and source_size==r['size_bytes'],'Original source SHA/size: '+rel)
            if r.get('gzipchunks')is None:
                require(r['status']=='PUBLISHED_RAW','Raw status')
                stored_hash,stored_size=digest(file_chunks(PUB/rel));require(stored_hash==r['published_sha256']==source_hash and stored_size==source_size,'Stored raw identity '+rel)
                require(rel not in physical_expected,'Duplicate raw/part physical path');physical_expected[rel]=(stored_hash,stored_size);stream=lambda rel=rel:file_chunks(PUB/rel)
            else:
                require(r['status']=='PUBLISHED_GZIP_CHUNKS','Gzip status');g=r['gzipchunks'];require(g['format']=='concatenated-bytes-v1','Gzip concatenation convention')
                for part in g['parts']:
                    pr=relative(part['path']);h,n=digest(file_chunks(PUB/pr));require(h==part['sha256']and n==part['stored_size_bytes']and n<LIMIT,'Stored gzip part SHA/size')
                    require(pr not in physical_expected,'Duplicate stored gzip part');physical_expected[pr]=(h,n)
                restored_hash,restored_size=digest(gunzip_parts(g['parts']))
                require(restored_hash==g['content_sha256']==source_hash and restored_size==g['content_size_bytes']==source_size,'Gzip original roundtrip '+rel)
                desc=relative(r['gzip_descriptor_path']);require(desc==rel+'.parts.json'and desc in logical and logical[desc]['status']=='PUBLISHED_RAW','Physical descriptor separate raw record')
                dh,dn=digest(file_chunks(PUB/desc));require(dh==r['gzip_descriptor_sha256']and dn==r['gzip_descriptor_size_bytes'],'Descriptor SHA/size')
                descriptor=read(PUB/desc);require(descriptor==g,'Descriptor exact gzip contract')
                stream=lambda parts=g['parts']:gunzip_parts(parts)
            if Path(rel).suffix.casefold()in TEXT:
                scan=scan_text(stream());scanned.add(rel)
                credential_hits.extend(dict(v,published_relative=rel)for v in scan['format_hits_redacted'])
            file_results.append({'published_relative':rel,'status':r['status'],'source_root_id':r['source_root_id'],'source_sha256':source_hash,'source_size_bytes':source_size,'PASS':True})
        for rel,(h,n)in physical_expected.items():require(n<LIMIT,'Physical Git file must be strictly<100MiB')
        buildfiles={}
        for f in build['physical_files']:
            rel=relative(f['path']);require(rel not in buildfiles,'Unique physical build path');buildfiles[rel]=f
            require(rel.startswith(PREFIX)or rel=='readme.md','Only publication delta or global README')
            require(0<=f['size_bytes']<LIMIT,'Physical Git<100MiB');h,n=digest(file_chunks(REPO/rel));require(h==f['sha256']and n==f['size_bytes'],'All physical BUILD bytes')
            logical_rel=rel[len(PREFIX):]if rel.startswith(PREFIX)else None
            if logical_rel is not None:require(logical_rel not in old,'Old covered path in new physical BUILD')
            if logical_rel not in scanned and Path(rel).suffix.casefold()in TEXT:
                credential_hits.extend(dict(v,published_relative=rel)for v in scan_text(file_chunks(REPO/rel))['format_hits_redacted']);scanned.add(logical_rel or rel)
        expected_repo={PREFIX+k for k in physical_expected}|{PREFIX+DELTA+'DELTA_MANIFEST.json',PREFIX+'MANIFEST_CHAIN_AFTER_V187.json','readme.md'}
        require(set(buildfiles)==expected_repo,'Exact physical BUILD inventory, no omissions/extras')
        for rel,(h,n)in physical_expected.items():require(buildfiles[PREFIX+rel]['sha256']==h and buildfiles[PREFIX+rel]['size_bytes']==n,'Manifest-to-BUILD physical equality')
        chain=read(checked(build['new_chain']));require(chain['base']=={'path':'PUBLICATION_MANIFEST.json','sha256':BASE_SHA}and chain['deltas']==[{'path':DELTA+'DELTA_MANIFEST.json','sha256':e['delta_manifest']['sha256']}],'Manifest chain closure')
        require(chain['index']['path']=='EXPERIMENT_INDEX_THROUGH_V187_KO.md'and chain['index']['sha256']==build['new_index']['sha256'],'Versioned index chain binding')
        checked(build['new_index']);old_index=bytes_of(PUB/'EXPERIMENT_INDEX_KO.md');new_index=bytes_of(PUB/'EXPERIMENT_INDEX_THROUGH_V187_KO.md')
        old_index_rows=[r for r in base['entries']if r.get('published_relative')=='EXPERIMENT_INDEX_KO.md'and r['status']!='EXCLUDED']
        require(len(old_index_rows)==1 and hashlib.sha256(old_index).hexdigest()==old_index_rows[0]['published_sha256']and new_index.endswith(old_index),'Original index byte-exact suffix in same parent')
        backup=bytes_of(PUB/DELTA/'GLOBAL_README_BEFORE_V187.md');require(bytes_of(REPO/'readme.md').endswith(backup),'Prior global README byte-exact suffix')
        links=[portable_audit(doc,bytes_of(PUB/doc).decode('utf-8'),exists_file_or_directory)for doc in PORTABLE]
        require(not credential_hits,'Credential format hits present; values are redacted')
        require(digest(file_chunks(e['base_manifest']['path']))[0]==BASE_SHA,'Immutable base unchanged after audit')
        result={'completed':True,'PASS':True,'review_type':'INDEPENDENT_IMMUTABLE_PUBLICATION_DELTA_BYTE_AND_PORTABLE_TEXT_AUDIT','entry':bind(entry),'code':bind(__file__),'protocol':bind(PROTOCOL),'base_manifest':e['base_manifest'],'delta_manifest':e['delta_manifest'],'build_complete':e['build_complete'],
                'coverage':{'source_archive_records':len(archive),'included_logical_records':len(file_results),'physical_BUILD_files':len(buildfiles),'logical_text_records_scanned':len(scanned),'portable_entry_pages':len(links),'portable_local_links':sum(r['local_link_count']for r in links),'old_manifest_covered_paths_untouched':True},
                'file_results':file_results,'portable_link_QA':links,'credential_format_hits_redacted':credential_hits,'old_index_and_global_readme_suffix_preserved':True,'physical_Git_limit_bytes_exclusive':LIMIT,
                'limits':['No Git index/commit/push or remote-publication check; BUILD physical plan is verified before later review-bound push-plan creation.','Only administrative manifests/chain/descriptors parsed as JSON. Other new text received credential/link checks only; image/model/evaluation data were never decoded or semantically parsed.','Credential-format scan does not prove absence of arbitrary secrets. Copied provenance Markdown was not treated as a portable entry.','No old raw dataset/model/evaluation bodies read or old manifest-covered files rehashed; base/index metadata and old index text only. Publication consistency does not revalidate research numerics or semantic labels.'],
                'phase_source_or_published_files_modified':False,'Git_calls':0,'model_or_eval_or_image_decode':0,'V112_access':0,'automatic_retry':False}
        write(OUT/'REVIEW.json',result);print(json.dumps({'completed':True,'PASS':True,'review':bind(OUT/'REVIEW.json'),'coverage':result['coverage']}),flush=True)
    except Exception:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',{'completed':False,'PASS':False,'entry':bind(entry),'automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:run(a.entry)
    else:ap.error('--prepare or root --entry required')
