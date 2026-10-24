"""One completed phase publication inventory + pure immutable-delta helpers.
--prepare uses only invented metadata/bytes. --entry inventories ONLY V189 after
root freezes the completed phase. This tool never copies or accesses Git/repo.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, os, stat

H=Path(__file__).absolute().parent;P=H.parent.parent;PHASE=P/'query_visibility_sampled_domain_direct_end_20261024_v189'
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PHASE_NAME='query_visibility_sampled_domain_direct_end_20261024_v189'
TEXT=set('.py .sh .md .txt .json .jsonl .csv .tsv .html .htm .yaml .yml .toml .log .patch .diff .svg .pdf'.split())
VISUAL={'.png','.jpg','.jpeg','.webp','.gif'}
PRUNE={'.git','.venv','venv','node_modules','__pycache__','.pytest_cache','.mypy_cache'}

def require(v,msg):
    if not bool(v):raise AssertionError(msg)
def safe_parts(path):
    p=Path(path)
    require('..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Forbidden name before filesystem access')
    return p
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def open_nofollow(path):
    p=safe_parts(path);require(p.is_absolute() and P in p.parents,'Explicit project path')
    fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        parts=p.relative_to(P).parts
        for part in parts[:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        final=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Regular source only')
    return os.fdopen(final,'rb')
def sha_file(path):
    h=hashlib.sha256()
    with open_nofollow(path) as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def read_checked(binding):
    require(len(binding['sha256'])==64,'Bound metadata SHA')
    with open_nofollow(binding['path']) as f:data=f.read()
    require(sha_bytes(data)==binding['sha256'],'Bound metadata drift')
    return json.loads(data)
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def physical_paths(entry):
    if entry['status']=='EXCLUDED':return set()
    if entry.get('gzipchunks') is None:return {entry['published_relative']}
    paths={p['path'] for p in entry['gzipchunks']['parts']}
    if entry.get('gzip_descriptor_path'):paths.add(entry['gzip_descriptor_path'])
    return paths
def assert_new_paths(base_entries,delta_entries):
    old=set();new=set()
    for e in base_entries:
        old.update(physical_paths(e))
        if e['status']!='EXCLUDED':old.add(e['published_relative'])
    for e in delta_entries:
        if e['status']=='EXCLUDED':continue
        pp=physical_paths(e)|{e['published_relative']}
        for path in pp:safe_parts(path);require(not Path(path).is_absolute(),'Relative publication path')
        require(not (old & pp),'Old manifest-covered path must remain immutable')
        new.update(pp)
    return True
ANCESTOR_PHASES=(('V187','query_visibility_source_selected_part_correspondence_20261024_v187'),
                 ('V188','query_visibility_direct_end_local_counterfactual_20261024_v188'))


def validate_parent_closure(base_binding,base,ancestors):
    """Pure ordered BASE -> V187 -> V188 closure; no filesystem operations.

    Each item contains bound metadata objects, not archive payloads. Legacy V188
    lacks an explicit previous_index field; its previous_chain binds V187 index.
    V189 writes previous_index explicitly for future readers.
    """
    require(isinstance(base.get('entries'),list),'Initial manifest entries')
    require(len(ancestors)==len(ANCESTOR_PHASES),'All required ancestors, no missing earlier delta')
    entries=list(base['entries']);delta_refs=[];protected=[('PUBLICATION_MANIFEST.json',base_binding)]
    previous=None
    for record,(tag,phase) in zip(ancestors,ANCESTOR_PHASES):
        require(record['tag']==tag,'Exact chronological ancestor order')
        dr='publication/deltas/'+tag+'_20261024_v1/DELTA_MANIFEST.json'
        cr='MANIFEST_CHAIN_AFTER_'+tag+'.json';ir='EXPERIMENT_INDEX_THROUGH_'+tag+'_KO.md'
        manifest=record['manifest'];chain=record['chain_data']
        require(manifest['format']=='immutable-completed-phase-delta-v1' and manifest['phase']==phase,'Ancestor phase authority')
        require(manifest['previous_base_manifest']==base_binding,'Every delta exact initial base binding')
        if previous is not None:
            require(manifest['previous_delta_manifest']==previous['delta_manifest'],'Latest delta binds earlier ancestor manifest')
            require(manifest['previous_chain']==previous['chain'],'Latest delta binds earlier ancestor chain')
            if 'previous_index' in manifest:require(manifest['previous_index']==previous['index'],'Explicit previous index binding')
            require(chain['previous_chain']=={'path':previous['chain_relative'],'sha256':previous['chain']['sha256']},'Chain previous ancestor SHA')
        delta_refs.append({'path':dr,'sha256':record['delta_manifest']['sha256']})
        require(chain['format']=='immutable-publication-manifest-chain-v1' and chain['base']=={'path':'PUBLICATION_MANIFEST.json','sha256':base_binding['sha256']},'Ancestor chain initial base SHA')
        require(chain['deltas']==delta_refs,'Exact full ancestor chain order and SHA')
        require(chain['index']=={'path':ir,'sha256':record['index']['sha256']},'Each ancestor chain binds its versioned index')
        matching=[r for r in manifest['entries'] if r.get('published_relative')==ir and r['status']!='EXCLUDED']
        require(len(matching)==1 and matching[0]['status']=='PUBLISHED_RAW' and matching[0]['published_sha256']==record['index']['sha256'],'Versioned index bytes bound by corresponding delta')
        entries.extend(manifest['entries']);protected.extend([(dr,record['delta_manifest']),(cr,record['chain'])])
        previous=dict(record,chain_relative=cr)
    for relative,binding in protected:
        entries.append(dict(status='PUBLISHED_RAW',published_relative=relative,gzipchunks=None,published_sha256=binding['sha256']))
    return entries


def versioned_index_bytes(previous_bytes,phase_name,portable_relative):
    safe_parts(portable_relative)
    header=('# 추가 완료 기록: '+phase_name+'\n\n'
            '[새 실험 보고서·그림·표·실패 기록]('+portable_relative+')\n\n'
            '아래는 이전 게시 인덱스의 원문입니다. 기존 manifest와 모든 기존 게시 경로는 유지됩니다. '
            '이번 추가 게시가 모델 채택 또는 전체 목표 해결을 뜻하지 않습니다.\n\n---\n\n').encode()
    result=header+previous_bytes;require(result.endswith(previous_bytes),'Exact old index suffix');return result
def bounded_batches(files,limit=480*1024**2):
    require(0<limit<=512*1024**2,'Batch bound');seen=set();out=[];batch=[];size=0
    for r in sorted(files,key=lambda r:r['path']):
        name=r['path'];safe_parts(name)
        require(name.startswith(PREFIX) or name=='readme.md','Only publication or global readme')
        require(name not in seen,'No duplicate physical push path');seen.add(name)
        require(0<=r['size_bytes']<100*1024**2 and len(r['sha256'])==64,'Physical size/hash schema')
        if batch and size+r['size_bytes']>limit:out.append(batch);batch=[];size=0
        batch.append(r);size+=r['size_bytes']
    if batch:out.append(batch)
    require(all(sum(r['size_bytes'] for r in b)<=limit for b in out),'All bounded')
    return out

def inventory(entry_path):
    with open_nofollow(entry_path) as f:e=json.load(f)
    require(e['approved_completed_phase_inventory'] is True and e['completed_phase_frozen'] is True,'Root completed-phase entry required')
    require(e['code']=={'path':str(Path(__file__).absolute()),'sha256':sha_file(Path(__file__).absolute())},'Exact inventory code')
    require(e['protocol']['path']==str(H/'PROTOCOL_BEFORE_V189_PUBLICATION_ENUMERATION_OR_COPY.json'),'Exact inventory protocol path')
    frozen_protocol=read_checked(e['protocol']);require(frozen_protocol['code']==e['code'],'Inventory protocol/code closure')
    require(e['source_phase']==str(PHASE) and PHASE.name==PHASE_NAME,'Only exact V189 source root')
    scope=read_checked(e['root_scope']);require(bool(scope),'Explicit root publication scope')
    require(len(e['phase_completion_receipts'])>=2,'Final phase result and independent report QA barriers required')
    require(any(Path(r['binding']['path']).name=='ROOT_PHASE_RESULT.json' for r in e['phase_completion_receipts']),'Final V189 phase result must be bound before enumeration')
    require(any(r.get('expected_fields',{}).get('PASS') is True for r in e['phase_completion_receipts'] if Path(r['binding']['path']).name!='ROOT_PHASE_RESULT.json'),'Separate completed report QA PASS required')
    for r in e['phase_completion_receipts']:
        d=read_checked(r['binding'])
        for k,value in r['expected_fields'].items():require(d[k]==value,'Completion receipt expectation '+k)
    output=safe_parts(e['output_directory']);require(output.is_absolute() and P/'publication_root_preparation' in output.parents and PHASE not in output.parents,'Separate root-admin output, never phase self-recursion')
    output.mkdir(parents=True,exist_ok=False)
    numerical=set(e.get('explicit_small_numeric_relative_paths',[]))
    require(not numerical,'V189 raw arrays/weights/banks excluded as provenance; no numerical NPZ admission in this delta')
    for p in numerical:
        safe_parts(p);require(not Path(p).is_absolute(),'Numerical path relative to phase')
        require(not any(x in p.casefold() for x in ('input','control_triplet','cache','feature','weight','raw_bank','full_rgb','source_bank')),'Raw control/data/model banks are excluded provenance, never numerical-result whitelist')
    rows=[];pruned=[];stack=[PHASE]
    while stack:
        directory=stack.pop()
        for ent in sorted(os.scandir(directory),key=lambda r:r.name):
            rel=str(Path(ent.path).relative_to(PHASE));name=ent.name
            # No stat/open/read/hash/resolve for V112 names.
            if 'v112' in name.casefold():pruned.append({'relative':rel,'reason':'V112_name_pruned_before_stat','size':None});continue
            if name.casefold() in PRUNE or name.startswith('.') or name.endswith('.pyc'):pruned.append({'relative':rel,'reason':'environment_or_hidden_or_bytecode','size':None});continue
            s=ent.stat(follow_symlinks=False)
            if stat.S_ISLNK(s.st_mode):pruned.append({'relative':rel,'reason':'symlink_not_read_resolved_or_followed','target_size':None});continue
            if stat.S_ISDIR(s.st_mode):stack.append(Path(ent.path));continue
            if not stat.S_ISREG(s.st_mode):pruned.append({'relative':rel,'reason':'nonregular'});continue
            ext=Path(rel).suffix.casefold();action='INCLUDE' if ext in TEXT|VISUAL else ('NUMERIC_REVIEW' if ext=='.npz' and rel in numerical else 'EXCLUDE')
            row={'source_root_id':'P','source_absolute':ent.path,'source_relative_path':PHASE_NAME+'/'+rel,'source_sha256':None,'size_bytes':s.st_size,'published_relative':'archive/task-2/pallet-keypoint-visibility/'+PHASE_NAME+'/'+rel,'published_sha256':None,'status':'CANDIDATE_'+action,'action':action,'category':'completed_phase_research_record' if action=='INCLUDE' else ('explicit_small_numeric_result' if action=='NUMERIC_REVIEW' else 'raw_array_weight_cache_or_unselected_payload'),'exclusion_reason':None if action!='EXCLUDE' else 'Array/weight/cache or nonresearch payload; root may admit only explicitly reviewed small numerical results.','suffix':ext,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'gzipchunks':None}
            if action!='EXCLUDE':
                row['source_sha256']=sha_file(Path(ent.path));after=ent.stat(follow_symlinks=False)
                require((after.st_size,after.st_mtime_ns,after.st_dev,after.st_ino)==(s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino),'Frozen source drift during inventory')
            rows.append(row)
    rows.sort(key=lambda r:r['source_relative_path'])
    with (output/'V189_DELTA_CANDIDATE_MANIFEST.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    write(output/'DIRECTORY_PRUNE_RECORDS.json',pruned)
    manifest=output/'V189_DELTA_CANDIDATE_MANIFEST.jsonl'
    done={'completed':True,'PASS':True,'scope':e['root_scope'],'completion_receipts':e['phase_completion_receipts'],'candidate_manifest':{'path':str(manifest),'sha256':sha_file(manifest)},'source_phase':str(PHASE),'files':len(rows),'actions':dict(Counter(r['action'] for r in rows)),'included_logical_bytes':sum(r['size_bytes'] for r in rows if r['action']!='EXCLUDE'),'repo_access_or_copy_or_Git':0,'source_modified':False,'old_manifest_or_published_paths_modified':False}
    write(output/'INVENTORY_COMPLETE.json',done);print(json.dumps(done),flush=True)

def prepare():
    raw={'status':'PUBLISHED_RAW','published_relative':'archive/old.json','gzipchunks':None}
    gz={'status':'PUBLISHED_GZIP_CHUNKS','published_relative':'archive/old.jsonl','gzipchunks':{'parts':[{'path':'archive/old.jsonl.parts/part00000.gz'}]},'gzip_descriptor_path':'archive/old.jsonl.parts.json'}
    new={'status':'PUBLISHED_RAW','published_relative':'archive/new.json','gzipchunks':None}
    checks=[]
    checks.append(('new_path_disjoint',assert_new_paths([raw,gz],[new])))
    for name,candidate in [('old_raw_rejected',raw),('old_chunk_rejected',{'status':'PUBLISHED_RAW','published_relative':'archive/old.jsonl.parts/part00000.gz','gzipchunks':None}),('old_descriptor_rejected',{'status':'PUBLISHED_RAW','published_relative':'archive/old.jsonl.parts.json','gzipchunks':None})]:
        try:assert_new_paths([raw,gz],[candidate]);ok=False
        except AssertionError:ok=True
        checks.append((name,ok))
    original='이전 인덱스\n'.encode();checks.append(('old_index_exact_suffix',versioned_index_bytes(original,'V189','archive/new/PORTABLE_REPORT_KO.md').endswith(original)))
    try:safe_parts('archive/forbidden_v112/file');ok=False
    except AssertionError:ok=True
    checks.append(('forbidden_name_before_any_stat',ok))
    files=[{'path':PREFIX+f'new/{i}.bin','size_bytes':90*1024**2,'sha256':'a'*64} for i in range(6)]
    bb=bounded_batches(files);checks.append(('480MiB_split_metadata_only',len(bb)==2 and [len(b) for b in bb]==[5,1]))
    try:bounded_batches(files+[files[0]]);ok=False
    except AssertionError:ok=True
    checks.append(('duplicate_physical_path_rejected',ok))
    receipt={'completed':True,'PASS':all(v for _,v in checks),'checks':[{'name':k,'PASS':v} for k,v in checks],'invented_only':True,'actual_phase_enumerations_hashes_copies':0,'repository_or_Git_access':0}
    write(H/'GENERATED_INVENTED_DELTA_CHECKS.json',receipt)
    protocol={'completed':True,'prepared_only':True,'phase':'V189','source_phase':str(PHASE),'old_publication':'Existing immutable BASE plus completed V187 AND V188 delta/chain/index and every covered source/derived path immutable. No rerun of initial full snapshot helpers.','inventory':'Exact V189 only after final completion receipts/root scope; V112 name prune before stat, nofollow symlinks, pycache/pyc/env excluded, raw arrays/weights excluded unless explicit numeric result whitelist. No broad P scan.','copy':'Use existing immutable package_publication_snapshot.py with this new phase-only candidate,1MiB text losslessgzip32MiBchunks, fresh publication/deltas/V189/SOURCE_ARCHIVE_MANIFEST.jsonl and COPY_COMPLETE.json names. Root approves plan; no overwrite.','delta_manifest':'Fresh publication/deltas/V189/DELTA_MANIFEST.json {entries} includes new source records, physical gzipdescriptors, portable phase report/newindex/newtools only. Reject all collisions with base logical/physical paths. Old PUBLICATION_MANIFEST.json never rewritten.','portable':'Fresh archive/task-2/pallet-keypoint-visibility/'+PHASE_NAME+'/PORTABLE_REPORT_KO.md; short root-approved source-only summary + relative original ROOT_REPORT/RESULTS/PHASE, all local literal report images and all tables/code/audit/failure provenance links; gzip logicallinks point actual .parts.json. No absolute local links or invented data.','index':'New root-level EXPERIMENT_INDEX_THROUGH_V189_KO.md prepends phase link and preserves previous index bytes as suffix; old EXPERIMENT_INDEX_KO.md/targetREADME unchanged. Global repo readme.md outside base manifest may prepend newlatestlink and retain prior bytes suffix.','chain':'New MANIFEST_CHAIN_AFTER_V189.json binds immutable base and delta manifest hashes. Chain and independent QA outputs are separately Git-plan-bound, avoiding self-reference. Existing restore tool takes base or delta manifest explicitly.','verify':'Independent original byte roundtrip/gzip sourceSHA+storedpartSHA; credential/relative-link/storedsize checks; actual source numeric reviews remain distinct from publication consistency. Verify oldmanifestSHA unchanged before/after; source algorithm/model unchanged.','push':'Fresh root-approved delta-only plan uses existing strict push_publication_batches.py runner with expected currentmain/localHEAD/commitblob checks;480MiB maximum physical batches; no force, no unrelated files. Initial make_push_plan.py is full-snapshot-specific and must not be rerun for delta.','closing':'Fresh V189-specific publication receipt/state/plan filenames; bind actual payload tip after remote checks, no circular owncommit and no collisions with initial closing records.','claims':'Publish only completed root-frozen V189 scientific/code/result/UNKNOWN/failure records. Actual phase report controls whether source controls/fit preparation/fit results completed; publisher infers no learning/transfer/adoption/fullgoal claim. No V189 phase body or outcome reads during preparation.','actual_artifact_reads_before_root_entry':0,'checks_file':'GENERATED_INVENTED_DELTA_CHECKS.json'}
    protocol['code']={'path':str(Path(__file__).absolute()),'sha256':sha_file(Path(__file__).absolute())}
    protocol_path=H/'PROTOCOL_BEFORE_V189_PUBLICATION_ENUMERATION_OR_COPY.json';write(protocol_path,protocol)
    template={'approved_completed_phase_inventory':False,'completed_phase_frozen':False,'code':protocol['code'],'protocol':{'path':str(protocol_path),'sha256':sha_file(protocol_path)},'source_phase':str(PHASE),'root_scope':{'path':None,'sha256':None},'phase_completion_receipts':[],'explicit_small_numeric_relative_paths':[],'output_directory':str(P/'publication_root_preparation/V189_DELTA_INVENTORY_v1')}
    write(H/'ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:inventory(a.entry)
    else:ap.error('--prepare or root --entry required')
