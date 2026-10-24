"""One completed phase publication inventory + pure immutable-delta helpers.
--prepare uses only invented metadata/bytes. --entry inventories ONLY V198 after
root freezes the completed phase. This tool never copies or accesses Git/repo.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, os, stat

H=Path(__file__).absolute().parent;P=H.parent.parent;PHASE=P/'query_visibility_native_query_conditioning_feasibility_20261024_v198'
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PHASE_NAME='query_visibility_native_query_conditioning_feasibility_20261024_v198'
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
ANCESTOR_PHASES=(('V187', 'query_visibility_source_selected_part_correspondence_20261024_v187'), ('V188', 'query_visibility_direct_end_local_counterfactual_20261024_v188'), ('V189', 'query_visibility_sampled_domain_direct_end_20261024_v189'), ('V190', 'query_visibility_direct_end_paired_source_training_20261024_v190'), ('V191', 'query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191'), ('V192', 'query_visibility_source_direct_end_coverage_audit_20261024_v192'), ('V193', 'query_visibility_source_end_pairs_opposite_sham_20261024_v193'), ('V194', 'query_visibility_source_group_split_direct_end_training_20261024_v194'), ('V195', 'query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195'), ('V196', 'query_visibility_grouped_source_holdout_target_readout_20261024_v196'), ('V197', 'query_visibility_train_only_target_contract_readout_20261024_v197'))
CLOSING_TAGS=('BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197')

def closing_protected_entries(records,repository,readme_exceptions=None):
    """Metadata-only exact closing pushplans. No old file bodies are reopened."""
    require([r['tag'] for r in records]==list(CLOSING_TAGS),'All prior closing plans ordered BASE through V197')
    covered=[];seen=set();historical_readme_seen=False
    for record in records:
        plan=record['plan_data']
        require(plan['authorized_main_push'] is True and plan['repository']==repository and plan['branch']=='main' and plan['remote']=='https://github.com/CanelE452/pallet-6d-pose.git','Prior strict closing plan authority')
        require(plan['batches'],'Nonempty prior closing plan')
        for batch in plan['batches']:
            for item in batch['files']:
                path=str(safe_parts(item['path']))
                if path=='readme.md':
                    require(record['tag']=='BASE' and not historical_readme_seen,'Only one BASE historical root README exception')
                    require(type(item['size_bytes']) is int and 0<=item['size_bytes']<100*1024**2 and isinstance(item['sha256'],str) and len(item['sha256'])==64 and all(c in '0123456789abcdef' for c in item['sha256']),'Historical README sealed SHA/size shape')
                    historical_readme_seen=True
                    if readme_exceptions is not None:readme_exceptions.append(dict(tag='BASE',path='readme.md',sha256=item['sha256'],size_bytes=item['size_bytes'],exception='Historical global README may change only under current global_readme_before exactSHA + prior-bytes suffix guard; not an immutable archived path.'))
                    continue
                require(path.startswith(PREFIX),'All other closing paths must be inside publication prefix')
                relative=path[len(PREFIX):]
                require(relative and not Path(path).is_absolute() and relative not in seen,'Unique prior closing physical paths')
                require(type(item['size_bytes']) is int and 0<=item['size_bytes']<100*1024**2 and len(item['sha256'])==64,'Sealed closing physical hash/size')
                seen.add(relative);covered.append(dict(status='PUBLISHED_RAW',published_relative=relative,gzipchunks=None,published_sha256=item['sha256'],size_bytes=item['size_bytes']))
    return covered


def validate_parent_closure(base_binding,base,ancestors):
    """Pure ordered BASE through V197 closure; no filesystem operations.

    Each item contains bound metadata objects, not archive payloads. Legacy V188
    lacks an explicit previous_index field; its previous_chain binds V187 index.
    V198 writes previous_index explicitly for future readers.
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

def require_completion_barriers(receipts):
    required={'phase_result','independent_report_QA','final_phase_freeze'}
    require(isinstance(receipts,list) and len(receipts)>=3,'Final phase, independent QA and freeze barriers')
    roles=[r.get('role') for r in receipts]
    require(required<=set(roles) and all(roles.count(k)==1 for k in required),'Each required completion role exactly once')
    for r in receipts:
        require(isinstance(r.get('expected_fields'),dict) and bool(r['expected_fields']),'Explicit receipt field checks')
    phase=next(r for r in receipts if r.get('role')=='phase_result')
    qa=next(r for r in receipts if r.get('role')=='independent_report_QA')
    freeze=next(r for r in receipts if r.get('role')=='final_phase_freeze')
    require(Path(phase['binding']['path']).name=='ROOT_PHASE_RESULT.json','Final phase result exact basename')
    require(qa['expected_fields'].get('PASS') is True,'Independent report QA PASS required')
    require('FREEZE' in Path(freeze['binding']['path']).name and any(v is True for v in freeze['expected_fields'].values()),'Explicit final freeze receipt and true frozen-state expectation')
    return True

def classify_file(relative,suffix,size,own,saved):
    require(not own and not saved,'V198 has empty own-head and saved-array whitelists')
    return 'INCLUDE' if suffix in TEXT|VISUAL else 'EXCLUDE'

def inventory(entry_path):
    with open_nofollow(entry_path) as f:e=json.load(f)
    require(e['approved_completed_phase_inventory'] is True and e['completed_phase_frozen'] is True,'Root completed-phase entry required')
    require(e['code']=={'path':str(Path(__file__).absolute()),'sha256':sha_file(Path(__file__).absolute())},'Exact inventory code')
    require(e['protocol']['path']==str(H/'PROTOCOL_BEFORE_V198_PUBLICATION_ENUMERATION_OR_COPY.json'),'Exact inventory protocol path')
    frozen_protocol=read_checked(e['protocol']);require(frozen_protocol['code']==e['code'],'Inventory protocol/code closure')
    require(e['source_phase']==str(PHASE) and PHASE.name==PHASE_NAME,'Only exact V198 source root')
    scope=read_checked(e['root_scope']);require(bool(scope),'Explicit root publication scope')
    require_completion_barriers(e['phase_completion_receipts'])
    for r in e['phase_completion_receipts']:
        d=read_checked(r['binding'])
        for k,value in r['expected_fields'].items():require(d[k]==value,'Completion receipt expectation '+k)
    output=safe_parts(e['output_directory']);require(output.is_absolute() and P/'publication_root_preparation' in output.parents and PHASE not in output.parents,'Separate root-admin output, never phase self-recursion')
    output.mkdir(parents=True,exist_ok=False)
    # Source-only control-data diagnostic: no checkpoint/score/array inclusion or seal dependency.
    own=set(e.get('explicit_own_head_relative_paths',[]));saved=set(e.get('explicit_saved_output_relative_paths',[]))
    require(not own and not saved,'No weights or saved arrays in V198 publication')
    require('source_prediction_seal' not in e and 'source_fit_complete' not in e,'No prediction or fit seal fields in source-control entry')
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
            ext=Path(rel).suffix.casefold();action=classify_file(rel,ext,s.st_size,own,saved)
            row={'source_root_id':'P','source_absolute':ent.path,'source_relative_path':PHASE_NAME+'/'+rel,'source_sha256':None,'size_bytes':s.st_size,'published_relative':'archive/task-2/pallet-keypoint-visibility/'+PHASE_NAME+'/'+rel,'published_sha256':None,'status':'CANDIDATE_'+action,'action':action,'category':{'INCLUDE':'completed_phase_research_record','EXCLUDE':'all_arrays_weights_cache_or_nonresearch_payload'}[action],'exclusion_reason':None if action!='EXCLUDE' else 'All array/weight/cache payloads excluded for this source-only control-data diagnostic; no numerical-array or weight whitelist.','suffix':ext,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'gzipchunks':None}
            if action!='EXCLUDE':
                row['source_sha256']=sha_file(Path(ent.path));after=ent.stat(follow_symlinks=False)
                require((after.st_size,after.st_mtime_ns,after.st_dev,after.st_ino)==(s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino),'Frozen source drift during inventory')
            rows.append(row)
    rows.sort(key=lambda r:r['source_relative_path'])
    with (output/'V198_DELTA_CANDIDATE_MANIFEST.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    write(output/'DIRECTORY_PRUNE_RECORDS.json',pruned)
    manifest=output/'V198_DELTA_CANDIDATE_MANIFEST.jsonl'
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
    original='이전 인덱스\n'.encode();checks.append(('old_index_exact_suffix',versioned_index_bytes(original,'V198','archive/new/PORTABLE_REPORT_KO.md').endswith(original)))
    try:safe_parts('archive/forbidden_v112/file');ok=False
    except AssertionError:ok=True
    checks.append(('forbidden_name_before_any_stat',ok))
    files=[{'path':PREFIX+f'new/{i}.bin','size_bytes':90*1024**2,'sha256':'a'*64} for i in range(6)]
    bb=bounded_batches(files);checks.append(('480MiB_split_metadata_only',len(bb)==2 and [len(b) for b in bb]==[5,1]))
    try:bounded_batches(files+[files[0]]);ok=False
    except AssertionError:ok=True
    checks.append(('duplicate_physical_path_rejected',ok))
    checks.append(('own_head_excluded',classify_file('own.pt','.pt',12,set(),set())=='EXCLUDE'))
    checks.append(('saved_output_excluded',classify_file('out.npz','.npz',12,set(),set())=='EXCLUDE'))
    checks.append(('raw_cache_excluded',classify_file('RAW_INPUTS.npz','.npz',12,set(),set())=='EXCLUDE'))
    for name,args in [('any_own_head',('own.pt','.pt',12,{'own.pt'},set())),('any_saved_array',('out.npz','.npz',12,set(),{'out.npz'}))]:
        try:classify_file(*args);ok=False
        except AssertionError:ok=True
        checks.append((name+'_rejected',ok))
    barriers=[dict(role='phase_result',binding={'path':'/invented/ROOT_PHASE_RESULT.json'},expected_fields={'completed':True}),dict(role='independent_report_QA',binding={'path':'/invented/REVIEW.json'},expected_fields={'PASS':True}),dict(role='final_phase_freeze',binding={'path':'/invented/ROOT_FINAL_PHASE_FREEZE.json'},expected_fields={'frozen':True})]
    checks.append(('phase_QA_freeze_barriers_accept',require_completion_barriers(barriers)))
    alternate=[dict(r) for r in barriers];alternate[-1]=dict(alternate[-1],expected_fields={'scientific_phase_frozen_no_further_phase_writes':True})
    checks.append(('explicit_alternate_freeze_field_accept',require_completion_barriers(alternate)))
    for name,r in [('missing_freeze',barriers[:2]),('wrong_phase_basename',[dict(barriers[0],binding={'path':'/invented/OTHER.json'})]+barriers[1:]),('QA_without_PASS',[barriers[0],dict(barriers[1],expected_fields={'completed':True}),barriers[2]])]:
        try:require_completion_barriers(r);ok=False
        except AssertionError:ok=True
        checks.append((name+'_rejected',ok))
    receipt={'completed':True,'PASS':all(v for _,v in checks),'checks':[{'name':k,'PASS':v} for k,v in checks],'invented_only':True,'actual_phase_enumerations_hashes_copies':0,'repository_or_Git_access':0}
    write(H/'GENERATED_INVENTED_DELTA_CHECKS.json',receipt)
    scope=H.parent/'ROOT_V198_PUBLICATION_CODE_PREPARATION_SCOPE_B.json'
    protocol={'completed':True,'prepared_only':True,'phase':'V198','source_phase':str(PHASE),'code':{'path':str(Path(__file__).absolute()),'sha256':sha_file(Path(__file__).absolute())},'root_code_preparation_scope':{'path':str(scope),'sha256':sha_file(scope)},'ancestor_tags':[t for t,_ in ANCESTOR_PHASES],'immutable_closing_tags':list(CLOSING_TAGS),'own_head_whitelist':[],'saved_array_whitelist':[],'direct_target_whitelist':[],'inventory':'Only exact root-frozen V198 after final phase_result, independent_report_QA and final_phase_freeze receipts. V112 name prune before stat; componentwise NOFOLLOW; exclude arrays, weights, rawRGB banks, SAM/features/cache and environments. No prediction/fit seal dependency or broad project scan.','copy':'Original bytes;2MiB text threshold then lossless gzip32MiB chunks with physical descriptor and reversible logical path. No model/tensor/image decode.','immutable_policy':'BASE and all11 V187..V197 delta/chain/index plus all12closing paths, logical/raw/chunk/descriptor/cycle-excluded paths immutable. One historical BASE README entry shape exception; current README exact-before-SHA and old-byte suffix remain required.','actual_inventory_or_repo_or_parent_reads':0,'automatic_retry':False}
    write(H/'PROTOCOL_BEFORE_V198_PUBLICATION_ENUMERATION_OR_COPY.json',protocol)
    write(H/'ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json',{'approved_completed_phase_inventory':False,'completed_phase_frozen':False,'code':protocol['code'],'protocol':{'path':str(H/'PROTOCOL_BEFORE_V198_PUBLICATION_ENUMERATION_OR_COPY.json'),'sha256':sha_file(H/'PROTOCOL_BEFORE_V198_PUBLICATION_ENUMERATION_OR_COPY.json')},'source_phase':str(PHASE),'root_scope':{'path':None,'sha256':None},'phase_completion_receipts':[dict(role=role,binding={'path':None,'sha256':None},expected_fields=None)for role in ('phase_result','independent_report_QA','final_phase_freeze')],'explicit_own_head_relative_paths':[],'explicit_saved_output_relative_paths':[],'output_directory':str(P/'publication_root_preparation/V198_DELTA_INVENTORY_v1'),'expected_remote_main':None,'automatic_retry':False})
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:inventory(a.entry)
    else:ap.error('--prepare or root --entry required')
