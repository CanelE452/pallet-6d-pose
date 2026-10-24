"""One completed phase publication inventory + pure immutable-delta helpers.
--prepare uses only invented metadata/bytes. --entry inventories ONLY V200 after
root freezes the completed phase. This tool never copies or accesses Git/repo.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, os, stat

H=Path(__file__).absolute().parent;P=H.parent.parent;PHASE=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200'
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PHASE_NAME='query_visibility_native_heads_frozen_real_benchmark_20261024_v200'
TEXT=set('.py .sh .md .txt .json .jsonl .csv .tsv .html .htm .yaml .yml .toml .log .patch .diff .svg .pdf'.split())
VISUAL={'.png','.jpg','.jpeg','.webp','.gif'}
PRUNE={'.git','.venv','venv','node_modules','__pycache__','.pytest_cache','.mypy_cache'}

ARMS=('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL', 'C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT')
DOMAINS=('CURRENT223_1784','PUBLIC_HELD45_360')
PRED_RUN='frozen_C58_C59_real_predictions_v1'
AUTHORITY_KEYS=('source_prediction_seal','evaluation_complete','independent_saved_result_review','final_phase_freeze')
ROOT_SEAL={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/PREDICTION_SEAL.json', 'sha256': 'ff28aa2047e73af2b843f93a2c29f6789490504d00494f1188bb746c171a69fa'}
ROOT_BINDINGS={'frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_CURRENT223_1784_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_CURRENT223_1784_OUTPUTS.npz', 'sha256': '49e9767048f35d5d06427cc13776449ffacb19a5f7d8aeeb8fc86bc5a1f2e33c'}, 'frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz', 'sha256': 'bc962112f0dad0ba0f8926cae720a83c95eb2fee1ae6f87f3a1d6f3bb1acde1d'}, 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_CURRENT223_1784_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_CURRENT223_1784_OUTPUTS.npz', 'sha256': '30c663865e46af910b9da9230b7174031fd59bafd664b227001930e4b56178e2'}, 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz', 'sha256': 'cabb4bd02558e4136ca30465bef7774a5e961e8c5ba794c4fe44928d361f214e'}}
ROOT_OWNED=()
ROOT_SAVED=('frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_CURRENT223_1784_OUTPUTS.npz', 'frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz', 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_CURRENT223_1784_OUTPUTS.npz', 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz')

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
ANCESTOR_PHASES=(('V187', 'query_visibility_source_selected_part_correspondence_20261024_v187'), ('V188', 'query_visibility_direct_end_local_counterfactual_20261024_v188'), ('V189', 'query_visibility_sampled_domain_direct_end_20261024_v189'), ('V190', 'query_visibility_direct_end_paired_source_training_20261024_v190'), ('V191', 'query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191'), ('V192', 'query_visibility_source_direct_end_coverage_audit_20261024_v192'), ('V193', 'query_visibility_source_end_pairs_opposite_sham_20261024_v193'), ('V194', 'query_visibility_source_group_split_direct_end_training_20261024_v194'), ('V195', 'query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195'), ('V196', 'query_visibility_grouped_source_holdout_target_readout_20261024_v196'), ('V197', 'query_visibility_train_only_target_contract_readout_20261024_v197'), ('V198', 'query_visibility_native_query_conditioning_feasibility_20261024_v198'), ('V199', 'query_visibility_native_conditioned_grouped_source_training_20261024_v199'))
CLOSING_TAGS=('BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199')

def closing_protected_entries(records,repository,readme_exceptions=None):
    """Metadata-only exact closing pushplans. No old file bodies are reopened."""
    require([r['tag'] for r in records]==list(CLOSING_TAGS),'All prior closing plans ordered BASE through V198')
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
    """Pure ordered BASE through V198 closure; no filesystem operations.

    Each item contains bound metadata objects, not archive payloads. Legacy V188
    lacks an explicit previous_index field; its previous_chain binds V187 index.
    V199 writes previous_index explicitly for future readers.
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
    require('FREEZE' in Path(freeze['binding']['path']).name and freeze['expected_fields'].get('scientific_phase_frozen_no_further_phase_writes') is True,'Exact scientific phase frozen flag, not generic completed/PASS')
    return True

# replaced by root-declared V200 metadata constants
# replaced by root-declared V200 metadata constants
# replaced by root-declared V200 metadata constants
# replaced by root-declared V200 metadata constants
# replaced by root-declared V200 metadata constants
# replaced by root-declared V200 metadata constants

def declared_prediction_artifacts(e,pred,ev,audit):
    require(e['source_prediction_seal']==ROOT_SEAL,'Root-fixed exact V200 prediction seal')
    require(e['evaluation_complete']['path']==str(PHASE/'saved_real_head_evaluation_v1/COMPLETE.json')and e['independent_saved_result_review']['path']==str(PHASE/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),'Exact V200 saved evaluation/numeric audit')
    require(e['final_phase_freeze']['path']==str(PHASE/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),'Exact completed V200 phasefreeze')
    require(pred['completed']is True and pred['PASS']is True and pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and pred['all_batch_repeat2_state_and_input_bytes_exact']is True,'Both frozen predictions/repeat2/state/input closure before annotations')
    require(pred['original_rows2144']==2144 and pred['model_domain_rows4288']==4288 and pred['original_currentU45_publicU108']is True,'Complete original real-domain rows/U')
    require(set(pred['arms'])==set(ARMS),'Both root-fixed RAW arms, no outcome subset')
    require(ev['completed']is True and ev['PASS']is True and ev['prediction_seal']==e['source_prediction_seal']and ev['all_both_predictions_closed_before_any_annotation_or_score_body']is True,'Saved evaluation closed exact both prediction seal')
    require(audit['completed']is True and audit['PASS']is True and audit['complete']==e['evaluation_complete'],'Independent exact saved evaluation review')
    for d in (pred,ev,audit):require(d['selected_model']is None and d['adopted']is False and d['fully_solved']is False,'No winner/adoption/solved authority')
    artifacts={}
    for i,arm in enumerate(ARMS):
        r=pred['arms'][arm]
        require(r['updates']==8000 and r['use_sam']is False and type(r['native_condition'])is bool and r['native_condition']==bool(i),'Both RAW strict zero8/onehot flags and final8000')
        require(set(r['outputs'])==set(DOMAINS),'Both original domains per arm')
        for domain in DOMAINS:
            rel=PRED_RUN+'/'+arm+'/FINAL_'+domain+'_OUTPUTS.npz';b=r['outputs'][domain]
            require(b==ROOT_BINDINGS[rel]and b['path']==str(PHASE/rel),'Exact root-declared four output path/SHA bindings')
            require(len(b['sha256'])==64 and all(c in '0123456789abcdef'for c in b['sha256']),'Output SHA shape')
            artifacts[rel]=b
    require(set(artifacts)==set(ROOT_SAVED)and len(artifacts)==4,'All4 target-free saved16key packets; opaque bytes only')
    require(not e.get('explicit_own_head_relative_paths')and not e.get('explicit_direct_target_relative_paths')and e.get('direct_target_cache_complete')is None,'No new owned weights/sourcefit/targets')
    require(set(e['explicit_saved_output_relative_paths'])==set(artifacts),'No missing/subset/extra saved packet')
    receipts=e.get('phase_completion_receipts',e.get('completion_reviews',[]));require_completion_barriers(receipts)
    freeze=next(r for r in receipts if r['role']=='final_phase_freeze');require(freeze['binding']==e['final_phase_freeze'],'Exact freeze binding in completion barriers')
    return dict(**{k:e[k]for k in AUTHORITY_KEYS},explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=sorted(artifacts),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None,declared_artifact_bindings=artifacts)

def source_artifact_authority(e):
    docs=[read_checked(e[k])for k in AUTHORITY_KEYS[:3]];read_checked(e['final_phase_freeze'])
    return declared_prediction_artifacts(e,*docs)

def authority_template():
    return dict(source_prediction_seal=dict(ROOT_SEAL),evaluation_complete=dict(path=str(PHASE/'saved_real_head_evaluation_v1/COMPLETE.json'),sha256=None),independent_saved_result_review=dict(path=str(PHASE/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),sha256=None),final_phase_freeze=dict(path=str(PHASE/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),sha256=None),explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=list(ROOT_SAVED),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None)

def invented_prediction_metadata():
    e=authority_template()
    for k in AUTHORITY_KEYS[1:]:e[k]['sha256']='a'*64
    e['phase_completion_receipts']=[dict(role=r,binding=e['final_phase_freeze']if r=='final_phase_freeze'else dict(path='/invented/'+('ROOT_PHASE_RESULT.json'if r=='phase_result'else'REVIEW.json'),sha256='b'*64),expected_fields={'completed':True,'PASS':True,**({'scientific_phase_frozen_no_further_phase_writes':True}if r=='final_phase_freeze'else{})})for r in ('phase_result','independent_report_QA','final_phase_freeze')]
    p=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,original_currentU45_publicU108=True,arms={},selected_model=None,adopted=False,fully_solved=False)
    for i,a in enumerate(ARMS):p['arms'][a]=dict(updates=8000,use_sam=False,native_condition=bool(i),outputs={d:dict(ROOT_BINDINGS[PRED_RUN+'/'+a+'/FINAL_'+d+'_OUTPUTS.npz'])for d in DOMAINS})
    ev=dict(completed=True,PASS=True,prediction_seal=e['source_prediction_seal'],all_both_predictions_closed_before_any_annotation_or_score_body=True,selected_model=None,adopted=False,fully_solved=False)
    au=dict(completed=True,PASS=True,complete=e['evaluation_complete'],selected_model=None,adopted=False,fully_solved=False)
    return e,p,ev,au

def classify_file(relative,suffix,size,own,saved):
    require(not own,'Evaluation publication owns no new checkpoints')
    action='INCLUDE' if suffix in TEXT|VISUAL else ('SMALL_OWN_HEAD_REVIEW' if relative in own else ('SAVED_OUTPUT_REVIEW' if relative in saved else 'EXCLUDE'))
    if action=='SMALL_OWN_HEAD_REVIEW':require(suffix=='.pt' and size<16*1024**2,'Owned final head below16MiB')
    if action=='SAVED_OUTPUT_REVIEW':require(suffix=='.npz' and size<32*1024**2,'Explicit source predictions/direct target below32MiB')
    return action

def inventory(entry_path):
    with open_nofollow(entry_path) as f:e=json.load(f)
    require(e['approved_completed_phase_inventory'] is True and e['completed_phase_frozen'] is True,'Root completed-phase entry required')
    require(e['code']=={'path':str(Path(__file__).absolute()),'sha256':sha_file(Path(__file__).absolute())},'Exact inventory code')
    require(e['protocol']['path']==str(H/'PROTOCOL_BEFORE_V200_PUBLICATION_ENUMERATION_OR_COPY.json'),'Exact inventory protocol path')
    frozen_protocol=read_checked(e['protocol']);require(frozen_protocol['code']==e['code'],'Inventory protocol/code closure')
    require(e['source_phase']==str(PHASE) and PHASE.name==PHASE_NAME,'Only exact V200 source root')
    scope=read_checked(e['root_scope']);require(bool(scope),'Explicit root publication scope')
    require_completion_barriers(e['phase_completion_receipts'])
    for r in e['phase_completion_receipts']:
        d=read_checked(r['binding'])
        for k,value in r['expected_fields'].items():require(d[k]==value,'Completion receipt expectation '+k)
    output=safe_parts(e['output_directory']);require(output.is_absolute() and P/'publication_root_preparation' in output.parents and PHASE not in output.parents,'Separate root-admin output, never phase self-recursion')
    output.mkdir(parents=True,exist_ok=False)
    authority=source_artifact_authority(e)
    own=set(authority['explicit_own_head_relative_paths']);pred=set(authority['explicit_saved_output_relative_paths']);target=set(authority['explicit_direct_target_relative_paths']);saved=pred|target
    for relative in own|saved:safe_parts(relative);require(not Path(relative).is_absolute(),'Explicit phase-relative artifact whitelist')
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
            row={'source_root_id':'P','source_absolute':ent.path,'source_relative_path':PHASE_NAME+'/'+rel,'source_sha256':None,'size_bytes':s.st_size,'published_relative':'archive/task-2/pallet-keypoint-visibility/'+PHASE_NAME+'/'+rel,'published_sha256':None,'status':'CANDIDATE_'+action,'action':action,'category':{'INCLUDE':'completed_phase_research_record','SMALL_OWN_HEAD_REVIEW':'explicit_owned_final_head','SAVED_OUTPUT_REVIEW':'explicit_direct_target_sidecar' if rel in target else 'explicit_saved_real_prediction','EXCLUDE':'raw_array_weight_cache_or_nonresearch_payload'}[action],'exclusion_reason':None if action!='EXCLUDE' else 'Raw RGB/SAM/feature/cache arrays, external weights and unselected payload excluded; exact4 new real prediction packets whitelisted; no owned checkpoints/direct-target arrays.','suffix':ext,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'gzipchunks':None}
            if action!='EXCLUDE':
                row['source_sha256']=sha_file(Path(ent.path))
                if rel in authority['declared_artifact_bindings']:require(row['source_sha256']==authority['declared_artifact_bindings'][rel]['sha256'],'Final owned artifact bytes match sealed fit/source-target binding')
                after=ent.stat(follow_symlinks=False)
                require((after.st_size,after.st_mtime_ns,after.st_dev,after.st_ino)==(s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino),'Frozen source drift during inventory')
            rows.append(row)
    rows.sort(key=lambda r:r['source_relative_path'])
    with (output/'V200_DELTA_CANDIDATE_MANIFEST.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    write(output/'DIRECTORY_PRUNE_RECORDS.json',pruned)
    manifest=output/'V200_DELTA_CANDIDATE_MANIFEST.jsonl'
    done={'completed':True,'PASS':True,'scope':e['root_scope'],'completion_receipts':e['phase_completion_receipts'],'candidate_manifest':{'path':str(manifest),'sha256':sha_file(manifest)},'source_phase':str(PHASE),'artifact_authority':authority,'files':len(rows),'actions':dict(Counter(r['action'] for r in rows)),'included_logical_bytes':sum(r['size_bytes'] for r in rows if r['action']!='EXCLUDE'),'repo_access_or_copy_or_Git':0,'source_modified':False,'old_manifest_or_published_paths_modified':False}
    write(output/'INVENTORY_COMPLETE.json',done);print(json.dumps(done),flush=True)

def prepare():
    import copy
    checks=[]
    def add(n,v):checks.append(dict(name=n,PASS=bool(v)))
    raw=dict(status='PUBLISHED_RAW',published_relative='old/raw',gzipchunks=None)
    gz=dict(status='PUBLISHED_GZIP_CHUNKS',published_relative='old/logical',gzip_descriptor_path='old/descriptor',gzipchunks={'parts':[{'path':'old/chunk'}]})
    for path in ['old/raw','old/logical','old/descriptor','old/chunk']:
        try:assert_new_paths([raw,gz],[dict(raw,published_relative=path)]);ok=False
        except AssertionError:ok=True
        add('old_collision_'+path,ok)
    add('newpath',assert_new_paths([raw,gz],[dict(raw,published_relative='new/raw')]))
    add('old_index_suffix',versioned_index_bytes(b'old bytes','V200','new/report.md').endswith(b'old bytes'))
    try:safe_parts('forbidden_v112/file');ok=False
    except AssertionError:ok=True
    add('V112_preaccess',ok)
    e,p,ev,a=invented_prediction_metadata();au=declared_prediction_artifacts(e,p,ev,a);add('root4_RAW_zero8_onehot_authority',len(au['declared_artifact_bindings'])==4)
    add('exact_scientific_freeze_true_accepted',e['phase_completion_receipts'][-1]['expected_fields']['scientific_phase_frozen_no_further_phase_writes'] is True)
    fake=copy.deepcopy((e,p,ev,a));fake[0]['phase_completion_receipts'][-1]['expected_fields']={'completed':True,'PASS':True}
    try:declared_prediction_artifacts(*fake);ok=False
    except AssertionError:ok=True
    add('completed_PASS_only_freeze_without_exact_key_rejected',ok)
    mutations=[('onearm',lambda e,p,v,a:p['arms'].pop(ARMS[1])),('partialdomain',lambda e,p,v,a:p['arms'][ARMS[0]]['outputs'].pop(DOMAINS[1])),('wrongSHA',lambda e,p,v,a:p['arms'][ARMS[0]]['outputs'][DOMAINS[0]].update(sha256='f'*64)),('oldSAMarm',lambda e,p,v,a:p['arms'][ARMS[1]].update(use_sam=True)),('native_intnotbool',lambda e,p,v,a:p['arms'][ARMS[1]].update(native_condition=1)),('nativewrong',lambda e,p,v,a:p['arms'][ARMS[1]].update(native_condition=False)),('earlypred',lambda e,p,v,a:p.update(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False)),('badrepeat',lambda e,p,v,a:p.update(all_batch_repeat2_state_and_input_bytes_exact=False)),('rowcount',lambda e,p,v,a:p.update(original_rows2144=2143)),('domaincount',lambda e,p,v,a:p.update(model_domain_rows4288=4287)),('wrongseal',lambda e,p,v,a:e['source_prediction_seal'].update(sha256='f'*64)),('wrong_eval',lambda e,p,v,a:v.update(prediction_seal={'path':'bad','sha256':'f'*64})),('wrong_audit',lambda e,p,v,a:a.update(complete={'path':'bad','sha256':'f'*64})),('ownweights',lambda e,p,v,a:e.update(explicit_own_head_relative_paths=['head.pt'])),('targets',lambda e,p,v,a:e.update(explicit_direct_target_relative_paths=['target.npz'])),('missingfreeze',lambda e,p,v,a:e['phase_completion_receipts'].pop()),('selected',lambda e,p,v,a:p.update(selected_model=ARMS[1]))]
    for name,mutate in mutations:
        args=copy.deepcopy((e,p,ev,a));mutate(*args)
        try:declared_prediction_artifacts(*args);ok=False
        except (AssertionError,KeyError):ok=True
        add(name+'_rejected',ok)
    add('rawcacheexcluded',classify_file('INPUTS.npz','.npz',1,set(),set())=='EXCLUDE')
    add('4savedopaque',classify_file(ROOT_SAVED[0],'.npz',1,set(),set(ROOT_SAVED))=='SAVED_OUTPUT_REVIEW')
    require(all(x['PASS']for x in checks),'Invented checks')
    write(H/'GENERATED_INVENTED_DELTA_CHECKS.json',dict(completed=True,PASS=True,checks=checks,invented_only=True,actual_source_repo_hash_copy=0))
    proto=dict(completed=True,prepared_only=True,code=dict(path=str(Path(__file__).absolute()),sha256=sha_file(Path(__file__).absolute())),root_code_preparation_scope=dict(path=str(H.parent/'ROOT_V200_PUBLICATION_CODE_PREPARATION_SCOPE_C.json'),sha256='513f01ce27db2aae14d629afc3cb15e43b8579d03a9376a10862acc1d1ff6ffd'),ancestor_tags=[t for t,_ in ANCESTOR_PHASES],immutable_closing_tags=list(CLOSING_TAGS),source_phase=str(PHASE),source_artifacts='Root-declared exact four saved target-free new real prediction packets; bothRAW strict native false/true; old weights/sourcefit not recopied or reopened.',owned_heads=[],saved_arrays=list(ROOT_SAVED),target_arrays=[],original2144_model4288=True,expected_remote_main=None,gzip_text_threshold_bytes=2*1024**2,immutable_policy='BASE+13ancestor deltas/chains/index+14priorclosing plans immutable; BASE historical README sole exception; current README exact suffix; no logical/chunk/descriptor collision.',actual_before_separate_root_entry=0)
    write(H/'PROTOCOL_BEFORE_V200_PUBLICATION_ENUMERATION_OR_COPY.json',proto)
    entry=dict(approved_completed_phase_inventory=False,completed_phase_frozen=False,code=proto['code'],protocol=dict(path=str(H/'PROTOCOL_BEFORE_V200_PUBLICATION_ENUMERATION_OR_COPY.json'),sha256=sha_file(H/'PROTOCOL_BEFORE_V200_PUBLICATION_ENUMERATION_OR_COPY.json')),source_phase=str(PHASE),root_scope=dict(path=None,sha256=None),phase_completion_receipts=[dict(role=r,binding=dict(path=None,sha256=None),expected_fields={'completed':True,'PASS':True,**({'scientific_phase_frozen_no_further_phase_writes':True}if r=='final_phase_freeze'else{})})for r in ('phase_result','independent_report_QA','final_phase_freeze')],output_directory=str(P/'publication_root_preparation/V200_DELTA_INVENTORY_v1'),expected_remote_main=None,**authority_template())
    write(H/'ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json',entry);print(json.dumps(dict(completed=True,PASS=True,invented=len(checks))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:inventory(a.entry)
    else:ap.error('--prepare or root --entry required')
