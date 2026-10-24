"""One completed phase publication inventory + pure immutable-delta helpers.
--prepare uses only invented metadata/bytes. --entry inventories ONLY V195 after
root freezes the completed phase. This tool never copies or accesses Git/repo.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, os, stat

H=Path(__file__).absolute().parent;P=H.parent.parent;PHASE=P/'query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195'
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PHASE_NAME='query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195'
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
                 ('V188','query_visibility_direct_end_local_counterfactual_20261024_v188'),
                 ('V189','query_visibility_sampled_domain_direct_end_20261024_v189'),
                 ('V190','query_visibility_direct_end_paired_source_training_20261024_v190'),
                 ('V191','query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191'),
                 ('V192','query_visibility_source_direct_end_coverage_audit_20261024_v192'),
                 ('V193','query_visibility_source_end_pairs_opposite_sham_20261024_v193'),
                 ('V194','query_visibility_source_group_split_direct_end_training_20261024_v194'))
CLOSING_TAGS=('BASE','V187','V188','V189','V190','V191','V192','V193','V194')

def closing_protected_entries(records,repository,readme_exceptions=None):
    """Metadata-only exact closing pushplans. No old file bodies are reopened."""
    require([r['tag'] for r in records]==list(CLOSING_TAGS),'All prior closing plans ordered BASE/V187/V188/V189/V190/V191/V192/V193/V194')
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
    """Pure ordered BASE -> V187 -> V188 -> V189 -> V190 -> V191 -> V192 closure; no filesystem operations.

    Each item contains bound metadata objects, not archive payloads. Legacy V188
    lacks an explicit previous_index field; its previous_chain binds V187 index.
    V195 writes previous_index explicitly for future readers.
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

ARMS=('C56_RAW96_DIRECT_END_GROUPED','C57_RAW96_PLUS_FROZEN_SAM_DIRECT_END_GROUPED')
DOMAINS=('CURRENT223_1784','PUBLIC_HELD45_360')
PRED_RUN='frozen_C56_C57_real_predictions_v1'
FIT_PHASE=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
FIT_RUN='runs/C56_C57_DIRECT_END_GROUPED_SOURCE_FIXED8000'
AUTHORITY_KEYS=('source_prediction_seal','source_fit_complete','source_fit_integrity_review','all2_final_states_freeze','evaluation_complete','independent_saved_result_review','final_phase_freeze')

def declared_prediction_artifacts(e,pred,fit,review,bar,ev,audit,phase=None,fitphase=None):
    if phase is None:phase=PHASE
    if fitphase is None:fitphase=FIT_PHASE
    require(e['source_fit_integrity_review']['path']==str(fitphase/'independent_grouped_source_fit_review_a/actual_review_v1/REVIEW.json'),'Exact already-published V194 integrity receipt')
    require(e['evaluation_complete']['path']==str(phase/'saved_real_head_evaluation_v1/COMPLETE.json') and e['independent_saved_result_review']['path']==str(phase/'independent_saved_result_review_preparation_a/actual_review_v1/REVIEW.json'),'Exact closed V195 evaluation/audit receipt paths')
    require(e['final_phase_freeze']['path']==str(phase/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),'Exact final V195 freeze path')
    require(e['source_fit_complete']['path']==str(fitphase/FIT_RUN/'COMPLETE.json'),'Exact already-published V194 sourcefit')
    require(fit['completed'] is True and fit['PASS'] is True and fit['source_only'] is True and fit['from_scratch'] is True and fit['steps_each']==8000 and fit['total_updates']==16000,'Both fixed8000 complete before real predictions')
    require(fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs'] is True and fit['all2_freeze']==e['all2_final_states_freeze'],'Exact both-final source barrier')
    require(set(fit['arms'])==set(ARMS) and fit['protected_real_COMMON_VAL_access']==0,'Both sourcearms/protected-sourcefit contract')
    require(bar['completed'] is True and bar['source_TRAIN_or_HOLDOUT_prediction_emitted'] is False and bar['protected_real_COMMON_VAL_access'] is False and bar['checkpoint_or_arm_selection'] is False,'Both states frozen before outputs, no choice')
    require(bar['checkpoints']=={name:fit['arms'][name]['checkpoint'] for name in ARMS},'Final source checkpoints referenced metadata-only')
    require(review['completed'] is True and review['PASS'] is True and review['source_fit_complete']==e['source_fit_complete'],'Independent exact-sourcefit review')
    require(e['source_prediction_seal']['path']==str(phase/PRED_RUN/'PREDICTION_SEAL.json'),'Exact new real-prediction seal')
    require(pred['completed'] is True and pred['PASS'] is True and pred['source_fit_complete']==e['source_fit_complete'] and pred['source_fit_integrity_review']==e['source_fit_integrity_review'],'Real seal exactfit/review provenance')
    require(pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'] is True and pred['all_batch_repeat2_state_and_input_bytes_exact'] is True,'Bothpredictionseal and repeat/input/state closure')
    require(pred['original_rows2144']==2144 and pred['model_domain_rows4288']==4288 and pred['original_currentU45_publicU108'] is True,'Full original real domains/U')
    require(set(pred['arms'])==set(ARMS),'Both real arm outputs, no winner subset')
    require(ev['completed'] is True and ev['PASS'] is True and ev['prediction_seal']==e['source_prediction_seal'] and ev['all_both_predictions_closed_before_any_annotation_or_score_body'] is True,'Separate evaluation after both prediction seal')
    require(audit['completed'] is True and audit['PASS'] is True and audit['complete']==e['evaluation_complete'],'Independent exact saved-evaluation review')
    for d in (fit,pred,ev,audit):require(d['selected_model'] is None and d['adopted'] is False and d['fully_solved'] is False,'No adoption/winner/fullgoal claim')
    saved=set();bindings={}
    for ai,name in enumerate(ARMS):
        f=fit['arms'][name];r=pred['arms'][name]
        require(f['updates']==r['updates']==8000 and f['use_sam']==r['use_sam']==bool(ai),'Fixed source/prediction arms')
        require(r['checkpoint']==f['checkpoint'] and r['source_final_state_sha']==f['final_state_sha'],'Same final source state; never reopen weights')
        require(set(r['outputs'])==set(DOMAINS),'Exactly current/public original domains')
        for domain in DOMAINS:
            rel=PRED_RUN+'/'+name+'/FINAL_'+domain+'_OUTPUTS.npz';binding=r['outputs'][domain]
            require(binding['path']==str(phase/rel) and len(binding['sha256'])==64,'Exact new real packet binding')
            saved.add(rel);bindings[rel]=binding
    require(not e.get('explicit_own_head_relative_paths') and not e.get('explicit_direct_target_relative_paths'),'No owned weights/direct targets in this evaluation phase')
    require(set(e['explicit_saved_output_relative_paths'])==saved and len(saved)==4,'Exactly all4 new real saved outputs')
    receipts=e.get('phase_completion_receipts',e.get('completion_reviews',[]))
    require_completion_barriers(receipts)
    freeze=next(r for r in receipts if r['role']=='final_phase_freeze')
    require(freeze['binding']==e['final_phase_freeze'],'Same final frozen phase exactbinding')
    return dict(**{k:e[k] for k in AUTHORITY_KEYS},explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=sorted(saved),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None,declared_artifact_bindings=bindings)

def source_artifact_authority(e):
    pred=read_checked(e['source_prediction_seal']);fit=read_checked(e['source_fit_complete']);review=read_checked(e['source_fit_integrity_review']);bar=read_checked(e['all2_final_states_freeze']);ev=read_checked(e['evaluation_complete']);audit=read_checked(e['independent_saved_result_review'])
    read_checked(e['final_phase_freeze'])
    return declared_prediction_artifacts(e,pred,fit,review,bar,ev,audit)

def authority_template():
    refs={k:dict(path=None,sha256=None) for k in AUTHORITY_KEYS}
    refs['source_prediction_seal']['path']=str(PHASE/PRED_RUN/'PREDICTION_SEAL.json')
    refs['source_fit_complete']['path']=str(FIT_PHASE/FIT_RUN/'COMPLETE.json')
    refs['source_fit_integrity_review']['path']=str(FIT_PHASE/'independent_grouped_source_fit_review_a/actual_review_v1/REVIEW.json')
    refs['evaluation_complete']['path']=str(PHASE/'saved_real_head_evaluation_v1/COMPLETE.json')
    refs['independent_saved_result_review']['path']=str(PHASE/'independent_saved_result_review_preparation_a/actual_review_v1/REVIEW.json')
    refs['final_phase_freeze']['path']=str(PHASE/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json')
    return dict(**refs,explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=[PRED_RUN+'/'+a+'/FINAL_'+d+'_OUTPUTS.npz' for a in ARMS for d in DOMAINS],explicit_direct_target_relative_paths=[],direct_target_cache_complete=None)

def invented_prediction_metadata(phase=PHASE,fitphase=FIT_PHASE):
    e=authority_template()
    for k in AUTHORITY_KEYS:e[k]=dict(path=str(phase/('INVENTED_'+k+'.json')),sha256='a'*64)
    e['source_fit_integrity_review']['path']=str(fitphase/'independent_grouped_source_fit_review_a/actual_review_v1/REVIEW.json');e['evaluation_complete']['path']=str(phase/'saved_real_head_evaluation_v1/COMPLETE.json');e['independent_saved_result_review']['path']=str(phase/'independent_saved_result_review_preparation_a/actual_review_v1/REVIEW.json');e['final_phase_freeze']['path']=str(phase/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json');e['source_fit_complete']['path']=str(fitphase/FIT_RUN/'COMPLETE.json');e['source_prediction_seal']['path']=str(phase/PRED_RUN/'PREDICTION_SEAL.json')
    e['phase_completion_receipts']=[dict(role=r,binding=e['final_phase_freeze'] if r=='final_phase_freeze' else dict(path='/invented/'+('ROOT_PHASE_RESULT.json' if r=='phase_result' else 'REVIEW.json'),sha256='b'*64),expected_fields={'PASS':True}) for r in ('phase_result','independent_report_QA','final_phase_freeze')]
    fit=dict(completed=True,PASS=True,source_only=True,from_scratch=True,steps_each=8000,total_updates=16000,all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs=True,all2_freeze=e['all2_final_states_freeze'],arms={},selected_model=None,adopted=False,fully_solved=False,protected_real_COMMON_VAL_access=0)
    pred=dict(completed=True,PASS=True,source_fit_complete=e['source_fit_complete'],source_fit_integrity_review=e['source_fit_integrity_review'],all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,original_currentU45_publicU108=True,arms={},selected_model=None,adopted=False,fully_solved=False)
    for ai,name in enumerate(ARMS):
        cp=dict(path=str(fitphase/FIT_RUN/name/'FINAL_CHECKPOINT.pt'),sha256='c'*64)
        fit['arms'][name]=dict(updates=8000,use_sam=bool(ai),checkpoint=cp,final_state_sha='d'*64)
        pred['arms'][name]=dict(updates=8000,use_sam=bool(ai),checkpoint=cp,source_final_state_sha='d'*64,outputs={d:dict(path=str(phase/PRED_RUN/name/('FINAL_'+d+'_OUTPUTS.npz')),sha256='e'*64) for d in DOMAINS})
    bar=dict(completed=True,source_TRAIN_or_HOLDOUT_prediction_emitted=False,protected_real_COMMON_VAL_access=False,checkpoint_or_arm_selection=False,checkpoints={a:r['checkpoint'] for a,r in fit['arms'].items()})
    review=dict(completed=True,PASS=True,source_fit_complete=e['source_fit_complete'])
    ev=dict(completed=True,PASS=True,prediction_seal=e['source_prediction_seal'],all_both_predictions_closed_before_any_annotation_or_score_body=True,selected_model=None,adopted=False,fully_solved=False)
    audit=dict(completed=True,PASS=True,complete=e['evaluation_complete'],selected_model=None,adopted=False,fully_solved=False)
    return e,pred,fit,review,bar,ev,audit

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
    require(e['protocol']['path']==str(H/'PROTOCOL_BEFORE_V195_PUBLICATION_ENUMERATION_OR_COPY.json'),'Exact inventory protocol path')
    frozen_protocol=read_checked(e['protocol']);require(frozen_protocol['code']==e['code'],'Inventory protocol/code closure')
    require(e['source_phase']==str(PHASE) and PHASE.name==PHASE_NAME,'Only exact V195 source root')
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
    with (output/'V195_DELTA_CANDIDATE_MANIFEST.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    write(output/'DIRECTORY_PRUNE_RECORDS.json',pruned)
    manifest=output/'V195_DELTA_CANDIDATE_MANIFEST.jsonl'
    done={'completed':True,'PASS':True,'scope':e['root_scope'],'completion_receipts':e['phase_completion_receipts'],'candidate_manifest':{'path':str(manifest),'sha256':sha_file(manifest)},'source_phase':str(PHASE),'artifact_authority':authority,'files':len(rows),'actions':dict(Counter(r['action'] for r in rows)),'included_logical_bytes':sum(r['size_bytes'] for r in rows if r['action']!='EXCLUDE'),'repo_access_or_copy_or_Git':0,'source_modified':False,'old_manifest_or_published_paths_modified':False}
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
    original='이전 인덱스\n'.encode();checks.append(('old_index_exact_suffix',versioned_index_bytes(original,'V195','archive/new/PORTABLE_REPORT_KO.md').endswith(original)))
    try:safe_parts('archive/forbidden_v112/file');ok=False
    except AssertionError:ok=True
    checks.append(('forbidden_name_before_any_stat',ok))
    files=[{'path':PREFIX+f'new/{i}.bin','size_bytes':90*1024**2,'sha256':'a'*64} for i in range(6)]
    bb=bounded_batches(files);checks.append(('480MiB_split_metadata_only',len(bb)==2 and [len(b) for b in bb]==[5,1]))
    try:bounded_batches(files+[files[0]]);ok=False
    except AssertionError:ok=True
    checks.append(('duplicate_physical_path_rejected',ok))
    checks.append(('ownweights_excluded',classify_file('own.pt','.pt',12,set(),set())=='EXCLUDE'))
    checks.append(('saved4_whitelist',classify_file('out.npz','.npz',12,set(),{'out.npz'})=='SAVED_OUTPUT_REVIEW'))
    checks.append(('raw_cache_excluded',classify_file('RAW_INPUTS.npz','.npz',12,set(),set())=='EXCLUDE'))
    e,pred,fit,review,bar,ev,audit=invented_prediction_metadata()
    au=declared_prediction_artifacts(e,pred,fit,review,bar,ev,audit)
    checks.append(('two_frozen8000_real4_no_weights_closure',len(au['declared_artifact_bindings'])==4 and not au['explicit_own_head_relative_paths']))
    import copy
    cases=[('partial_outputs',lambda ee,pp,ff,rr,bb,vv,aa:pp['arms'][ARMS[0]]['outputs'].pop(DOMAINS[1])),('early_outputs',lambda ee,pp,ff,rr,bb,vv,aa:pp.update(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False)),('wrong_fit_review',lambda ee,pp,ff,rr,bb,vv,aa:rr.update(source_fit_complete=dict(ee['source_fit_complete'],sha256='9'*64))),('wrong_evaluation_seal',lambda ee,pp,ff,rr,bb,vv,aa:vv.update(prediction_seal=dict(ee['source_prediction_seal'],sha256='9'*64))),('wrong_numeric_complete',lambda ee,pp,ff,rr,bb,vv,aa:aa.update(complete=dict(ee['evaluation_complete'],sha256='9'*64))),('own_head',lambda ee,pp,ff,rr,bb,vv,aa:ee.update(explicit_own_head_relative_paths=['head.pt']))]
    for name,change in cases:
        args=copy.deepcopy((e,pred,fit,review,bar,ev,audit));change(*args)
        try:declared_prediction_artifacts(*args);ok=False
        except (AssertionError,KeyError):ok=True
        checks.append((name+'_rejected',ok))
    barriers=[dict(role='phase_result',binding={'path':'/invented/ROOT_PHASE_RESULT.json'},expected_fields={'completed':True}),dict(role='independent_report_QA',binding={'path':'/invented/REVIEW.json'},expected_fields={'PASS':True}),dict(role='final_phase_freeze',binding={'path':'/invented/ROOT_FINAL_PHASE_FREEZE.json'},expected_fields={'frozen':True})]
    checks.append(('phase_QA_freeze_barriers_accept',require_completion_barriers(barriers)))
    for name,r in [('missing_freeze',barriers[:2]),('wrong_phase_basename',[dict(barriers[0],binding={'path':'/invented/OTHER.json'})]+barriers[1:]),('QA_without_PASS',[barriers[0],dict(barriers[1],expected_fields={'completed':True}),barriers[2]])]:
        try:require_completion_barriers(r);ok=False
        except AssertionError:ok=True
        checks.append((name+'_rejected',ok))
    receipt={'completed':True,'PASS':all(v for _,v in checks),'checks':[{'name':k,'PASS':v} for k,v in checks],'invented_only':True,'actual_phase_enumerations_hashes_copies':0,'repository_or_Git_access':0}
    write(H/'GENERATED_INVENTED_DELTA_CHECKS.json',receipt)
    protocol={'completed':True,'prepared_only':True,'phase':'V195','source_phase':str(PHASE),'historical_README_exception':'Only one BASE closingplan root readme.md is SHA/size-shape verified and explicitly reported but not path-protected; current global_readme_before exactSHA+suffix guard remains. NonBASE readme and all other outsideprefix paths failclosed.','old_publication':'Existing immutable BASE plus completed V187/V188/V189/V190/V191/V192/V193/V194 delta/chain/index AND closing physical paths and every covered source/derived path immutable. No rerun of initial full snapshot helpers.','inventory':'Exact V195 only after final completion receipts/root scope; V112 name prune before stat, nofollow symlinks, pycache/pyc/env excluded, all raw input/SAM/feature/cache arrays excluded; 0ownedheads and exactly4 real predictions<32MiB; sourcefit/review/freeze + predictionseal/evaluation/audit/finalphase closure required; all raw/cache/externalweights excluded; exactfinalCOMPLETE sourcefit-integrity + new-real prediction/eval/audit/finalfreeze closure required. Final ROOT_PHASE, independent reportQA and freeze receipts are mandatory. No broad P scan.','copy':'Use existing immutable package_publication_snapshot.py with this new phase-only candidate,1MiB text losslessgzip32MiBchunks, fresh publication/deltas/V195/SOURCE_ARCHIVE_MANIFEST.jsonl and COPY_COMPLETE.json names. Root approves plan; no overwrite.','delta_manifest':'Fresh publication/deltas/V195/DELTA_MANIFEST.json {entries} includes new source records, physical gzipdescriptors, portable phase report/newindex/newtools only. Reject all collisions with base logical/physical paths. Old PUBLICATION_MANIFEST.json never rewritten.','portable':'Fresh archive/task-2/pallet-keypoint-visibility/'+PHASE_NAME+'/PORTABLE_REPORT_KO.md; short root-approved frozen real comparison; D/E human and public weak owner separate; source includes public real + relative original ROOT_REPORT/RESULTS/PHASE, all local literal report images and all tables/code/audit/failure provenance links; gzip logicallinks point actual .parts.json. No absolute local links or invented data.','index':'New root-level EXPERIMENT_INDEX_THROUGH_V195_KO.md prepends phase link and preserves previous index bytes as suffix; old EXPERIMENT_INDEX_KO.md/targetREADME unchanged. Global repo readme.md outside base manifest may prepend newlatestlink and retain prior bytes suffix.','chain':'New MANIFEST_CHAIN_AFTER_V195.json binds immutable base and delta manifest hashes. Chain and independent QA outputs are separately Git-plan-bound, avoiding self-reference. Existing restore tool takes base or delta manifest explicitly.','verify':'Independent original byte roundtrip/gzip sourceSHA+storedpartSHA; credential/relative-link/storedsize checks; actual source numeric reviews remain distinct from publication consistency. Verify oldmanifestSHA unchanged before/after; source algorithm/model unchanged.','push':'Fresh root-approved delta-only plan uses existing strict push_publication_batches.py runner with expected currentmain/localHEAD/commitblob checks;480MiB maximum physical batches; no force, no unrelated files. Initial make_push_plan.py is full-snapshot-specific and must not be rerun for delta.','closing':'Fresh V195-specific publication receipt/state/plan filenames; bind actual payload tip after remote checks, no circular owncommit and no collisions with initial closing records.','claims':'Publish only completed root-frozen V195 scientific/code/result/UNKNOWN/failure records. Completed evaluation-only old6+new2 frozen comparison and fixed68 review gallery; repeatedly exposed/nonblind and legacy realCAL provenance. No causal/transfer/improvement/adoption/fullgoal claim; historical source exposure and weak authority remain explicit. No V195 phase body or outcome reads during preparation.','actual_artifact_reads_before_root_entry':0,'checks_file':'GENERATED_INVENTED_DELTA_CHECKS.json'}
    protocol['code']={'path':str(Path(__file__).absolute()),'sha256':sha_file(Path(__file__).absolute())}
    protocol_path=H/'PROTOCOL_BEFORE_V195_PUBLICATION_ENUMERATION_OR_COPY.json';write(protocol_path,protocol)
    template={'approved_completed_phase_inventory':False,'completed_phase_frozen':False,'code':protocol['code'],'protocol':{'path':str(protocol_path),'sha256':sha_file(protocol_path)},'source_phase':str(PHASE),'root_scope':{'path':None,'sha256':None},'phase_completion_receipts':[{'role':role,'binding':{'path':None,'sha256':None},'expected_fields':None} for role in ('phase_result','independent_report_QA','final_phase_freeze')],**authority_template(),'output_directory':str(P/'publication_root_preparation/V195_DELTA_INVENTORY_v1')}
    write(H/'ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:inventory(a.entry)
    else:ap.error('--prepare or root --entry required')
