"""Mechanical, code/metadata-only derivation. Never read any scientific body."""
from pathlib import Path
import ast, hashlib, json, re

H=Path(__file__).absolute().parent;R=H.parent;P=R.parent
Q=R/'V200_DELTA_REVIEW_PREPARATION_C'
S=R/'ROOT_V200_PUBLICATION_CODE_PREPARATION_SCOPE_C.json'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):
    p=Path(b['path']);assert 'v112'not in str(p).casefold() and not p.is_symlink();assert bind(p)['sha256']==b['sha256'];return p
def write(p,x):
    with p.open('x')as f:f.write(x if isinstance(x,str)else json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert bind(S)['sha256']=='513f01ce27db2aae14d629afc3cb15e43b8579d03a9376a10862acc1d1ff6ffd'
scope=json.loads(S.read_text());ready={k:json.loads(checked(scope[k]).read_text()) for k in ['generic_publisher_ready','generic_auditor_ready','prior_real_benchmark_publisher_ready','prior_real_benchmark_auditor_ready']}
def tool(k,name):
    r=ready[k];b=r['files'][name]if 'files'in r else r[name];return checked(b).read_text()
base=tool('prior_real_benchmark_publisher_ready','prepare_completed_phase_delta.py')
generic=tool('generic_publisher_ready','prepare_completed_phase_delta.py')
builder=tool('generic_publisher_ready','build_completed_phase_delta.py')
auditor=tool('prior_real_benchmark_auditor_ready','code')
gauditor=tool('generic_auditor_ready','code')
def node(t,name,typ=ast.FunctionDef):
    return next(n for n in ast.parse(t).body if isinstance(n,typ)and(getattr(n,'name',None)==name or (isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id==name for x in n.targets))))
def segment(t,name,typ=ast.FunctionDef):return ast.get_source_segment(t,node(t,name,typ))
def replace(t,name,new,typ=ast.FunctionDef):
    n=node(t,name,typ);lines=t.splitlines(keepends=True);return ''.join(lines[:n.lineno-1])+new.rstrip()+'\n'+''.join(lines[n.end_lineno:])
oldphase=ast.literal_eval(node(generic,'PHASE_NAME',ast.Assign).value)
phase=Path(scope['phase']).name
anc=ast.literal_eval(node(generic,'ANCESTOR_PHASES',ast.Assign).value)+(('V199',oldphase),)
tags=tuple(scope['prior_closing_tags'])
assert len(anc)==13 and tuple(x[0]for x in anc)==tuple(scope['ancestor_tags'])
arms=('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL','C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT')
run='frozen_C58_C59_real_predictions_v1'
relative=scope['explicit_saved_prediction_relative_paths']
declared={str(Path(b['path']).relative_to(scope['phase'])):b for b in scope['declared_saved_prediction_bindings']}
constants='\n'.join([f'ARMS={arms!r}',"DOMAINS=('CURRENT223_1784','PUBLIC_HELD45_360')",f'PRED_RUN={run!r}',"AUTHORITY_KEYS=('source_prediction_seal','evaluation_complete','independent_saved_result_review','final_phase_freeze')",f'ROOT_SEAL={scope["source_pred_seal_declared_by_root"]!r}',f'ROOT_BINDINGS={declared!r}',f'ROOT_OWNED=()',f'ROOT_SAVED={tuple(relative)!r}'])
policy='''def declared_prediction_artifacts(e,pred,ev,audit):
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
'''
source='''def source_artifact_authority(e):
    docs=[read_checked(e[k])for k in AUTHORITY_KEYS[:3]];read_checked(e['final_phase_freeze'])
    return declared_prediction_artifacts(e,*docs)
'''
template='''def authority_template():
    return dict(source_prediction_seal=dict(ROOT_SEAL),evaluation_complete=dict(path=str(PHASE/'saved_real_head_evaluation_v1/COMPLETE.json'),sha256=None),independent_saved_result_review=dict(path=str(PHASE/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),sha256=None),final_phase_freeze=dict(path=str(PHASE/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),sha256=None),explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=list(ROOT_SAVED),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None)
'''
toy='''def invented_prediction_metadata():
    e=authority_template()
    for k in AUTHORITY_KEYS[1:]:e[k]['sha256']='a'*64
    e['phase_completion_receipts']=[dict(role=r,binding=e['final_phase_freeze']if r=='final_phase_freeze'else dict(path='/invented/'+('ROOT_PHASE_RESULT.json'if r=='phase_result'else'REVIEW.json'),sha256='b'*64),expected_fields={'completed':True,'PASS':True,**({'scientific_phase_frozen_no_further_phase_writes':True}if r=='final_phase_freeze'else{})})for r in ('phase_result','independent_report_QA','final_phase_freeze')]
    p=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,original_currentU45_publicU108=True,arms={},selected_model=None,adopted=False,fully_solved=False)
    for i,a in enumerate(ARMS):p['arms'][a]=dict(updates=8000,use_sam=False,native_condition=bool(i),outputs={d:dict(ROOT_BINDINGS[PRED_RUN+'/'+a+'/FINAL_'+d+'_OUTPUTS.npz'])for d in DOMAINS})
    ev=dict(completed=True,PASS=True,prediction_seal=e['source_prediction_seal'],all_both_predictions_closed_before_any_annotation_or_score_body=True,selected_model=None,adopted=False,fully_solved=False)
    au=dict(completed=True,PASS=True,complete=e['evaluation_complete'],selected_model=None,adopted=False,fully_solved=False)
    return e,p,ev,au
'''
# V195 real-packet policy surface; V199 immutable parent/closing helpers.
t=base.replace('V195','V200').replace('v195','v200').replace('V195_DELTA_PREPARATION_A','V200_DELTA_PREPARATION_C').replace('query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v200',phase)
t=t.replace('V200_DELTA_PREPARATION_A','V200_DELTA_PREPARATION_C')
for name in ['ANCESTOR_PHASES','CLOSING_TAGS']:t=replace(t,name,f'{name}={anc!r}'if name=='ANCESTOR_PHASES'else f'{name}={tags!r}',ast.Assign)
for name in ['closing_protected_entries','validate_parent_closure']:t=replace(t,name,segment(generic,name))
for name in ['FIT_PHASE','FIT_RUN','ARMS','DOMAINS','PRED_RUN','AUTHORITY_KEYS']:
    t=replace(t,name,'# replaced by root-declared V200 metadata constants',ast.Assign)
t=t.replace('def require(v,msg):',constants+'\n\ndef require(v,msg):',1)
for name,new in [('declared_prediction_artifacts',policy),('source_artifact_authority',source),('authority_template',template),('invented_prediction_metadata',toy)]:t=replace(t,name,new)
prepare='''def prepare():
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
'''
t=replace(t,'prepare',prepare);write(H/'prepare_completed_phase_delta.py',t)
# Preserve the V199 copy/hash/gzip/finalize engine. Only phase/chain constants,
# explicit numeric authority call and prospective truthful publication prose change.
b=builder.replace('V199','V200').replace('v199','v200').replace(oldphase,phase).replace('V200_DELTA_PREPARATION_B','V200_DELTA_PREPARATION_C')
b=b.replace("PARENT_DELTA='publication/deltas/V198_20261024_v1/DELTA_MANIFEST.json'","PARENT_DELTA='publication/deltas/V199_20261024_v1/DELTA_MANIFEST.json'").replace("PARENT_CHAIN='MANIFEST_CHAIN_AFTER_V198.json'","PARENT_CHAIN='MANIFEST_CHAIN_AFTER_V199.json'").replace("PARENT_INDEX='EXPERIMENT_INDEX_THROUGH_V198_KO.md'","PARENT_INDEX='EXPERIMENT_INDEX_THROUGH_V199_KO.md'")
b=b.replace('len(requested)==12','len(requested)==13').replace('All12 V187..V198','All13 V187..V199')
b=b.replace('All2owned+24source outputs and 0directtarget selected','All4 saved real predictions;0owned0targets selected')
b=b.replace("'V200 matched native input source fit and weak heldout limits'","'V200 frozen native heads real comparison, no improvement/adoption'")
tree=ast.parse(b)
replacements=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Constant)and isinstance(n.value,str):
        value=n.value
        if value.startswith('Completed V200 source-only matched fresh RAW'):new=scope['research_prose']+' Exact4 new saved prediction NPZ opaque bytes;0ownedweights0targets; sourcefit belongs to V199.'
        elif value.startswith('# V200 source-only native 조건부'):new='# V200 동결 native 조건 head 실사 비교\n\n[최신 실험 인덱스]('
        elif value.startswith(') · [V200 원문·방법·출력·한계]('):new=') · [V200 결과·코드·같은68 gallery]('
        elif value.startswith(')\n\n새 RAW 두팔은'):new=')\n\n같은 원query 실사1237(D1147/E90)에서 새 native C58/C59는 이전 C57보다 D/E 정답 수가 모두 적었습니다. 두팔 모두 고정 평가했고 새 winner·threshold·학습자료 선택, 채택·실패원인 확정·freshblind 일반화·전체 목표 해결을 주장하지 않습니다. Source 학습은 이전 V199의 합성+허용공개실사이며 이번V200은 평가만 수행했습니다.4새 저장 예측은 바이트만 보존하고 weights·원input/SAM/feature/target cache는 제외합니다.\n\n---\n\n'
        else:continue
        replacements.append((ast.get_source_segment(b,n),repr(new)))
for oldliteral,newliteral in replacements:
    assert oldliteral in b
    b=b.replace(oldliteral,newliteral,1)
builderprepare=segment(builder,'prepare').replace('V199','V200').replace('v199','v200').replace('V200_DELTA_PREPARATION_B','V200_DELTA_PREPARATION_C').replace('ROOT_V200_PUBLICATION_CODE_PREPARATION_SCOPE_B','ROOT_V200_PUBLICATION_CODE_PREPARATION_SCOPE_C')
start=builderprepare.index("    protocol=");end=builderprepare.index("    pure.write(BUILDER_PROTOCOL,protocol)")
builderprepare=builderprepare[:start]+"    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),pure_helper=bind(H/'prepare_completed_phase_delta.py'),root_code_preparation_scope=bind(H.parent/'ROOT_V200_PUBLICATION_CODE_PREPARATION_SCOPE_C.json'),ancestor_tags=[t for t,_ in pure.ANCESTOR_PHASES],immutable_closing_tags=list(pure.CLOSING_TAGS),source_scope=V.name,own_head_whitelist=[],saved_array_whitelist=list(pure.ROOT_SAVED),direct_target_whitelist=[],gzip_text_threshold_bytes=2*1024**2,portable_pages="+repr(scope['portable_pages'])+",immutable_policy='BASE+13ancestor deltas/chains/index and14closing plans immutable; BASE historical rootREADME exception only; current README exact prior bytes suffix; physical descriptors and chunks protected.',research_prose="+repr(scope['research_prose'])+",actual_before_root_entry=0,expected_remote_main=None,automatic_retry=False)\n"+builderprepare[end:]
start=builderprepare.index("'source_fit_complete':");end=builderprepare.index("'admin_output':",start)
builderprepare=builderprepare[:start]+"**pure.authority_template(),"+builderprepare[end:]
b=replace(b,'prepare',builderprepare);write(H/'build_completed_phase_delta.py',b)
# Full original old tests remain metadata evidence; replay pure parent/closing
# fixtures in this phase without reading copier dependencies or source packets.
fixture=tool('generic_publisher_ready','run_invented_delta_roundtrip.py')
fc=fixture[:fixture.index('\ndef run(')]
fc=fc.replace('V199','V200').replace('v199','v200').replace('V198','V199').replace('thirteen','fourteen')
fc+='\nif __name__==\'__main__\':\n    x=pure_checks()\n    out=dict(completed=True,PASS=all(x.values()),checks=x,count=len(x),invented_only=True,actual_phase_repo_hash_copy_Git=0)\n    with(H/\'INVENTED_PARENT_AND_CLOSING_ROUNDTRIP.json\').open(\'x\')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write(\'\\n\')\n    print(json.dumps(out))\n'
write(H/'run_invented_parent_closure.py',fc)
write(H/'INHERITED_SOURCE_BINDINGS.json',dict(scope=bind(S),parents=scope,code_bindings={k:{kk:vv for kk,vv in r.items()if kk in ['inventory_code','builder_code','code','protocol','checks']}for k,r in ready.items()},actual_scientific_or_parent_repo_reads=0))
write(H/'PORTABLE_ENTRY_POINTS_BEFORE_PHASE_READS.json',dict(portable_pages=scope['portable_pages'],source='Root prospective declarations only;no actual phase reads'))
write(H/'DELTA_OPERATOR_WORKFLOW_KO.md','# V200 평가 완료 phase 추가 게시\n\n별도 root 실제 scope/동결 완료 후 inventory→build→독립 byte QA→finalize 순서만 실행한다. 각 exit0 다음 단계, 실패 보존 후 중지한다. Git push는 root 담당이다. BASE+V187..V199 13delta 및14closing 보호, V200 정확4prediction만 opaque 보존한다. ownedhead/sourcefit/targets0, rawRGB/SAM/feature/cache는 제외 사유 기록한다.2MiB text gzip은 원SHA/physicaldescriptor/chunk를 보존한다.6 portable entrypages, 새 THROUGH_V200/AFTER_V200/delta만 추가한다. README는 원문 suffix를 보존한다. 새 두팔 결과는 개선·채택·freshblind·원인·goal완료를 입증하지 않는다.\n')
# Auditor V195 byte/credential/link kernels with V199 cumulative closure helpers.
a=auditor.replace('V195','V200').replace('v195','v200').replace('query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v200',phase).replace('V200_DELTA_PREPARATION_A','V200_DELTA_PREPARATION_C').replace('V200_DELTA_INDEPENDENT_REVIEW_A','V200_DELTA_INDEPENDENT_REVIEW_C')
a=a.replace("PARENT_DELTA='publication/deltas/V194_20261024_v1/DELTA_MANIFEST.json'","PARENT_DELTA='publication/deltas/V199_20261024_v1/DELTA_MANIFEST.json'").replace("PARENT_CHAIN='MANIFEST_CHAIN_AFTER_V194.json'","PARENT_CHAIN='MANIFEST_CHAIN_AFTER_V199.json'").replace("PARENT_INDEX='EXPERIMENT_INDEX_THROUGH_V194_KO.md'","PARENT_INDEX='EXPERIMENT_INDEX_THROUGH_V199_KO.md'")
for name in ['ANCESTORS','PORTABLE']:a=replace(a,name,name+'='+repr(anc if name=='ANCESTORS'else scope['portable_pages']),ast.Assign)
for name in ['cumulative_parent_entries','closing_entries','same_exceptions']:
    a=replace(a,name,segment(gauditor,name).replace('V199_DELTA_PREPARATION_B','V200_DELTA_PREPARATION_C'))
cs=segment(a,'cumulative_parent_entries').replace('==12','==13').replace('V198','V199');a=replace(a,'cumulative_parent_entries',cs)
cs=segment(a,'closing_entries').replace("['BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198']",repr(list(tags)));a=replace(a,'closing_entries',cs)
for name in ['FITV','FITRUN','RUN','ARMS','DOMAINS','AUTHORITY_KEYS']:a=replace(a,name,'# V200 root-declared output authority below',ast.Assign)
a=a.replace('def require(v,s):',constants.replace('PRED_RUN=','RUN=')+'\n\ndef require(v,s):',1)
indpolicy=policy.replace('def declared_prediction_artifacts(e,pred,ev,audit):','def numeric_metadata_policy(e,pred,ev,audit):').replace('PHASE/', 'V/').replace('PRED_RUN','RUN')
# Separate implementation: infer exact saved sets from the immutable root map,
# use strictly typed arm declarations and metadata checks; no publisher imports.
indpolicy=indpolicy.replace('require_completion_barriers(receipts)',"require({'phase_result','independent_report_QA','final_phase_freeze'}<=set(r.get('role')for r in receipts),'All completion barrier roles')")
a=replace(a,'numeric_metadata_policy',indpolicy)
a=replace(a,'read_numeric_authority',"""def read_numeric_authority(e):
    docs=[read(checked(e[k]))for k in AUTHORITY_KEYS[:3]];read(checked(e['final_phase_freeze']))
    req=e.get('phase_completion_receipts',e.get('completion_reviews',[]))
    for item in req:
        o=read(checked(item['binding']));require(item['expected_fields']and all(o[k]==x for k,x in item['expected_fields'].items()),'Exact completion fields')
    return numeric_metadata_policy(e,*docs)
""")
a=a.replace("len(e['ancestor_publications'])==8","len(e['ancestor_publications'])==13").replace("['BASE','V187','V188','V189','V190','V191','V192','V193','V194']",repr(list(tags))).replace("['BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194']",repr(list(tags)))
a=a.replace("fixed_direct_end_real_visual_v1",'fixed_native_real_visual_v1')
runtext=segment(a,'run').replace('old_BASE_through_V194','old_BASE_through_V199').replace('Nine explicit','Fourteen explicit').replace('Exactly eight immutable','Exactly thirteen immutable').replace('nine-ancestor','thirteen-ancestor')
a=replace(a,'run',runtext)
inv=segment(a,'invented');inv=inv[:inv.index('    import copy\n    e=')]
inv=inv.replace("if tag in ('V189','V190','V191','V192','V193','V194'):","if tag not in ('V187','V188'):").replace('len(records)==8 and len(cumulative)>8','len(records)==13 and len(cumulative)>13').replace('full_eight_ancestor','full_thirteen_ancestor').replace('len(old_closing)==9','len(old_closing)==14').replace('nine_closing','fourteen_closing').replace("str(i+1)*64","hashlib.sha256(str(i).encode()).hexdigest()")
inv+='''    import copy
    e=dict(source_prediction_seal=dict(ROOT_SEAL),evaluation_complete=dict(path=str(V/'saved_real_head_evaluation_v1/COMPLETE.json'),sha256='d'*64),independent_saved_result_review=dict(path=str(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),sha256='e'*64),final_phase_freeze=dict(path=str(V/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),sha256='f'*64),explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=list(ROOT_SAVED),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None)
    e['completion_reviews']=[dict(role=x,binding=e['final_phase_freeze']if x=='final_phase_freeze'else dict(path='/invented/'+x+'.json',sha256='b'*64),expected_fields={'completed':True,'PASS':True})for x in ('phase_result','independent_report_QA','final_phase_freeze')]
    pred=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,original_currentU45_publicU108=True,arms={arm:dict(updates=8000,use_sam=False,native_condition=bool(i),outputs={d:dict(ROOT_BINDINGS[RUN+'/'+arm+'/FINAL_'+d+'_OUTPUTS.npz'])for d in DOMAINS})for i,arm in enumerate(ARMS)},selected_model=None,adopted=False,fully_solved=False)
    ev=dict(completed=True,PASS=True,prediction_seal=e['source_prediction_seal'],all_both_predictions_closed_before_any_annotation_or_score_body=True,selected_model=None,adopted=False,fully_solved=False)
    au=dict(completed=True,PASS=True,complete=e['evaluation_complete'],selected_model=None,adopted=False,fully_solved=False)
    authority=numeric_metadata_policy(e,pred,ev,au);add('exact4RAW_native_flags_rootSHA_authority',len(authority['declared_artifact_bindings'])==4)
    selected=[dict(source_root_id='P',source_relative_path=PHASE+'/'+r,source_absolute=b['path'],source_sha256=b['sha256'],published_sha256=b['sha256'],published_relative=ARCHIVE+r,status='PUBLISHED_RAW',gzipchunks=None,size_bytes=1)for r,b in authority['declared_artifact_bindings'].items()]
    add('opaque4packetpolicy',source_archive_policy(selected,(),set(ROOT_SAVED),(),authority['declared_artifact_bindings'])==0)
    changes=[('wrongSHA',lambda e,p,v,a:p['arms'][ARMS[0]]['outputs'][DOMAINS[0]].update(sha256='9'*64)),('nativeint',lambda e,p,v,a:p['arms'][ARMS[1]].update(native_condition=1)),('wrongnative',lambda e,p,v,a:p['arms'][ARMS[1]].update(native_condition=False)),('SAMalias',lambda e,p,v,a:p['arms'][ARMS[1]].update(use_sam=True)),('partial4',lambda e,p,v,a:p['arms'][ARMS[0]]['outputs'].pop(DOMAINS[1])),('selected',lambda e,p,v,a:p.update(selected_model=ARMS[0])),('early',lambda e,p,v,a:p.update(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False)),('badrepeat',lambda e,p,v,a:p.update(all_batch_repeat2_state_and_input_bytes_exact=False)),('badseal',lambda e,p,v,a:e['source_prediction_seal'].update(sha256='9'*64)),('bad_eval',lambda e,p,v,a:v.update(prediction_seal={'path':'bad','sha256':'9'*64})),('bad_audit',lambda e,p,v,a:a.update(complete={'path':'bad','sha256':'9'*64})),('ownhead',lambda e,p,v,a:e.update(explicit_own_head_relative_paths=['head.pt'])),('target',lambda e,p,v,a:e.update(explicit_direct_target_relative_paths=['target.npz']))]
    for name,change in changes:
        args=copy.deepcopy((e,pred,ev,au));change(*args)
        try:numeric_metadata_policy(*args);ok=False
        except (AssertionError,KeyError):ok=True
        add(name+'_refused',ok)
    for name,rows in [('wrongpacketSHA',[dict(selected[0],source_sha256='9'*64)]+selected[1:]),('missingpacket',selected[:-1]),('excludedpacket',[dict(selected[0],status='EXCLUDED',exclusion_reason='bad')]+selected[1:])]:
        try:source_archive_policy(rows,(),set(ROOT_SAVED),(),authority['declared_artifact_bindings']);ok=False
        except AssertionError:ok=True
        add(name+'_refused',ok)
    text='<pre>[notlink](missing.txt)</pre><a href="actual.md">actual</a>'
    add('HTMLattributes_only',portable_audit('archive/p/x.html',text,lambda p:p=='archive/p/actual.md')['local_links']==['archive/p/actual.md'])
    return dict(completed=True,PASS=all(x['PASS']for x in tests),checks=tests,invented_only=True,actual_source_repo_hash_copy_Git=0)
'''
a=replace(a,'invented',inv)
# Final prepare will bind the finished publisher once; old actuals never read.
prep=segment(a,'prepare')
prep=prep.replace("'inherited_auditor_code':bind(P/'publication_root_preparation/V194_DELTA_REVIEW_PREPARATION_C/audit_immutable_v194_delta.py'),","'inherited_auditor_ready':"+repr(scope['prior_real_benchmark_auditor_ready'])+",")
prep=prep.replace("owned_heads=0,saved_array_packets=4,optional_direct_target_packets_max=0","owned_heads=0,saved_array_packets=4,optional_direct_target_packets_max=0")
prep=prep.replace("write(H/'PREPARATION_COMPLETE.json',ready)","ready.update(PASS=True,FINAL_READY_FROZEN=True,invented_check_count=len(tests['checks']),publisher_final=bind(C/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json'));write(H/'FINAL_READY.json',ready)")
prep=prep.replace("write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)","template.update(source_prediction_seal=dict(ROOT_SEAL),evaluation_complete=dict(path=str(V/'saved_real_head_evaluation_v1/COMPLETE.json'),sha256=None),independent_saved_result_review=dict(path=str(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),sha256=None),final_phase_freeze=dict(path=str(V/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),sha256=None),expected_remote_main=None);write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)")
a=replace(a,'prepare',prep)
write(Q/'audit_immutable_v200_delta.py',a)
write(Q/'RULES_BEFORE_ACTUAL_PUBLISHED_BYTES.json',dict(scope=bind(S),owned_heads=0,saved_arrays=4,targets=0,ancestor13=True,closing14=True,body_interpretation='Only opaque exact4 saved output files; no sourcefit/oldweights reopened; no scientific/image/array/model decode',actual_reads=0))
write(Q/'INHERITED_CODE_BINDINGS.json',dict(scope=bind(S),generic_auditor=scope['generic_auditor_ready'],real_auditor=scope['prior_real_benchmark_auditor_ready']))
print(json.dumps(dict(completed=True,code_only=True,inventory=bind(H/'prepare_completed_phase_delta.py'),builder=bind(H/'build_completed_phase_delta.py'),auditor=bind(Q/'audit_immutable_v200_delta.py'))))
