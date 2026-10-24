"""V198 code-only publication derivation from exact V197 frozen tools."""
from pathlib import Path
import ast,hashlib,json,os,stat,subprocess,sys
R=Path(__file__).absolute().parent.parent;H=Path(__file__).absolute().parent;Q=R/'V198_DELTA_REVIEW_PREPARATION_B';OLD=R/'V197_DELTA_PREPARATION_C';OQ=R/'V197_DELTA_REVIEW_PREPARATION_C'
SCOPE=R/'ROOT_V198_PUBLICATION_CODE_PREPARATION_SCOPE_B.json'
OLD_PHASE='query_visibility_train_only_target_contract_readout_20261024_v197';NEW_PHASE='query_visibility_native_query_conditioning_feasibility_20261024_v198'
def read(p):
 assert 'v112'not in str(p).casefold()and'..'not in p.parts
 cur=Path(p.anchor)
 for x in p.parts[1:]:
  cur/=x;assert not stat.S_ISLNK(os.lstat(cur).st_mode)
 return p.read_bytes()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(read(p)).hexdigest(),size_bytes=len(read(p)))
def put(p,value):
 b=value.encode()if isinstance(value,str)else(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
 with p.open('xb')as f:f.write(b)
 return bind(p)
def swap(s):return s.replace(OLD_PHASE,'__CURRENT__').replace('V197','V198').replace('v197','v198').replace('__CURRENT__',NEW_PHASE).replace('V198_DELTA_PREPARATION_C','V198_DELTA_PREPARATION_B').replace('V198_DELTA_REVIEW_PREPARATION_C','V198_DELTA_REVIEW_PREPARATION_B').replace('V198_DELTA_INDEPENDENT_REVIEW_C','V198_DELTA_INDEPENDENT_REVIEW_B').replace('ROOT_V198_PUBLICATION_CODE_PREPARATION_SCOPE_C','ROOT_V198_PUBLICATION_CODE_PREPARATION_SCOPE_B')
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def assign(s,name,value):
 lines=s.splitlines();ii=[i for i,l in enumerate(lines)if l.startswith(name+'=')];assert len(ii)==1,(name,ii);lines[ii[0]]=name+'='+repr(value);return '\n'.join(lines)+'\n'
root=json.loads(read(SCOPE));assert bind(SCOPE)['sha256']=='394ff3b8f989f6dd6d1ac00fb0c10964538a7a03a3e0f70fbcbff69340ade677'
for b in root['inherited_preparations']:assert bind(Path(b['path']))['sha256']==b['sha256']
oldhelper=read(OLD/'prepare_completed_phase_delta.py').decode();oldbuilder=read(OLD/'build_completed_phase_delta.py').decode();oldfixture=read(OLD/'run_invented_delta_roundtrip.py').decode()
cons={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse(oldhelper).body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('ANCESTOR_PHASES','CLOSING_TAGS')}
anc=cons['ANCESTOR_PHASES']+(('V197',OLD_PHASE),);closing=cons['CLOSING_TAGS']+('V197',)
assert len(anc)==11 and len(closing)==12
helper=assign(assign(swap(oldhelper),'ANCESTOR_PHASES',anc),'CLOSING_TAGS',closing).replace('BASE through V196','BASE through V197')
helper=helper.replace('BASE and all9 V187..V195 delta/chain/index plus all10closing paths','BASE and all11 V187..V197 delta/chain/index plus all12closing paths')
put(H/'prepare_completed_phase_delta.py',helper)
builder=swap(oldbuilder)
for name,value in [('PARENT_DELTA','publication/deltas/V197_20261024_v1/DELTA_MANIFEST.json'),('PARENT_CHAIN','MANIFEST_CHAIN_AFTER_V197.json'),('PARENT_INDEX','EXPERIMENT_INDEX_THROUGH_V197_KO.md')]:builder=assign(builder,name,value)
builder=once(builder,"require(len(requested)==10,'All10 V187..V196 ancestor metadata mandatory')","require(len(requested)==11,'All11 V187..V197 ancestor metadata mandatory')")
builder=builder.replace('Latest parent aliases exact V196 bindings','Latest parent aliases exact V197 bindings').replace('BASE+10ancestor deltas/index/chains and11closing plans','BASE+11ancestor deltas/index/chains and12closing plans')
lines=builder.splitlines()
claim='Publication of root-frozen V198 code-only nativeID/originalq8 input feasibility and independent hypothesis review. Native ID is stored but not fed to the reviewed prior raw head. SELF lacks a simultaneous jittered initializer8; old active anchor may reveal artificial jitter. No new source pixels/generation, labels, model, fit/inference/scores, physical native gold, improvement, causal effect, data/architecture/threshold selection, adoption or goal-success claim. A separate prospective trial belongs to V199 and is not a V198 result. All numeric arrays and weights excluded.'
for i,line in enumerate(lines):
 if 'index=links+pure.versioned_index_bytes' in line:lines[i]="    index=links+pure.versioned_index_bytes(old_index,'V198 nativeID / originalq8 code feasibility',portable)"
 if 'manifest={'in line and "'scope':'Publication of root-frozen"in line:
  a=line.index("'scope':")+len("'scope':");b=line.index(",'model_adopted'",a);lines[i]=line[:a]+repr(claim)+line[b:]
 if line.lstrip().startswith('new_readme='):
  lines[i]="    new_readme=('# V198 nativeID·originalq8 입력 계약 코드 검토\\n\\n[최신 실험 인덱스]('+REPO_PREFIX+NEW_INDEX+') · [V198 원문·독립 검토·한계]('+REPO_PREFIX+portable+')\\n\\n이 추가 기록은 nativeID 저장·forward 여부와 원q8 문맥의 증강량 누출 위험, native prior 및 비선형 대조의 의미를 코드만으로 검토합니다. 새 모델·학습·추론·점수·라벨·실사 개선·물리 native gold·인과 효과·선택·채택·전체 목표 해결을 주장하지 않습니다. 별도 V199 진행은 이 V198의 성능 결과가 아닙니다. 모든 배열·가중치를 제외하고 원문·실패·가상 검사를 보존합니다.\\n\\n---\\n\\n').encode()+old_readme"
 if "'portable':'Exactly root-fixed"in line:
  a=line.index("'portable':")+len("'portable':");b=line.index(",'historical_README_exception'",a);desc='Exactly6 root-fixed pages: PORTABLE_REPORT,ROOT_REPORT,ROOT_READOUT,source_contract_review_a/KO.md,independent_hypothesis_review_b/KO.md,THROUGH_V198 index. ROOT_RESULTS JSON direct target;2MiB text threshold preserves small links; no logical gzip fallback.';lines[i]=line[:a]+repr(desc)+line[b:]
 if "'message':'Publish V198"in line:lines[i]="    plan['batches']=pure.bounded_batches(files)" if False else line.replace('TRAIN-only saved target correspondence and preserved limits','code-only native input feasibility and preserved limits')
builder='\n'.join(lines)+'\n';put(H/'build_completed_phase_delta.py',builder)
fixture=swap(oldfixture).replace('BASE+V187+V188+V189+V190+V191+V192 -> V198','BASE+V187..V197 -> V198').replace('through_V196','through_V197').replace('latest_V196','latest_V197').replace('all_eleven_closing','all_twelve_closing').replace('newindex_exact_V196_suffix','newindex_exact_V197_suffix')
fixture=once(fixture,"chain['deltas'][:10]==refs and len(refs)==10 and len(chain['deltas'])==11","chain['deltas'][:11]==refs and len(refs)==11 and len(chain['deltas'])==12")
fixture=fixture.replace("'latest_index_SHA_drift':lambda r:r[-1]['chain_data']['index'].update(sha256='f'*64)","'latest_index_SHA_drift':lambda r:r[-1]['chain_data']['index'].update(sha256='0'*64)")
put(H/'run_invented_delta_roundtrip.py',fixture)
# Storage/security/runtime kernels remain exact apart from strings and closure-size guards.
class Strip(ast.NodeTransformer):
 def visit_Constant(self,n):return ast.copy_location(ast.Constant('<STRING>'),n)if isinstance(n.value,str)else n
def dump(s,name):
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name);return ast.dump(Strip().visit(n),include_attributes=False)
parity={}
for key,old,new,names in [('helper',oldhelper,helper,['require','safe_parts','sha_bytes','open_nofollow','sha_file','read_checked','write','physical_paths','assert_new_paths','closing_protected_entries','validate_parent_closure','versioned_index_bytes','bounded_batches','require_completion_barriers','classify_file','inventory']),('builder',oldbuilder,builder,['require','opened','read_bytes','sha','bind','checked','checked_json','write_new','json_bytes','raw_record','check_reviews','recheck_parent_bindings','build','finalize'])]:
 for name in names:parity[key+'.'+name]=dump(old,name)==dump(new,name)
assert all(parity.values())
put(H/'LIMITED_DERIVATION_AND_GENERIC_AST_PARITY.json',dict(completed=True,PASS=True,generic_functions_stripped_strings_exact=parity,changes=['V198/code-only phase identity and claims','11ancestors V187..V197/12closing BASE..V197','LatestparentV197','0heads/arrays/targets, unchanged2MiB','All stale9/10 prose corrected11/12'],actual_phase_repo_parent_API_Git_model_access=0))
put(H/'INHERITED_PUBLISHER_SOURCE_BINDINGS.json',dict(completed=True,source_final=bind(OLD/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json'),source_code=[bind(OLD/n)for n in ['prepare_completed_phase_delta.py','build_completed_phase_delta.py','run_invented_delta_roundtrip.py']],root_scope=bind(SCOPE),actual_phase_reads=0))
put(H/'DELTA_OPERATOR_WORKFLOW_KO.md','# V198 코드 검토 기록 게시 절차\n\n실제 phase 열거·해시·복사·repo·부모 manifest·API·Git 접근 없이 준비한다. Root가 phase_result·독립 report QA·finalfreeze를 정확 SHA와 필드로 동결한 별도 actual entry를 제공해야 한다.\n\nBASE+V187..V197의11개 prior delta/chain/index와12개 closing plan을 순서대로 보존한다. 과거 logical/raw/chunk/descriptor/cycle 경로는 immutable이며 BASE 역사 README 예외 하나 외에 덮어쓰지 않는다. 현재 README는 원문 SHA와 suffix로 보존한다.\n\n정확 V198 문서·코드·표·PNG/HTML만 포함하고 모든 배열·가중치·rawRGB/SAM/features/cache는 제외 이유를 남긴다. ownhead/savedarray/directtarget whitelist는0이다. 2MiB 초과 text만 lossless gzip chunk+descriptor로 저장한다. Root 고정6페이지의 literal 상대링크를 검산한다.\n\n새 THROUGH_V198/AFTER_V198와 delta만 만들고 독립 byte/CRC/SHA/link/credential PASS 후 strict480MiB pushplan을 봉인한다. 실제 Git/push는 root가 별도로 수행한다. V198는 native input의 code-only 검토이며 새 fit/score/개선/선택/물리 gold/채택/목표완료를 주장하지 않는다.\n')
for name in ['prepare_completed_phase_delta.py','build_completed_phase_delta.py']:
 subprocess.run([sys.executable,'-B',str(H/name),'--prepare'],check=True)
subprocess.run([sys.executable,'-B',str(H/'run_invented_delta_roundtrip.py')],check=True)
put(H/'PORTABLE_ENTRY_POINTS_BEFORE_PHASE_READS.json',dict(completed=True,prepared_only=True,phase='V198',portable_pages=root['portable_pages'],actual_phase_reads=0))
put(H/'CUMULATIVE_PARENT_CLOSURE_PROTOCOL.json',dict(completed=True,prepared_only=True,ancestor_tags=[t for t,_ in anc],immutable_closing_tags=list(closing),ancestor_delta_count=11,prior_closing_plan_count=12,policy='Exact order/SHA/fullchain/previousindex and immutable logical/raw/chunk/descriptor/closing paths; BASE historical README sole exception; expectedactualmain null until rootentry.'))
names=['derive_v198_code_only.py','prepare_completed_phase_delta.py','build_completed_phase_delta.py','run_invented_delta_roundtrip.py','LIMITED_DERIVATION_AND_GENERIC_AST_PARITY.json','INHERITED_PUBLISHER_SOURCE_BINDINGS.json','DELTA_OPERATOR_WORKFLOW_KO.md','GENERATED_INVENTED_DELTA_CHECKS.json','GENERATED_BUILDER_CHECKS.json','INVENTED_COPY_BUILD_PLAN_ROUNDTRIP.json','PROTOCOL_BEFORE_V198_PUBLICATION_ENUMERATION_OR_COPY.json','BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json','ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json','ROOT_DELTA_BUILD_ENTRY_TEMPLATE_APPROVED_FALSE.json','ROOT_DELTA_PUSH_PLAN_ENTRY_TEMPLATE_APPROVED_FALSE.json','PORTABLE_ENTRY_POINTS_BEFORE_PHASE_READS.json','CUMULATIVE_PARENT_CLOSURE_PROTOCOL.json']
for n in ['PROTOCOL_BEFORE_V198_PUBLICATION_ENUMERATION_OR_COPY.json','BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json']:
 text=read(H/n).decode();assert not any(s in text for s in ['all9 V187..V195','all10closing','10ancestor deltas','11closing plans'])
inv=json.loads(read(H/'GENERATED_INVENTED_DELTA_CHECKS.json'));bcheck=json.loads(read(H/'GENERATED_BUILDER_CHECKS.json'));roundtrip=json.loads(read(H/'INVENTED_COPY_BUILD_PLAN_ROUNDTRIP.json'));assert inv['PASS']and bcheck['PASS']and roundtrip['PASS']
final=dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,phase='V198',files={n:bind(H/n)for n in names},inventory_code=bind(H/'prepare_completed_phase_delta.py'),builder_code=bind(H/'build_completed_phase_delta.py'),checks=dict(inventory=len(inv['checks']),builder=len(bcheck['checks']),fullcopy_gzip_pushplan=roundtrip['count']),owned_heads=0,saved_numeric_arrays=0,direct_target_arrays=0,ancestor_delta_count=11,prior_closing_plan_count=12,gzip_text_threshold_bytes=2097152,portable_pages=root['portable_pages'],expected_remote_main=None,root_declared_future_main_not_remotely_queried=root['expected_remote_main_for_future_entry'],future_actual_requires=['phase_result','independent_report_QA','final_phase_freeze','separate_actual_root_scope'],actual_phase_inventory_source_hash_copy_parent_repo_API_Git_or_model_calls=0,root_code_preparation_scope=bind(SCOPE),no_edits_after_final_ready_without_root_request=True)
put(H/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json',final)
# Independent scalar/stream auditor copied separately, not importing publisher kernels.
old_audit=read(OQ/'audit_immutable_v197_delta.py').decode();audit=swap(old_audit)
audit=assign(audit,'ANCESTORS',anc);audit=assign(audit,'PORTABLE',root['portable_pages'])
for name,value in [('C_CODE',bind(H/'build_completed_phase_delta.py')),('C_PROTO',bind(H/'BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json')),('C_FINAL',bind(H/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json')),('ROOT_PREPARATION_SCOPE',bind(SCOPE)),('PARENT_DELTA','publication/deltas/V197_20261024_v1/DELTA_MANIFEST.json'),('PARENT_CHAIN','MANIFEST_CHAIN_AFTER_V197.json'),('PARENT_INDEX','EXPERIMENT_INDEX_THROUGH_V197_KO.md')]:
 if isinstance(value,dict):value={k:v for k,v in value.items()if k in ('path','sha256')}
 audit=assign(audit,name,value)
# Exact finite count/list changes, no probability/byte/link/credential kernel changes.
audit=audit.replace("len(records)==10,'All10 original V187..V196 ancestors required'","len(records)==11,'All11 original V187..V197 ancestors required'")
audit=audit.replace("'V195','V196')","'V195','V196','V197')")
audit=audit.replace("'V195', 'V196']","'V195', 'V196', 'V197']").replace("'V195', 'V196')","'V195', 'V196', 'V197')")
audit=audit.replace('eleven exact ordered closing','twelve exact ordered closing').replace('eleven closing metadata','twelve closing metadata').replace('Eleven explicit closing','Twelve explicit closing')
audit=audit.replace('BASE + 10-ancestor','BASE + 11-ancestor').replace('full_ten_ancestor_chain','full_eleven_ancestor_chain').replace('len(records)==10 and len(cumulative)>10','len(records)==11 and len(cumulative)>11')
audit=audit.replace('eleven_closing_plans','twelve_closing_plans').replace('len(old_closing)==11','len(old_closing)==12')
audit=audit.replace('len(PORTABLE) == 7','len(PORTABLE) == 6').replace('phase and seven portable','phase and six portable').replace('Seven fixed portable','Six fixed portable')
audit=audit.replace("final['ancestor_delta_count'] == 10 and final['prior_closing_plan_count'] == 11","final['ancestor_delta_count'] == 11 and final['prior_closing_plan_count'] == 12")
audit=audit.replace('BASE+10ancestor/11closing','BASE+11ancestor/12closing').replace("len(e['ancestor_publications'])==10,'Exactly ten immutable ancestor metadata sets'","len(e['ancestor_publications'])==11,'Exactly eleven immutable ancestor metadata sets'")
audit=audit.replace('Prior V196 index byte-exact','Prior V197 index byte-exact').replace('old_BASE_through_V196','old_BASE_through_V197')
put(Q/'audit_immutable_v198_delta.py',audit)
# Compare only independent stream/security/numerical kernels which must not drift.
qs=['require','safe_name','relative','opened','bytes_of','digest','checked','write','file_chunks','gunzip_parts','physical','assert_disjoint','source_archive_policy','resolve_target','portable_audit','credential_scan']
oldnames={n.name for n in ast.parse(old_audit).body if isinstance(n,ast.FunctionDef)};newnames={n.name for n in ast.parse(audit).body if isinstance(n,ast.FunctionDef)}
qp={n:dump(old_audit,n)==dump(audit,n)for n in qs if n in oldnames&newnames};assert all(qp.values())
put(Q/'LIMITED_DERIVATION_AND_GENERIC_AST_PARITY.json',dict(completed=True,PASS=True,kernels=qp,structural_changes=['V198exactphase/newBdirs','11ancestors/12closing andpreviousindexV197','6portablepages','newpublisherexactbindings','sourcecodefeasibilityprose'],actual_reads=0))
put(Q/'INHERITED_AUDITOR_SOURCE_BINDINGS.json',dict(completed=True,source_final=bind(OQ/'FINAL_READY.json'),source_code=bind(OQ/'audit_immutable_v197_delta.py'),source_protocol=bind(OQ/'PROTOCOL_BEFORE_ACTUAL_V197_DELTA_BYTES.json'),root_scope=bind(SCOPE)))
put(Q/'CODE_ONLY_PREPARATION_SCOPE.json',dict(completed=True,prepared_code_only=True,source_inventory_expected=dict(total='Future root-frozen count; no phase inventory during preparation',owned_heads=0,saved_array_packets=0,direct_target_arrays=0),source_facts_declared_by_parent=dict(code_only_nativeID_originalq8_feasibility=True,native_ID_available=True,SELF_has_no_simultaneous_jittered8=True,new_models=0,new_fit=0,new_scores=0,new_source_generation=0,physicalnativegold=False,performance_improvement=False,adopted=False,goal_solved=False),root_scope=bind(SCOPE),actual_source_phase_repo_parent_API_Git_access=0))
subprocess.run([sys.executable,'-B',str(Q/'audit_immutable_v198_delta.py'),'--prepare'],check=True)
print(json.dumps(dict(completed=True,PASS=True,publisher=bind(H/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json'),auditor=bind(Q/'FINAL_READY.json'),actualsource_repo_API_Git=0)))
