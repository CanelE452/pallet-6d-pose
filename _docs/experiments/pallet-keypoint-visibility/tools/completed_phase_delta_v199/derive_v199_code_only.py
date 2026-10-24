"""Code-only V199 source-fit publication adaptation. No actual phase/repo reads."""
from pathlib import Path
import ast,hashlib,json,os,stat,subprocess,sys
R=Path(__file__).absolute().parent.parent;H=Path(__file__).absolute().parent;Q=R/'V199_DELTA_REVIEW_PREPARATION_B';OLD=R/'V198_DELTA_PREPARATION_B';OQ=R/'V198_DELTA_REVIEW_PREPARATION_B';N=R/'V194_DELTA_PREPARATION_C';SCOPE=R/'ROOT_V199_PUBLICATION_CODE_PREPARATION_SCOPE_B.json'
OP='query_visibility_native_query_conditioning_feasibility_20261024_v198';NP='query_visibility_native_conditioned_grouped_source_training_20261024_v199'
def read(p):
 assert 'v112'not in str(p).lower()and'..'not in p.parts;cur=Path(p.anchor)
 for x in p.parts[1:]:cur/=x;assert not stat.S_ISLNK(os.lstat(cur).st_mode)
 return p.read_bytes()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(read(p)).hexdigest(),size_bytes=len(read(p)))
def put(p,o):
 b=o.encode()if isinstance(o,str)else(json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode()
 with p.open('xb')as f:f.write(b)
 return bind(p)
def swap(s):return s.replace(OP,'__CURRENT__').replace('V198','V199').replace('v198','v199').replace('__CURRENT__',NP)
def assign(s,n,v):
 ls=s.splitlines();ix=[i for i,l in enumerate(ls)if l.startswith(n+'=')];assert len(ix)==1,n;ls[ix[0]]=n+'='+repr(v);return '\n'.join(ls)+'\n'
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def fn(s,n):return ast.get_source_segment(s,next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)and x.name==n))
def replacefn(s,n,z):return s.replace(fn(s,n),z)
root=json.loads(read(SCOPE));assert bind(SCOPE)['sha256']=='dc81d797df41576fe52f9f2922adb2b985f0c5ae7d8d7d11f867a3ce6d1725bf'
for b in root['inherited_preparations']:assert bind(Path(b['path']))['sha256']==b['sha256']
heads=root['explicit_owned_head_relative_paths'];saved=root['explicit_saved_output_relative_paths'];assert len(heads)==2 and len(saved)==24 and not root['explicit_direct_target_paths']
run=heads[0].split('/'+heads[0].split('/')[2]+'/')[0];arms=tuple(x.split('/')[2]for x in heads);domains=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')
assert run=='runs/C58_C59_NATIVE_CONDITIONED_GROUPED_SOURCE_FIXED8000'
old=read(OLD/'prepare_completed_phase_delta.py').decode();oldb=read(OLD/'build_completed_phase_delta.py').decode();num=read(N/'prepare_completed_phase_delta.py').decode();numb=read(N/'build_completed_phase_delta.py').decode();auditold=read(OQ/'audit_immutable_v198_delta.py').decode()
cs={x.targets[0].id:ast.literal_eval(x.value)for x in ast.parse(old).body if isinstance(x,ast.Assign)and len(x.targets)==1 and isinstance(x.targets[0],ast.Name)and x.targets[0].id in('ANCESTOR_PHASES','CLOSING_TAGS')};anc=cs['ANCESTOR_PHASES']+(('V198',OP),);clos=cs['CLOSING_TAGS']+('V198',);assert len(anc)==12 and len(clos)==13
helper=assign(assign(swap(old),'ANCESTOR_PHASES',anc),'CLOSING_TAGS',clos).replace('BASE through V197','BASE through V198').replace('all11 V187..V197','all12 V187..V198').replace('all12closing','all13closing')
constants='\nFIT_RUN='+repr(run)+'\nFIT_ARMS='+repr(arms)+'\nFIT_DOMAINS='+repr(domains)+'\nROOT_OWNED='+repr(tuple(heads))+'\nROOT_SAVED='+repr(tuple(saved))+'\n'
helper=helper.replace('\ndef require(v,msg):',constants+'\ndef require(v,msg):')
extra='\n\n'.join(fn(num,n)for n in ('declared_fit_artifacts','source_artifact_authority','invented_fit_metadata'))
extra=extra.replace("r['updates']==8000 and r['use_sam']==bool(ai)","r['updates']==8000 and r['use_sam']is False and type(r['native_condition'])is bool and r['native_condition']==bool(ai)")
extra=extra.replace("'Both fixed source arms complete')","'Both fixed source arms complete');require(fit['parameters_each']==116407,'Native source head116407')")
extra=extra.replace('use_sam=bool(ai),','use_sam=False,native_condition=bool(ai),').replace('total_updates=16000,','total_updates=16000,parameters_each=116407,')
a=extra.index("    target=set(e.get('explicit_direct_target_relative_paths',[]));");b=extra.index("    require(not own.intersection",a)
extra=extra[:a]+"    target=set(e.get('explicit_direct_target_relative_paths',[]));require(not target and e.get('direct_target_cache_complete')is None,'No direct target publication')\n"+extra[b:]
extra=extra.replace("    own=set();saved=set();checkpoints={}","    require(fit_binding['path']==str(phase/FIT_RUN/'COMPLETE.json') and barrier_binding['path']==str(phase/FIT_RUN/'ALL2_FINAL8000_FROZEN_BEFORE_SOURCE_TRAIN_OR_HOLDOUT_OUTPUTS.json'),'Exact source fit/barrier metadata paths')\n    own=set();saved=set();checkpoints={}")
extra=extra.replace("    require(not own.intersection(saved|target)","    require(own==set(ROOT_OWNED)and saved==set(ROOT_SAVED),'Exact scope2head24output whitelist')\n    require(not own.intersection(saved|target)")
helper=helper.replace('\ndef classify_file(',extra+'\n\ndef classify_file(');helper=replacefn(helper,'classify_file',fn(num,'classify_file'))
inv=fn(num,'inventory').replace('V194','V199').replace('optional1directtarget','0directtarget').replace('all24source predictions/0directtarget','all24source predictions/0directtarget');helper=replacefn(helper,'inventory',inv)
# Keep the inherited generic18tests; replace only whitelist assertions and add source seal fixtures.
prep=fn(helper,'prepare')
start=prep.index("    checks.append(('own_head_excluded'");end=prep.index('    barriers=',start)
numprep=fn(num,'prepare');a=numprep.index("    checks.append(('owned_head_whitelist'");b=numprep.index('    barriers=',a)
prep=prep[:start]+numprep[a:b]+prep[end:]
prep=prep.replace("'own_head_whitelist':[],'saved_array_whitelist':[],'direct_target_whitelist':[]","'own_head_whitelist':list(ROOT_OWNED),'saved_array_whitelist':list(ROOT_SAVED),'direct_target_whitelist':[]")
prep=prep.replace('exclude arrays, weights, rawRGB banks, SAM/features/cache and environments. No prediction/fit seal dependency or broad project scan.','Exclude raw inputs/targets/external weights; only exact2owned heads<16MiB+24saved outputs<32MiB permitted as opaque bytes after completed fit/source-output seal and all2 barrier metadata. No broad project scan.')
prep=prep.replace("'explicit_own_head_relative_paths':[],'explicit_saved_output_relative_paths':[]", "'source_fit_complete':{'path':str(PHASE/FIT_RUN/'COMPLETE.json'),'sha256':None},'source_prediction_seal':{'path':str(PHASE/FIT_RUN/'COMPLETE.json'),'sha256':None},'all2_final_states_freeze':{'path':str(PHASE/FIT_RUN/'ALL2_FINAL8000_FROZEN_BEFORE_SOURCE_TRAIN_OR_HOLDOUT_OUTPUTS.json'),'sha256':None},'explicit_own_head_relative_paths':list(ROOT_OWNED),'explicit_saved_output_relative_paths':list(ROOT_SAVED),'explicit_direct_target_relative_paths':[],'direct_target_cache_complete':None")
helper=replacefn(helper,'prepare',prep);put(H/'prepare_completed_phase_delta.py',helper)
builder=swap(oldb)
for n,v in [('PARENT_DELTA','publication/deltas/V198_20261024_v1/DELTA_MANIFEST.json'),('PARENT_CHAIN','MANIFEST_CHAIN_AFTER_V198.json'),('PARENT_INDEX','EXPERIMENT_INDEX_THROUGH_V198_KO.md')]:builder=assign(builder,n,v)
builder=once(builder,"len(requested)==11,'All11 V187..V197 ancestor metadata mandatory'","len(requested)==12,'All12 V187..V198 ancestor metadata mandatory'")
builder=builder.replace('Latest parent aliases exact V197 bindings','Latest parent aliases exact V198 bindings').replace('BASE+11ancestor deltas/index/chains and12closing plans','BASE+12ancestor deltas/index/chains and13closing plans')
build=fn(builder,'build');nb=fn(numb,'build');start=build.index('    candidate=checked(');end=build.index('    pure.assert_new_paths',start);a=nb.index('    authority=pure.source_artifact_authority');b=nb.index('    pure.assert_new_paths',a);build=build[:start]+nb[a:b].replace('anyoptional target','0directtarget')+build[end:]
build=build.replace("'publisher_tools':tools,","'publisher_tools':tools,'artifact_authority':authority,").replace("'entries':entries,'source_phase_original_records'","'entries':entries,'source_phase_original_records'")
build=build.replace("'scope':'Publication of root-frozen","'artifact_authority':authority,'scope':'Publication of root-frozen")
lines=build.splitlines()
claim='Completed V199 source-only matched fresh RAW zero8/onehot8 heads116407 each,8000 each,with independent source integrity audit and fixed source TRAIN/HOLDOUT readout. SourceTRAIN327 perfect both; weak synthetic-only45 heldout D28/30 both and E5/15 vs4/15; no native-ID improvement established.6sourcebases,real-end holdout0,weaknotphysicalgold,notfreshblind. No new real result, arm/checkpoint/data/threshold selection, adoption or fullgoal success. New V200 private metric body was unread at source freeze; this does not deny historical evaluation exposure. Exact2new owned checkpoints and24saved source outputs copied opaquely; no input/target/third-party weights.'
for i,l in enumerate(lines):
 if 'index=links+pure.versioned_index_bytes'in l:lines[i]="    index=links+pure.versioned_index_bytes(old_index,'V199 matched native input source fit and weak heldout limits',portable)"
 if 'manifest={'in l and "'scope':"in l:
  a=l.index("'scope':")+len("'scope':");b=l.index(",'model_adopted'",a);lines[i]=l[:a]+repr(claim)+l[b:]
 if l.lstrip().startswith('new_readme='):lines[i]="    new_readme=('# V199 source-only native 조건부 대조 학습 기록\\n\\n[최신 실험 인덱스]('+REPO_PREFIX+NEW_INDEX+') · [V199 원문·방법·출력·한계]('+REPO_PREFIX+portable+')\\n\\n새 RAW 두팔은 각116407parameter·8000step으로 고정됐습니다. Source TRAIN327은 두팔 모두 맞혔지만, weak 합성-only HOLDOUT45는 D28/30씩·E5/15 대4/15이며 native-ID 개선을 입증하지 못했습니다.6base·real-end HOLDOUT0·weak authority·기존 노출된 설계 한계를 보존하며 물리 gold·freshblind·실사 개선·선택·채택·전체 목표 해결을 주장하지 않습니다. Source freeze의 신규 실사 결과 미열람은 V200 새 metric에 한정되며 과거 실사 평가 노출을 부정하지 않습니다. 정확2개 자체 checkpoint와24저장 출력은 바이트만 보존하고 input/target/외부모델은 제외합니다.\\n\\n---\\n\\n').encode()+old_readme"
builder=replacefn(builder,'build','\n'.join(lines));bp=fn(builder,'prepare')
bp=bp.replace("'own_head_whitelist':[],'saved_array_whitelist':[],'direct_target_whitelist':[]","'own_head_whitelist':list(pure.ROOT_OWNED),'saved_array_whitelist':list(pure.ROOT_SAVED),'direct_target_whitelist':[]")
bp=bp.replace('source_contract_review_a/KO.md,independent_hypothesis_review_b/KO.md','independent_final_source_report_review_a/KO.md,independent_code_review_b/KO.md')
needle="'expected_remote_main':None,'publisher_tool_bindings':[],'admin_output'"
replacement="'expected_remote_main':None,'publisher_tool_bindings':[],'source_fit_complete':{'path':str(V/pure.FIT_RUN/'COMPLETE.json'),'sha256':None},'source_prediction_seal':{'path':str(V/pure.FIT_RUN/'COMPLETE.json'),'sha256':None},'all2_final_states_freeze':{'path':str(V/pure.FIT_RUN/'ALL2_FINAL8000_FROZEN_BEFORE_SOURCE_TRAIN_OR_HOLDOUT_OUTPUTS.json'),'sha256':None},'explicit_own_head_relative_paths':list(pure.ROOT_OWNED),'explicit_saved_output_relative_paths':list(pure.ROOT_SAVED),'explicit_direct_target_relative_paths':[],'direct_target_cache_complete':None,'admin_output'"
assert needle in bp;bp=bp.replace(needle,replacement);builder=replacefn(builder,'prepare',bp);put(H/'build_completed_phase_delta.py',builder)
# Adapt mock source fit metadata and all26 opaque numeric byte fixtures before build.
fixture=swap(read(OLD/'run_invented_delta_roundtrip.py').decode()).replace('through_V197','through_V198').replace('latest_V197','latest_V198').replace('all_twelve_closing','all_thirteen_closing').replace('newindex_exact_V197_suffix','newindex_exact_V198_suffix')
fixture=fixture.replace("format(i+1,'x')*64","hashlib.sha256(('delta'+str(i)).encode()).hexdigest()").replace("format(i+3,'x')*64","hashlib.sha256(('chain'+str(i)).encode()).hexdigest()").replace("format(i+5,'x')*64","hashlib.sha256(('index'+str(i)).encode()).hexdigest()")
fixture=once(fixture,"chain['deltas'][:11]==refs and len(refs)==11 and len(chain['deltas'])==12","chain['deltas'][:12]==refs and len(refs)==12 and len(chain['deltas'])==13")
needle="        rows=[]\n        for name in ['PORTABLE_REPORT_KO.md'";insert="""        fit,barrier,fb,bb=m.pure.invented_fit_metadata(phase)
        for arm,record in fit['arms'].items():
            bindings=[record['checkpoint']]+list(record['outputs'].values())+[q['outputs']for groups in record['source_cohorts'].values()for q in groups.values()]
            for z in bindings:
                path=Path(z['path']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('OPAQUE-'+path.name).encode());z.update(b(path))
        barrier['checkpoints']={a:r['checkpoint']for a,r in fit['arms'].items()};w(Path(bb['path']),barrier);bb=b(Path(bb['path']));fit['all2_freeze']=bb;w(Path(fb['path']),fit);fb=b(Path(fb['path']))
        art=dict(source_fit_complete=fb,source_prediction_seal=fb,all2_final_states_freeze=bb,explicit_own_head_relative_paths=list(m.pure.ROOT_OWNED),explicit_saved_output_relative_paths=list(m.pure.ROOT_SAVED),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None)
        rows=[]
        for name in ['PORTABLE_REPORT_KO.md'"""
assert needle in fixture;fixture=fixture.replace(needle,insert)
fixture=fixture.replace("'RESULT.json','RAW_INPUTS.npz']:","'RESULT.json','RAW_INPUTS.npz']+list(m.pure.ROOT_OWNED)+list(m.pure.ROOT_SAVED):")
fixture=fixture.replace("exclude=name.endswith('.npz');rows.append", "exclude=name=='RAW_INPUTS.npz';action='EXCLUDE'if exclude else m.pure.classify_file(name,p.suffix,s.st_size,set(m.pure.ROOT_OWNED),set(m.pure.ROOT_SAVED));rows.append")
fixture=fixture.replace("status='CANDIDATE_EXCLUDE' if exclude else 'CANDIDATE_INCLUDE',action='EXCLUDE' if exclude else 'INCLUDE'","status='CANDIDATE_'+action,action=action")
fixture=fixture.replace("m.P=source;m.V=phase;m.BUILDER_PROTOCOL=proto;m.build(entry)","m.P=source;m.V=phase;m.pure.PHASE=phase;m.BUILDER_PROTOCOL=proto;e.update(art);w(entry,e);inventory=json.loads(inv.read_text());inventory['artifact_authority']=m.pure.source_artifact_authority(e);w(inv,inventory);e['inventory_complete']=b(inv);w(entry,e);m.build(entry)")
put(H/'run_invented_delta_roundtrip.py',fixture)
put(H/'LIMITED_DERIVATION_AND_GENERIC_AST_PARITY.json',dict(completed=True,PASS=True,source_generic='V198 immutable byte/security kernels; V194 declared_fit_artifacts/source_artifact_authority/classifier restored with strict bothRAW/nativebool and2/24/0 whitelist',changes=['12ancestors/13closing','116407 RAW native twoarms','Completed sourcefit prose not newV200 realmetrics','sourcefit metadata closure,2new owned checkpoint+24source outputs byteonly;0targets'],actual_phase_repo_API_Git_model_access=0))
put(H/'INHERITED_PUBLISHER_SOURCE_BINDINGS.json',dict(completed=True,source_final=bind(OLD/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json'),source_code=[bind(OLD/n)for n in ['prepare_completed_phase_delta.py','build_completed_phase_delta.py','run_invented_delta_roundtrip.py']],numeric_authority_code=[bind(N/n)for n in ['prepare_completed_phase_delta.py','build_completed_phase_delta.py']],scope=bind(SCOPE)))
put(H/'DELTA_OPERATOR_WORKFLOW_KO.md','# V199 source-fit 게시\n\n실제phase/repo/parent/모델·배열 읽기 없이 코드만 준비한다. Root actual entry에서 과학단계완료·보고서QA·freeze·fit COMPLETE/source-output seal(동일binding)·양FINALbarrier를 먼저 확인한다.\n\n정확2개 새owned checkpoint<16MiB와24개 저장출력<32MiB만 rawbyte로 복사한다. 모든 원input/target/RGB/SAM/feature/외부모델은 제외한다. 2MiB textgzip,6portablepages,BASE+187..198의12ancestors/13closing 보존과 oldREADME/index suffix 정책은 같다.\n\nsource TRAIN apparent와 약한 합성HOLDOUT 한계·realendheld0·과거설계노출·신규V200metric미열람의좁은의미를 명시한다. 원격Git/실사결과에 따른선택은 없다. 실패를보존하고자동retry하지않는다.\n')
for n in ['prepare_completed_phase_delta.py','build_completed_phase_delta.py']:subprocess.run([sys.executable,'-B',str(H/n),'--prepare'],check=True)
subprocess.run([sys.executable,'-B',str(H/'run_invented_delta_roundtrip.py')],check=True)
put(H/'PORTABLE_ENTRY_POINTS_BEFORE_PHASE_READS.json',dict(completed=True,prepared_only=True,phase='V199',portable_pages=root['portable_pages'],actual_phase_reads=0))
put(H/'CUMULATIVE_PARENT_CLOSURE_PROTOCOL.json',dict(completed=True,ancestor_tags=[t for t,_ in anc],immutable_closing_tags=list(clos),ownheads=heads,savedoutputs=saved,directtargets=[],actualphase_reads=0))
names=['derive_v199_code_only.py','prepare_completed_phase_delta.py','build_completed_phase_delta.py','run_invented_delta_roundtrip.py','LIMITED_DERIVATION_AND_GENERIC_AST_PARITY.json','INHERITED_PUBLISHER_SOURCE_BINDINGS.json','DELTA_OPERATOR_WORKFLOW_KO.md','GENERATED_INVENTED_DELTA_CHECKS.json','GENERATED_BUILDER_CHECKS.json','INVENTED_COPY_BUILD_PLAN_ROUNDTRIP.json','PROTOCOL_BEFORE_V199_PUBLICATION_ENUMERATION_OR_COPY.json','BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json','ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json','ROOT_DELTA_BUILD_ENTRY_TEMPLATE_APPROVED_FALSE.json','ROOT_DELTA_PUSH_PLAN_ENTRY_TEMPLATE_APPROVED_FALSE.json','PORTABLE_ENTRY_POINTS_BEFORE_PHASE_READS.json','CUMULATIVE_PARENT_CLOSURE_PROTOCOL.json']
checks=[json.loads(read(H/n))for n in ['GENERATED_INVENTED_DELTA_CHECKS.json','GENERATED_BUILDER_CHECKS.json','INVENTED_COPY_BUILD_PLAN_ROUNDTRIP.json']];assert all(c['PASS']for c in checks)
final=dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,phase='V199',files={n:bind(H/n)for n in names},inventory_code=bind(H/'prepare_completed_phase_delta.py'),builder_code=bind(H/'build_completed_phase_delta.py'),checks=dict(inventory=len(checks[0]['checks']),builder=len(checks[1]['checks']),fullcopy_gzip_pushplan=checks[2]['count']),owned_heads=2,saved_numeric_arrays=24,direct_target_arrays=0,ancestor_delta_count=12,prior_closing_plan_count=13,gzip_text_threshold_bytes=2097152,portable_pages=root['portable_pages'],expected_remote_main=None,root_code_preparation_scope=bind(SCOPE),actual_phase_inventory_source_hash_copy_parent_repo_API_Git_or_model_calls=0,no_edits_after_final_ready_without_root_request=True);put(H/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json',final)
print('PUBLISHER_FINAL',json.dumps(bind(H/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json')))
