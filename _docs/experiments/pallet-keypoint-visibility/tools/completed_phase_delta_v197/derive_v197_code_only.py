"""Finite code-only V196->V197 publication adaptation; no phase/repo reads."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).absolute().parent.parent
OLD=R/'V196_DELTA_PREPARATION_B';H=Path(__file__).absolute().parent
OLD_PHASE='query_visibility_grouped_source_holdout_target_readout_20261024_v196'
NEW_PHASE='query_visibility_train_only_target_contract_readout_20261024_v197'
def put(name,value):
 p=H/name;data=value.encode() if isinstance(value,str) else (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
 with p.open('xb') as f:f.write(data)
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def swap(text):
 return text.replace(OLD_PHASE,'__EXACT_CURRENT_PHASE__').replace('V196','V197').replace('v196','v197').replace('__EXACT_CURRENT_PHASE__',NEW_PHASE).replace('V197_DELTA_PREPARATION_B','V197_DELTA_PREPARATION_C').replace('ROOT_V197_PUBLICATION_CODE_PREPARATION_SCOPE_B','ROOT_V197_PUBLICATION_CODE_PREPARATION_SCOPE_C')
def replace_once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
oldhelper=(OLD/'prepare_completed_phase_delta.py').read_text();helper=swap(oldhelper)
tree=ast.parse(oldhelper);constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'ANCESTOR_PHASES','CLOSING_TAGS'}}
for name,value in [('ANCESTOR_PHASES',constants['ANCESTOR_PHASES']+(('V196',OLD_PHASE),)),('CLOSING_TAGS',constants['CLOSING_TAGS']+('V196',))]:
 lines=helper.splitlines();indexes=[i for i,l in enumerate(lines) if l.startswith(name+'=')];assert len(indexes)==1;lines[indexes[0]]=name+'='+repr(value);helper='\n'.join(lines)+'\n'
helper=helper.replace('BASE through V195','BASE through V196')
needle="    checks.append(('phase_QA_freeze_barriers_accept',require_completion_barriers(barriers)))"
helper=replace_once(helper,needle,needle+"\n    alternate=[dict(r) for r in barriers];alternate[-1]=dict(alternate[-1],expected_fields={'scientific_phase_frozen_no_further_phase_writes':True})\n    checks.append(('explicit_alternate_freeze_field_accept',require_completion_barriers(alternate)))")
put('prepare_completed_phase_delta.py',helper)
oldbuilder=(OLD/'build_completed_phase_delta.py').read_text();builder=swap(oldbuilder)
for name,value in [('PARENT_DELTA','publication/deltas/V196_20261024_v1/DELTA_MANIFEST.json'),('PARENT_CHAIN','MANIFEST_CHAIN_AFTER_V196.json'),('PARENT_INDEX','EXPERIMENT_INDEX_THROUGH_V196_KO.md')]:
 lines=builder.splitlines();ix=[i for i,l in enumerate(lines) if l.startswith(name+'=')];assert len(ix)==1;lines[ix[0]]=name+'='+repr(value);builder='\n'.join(lines)+'\n'
builder=replace_once(builder,"require(len(requested)==9,'All9 V187..V195 ancestor metadata mandatory')","require(len(requested)==10,'All10 V187..V196 ancestor metadata mandatory')")
builder=builder.replace('Latest parent aliases exact V195 bindings','Latest parent aliases exact V196 bindings')
builder=builder.replace('BASE+9ancestor deltas/index/chains and10closing plans','BASE+10ancestor deltas/index/chains and11closing plans')
lines=builder.splitlines()
for i,line in enumerate(lines):
 if 'index=links+pure.versioned_index_bytes' in line:lines[i]="    index=links+pure.versioned_index_bytes(old_index,'V197 TRAIN-only saved target/input correspondence readout',portable)"
 if 'manifest={' in line and "'scope':'Publication of root-frozen" in line:
  start=line.index("'scope':")+len("'scope':");end=line.index(",'model_adopted'",start)
  claim='Publication of root-frozen V197 TRAIN-only saved target/input correspondence readout:109 source metadata records, outcome-blind fixed12 selected triplets/288 native rows/inputU24/36 weak labels/252UNKNOWN, and36 full-context review images. No new source generation, labels, fit/inference/scores, native physical gold, causal-mechanism proof, model/data/threshold selection, adoption or goal-success claim. All numeric arrays and weights excluded.'
  lines[i]=line[:start]+repr(claim)+line[end:]
 if line.lstrip().startswith('new_readme='):
  lines[i]="    new_readme=('# V197 TRAIN-only saved target/input correspondence records update\\n\\n[최신 실험 인덱스]('+REPO_PREFIX+NEW_INDEX+') · [V197 원문·그림·표·실패기록]('+REPO_PREFIX+portable+')\\n\\n이 추가 기록은 TRAIN109개 source metadata와 예측 오류로 선택하지 않은 고정12묶음288행·원입력U24·약한 표적36개/UNKNOWN252개 및 full-context36그림의 입력-표적 대응 진단입니다. 새 학습·추론·점수·라벨·실사 성능개선·원인 확정·native 물리 정답·모델 선택·채택이나 전체 목표 해결을 뜻하지 않습니다. 원본 배열은 제외하고 원문·표·그림·한계를 보존합니다.\\n\\n---\\n\\n').encode()+old_readme"
 if "'portable':'Explicit final ROOT_RESULTS.md" in line:
  a=line.index("'portable':")+len("'portable':");b=line.index(",'historical_README_exception'",a)
  desc='Exactly root-fixed PORTABLE_REPORT/ROOT_REPORT/ROOT_READOUT, fixed_train_target_contract_viewer_v1 portable gallery+HTML, independent_train_target_interpretation_a/KO.md and THROUGH_V197 index are independently link checked. ROOT_RESULTS_KO.json is a direct target only; no ROOT_RESULTS_KO.md is assumed.2MiB preserves small linked JSON; gzip descriptors retain larger originals. No silent link fallback.'
  lines[i]=line[:a]+repr(desc)+line[b:]
 if "'message':'Publish V197 saved source holdout" in line:lines[i]=line.replace('Publish V197 saved source holdout diagnostics and preserved limits','Publish V197 TRAIN-only saved target correspondence and preserved limits')
builder='\n'.join(lines)+'\n';put('build_completed_phase_delta.py',builder)
fixture=swap((OLD/'run_invented_delta_roundtrip.py').read_text())
fixture=fixture.replace("str(i+1)*64","format(i+1,'x')*64").replace('through_V195','through_V196').replace('latest_V195','latest_V196').replace('all_ten_closing','all_eleven_closing').replace('newindex_exact_V195_suffix','newindex_exact_V196_suffix')
fixture=replace_once(fixture,"chain['deltas'][:9]==refs and len(chain['deltas'])==10","chain['deltas'][:10]==refs and len(refs)==10 and len(chain['deltas'])==11")
put('run_invented_delta_roundtrip.py',fixture)
# New phase identity/ancestor lengths are the only runtime structural changes;
# storage, security, collision, credential and plan arithmetic remain inherited.
def funcs(s):return {n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
class IgnoreStrings(ast.NodeTransformer):
 def visit_Constant(self,node):
  if isinstance(node.value,str):return ast.copy_location(ast.Constant(value='<STRING>'),node)
  return node
parity={}
for name,old,new,wanted in [('helper',oldhelper,helper,['require','safe_parts','sha_bytes','open_nofollow','sha_file','read_checked','write','physical_paths','assert_new_paths','closing_protected_entries','validate_parent_closure','versioned_index_bytes','bounded_batches','require_completion_barriers','classify_file','inventory']),('builder',oldbuilder,builder,['require','opened','read_bytes','sha','bind','checked','checked_json','write_new','json_bytes','raw_record','check_reviews','recheck_parent_bindings','build','finalize'])]:
 oldf,newf=funcs(old),funcs(new)
 for fn in wanted:
  a=ast.dump(IgnoreStrings().visit(ast.parse(ast.unparse(oldf[fn]))),include_attributes=False);b=ast.dump(IgnoreStrings().visit(ast.parse(ast.unparse(newf[fn]))),include_attributes=False);parity[name+'.'+fn]=a==b
assert all(parity.values()),parity
put('LIMITED_DERIVATION_AND_GENERIC_AST_PARITY.json',{'completed':True,'PASS':True,'generic_functions_stripped_strings_exact':parity,'structural_changes':['Exact V197 phase and names','V196 latest parent;10ancestors/11closings','No whitelist:0heads0arrays0targets','Updated source-only prose, unchanged2MiB policy','Invented10th ancestor uses64hex SHA fixture','Invented alternate explicit freeze-field acceptance'],'actual_phase_repo_parentmanifest_Git_API_model_calls':0})
put('INHERITED_PUBLISHER_SOURCE_BINDINGS.json',{'completed':True,'source_final':bind(OLD/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json'),'source_code':[bind(OLD/n)for n in ['prepare_completed_phase_delta.py','build_completed_phase_delta.py','run_invented_delta_roundtrip.py']],'scope':bind(R/'ROOT_V197_PUBLICATION_CODE_PREPARATION_SCOPE_C.json'),'no_actual_source_phase_reads':True})
put('DELTA_OPERATOR_WORKFLOW_KO.md','# V197 완료 단계 게시 절차\n\n이 폴더는 코드·프로토콜·가상 검사만 준비한다. 현재 parent main과 미래 phase_result/독립 보고서 QA/freeze 바인딩은 null이며, 별도 root actual scope 전에는 실제 V197 열거·해시·복사·저장소·부모 manifest·API 접근을 하지 않는다.\n\n1. 완료된 V197의 실제 ROOT_PHASE_RESULT/독립 report QA/명시적 freeze를 root entry의 정확한 필드와 SHA로 바인딩한다.\n2. BASE+V187..V196의 10개 delta/chain/index와 11개 closing plan을 순서대로 보존한다. BASE 과거 README 한 개 예외 외의 old logical/raw/gzip descriptor/chunk/cycle 경로는 덮어쓰지 않는다. 현재 README는 before SHA와 원문 suffix로만 갱신한다.\n3. 정확 V197만 inventory한다. 코드·문서·표·검토 PNG/full-context 그림은 포함하고 자체/외부 가중치·모든 NPZ/NPY/feature/SAM/raw cache는 제외 이유를 남긴다. ownhead/savedarray/target whitelist는 빈 집합이다.\n4. 원문 바이트는 보존하며 2MiB 초과 text만 lossless gzip32MiB chunks와 physical descriptor로 저장한다. ROOT_RESULTS JSON 직접링크와 root 고정7개 portable page는 독립 링크 검산을 통과해야 한다.\n5. fresh delta/THROUGH_V197 index/AFTER_V197 chain을 쓰고 독립 completed/PASS byte·credential·link 검산 후에만 480MiB 이하 exact physical pushplan을 봉인한다. Git commit/push는 root 별도 승인·실행이다.\n\nTRAIN-only 저장 입력-약한 표적 진단이며109metadata/고정12묶음288행/U24/약한표적36·UNKNOWN252/full-context36을 보존한다. 새 fit/forward/score/label/native physical gold/인과·일반화·채택·목표완료를 주장하지 않는다. 실패는 보존하며 자동 retry하지 않는다.\n')
print(json.dumps({'completed':True,'code_only':True,'helper':bind(H/'prepare_completed_phase_delta.py'),'builder':bind(H/'build_completed_phase_delta.py'),'generic_functions':len(parity)}))
