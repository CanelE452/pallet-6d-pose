"""C independent saved-report QA only; stdlib/no images decode/arrays/models."""
from pathlib import Path
from html.parser import HTMLParser
import ast,datetime,hashlib,json,os,re,stat
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
def need(v,m):
 if not v:raise AssertionError(m)
def data(path):
 p=Path(path);need(p.is_absolute() and P in p.parents and '..' not in p.parts and all('v112' not in s.casefold() for s in p.parts),'Safe exact path before access')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for name in p.parts[1:-1]:
   nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(last).st_mode),'Regular saved text/imagehash only')
 with os.fdopen(last,'rb') as f:return f.read()
def sha(d):return hashlib.sha256(d).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(data(p)))
def checked(b):
 d=data(b['path']);need(sha(d)==b['sha256'],'Exact bound saved bytes');return d
def obj(b):return json.loads(checked(b))
checks=[]
def test(name,v):need(v,name);checks.append(dict(name=name,PASS=True))
scopepath=V/'ROOT_ACTUAL_FINAL_REPORT_QA_EXPLICIT_U_RATE_SCOPE.json'
test('Latest explicit_U scope binding',sha(data(scopepath))=='8beb53593c053630c71eddc5780096cbd099247d7171bbeb9b82ef6ead7d8707')
scope=json.loads(data(scopepath));need(scope['approved_independent_final_report_QA'] is True,'Root actual QA entry')
reports={Path(b['path']).name:b for b in scope['final_reports']};need(len(reports)==5,'Exactly five top reports')
actual={n:checked(b) for n,b in reports.items()};r=json.loads(actual['ROOT_RESULTS_KO.json']);phase=json.loads(actual['ROOT_PHASE_RESULT.json']);md=actual['ROOT_REPORT_KO.md'].decode();html=actual['ROOT_READOUT_KO.html'].decode()
test('Five current top hashes and identical portable_MD',actual['PORTABLE_REPORT_KO.md']==actual['ROOT_REPORT_KO.md'])
for name,b in phase['reports'].items():test('Phase report binding '+name,reports[name]==b)
proto=obj(phase['report_protocol']);code=checked(phase['report_code']);inputs={k:obj(b) for k,b in proto['inputs'].items()}
test('Source/code/protocol/input closure',proto['code']==r['report_code']==phase['report_code'] and r['report_protocol']==phase['report_protocol'] and r['entry']==phase['entry'] and r['evidence']==phase['evidence']==proto['inputs'])
preserved=obj(scope['preservation_receipt']);old={};preserved_records=[]
for item in preserved['records']:
 b=item['preserved'];raw=checked(b);need(len(raw)==item['original']['size_bytes']==b['size_bytes'] and b['sha256']==item['original']['sha256'],'Preserved original SHA/size equality')
 old[item['original']['path']]=raw;preserved_records.append(b)
test('All13 originals bytepreserved via receipt actual paths',len(preserved_records)==13 and preserved['underlying_summary_evaluation_prediction_or_images_changed'] is False)
before=json.loads(old[str(V/'ROOT_RESULTS_KO.json')]);added=set(r)-set(before);changed={k for k in set(r)&set(before) if r[k]!=before[k]}
test('Metric payload unchanged across presentation repair',added=={'primary_U_rate_by_model','original_current_input_U_rate'} and changed=={'entry','report_protocol','limitations_KO','report_code'})
oldphase=json.loads(old[str(V/'ROOT_PHASE_RESULT.json')]);test('Original evidence bindings unchanged',oldphase['evidence']==phase['evidence'])
oldproto=json.loads(old[phase['report_protocol']['path']]);test('Protocol exact evidence unchanged',oldproto['inputs']==proto['inputs'])
aa={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
bb={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old[phase['report_code']['path']]).body if isinstance(n,ast.FunctionDef)}
test('All other generator helper AST unchanged',set(aa)==set(bb) and all(aa[n]==bb[n] for n in aa if n not in {'short_rows','compose','invented'}))
s=inputs['summary'];head=inputs['root_headline'];numeric=inputs['independent_numeric'];models=['v61','legacy','C52','C53','C54','C55']
test('Independent saved evidence PASS',all(inputs[k]['completed'] is True and inputs[k]['PASS'] is True for k in ['gallery_complete','evaluation_complete','root_headline','reference_error','independent_numeric','root_visual','C_visual','source_fit_integrity']))
test('All6 and source summary exact',r['models']==models and r['table_all6_material18']==s['current_primary']['rows'] and r['table_primary6']==[x for x in s['current_primary']['rows'] if x['group']=='ALL'])
def ratio(a,b):return a/b if b else None
for x in r['table_all6_material18']:
 d=[x['D_D'],x['D_E'],x['D_U']];e=[x['E_D'],x['E_E'],x['E_U']];N=sum(d)+sum(e);h=head['results'][x['group']][x['model']];b=numeric['current']['primary'][x['group']][x['model']]
 values=dict(D_precision=ratio(d[0],d[0]+e[0]),D_recall=ratio(d[0],sum(d)),E_precision=ratio(e[1],d[1]+e[1]),E_recall=ratio(e[1],sum(e)))
 need(N==x['N'] and sum(d)==x['D_support'] and sum(e)==x['E_support'] and d[2]+e[2]==x['U'],'Full D/E/U denominators')
 need(h['CM_D_E_by_D_E_U']==b['CM_D_E_by_D_E_U']==[d,e] and all(x[k]==h[k]==b[k]==v for k,v in values.items()),'Independent integer CM/P/R vs headline and numeric review')
 if x['group']=='ALL':need(r['primary_U_rate_by_model'][x['model']]==x['U']/N==h['U_rate']==b['U_rate'],'Explicit primary U/N')
test('All18 CM/P/R/U and6 explicit_U_rates independent arithmetic',len(r['table_all6_material18'])==18 and len(r['primary_U_rate_by_model'])==6)
tablelines=md.splitlines();header=next(i for i,l in enumerate(tablelines) if l.startswith('|모델|N|D행'))
rows=[l.split('|')[1:-1] for l in tablelines[header+2:header+8]]
for cells,x in zip(rows,r['table_primary6']):
 need(cells[0]==x['model'] and int(cells[1])==1237 and cells[2]==str([x['D_D'],x['D_E'],x['D_U']]) and cells[3]==str([x['E_D'],x['E_E'],x['E_U']]),'Displayed CM rows')
 need(cells[4:8]==[f'{100*x[k]:.1f}%' for k in ['D_precision','D_recall','E_precision','E_recall']] and cells[8]==str(x['U']) and cells[9]==f'{100*x["U"]/x["N"]:.1f}%','Displayed percentages and Ucount/rate')
 for cell in cells:need('>'+cell+'<' in html,'HTML same primary numeric cells')
test('All6 displayed MD/HTML CM/P/R/Ucount/Urate match',len(rows)==6)
test('Factual input_U separate from primary_U',r['original_current_input_U_rate']==45/1784 and r['current_all1784']['N']==1784 and r['current_all1784']['input_U']==45 and '45/1784=2.52%' in md and '45/1784=2.52%' in html)
native=r['native8_material_full_denominators'];test('All native8 material original denominators',len(native)==16 and sum(x['all_original_slots'] for x in native)==1784 and sum(x['original_input_U'] for x in native)==45 and all(x['all_original_slots']==(105 if x['material']=='WOOD' else 118) for x in native))
test('Registered vs primary frames explicit distinct',r['registered_frames']=={'WOOD':105,'SQUARE':118} and r['primary_SQUARE_frames']==117 and 'SQUARE118' in md and 'primary는117프레임' in md)
for x in r['table_self6']:
 rr=[z for z in s['self_separate'] if z['cohort']=='paired_same_query_RGB' and z['model']==x['모델']]
 need(x['별도 human S N']==sum(z['human_S_support'] for z in rr)==21 and x['D 응답']==sum(z['visible_response'] for z in rr) and x['binary hidden 응답']==sum(z['binary_hidden_response'] for z in rr) and x['미결정']==sum(z['unresolved'] for z in rr),'Separate self responses arithmetic')
test('Self not external gold or learned-self success',len(r['table_self6'])==6 and 'learned-self classifier가 아니다' in md)
public=r['public_weak_center_only'];test('Full360 weak coverage separate',public==s['public_weak_owner_only'] and public['rows']==360 and public['coverage']['known_center204']==204 and public['coverage']['unknown_center156']==156 and public['coverage']['original_input_U']==108 and public['coverage']['center4_invalid']==112)
for x in r['table_public_weak5']:
 m=public['cohorts']['FULL360']['ALL']['models'][x['model']];b=m['binary_T_nonT'];cm=b['known_truth_confusion'];need(sum(map(sum,cm))==204,'Known weak truth full denominator')
 need(x['T_precision_known']==ratio(cm[0][0],cm[0][0]+cm[1][0]) and x['T_recall_including_U']==ratio(cm[0][0],sum(cm[0])) and x['nonT_recall_including_U']==ratio(cm[1][1],sum(cm[1])),'Weak binary precision/recall scalar')
 if 'owner3' in m:
  o=m['owner3']['known_truth_confusion'];need(x['ownerF_precision_known']==ratio(o[1][1],sum(z[1] for z in o)) and x['ownerF_recall_including_U']==ratio(o[1][1],sum(o[1])),'Weak F scalar')
 else:need(x['ownerF_precision_known'] is None and x['ownerF_recall_including_U'] is None,'No invented v61owner3')
test('Public5 scalar summary and missing legacy separated',len(r['table_public_weak5'])==5 and 'legacy public owner 경로는 없어서 만들지 않았다' in md)
ref=r['reference_error_LE3_humanD'];test('Reference461 vs manual376 authority kept distinct',ref==inputs['reference_error']['counts']['PRIMARY_HUMAN_D']['LE3'] and ref['N']==461 and r['manual_near_D_distinct_cohort']['N']==376 and r['manual_near_D_distinct_cohort']['WOOD']==153 and r['manual_near_D_distinct_cohort']['SQUARE']==223 and all(sum(z.values())==461 for z in ref['models'].values()))
class Refs(HTMLParser):
 def __init__(self):super().__init__();self.all=[];self.images=[]
 def handle_starttag(self,t,a):
  for k,v in a:
   if k in ['href','src']:self.all.append(v)
   if t=='img' and k=='src':self.images.append(v)
hr=Refs();hr.feed(html);mdrefs=re.findall(r'\]\(([^)]+)\)',md)
for relative in mdrefs+hr.all:
 need(not relative.startswith('/') and not re.match(r'^[A-Za-z]+:',relative) and '..' not in Path(relative).parts and all('v112' not in p.casefold() for p in Path(relative).parts),'Safe portable local literal links')
 need(stat.S_ISREG((V/relative).lstat().st_mode),'Existing local link regularfile')
test('Exactly24MD_9HTML local links exist',len(mdrefs)==24 and len(hr.all)==9)
inline=r['six_fixed_inline_cards'];manifest=inputs['image_manifest'];lookup={x['original_global_index']:x for x in manifest['images']}
for x in inline:need(x['card']==lookup[x['original_global_index']]['card'] and sha(checked(x['card']))==x['card']['sha256'],'Exact existing image manifest binding/hash only')
test('Inline6 unchanged files hashonly no decode',len(inline)==6 and len(hr.images)==6 and set(hr.images)=={str(Path(x['card']['path']).relative_to(V)) for x in inline})
test('Actual view and browser scope preserved',r['actual_visual_scope']=={'root':24,'independent_C':12,'overlap_possible':True,'all68_or1784_viewed':False,'browser_rendered':False} and inputs['gallery_complete']['selected_unique_cases']==68)
test('No adoption/goal/causal or fresh-blind claim',r['selected_model'] is None and r['adopted'] is False and r['fully_solved'] is False and r['goal_status']=='active_unsolved' and phase['source_fit_V190_phase_scope_separate'] is True and 'fresh blind 검증이 아니다' in md and '원인을 분리하거나' in md and '모델을 채택하지 않았다' in md)
pair=obj(scope['optional_fixed_pair_diagnostic']);test('Optional fixed_pair inequality only no newthreshold',pair['completed'] is True and pair['PASS'] is True and pair['visible_case']['original_global_index']==2 and pair['external_case']['original_global_index']==996 and pair['visible_case']['score']>pair['external_case']['score'] and 'actualprospective t remains0' in pair['decoder'])
receipt=dict(completed=True,PASS=True,review_type='Independent saved finalreport presentation/numeric consistency and provenance QA',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=bind(scopepath),current_reports=scope['final_reports'],QA_code=bind(__file__),checks=checks,check_count=len(checks),presentation_repair_originals_verified=13,underlying_metric_payload_unchanged=True,CM_P_R_U_rows_independently_recomputed=18,primary_U_N_rates_independently_recomputed=6,factual_input_U45_of1784_separate=True,portable_local_links=dict(MD=24,HTML=9),inline_image_file_SHA_only=6,new_image_decode_or_toolview=0,source_arrays_weights_models_GT_metric_execution_Git_network_reads_or_runs=0,frame_scope='Registered105WOOD/118SQUARE and declared117SQUAREprimary distinction checked in frozen report metadata/text; no new raw frame membership reconstruction.',limits=['Report arithmetic/provenance/link consistency, not new independent performance data or physical/native gold proof.','Existing separate numeric review authority retained; no raw ledger/NPZ or annotation reread, no evaluation rerun.','All13 pre-rate-fix records read only from PRESERVATION_RECEIPT.records[].preserved bindings.','Fixed-pair inequality rules out simultaneously correcting these two examples by one monotone scalar threshold, not a population cause proof or threshold choice.'],selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved')
with (H/'REVIEW.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,checks=len(checks),review=bind(H/'REVIEW.json'))))
