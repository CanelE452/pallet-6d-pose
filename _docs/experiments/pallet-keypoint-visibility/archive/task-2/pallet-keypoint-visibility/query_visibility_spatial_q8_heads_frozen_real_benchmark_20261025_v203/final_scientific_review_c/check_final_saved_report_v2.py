"""Bounded V203 report/table/link QA; no packet/model/input or metric rerun."""
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,math,csv,io,re,os,stat
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent;G=V/'fixed12_spatial_q8_real_visual_v1';checks=[]
def need(x,m):
 if not x:raise AssertionError(m)
 checks.append(m)
def raw(p):
 p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Finite lexical source path')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for s in p.parts[1:-1]:n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'rb')as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'No symlink regular file');return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def bound(b):
 a=raw(b['path']);need(hashlib.sha256(a).hexdigest()==b['sha256'],'Exact bound file SHA');return a

def js(p):return json.loads(raw(p))
def equal(a,b):
 if isinstance(a,(float,int))and isinstance(b,(float,int)):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 return a==b
class Page(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.tables=[];self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  self.refs +=[v for k,v in a if k in('href','src')and v]
  if t=='table':self.tables.append([])
  if t=='tr':self.row=[]
  if t in('th','td'):self.cell=''
 def handle_data(self,s):
  if self.cell is not None:self.cell+=s
 def handle_endtag(self,t):
  if t in('th','td'):self.row.append(self.cell);self.cell=None
  if t=='tr':self.tables[-1].append(self.row);self.row=None
qa=js(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json');need(qa['completed']and qa['PASS'],'Closed independent numeric PASS')
build=js(V/'report_preparation_a/ROOT_REPORT_BUILD_COMPLETE.json');need(build['completed']and build['PASS'],'Final report build PASS')
for b in build['outputs'].values():bound(b)
r=js(V/'ROOT_RESULTS_KO.json');gal=js(G/'COMPLETE.json');gs=js(G/'SUMMARY_AND_TABLES.json');ev=js(V/'saved_real_head_evaluation_v1/COMPLETE.json')
need(r['independent_numeric_review']==bind(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json')and r['evaluation_complete']==qa['evaluation_complete']and r['prediction_seal']==qa['prediction_seal'],'Exact closed numeric/evaluation/prediction bindings')
need(r['all_models12']==list(qa['current']['primary']['ALL'])and len(r['table_primary36'])==36,'All12 original and new model primary36 rows')
for row in r['table_primary36']:
 q=qa['current']['primary'][row['group']][row['model']]
 for k,v in row.items():
  if k in('group','model'):continue
  need(equal(v,q[{'D':'D_N','E':'E_N'}.get(k,k)]),'Primary ALL/material CM P/R/U exact '+k)
need(r['primary1237']==dict(D=1147,E=90)and sum(r['primary1237'].values())==1237,'Primary1237 D1147/E90 never same-slot1248')
need(r['original_input_U']==dict(current_N=1784,current_U=45,current_U_rate=45/1784,public_N=360,public_U=108),'Original inputU45/1784 distinct from primary U')
source=json.loads(bound(ev['outputs']['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json']));pub=json.loads(bound(ev['outputs']['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json']))
need(source['same_slot_N']==1248 and source['primary_D_E_N']==1237,'Saved cohort distinction retained')
for row in r['table_SELF24']:
 rr=[s for s in source['tables']['human_self_binary_response']if s['cohort']==row['cohort']and s['model']==row['model']];sums={k:sum(s[k]for s in rr)for k in('human_S_support','binary_hidden_response','visible_response','unresolved')}
 need(row['N']==sums['human_S_support']==(21 if row['cohort']=='paired_same_query_RGB'else 136)and row['binary_hidden_response']==sums['binary_hidden_response']and row['visible_response']==sums['visible_response']and row['U']==sums['unresolved']and row['U_rate']==row['U']/row['N']and row['learned_self_or_cause']is False,'SELF21/136 original responses separate')
for row in r['table_public_weak11']:
 m=pub['cohorts']['FULL360']['ALL']['models'][row['model']];t=m['binary_T_nonT']['per_class'];f=m.get('owner3',{}).get('per_class',{}).get('FOREIGN_VISIBLE_SUPPORTED',{})
 need((row['all_original_rows'],row['known_center_truth'],row['unknown_center_truth'],row['original_input_U'])==(360,204,156,108),'Public weak360/204/156/inputU108')
 for k,x in [('T_precision_known',t['T']['precision_on_known_truth']),('T_recall_including_U',t['T']['recall_including_prediction_U']),('nonT_recall_including_U',t['NON_T']['recall_including_prediction_U']),('ownerF_precision_known',f.get('precision_on_known_truth')),('ownerF_recall_including_U',f.get('recall_including_prediction_U'))]:need(equal(row[k],x),'Public weak ratios exact')
need(qa['old10_saved_ledger_fields_and_metric_counts_ratios_unchanged']and qa['all_existing_tables']['self136_paired21']and qa['all_existing_tables']['reference461_separate']and qa['all_existing_tables']['manual376_membership_exact']and qa['all_existing_tables']['native8_and_transition_tables'],'Closed native8/reference461/manual376/old10 full audit retained')
need(r['table_primary36']==gs['primary36']and r['table_SELF24']==gs['self24']and r['table_public_weak11']==gs['public_weak11'],'Report/gallery all71 table rows identical')
md=raw(V/'PORTABLE_REPORT_KO.md').decode();need(raw(V/'ROOT_REPORT_KO.md')==md.encode(),'ROOT MD/PORTABLE exact bytes');ht=raw(V/'ROOT_READOUT_KO.html').decode();page=Page();page.feed(ht)
expected=[r['table_primary36'][:12],r['table_primary36'][12:],r['table_SELF24'],r['table_public_weak11']]
need(page.tables==[[list(rows[0])]+[[str(x[k])for k in rows[0]]for x in rows]for rows in expected],'HTML source all primary/material/SELF/public cells exact')
mdtables=[]
for block in re.findall(r'(?m)(^\|.*(?:\n\|.*)+)',md):
 rows=[x.strip('|').split('|')for x in block.splitlines()];mdtables.append([rows[0]]+rows[2:])
need(mdtables==page.tables,'Markdown all table cells exact HTML/JSON')
manifest=js(G/'IMAGE_MANIFEST.json');need(r['fixed12_ids']==gal['exact_original_ids']==[0,1,2,3,4,5,7,8,256,300,840,996],'Original fixed12 no new outcome selection')
for im in manifest['images']:
 need(im['clean']['sha256']==im['old_clean']['sha256']and im['valid']['sha256']==im['old_valid']['sha256'],'Recorded original saved clean/valid encoded copy parity')
 for k in('card','clean','valid'):bound(im[k])
for b in manifest['contacts']:bound(b)
linkcounts={}
for p,html in [(V/'ROOT_REPORT_KO.md',False),(V/'PORTABLE_REPORT_KO.md',False),(V/'ROOT_READOUT_KO.html',True),(G/'PORTABLE_GALLERY_KO.md',False),(G/'index.html',True)]:
 t=raw(p).decode();z=Page();z.feed(t)if html else None;refs=z.refs if html else re.findall(r'\]\(([^)]+)\)',t)
 for ref in refs:
  need(not ref.startswith(('/','//'))and not re.match(r'^[A-Za-z]+:',ref)and 'v112'not in ref.casefold(),'Portable relative ref only');q=Path(os.path.normpath(p.parent/ref));need(q.suffix.lower()not in('.npz','.npy','.pt','.pth'),'No raw array or weights linked/read');raw(q)
 linkcounts[str(p.relative_to(V))]=len(refs)
visual=js(H/'ACTUAL_FIXED6_CARD_VISUAL_QA.json');need(visual['actual_fixed_ids']==[1,3,4,5,7,8]and visual['actual_toolview_card_count']==6,'Own6 actual tools.view_image retained')
for b in r['actual_visual_review_bindings']:bound(b)
need(r['selected_model']is None and r['adopted']is False and r['fully_solved']is False and r['source_results_selected_arm']is False and r['source_weak_not_native_gold']and r['repeated_exposure_not_freshblind']and r['legacy_realCAL_history_preserved']and r['original_qL_native_and_sampler_unchanged'],'No adoption/selection/physicalgold/freshblind claim')
result=dict(completed=True,PASS=True,code=bind(__file__),report_build_complete=bind(V/'report_preparation_a/ROOT_REPORT_BUILD_COMPLETE.json'),root_results=bind(V/'ROOT_RESULTS_KO.json'),root_portable=bind(V/'PORTABLE_REPORT_KO.md'),root_HTML=bind(V/'ROOT_READOUT_KO.html'),independent_numeric_review=r['independent_numeric_review'],gallery_complete=r['gallery_complete'],actual_visual_review=bind(H/'ACTUAL_FIXED6_CARD_VISUAL_QA.json'),check_count=len(checks),table_rows=dict(primary36=36,SELF24=24,public_weak11=11),relative_link_occurrences=linkcounts,primary_vs_sameslot=dict(primary1237=1237,D1147=1147,E90=90,same_slot1248=1248),all_new2_and_old10_closed_numeric_tables_exact=True,own_actual_CARD_toolviews6=True,figure_count_other_than_cards=0,browser_rendered=False,new_packet_or_input_array_weight_model_header_forward_or_metric_reexecution=0,semantic_review_ko='주평가1237와 same-slot1248을 혼합하지 않는다. SELF21/136은 binary response이며 원인 인증이 아니다. ref461과 manual376은 고정연관표다. Public weak owner/crop은 native gold가 아니다. q8은 원initializer 정보이며 GT를 주입한 입력으로 서술하지 않는다. 두팔tradeoff를 보고하고 승자/threshold/causal factor를 선택하지 않았다.',limits=['C 직접view6개만. Root 별도6과 합친 범위는 별도영수증이며 C가 all12 또는 all1784를 봤다는 뜻이 아니다.','6카드와 저장 수치/문서만 검증했다. 원crop NPZ·source RGB·checkpoint·새forward·browser 검증0.','PNG source parity는 encoded SHA/저장manifest 확인이다. 이미 완료된 원human authority+새4288packet 독립 수치감사는 재실행하지 않았다.','ROOT_RESULTS의 new_model_forward_or_fit=0은 report generator의 실행범위로 해석하며 실제V203 prediction실행0을 뜻하지 않는다.'])
with (H/'REVIEW.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
with (H/'KO.md').open('x')as f:f.write('V203 최종 수치·문서·고정갤러리 QA PASS. 36개 주평가/재질 표, 24개 SELF 응답 표, 11개 public weak 표 및 MD/HTML/JSON 셀을 완료된 독립 감사와 대조했다. Primary1237(D1147/E90)와 same-slot1248은 별개다. C가 실제 본 것은 지정6카드이며 물리 native identity나 실사 실패 원인을 crop으로 새로 확정하지 않았다. 새모델 채택·threshold 선택·목표완료 주장은 없다.\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),checks=result['check_count'],links=linkcounts),ensure_ascii=False))
