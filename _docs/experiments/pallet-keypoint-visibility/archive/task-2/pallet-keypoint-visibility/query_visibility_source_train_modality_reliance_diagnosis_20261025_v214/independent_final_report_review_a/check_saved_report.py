"""Closed V214 report/CSV/provenance/link QA only; no images, tensors or raw GT."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from collections import Counter
import os,stat,json,hashlib,csv,io,ast,re,math,traceback
H=Path(__file__).parent;V=H.parent;P=V.parent

def need(v,m):
 if not bool(v):raise AssertionError(m)
def safe(p):
 need('v112'not in str(p).casefold(),'Lexical guard before FS');p=Path(p);need(p.is_absolute()and P in p.parents and '..'not in p.parts,'Bound project path');return p
def opened(p):
 p=safe(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for c in p.parts[1:-1]:n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 return n
def raw(p):
 p=safe(p);need(p.suffix in('.json','.md','.html','.py','.csv'),'Closed text only; no images/arrays/rawsource');n=opened(p);need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def checked(b):
 z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact text binding '+b['path']);return z
def get(b):return json.loads(checked(b))
def table_md(s):
 return[[c for c in line[1:-1].split('|')]for line in s.splitlines()if line.startswith('|')and not re.match(r'^\|[-:| ]+\|$',line)]
class Parse(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None;self.links=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  for key in('href','src'):
   if key in d:self.links.append(d[key])
  if t=='tr':self.row=[]
  if t in('td','th'):self.cell=''
 def handle_data(self,d):
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if t in('td','th')and self.cell is not None:self.row.append(self.cell);self.cell=None
  if t=='tr'and self.row is not None:self.rows.append(self.row);self.row=None
def links(base,s,ishtml):
 parser=Parse();parser.feed(s);urls=parser.links if ishtml else re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s)
 for url in urls:
  need('v112'not in url.casefold(),'Forbidden URL before filesystem');u=urlsplit(url);need(not u.scheme and not u.netloc and not u.path.startswith('/')and'\\'not in u.path,'Portable relative link only')
  if u.path:
   path=safe(os.path.normpath(str(base/unquote(u.path))));n=opened(path)
   try:need(stat.S_ISREG(os.fstat(n).st_mode)or stat.S_ISDIR(os.fstat(n).st_mode),'Target exists without symlinks')
   finally:os.close(n)
 return len(urls)
def pct(v):return '—'if v is None else '%.2f%%'%(100*v)
def close(a,b):return(a is None and(b is None or b==''))or(a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12))
def run():
 r=get(bind(V/'ROOT_RESULTS_KO.json'));complete=get(r['diagnostic_complete']);s=get(r['summary']);q=get(r['numeric_QA']);vis=get(r['visual_QA']);fig=get(r['figures_complete']);rb=get(bind(V/'saved_readout_root/REPORT_BUILD_COMPLETE.json'));evo=get(bind(V/'ROOT_FIGURE_CODE_EVOLUTION_AND_EXACT_EXECUTED_SNAPSHOT.json'))
 need(all(x['completed']and x['PASS']for x in(r,complete,s,q,vis,fig,rb,evo)),'All closed receipts PASS')
 need(complete['outputs']['SUMMARY.json']==r['summary']==q['summary']and q['diagnostic_complete']==r['diagnostic_complete'],'Exact diagnostic/summary/independent QA closure')
 need(r['code']==rb['code']==evo['current_saved_report_code'],'Current report builder binding');current=checked(evo['current_saved_report_code']).decode();snapshot=checked(evo['exact_executed_code_snapshot']).decode()
 need(fig['code']==evo['figure_code_at_execution_historical']and evo['exact_executed_code_snapshot']['sha256']==fig['code']['sha256'],'Figure historical executed SHA resolved by exact snapshot, not current source')
 def function(s,name):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
 need(ast.dump(function(current,'figures'),include_attributes=False)==ast.dump(function(snapshot,'figures'),include_attributes=False),'Figure function AST unchanged')
 need(evo['figure_code_binding_is_historical_not_current_source']and not evo['statistics_models_or_source_data_changed'],'Historical/current distinction preserved')
 text={Path(b['path']).name:checked(b).decode()for b in rb['outputs']};need(text['ROOT_REPORT_KO.md']==text['PORTABLE_REPORT_KO.md'],'MD and portable byteequal');md=text['PORTABLE_REPORT_KO.md'];html=Parse();html.feed(text['ROOT_READOUT_KO.html'])
 labels=['Factual','RGB127','No q8 XY','No q8','No native','No condition'];rows=[]
 for arm in r['models']:
  for pop in('POINT','WEAK'):
   for mode,label in zip(r['conditions'],labels):
    z=s['models'][arm][pop][mode];N,D,E=(6096,5853,243)if pop=='POINT'else(44,26,18)
    need((z['N'],z['D'],z['E'])==(N,D,E)and sum(map(sum,z['CM_D_E_by_D_E_U']))==N,'Separate source point/weak fixed denominators')
    rows.append([arm[:3],pop,label,str(z['CM_D_E_by_D_E_U']),pct(z['D_recall']),pct(z['E_recall']),pct(z['E_precision']),str(z['sign_flips']),str(z['U'])])
 need(len(rows)==24 and r['main_rows']==rows,'All24 root displayed values exact SUMMARY')
 need(table_md(md)[1:]==rows and html.rows[1:]==rows,'Every MD and HTML data cell exact')
 need(r['scalar_rows']==complete['scalar_rows']==q['checks']['scalar_rows']==73680 and q['checks']['all_CM_P_R_U_delta_quantile_transition_and_CSV_cells_recomputed'],'Independent allscalar numerical QA')
 full=list(csv.DictReader(io.StringIO(checked(complete['outputs']['FULL_AND_NATIVE8_METRICS.csv']).decode())));native=list(csv.DictReader(io.StringIO(checked(complete['outputs']['NATIVE8_METRICS.csv']).decode())))
 need(len(full)==24 and len(native)==192,'Full24/native192 export denominators')
 for row in full+native:
  z=s['models'][row['arm']][row['population']][row['condition']]
  if'native_slot'in row:z=z['by_native'][row['native_slot']]
  for k,v in z.items():
   if isinstance(v,(list,dict)):need(json.loads(row[k])==v,'Saved CSV structured cell '+k)
   else:need(close(v,row[k]),'Saved CSV scalar '+k)
 for pop in('POINT','WEAK'):
  z=s['models'][r['models'][1]][pop][r['conditions'][-1]];need(z['sign_flips']==0 and z['mean_abs_delta']>0,'C63 zero flips not zero contribution');need(abs(z['mean_abs_delta']-q['zero_all_conditions'][r['models'][1]][pop]['mean_abs_delta'])<1e-12,'C63 nonzero delta matches numericQA')
 for arm in r['models']:
  for pop in('POINT','WEAK'):
   z=s['models'][arm][pop][r['conditions'][1]];need(z['CM_D_E_by_D_E_U']==[[0,z['D'],0],[0,z['E'],0]],'RGB127 allknown E')
 need(s['all7040_point_metadata_retained']and s['point_unknown944_not_forwarded']and s['point_originalU64_not_forwarded']and'UNKNOWN944'in md and'inputU64'in md,'UNKNOWN944/U64 retained separately')
 need(vis['actual_fixed_source_cases_seen']==4 and vis['actual_views']==2 and vis['figure_complete']==r['figures_complete']and vis['actual_figures_viewed']==fig['figures'],'Actual root figures/montage receipt only, no own new views')
 need(vis['same_pair_pixel_comparison']['changed_pixels']==4977 and vis['same_pair_pixel_comparison']['total_pixels']==9216 and vis['same_pair_pixel_comparison']['qL_differ']is True and'원q/L가 서로 달라'in md,'Fixed D/E image examples not pure sameq pair')
 pair=q['pair_counts_from_saved_input_matching_flags'];need(pair['POINT']==dict(all_pairs=4570,exact_nonRGB_matched=2310,matched_different_RGB=462,matched_same_RGB=1848)and pair['WEAK']['all_pairs']==pair['WEAK']['exact_nonRGB_matched']==18,'Report paircounts exact closed QA')
 need(all(q['factual_matched_different_RGB_logit_dependence_counts'][arm]==dict(POINT=462,WEAK=18)for arm in r['models'])and'point2310/4570'in md and'weak18/18'in md,'Bounded matched RGB dependence counts')
 native0=s['models'][r['models'][0]]['POINT'][r['conditions'][0]]['by_native'];need(native0['6']['E']==2 and native0['7']['E']==4,'Known source E native6/7 sparsity')
 wn=s['models'][r['models'][0]]['WEAK'][r['conditions'][0]]['by_native'];need(all(wn[str(k)]['N']==0 for k in(5,6,7)),'Weak native5/6/7 missing preserved')
 counts={n:links(V,text[n],n.endswith('.html'))for n in('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html')}
 need(r['source_apparent_OOD_not_generalization']and r['training']==r['PnP']==r['protected_new_image_row_score_or_label_artifacts']==0 and r['selected_model']is None and not r['adopted']and not r['fully_solved'],'No new protected access/fit/selection/adoption')
 need('수학적으로0이라는 뜻은 아니다'in md and'OOD'in md and'실사 전이 실패의 원인을 확정하지 않는다'in md and'인간 native 의미나 실사 일반화의 증거가 아니다'in md and'validity와 고정 XY grid'in md,'Bounded mechanism interpretation')
 result=dict(completed=True,PASS=True,code=bind(__file__),root_report_build_complete=bind(V/'saved_readout_root/REPORT_BUILD_COMPLETE.json'),reports=rb['outputs'],diagnostic_complete=r['diagnostic_complete'],summary=r['summary'],numeric_QA=r['numeric_QA'],root_actual_visual_QA=r['visual_QA'],figure_code_evolution=bind(V/'ROOT_FIGURE_CODE_EVOLUTION_AND_EXACT_EXECUTED_SNAPSHOT.json'),exact_executed_figure_code_snapshot=evo['exact_executed_code_snapshot'],current_report_code=evo['current_saved_report_code'],figure_AST_unchanged=True,main24_MD_HTML_JSON_SUMMARY_rows_exact=True,full24_native192_CSV_cells_exact=True,point6096_D5853_E243_and_weak44_D26_E18_separate=True,all7040_UNKNOWN944_inputU64_retained_not_relabelled=True,C63_no_sign_flips_not_zero_logit_contribution=True,RGB127_all_E_ood_not_transfer_cause=True,matched_SOURCE_RGB_pair_counts_exact_to_independent_QA=True,native6_7_source_E_sparse_weak5_6_7_absent=True,portable_relative_link_occurrences=counts,all_link_targets_exist_nosymlink=True,root_actual_four_case_two_figure_receipt_bound=True,own_new_actual_image_views=0,material_errors=[],source_RGB_GT_NPZ_weights_protected_scores_new_forward_or_fit_reads_calls=0,HTML_browser_rendered=False,limits_KO=['완료된 SUMMARY/CSV/보고서와독립 saved-scalar QA를 대조했다. scalar원장73,680행이나 pair55,056행을 다시 재집계하지 않았다.','원 source/GT·checkpoint·NPZ·이미지·보호셋은 열지 않았다. 그림은 root가 실제 본 receipt를 인용하며 본인 새toolview는0이다.','같은SOURCE에서의 RGB계산의존과 OOD민감도이며 human/nativegold·보호실사 전이원인·모델/문턱/학습선택·채택/해결로확대하지 않는다.','그림 실행당시99905b… code는 정확 snapshot으로 검증했다. 현재a79b80… 보고서코드가 그시점에 실행됐다고 주장하지 않는다.'],frozen_after_QA=True)
 with(H/'REVIEW.json').open('x',encoding='utf8')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),link_counts=counts),ensure_ascii=False))
if __name__=='__main__':
 try:run()
 except BaseException:
  with(H/'FAILURE.json').open('x')as f:json.dump(dict(completed=False,PASS=False,traceback=traceback.format_exc()),f,indent=2);f.write('\n')
  raise
