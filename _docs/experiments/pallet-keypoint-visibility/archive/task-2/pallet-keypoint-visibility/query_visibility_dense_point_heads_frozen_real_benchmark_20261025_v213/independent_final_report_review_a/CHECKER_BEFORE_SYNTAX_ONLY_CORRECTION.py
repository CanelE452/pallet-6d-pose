"""Independent closed-report/table/link QA. No tensor, image, score-ledger or GT reads."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import os,stat,json,hashlib,csv,io,re,math,ast,traceback
H=Path(__file__).parent;V=H.parent;P=V.parent
allowed={'.json','.jsonl','.md','.html','.csv','.py'}
def need(v,m):
 if not bool(v):raise AssertionError(m)
def lexical(p):
 need('v112'not in str(p).casefold(),'Forbidden lexical path before access');p=Path(p);need(p.is_absolute()and P in p.parents and '..'not in p.parts,'Exact project path');return p
def opened(p):
 p=lexical(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for c in p.parts[1:-1]:n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 return n
def raw(p):
 p=lexical(p);need(p.suffix in allowed,'Closed text only; no image/array/model');n=opened(p);need(stat.S_ISREG(os.fstat(n).st_mode),'Regular text')
 with os.fdopen(n,'rb')as f:return f.read()
def b(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def checked(binding):
 z=raw(binding['path']);need(hashlib.sha256(z).hexdigest()==binding['sha256'],'Exact closed binding '+binding['path']);return z
def get(binding):return json.loads(checked(binding))
def csvread(binding):return list(csv.DictReader(io.StringIO(checked(binding).decode())))
def equal(a,b_,name):need(a==b_,name)
def metric_equal(r,q):
 for k,kk in [('N','N'),('D','D_N'),('E','E_N'),('CM_D_E_by_D_E_U','CM_D_E_by_D_E_U'),('D_precision','D_precision'),('D_recall','D_recall'),('E_precision','E_precision'),('E_recall','E_recall'),('U','U'),('U_rate','U_rate')]:equal(r[k],q[kk],'Independent numeric review primary '+k)
class Tables(HTMLParser):
 def __init__(self):super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None;self.refs=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  for k in('href','src'):
   if k in d:self.refs.append(d[k])
  if t=='table':self.table=[]
  if t=='tr'and self.table is not None:self.row=[]
  if t in('td','th')and self.row is not None:self.cell=''
 def handle_data(self,d):
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if t in('td','th')and self.cell is not None:self.row.append(self.cell);self.cell=None
  if t=='tr'and self.row is not None:self.table.append(self.row);self.row=None
  if t=='table'and self.table is not None:self.tables.append(self.table);self.table=None
def md_tables(text):
 out=[];rows=[]
 for line in text.splitlines()+['']:
  if line.startswith('|')and line.endswith('|'):
   vals=line[1:-1].split('|')
   if all(v and set(v)<=set('-: ')for v in vals):continue
   rows.append(vals)
  elif rows:out.append(rows);rows=[]
 return out
def expected_table(rows):return[list(rows[0])]+[[str(r[k])for k in rows[0]]for r in rows]
def refs_md(text):return re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
def linkcheck(base,url):
 need('v112'not in url.casefold(),'Forbidden lexical URL before filesystem');u=urlsplit(url)
 if u.scheme or u.netloc:need(u.scheme in('https','http','mailto'),'No file/data absolute portable link');return 'external'
 if not u.path:return 'anchor'
 path=unquote(u.path);need(not path.startswith('/')and '\\'not in path,'Relative portable path only')
 target=lexical(os.path.normpath(str(base/path)));n=opened(target)
 try:need(stat.S_ISREG(os.fstat(n).st_mode)or stat.S_ISDIR(os.fstat(n).st_mode),'Existing ordinary target')
 finally:os.close(n)
 return 'local'
def run():
 build=get(b(V/'report_preparation_b/ROOT_REPORT_BUILD_COMPLETE.json'));need(build['completed']and build['PASS'],'Closed actual root report')
 files={n:checked(x)for n,x in build['outputs'].items()};root=json.loads(files['ROOT_RESULTS_KO.json']);equal(files['ROOT_REPORT_KO.md'],files['PORTABLE_REPORT_KO.md'],'Portable MD byteexact')
 for k in('code','protocol'):equal(root[k],build[k],'Root builder/protocol binding');checked(build[k])
 code=raw(build['code']['path']).decode();ast.parse(code);need('source procedural point'in' '.join(root['limits'])and root['local_crop_cause_or_physical_native_gold_claim']is False,'Distinct weak/source authority')
 gal=get(root['gallery_complete']);data=get(gal['summary']);qa=get(root['independent_numeric_review']);ev=get(root['evaluation_complete'])
 need(all(x['completed']and x['PASS']for x in[gal,data,qa,ev]),'Closed dependencies PASS');equal(qa['evaluation_complete'],root['evaluation_complete'],'Independent numeric exact evaluation');equal(gal['evaluation_complete'],root['evaluation_complete'],'Gallery exact evaluation')
 equal(root['all_models14'],ev['models'],'All14 fixed models');equal(len(root['all_models14']),14,'All14');equal(root['table_primary42'],data['primary42'],'Root all42 primary rows');equal(root['table_SELF28'],data['self28'],'Root all28 SELF rows');equal(root['table_public_weak13'],data['public_weak13'],'Root public13');equal(root['supplementary'],data['supplementary'],'All supplementary tables preserved')
 for r in root['table_primary42']:metric_equal(r,qa['current']['primary'][r['group']][r['model']]);need(sum(map(sum,r['CM_D_E_by_D_E_U']))==r['N']and sum(r['CM_D_E_by_D_E_U'][0])==r['D']and sum(r['CM_D_E_by_D_E_U'][1])==r['E'],'Primary D/E/U conservation')
 for group,n,d,e in[('ALL',1237,1147,90),('WOOD_PRIMARY105',645,616,29),('SQUARE118',592,531,61)]:
  rr=[r for r in root['table_primary42']if r['group']==group];need(len(rr)==14 and all((r['N'],r['D'],r['E'])==(n,d,e)for r in rr),'Fixed full/material primary denominator')
 current=get(ev['outputs']['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json']);public=get(ev['outputs']['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json']);t=current['tables'];sup=root['supplementary']
 for key,src in [('native8','primary_native8_D_E_with_U'),('type_U_inventory','all1784_model_inventory'),('manual_near_D','manual_near_D'),('coordinate_error_bins','coordinate_error_bins'),('paired_recovery_damage','paired_recovery_and_damage')]:equal(sup[key],t[src],'Closed saved table unchanged '+key)
 equal(root['human_state_inventory'],current['state_inventory'],'Human inventory unchanged')
 for r in root['table_SELF28']:
  rr=[x for x in t['human_self_binary_response']if x['cohort']==r['cohort']and x['model']==r['model']];need(len(rr)==2,'Two material SELF rows')
  expected=21 if r['cohort']=='paired_same_query_RGB'else136;equal(r['N'],expected,'SELF21/136 separate')
  for dst,src in[('N','human_S_support'),('binary_hidden_response','binary_hidden_response'),('visible_response','visible_response'),('U','unresolved')]:equal(r[dst],sum(int(x[src])for x in rr),'SELF all response cells')
  need(r['binary_hidden_response']+r['visible_response']+r['U']==r['N']and r['U_rate']==r['U']/r['N']and not r['learned_self_or_cause'],'SELF response conservation not cause')
 for r in sup['reference_association']:
  if r['association']=='HUMAN_D_EXISTING_REFERENCE_LE3':equal({k:r[k]for k in('N','D','E','U')},qa['current']['all_registered_humanD_existing_reference_LE3_association'][r['model']],'Reference461 exact independent census')
  else:need(r['association']=='HUMAN_D_MISSING_REFERENCE'and r['N']==r['D']==r['E']==r['U']==0,'Missing reference explicit zero')
 for m in root['all_models14']:
  need(sum(x['N']for x in sup['manual_near_D']if x['model']==m)==376,'Manual376 separate')
  rr=[x for x in sup['native8']if x['group']=='ALL'and x['model']==m];need(len(rr)==8 and sum(x['N']for x in rr)==1237,'Native8 full denominator')
 for r in root['table_public_weak13']:
  z=public['cohorts']['FULL360']['ALL']['models'][r['model']];per=z['binary_T_nonT']['per_class'];f=z.get('owner3',{}).get('per_class',{}).get('FOREIGN_VISIBLE_SUPPORTED',{})
  equal((r['all_original_rows'],r['known_center_truth'],r['unknown_center_truth'],r['original_input_U']),(360,204,156,108),'Public weak denominator')
  for k,v in [('T_precision_known',per['T']['precision_on_known_truth']),('T_recall_including_U',per['T']['recall_including_prediction_U']),('nonT_recall_including_U',per['NON_T']['recall_including_prediction_U']),('ownerF_precision_known',f.get('precision_on_known_truth')),('ownerF_recall_including_U',f.get('recall_including_prediction_U'))]:equal(r[k],v,'Public weak scalar '+k)
 need(len(root['table_public_weak13'])==13 and 'legacy'not in{r['model']for r in root['table_public_weak13']}and public['legacy_baseline']['status']=='MISSING_IN_REVIEWED_LINEAGE','Public legacy missing')
 equal(root['original_input_U'],data['original_input_U'],'Input U metadata unchanged');need(root['original_input_U']['current_U']==45 and root['original_input_U']['current_N']==1784 and root['original_input_U']['current_U_rate']==45/1784,'Input U denominator distinct')
 refs=get(gal['manifest']);im={x['original_global_index']:x for x in refs['images']};viewids=[]
 for binding in root['actual_visual_review_bindings']:
  r=get(binding);need(r['completed']and r['PASS']and r['actual_view_count']==6,'Actual six receipt PASS');equal(r['image_manifest'],gal['manifest'],'Receipt manifest binding')
  equal([x['original_global_index']for x in r['actually_viewed_images']],r['actual_viewed_ids'],'Actual receipt image IDs')
  for x in r['actually_viewed_images']:need(x['actual_tool_image_seen']and x['image']==im[x['original_global_index']]['card'],'Actual image binding matches manifest')
  viewids+=r['actual_viewed_ids']
 need(len(viewids)==len(set(viewids))==12 and sorted(viewids)==root['fixed12_ids']==gal['exact_original_ids'],'Root+A actual6+6 disjoint union12')
 tables=[root['table_primary42'][:14],root['table_primary42'][14:],root['table_SELF28'],sup['reference_association'],root['table_public_weak13']];expected=[expected_table(x)for x in tables]
 for name in('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md'):equal(md_tables(files[name].decode()),expected,'Every Markdown displayed table cell '+name)
 html=Tables();html.feed(files['ROOT_READOUT_KO.html'].decode());equal(html.tables,expected,'Every HTML displayed table cell')
 counts={};targets=set()
 for name in('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'):
  s=files[name].decode();parser=Tables();parser.feed(s);links=parser.refs if name.endswith('.html')else refs_md(s);counts[name]=Counter(linkcheck(V,u)for u in links);targets.update(links)
 for name in('PORTABLE_GALLERY_KO.md','index.html'):
  path=V/'fixed12_dense_point_real_visual_v1'/name;s=raw(path).decode();parser=Tables();parser.feed(s);links=parser.refs if name.endswith('.html')else refs_md(s);counts['gallery/'+name]=Counter(linkcheck(path.parent,u)for u in links)
 for x in root['links']:linkcheck(V,x['relative'])
 need(sum(sum(v.values())for k,v in counts.items()if not k.startswith('gallery/'))==build['relative_link_occurrences'],'Root link occurrences independent')
 need(root['primary_a_priori']=='C63'and root['ablation_a_priori']=='C62'and root['source_results_selected_arm']is False and root['adopted']is False and root['fully_solved']is False and root['selected_model']is None,'No result-guided selection/adoption/goal claim')
 need(root['repeated_exposure_not_freshblind']and root['legacy_realCAL_history_preserved']and root['v61_presence_proxy']and root['zero_U_is_not_confidence']and root['reference461_not_manual376'],'Necessary limitations preserved')
 result=dict(completed=True,PASS=True,code=b(__file__),report_build_complete=b(V/'report_preparation_b/ROOT_REPORT_BUILD_COMPLETE.json'),reports=build['outputs'],builder=build['code'],builder_protocol=build['protocol'],gallery_complete=root['gallery_complete'],independent_numeric_review=root['independent_numeric_review'],actual_visual_review_bindings=root['actual_visual_review_bindings'],checked_primary_rows=42,checked_model_count=14,primary_N_D_E=[1237,1147,90],material_denominators={'WOOD':[645,616,29],'SQUARE':[592,531,61]},checked_SELF_rows=28,SELF_paired21_and_all136_separate=True,reference461_manual376_separate=True,native8_type_U_and_paired_tables_closed_saved_exact=True,public13_centerweak_only_and_legacy_MISSING=True,all_displayed_MD_HTML_table_cells_exact=True,actual_root_A_visual_union_ids=sorted(viewids),own_actual_viewed_ids=[1,3,4,5,7,8],relative_link_counts={k:dict(v)for k,v in counts.items()},all_local_targets_exist_without_symlink=True,portable_absolute_home_or_file_links=0,source_procedural_point_publicweak_and_human_native_authority_separate=True,material_errors=[],original_RGB_NPZ_ledger_weight_model_or_GT_reads=0,new_fit_forward_or_label_changes=0,HTML_browser_rendered=False,limits_KO=['완료된 saved summary/table/숫자감사 receipt에 대한 독립 보고서 검토다. 원GT·원장·예측NPZ·가중치·이미지를 다시 열지 않았다.','표/링크 QA가 실제 HTML browser 렌더나 전체 영상 검사를 뜻하지 않는다. 본인 실제 view는 앞서 봉인한6개이며 root의 별도6과 합해12다.','source procedural point 및 공개 weak feature는 human native gold가 아니며 localcrop으로 물리 코너/선택instance/가림원인을 확정하지 않는다. 반복 노출·비대표 고정 사례·미채택/미해결 한계를 유지한다.'],frozen_after_QA=True)
 with(H/'FINAL_REPORT_QA.json').open('x',encoding='utf8')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,receipt=b(H/'FINAL_REPORT_QA.json'),relative_link_counts=result['relative_link_counts']),ensure_ascii=False))
if __name__=='__main__':
 try:run()
 except BaseException:
  with(H/'FAILURE.json').open('x')as f:json.dump(dict(completed=False,PASS=False,traceback=traceback.format_exc()),f,indent=2);f.write('\n')
  raise
