"""Independent final report factual/table/link QA; generated chart viewed separately."""
import json,hashlib,re,os,stat
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from datetime import datetime,timezone
V=Path(__file__).absolute().parent.parent;H=V/'final_scientific_review_c';checks=0;accesses={}
def need(v,m):
 global checks
 checks+=1
 if not v:raise AssertionError(m)
def rb(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'safe before access')
 need(p.suffix.casefold()not in('.npz','.npy','.pt','.pth','.safetensors','.bin'),'No scientific binary')
 for q in(p,*p.parents):need(not q.is_symlink(),'no symlinks')
 z=p.read_bytes();accesses[str(p)]={'sha256':hashlib.sha256(z).hexdigest(),'size_bytes':len(z)};return z
def bind(p):z=rb(p);return {'path':str(p),'sha256':hashlib.sha256(z).hexdigest()}
def jd(p):return json.loads(rb(p))
ready=jd(V/'report_b/FINAL_READY.json');need(bind(V/'report_b/FINAL_READY.json')['sha256']=='4cd99c2a78e772e9145fd96067fe4580e63716a104a8707a443cf048b09ccfd2','ExactREADY')
for b in ready['report_ROOT_artifacts'].values():need(bind(Path(b['path']))==b,'Report exactfrozenbinding')
for b in ready['figure'].values():need(bind(Path(b['path']))==b,'Figure exactbinding')
result=jd(V/'ROOT_RESULTS.json');aud=jd(V/'independent_saved_fit_review_c/actual_review_v1/REVIEW.json');own=jd(V/'independent_saved_fit_review_c/actual_review_v1/SOURCE_GROUPED_DIRECT_READOUT.json');need(bind(V/'independent_saved_fit_review_c/actual_review_v1/REVIEW.json')['sha256']=='7def791a9f153bc8491825a3af7406e1871718e97f9354a7411b2c536d3e168f','Actual independent audit')
need(result['cohorts']==own['cohorts']and result['triplets']==own['triplets'],'Allreport cohort/domain/native/authority/paired/UNKNOWN/triplet cells exactly independent readout')
need(result['source_fit_complete']==aud['fit_complete']and result['source_evaluation_complete']==aud['source_evaluation_complete']and result['independent_actual_fit_audit']['binding']==ready['independent_actual_fit_audit'],'Report provenance')
need(result['architecture_parameters_each']==118359 and result['updates_each']==4000 and result['total_updates']==8000 and result['runtime_seconds']==382.95 and result['original_rows']==3144 and result['original_inputU']==112 and result['target_UNKNOWN']==2772 and result['full_source_rows_each']==73976,'Declared counts/config')
need(result['model_adopted']is False and result['fully_solved']is False and result['new_forward_fit_or_input_array_reads_for_report']==0,'Unsolved/noadoption/no newrun')
md=rb(V/'PORTABLE_REPORT_KO.md').decode();html=rb(V/'ROOT_READOUT_KO.html').decode();arms=result['arms']
rows=[[s.strip()for s in line.strip().strip('|').split('|')]for line in md.splitlines()if line.startswith('|')]
def cm(x):return json.dumps(x['CM_D_E_by_D_E_U'],separators=(',',':'))
def pct(x):return '—'if x is None else f'{100*x:.2f}%'
expected=[]
for role in('FULL','TRAIN','HOLDOUT'):
 for arm in arms:
  x=own['cohorts'][role]['all']['models'][arm];expected.append([role,arm[:3],f"{x['D']}/{x['E']}",cm(x),pct(x['D_precision']),pct(x['E_precision']),pct(x['D_recall']),pct(x['E_recall']),str(x['known_prediction_U'])])
for role in('FULL','TRAIN','HOLDOUT'):
 x=own['cohorts'][role]['all']['paired_correctness'];t=own['triplets'][role];expected.append([role,str(x['known']),str(x['recovered_by_XY']),str(x['damaged_by_XY']),str(x['net_correct_change']),str(t['known_D_E_D_triplets']),str(t['models'][arms[0]]['all3_correct']),str(t['models'][arms[1]]['all3_correct'])])
for role in('TRAIN','HOLDOUT'):
 for domain,x in own['cohorts'][role]['by_domain'].items():
  for arm in arms:
   m=x['models'][arm];expected.append([role,domain,arm[:3],str(m['known']),cm(m),pct(m['D_recall']),pct(m['E_recall'])])
for k in range(8):
 a,b=[own['cohorts']['HOLDOUT']['by_native'][str(k)]['models'][arm]for arm in arms];expected.append([str(k),f"{a['D']}/{a['E']}",cm(a),cm(b),pct(a['E_recall']),pct(b['E_recall']),str(a['target_UNKNOWN'])])
for row in expected:need(row in rows,'Rendered table exact '+str(row[:3]))
need(len(expected)==25,'All25 rendereddata rows checked')
class HTML(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.pre=False;self.pretext=''
 def handle_starttag(self,tag,attrs):
  if tag=='pre':self.pre=True
  self.links.extend(v for k,v in attrs if k in('href','src')and v)
 def handle_endtag(self,tag):
  if tag=='pre':self.pre=False
 def handle_data(self,data):
  if self.pre:self.pretext+=data
parser=HTML();parser.feed(html);need(parser.pretext==md,'HTML embeds originalMarkdown verbatim')
mdrefs=[a or b for a,b in re.findall(r'!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)',md)]
links=[]
for doc,refs in [('PORTABLE_REPORT_KO.md',mdrefs),('ROOT_READOUT_KO.html',parser.links)]:
 for r in refs:
  u=urlsplit(r);need(not u.scheme and not u.netloc and not u.path.startswith('/'),'Relative local reportlink')
  p=V/unquote(u.path)
  need('..'not in p.parts and'v112'not in str(p).casefold(),'Finite ownphase link')
  for q in(p,*p.parents):need(not q.is_symlink(),'No linksymlink')
  need(p.is_file(),'Link exists')
  need(p.suffix.lower()not in('.npz','.pt','.pth','.npy'),'Report does not directly link model/arrays')
  need(p.stat().st_size<=2*1024**2,'Directreporttarget under2MiB')
  links.append({'page':doc,'reference':r})
need(len(links)==17,'11Markdown+6actualHTMLattribute links')
# Six bases/15 known source attempt IDs from already-generated saved text, not targets or pixels.
ledger=[json.loads(line)for line in rb(Path(result['saved_source_bindings']['ledger']['path'])).splitlines()if line.strip()]
held=[r for r in ledger if r['role']=='HOLDOUT'and r['direct_weak_target']>=0];attempts={r['attempt_id']for r in held};bases={int(re.search(r'BASE(\d+)',r['attempt_id']).group(1))for r in held}
need(len(held)==45 and len(attempts)==15 and len(bases)==6 and all(r['domain']=='SYNTH_FINITE_END'for r in held),'H45=15triplets/6syntheticbases/real0')
need(result['HOLDOUT_known_synthetic_bases']==6 and result['HOLDOUT_known_triplets']==15 and result['HOLDOUT_real_known']==0,'JSONscope exact')
for role in('FULL','TRAIN','HOLDOUT'):
 for arm in arms:
  x=own['cohorts'][role]['all']['models'][arm];need(sum(sum(r)for r in x['CM_D_E_by_D_E_U'])==x['known']and x['known']+x['target_UNKNOWN']==x['rows']and sum(x['UNKNOWN_target_prediction_counts'].values())==x['target_UNKNOWN'],'Denominator/unknown arithmetic')
# Caption meaning was read directly; these are factual confirmations, not lexical acceptance guards.
semantic=dict(E_gain_with_equal_D_loss_net0_not_overall_improvement=True,not_natural_real_validation=True,source_HOLD_exposed_synthetic_only6bases15triplets=True,public_real_weak_TRAIN_separate=True,weak_not_physical_nativegold=True,unknown_inputU_not_solved=True,nominal_fit_exclusion_preserves_original_roles=True,V199_recipe_difference_no_q8_causality_claim=True,no_model_checkpoint_threshold_arm_selection_both_to_V203=True,independent_audit_limits_not_full_neural_training_reexecution=True,v1_not_actualrun_v2partialUNKNOWN_preserved=True)
review=dict(completed=True,PASS=True,created_utc=datetime.now(timezone.utc).isoformat(),review_kind='Independent saved-final-report numeric/table/link and one actualchart view; no new fit audit',code=bind(Path(__file__).absolute()),report_READY=bind(V/'report_b/FINAL_READY.json'),report_artifacts=ready['report_ROOT_artifacts'],figures=ready['figure'],prior_actual_independent_fit_audit=ready['independent_actual_fit_audit'],all_saved_report_cohort_and_triplet_values_equal_independent_audit=True,rendered_MD_table_rows25_exact=True,HTML_pre_verbatim_MD=True,relative_MD_links=len(mdrefs),relative_HTML_attribute_links=len(parser.links),all_relative_links17_exist_and_under2MiB=True,links=links,scope_counts={'H_known45':True,'H_triplets15':True,'H_synthetic_bases6':True,'H_real_known0':True,'physical_attempt_native1160':True,'complete992':True,'partial_UNKNOWN168':True},semantic_review=semantic,actual_C_chart_view={'count':1,'path':str(V/'report_b/SOURCE_DIRECT_CM_AND_PAIRED.png'),'sha256':'595ecdde1f62514f9c4436c56a023f429a937739c42c74863d8ba6d876e33ff1','original_size':'2790x1260','tool_display':'2048x925','observations':['All6 CM panels and weakD/E row versus D/E/U prediction axes readable, numerical labels including HOLD C60[28,2;11,4]andC61[26,4;9,6] agree.','Paired row recovery/damage3/3 labels readable; TRAIN109/109 and HOLD3/15vs4/15 DED bars clearly annotated.','Title/subtitle/native+same-mask/118359 and sourceweak-notnativegold/footer45known15triplets6syntheticbases/real0/noadoption readable; no obvious clipping or overlap.'],'browser_HTML_rendered':False,'source_RGB_or_crop_viewed':False},actual_NPZ_headers_arrays_weights_models_originalpixels_or_new_metrics=0,source_or_report_files_modified=False,new_model_selection_or_adoption=False,goal_solved=False,checks=checks,accesses=accesses,limits=['This closes savedreport facts/links and onegenerated scalar chart only; no neural training rerun, model forward, original pixel review or natural-real performance claim.','Underlying source-only numeric audit is the prior closed7def791a receipt; its limitations remain.','No wholephase freeze or publication action performed by this review.'])
z=(json.dumps(review,ensure_ascii=False,indent=2)+'\n').encode()
with(H/'REVIEW.json').open('xb')as f:f.write(z)
print(json.dumps({'PASS':True,'checks':checks,'review_sha256':hashlib.sha256(z).hexdigest(),'links':len(links),'table_rows':len(expected)},ensure_ascii=False))
