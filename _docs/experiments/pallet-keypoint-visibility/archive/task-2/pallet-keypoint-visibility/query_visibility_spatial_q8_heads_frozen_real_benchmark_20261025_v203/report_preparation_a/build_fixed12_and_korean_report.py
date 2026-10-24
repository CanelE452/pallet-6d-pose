"""V203 closed-result Korean report + fixed12 saved-PNG gallery.
CODEONLY prepare: only frozen scripts/protocols and invented CPU arrays.
Actual modes require separate root entries after both real prediction seal.
Never loads NPZ, checkpoint, SAM, raw-source RGB or an operational model.
"""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
import argparse,csv,hashlib,html,json,math,os,re,stat,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_FROZEN_REAL_BENCHMARK_SCOPE.json'
OLD=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200'
OLD_READY=OLD/'fixed_visual_preparation_a/FINAL_READY.json'
EVAL_READY=V/'evaluation_preparation_a/FINAL_READY.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_RESULTS_OR_FIXED_PNG_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'fixed12_spatial_q8_real_visual_v1'
MODELS=['v61','legacy','C52','C53','C54','C55','C56','C57','C58','C59','C60','C61']
DIRECT=MODELS[4:]
FIXED_IDS=[0,1,2,3,4,5,7,8,256,300,840,996]
ROOT6=[0,2,256,300,840,996];PRIOR_A6=[1,3,4,5,7,8]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def require(x,m):
 if not bool(x):raise AssertionError(m)

def safe(path):
 p=Path(path).absolute();require(P in p.parents and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Explicit project path before access');return p

def opened(path,create=False):
 p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
  flags=(os.O_WRONLY|os.O_CREAT|os.O_EXCL) if create else os.O_RDONLY
  last=os.open(p.name,flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 require(stat.S_ISREG(os.fstat(last).st_mode),'No symlink/regular file');return os.fdopen(last,'wb' if create else 'rb')

def sha(path):
 h=hashlib.sha256()
 with opened(path) as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def bind(p):return {'path':str(safe(p)),'sha256':sha(p)}

def checked(b):require(bind(b['path'])=={k:b[k] for k in ['path','sha256']},'Bound artifact changed');return safe(b['path'])

def read(p):
 with opened(p)as f:return json.load(f)

def lines(p):
 with opened(p)as f:return [json.loads(s) for s in f if s.strip()]

def text(p,s):
 with opened(p,True)as f:f.write(s.encode())

def write(p,d):text(p,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csvfile(p,rows):
 if not rows:text(p,'EMPTY\n');return
 import io
 f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);text(p,f.getvalue())

def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def yes(x):return x is True or str(x).lower()=='true'

def number(x):
 try:v=float(x)
 except(TypeError,ValueError):return None
 return v if math.isfinite(v) else None

def fmt(x):
 v=number(x);return 'UNKNOWN' if v is None else '%.6g'%v

def ratio(a,b):return a/b if b else None

def material(r):
 require(r['group'] in ['WOOD_PRIMARY105','SQUARE118'],'Original material groups')
 return 'WOOD' if r['group']=='WOOD_PRIMARY105' else 'SQUARE'

def primary(r):return yes(r['paired_same_query_RGB_D_vs_E']) and r['human_state'] in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']

def zoom_position(q,scale):return (q+.5)*scale-.5

def render_card(r,rgb,valid):
 require(rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and valid.dtype==bool and valid.shape==(96,96),'Exact original crop')
 im=Image.new('RGB',(1600,1440),'white');d=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n)
 def label(x,y,s,n=22):d.text((x,y),s,font=font(n),fill='black')
 label(24,16,'V203 | original %04d | %s | native %d | %s'%(r['global_index'],r['id'],r['native_corner'],material(r)),27)
 label(24,56,'Human: %s | mapping: %s | primary-paired: %s'%(r['human_state'],r['mapping_status'],primary(r)),22)
 label(24,90,'Initial q=(%s,%s), L=%s | input available=%s | existing error=%s px'%(fmt(r['query_x']),fmt(r['query_y']),fmt(r['roi_side']),r['query_available'],fmt(r['initial_reference_error_px'])),22)
 label(24,124,'Reference=(%s,%s), authority=%s; numeric proxy only, no GT crop/marker'%(fmt(r['human_reference_x']),fmt(r['human_reference_y']),str(r['annotation_coordinate_source'])[:55]),20)
 raw=Image.fromarray(rgb,'RGB');large=raw.resize((576,576),Image.Resampling.NEAREST);ld=ImageDraw.Draw(large);q=zoom_position(47.5,6)
 ld.line([(q-16,q),(q+16,q)],fill=(0,255,255),width=2);ld.line([(q,q-16),(q,q+16)],fill=(0,255,255),width=2)
 im.paste(large,(24,202));im.paste(Image.fromarray(valid.astype(np.uint8)*255).convert('RGB').resize((288,288),Image.Resampling.NEAREST),(640,202));im.paste(raw,(980,202))
 label(24,170,'Nearest x6 DISPLAY copy; cyan initial q=47.5',21);label(640,170,'Original valid96 x3',21);label(980,170,'Clean96 (exact)',21)
 d.rectangle([640,202,927,489],outline=(150,150,150),width=1)
 label(640,504,'Black validity = invalid/padding, not background.',19)
 xs=[640,790,885,1120];label(xs[0],550,'Model',23);label(xs[1],550,'D/E/U',23);label(xs[2],550,'raw direct',23);label(xs[3],550,'Fixed route',23)
 for i,m in enumerate(MODELS):
  y=594+i*47;label(xs[0],y,m,24);label(xs[1],y,r[m+'_D_E_U'],24)
  label(xs[2],y,fmt(r.get(m+'_raw_direct_end_external_logit')) if m in DIRECT else '--',22)
  label(xs[3],y,'direct raw >=0' if m in DIRECT else ('regional E-first' if m in ['C52','C53'] else 'original saved'),21)
 notes=['Exact clean96 file remains unchanged. Cross is display-only; input q/L/native were never moved.',
 'D/E/S human semantics and point/part location differ. A local crop alone cannot establish selected-instance or true cause.',
 'Error <=3px is existing reference association, not physical coordinate gold. Repeated real-set exposure: not blind.',
 'Fixed post-result strata are descriptive, not random prevalence samples or a training/model/threshold selection.']
 for i,s in enumerate(notes):label(24,1200+i*39,s,21)
 require(np.array_equal(np.asarray(im)[202:298,980:1076],rgb),'Native clean96 ROI exact in card')
 return im

def primary_tables(current):
 metric=[r for r in current['tables']['metrics'] if r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E']
 require(len(metric)==24 and {r['model'] for r in metric}==set(MODELS),'All12 x2 material primary metrics')
 result=[]
 for group in ['ALL','WOOD_PRIMARY105','SQUARE118']:
  for m in MODELS:
   rr=[r for r in metric if r['model']==m and (group=='ALL' or r['group']==group)];require(len(rr)==(2 if group=='ALL' else 1),'Fixed model/material rows')
   c={k:sum(int(r[k]) for r in rr) for k in ['N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']}
   du=c['U_visible']+c['out_of_task_S_visible']+c['other_unresolved_visible'];eu=c['U_occluded']+c['out_of_task_S_occluded']+c['other_unresolved_occluded']
   require(c['TP']+c['FN_decided']+du==c['visible_support'] and c['FP']+c['TN']+eu==c['occluded_support'],'Original known denominators incl U')
   result.append(dict(group=group,model=m,N=c['N'],D_support=c['visible_support'],E_support=c['occluded_support'],D_D=c['TP'],D_E=c['FN_decided'],D_U=du,E_D=c['FP'],E_E=c['TN'],E_U=eu,D_precision=ratio(c['TP'],c['TP']+c['FP']),D_recall=ratio(c['TP'],c['visible_support']),E_precision=ratio(c['TN'],c['TN']+c['FN_decided']),E_recall=ratio(c['TN'],c['occluded_support']),U=du+eu))
 return result

def public_table(public):
 rows=[]
 for model,metrics in public['cohorts']['FULL360']['ALL']['models'].items():
  binary=metrics['binary_T_nonT']['per_class'];owner=metrics.get('owner3',{}).get('per_class',{})
  t=binary['T'];non=binary['NON_T'];f=owner.get('FOREIGN_VISIBLE_SUPPORTED',{})
  rows.append(dict(model=model,all_original_rows=public['rows'],known_center_truth=public['coverage']['known_center204'],unknown_center_truth=public['coverage']['unknown_center156'],original_input_U=public['coverage']['original_input_U'],T_precision_known=t['precision_on_known_truth'],T_recall_including_U=t['recall_including_prediction_U'],nonT_recall_including_U=non['recall_including_prediction_U'],ownerF_precision_known=f.get('precision_on_known_truth'),ownerF_recall_including_U=f.get('recall_including_prediction_U')))
 require({r['model'] for r in rows}==set(MODELS)-{'legacy'},'Public weak comparison all eleven available owner routes, legacy missing')
 return rows

def table_html(rows):
 if not rows:return '<p>EMPTY (no replacement)</p>'
 cols=list(rows[0]);return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in cols)+'</tr>' for r in rows)+'</tbody></table>'

def table_md(rows):
 if not rows:return 'EMPTY (no replacement)\n'
 cols=list(rows[0]);s='|'+'|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'
 for r in rows:s+='|'+'|'.join(str(r[k]).replace('|','/') for k in cols)+'|\n'
 return s

class LocalRefs(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ['href','src'] and v)

def local_links(text,html_mode=False):
 if html_mode:
  parser=LocalRefs();parser.feed(text);return parser.refs
 return re.findall(r'\]\(([^)]+)\)',text)

def fixed_rows(rows):
 require(len(rows)==1784 and [r['global_index']for r in rows]==list(range(1784)),'All original1784 ledger identity/order')
 for r in rows:require(all(r[m+'_D_E_U']in ('D','E','U')for m in MODELS),'Every saved twelve-model response exists')
 require(sum(primary(r)for r in rows)==1237 and sum(primary(r)and r['human_state']=='DIRECT_VISIBLE'for r in rows)==1147 and sum(primary(r)and r['human_state']=='EXTERNAL_OCCLUDED'for r in rows)==90,'Fixed primary1237 D1147 E90')
 require(sum(not yes(r['query_available'])for r in rows)==45,'Original U45 preserved')
 return [rows[i]for i in FIXED_IDS]

def self_tables(current):
 rows=current['tables']['human_self_binary_response'];out=[]
 for cohort,expected in [('paired_same_query_RGB',21),('same_slot_pipeline',136)]:
  for m in MODELS:
   rr=[r for r in rows if r['cohort']==cohort and r['model']==m];require(len(rr)==2,'Both original material SELF rows')
   fields=('human_S_support','binary_hidden_response','visible_response','unresolved');d={k:sum(int(r[k])for r in rr)for k in fields}
   require(d['human_S_support']==expected and d['binary_hidden_response']+d['visible_response']+d['unresolved']==expected,'Separate SELF cohort support and response conservation')
   out.append(dict(cohort=cohort,model=m,N=expected,binary_hidden_response=d['binary_hidden_response'],visible_response=d['visible_response'],U=d['unresolved'],U_rate=ratio(d['unresolved'],expected),learned_self_or_cause=False))
 return out

def pretty_primary(pt):
 return [dict(group=r['group'],model=r['model'],N=r['N'],D=r['D_support'],E=r['E_support'],CM_D_E_by_D_E_U=[[r['D_D'],r['D_E'],r['D_U']],[r['E_D'],r['E_E'],r['E_U']]],D_precision=r['D_precision'],D_recall=r['D_recall'],E_precision=r['E_precision'],E_recall=r['E_recall'],U=r['U'],U_rate=ratio(r['U'],r['N']))for r in pt]

def relative(path,base):return Path(os.path.relpath(path,base)).as_posix()

def verify_links(base,content,html_mode=False):
 require('/home/'not in content and 'file://'not in content,'Portable content no absolute local paths')
 refs=local_links(content,html_mode)
 for ref in refs:
  require(ref and not ref.startswith(('/','//'))and not re.match(r'^[A-Za-z]+:',ref)and 'v112'not in ref.casefold(),'Relative local portable ref only')
  path=safe(os.path.normpath(base/ref))
  with opened(path):pass
 return len(refs)

def completed(binding):
 x=read(checked(binding));require(x['completed']is True and x['PASS']is True,'Completed exact frozen evidence');return x

def evaluation_gate(e,p):
 pred=completed(e['prediction_seal']);ev=completed(e['evaluation_complete'])
 require(pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and pred['annotations_reference_GT_oldreal_scores_or_fullRGB_read']is False,'Both predictions sealed before labels')
 require(ev['prediction_seal']==e['prediction_seal']and ev['code']==p['evaluation_code']and ev['protocol']==p['evaluation_protocol']and ev['models']==MODELS,'Exact closed saved evaluation all12 models')
 require(pred['source_fit_complete']['path']==p['source_fit_complete_path']and pred['original_rows2144']==2144 and pred['model_domain_rows4288']==4288,'Source frozen finals/full original real input census')
 review=completed(e['independent_numeric_review']);require(review['evaluation_complete']==e['evaluation_complete'],'Independent scalar audit exact saved evaluation binding')
 return pred,ev,review

def invented():
 tests=[]
 def test(name,ok):require(ok,name);tests.append(dict(name=name,PASS=True))
 test('Fixed12_original_sorted_IDs_no_score_selection',FIXED_IDS==sorted(set(ROOT6+PRIOR_A6))and len(FIXED_IDS)==12)
 r=dict(global_index=0,id='INVENTED',native_corner=0,group='WOOD_PRIMARY105',human_state='DIRECT_VISIBLE',mapping_status='invented',paired_same_query_RGB_D_vs_E='True',initial_reference_error_px='3',query_available=True,query_x=1.,query_y=2.,roi_side=8.,human_reference_x='',human_reference_y='',annotation_coordinate_source='UNKNOWN')
 for m in MODELS:r[m+'_D_E_U']='D'
 for m in DIRECT:r[m+'_raw_direct_end_external_logit']=-.1
 rgb=np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8);before=rgb.copy();valid=np.ones((96,96),bool);card=render_card(r,rgb,valid)
 test('Actual_renderer_clean96_ROI_exact_without_input_cross',np.array_equal(rgb,before)and np.array_equal(np.asarray(card)[202:298,980:1076],rgb))
 test('Twelve_model_rows_do_not_overlap_footer',594+11*47+30<1200 and 1200+3*39+30<1440)
 test('Display_halfpixel_q47p5_fixed',zoom_position(47.5,6)==287.5)
 metrics=[]
 for g in ['WOOD_PRIMARY105','SQUARE118']:
  for m in MODELS:metrics.append(dict(cohort='paired_same_query_RGB',task='D_vs_E',group=g,model=m,N=4,visible_support=2,occluded_support=2,TP=1,FN_decided=0,FP=1,TN=0,U_visible=1,U_occluded=1,out_of_task_S_visible=0,out_of_task_S_occluded=0,other_unresolved_visible=0,other_unresolved_occluded=0))
 pt=primary_tables(dict(tables=dict(metrics=metrics)));pp=pretty_primary(pt)
 test('Actual_primary_adapter_all12_ALLplusmaterial_U_rate_and_CM_conservation',len(pp)==36 and pp[0]['N']==8 and pp[0]['CM_D_E_by_D_E_U']==[[2,0,2],[2,0,2]]and pp[0]['U_rate']==.5)
 bad=[x for x in metrics if x['model']!='C61']
 try:primary_tables(dict(tables=dict(metrics=bad)));ok=False
 except AssertionError:ok=True
 test('Missing_new_model_metrics_rejected',ok)
 sr=[]
 for cohort,n in [('paired_same_query_RGB',21),('same_slot_pipeline',136)]:
  for m in MODELS:
   for g,nn in [('WOOD_PRIMARY105',n),('SQUARE118',0)]:sr.append(dict(cohort=cohort,group=g,model=m,human_S_support=nn,binary_hidden_response=0,visible_response=nn,unresolved=0))
 st=self_tables(dict(tables=dict(human_self_binary_response=sr)))
 test('Actual_SELF_adapter_distinguishes21ES_and136_same_slot',len(st)==24 and st[0]['N']==21 and st[12]['N']==136)
 wrong=[dict(x,human_S_support=21,visible_response=21)if x['cohort']=='same_slot_pipeline'and x['group']=='WOOD_PRIMARY105'else x for x in sr]
 try:self_tables(dict(tables=dict(human_self_binary_response=wrong)));ok=False
 except AssertionError:ok=True
 test('SELF_denominator_conflation_rejected',ok)
 pc=lambda p,r:dict(precision_on_known_truth=p,recall_including_prediction_U=r)
 mm={m:dict(binary_T_nonT=dict(per_class={'T':pc(.5,.25),'NON_T':pc(.4,.75)}),**(dict(owner3=dict(per_class={'FOREIGN_VISIBLE_SUPPORTED':pc(.6,.7)}))if m!='v61'else {}))for m in MODELS if m!='legacy'}
 wt=public_table(dict(rows=360,coverage=dict(known_center204=204,unknown_center156=156,original_input_U=108),cohorts=dict(FULL360=dict(ALL=dict(models=mm)))))
 test('Actual_public11_adapter_keeps_weak_unknown_and_missing_legacy',len(wt)==11 and wt[0]['unknown_center_truth']==156 and wt[0]['ownerF_precision_known']is None)
 test('MD_HTML_literal_link_parsing',local_links('![x](CASE_0000/CARD.png)')==['CASE_0000/CARD.png']and local_links('<img src="CASE_0000/CARD.png">',True)==['CASE_0000/CARD.png'])
 for pth in (P/'never_v112_access',Path('/tmp/outside')):
  try:safe(pth);ok=False
  except AssertionError:ok=True
  test('Lexical_forbidden_or_outside_refused_'+str(len(tests)),ok)
 return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_source_or_real_saved_body_reads=0,model_or_NPZ_or_weight_reads=0),card

def prepare():
 scope=read(SCOPE);require(scope['prospective_evaluation_scope']and scope['actual_data_or_models_allowed_at_preparation']is False,'Prospective code-only scope')
 old=read(OLD_READY);oldp=read(checked(old['protocol']));checked(old['code'])
 er=read(EVAL_READY);ep=read(checked(er['evaluation']['protocol']));require(ep['models']==MODELS,'All twelve evaluator schema')
 checks,card=invented();write(CHECKS,checks);card.save(H/'INVENTED_LAYOUT_ONLY.png')
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(CHECKS),inherited_visual_code=old['code'],inherited_visual_protocol=old['protocol'],evaluation_code=er['evaluation']['code'],evaluation_protocol=er['evaluation']['protocol'],evaluation_schema=er['evaluation']['schema'],source_fit_complete_path=scope['fit_complete']['path'],fixed_original_ids=FIXED_IDS,fixed_rule='Root six 0,2,256,300,840,996 plus previous A six1,3,4,5,7,8, sorted union. Fixed before V203 actual scores; original V191 fixed68 membership retained.',prior_selection=oldp['fixed_prior_selection'],prior_gallery_complete_path=str(OLD/'fixed_native_real_visual_v1/COMPLETE.json'),output=str(OUT),models=MODELS,original_clean96_source='Only existing V200 gallery CLEAN_RAW96/VALID96 PNG bytes from its sealed manifest; no source RGB or raw cache NPZ opened. Decode after root entry, preserve copied PNG bytes and clean card ROI.',context_contract='Both RAW; native+availabilitymask same, C60 XY zero vs C61 original q8 F64 relative XY. Original q/L/crops/native/U unchanged. No label/GT/nominal geometry features.',report_targets=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'],limits=['Existing human D/E/S remains authority, no physical native/occlusion cause certification.','Repeatedly exposed current223/publicheld45, not fresh blind. Legacy realCAL history preserved.','Existing reference <=3px humanD461 is an association proxy, distinct from manualnear376; no GT crop or coordinate correction.','Self paired21 uses D_vs_ES; full humanSELF136/same-slot response is separate. Difference115 exclusion detail not supplied.','Public360/U108 center-owner weak consensus is separate, legacy owner missing; never native D/E gold.','Twelve fixed cards are descriptive, not representative/all1784 inspection. No actual toolview or browser claim in generation.','No arm adoption, model/data/architecture/threshold selection or resolved-goal claim. Source weak labels/source-held do not establish real transfer.'],selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p)
 shared=dict(code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],prediction_seal=None,evaluation_complete=None,independent_numeric_review={'path':str(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),'sha256':None})
 gallery=dict(shared,approved_fixed12_saved_png_gallery=False,source_RGB_cache_NPZ_weights_models_or_forward_allowed=False,prior_gallery_complete={'path':p['prior_gallery_complete_path'],'sha256':None})
 report=dict(shared,approved_closed_saved_report=False,gallery_complete=None,actual_visual_review_bindings=[],new_metrics_fit_or_source_arrays_allowed=False)
 write(H/'ROOT_ACTUAL_FIXED12_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json',gallery);write(H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json',report)
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],scope=p['scope'],gallery_entry_template=bind(H/'ROOT_ACTUAL_FIXED12_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json'),report_entry_template=bind(H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),invented_layout=bind(H/'INVENTED_LAYOUT_ONLY.png'),actual_metadata_array_header_weight_image_label_score_result_reads=0,new_model_forward_fit=0,automatic_retry=False)
 write(H/'FINAL_READY.json',ready);print(json.dumps(ready))

def read_entry(path,approval):
 e=read(path);require(e[approval]is True,'Separate root actual entry')
 p=read(checked(e['protocol']));require(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['scope']==p['scope']==bind(SCOPE),'Frozen report code/protocol/scope')
 return e,p

def gallery(entry):
 e,p=read_entry(entry,'approved_fixed12_saved_png_gallery');require(e['source_RGB_cache_NPZ_weights_models_or_forward_allowed']is False,'Saved PNG only')
 pred,ev,review=evaluation_gate(e,p)
 old=completed(e['prior_gallery_complete']);require(e['prior_gallery_complete']['path']==p['prior_gallery_complete_path']and old['code']==p['inherited_visual_code']and old['protocol']==p['inherited_visual_protocol'],'Exact prior V200 gallery')
 oldentry=read(checked(old['entry']));require(oldentry['input_complete']==pred['input_complete']and old['all_selected_original_RGB96_valid96_png_parity']is True,'Same original raw input cache, old exact clean/valid PNG parity')
 require(not OUT.exists(),'Fresh exclusive fixed12 output/no retry');OUT.mkdir()
 try:
  ob=ev['outputs'];rows=lines(checked(ob['current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl']));selected=fixed_rows(rows)
  current=read(checked(ob['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json']));public=read(checked(ob['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json']))
  priorselection=read(checked(p['prior_selection']));require(all(i in priorselection['unique_original_indices']for i in FIXED_IDS),'All12 belong to immutable old68')
  selection=OUT/'SELECTION_BEFORE_SAVED_PNG_READS.json';write(selection,dict(completed=True,original_global_indices=FIXED_IDS,rule=p['fixed_rule'],evaluation_complete=e['evaluation_complete'],prior_selection=p['prior_selection'],no_PNG_bytes_or_pixels_read_so_far=True))
  oldimages=read(checked(old['manifest']));byid={r['original_global_index']:r for r in oldimages['images']};require(len(byid)==68,'Old68 unique manifest images')
  images=[];cards=[]
  for r in selected:
   i=r['global_index'];src=byid[i];require(src['native_slot']==r['native_corner']and src['id']==r['id']and src['clean_pixel_parity']is True,'Old clean PNG input identity')
   raw=checked(src['clean']);vp=checked(src['valid'])
   with opened(raw)as f:cleanbytes=f.read()
   with opened(vp)as f:validbytes=f.read()
   import io
   with Image.open(io.BytesIO(cleanbytes))as im:rgb=np.array(im)
   with Image.open(io.BytesIO(validbytes))as im:valid8=np.array(im)
   require(rgb.dtype==np.uint8 and rgb.shape==(96,96,3)and valid8.dtype==np.uint8 and valid8.shape==(96,96)and np.isin(valid8,[0,255]).all(),'Exact saved PNG schemas')
   valid=valid8==255;require(ash(rgb)==src['original_RGB96_array_sha256']and ash(valid)==src['original_valid96_array_sha256'],'Decoded original clean96/valid byte parity')
   folder=OUT/('CASE_%04d'%i);folder.mkdir();cp=folder/'CARD.png';newraw=folder/'CLEAN_RAW96.png';newvalid=folder/'VALID96.png'
   with opened(newraw,True)as f:f.write(cleanbytes)
   with opened(newvalid,True)as f:f.write(validbytes)
   card=render_card(r,rgb,valid);card.save(cp);cards.append(card)
   require(sha(newraw)==src['clean']['sha256']and sha(newvalid)==src['valid']['sha256'],'Original saved PNG encoded bytes preserved')
   images.append(dict(original_global_index=i,id=r['id'],native_slot=r['native_corner'],human_state=r['human_state'],reference_error_px=r['initial_reference_error_px'],input_available=r['query_available'],old_clean=src['clean'],old_valid=src['valid'],clean=bind(newraw),valid=bind(newvalid),card=bind(cp),original_clean_and_valid_encoded_and_pixel_parity=True,clean96_card_ROI_exact=True))
  contacts=[]
  for j in range(0,len(cards),2):
   tile=Image.new('RGB',(1600,1440*len(cards[j:j+2])),'white')
   for k,card in enumerate(cards[j:j+2]):tile.paste(card,(0,1440*k))
   path=OUT/('CONTACT_%02d.png'%(j//2+1));tile.save(path);contacts.append(bind(path))
  pt=pretty_primary(primary_tables(current));st=self_tables(current);wt=public_table(public)
  csvfile(OUT/'PRIMARY_ALL12_CM_P_R_U.csv',pt);csvfile(OUT/'SELF_TWO_COHORTS_ALL12.csv',st);csvfile(OUT/'PUBLIC_WEAK_CENTER_ONLY11.csv',wt)
  summary=dict(completed=True,PASS=True,models=MODELS,primary36=pt,self24=st,public_weak11=wt,original_input_U=dict(current_N=1784,current_U=45,current_U_rate=45/1784,public_N=360,public_U=108),primary1237=dict(D=1147,E=90),human_state_inventory=current['state_inventory'],source_evaluation=e['evaluation_complete'],independent_numeric_review=e['independent_numeric_review'],prediction_seal=e['prediction_seal'],selected12=FIXED_IDS,limits=p['limits'],selected_model=None,adopted=False,fully_solved=False)
  write(OUT/'SUMMARY_AND_TABLES.json',summary);write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,images=images,contacts=contacts,actual_tools_view_image='PENDING_SEPARATE_QA',no_all68_or1784_view_claim=True))
  md=['# V203 고정12 저장 crop 진단','','이전 root6+A6의 동일12 ID다. 새 점수로 선택·대체하지 않았다. clean96/valid PNG 원문과 픽셀은 V200과 같다. cross는 확대 표시 사본에만 있다. local crop은 selected-instance/native 끝부위나 실제 차폐 원인을 확정하지 않는다.','', '[전체 표·한계](SUMMARY_AND_TABLES.json) · [12모델 주평가 CSV](PRIMARY_ALL12_CM_P_R_U.csv) · [SELF 두 cohort](SELF_TWO_COHORTS_ALL12.csv) · [public weak만](PUBLIC_WEAK_CENTER_ONLY11.csv)','',table_md([x for x in pt if x['group']=='ALL'])]
  body='<h1>V203 fixed12 saved original crops</h1><p>'+html.escape(md[2])+'</p><p><a href="SUMMARY_AND_TABLES.json">Full tables</a></p>'+table_html([x for x in pt if x['group']=='ALL'])
  for rec in images:
   rr=relative(rec['card']['path'],OUT);clean=relative(rec['clean']['path'],OUT);i=rec['original_global_index']
   md+=['## original %04d'%i,'',f'![고정 original{i} card]({rr})','',f'[변경 없는 clean96]({clean})',''];body+='<h2>original %04d</h2><img width="1500" src="%s"><p><a href="%s">clean96</a></p>'%(i,rr,clean)
  md+=['## 한계','',*['- '+s for s in p['limits']],'']
  text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(md));text(OUT/'index.html','<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{max-width:100%;height:auto}</style>'+body)
  links=verify_links(OUT,(OUT/'PORTABLE_GALLERY_KO.md').read_text())+verify_links(OUT,(OUT/'index.html').read_text(),True)
  done=dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),prediction_seal=e['prediction_seal'],evaluation_complete=e['evaluation_complete'],independent_numeric_review=e['independent_numeric_review'],prior_gallery_complete=e['prior_gallery_complete'],summary=bind(OUT/'SUMMARY_AND_TABLES.json'),selection=bind(selection),manifest=bind(OUT/'IMAGE_MANIFEST.json'),gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),contacts=contacts,selected_unique_cases=12,exact_original_ids=FIXED_IDS,all12_original_encoded_and_decoded_clean_valid_parity=True,clean_card_ROI_exact=True,local_link_occurrences=links,HTML_browser_rendered=False,actual_toolview='PENDING_SEPARATE_QA',source_cache_NPZ_rawRGB_weights_or_models_opened=0,selected_model=None,adopted=False,fully_solved=False)
  write(OUT/'COMPLETE.json',done);print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

def report(entry):
 e,p=read_entry(entry,'approved_closed_saved_report');require(e['new_metrics_fit_or_source_arrays_allowed']is False,'Closed saved report only')
 pred,ev,qa=evaluation_gate(e,p);gal=completed(e['gallery_complete']);require(gal['evaluation_complete']==e['evaluation_complete']and gal['prediction_seal']==e['prediction_seal']and gal['exact_original_ids']==FIXED_IDS,'Exact current fixed12 gallery')
 data=read(checked(gal['summary']));require(data['models']==MODELS and data['selected12']==FIXED_IDS,'All twelve model report schema')
 visual=[dict(binding=b,receipt=completed(b))for b in e['actual_visual_review_bindings']]
 targets=[V/x for x in p['report_targets']];require(all(not x.exists()for x in targets),'Exclusive root report targets')
 main=[x for x in data['primary36']if x['group']=='ALL'];by={x['model']:x for x in main};delta={m:dict(D_correct=by[m]['CM_D_E_by_D_E_U'][0][0],E_correct=by[m]['CM_D_E_by_D_E_U'][1][1],D_recall=by[m]['D_recall'],E_recall=by[m]['E_recall'],primary_U=by[m]['U'],primary_U_rate=by[m]['U_rate'])for m in ['C60','C61']}
 links=[('12개 고정 gallery',relative(gal['gallery']['path'],V)),('12개 gallery HTML',relative(gal['HTML']['path'],V)),('모든 주평가 표',relative(OUT/'PRIMARY_ALL12_CM_P_R_U.csv',V)),('SELF 두 cohort',relative(OUT/'SELF_TWO_COHORTS_ALL12.csv',V)),('public weak만',relative(OUT/'PUBLIC_WEAK_CENTER_ONLY11.csv',V)),('저장 evaluator',relative(ev['code']['path'],V)),('저장 평가 protocol',relative(ev['protocol']['path'],V)),('독립 수치감사',relative(e['independent_numeric_review']['path'],V)),('prediction 봉인',relative(e['prediction_seal']['path'],V)),('본 보고서 코드',relative(__file__,V)),('본 보고서 protocol',relative(PROTOCOL,V))]
 intro='두 고정 spatial q8 head C60/C61을 같은 원초기 q/L/native·RAW96에서 평가했다. C60은 native+other availability mask와 zero XY, C61은 같은 native/mask와 original q8 상대 XY다. 두 팔 모두 결과와 무관하게 사전 고정되었고 source 점수로 선택하지 않았다. 이번 보고서는 결과를 기술하며 모델을 채택하거나 전체 목표 해결을 주장하지 않는다.'
 quantitative='C60은 주평가에서 D%d/1147, E%d/90, C61은 D%d/1147, E%d/90을 맞혔다. 이는 저장 confusion matrix의 정답 수이며 개별 사례의 paired recovery나 단일 원인 효과를 뜻하지 않는다.'%(delta['C60']['D_correct'],delta['C60']['E_correct'],delta['C61']['D_correct'],delta['C61']['E_correct'])
 doc=['# V203 두 동결 spatial q8 head 실사 비교','',intro,'',quantitative,'','주평가 분모는1237(D1147/E90)이며 기존10 저장모델과 새2모델을 모두 표시한다. CM 행은 기존 human D/E, 열은 D/E/U다. 원 전체 input U45/1784=2.52%는 주평가 U/N과 별개다. 목재/정사각 재질과 native8·SELF·public weak를 섞지 않는다.','',table_md(main),'','## 재질별 주평가','',table_md([x for x in data['primary36']if x['group']!='ALL']),'','## SELF 두 별도 cohort','', 'paired same-query의21은 D_vs_ES eligibility 기준이며 전체 humanSELF136/same-slot 응답과 구분한다. 차이115의 세부 제외사유를 이 결과에서 새로 추측하지 않는다. 아래 binary hidden 응답은 학습된 SELF 판정이나 외부/self 원인 인증이 아니다.','',table_md(data['self24']),'','## public weak center-owner만','', '원360/inputU108/knowncenter204/UNKNOWN156의 기존 weak center-owner 관찰이다. crop-anytarget나 native 가시성 D/E gold로 승격하지 않는다. legacy owner는 없는 것으로 유지한다.','',table_md(data['public_weak11']),'','## 고정 입력 관찰 범위','', '사전 고정12 crop을 V200 clean96/valid PNG와 byte·pixel parity로 재사용했다. 렌더 완료와 실제 toolview는 다르다. actual visual receipt '+str(len(visual))+'개만 인용하며 unread68/1784 전체를 봤다고 주장하지 않는다. CASE0840·0996은 기존 고정 진단 사례이며 새로운 학습 선택이 아니다.','']
 images=read(checked(gal['manifest']))['images']
 for im in images:
  if im['original_global_index']in (840,996):doc += [f"![기존 고정 진단 CASE_{im['original_global_index']:04d}]({relative(im['card']['path'],V)})",'']
 doc+=['## 한계','',*['- '+x for x in p['limits']],'','공개 source 직접 end 감독은 합성/기존 RGB weak 관찰이며 자연실사 native gold와 같지 않다. source TRAIN apparent fit이나 source HOLDOUT은 본 실사 전이의 증명이 아니다. q8 입력은 원 initializer 정보이며 GT/nominal endpoint/pose/PnP/새좌표 추정 입력이 아니다. 원 sampler와 q/L은 변경하지 않았다.','', '## 표·코드·감사 연결','',*['- [%s](%s)'%(a,b)for a,b in links],'']
 md='\n'.join(doc);text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md)
 page='<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{max-width:100%;height:auto}</style><h1>V203 두 동결 spatial q8 head 실사 비교</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(quantitative)+'</p><p>주평가1237 D1147/E90. CM 행 D/E, 열 D/E/U. 전체 inputU45/1784와 주평가 U율은 별개다.</p>'+table_html(main)+'<h2>재질별</h2>'+table_html([x for x in data['primary36']if x['group']!='ALL'])+'<h2>SELF paired21 / full136 별도</h2><p>115의 세부 제외사유 미제공. binary hidden은 SELF 원인 인증이 아니다.</p>'+table_html(data['self24'])+'<h2>Public weak center-owner만</h2>'+table_html(data['public_weak11'])+'<h2>고정12 / 실제 view 범위</h2><p>원 saved PNG parity와 렌더는 실제 toolview와 다르다. actual visual receipts '+str(len(visual))+'개. 전체1784 도구 관찰 주장 없음.</p><ul>'+''.join('<li>'+html.escape(x)+'</li>'for x in p['limits'])+'</ul><ul>'+''.join('<li><a href="'+b+'">'+html.escape(a)+'</a></li>'for a,b in links)+'</ul>'
 text(V/'ROOT_READOUT_KO.html',page)
 result=dict(completed=True,PASS=True,phase='V203',code=bind(__file__),protocol=bind(PROTOCOL),scope=p['scope'],prediction_seal=e['prediction_seal'],evaluation_complete=e['evaluation_complete'],independent_numeric_review=e['independent_numeric_review'],gallery_complete=e['gallery_complete'],actual_visual_review_bindings=e['actual_visual_review_bindings'],all_models12=MODELS,table_primary36=data['primary36'],table_SELF24=data['self24'],table_public_weak11=data['public_weak11'],original_input_U=data['original_input_U'],primary1237=data['primary1237'],headline=delta,human_state_inventory=data['human_state_inventory'],fixed12_ids=FIXED_IDS,local_crop_cause_or_physical_native_gold_claim=False,source_results_selected_arm=False,source_weak_not_native_gold=True,repeated_exposure_not_freshblind=True,legacy_realCAL_history_preserved=True,reference461_not_manual376=True,original_qL_native_and_sampler_unchanged=True,new_model_forward_or_fit=0,limits=p['limits'],links=[dict(label=a,relative=b)for a,b in links],selected_model=None,adopted=False,fully_solved=False)
 write(V/'ROOT_RESULTS_KO.json',result)
 n=sum(verify_links(V,(V/x).read_text(),x.endswith('.html'))for x in ['ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'])
 require((V/'ROOT_REPORT_KO.md').read_bytes()==(V/'PORTABLE_REPORT_KO.md').read_bytes(),'MD portable exact')
 write(H/'ROOT_REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),outputs={x.name:bind(x)for x in targets},relative_link_occurrences=n,actual_model_array_or_original_pixel_reads=0,browser_rendered=False,actual_visual_receipts_count=len(visual),selected_model=None,adopted=False,fully_solved=False))
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(H/'ROOT_REPORT_BUILD_COMPLETE.json'))))

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--gallery-entry',type=Path);g.add_argument('--report-entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.gallery_entry:gallery(a.gallery_entry)
 else:report(a.report_entry)
