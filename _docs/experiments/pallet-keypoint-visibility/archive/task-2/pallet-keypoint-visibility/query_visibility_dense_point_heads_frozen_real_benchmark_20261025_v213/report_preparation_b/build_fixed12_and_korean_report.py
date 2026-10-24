"""V213 closed-result Korean report + fixed12 saved-PNG gallery.
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
SCOPE=V/'ROOT_PROSPECTIVE_DENSE_POINT_HEADS_FROZEN_BENCHMARK_SCOPE.json'
OLD=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203'
OLD_PROTOCOL=OLD/'report_preparation_a/PROTOCOL_BEFORE_SAVED_RESULTS_OR_FIXED_PNG_BODIES.json'
SOURCE_PROTOCOL=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json'
EVAL_READY=V/'inference_evaluation_preparation_b/FINAL_READY.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_RESULTS_OR_FIXED_PNG_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'fixed12_dense_point_real_visual_v1'
MODELS=['v61','legacy','C52','C53','C54','C55','C56','C57','C58','C59','C60','C61','C62','C63']
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
 im=Image.new('RGB',(1600,1540),'white');d=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n)
 def label(x,y,s,n=22):d.text((x,y),s,font=font(n),fill='black')
 label(24,16,'V213 | original %04d | %s | native %d | %s'%(r['global_index'],r['id'],r['native_corner'],material(r)),27)
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
 for i,s in enumerate(notes):label(24,1300+i*39,s,21)
 require(np.array_equal(np.asarray(im)[202:298,980:1076],rgb),'Native clean96 ROI exact in card')
 return im

def primary_tables(current):
 metric=[r for r in current['tables']['metrics'] if r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E']
 require(len(metric)==28 and {r['model'] for r in metric}==set(MODELS),'All14 x2 material primary metrics')
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
 require({r['model'] for r in rows}==set(MODELS)-{'legacy'},'Public weak comparison all thirteen available owner routes, legacy missing')
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
 for r in rows:require(all(r[m+'_D_E_U']in ('D','E','U')for m in MODELS),'Every saved fourteen-model response exists')
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


def complete_alias(review,expected):
 bindings=[review[k]for k in ('evaluation_complete','complete')if k in review]
 require(bindings and all(b==expected for b in bindings),'Independent scalar review exact evaluation binding, no conflicting alias')
 return True

def supplementary_tables(current,rows):
 t=current['tables'];refs=[]
 for name,predicate in [('HUMAN_D_EXISTING_REFERENCE_LE3',lambda r:r['human_state']=='DIRECT_VISIBLE'and number(r['initial_reference_error_px'])is not None and number(r['initial_reference_error_px'])<=3),('HUMAN_D_MISSING_REFERENCE',lambda r:r['human_state']=='DIRECT_VISIBLE'and number(r['initial_reference_error_px'])is None)]:
  rr=[r for r in rows if predicate(r)]
  if name=='HUMAN_D_EXISTING_REFERENCE_LE3':require(len(rr)==461,'Existing reference association461 is not manualnear376')
  for m in MODELS:
   c=Counter(r[m+'_D_E_U']for r in rr);refs.append(dict(association=name,model=m,N=len(rr),D=c['D'],E=c['E'],U=c['U'],scope='Existing reference association only; not coordinate error gold or primary gate'))
 return dict(native8=t['primary_native8_D_E_with_U'],type_U_inventory=t['all1784_model_inventory'],reference_association=refs,manual_near_D=t['manual_near_D'],coordinate_error_bins=t['coordinate_error_bins'],paired_recovery_damage=t['paired_recovery_and_damage'])


def evaluation_gate(e,p):
 pred=completed(e['prediction_seal']);ev=completed(e['evaluation_complete'])
 require(pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and pred['annotations_reference_GT_oldreal_scores_or_fullRGB_read']is False,'Both predictions sealed before labels')
 require(ev['prediction_seal']==e['prediction_seal']and ev['code']==p['evaluation_code']and ev['protocol']==p['evaluation_protocol']and ev['models']==MODELS and ev['old_twelve_primary_metrics_exact']is True,'Exact V213 all14 evaluation and old12 saved metrics')
 require(pred['source_fit_complete']['path']==p['source_fit_complete_path']and pred['original_rows2144']==2144 and pred['model_domain_rows4288']==4288,'Source finals/full original input census')
 require(set(pred['arms'])=={r[0]for r in p['new_arms'].values()},'Both new fixed arms')
 for a,weak,weight in p['new_arms'].values():
  r=pred['arms'][a];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is True and type(r['weak_bridge'])is bool and r['weak_bridge']is weak and r['weak_loss_weight']==weight,'Same C61 RAW/q8, fixed point/weak roles')
 review=completed(e['independent_numeric_review']);complete_alias(review,e['evaluation_complete'])
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
 test('Fourteen_model_rows_do_not_overlap_footer',594+13*47+30<1300 and 1300+3*39+30<1540)
 test('Display_halfpixel_q47p5_fixed',zoom_position(47.5,6)==287.5)
 metrics=[]
 for g in ['WOOD_PRIMARY105','SQUARE118']:
  for m in MODELS:metrics.append(dict(cohort='paired_same_query_RGB',task='D_vs_E',group=g,model=m,N=4,visible_support=2,occluded_support=2,TP=1,FN_decided=0,FP=1,TN=0,U_visible=1,U_occluded=1,out_of_task_S_visible=0,out_of_task_S_occluded=0,other_unresolved_visible=0,other_unresolved_occluded=0))
 pt=primary_tables(dict(tables=dict(metrics=metrics)));pp=pretty_primary(pt)
 test('Actual_primary_adapter_all14_ALLplusmaterial_U_rate_and_CM_conservation',len(pp)==42 and pp[0]['N']==8 and pp[0]['CM_D_E_by_D_E_U']==[[2,0,2],[2,0,2]]and pp[0]['U_rate']==.5)
 bad=[x for x in metrics if x['model']!='C63']
 try:primary_tables(dict(tables=dict(metrics=bad)));ok=False
 except AssertionError:ok=True
 test('Missing_new_model_metrics_rejected',ok)
 sr=[]
 for cohort,n in [('paired_same_query_RGB',21),('same_slot_pipeline',136)]:
  for m in MODELS:
   for g,nn in [('WOOD_PRIMARY105',n),('SQUARE118',0)]:sr.append(dict(cohort=cohort,group=g,model=m,human_S_support=nn,binary_hidden_response=0,visible_response=nn,unresolved=0))
 st=self_tables(dict(tables=dict(human_self_binary_response=sr)))
 test('Actual_SELF_adapter_distinguishes21ES_and136_same_slot',len(st)==28 and st[0]['N']==21 and st[14]['N']==136)
 wrong=[dict(x,human_S_support=21,visible_response=21)if x['cohort']=='same_slot_pipeline'and x['group']=='WOOD_PRIMARY105'else x for x in sr]
 try:self_tables(dict(tables=dict(human_self_binary_response=wrong)));ok=False
 except AssertionError:ok=True
 test('SELF_denominator_conflation_rejected',ok)
 pc=lambda p,r:dict(precision_on_known_truth=p,recall_including_prediction_U=r)
 mm={m:dict(binary_T_nonT=dict(per_class={'T':pc(.5,.25),'NON_T':pc(.4,.75)}),**(dict(owner3=dict(per_class={'FOREIGN_VISIBLE_SUPPORTED':pc(.6,.7)}))if m!='v61'else {}))for m in MODELS if m!='legacy'}
 wt=public_table(dict(rows=360,coverage=dict(known_center204=204,unknown_center156=156,original_input_U=108),cohorts=dict(FULL360=dict(ALL=dict(models=mm)))))
 test('Actual_public13_adapter_keeps_weak_unknown_and_missing_legacy',len(wt)==13 and wt[0]['unknown_center_truth']==156 and wt[0]['ownerF_precision_known']is None)
 test('MD_HTML_literal_link_parsing',local_links('![x](CASE_0000/CARD.png)')==['CASE_0000/CARD.png']and local_links('<img src="CASE_0000/CARD.png">',True)==['CASE_0000/CARD.png'])
 for pth in (P/'never_v112_access',Path('/tmp/outside')):
  try:safe(pth);ok=False
  except AssertionError:ok=True
  test('Lexical_forbidden_or_outside_refused_'+str(len(tests)),ok)
 return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_source_or_real_saved_body_reads=0,model_or_NPZ_or_weight_reads=0),card


def prepare():
 scope=read(SCOPE);require(scope['approved_actual_forward_or_evaluation']is False and scope['status']=='PROSPECTIVE_CODE_PREPARATION_ONLY','Prospective code-only scope')
 oldp=read(OLD_PROTOCOL);require(oldp['code']['sha256']=='c3560149cfd79fee14e8d3d1d662432d81c1a3e2fb9e7202c7bcfc234d75d3dc','Exact inherited report code');checked(oldp['code'])
 er=read(EVAL_READY);ep=read(checked(er['evaluation']['protocol']));require(ep['models']==MODELS,'All14 evaluator schema')
 source=read(SOURCE_PROTOCOL);require(source['primary_predeclared']=='C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8'and source['parameters_each']==118359 and source['training']['steps_each']==4000,'Declared source loss authority only')
 extra=read(H/'NEW_REPORT_GATE_CHECKS.json');require(extra['completed']and extra['PASS']and extra['code']==bind(__file__),'New report helpers checked against exact current code')
 checks,card=invented();write(CHECKS,checks);card.save(H/'INVENTED_LAYOUT_ONLY.png')
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(CHECKS),inherited_visual_code=oldp['code'],inherited_visual_protocol=bind(OLD_PROTOCOL),evaluation_code=er['evaluation']['code'],evaluation_protocol=er['evaluation']['protocol'],evaluation_schema=er['schema'],source_fit_complete_path=ep['source_fit_complete_path'],fixed_original_ids=FIXED_IDS,fixed_rule=oldp['fixed_rule'],prior_gallery_complete_path=str(OLD/'fixed12_spatial_q8_real_visual_v1/COMPLETE.json'),output=str(OUT),models=MODELS,
  new_arms={'C62':['C62_DENSE_POINT_RAW_Q8',False,0.],'C63':['C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',True,.25]},primary_a_priori='C63',ablation_a_priori='C62',
  original_clean96_source='Only exact V203 fixed12 CLEAN_RAW96/VALID96 PNG bytes via sealed gallery COMPLETE/manifest/selection; no source RGB/cache NPZ. Actual root entry only. Encoded bytes unchanged, decoded schema/hash recorded, clean card ROI exact.',
  future_metadata_whitelist='New sealed evaluation+prediction+independent scalar review; V203 gallery COMPLETE/manifest/selection/prediction-seal metadata; twelve bound CLEAN/VALID PNG pairs. Never follow source/model/cache payloads.',
  context_contract='Both same C61 RAW/native/original availability mask/original q8 F64 XY; no GT/reference/region/teacher feature, no crop/recenter/permutation changes.',
  source_protocol=bind(SOURCE_PROTOCOL),source_losses=source['losses'],source_point_mask_authority=source['masks'],source_weak_authority=source['weak_authority'],
  report_targets=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'],limits=[
  'Existing human D/E/S remains authority; source procedural point and public weak visible-feature labels are distinct and are not human native gold.',
  'Repeatedly exposed current223/publicheld45, not fresh blind. Legacy realCAL history remains; v61 presence is a proxy, not an independently trained D/E classifier.',
  'Zero output U means the fixed decoder returned a finite decision, not confidence, calibration or solved uncertainty. Original inputU45/1784 remains distinct from primary U.',
  'Existing reference <=3px humanD461 is association only, distinct from manualnear376; no GT recrop or coordinate correction. Cyan marks initial q only; reference is numeric text, no reference marker.',
  'Self paired21 uses D_vs_ES; full humanSELF136/same-slot response is separate. Binary E/ownerF does not certify external-vs-self cause.',
  'Public360/U108 weakcenter204/UNKNOWN156 is separate, legacy owner missing; never native D/E gold.',
  'Twelve frozen descriptive cases, not representative/all1784 inspection. Rendering does not claim actual toolview or browser rendering.',
  'C63 primary and C62 ablation fixed before results. No arm adoption or model/data/architecture/threshold selection. No established performance/causal improvement or solved-goal claim.'
  ],selected_model=None,adopted=False,fully_solved=False)
 p['new_report_gate_checks']=bind(H/'NEW_REPORT_GATE_CHECKS.json');write(PROTOCOL,p)
 shared=dict(code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],prediction_seal=None,evaluation_complete=None,independent_numeric_review=None)
 write(H/'ROOT_ACTUAL_FIXED12_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(shared,approved_fixed12_saved_png_gallery=False,source_RGB_cache_NPZ_weights_models_or_forward_allowed=False,prior_gallery_complete=dict(path=p['prior_gallery_complete_path'],sha256=None)))
 write(H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(shared,approved_closed_saved_report=False,gallery_complete=None,actual_visual_review_bindings=[],root_confirmed_fixed12_actual_visual_QA=False,new_metrics_fit_or_source_arrays_allowed=False))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],scope=p['scope'],gallery_entry_template=bind(H/'ROOT_ACTUAL_FIXED12_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json'),report_entry_template=bind(H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),invented_layout=bind(H/'INVENTED_LAYOUT_ONLY.png'),actual_source_or_real_metadata_array_header_weight_image_label_score_result_reads=0,new_model_forward_fit=0,automatic_retry=False)
 ready['new_report_gate_checks']=p['new_report_gate_checks'];ready['new_report_gate_check_code']=bind(H/'check_report_gates.py');ready['initial_derive_failure_preserved']=bind(H/'provenance/INITIAL_DERIVE_FAILURE.json')
 write(H/'FINAL_READY.json',ready);print(json.dumps(ready))


def read_entry(path,approval):
 e=read(path);require(e[approval]is True,'Separate root actual entry')
 p=read(checked(e['protocol']));require(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['scope']==p['scope']==bind(SCOPE),'Frozen report code/protocol/scope')
 return e,p

def gallery(entry):
 e,p=read_entry(entry,'approved_fixed12_saved_png_gallery');require(e['source_RGB_cache_NPZ_weights_models_or_forward_allowed']is False,'Saved PNG only')
 pred,ev,review=evaluation_gate(e,p)
 old=completed(e['prior_gallery_complete']);require(e['prior_gallery_complete']['path']==p['prior_gallery_complete_path']and old['code']==p['inherited_visual_code']and old['protocol']==p['inherited_visual_protocol']and old['exact_original_ids']==FIXED_IDS and old['all12_original_encoded_and_decoded_clean_valid_parity']is True,'Exact prior V203 fixed12 gallery')
 oldpred=completed(old['prediction_seal']);require(oldpred['input_complete']==pred['input_complete']and oldpred['original_raw_complete']==pred['original_raw_complete'],'Same immutable original input caches; metadata only')
 require(not OUT.exists(),'Fresh exclusive fixed12 output/no retry');OUT.mkdir()
 try:
  ob=ev['outputs'];rows=lines(checked(ob['current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl']));selected=fixed_rows(rows)
  current=read(checked(ob['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json']));public=read(checked(ob['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json']))
  priorselection=read(checked(old['selection']));require(priorselection['original_global_indices']==FIXED_IDS,'Exact inherited12, no new score selection')
  selection=OUT/'SELECTION_BEFORE_SAVED_PNG_READS.json';write(selection,dict(completed=True,original_global_indices=FIXED_IDS,rule=p['fixed_rule'],evaluation_complete=e['evaluation_complete'],prior_selection=old['selection'],no_PNG_bytes_or_pixels_read_so_far=True))
  oldimages=read(checked(old['manifest']));byid={r['original_global_index']:r for r in oldimages['images']};require(len(byid)==12 and sorted(byid)==FIXED_IDS,'Old fixed12 unique manifest images')
  images=[];cards=[]
  for r in selected:
   i=r['global_index'];src=byid[i];require(src['native_slot']==r['native_corner']and src['id']==r['id']and src['input_available']==r['query_available']and src['original_clean_and_valid_encoded_and_pixel_parity']is True,'Old clean PNG input identity')
   raw=checked(src['clean']);vp=checked(src['valid'])
   with opened(raw)as f:cleanbytes=f.read()
   with opened(vp)as f:validbytes=f.read()
   import io
   with Image.open(io.BytesIO(cleanbytes))as im:rgb=np.array(im)
   with Image.open(io.BytesIO(validbytes))as im:valid8=np.array(im)
   require(rgb.dtype==np.uint8 and rgb.shape==(96,96,3)and valid8.dtype==np.uint8 and valid8.shape==(96,96)and np.isin(valid8,[0,255]).all(),'Exact saved PNG schemas')
   valid=valid8==255;decoded_rgb=ash(rgb);decoded_valid=ash(valid) # Exact encoded source PNG identity already checked; record decoded hashes without reopening original cache.
   folder=OUT/('CASE_%04d'%i);folder.mkdir();cp=folder/'CARD.png';newraw=folder/'CLEAN_RAW96.png';newvalid=folder/'VALID96.png'
   with opened(newraw,True)as f:f.write(cleanbytes)
   with opened(newvalid,True)as f:f.write(validbytes)
   card=render_card(r,rgb,valid);card.save(cp);cards.append(card)
   require(sha(newraw)==src['clean']['sha256']and sha(newvalid)==src['valid']['sha256'],'Original saved PNG encoded bytes preserved')
   images.append(dict(original_global_index=i,id=r['id'],native_slot=r['native_corner'],human_state=r['human_state'],reference_error_px=r['initial_reference_error_px'],input_available=r['query_available'],old_clean=src['clean'],old_valid=src['valid'],clean=bind(newraw),valid=bind(newvalid),card=bind(cp),original_clean_and_valid_encoded_and_pixel_parity=True,clean96_card_ROI_exact=True,original_RGB96_array_sha256=decoded_rgb,original_valid96_array_sha256=decoded_valid))
  contacts=[]
  for j in range(0,len(cards),2):
   tile=Image.new('RGB',(1600,1540*len(cards[j:j+2])),'white')
   for k,card in enumerate(cards[j:j+2]):tile.paste(card,(0,1540*k))
   path=OUT/('CONTACT_%02d.png'%(j//2+1));tile.save(path);contacts.append(bind(path))
  pt=pretty_primary(primary_tables(current));st=self_tables(current);wt=public_table(public);supp=supplementary_tables(current,rows)
  for key,name in [('native8','PRIMARY_NATIVE8_ALL14.csv'),('type_U_inventory','ALL1784_TYPE_U_INVENTORY.csv'),('reference_association','REFERENCE_ASSOCIATIONS_ALL14.csv'),('manual_near_D','MANUAL_NEAR_D376_ALL14.csv'),('coordinate_error_bins','COORDINATE_ERROR_ASSOCIATIONS_ALL14.csv'),('paired_recovery_damage','PAIRED_RECOVERY_DAMAGE_ALL14.csv')]:csvfile(OUT/name,supp[key])
  csvfile(OUT/'PRIMARY_ALL14_CM_P_R_U.csv',pt);csvfile(OUT/'SELF_TWO_COHORTS_ALL14.csv',st);csvfile(OUT/'PUBLIC_WEAK_CENTER_ONLY13.csv',wt)
  summary=dict(completed=True,PASS=True,models=MODELS,primary42=pt,self28=st,public_weak13=wt,supplementary=supp,original_input_U=dict(current_N=1784,current_U=45,current_U_rate=45/1784,public_N=360,public_U=108),primary1237=dict(D=1147,E=90),human_state_inventory=current['state_inventory'],source_evaluation=e['evaluation_complete'],independent_numeric_review=e['independent_numeric_review'],prediction_seal=e['prediction_seal'],selected12=FIXED_IDS,limits=p['limits'],selected_model=None,adopted=False,fully_solved=False)
  write(OUT/'SUMMARY_AND_TABLES.json',summary);write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,images=images,contacts=contacts,actual_tools_view_image='PENDING_SEPARATE_QA',no_all68_or1784_view_claim=True))
  md=['# V213 고정12 저장 crop 진단','','이전 root6+A6의 동일12 ID다. 새 점수로 선택·대체하지 않았다. clean96/valid PNG 원문과 픽셀은 V203과 같다. cross는 확대 표시 사본에만 있다. local crop은 selected-instance/native 끝부위나 실제 차폐 원인을 확정하지 않는다.','', '[전체 표·한계](SUMMARY_AND_TABLES.json) · [14모델 주평가 CSV](PRIMARY_ALL14_CM_P_R_U.csv) · [SELF 두 cohort](SELF_TWO_COHORTS_ALL14.csv) · [public weak만](PUBLIC_WEAK_CENTER_ONLY13.csv)','',table_md([x for x in pt if x['group']=='ALL'])]
  body='<h1>V213 fixed12 saved original crops</h1><p>'+html.escape(md[2])+'</p><p><a href="SUMMARY_AND_TABLES.json">Full tables</a></p>'+table_html([x for x in pt if x['group']=='ALL'])
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
 data=read(checked(gal['summary']));require(data['models']==MODELS and data['selected12']==FIXED_IDS,'All14 model report schema')
 require(e['root_confirmed_fixed12_actual_visual_QA']is True and e['actual_visual_review_bindings'],'Separate root actual fixed12 visual QA required')
 visual=[dict(binding=b,receipt=completed(b))for b in e['actual_visual_review_bindings']]
 targets=[V/x for x in p['report_targets']];require(all(not x.exists()for x in targets),'Exclusive report targets')
 main=[x for x in data['primary42']if x['group']=='ALL'];by={x['model']:x for x in main}
 headline={m:dict(D_correct=by[m]['CM_D_E_by_D_E_U'][0][0],E_correct=by[m]['CM_D_E_by_D_E_U'][1][1],D_recall=by[m]['D_recall'],E_recall=by[m]['E_recall'],primary_U=by[m]['U'],primary_U_rate=by[m]['U_rate'])for m in ['C62','C63']}
 filelinks=[('고정12 gallery',gal['gallery']['path']),('gallery HTML',gal['HTML']['path']),('14모델 CM/P/R/U',OUT/'PRIMARY_ALL14_CM_P_R_U.csv'),('native8 원표',OUT/'PRIMARY_NATIVE8_ALL14.csv'),('전체1784 유형/U',OUT/'ALL1784_TYPE_U_INVENTORY.csv'),('SELF 두 cohort',OUT/'SELF_TWO_COHORTS_ALL14.csv'),('reference461/missing 연관',OUT/'REFERENCE_ASSOCIATIONS_ALL14.csv'),('manualnear376 원표',OUT/'MANUAL_NEAR_D376_ALL14.csv'),('기존 error-bin 연관',OUT/'COORDINATE_ERROR_ASSOCIATIONS_ALL14.csv'),('paired recovery/damage 원표',OUT/'PAIRED_RECOVERY_DAMAGE_ALL14.csv'),('public weak13',OUT/'PUBLIC_WEAK_CENTER_ONLY13.csv'),('저장 evaluator',ev['code']['path']),('평가 protocol',ev['protocol']['path']),('독립 수치감사',e['independent_numeric_review']['path']),('prediction 봉인',e['prediction_seal']['path']),('source 감독 protocol',p['source_protocol']['path']),('보고서 코드',__file__),('보고서 protocol',PROTOCOL)]
 links=[(label,relative(path,V))for label,path in filelinks]
 intro='C63은 결과 전 지정한 주 모델이고 C62는 point-only 대조 팔이다. 두 모델은 같은 C61 RAW/native/original q8 구조(118,359개 파라미터)로 각각 4,000회 학습했다. 두 SOURCE 최종 상태와 두 실사 출력을 먼저 봉인하고 같은 고정 평가에서 모두 비교했다. 결과에 따른 채택·선택을 하지 않는다.'
 quantitative='C62: D%d/1147·E%d/90, C63: D%d/1147·E%d/90. 저장 CM의 정답 수이며, 단일 원인의 효과나 성능향상 입증을 뜻하지 않는다.'%(headline['C62']['D_correct'],headline['C62']['E_correct'],headline['C63']['D_correct'],headline['C63']['E_correct'])
 authority='두 팔은 기존 old5 평균과 원영상 own-scene procedural point D/E BCE를 사용하고, C63만 공개 SOURCE TRAIN의 local-feature weak D/E bridge BCE를 가중치0.25로 추가한다. Point target은 자체 합성 장면의 점 상태이고 public weak target은 보이는 국소 feature 개입이다. Region/finite-end/center-owner를 인간 native D/E 정답으로 복사하지 않는다. 원 q/L/native/sampler를 유지하며 GT reference·라벨은 입력 특징이 아니다.'
 doc=['# V213 두 동결 dense-point head 실사 비교','',intro,'',quantitative,'',authority,'','주평가1237(D1147/E90)의 기존12+신규2모델 전체를 표시한다. CM 행은 기존 human D/E, 열은 D/E/U다. 원 inputU45/1784와 주평가 U율은 별개다.','',table_md(main),'','## 목재·정사각형별 주평가','',table_md([x for x in data['primary42']if x['group']!='ALL']),'','## SELF 두 별도 cohort','','paired21은 D_vs_ES eligibility이며 전체 humanSELF136/same-slot과 다르다. Binary hidden 응답은 SELF 학습판정이나 실제 차폐 원인 인증이 아니다.','',table_md(data['self28']),'','## reference 연관만','','기존 human-D/reference≤3px461과 manualnear376은 다르다. 아래 표는 위치오류 gold나 주평가 gate가 아니다.','',table_md(data['supplementary']['reference_association']),'','## public weak center-owner만','','원360/inputU108/known204/UNKNOWN156이다. v61 presence proxy 및 owner 응답을 native D/E 성능과 섞지 않는다. legacy owner는 결측을 유지한다.','',table_md(data['public_weak13']),'','## 고정 입력 관찰 범위','','V203에서 이미 고정한 동일12 crop/valid PNG를 원byte 그대로 사용했다. Cyan은 초기 q47.5 표시이고 기존 reference는 숫자로만 별도 표시한다. GT recenter는 없다. 실제 visual receipt '+str(len(visual))+'개를 인용하며, 렌더링과 실제 toolview 및 HTML browser 렌더링을 구분한다. 전체68/1784를 봤다고 주장하지 않는다.','']
 images=read(checked(gal['manifest']))['images']
 for im in images:
  if im['original_global_index']in ROOT6:doc += [f"![고정 CASE_{im['original_global_index']:04d}]({relative(im['card']['path'],V)})",'']
 doc+=['## 한계','',*['- '+x for x in p['limits']],'','## 표·코드·감사 연결','',*['- [%s](%s)'%(a,b)for a,b in links],'']
 md='\n'.join(doc);text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md)
 page='<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{max-width:100%;height:auto}</style><h1>V213 고정14 모델 실사 비교</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(quantitative)+'</p><p>'+html.escape(authority)+'</p>'+table_html(main)+'<h2>목재·정사각형</h2>'+table_html([x for x in data['primary42']if x['group']!='ALL'])+'<h2>SELF paired21 / full136</h2>'+table_html(data['self28'])+'<h2>기존 reference association</h2>'+table_html(data['supplementary']['reference_association'])+'<h2>public weak center-owner</h2>'+table_html(data['public_weak13'])+'<h2>한계</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>'for x in p['limits'])+'</ul><ul>'+''.join('<li><a href="'+b+'">'+html.escape(a)+'</a></li>'for a,b in links)+'</ul>'
 text(V/'ROOT_READOUT_KO.html',page)
 result=dict(completed=True,PASS=True,phase='V213',code=bind(__file__),protocol=bind(PROTOCOL),scope=p['scope'],prediction_seal=e['prediction_seal'],evaluation_complete=e['evaluation_complete'],independent_numeric_review=e['independent_numeric_review'],gallery_complete=e['gallery_complete'],actual_visual_review_bindings=e['actual_visual_review_bindings'],all_models14=MODELS,table_primary42=data['primary42'],table_SELF28=data['self28'],table_public_weak13=data['public_weak13'],supplementary=data['supplementary'],original_input_U=data['original_input_U'],primary1237=data['primary1237'],headline=headline,human_state_inventory=data['human_state_inventory'],fixed12_ids=FIXED_IDS,primary_a_priori='C63',ablation_a_priori='C62',source_authority=authority,source_protocol=p['source_protocol'],local_crop_cause_or_physical_native_gold_claim=False,source_results_selected_arm=False,repeated_exposure_not_freshblind=True,legacy_realCAL_history_preserved=True,v61_presence_proxy=True,zero_U_is_not_confidence=True,reference461_not_manual376=True,original_qL_native_and_sampler_unchanged=True,new_model_forward_or_fit=0,limits=p['limits'],links=[dict(label=a,relative=b)for a,b in links],selected_model=None,adopted=False,fully_solved=False)
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
