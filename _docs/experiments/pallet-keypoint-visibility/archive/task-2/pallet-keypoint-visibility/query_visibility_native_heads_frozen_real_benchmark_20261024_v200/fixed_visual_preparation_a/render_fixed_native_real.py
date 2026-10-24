"""Fixed V200 saved-result gallery/readout; no model, initializer or recrop.
Prepare uses only prospective plan/schema and authored invented arrays/records.
Run needs both prediction seal + terminal saved evaluation + separate root entry.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,html,json,math,os,re,stat,traceback
from html.parser import HTMLParser
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
OLD191=P/'query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191'
PLAN=OLD191/'ROOT_VISUAL_DIAGNOSTIC_PLAN_BEFORE_REAL_OUTPUT_OR_LABEL_READS.json'
FIXED_SELECTION=dict(path=str(OLD191/'fixed_direct_end_real_visual_v1/SELECTION_BEFORE_INPUT_PIXELS.json'),sha256='69f558908cd5efbbb692bf998f75b2c5e93b74691011207b496f2e5ea04b9769')
SCHEMA=H/'PREPARED_GALLERY_OUTPUT_SCHEMA.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_EVALUATION_OR_CROP_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'fixed_native_real_visual_v1'
OLD=P/'query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184'
INPUT_COMPLETE=OLD/'real_raw10_SAM96_inputs_v1/COMPLETE.json'
EVAL_CODE={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/evaluation_preparation_a/evaluate_saved_native_heads_real.py', 'sha256': '2778ea6ded31ca8369803fa7dff41739d81a788c09e8645595541382fb5b1aa5'}
EVAL_PROTO={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/evaluation_preparation_a/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json', 'sha256': 'a04ea374edff991b03c9865a84ae12558c7673b69e77504be766dcf6ad8b3605'}
MODELS=['v61','legacy','C52','C53','C54','C55','C56','C57','C58','C59'];TAGS=MODELS[2:]
CELLS=['BOTH_C54_C55_CORRECT_VISIBLE_D','BOTH_C54_C55_CORRECT_EXTERNAL_E','EITHER_DIRECT_MODEL_MISSES_EXTERNAL_E','EITHER_DIRECT_MODEL_REJECTS_VISIBLE_D_ERROR_LE3PX','EITHER_DIRECT_MODEL_REJECTS_VISIBLE_D_ERROR_GT3PX','EITHER_DIRECT_MODEL_INPUT_OR_OUTPUT_UNKNOWN']
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
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
def matches(r,cell):
 a,b=r['C54_D_E_U'],r['C55_D_E_U'];d=r['human_state']=='DIRECT_VISIBLE';e=r['human_state']=='EXTERNAL_OCCLUDED';err=number(r['initial_reference_error_px'])
 require(a in {'D','E','U'} and b in {'D','E','U'},'Original direct outcomes')
 if cell==CELLS[5]:return a=='U' or b=='U'
 if not primary(r):return False
 return {CELLS[0]:d and a==b=='D',CELLS[1]:e and a==b=='E',CELLS[2]:e and (a!='E' or b!='E'),CELLS[3]:d and (a=='E' or b=='E') and err is not None and err<=3,CELLS[4]:d and (a=='E' or b=='E') and err is not None and err>3}[cell]
def choose(rows):
 ordered=sorted(rows,key=lambda r:r['global_index']);require(len({r['global_index'] for r in rows})==len(rows),'Unique original IDs')
 selected=set();cells=[]
 for mat in ['WOOD','SQUARE']:
  for cell in CELLS:
   eligible=[r['global_index'] for r in ordered if material(r)==mat and matches(r,cell)];first=eligible[:6];selected.update(first)
   cells.append(dict(material=mat,cell=cell,eligible_N=len(eligible),selected_original_global_indices=first,missing_cell=len(eligible)==0,no_replacement=True))
 return cells,[r for r in ordered if r['global_index'] in selected]
def frozen_selection(rows,selection,expected=68):
 require(selection.get('completed')is True and selection['unique_count']==expected,'Exact previously frozen case count')
 ids=selection['unique_original_indices'];require(len(ids)==expected and ids==sorted(set(ids)),'Fixed source IDs unique and ordered')
 byid={r['global_index']:r for r in rows};require(len(byid)==len(rows)and all(i in byid for i in ids),'Every fixed case present in full original ledger')
 # Old cells are provenance only. No predicate is recomputed on new predictions.
 return selection['cells'],[byid[i]for i in ids]


def ledger_contract(rows):
 require(len(rows)==1784 and [r['global_index']for r in rows]==list(range(1784))and {r['native_corner']for r in rows}==set(range(8)),'Original full1784 order/native8')
 for r in rows:
  require(all(r[m+'_D_E_U']in {'D','E','U'}for m in MODELS),'All10 saved D/E/U columns without imputation')
  for m in ['C54','C55','C56','C57','C58','C59']:
   require(m+'_raw_direct_end_external_logit'in r,'Explicit saved direct logit column')
 return True

def zoom_position(q,scale):return (q+.5)*scale-.5

def render_card(r,rgb,valid):
 require(rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and valid.dtype==bool and valid.shape==(96,96),'Exact original crop')
 im=Image.new('RGB',(1600,1280),'white');d=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n)
 def label(x,y,s,n=22):d.text((x,y),s,font=font(n),fill='black')
 label(24,16,'V200 | original %04d | %s | native %d | %s'%(r['global_index'],r['id'],r['native_corner'],material(r)),27)
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
  label(xs[2],y,fmt(r.get(m+'_raw_direct_end_external_logit')) if m in ['C54','C55','C56','C57','C58','C59'] else '--',22)
  label(xs[3],y,'direct raw >=0' if m in ['C54','C55','C56','C57','C58','C59'] else ('regional E-first' if m in ['C52','C53'] else 'original saved'),21)
 notes=['Exact clean96 file remains unchanged. Cross is display-only; input q/L/native were never moved.',
 'D/E/S human semantics and point/part location differ. A local crop alone cannot establish selected-instance or true cause.',
 'Error <=3px is existing reference association, not physical coordinate gold. Repeated real-set exposure: not blind.',
 'Fixed post-result strata are descriptive, not random prevalence samples or a training/model/threshold selection.']
 for i,s in enumerate(notes):label(24,1060+i*39,s,21)
 require(np.array_equal(np.asarray(im)[202:298,980:1076],rgb),'Native clean96 ROI exact in card')
 return im

def primary_tables(current):
 metric=[r for r in current['tables']['metrics'] if r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E']
 require(len(metric)==20 and {r['model'] for r in metric}==set(MODELS),'All10 x2 material primary metrics')
 result=[]
 for group in ['ALL','WOOD_PRIMARY105','SQUARE118']:
  for m in MODELS:
   rr=[r for r in metric if r['model']==m and (group=='ALL' or r['group']==group)];require(len(rr)==(2 if group=='ALL' else 1),'Fixed model/material rows')
   c={k:sum(int(r[k]) for r in rr) for k in ['N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']}
   du=c['U_visible']+c['out_of_task_S_visible']+c['other_unresolved_visible'];eu=c['U_occluded']+c['out_of_task_S_occluded']+c['other_unresolved_occluded']
   require(c['TP']+c['FN_decided']+du==c['visible_support'] and c['FP']+c['TN']+eu==c['occluded_support'],'Original known denominators incl U')
   result.append(dict(group=group,model=m,N=c['N'],D_support=c['visible_support'],E_support=c['occluded_support'],D_D=c['TP'],D_E=c['FN_decided'],D_U=du,E_D=c['FP'],E_E=c['TN'],E_U=eu,D_precision=ratio(c['TP'],c['TP']+c['FP']),D_recall=ratio(c['TP'],c['visible_support']),E_precision=ratio(c['TN'],c['TN']+c['FN_decided']),E_recall=ratio(c['TN'],c['occluded_support']),U=du+eu))
 return result

def inventories(rows):
 out=[]
 for mat in ['WOOD','SQUARE']:
  for slot in range(8):
   rr=[r for r in rows if material(r)==mat and r['native_corner']==slot];c=Counter(r['human_state'] for r in rr)
   out.append(dict(material=mat,native_slot=slot,all_original_slots=len(rr),original_input_U=sum(not yes(r['query_available']) for r in rr),primary_D_E_N=sum(primary(r) for r in rr),human_D=c['DIRECT_VISIBLE'],human_E=c['EXTERNAL_OCCLUDED'],human_S=c['SELF_OCCLUDED'],human_OTHER_UNKNOWN=len(rr)-c['DIRECT_VISIBLE']-c['EXTERNAL_OCCLUDED']-c['SELF_OCCLUDED']))
 require(sum(r['all_original_slots'] for r in out)==1784,'All8/allmaterials full slots');return out

def public_table(public):
 rows=[]
 for model,metrics in public['cohorts']['FULL360']['ALL']['models'].items():
  binary=metrics['binary_T_nonT']['per_class'];owner=metrics.get('owner3',{}).get('per_class',{})
  t=binary['T'];non=binary['NON_T'];f=owner.get('FOREIGN_VISIBLE_SUPPORTED',{})
  rows.append(dict(model=model,all_original_rows=public['rows'],known_center_truth=public['coverage']['known_center204'],unknown_center_truth=public['coverage']['unknown_center156'],original_input_U=public['coverage']['original_input_U'],T_precision_known=t['precision_on_known_truth'],T_recall_including_U=t['recall_including_prediction_U'],nonT_recall_including_U=non['recall_including_prediction_U'],ownerF_precision_known=f.get('precision_on_known_truth'),ownerF_recall_including_U=f.get('recall_including_prediction_U')))
 require({r['model'] for r in rows}=={'v61','C52','C53','C54','C55','C56','C57','C58','C59'},'Public weak comparison all nine available owner routes, legacy missing')
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
def check_links(root,content,html_mode=False):
 require('/home/' not in content and 'file://' not in content,'Portable text has no local absolute paths')
 refs=local_links(content,html_mode)
 for ref in refs:
  require(not ref.startswith('/') and not re.match(r'^[A-Za-z]+:',ref) and '..' not in Path(ref).parts,'Only local generated gallery paths')
  require((root/ref).is_file(),'Every literal gallery image/table link exists')
 return len(refs)

def invented():
 tests=[]
 def test(n,x):require(x,n);tests.append(dict(name=n,PASS=True))
 rows=[]
 for i in range(18):
  r=dict(global_index=i,id='TOY',native_corner=i%8,group='WOOD_PRIMARY105',human_state='DIRECT_VISIBLE',mapping_status='invented',paired_same_query_RGB_D_vs_E='True',initial_reference_error_px='3',query_available=True,query_x=1.,query_y=1.,roi_side=8.,human_reference_x='',human_reference_y='',annotation_coordinate_source='UNKNOWN')
  for m in MODELS:r[m+'_D_E_U']='D'
  for m in ['C54','C55','C56','C57','C58','C59']:r[m+'_raw_direct_end_external_logit']=-.1
  rows.append(r)
 cells,sel=choose(rows[::-1]);test('First6_sort_original_index_and_dedup', [r['global_index'] for r in sel]==list(range(6)))
 test('Missing_cells_retained_no_replacement',sum(x['missing_cell'] for x in cells)==11)
 r=dict(rows[0],C54_D_E_U='U',human_state='UNKNOWN',paired_same_query_RGB_D_vs_E='False');test('U_cell_all1784_scope_separate_from_primary',matches(r,CELLS[5]) and not any(matches(r,c) for c in CELLS[:5]))
 e=dict(rows[0],human_state='EXTERNAL_OCCLUDED',C54_D_E_U='U',C55_D_E_U='E');test('E_miss_includes_U_not_silently_dropped',matches(e,CELLS[2]))
 r=dict(rows[0],C54_D_E_U='E');test('Error3_closed_boundary_and_missing_not_imputed',matches(r,CELLS[3]) and not matches(dict(r,initial_reference_error_px=''),CELLS[3]) and matches(dict(r,initial_reference_error_px='3.01'),CELLS[4]))
 rgb=np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8);copy=rgb.copy();valid=np.ones((96,96),bool);card=render_card(rows[0],rgb,valid)
 test('Clean96_native_card_ROI_and_input_bytes_unchanged',np.array_equal(rgb,copy) and np.array_equal(np.asarray(card)[202:298,980:1076],rgb))
 test('Q47p5_pixelcenter_zoom_halfpixel',zoom_position(47.5,6)==287.5)
 test('Last_model_row_and_footer_vertical_bboxes_disjoint',594+9*47+30<1060)
 d=ImageDraw.Draw(card);f=ImageFont.truetype(FONT,21);test('Image_footer_caption_width_fits',d.textbbox((24,0),'Fixed post-result strata are descriptive, not random prevalence samples or a training/model/threshold selection.',font=f)[2]<1580)
 # Exercise the same report adapters, using authored numbers only.
 metrics=[]
 for group in ['WOOD_PRIMARY105','SQUARE118']:
  for model in MODELS:metrics.append(dict(cohort='paired_same_query_RGB',task='D_vs_E',group=group,model=model,N=4,visible_support=2,occluded_support=2,TP=1,FN_decided=0,FP=1,TN=0,U_visible=1,U_occluded=1,out_of_task_S_visible=0,out_of_task_S_occluded=0,other_unresolved_visible=0,other_unresolved_occluded=0))
 pt=primary_tables({'tables':{'metrics':metrics}})
 test('Actual_primary_report_adapter_all10_including_U_and_zero_precision_denominator',len(pt)==30 and pt[0]['N']==8 and pt[0]['D_recall']==.5 and pt[0]['E_precision'] is None and pt[0]['U']==4)
 pc=lambda p,r:{'precision_on_known_truth':p,'recall_including_prediction_U':r}
 mm={m:{'binary_T_nonT':{'per_class':{'T':pc(.5,.25),'NON_T':pc(.4,.75)}},**({'owner3':{'per_class':{'FOREIGN_VISIBLE_SUPPORTED':pc(.6,.7)}}} if m!='v61' else {})} for m in ['v61','C52','C53','C54','C55','C56','C57','C58','C59']}
 wt=public_table({'rows':360,'coverage':{'known_center204':204,'unknown_center156':156,'original_input_U':108},'cohorts':{'FULL360':{'ALL':{'models':mm}}}})
 test('Actual_public_report_adapter_keeps_weak_unknown_truth_separate_from_native',len(wt)==9 and wt[0]['unknown_center_truth']==156 and wt[0]['ownerF_precision_known'] is None)
 fixed=dict(completed=True,unique_count=2,unique_original_indices=[0,3],cells=[dict(material='WOOD',cell='ORIGINAL_V191_FIXED',selected_original_global_indices=[0,3])]);_,ff=frozen_selection(rows,fixed,2)
 test('Frozen_ID_selection_ignores_new_model_predictions',[r['global_index']for r in ff]==[0,3] and [r['global_index']for r in frozen_selection([dict(r,C58_D_E_U='E',C59_D_E_U='U')for r in rows],fixed,2)[1]]==[0,3])
 for name,bad in [('duplicate',dict(fixed,unique_original_indices=[0,0])),('missing',dict(fixed,unique_original_indices=[0,999]))]:
  try:frozen_selection(rows,bad,2);ok=False
  except AssertionError:ok=True
  test('Fixed_selection_'+name+'_rejected',ok)
 test('Literal_HTML_image_table_links_parse_without_browser',local_links('<img src="a.png"><a href="table.csv">x</a>',True)==['a.png','table.csv'])
 test('All10_native_card_rows_and_RAW_routes',len(MODELS)==10 and MODELS[-2:]==['C58','C59'])
 badmetrics=[r for r in metrics if r['model']!='C59']
 try:primary_tables({'tables':{'metrics':badmetrics}});ok=False
 except AssertionError:ok=True
 test('Missing_tenth_model_not_imputed',ok)
 badpublic={'rows':360,'coverage':{'known_center204':204,'unknown_center156':156,'original_input_U':108},'cohorts':{'FULL360':{'ALL':{'models':dict(mm,legacy=mm['v61'])}}}}
 try:public_table(badpublic);ok=False
 except AssertionError:ok=True
 test('No_public_legacy_owner_imputation',ok)
 full=[dict(rows[i%len(rows)],global_index=i,native_corner=i%8)for i in range(1784)]
 test('Actual_all1784_ledger_guard_all10_columns',ledger_contract(full))
 broken=[dict(r)for r in full];del broken[0]['C59_D_E_U']
 try:ledger_contract(broken);ok=False
 except(KeyError,AssertionError):ok=True
 test('Missing_new_prediction_column_rejected',ok)
 for badpath in [str(P/'lexical_v112_forbidden'/'never_stat'),'/tmp/outside_project_file']:
  try:safe(badpath);ok=False
  except AssertionError:ok=True
  test('Lexical_path_guard_'+str(len(tests)),ok)
 return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_saved_evaluation_crop_or_score_body_reads=0),card

def prepare():
 scope_path=V/'ROOT_PROSPECTIVE_FIXED68_TEN_MODEL_VISUAL_CODE_PREPARATION_SCOPE_A.json';scope=read(scope_path)
 require(bind(scope_path)['sha256']=='123057e55402c7d8ed4ec0761e1563be3e441e8c720cc11f044bc2cb7da7f772'and scope['approved_code_preparation_only'],'Exact gallery code-only scope')
 oldp=read(checked(scope['old_visual_protocol']));oldc=read(checked(scope['old_fixed_visual_complete']));checked(scope['old_visual_code'])
 require(oldc['completed']and oldc['PASS']and oldc['code']==scope['old_visual_code']and oldc['protocol']==scope['old_visual_protocol'],'Closed old renderer metadata lineage')
 require(oldp['fixed_prior_selection']==FIXED_SELECTION and oldp['max_unique_current_cases']==68,'Exact same fixed V191 IDs; body unopened')
 require(scope['new_evaluator_code']==EVAL_CODE and scope['new_evaluator_protocol']==EVAL_PROTO,'Prospective evaluator binding')
 newp=read(checked(EVAL_PROTO));checked(EVAL_CODE);require(newp['models']==MODELS,'All10 current evaluator schema')
 tests,card=invented();write(CHECKS,tests);card.save(H/'INVENTED_LAYOUT_ONLY.png')
 schema=dict(completed=True,prepared_only=True,models=MODELS,public_owner_models=[m for m in MODELS if m!='legacy'],current_ledger=dict(path='saved_real_head_evaluation_v1/current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl',rows=1784,required_fields=['global_index','id','native_corner','group','human_state','mapping_status','paired_same_query_RGB_D_vs_E','query_available','query_x','query_y','roi_side','initial_reference_error_px','human_reference_x','human_reference_y','annotation_coordinate_source']+[m+'_D_E_U'for m in MODELS]+[m+'_raw_direct_end_external_logit'for m in ['C54','C55','C56','C57','C58','C59']]),current_primary_rows=1237,current_D=1147,current_E=90,current_inputU=45,public_rows=360,public_inputU=108,exact_original_META6=META,selection_count=68,selection_frozen_before_new_prediction_bodies=True,future_output=str(OUT),actual_bodies_opened=0)
 write(SCHEMA,schema)
 p=dict(completed=True,prepared_only=True,code=bind(Path(__file__)),scope=bind(scope_path),plan=oldp['plan'],schema=bind(SCHEMA),checks=bind(CHECKS),inherited_old_visual_code=scope['old_visual_code'],inherited_old_visual_protocol=scope['old_visual_protocol'],inherited_old_visual_complete=scope['old_fixed_visual_complete'],evaluation_code=EVAL_CODE,evaluation_protocol=EVAL_PROTO,output=str(OUT),input_complete_path=str(INPUT_COMPLETE),current_ledger_path=schema['current_ledger']['path'],fixed_prior_selection=FIXED_SELECTION,selection=oldp['selection'],max_unique_current_cases=68,all_models=MODELS,all_rows_tables='Full1784/U45/material/native8 and primary1237 D1147/E90 all10 preserved; self separate paired21/full136. Public360/U108 weak-owner9 legacy unavailable, not native D/E. Existing <=3 reference461 differs manualnear376.',pixels=oldp['pixels'],limits=oldp['limits']+['C58/C59 are both fixed RAW native-conditioned arms. Gallery imports no models and reads only already saved outputs after both prediction/evaluation seal.','Source-held results do not select or replace gallery IDs, models, thresholds or training data.'],actual_before_rootentry=0,model_fit_forward_threshold_or_data_choice=False,selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p);template=H/'ROOT_ACTUAL_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_saved_real_gallery_readout=False,both_predictions_and_saved_evaluation_completed=False,no_model_training_data_architecture_threshold_selection=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],plan=p['plan'],schema=p['schema'],prediction_seal=None,evaluation_complete=None,input_complete=None))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],schema=p['schema'],checks=p['checks'],template=bind(template),invented_layout=bind(H/'INVENTED_LAYOUT_ONLY.png'),actual_saved_result_annotation_array_header_crop_or_weight_reads=0,new_forward_fit=0,automatic_retry=False)
 write(H/'PREPARATION_COMPLETE.json',ready);write(H/'FINAL_READY.json',dict(ready,preparation=bind(H/'PREPARATION_COMPLETE.json')));print(json.dumps(ready))

def run(entry_path):
 e=read(entry_path);require(e['approved_saved_real_gallery_readout'] and e['both_predictions_and_saved_evaluation_completed'] and e['no_model_training_data_architecture_threshold_selection'],'Separate postterminal saved-only visual entry')
 p=read(checked(e['protocol']));require(e['scope']==p['scope']and e['scope']==bind(V/'ROOT_PROSPECTIVE_FIXED68_TEN_MODEL_VISUAL_CODE_PREPARATION_SCOPE_A.json'),'Exact code-only scope lineage');require(e['code']==bind(Path(__file__))==p['code'] and e['plan']==bind(PLAN)==p['plan'] and e['schema']==bind(SCHEMA)==p['schema'] and e['protocol']==bind(PROTOCOL),'Frozen renderer/selection')
 pred=read(checked(e['prediction_seal']));done=read(checked(e['evaluation_complete']));source=read(checked(e['input_complete']))
 require(pred['completed'] and pred['PASS'] and pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'] and done['completed'] and done['PASS'] and done['prediction_seal']==e['prediction_seal'],'Both final predictions + saved evaluation barrier')
 require(done['code']==p['evaluation_code'] and done['protocol']==p['evaluation_protocol'] and source['completed'] and source['PASS'] and source['evaluation_inputs_only'] and e['input_complete']['path']==p['input_complete_path'] and pred['input_complete']==e['input_complete'],'Exact evaluation and original target-free cache')
 require(not OUT.exists(),'Exclusive new output/no retry');OUT.mkdir()
 try:
  ob=done['outputs'];ledger_key='current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl';rows=lines(checked(ob[ledger_key]));current=read(checked(ob['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json']));public=read(checked(ob['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json']))
  ledger_contract(rows)
  require(sum(not yes(r['query_available']) for r in rows)==45 and sum(primary(r) for r in rows)==1237 and sum(primary(r) and r['human_state']=='DIRECT_VISIBLE' for r in rows)==1147 and sum(primary(r) and r['human_state']=='EXTERNAL_OCCLUDED' for r in rows)==90,'Original full/primary denominators')
  prior_selection=read(checked(p['fixed_prior_selection']));cells,selected=frozen_selection(rows,prior_selection);selection=OUT/'SELECTION_BEFORE_INPUT_PIXELS.json';write(selection,dict(completed=True,plan=p['plan'],evaluation_complete=e['evaluation_complete'],ledger=ob[ledger_key],fixed_prior_selection=p['fixed_prior_selection'],cells=cells,unique_original_indices=[r['global_index'] for r in selected],unique_count=len(selected),no_cache_NPZ_bytes_headers_hash_or_pixels_read_yet=True))
  # First cache byte/hash/header access occurs after the selection manifest above.
  record=source['domains']['CURRENT223_1784'];ib=record['inputs']
  with opened(checked(ib)) as f:
   with np.load(f,allow_pickle=False)as z:x={k:z[k] for k in ['rgb96','valid96']+META}
  require(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(1784,96,96,3) and x['valid96'].dtype==bool and x['valid96'].shape==(1784,96,96),'Original96 RGB/valid schema')
  for i,r in enumerate(rows):
   require(str(x['row_id'][i])==r['id']+':native'+str(r['native_corner']) and int(x['original_global_index'][i])==i and int(x['native_slot'][i])==r['native_corner'] and bool(x['original_query_available'][i])==yes(r['query_available']),'Original all1784 META identity')
   q=np.array([r['query_x'],r['query_y']],np.float64);L=np.nan if r['roi_side'] is None else r['roi_side'];require(np.array_equal(q,x['query_xy'][i],equal_nan=True) and np.array_equal(np.asarray(L),x['L'][i],equal_nan=True),'Original F64 q/L')
  images=[];cards=[]
  for r in selected:
   i=r['global_index'];d=OUT/('CASE_%04d'%i);d.mkdir();rgb=x['rgb96'][i];valid=x['valid96'][i];raw=d/'CLEAN_RAW96.png';vp=d/'VALID96.png';cp=d/'CARD.png'
   Image.fromarray(rgb).save(raw);Image.fromarray(valid.astype(np.uint8)*255).save(vp);card=render_card(r,rgb,valid);card.save(cp)
   with Image.open(raw)as a:require(np.array_equal(np.asarray(a),rgb),'Saved clean PNG exact original RGB96')
   with Image.open(vp)as a:require(np.array_equal(np.asarray(a),valid.astype(np.uint8)*255),'Saved valid PNG exact')
   membership=[dict(material=c['material'],cell=c['cell']) for c in cells if i in c['selected_original_global_indices']]
   images.append(dict(original_global_index=i,native_slot=r['native_corner'],id=r['id'],memberships=membership,clean=bind(raw),valid=bind(vp),card=bind(cp),original_RGB96_array_sha256=ash(rgb),original_valid96_array_sha256=ash(valid),clean_pixel_parity=True));cards.append(card)
  contacts=[]
  for j in range(0,len(cards),2):
   tile=Image.new('RGB',(1600,1280*len(cards[j:j+2])),'white')
   for k,card in enumerate(cards[j:j+2]):tile.paste(card,(0,1280*k))
   path=OUT/('CONTACT_%02d.png'%(j//2+1));tile.save(path);contacts.append(bind(path))
  pt=primary_tables(current);nt=inventories(rows);wt=public_table(public);csvfile(OUT/'PUBLIC_WEAK_CENTER_OWNER_ONLY.csv',wt);csvfile(OUT/'PRIMARY_ALL10_CM_PRECISION_RECALL_U.csv',pt);csvfile(OUT/'ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv',nt);csvfile(OUT/'FIXED_CELL_SELECTION.csv',[dict(material=c['material'],cell=c['cell'],eligible_N=c['eligible_N'],selected_N=len(c['selected_original_global_indices']),indices=';'.join(map(str,c['selected_original_global_indices']))) for c in cells]);csvfile(OUT/'SELF_SEPARATE_BINARY_RESPONSES.csv',current['tables']['human_self_binary_response'])
  summary=dict(completed=True,source_evaluation=e['evaluation_complete'],prediction_seal=e['prediction_seal'],all_models=MODELS,current_all1784=dict(N=1784,input_U=45,materials_native8=nt,human_state_counts=dict(Counter(r['human_state'] for r in rows)),mapping_status_counts=dict(Counter(r['mapping_status'] for r in rows))),current_primary=dict(N=1237,D=1147,E=90,rows=pt),self_separate=current['tables']['human_self_binary_response'],public_weak_owner_only=public,public_weak_headline_table=wt,selected_images=len(images),selection=bind(selection),limits=p['limits'],selected_model=None,adopted=False,fully_solved=False)
  write(OUT/'SUMMARY_AND_TABLES.json',summary);write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,images=images,contacts=contacts,actual_tools_view_image='PENDING_SEPARATE_QA',no_all1784_image_inspection_claim=True))
  intro='Current223는 반복 노출된 평가이며 blind 검증이 아니다. legacy realCAL 이력을 보존한다. 새 direct 판단은 기존 인간 D/E와 비교하며 물리 외부/self 원인 인증이 아니다. public360의 약한 center-owner/crop 관찰은 별도다. 모델 채택·전체 목표 해결·새 학습 선택은 없다.'
  md=['# V200 고정 실사 입력 진단','',intro,'','전체1784 / 원 input U45. primary1237(D1147/E90)의 모든10모델 표를 보존했다. 선택은 V191의 동일68 ID를 그대로 사용했으며 새 예측에 따라 바꾸지 않았다. 과거 post-result strata이며 유병률 대표표본이 아니다. D/E cell은 primary paired, U cell은 전체1784 범위다. 빈 cell 대체가 없다. 갤러리의 기존 reference 오차≤3px 집단은 manual-click near376 집단과 다르며, 물리 좌표 gold가 아니다.','', '[전체 요약·public weak·self 표](SUMMARY_AND_TABLES.json) · [primary CM/P/R/U CSV](PRIMARY_ALL10_CM_PRECISION_RECALL_U.csv) · [native/material 전체분모](ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv) · [선택표](FIXED_CELL_SELECTION.csv) · [self 별도](SELF_SEPARATE_BINARY_RESPONSES.csv) · [public weak center-owner만](PUBLIC_WEAK_CENTER_OWNER_ONLY.csv)','',table_md([r for r in pt if r['group']=='ALL']),'','## 고정 선택 전체','']
  body='<h1>V200 fixed original input diagnostics</h1><p>'+html.escape(intro)+'</p><p><a href="SUMMARY_AND_TABLES.json">Full summary and public weak tables</a> | <a href="PRIMARY_ALL10_CM_PRECISION_RECALL_U.csv">Primary CSV</a> | <a href="ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv">Native/material CSV</a> | <a href="SELF_SEPARATE_BINARY_RESPONSES.csv">Self separate CSV</a> | <a href="PUBLIC_WEAK_CENTER_OWNER_ONLY.csv">Public weak center CSV</a></p>'+table_html([r for r in pt if r['group']=='ALL'])+'<h2>All original native/material denominators</h2>'+table_html(nt)+'<h2>Public weak center-owner only (not native D/E)</h2>'+table_html(wt)+'<h2>Fixed cells</h2>'+table_html([dict(material=c['material'],cell=c['cell'],eligible=c['eligible_N'],selected=c['selected_original_global_indices']) for c in cells])
  for rec in images:
   rel=Path(rec['card']['path']).relative_to(OUT).as_posix();clean=Path(rec['clean']['path']).relative_to(OUT).as_posix();i=rec['original_global_index'];md+=['### original %04d'%i,'',f'![original{i} fixed input card]({rel})','',f'[exact clean96]({clean})',''];body+='<h3>original %04d</h3><img width="1500" src="%s"><p><a href="%s">exact clean96</a></p>'%(i,html.escape(rel),html.escape(clean))
  md+=['## Contact sheets','','']
  for contact in contacts:
   rel=Path(contact['path']).relative_to(OUT).as_posix();md += [f'![fixed contact sheet]({rel})','']
  md+=['## 한계','','원 crop을 중심 이동하지 않았다. cyan q cross는 표시 사본에만 있다. 원 reference 오류 <=3/>3은 관측 연관이며 물리 좌표 gold/인과 분리가 아니다. local crop만으로 선택 instance나 실제 가림 원인을 확정하지 않는다. 전체1784개의 그림을 봤다는 주장은 하지 않는다.',''];text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(md));text(OUT/'index.html','<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:7px}img{max-width:100%;height:auto}</style>'+body)
  content=(OUT/'PORTABLE_GALLERY_KO.md').read_text();md_links=check_links(OUT,content);html_links=check_links(OUT,(OUT/'index.html').read_text(),True)
  checked(ib);checked(ob[ledger_key]);write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry_path),selection=bind(selection),summary=bind(OUT/'SUMMARY_AND_TABLES.json'),manifest=bind(OUT/'IMAGE_MANIFEST.json'),gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),contacts=contacts,selected_unique_cases=len(images),all_selected_original_RGB96_valid96_png_parity=True,all1784_META_qL_native_U_preserved=True,portable_local_links=md_links,HTML_local_links=html_links,HTML_browser_rendered=False,actual_image_tool_view='PENDING_SEPARATE_QA',no_new_labels_model_fit_or_selection=True,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
