"""V184 saved-only fixed19 cards and whole-primary scientific figure.
Preparation: source/receipt/case-list metadata and invented arrays only.
Execution: separate root entry after both prediction seal and evaluation complete.
"""
from pathlib import Path
import argparse,hashlib,html,json,math,os,shutil,sys,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json';SCOPE_SHA='05c628ad960a5715fedd96fb4e3cde7b2a67ee3ce188e49eb644867f4e3e0b25'
CLARIFY=V/'ROOT_REFERENCE_AUTHORITY_CLARIFICATION_BEFORE_REAL_BODIES.json';CLARIFY_SHA='3e60202c1a000b2c5837f1fe826f26d05bc1201edc019e8e3ecf37f709f33fe6'
OLD=P/'query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/fixed19_current_crop_review_v1'
OLD_RULES=OLD/'DISPLAY_RULES_AND_INPUT_BINDINGS.json';OLD_RULES_SHA='b8506f70677e8efa9809d4b269464299e77ffca0bf4843110fd67f481c853c87'
OLD_COMPLETE=OLD/'COMPLETE.json';OLD_COMPLETE_SHA='7eb49502b5277beef93b4087ea578830d5cc42286680ee480d6c8c8e22f6da9d'
RAW_COMPLETE=V/'original_real_raw8_inputs_v1/COMPLETE.json';RAW_COMPLETE_SHA='2d90a7863fe80cfce95e3d54ee1ce529b2b800bcc85373cd7b1f0551c5e68a37'
EVAL_CODE=V/'evaluation_preparation_b/evaluate_saved_real_heads.py';EVAL_CODE_SHA='cbe63b53905c7f088a4534c1e6551396472331550f96c0941c9abb8e6b0f6de0'
EVAL_PROTOCOL=V/'evaluation_preparation_b/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json';EVAL_PROTOCOL_SHA='44aae11dccc0b10c07c25c7502e5c54b4cca54ce31d3fab47ecd05790266a586'
PRODUCER_CODE=V/'frozen_prediction_preparation_c/compose_SAM96_and_infer_both_frozen_heads.py';PRODUCER_SHA='ead5654961db0bca0dedb457a9212adc22a348a8b98be9b0abd83c7a15fbeaa0'
PRODUCER_PROTOCOL=V/'frozen_prediction_preparation_c/PROTOCOL_BEFORE_REAL_RAW_SAM_OR_HEAD_WEIGHT_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_REAL_SAVED_LEDGER_OR_DISPLAY_PIXELS.json';CHECKS=H/'GENERATED_MEANINGFUL_CHECKS.json'
OUT=V/'fixed19_real_visual_v1';MODELS=['v61','legacy','C52','C53'];NEW=['C52','C53']
GROUPS={'WOOD_PRIMARY105':'WOOD','SQUARE118':'SQUARE'}
FIXED=[0,5,8,32,56,219,256,296,300,324,401,840,841,858,898,908,914,996,1009]
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';BG='#f5f8fb';INK='#20364c';AMBER='#965111';META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']

def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def safe(p):
 p=Path(p);require('_v112' not in str(p).lower(),'Forbidden path before access');return p
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def checked(b):require(bind(b['path'])=={k:b[k] for k in ['path','sha256']},'Exact binding '+b['path']);return Path(b['path'])
def read(p):return json.loads(safe(p).read_text())
def lines(p):return [json.loads(s) for s in safe(p).read_text().splitlines() if s]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pixsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def font(n):return ImageFont.truetype(FONT,n)
def text(dr,xy,s,size=25,maxw=2025,color=INK,maxh=1450):
 box=dr.textbbox(xy,s,font=font(size));require(box[2]-xy[0]<=maxw and box[3]<maxh,'Text outside fixed readable region: '+s);dr.text(xy,s,font=font(size),fill=color)
def number(x):
 if x is None or x=='':return 'UNKNOWN'
 try:return format(float(x),'.6g') if math.isfinite(float(x)) else 'NaN'
 except (ValueError,TypeError):return str(x)
def vector(v):return '['+', '.join(number(x) for x in v)+']'
def boolvalue(x):return x is True or str(x)=='True'

def validate_ledger(rows):
 require(len(rows)==1784 and [r['global_index'] for r in rows]==list(range(1784)),'Full original ledger1784 order')
 require(len({(r['id'],r['native_corner']) for r in rows})==1784,'Unique original native keys')
 require(sum(not r['query_available'] for r in rows)==45,'Original inputU45')
 for i,r in enumerate(rows):
  require(r['native_corner']==i%8 and r['group'] in GROUPS,'Original slot/material mapping')
  for m in MODELS:require(r[m+'_D_E_U'] in ['D','E','U'],'Saved fixed decoder '+m)
  for m in NEW:
   s=r[m+'_raw_support_logits'];o=r[m+'_raw_owner_logits'];sp=r[m+'_support_prediction'];owner=r[m+'_owner_prediction'];proxy=r[m+'_regional_support_proxy_D0_E1_Uminus1']
   require(len(s)==2 and len(o)==3 and len(sp)==2,'Saved support2/owner3 shape')
   if r['query_available']:
    require(all(x is not None and math.isfinite(x) for x in s+o),'Available finite saved logits')
    require(sp==[int(x>0) for x in s] and owner==max(range(3),key=lambda k:o[k]),'Saved >0 independent support/firsttieowner')
    require(proxy==(1 if sp[1]==1 else 0 if sp[0]==1 else -1),'Saved foreign-priority proxy')
   else:require(s==[None,None] and o==[None,None,None] and sp==[-1,-1] and owner==-1 and proxy==-1,'InputU first rawNaN normalized to null')
   require(r[m+'_D_E_U']=={0:'D',1:'E',-1:'U'}[proxy],'Saved displayed D/E/U equals proxy')

COUNTS=['N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']
def metric_counts(r):
 d={k:int(r[k]) for k in COUNTS};require(d['N']==d['visible_support']+d['occluded_support'],'Fixed metric full denominator')
 require(d['TP']+d['FN_decided']+d['U_visible']+d['out_of_task_S_visible']+d['other_unresolved_visible']==d['visible_support'],'Visible denominator with U')
 require(d['FP']+d['TN']+d['U_occluded']+d['out_of_task_S_occluded']+d['other_unresolved_occluded']==d['occluded_support'],'External denominator with U');return d
def rates(d):
 ratio=lambda a,b:None if b==0 else a/b
 return [ratio(d['TP'],d['TP']+d['FP']),ratio(d['TP'],d['visible_support']),ratio(d['TN'],d['TN']+d['FN_decided']),ratio(d['TN'],d['occluded_support']),ratio(d['N']-d['TP']-d['FN_decided']-d['FP']-d['TN'],d['N'])]
def primary_metrics(report):
 rows=report['tables']['metrics'];out={}
 for m in MODELS:
  for group in GROUPS:
   match=[r for r in rows if r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E' and r['group']==group and r['model']==m]
   require(len(match)==1,'Unique fixed primary saved metrics');d=metric_counts(match[0]);values=rates(d)
   for key,v in zip(['visible_precision','visible_recall_all_GT','occluded_precision','occluded_recall_all_GT','unresolved_rate'],values):require(match[0][key]==v,'Published saved count/rate closure '+key)
   out[m,group]=d
  out[m,'ALL']={k:sum(out[m,g][k] for g in GROUPS) for k in COUNTS}
  require(out[m,'ALL']['N']==1237 and out[m,'ALL']['visible_support']==1147 and out[m,'ALL']['occluded_support']==90,'Unchanged allprimary1237 D1147/E90')
 return out

def plot_primary(data,path):
 labels=['D precision','D recall','E precision','E recall','Unresolved U'];colors=['#738596','#ad875e','#377ca5','#c45550'];fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True,layout='constrained')
 for ax,group in zip(axes,['ALL',*GROUPS]):
  xx=np.arange(5);width=.19
  for j,m in enumerate(MODELS):
   vv=rates(data[m,group]);ax.bar(xx+(j-1.5)*width,[np.nan if x is None else x*100 for x in vv],width,label=m,color=colors[j])
  d=data[MODELS[0],group];ax.set_title(f'{GROUPS.get(group,group)}: fixed primary N={d["N"]}, human D={d["visible_support"]}, E={d["occluded_support"]}',loc='left');ax.set_ylim(0,105);ax.set_ylabel('Percent');ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.set_xticks(xx,labels)
 axes[0].legend(ncol=4,loc='upper right');fig.suptitle('Frozen real benchmark: original human semantic D/E\nBoth new heads fixed before real evaluation; repeated development exposure',fontsize=16)
 fig.supxlabel('All truth rows retained; unresolved includes abstention / geometry / operator U. Regional proxy is a different training target.',fontsize=11);fig.savefig(path,dpi=150);plt.close(fig)

def draw_card(r,context,raw,c32,valid,case_order):
 board=Image.new('RGB',(2080,1450),BG);dr=ImageDraw.Draw(board);g=r['global_index']
 text(dr,(24,15),f'V184 fixed real case {case_order+1}/19 · g{g} · {r["group"]} · native{r["native_corner"]}',32)
 text(dr,(24,64),f'Original ID: {r["id"]}',25)
 text(dr,(24,106),f'Human semantic: {r["human_state"]} | input available: {r["query_available"]} | original q={vector([r["query_x"],r["query_y"]])}, L={number(r["roi_side"])}',24)
 text(dr,(24,150),'Original saved context (display overlay)',22,maxw=1050);text(dr,(1120,150),'Clean96 ×4 nearest',22,maxw=400);text(dr,(1595,150),'Central32 ×8 nearest',22,maxw=460)
 board.paste(context,(24,202));board.paste(raw.resize((384,384),Image.Resampling.NEAREST),(1120,202));board.paste(c32.resize((256,256),Image.Resampling.NEAREST),(1595,202))
 text(dr,(1885,165),'Center2 ×64',17,maxw=180);board.paste(raw.crop((47,47,49,49)).resize((128,128),Image.Resampling.NEAREST),(1885,202))
 text(dr,(1885,370),'Center16 ×8',17,maxw=180);board.paste(raw.crop((40,40,56,56)).resize((128,128),Image.Resampling.NEAREST),(1885,412))
 text(dr,(1120,610),'Original valid96 ×2 nearest',21,maxw=500);board.paste(valid.resize((192,192),Image.Resampling.NEAREST).convert('RGB'),(1120,650))
 text(dr,(1350,665),'Display-only enlargements;',22,maxw=700);text(dr,(1350,702),'clean pixels remain unchanged.',22,maxw=700)
 text(dr,(24,860),'Model',24,maxw=150);text(dr,(195,860),'D/E/U',24,maxw=150);text(dr,(395,860),'Support [V,F]',24,maxw=270);text(dr,(690,860),'Owner',24,maxw=200);text(dr,(915,860),'Raw support logits',24,maxw=380);text(dr,(1365,860),'Raw owner logits [T,F,BG]',24,maxw=680)
 for j,m in enumerate(MODELS):
  yy=915+j*57;text(dr,(24,yy),m,28,maxw=150);text(dr,(195,yy),r[m+'_D_E_U'],28,maxw=150)
  sp=r[m+'_support_prediction'] if m in NEW else None;owner=r[m+'_owner_prediction'] if m in NEW else None
  text(dr,(395,yy),vector(sp) if sp is not None else 'different task',25,maxw=280);text(dr,(690,yy),{-1:'U',0:'T',1:'F',2:'BG'}[owner] if m in NEW else 'not owner3',25,maxw=220)
  text(dr,(915,yy),vector(r[m+'_raw_support_logits']) if m in NEW else '—',24,maxw=420);text(dr,(1365,yy),vector(r[m+'_raw_owner_logits']) if m in NEW else '—',24,maxw=680)
 text(dr,(24,1168),f'Existing reference error: {number(r["initial_reference_error_px"])} px | authority: {r.get("annotation_coordinate_source","UNKNOWN")}',23)
 text(dr,(24,1214),'Reference distance is a frozen proxy association; no q/L change or new point-quality gold.',23,color=AMBER)
 text(dr,(24,1260),'19 cases were selected by historical outcomes. No new-score selection; not a representative sample.',23,color=AMBER)
 text(dr,(24,1306),'New supports are independent (>0), BOTH/NONE retained. Proxy: foreign YES→E, else visible YES→D, else U.',23)
 text(dr,(24,1352),'Owner T/F/BG is separate from human semantic visibility; F/BG alone does not establish native external occlusion.',23)
 text(dr,(24,1398),'Saved RGB/valid are original model inputs. Context overlay and enlargement panels are display only.',23)
 return board

def external90(rows):
 selected=[r for r in rows if boolvalue(r['paired_same_query_RGB_D_vs_E']) and r['human_state']=='EXTERNAL_OCCLUDED']
 require(len(selected)==90 and [r['global_index'] for r in selected]==sorted(r['global_index'] for r in selected),'All original primaryE90 ascending, no prediction-based selection');return selected
def errorbin(r):
 try:x=float(r['initial_reference_error_px'])
 except (ValueError,TypeError):return 'missing ref'
 return ('ref<=3' if x<=3 else 'ref>3') if math.isfinite(x) else 'missing ref'
def contact_page(rows,images,page):
 require(len(rows)==len(images)==30,'Exactly30 original E rows/contact');board=Image.new('RGB',(2400,2070),BG);dr=ImageDraw.Draw(board)
 text(dr,(24,15),f'V184 all original primary external-E cases · page {page+1}/3 · 30/90',30,maxw=2350,maxh=2070)
 text(dr,(24,62),'Original human semantic E cohort, ascending global index; no selection by new predictions. Clean96 ×3 nearest.',23,maxw=2350,maxh=2070)
 for j,(r,raw) in enumerate(zip(rows,images)):
  x=(j%6)*400;y=110+(j//6)*390
  text(dr,(x+10,y),f'g{r["global_index"]} {GROUPS[r["group"]]} n{r["native_corner"]}',21,maxw=380,maxh=2070)
  ident=str(r['id']);require(len(ident)<=100,'Original ID unexpectedly long for fixed contact layout')
  for ln in range((len(ident)+49)//50):text(dr,(x+10,y+29+ln*14),ident[ln*50:(ln+1)*50],13,maxw=380,maxh=2070)
  board.paste(raw.resize((288,288),Image.Resampling.NEAREST),(x+56,y+62))
  s52=r['C52_support_prediction'];s53=r['C53_support_prediction'];text(dr,(x+10,y+352),f'C52 {r["C52_D_E_U"]} V{s52[0]}F{s52[1]} | C53 {r["C53_D_E_U"]} V{s53[0]}F{s53[1]} | {errorbin(r)}',16,maxw=385,maxh=2070)
 return board

def checks():
 passed=[]
 def rejects(fn):
  try:fn()
  except AssertionError:return True
  return False
 r=dict(global_index=0,id='INVENTED_CASE',native_corner=0,group='WOOD_PRIMARY105',human_state='DIRECT_VISIBLE',query_available=True,query_x=1.25,query_y=2.75,roi_side=96.,initial_reference_error_px='3',annotation_coordinate_source='manual_click',paired_same_query_RGB_D_vs_E='True')
 for m in MODELS:r[m+'_D_E_U']='D'
 for m in NEW:r.update({m+'_raw_support_logits':[1.,0.],m+'_raw_owner_logits':[0.,0.,0.],m+'_support_prediction':[1,0],m+'_owner_prediction':0,m+'_regional_support_proxy_D0_E1_Uminus1':0})
 raw=Image.fromarray(np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3));c32=raw.crop((32,32,64,64));valid=Image.new('L',(96,96),255);context=Image.new('RGB',(1050,590),'#dae1e8');board=draw_card(r,context,raw,c32,valid,0)
 require(board.crop((1120,202,1504,586)).tobytes()==raw.resize((384,384),Image.Resampling.NEAREST).tobytes() and board.crop((1595,202,1851,458)).tobytes()==c32.resize((256,256),Image.Resampling.NEAREST).tobytes(),'Actual card layout cleanRGB nearest ROIs exact');passed.append('Invented actualcard alltext bounds and raw96/central32 nearest pixels exact')
 require(board.crop((24,202,1074,792)).tobytes()==context.tobytes(),'Exact saved context ROI');passed.append('Context display copied without sourceRGB reads or modelinput overlay')
 d=dict(N=10,visible_support=7,occluded_support=3,TP=4,FN_decided=1,FP=1,TN=1,U_visible=1,U_occluded=1,out_of_task_S_visible=0,out_of_task_S_occluded=0,other_unresolved_visible=1,other_unresolved_occluded=0);metric_counts(d)
 require(rates(d)==[4/5,4/7,1/2,1/3,3/10] and rejects(lambda:metric_counts(dict(d,N=9))),'Unknown/unresolved counted in denominator');passed.append('Independent saved metric count closure preserves allU and rejects denominator drift')
 toy=[]
 for g in GROUPS:
  for m in MODELS:toy.append(dict(d,cohort='paired_same_query_RGB',task='D_vs_E',group=g,model=m,**dict(zip(['visible_precision','visible_recall_all_GT','occluded_precision','occluded_recall_all_GT','unresolved_rate'],rates(d)))))
 require(rejects(lambda:primary_metrics(dict(tables=dict(metrics=toy)))),'Fixed1237 guard');passed.append('Wrong wholeprimary denominator rejected rather than relabeled1237')
 require(rejects(lambda:validate_ledger([r])),'All1784 guard');passed.append('Incomplete saved ledger rejected before case rendering')
 toyrows=[]
 for i in range(1784):
  row=dict(r,global_index=i,id='INVENTED_FRAME_'+str(i//8),native_corner=i%8,query_available=i>=45)
  for m in MODELS:row[m+'_D_E_U']='D' if i>=45 else 'U'
  for m in NEW:
   if i<45:row.update({m+'_raw_support_logits':[None,None],m+'_raw_owner_logits':[None,None,None],m+'_support_prediction':[-1,-1],m+'_owner_prediction':-1,m+'_regional_support_proxy_D0_E1_Uminus1':-1})
  toyrows.append(row)
 validate_ledger(toyrows);swapped=list(toyrows);swapped[0],swapped[1]=swapped[1],swapped[0]
 require(rejects(lambda:validate_ledger(swapped)),'Order drift rejected');passed.append('Full invented1784 savedrow Ufirst/sign/argmax/proxy guard passes and swappedorder rejects')
 full=[]
 for group,n,dn,en in [('WOOD_PRIMARY105',700,650,50),('SQUARE118',537,497,40)]:
  for m in MODELS:
   dd=dict(N=n,visible_support=dn,occluded_support=en,TP=dn-4,FN_decided=2,FP=2,TN=en-4,U_visible=1,U_occluded=1,out_of_task_S_visible=0,out_of_task_S_occluded=0,other_unresolved_visible=1,other_unresolved_occluded=1);metric_counts(dd)
   full.append(dict(dd,cohort='paired_same_query_RGB',task='D_vs_E',group=group,model=m,**dict(zip(['visible_precision','visible_recall_all_GT','occluded_precision','occluded_recall_all_GT','unresolved_rate'],rates(dd)))))
 mm=primary_metrics(dict(tables=dict(metrics=full)));plot_primary(mm,H/'INVENTED_PLOT_ONLY.png');passed.append('Actual scientific plot adapter runs on invented count-consistent1237 only')
 ee=[dict(r,global_index=i,id='INVENTED_'+str(i),human_state='EXTERNAL_OCCLUDED') for i in range(90)];require([z['global_index'] for z in external90(ee)]==list(range(90)) and rejects(lambda:external90(ee[:-1])),'All90 cohort retained')
 sheet=contact_page(ee[:30],[raw]*30,0);require(sheet.crop((56,172,344,460)).tobytes()==raw.resize((288,288),Image.Resampling.NEAREST).tobytes(),'Contact rawRGB3x nearest');sheet.save(H/'INVENTED_ALL90_CONTACT_LAYOUT_ONLY.png');passed.append('All90 predeclared truth cohort and actual6x5 contact layout guards pass, no prediction selection')
 board.save(H/'INVENTED_LAYOUT_ONLY.png');return passed

def prepare():
 for p,s in [(SCOPE,SCOPE_SHA),(CLARIFY,CLARIFY_SHA),(OLD_RULES,OLD_RULES_SHA),(OLD_COMPLETE,OLD_COMPLETE_SHA),(RAW_COMPLETE,RAW_COMPLETE_SHA),(EVAL_CODE,EVAL_CODE_SHA),(EVAL_PROTOCOL,EVAL_PROTOCOL_SHA),(PRODUCER_CODE,PRODUCER_SHA)]:require(sha(p)==s,'Pinned allowed metadata/code '+str(p))
 old=read(OLD_RULES);c=read(OLD_COMPLETE);require(old['fixed19']==FIXED and c['fixed19_indices']==FIXED and c['selected_cases']==19 and c['completed'] and c['PASS'],'Allold19 exact order/outcome-oldselected')
 cc=checks();write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),checks=cc,count=len(cc),actual_private_public_held_pixels_annotations_scores_or_NPZ_body_reads=0))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),old_display_rules=bind(OLD_RULES),old_complete=bind(OLD_COMPLETE),old_source_render=old['inputs']['original_render'],original_raw_complete=bind(RAW_COMPLETE),evaluator_code=bind(EVAL_CODE),evaluator_protocol=bind(EVAL_PROTOCOL),prediction_producer_code=bind(PRODUCER_CODE),prediction_producer_protocol=bind(PRODUCER_PROTOCOL),checks=bind(CHECKS),fixed_case_ids=FIXED,output=str(OUT),
  case_policy='All original V17819 in exactorder, inherited V169 historical outcome-selected illustrations, notrepresentative; no newoutcome selection.',
  pixels='Reuse existing clean96/central32/valid96 PNG by exact encoded-byte copy plus decodedRGB/hash/subset parity. Context only existing V178 card ROI. If cleanPNG unavailable, extract originalV184raw8 unchanged; never resamplecrop/sourcefullRGB. Enlargements NEAREST; contextoverlay is displaycopy only.',
  model_display='v61/legacy/C52/C53 D/E/U from complete savedledger; new2support bits/logits andowner3 distinct. Humansemantic/referenceerror label outside cleanpixels; no nativecause inference fromownerF/BG.',
  figure='Dynamic savedprimary metrics only: count-derived ALL1237 D1147E90 and WOOD/SQUARE; all4models D/Eprecision/recall plus allunresolvedU. Publicassistantweakowner separately linked/table, never pooled with humansemantic.',
  external90_census='All original paired_same_query_RGB_D_vs_E and humanstateEXTERNAL_OCCLUDED rows, exactly90, originalglobalindex ascending,3pages x30(6columns x5rows). Savedraw8 RGB96 exactextraction only; eachtilecaptionmaterial/ID/native/C52C53proxyand2support/referencebin outsidecleanpixels. No outcome selection, no recrops.',
  runtime_contract='Rootentry only after bothpredictionseal plus savedrealevaluation COMPLETE/PASS exactclosure. Reads savedCSV/JSONL/JSON and savedPNG, optionalraw8 cleanextraction only; no sourcefullRGB/weights/model/cropsampling/fit/threshold/labelchanges.',
  validation='PNG parity automated all19; actualtoolview separately pending. HTML source/links cancheck offline, no browserrender claim.',selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p);template=H/'ROOT_ACTUAL_FIXED19_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_saved_real_visual=False,evaluation_only=True,code=bind(__file__),protocol=bind(PROTOCOL),prediction_seal=None,evaluation_complete=None))
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_real_pixel_annotation_score_NPZ_body_reads=0));print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))

def run(entry):
 e=read(entry);require(e.get('approved_saved_real_visual') is True and e.get('evaluation_only') is True,'Separate root actual visual entry')
 require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Frozen visual source/protocol');p=read(PROTOCOL)
 for key in ['scope','authority_clarification','old_display_rules','old_complete','original_raw_complete','evaluator_code','evaluator_protocol','prediction_producer_code','prediction_producer_protocol','checks']:checked(p[key])
 pred=read(checked(e['prediction_seal']));ev=read(checked(e['evaluation_complete']))
 require(pred['completed'] and pred['PASS'] and pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'] and pred['code']==p['prediction_producer_code'] and pred['protocol']==p['prediction_producer_protocol'],'Both exact fixed realpredictions closed')
 require(ev['completed'] and ev['PASS'] and ev['prediction_seal']==e['prediction_seal'] and ev['code']==p['evaluator_code'] and ev['protocol']==p['evaluator_protocol'] and ev['selected_model'] is None and not ev['adopted'],'Closed exact savedevaluation, noselection')
 require(not OUT.exists(),'Preserve all attempts, no overwrite or automaticretry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),both_prediction_and_saved_evaluation_closed=True))
 try:
  outputs=ev['outputs'];get=lambda name:checked(outputs[name]);report=read(get('current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json'));rows=lines(get('current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl'));validate_ledger(rows);metrics=primary_metrics(report)
  external=external90(rows);selection90=OUT/'ALL90_EXTERNAL_E_CASE_IDS_BEFORE_IMAGE_READS.json';write(selection90,dict(completed=True,original_primary_external_E90=True,global_indices=[r['global_index'] for r in external],selection='Original fixed paired primary humanEXTERNAL_OCCLUDED cohort, ascendingglobalindex, all90, no predictions used'))
  old=read(checked(p['old_complete']));source=read(checked(p['old_source_render']));panels={r['global_index']:r for r in source['panels']};require(set(panels)==set(FIXED),'Original exact19 cleanpanel inventory')
  contexts={int(Path(b['path']).name.split('_')[1]):b for b in old['cards']};require(set(contexts)==set(FIXED),'OriginalV178 contextcard19 inventory');records=[];rawsource=None;rawcomplete=read(checked(p['original_raw_complete']))
  body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V184 고정19 실사 비교</title><style>body{font:18px system-ui;max-width:2080px;margin:auto;padding:24px;background:#f5f8fb;color:#20364c}img{width:100%}article{margin:40px 0}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>고정19 사례 · 전체1,237 주평가와 구분</h1><p>과거 결과로 선택했던 19건을 같은 순서로 전부 재사용했습니다. 새 점수 기반 사례 선택은 없으며 대표 표본이 아닙니다. 현재223과 공개held45는 이전 개발 노출이 있어 새 독립 검증이 아닙니다.</p><p>사람 semantic D/E/S 평가 authority, 지원부위 두 목표, literal 점 owner, 기존 reference 오차를 구분합니다. 모델 선택·새문턱·라벨교정·학습편입은 없습니다.</p><p>원 clean96/중앙32/valid PNG와 원 context display만 재사용했습니다. 확대는 nearest, clean input에는 overlay를 넣지 않았습니다. HTML browser 렌더링 검증은 주장하지 않습니다.</p><h2>전체 고정 주평가</h2><img src="FIXED_PRIMARY_REAL_METRICS.png"><p>원 1,237건(D1,147/E90), U 포함 분모 유지. 공개 assistantweak owner와 합산하지 않습니다.</p>']
  for order,i in enumerate(FIXED):
   r=rows[i];a=panels[i];require(a['id']==r['id'] and a['native_corner']==r['native_corner'] and a['material']==GROUPS[r['group']],'Same original case/material/native')
   require(a['original_query_xy']==[r['query_x'],r['query_y']] and a['original_L']==r['roi_side'],'No query/L correction or newcrop');clean={};parity={}
   for key in ['raw96','central32','valid96']:
    target=OUT/f'CASE_{i:04d}_{key.upper()}_CLEAN.png'
    if Path(a[key]['path']).is_file():shutil.copy2(checked(a[key]),target);require(bind(target)['sha256']==a[key]['sha256'],'CleanPNG encoded-byte copy')
    else:
     if rawsource is None:
      b=rawcomplete['domains']['CURRENT223_1784']['inputs']
      with np.load(checked(b),allow_pickle=False) as z:rawsource={k:z[k] for k in ['rgb96','valid96',*META]}
     require(int(rawsource['original_global_index'][i])==i and int(rawsource['native_slot'][i])==r['native_corner'] and np.array_equal(rawsource['query_xy'][i],np.array([r['query_x'],r['query_y']],np.float64),equal_nan=True) and np.array_equal(np.array(rawsource['L'][i]),np.array(np.nan if r['roi_side'] is None else r['roi_side']),equal_nan=True) and bool(rawsource['original_query_available'][i])==r['query_available'],'Unchanged raw8 extraction identity/qL/availability')
     arr=rawsource['rgb96'][i] if key=='raw96' else rawsource['rgb96'][i,32:64,32:64] if key=='central32' else rawsource['valid96'][i].astype(np.uint8)*255;Image.fromarray(arr).save(target)
    clean[key]=bind(target)
   raw=Image.open(clean['raw96']['path']).convert('RGB');c32=Image.open(clean['central32']['path']).convert('RGB');valid=Image.open(clean['valid96']['path']).convert('L');require(raw.size==(96,96) and c32.size==(32,32) and valid.size==(96,96),'Original raw/c32/valid dimensions')
   require(pixsha(np.array(raw))==a['RGB96_HWC_sha256'] and pixsha(np.array(c32))==a['central32_HWC_sha256'] and raw.crop((32,32,64,64)).tobytes()==c32.tobytes(),'All19 decodedclean hashes and exactcentral32 subset')
   oldcard=Image.open(checked(contexts[i])).convert('RGB');context=oldcard.crop((24,202,1074,792));board=draw_card(r,context,raw,c32,valid,order);card=OUT/f'CASE_{i:04d}_CARD.png';board.save(card)
   with Image.open(card) as saved:
    parity=dict(context_display_exact=saved.crop((24,202,1074,792)).tobytes()==context.tobytes(),raw96_NEAREST_exact=saved.crop((1120,202,1504,586)).tobytes()==raw.resize((384,384),Image.Resampling.NEAREST).tobytes(),central32_NEAREST_exact=saved.crop((1595,202,1851,458)).tobytes()==c32.resize((256,256),Image.Resampling.NEAREST).tobytes(),valid96_NEAREST_exact=saved.crop((1120,650,1312,842)).convert('L').tobytes()==valid.resize((192,192),Image.Resampling.NEAREST).tobytes())
   require(all(parity.values()),'All actual19 pixel display parity');records.append(dict(global_index=i,id=r['id'],native_slot=r['native_corner'],group=r['group'],old_original_context=contexts[i],old_clean_sources={k:a[k] for k in clean},clean=clean,card=bind(card),pixel_parity=parity,selection='historical outcome-selected fixed19, notrepresentative',training_admission=False))
   body.append(f'<article id="g{i}"><h2>{order+1}/19 · g{i} · {html.escape(r["id"])} · native{r["native_corner"]}</h2><img src="{card.name}"><p>'+ ' · '.join(f'<a href="{Path(b["path"]).name}">{k} clean</a>' for k,b in clean.items())+'</p></article>')
  if rawsource is None:
   b=rawcomplete['domains']['CURRENT223_1784']['inputs']
   with np.load(checked(b),allow_pickle=False) as z:
    require(set(z.files)=={'rgb96','valid96',*META},'Only original target-free raw8 cache for all90')
    rawsource={k:z[k] for k in z.files}
  require(rawsource['rgb96'].dtype==np.uint8 and rawsource['rgb96'].shape==(1784,96,96,3) and np.array_equal(rawsource['original_global_index'],np.arange(1784,dtype=np.int64)),'All90 original RGB96 source identity/schema')
  all90=[];all90images=[]
  for r in external:
   i=r['global_index'];require(int(rawsource['native_slot'][i])==r['native_corner'] and np.array_equal(rawsource['query_xy'][i],np.array([r['query_x'],r['query_y']],np.float64),equal_nan=True) and bool(rawsource['original_query_available'][i])==r['query_available'] and np.array_equal(np.array(rawsource['L'][i]),np.array(np.nan if r['roi_side'] is None else r['roi_side']),equal_nan=True),'All90 source originalq/L/native/availability preserved')
   arr=rawsource['rgb96'][i];image=Image.fromarray(arr);cp=OUT/f'EXTERNAL90_G{i:04d}_CLEAN96.png';image.save(cp)
   with Image.open(cp) as saved:require(np.array_equal(np.array(saved.convert('RGB')),arr),'All90 cleanPNG decoded bytes exact')
   all90images.append(image);all90.append(dict(global_index=i,id=r['id'],native_slot=r['native_corner'],group=r['group'],original_query_available=r['query_available'],clean=bind(cp),decoded_RGB_sha256=pixsha(arr),selection='all original primaryE, no prediction-based selection'))
  sheets=[]
  for page in range(3):
   rr=external[page*30:(page+1)*30];im=all90images[page*30:(page+1)*30];sheet=contact_page(rr,im,page);sp=OUT/f'ALL90_EXTERNAL_E_CONTACT_{page+1:02d}.png';sheet.save(sp)
   with Image.open(sp) as saved:
    for j,clean in enumerate(im):
     x=(j%6)*400+56;y=110+(j//6)*390+62;require(saved.crop((x,y,x+288,y+288)).tobytes()==clean.resize((288,288),Image.Resampling.NEAREST).tobytes(),'All90 contact NEAREST pixels exact')
   sheets.append(bind(sp))
  write(OUT/'ALL90_EXTERNAL_E_DISPLAY_RECORDS.json',all90)
  body.insert(5,'<h2>기존 주평가 external E 전체 90건</h2><p>예측으로 선택하지 않고, 기존 사람 semantic E 90개 행 전체를 global index 오름차순으로 표시합니다. 각 raw96은 기존 캐시에서 변경 없이 추출하며 nearest ×3으로 확대합니다.</p>'+''.join(f'<p><a href="{Path(b["path"]).name}">page {j+1}/3</a></p><img src="{Path(b["path"]).name}">' for j,b in enumerate(sheets)))
  fig=OUT/'FIXED_PRIMARY_REAL_METRICS.png';plot_primary(metrics,fig);write(OUT/'FIXED_PRIMARY_PLOT_COUNTS.json',dict(rows=[dict(model=m,group=g,counts=d,rates=rates(d)) for (m,g),d in metrics.items()],primary1237=True))
  public=get('publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json');publicrel=os.path.relpath(public,OUT);weak=read(public);known=weak['cohorts']['KNOWN_CENTER_TRUTH']['ALL'];require(known['N']==204 and weak['coverage']['unknown_center156']==156,'Separate original weakowner denominators')
  weak_table=['<h2>공개 held45 · 별도 assistantweak 중심 owner 진단</h2><p>단일 영상 45프레임/360슬롯, known204/U156. 지원부위·native 외부가림 정답이 아니며 위 human semantic 지표에 합산하지 않습니다. legacy 동일입력 출력은 미확인으로 표시하지 않습니다.</p><table><tr><th>모델</th><th>T precision</th><th>T recall (U 포함)</th><th>NON-T recall</th><th>known truth의 prediction U</th></tr>']
  percent=lambda x:'—' if x is None else f'{100*x:.2f}%'
  for m in ['v61','C52','C53']:
   wm=known['models'][m]['binary_T_nonT'];t=wm['per_class']['T'];nt=wm['per_class']['NON_T'];weak_table.append('<tr>'+''.join('<td>'+str(x)+'</td>' for x in [m,percent(t['precision_on_known_truth']),percent(t['recall_including_prediction_U']),percent(nt['recall_including_prediction_U']),wm['prediction_U_on_known_truth']])+'</tr>')
  weak_table.append('</table><p><a href="'+html.escape(publicrel)+'">공개 weak owner/crop 전체 원기록</a></p>');body.insert(5,''.join(weak_table));body.append('</html>');index=OUT/'index.html';index.write_text('\n'.join(body));write(OUT/'ALL19_DISPLAY_RECORDS.json',records)
  for b in [e['prediction_seal'],e['evaluation_complete'],p['old_source_render'],p['old_complete']]:checked(b)
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),prediction_seal=e['prediction_seal'],evaluation_complete=e['evaluation_complete'],fixed_case_ids=FIXED,selected_cases=19,all_original_order_preserved=True,records=bind(OUT/'ALL19_DISPLAY_RECORDS.json'),figure=bind(fig),plot_counts=bind(OUT/'FIXED_PRIMARY_PLOT_COUNTS.json'),index=bind(index),cards=[r['card'] for r in records],all19_clean_and_display_pixel_parity=True,external_E90_selection=bind(selection90),external_E90_records=bind(OUT/'ALL90_EXTERNAL_E_DISPLAY_RECORDS.json'),external_E90_contact_sheets=sheets,external_E90_clean_and_NEAREST_pixel_parity=True,wholeprimary_N1237=True,public_weak_metrics_separate=True,actual_visual_inspection='pending separate tool-view receipt',HTML_browser_rendered=False,new_source_fullRGB_crop_sampler_model_fit_threshold_or_label_changes=0,selected_model=None,adopted=False,fully_solved=False))
  print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),figure=bind(fig),index=bind(index))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
