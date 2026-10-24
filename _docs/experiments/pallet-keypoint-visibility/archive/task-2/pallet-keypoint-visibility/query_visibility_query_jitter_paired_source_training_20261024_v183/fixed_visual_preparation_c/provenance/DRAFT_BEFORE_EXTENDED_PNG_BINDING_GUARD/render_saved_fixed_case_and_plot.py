"""V183 fixed old-error illustration and saved-metric scientific plot only.

Prepare: code/metadata and invented arrays only. Actual: separate root entry
AFTER both new heads freeze and saved evaluation completes. No new crop sampling,
source RGB, model, checkpoint, GT buffer, initializer, fitting or label creation.
"""
from pathlib import Path
import argparse,hashlib,io,json,math,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json';SCOPE_SHA='d8394deb1c0a5844ecb987447391d7fd61ce5aa80d073cf33adf76ff385fcb95'
OLD=P/'query_visibility_source_same_query_causal_diagnosis_20261024_v182/crop_geometry_visual_v1/COMPLETE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_SAVED_OUTPUT_OR_PIXEL_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'fixed_source_case_and_plot_v1'
OLD_ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
NEW_ARMS=['C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE']
PLOT_ARMS=[OLD_ARMS[0],NEW_ARMS[0],OLD_ARMS[1],NEW_ARMS[1]]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLUMNS=[dict(title='BEFORE: original production q/L',newdomain='SYNTH_AUG65536',newindex=28335,original_global_index=1889),dict(title='SAME BEFORE: common COVER q/L',newdomain='COMMON4096',newindex=1889,original_global_index=1889),dict(title='COVER image: common q/L',newdomain='COMMON4096',newindex=1897,original_global_index=1897)]


def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def sha(p):
 p=Path(p).resolve();require(p.is_relative_to(P) and not any(s in str(p).lower() for s in ['_v112','_v169','_v177','_v178','current223','private_evaluation']),'Forbidden source path')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):require(bind(b['path'])==b,'Exact sealed binding '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def font(n):return ImageFont.truetype(FONT,n)
def wrapped(d,xy,text,width,size=19):
 x,y=xy;f=font(size);line=''
 for word in str(text).split():
  new=word if not line else line+' '+word
  if line and d.textlength(new,font=f)>width:d.text((x,y),line,font=f,fill='#dce7f5');y+=size+6;line=word
  else:line=new
 if line:d.text((x,y),line,font=f,fill='#dce7f5');y+=size+6
 return y


def card_image(columns):
 im=Image.new('RGB',(2100,1510),'#0f192a');d=ImageDraw.Draw(im)
 wrapped(d,(25,18),'Fixed BASE59 / native1: same saved crops, old and new frozen heads',2050,32)
 wrapped(d,(25,65),'Selected from an OLD C50 source error before NEW outcomes. Error illustration, not a representative success rate.',2050,22)
 wrapped(d,(25,107),'New own nominal row28335 = original1889 x15; COMMON rows1889 and1897. q/L, and SAM prompt, can differ; no isolated q-only causal claim.',2050,20)
 for j,c in enumerate(columns):
  x=25+j*695;y=170
  y=wrapped(d,(x,y),c['title'],650,24)
  wrapped(d,(x,220),'q=(%.6f, %.6f); L=%.6f'%(*c['q'],c['L']),650,18)
  im.paste(Image.fromarray(c['RGB']).resize((384,384),Image.Resampling.NEAREST),(x+135,267))
  wrapped(d,(x+135,662),'Unmodified saved RGB96 x4 nearest',500,18)
  center=c['RGB'][32:64,32:64]
  im.paste(Image.fromarray(center).resize((256,256),Image.Resampling.NEAREST),(x+199,708))
  wrapped(d,(x+110,975),'Exact central32 [32:64] x8 nearest',560,18)
  y=1024
  for arm in OLD_ARMS+NEW_ARMS:
   a=c['models'][arm];tag=arm[:3]
   y=wrapped(d,(x+9,y),tag+' target(V,F)='+str(a['target'])+'; prediction='+str(a['prediction']),660,18)
   y=wrapped(d,(x+9,y),'logits(V,F)=[%.6f, %.6f]; regional proxy=%d'%(*a['logits'],a['proxy']),660,17)+8
 wrapped(d,(25,1387),'V/F are two independent experimental source supports, not certified physical native visibility. Changed or UNKNOWN new targets are shown as stored; they are not forced equal to old targets.',2050,19)
 wrapped(d,(25,1450),'RGB is the saved model input. Enlargements and text are display only. No source RGB reread, new crop sampling, model call, label correction, or model selection.',2050,18)
 return im


def plot_values(metrics):
 result=[]
 for arm in PLOT_ARMS:
  pair=metrics['COMMON_matched_pairs'][arm]['BEFORE_to_COVER']['joint_visible_only_to_foreign_only']
  g=metrics['original_fixed401_geometry_sensitivity'][arm];eligible=g['fixed_old401_current_eligible']
  val=metrics['results'][arm]['SYNTH_VAL2048']['ALL']
  support={}
  for key in ['visible_end_support','foreign_front_support']:
   z=val[key]['primary_known_available'];support[key]=dict(precision=z['precision'][1],recall=z['recall'][1],known=z['truth_known'],class_support=z['class_support'],prediction_U=z['prediction_U_among_known'])
  result.append(dict(arm=arm,pair=pair,fixed401=g,eligible=eligible,VAL_known_available=support))
 return result


def scientific_plot(values):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
 fig,axes=plt.subplots(2,2,figsize=(14,10));colors=['#5480b5','#d28236','#5480b5','#d28236'];x=np.arange(4);tags=[r['arm'][:3] for r in values]
 ax=axes[0,0];den=[r['pair']['pairs'] for r in values];num=[r['pair']['both_supports_correct_at_both_images'] for r in values]
 require(len(set(den))==1,'Same fixed COMMON pair target denominator across all4 models')
 rate=[np.nan if n==0 else c/n*100 for c,n in zip(num,den)];ax.bar(x,rate,color=colors)
 for i,r in enumerate(values):ax.text(i,(0 if math.isnan(rate[i]) else rate[i])+2,'%d/%d\navailable %d'%(num[i],den[i],r['pair']['both_available']),ha='center',fontsize=9)
 ax.set(title='A. COMMON: both supports correct at both paired images',ylabel='% of known canonical BEFORE->COVER pairs',ylim=(0,112),xticks=x,xticklabels=tags)
 ax=axes[0,1]
 for i,r in enumerate(values):
  z=r['eligible'];n=z['rows'];yy=[np.nan if n==0 else z[k]/n*100 for k in ['nominal_correct','common_correct']]
  ax.plot([0,1],yy,'o-',label='%s n=%d'%(tags[i],n),color=colors[i],linestyle='--' if i>=2 else '-')
 ax.set(title='B. Fixed old401 same-image geometry sensitivity',ylabel='% both supports correct (currently eligible subset)',xticks=[0,1],xticklabels=['Own nominal q/L','Common COVER q/L'],ylim=(0,105));ax.legend(fontsize=9)
 for ax,key,title in [(axes[1,0],'visible_end_support','C. Reused synthetic VAL: visible-end support'),(axes[1,1],'foreign_front_support','D. Reused synthetic VAL: foreign-front support')]:
  pp=[r['VAL_known_available'][key]['precision'] for r in values];rr=[r['VAL_known_available'][key]['recall'] for r in values]
  ax.bar(x-.18,[np.nan if z is None else z*100 for z in pp],.36,label='Positive precision',color='#517ea9')
  ax.bar(x+.18,[np.nan if z is None else z*100 for z in rr],.36,label='Positive recall',color='#d08438')
  ax.set(title=title,ylabel='% (known support and original input available)',xticks=x,xticklabels=tags,ylim=(0,106));ax.legend(fontsize=9)
  for i,r in enumerate(values):ax.text(i,3,'NO/YES\n'+str(r['VAL_known_available'][key]['class_support']),ha='center',fontsize=8,color='#30343b')
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('Frozen source-only comparison: RAW C50->C52, SAM C51->C53',fontsize=16,y=.98)
 fig.text(.02,.022,'COMMON and old401 are source TRAIN-derived diagnostics. VAL2048 is historically reused, not blind natural-real validation.\nFixed401 excludes new unknown/changed truth from eligible scoring and retains those counts in JSON. No native/physical gold or model adoption is claimed.',fontsize=10)
 fig.tight_layout(rect=(0,.075,1,.95));return fig


def invented():
 checks=[]
 def passed(name,ok):require(ok,name);checks.append(dict(name=name,PASS=True))
 a=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);before=a.copy()
 columns=[dict(title=c['title'],q=[1.,2.],L=4.,RGB=a.copy(),models={arm:dict(logits=[.25,-.5],prediction=[1,0],target=[1,0],proxy=0) for arm in OLD_ARMS+NEW_ARMS}) for c in COLUMNS]
 im=card_image(columns);passed('Three-column PIL layout fixed and no source pixel mutation',im.size==(2100,1510) and np.array_equal(a,before))
 for j in range(3):
  x=25+j*695+135;passed('Clean96 nearest4 display panel exact column'+str(j),np.array_equal(np.asarray(im)[267:651,x:x+384],np.repeat(np.repeat(a,4,0),4,1)))
 stream=io.BytesIO();Image.fromarray(a).save(stream,format='PNG');stream.seek(0);back=np.asarray(Image.open(stream).convert('RGB'))
 passed('CleanPNG decoded RGB exact and exact central32 slice',np.array_equal(back,a) and np.array_equal(back[32:64,32:64],a[32:64,32:64]))
 values=[dict(arm=arm,pair=dict(pairs=10,both_supports_correct_at_both_images=3,both_available=9),fixed401={},eligible=dict(rows=8,nominal_correct=6,common_correct=5),VAL_known_available={k:dict(precision=.6,recall=.5,class_support=[4,5],known=9,prediction_U=0) for k in ['visible_end_support','foreign_front_support']}) for arm in PLOT_ARMS]
 fig=scientific_plot(values);stream=io.BytesIO();fig.savefig(stream,format='png',dpi=70);plt.close(fig);stream.seek(0)
 passed('Actual scientific plot renders invented counts with explicit denominator and source caveats',Image.open(stream).size==(980,700))
 passed('Fixed ownnominal mapping original1889times15 and fixed COMMON columns',COLUMNS[0]['newindex']==1889*15 and [c['newindex'] for c in COLUMNS[1:]]==[1889,1897])
 return checks


def prepare():
 require(sha(SCOPE)==SCOPE_SHA,'Frozen source scope')
 old=read(OLD);require(old['completed'] and old['PASS'] and old['outcome_selected_error_illustration'],'Historical COMPLETE metadata only')
 checks=invented();write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),checks=checks,actual_saved_RGB_score_target_or_metric_bodies_read=0))
 protocol=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),checks=bind(CHECKS),old_fixed_visual_complete_metadata=bind(OLD),columns=COLUMNS,output=str(OUT),
 fixed_case='BASE59/native1 selected using old C50 lost event before new outcomes; outcome-selected error illustration, not representative success or a new selection by results.',
 pixels='Only existing V1823clean96 and3central32 PNGs, SHA and decoded-array SHA; copies byteexact, display enlarged nearest. No source RGB, sourcebuffers, new crops or GTinput.',
 facts='Old C50/C51 facts from sealed V182 case; new C52/C53 raw saved support logits/pred and evaluator-ledger targets. New changed/UNKNOWN target not forced equal.',
 plot='Four panels: dynamic known canonical COMMON pair numerator/denominator, fixedold401 eligible nominal/common sensitivity with changed/unknown retained, historicallyreusedVAL knownavailable visible and foreign positive P/R.',
 root_entry='Separate both-final8000 fit COMPLETE plus saved evaluation COMPLETE required before any pixel/score/target/metric body.',
 actual_PNG_visual_QA='Pending separate root actual view after execution',browser_render_verified=False,no_new_model_or_fit_or_label_or_threshold_or_sampler=True,selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,protocol)
 template=H/'ROOT_ACTUAL_SAVED_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_saved_fixed_source_visual=False,source_only=True,all_two_models_frozen=False,code=bind(__file__),protocol=bind(PROTOCOL),fit_complete=None,evaluation_complete=None,output_dir=str(OUT)))
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_new_model_output_or_pixel_bodies_read=0,actual_model_or_render=0))
 print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))


def run(entry_path):
 e=read(entry_path);require(e.get('approved_saved_fixed_source_visual') is True and e.get('source_only') is True and e.get('all_two_models_frozen') is True,'Root postfreeze saved visual entry')
 require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Frozen sourcevisual code/protocol');p=read(PROTOCOL)
 require(p['scope']==bind(SCOPE) and sha(SCOPE)==SCOPE_SHA and p['checks']==bind(CHECKS) and p['old_fixed_visual_complete_metadata']==bind(OLD),'Frozen rules/old fixed case')
 fit=read(checked(e['fit_complete']));ev=read(checked(e['evaluation_complete']))
 require(fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_evaluation_access'] and fit['updates_each']==8000 and fit['total_updates']==16000 and set(fit['arms'])==set(NEW_ARMS),'Both new heads fixed final8000')
 require(ev['completed'] and ev['PASS'] and ev['fit_complete']==e['fit_complete'],'Closed saved evaluation of these two heads')
 old=read(OLD);facts=read(checked(old['facts']));require(len(facts['columns'])==3,'Three original saved source case columns')
 selected=facts['selected_case']['selected'];require((selected['base_index'],selected['native_slot'],selected['diagnostic_row_index'],selected['anchor_original_global_index'])==(59,1,1889,1897),'Fixed case only')
 out=Path(e['output_dir']);require(out==OUT and not out.exists(),'New fixed visual output only');out.mkdir()
 # One pass over the already saved evaluation ledger, never target or RGB caches.
 wants={(a,c['newdomain'],c['newindex']) for a in NEW_ARMS for c in COLUMNS}
 ledger={}
 with checked(ev['rows']).open() as f:
  for line in f:
   r=json.loads(line);key=(r['arm'],r['domain'],r['row_index'])
   if key in wants:require(key not in ledger,'Unique fixed model/domain/row');ledger[key]=r
 require(set(ledger)==wants,'All six new saved fixed-case target/pred rows')
 score={}
 keys=['row_id','original_global_index','native_slot','query_xy','L','original_query_available','raw_support_logits','support_prediction','regional_support_proxy_D0_E1_Uminus1','support_order']
 for arm in NEW_ARMS:
  for domain in ['SYNTH_AUG65536','COMMON4096']:
   with np.load(checked(fit['arms'][arm]['outputs'][domain]),allow_pickle=False) as z:score[(arm,domain)]={k:z[k] for k in keys}
 columns=[];published=[]
 for j,c in enumerate(COLUMNS):
  source=facts['columns'][j];rgb=np.asarray(Image.open(checked(source['clean96'])).convert('RGB'));center=np.asarray(Image.open(checked(source['central32'])).convert('RGB'))
  require(rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and ash(rgb)==source['clean96']['decoded_array_sha256'] and np.array_equal(center,rgb[32:64,32:64]) and ash(center)==source['central32']['decoded_array_sha256'],'Original clean96/central32 pixel hashes exact')
  prefix='COLUMN%d'%(j+1);shutil.copyfile(source['clean96']['path'],out/(prefix+'_CLEAN96.png'));shutil.copyfile(source['central32']['path'],out/(prefix+'_CENTRAL32.png'))
  require(sha(out/(prefix+'_CLEAN96.png'))==source['clean96']['sha256'] and sha(out/(prefix+'_CENTRAL32.png'))==source['central32']['sha256'],'PNG source bytes preserved')
  models={arm:dict(source['models'][arm],target=source['support_target']) for arm in OLD_ARMS}
  for arm in NEW_ARMS:
   a=score[(arm,c['newdomain'])];i=c['newindex'];r=ledger[(arm,c['newdomain'],i)]
   require(int(a['original_global_index'][i])==i and int(a['native_slot'][i])==1 and str(a['row_id'][i])==r['row_id'] and r['original_global_index']==i,'New fixed row/native identity')
   require(np.array_equal(a['query_xy'][i],np.asarray(source['query_xy'],np.float64)) and float(a['L'][i])==source['L'] and bool(a['original_query_available'][i])==source['original_query_available'],'Original saved crop q/L/availability preserved')
   require(a['support_order'].tolist()==['visible_end_support','foreign_front_support'],'Support task order')
   raw=a['raw_support_logits'][i];pred=a['support_prediction'][i];require(raw.dtype==np.float32 and np.isfinite(raw).all() and np.array_equal((raw>0).astype(np.int8),pred),'Saved FP32 strict>0 decoder')
   require(pred.tolist()==r['support_prediction'],'Exact saved evaluator prediction')
   proxy=1 if pred[1]==1 else 0 if pred[0]==1 else -1;require(proxy==int(a['regional_support_proxy_D0_E1_Uminus1'][i]),'Saved diagnostic proxy')
   models[arm]=dict(logits=raw.tolist(),prediction=pred.tolist(),target=r['support_truth'],proxy=proxy)
  columns.append(dict(title=c['title'],q=source['query_xy'],L=source['L'],RGB=rgb,models=models))
  published.append(dict(column=j+1,selection=c,q=source['query_xy'],L=source['L'],models=models,clean96=bind(out/(prefix+'_CLEAN96.png')),central32=bind(out/(prefix+'_CENTRAL32.png')),decoded_RGB96_sha256=ash(rgb),source_clean96=source['clean96'],source_central32=source['central32']))
 card_image(columns).save(out/'CARD.png')
 metrics=read(checked(ev['metrics']));values=plot_values(metrics);fig=scientific_plot(values);fig.savefig(out/'SCIENTIFIC_SOURCE_COMPARISON.png',dpi=160);plt.close(fig)
 write(out/'DISPLAY_FACTS_AND_PLOT_DATA.json',dict(fixed_old_case=True,outcome_selected_old_error_illustration=True,not_representative=True,columns=published,plot_data=values,saved_evaluation_metrics=ev['metrics'],no_new_source_query_sampler_model_or_labels=True,selected_model=None,fully_solved=False))
 page='<!doctype html><meta charset="utf-8"><title>Fixed source case and saved comparison</title><style>body{font:18px/1.6 sans-serif;margin:24px;max-width:1700px;background:#101b2d;color:#e2ebf6}img{max-width:100%;height:auto}a{color:#9cd0ff}</style><h1>Fixed source error case and frozen comparisons</h1><p>BASE59/native1 was selected from an old C50 source error before the new outcomes. It is an error illustration, not a representative success rate. All clean96/central32 PNG bytes are copied unchanged from the earlier saved case. No new crop sampling or model call.</p><a href="CARD.png"><img src="CARD.png"></a><h2>Source diagnostics and reused synthetic validation</h2><p>COMMON and fixedold401 are TRAIN-derived diagnostics. VAL2048 is historically reused, not blind natural-real validation. Independent source support targets and literal owner are not certified physical native visibility. No model adoption or full-goal solution is claimed.</p><a href="SCIENTIFIC_SOURCE_COMPARISON.png"><img src="SCIENTIFIC_SOURCE_COMPARISON.png"></a><p><a href="DISPLAY_FACTS_AND_PLOT_DATA.json">Exact saved facts, targets, plot counts and pixel hashes</a></p>'
 (out/'index.html').write_text(page)
 write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),fit_complete=e['fit_complete'],evaluation_complete=e['evaluation_complete'],old_case_complete=bind(OLD),card=bind(out/'CARD.png'),scientific_plot=bind(out/'SCIENTIFIC_SOURCE_COMPARISON.png'),display_facts=bind(out/'DISPLAY_FACTS_AND_PLOT_DATA.json'),index=bind(out/'index.html'),clean96_count=3,central32_count=3,all6PNG_sourcebyte_and_decodedpixel_parity=True,actual_PNG_view_QA='PENDING separate actual view_image',browser_render_verified=False,new_source_RGB_read_crop_sampling_model_forward_fit_or_label_creation=0,selected_model=None,adopted=False,fully_solved=False))
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(out/'COMPLETE.json'))))


if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.entry)
