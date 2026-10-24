"""Saved-only V131 three-model diagnosis. No model import, forward or GT before seals.

--prepare reads code/metadata and invented arrays only. Root owns --entry.
"""
from pathlib import Path
import argparse,csv,hashlib,html,json,sys,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SMALL_QUERY_DIAGNOSIS_POLICY_BEFORE_INPUTS.json'
PROTOCOL=H/'SAVED_EVALUATION_PROTOCOL.json';GENERATED=H/'GENERATED_CHECKS.json'
OUT=V/'saved_three_model_evaluation_v1';MODELS=('C16','C23','v61')
OFFSETS=np.array([[-3,-3],[0,-3],[3,-3],[-3,0],[0,0],[3,0],[-3,3],[0,3],[3,3]],np.float64)
AXIS=(1,3,5,7);DIAGONAL=(0,2,6,8)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float):assert np.isfinite(x)
 return x
def write(p,x):
 with Path(p).open('x')as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ratio(a,b):return float(a/b)if b else None
def decode(z,a,m):return np.where(a,np.where(z>=0,0 if m=='v61'else 1,1 if m=='v61'else 0),-1).astype(np.int8)
def metric(pred,y,state,mask):
 known=np.asarray(mask,bool)&(y>=0)
 cm=np.array([[np.sum(known&(y==r)&(pred==c))for c in (0,1,-1)]for r in (0,1)],np.int64)
 return {'CM':cm,'rows':['D','EXTERNAL_E'],'columns':['D','E','prediction_U'],'planned_rows':int(np.sum(mask)),'known_DE':int(known.sum()),'D_precision':ratio(cm[0,0],cm[:,0].sum()),'D_recall':ratio(cm[0,0],cm[0].sum()),'E_precision':ratio(cm[1,1],cm[:,1].sum()),'E_recall':ratio(cm[1,1],cm[1].sum()),'known_prediction_U':int(cm[:,2].sum()),'known_prediction_U_rate':ratio(cm[:,2].sum(),cm.sum()),'teacher_state_counts':{s:int(np.sum(mask&(state==s)))for s in 'DESU'}}
def authorize(entry):
 e=read(entry);assert e['approved_saved_three_model_evaluation']is True and e['source_only']is True
 assert all(e[k]is False for k in ('model_forward','fit','current223','V112','PnP','automatic_retry'))
 assert not OUT.exists(),'Preserve prior attempt; no automatic restart.'
 for k in ('code','protocol','generated','paired_prediction_seal','v61_prediction_seal','input_complete','input_manifest'):check(e[k])
 p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['generated']==p['generated']
 assert read(e['generated']['path'])['PASS']and read(e['generated']['path'])['code']==e['code'];check(p['policy'])
 a=read(e['paired_prediction_seal']['path']);b=read(e['v61_prediction_seal']['path'])
 assert all(s['completed']and s['PASS']and s['rows']==918 for s in (a,b))
 assert a['input_complete']==e['input_complete']and a['input_manifest']==e['input_manifest']and a['GT_read']is False and a['threshold_logit']==0
 assert set(a['models'])=={'C16','C23'}
 assert all(a['models'][m]['head_state_unchanged']and a['models'][m]['repeat_exact']for m in ('C16','C23'))
 assert b['GT_body_or_hash_read']is False and b['threshold_visible_logit']==0 and b['head_state_unchanged']and b['backbone_state_unchanged']and b['head_repeat_exact']
 assert b['all102_centre_ROI_logit_availability_exact']and len(b['corpora'])==1
 d=b['corpora'][0];assert d==dict(name='PUBLIC_SMALL_JITTER',expected_rows=918,complete=e['input_complete'],input_rows=e['input_manifest'])
 for s in (a,b):
  cp=read(check(s['centre_parity']));assert cp['completed']and cp['PASS']
 done=read(e['input_complete']['path']);assert done['completed']and done['PASS']and done['input_manifest']==e['input_manifest']
 # Both complete prediction seals and exact input equality are established above.
 # Only below this line are GT or prediction/crop payloads hashed/read.
 assert done['GT_only']==e['GT_rows'];check(e['GT_rows'])
 for m in ('C16','C23'):check(a['models'][m]['predictions'])
 check(b['predictions'])
 for x in p['wide_parent_results'].values():check(x)
 return e,p,a,b
def inputs_and_GT(e):
 a=read(e['input_manifest']['path']);g=read(e['GT_rows']['path']);rows=a['rows'];gt=g['rows']
 assert a['completed']and g['completed']and g['input_row_identity']==e['input_manifest']and len(rows)==len(gt)==918
 assert len({r['id']for r in rows})==918
 for i,(r,t)in enumerate(zip(rows,gt)):
  assert r['occurrence_index']==t['occurrence_index']==i and r['id']==t['id']and r['split']==t['split']
  assert r['base_index']==i//153 and r['condition_index']==(i%153)//17 and r['carrier_slot']==0
  assert np.array_equal(r['offset_xy'],OFFSETS[r['condition_index']])
  assert np.array_equal(r['query_xy'],np.asarray(r['initial_points'])[0])
  assert r['query_available']==(r['factual_query_available']and r['context_available'])
  assert t['raw_state']in 'DESU'and t['external_target_D0_E1_else_minus1']=={'D':0,'E':1}.get(t['raw_state'],-1)
  assert t['human_gold']is False and t['training_admission']is False
 assert sum(r['split']=='TRAIN'for r in rows)==612 and sum(r['split']=='VALIDATION'for r in rows)==306
 return rows,gt
def scores(a,b,rows):
 pred={};raw={};avail={};N=len(rows);ix=np.arange(N);ids=np.array([r['id']for r in rows]);q=np.array([r['query_xy']for r in rows],np.float64);L=np.array([r['L']for r in rows],np.float64)
 factual=np.array([r['query_available']for r in rows],bool);native=np.array([r['original_input_availability']['continuous']and r['original_input_availability']['nearest']for r in rows],bool);roi=np.array([r['original_input_availability']['ROI_any_valid']for r in rows],bool)
 for m in MODELS:
  binding=b['predictions']if m=='v61'else a['models'][m]['predictions']
  with np.load(binding['path'],allow_pickle=False)as f:
   for k,value in [('global_index',ix),('source_occurrence_index',ix),('occurrence_id',ids),('query_xy',q),('L',L)]:assert np.array_equal(f[k],value),(m,k)
   assert np.all(f['corpus']=='PUBLIC_SMALL_JITTER')
   raw[m]=np.array(f['raw_visible_logit'if m=='v61'else'raw_head_logit_before_U'],copy=True);avail[m]=np.array(f['query_available'],copy=True);pred[m]=np.array(f['prediction_D0_E1_Uminus1'if m=='v61'else'prediction'],copy=True)
   if m=='v61':
    assert np.array_equal(f['raw_hidden_logit'],-raw[m])and np.array_equal(f['native_query_valid'],native)and np.array_equal(f['ROI_any_valid'],roi)and np.array_equal(f['shared_C16_query_available'],factual)
   else:assert np.array_equal(f['query_logit'],np.where(avail[m],raw[m],0))
  assert avail[m].dtype==bool and raw[m].dtype==np.float32 and pred[m].dtype==np.int8 and raw[m].shape==avail[m].shape==pred[m].shape==(N,)and np.isfinite(raw[m]).all()
  assert np.array_equal(avail[m],native&roi if m=='v61'else factual)
  assert np.array_equal(pred[m],decode(raw[m],avail[m],m))
 return pred,raw,avail
def masks(rows):
 c=np.array([r['condition_index']for r in rows]);fold=np.array([r['split']for r in rows]);cue=np.array([r['base_index']for r in rows]);g={'ALL':np.ones(len(rows),bool)}
 for key,mask in [('CENTRE_0PX',c==4),('AXIS_3PX',np.isin(c,AXIS)),('DIAGONAL_SQRT18PX',np.isin(c,DIAGONAL)),('LE3_INCLUDES_CENTRE_5CONDITIONS',np.isin(c,(1,3,4,5,7)))]:g[key]=mask
 for j in range(9):g[f'condition/{j}']=c==j
 for j in range(6):g[f'cue/{j}']=cue==j
 for variant in sorted({r['variant']for r in rows}):g['variant/'+variant]=np.array([r['variant']==variant for r in rows])
 for f in ('TRAIN','VALIDATION'):
  for key,mask in list(g.copy().items()):
   if not key.startswith('fold/'):g[f'fold/{f}/{key}']=mask&(fold==f)
 return g
def triples_and_scenes(rows,gt,pred,raw,avail):
 triples=[];scenes=[]
 for cue in range(6):
  for condition in range(9):
   for rank in range(8):
    ii=[cue*153+condition*17, cue*153+condition*17+1+2*rank,cue*153+condition*17+2+2*rank]
    for k in ('query_xy','initial_points','L','context_box_xyxy'):assert all(rows[i][k]==rows[ii[0]][k]for i in ii)
    assert [rows[i]['variant']for i in ii]==['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']
    state=[gt[i]['raw_state']for i in ii];known=state==['D','E','D'];r=rows[ii[0]]
    triples.append(dict(cue=cue,condition_index=condition,donor_rank=rank,split=r['split'],indices=ii,states=state,known_D_E_D=known,models={m:{'prediction':pred[m][ii],'raw_logit':raw[m][ii],'all3_available':bool(avail[m][ii].all()),'all3_correct':bool(known and np.array_equal(pred[m][ii],[0,1,0]))}for m in MODELS}))
  for role in range(17):
   ii=[cue*153+j*17+role for j in range(9)];centre=ii[4];st=gt[centre]['raw_state'];assert all(gt[i]['raw_state']==st for i in ii)
   scenes.append(dict(scene_index=rows[centre]['scene_index'],cue=cue,role=role,variant=rows[centre]['variant'],donor_rank=rows[centre]['donor_rank'],split=rows[centre]['split'],centre_index=centre,indices=ii,state=st,models={m:{'predictions':pred[m][ii],'centre_correct':bool(st in'DE'and pred[m][centre]=={'D':0,'E':1}[st]),'axis_changed_from_centre':sum(pred[m][ii[j]]!=pred[m][centre]for j in AXIS),'axis_wrong_after_correct_centre':bool(st in'DE'and pred[m][centre]=={'D':0,'E':1}[st]and any(pred[m][ii[j]]!=pred[m][centre]for j in AXIS)),'all_axis_stable':bool(all(pred[m][ii[j]]==pred[m][centre]for j in AXIS)),'centre_raw_logit':float(raw[m][centre])}for m in MODELS}))
 def summary(rr):
  known=[r for r in rr if r['known_D_E_D']]
  return {'planned_triplets':len(rr),'known_D_E_D':len(known),'models':{m:{'all3_correct':sum(r['models'][m]['all3_correct']for r in known),'all3_available':sum(r['models'][m]['all3_available']for r in known)}for m in MODELS}}
 ts={'ALL':summary(triples)}
 for f in ('TRAIN','VALIDATION'):
  rr=[r for r in triples if r['split']==f];ts[f]=summary(rr)
  for j in range(9):ts[f'fold/{f}/condition/{j}']=summary([r for r in rr if r['condition_index']==j])
 flip={f:{m:{'scenes':sum(r['split']==f for r in scenes),'known_DE_scenes':sum(r['split']==f and r['state']in'DE'for r in scenes),'correct_centre_scenes':sum(r['split']==f and r['models'][m]['centre_correct']for r in scenes),'correct_centre_then_axis_failure_scenes':sum(r['split']==f and r['models'][m]['axis_wrong_after_correct_centre']for r in scenes),'axis_prediction_flips':int(sum(r['models'][m]['axis_changed_from_centre']for r in scenes if r['split']==f))}for m in MODELS}for f in ('TRAIN','VALIDATION')}
 return triples,ts,scenes,flip
def select_gallery(scenes):
 eligible=[s for s in scenes if s['split']=='VALIDATION'and any(s['models'][m]['axis_wrong_after_correct_centre']for m in MODELS)]
 selected=eligible[:4];missing=[]
 if not eligible:
  for state in 'DE':
   options=[s for s in scenes if s['split']=='VALIDATION'and s['state']==state and s['models']['C23']['centre_correct']and s['models']['C23']['all_axis_stable']]
   if options:selected.append(options[0])
   else:missing.append('FIRST_C23_CENTRE_CORRECT_ALL_AXIS_STABLE_'+state)
 return {'selection_rule':'First up to4 original-order VAL scenes with any-model centre correct then axis failure. If none, first C23 correct/stable D and E, missing explicit. No model/threshold changes.','eligible_VAL_axis_failure_scenes':len(eligible),'selected_scenes':selected,'missing_fallback_roles':missing}
def font(size):
 from PIL import ImageFont
 return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def gallery(rows,gt,selected,pred,raw):
 from PIL import Image,ImageDraw
 figs=[];cache={}
 for selection in selected['selected_scenes']:
  im=Image.new('RGB',(942,884),'#edf3f7');d=ImageDraw.Draw(im);d.text((10,8),f'PUBLIC weak AUX / scene {selection["scene_index"]} / {selection["split"]} / {selection["state"]}',font=font(18),fill='#183345');d.text((10,35),'RAW96 nearest-display / separate marked copy: q yellow, weak reference cyan',font=font(14),fill='#183345')
  for pos,i in enumerate(selection['indices']):
   r=rows[i];b=r['rgb96_chunk'];key=(b['path'],b['sha256'])
   if key not in cache:
    check(b)
    with np.load(b['path'],allow_pickle=False)as f:cache[key]=(f['rgb96'].copy(),f['valid96'].copy(),f['occurrence_index'].copy())
   aa,vv,ix=cache[key];j=r['rgb96_row'];a=aa[j];valid=vv[j]
   assert ix[j]==i and a.dtype==np.uint8 and a.shape==(96,96,3)and valid.dtype==bool and valid.shape==(96,96)and ash(a)==r['rgb96_sha256']and ash(valid)==r['valid96_sha256']
   rawim=Image.fromarray(a).resize((144,144),Image.Resampling.NEAREST);marked=rawim.copy();md=ImageDraw.Draw(marked)
   def cross(xy,color):
    x,y=xy
    if 0<=x<96 and 0<=y<96:
     x=(x+.5)*1.5-.5;y=(y+.5)*1.5-.5;md.line((x-5,y,x+5,y),fill=color,width=2);md.line((x,y-5,x,y+5),fill=color,width=2)
   cross([47.5,47.5],'yellow');cross((np.array(gt[i]['reference_xy'])-np.array(r['query_xy']))*96/r['L']+47.5,'cyan')
   x=8+(pos%3)*312;y=65+(pos//3)*272;im.paste(rawim,(x,y+20));im.paste(marked,(x+151,y+20));d.text((x,y),f'condition{r["condition_index"]} offset{r["offset_xy"]}',font=font(13),fill='#183345')
   for k,m in enumerate(MODELS):d.text((x,y+170+22*k),f'{m}: {int(pred[m][i])} z={raw[m][i]:+.4f}  (0D 1E -1U)',font=font(13),fill='#183345')
  path=OUT/'figures'/f'SCENE_{selection["scene_index"]:03d}_ALL9_RAW_AND_MARKED.png';im.save(path);figs.append(bind(path))
 return figs
def chart(reports):
 from PIL import Image,ImageDraw
 im=Image.new('RGB',(1440,600),'white');d=ImageDraw.Draw(im);d.text((15,10),'Frozen source diagnosis: raw0 unchanged / true D,E rows; prediction D,E,U columns',font=font(20),fill='#183345')
 for fi,fold in enumerate(('TRAIN','VALIDATION')):
  for mi,m in enumerate(MODELS):
   x=15+mi*475;y=55+fi*267;r=reports[f'fold/{fold}/ALL'][m]['full_known_denominator'];d.text((x,y),f'{m} {fold} known{r["known_DE"]}',font=font(18),fill='#183345')
   for a in range(2):
    for b in range(3):
     xx=x+b*108;yy=y+30+a*40;d.rectangle((xx,yy,xx+102,yy+35),fill='#e1edf5');d.text((xx+9,yy+5),str(r['CM'][a][b]),font=font(18),fill='#183345')
   for j,key in enumerate(('CENTRE_0PX','AXIS_3PX','DIAGONAL_SQRT18PX','LE3_INCLUDES_CENTRE_5CONDITIONS')):
    a=reports[f'fold/{fold}/{key}'][m]['full_known_denominator'];fmt=lambda v:'NA'if v is None else f'{100*v:.1f}%';d.text((x,y+122+j*26),f'{key.split("_")[0]} D {fmt(a["D_recall"])} / E {fmt(a["E_recall"])}',font=font(15),fill='#183345')
 path=OUT/'figures/THREE_MODELS_SMALL_JITTER_SUMMARY.png';im.save(path);return bind(path)
def wide_comparison(p,reports):
 a=read(p['wide_parent_results']['C16_v61']['path']);b=read(p['wide_parent_results']['C23']['path']);out={}
 for m in MODELS:
  prior=b['groups']['PUBLIC_AUX::VALIDATION']['all_planned']if m=='C23'else a['semantic_reports']['PUBLIC_AUX/fold/VALIDATION'][m]['own_full_denominator']
  out[m]={'prior_wide_VAL':prior,'small_jitter_VAL':reports['fold/VALIDATION/ALL'][m]['full_known_denominator'],'same918_scenes_andfolds_but_changed_q8_L_context_crops':True}
 return out
def report_html(result,figs):
 def pct(x):return '—'if x is None else f'{100*x:.1f}%'
 table=''
 for fold in ('TRAIN','VALIDATION'):
  for m in MODELS:
   for band in ('ALL','CENTRE_0PX','AXIS_3PX','DIAGONAL_SQRT18PX','LE3_INCLUDES_CENTRE_5CONDITIONS'):
    a=result['reports'][f'fold/{fold}/{band}'][m]['full_known_denominator'];table+=f'<tr><td>{fold}</td><td>{m}</td><td>{band}</td><td>{a["known_DE"]}</td><td>{a["CM"]}</td><td>{pct(a["D_precision"])}</td><td>{pct(a["D_recall"])}</td><td>{pct(a["E_precision"])}</td><td>{pct(a["E_recall"])}</td><td>{a["known_prediction_U"]}</td></tr>'
 images=''.join(f'<figure><a href="figures/{Path(b["path"]).name}"><img src="figures/{Path(b["path"]).name}"></a></figure>'for b in figs)
 txt=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V131 고정 모델 작은 query 이동 진단</title><style>body{{font:16px Arial,sans-serif;line-height:1.65;max-width:1460px;margin:24px auto;padding:0 18px;color:#183345}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:6px;border:1px solid #c8d7e2}}img{{max-width:100%;height:auto}}.note{{padding:16px;background:#fff0d8}}a{{color:#07568b}}</style><h1>공개 실사 6cue: 작은 query 이동에 대한 고정 3모델 진단</h1><p class="note">학습·체크포인트·임계값 변경 없이 저장 예측만 평가했습니다. 모든 query는 source emulation이며 weak conditional AUX입니다. native/human gold나 실제 initializer 정확도 평가가 아닙니다. 전체 목표와 source gate는 그대로이며, 쉬운 부분집합으로 모델을 채택하지 않습니다.</p><p>총918행: TRAIN612 / 내부 VALIDATION306. 원본6프레임·102합성 장면을9번 query 이동한 반복이며 독립918이미지가 아닙니다. VAL은2cue/2프레임이고 case23 take는 개발 노출이 있습니다. 중심0px, 축3px, 대각√18≈4.243px입니다. ≤3px는 중심과 축의5조건을 포함합니다. 원 weakcue uncertainty를 줄이지 않았습니다. teacher U와 입력 불가는 별도이며 recall 분모에는 예측 U를 포함합니다.</p><table><tr><th>fold</th><th>모델</th><th>조건</th><th>known</th><th>CM D/E×D/E/U</th><th>D P</th><th>D R</th><th>E P</th><th>E R</th><th>U</th></tr>{table}</table>{images}<p>raw crop은 수정하지 않았고 q/r 마크는 별도 사본입니다. 시각화는 VAL에서 중심 정답 후 축 방향 실패가 생긴 첫 최대4장면이며, 없을 때만 C23의 첫 D/E 안정 장면을 사용합니다. 이는 실패 원인의 단독 인과 증명이 아닙니다. native zero-error 실패 등 기존 전체 한계는 이 진단으로 없어지지 않습니다.</p><p><a href="SOURCE_RESULTS.json">전체 cue/조건/가용성·기존 wide 비교</a> · <a href="ALL918_SCORES.csv">918행</a> · <a href="ALL432_SAME_QUERY_TRIPLETS.json">동일q8 D/E/D triplets</a> · <a href="SCENE102_CENTRE_AXIS_TRANSITIONS.json">장면별 중심→축 변화</a> · <a href="GALLERY_SELECTION_BEFORE_CROP_READ.json">그림 선택 규칙</a></p></html>'''
 (OUT/'READOUT_KO.html').write_text(txt)
def run(entry):
 e,p,a,b=authorize(entry);rows,gt=inputs_and_GT(e);pred,raw,avail=scores(a,b,rows);states=np.array([g['raw_state']for g in gt]);y=np.array([g['external_target_D0_E1_else_minus1']for g in gt],np.int8);shared=np.logical_and.reduce([avail[m]for m in MODELS]);g=masks(rows)
 OUT.mkdir();(OUT/'figures').mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'both_completed_prediction_seals_and_input_identity_before_GT':True,'model_forward':False})
 try:
  reports={k:{m:{'full_known_denominator':metric(pred[m],y,states,mask),'own_available_subset':metric(pred[m],y,states,mask&avail[m]),'shared_available_subset':metric(pred[m],y,states,mask&shared)}for m in MODELS}for k,mask in g.items()}
  tr,ts,sc,flips=triples_and_scenes(rows,gt,pred,raw,avail);selected=select_gallery(sc)
  write(OUT/'ALL432_SAME_QUERY_TRIPLETS.json',{'completed':True,'summary':ts,'rows':tr});write(OUT/'SCENE102_CENTRE_AXIS_TRANSITIONS.json',{'completed':True,'summary':flips,'rows':sc});write(OUT/'GALLERY_SELECTION_BEFORE_CROP_READ.json',selected)
  figs=[chart(reports)]+gallery(rows,gt,selected,pred,raw)
  result={'completed':True,'source_only':True,'rows':918,'TRAIN_rows':612,'VALIDATION_rows':306,'original_frames_cues':6,'native_fullRGB_scenes':102,'independent_VAL_frames_cues':2,'reports':reports,'same_q_triplets':ts,'centre_axis_transitions':flips,'availability':{m:{'available':int(avail[m].sum()),'factual_prediction_U':int((~avail[m]).sum()),'known_DE_input_U':int(((y>=0)&~avail[m]).sum())}for m in MODELS},'teacher_states':{s:int((states==s).sum())for s in'DESU'},'wide_grid_comparison':wide_comparison(p,reports),'figures':figs,'thresholds_unchanged_zero':True,'C23_application_admitted':False,'whole_application_objective_and_source_gates_unchanged':True,'teacher_U_not_input_availability':True,'native_human_gold':False,'model_fit_current223_V112_PnP_calls':0,'fully_solved':False}
  with(OUT/'ALL918_SCORES.csv').open('x',newline='')as f:
   fields=['occurrence_index','id','cue','scene_index','selection_index','split','variant','donor_rank','condition_index','dx','dy','qx','qy','L','raw_teacher_state','external_target']+[m+'_'+k for m in MODELS for k in ('raw_logit','prediction','available')];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for i,r in enumerate(rows):w.writerow(dict(zip(fields,[i,r['id'],r['base_index'],r['scene_index'],r['selection_index'],r['split'],r['variant'],r['donor_rank'],r['condition_index'],*r['offset_xy'],*r['query_xy'],r['L'],states[i],int(y[i])]+[v for m in MODELS for v in(float(raw[m][i]),int(pred[m][i]),bool(avail[m][i]))])))
  write(OUT/'SOURCE_RESULTS.json',result);report_html(result,figs)
  outputs=[bind(x)for x in sorted(OUT.rglob('*'))if x.is_file()];seal=OUT/'EVALUATION_SEAL.json';write(seal,{'completed':True,'PASS':True,'entry':bind(entry),'code':e['code'],'protocol':e['protocol'],'paired_prediction_seal':e['paired_prediction_seal'],'v61_prediction_seal':e['v61_prediction_seal'],'input_complete':e['input_complete'],'input_manifest':e['input_manifest'],'GT_rows':e['GT_rows'],'outputs':outputs,'all918_retained':True,'no_model_fit_current223_V112_PnP':True,'human_actual_visual_review_not_claimed':True})
  print(json.dumps(clean({'completed':True,'seal':bind(seal),'VAL':{m:reports['fold/VALIDATION/ALL'][m]['full_known_denominator']for m in MODELS},'gallery_scene_count':len(selected['selected_scenes'])})),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
def prepare():
 z=np.array([-1.,-0.,0.,1.]);assert decode(z,np.ones(4,bool),'C23').tolist()==[0,1,1,1]and decode(z,np.ones(4,bool),'v61').tolist()==[1,0,0,0]
 q=np.array([0,1,-1,1],np.int8);y=np.array([0,0,0,1]);m=metric(q,y,np.array(list('DDDE')),np.ones(4,bool));assert m['CM'].tolist()==[[1,1,1],[0,1,0]]and m['D_recall']==1/3
 rr=[{'condition_index':i%9,'split':'TRAIN'if i<9 else'VALIDATION','base_index':i//9,'variant':'BEFORE'}for i in range(18)];gg=masks(rr);assert gg['LE3_INCLUDES_CENTRE_5CONDITIONS'].sum()==10 and gg['AXIS_3PX'].sum()==8 and gg['DIAGONAL_SQRT18PX'].sum()==8
 assert len(np.ravel([[c*153+j*17+k for j in range(9)for k in range(17)]for c in range(6)]))==918
 rr=[];tt=[];pp=[]
 for cue in range(6):
  for condition in range(9):
   for role in range(17):
    state='E'if role%2==1 else'D';i=len(rr)
    rr.append(dict(query_xy=[5.,5.],initial_points=[[5.,5.]]*8,L=16.,context_box_xyxy=[0.,0.,10.,10.],variant='BEFORE'if role==0 else'OPAQUE_COVER'if role%2 else'CUE_PRESERVING_SHAM',split='TRAIN'if cue<4 else'VALIDATION',scene_index=cue*17+role,condition_index=condition,base_index=cue,donor_rank=None if role==0 else(role-1)//2));tt.append({'raw_state':state});pp.append(1 if state=='E'else 0)
 pp=np.array(pp,np.int8);pd={x:pp.copy()for x in MODELS};pd['C23'][4*153+17]=1
 tr,ts,sc,fl=triples_and_scenes(rr,tt,pd,{x:pp.astype(np.float32)for x in MODELS},{x:np.ones(918,bool)for x in MODELS});sel=select_gallery(sc)
 assert len(tr)==432 and len(sc)==102 and ts['TRAIN']['planned_triplets']==288 and ts['VALIDATION']['planned_triplets']==144
 assert len(sel['selected_scenes'])==1 and sel['selected_scenes'][0]['scene_index']==68 and fl['VALIDATION']['C23']['correct_centre_then_axis_failure_scenes']==1
 json.dumps(clean({'int':np.int64(5),'float':np.float32(.5),'bool':np.bool_(True),'CM':m,'tr':tr,'sc':sc,'selection':sel}),allow_nan=False)
 import tempfile
 with tempfile.TemporaryDirectory()as d:
  x=Path(d)/'unapproved.json';x.write_text(json.dumps({'approved_saved_three_model_evaluation':False}))
  try:authorize(x)
  except AssertionError:pass
  else:raise AssertionError('entry failed open')
 write(GENERATED,{'completed':True,'PASS':True,'code':bind(__file__),'checks':['opposite_raw_polarity_signedzero','recall_U_denominator','centre_axis_diagonal_LE3_partition','all918_order','invented432_DED_triplets_102scenes_and_VAL_flip_selection','NumPy_clean_JSON','denied_entry_before_payload'],'actual_prediction_GT_crop_model_current_V112_reads':0})
 wide={'C16_v61':bind(P/'query_visibility_registered_native_diversity_20261023_v129/saved_source_baseline_evaluation_v1/SOURCE_RESULTS.json'),'C23':bind(P/'query_visibility_source_coverage_duration_20261024_v130/runs/C22_C23_FIXED8000_SOURCE_CONTROL/C23_PUBLIC_PLUS_NATIVE_FIXED8000/SOURCE_RESULTS.json')}
 p={'completed':True,'code':bind(__file__),'generated':bind(GENERATED),'policy':bind(POLICY),'wide_parent_results':wide,'output':str(OUT),'rows':918,'models':list(MODELS),'GT_rule':'Both prediction seals completed/PASS/918/state frozen/raw0/centreparity/inputidentity before GT hash/read. Teacher U independent from factual input availability.','metrics':'full known denominator with predictionU; own/shared availability; ALL/TR612/VAL306/cue/condition/centre0-axis3-diagonalSqrt18-LE3(centre included); sameq8 DED triples and102scene centre→axis flips.','gallery_rule':'First up to4 VAL original-order any-model correctcentre→axisfailure scenes; if none, first C23 correct/stable D and E. Lock before stored crop reads. All9 raw crops plus separate q/r marks.','source_emulation_not_initializer':True,'no_threshold_or_arm_or_model_admission':True,'no_model_fit_current_V112':True};write(PROTOCOL,p)
 def placeholder(path):return {'path':str(path),'sha256':'ROOT_FILL_COMPLETED_SHA'}
 t={'approved_saved_three_model_evaluation':False,'source_only':True,'model_forward':False,'fit':False,'current223':False,'V112':False,'PnP':False,'automatic_retry':False,'code':p['code'],'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'paired_prediction_seal':placeholder(V/'frozen_C16_C23_small_jitter_v1/PREDICTION_SEAL.json'),'v61_prediction_seal':placeholder(V/'v61_frozen_source_baseline_v1/PREDICTION_SEAL.json'),'input_complete':placeholder(V/'source_crops_v1/CROP_COMPLETE.json'),'input_manifest':placeholder(V/'source_crops_v1/INPUT_ROWS.json'),'GT_rows':placeholder(V/'source_crops_v1/GT_ROWS.json')};write(H/'ROOT_SAVED_EVAL_ENTRY_TEMPLATE.json',t)
 receipt={'completed':True,'PASS':True,'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'template':bind(H/'ROOT_SAVED_EVAL_ENTRY_TEMPLATE.json'),'command':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).resolve())+' --entry '+str(V/'ROOT_ACTUAL_SAVED_THREE_MODEL_EVAL_ENTRY.json'),'actual_forward_or_prediction_GT_crop_body_reads':0};write(H/'PREPARATION.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use --prepare or root --entry.')
