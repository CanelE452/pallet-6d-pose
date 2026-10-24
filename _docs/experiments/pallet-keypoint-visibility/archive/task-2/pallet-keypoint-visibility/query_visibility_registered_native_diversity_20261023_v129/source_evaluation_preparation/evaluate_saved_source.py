"""Saved-only two-baseline source evaluation; no model or native RGB decoding.

Both completed root-bound prediction seals precede GT joins and prediction/crop
payload access. Import and --prepare use code/metadata and invented arrays only.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,html,json,sys,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent
PROTOCOL=H/'SAVED_EVALUATION_PROTOCOL.json';GENERATED=H/'GENERATED_EVALUATION_CHECKS.json'
OUT=V/'saved_source_baseline_evaluation_v1';MODELS=('C16','v61');EXPECTED={'NATIVE':55296,'PUBLIC_AUX':918}
FACTORS=('geometry_family','aspect','material','asset_seed','camera_pair_deg','signed_quarter','foreign_family','target_role','distance_scale')
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
 if isinstance(x,(tuple,list)):return [clean(v)for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float):assert np.isfinite(x);return x
 return x
def write(p,x):
 with Path(p).open('x')as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ratio(a,b):return float(a/b)if b else None
def metric(pred,y,mask,states):
 mask=np.asarray(mask,bool);known=mask&(y>=0)
 cm=np.array([[np.sum(known&(y==r)&(pred==c))for c in (0,1,-1)]for r in (0,1)],np.int64)
 return {'CM':cm.tolist(),'rows':['D','EXTERNAL_E'],'columns':['D','E','prediction_U'],'planned_rows':int(mask.sum()),'known_DE':int(known.sum()),'D_precision':ratio(cm[0,0],cm[:,0].sum()),'D_recall':ratio(cm[0,0],cm[0].sum()),'E_precision':ratio(cm[1,1],cm[:,1].sum()),'E_recall':ratio(cm[1,1],cm[1].sum()),'known_prediction_U':int(cm[:,2].sum()),'known_prediction_U_rate':ratio(cm[:,2].sum(),cm.sum()),'typed_outside_external_task':{s:{'N':int((mask&(states==s)).sum()),'D':int((mask&(states==s)&(pred==0)).sum()),'E':int((mask&(states==s)&(pred==1)).sum()),'prediction_U':int((mask&(states==s)&(pred==-1)).sum())}for s in ('S','U')}}
def decode(z,a,model):return np.where(a,np.where(z>=0,1 if model=='C16'else 0,0 if model=='C16'else 1),-1).astype(np.int8)
def authorize(entry):
 e=read(entry);assert e['approved_saved_source_evaluation']is True and e['source_only']is True
 assert all(e[k]is False for k in ('model_forward','fit','current223','V112','PnP','automatic_retry'))
 for k in ('code','protocol','generated','C16_prediction_seal','v61_prediction_seal'):check(e[k])
 p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['generated']==p['generated']
 assert read(e['generated']['path'])['PASS']and read(e['generated']['path'])['code']==e['code']
 seals={m:read(e[m+'_prediction_seal']['path'])for m in MODELS}
 assert all(s['completed']and s['PASS']and s['rows']==sum(EXPECTED.values())for s in seals.values())
 assert seals['C16']['head_state_unchanged']and seals['v61']['head_state_unchanged']and seals['v61']['backbone_state_unchanged']
 assert seals['C16']['GT_read']is False and seals['v61']['GT_body_or_hash_read']is False
 assert seals['C16']['threshold_logit']==seals['v61']['threshold_visible_logit']==0
 assert [d['name']for d in e['corpora']]==list(EXPECTED)
 for m,s in seals.items():
  assert len(s['corpora'])==2
  for d,prior in zip(e['corpora'],s['corpora']):
   assert d['name']==prior['name']and d['expected_rows']==prior['expected_rows']==EXPECTED[d['name']]
   assert d['complete']==prior['complete']and d['input_rows']==prior['input_rows']
 # Only now can GT be hashed/read and prediction/crop payloads be opened.
 for d in e['corpora']:
  for k in ('complete','input_rows','GT_rows'):check(d[k])
  done=read(d['complete']['path']);assert done['completed']and done['PASS']and done['input_manifest']==d['input_rows']and done['GT_only']==d['GT_rows']
 for s in seals.values():check(s['predictions'])
 assert not OUT.exists(),'Preserve prior results/failure; no automatic restart.'
 return e,seals
def load_rows(e):
 rows=[];gt=[];corpus=[];local=[]
 for d in e['corpora']:
  a=read(d['input_rows']['path']);b=read(d['GT_rows']['path']);rr=a['rows'];gg=b['rows'];assert a['completed']and b['completed']and b['input_row_identity']==d['input_rows']
  assert len(rr)==len(gg)==d['expected_rows']and len({r['id']for r in rr})==len(rr)
  assert [r['occurrence_index']for r in rr]==[g['occurrence_index']for g in gg]==list(range(len(rr)))
  for r,g in zip(rr,gg):
   assert r['id']==g['id']and r['split']==g['split']and r['variant']in ('BEFORE','COVER','NEAR_SHAM','FAR_SHAM','OPAQUE_COVER','CUE_PRESERVING_SHAM')
   assert g['raw_state']in 'DESU'and g['external_target_D0_E1_else_minus1']==({'D':0,'E':1}.get(g['raw_state'],-1))
   assert g['human_gold']is False and g['training_admission']is False
   assert r['query_available']==(r['factual_query_available']and r['context_available'])
  rows.extend(rr);gt.extend(gg);corpus.extend([d['name']]*len(rr));local.extend(range(len(rr)))
 return rows,gt,np.array(corpus),np.array(local,np.int32)
def load_predictions(seals,rows,corpus,local):
 N=len(rows);ids=np.array([r['id']for r in rows]);q=np.array([r['query_xy']for r in rows],np.float64);L=np.array([r['L']for r in rows],np.float64)
 keys={'C16':('query_logit','raw_head_logit_before_U','prediction','query_available','global_index','corpus','source_occurrence_index'),'v61':('raw_visible_logit','raw_hidden_logit','prediction_D0_E1_Uminus1','query_available','native_query_valid','ROI_any_valid','shared_C16_query_available','global_index','corpus','source_occurrence_index','occurrence_id','global_occurrence_id','query_xy','L')}
 data={}
 for m in MODELS:
  with np.load(seals[m]['predictions']['path'],allow_pickle=False)as f:data[m]={k:np.array(f[k],copy=True)for k in keys[m]}
  d=data[m];assert np.array_equal(d['global_index'],np.arange(N))and np.array_equal(d['corpus'],corpus)and np.array_equal(d['source_occurrence_index'],local)
 a=data['C16'];b=data['v61'];available={'C16':a['query_available'],'v61':b['query_available']};raw={'C16':a['raw_head_logit_before_U'],'v61':b['raw_visible_logit']};pred={'C16':a['prediction'],'v61':b['prediction_D0_E1_Uminus1']}
 assert np.array_equal(b['occurrence_id'],ids)and np.array_equal(b['global_occurrence_id'],np.char.add(np.char.add(corpus,'::'),ids))and np.array_equal(b['query_xy'],q)and np.array_equal(b['L'],L)
 assert np.array_equal(a['query_logit'],np.where(a['query_available'],raw['C16'],0))and np.array_equal(b['raw_hidden_logit'],-raw['v61'])
 expected16=np.array([r['query_available']for r in rows],bool);native=np.array([r['original_input_availability']['continuous']and r['original_input_availability']['nearest']for r in rows],bool);roi=np.array([r['original_input_availability']['ROI_any_valid']for r in rows],bool)
 assert np.array_equal(available['C16'],expected16)and np.array_equal(b['shared_C16_query_available'],expected16)and np.array_equal(b['native_query_valid'],native)and np.array_equal(b['ROI_any_valid'],roi)and np.array_equal(available['v61'],native&roi)
 for m in MODELS:
  assert available[m].shape==raw[m].shape==pred[m].shape==(N,)and available[m].dtype==bool and raw[m].dtype==np.float32 and pred[m].dtype==np.int8 and np.isfinite(raw[m]).all()
  assert np.array_equal(pred[m],decode(raw[m],available[m],m))
 return pred,raw,available
def groups(rows,gt,corpus):
 g={'ALL':np.ones(len(rows),bool)};error=np.array([np.nan if r['error_px']is None else r['error_px']for r in gt]);g['error/0']=np.isfinite(error)&(error==0);g['error/(0,3]']=np.isfinite(error)&(error>0)&(error<=3);g['error/>3']=np.isfinite(error)&(error>3);g['error/missing']=~np.isfinite(error)
 assert np.array([g[k]for k in ('error/0','error/(0,3]','error/>3','error/missing')]).sum(0).tolist()==[1]*len(rows)
 for name in EXPECTED:
  mask=corpus==name;g[name]=mask
  for key in ('split','variant','condition_id','logical_slot','cue_index','donor_rank','selection_index','source_frame_group'):
   values={json.dumps(r[key],sort_keys=True)for i,r in enumerate(rows)if mask[i]and key in r}
   for val in sorted(values):
    selected=mask&np.array([json.dumps(r.get(key),sort_keys=True)==val for r in rows]);g[f'{name}/{key}/{val}']=selected
    for fold in ('TRAIN','VALIDATION'):g[f'{name}/fold/{fold}/{key}/{val}']=selected&np.array([r['split']==fold for r in rows])
  for fold in ('TRAIN','VALIDATION'):g[f'{name}/fold/{fold}']=mask&np.array([r['split']==fold for r in rows])
  if name=='NATIVE':
   for key in FACTORS:
    values={json.dumps(r['factors'][key],sort_keys=True)for i,r in enumerate(rows)if mask[i]}
    for val in sorted(values):
     selected=mask&np.array([json.dumps(r.get('factors',{}).get(key),sort_keys=True)==val for r in rows]);g[f'{name}/factor/{key}/{val}']=selected
     for fold in ('TRAIN','VALIDATION'):g[f'{name}/fold/{fold}/factor/{key}/{val}']=selected&np.array([r['split']==fold for r in rows])
 return g
def metric_bundle(pred,y,states,available,shared,mask):
 return {'own_full_denominator':metric(pred,y,mask,states),'own_available_subset':metric(pred,y,mask&available,states),'shared_available_subset':metric(pred,y,mask&shared,states)}
def triplets(rows,gt,corpus,pred,raw,available):
 gg=defaultdict(list)
 for i,r in enumerate(rows):gg[(str(corpus[i]),r['query_group_id'])].append(i)
 records=[]
 for(name,gid),ix in gg.items():
  if name=='NATIVE':
   lookup={rows[i]['variant']:i for i in ix};assert len(ix)==4 and len(lookup)==4 and set(lookup)=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
   triples=[(None,sh,[lookup['BEFORE'],lookup['COVER'],lookup[sh]])for sh in ('NEAR_SHAM','FAR_SHAM')]
  else:
   assert len(ix)==17;before=[i for i in ix if rows[i]['variant']=='BEFORE'];assert len(before)==1
   triples=[]
   for rank in range(8):
    cover=[i for i in ix if rows[i]['donor_rank']==rank and rows[i]['variant']=='OPAQUE_COVER'];sham=[i for i in ix if rows[i]['donor_rank']==rank and rows[i]['variant']=='CUE_PRESERVING_SHAM'];assert len(cover)==len(sham)==1;triples.append((rank,'CUE_PRESERVING_SHAM',[before[0],cover[0],sham[0]]))
  for rank,sh,ii in triples:
   for k in ('query_xy','initial_points','L','context_box_xyxy'):assert all(rows[i][k]==rows[ii[0]][k]for i in ii)
   st=[gt[i]['raw_state']for i in ii];known=st==['D','E','D'];r=rows[ii[0]]
   error=gt[ii[0]]['error_px'];error_bin='MISSING'if error is None else'ZERO'if error==0 else'LE3'if error<=3 else'GT3'
   record={'corpus':name,'query_group_id':gid,'indices':ii,'source_occurrence_indices':[rows[i]['occurrence_index']for i in ii],'split':r['split'],'condition_id':r['condition_id'],'condition_index':r['condition_index'],'sham_variant':sh,'donor_rank':rank,'states':st,'known_D_E_D':known,'source_error_px':error,'error_bin':error_bin,'factors':r.get('factors',{}),'base_index':r.get('base_index'),'cue_index':r.get('cue_index')}
   for m in MODELS:record[m]={'predictions':pred[m][ii].tolist(),'all3_correct':bool(known and np.array_equal(pred[m][ii],[0,1,0])),'all3_available':bool(available[m][ii].all()),'hidden_direction_logits':(raw[m][ii]*(1 if m=='C16'else -1)).tolist()}
   records.append(record)
 summaries={}
 def summarize(rr):
  known=[r for r in rr if r['known_D_E_D']];return {'planned_triplets':len(rr),'known_D_E_D':len(known),**{m:{'all3_correct':sum(r[m]['all3_correct']for r in known),'all3_available':sum(r[m]['all3_available']for r in known),'shared_all3_available_N':sum(all(r[n]['all3_available']for n in MODELS)for r in known),'shared_all3_correct':sum(r[m]['all3_correct']and all(r[n]['all3_available']for n in MODELS)for r in known)}for m in MODELS}}
 for name in EXPECTED:
  rr=[r for r in records if r['corpus']==name];summaries[name]=summarize(rr)
  for key in ('split','condition_id','sham_variant','donor_rank','error_bin'):
   for val in sorted({json.dumps(r[key])for r in rr}):summaries[f'{name}/{key}/{val}']=summarize([r for r in rr if json.dumps(r[key])==val])
  for fold in ('TRAIN','VALIDATION'):
   for key in ('condition_id','sham_variant','error_bin'):
    for val in sorted({json.dumps(r[key])for r in rr}):summaries[f'{name}/fold/{fold}/{key}/{val}']=summarize([r for r in rr if r['split']==fold and json.dumps(r[key])==val])
 return records,summaries
def pixel_diagnostics(rows,gt,corpus,pred,states,available,shared):
 N=len(rows);native=corpus=='NATIVE';y=np.full(N,-1,np.int8);category=np.full(N,'NOT_NATIVE',dtype='<U24')
 for i,g in enumerate(gt):
  if native[i]:
   category[i]=g['q_nearest_pixel_owner_diagnostic']['category'];assert category[i]in ('TARGET','FOREIGN','BACKGROUND','U');y[i]=0 if category[i]=='TARGET'else(1 if category[i]in ('FOREIGN','BACKGROUND')else-1)
 reports={m:metric_bundle(pred[m],y,states,available[m],shared,native)for m in MODELS}
 cross=[]
 for name in EXPECTED:
  cats=category if name=='NATIVE'else np.array(['IN_PASTED_FOREGROUND'if g.get('nearest_q_foreground_alpha_diagnostic')else 'CLEAR_OF_PASTED_FOREGROUND'if g.get('nearest_q_inside_image_diagnostic')else 'OOB'for g in gt])
  for fold in ('TRAIN','VALIDATION'):
   for st in 'DESU':
    for cat in sorted(set(cats[corpus==name])):
     mask=(corpus==name)&np.array([r['split']==fold for r in rows])&(states==st)&(cats==cat)
     if mask.any():cross.append({'corpus':name,'split':fold,'state':st,'q_pixel_diagnostic':cat,'N':int(mask.sum()),**{m:{p:int((mask&(pred[m]==v)).sum())for p,v in [('D',0),('E',1),('prediction_U',-1)]}for m in MODELS}})
 return {'native_q_pixel_target_D_vs_foreign_or_background_E':reports,'cross_semantic_state_pixel_category_prediction':cross,'scope':'Native nearest pixel owner is a separate objective from continuous structural-region D/E. Public alpha outside means clear of pasted foreground only, NOT known target presence; no public q-pixel accuracy is fabricated.'},y,category
def select_gallery(rows,gt,corpus,pred,y):
 roles=[];chosen=[]
 for name in EXPECTED:
  for m in MODELS:
   for role in ('FIRST_D_CORRECT','FIRST_E_CORRECT','FIRST_KNOWN_DE_FAILURE'):
    eligible=[i for i in range(len(rows))if corpus[i]==name and ((y[i]==0 and pred[m][i]==0)if role=='FIRST_D_CORRECT'else(y[i]==1 and pred[m][i]==1)if role=='FIRST_E_CORRECT'else(y[i]>=0 and pred[m][i]!=y[i]))]
    i=eligible[0]if eligible else None;roles.append({'corpus':name,'model':m,'role':role,'index':i,'absent':i is None})
    if i is not None and i not in chosen:chosen.append(i)
 assert len(chosen)<=12
 return roles,chosen
def font(size):
 from PIL import ImageFont
 return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def gallery(rows,gt,corpus,pred,raw,roles,selected):
 from PIL import Image,ImageDraw
 root=OUT/'figures';root.mkdir();write(OUT/'FIXED_GALLERY_SELECTION_BEFORE_CROP_READ.json',{'completed':True,'selection_rule':'For each corpus x model, first D correct, first E correct, first known D/E failure in original row order; missing explicit; deduplicate source rows, <=12. Posthoc display only.','roles':roles,'selected_indices':selected,'selected_rows':[{'combined_index':i,'corpus':corpus[i],'input_row':rows[i],'teacher_row':gt[i]}for i in selected]})
 board=Image.new('RGB',(3*414,70+max(1,(len(selected)+2)//3)*292),'#eef3f6');d=ImageDraw.Draw(board);d.text((12,8),'Fixed saved source diagnostics: RAW96 x2 / separate q-yellow, reference-cyan copy',font=font(17),fill='#123044');d.text((12,35),'Posthoc first-row rules; no model, label, coordinate or training selection.',font=font(14),fill='#344c5e');checked=set()
 for pos,i in enumerate(selected):
  r=rows[i];b=r['rgb96_chunk'];key=(b['path'],b['sha256'])
  if key not in checked:check(b);checked.add(key)
  with np.load(b['path'],allow_pickle=False)as f:
   j=r['rgb96_row'];a=np.array(f['rgb96'][j],copy=True);v=np.array(f['valid96'][j],copy=True);assert int(f['occurrence_index'][j])==r['occurrence_index']
  assert a.dtype==np.uint8 and a.shape==(96,96,3)and v.dtype==bool and v.shape==(96,96)and ash(a)==r['rgb96_sha256']and ash(v)==r['valid96_sha256']
  panel=Image.new('RGB',(408,282),'white');pd=ImageDraw.Draw(panel);rawim=Image.fromarray(a).resize((192,192),Image.Resampling.NEAREST);marked=rawim.copy();md=ImageDraw.Draw(marked)
  def cross(xy,color):
   x,y=xy
   if 0<=x<96 and 0<=y<96:
    x=(x+.5)*2-.5;y=(y+.5)*2-.5;md.line((x-5,y,x+5,y),fill=color,width=2);md.line((x,y-5,x,y+5),fill=color,width=2)
  cross([47.5,47.5],'yellow');ref=gt[i]['reference_xy']
  if ref is not None:cross((np.asarray(ref)-np.asarray(r['query_xy']))*96/r['L']+47.5,'cyan')
  panel.paste(rawim,(6,18));panel.paste(marked,(206,18));pd.text((6,1),f'{corpus[i]} row {r["occurrence_index"]}: {gt[i]["raw_state"]}',font=font(12),fill='#123044')
  text=[f'{r["split"]} {r["variant"]} c{r["condition_index"]}',f'q={np.round(r["query_xy"],2).tolist()} L={r["L"]:.3f}',f'C16 {int(pred["C16"][i])} hidden z={raw["C16"][i]:+.3f}',f'v61 {int(pred["v61"][i])} visible z={raw["v61"][i]:+.3f} | 0D 1E -1U']
  for line,t in enumerate(text):pd.text((6,213+16*line),t,font=font(12),fill='#123044')
  board.paste(panel,(5+(pos%3)*414,66+(pos//3)*292))
 path=root/'FIXED_SOURCE_SUCCESS_FAILURE_MAX12.png';board.save(path);return bind(path)
def chart(reports):
 from PIL import Image,ImageDraw
 im=Image.new('RGB',(1160,540),'white');d=ImageDraw.Draw(im);d.text((15,12),'All planned semantic D/E: rows true D/E, columns predicted D/E/U',font=font(21),fill='#123044')
 for k,name in enumerate(EXPECTED):
  for j,m in enumerate(MODELS):
   x=15+j*575;y=58+k*235;v=reports[name][m]['own_full_denominator'];d.text((x,y),f'{name} / {m}  known={v["known_DE"]}',font=font(18),fill='#123044')
   for r in range(2):
    for c in range(3):
     xx=x+42+c*132;yy=y+35+r*55;d.rectangle((xx,yy,xx+124,yy+48),fill='#dceaf3');d.text((xx+12,yy+12),str(v['CM'][r][c]),font=font(20),fill='#14334b')
   d.text((x,y+153),f'D recall={v["D_recall"]}   E recall={v["E_recall"]}',font=font(14),fill='#123044')
 p=OUT/'figures/SEMANTIC_DE_CONFUSION.png';im.save(p);return bind(p)
def html_report(result,figs):
 def percent(x):return '—'if x is None else f'{100*x:.2f}%'
 table=''
 for name in EXPECTED:
  for m in MODELS:
   for den in ('own_full_denominator','shared_available_subset'):
    x=result['semantic_reports'][name][m][den];table+=f'<tr><td>{name}</td><td>{m}</td><td>{den}</td><td>{x["CM"]}</td><td>{percent(x["D_precision"])}</td><td>{percent(x["D_recall"])}</td><td>{percent(x["E_precision"])}</td><td>{percent(x["E_recall"])}</td><td>{x["known_prediction_U"]}</td></tr>'
 images=''.join(f'<figure><a href="figures/{Path(b["path"]).name}"><img src="figures/{Path(b["path"]).name}"></a></figure>'for b in figs)
 (OUT/'READOUT_KO.html').write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>V129 source 고정 baseline</title><style>body{{font:16px Arial,sans-serif;line-height:1.6;max-width:1450px;margin:25px auto;padding:0 20px;color:#183345}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ccd9e1;padding:7px}}img{{max-width:100%;height:auto}}.note{{background:#fff0d8;padding:15px}}code{{overflow-wrap:anywhere}}</style><h1>등록 합성 다양성 + 공개 실사 AUX: 고정 source 진단</h1><p class="note">새 학습·모델 선택·임계값 변경 없이 저장된 C16/v61 결과를 비교했습니다. 실제 현재 평가나 전이 성공의 증거가 아닙니다. 합성 wood/polymer 재질과 rectangular/square 형상은 독립 축이며 실제 WOOD/SQUARE 평가군과 다릅니다.</p><p>행 D/E, 열 예측 D/E/U. 전체 known D/E의 recall 분모에는 예측 U를 포함합니다. 모델별 전체·자체 가용·공통 가용 분모를 JSON에 모두 보존했습니다. S 응답은 외부 E 정답으로 해석하지 않으며 teacher U는 미정 상태입니다.</p><table><tr><th>자료</th><th>모델</th><th>분모</th><th>CM</th><th>D P</th><th>D R</th><th>E P</th><th>E R</th><th>예측 U</th></tr>{table}</table>{images}<h2>해석 범위</h2><p>Native 192개 설정·768개 장면, 공개 실사 6cue·6원본 프레임의 반복 조건입니다. query/variant 반복은 독립 새 이미지가 아닙니다. 등록 합성의 연속 구조영역 teacher와 q 픽셀의 owner 목적을 별도 계산했습니다. 공개 pasted alpha 밖은 target 확인을 뜻하지 않습니다. q와 L은 그대로이며 reference 마크는 표시 사본에만 있습니다. 모든 source query는 emulation이며 실제 initializer 정확도 인증이 아닙니다. Native U/부분/aliasing 및 공개 weak cue의 불확실성을 유지했습니다. 새로운 학습 승인이나 모델 추천을 만들지 않습니다.</p><p><a href="SOURCE_RESULTS.json">전체 factor/availability/S/U/qpixel JSON</a> · <a href="ALL56214_SCORES.csv">전체 행</a> · <a href="ALL_SAME_QUERY_TRIPLETS.json">모든 계획 triplet</a> · <a href="FIXED_GALLERY_SELECTION_BEFORE_CROP_READ.json">그림 선택 규칙·원행</a></p></html>')
def run(entry):
 e,seals=authorize(entry);rows,gt,corpus,local=load_rows(e);pred,raw,available=load_predictions(seals,rows,corpus,local);shared=available['C16']&available['v61'];states=np.array([g['raw_state']for g in gt]);y=np.array([g['external_target_D0_E1_else_minus1']for g in gt],np.int8);masks=groups(rows,gt,corpus)
 OUT.mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'both_completed_prediction_seals_before_GT_join':True,'source_only':True,'model_forward':False})
 try:
  reports={key:{m:metric_bundle(pred[m],y,states,available[m],shared,mask)for m in MODELS}for key,mask in masks.items()};triples,triplesummary=triplets(rows,gt,corpus,pred,raw,available);pixel,py,pc=pixel_diagnostics(rows,gt,corpus,pred,states,available,shared);roles,selected=select_gallery(rows,gt,corpus,pred,y)
  write(OUT/'ALL_SAME_QUERY_TRIPLETS.json',{'completed':True,'scope':'Same q8/q/L/context; planned non-D/E/D states retained; no stage-name label coercion. Native NEAR and FAR are separate triple diagnostics, not independent BEFORE/COVER examples.','summary':triplesummary,'rows':triples})
  counts={name:{'planned_rows':int((corpus==name).sum()),'teacher_states':{s:int(((corpus==name)&(states==s)).sum())for s in 'DESU'},'unique_fullRGB_bindings':len({(r['full_RGB']['path'],r['full_RGB']['sha256'])for i,r in enumerate(rows)if corpus[i]==name}),'source_frame_or_base_groups':len({r.get('base_id',r.get('source_frame_group',r.get('cue_index')))for i,r in enumerate(rows)if corpus[i]==name})}for name in EXPECTED}
  result={'completed':True,'source_only':True,'rows':len(rows),'counts':counts,'availability':{m:{'available':int(available[m].sum()),'unavailable':int((~available[m]).sum()),'known_DE_unavailable':int(((y>=0)&~available[m]).sum())}for m in MODELS},'shared_available':int(shared.sum()),'availability_disagreement':int((available['C16']!=available['v61']).sum()),'semantic_reports':reports,'same_q_triplet_summary':triplesummary,'q_pixel_diagnostics':pixel,'teacher_U_and_factual_U_separate':True,'material_aspect_independent_from_real_WOOD_SQUARE_categories':True,'new_model_forward_fit_current223_V112_PnP':0,'training_admission_or_model_selection':False,'correlated_rows_not_independent_images':True,'input_bindings':e['corpora'],'prediction_seals':{m:e[m+'_prediction_seal']for m in MODELS}}
  with(OUT/'ALL56214_SCORES.csv').open('x',newline='')as f:
   fields=['combined_index','corpus','source_occurrence_index','id','split','variant','condition_id','logical_slot','query_group_id','qx','qy','L','teacher_state','external_target','error_px','native_q_pixel_category','native_q_pixel_target','C16_raw_hidden','C16_prediction','C16_available','v61_raw_visible','v61_prediction','v61_available','shared_available'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for i,r in enumerate(rows):w.writerow(dict(zip(fields,[i,str(corpus[i]),int(local[i]),r['id'],r['split'],r['variant'],r['condition_id'],r['logical_slot'],r['query_group_id'],*r['query_xy'],r['L'],str(states[i]),int(y[i]),gt[i]['error_px'],str(pc[i]),int(py[i]),float(raw['C16'][i]),int(pred['C16'][i]),bool(available['C16'][i]),float(raw['v61'][i]),int(pred['v61'][i]),bool(available['v61'][i]),bool(shared[i])])) )
  figs=[gallery(rows,gt,corpus,pred,raw,roles,selected)];figs.insert(0,chart(reports));result['figures']=figs;write(OUT/'SOURCE_RESULTS.json',result);html_report(result,figs)
  outputs=[bind(p)for p in sorted(OUT.rglob('*'))if p.is_file()];seal=OUT/'EVALUATION_SEAL.json';write(seal,{'completed':True,'PASS':True,'entry':bind(entry),'code':e['code'],'protocol':e['protocol'],'C16_prediction_seal':e['C16_prediction_seal'],'v61_prediction_seal':e['v61_prediction_seal'],'corpora':e['corpora'],'all56214_retained':True,'both_prediction_seals_before_GT_payload':True,'saved_crop_hashes_exact_for_all_selected':True,'selected_gallery_rows':len(selected),'outputs':outputs,'no_model_forward_fit_current223_V112_PnP':True,'human_visual_review_not_performed_by_script':True})
  print(json.dumps({'completed':True,'seal':bind(seal),'rows':len(rows),'CM':{n:{m:reports[n][m]['own_full_denominator']['CM']for m in MODELS}for n in EXPECTED},'gallery_rows':len(selected)}),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
def generated():
 pred=np.array([0,1,-1,0,1,1,-1,0],np.int8);y=np.array([0,0,0,1,1,1,-1,-1]);st=np.array(list('DDDEEESU'));m=metric(pred,y,np.ones(8,bool),st);assert m['CM']==[[1,1,1],[1,2,0]]and m['D_recall']==1/3 and m['known_prediction_U_rate']==1/6 and m['typed_outside_external_task']['S']['prediction_U']==1
 z=np.array([-1.,-np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.),1.]);assert decode(z,np.ones(6,bool),'C16').tolist()==[0,0,1,1,1,1]and decode(z,np.ones(6,bool),'v61').tolist()==[1,1,0,0,0,0]and(decode(z,np.zeros(6,bool),'C16')==-1).all()
 rr=[];gg=[];cc=[];pp=[]
 for name in EXPECTED:
  stages=[('BEFORE',None,'D')]+([('COVER',None,'E'),('NEAR_SHAM',None,'D'),('FAR_SHAM',None,'D')]if name=='NATIVE'else[(v,k,s)for k in range(8)for v,s in [('OPAQUE_COVER','E'),('CUE_PRESERVING_SHAM','D')]])
  for v,k,s in stages:
   i=len(rr);rr.append({'id':str(i),'occurrence_index':i,'query_group_id':'a','variant':v,'donor_rank':k,'query_xy':[1.,2.],'initial_points':[[1.,2.]]*8,'L':16.,'context_box_xyxy':[0.,0.,4.,5.],'split':'TRAIN','condition_id':'IDENTITY','condition_index':0});gg.append({'raw_state':s,'error_px':0.});cc.append(name);pp.append(0 if s=='D'else 1)
 pp=np.array(pp,np.int8);pr={m:pp.copy()for m in MODELS};av={m:np.ones(len(rr),bool)for m in MODELS};tr,ts=triplets(rr,gg,np.array(cc),pr,{m:pp.astype(np.float32)for m in MODELS},av);assert len(tr)==10 and ts['NATIVE']['known_D_E_D']==2 and ts['PUBLIC_AUX']['known_D_E_D']==8
 roles,sel=select_gallery(rr,gg,np.array(cc),pr,pp);assert len(roles)==12 and len(sel)<=12 and sum(r['absent']for r in roles)==4
 shared=np.array([True,False,True,True,True,True,False,True]);b=metric_bundle(pred,y,st,np.ones(8,bool),shared,np.ones(8,bool));assert b['own_full_denominator']['known_DE']==6 and b['shared_available_subset']['known_DE']==5
 assert clean({'int':np.int64(4),'bool':np.bool_(True),'float':np.float32(.5)})=={'int':4,'bool':True,'float':.5};json.dumps(clean({'m':m,'triples':tr,'summary':ts,'roles':roles}),allow_nan=False)
 return {'completed':True,'PASS':True,'code':bind(__file__),'checks':{'semantic_CM_U_recall_and_typed_S_U':True,'opposite_raw_polarities_signedzero_subnormal':True,'shared_vs_full_denominators_preserved':True,'native4_and_public17_grouped_sameq_triplets':True,'all_planned_non_DED_triplets_retained':True,'first_row_transparent_gallery_max12_with_absent_categories':True,'all_NumPy_scalars_clean_JSON':True,'no_model_import_or_native_RGB_decode':True,'both_completed_seals_checked_before_GT_join':True,'public_alpha_clear_not_assumed_target':True},'actual_source_predictions_arrays_RGB_model_weights_current_V112_reads':0}
def prepare():
 g=generated();write(GENERATED,g);p={'completed':True,'code':bind(__file__),'generated':bind(GENERATED),'output':str(OUT),'expected_rows':EXPECTED,'GT_boundary':'Both root-bound completed PASS prediction seals and their corpus input bindings are verified before GT hash/body joins or prediction/crop payload reads. No model/encoder imports.','metrics':'2x3 semanticD/E CM/P/R/U with modelownfull,ownavailable,sharedavailable; S response and teacherU separate. Factor/fold/condition/variant/error groups. Native qpixel target vsforeign/background separate; public alpha outside neverpromotedtarget. SameqD/E/D planned/known/availabletriples and sourceemulationerrors.','factor_warning':'Geometry/aspect/material independently crossed in synthetic source; not real WOOD/SQUARE categories.','gallery':'Atmost12deduplicated originalorder firstDcorrect/Ecorrect/knownDEfailure permodelpercorpus. Selectionfile before selected savedRGB96 chunkopen. Raw and markedcopy separate. No newnativeRGBdecode.','new_model_fit_current223_V112_PnP':False};write(PROTOCOL,p)
 corpora=[]
 for name,folder,file in [('NATIVE','source_crops_v1','CROP_COMPLETE.json'),('PUBLIC_AUX','public_corpus_v1','CORPUS_COMPLETE.json')]:
  complete=bind(V/folder/file);done=read(complete['path']);assert done['completed']and done['PASS'];corpora.append({'name':name,'expected_rows':EXPECTED[name],'complete':complete,'input_rows':done['input_manifest'],'GT_rows':done['GT_only']})
 t={'approved_saved_source_evaluation':False,'source_only':True,'model_forward':False,'fit':False,'current223':False,'V112':False,'PnP':False,'automatic_retry':False,'code':p['code'],'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'C16_prediction_seal':None,'v61_prediction_seal':None,'corpora':corpora};tp=H/'ROOT_SAVED_EVAL_ENTRY_TEMPLATE.json';write(tp,t)
 print(json.dumps({'code':p['code'],'protocol':bind(PROTOCOL),'generated':bind(GENERATED),'template':bind(tp),'command':str(Path(sys.executable).resolve())+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SAVED_SOURCE_EVAL_ENTRY.json','actual_prediction_or_model_reads':0}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
 if o.prepare:prepare()
 elif o.entry:run(o.entry)
 else:raise SystemExit('Use --prepare or root --entry.')
