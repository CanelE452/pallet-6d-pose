"""Post-freeze saved-output MR SOURCE_VALIDATION gallery; preparation opens code/metadata only.
No encoder/classifier/checkpoint/fit calls. Every original known weak D/E row is retained.
"""
from pathlib import Path
import argparse,ast,hashlib,html,importlib.util,json,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V153=P/'query_visibility_mr6d_full_public_source_20261024_v153'
V162=P/'query_visibility_counterfactual_frozen_features_20261024_v162'
PARENT=V162/'controlled_source_visual_preparation/render_saved_C43_C44_caption_v2.py'
MATH=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
TRAIN_PROTOCOL=V/'controlled_fit_preparation/PROTOCOL_BEFORE_SOURCE_FIT_FEATURE_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_POSTFIT_SOURCE_VALIDATION_BODIES.json'
OUT=V/'source_validation_full154_visual_v1';RUN=V/'runs/C45_C46_PART_NATIVE_FIXED8000'
ARMS=('C45_REPLAY_C43_OWNER253_NATIVE0','C46_PART_NATIVE92_OWNER253_NATIVE0')
FONT=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
CONFUSIONS=('TP','FN','FP','TN','U')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,r):
 with Path(p).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rows):
 with Path(p).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 return bind(p)
def font(n):return ImageFont.truetype(str(FONT),n)
def helpers(p):
 check(p['parent_renderer']);spec=importlib.util.spec_from_file_location('exact_V162_saved_display_helpers',PARENT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 check(p['C9_pure_math']);ns={'np':np};nodes=[n for n in ast.parse(MATH.read_text()).body if isinstance(n,ast.FunctionDef)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(MATH)+'::pure_numpy_only','exec'),ns)
 return m,ns

def confusion(truth,pred):
 assert truth in ('D','E') and pred in ('D','E','U')
 if pred=='U':return 'U'
 return {('E','E'):'TP',('E','D'):'FN',('D','E'):'FP',('D','D'):'TN'}[(truth,pred)]
def fixed_representatives(rows):
 # Lowest global index only; outcome categories are diagnostic strata, not success selection.
 ordered=sorted(rows,key=lambda r:r['global_index']);chosen={};categories={}
 for arm in ARMS:
  for category in CONFUSIONS:
   key=arm+'::'+category;rr=next((r for r in ordered if r['measurements'][arm]['confusion']==category),None)
   categories[key]=None if rr is None else rr['global_index']
   if rr is not None:chosen.setdefault(rr['global_index'],[]).append(key)
 for truth in ('D','E'):
  for old in ('D','E','U'):
   for new in ('D','E','U'):
    if old==new:continue
    key=f'CHANGE::{truth}::{old}->{new}'
    rr=next((r for r in ordered if r['fixed_weak_truth']==truth and r['measurements'][ARMS[0]]['prediction']==old and r['measurements'][ARMS[1]]['prediction']==new),None)
    categories[key]=None if rr is None else rr['global_index']
    if rr is not None:chosen.setdefault(rr['global_index'],[]).append(key)
 return categories,chosen

def prepare():
 p=read(TRAIN_PROTOCOL)
 fc=read(V153/'full230_native_DINO_v1/FEATURE_COMPLETE.json')
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),trainer_protocol=bind(TRAIN_PROTOCOL),parent_renderer=bind(PARENT),C9_pure_math=bind(MATH),font=bind(FONT),
  source_labels=p['source_labels'],source_inputfacts=p['MR_inputfacts'],source_frames=fc['inputs'],saved96_packets=fc['crop_manifest'],
  arms=list(ARMS),run=str(RUN),output=str(OUT),required_terminal_trigger='Root actual terminal exit0 and all2 completed saved-readout trigger; verify CONTROL_COMPLETE and ALL2_FROZEN before labels/output/RGB reads',
  scope='All original MR SOURCE_VALIDATION360 census; all known weak D/E154 cards D102/E52, retaining9D inputU;145 input-valid diagnostic shown separately. Other206 truth-outside-D/E rows remain census and are not called error-free.',
  selection='After both models freeze, before any RGB read: lowest global_index per arm TP/FN/FP/TN/U and each truth-specific changed D/E/U transition. No fallback if empty. All154 cards plus ALL changed known154 appendix retained; no success-only52E selection.',
  rendering='Original public SOURCE_VALIDATION RGB only; original saved raw96 and literal raw96[32:64,32:64]; unchanged q/L/native. Context224 recomputed by exact V107/C9 pure bilinear math at original contextbox, hash-equal original float32 target224/valid facts. Display rounds uint8 only; save exact float32 target and valid.',
  availability='Original continuous-only availability is sole primary U rule. nearest/central4/context/construct diagnostics do not add gates. Logit threshold0 unchanged. Native weak role and literal q RGB are separate; no new owner/coordinate gold.',
  QA='Actually view unique fixed representatives plus every changed known row; all154 link/hash/census verified. No claim of actual viewing every unselected card or browser HTML rendering.',
  source_held_labels_outputs_or_RGB_bodies_read_during_preparation=0,encoder_classifier_fit_checkpoint_or_private_calls=0,new_labels_or_training_admission=False,current_V112_PnP_allowed=False)
 write(PROTOCOL,protocol)
 print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL))))

def card(row,rgb,crop,valid,target_display):
 c=Image.new('RGB',(2200,1410),'#f5f8fb');d=ImageDraw.Draw(c);q=row['query_xy']
 d.text((24,18),f"MR SOURCE_VALIDATION · g{row['global_index']:04d} · native{row['native_slot']} · 기존 native 약역할 {row['fixed_weak_truth']}",font=font(33),fill='#20364c')
 d.text((24,66),row['query_id'],font=font(25),fill='#3b526a')
 d.text((24,107),f"원래 q={q} · L={row['L']} · source={row['source_id']} · input_available={row['query_available']}",font=font(25),fill='#965111')
 d.text((24,150),'두 모델에 같은 원래 입력 · 아래 model 출력만 비교 · q/L/native/문턱 변경 없음',font=font(26),fill='#20364c')
 d.text((24,202),'전체 RGB: q 십자는 표시 복사본에만',font=font(23),fill='#20364c')
 im=Image.fromarray(rgb);im.thumbnail((960,540),Image.Resampling.BILINEAR)
 if q is not None and len(q)==2 and all(x is not None and np.isfinite(x) for x in q) and q!=[-1,-1]:
  x,y=q[0]*im.width/rgb.shape[1],q[1]*im.height/rgb.shape[0]
  if 0<=x<im.width and 0<=y<im.height:
   dd=ImageDraw.Draw(im);dd.line((x-7,y,x+7,y),fill='#ef00db',width=2);dd.line((x,y-7,x,y+7),fill='#ef00db',width=2)
 c.paste(im,(24,244))
 d.text((1020,202),'저장 raw RGB96 · 4×',font=font(23),fill='#20364c');c.paste(Image.fromarray(crop).resize((384,384),Image.Resampling.NEAREST),(1020,244))
 d.text((1450,202),'같은 중앙32 · 8×',font=font(23),fill='#20364c');c.paste(Image.fromarray(crop[32:64,32:64]).resize((256,256),Image.Resampling.NEAREST),(1450,244))
 d.text((1740,202),'원래 target224 · 표시2×',font=font(22),fill='#20364c');c.paste(Image.fromarray(target_display).resize((448,448),Image.Resampling.NEAREST),(1740,244))
 d.text((1020,658),f"center4 valid: {int(valid[47:49,47:49].sum())}/4 · construct={row['feature_constructed']}",font=font(24),fill='#3b526a')
 d.text((1020,702),f"nearest={row['nearest_valid']} · context={row['context_available']}",font=font(24),fill='#3b526a')
 d.text((24,810),'E는 external 양성 · TP/FN/FP/TN은 기존 native 약역할 기준 · U는 원래 입력 사실',font=font(27),fill='#965111')
 for j,arm in enumerate(ARMS):
  x=24+j*1090;yy=866;m=row['measurements'][arm]
  d.rounded_rectangle((x,yy,x+1050,yy+247),radius=12,fill='#e4ebf3')
  d.text((x+18,yy+16),('C45 원 owner253 · native 추가0' if j==0 else 'C46 원 owner253 + weak native92'),font=font(30),fill='#20364c')
  logit_text='undefined' if m['raw_external_E_logit'] is None else f"{m['raw_external_E_logit']:.7g}"
  d.text((x+18,yy+70),f"예측 {m['prediction']} · {m['confusion']} · raw logit(E)={logit_text}",font=font(28),fill='#20364c')
  pe='undefined' if m['raw_pE'] is None else f"{m['raw_pE']:.7f}"
  d.text((x+18,yy+119),'raw p(E)='+pe+' · 고정 문턱 logit0',font=font(27),fill='#20364c')
  d.text((x+18,yy+165),'entity/quality는 primary의 필터가 아님',font=font(26),fill='#3b526a')
 d.text((24,1156),'Native D/E 약역할과 literal q 중심의 RGB/소유는 다른 관찰입니다. 배경 중심도 좌표오차 gold를 뜻하지 않습니다.',font=font(28),fill='#965111')
 d.text((24,1206),'기존 prior assistant/root 구조 약라벨 · 공식 사람 keypoint gold/3px 품질 정답 아님 · 새 owner/native/quality 라벨 생성 없음',font=font(27),fill='#965111')
 d.text((24,1256),'원래 full154 = D102/E52 (D9 inputU 포함) · input-valid145 = D93/E52 별도 · 전체 SOURCE_VAL360 중206은 D/E 밖',font=font(27),fill='#3b526a')
 d.text((24,1306),'SOURCE 평가 설명용 · 모델/문턱/epoch 선택·추가 학습 편입 없음 · EuroWOOD 예시이며 전체 WOOD/SQUARE/native8 목표 유지',font=font(26),fill='#3b526a')
 return c

def render(trigger_path):
 p=read(PROTOCOL);assert p['code']==bind(__file__) and p['trainer_protocol']==bind(TRAIN_PROTOCOL)
 trigger=read(trigger_path)
 assert trigger.get('all2_completed',trigger.get('both_models_completed',False)) is True
 assert trigger.get('approved_saved_output_readout',False) is True
 control=read(RUN/'CONTROL_COMPLETE.json');freeze=read(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json')
 assert control['completed'] and control['PASS'] and list(control['arms'])==list(ARMS) and freeze['completed'] and freeze['arms']==list(ARMS)
 assert control['updates_each']==8000 and len(freeze['checkpoints'])==2
 assert not OUT.exists();OUT.mkdir();(OUT/'cards').mkdir();(OUT/'assets').mkdir()
 helper,op=helpers(p);labels=lines(check(p['source_labels']));facts=lines(check(p['source_inputfacts']))
 frames=read(check(p['source_frames']))['frames'];crops={r['frame_index']:r['crop_packet'] for r in read(check(p['saved96_packets']))['sheets']}
 assert len(labels)==len(facts)==1840 and len(frames)==230
 packets={};packet_bindings={}
 fields=('global_index','query_id','source_id','split','native_slot','query_xy','L','query_available','feature_constructed','raw_three_logits','primary_visibility_prediction')
 for arm in ARMS:
  b=bind(RUN/arm/'FINAL_MR1840_OUTPUTS.npz');packet_bindings[arm]=b
  with np.load(check(b),allow_pickle=False) as z:packets[arm]={k:z[k].copy() for k in fields}
  assert packets[arm]['global_index'].tolist()==list(range(1840))
 held=[i for i,r in enumerate(labels) if r['split']=='SOURCE_VALIDATION'];assert len(held)==360
 known=[i for i in held if labels[i]['role_target'] in (0,1)];assert len(known)==154
 assert sum(labels[i]['role_target']==0 for i in known)==102 and sum(labels[i]['role_target']==1 for i in known)==52
 rows=[];all360=[]
 for i in held:
  l,f=labels[i],facts[i];assert f['global_index']==i and f['split']=='SOURCE_VALIDATION' and l['query_id']==f['query_id']
  assert l['query_xy']==f['query_xy'] and l['L']==f['L'] and l['source_id']==f['source_id']
  truth=('D','E')[l['role_target']] if l['role_target'] in (0,1) else None
  if truth is not None:assert l['role_state']==truth
  row=dict(global_index=i,query_id=f['query_id'],frame_index=f['frame_index'],source_id=f['source_id'],split=f['split'],native_slot=f['native_slot'],query_xy=f['query_xy'],L=f['L'],fixed_weak_truth=truth,query_available=bool(f['native_continuous_valid']),feature_constructed=bool(f['feature_constructed']),nearest_valid=bool(f['nearest_valid']),context_available=bool(f['context_available']),measurements={})
  assert row['query_available']==l['original_input_valid']
  for arm in ARMS:
   z=packets[arm];assert str(z['query_id'][i])==f['query_id'] and z['native_slot'][i]==f['native_slot'] and str(z['split'][i])==f['split'] and str(z['source_id'][i])==f['source_id']
   q=np.asarray(f['query_xy'] if f['query_xy'] is not None else [np.nan,np.nan],np.float64);side=np.nan if f['L'] is None else f['L']
   assert np.array_equal(z['query_xy'][i],q,equal_nan=True) and np.array_equal(np.asarray(z['L'][i]),np.asarray(side),equal_nan=True)
   assert bool(z['query_available'][i])==row['query_available'] and bool(z['feature_constructed'][i])==row['feature_constructed']
   logit=float(z['raw_three_logits'][i,0]);pred=helper.primary(logit,row['query_available']);assert int(z['primary_visibility_prediction'][i])=={'U':-1,'D':0,'E':1}[pred]
   row['measurements'][arm]=dict(prediction=pred,raw_external_E_logit=logit if np.isfinite(logit) else None,raw_pE=helper.sigmoid(logit),confusion=confusion(truth,pred) if truth else None)
  all360.append(row)
  if truth is not None:rows.append(row)
 assert sum(not r['query_available'] for r in rows)==9 and all(r['fixed_weak_truth']=='D' for r in rows if not r['query_available'])
 categories,chosen=fixed_representatives(rows);changed=[r['global_index'] for r in rows if r['measurements'][ARMS[0]]['prediction']!=r['measurements'][ARMS[1]]['prediction']]
 # Freeze all row identities and deterministic diagnostic sample IDs before any RGB body.
 write(OUT/'DIAGNOSTIC_SELECTION_BEFORE_RGB.json',dict(completed=True,protocol=bind(PROTOCOL),root_trigger=bind(trigger_path),categories=categories,selected_global_indices_and_categories={str(k):v for k,v in chosen.items()},all_changed_known_global_indices=changed,rule='Lowest global_index per fixed category, plus exhaustive allchanged appendix; no visual/model optimization',all154_global_indices=[r['global_index'] for r in rows]))
 census=jsonl(OUT/'ALL360_SOURCE_VALIDATION_CENSUS.jsonl',all360)
 assets={};records=[]
 for row in rows:
  i=row['global_index'];f=facts[i];fi=row['frame_index'];fr=frames[fi];assert fr['frame_index']==fi and fr['source_id']==row['source_id'] and fr['split']=='SOURCE_VALIDATION'
  if fi not in assets:
   rgb=np.load(check(fr['canonical_RGB_NPY']),allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3] and ash(rgb)==fr['decoded_RGB_sha256']
   with np.load(check(crops[fi]),allow_pickle=False) as z:cp={k:z[k].copy() for k in ('global_index','rgb96','valid96','crop_constructed')}
   box=f['context_box_xyxy']
   if op['valid_box'](box):target,vt=op['sample_target_context'](rgb,box)
   else:target=np.zeros((224,224,3),np.float32);vt=np.zeros((224,224),bool)
   assert ash(target)==f['target224_F32_HWC_sha256'] and ash(vt)==f['target_valid_sha256']
   ad=OUT/'assets'/f'FRAME_{fi:04d}';ad.mkdir();Image.fromarray(rgb).save(ad/'ORIGINAL_FULL_RGB.png');np.save(ad/'TARGET224_EXACT_FLOAT32.npy',target,allow_pickle=False)
   # C9 fills unsupported samples with ImageNet RGB mean before normalization.
   meanfill=np.where(vt[...,None],target,np.array([.485,.456,.406],np.float32)*np.float32(255))
   td=np.rint(meanfill).clip(0,255).astype(np.uint8);Image.fromarray(td).save(ad/'TARGET224_DISPLAY_ONLY.png');Image.fromarray(vt.astype(np.uint8)*255).save(ad/'TARGET224_VALID.png')
   ab=dict(canonical_RGB=fr['canonical_RGB_NPY'],decoded_RGB_sha256=fr['decoded_RGB_sha256'],saved96_packet=crops[fi],fullRGB_PNG=bind(ad/'ORIGINAL_FULL_RGB.png'),target224_exact_float32=bind(ad/'TARGET224_EXACT_FLOAT32.npy'),target224_raw_array_sha256=ash(target),target_valid_array_sha256=ash(vt),target224_display_only=bind(ad/'TARGET224_DISPLAY_ONLY.png'),target224_valid_PNG=bind(ad/'TARGET224_VALID.png'))
   assets[fi]=(rgb,cp,target,vt,td,ab)
  rgb,cp,target,vt,td,ab=assets[fi];kk=np.flatnonzero(cp['global_index']==i);assert len(kk)==1;k=int(kk[0]);crop=cp['rgb96'][k];valid=cp['valid96'][k]
  assert ash(crop)==f['RGB96_HWC_sha256'] and ash(valid)==f['valid96_sha256'] and bool(cp['crop_constructed'][k])==row['feature_constructed']
  assert ash(target)==f['target224_F32_HWC_sha256'] and ash(vt)==f['target_valid_sha256']
  ad=OUT/'assets'/f'ROW_{i:04d}';ad.mkdir();Image.fromarray(crop).save(ad/'RAW96.png');Image.fromarray(crop[32:64,32:64]).save(ad/'CENTRAL32.png');Image.fromarray(valid.astype(np.uint8)*255).save(ad/'VALID96.png')
  image=card(row,rgb,crop,valid,td);path=OUT/'cards'/f'ROW_{i:04d}.png';image.save(path)
  record={**row,'card':bind(path),'assets':ab,'raw96':bind(ad/'RAW96.png'),'central32':bind(ad/'CENTRAL32.png'),'valid96':bind(ad/'VALID96.png'),'raw96_array_sha256':ash(crop),'central32_array_sha256':ash(crop[32:64,32:64]),'valid96_array_sha256':ash(valid),'fixed_representative_categories':chosen.get(i,[]),'postresult_allchanged_appendix':i in changed,'new_label':False,'training_admitted':False}
  records.append(record)
 manifest=jsonl(OUT/'ALL154_GALLERY_MANIFEST.jsonl',records)
 def page(title,selected,extra=''):
  return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:21px system-ui;max-width:2200px;margin:auto;padding:24px}img{width:100%;height:auto}article{margin:45px 0}a{margin-right:20px}</style><h1>'+html.escape(title)+'</h1><p>Full SOURCE_VAL360; 기존 native weak D/E154=D102/E52, D9 inputU 포함. Input-valid145=D93/E52는 별도 진단. 206행은 D/E 정답 밖입니다. E는 external 양성, logit0 고정이며 새 label/학습 편입/모델 선택은 없습니다.</p><p>Native role과 literal q 중심 RGB/소유는 다릅니다. 좌표오차/3px gold가 아닙니다. 모든154 카드와 변경 전수 부록을 보존합니다.</p><p><a href="index.html">전체154</a><a href="diagnostic_representatives.html">고정규칙 대표</a><a href="all_changed_appendix.html">변경 전수 부록</a></p>'+extra+''.join('<article id="row'+str(r['global_index'])+'"><h2>g'+str(r['global_index'])+' · '+html.escape(r['query_id'])+'</h2><p>'+html.escape(', '.join(r['fixed_representative_categories']))+'</p><a href="cards/'+Path(r['card']['path']).name+'"><img loading="lazy" src="cards/'+Path(r['card']['path']).name+'" alt="saved C45 C46 source comparison"></a></article>' for r in selected)+'</html>'
 (OUT/'index.html').write_text(page('C45/C46 · MR 알려진 D/E154 전체',records))
 (OUT/'diagnostic_representatives.html').write_text(page('사전 고정 규칙: 각 confusion/delta의 lowest global_index',[r for r in records if r['global_index'] in chosen],'<p>대표 strata는 결과 후 규칙 적용 진단이며 성공 사례만 선택하지 않습니다. 빈 category는 대체하지 않습니다.</p>'))
 (OUT/'all_changed_appendix.html').write_text(page('알려진 D/E154 중 예측 변경 전수 '+str(len(changed)),[r for r in records if r['global_index'] in changed],'<p>사후 전수 부록이며 고정 대표를 대체하지 않습니다. 변경 없음 행도 전체154 gallery에 남습니다.</p>'))
 qa=[r for r in records if r['global_index'] in chosen or r['global_index'] in changed]
 write(OUT/'ACTUAL_VIEW_QA_REQUIRED.json',dict(completed=False,rule='Actually view every unique fixed representative and all changed known154 rows; remaining154 checked by hash/link/census only',cards=[dict(global_index=r['global_index'],query_id=r['query_id'],card=r['card'],representative_categories=r['fixed_representative_categories'],allchanged=r['postresult_allchanged_appendix']) for r in qa],actually_viewed_images=[]))
 write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),root_trigger=bind(trigger_path),control_complete=bind(RUN/'CONTROL_COMPLETE.json'),all2_freeze=bind(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json'),prediction_packets=packet_bindings,census360=census,gallery154=manifest,selection_before_RGB=bind(OUT/'DIAGNOSTIC_SELECTION_BEFORE_RGB.json'),all154_retained=True,full360=360,known_D_E154=[102,52],known_inputU9_D=True,input_valid145=[93,52],truth_outside_DE206=True,representative_unique_count=len(chosen),allchanged_known154_count=len(changed),QA_required=bind(OUT/'ACTUAL_VIEW_QA_REQUIRED.json'),html=[bind(OUT/n) for n in ('index.html','diagnostic_representatives.html','all_changed_appendix.html')],model_encoder_classifier_checkpoint_fit_calls=0,new_labels=False,training_admitted=False,HTML_browser_rendered=False,actual_visual_QA_pending=True,source_only=True))
 print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'),qa_images=len(qa))))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--trigger',type=Path);a=ap.parse_args()
 if a.prepare and not a.trigger:prepare()
 elif a.trigger and not a.prepare:render(a.trigger)
 else:raise SystemExit('Metadata-only --prepare first; --trigger only after root terminal/all2 completion')
