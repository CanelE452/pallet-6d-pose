"""Fixed prospective current-case evidence, entry-gated and no model calls.

Preparation reads code/JSON descriptors and invented arrays only. Actual order:
completed evaluation -> deterministic IDs seal -> cached RGB/fullframes -> PNGs.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,html,json,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_FIXED_VISUAL_CASE_POLICY_BEFORE_SAVED_EVALUATION.json'
PSHA='4b9178f69666e8f3f36cad3c4f774e09758370c03d7e63833d30331a7869a95f'
B=V/'evaluation_preparation';BCODE=B/'evaluate_saved_seven.py';BPROTOCOL=B/'PROTOCOL.json'
EVAL=V/'saved_seven_current_evaluation_v1/EVALUATION_SEAL.json'
LEDGER=EVAL.parent/'ALL1784_NATIVE_LEDGER.csv'
C11=P/'query_visibility_spatial_target_query_features_20261023_v111/private_evaluation/C11_FROZEN_CURRENT223'
AUDIT=C11/'INPUT_PREPARATION_AUDIT.json';AUDIT_SHA='e282d01560c523ab8d3b95e79cf19881364a71ad63e0d42d6a75aa7f78757a6d'
SEAL=C11/'INFERENCE_SEAL.json';SEAL_SHA='d49f56f81cb59ee76d9a571612acbe945a1148300c1f5a235c939e4cb0764498'
PROTOCOL=H/'PROTOCOL_BEFORE_CROP_LABEL_SCORE_OR_RGB_BODIES.json';OUT=V/'fixed_current_crop_review_v1'
MODELS=('v61','legacy','C16','C35','C36','C37','C38');NEW=MODELS[3:]
FIELDS=('global_index','id','native_corner','group','human_state','query_x','query_y','roi_side','query_available',
 'original_query_valid','initial_reference_error_px','annotation_coordinate_source')
RGB_KEYS=('rgb96','crop_valid96','query_xy','side_L')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):
 path=Path(p)
 with(gzip.open if path.suffix=='.gz'else open)(path,'rt')as f:return json.load(f)
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def check(b):
 assert '_v112'not in str(b['path']).lower();assert bind(b['path'])==b;return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def boolean(x):return x is True or x=='True'
def number(x):
 try:v=float(x);return v if np.isfinite(v)else None
 except (ValueError,TypeError):return None
def material(r):
 if r['group'].startswith('WOOD'):return'WOOD'
 assert r['group'].startswith('SQUARE');return'SQUARE'
def selected_buckets(rows):
 """Four declared buckets per material, global ascending, distinct frame preferred."""
 selected=[];used=set()
 for mat in('WOOD','SQUARE'):
  for bucket in range(4):
   candidates=[]
   for r in rows:
    if material(r)!=mat:continue
    direct=r['human_state']=='DIRECT_VISIBLE';external=r['human_state']=='EXTERNAL_OCCLUDED'
    err=number(r['initial_reference_error_px']);near=direct and r['annotation_coordinate_source']=='manual_click'and err is not None and err<=3 and boolean(r['original_query_valid'])
    newD=any(r[m+'_D_E_U']=='D'for m in NEW);newE=any(r[m+'_D_E_U']=='E'for m in NEW)
    ok=(near and newD and r['v61_D_E_U']=='E',near and r['C16_D_E_U']=='D'and newE,
        external and r['C16_D_E_U']=='E'and newD,external and r['C16_D_E_U']=='D'and newE)[bucket]
    if ok:candidates.append(r)
   candidates.sort(key=lambda r:int(r['global_index']))
   distinct=[r for r in candidates if r['id']not in used];choice=(distinct or candidates or[None])[0]
   if choice is not None:used.add(choice['id'])
   selected.append(dict(material=mat,bucket_index=bucket,eligible_rows=len(candidates),empty=choice is None,
    selected_global_index=None if choice is None else int(choice['global_index']),selected_id=None if choice is None else choice['id'],
    selected_native=None if choice is None else int(choice['native_corner']),duplicate_frame_only_if_necessary=bool(choice is not None and not distinct)))
 return selected
def generated():
 def row(i,id='A',state='DIRECT_VISIBLE',near='1.0',source='manual_click',mat='WOOD_PRIMARY105',**changes):
  a=dict(global_index=str(i),id=id,native_corner=str(i%8),group=mat,human_state=state,initial_reference_error_px=near,annotation_coordinate_source=source,original_query_valid='True')
  for m in MODELS:a[m+'_D_E_U']='D'
  a.update(changes);return a
 rr=[row(0,**{'v61_D_E_U':'E'}),row(1,**{'C35_D_E_U':'E'}),row(8,id='B',**{'C35_D_E_U':'E'}),
  row(9,id='C',state='EXTERNAL_OCCLUDED',**{'C16_D_E_U':'E'}),row(10,id='D',state='EXTERNAL_OCCLUDED',**{'C35_D_E_U':'E'})]
 a=selected_buckets(rr);assert[a[i]['selected_global_index']for i in range(4)]==[0,8,9,10]and all(r['empty']for r in a[4:])
 rr=[row(0,source='projected',**{'v61_D_E_U':'E'}),row(1,near='3.1',**{'v61_D_E_U':'E'})];assert all(r['empty']for r in selected_buckets(rr))
 a=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);copy=a.copy();center=a[32:64,32:64].copy()
 from PIL import Image,__version__
 zoom=np.array(Image.fromarray(a).resize((192,192),Image.Resampling.NEAREST));assert np.array_equal(zoom,np.repeat(np.repeat(a,2,0),2,1))and np.array_equal(a,copy)and center.shape==(32,32,3)
 return dict(completed=True,PASS=True,invented_only=True,distinct_frame_preference_global_order_empty_buckets=True,manual_near_requires_original_manual_provenance_and3px=True,
  raw96_nearest_display_and_center32_exact_no_source_mutation=True,Pillow=__version__,private_labels_scores_RGB_or_models_read=0)
def prepare():
 assert sha(POLICY)==PSHA and sha(AUDIT)==AUDIT_SHA and sha(SEAL)==SEAL_SHA
 policy=read(POLICY);assert policy['models']==list(MODELS)and policy['maximum_cases']==8
 audit=read(AUDIT);seal=read(SEAL);facts=next(b for b in seal['outputs']if Path(b['path']).name=='CONTEXT_FRAME_FACTS_INPUT_ONLY.jsonl')
 b=audit['prior_artifact_bindings'];proof=generated();assert proof['Pillow']==audit['decoder']['Pillow']=='12.2.0'
 write(H/'GENERATED_INVENTED_CHECKS.json',proof)
 evaluation=read(EVAL);assert evaluation['completed']and evaluation['PASS']and evaluation['selected_model']is None
 p=dict(completed=True,prepared_only=True,code=bind(__file__),case_policy=bind(POLICY),evaluator_code=bind(BCODE),evaluator_protocol=bind(BPROTOCOL),evaluation_seal_metadata=bind(EVAL),
  required_evaluation_seal=str(EVAL),required_ledger=str(LEDGER),required_ledger_fields=list(FIELDS),models=list(MODELS),
  original_cache=b['current_model_input'],prepared_frames=b['prepared_inputs'],feature_metadata=b['feature_metadata'],
  C11_input_audit=bind(AUDIT),C11_inference_seal=bind(SEAL),C11_frame_facts=facts,
  actual_cache_array_whitelist=list(RGB_KEYS),P3_and_box_array_deserialized=False,
  pipeline='Root entry+completedEVALseal -> sealed1784ledger+original q/L metadata -> CASE_IDS_BEFORE_ANY_RGB.json -> only then cachedRGB/fullframe reads -> original96/center32 pixelsha+frame8batchhash and decodedRGBsha checks ->boards/index/receipt.',
  q_contract='Final B ledger q/L are original F64 metadata. Verify exact original F64 prepared q/L, then overlay/display these unchanged. Cached working q/L separately match FP32 casts. No q correction, GTcenter, or target recrop.',
  crop_contract='Original uint8CHW96 cache, center32 exact indices32:64, raw PNG bytes decode to exactsavedpixels. Nearest2x96 and6x32 display only; validmasks separate. Fullframe wholeimage display is not the DINOtarget224 tensor.',
  fullframe_decoder=audit['decoder'],selection='Exactly four prospectivebuckets perWOOD/SQUARE, firstglobalidx preferringnotusedframe; emptyexplicit. No image/score magnitude based replacement.',
  baseline_scope='v61/legacy used same query/cohort but different operators; these are DINOactualinput crops for C16/C35-C38.',
  labels_scores_RGB_or_caches_or_weights_body_hash_reads_during_prepare=0,actual_model_or_encoder_R0_PnP_calls=0,selected_model=None)
 write(PROTOCOL,p)
 template=dict(approved_saved_crop_visualization=False,code=bind(__file__),protocol=bind(PROTOCOL),case_policy=bind(POLICY),
  evaluation_seal=bind(EVAL),
  allow_saved_labels_scores_and_original_RGB_cache_for_render=True,case_ID_seal_before_RGB_required=True,
  new_model_encoder_R0_or_fit=False,private_weights=False,V112=False,PnP=False,q_L_correction=False,model_threshold_or_case_rule_choice=False,automatic_retry=False)
 write(H/'ROOT_CROP_VISUAL_ENTRY_TEMPLATE.json',template)
 ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(H/'GENERATED_INVENTED_CHECKS.json'),template=bind(H/'ROOT_CROP_VISUAL_ENTRY_TEMPLATE.json'),
  labels_scores_RGB_cache_or_weight_body_hash_reads=0,actual_model_calls=0,command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).resolve())+' --entry '+str(V/'ROOT_APPROVED_FIXED_CROP_VISUAL_ENTRY.json'))
 write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)
def font(size):
 from PIL import ImageFont
 return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def textlines(draw,xy,text,width=115,size=14):
 import textwrap
 x,y=xy
 for line in textwrap.wrap(text,width=width):draw.text((x,y),line,font=font(size),fill='black');y+=size+5
 return y
def draw_case(case,row,frame,q,L,rgb96,valid96,full,folder):
 from PIL import Image,ImageDraw,ImageOps
 k=int(row['native_corner']);idx=int(row['global_index']);image=Image.fromarray(full.copy())
 # Overlay is confined to this wholeimage canvas before any paste.
 d=ImageDraw.Draw(image);hh,ww=full.shape[:2]
 for slot,point in enumerate(q):
  if not np.isfinite(point).all()or(point==[-1.,-1.]).all():continue
  x,y=point;color='#ff22bb'if slot==k else'#00dddd';rad=max(4,ww/250)
  d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=3);d.text((x+rad,y-rad),str(slot),font=font(max(20,int(ww/55))),fill=color)
 raw=rgb96[k].transpose(1,2,0).copy();central=raw[32:64,32:64].copy();mask=valid96[k,0]
 rawpath=folder/f'CASE_{idx:04d}_ORIGINAL_RGB96.png';centerpath=folder/f'CASE_{idx:04d}_CENTRAL32.png'
 Image.fromarray(raw).save(rawpath);Image.fromarray(central).save(centerpath)
 assert np.array_equal(np.asarray(Image.open(rawpath)),raw)and np.array_equal(np.asarray(Image.open(centerpath)),central)
 board=Image.new('RGB',(1120,850),'white');draw=ImageDraw.Draw(board)
 header=f"{case['material']} bucket{case['bucket_index']+1} | global {idx} | {row['id']} native{k}"
 draw.text((16,12),header,font=font(18),fill='black')
 err=number(row['initial_reference_error_px']);state=row['human_state'];draw.text((16,42),f'Existing human state: {state}; fixed-reference error: {err}; provenance: {row["annotation_coordinate_source"]}',font=font(13),fill='black')
 draw.text((16,65),f'Original q=({q[k,0]:.6f}, {q[k,1]:.6f}), L={L:.6f}; inputvalid={row["query_available"]}; magenta=selected q, cyan=other original q',font=font(13),fill='black')
 # Whole-frame display preserves the native aspect; crop displays remain exact.
 fitted=ImageOps.contain(image,(640,360),Image.Resampling.BILINEAR)
 letterbox=Image.new('RGB',(640,360),'#eeeeee')
 letterbox.paste(fitted,((640-fitted.width)//2,(360-fitted.height)//2))
 board.paste(letterbox,(16,112))
 for x,title,data,scale in((680,'Original saved RGB96',raw,2),(888,'Exact central32 slice',central,6)):
  draw.text((x,92),title,font=font(14),fill='black');board.paste(Image.fromarray(data).resize((192,192),Image.Resampling.NEAREST),(x,112))
 board.paste(Image.fromarray((mask*255).astype(np.uint8)).resize((96,96),Image.Resampling.NEAREST),(680,330))
 board.paste(Image.fromarray((mask[32:64,32:64]*255).astype(np.uint8)).resize((96,96),Image.Resampling.NEAREST),(888,330))
 draw.text((680,435),'Valid support (white)',font=font(12),fill='black');draw.text((888,435),'Central support',font=font(12),fill='black')
 draw.text((16,490),'Model     decision / original response                 saved visibility score         entity z       quality z',font=font(15),fill='black')
 for j,m in enumerate(MODELS):
  response=row[m+'_original_response'];decision=row[m+'_D_E_U']
  if m in('C16',*NEW):score=f'z={float(row[m+"_raw_visibility_logit"]):+.7g}'
  else:score=('pPresent='+row.get('v61_saved_p_present',''))if m=='v61'else('pD='+row.get('legacy_saved_p_DIRECT',''))
  draw.text((16,520+j*27),f'{m:<7} {decision} / {response}',font=font(13),fill='black')
  draw.text((590,520+j*27),score,font=font(13),fill='black')
  if m in NEW:
   draw.text((835,520+j*27),f'{float(row[m+"_raw_entity_logit"]):+.7g}',font=font(13),fill='black')
   draw.text((975,520+j*27),f'{float(row[m+"_raw_quality_logit"]):+.7g}',font=font(13),fill='black')
 textlines(draw,(16,722),'Actual saved DINO query pixels; display enlargement only. Raw96/central32 files have no overlays. v61/legacy consumed different operators. Human reference error is a fixed diagnostic, not new pixel-quality gold. Auxiliary scores never gate primary visibility. No GT-centering or q/L correction.',width=145,size=13)
 panel=folder/f'CASE_{idx:04d}_PANEL.png';board.save(panel)
 # Independently verify display regions remain exact nearest replicas of the saved crop.
 rendered=np.asarray(Image.open(panel));assert np.array_equal(rendered[112:304,680:872],np.repeat(np.repeat(raw,2,0),2,1))
 assert np.array_equal(rendered[112:304,888:1080],np.repeat(np.repeat(central,6,0),6,1))
 return dict(global_index=idx,id=row['id'],native_corner=k,case=case,original_query_xy=q[k].tolist(),original_L=L,
  RGB96_CHW_sha256=array_sha(rgb96[k]),RGB96_HWC_sha256=array_sha(raw),valid96_sha256=array_sha(valid96[k]),central32_HWC_sha256=array_sha(central),
  raw96=bind(rawpath),central32=bind(centerpath),panel=bind(panel),display_crop_pixels_verified_exact_nearest=True)
def run(entry):
 e=read(entry);assert e['approved_saved_crop_visualization']and e['allow_saved_labels_scores_and_original_RGB_cache_for_render']and e['case_ID_seal_before_RGB_required']
 assert all(e[k]is False for k in('new_model_encoder_R0_or_fit','private_weights','V112','PnP','q_L_correction','model_threshold_or_case_rule_choice','automatic_retry'))
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['case_policy']==bind(POLICY)and sha(POLICY)==PSHA and not OUT.exists()
 p=read(PROTOCOL);assert p['code']==e['code']and p['case_policy']==e['case_policy']
 for k in('evaluator_code','evaluator_protocol','C11_input_audit','C11_inference_seal'):check(p[k])
 assert Path(e['evaluation_seal']['path']).resolve()==EVAL.resolve()
 seal=read(check(e['evaluation_seal']));assert seal['completed']and seal['PASS']and seal['selected_model']is None and not seal['source_admission_changed']
 assert seal['code']==p['evaluator_code']and seal['protocol']==p['evaluator_protocol']
 lb=next(b for b in seal['outputs']if Path(b['path']).resolve()==LEDGER.resolve())
 with check(lb).open()as f:rows=list(csv.DictReader(f))
 assert len(rows)==1784 and all(int(r['global_index'])==i and int(r['native_corner'])==i%8 for i,r in enumerate(rows))
 assert all(set(FIELDS).issubset(r)and all(r[m+'_D_E_U']in('D','E','U')for m in MODELS)for r in rows)
 frames=read(check(p['prepared_frames']));metadata=read(check(p['feature_metadata']));facts=lines(check(p['C11_frame_facts']))
 assert len(frames)==len(metadata)==len(facts)==223
 for i,(frame,meta,fact)in enumerate(zip(frames,metadata,facts)):
  assert frame['id']==meta['id']==fact['id']and fact['frame_index']==i
  for k in range(8):
   r=rows[i*8+k];assert r['id']==frame['id']
   assert np.array_equal(np.asarray([float(r['query_x']),float(r['query_y'])],np.float64),np.asarray(frame['initial_points'],np.float64)[k],equal_nan=True)
   assert float(r['roi_side'])==float(meta['side'])
 cases=selected_buckets(rows);OUT.mkdir()
 casefile=OUT/'CASE_IDS_BEFORE_ANY_RGB.json';write(casefile,dict(completed=True,PASS=True,code=bind(__file__),entry=bind(entry),case_policy=p['case_policy'],evaluation_seal=e['evaluation_seal'],ledger=lb,
  cases=cases,selected_global_indices=[r['selected_global_index']for r in cases if not r['empty']],RGB_body_or_cache_reads_so_far=0,selection_rule_unchanged=True,selected_model=None))
 try:
  # First RGB/cached-pixel file hash/read is strictly after the durable ID seal.
  from PIL import Image,__version__
  assert __version__==p['fullframe_decoder']['Pillow']=='12.2.0'
  with np.load(check(p['original_cache']),allow_pickle=False)as z:data={k:z[k].copy()for k in RGB_KEYS}
  assert data['rgb96'].shape==(1784,3,96,96)and data['rgb96'].dtype==np.uint8
  assert data['crop_valid96'].shape==(1784,1,96,96)and data['crop_valid96'].dtype==bool
  artifacts=[];images={};verifiedframes={}
  for c in cases:
   if c['empty']:continue
   idx=c['selected_global_index'];fi=idx//8;frame=frames[fi];meta=metadata[fi];fact=facts[fi]
   q=np.asarray(frame['initial_points'],np.float64)[:8];L=float(meta['side']);sl=slice(fi*8,fi*8+8)
   assert np.array_equal(data['query_xy'][sl],q.astype(np.float32),equal_nan=True)and np.array_equal(data['side_L'][sl],np.full(8,np.float32(L)))
   rgb=data['rgb96'][sl];valid=data['crop_valid96'][sl]
   assert array_sha(rgb)==fact['RGB96_original_cached_sha256']and array_sha(valid)==fact['valid96_original_cached_sha256']
   if fi not in images:
    imagebinding=dict(path=frame['image'],sha256=frame['image_sha256']);check(imagebinding)
    full=np.asarray(Image.open(frame['image']).convert('RGB'),dtype=np.uint8).copy()
    assert list(full.shape[:2])==frame['raw_hw']and imagebinding==fact['image']
    if fact['decoded_RGB_sha256']is not None:assert array_sha(full)==fact['decoded_RGB_sha256']
    images[fi]=full;verifiedframes[str(fi)]=dict(id=frame['id'],image=imagebinding,decoded_RGB_sha256=array_sha(full),
      C11_decoded_hash_available=fact['decoded_RGB_sha256']is not None,original_frame8_RGB96_CHW_sha256=array_sha(rgb),original_frame8_valid96_sha256=array_sha(valid))
   artifacts.append(draw_case(c,rows[idx],frame,q,L,rgb,valid,images[fi],OUT))
  content=['<!doctype html><meta charset="utf-8"><title>Fixed original-query crop evidence</title><h1>Fixed original-query crop evidence</h1><p>Prospective 4 buckets × WOOD/SQUARE; all seven frozen predictions. Saved input evidence, no new inference.</p>']
  for c in cases:
   content.append('<h2>'+html.escape(c['material']+' bucket '+str(c['bucket_index']+1))+'</h2>')
   if c['empty']:content.append('<p>EMPTY — no eligible row; no replacement.</p>')
   else:
    a=next(a for a in artifacts if a['global_index']==c['selected_global_index']);content.append('<img style="max-width:1120px;width:100%" src="'+Path(a['panel']['path']).name+'">')
  index=OUT/'index.html';index.write_text('\n'.join(content))
  receipt=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),case_ids_before_RGB=bind(casefile),evaluation_seal=e['evaluation_seal'],
    cases=cases,panels=artifacts,verified_original_frames=verifiedframes,index=bind(index),selected_cases=len(artifacts),empty_buckets=sum(c['empty']for c in cases),
    original_cache=p['original_cache'],cache_deserialized_keys=list(RGB_KEYS),P3_box_model_features_or_weights_deserialized=False,
    new_model_encoder_R0_optimizer_PnP_or_q_L_changes=0,case_selection_changed_after_RGB=False,all7_and_new_aux_scores_reported=True,selected_model=None)
  write(OUT/'CROP_RENDER_COMPLETE.json',receipt);print(json.dumps(dict(completed=True,PASS=True,receipt=bind(OUT/'CROP_RENDER_COMPLETE.json'))),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,case_ID_seal_preserved=bind(casefile),automatic_retry=False,traceback=traceback.format_exc()));raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry',type=Path)
 a=parser.parse_args();prepare()if a.prepare else run(a.entry)
