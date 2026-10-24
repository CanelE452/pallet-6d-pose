"""Prospective V169 fixed crop readout. No current bodies during preparation.
Evaluation must freeze first; separate root visual entry then seals IDs before pixels.
"""
from pathlib import Path
import argparse,csv,hashlib,html,importlib.util,itertools,json,sys
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PARENT=P/'query_visibility_separated_frozen_current_benchmark_20261024_v156/crop_review_preparation/render_fixed_saved_crops_alias_repair.py'
PARENT_PROTOCOL=PARENT.parent/'PROTOCOL_BEFORE_ALIAS_REPAIR_RENDER.json'
PARENT_SHA='1489d086d1c72d9ff92c6b5677fbd6eeb8d56ff663f831134f8e58adfb0ff80c'
PARENT_PROTOCOL_SHA='f0fe65cf9e677ec68598c5064bb94e780542d9cf60d848beaff8b7cd45c0f8c5'
POLICY=H/'SELECTION_RULE_BEFORE_CURRENT_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_CURRENT_CROP_BODIES.json'
EVAL=V/'saved_seven_condition_evaluation_v1/EVALUATION_SEAL.json'
LEDGER=EVAL.parent/'ALL1784_NATIVE_LEDGER.csv'
OUT=V/'fixed_current_crop_review_v1'
CONDITIONS=('C45_ORIGINAL0','C45_PUBLICCAL_FULL','C46_ORIGINAL0','C46_PUBLICCAL_FULL')
BASELINES=('v61','legacy','C16');ALL_CONDITIONS=BASELINES+CONDITIONS
STRATA=('TP','FN','FP','TN','U')
RGB_KEYS=('rgb96','crop_valid96','query_xy','side_L')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 with Path(p).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def number(x):
 try:v=float(x);return v if np.isfinite(v) else None
 except (TypeError,ValueError):return None
def boolean(x):return x is True or x=='True'
def material(r):
 if r['group'].startswith('WOOD'):return 'WOOD'
 assert r['group'].startswith('SQUARE');return 'SQUARE'
def stratum(r,condition):
 pred=r[condition+'_D_E_U'];assert pred in ('D','E','U')
 if pred=='U':return 'U' # Any original row with factual model U; unknown truth is not recoded.
 truth={'DIRECT_VISIBLE':'D','EXTERNAL_OCCLUDED':'E'}.get(r['human_state'])
 return {('E','E'):'TP',('E','D'):'FN',('D','E'):'FP',('D','D'):'TN'}.get((truth,pred))
def select_fixed(rows):
 rows=sorted(rows,key=lambda r:int(r['global_index']));buckets=[];selected={}
 for condition in CONDITIONS:
  for mat in ('WOOD','SQUARE'):
   for category in STRATA:
    candidates=[r for r in rows if material(r)==mat and stratum(r,condition)==category]
    key=f'{condition}::{mat}::{category}';choice=candidates[0] if candidates else None
    buckets.append(dict(key=key,eligible_rows=len(candidates),selected_global_index=None if choice is None else int(choice['global_index']),empty=choice is None))
    if choice is not None:selected.setdefault(int(choice['global_index']),[]).append(key)
   candidates=[r for r in rows if material(r)==mat and r['human_state']=='DIRECT_VISIBLE' and r['annotation_coordinate_source']=='manual_click' and boolean(r['original_query_valid']) and number(r['initial_reference_error_px']) is not None and number(r['initial_reference_error_px'])<=3 and r[condition+'_D_E_U']=='E']
   key=f'{condition}::{mat}::MANUAL_REFERENCE_LE3_DIRECT_PRED_E';choice=candidates[0] if candidates else None
   buckets.append(dict(key=key,eligible_rows=len(candidates),selected_global_index=None if choice is None else int(choice['global_index']),empty=choice is None,physical_3px_gold=False))
   if choice is not None:selected.setdefault(int(choice['global_index']),[]).append(key)
 assert len(buckets)==48 and len(selected)<=48
 return buckets,selected

def prepare():
 assert sha(PARENT)==PARENT_SHA and sha(PARENT_PROTOCOL)==PARENT_PROTOCOL_SHA
 old=read(PARENT_PROTOCOL) # Descriptor/code only; never check/hash/read its current inputs here.
 policy=dict(completed=True,prepared_before_current_bodies=True,conditions=list(CONDITIONS),materials=['WOOD','SQUARE'],strata=list(STRATA),
  selection='For each condition×material×TP/FN/FP/TN/U choose lowest original global_index; empty remains empty, no replacement or frame preference. Dedup exact global_index across48 buckets.',
  U='U stratum includes any original row with decodedU. TP/FN/FP/TN require existing DIRECT_VISIBLE or EXTERNAL_OCCLUDED; self/uncertain are not negatives.',
  extra8='Per condition/material lowest global_index with existing DIRECT_VISIBLE, manual_click reference, finite existing initial_reference_error_px<=3, original_query_valid and predictionE. Fixed reference diagnostic, not physical3px gold.',
  maximum_unique_cases=48,selection_before_RGB=True,changed_ledger='Keep full1784 ledger plus exhaustive differences for every unordered pair of all7 baseline/new conditions; no omitted changed row. Changed ledger is not extra RGB selection.',
  no_new_labels_or_q_L_correction=True,no_actual_current_bodies_read=True)
 if not POLICY.exists():write(POLICY,policy)
 else:assert read(POLICY)==policy
 keys=('original_cache','prepared_frames','feature_metadata','C11_frame_facts','fullframe_decoder')
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),parent_alias_renderer=bind(PARENT),parent_alias_protocol=bind(PARENT_PROTOCOL),
  **{k:old[k] for k in keys},conditions=list(CONDITIONS),baselines=list(BASELINES),evaluation_seal_path=str(EVAL),ledger_path=str(LEDGER),output=str(OUT),
  pipeline='Separate approved visual entry + frozen completed evaluation seal -> unchanged1784 ledger -> selection manifest durable BEFORE any RGB/cache hash/read -> exact original q/L metadata -> original pixel-cache whitelist only -> safe original alias SHA decode -> clean PNGs/card.',
  fullRGB_alias='Reuse V156 safe_original_image_path exactly; original descriptor SHA, symlink expansion guard, original alias descriptor equality, decoded RGB SHA. Do not require alias string to equal canonical spelling.',
  clean_crop='Original uint8CHW96 and boolvalid; save clean HWC raw96/central32[32:64]/valid PNG before overlays; verify decoded bytes. Whole-frame context with all originalnativeq is display only, not a new DINOtarget224 claim.',
  evaluator_schema='C45_ORIGINAL0/C45_PUBLICCAL_FULL/C46_ORIGINAL0/C46_PUBLICCAL_FULL *_D_E_U,*_original_response,*_raw_visibility_logit,*_threshold_float64,*_decision_score64 plus unchanged baseline responses and original identity/truth/qL/reference fields; final evaluator binding supplied only after freeze.',
  current_feature_label_score_RGB_body_or_cache_hash_reads=0,new_model_encoder_fit_PnP_qL_calls=0,actual_rendering=False)
 write(PROTOCOL,protocol);print(json.dumps(dict(prepared_only=True,code=bind(__file__),policy=bind(POLICY),protocol=bind(PROTOCOL))))

def parent_helpers(protocol):
 assert bind(PARENT)==protocol['parent_alias_renderer'] and bind(PARENT_PROTOCOL)==protocol['parent_alias_protocol']
 spec=importlib.util.spec_from_file_location('V156_exact_alias_and_pixel_helpers',PARENT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write_csv(path,rows,fields):
 with Path(path).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 return bind(path)

def draw_case(row,memberships,q,L,rgb96,valid96,full,folder,parent):
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',n)
 idx=int(row['global_index']);k=int(row['native_corner']);raw=rgb96[k].transpose(1,2,0).copy();center=raw[32:64,32:64].copy();vm=valid96[k,0]
 rawpath=folder/f'CASE_{idx:04d}_RAW96.png';cp=folder/f'CASE_{idx:04d}_CENTRAL32.png';vp=folder/f'CASE_{idx:04d}_VALID96.png'
 Image.fromarray(raw).save(rawpath);Image.fromarray(center).save(cp);Image.fromarray(vm.astype(np.uint8)*255).save(vp)
 assert np.array_equal(np.asarray(Image.open(rawpath)),raw) and np.array_equal(np.asarray(Image.open(cp)),center)
 fullcopy=Image.fromarray(full.copy());fd=ImageDraw.Draw(fullcopy);hh,ww=full.shape[:2]
 for slot,point in enumerate(q):
  if not np.isfinite(point).all() or (point==[-1.,-1.]).all():continue
  x,y=point;color='#ff22bb' if slot==k else '#00dddd';rad=max(4,ww/250)
  fd.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=3);fd.text((x+rad,y-rad),str(slot),font=font(max(20,int(ww/55))),fill=color)
 if np.isfinite(q[k]).all() and not (q[k]==[-1.,-1.]).all():
  x,y=q[k];fd.rectangle((x-L/2,y-L/2,x+L/2,y+L/2),outline='#ff22bb',width=2)
 board=Image.new('RGB',(2080,1400),'#f5f8fb');dd=ImageDraw.Draw(board)
 dd.text((24,18),f"{material(row)} · g{idx:04d} · {row['id']} · native{k} · 기존상태 {row['human_state']}",font=font(30),fill='#20364c')
 dd.text((24,65),f"원래 q=({q[k,0]:.8g},{q[k,1]:.8g}) L={L:.8g} · inputvalid={row['query_available']} · 선택 stratum 수={len(memberships)}",font=font(25),fill='#3b526a')
 dd.text((24,107),f"기존 reference오차={row['initial_reference_error_px']}px · 출처={row['annotation_coordinate_source']} · physical3px gold 아님",font=font(24),fill='#965111')
 dd.text((24,160),'전체 원본 표시복사본: 자홍=선택q/원래qL ROI, 청록=다른nativeq',font=font(23),fill='#20364c')
 fitted=ImageOps.contain(fullcopy,(1050,590),Image.Resampling.BILINEAR);bg=Image.new('RGB',(1050,590),'#e3e7eb');bg.paste(fitted,((1050-fitted.width)//2,(590-fitted.height)//2));board.paste(bg,(24,202))
 dd.text((1120,160),'overlay 없는 실제 raw96 · 4×',font=font(23),fill='#20364c');board.paste(Image.fromarray(raw).resize((384,384),Image.Resampling.NEAREST),(1120,202))
 dd.text((1595,160),'동일 중앙32 · 8×',font=font(23),fill='#20364c');board.paste(Image.fromarray(center).resize((256,256),Image.Resampling.NEAREST),(1595,202))
 dd.text((1120,620),'원래 validity96 · 흰색=supported',font=font(22),fill='#20364c');board.paste(Image.fromarray(vm.astype(np.uint8)*255).resize((192,192),Image.Resampling.NEAREST),(1120,660))
 dd.text((1390,680),f"center4 valid={int(vm[47:49,47:49].sum())}/4",font=font(24),fill='#3b526a')
 dd.text((24,850),'조건                       예측  raw visibility logit       고정 threshold64              decision_score64',font=font(25),fill='#20364c')
 for j,condition in enumerate(CONDITIONS):
  yy=901+j*49;dd.text((24,yy),condition,font=font(25),fill='#20364c');dd.text((465,yy),row[condition+'_D_E_U'],font=font(27),fill='#20364c')
  for x,key in ((590,'raw_visibility_logit'),(1025,'threshold_float64'),(1510,'decision_score64')):
   dd.text((x,yy),str(row[condition+'_'+key]),font=font(24),fill='#3b526a')
 for j,name in enumerate(BASELINES):
  yy=1113+j*40;extra=('saved pPresent='+row.get('v61_saved_p_present','undefined')) if name=='v61' else ('saved pDIRECT='+row.get('legacy_saved_p_DIRECT','undefined')) if name=='legacy' else ('raw external logit='+row.get('C16_raw_visibility_logit','undefined'))
  dd.text((24,yy),f"{name}: {row[name+'_D_E_U']} / {row[name+'_original_response']} · {extra}",font=font(23),fill='#3b526a')
 dd.text((24,1245),'원래 native 상태·literal q 중심·fixed reference 오차는 서로 다른 진단입니다. 새 owner/native/quality gold·q보정 없음.',font=font(25),fill='#965111')
 dd.text((24,1293),'FullRGB context는 표시용이며 DINO target224 tensor 주장 아님 · raw96/중앙32에는 overlay 없음 · baseline별 입력 operator 차이 유지',font=font(23),fill='#3b526a')
 dd.text((24,1337),'고정 strata 설명 사례만 · empty 대체 없음 · 변경 전수 ledger는 별도 · 모델/문턱/학습 선택 및 전수 이미지 검사 주장 없음',font=font(23),fill='#3b526a')
 path=folder/f'CASE_{idx:04d}_CARD.png';board.save(path)
 a=np.asarray(Image.open(path));assert np.array_equal(a[202:586,1120:1504],np.repeat(np.repeat(raw,4,0),4,1));assert np.array_equal(a[202:458,1595:1851],np.repeat(np.repeat(center,8,0),8,1))
 return dict(global_index=idx,id=row['id'],native_corner=k,material=material(row),memberships=memberships,original_query_xy=q[k].tolist(),original_L=L,raw96=bind(rawpath),central32=bind(cp),valid96=bind(vp),card=bind(path),RGB96_CHW_sha256=parent.array_sha(rgb96[k]),RGB96_HWC_sha256=parent.array_sha(raw),central32_HWC_sha256=parent.array_sha(center),valid96_sha256=parent.array_sha(valid96[k]),clean_saved_pixels_and_nearest_display_exact=True,physical3px_gold=False,new_labels=False)

def run(entry):
 e=read(entry);assert e['approved_saved_crop_visualization'] is True and e['allow_saved_labels_scores_and_original_RGB_cache_for_render'] is True
 assert all(e[k] is False for k in ('new_model_encoder_R0_or_fit','private_weights','V112','PnP','q_L_correction','model_threshold_or_case_rule_choice','automatic_retry'))
 assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['case_policy']==bind(POLICY) and not OUT.exists()
 p=read(PROTOCOL);assert p['code']==e['code'] and p['policy']==e['case_policy'];parent=parent_helpers(p)
 assert Path(e['evaluation_seal']['path']).resolve()==EVAL.resolve()
 seal=parent.read(parent.check(e['evaluation_seal']));assert seal['completed'] and seal['PASS'] and seal.get('selected_model') is None
 assert seal['code']==e['evaluator_code'] and seal['protocol']==e['evaluator_protocol']
 parent.check(e['evaluator_code']);parent.check(e['evaluator_protocol'])
 lb=next(b for b in seal['outputs'] if Path(b['path']).resolve()==LEDGER.resolve())
 with parent.check(lb).open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==1784 and all(int(r['global_index'])==i and int(r['native_corner'])==i%8 for i,r in enumerate(rows))
 required=('global_index','id','native_corner','group','human_state','query_x','query_y','roi_side','query_available','original_query_valid','initial_reference_error_px','annotation_coordinate_source')
 assert all(set(required).issubset(r) and all(r[c+'_D_E_U'] in ('D','E','U') for c in ALL_CONDITIONS) for r in rows)
 assert all(all(c+'_'+key in r for c in CONDITIONS for key in ('raw_visibility_logit','threshold_float64','decision_score64')) for r in rows)
 buckets,selected=select_fixed(rows);OUT.mkdir();selected_ids=sorted(selected)
 # Durable IDs are the first output; no current RGB/cache hash, load, or decode occurs above.
 casefile=OUT/'CASE_IDS_BEFORE_ANY_RGB.json';write(casefile,dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),policy=bind(POLICY),entry=bind(entry),evaluation_seal=e['evaluation_seal'],ledger=lb,buckets=buckets,selected_global_indices=selected_ids,memberships={str(k):v for k,v in selected.items()},maximum48=True,RGB_or_cache_body_hash_reads_so_far=0,selection_rule_unchanged=True))
 allcsv=OUT/'ALL1784_CONDITION_COMPARISON_LEDGER.csv';write_csv(allcsv,rows,list(rows[0]));alljson=OUT/'ALL1784_CONDITION_COMPARISON_LEDGER.jsonl'
 with alljson.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 changes=[]
 for r in rows:
  for left,right in itertools.combinations(ALL_CONDITIONS,2):
   if r[left+'_D_E_U']==r[right+'_D_E_U']:continue
   changes.append({**{k:r[k] for k in ('global_index','id','native_corner','group','human_state')},'left_condition':left,'right_condition':right,'left_D_E_U':r[left+'_D_E_U'],'right_D_E_U':r[right+'_D_E_U'],'left_original_response':r[left+'_original_response'],'right_original_response':r[right+'_original_response']})
 fields=['global_index','id','native_corner','group','human_state','left_condition','right_condition','left_D_E_U','right_D_E_U','left_original_response','right_original_response'];change_csv=OUT/'ALL_CHANGED_ROW_PAIR_LEDGER.csv';write_csv(change_csv,changes,fields);change_json=OUT/'ALL_CHANGED_ROW_PAIR_LEDGER.jsonl'
 with change_json.open('x') as f:
  for r in changes:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 try:
  frames=parent.read(parent.check(p['prepared_frames']));metadata=parent.read(parent.check(p['feature_metadata']));facts=parent.lines(parent.check(p['C11_frame_facts']))
  assert len(frames)==len(metadata)==len(facts)==223
  for i,(frame,meta,fact) in enumerate(zip(frames,metadata,facts)):
   assert frame['id']==meta['id']==fact['id'] and fact['frame_index']==i
   for k in range(8):
    r=rows[i*8+k];assert r['id']==frame['id'];assert np.array_equal(np.asarray([float(r['query_x']),float(r['query_y'])],np.float64),np.asarray(frame['initial_points'],np.float64)[k],equal_nan=True);assert float(r['roi_side'])==float(meta['side'])
  from PIL import Image,__version__
  assert __version__==p['fullframe_decoder']['Pillow']=='12.2.0'
  with np.load(parent.check(p['original_cache']),allow_pickle=False) as z:data={k:z[k].copy() for k in RGB_KEYS}
  assert data['rgb96'].shape==(1784,3,96,96) and data['rgb96'].dtype==np.uint8 and data['crop_valid96'].shape==(1784,1,96,96) and data['crop_valid96'].dtype==bool
  images={};verified={};artifacts=[]
  for idx in selected_ids:
   fi=idx//8;frame=frames[fi];meta=metadata[fi];fact=facts[fi];q=np.asarray(frame['initial_points'],np.float64)[:8];L=float(meta['side']);sl=slice(fi*8,fi*8+8)
   assert np.array_equal(data['query_xy'][sl],q.astype(np.float32),equal_nan=True) and np.array_equal(data['side_L'][sl],np.full(8,np.float32(L)))
   rgb=data['rgb96'][sl];valid=data['crop_valid96'][sl];assert parent.array_sha(rgb)==fact['RGB96_original_cached_sha256'] and parent.array_sha(valid)==fact['valid96_original_cached_sha256']
   if fi not in images:
    imagebinding=dict(path=frame['image'],sha256=frame['image_sha256']);safe=parent.safe_original_image_path(imagebinding)
    full=np.asarray(Image.open(safe).convert('RGB'),dtype=np.uint8).copy();assert list(full.shape[:2])==frame['raw_hw'] and imagebinding==fact['image']
    if fact['decoded_RGB_sha256'] is not None:assert parent.array_sha(full)==fact['decoded_RGB_sha256']
    images[fi]=full;verified[str(fi)]=dict(id=frame['id'],image=imagebinding,safe_canonical_path=str(safe),decoded_RGB_sha256=parent.array_sha(full),original_frame8_RGB96_sha256=parent.array_sha(rgb),original_frame8_valid96_sha256=parent.array_sha(valid))
   artifacts.append(draw_case(rows[idx],selected[idx],q,L,rgb,valid,images[fi],OUT,parent))
  index=OUT/'index.html';by={r['global_index']:r for r in artifacts};content=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V169 fixed current diagnostic crops</title><style>body{font:20px system-ui;max-width:2080px;margin:auto;padding:24px}img{width:100%}</style><h1>4 고정조건 × WOOD/SQUARE strata</h1><p>각 TP/FN/FP/TN/U 최저 global index와 기존 manual-reference≤3 D→E 진단. Empty 대체·사진 후 재선택 없음. Native 상태·literal q·좌표품질은 별개이며 physical3px gold가 아닙니다. 전체1784와 변경 전수 ledger는 별도 보존합니다.</p>']
  for b in buckets:
   content.append('<h2>'+html.escape(b['key'])+'</h2>')
   if b['empty']:content.append('<p>EMPTY — no replacement</p>')
   else:content.append('<p><a href="#g'+str(b['selected_global_index'])+'">g'+str(b['selected_global_index'])+'</a></p>')
  for r in artifacts:content.append('<article id="g'+str(r['global_index'])+'"><h2>g'+str(r['global_index'])+'</h2><p>'+html.escape(', '.join(r['memberships']))+'</p><img src="'+Path(r['card']['path']).name+'"></article>')
  index.write_text('\n'.join(content)+'</html>')
  write(OUT/'CROP_RENDER_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),case_ids_before_RGB=bind(casefile),evaluation_seal=e['evaluation_seal'],panels=artifacts,verified_original_frames=verified,index=bind(index),selected_cases=len(artifacts),empty_buckets=sum(b['empty'] for b in buckets),all1784_csv=bind(allcsv),all1784_jsonl=bind(alljson),all_changed_pairs_csv=bind(change_csv),all_changed_pairs_jsonl=bind(change_json),changed_row_pair_count=len(changes),changed_unique_rows=len({r['global_index'] for r in changes}),all21_condition_pairs_covered=True,original_cache=p['original_cache'],cache_deserialized_keys=list(RGB_KEYS),P3_boxes_features_or_weights_deserialized=False,new_model_encoder_R0_fit_PnP_qL_changes=0,actual_visual_QA_pending=True,all1784_images_actually_viewed_claimed=False,case_selection_changed_after_RGB=False,selected_model=None,training_admitted=False))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'CROP_RENDER_COMPLETE.json'))),flush=True)
 except BaseException:
  import traceback
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,case_ID_seal_preserved=bind(casefile),automatic_retry=False,traceback=traceback.format_exc()));raise


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare and not a.entry:prepare()
 elif a.entry and not a.prepare:run(a.entry)
 else:raise SystemExit('Metadata-only --prepare; actual visualization requires separately approved frozen-evaluation entry')
