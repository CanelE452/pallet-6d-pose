#!/usr/bin/env python3
"""Three preselected SOURCE internalDEV context views. No recrop/label/model code."""
import hashlib,html,io,json,math,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).parent
R=H.parent
SCOPE=R/'V236_SOURCE_CONTEXT_DIAGNOSTIC/THREE_FRAME_SCOPE_BEFORE_SOURCE_RGB.json'
SCOPE_SHA='d9108bac55768b005e5b99a9944631034b1a41423f208d99da19eb289337e3a2'
RAW=R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/raw_rgb96.npy'
RAW_SHA='f9c4dac0a9a021ce248732c335014d7c88d39539ff41a20cc941e6acaff7020a'
access=[]
def lexical(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 for bad in ('v112','__pycache__','current223','publicheld','oldval','source_holdout','v177','v178','v182'):assert bad not in str(p).lower()
 return p
def checked(p,sha,white):
 p=lexical(p);assert str(p) in white
 for q in [p]+list(p.parents):assert not q.is_symlink()
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b);n+=len(b)
 assert h.hexdigest()==sha
 access.append({'path':str(p),'sha256':sha,'size_bytes':n});return p
checked(SCOPE,SCOPE_SHA,{str(SCOPE)})
scope=json.loads(SCOPE.read_text());assert scope['approved_source_internal_DEV_RGB_diagnostic'] is True
assert len(scope['cases'])==len(scope['exact_RGB_whitelist'])==3
white={b['path'] for b in scope['exact_RGB_whitelist']}|{str(RAW)}
for b in scope['exact_RGB_whitelist']:checked(b['path'],b['sha256'],white)
checked(RAW,RAW_SHA,white)
raw=np.load(RAW,mmap_mode='r',allow_pickle=False);assert raw.dtype==np.uint8 and raw.shape==(7617,3,96,96)
indices=[int(c['row']['sample_index']) for c in scope['cases']];assert indices==[7536,7220,7348] and len(set(indices))==3
out=H/'three_source_context_v1';out.mkdir(exist_ok=False)
font=lambda n:ImageFont.truetype('DejaVuSans.ttf',n)
F16,F20,F24,F30=font(16),font(20),font(24),font(30)
CYAN=(0,255,255);ORANGE=(255,175,20);MAGENTA=(255,40,225);WHITE=(245,245,245);BG=(16,24,39)
def cross(d,x,y,color,r=7,w=2):
 d.line((x-r,y,x+r,y),fill=color,width=w);d.line((x,y-r,x,y+r),fill=color,width=w)
def save(im,name):im.save(out/name)
def binding(p):
 b=p.read_bytes();return {'path':p.name,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
coords=[];cards=[]
for j,c in enumerate(scope['cases']):
 row=c['row'];inp=c['input_original'];ix=int(row['sample_index']);aux=int(row['aux_slot']);obj=int(row['object_id'])
 assert inp['split']=='TRAIN' and inp['internal_source_role']=='INTERNAL_SOURCE_DEV' and inp['native_mapping']=='UNKNOWN' and inp['source_emulation'] is True
 assert inp['image']==c['image'] and [obj,aux]==c['case_key']
 q=[float(row['q_x']),float(row['q_y'])];L=float(row['L']);ref=[float(row['r_x_GT_diagnostic_only']),float(row['r_y_GT_diagnostic_only'])]
 assert q==inp['initial_points'][aux] and L==inp['roi_side']
 src=Path(c['image']['path']);encoded=src.read_bytes();assert hashlib.sha256(encoded).hexdigest()==c['image']['sha256']
 (out/f'CASE{j:02d}_SOURCE_CLEAN.png').write_bytes(encoded)
 with Image.open(io.BytesIO(encoded)) as im:
  source=im.convert('RGB');assert source.size==(inp['raw_hw'][1],inp['raw_hw'][0])==(640,480)
 rgb=np.array(raw[ix],copy=True);assert rgb.shape==(3,96,96)
 crop=Image.fromarray(np.ascontiguousarray(rgb.transpose(1,2,0)));save(crop,f'CASE{j:02d}_RAW96_CLEAN.png')
 with Image.open(out/f'CASE{j:02d}_RAW96_CLEAN.png') as im:assert np.array_equal(np.asarray(im).transpose(2,0,1),rgb)
 full=source.copy();d=ImageDraw.Draw(full)
 box=inp['query_box_xyxy'];d.rectangle(box,outline=CYAN,width=2)
 for k,(x,y) in enumerate(inp['initial_points']):
  d.ellipse((x-3,y-3,x+3,y+3),outline=CYAN,width=2)
  d.text((x+5,y-15),str(k),font=F16,fill=CYAN,stroke_width=1,stroke_fill=(0,0,0))
 cropbox=[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2];d.rectangle(cropbox,outline=ORANGE,width=3)
 cross(d,*q,WHITE,9,2);d.ellipse((ref[0]-6,ref[1]-6,ref[0]+6,ref[1]+6),outline=MAGENTA,width=2)
 save(full,f'CASE{j:02d}_FULL_Q8_DISPLAY.png')
 uv=[(ref[k]-q[k])/L*96.+47.5 for k in (0,1)]
 assert abs(uv[0]-float(row['raw96_ref_x']))<1e-12 and abs(uv[1]-float(row['raw96_ref_y']))<1e-12
 overlay=crop.resize((384,384),Image.Resampling.NEAREST);d=ImageDraw.Draw(overlay)
 # Half-pixel pixel-centre mapping: (x+.5)*scale-.5.
 center=(47.5+.5)*4-.5;cross(d,center,center,CYAN,13,3)
 rx=(uv[0]+.5)*4-.5;ry=(uv[1]+.5)*4-.5;d.ellipse((rx-10,ry-10,rx+10,ry+10),outline=MAGENTA,width=3)
 save(overlay,f'CASE{j:02d}_RAW96_DISPLAY.png')
 canvas=Image.new('RGB',(1900,910),BG);dr=ImageDraw.Draw(canvas)
 dr.text((22,16),f'CASE {j+1} | object_id {obj} / AUX slot {aux} | SOURCE internalDEV | sample {ix}',font=F30,fill=WHITE)
 dr.text((22,60),'GT-side intended object ID only; no mask-free visual identity certification. Stored associated-r state E.',font=F20,fill=WHITE)
 canvas.paste(full.resize((960,720),Image.Resampling.NEAREST),(20,105))
 dr.text((1010,104),'UNCHANGED original RAW96',font=F24,fill=WHITE)
 canvas.paste(crop.resize((384,384),Image.Resampling.NEAREST),(1010,142))
 dr.text((1430,104),'DISPLAY only: q + reference',font=F24,fill=WHITE);canvas.paste(overlay,(1430,142))
 lines=[f"row: {row['row_id']}",f"scene {int(row['scene']):06d} / frame {int(row['frame_id']):06d}; input {inp['input_id']}",f'q=({q[0]:.5f}, {q[1]:.5f})  L={L:.5f} source pixels',f'r=({ref[0]:.5f}, {ref[1]:.5f}) DIAGNOSTIC ONLY',f'r in RAW96=({uv[0]:.5f}, {uv[1]:.5f}); q=(47.5,47.5)', 'CYAN: original emulated q8 and q-box; ORANGE: original L canvas.', 'WHITE on full frame: selected q; MAGENTA: SOURCE teacher r.', 'q8 is GT-jitter emulated input. Native mapping UNKNOWN.', 'No GT recrop/recenter, new labels, model inference or training.']
 for k,t in enumerate(lines):dr.text((1010,552+k*29),t,font=F20,fill=WHITE if k<5 else (194,207,229))
 oob=[k for k,(x,y) in enumerate(inp['initial_points']) if not(0<=x<640 and 0<=y<480)]
 dr.text((22,843),f'q8 coordinates outside frame: {oob or "none"}; retained exactly, not relocated. Image drawing may clip them.',font=F20,fill=ORANGE)
 save(canvas,f'CASE{j:02d}_CONTEXT_VS_CROP.png');cards.append(canvas)
 coords.append({'case_position':j,'case_key':c['case_key'],'row_id':row['row_id'],'sample_index':ix,'original_role':'TRAIN','internal_source_role':'INTERNAL_SOURCE_DEV','source_image':c['image'],'source_clean_copy':f'CASE{j:02d}_SOURCE_CLEAN.png','source_decoded_RGB_HWC_SHA256':hashlib.sha256(np.array(source).tobytes()).hexdigest(),'original_crop_raw96_CHW_SHA256':hashlib.sha256(rgb.tobytes()).hexdigest(),'original_q8_xy':inp['initial_points'],'q8_availability_field_in_scope':None,'q8_coordinate_outside_frame_indices':oob,'original_query_box_xyxy':box,'original_query_xy':q,'original_L':L,'L_canvas_xyxy':cropbox,'teacher_reference_xy_diagnostic_only':ref,'reference_RAW96_xy_diagnostic_only':uv,'raw96_query_marker':[47.5,47.5],'object_id_GT_side_only':obj,'logical_aux_slot':aux,'native_mapping':'UNKNOWN','query_origin':'PUBLIC_YCB_GT_JITTER_EMULATION','new_visual_label':None,'new_training_admission':False})
board=Image.new('RGB',(1900,3*910+150),BG);d=ImageDraw.Draw(board)
d.text((22,18),'V236 | Three fixed SOURCE internalDEV context/crop comparisons',font=F30,fill=WHITE)
d.text((22,62),'Illustrative, reused SOURCE. No native human gold or causal proof; original input/targets unchanged.',font=F24,fill=WHITE)
for j,card in enumerate(cards):board.paste(card,(0,150+j*910))
save(board,'THREE_SOURCE_CONTEXT_BOARD.png')
(out/'CASE_COORDINATES.json').write_text(json.dumps({'completed':True,'cases':coords,'scope_SHA256':SCOPE_SHA,'selected_numeric_RAW96_rows':indices,'raw_NPY_whole_SHA_opaque_checked':RAW_SHA,'raw_GT_mask_depth_weights_prefix_current_held_access':0,'new_model_forward_or_fit':0},ensure_ascii=False,indent=2)+'\n')
body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V236 SOURCE 문맥 진단</title><style>body{background:#101827;color:#eef3ff;font:18px system-ui;margin:24px}img{max-width:100%}a{color:#8eeaff}article{margin:30px 0;padding:16px;background:#1d293f}pre{white-space:pre-wrap}</style><h1>V236: 고정 SOURCE 세 사례의 fullframe과 원 crop</h1><p>기존 TRAIN의 internalDEV 자료이며 새 학습용 자료가 아닙니다. q8은 GT-jitter emulation 입력이고 native mapping은 UNKNOWN입니다. object_id는 GT-side 메타데이터로서 mask 없는 시각만으로 해당 물체를 확정하지 않습니다. magenta reference는 진단 표시뿐이며 crop을 다시 맞추지 않았습니다.</p><p><a href="CASE_COORDINATES.json">원 좌표와 바인딩</a> · <a href="THREE_SOURCE_CONTEXT_BOARD.png">전체 board</a></p>']
for j,c in enumerate(coords):
 body.append(f'<article><h2>고정 사례 {j+1}: object_id {c["object_id_GT_side_only"]}, aux {c["logical_aux_slot"]}</h2><img src="CASE{j:02d}_CONTEXT_VS_CROP.png" alt="원 문맥과 원crop의 비교"><p><a href="CASE{j:02d}_SOURCE_CLEAN.png">원 fullRGB byte copy</a> · <a href="CASE{j:02d}_RAW96_CLEAN.png">원 RAW96 clean</a></p><pre>'+html.escape(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></article>')
body.append('<p>target 정체가 crop 밖에 있을 수 있다는 가설을 살피는 표시입니다. 세 사례로 global context의 필요·충분성이나 실제 실패 원인을 인증하지 않으며 새 D/E/SELF·human 라벨을 부여하지 않습니다. 원자료·라벨·모델·학습을 바꾸지 않았습니다.</p></html>')
(out/'PORTABLE_CONTEXT_GALLERY_KO.html').write_text(''.join(body))
# Scope-bound source bytes are immutable; raw NPY rehash is opaque only.
for b in scope['exact_RGB_whitelist']:checked(b['path'],b['sha256'],white)
checked(RAW,RAW_SHA,white)
outputs=[binding(p) for p in sorted(out.iterdir()) if p.is_file()]
complete={'completed':True,'PASS':True,'code':{'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'scope':{'path':str(SCOPE),'sha256':SCOPE_SHA},'source_access':access,'source_RGB_unique':3,'source_RAW96_numeric_rows_only':indices,'all_original_qL_q8_native_UNKNOWN_preserved':True,'new_label_or_training_or_model_forward':0,'original_GT_mask_depth_weights_prefix_current_held_access':0,'outputs':outputs,'toolview_pending':True}
(out/'COMPLETE.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'completed':True,'PASS':True,'COMPLETE':str(out/'COMPLETE.json'),'board':str(out/'THREE_SOURCE_CONTEXT_BOARD.png')}))
