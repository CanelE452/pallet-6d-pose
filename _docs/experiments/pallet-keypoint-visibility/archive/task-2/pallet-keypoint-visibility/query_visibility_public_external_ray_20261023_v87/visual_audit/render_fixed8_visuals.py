"""Inspection-only fixed-eight native RGB/M4-outline panels, no model/M0."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime,timezone
import hashlib,json,math,io
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent
rd=lambda p:json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def txt(im,xy,s,size=13):ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size),fill='white')
sc=rd(R/'VISUAL_SCOPE_BEFORE_RGB.json');sel=rd(sc['bindings']['fixed_TRAIN_minSHA_selection']['path']);assert len(sel['records'])==8
inventory=rd(V/'corpus_build_v1/SOURCE_DEPTH_MASK_BYTE_BINDINGS.json');msha={x['path']:x['sha256']for x in inventory['files']}
compact=Image.new('RGB',(768,4*270),(25,25,25));outputs=[];cases=[]
for n,(reason,case)in enumerate(sel['records'].items()):
 row=case['source_metadata'];assert row['split']=='train';rec=case['record'];assert reason==rec['reason']
 blob=Path(row['image']).read_bytes();assert hashlib.sha256(blob).hexdigest()==row['image_sha256'];raw=Image.open(io.BytesIO(blob)).convert('RGB');w,h=raw.size;assert[w,h]==[row['width'],row['height']]
 mb=Path(row['mask_vis']).read_bytes();assert hashlib.sha256(mb).hexdigest()==msha[row['mask_vis']];mask=np.asarray(Image.open(io.BytesIO(mb)))>0;assert mask.shape==(h,w)
 interior=mask.copy();interior[1:,:]&=mask[:-1,:];interior[:-1,:]&=mask[1:,:];interior[:,1:]&=mask[:,:-1];interior[:,:-1]&=mask[:,1:]
 boundary=mask&~interior;a=np.array(raw);a[boundary]=[0,255,0];marked=Image.fromarray(a);d=ImageDraw.Draw(marked)
 x,y=case['reference_xy'];x0=math.floor(x)-48;y0=math.floor(y)-48
 if 0<=x<w and 0<=y<h:
  d.line((x-5,y,x+5,y),fill='red',width=1);d.line((x,y-5,x,y+5),fill='red',width=1)
  d.text((x+7,y-8),f'r k{case["slot"]}',fill='red')
 rawcrop=raw.crop((x0,y0,x0+96,y0+96));markcrop=marked.crop((x0,y0,x0+96,y0+96));dd=ImageDraw.Draw(markcrop)
 px=x-x0;py=y-y0;dd.line((px-5,py,px+5,py),fill='red');dd.line((px,py-5,px,py+5),fill='red');dd.rectangle((-x0,-y0,w-1-x0,h-1-y0),outline='cyan',width=1)
 folder=R/f'{n:02d}_{reason}';folder.mkdir(exist_ok=False)
 rawcrop.save(folder/'RAW_NATIVE96.png');markcrop.save(folder/'MARKED_NATIVE96.png')
 full=Image.new('RGB',(2*w,h+55),(25,25,25));full.paste(raw,(0,55));full.paste(marked,(w,55));txt(full,(5,3),f'{n} {reason} {case["id"]} k{case["slot"]}',16);txt(full,(5,27),'raw full | target M4 outline GREEN, unchanged r RED; not semantic corner GT',14)
 fp=folder/'FULL_RAW_AND_TARGET_OUTLINE.png';full.save(fp)
 pair=Image.new('RGB',(576,390),(25,25,25));pair.paste(rawcrop.resize((288,288),Image.Resampling.NEAREST),(0,102));pair.paste(markcrop.resize((288,288),Image.Resampling.NEAREST),(288,102))
 txt(pair,(5,2),reason,14);txt(pair,(5,23),f'{case["id"]} k{case["slot"]} r=({x:.3f},{y:.3f})',13)
 txt(pair,(5,43),f'owner={rec["owner_type"]} count={rec["owner_count"]} inst={rec["unique_owner_id"]} depthRaw={rec["depth_raw"]}',12)
 txt(pair,(5,62),f'refRange={rec["ref_range_mm"]} obsLo={rec["observed_range_lo_mm"]}',11)
 txt(pair,(5,79),f'obsHi={rec["observed_range_hi_mm"]} OOB={not(0<=x<w and 0<=y<h)} raw|marked',11)
 cp=folder/'RAW_MARKED_NATIVE96_NUMERICS.png';pair.save(cp)
 lx=n%2*384;ly=n//2*270;txt(compact,(lx+3,ly+2),f'{n} '+reason,10);txt(compact,(lx+3,ly+18),f'{case["id"]} k{case["slot"]}',10);txt(compact,(lx+3,ly+34),f'{rec["owner_type"]} depthRaw{rec["depth_raw"]} OOB{not(0<=x<w and 0<=y<h)}',10)
 compact.paste(rawcrop.resize((192,192),Image.Resampling.NEAREST),(lx,ly+65));compact.paste(markcrop.resize((192,192),Image.Resampling.NEAREST),(lx+192,ly+65))
 fs=[folder/'RAW_NATIVE96.png',folder/'MARKED_NATIVE96.png',fp,cp];outputs.extend(bind(f)for f in fs)
 cases.append({'index':n,'reason':reason,'id':case['id'],'slot':case['slot'],'original_reference_xy':case['reference_xy'],'crop_origin':[x0,y0],'crop_native_side':96,'source_RGB':{'path':row['image'],'sha256':row['image_sha256']},'decoded_hw':[h,w],'target_visible_M4':bind(Path(row['mask_vis'])),'saved_numeric_diagnostic':rec,'files':[bind(f)for f in fs]})
f=R/'ALL8_FIXED_REASON_RAW_MARKED_NATIVE96_CONTACT.png';compact.save(f);outputs.append(bind(f))
save(R/'ACTUAL_RENDER_RECEIPT.json',{'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'scope':bind(R/'VISUAL_SCOPE_BEFORE_RGB.json'),'code':bind(__file__),'cases':cases,'outputs':outputs,'source_RGB_decoded':8,'source_M4_decoded':8,'M0_depth_arrays_ML_model_current_reads':0,'new_forward_fit_PnP':0,'r_unchanged_all8':True,'inspection_only_not_training_inputs':True})
print(json.dumps({'rendered_cases':8,'contact':bind(f),'receipt':bind(R/'ACTUAL_RENDER_RECEIPT.json')}))
