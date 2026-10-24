"""Fixed metadata-driven source RGB inspection only; no annotations/proxies."""
from pathlib import Path
import json,hashlib,math
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;V=R.parent
scope=json.loads((V/'SOURCE_NATIVE8_VISUAL_SCOPE_BEFORE_RGB.json').read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def text(im,xy,t,fill='white',size=14):
 f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size);ImageDraw.Draw(im).text(xy,t,font=f,fill=fill)
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
policy=json.loads((R/'RAW_REVIEW_POLICY_BEFORE_RGB.json').read_text());assert policy['frozen_before_RGB_proxy_reads']
targets=json.loads(Path(scope['targets']['path']).read_text())['targets'];outputs=[];verified=[]
for row in targets:
 assert bind(row['image'])['sha256']==row['image_sha256']
 raw=Image.open(row['image']).convert('RGB');W,H=raw.size;assert [H,W]==row['raw_hw']
 name=row['id'];r=row['fixed_existing_renderer_r8'];q=row['preserved_input_q8'];marked=raw.copy();d=ImageDraw.Draw(marked)
 for k,(rr,qq)in enumerate(zip(r,q)):
  x,y=rr;u,v=qq
  if 0<=x<W and 0<=y<H:
   d.line((x-5,y,x+5,y),fill='red',width=1);d.line((x,y-5,x,y+5),fill='red',width=1)
   d.text((x+5,y-12),str(k),fill='yellow')
  if 0<=u<W and 0<=v<H:d.ellipse((u-2,v-2,u+2,v+2),outline='cyan',width=1)
 full=Image.new('RGB',(W*2,H+42),(25,25,25));full.paste(raw,(0,42));full.paste(marked,(W,42));text(full,(8,6),name+' RAW | unchanged r=red cross / q=cyan ring',size=18)
 fp=R/(name+'_FULL_RAW_MARKED.png');full.save(fp);outputs.append(bind(fp))
 sheet=Image.new('RGB',(2*576,4*330),(25,25,25))
 for k,(rr,qq)in enumerate(zip(r,q)):
  x,y=rr;cx0=math.floor(x)-48;cy0=math.floor(y)-48;crop=raw.crop((cx0,cy0,cx0+96,cy0+96));mark=crop.copy();dd=ImageDraw.Draw(mark)
  px=x-cx0;py=y-cy0;qx=qq[0]-cx0;qy=qq[1]-cy0
  dd.line((px-5,py,px+5,py),fill='red',width=1);dd.line((px,py-5,px,py+5),fill='red',width=1)
  if 0<=qx<96 and 0<=qy<96:dd.ellipse((qx-2,qy-2,qx+2,qy+2),outline='cyan',width=1)
  # Native image border location in the padded diagnostic crop.
  dd.rectangle((-cx0,-cy0,W-1-cx0,H-1-cy0),outline='lime',width=1)
  left=(k%2)*576;top=(k//2)*330
  text(sheet,(left+5,top+2),f'{name} k{k} r=({x:.2f},{y:.2f})',size=14)
  text(sheet,(left+5,top+20),f'raw96 then marked; origin=({cx0},{cy0}) OOB={not(0<=x<W and 0<=y<H)}',size=13)
  sheet.paste(crop.resize((288,288),Image.Resampling.NEAREST),(left,top+42));sheet.paste(mark.resize((288,288),Image.Resampling.NEAREST),(left+288,top+42))
 sp=R/(name+'_ALL8_RAW96_MARKED.png');sheet.save(sp);outputs.append(bind(sp))
 verified.append({'id':name,'RGB':bind(row['image']),'decoded_HW':[H,W],'fixed_r8':r,'unchanged_q8':q,'source_annotation_opened':False})
write(R/'RAW_INSPECTION_RENDER_RECEIPT.json',{'completed':True,'policy':bind(R/'RAW_REVIEW_POLICY_BEFORE_RGB.json'),'code':bind(__file__),'verified_source_RGB':verified,'inspection_images':outputs,'source_annotation_fraction_mask_depth_proxy_model_current_reads':0,'new_forward_fit_PnP':0,'inspection_crops_are_not_inputs':True})
print(json.dumps({'rendered':len(outputs),'receipt':bind(R/'RAW_INSPECTION_RENDER_RECEIPT.json')}))
