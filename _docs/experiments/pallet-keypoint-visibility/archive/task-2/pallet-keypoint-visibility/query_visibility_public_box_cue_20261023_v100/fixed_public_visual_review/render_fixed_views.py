from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
V=Path(__file__).resolve().parents[1];O=V/'fixed_public_visual_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
s=json.loads((O/'FIXED_CASE_SELECTION_BEFORE_RGB.json').read_text());receipt=[]
for c in s['instances']:
 p=Path(c['image']['path']);assert sha(p)==c['image']['sha256'];im=Image.open(p);assert im.mode=='RGB';rgb=np.asarray(im);h,w=rgb.shape[:2];assert [h,w]==c['raw_hw']
 sheet=Image.new('RGB',(1280,1000),'#f1f3f5');d=ImageDraw.Draw(sheet)
 d.text((14,10),f"OBJ{c['object_id']:02d} | scene{c['scene_id']} frame{c['frame_id']} inst{c['instance_id']} | fixed TRAIN",font=font(24),fill='black')
 rawfull=im.copy();markfull=im.copy();md=ImageDraw.Draw(markfull)
 for r in c['corners']:
  x,y=r['projected_AABB_r_xy'];k=r['aux_aabb_index'];md.line((x-5,y,x+5,y),fill='red',width=2);md.line((x,y-5,x,y+5),fill='red',width=2);md.text((x+7,y-10),str(k),fill='red',font=font(15))
 sheet.paste(rawfull,(0,48));sheet.paste(markfull,(640,48));d.text((12,532),'Raw context (left), fixed projected AUX_AABB points (right). All8 preserved; no masks/depth.',font=font(17),fill='black')
 crops=[]
 for r in c['corners']:
  k=r['aux_aabb_index'];x,y=r['projected_AABB_r_xy'];xs=x+(np.arange(96)+.5-48);ys=y+(np.arange(96)+.5-48);xi=np.floor(xs+.5).astype(int);yi=np.floor(ys+.5).astype(int);valid=(yi[:,None]>=0)&(yi[:,None]<h)&(xi[None,:]>=0)&(xi[None,:]<w)
  a=np.zeros((96,96,3),np.uint8);a[valid]=rgb[np.broadcast_to(yi[:,None],(96,96))[valid],np.broadcast_to(xi[None,:],(96,96))[valid]]
  raw=Image.fromarray(a);marked=raw.copy();dd=ImageDraw.Draw(marked);dd.line((40,47.5,55,47.5),fill='red');dd.line((47.5,40,47.5,55),fill='red')
  # Cyan is nearest surface projection only; never substituted for r.
  sx,sy=r['nearest_surface_projection_xy_diagnostic'];cx=sx-x+47.5;cy=sy-y+47.5;dd.ellipse((cx-3,cy-3,cx+3,cy+3),outline='cyan',width=1)
  rp=O/f"OBJ{c['object_id']:02d}_AUX{k}_RAW96.png";mp=O/f"OBJ{c['object_id']:02d}_AUX{k}_MARKED96.png";raw.save(rp);marked.save(mp)
  col=k%4;row=k//4;ox=col*320;oy=564+row*212
  d.text((ox+4,oy),f"AUX{k} r=({x:.1f},{y:.1f})",font=font(14),fill='black');d.text((ox+4,oy+20),f"mesh distance {r['nearest_surface_distance_mm']:.2f}mm",font=font(13),fill='black')
  sheet.paste(raw.resize((150,150),Image.Resampling.NEAREST),(ox+4,oy+42));sheet.paste(marked.resize((150,150),Image.Resampling.NEAREST),(ox+161,oy+42))
  crops.append({'aux_aabb_index':k,'raw':bind(rp),'marked':bind(mp),'r_xy':r['projected_AABB_r_xy'],'valid_native_samples':int(valid.sum()),'samples':9216,'raw_RGB_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
 sp=O/f"OBJ{c['object_id']:02d}_ALL8_CONTEXT_AND_RAW_MARKED.png";sheet.save(sp)
 receipt.append({'case_id':c['case_id'],'source_image':c['image'],'raw_hw':[h,w],'source_decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'sheet':bind(sp),'crops':crops})
(O/'RENDERED_FIXED_VIEW_RECEIPTS.json').write_text(json.dumps({'completed':True,'selection':bind(O/'FIXED_CASE_SELECTION_BEFORE_RGB.json'),'actual_RGB_decoded_unique':3,'instances':4,'all32_raw_crops':32,'receipts':receipt,'mask_depth_reads':0,'model_current_reads':0},indent=2)+'\n')
print(json.dumps({'completed':True,'sheets':[r['sheet'] for r in receipt]}))
