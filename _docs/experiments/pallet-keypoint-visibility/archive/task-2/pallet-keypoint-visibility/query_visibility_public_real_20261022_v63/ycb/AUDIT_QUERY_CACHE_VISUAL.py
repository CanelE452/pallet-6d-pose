import csv,json,pathlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb';CACHE=ROOT/'source_cache/v2'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
def yes(x):return x=='True'
items=[]
for split,kind in [('VALIDATION','visible_boundary'),('VALIDATION','external'),('TRAIN','outside'),('TRAIN','visible_interior')]:
 rows=list(csv.DictReader((CACHE/f'YCB_{split}_QUERIES.csv').open()))
 if kind=='visible_boundary': candidates=[r for r in rows if r['sampling']=='visible_boundary'and r['presence_target']=='1'and yes(r['boundary'])]
 elif kind=='external':candidates=[r for r in rows if yes(r['external_proxy'])and r['sampling']=='external_if_available']
 elif kind=='outside':candidates=[r for r in rows if yes(r['outside_full'])and r['sampling']=='outside_full_boundary']
 else:candidates=[r for r in rows if r['sampling']=='visible_interior'and r['presence_target']=='1']
 candidates=[r for r in candidates if 40<float(r['query_x'])<600 and 40<float(r['query_y'])<440]
 # Fixed middle candidate avoids choosing based on model scores.
 r=candidates[len(candidates)//2];r['diagnostic_kind']=kind;r['cache_split']=split;items.append(r)
canvas=Image.new('RGB',(1020,1010),'white');d=ImageDraw.Draw(canvas)
d.text((12,8),'Public source query audit: actual RGB, marked query, target masks, 9x9 dense labels',font=font,fill='black')
d.text((12,30),'Green=visible target; orange=full minus visible; gray=outside support. Cyan cross=query. No model scores used.',font=small,fill='black')
for j,r in enumerate(items):
 top=60+j*235;im=Image.open(r['image']).convert('RGB');full=np.asarray(Image.open(r['mask_full']))>0;vis=np.asarray(Image.open(r['mask_vis']))>0
 qx,qy=float(r['query_x']),float(r['query_y']);L=float(r['roi_side']);px,py=np.floor(np.array([qx,qy])+.5).astype(int)
 assert str(int(vis[py,px]))==r['presence_target'];assert bool(full[py,px]and not vis[py,px])==yes(r['external_proxy'])
 d.text((12,top),f'{r["cache_split"]} {r["diagnostic_kind"]} | scene {r["scene_id"]}, frame {r["frame_id"]}, obj {r["object_id"]}, inst {r["instance_id"]}, slot {r["query_index"]}',font=font,fill='black')
 d.text((12,top+20),f'q=({qx:.3f}, {qy:.3f}), pixel=({px},{py}), ROI={L:.2f}px | y={r["presence_target"]}, full={r["full_member"]}, external={r["external_proxy"]}, self=UNKNOWN',font=small,fill='black')
 whole=im.resize((256,192));wd=ImageDraw.Draw(whole);sx,sy=qx*.4,qy*.4;wd.rectangle(((qx-L/2)*.4,(qy-L/2)*.4,(qx+L/2)*.4,(qy+L/2)*.4),outline=(0,255,255),width=1);wd.line((sx-5,sy,sx+5,sy),fill=(0,255,255),width=1);wd.line((sx,sy-5,sx,sy+5),fill=(0,255,255),width=1);canvas.paste(whole,(12,top+40))
 box=(qx-L/2,qy-L/2,qx+L/2,qy+L/2);crop=im.transform((192,192),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)
 color=np.zeros((480,640,3),np.uint8);color[:]=(75,75,75);color[full&~vis]=(255,95,30);color[vis]=(35,200,100)
 rgb=np.asarray(im).copy();overlay=(rgb*.55+color*.45).astype(np.uint8);ov=Image.fromarray(overlay).transform((192,192),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)
 for p,left in [(crop,282),(ov,488)]:
  pd=ImageDraw.Draw(p);pd.line((88,96,104,96),fill=(0,255,255),width=2);pd.line((96,88,96,104),fill=(0,255,255),width=2);canvas.paste(p,(left,top+40))
 axis=((np.arange(9)+.5)/9-.5)*L;xx,yy=np.meshgrid(axis,axis);grid=np.stack([qx+xx,qy+yy],-1);pix=np.floor(grid+.5).astype(int);valid=(grid[...,0]>=0)&(grid[...,0]<640)&(grid[...,1]>=0)&(grid[...,1]<480)&(pix[...,0]>=0)&(pix[...,0]<640)&(pix[...,1]>=0)&(pix[...,1]<480)
 dense=np.zeros((9,9,3),np.uint8);dense[:]=(75,75,75);v=vis[np.clip(pix[...,1],0,479),np.clip(pix[...,0],0,639)];f=full[np.clip(pix[...,1],0,479),np.clip(pix[...,0],0,639)];dense[f&~v]=(255,95,30);dense[v]=(35,200,100);dense[~valid]=(0,0,0)
 panel=Image.fromarray(dense).resize((180,180),Image.Resampling.NEAREST);pd=ImageDraw.Draw(panel);pd.rectangle((80,80,99,99),outline=(0,255,255),width=3)
 canvas.paste(panel,(700,top+46));d.text((890,top+80),'9x9 target',font=small,fill='black');d.text((890,top+98),'mask labels',font=small,fill='black');d.text((890,top+118),'center [4,4]',font=small,fill='black')
canvas.save(OUT/'AUDIT_QUERY_CACHE_ACTUAL_CROPS.png');(OUT/'AUDIT_QUERY_CACHE_ACTUAL_CROPS.json').write_text(json.dumps(items,indent=2))
print(OUT/'AUDIT_QUERY_CACHE_ACTUAL_CROPS.png')
