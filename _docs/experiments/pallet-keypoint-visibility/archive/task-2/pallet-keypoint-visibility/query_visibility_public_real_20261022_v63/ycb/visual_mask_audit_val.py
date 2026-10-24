import json,pathlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
ROOT=pathlib.Path(__file__).resolve().parent
rows=[json.loads(x) for x in (ROOT/'SOURCE_VAL_LOCKED_200.jsonl').read_text().splitlines()]
r=max([r for r in rows if r['object_id']==16],key=lambda x:x['external_proxy_pixels'])
im=Image.open(r['image']).convert('RGB');a=np.asarray(im).copy();full=np.asarray(Image.open(r['mask_full']))>0;vis=np.asarray(Image.open(r['mask_vis']))>0;ext=full&~vis
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
font2=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
canvas=Image.new('RGB',(1560,540),(250,250,250));d=ImageDraw.Draw(canvas)
canvas.paste(im,(10,50))
over=a.copy();over[vis]=(0.55*over[vis]+0.45*np.array([0,210,100])).astype(np.uint8);over[ext]=(0.45*over[ext]+0.55*np.array([255,75,20])).astype(np.uint8)
canvas.paste(Image.fromarray(over),(660,50))
d.text((10,12),f'YCB-V train_real scene={r["scene_id"]}, frame={r["frame_id"]}, target obj={r["object_id"]}, inst={r["instance_id"]}',fill=(0,0,0),font=font)
d.text((660,12),'Green: visible target | Orange: full minus visible',fill=(0,0,0),font=font2)
items=[]
for label,m in [('VISIBLE',vis),('EXTERNAL PROXY',ext),('OUTSIDE SUPPORT',~full & (distance_transform_edt(~full)<20))]:
 safe=m.copy();safe[:32,:]=False;safe[-32:,:]=False;safe[:,:32]=False;safe[:,-32:]=False
 if safe.any():m=safe
 dist=distance_transform_edt(np.pad(m,1))[1:-1,1:-1];y,x=np.unravel_index(dist.argmax(),dist.shape);items.append({'label':label,'query_xy':[int(x),int(y)],'full':bool(full[y,x]),'visible':bool(vis[y,x])})
 crop=im.crop((x-32,y-32,x+32,y+32)).resize((150,150),Image.Resampling.NEAREST)
 cd=ImageDraw.Draw(crop);cd.line([(66,75),(84,75)],fill=(0,255,255),width=2);cd.line([(75,66),(75,84)],fill=(0,255,255),width=2)
 j=len(items)-1;canvas.paste(crop,(1320,50+j*160));d.text((1320,32+j*160),label,fill=(0,0,0),font=font2)
 d.ellipse((10+x-4,50+y-4,10+x+4,50+y+4),outline=(0,255,255),width=2);d.text((10+x+6,50+y),str(j+1),fill=(0,255,255),font=font)
canvas.save(ROOT/'YCB_WOOD_BLOCK_VAL_LABEL_AUDIT.png')
(ROOT/'YCB_WOOD_BLOCK_VAL_LABEL_AUDIT.json').write_text(json.dumps({'row':r,'queries':items},indent=2))
print(ROOT/'YCB_WOOD_BLOCK_VAL_LABEL_AUDIT.png')
