import csv,json,pathlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb'
rows=list(csv.DictReader((OUT/'SOURCE_FAILURE_CONTEXT_SWAP_QUERIES.csv').open()));byid={r['id']:r for r in rows};sam={r['sample_id']:r for r in csv.DictReader((OUT/'SOURCE_FAILURE_SAM_BOX_SWAP.csv').open())};pred={r['id']:r for r in [json.loads(x)for x in(ROOT/'sam/TARGET_PREDICTIONS.jsonl').read_text().splitlines()]if r['source']=='YCB'}
manifest=[json.loads(x)for x in(OUT/'val_instances.jsonl').read_text().splitlines()];boxes={f'YCB:{r["scene_id"]:06d}:{r["frame_id"]:06d}:{r["instance_id"]:06d}':r['bbox_obj']for r in manifest}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
items=[]
for kind in ['visible_A_outside_B_context_ignored','external_A_visible_B_context_ignored','SAM_external_false_positive']:
 cand=[]
 for r in rows:
  if r['sample_id']not in sam or r['boundary']!='False':continue
  s=sam[r['sample_id']];pa=float(r['MIXED_CONTEXT_V2_p']);pb=float(r['MIXED_CONTEXT_V2_swapped_p'])
  if kind=='visible_A_outside_B_context_ignored':ok=r['presence_target']=='1'and r['other_target_y']=='0'and s['SAM_own']=='1'and s['SAM_other_box']=='0'and pa>.55 and pb>.55
  elif kind=='external_A_visible_B_context_ignored':ok=r['external_proxy']=='True'and r['other_target_y']=='1'and s['SAM_own']=='0'and s['SAM_other_box']=='1'and pa>.55 and pb>.55
  else:ok=r['external_proxy']=='True'and s['SAM_own']=='1'
  if ok:cand.append(r)
 assert cand,kind
 cand.sort(key=lambda r:abs(float(r['MIXED_CONTEXT_V2_own_minus_swap'])))
 r=dict(cand[len(cand)//2]);r['diagnostic_kind']=kind;r.update(SAM_own=sam[r['sample_id']]['SAM_own'],SAM_other=sam[r['sample_id']]['SAM_other_box']);items.append(r)
canvas=Image.new('RGB',(1120,870),'white');d=ImageDraw.Draw(canvas)
d.text((12,10),'Ownership counterfactual: same RGB/query; change target A to B',font=font,fill='black');d.text((12,34),'Source-only. Cyan=A, magenta=B boxes. Mask overlay: green=visible, orange=external, gray=outside.',font=small,fill='black')
for j,r in enumerate(items):
 top=65+265*j;im=Image.open(r['image']).convert('RGB');other=byid[r['other_target_id']];qx,qy=float(r['query_x']),float(r['query_y']);L=float(r['roi_side']);box=(qx-L/2,qy-L/2,qx+L/2,qy+L/2)
 d.text((12,top),f'{r["diagnostic_kind"]} | scene {r["scene_id"]}, frame {r["frame_id"]}, A inst {r["instance_id"]}, B {r["other_target_id"].split(":")[-1]}',font=font,fill='black')
 d.text((12,top+22),f'y(A,B)=({r["presence_target"]},{r["other_target_y"]}) | V2 p(A,B)=({float(r["MIXED_CONTEXT_V2_p"]):.3f},{float(r["MIXED_CONTEXT_V2_swapped_p"]):.3f}) | SAM(A,B)=({r["SAM_own"]},{r["SAM_other"]}) | boundary=False',font=small,fill='black')
 whole=im.resize((320,240));wd=ImageDraw.Draw(whole)
 for fid,color in [(r['id'],(0,255,255)),(r['other_target_id'],(255,0,255))]:
  bx,by,bw,bh=boxes[fid];wd.rectangle((bx*.5,by*.5,(bx+bw)*.5,(by+bh)*.5),outline=color,width=2)
 wd.ellipse((qx*.5-3,qy*.5-3,qx*.5+3,qy*.5+3),outline=(255,255,0),width=2)
 canvas.paste(whole.resize((256,192)),(12,top+48))
 panels=[];panels.append(('same RGB crop',im.transform((174,174),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)))
 for label,rr in [('A truth',r),('B truth',other)]:
  vis=np.asarray(Image.open(rr['mask_vis']))>0;full=np.asarray(Image.open(rr['mask_full']))>0;color=np.full((480,640,3),75,np.uint8);color[full&~vis]=(255,95,30);color[vis]=(35,200,100);ov=(np.asarray(im)*.45+color*.55).astype(np.uint8);panels.append((label,Image.fromarray(ov).transform((174,174),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)))
 with np.load(pred[r['id']]['mask_path'])as z:m=z['mask'].astype(bool)
 color=np.full((480,640,3),75,np.uint8);color[m]=(35,200,100);ov=(np.asarray(im)*.45+color*.55).astype(np.uint8);panels.append(('SAM A prediction',Image.fromarray(ov).transform((174,174),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)))
 for k,(label,panel)in enumerate(panels):
  pd=ImageDraw.Draw(panel);pd.line((79,87,95,87),fill=(0,255,255),width=2);pd.line((87,79,87,95),fill=(0,255,255),width=2);left=285+202*k;d.text((left,top+48),label,font=small,fill='black');canvas.paste(panel,(left,top+66))
canvas.save(OUT/'SOURCE_FAILURE_OWNERSHIP_CROPS.png');(OUT/'SOURCE_FAILURE_OWNERSHIP_CROPS.json').write_text(json.dumps(items,indent=2));print(OUT/'SOURCE_FAILURE_OWNERSHIP_CROPS.png')
