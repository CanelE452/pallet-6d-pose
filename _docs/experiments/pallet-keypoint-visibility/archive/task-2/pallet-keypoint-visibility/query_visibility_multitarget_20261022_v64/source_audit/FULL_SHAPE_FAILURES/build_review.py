#!/usr/bin/env python3
"""Deterministic source failure contact sheet, saved outputs only."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V63=ROOT.parent/'query_visibility_public_real_20261022_v63';DATA=(V63/'ycb/data/train_real').resolve()
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
 p=Path(p).resolve();assert p.is_relative_to(DATA);return p
def readmask(p):
 m=cv2.imread(str(source(p)),cv2.IMREAD_UNCHANGED);assert m.shape==(480,640)and set(np.unique(m))<={0,255};return m>0
def csvrows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def fit(image,hw):
 out=Image.new('RGB',hw,'#f3f3f3');scale=min(hw[0]/image.width,hw[1]/image.height);im=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.NEAREST);out.paste(im,((hw[0]-im.width)//2,(hw[1]-im.height)//2));return out
def marked(image,q,box=None):
 im=image.copy();d=ImageDraw.Draw(im)
 if box is not None:d.rectangle(box,outline='deepskyblue',width=2)
 x,y=q;d.rectangle((x-2,y-2,x+2,y+2),outline='yellow',width=1);d.ellipse((x-.7,y-.7,x+.7,y+.7),fill='white');return im

protocol=OUT/'PROTOCOL.json';assert protocol.exists()and not(OUT/'REVIEW_CASES.json').exists()
rows=csvrows(ROOT/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv');inputs_path=V63/'sam_source/INPUTS.jsonl';outputs_path=ROOT/'runs/MIXED_MULTITARGET_UNWEIGHTED/VAL_SOURCE_OUTPUTS.npz'
inp={r['id']:r for r in [json.loads(s)for s in inputs_path.read_text().splitlines()]if r['source']=='YCB'}
instances=[json.loads(s)for s in(V63/'ycb/val_instances.jsonl').read_text().splitlines()];meta={(r['scene_id'],r['frame_id'],r['instance_id']):r for r in instances};frameall={}
for r in instances:frameall.setdefault((r['scene_id'],r['frame_id']),[]).append(r)
with np.load(outputs_path)as z:
 take=z['source']=='YCB';p=z['joint_probability'][take];pv=z['visible_probability'][take];pf=z['full_probability'][take];gt=z['truth_classes'][take];hard=np.where(pv>=.5,0,np.where(pf>=.5,1,2))
assert len(rows)==len(p)==3200
strict=np.array([bool(int(r['strict_external_corroborated_5x5']))for r in rows]);assert(gt[strict,0]==1).all()
selection=[]
for predicted in [0,2]:
 for scene,wanted in [(8,2),(9,1)]:
  objects=set();frames=set()
  eligible=sorted([i for i,r in enumerate(rows)if strict[i]and hard[i,0]==predicted and int(r['scene_id'])==scene],key=lambda i:rows[i]['sample_id'])
  for i in eligible:
   r=rows[i];frame=(r['scene_id'],r['frame_id'])
   if r['object_id']in objects or frame in frames:continue
   objects.add(r['object_id']);frames.add(frame);selection.append(i)
   if len(objects)==wanted:break
  assert len(objects)==wanted
names=['VISIBLE','EXTERNAL_PROXY','OUTSIDE'];casew,caseh=1050,410;canvas=Image.new('RGB',(2*casew,3*caseh+100),'white');draw=ImageDraw.Draw(canvas)
draw.text((15,10),'Public YCB: strict corroborated external-support failures',font=font(28),fill='black')
draw.text((15,48),'All25 pixels full1/visible0/other-visible1. Same saved MIXED head; no new inference. Green=target-visible; magenta=hidden-support proxy.',font=font(16),fill='black')
records=[]
for c,i in enumerate(selection):
 r=rows[i];ident=r['id'];prompt=inp[ident];key=(int(r['scene_id']),int(r['frame_id']));target=int(r['instance_id']);m=meta[key+(target,)];rgb=Image.open(source(r['image'])).convert('RGB');assert sha(source(r['image']))==r['image_sha256'];v=readmask(m['mask_vis']);f=readmask(m['mask_full'])
 bx0,by0,bx1,by1=prompt['box'];cx,cy=(bx0+bx1)/2,(by0+by1)/2;bw,bh=(bx1-bx0)*1.25,(by1-by0)*1.25;global_box=(int(np.floor(cx-bw/2)),int(np.floor(cy-bh/2)),int(np.ceil(cx+bw/2)),int(np.ceil(cy+bh/2)))
 qx,qy=float(r['query_x']),float(r['query_y']);px,py=int(r['pixel_x']),int(r['pixel_y']);L=float(r['roi_side']);local=(int(np.floor(qx-L/2)),int(np.floor(qy-L/2)),int(np.ceil(qx+L/2)),int(np.ceil(qy+L/2)))
 overlay=np.asarray(rgb).astype(float);overlay[v]=.45*overlay[v]+.55*np.array([0,255,80]);overlay[f&~v]=.45*overlay[f&~v]+.55*np.array([255,0,180]);overlay=Image.fromarray(np.uint8(overlay))
 image_marked=marked(rgb,(px,py),prompt['box']);overlay_marked=marked(overlay,(px,py),prompt['box'])
 views=[image_marked,image_marked.crop(global_box),overlay_marked.crop(global_box),image_marked.crop(local)];titles=['Full image / rough box','Input-only global RGB x1.25','Diagnostic GT overlay','Original local RGB']
 x0=(c%2)*casew;y0=100+(c//2)*caseh;draw.text((x0+12,y0+4),f"{ident.split(':',1)[1]} q{r['query_index']}  obj{r['object_id']}  GT=E -> {names[int(hard[i,0])]}",font=font(17),fill='black')
 for col,(view,title)in enumerate(zip(views,titles)):
  x=x0+12+col*258;draw.text((x,y0+34),title,font=font(15),fill='black');canvas.paste(fit(view,(246,245)),(x,y0+57))
 full_bbox=m['bbox_obj'];vis_bbox=m['bbox_visib'];full_pixels=int(f.sum());visible_pixels=int(v.sum());qinside=bx0<=qx<=bx1 and by0<=qy<=by1;objvis=visible_pixels/max(full_pixels,1);others=[]
 for mm in frameall[key]:
  if mm['instance_id']==target:continue
  vv=readmask(mm['mask_vis'])
  if vv[py,px]:others.append(dict(instance_id=mm['instance_id'],object_id=mm['object_id']))
 draw.text((x0+12,y0+313),f"saved pV={pv[i,0]:.3f}, pFull={pf[i,0]:.3f}; maxjoint={p[i].max():.3f}; U={p[i].max()<.5}; target visible/full area={objvis:.1%}",font=font(16),fill='black')
 draw.text((x0+12,y0+342),f"native q=({qx:.2f},{qy:.2f}), ROI={L:.1f}px; other visible object(s) at q="+','.join(str(x['object_id'])for x in others),font=font(15),fill='black')
 draw.text((x0+12,y0+370),'Global RGB can constrain identity; hidden full-support membership may still require shape/pose.',font=font(14),fill='#444444')
 records.append(dict(cache_row_index=i,sample_id=r['sample_id'],id=ident,scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),instance_id=target,object_id=int(r['object_id']),truth_exact='EXTERNAL_PROXY',saved_logical_prediction=names[int(hard[i,0])],p_visible=float(pv[i,0]),p_full=float(pf[i,0]),max_joint=float(p[i].max()),confidence_U=bool(p[i].max()<.5),source_image=r['image'],image_sha256=r['image_sha256'],query=[qx,qy],nearest_pixel=[px,py],roi_side=L,prompt_box=prompt['box'],expanded_box_1p25=global_box,global_crop_pixels=[global_box[2]-global_box[0],global_box[3]-global_box[1]],query_inside_prompt_box=qinside,target_visible_pixels=visible_pixels,target_full_pixels=full_pixels,target_visible_fraction_in_frame_full=objvis,source_bbox_obj=full_bbox,source_bbox_visib=vis_bbox,source_visib_fract=m.get('visib_fract'),other_visible_objects_at_query=others,strict_external_5x5=True,strict_external_corroborated_5x5=True,source_GT_physical_confirmation=False,manual_visual_assessment='PENDING'))
canvas.save(OUT/'CONTACT_SHEET.png')
report=dict(schema='PUBLIC_SOURCE_GLOBAL_SHAPE_FAILURE_REVIEW_CASES_V1',protocol_sha256=sha(protocol),code_sha256=sha(__file__),source_outputs_sha256=sha(outputs_path),source_SAM_input_sha256=sha(inputs_path),source_only=True,current_private_data_read=False,GPU_used=False,new_model_inference=False,training=False,PnP=False,strict_corroborated_E_population=176,population_saved_logical_counts={names[c]:int((hard[strict,0]==c).sum())for c in range(3)},qualitative_selection=selection,selection_not_representative=True,cases=records)
(OUT/'REVIEW_CASES.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'selection':[(r['sample_id'],r['object_id'],r['saved_logical_prediction'])for r in records],'contact_sheet':str(OUT/'CONTACT_SHEET.png')},indent=2))
