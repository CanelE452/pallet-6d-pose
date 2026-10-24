#!/usr/bin/env python3
import pathlib,json,hashlib
from PIL import Image
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
plan=json.loads((ROOT/'split_plan.json').read_text())
deny=set(json.loads((ROOT.parent/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
rows=[];counts={}
for scene,p in plan.items():
 prefix=ROOT/'data'/'train_real'/scene
 gt=json.loads((prefix/'scene_gt.json').read_text());camera=json.loads((prefix/'scene_camera.json').read_text())
 for frame in p['frames']:
  image=prefix/'rgb'/f'{frame:06d}.png'
  if not image.exists():continue
  objects=gt[str(frame)];paths=[prefix/k/f'{frame:06d}_{inst:06d}.png' for inst in range(len(objects)) for k in ['mask','mask_visib']]
  if not all(x.exists() for x in paths):continue
  sha=hashlib.sha256(image.read_bytes()).hexdigest();assert sha not in deny
  for inst,obj in enumerate(objects):
   full=prefix/'mask'/f'{frame:06d}_{inst:06d}.png';vis=prefix/'mask_visib'/f'{frame:06d}_{inst:06d}.png'
   mf=np.asarray(Image.open(full))>0;mv=np.asarray(Image.open(vis))>0;assert mf.shape==mv.shape==(480,640);assert not np.any(mv&~mf)
   rows.append({'split':p['split'],'image':str(image),'mask_vis':str(vis),'mask_full':str(full),'scene_id':int(scene),'frame_id':frame,'object_id':obj['obj_id'],'instance_id':inst,'K':camera[str(frame)]['cam_K'],'cam_R_m2c':obj['cam_R_m2c'],'cam_t_m2c':obj['cam_t_m2c'],'depth':str(prefix/'depth'/f'{frame:06d}.png'),'depth_scale':camera[str(frame)]['depth_scale'],'height':480,'width':640,'vis_pixels':int(mv.sum()),'full_pixels':int(mf.sum()),'external_proxy_pixels':int((mf&~mv).sum()),'image_sha256':sha,'source':'BOP YCB-V official train_real','license':'MIT'})
  counts[p['split']]=counts.get(p['split'],0)+1
for split in ['train','val']:
 out=ROOT/f'{split}_instances_incremental.jsonl';tmp=out.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']==split));tmp.rename(out)
print(json.dumps({'complete_frames':counts,'instance_rows':len(rows)}))
