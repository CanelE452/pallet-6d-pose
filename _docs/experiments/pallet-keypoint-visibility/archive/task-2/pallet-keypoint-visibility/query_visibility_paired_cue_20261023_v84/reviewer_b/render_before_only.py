from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');B=P/'query_visibility_paired_cue_20261023_v84/reviewer_b';meta=P/'query_visibility_paired_cue_20261023_v84/FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert (B/'REVIEW_POLICY_BEFORE_RGB.json').exists();z=json.loads(meta.read_text())['raw_files']['before']['image'];p=Path(z['path']);assert b(p)['sha256']==z['sha256'];im=Image.open(p).convert('RGB');assert im.size==(640,480)
out=B/'inspection';out.mkdir(exist_ok=True)
# Crop centers are inspection-window choices only; no cue coordinates finalized here.
windows=[('UPPER_LEFT',[29,137,125,233]),('UPPER_RIGHT',[524,134,620,230]),('LOWER_LEFT',[32,372,128,468]),('LOWER_RIGHT',[522,372,618,468])]
files=[]
for name,xy in windows:
 crop=im.crop(xy);raw=out/(name+'_UNMARKED_NATIVE96.png');crop.save(raw);zoom=out/(name+'_UNMARKED_4X_NEAREST.png');crop.resize((384,384),Image.Resampling.NEAREST).save(zoom)
 grid=crop.resize((384,384),Image.Resampling.NEAREST);d=ImageDraw.Draw(grid)
 for i in range(0,96,8):
  d.line([(i*4,0),(i*4,383)],fill=(0,180,180),width=1);d.text((i*4+1,1),str(xy[0]+i),fill=(255,255,0))
  d.line([(0,i*4),(383,i*4)],fill=(0,180,180),width=1);d.text((1,i*4+10),str(xy[1]+i),fill=(255,255,0))
 gp=out/(name+'_NATIVE8_GRID.png');grid.save(gp)
 files.append({'window_name':name,'native_crop_xyxy':xy,'raw':b(raw),'unmarked_zoom':b(zoom),'grid_for_localization':b(gp)})
r={'completed':True,'before_only':True,'policy':b(B/'REVIEW_POLICY_BEFORE_RGB.json'),'fixed_metadata':b(meta),'raw_before':b(p),'raw_hw':[480,640],'source_RGB_bodies_read':[str(p)],'after_RGB_masks_depth_pose_bodies_read':0,'no_coordinate_finalized_by_renderer':True,'inspection_crops':files,'code':b(Path(__file__))}
f=B/'INSPECTION_RENDER_MANIFEST.json';assert not f.exists();f.write_text(json.dumps(r,indent=2)+'\n');print(b(f))
