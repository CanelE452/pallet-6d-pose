from pathlib import Path
import hashlib
import json
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
H=Path(__file__).resolve().parent;V=H.parent
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def cross(draw,xy,color='yellow',s=6):
 x,y=xy;draw.line((x-s,y,x+s,y),fill=color,width=1);draw.line((x,y-s,x,y+s),fill=color,width=1)
def main():
 policy=json.loads((H/'VIEW_POLICY_BEFORE_RGB.json').read_text());data=json.loads((V/'SURFACE_CANDIDATE_RESULT.json').read_text());receipts=[]
 for obj in data['objects']:
  pilot=obj['fixed_pilot'];oid=obj['object_id'];source=Path(pilot['image']['path']);assert bind(source)==pilot['image']
  raw=Image.open(source).convert('RGB');rgb=np.asarray(raw);height,width=rgb.shape[:2];pts=np.array([p['projected_xy']for p in pilot['projected_surface_candidates']]);L=int(math.ceil(float(np.ptp(pts,axis=0).max())/2))
  folder=H/f'OBJ{oid:02d}';folder.mkdir();shutil.copyfile(source,folder/'FULL_ORIGINAL.png')
  marked=raw.copy();draw=ImageDraw.Draw(marked)
  for k,xy in enumerate(pts):cross(draw,xy);draw.text((xy[0]+6,xy[1]+3),f'S{k}',font=font(14),fill='yellow',stroke_width=2,stroke_fill='black')
  marked.save(folder/'FULL_PROJECTIONS_MARKED.png')
  sheet=Image.new('RGB',(1280,1280),'#101821');d=ImageDraw.Draw(sheet)
  d.text((18,10),f'YCB object {oid:02d}: ALL8 model-surface AUX points | fixed TRAIN scene {pilot["scene_id"]}, frame {pilot["frame_id"]}',font=font(23),fill='white')
  d.text((18,42),'GT-reference-centred ORACLE feasibility | NOT initializer q / NOT training input / native mapping UNKNOWN',font=font(18),fill='#ffdf70')
  sheet.paste(raw,(0,96));sheet.paste(marked,(640,96));d.text((10,73),'ORIGINAL RGB',font=font(18),fill='white');d.text((650,73),'SEPARATE PROJECTION OVERLAY',font=font(18),fill='white')
  cases=[]
  offsets=(np.arange(96)+.5-48)*L/96
  for k,r in enumerate(pts):
   xx,yy=np.meshgrid(r[0]+offsets,r[1]+offsets);xi=np.floor(xx+.5).astype(int);yi=np.floor(yy+.5).astype(int);valid=(xi>=0)&(xi<width)&(yi>=0)&(yi<height);crop=np.zeros((96,96,3),dtype=np.uint8);crop[valid]=rgb[yi[valid],xi[valid]]
   im=Image.fromarray(crop);rawpath=folder/f'S{k}_RAW96.png';im.save(rawpath);cm=im.copy();cross(ImageDraw.Draw(cm),(47.5,47.5),s=5);cm.save(folder/f'S{k}_MARKED96.png')
   x=(k%4)*320;y=592+(k//4)*330;d.text((x+8,y),f'S{k} raw | L={L}px | OOB {int((~valid).sum())}/9216',font=font(16),fill='white');sheet.paste(im.resize((288,288),Image.Resampling.NEAREST),(x+16,y+30))
   cases.append(dict(aux_surface_id=pilot['projected_surface_candidates'][k]['aux_surface_id'],original_projected_r_xy=r.tolist(),source_L_native_px=L,source_point_or_native_mapping='UNKNOWN',raw96=bind(rawpath),marked96=bind(folder/f'S{k}_MARKED96.png'),out_of_image_samples=int((~valid).sum()),total_samples=9216,raw_pixels_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
  path=H/f'OBJ{oid:02d}_ALL8_ORACLE_CONTEXT_RAW_CROPS.png';sheet.save(path)
  receipts.append(dict(object_id=oid,source=pilot['image'],full_original=bind(folder/'FULL_ORIGINAL.png'),full_marked=bind(folder/'FULL_PROJECTIONS_MARKED.png'),contact=bind(path),source_L_native_px=L,all8=cases))
 result=dict(completed=True,policy=bind(H/'VIEW_POLICY_BEFORE_RGB.json'),code=bind(Path(__file__)),all4_instances=receipts,total_raw_crops=32,unique_source_RGBs=len({r['source']['sha256']for r in receipts}),no_point_shift_or_reselection=True,reference_centred_oracle_only=True,mask_depth_pixels_read=0,model_fit_private_PnP=0,labels_or_training_admission=False)
 with(H/'RENDER_RECEIPTS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(completed=True,receipt=bind(H/'RENDER_RECEIPTS.json'),contacts=[r['contact']['path']for r in receipts])))
if __name__=='__main__':main()
