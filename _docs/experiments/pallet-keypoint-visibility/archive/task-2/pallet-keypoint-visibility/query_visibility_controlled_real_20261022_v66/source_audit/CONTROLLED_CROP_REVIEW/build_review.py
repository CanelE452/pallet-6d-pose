"""Fixed source-only, score-free controlled-image/certificate contact sheets."""
from pathlib import Path
import hashlib
import json
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];DATA=ROOT/'controlled'
SEAL_SHA='8ba83649a4b99650afc384840f40108e39763c49369d93c4343bb3538f02a2f4'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PALETTE={-1:(72,105,205),0:(40,170,80),1:(225,45,150),2:(205,205,205)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def checked(entry):
 if sha(entry['path'])!=entry['sha256']:raise AssertionError('Bound review input changed: '+entry['path'])
 return Path(entry['path'])


def font(size):return ImageFont.truetype(FONT,size)


def mark_local(image,pixel,query,origin):
 im=image.resize((192,192),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
 # Integer pixels occupy2x2display blocks; +.5marks their center.
 x=(pixel[0]-origin[0]+.5)*2;y=(pixel[1]-origin[1]+.5)*2
 d.rectangle((x-5,y-5,x+5,y+5),outline='yellow',width=1)
 d.line((x-8,y,x+8,y),fill='yellow',width=1);d.line((x,y-8,x,y+8),fill='yellow',width=1)
 qx=(query[0]-origin[0]+.5)*2;qy=(query[1]-origin[1]+.5)*2
 d.ellipse((qx-1,qy-1,qx+1,qy+1),fill='cyan')
 return im


def full_marked(image,box,cropbox,pixel,queries):
 im=image.copy();d=ImageDraw.Draw(im)
 d.rectangle(box,outline='deepskyblue',width=2);d.rectangle(cropbox,outline='yellow',width=2)
 for q in queries:
  x,y=q;d.ellipse((x-1,y-1,x+1,y+1),fill='white')
 px,py=pixel;d.line((px-5,py,px+5,py),fill='yellow',width=2);d.line((px,py-5,px,py+5),fill='yellow',width=2)
 return im.resize((288,216),Image.Resampling.LANCZOS)


def main():
 if(OUT/'SELECTION.json').exists():raise FileExistsError('Preserve supplemental selection/artifacts')
 if sha(DATA/'SEAL.json')!=SEAL_SHA:raise AssertionError('Generation seal changed')
 seal=json.loads((DATA/'SEAL.json').read_text())
 manifest_path=DATA/'COMPOSITES.jsonl';manifest_binding=next(b for b in seal['outputs']if b['path']==str(manifest_path))
 checked(manifest_binding)
 rows=[json.loads(s)for s in manifest_path.read_text().splitlines()];first={};manifest_rows={}
 for index,r in enumerate(rows):
  scene=r['target_original']['scene_id']
  if scene not in first:first[scene]=r;manifest_rows[scene]=index
 if set(first)!=set(range(10)):raise AssertionError('One accepted case in each original scene0..9 required')
 records=[];raw_outputs=[]
 for scene,r in sorted(first.items()):
  original_path=checked(r['target_original']['image']);composite_path=checked(r['image']);checked(r['labels']);checked(r['query_labels'])
  original=Image.open(original_path).convert('RGB');composite=Image.open(composite_path).convert('RGB')
  k=r['acceptance']['anchor_query'];q=r['query'][k];px,py=r['acceptance']['anchor_pixel_xy'];crop=(px-48,py-48,px+48,py+48)
  label=json.loads(Path(r['query_labels']['path']).read_text())['controlled_composite'][k]
  if not label['radius2_strict_new_controlled_E_core']:raise AssertionError('Fixed selected anchor is not certified25pxE')
  with np.load(r['labels']['path'],allow_pickle=False)as z:
   states=z['exact_state'];origvis=z['original_visible'];paste=z['paste_mask'];fp=z['full_positive'];fn=z['full_negative'];unknown=z['full_unknown']
  if not(origvis[py-2:py+3,px-2:px+3]&paste[py-2:py+3,px-2:px+3]).all():raise AssertionError('Strict certificate rawmask mismatch')
  cert=np.zeros((*states.shape,3),np.uint8)
  for state,color in PALETTE.items():cert[states==state]=color
  raw_original=original.crop(crop);raw_composite=composite.crop(crop);raw_cert=Image.fromarray(cert).crop(crop)
  paths=[]
  for name,im in [('ORIGINAL_NATIVE96',raw_original),('COMPOSITE_NATIVE96',raw_composite),('CERTIFICATE_NATIVE96',raw_cert)]:
   path=OUT/f'SCENE_{scene:02d}_{name}.png';im.save(path);paths.append(binding(path));raw_outputs.append(binding(path))
  record=dict(scene_id=scene,split=r['split'],manifest_row=manifest_rows[scene],composite_id=r['id'],source_target_id=r['target_original']['id'],
   target_object_id=r['target_original']['object_id'],occluder_object_id=r['occluder_original']['object_id'],anchor_query=k,
   native_query_xy=q,nearest_pixel_xy=[px,py],crop96_xyxy=list(crop),diagnostic_crop_native_size=[96,96],model_roi_side=r['side'],
   original_prompt_box=r['box'],new_covered_original_visible_fraction=r['acceptance']['covered_fraction'],
   anchor_bits=dict(original_visible=bool(origvis[py,px]),paste=bool(paste[py,px]),controlled_visible=bool(states[py,px]==0),
    known_full_positive=bool(fp[py,px]),known_full_negative=bool(fn[py,px]),unknown_full=bool(unknown[py,px]),
    exact_controlled_state=int(states[py,px]),radius2_strict25_new_E=label['radius2_strict_new_controlled_E_core']),
   original_image=r['target_original']['image'],composite_image=r['image'],label_arrays=r['labels'],query_labels=r['query_labels'],raw_crop_outputs=paths)
  record['_views']=[full_marked(original,r['box'],crop,[px,py],r['query']),full_marked(composite,r['box'],crop,[px,py],r['query']),
   mark_local(raw_original,[px,py],q,crop[:2]),mark_local(raw_composite,[px,py],q,crop[:2]),mark_local(raw_cert,[px,py],q,crop[:2])]
  records.append(record)
 sheets=[]
 for start in (0,5):
  width,height=1252,1740;canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
  draw.text((16,10),f'Controlled public YCB source: scenes{start:02d}-{start+4:02d} / fixed first accepted',font=font(24),fill='black')
  draw.text((16,43),'No model scores. Original and composite crops are the SAME96 native pixels, shown2x nearest.',font=font(16),fill='black')
  draw.text((16,66),'Yellow=nearest anchor/diagnostic96box; cyan=unchanged roughbox or fractional query; white=other queries.',font=font(15),fill='#333333')
  draw.text((16,88),'Certificate: greenV / magenta NEW controlledE / grayO / blueU(original-hidden) / black=outside frame.',font=font(15),fill='#333333')
  for j,r in enumerate(records[start:start+5]):
   top=124+j*320;draw.rectangle((8,top-3,width-8,top+309),outline='#c7c7c7',width=1)
   q=r['native_query_xy'];p=r['nearest_pixel_xy'];cov=r['new_covered_original_visible_fraction']
   draw.text((16,top),f"Scene{r['scene_id']:02d} {r['split']} / obj{r['target_object_id']} <- occluder obj{r['occluder_object_id']} / q{r['anchor_query']}=({q[0]:.3f},{q[1]:.3f}) -> pixel({p[0]},{p[1]}) / coverage{cov:.2%}",font=font(17),fill='black')
   xlist=[16,316,618,826,1034];titles=['Original fullframe','Composite fullframe','Original96native','Composite96native','Controlled certificate']
   for x,title,view in zip(xlist,titles,r['_views']):
    draw.text((x,top+27),title,font=font(15),fill='black');canvas.paste(view,(x,top+49))
   draw.text((16,top+274),f"Anchor: origVis=1 paste=1 newVis=0 fullKnown+=1 fullKnown-=0 U=0; strict E={25}/25 pixels. Model ROI sideL={r['model_roi_side']:.3f}px.",font=font(16),fill='black')
   draw.text((16,top+295),r['source_target_id']+' / unknown original hidden support remains U; diagnostic crop is not a model crop resize.',font=font(13),fill='#555555')
  path=OUT/f'SCENES_{start:02d}_{start+4:02d}_CONTACT_SHEET.png';canvas.save(path);sheets.append(binding(path))
 for r in records:r.pop('_views')
 selection=dict(schema='V66_FIXED_CONTROLLED_SOURCE_SCORE_FREE_CROP_REVIEW',selection='first accepted row per original target source scene in sealed COMPOSITES.jsonl; scenes0..9 ascending',
  selection_not_representative=True,selection_used_model_scores=False,split_policy='TRAINscenes0..7; VALIDATIONscenes8..9',
  diagnostic_crop='96x96 native pixels, [px-48,py-48,px+48,py+48); lossless raw crops saved; display nearest2x; black padding outside canvas',
  source_seal=binding(DATA/'SEAL.json'),source_manifest=manifest_binding,code=binding(__file__),cases=records,contact_sheets=sheets,
  model_or_feature_extraction_calls=0,GPU_training_PnP_calls=0,current_private_RGB_GT_scores_read=False,sealed_generation_files_modified=False)
 (OUT/'SELECTION.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
 if sha(DATA/'SEAL.json')!=SEAL_SHA:raise AssertionError('Sealed generation changed during supplemental review')
 print(json.dumps({'cases':len(records),'contact_sheets':sheets,'selection':binding(OUT/'SELECTION.json'),'model_calls':0}))

if __name__=='__main__':main()
