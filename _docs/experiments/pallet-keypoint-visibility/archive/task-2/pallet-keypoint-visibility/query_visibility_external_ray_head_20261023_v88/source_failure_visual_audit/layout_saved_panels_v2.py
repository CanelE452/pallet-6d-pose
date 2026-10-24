"""Additive visual layout repair from saved panels only. No raw/source array I/O."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;D=O/'LAYOUT_QA_V2';D.mkdir(exist_ok=False)
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def w(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
receipt=json.loads((O/'RENDER_AND_PIXEL_PROVENANCE_RECEIPT.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);imgs=[];proof=[]
for c in receipt['cases']:
 oldpath=Path(c['figure']['path']);assert b(oldpath)['sha256']==c['figure']['sha256'];old=Image.open(oldpath).convert('RGB');out=Image.new('RGB',(1510,1130),'#14212d');draw=ImageDraw.Draw(out)
 # Identical original title and source/data panels, no recomputation or resizing.
 components=[((0,0,1510,72),(0,0)),((15,76,655,588),(15,76)),((680,76,1064,492),(680,76)),((1080,76,1464,492),(1080,76)),((680,505,1510,535),(680,505)),((680,550,872,742),(1260,730))]
 for box,xy in components:
  tile=old.crop(box);out.paste(tile,xy);assert out.crop((xy[0],xy[1],xy[0]+tile.width,xy[1]+tile.height)).tobytes()==tile.tobytes()
 draw.text((15,610),'Yellow: original q-centered square. Cyan: initial query q. Magenta: unchanged virtual reference r.',font=font,fill='white')
 draw.text((15,640),'Green: target visibleM4 outline (GT display only). Mask absence is NOT semantic external/self occlusion.',font=font,fill='white')
 draw.text((15,670),'RGB96 uses original modeled native span L; the separate native96 diagnostic has a different span.',font=font,fill='white')
 draw.text((1260,700),'Cached valid96',font=small,fill='white');draw.text((15,720),f"Canvas valid fraction={c['local_canvas_valid_fraction']:.6f}; qvalid=1; saved input_available=1.",font=font,fill='white')
 t=c['original_teacher_numeric_metadata'];lines=[f"Teacher: {t['reason8']} / relation={t['relation8']}",f"Visible-mask owner={t['owner_type8']} count={t['owner_count8']} unique_instance_ordinal={t['unique_owner_id8']}",f"depth_raw={t['depth_raw8']}; observed_Z_mm={t['observed_Z_mm8']:.12g}; gamma={t['gamma8']:.12g}",f"ref_range_mm={t['ref_range_mm8']:.12g}",f"observed_range_mm interval=[{t['observed_range_lo_mm8']:.12g}, {t['observed_range_hi_mm8']:.12g}]"]
 for j,line in enumerate(lines):draw.text((15,765+j*33),line,font=small,fill='#d5e3eb')
 for j,line in enumerate(['CLEAR means clear-to-virtual-endpoint, not physical cue visibility or human D.','External-ray proxy is not human E; convex self proxy is not human S.','Emulated source q error <=3px is NOT the current manual-click <=3px cohort.','Fixed minSHA diagnostic cases; no replacement, coordinates/labels/score changes, new model/fit/current/PnP.']):draw.text((15,965+j*34),line,font=small,fill='#d5e3eb')
 p=D/oldpath.name;out.save(p);imgs.append(out);proof.append({'case_number':c['selection']['case_number'],'original':c['figure'],'layoutV2':b(p),'all_source_data_panels_identical_crop_bytes':True,'only_legends_and_validity_panel_position_changed':True})
contact=Image.new('RGB',(1510,1695),'#14212d')
for i,im in enumerate(imgs):contact.paste(im.resize((755,565),Image.Resampling.LANCZOS),((i%2)*755,(i//2)*565))
contact.save(D/'ALL_FIXED_SIX_CONTACT.png')
w(D/'LAYOUT_REPAIR_RECEIPT.json',{'completed':True,'PASS':True,'reason':'V1 legend overlapped cached-validity panel; all V1 files preserved unchanged.','code':b(Path(__file__)),'original_render_receipt':b(O/'RENDER_AND_PIXEL_PROVENANCE_RECEIPT.json'),'cases':proof,'contact':b(D/'ALL_FIXED_SIX_CONTACT.png'),'raw_RGB_mask_npz_reads':0,'P3_weights_current_model_forward_fit_PnP':0})
print(json.dumps({'completed':True,'contact':b(D/'ALL_FIXED_SIX_CONTACT.png'),'receipt':b(D/'LAYOUT_REPAIR_RECEIPT.json')}))
