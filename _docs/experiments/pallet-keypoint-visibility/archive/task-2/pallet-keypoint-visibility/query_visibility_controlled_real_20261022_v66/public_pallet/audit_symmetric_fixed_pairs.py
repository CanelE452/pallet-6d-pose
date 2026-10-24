#!/usr/bin/env python3
"""Fixed-six symmetric paired source evidence; no negative-label assumption."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mask(p):return cv2.imread(str(p),cv2.IMREAD_UNCHANGED)>0
def erode2(m):return cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
if __name__=='__main__':
 out=ROOT/'FIXED_SIX_SYMMETRIC_PAIR_EVIDENCE.json';assert not out.exists();protocol=ROOT/'SYMMETRIC_PAIR_PROTOCOL_ADDENDUM.json';assert protocol.exists()
 manifest=json.loads((ROOT/'EXACT_POSE_PAIR_MANIFEST.json').read_text());byid={p['pair_id']:p for p in manifest['pairs']};selection=json.loads((ROOT/'FIXED_SIX_PAIR_SELECTION.json').read_text())['pair_ids'];assert len(selection)==6
 canvas=Image.new('RGB',(1140,100+len(selection)*315),'white');d=ImageDraw.Draw(canvas);d.text((12,8),'Fixed real pairs: neutral A/B order, symmetric supplied-geometry evidence',font=font(22),fill='black');d.text((12,38),'A / B / evidence map: green=strict stable target; orange=A target covered in B; cyan=B target covered in A.',font=font(17),fill='black');d.text((12,65),'Mask0 is never E/O. Source manual-prompt SAM positives + measured depth + mesh checks; no fitting or new alignment.',font=font(16),fill='black');records=[]
 for i,ident in enumerate(selection):
  p=byid[ident];a,b=p['before'],p['after'];ma=mask(DATA/a['mask_teacher']);mb=mask(DATA/b['mask_teacher']);Da=cv2.imread(str(DATA/a['depth']),cv2.IMREAD_UNCHANGED).astype(float);Db=cv2.imread(str(DATA/b['depth']),cv2.IMREAD_UNCHANGED).astype(float);ca=erode2(ma);cb=erode2(mb)
  with np.load(ROOT/f"FIXED_PAIR_{p['session']}.npz")as z:Dm=z['render_depth_before_m'].astype(float)*1000
  mv=np.isfinite(Dm);agreeA=mv&(Da>0)&(np.abs(Dm-Da)<=np.maximum(20,.01*Da));agreeB=mv&(Db>0)&(np.abs(Dm-Db)<=np.maximum(20,.01*Db));seedA=ca&agreeA;seedB=cb&agreeB;both=(Da>0)&(Db>0);delta=Da-Db
  AB=seedA&(Db>0)&(delta>=50)&~agreeB;BA=seedB&(Da>0)&(-delta>=50)&~agreeA;stable=seedA&seedB&both&(np.abs(delta)<=20)
  baseStableA=ca&both&mb&(np.abs(delta)<=20);baseStableB=cb&both&ma&(np.abs(delta)<=20)
  knownA=seedA|BA;knownB=seedB|AB
  values=dict(seed_target_A=seedA,seed_target_B=seedB,A_target_externally_covered_in_B=AB,B_target_externally_covered_in_A=BA,strict_stable_visible_both=stable,mesh_depth_agrees_A=agreeA,mesh_depth_agrees_B=agreeB,depth_difference_A_minus_B_mm=delta.astype(np.float32))
  np.savez_compressed(ROOT/f"SYMMETRIC_FIXED_PAIR_{p['session']}.npz",**values)
  count=dict(seed_A=int(seedA.sum()),seed_B=int(seedB.sum()),strict_stable_visible_both=int(stable.sum()),base_stable_visible_A_to_B=int(baseStableA.sum()),base_stable_visible_B_to_A=int(baseStableB.sum()),A_to_B_external_50mm=int(AB.sum()),B_to_A_external_50mm=int(BA.sum()),A_to_B_external_100mm=int((seedA&(Db>0)&(delta>=100)&~agreeB).sum()),A_to_B_external_200mm=int((seedA&(Db>0)&(delta>=200)&~agreeB).sum()),B_to_A_external_100mm=int((seedB&(Da>0)&(-delta>=100)&~agreeA).sum()),B_to_A_external_200mm=int((seedB&(Da>0)&(-delta>=200)&~agreeA).sum()),A_to_B_external_but_B_teacher_positive=int((AB&mb).sum()),B_to_A_external_but_A_teacher_positive=int((BA&ma).sum()),A_to_B_external_eroded5x5=int(erode2(AB).sum()),B_to_A_external_eroded5x5=int(erode2(BA).sum()),known_A_including_E_evidence=int(knownA.sum()),known_B_including_E_evidence=int(knownB.sum()))
  record=dict(pair_id=ident,session=p['session'],split=p['split'],A_frame=a['frame_id'],B_frame=b['frame_id'],counts=count,source_pair_has_identical_supplied_pose_and_K=True,label_scope='Partial target visible/external source evidence; all remaining pixels U, no O/self GT',teacher_visible_independence=False,visual_review='A RGB is unobstructed bin, B adds front cardboard; higher-camera occlusion lies partly outside frame. Tiny reverse evidence is not automatically certified physical occlusion.',native_evidence_npz_sha256=sha(ROOT/f"SYMMETRIC_FIXED_PAIR_{p['session']}.npz"));records.append(record)
  y=100+i*315;d.text((12,y),ident+'   A/B are filenames, not a globally assumed ordering',font=font(18),fill='black')
  rgbA=np.array(Image.open(DATA/a['image']).convert('RGB'));rgbB=np.array(Image.open(DATA/b['image']).convert('RGB'));ev=np.full_like(rgbA,35);ev[stable]=[0,220,70];ev[AB]=[255,110,0];ev[BA]=[0,200,255]
  for j,image in enumerate([rgbA,rgbB,ev]):canvas.paste(Image.fromarray(image).resize((320,240),Image.Resampling.NEAREST),(12+380*j,y+26))
  d.text((12,y+273),f"strict stable {count['strict_stable_visible_both']}    A->B E50 {count['A_to_B_external_50mm']}    B->A E50 {count['B_to_A_external_50mm']}    all unsupported pixels U",font=font(17),fill='black')
  print('SYMMETRIC',ident,count,flush=True)
 canvas.save(ROOT/'FIXED_SIX_SYMMETRIC_PAIRED_RGBD_CONTACT_SHEET.png');result=dict(schema='PUBLIC_PALLOX_FIXED_SIX_SYMMETRIC_PAIR_EVIDENCE_V1',protocol_sha256=sha(protocol),pair_manifest_sha256=sha(ROOT/'EXACT_POSE_PAIR_MANIFEST.json'),fixed_selection_sha256=sha(ROOT/'FIXED_SIX_PAIR_SELECTION.json'),source_only=True,private_RGB_GT_scores_read=False,GPU_used=False,model_inference=False,training=False,pose_alignment_optimization=False,PnP_calls=0,margin50mm_predeclared=True,sensitivity_margins100_200_mm_only=True,cases=records,fit_admission='PENDING_REGISTRATION_AND_LABEL_QUALITY; THESE_ARE_EVIDENCE_NOT_HUMAN_PIXEL_GT',code_sha256=sha(__file__));out.write_text(json.dumps(result,indent=2)+'\n')
