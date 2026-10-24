"""First-by-state source RGB review, no re-selection/model/label admission."""
from pathlib import Path
import ast,hashlib,json,math,shutil,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SEAL=V/'corpus_build_v1/CORPUS_SEAL.json';SEAL_SHA='934a9a3aa88854cf0632fc7b3f4c89b5f0630cc435cfc1f64e1c4eea220903db'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def gridtext(a):return ' / '.join(' '.join('--'if x is None else f'{x:g}'for x in row)for row in np.asarray(a,dtype=object).reshape(3,3))
def select():
 assert bind(SEAL)['sha256']==SEAL_SHA
 s=load(SEAL);inp=rows(s['inputs']['path']);gt=rows(s['reference_GT_only']['path']);evidence=rows(V/'corpus_build_v1/SLOT_EVIDENCE8000_GT_ONLY.jsonl');chosen=[];missing=[]
 for oid in (2,3,7,8):
  for state in ('D','E'):
   options=[x for x in evidence if x['object_id']==oid and x['raw_surface_teacher_state']==state]
   if not options:missing.append(dict(object_id=oid,state=state));continue
   chosen.append(dict(requested_stratum=f'OBJ{oid:02d}_{state}',slot_index=options[0]['slot_index']))
 for state in ('S','U'):
  options=[x for x in evidence if x['raw_surface_teacher_state']==state]
  if not options:missing.append(dict(object_id='ALL',state=state));continue
  chosen.append(dict(requested_stratum=f'GLOBAL_{state}',slot_index=options[0]['slot_index']))
 for c in chosen:
  j=c['slot_index'];i,k=divmod(j,8);c.update(instance_row_index=i,logical_aux_slot=k,input=inp[i],GT_metadata=gt[i],saved_numeric_evidence=evidence[j]);assert evidence[j]['slot_index']==j
 lock=dict(completed=True,source_seal=bind(SEAL),code=bind(Path(__file__)),crop_code=bind(CROP),selection_rule='First original slot_index perD/E within eachobject2/3/7/8, then firstS/Uglobal; no score/quality/alternate selection.',selected=chosen,missing_strata=missing,RGB_pixels_read_before_lock=0,all_existing_TRAIN=True,source_emulation=True,native_mapping='UNKNOWN',human_gold=False,training_admission=False)
 write(H/'FIXED_SELECTION_BEFORE_RGB.json',lock);return lock
def main():
 assert not(H/'FIXED_SELECTION_BEFORE_RGB.json').exists();lock=select()
 node=next(n for n in ast.parse(CROP.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(CROP),'exec'),ns);crop=ns['crop']
 with np.load(V/'corpus_build_v1/RAW_SURFACE_TEACHER8000_GT_ONLY.npz',allow_pickle=False)as f:pack={k:np.array(f[k],copy=True)for k in f.files}
 bindings=load(V/'corpus_build_v1/SOURCE_DEPTH_MASK_BINDINGS.json');frames={x['frame_index']:x for x in bindings['frames']};rendered=[]
 for number,c in enumerate(lock['selected'],1):
  i,k=c['instance_row_index'],c['logical_aux_slot'];inp=c['input'];gt=c['GT_metadata'];ev=c['saved_numeric_evidence'];state=ev['raw_surface_teacher_state'];oid=ev['object_id'];q=np.array(inp['initial_points'][k]);r=np.array(gt['reference_xy'][k]);L=inp['roi_side'];image=Path(inp['image']['path']);assert bind(image)==inp['image']
  original=Image.open(image).convert('RGB');rgb=np.asarray(original);a,valid=crop(rgb,q,L,96);raw=Image.fromarray(a.transpose(1,2,0));folder=H/f'CASE{number:02d}_{c["requested_stratum"]}_SLOT{c["slot_index"]}';folder.mkdir();shutil.copyfile(image,folder/'FULL_ORIGINAL.png');raw.save(folder/'QUERY96_RAW.png')
  marked=original.copy();d=ImageDraw.Draw(marked);x,y=q;d.line((x-6,y,x+6,y),fill='#ff3e56',width=2);d.line((x,y-6,x,y+6),fill='#ff3e56',width=2);x,y=r;d.ellipse((x-5,y-5,x+5,y+5),outline='#00ffff',width=2);d.line((*q,*r),fill='yellow',width=1);d.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline='#ff3e56',width=1);marked.save(folder/'FULL_Q_R_MARKED.png')
  cm=raw.copy();d=ImageDraw.Draw(cm);centre=np.array([47.5,47.5]);d.line((centre[0]-4,centre[1],centre[0]+4,centre[1]),fill='#ff3e56');d.line((centre[0],centre[1]-4,centre[0],centre[1]+4),fill='#ff3e56');rp=centre+(r-q)*96/L;d.ellipse((rp[0]-3,rp[1]-3,rp[0]+3,rp[1]+3),outline='#00ffff');cm.save(folder/'QUERY96_Q_R_MARKED.png')
  card=Image.new('RGB',(1280,830),'#101821');d=ImageDraw.Draw(card);d.text((12,8),f'CASE {number:02d} | {c["requested_stratum"]} | original slot {c["slot_index"]} | scene {ev["source_scene_id"]}, frame {ev["source_frame_id"]}',font=font(22),fill='white');d.text((12,39),'CONDITIONAL SURFACE TEACHER | source q=GT reference+jitter | NOT native/human gold',font=font(18),fill='#ffdf70');card.paste(original,(0,82));card.paste(marked,(640,82));d.text((12,62),'UNMODIFIED FULL RGB',font=font(15),fill='white');d.text((650,62),'SEPARATE OVERLAY: q=red cross/square; r=cyan ring',font=font(15),fill='white');card.paste(raw.resize((240,240),Image.Resampling.NEAREST),(10,583));card.paste(cm.resize((240,240),Image.Resampling.NEAREST),(260,583));d.text((14,564),'ORIGINAL q/L RGB96',font=font(14),fill='white');d.text((265,564),'SEPARATE q/r OVERLAY',font=font(14),fill='white')
  depth=pack['neighbour_depth_raw_uint16'][i,k].astype(float)*pack['depth_scale_to_mm'][i];inside=pack['neighbour_inbounds'][i,k];depth=[None if not inside[j]else float(z)for j,z in enumerate(depth)];fr=frames[int(pack['frame_index'][i])];own=gt['source_instance']['instance_id'];ownermap={str(j):o for j,o in enumerate(fr['instance_object_ids'])};rays=ev['fixed_ray_firsthits'];gaps=pack['geometry_point_minus_first_range_mm'][i,k];finitegap=[None if not np.isfinite(x)else float(x)for x in gaps]
  lines=[f'Teacher state {state}; geometry {ev["geometry_state"]}; L={L:.3f}px; |q-r|={np.linalg.norm(q-r):.3f}px',f'q=({q[0]:.3f},{q[1]:.3f}), r=({r[0]:.3f},{r[1]:.3f})',f'Fixed reference cameraZ {ev["reference_cameraZ_mm"]:.3f}mm; ray gaps {finitegap}',f'3x3 depth mm: {gridtext(depth)}',f'3x3 own FULL: {gridtext(pack["neighbour_own_full"][i,k].astype(int))}',f'3x3 own VIS: {gridtext(pack["neighbour_own_visible"][i,k].astype(int))}',f'3x3 owner counts: {gridtext(pack["neighbour_visible_owner_count"][i,k])}',f'3x3 owner bits: {gridtext(pack["neighbour_owner_bits_uint64"][i,k])}; own index={own}',f'Instance index->objectID: {ownermap}',f'Reason: {ev["reason"]}']
  yy=569
  for line in lines:
   for wrapped in textwrap.wrap(line,width=99):d.text((517,yy),wrapped,font=font(13),fill='white');yy+=20
  path=folder/'CASE_CARD.png';card.save(path);rendered.append(dict(case_number=number,stratum=c['requested_stratum'],slot_index=c['slot_index'],source_image=inp['image'],case_card=bind(path),full_original=bind(folder/'FULL_ORIGINAL.png'),full_overlay=bind(folder/'FULL_Q_R_MARKED.png'),query96_raw=bind(folder/'QUERY96_RAW.png'),query96_overlay=bind(folder/'QUERY96_Q_R_MARKED.png'),query96_decoded_pixels_sha256=hashlib.sha256(a.tobytes()).hexdigest(),source_q=q.tolist(),fixed_r=r.tolist(),L=L,teacher_state=state,geometry_state=ev['geometry_state'],numeric_evidence=ev,depth_mm_3x3=depth,own_full_3x3=pack['neighbour_own_full'][i,k].tolist(),own_visible_3x3=pack['neighbour_own_visible'][i,k].tolist(),owner_bits_3x3=pack['neighbour_owner_bits_uint64'][i,k].tolist(),owner_index_to_objectID=ownermap,query_valid_fraction=float(valid.mean()),points_shifted=False))
 contacts=[]
 for page,start in enumerate(range(0,len(rendered),6),1):
  selected=rendered[start:start+6];sheet=Image.new('RGB',(1280,415*math.ceil(len(selected)/2)),'#101821')
  for j,c in enumerate(selected):sheet.paste(Image.open(c['case_card']['path']).resize((640,415),Image.Resampling.LANCZOS),((j%2)*640,(j//2)*415))
  p=H/f'CONTACT_{page:02d}_FIXED_CASES.png';sheet.save(p);contacts.append(bind(p))
 result=dict(completed=True,selection=bind(H/'FIXED_SELECTION_BEFORE_RGB.json'),rendered=rendered,contacts=contacts,selected_cases=len(rendered),missing_strata=lock['missing_strata'],unique_RGB_frames=len({c['source_image']['sha256']for c in rendered}),repeated_scene_frames=True,no_label_admission=True,no_models_fit_current_PnP=True)
 write(H/'RENDER_RECEIPTS.json',result);print(json.dumps(dict(completed=True,receipt=bind(H/'RENDER_RECEIPTS.json'),contacts=contacts,cards=[c['case_card']['path']for c in rendered])))
if __name__=='__main__':main()
