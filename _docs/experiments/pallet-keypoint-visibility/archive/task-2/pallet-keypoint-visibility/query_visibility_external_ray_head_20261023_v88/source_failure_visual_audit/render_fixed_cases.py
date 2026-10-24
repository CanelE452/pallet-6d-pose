"""Render only previously sealed source cases; never loads P3/checkpoints/models."""
from pathlib import Path
import json,hashlib,zipfile,math,datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;ROOT=O.parent

def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def selected_member(path,member,local_i,k,queryshape,dtype):
 # Stream only the two allowed .npy members, not P3 nor whole query tensors.
 with zipfile.ZipFile(path) as z:
  with z.open(member+'.npy') as f:
   version=np.lib.format.read_magic(f);assert version in ((1,0),(2,0))
   shape,fortran,dt=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(f)
   assert not fortran and tuple(shape[1:])==(8,*queryshape) and dt==np.dtype(dtype)
   n=int(np.prod(queryshape));data_start=f.tell();f.seek(data_start+(local_i*8+k)*n*dt.itemsize);raw=f.read(n*dt.itemsize);assert len(raw)==n*dt.itemsize
 return np.frombuffer(raw,dtype=dt).copy().reshape(queryshape)
entry=read(O/'RENDER_CODE_LOCK_BEFORE_RGB.json');assert entry['code']==bind(__file__);check(entry['selection_seal'])
seal=read(entry['selection_seal']['path']);assert seal['completed'] and seal['PASS']
for b in seal['files']:check(b)
selection=read(O/'SELECTION_AND_COHORT_LOCK_BEFORE_RGB.json');metadata=read(O/'SELECTED_CASE_METADATA_GT_ONLY.json');assert selection['selected_case_N']==len(metadata['cases'])<=6
pack=selection['input_pack'];manifest=read(pack['manifest']['path']);check(pack['manifest'])
V87=Path(selection['teacher']['path']).parent;maskinventory=V87/'SOURCE_DEPTH_MASK_BYTE_BINDINGS.json';maskbindings={x['path']:x for x in read(maskinventory)['files']}
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,18);small=ImageFont.truetype(fontpath,15);title=ImageFont.truetype(fontpath,22)
rendered=[];chunk_checked=set();images=[];crops=[];valids=[]
for case in metadata['cases']:
 c=case['selection'];i,k=c['frame_index'],c['native_k'];n=c['case_number'];q=np.array([c['q_x'],c['q_y']]);r=np.array([c['r_x'],c['r_y']]);L=c['L'];m=case['source_metadata_original']['metadata'];inp=case['input_query_metadata'];line=case['input_lineage_metadata'];teacher=case['teacher_original_numeric_values'];assert i==inp['frame_index'] and c==selection['selected_cases'][n]
 imagepath=Path(m['image']);assert m['image_sha256']==inp['image_sha256']==line['image_sha256'];assert bind(imagepath)['sha256']==m['image_sha256']
 with Image.open(imagepath) as im:rgb=np.asarray(im.convert('RGB')).copy()
 assert rgb.shape==(480,640,3) and list(inp['raw_hw'])==[480,640]
 maskpath=Path(m['mask_vis']);check(maskbindings[str(maskpath)])
 with Image.open(maskpath) as im:mask=np.asarray(im).copy()>0
 assert mask.shape==rgb.shape[:2]
 chunk=next(x for x in manifest['chunks'] if x['start']<=i<x['end']);cp=chunk['file']
 if cp['path'] not in chunk_checked:check(cp);chunk_checked.add(cp['path'])
 rgb96=selected_member(cp['path'],'rgb96',i-chunk['start'],k,(3,96,96),'uint8').transpose(1,2,0);valid=selected_member(cp['path'],'crop_valid96',i-chunk['start'],k,(1,96,96),'bool')[0]
 assert rgb96.shape==(96,96,3) and valid.any();crops.append(rgb96.copy());valids.append(valid.copy())
 qpix=np.floor(q+.5).astype(int);rpix=np.floor(r+.5).astype(int);assert np.array_equal(rpix,np.asarray(teacher['pixel_xy8']))
 # Green target visible-mask outline is a GT-only display, never input.
 eroded=mask.copy();eroded[1:]&=mask[:-1];eroded[:-1]&=mask[1:];eroded[:,1:]&=mask[:,:-1];eroded[:,:-1]&=mask[:,1:];edge=mask&~eroded
 overlay=rgb.copy();overlay[edge]=[30,255,100];full=Image.fromarray(overlay);d=ImageDraw.Draw(full)
 qs=[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2];d.rectangle(qs,outline='#ffff00',width=2)
 for xy,color,kind in [(q,'#00e5ff','q'),(r,'#ff4ada','r')]:
  x,y=xy;d.line((x-5,y,x+5,y),fill=color,width=2);d.line((x,y-5,x,y+5),fill=color,width=2);d.text((x+7,y-15),kind,fill=color,font=small)
 # Exactly cached modeled RGB; only display enlargement, no new resampling as model input.
 modeled=Image.fromarray(rgb96).resize((384,384),Image.Resampling.NEAREST);md=ImageDraw.Draw(modeled)
 for xy,color,kind in [(q,'#00e5ff','q'),(r,'#ff4ada','r')]:
  ix=(xy-q)/L*96+47.5;x,y=(ix+.5)*4-.5;md.line((x-9,y,x+9,y),fill=color,width=2);md.line((x,y-9,x,y+9),fill=color,width=2);md.text((x+9,y-24),kind,fill=color,font=font)
 # Different diagnostic native span96 about nearest q, explicitly not model crop.
 box=[int(qpix[0])-48,int(qpix[1])-48,int(qpix[0])+48,int(qpix[1])+48];native=Image.fromarray(rgb).crop(box).resize((384,384),Image.Resampling.NEAREST);nd=ImageDraw.Draw(native)
 for xy,color,kind in [(q,'#00e5ff','q'),(r,'#ff4ada','r')]:
  x=(xy[0]-box[0]+.5)*4-.5;y=(xy[1]-box[1]+.5)*4-.5;nd.line((x-9,y,x+9,y),fill=color,width=2);nd.line((x,y-9,x,y+9),fill=color,width=2);nd.text((x+9,y-24),kind,fill=color,font=font)
 canvas=Image.new('RGB',(1510,960),'#14212d');cd=ImageDraw.Draw(canvas);cd.text((15,10),f"{n:02d} {c['stratum']} / {c['id']} / native k{k}",font=title,fill='white')
 cd.text((15,40),f"GT ray={c['GT']} pred={c['pred']}  p_external={c['p_external_derived_display']:.4f}  selfproxy={c['convexselfproxy']}  |q-r|={c['Euclidean_q_r_native_px']:.4f}px  L={L:.4f}px",font=font,fill='white')
 cd.text((15,76),'FULL native RGB + GT target M4 outline',font=font,fill='white');canvas.paste(full,(15,108))
 cd.text((680,76),'ACTUAL cached MODEL INPUT RGB96',font=font,fill='#ffff00');canvas.paste(modeled,(680,108));cd.text((680,505),f'Modeled native span L={L:.3f}px; display4x',font=small,fill='#ffff00')
 cd.text((1080,76),'Separate NATIVE96 diagnostic',font=font,fill='white');canvas.paste(native,(1080,108));cd.text((1080,505),'Fixed96 native px; NOT model input',font=small,fill='white')
 vm=Image.fromarray((valid.astype(np.uint8)*255)).convert('RGB').resize((192,192),Image.Resampling.NEAREST);canvas.paste(vm,(680,550));cd.text((890,565),f'Cached canvas-valid fraction={valid.mean():.6f}',font=small,fill='white');cd.text((890,595),f'qvalid=1 input_available=1 ROIany={int(line["ROI_anyvalid8"][k])}',font=small,fill='white')
 cd.text((15,610),'Yellow: unchanged q-centered square. Cyan: initial q. Magenta: original virtual reference r.',font=font,fill='white');cd.text((15,640),'Green: target visibleM4 outline (GT-only display); absence of M4 is NOT semantic E/S.',font=font,fill='white')
 desc=[f"Teacher reason={teacher['reason8']} / relation={teacher['relation8']} / owner={teacher['owner_type8']} count={teacher['owner_count8']} unique_id={teacher['unique_owner_id8']}",f"depth_raw={teacher['depth_raw8']}  Zobs_mm={teacher['observed_Z_mm8']:.9g}  gamma={teacher['gamma8']:.12g}",f"ref_range_mm={teacher['ref_range_mm8']:.12g} / obs_range_interval=[{teacher['observed_range_lo_mm8']:.12g},{teacher['observed_range_hi_mm8']:.12g}]",'CLEAR is clear-to-virtual-endpoint, not humanD. External proxy is not humanE. Self proxy is not humanS.','q is source emulation; this <=3px cohort is NOT current manual-click<=3px. No new model/fit/PnP.']
 for j,t in enumerate(desc):cd.text((15,765+j*34),t,font=small,fill='#d5e3eb')
 figure=O/f'CASE_{n:02d}_{c["stratum"]}.png';canvas.save(figure);images.append(canvas)
 Image.fromarray(rgb96).save(O/f'CASE_{n:02d}_ACTUAL_CACHED_RGB96.png');Image.fromarray(valid.astype(np.uint8)*255).save(O/f'CASE_{n:02d}_ACTUAL_CACHED_VALID96.png')
 rec=dict(completed=True,selection=c,source_RGB=bind(imagepath),decoded_RGB_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_maskM4=maskbindings[str(maskpath)],cached_container=cp,whitelisted_members_read=['rgb96','crop_valid96'],P3_members_deserialized=0,RGB96_array_SHA256=hashlib.sha256(rgb96.tobytes()).hexdigest(),valid96_array_SHA256=hashlib.sha256(valid.tobytes()).hexdigest(),local_canvas_valid_fraction=float(valid.mean()),q_native=q.tolist(),reference_native_GT_only=r.tolist(),original_qsquare_xyxy=qs,L=L,model_sample_formula='q+((j+.5)/96-.5)*L; q maps to index47.5, no exactcenter sample in even96',separate_native96_box=box,actual_targetM4_at_q=bool(mask[qpix[1],qpix[0]]),actual_targetM4_at_reference_pixel=bool(mask[rpix[1],rpix[0]]),original_teacher_numeric_metadata=teacher,source_target_object_id=m['object_id'],source_target_instance_ordinal=m['instance_id'],figure=bind(figure),pixels_or_mask_used_to_reselect=False,new_forward_fit_current_PnP=0)
 write(O/f'CASE_{n:02d}_PIXEL_AND_PROVENANCE.json',rec);rendered.append(rec)
contact=Image.new('RGB',(1510,3*480),'#14212d')
for i,im in enumerate(images):contact.paste(im.resize((755,480),Image.Resampling.LANCZOS),((i%2)*755,(i//2)*480))
contact.save(O/'ALL_FIXED_SIX_CONTACT.png')
np.savez_compressed(O/'SELECTED_ACTUAL_CACHED_RGB96_VALID_ONLY.npz',rgb96=np.stack(crops),valid96=np.stack(valids),q=np.array([[x['selection']['q_x'],x['selection']['q_y']] for x in metadata['cases']]),reference_GT_only=np.array([[x['selection']['r_x'],x['selection']['r_y']] for x in metadata['cases']]),L=np.array([x['selection']['L'] for x in metadata['cases']]))
write(O/'RENDER_AND_PIXEL_PROVENANCE_RECEIPT.json',dict(completed=True,code=bind(__file__),selection_seal=entry['selection_seal'],cases=rendered,contact=bind(O/'ALL_FIXED_SIX_CONTACT.png'),selected_input_arrays=bind(O/'SELECTED_ACTUAL_CACHED_RGB96_VALID_ONLY.npz'),selected_raw_RGB_decodes=6,selected_M4_decodes=6,selected_cached_query_RGB_reads=6,selected_cached_query_validity_reads=6,P3_members_deserialized=0,checkpointweights_reads=0,NPZ_container_SHA_byte_reads=len(chunk_checked),model_forward_fit_current_PnP=0,visual_judgments_not_yet_written=True))
print(json.dumps({'completed':True,'N':len(rendered),'contact':bind(O/'ALL_FIXED_SIX_CONTACT.png'),'cases':[{k:x[k] for k in ('selection','source_target_object_id','actual_targetM4_at_q','actual_targetM4_at_reference_pixel','local_canvas_valid_fraction')} for x in rendered]},ensure_ascii=False))
