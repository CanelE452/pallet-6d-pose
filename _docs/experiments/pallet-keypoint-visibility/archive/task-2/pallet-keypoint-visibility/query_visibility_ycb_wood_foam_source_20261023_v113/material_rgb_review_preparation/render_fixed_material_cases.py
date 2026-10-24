"""Prospective fixed material-source RGB diagnostics. No model/fit/private access."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,math,shutil,sys,textwrap
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_EXACT3200_MATERIAL_TEACHER_POLICY_BEFORE_LABELS.json'
POLICY_SHA='2b492b0334ec6c037205b44cc71b03f7a60f84afe9c0250f58e91fd03425499b'
TEACHER_SEAL_SHA='573b2b3fc93869eb11f699a796bfb8ed3086e7696851f56db929a854f0ac0634'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CONTEXT=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
OUT=V/'fixed_material_rgb_review'
STRATA=((16,'TRAIN'),(16,'VALIDATION'),(21,'TRAIN'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def samplers():
 node=next(x for x in ast.parse(CROP.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='crop')
 ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(CROP),'exec'),ns)
 spec=importlib.util.spec_from_file_location('v113_context',CONTEXT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return ns['crop'],m.sample_target_context
def choose(evidence,inputs):
 ordered=sorted(evidence,key=lambda x:x['slot_index']);selected=[];missing=[]
 for oid,split in STRATA:
  for state in('D','E'):
   found=[x for x in ordered if x['object_id']==oid and inputs[x['slot_index']//8]['split']==split and x['raw_surface_teacher_state']==state]
   if found:selected.append(dict(stratum=f'OBJ{oid:02d}_{split}_{state}',slot_index=found[0]['slot_index']))
   else:missing.append(dict(object_id=oid,split=split,state=state))
 for state in('S','U'):
  found=[x for x in ordered if x['raw_surface_teacher_state']==state]
  if found:selected.append(dict(stratum=f'GLOBAL_{state}',slot_index=found[0]['slot_index']))
  else:missing.append(dict(object_id='ALL',split='ALL',state=state))
 assert len(selected)<=8
 return selected,missing
def prepare():
 assert sha(POLICY)==POLICY_SHA and not(H/'RENDER_PROTOCOL_BEFORE_RGB.json').exists()
 proof=H/'GENERATED_RENDER_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 query=V/'query_preparation/PRODUCER_PROTOCOL.json';q=read(query)
 teacher=dict(path=str(V/'corpus_build_v1/CORPUS_SEAL.json'),sha256=TEACHER_SEAL_SHA);check(teacher)
 protocol=dict(completed=True,root_teacher_policy=bind(POLICY),code=bind(__file__),generated=bind(proof),
               exact_V90_crop=bind(CROP),exact_V107_context=bind(CONTEXT),query_protocol=bind(query),inputs=q['inputs'],reference_GT_only=q['reference_GT_only'],completed_teacher_seal=teacher,
               selection_rule='First originalglobalrow D and E within (obj16TRAIN,obj16VALIDATION,obj21TRAIN), then globalfirst S/U. Missing strata remain missing, no replacement.',
               maximum_cases=8,context_box='Exact frozen query_box_xyxy, no clipping/expansion/GT recenter.',
               raw96='Exact V90crop unchanged q/L; raw/valid stored, marker only in separate copy.',
               context224='Exact V107 sample_target_context; FP32/valid retained in NPZ, PNG is rounded display only.',
               native_mapping='UNKNOWN',human_gold=False,source_VAL_is_development_exposed=True,
               actual_source_RGB_reads=0,model_fit_private_V112_PnP=False,output=str(OUT))
 write(H/'RENDER_PROTOCOL_BEFORE_RGB.json',protocol)
 write(H/'ROOT_RENDER_ENTRY_TEMPLATE.json',dict(approved_actual_public_RGB_diagnostic=False,model_fit_private_V112_PnP_allowed=False,
       protocol=bind(H/'RENDER_PROTOCOL_BEFORE_RGB.json'),code=bind(__file__),
       completed_teacher_seal=teacher,output=str(OUT)))
 print(json.dumps(dict(protocol=bind(H/'RENDER_PROTOCOL_BEFORE_RGB.json'),template=bind(H/'ROOT_RENDER_ENTRY_TEMPLATE.json'),command=f'{sys.executable} {Path(__file__).resolve()} --entry EXACT_ROOT_RENDER_ENTRY.json')))
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def gridtext(values):return ' / '.join(' '.join('--'if x is None else f'{x:g}'for x in rr)for rr in np.asarray(values,dtype=object).reshape(3,3))
def cross(draw,xy,color,radius=5,width=2):
 x,y=map(float,xy);draw.line((x-radius,y,x+radius,y),fill=color,width=width);draw.line((x,y-radius,x,y+radius),fill=color,width=width)
def mark(image,q,r,L=None,box=None):
 out=image.copy();d=ImageDraw.Draw(out);cross(d,q,'#ff3656');x,y=r;d.ellipse((x-4,y-4,x+4,y+4),outline='#00ffff',width=2)
 if L is not None:d.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline='#ff3656',width=1)
 if box is not None:d.rectangle(tuple(box),outline='#ffff00',width=1)
 return out
def run(entry_path):
 e=read(entry_path);assert e['approved_actual_public_RGB_diagnostic']is True and e['model_fit_private_V112_PnP_allowed']is False
 p=read(check(e['protocol']));assert p['code']==e['code']and check(e['code'])==Path(__file__).resolve()
 assert sha(POLICY)==POLICY_SHA and not OUT.exists()and Path(e['output']).resolve()==OUT.resolve()
 for key in('root_teacher_policy','generated','exact_V90_crop','exact_V107_context','query_protocol','inputs','reference_GT_only'):check(p[key])
 assert e['completed_teacher_seal']==p['completed_teacher_seal']
 seal=read(check(e['completed_teacher_seal']));assert seal['completed']and seal['all3200_preserved']
 assert seal['inputs']==p['inputs']and seal['reference_GT_only']==p['reference_GT_only']
 for b in seal['outputs']:check(b)
 inputs=rows(p['inputs']['path']);gt=rows(p['reference_GT_only']['path']);evidence=rows(V/'corpus_build_v1/SLOT_EVIDENCE3200_GT_ONLY.jsonl')
 assert len(inputs)==len(gt)==400 and len(evidence)==3200
 assert all(x['slot_index']==i for i,x in enumerate(evidence))
 chosen,missing=choose(evidence,inputs)
 OUT.mkdir();write(OUT/'FIXED_SELECTION_BEFORE_RGB.json',dict(completed=True,entry=bind(entry_path),protocol=e['protocol'],teacher_seal=e['completed_teacher_seal'],
       selected=chosen,missing_strata=missing,selection_uses_scores=False,RGB_pixels_read_before_selection_lock=0,
       all3200_retained=True,source_VAL_is_development_exposed=True,human_gold=False,native_mapping='UNKNOWN'))
 crop,context=samplers()
 with np.load(V/'corpus_build_v1/RAW_SURFACE_TEACHER3200_GT_ONLY.npz',allow_pickle=False)as f:pack={k:np.array(f[k],copy=True)for k in f.files}
 frames={x['frame_index']:x for x in read(V/'corpus_build_v1/SOURCE_DEPTH_MASK_BINDINGS.json')['frames']};rendered=[]
 for n,selected in enumerate(chosen,1):
  index,k=divmod(selected['slot_index'],8);inp=inputs[index];g=gt[index];ev=evidence[selected['slot_index']]
  assert inp['input_id']==g['input_id']==ev['input_id']
  q=np.asarray(inp['initial_points'][k]);r=np.asarray(g['reference_xy'][k]);L=inp['roi_side'];box=np.asarray(inp['query_box_xyxy'])
  path=check(inp['image']);original=Image.open(path).convert('RGB');rgb=np.asarray(original)
  assert rgb.shape[:2]==tuple(inp['raw_hw'])
  a,valid=crop(rgb,q,L,96);ctx,cv=context(rgb,box,(224,224))
  raw=Image.fromarray(a.transpose(1,2,0));valid_image=Image.fromarray(valid[0].astype(np.uint8)*255)
  ctx_image=Image.fromarray(np.floor(np.clip(ctx,0,255)+.5).astype(np.uint8))
  centre=np.array([47.5,47.5]);rp=centre+(r-q)*96/L;raw_mark=mark(raw,centre,rp)
  qctx=(q-box[:2])*224/(box[2:]-box[:2])-.5;rctx=(r-box[:2])*224/(box[2:]-box[:2])-.5
  ctx_mark=mark(ctx_image,qctx,rctx);full_mark=mark(original,q,r,L,box)
  folder=OUT/f'CASE{n:02d}_{selected["stratum"]}_SLOT{selected["slot_index"]}';folder.mkdir()
  shutil.copyfile(path,folder/'FULL_ORIGINAL.png')
  images={'FULL_Q_R_BOX_MARKED.png':full_mark,'QUERY96_RAW.png':raw,'QUERY96_VALID.png':valid_image,'QUERY96_Q_R_MARKED.png':raw_mark,'CONTEXT224_DISPLAY.png':ctx_image,'CONTEXT224_Q_R_MARKED.png':ctx_mark}
  for name,image in images.items():image.save(folder/name)
  np.savez_compressed(folder/'EXACT_RAW_CROPS_AND_VALID.npz',rgb96=a,crop_valid96=valid,context224_fp32=ctx,context_valid224=cv)
  depth=pack['neighbour_depth_raw_uint16'][index,k].astype(float)*pack['depth_scale_to_mm'][index];inside=pack['neighbour_inbounds'][index,k]
  depth=[None if not inside[j]else float(z)for j,z in enumerate(depth)]
  gaps=[None if not np.isfinite(z)else float(z)for z in pack['geometry_point_minus_first_range_mm'][index,k]]
  frame=frames[int(pack['frame_index'][index])];owner_map={str(j):oid for j,oid in enumerate(frame['instance_object_ids'])}
  lines=[f'State {ev["raw_surface_teacher_state"]}; geom {ev["geometry_state"]}',f'Original split {inp["original_split"]}; VAL development-exposed',
         f'L {L:.3f}px; |q-r| {np.linalg.norm(q-r):.3f}px',f'q {q.tolist()}',f'r {r.tolist()}',f'RefZ {ev["reference_cameraZ_mm"]:.3f}mm',f'Ray gaps mm {gaps}',
         f'Depth mm {gridtext(depth)}',f'Own FULL {gridtext(pack["neighbour_own_full"][index,k].astype(int))}',f'Own VIS {gridtext(pack["neighbour_own_visible"][index,k].astype(int))}',
         f'Owner counts {gridtext(pack["neighbour_visible_owner_count"][index,k])}',f'Owner bits {gridtext(pack["neighbour_owner_bits_uint64"][index,k])}',
         f'Own instance {g["source_instance"]["instance_id"]}; index->obj {owner_map}',f'Reason {ev["reason"]}']
  card=Image.new('RGB',(1600,880),'#101821');d=ImageDraw.Draw(card)
  d.text((10,7),f'CASE{n:02d} | {selected["stratum"]} | global slot{selected["slot_index"]} | scene{ev["source_scene_id"]}/frame{ev["source_frame_id"]}',font=font(23),fill='white')
  d.text((10,38),'CONDITIONAL SURFACE teacher | q=GT+jitter emulation | nativeUNKNOWN / humanGoldFALSE | no new labels',font=font(18),fill='#ffdc70')
  d.text((10,66),'NATIVE FULL RGB (unmodified)',font=font(15),fill='white');d.text((650,66),'SEPARATE OVERLAY: red q/L, cyan r, yellow frozen q8box',font=font(15),fill='white')
  card.paste(original,(0,88));card.paste(full_mark,(640,88))
  for x,label in((10,'EXACT q/L RGB96'),(260,'SEPARATE q/r marker'),(510,'RGB96 valid'),(715,'V107 context224 display'),(950,'SEPARATE context q/r')):d.text((x,584),label,font=font(14),fill='white')
  card.paste(raw.resize((240,240),Image.Resampling.NEAREST),(10,610));card.paste(raw_mark.resize((240,240),Image.Resampling.NEAREST),(260,610));card.paste(valid_image.resize((192,192),Image.Resampling.NEAREST),(510,610));card.paste(ctx_image,(715,610));card.paste(ctx_mark,(950,610))
  y=90
  for line in lines:
   for t in textwrap.wrap(line,width=39):d.text((1290,y),t,font=font(13),fill='white');y+=18
  card.save(folder/'CASE_CARD.png')
  rendered.append(dict(case_number=n,**selected,input_id=inp['input_id'],split=inp['split'],original_split=inp['original_split'],
                       source_image=inp['image'],q=q.tolist(),r=r.tolist(),L=L,frozen_context_box=box.tolist(),teacher_state=ev['raw_surface_teacher_state'],
                       case_card=bind(folder/'CASE_CARD.png'),files=[bind(x)for x in sorted(folder.iterdir())if x.is_file()],
                       exact_raw_crop_values_sha256=hashlib.sha256(a.tobytes()).hexdigest(),raw_valid_fraction=float(valid.mean()),context_valid_fraction=float(cv.mean()),
                       numeric_evidence=ev,depth_mm3x3=depth,owner_index_to_objectID=owner_map,no_point_query_change=True))
 contacts=[]
 for page,start in enumerate(range(0,len(rendered),8),1):
  subset=rendered[start:start+8];sheet=Image.new('RGB',(1600,440*math.ceil(len(subset)/2)),'#101821')
  for j,c in enumerate(subset):sheet.paste(Image.open(c['case_card']['path']).resize((800,440),Image.Resampling.LANCZOS),((j%2)*800,(j//2)*440))
  path=OUT/f'CONTACT_{page:02d}_FIXED_MATERIAL_CASES.png';sheet.save(path);contacts.append(bind(path))
 result=dict(completed=True,entry=bind(entry_path),protocol=e['protocol'],selection=bind(OUT/'FIXED_SELECTION_BEFORE_RGB.json'),teacher_seal=e['completed_teacher_seal'],
             selected_cases=len(rendered),missing_strata=missing,unique_source_RGB_frames=len({x['source_image']['sha256']for x in rendered}),rendered=rendered,contacts=contacts,
             all3200_retained=True,source_VAL_is_development_exposed=True,training_admission=False,human_gold=False,native_mapping='UNKNOWN',
             model_fit_private_V112_PnP=False)
 write(OUT/'RENDER_RECEIPTS.json',result);write(OUT/'RENDER_SEAL.json',dict(completed=True,receipt=bind(OUT/'RENDER_RECEIPTS.json'),outputs=contacts+[x for c in rendered for x in c['files']],actual_visual_observation_pending=True))
 print(json.dumps(dict(completed=True,receipt=bind(OUT/'RENDER_RECEIPTS.json'),seal=bind(OUT/'RENDER_SEAL.json'),contacts=contacts,selected_cases=len(rendered),missing_strata=missing)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or exact approved root --entry.')
