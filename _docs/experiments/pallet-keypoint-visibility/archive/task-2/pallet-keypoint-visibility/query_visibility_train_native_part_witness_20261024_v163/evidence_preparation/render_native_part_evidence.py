"""V163 input-only evidence. Preparation/invented checks never open source pixels."""
from pathlib import Path
import argparse, hashlib, json, traceback
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_train_native_part_witness_20261024_v163'
O=Path(__file__).resolve().parent
OUT=V/'evidence_v1'
V159=P/'query_visibility_native_part_support_feasibility_20261024_v159'
V158=P/'query_visibility_train_query_owner_witness_20261024_v158'/'review_evidence_preparation'
S=P/'query_visibility_mr6d_full_public_source_20261024_v153'
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
SALT='V163_NATIVE_PART_WITNESS_V1'
SCOPE=V/'ROOT_SCOPE_BEFORE_NATIVE_PART_WITNESS_PIXELS.json'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
CARD_SIZE=(1920,1400)

def utc(): return datetime.now(timezone.utc).isoformat()
def guard(p):
 p=Path(p).resolve();s=str(p).lower()
 assert p.is_relative_to(P) and not any(x in s for x in ('v112','current223','/private','/held'))
 assert p.suffix.lower() not in ('.pt','.pth','.ckpt','.safetensors')
 return p
def sha(p):return hashlib.sha256(guard(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(guard(p)),sha256=sha(p))
def verify(b):
 p=guard(b['path']);assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(guard(p).read_text())
def readb(b):return json.loads(verify(b).read_text())
def jsonl(b):return [json.loads(x) for x in verify(b).read_text().splitlines()]
def write(p,x):
 p=guard(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writejl(p,rows):
 with guard(p).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def digest(r):return hashlib.sha256((SALT+r['case_id']).encode()).hexdigest()
def assignments(cases):
 ordered=sorted(cases,key=lambda r:(digest(r),r['case_id']))
 groups={g:[r['case_id'] for j,r in enumerate(ordered) if j%3==i] for i,g in enumerate('ABC')}
 cross=[]
 for k in range(8):cross.extend(r['case_id'] for r in sorted((r for r in cases if r['native_slot']==k),key=lambda r:(digest(r),r['case_id']))[:5])
 for r in ordered:
  if len(cross)==40:break
  if r['case_id'] not in cross:cross.append(r['case_id'])
 return groups,cross

def project(corners,R,t,K):
 camera=np.asarray(corners)@np.asarray(R).T+np.asarray(t)
 assert np.all(camera[:,2]>0)
 h=camera@np.asarray(K).T
 return h[:,:2]/h[:,2,None]
def fnt(n):return ImageFont.truetype(str(FONT),n)
def text(d,xy,s,n=22,fill='#152536'):d.text(xy,str(s),font=fnt(n),fill=fill)
def wrapped(d,xy,s,width,n=21,line=29):
 words=str(s).split();out=[];cur=''
 for w in words:
  z=(cur+' '+w).strip()
  if d.textbbox((0,0),z,font=fnt(n))[2]>width and cur:out.append(cur);cur=w
  else:cur=z
 if cur:out.append(cur)
 for j,z in enumerate(out):text(d,(xy[0],xy[1]+j*line),z,n)
 return len(out)*line
def fit(im,wh):
 a=im.copy();a.thumbnail(wh,Image.Resampling.LANCZOS);return a
def marker(d,xy,color,kind):
 x,y=map(float,xy)
 if kind=='q':
  d.line((x-13,y,x+13,y),fill=color,width=3);d.line((x,y-13,x,y+13),fill=color,width=3)
 else:d.ellipse((x-9,y-9,x+9,y+9),outline=color,width=3)
def ref_crop(rgb,ref,side=224):
 cx,cy=np.floor(np.asarray(ref)+.5).astype(int);left=cx-side//2;top=cy-side//2
 out=np.zeros((side,side,3),np.uint8);valid=np.zeros((side,side),bool)
 h,w=rgb.shape[:2];x0=max(0,left);y0=max(0,top);x1=min(w,left+side);y1=min(h,top+side)
 if x1>x0 and y1>y0:
  out[y0-top:y1-top,x0-left:x1-left]=rgb[y0:y1,x0:x1]
  valid[y0-top:y1-top,x0-left:x1-left]=True
 return out,valid,[left,top,left+side,top+side]
def rawput(im,xy,a,scale):
 p=Image.fromarray(np.ascontiguousarray(a))
 im.paste(p.resize((p.width*scale,p.height*scale),Image.Resampling.NEAREST),xy)
def card(c,rgb,crop,valid):
 im=Image.new('RGB',CARD_SIZE,'#f7f9fb');d=ImageDraw.Draw(im);k=c['native_slot'];q=c['query_xy'];ref=c['reference_xy']
 text(d,(24,14),f"{c['case_id']} | native{k} {c['native_name']} | primary {c['primary_group']}",30)
 text(d,(24,54),f"Expected part: {c['nominal_expected_part']} | selected scene-GT list index {c['selected_gt_instance_index']} (obj1 Euro WOOD)",24)
 text(d,(24,90),f"q=({q[0]:.3f}, {q[1]:.3f}); L={c['L']:.3f} | source {c['source_id']} | nominal ref=({ref[0]:.3f}, {ref[1]:.3f})",21)
 text(d,(24,122),'Nominal CAD/pose is uncertain: no hard pixel bound. Markers and wireframe are display diagnostics, never visibility gold.',22,fill='#964200')
 text(d,(24,164),'ORIGINAL RGB: q magenta cross; nominal ref cyan circle',24)
 text(d,(990,164),'SELECTED INSTANCE: nominal wireframe/native0..7 only',24)
 a=Image.fromarray(rgb).copy();ad=ImageDraw.Draw(a);marker(ad,q,'#ff00ff','q');marker(ad,ref,'#00ffff','r')
 a=fit(a,(906,540));im.paste(a,(24,205))
 b=Image.fromarray(rgb).copy();bd=ImageDraw.Draw(b);points=c['dynamic_projected8']
 for u,w in EDGES:bd.line(tuple(points[u]+points[w]),fill='#ffe34d',width=2)
 for j,xy in enumerate(points):
  marker(bd,xy,'#00ffff' if j==k else '#ffe34d','r')
  text(bd,(xy[0]+10,xy[1]-25),str(j),27,fill='#00ffff' if j==k else '#ffe34d')
 marker(bd,q,'#ff00ff','q');b=fit(b,(906,540));im.paste(b,(990,205))
 text(d,(24,755),'RAW saved q RGB96 / 3x',22)
 text(d,(350,755),'DISPLAY COPY / q center',22)
 text(d,(676,755),'RAW central32 / 8x',22)
 text(d,(966,755),'VALID96 / 2x',22)
 text(d,(1190,755),'NOMINAL REF DIAGNOSTIC ONLY / 224px window / 2x',21)
 rawput(im,(24,794),crop,3)
 overlay=Image.fromarray(crop).resize((288,288),Image.Resampling.NEAREST);od=ImageDraw.Draw(overlay)
 marker(od,(143.5,143.5),'#00ffff','q');od.rectangle((141,141,146,146),outline='#ff00ff',width=1);im.paste(overlay,(350,794))
 rawput(im,(676,794),crop[32:64,32:64],8)
 rawput(im,(966,794),np.repeat((valid.astype(np.uint8)*255)[...,None],3,axis=2),2)
 text(d,(966,1002),'white=valid',18);text(d,(966,1028),'black=padding',18)
 v4=valid[47:49,47:49];text(d,(966,1062),f'center4: {int(v4.sum())}/4',20)
 text(d,(966,1095),str(v4.astype(int).tolist()),18)
 rc,rv,box=ref_crop(rgb,ref);rawput(im,(1190,794),rc,2)
 rd=ImageDraw.Draw(im);marker(rd,(1190+224,794+224),'#00ffff','r')
 wrapped(d,(1660,800),'Diagnostic view is centered on rounded nominal reference. It is not the original query crop, model input, corrected q, or exact physical native point.',224,19,27)
 wrapped(d,(1660,1060),'Black padding is invalid evidence, not natural background.',224,19,27)
 text(d,(24,1132),f"Raw center2x2 [47:49] / 40x; valid {int(v4.sum())}/4",20);rawput(im,(24,1167),crop[47:49,47:49],40)
 wrapped(d,(136,1167),'Original RGB96 and central32 remain untouched. Overlays occur on separate display copies. Full-frame and reference-window zooms provide structural context.',1000,21,29)
 text(d,(24,1290),'Assess intended selected-instance part and relation to unchanged q. Separate visible / external / self / boundary-or-gap / UNKNOWN.',21)
 text(d,(24,1328),'Assistant weak structural witness only; no old labels or model scores displayed. No native/point-quality gold or training admission.',21)
 text(d,(24,1364),'Rear is not automatically self-hidden. Same wood or target surface at a ray does not establish native-part correspondence.',20)
 return im,rc,rv,box,overlay

def geometry_only_cases(scope):
 geomrows=jsonl(scope['reviewer_input_geometry_only'])
 fb=bind(V159/'native_mask_support_audit'/'SELECTED88_TRAIN_FRAME_GEOMETRY_AND_MASK_BINDINGS.json')
 frames=readb(fb)['frames'];byfi={r['frame_index']:r for r in frames}
 parityb=bind(V158/'SOURCE_BINDINGS_AND_ARRAY_PARITY.json');parity=readb(parityb)
 factsb=parity['inputfacts'];facts={}
 for line in verify(factsb).read_text().splitlines():
  r=json.loads(line)
  if r['split']=='SOURCE_DEVELOPMENT':facts[r['global_index']]=r
 dictionary={r['native_slot']:r for r in jsonl(scope['native_dictionary'])}
 frameinputs={r['frame_index']:r for r in parity['selected_TRAIN_frames']}
 cases=[]
 for r in geomrows:
  assert r['split']=='SOURCE_DEVELOPMENT'
  fi=r['frame_index'];k=r['native_slot'];f=facts[r['global_index']];fr=byfi[fi]
  instances=fr['geometry']['instances'];g=next(g for g in instances if g['gt_instance_index']==r['selected_gt_instance_index'])
  assert g['split']=='SOURCE_DEVELOPMENT' and g['obj_id']==1
  assert all(r[z]==f[z] for z in ('global_index','query_id','frame_index','source_id','split','native_slot','query_xy','L'))
  assert f['feature_constructed'] and f['native_continuous_valid']
  assert r['reference_xy']==g['dynamic_projected8'][k]
  perm=g['dynamic_to_fixed_permutation'];assert sorted(perm)==list(range(8))
  assert np.array_equal(np.asarray(g['dynamic_projected8']),np.asarray(g['canonical_fixed_projected8'])[perm])
  projected=project(g['canonical_fixed_corners_mm'],g['provided_R_m2c'],g['provided_t_m2c_mm'],g['provided_K'])
  assert np.allclose(projected,g['canonical_fixed_projected8'],rtol=0,atol=1e-8)
  c={z:r[z] for z in ('case_id','global_index','query_id','frame_index','source_id','split','native_slot','native_name','query_xy','L','selected_gt_instance_index','reference_xy','reference_model_XYZ_mm')}
  c.update(nominal_expected_part=dictionary[k]['nominal_expected_part'],raw_hw=g['raw_hw'],obj_id=1,
   dynamic_projected8=g['dynamic_projected8'],dynamic_to_fixed_permutation=perm,
   canonical_fixed_projected8=g['canonical_fixed_projected8'],reference_is_nominal_not_pixel_gold=True,
   face_runner_margin=g['front_runner_margin'],front_top_imageX_separation_px=g['front_top_imageX_separation_px'],
   reference_continuous_inframe=r['reference_continuous_inframe'] if 'reference_continuous_inframe' in r else bool(0<=r['reference_xy'][0]<g['raw_hw'][1] and 0<=r['reference_xy'][1]<g['raw_hw'][0]),
   original_input_available=True,nearest_valid=f['nearest_valid'],local_central4_available=f['local_central4_available'],context_available=f['context_available'],
   RGB96_HWC_sha256=f['RGB96_HWC_sha256'],RGB96_CHW_sha256=f['RGB96_CHW_sha256'],valid96_sha256=f['valid96_sha256'])
  cases.append(c)
 cases.sort(key=lambda r:r['global_index']);assert len(cases)==390 and len({r['case_id'] for r in cases})==390
 assert len({r['frame_index'] for r in cases})==88
 groups,cross=assignments(cases)
 for c in cases:c['primary_group']=next(g for g,ids in groups.items() if c['case_id'] in ids);c['cross40']=c['case_id'] in cross
 cleanframes=[]
 for fi in sorted({r['frame_index'] for r in cases}):
  x=frameinputs[fi];assert x['split']=='SOURCE_DEVELOPMENT'
  assert Path(x['raw_RGB']['path']).parent==S/'full_source_v1'/'canonical_RGB'
  assert Path(x['crop_packet']['path'])==S/'full230_native_DINO_v1'/f'FRAME_{fi:04d}_FACTUAL96_INPUT_ONLY.npz'
  cleanframes.append({z:x[z] for z in ('frame_index','source_id','split','raw_RGB','decoded_RGB_sha256','raw_hw','crop_packet')})
 return cases,groups,cross,cleanframes,{'geometry88':fb,'input_parity':parityb,'inputfacts':factsb}

def generated_checks():
 checks={}
 corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
 K=np.array([[100,0,50],[0,100,40],[0,0,1]],float)
 a=project(corners,np.eye(3),[0,0,10],K)
 checks['invented_projection_known_front_points']=bool(np.array_equal(a[:4],[[50,40],[60,40],[60,50],[50,50]]))
 perm=[4,5,6,7,0,1,2,3];checks['dynamic_mapping_exact_one_application']=np.array_equal(a[perm][np.argsort(perm)],a)
 checks['wireframe_12_edges_degree3']=len(set(EDGES))==12 and all(sum(k in e for e in EDGES)==3 for k in range(8))
 rgb=np.zeros((720,1280,3),np.uint8);rgb[:,:,0]=np.arange(1280,dtype=np.uint16)%256;rgb[:,:,1]=np.arange(720,dtype=np.uint16)[:,None]%256
 crop=np.arange(96*96*3,dtype=np.uint16).reshape(96,96,3).astype(np.uint8);valid=np.ones((96,96),bool);valid[47,47]=False
 original=ah(crop);rc,rv,box=ref_crop(rgb,[0.2,.2]);checks['diagnostic_rounding_padding_count']=box==[-112,-112,112,112] and rv.sum()==112*112 and not rv[0,0]
 c=dict(case_id='INVENTED',native_slot=0,native_name='front_top_left',primary_group='C',nominal_expected_part='outer deck-board end corner',selected_gt_instance_index=0,query_xy=[640,360],L=200,source_id='INVENTED',reference_xy=[600,350],dynamic_projected8=[[400,300],[800,300],[800,500],[400,500],[500,250],[900,250],[900,450],[500,450]])
 im,*_=card(c,rgb,crop,valid);checks['raw_not_mutated_by_overlays']=ah(crop)==original
 checks['card_exact_1920x1400']=im.size==CARD_SIZE
 inv=O/'invented_checks';inv.mkdir(exist_ok=True);im.save(inv/'INVENTED_CARD.png')
 sheet=Image.new('RGB',(1920,2800),'white');sheet.paste(im,(0,0));sheet.paste(im,(0,1400));sheet.save(inv/'INVENTED_CONTACT.png')
 checks['two_card_contact_exact']=sheet.size==(1920,2800)
 checks['font_footer_within_card']=ImageDraw.Draw(im).textbbox((24,1364),'Rear is not automatically self-hidden. Same wood or target surface at a ray does not establish native-part correspondence.',font=fnt(20))[2]<1920
 rcpath=inv/'INVENTED_RAW96.png';Image.fromarray(crop).save(rcpath);checks['invented_raw_png_exact']=np.array_equal(np.asarray(Image.open(rcpath)),crop)
 assert all(checks.values()),checks
 return checks

def prepare():
 assert not (O/'PREPARATION_COMPLETE.json').exists()
 scope=read(SCOPE);assert bind(SCOPE)['sha256']=='875f28ae4633d83f80b598a6bc0812ae312131ea22d740d7e15f52b88f44481a'
 cases,groups,cross,frames,metab=geometry_only_cases(scope)
 assert all(len(groups[g])==130 for g in 'ABC') and len(cross)==len(set(cross))==40
 checks=generated_checks()
 checks['metadata_all390_88frames_ABC130']=len(cases)==390 and len(frames)==88 and all(len(groups[g])==130 for g in 'ABC')
 checks['cross40_native6_retained']=sum(c['case_id'] in cross and c['native_slot']==6 for c in cases)==2
 checks['strict_blinding_no_role_owner_score_keys']=not any(any(t in k.lower() for t in ('role_state','role_target','owner','score','prediction','reason','self_visible','semantic_state')) for c in cases for k in c)
 assert all(checks.values())
 write(O/'GENERATED_MEANINGFUL_CHECKS.json',dict(completed=True,PASS=True,checks=checks,source_pixel_or_crop_NPZ_body_reads=0,invented_card=bind(O/'invented_checks'/'INVENTED_CARD.png'),invented_contact=bind(O/'invented_checks'/'INVENTED_CONTACT.png')))
 write(O/'BLINDED_INPUT390_AND_FRAME88_BINDINGS.json',dict(schema='V163_INPUT_GEOMETRY_ONLY',completed=True,scope=bind(SCOPE),metadata_bindings=metab,geometry_only390=scope['reviewer_input_geometry_only'],native_dictionary=scope['native_dictionary'],full_mapping_lineage_only_not_parsed_by_this_code=scope['source_mapping'],cases=cases,frames=frames))
 write(O/'CASE_ASSIGNMENT_AND_CROSS40_BEFORE_PIXELS.json',dict(completed=True,source_pixels_or_NPZ_hashed_or_read=0,salt=SALT,primary_rule='ascending SHA256(salt+case_id), tie case_id; roundrobin A/B/C',cross_rule='per native first5 ascending same digest (native6 all2), then globally first remaining3 to40; allABC/root independently review same40',primary=groups,cross40=cross,cross40_native_counts={str(k):sum(c['case_id'] in cross and c['native_slot']==k for c in cases) for k in range(8)}))
 protocol=dict(schema='V163_RENDER_PROTOCOL_BEFORE_SOURCE_PIXELS',completed=True,code=bind(__file__),scope=bind(SCOPE),inputs=bind(O/'BLINDED_INPUT390_AND_FRAME88_BINDINGS.json'),assignment=bind(O/'CASE_ASSIGNMENT_AND_CROSS40_BEFORE_PIXELS.json'),generated_checks=bind(O/'GENERATED_MEANINGFUL_CHECKS.json'),source_pixel_or_NPZ_hash_or_read_in_preparation=0,
  allowed_pixel_assets='Only88 sealed TRAIN canonical_RGB NPY and88 exact FACTUAL96_INPUT_ONLY crop NPZ; no source masks/depth/fullfeatures/weights/held images.',
  card_size=list(CARD_SIZE),contact_size=[1920,2800],max_cards_per_contact=2,
  raw_policy='Raw RGB96 unchanged; central32 literal[32:64]; center2x2 literal[47:49]; fullvalid96/center4 bits/count preserved. Separate display copies only.',
  nominal_policy='Selected instance immutable dynamic8/perm from geometry-only frame metadata. Wireframe yellow/selectednativecyan only, no convex-self proxy or semantic states. No nearest instance/permutation optimization.',
  reference_diagnostic='224x224 originalRGB integer window centered floor(nominalref+.5), zero invalid padding, nearest2x; separately flagged diagnostic only, never input/q correction.',
  blinding='Explicit field whitelist; no original native D/E, old owner labels/votes/reasons or any model outputs. Native role dictionary and uncertain pose geometry are input diagnostics.',
  reviewer_authority='Reviewers have historical task context; observations are assistant weak witnesses, not naive/human/native/coordinate gold.',
  preservation='All390/88, original q/L/native/selectedinstance/splits/availability/edge flags retained; no labels/admission/encoder/model/fit/PnP.',
  outputs=['ALL390_BLINDED_EVIDENCE.jsonl','REVIEW_INPUT_GROUP_A.json','REVIEW_INPUT_GROUP_B.json','REVIEW_INPUT_GROUP_C.json','BLIND_CROSS40_ALL_REVIEWERS.json','ALL390_READOUT_INVENTORY.json','COMPLETE.json'],
  entry_required=True,failure_policy='Single attempt; preserve STARTED/FAILED, retain any partial artifacts. No blind retry/replacement.')
 write(O/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json',protocol)
 write(O/'ACTUAL_RENDER_ENTRY_TEMPLATE.json',dict(schema='V163_ROOT_ACTUAL_SOURCE_EVIDENCE_ENTRY',approved_actual_source_evidence_render=False,source_only=True,code=bind(__file__),protocol=bind(O/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json'),inputs=bind(O/'BLINDED_INPUT390_AND_FRAME88_BINDINGS.json'),assignment=bind(O/'CASE_ASSIGNMENT_AND_CROSS40_BEFORE_PIXELS.json'),no_encoder_model_fit_PnP_private_current_V112=True))
 write(O/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(O/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json'),inputs=bind(O/'BLINDED_INPUT390_AND_FRAME88_BINDINGS.json'),assignment=bind(O/'CASE_ASSIGNMENT_AND_CROSS40_BEFORE_PIXELS.json'),generated_checks=bind(O/'GENERATED_MEANINGFUL_CHECKS.json'),entry_template=bind(O/'ACTUAL_RENDER_ENTRY_TEMPLATE.json'),new_source_pixel_or_crop_NPZ_hash_or_read=0,actual_render_authorized=False))
 print(json.dumps(dict(completed=True,preparation=bind(O/'PREPARATION_COMPLETE.json'))),flush=True)

def render(entrypath):
 entry=read(entrypath);assert entry['approved_actual_source_evidence_render'] is True and entry['source_only'] is True
 assert entry['no_encoder_model_fit_PnP_private_current_V112'] is True
 protocol=readb(entry['protocol']);assert entry['code']==protocol['code']==bind(__file__)
 assert entry['inputs']==protocol['inputs'] and entry['assignment']==protocol['assignment']
 data=readb(protocol['inputs']);assignment=readb(protocol['assignment']);cases=data['cases']
 assert not (OUT/'STARTED.json').exists(),'Previous render attempt exists; do not blind retry.'
 write(OUT/'STARTED.json',dict(completed=False,UTC=utc(),entry=bind(entrypath),code=bind(__file__),protocol=entry['protocol'],automatic_retry=False))
 try:
  cache={};source_reads=[]
  for f in data['frames']:
   fi=f['frame_index'];assert f['split']=='SOURCE_DEVELOPMENT'
   rawpath=verify(f['raw_RGB']);assert rawpath==S/'full_source_v1'/'canonical_RGB'/f'RGB_{fi:03d}.npy'
   rgb=np.load(rawpath,allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==f['raw_hw']+[3] and ah(rgb)==f['decoded_RGB_sha256']
   cp=verify(f['crop_packet']);assert cp==S/'full230_native_DINO_v1'/f'FRAME_{fi:04d}_FACTUAL96_INPUT_ONLY.npz'
   with np.load(cp,allow_pickle=False) as z:
    assert set(z.files)=={'rgb96','valid96','global_index','crop_constructed'}
    packet={k:z[k] for k in z.files}
   assert packet['rgb96'].shape==(8,96,96,3) and packet['rgb96'].dtype==np.uint8 and packet['valid96'].shape==(8,96,96)
   assert np.array_equal(packet['global_index'],np.arange(fi*8,fi*8+8))
   cache[fi]=(rgb,packet);source_reads.append(f)
  records=[]
  for c in cases:
   rgb,packet=cache[c['frame_index']];k=c['native_slot'];crop=packet['rgb96'][k];valid=packet['valid96'][k]
   assert packet['crop_constructed'][k]
   assert ah(crop)==c['RGB96_HWC_sha256'] and ah(crop.transpose(2,0,1))==c['RGB96_CHW_sha256'] and ah(valid)==c['valid96_sha256']
   im,rc,rv,box,overlay=card(c,rgb,crop,valid);dest=OUT/'cases'/c['case_id'];dest.mkdir(parents=True)
   assets={'card':('CARD.png',im),'raw96':('RAW_RGB96.png',Image.fromarray(crop)),'central32':('RAW_CENTRAL32.png',Image.fromarray(crop[32:64,32:64])),'center2x2':('RAW_CENTER2X2.png',Image.fromarray(crop[47:49,47:49])),'valid96':('RAW_VALID96.png',Image.fromarray(valid.astype(np.uint8)*255)),'display96':('DISPLAY96_CENTER_OVERLAY.png',overlay),'nominal_ref_diagnostic':('NOMINAL_REFERENCE_DIAGNOSTIC.png',Image.fromarray(rc))}
   images={}
   for key,(name,image) in assets.items():image.save(dest/name);images[key]=bind(dest/name)
   for key,array in [('raw96',crop),('central32',crop[32:64,32:64]),('center2x2',crop[47:49,47:49]),('valid96',valid.astype(np.uint8)*255),('nominal_ref_diagnostic',rc)]:assert np.array_equal(np.asarray(Image.open(images[key]['path'])),array)
   rec=dict(c,images=images,center4_valid_bits=valid[47:49,47:49].tolist(),center4_valid_count=int(valid[47:49,47:49].sum()),center4_allvalid=bool(valid[47:49,47:49].all()),valid96_count=int(valid.sum()),crop_edge_or_padding=bool(not valid.all()),central32_array_sha256=ah(crop[32:64,32:64]),center2x2_array_sha256=ah(crop[47:49,47:49]),nominal_reference_diagnostic_raw_xyxy=box,nominal_reference_diagnostic_valid_count=int(rv.sum()),nominal_reference_diagnostic_array_sha256=ah(rc))
   records.append(rec)
  assert len(records)==390;lookup={r['case_id']:r for r in records};contacts=[]
  for g in 'ABC':
   ids=assignment['primary'][g];assert len(ids)==130
   for j in range(0,130,2):
    chunk=ids[j:j+2];im=Image.new('RGB',(1920,2800),'white')
    for t,cid in enumerate(chunk):im.paste(Image.open(lookup[cid]['images']['card']['path']),(0,t*1400))
    path=OUT/'contacts'/f'GROUP_{g}_PAGE_{j//2+1:03d}.png';path.parent.mkdir(exist_ok=True);im.save(path)
    contacts.append(dict(group=g,page=j//2+1,case_ids=chunk,image=bind(path)))
   write(OUT/f'REVIEW_INPUT_GROUP_{g}.json',dict(schema='V163_BLINDED_NATIVE_PART_INPUT',completed=True,group=g,count=130,cases=[lookup[cid] for cid in ids],contacts=[r for r in contacts if r['group']==g],cross40_case_ids=assignment['cross40'],historically_contextualized_assistant_weak=True,old_labels_or_outputs_exposed=False))
  crossids=assignment['cross40'];crosscontacts=[]
  for j in range(0,40,2):
   chunk=crossids[j:j+2];im=Image.new('RGB',(1920,2800),'white')
   for t,cid in enumerate(chunk):im.paste(Image.open(lookup[cid]['images']['card']['path']),(0,t*1400))
   path=OUT/'contacts'/f'CROSS40_PAGE_{j//2+1:03d}.png';im.save(path);crosscontacts.append(dict(page=j//2+1,case_ids=chunk,image=bind(path)))
  write(OUT/'BLIND_CROSS40_ALL_REVIEWERS.json',dict(completed=True,count=40,independent_reviewers=['A','B','C','ROOT_BEFORE_AGENT_VOTES'],cases=[lookup[cid] for cid in crossids],contacts=crosscontacts,other_reviewer_votes_exposed=False))
  writejl(OUT/'ALL390_BLINDED_EVIDENCE.jsonl',records)
  write(OUT/'ALL390_READOUT_INVENTORY.json',dict(completed=True,cases=390,frames=88,primary_groups={g:130 for g in 'ABC'},cross40=40,cards=390,primary_contacts=195,cross_contacts=20,max_cards_per_contact=2,source_reads=source_reads,raw_arrays_and_saved_PNG_exact=True,unchanged_qL_native_selectedinstance_split=True,geometry_source=data['geometry_only390'],no_oldlabels_owner_votes_modeloutputs=True))
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,UTC=utc(),entry=bind(entrypath),code=bind(__file__),protocol=entry['protocol'],inputs=protocol['inputs'],assignment=protocol['assignment'],inventory=bind(OUT/'ALL390_READOUT_INVENTORY.json'),all390=bind(OUT/'ALL390_BLINDED_EVIDENCE.jsonl'),review_groups={g:bind(OUT/f'REVIEW_INPUT_GROUP_{g}.json') for g in 'ABC'},cross40=bind(OUT/'BLIND_CROSS40_ALL_REVIEWERS.json'),cases=390,frames=88,individual_cards=390,groups={g:130 for g in 'ABC'},sourceRGB_and_crop_packets=176,source_masks_depth_fullfeatures_weights_or_models=0,held_private_current_V112_access=0,new_native_quality_owner_labels=0,automatic_training_admission=False))
  print(json.dumps(dict(completed=True,result=bind(OUT/'COMPLETE.json'))),flush=True)
 except Exception as e:
  write(OUT/'FAILED.json',dict(completed=False,UTC=utc(),error=str(e),traceback=traceback.format_exc(),actual_render_failure=True,automatic_retry=False));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','render']);parser.add_argument('--root-entry');args=parser.parse_args()
 if args.mode=='prepare':prepare()
 else:
  assert args.root_entry;render(args.root_entry)
