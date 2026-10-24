"""V205 source TRAIN input-only previews. No teacher, old weak label, GT or model."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,html,json,os,stat,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_PUBLIC_TRAIN_NATIVE_WEAK_OBSERVATION_SCOPE.json';SCOPE_SHA='78b8eb2cd37e0aa4cd7984269c0704940f153ac43830a6ae39daf6798a21fc53'
A=P/'query_visibility_local_support_raw_head_20261024_v181';D=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
B=P/'query_visibility_source_train_intervention_query_alignment_20261025_v204/public_source_coverage_b'
BINDINGS={'rows':{'path':str(A/'source_raw96_cache_v1/PUBLIC_TRAIN1480_ROWS.jsonl'),'sha256':'08754bf88a98fcdc1a56cd0d84d6dd40804ff9de2f1ad7693a9e5bdd1fa31ff0'},'roles':{'path':str(D/'source_component_split_v1/PUBLIC_TRAIN1480_SPLIT_ROWS.jsonl'),'sha256':'b04d5d7b1779b3df346a3fe6db26074b7376733317212471eef838439d72e0cc'},'frames':{'path':str(A/'sam_input_preparation_c/PUBLIC185_FRAMES_METADATA_ONLY.jsonl'),'sha256':'0cb2296e3dd786b294416a25d1c86e5af1331a43070270ae8f2828e245e74ca7'},'source_readiness':{'path':str(A/'sam_input_preparation_c/PUBLIC_INPUT_READINESS.json'),'sha256':'PENDING_PREP_BIND'},'existing_direct_attempts':{'path':str(B/'KNOWN_DIRECT_TRAIN_ATTEMPTS.csv'),'sha256':'515cc6d04d4fdada3437a37368340f57b6519039e5629cd7c6e0d7a85ed6b72a'}}
PREFIXES=('val/000000','val/000002','val/000005','val/000006');QUOTA=(2,2,1,1)
PROTO=H/'PROTOCOL_BEFORE_SELECTED_SOURCE_RGB_OR_FRAME_NPZ_BODIES.json';OUT=V/'fixed6_public_TRAIN_input_previews_v1';FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLORS=((0,255,255),(255,60,60),(50,255,50),(255,160,0),(255,0,255),(255,255,0),(70,150,255),(230,230,230))
def need(v,m):
 if not bool(v):raise AssertionError(m)
def safe(p):
 p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold(),'Forbidden/outside path rejected before filesystem');return p
def opened(p,write=False):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for x in p.parts[1:-1]:n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file/no symlink');return os.fdopen(n,'wb'if write else'rb')
def sha(p):
 d=hashlib.sha256()
 with opened(p)as f:
  for z in iter(lambda:f.read(1<<20),b''):d.update(z)
 return d.hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=sha(p))
def checked(b):need(bind(b['path'])==b,'Bound source metadata/body SHA drift');return Path(b['path'])
def read(p):
 with opened(p)as f:return json.load(f)
def lines(p):
 with opened(p)as f:return[json.loads(z)for z in f if z.strip()]
def write(p,d):
 with opened(p,True)as f:f.write((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def text(p,s):
 with opened(p,True)as f:f.write(s.encode())
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sameq(a,b):return np.array_equal(np.asarray([np.nan,np.nan]if a is None else a,np.float64),np.asarray([np.nan,np.nan]if b is None else b,np.float64),equal_nan=True)
def prefix(source_id):
 p=source_id.split(':')[0];need(p in PREFIXES,'Only existing four SOURCE prefixes');return p

def match_possible(slots,candidates,used):
 # Six-slot bipartite component matching, not pixel/outcome optimization.
 match={}
 def add(slot,seen):
  for comp in sorted({r['component_id']for r in candidates if r['prefix']==slot and r['component_id']not in used}):
   if comp in seen:continue
   seen.add(comp)
   if comp not in match or add(match[comp],seen):match[comp]=slot;return True
  return False
 return all(add(slot,set())for slot in slots)
def select_six(candidates):
 slots=[p for p,q in zip(PREFIXES,QUOTA)for _ in range(q)];unique_possible=match_possible(slots,candidates,set());selected=[];used=set()
 for i,slot in enumerate(slots):
  rr=sorted([r for r in candidates if r['prefix']==slot and r['frame_key']not in{x['frame_key']for x in selected}],key=lambda r:(r['stable_SHA256'],r['frame_key']))
  if unique_possible:rr=[r for r in rr if r['component_id']not in used and match_possible(slots[i+1:],candidates,used|{r['component_id']})]
  else:rr=([r for r in rr if r['component_id']not in used]or rr)
  need(rr,'Requested prefix slots must exist');selected.append(rr[0]);used.add(rr[0]['component_id'])
 need(len(selected)==6 and len({r['frame_key']for r in selected})==6,'Exactly six distinct original frames')
 return selected,dict(all6_distinct_components_possible=unique_possible,selected_distinct_components=len(used),selection_rule='Lexicographic SHA256(original source_id UTF8) then source_id within sortedprefix quotas2/2/1/1; prefer six unique graphcomponents using bounded matching feasibility; no pixels/scores/weaklabels used.')

def input_rows(rows,roles,frames):
 rm={r['row_id']:r for r in roles};need(len(rm)==len(roles)==len(rows)==1480,'Exact185x8 public source input/role census');fm={r['original_frame_index']:r for r in frames};need(len(fm)==185,'SOURCE185 only, not publicheld45');groups=defaultdict(list)
 for r in rows:
  z=rm[r['row_id']];need(z['role']in('TRAIN','HOLDOUT'),'Immutable source roles')
  if z['role']!='TRAIN':continue
  need(r['original_global_index']==z['original_global_index']and r['native_slot']==z['native_slot']and r['original_query_available']==z['original_query_available']and r['L']==z['L']and sameq(r['query_xy'],z['query_xy']),'Input META6 rolejoin exact')
  o=r['original'];need(o['split']=='SOURCE_DEVELOPMENT'and o['native_slot']==r['native_slot']and o['query_id']==r['row_id'],'Original source input IDs only')
  # Explicit whitelist: never expose SAM, teacher/owner, target or score fields.
  groups[o['frame_index']].append(dict(row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],query_xy=r['query_xy'],L=r['L'],original_query_available=r['original_query_available'],raw_packet=r['raw_packet'],raw_row=r['raw_row'],RGB96_sha256=r['actual_raw_RGB96_sha256'],valid96_sha256=r['actual_raw_valid96_sha256'],source_id=o['source_id'],frame_id=o['frame_id'],original_frame_index=o['frame_index'],component_id=z['component_id'],role='TRAIN',original_flags={k:o.get(k)for k in ('feature_constructed','native_continuous_valid','nearest_valid','Q_inside','local_central4_available','context_available')}))
 out=[]
 for fi,rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:r['native_slot']);f=fm[fi];need(len(rr)==8 and[r['native_slot']for r in rr]==list(range(8))and len({r['component_id']for r in rr})==1 and len({r['source_id']for r in rr})==1,'Whole coherent TRAIN originalnative8/frame/component')
  need(f['split']=='SOURCE_DEVELOPMENT'and f['source_id']==rr[0]['source_id']and f['frame_id']==rr[0]['frame_id'],'SOURCE185 fullframe identity')
  for k,r in enumerate(rr):need(r['raw_row']==k and r['raw_packet']==f['saved_rgb96_packet']and sameq(r['query_xy'],f['original_initial_points8'][k])and r['L']==f['original_L8'][k]and r['original_global_index']==f['original_global_indices'][k]and r['row_id']==f['original_query_ids'][k],'Frame originalq8/L8/RAW96 packet exact, no new sampling')
  source_id=rr[0]['source_id'];hint=f.get('original_selected_detection_bbox_xyxy') # No context/q8 bbox promoted to selected-instance box.
  out.append(dict(frame_key=source_id,source_id=source_id,original_frame_index=fi,frame_id=f['frame_id'],prefix=prefix(source_id),component_id=rr[0]['component_id'],role='TRAIN',stable_SHA256=hashlib.sha256(source_id.encode()).hexdigest(),image=f['image'],original_encoded_RGB_provenance=f['original_RGB'],original_RGB_decoder=f['original_RGB_decoder'],selected_detection_bbox_xyxy=hint,selected_detection_bbox_status='EXPLICIT_DEPLOYED_HINT'if hint is not None else'NOT_PRESENT_IN_BOUND_INPUT_ONLY_METADATA_NO_FALLBACK',native8=rr))
 need(len(out)==155 and sum(not r['original_query_available']for f in out for r in f['native8'])==161 and len({r['component_id']for r in out})==103,'Known155TRAIN/1240/U161/103 graphcomponents')
 return out

def metadata():
 bindings=dict(BINDINGS);bindings['source_readiness']=bind(bindings['source_readiness']['path']);ready=read(checked(bindings['source_readiness']));need(ready['completed']and ready['PASS']and ready['frames_manifest']==bindings['frames'],'Pinned SOURCE185 readiness')
 rows=lines(checked(bindings['rows']));roles=lines(checked(bindings['roles']));frames=lines(checked(bindings['frames']));alltrain=input_rows(rows,roles,frames)
 with opened(checked(bindings['existing_direct_attempts']))as f:direct=list(csv.DictReader(z.decode()for z in f))
 excluded=sorted({int(r['source_frame_index'])for r in direct if r['domain']=='REAL_SEMANTIC_END_WEAK'});need(len(excluded)==12,'Existing direct TRAIN12 frames fixed exclusion')
 candidates=[r for r in alltrain if r['original_frame_index']not in excluded];selected,method=select_six(candidates)
 return selected,dict(source_frames_metadata_read=185,source_rows_metadata_read=1480,original_TRAIN_frames=155,original_TRAIN_rows=1240,original_TRAIN_inputU=161,original_TRAIN_graph_components=103,excluded_existing_direct12_frame_indices=excluded,candidate_frames_after_exclusion=len(candidates),candidate_prefixes=dict(Counter(r['prefix']for r in candidates)),selected_prefixes=dict(Counter(r['prefix']for r in selected)),selected_original_U=sum(not r['original_query_available']for f in selected for r in f['native8']),**method),bindings

def marker(q,available,width,height):
 if q is None:return None
 a=np.asarray(q,np.float64)
 if a.shape!=(2,)or not np.isfinite(a).all()or not(0<=a[0]<width and 0<=a[1]<height):return None
 return float(a[0]),float(a[1]),bool(available)
def preview_frame(rgb,frame):
 im=Image.fromarray(rgb).copy();d=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,20)
 for r in frame['native8']:
  m=marker(r['query_xy'],r['original_query_available'],rgb.shape[1],rgb.shape[0])
  if m is None:continue
  x,y,av=m;color=COLORS[r['native_slot']];d.line([(x-9,y),(x+9,y)],fill=color,width=2);d.line([(x,y-9),(x,y+9)],fill=color,width=2);d.text((x+7,y-24),str(r['native_slot'])+(' U'if not av else''),font=font,fill=color)
 hint=frame['selected_detection_bbox_xyxy']
 if hint is not None:
  need(len(hint)==4 and np.isfinite(hint).all(),'Recorded deployed detectionbox finite only');d.rectangle(hint,outline='white',width=2)
 return im

def contact(frame,raw):
 im=Image.new('RGB',(1600,1020),'white');d=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n);d.text((18,10),'V205 TRAIN INPUT ONLY | '+frame['source_id'],font=font(25),fill='black');d.text((18,46),'Original native0..7, q/L and U. No GT/oldweak/teacher/SAM/classifier outcome. Cyan cross is DISPLAY ONLY at47.5.',font=font(20),fill='black')
 for r in frame['native8']:
  k=r['native_slot'];x=400*(k%4);y=86+450*(k//4);av=r['original_query_available'];q=r['query_xy'];d.text((x+8,y),'native %d | input %s'%(k,'AVAILABLE'if av else'U'),font=font(21),fill='black');d.text((x+8,y+28),'q='+str(q)+' L=%.6g'%r['L']if r['L']is not None else'q='+str(q)+' L=None',font=font(14),fill='black')
  clean=Image.fromarray(raw['rgb96'][k]);zoom=clean.resize((288,288),Image.Resampling.NEAREST);zd=ImageDraw.Draw(zoom);v=(47.5+.5)*3-.5;zd.line([(v-10,v),(v+10,v)],fill='cyan',width=2);zd.line([(v,v-10),(v,v+10)],fill='cyan',width=2);im.paste(zoom,(x+8,y+62));im.paste(clean,(x+300,y+62));im.paste(Image.fromarray(raw['valid96'][k].astype(np.uint8)*255).convert('RGB'),(x+300,y+190));d.text((x+300,y+162),'clean96',font=font(15),fill='black');d.text((x+300,y+290),'valid96',font=font(15),fill='black');d.text((x+8,y+365),'constructed='+str(bool(raw['crop_constructed'][k]))+'; black valid=padding/invalid',font=font(16),fill='black')
 d.text((18,990),'Unavailable/off-frame native is preserved, never moved or converted into externalE. Full-image context is separate; no target/GT fallback.',font=font(18),fill='black');return im

def observation_rows(selected):
 return[dict(frame_key=f['frame_key'],frame_id=f['frame_id'],original_frame_index=f['original_frame_index'],native_slot=r['native_slot'],row_id=r['row_id'],original_query_available=r['original_query_available'],query_xy=r['query_xy'],L=r['L'],observer=None,authority='INDEPENDENT_ASSISTANT_SOURCE_TRAIN_WEAK_NOT_GOLD',target_instance_identified='UNREVIEWED',intended_part_identified='UNREVIEWED',part_visibility_D_E_S_UNKNOWN='UNREVIEWED',query_relation_on_near_off_UNKNOWN='UNREVIEWED',external_occluder_evidence='',location_error_possible='UNREVIEWED',short_visual_evidence='',training_admission=False)for f in selected for r in f['native8']]

def tests():
 passed=[]
 def test(n,v):need(v,n);passed.append(n)
 fake=[dict(frame_key=p+':'+str(i),prefix=p,component_id=p+str(i),stable_SHA256=hashlib.sha256((p+':'+str(i)).encode()).hexdigest())for p in PREFIXES for i in range(4)];a,m=select_six(fake);b,_=select_six(fake[::-1]);test('Stable_prefix_quota_and_component_distinct_independent_order',a==b and Counter(r['prefix']for r in a)==dict(zip(PREFIXES,QUOTA))and m['selected_distinct_components']==6)
 repeated=[dict(x,component_id='one')for x in fake];a,m=select_six(repeated);test('Insufficient_components_retained_and_disclosed_no_frame_loss',len(a)==6 and m['selected_distinct_components']==1 and not m['all6_distinct_components_possible'])
 test('Finite_unavailable_query_not_replaced_or_made_available',marker([12.25,20.5],False,1280,720)==(12.25,20.5,False))
 test('Null_or_offframe_U_has_no_invented_marker',marker(None,False,1280,720)is None and marker([1,820],False,1280,720)is None)
 raw=dict(rgb96=np.arange(8*96*96*3,dtype=np.uint32).reshape(8,96,96,3).astype(np.uint8),valid96=np.ones((8,96,96),bool),crop_constructed=np.ones(8,bool));rr=[dict(native_slot=k,original_query_available=k!=7,query_xy=[12.25,20.5]if k!=7 else None,L=20.)for k in range(8)];before=raw['rgb96'].copy();im=contact(dict(source_id='INVENTED',native8=rr),raw);test('All8_clean96_contact_ROIs_exact_and_original_input_unchanged',np.array_equal(raw['rgb96'],before)and all(np.array_equal(np.asarray(im)[86+450*(k//4)+62:86+450*(k//4)+158,400*(k%4)+300:400*(k%4)+396],raw['rgb96'][k])for k in range(8)))
 template=observation_rows([dict(frame_key='INVENTED',frame_id='I',original_frame_index=0,native8=[dict(r,row_id='I'+str(r['native_slot']))for r in rr])]);test('Observation_template_is_unreviewed_no_new_labels_or_imputed_U',len(template)==8 and all(r['part_visibility_D_E_S_UNKNOWN']=='UNREVIEWED'and r['training_admission']is False for r in template)and template[-1]['original_query_available']is False)
 try:safe(P/'blocked_v112_before_stat');ok=False
 except AssertionError:ok=True
 test('Forbidden_lexical_precheck',ok)
 return dict(completed=True,PASS=True,invented_only=True,checks=passed,count=len(passed),actual_source_RGB_array_header_pixels_or_scores_read=0),im

def prepare():
 need(bind(SCOPE)['sha256']==SCOPE_SHA,'Exact root V205scope');selected,census,bindings=metadata();need(read(SCOPE)['actual_preview_requires_root_entry']is True,'Actualentry required')
 write(H/'SELECTED6_BEFORE_SOURCE_PIXELS.json',dict(completed=True,selection_before_pixels=True,frames=selected,census=census,bindings=bindings,old_weak_labels_or_predictions_used=False))
 binary=[]
 for f in selected:binary.extend([f['image']['binding'],f['native8'][0]['raw_packet']])
 need(len({x['path']for x in binary})==12,'Exactly6canonicalfullRGB+6originalframe8crop packets')
 check,img=tests();write(H/'INVENTED_CHECKS.json',check);img.save(H/'INVENTED_LAYOUT_ONLY.png')
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),selection=bind(H/'SELECTED6_BEFORE_SOURCE_PIXELS.json'),metadata_bindings=bindings,checks=bind(H/'INVENTED_CHECKS.json'),binary_input_whitelist=binary,source_RGB='Only selected6 SOURCE TRAIN full canonicalRGBNPY. Exact inherited decodedRGB SHA and originalJPEG metadata provenance/CV2 decoder preserved; no download, no new CV2 or image sampling.',crops='Only selected6 per-frame V153 INPUT_ONLY NPZ8native rgb96/valid96/crop_constructed/global_index. No shared SOURCE HOLDOUT raw container load, no feature/target/SAM packet read. Original qL/native/U from V181 inputmetadata unchanged.',annotation_authority='Independent assistant natural SOURCE TRAIN weak observation, not blinded/human evaluation gold or physicalpoint certification. D/E/SELF/location/UNKNOWN separate, externalE requires identifiedpart+foregroundevidence. No oldweak/GT/model score shown, no fit/admission.',original_selected_bbox_policy='Use only explicit original_selected_detection_bbox_xyxy when recorded. Bound SOURCE185 lacks it -> null/no fallback. Never use q8contextbox orGTbbox as selectedinstance.',output=str(OUT),actual_bodies_read_during_prepare=0,selected_model=None,fully_solved=False)
 write(PROTO,protocol);write(H/'ROOT_ACTUAL_SELECTED6_PREVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_selected6_TRAIN_input_preview=False,code=protocol['code'],protocol=bind(PROTO),scope=protocol['scope'],selection=protocol['selection'],binary_input_whitelist=binary,models_oldweak_or_GT_or_protected_data_allowed=False,automatic_retry=False));write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTO),checks=protocol['checks'],selection=protocol['selection'],false_entry=bind(H/'ROOT_ACTUAL_SELECTED6_PREVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_source_RGB_crop_NPZ_header_pixel_or_model_reads=0));print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),selected=[f['frame_key']for f in selected],census=census)))

def run(ep):
 e=read(ep);need(e['approved_selected6_TRAIN_input_preview']is True and e['models_oldweak_or_GT_or_protected_data_allowed']is False and e['automatic_retry']is False,'Root actual preview entry only');p=read(checked(e['protocol']));need(e['code']==p['code']==bind(__file__)and e['scope']==p['scope']==bind(SCOPE)and e['protocol']==bind(PROTO)and e['selection']==p['selection']and e['binary_input_whitelist']==p['binary_input_whitelist'],'Exact frozen code/selection/binary whitelist')
 plan=read(checked(p['selection']));selected,census,bindings=metadata();need(selected==plan['frames']and census==plan['census']and bindings==p['metadata_bindings'],'Immutable metadata-only source selection');need(not OUT.exists(),'Exclusive new preview/no retry');OUT.mkdir()
 try:
  records=[];slotrecords=[]
  for pos,f in enumerate(selected):
   with opened(checked(f['image']['binding']))as h:rgb=np.load(h,allow_pickle=False)
   need(f['image']['format']=='NPY_RGB_UINT8_HWC'and rgb.dtype==np.uint8 and rgb.shape==tuple(f['image']['raw_hw'])+(3,)and ash(rgb)==f['image']['decoded_RGB_sha256'],'Exact original canonicalfullRGB decodedbytes')
   binding=f['native8'][0]['raw_packet']
   with opened(checked(binding))as h:
    with np.load(h,allow_pickle=False)as z:raw={k:z[k]for k in ('rgb96','valid96','crop_constructed','global_index')}
   need(raw['rgb96'].dtype==np.uint8 and raw['rgb96'].shape==(8,96,96,3)and raw['valid96'].dtype==bool and raw['valid96'].shape==(8,96,96)and raw['crop_constructed'].dtype==bool and raw['crop_constructed'].shape==(8,),'Original rawframe8 schema')
   folder=OUT/('FRAME_%02d_%04d'%(pos,f['original_frame_index']));safe(folder);folder.mkdir();clean=folder/'ORIGINAL_CANONICAL_RGB.png';Image.fromarray(rgb).save(clean);overlay=folder/'ORIGINAL_Q8_INPUT_OVERLAY.png';preview_frame(rgb,f).save(overlay);card=folder/'ALL8_INPUT_CONTACT.png';contact(f,raw).save(card)
   with Image.open(clean)as im:need(np.array_equal(np.asarray(im),rgb),'Saved fulloriginalRGB PNG parity')
   for r in f['native8']:
    k=r['native_slot'];need(int(raw['global_index'][k])==r['original_global_index']and bool(raw['crop_constructed'][k])==r['original_flags']['feature_constructed']and ash(raw['rgb96'][k])==r['RGB96_sha256']and ash(raw['valid96'][k])==r['valid96_sha256'],'Every originalslot RGB/valid/globalindex/constructed flag exact')
    im=folder/('NATIVE_%d_CLEAN96.png'%k);vm=folder/('NATIVE_%d_VALID96.png'%k);Image.fromarray(raw['rgb96'][k]).save(im);Image.fromarray(raw['valid96'][k].astype(np.uint8)*255).save(vm)
    slotrecords.append(dict(frame_key=f['frame_key'],original_frame_index=f['original_frame_index'],native_slot=k,row_id=r['row_id'],query_xy=r['query_xy'],L=r['L'],original_query_available=r['original_query_available'],crop_constructed=bool(raw['crop_constructed'][k]),clean96=bind(im),valid96=bind(vm),q_marker_on_fullframe=marker(r['query_xy'],r['original_query_available'],rgb.shape[1],rgb.shape[0]),original_raw_input_hash_verified=True,oldweaklabels_GT_or_predictions_shown=False))
   records.append(dict(position=pos,frame_key=f['frame_key'],original_frame_index=f['original_frame_index'],prefix=f['prefix'],component_id=f['component_id'],fulloriginal=bind(clean),overlay=bind(overlay),all8contact=bind(card),original_image=f['image'],source_RGB_binding_verified=True,originalqL_native8_U_preserved=True,selected_detection_bbox_status=f['selected_detection_bbox_status']))
  need(len(records)==6 and len(slotrecords)==48,'All6frames/all48native/U retained');write(OUT/'ANNOTATOR_INPUT_MANIFEST.json',dict(completed=True,frames=records,slots48=slotrecords,selection=p['selection'],no_oldweak_GT_SAM_or_modeloutputs=True,actual_toolviews='PENDING_SEPARATE_IDENTITY_QA'));write(OUT/'UNREVIEWED_OBSERVATION_TEMPLATE48.json',dict(completed=True,observations_pending=True,rows=observation_rows(selected),no_generated_observation_label=True))
  doc=['# V205 공개 TRAIN native 의미 관찰 입력6frame','','사전 metadata 선택6개/48native다. 원 q/L/native/U와 원96crop을 그대로 둔다. 기존 weak owner/finite-end teacher/모델결과/GT는 화면에 보이지 않는다. fullRGB의 q8 overlay는 표시용이며 GT native 위치나 instance 증명이 아니다. unavailable/offframe q는 이동하거나E로 바꾸지 않는다.','', '독립 A/C source 약관찰이며 blind/human evaluation gold가 아니다. intendedpart와 targetinstance가 식별되고 실제crop에서보일때D, 해당부위를 externalforeground가 가린 증거가있을때만E, SELF·location error·UNKNOWN은분리한다. 새 학습·admission은이pilot에없다.','']
  body='<h1>V205 SOURCE TRAIN original inputs only</h1><p>Six metadata-fixed frames, native0..7 including U. No labels, teacher/model scores, GT or new sampling.</p>'
  for r in records:
   doc+=['## '+r['frame_key'],'',f"![원 fullRGB]({Path(r['fulloriginal']['path']).relative_to(OUT).as_posix()})",'',f"![원q8 표시 사본]({Path(r['overlay']['path']).relative_to(OUT).as_posix()})",'',f"![native8 원입력]({Path(r['all8contact']['path']).relative_to(OUT).as_posix()})",''];body+='<h2>'+html.escape(r['frame_key'])+'</h2>'+''.join('<img style="max-width:100%;height:auto" src="'+Path(r[k]['path']).relative_to(OUT).as_posix()+'">'for k in ('fulloriginal','overlay','all8contact'))
  text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(doc));text(OUT/'index.html','<!doctype html><meta charset="utf-8">'+body)
  for b in p['binary_input_whitelist']:checked(b)
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTO),entry=bind(ep),scope=p['scope'],selection=p['selection'],manifest=bind(OUT/'ANNOTATOR_INPUT_MANIFEST.json'),unreviewed_template=bind(OUT/'UNREVIEWED_OBSERVATION_TEMPLATE48.json'),gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),frames6=True,native_rows48=True,original_inputU=sum(not r['original_query_available']for r in slotrecords),all_original_fullRGB_decoded_and_crop96_valid_hashes_verified=True,source_HOLDOUT_current_publicheld_COMMON_VAL_pixels_read=0,oldweaklabels_GT_scores_model_or_initializer_reads=0,new_qL_sampling_labels_model_fit_or_admission=False,actual_toolview='PENDING_SEPARATE_IDENTITY_QA',automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)
