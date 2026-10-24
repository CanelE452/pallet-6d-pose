"""D48 SOURCE finite-part weak 대조. 실제 entry 전 PNG 접근 없음; 학습 실행 없음."""
import argparse,hashlib,io,json,math,os,stat,csv
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;R=H.parent;P=R.parent
PHASE=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
OUT=PHASE/'expected_part_counterfactual_pairs_v1'
FAMILY='PUBLIC_GT_QUERY_EXPECTED_FINITE_PART_COUNTERFACTUAL_V1'
PROTO=H/'PROTOCOL_BEFORE_D48_PNG_BODIES.json';PLAN=H/'D48_PLAN_AND_EXACT_PNG_WHITELIST_BEFORE_PIXELS.json'
AUTHORITY='CONTROLLED_EXPECTED_FINITE_PART_WEAK_NOT_PHYSICAL_NATIVE_POINT_GOLD'

def need(x,msg):
 if not x:raise AssertionError(msg)
def safe(path):
 s=str(path);need('v112'not in s.casefold(),'V112 lexical reject before FS');p=Path(s)
 need(p.is_absolute()and'..'not in p.parts and P in p.parents,'Exact project path');return p
def parentfd(path):
 p=safe(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise
def rb(path):
 p,fd=parentfd(path)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file only')
 with os.fdopen(n,'rb')as f:return f.read()
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def bind(p):b=rb(p);return dict(path=str(p),sha256=sha(b),size_bytes=len(b))
def checked(b):
 p=safe(b['path']);need(p.suffix=='.json','Declared metadata JSON only');z=rb(p)
 need(sha(z)==b['sha256']and('size_bytes'not in b or len(z)==b['size_bytes']),'Exact metadata binding');return json.loads(z)
def mkdir_new(p):
 p,fd=parentfd(p)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)
def save(p,b):
 p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(b)
 need(rb(p)==b,'Exact own output bytes');return dict(path=str(p),sha256=sha(b),size_bytes=len(b))
def savej(p,x):return save(p,enc(x))
def png_bytes(a):
 f=io.BytesIO();Image.fromarray(a).save(f,format='PNG');z=f.getvalue()
 with Image.open(io.BytesIO(z))as im:need(np.array_equal(np.asarray(im),a),'Own encoded PNG equals actual pixels')
 return z
def decode_RGB(z):
 with Image.open(io.BytesIO(z))as im:need(im.mode=='RGB'and im.size==(96,96),'Original unmarked96 RGB, no mode conversion/resize');return np.asarray(im).copy()
def mask(rle):
 x=np.zeros(9216,np.bool_);end=0
 for start,length in rle:
  need(type(start)is int and type(length)is int and start>=end and length>0 and start+length<=9216,'Canonical sorted96 RLE')
  x[start:start+length]=True;end=start+length
 return x.reshape(96,96)
def bbox(a):
 y,x=np.where(a);need(len(x)>0,'Nonempty weak bag');return [int(x.min()),int(y.min()),int(x.max()),int(y.max())]
def seed_for(parent,role):
 need(role in('SOURCE_FIT','SOURCE_DEV'),'Immutable role');return int(sha((FAMILY+'|'+role+'|'+parent).encode())[:16],16)
def patch_for(parent,role,w,h):
 """한 고정 절차family: 평탄색+저주파 색 변화/넓은 stripe. 영상 적응/색 선택 없음."""
 seed=seed_for(parent,role);rng=np.random.default_rng(seed)
 base=rng.integers(56,200,size=3);amp=rng.integers(8,21,size=3);phase=float(rng.uniform(0,2*np.pi));stripe_phase=int(rng.integers(0,16))
 yy,xx=np.mgrid[:h,:w];low=np.sin(2*np.pi*(xx+yy)/48+phase);stripe=np.where(((xx+2*yy+stripe_phase)//16)%2==0,1,-1)
 rgb=np.clip(np.rint(base[None,None,:]+low[:,:,None]*amp[None,None,:]+stripe[:,:,None]*10),0,255).astype(np.uint8)
 return rgb,dict(family='SEEDED_FLAT_LOW_FREQUENCY_STRIPE_V1',seed=seed,seed_namespace=role,base_RGB=base.tolist(),low_frequency_amplitude=amp.tolist(),phase=phase,stripe_phase=stripe_phase,period_lowfreq=48,stripe_width=16,real_occluder_donor_used=False,pattern_chosen_using_source_RGB=False)
def make_pair(before,valid,target,parent,role):
 need(before.shape==(96,96,3)and before.dtype==np.uint8 and valid.shape==target.shape==(96,96),'Exact parent96')
 x0,y0,x1,y1=bbox(target);rect=np.zeros((96,96),np.bool_);rect[y0:y1+1,x0:x1+1]=True
 geometry=dict(target_union_bbox=[x0,y0,x1,y1],COVER_rectangle=[x0,y0,x1,y1],rectangle_is_not_material_segmentation=True)
 if not np.all(valid[rect]):return dict(constructed=False,status='COVER_RECTANGLE_NOT_ALL_STRICT_VALID',geometry=geometry)
 candidates=[];w=x1-x0+1;h=y1-y0+1
 for sy in range(97-h):
  for sx in range(97-w):
   if np.any(target[sy:sy+h,sx:sx+w])or not np.all(valid[sy:sy+h,sx:sx+w]):continue
   dx,dy=sx-x0,sy-y0;candidates.append((abs(dx)+abs(dy),sy,sx,dx,dy))
 if not candidates:return dict(constructed=False,status='NO_ALLVALID_TARGET_DISJOINT_SHAM',geometry=geometry)
 _,sy,sx,dx,dy=min(candidates);geometry['SHAM_rectangle']=[sx,sy,sx+w-1,sy+h-1];patch,recipe=patch_for(parent,role,w,h)
 cover=before.copy();sham=before.copy();cover[y0:y1+1,x0:x1+1]=patch;sham[sy:sy+h,sx:sx+w]=patch
 ca=rect;sa=np.zeros_like(rect);sa[sy:sy+h,sx:sx+w]=True
 cd=np.any(cover!=before,axis=2);sd=np.any(sham!=before,axis=2)
 proof=dict(COVER_covers_full_exact_union=bool(np.all(ca[target])),SHAM_disjoint_exact_union=not bool(np.any(sa&target)),both_alpha_all_strict_valid=bool(np.all(valid[ca])and np.all(valid[sa])),equal_alpha_mass=int(ca.sum())==int(sa.sum()),same_rectangle_shape=True,same_actual_RGB_patch=bool(np.array_equal(cover[y0:y1+1,x0:x1+1],sham[sy:sy+h,sx:sx+w])),SHAM_target_RGB_byte_unchanged=bool(np.array_equal(sham[target],before[target])),COVER_outside_alpha_RGB_byte_unchanged=bool(np.array_equal(cover[~ca],before[~ca])),SHAM_outside_alpha_RGB_byte_unchanged=bool(np.array_equal(sham[~sa],before[~sa])),all_changes_inside_strict_valid=not bool(np.any((cd|sd)&~valid)),COVER_changed_target_pixels=int(cd[target].sum()),SHAM_changed_target_pixels=int(sd[target].sum()),COVER_changed_total=int(cd.sum()),SHAM_changed_total=int(sd.sum()),alpha_pixels=int(ca.sum()),target_union_pixels=int(target.sum()),SHAM_translation_xy=[dx,dy],geometry_search_candidates=len(candidates),native_point_gold_certified=False)
 ok=all(v for k,v in proof.items()if type(v)is bool and k!='native_point_gold_certified')and proof['COVER_changed_target_pixels']>0 and proof['SHAM_changed_total']>0
 return dict(constructed=ok,status='CONSTRUCTED_EXPECTED_FINITE_PART_WEAK'if ok else'ACTUAL_PIXEL_PROOF_FAILED_NO_ADMISSION',geometry=geometry,recipe=recipe,proof=proof,COVER=cover,SHAM=sham,COVER_alpha=ca,SHAM_alpha=sa)
def card(row,before,result):
 out=Image.new('RGB',(900,420),'white');d=ImageDraw.Draw(out)
 d.text((8,6),row['row_id']+' | '+row['source_role']+' | '+result['status'],fill='black')
 for j,key in enumerate(['BEFORE','COVER','SHAM']):
  d.text((8+j*300,28),key+' clean training RGB96',fill='black')
  a=before if key=='BEFORE'else result.get(key)
  if a is not None:out.paste(Image.fromarray(a).resize((288,288),Image.Resampling.NEAREST),(6+j*300,48))
  else:d.rectangle((6+j*300,48,294+j*300,336),fill='#dddddd');d.text((14+j*300,160),'NOT CONSTRUCTED',fill='black')
 d.text((8,347),'Weak finite-part bag only; native point target=-1. Input q/L/native/q8/valid fixed.',fill='black')
 d.text((8,369),'COVER rectangle includes exact ROI union; SHAM same patch/mass elsewhere. No GT recrop.',fill='black')
 d.text((8,391),'Parent seed namespace '+row['source_role']+' | no new fit / model / PnP',fill='black')
 f=io.BytesIO();out.save(f,format='PNG');return f.getvalue()
def run(entry_path):
 need(Path(entry_path).parent==H,'Root actual entry in exact prep folder');e=json.loads(rb(entry_path))
 need(e.get('approved_actual_SOURCE_counterfactual_production')is True and e.get('automatic_retry')is False and isinstance(e.get('root_authorization_message'),str)and e['root_authorization_message'].strip(),'Separate explicit root execution permission')
 need(e['code']==bind(__file__)and e['protocol']==bind(PROTO)and e['plan']==bind(PLAN),'Pinned code/protocol/48plan')
 proto=checked(e['protocol']);plan=checked(e['plan']);need(proto['code']==e['code']and checked(proto['checks'])['PASS']is True,'Frozen invented checks')
 rows=plan['parents48'];wl={b['path']:b for b in plan['exact_PNG_whitelist48']}
 need(len(rows)==len(wl)==48 and Counter(r['source_role']for r in rows)=={'SOURCE_FIT':39,'SOURCE_DEV':9},'Exact48/source split')
 cache={}
 for p,b in wl.items():
  need(p==str(safe(p))and p.startswith(plan['input_PNG_prefix'])and Path(p).suffix=='.png','Declared PNG membership before FS');z=rb(p)
  need(sha(z)==b['sha256']and len(z)==b['size_bytes'],'Every declared48 opaque source SHA before any decode');cache[p]=z
 # Output is fresh; root may already have written a prospective phase scope.
 p,fd=parentfd(PHASE)
 try:
  try:os.mkdir(p.name,dir_fd=fd)
  except FileExistsError:
   n=os.open(p.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(n)
 finally:os.close(fd)
 need(e['output_directory']==str(OUT),'Exact output');mkdir_new(OUT);savej(OUT/'STARTED.json',dict(completed=False,entry=bind(entry_path),source_PNGs48_all_SHA_verified_before_decode=True))
 try:
  attempts=[];variants=[];cards=[]
  for role in('SOURCE_FIT','SOURCE_DEV'):mkdir_new(OUT/role)
  for r in rows:
   parent=r['row_id'];before=decode_RGB(cache[r['RGB96_binding']['path']]);valid=mask(r['valid96_RLE']);target=mask(r['exact_union_RLE_rowmajor96']);need(np.all(valid[target]),'Exact weak union in parent valid')
   z=make_pair(before,valid,target,parent,r['source_role']);folder=OUT/r['source_role']/parent;mkdir_new(folder)
   images={'BEFORE':save(folder/'BEFORE.png',cache[r['RGB96_binding']['path']])};need(images['BEFORE']['sha256']==r['RGB96_binding']['sha256'],'Original encoded PNG copy equality')
   vb=save(folder/'VALID96.png',png_bytes(valid.astype(np.uint8)*255));tb=save(folder/'EXACT_TARGET_UNION.png',png_bytes(target.astype(np.uint8)*255));alpha={}
   for key in('COVER','SHAM'):
    if key in z:images[key]=save(folder/(key+'.png'),png_bytes(z[key]));alpha[key]=save(folder/(key+'_ALPHA.png'),png_bytes(z[key+'_alpha'].astype(np.uint8)*255))
   proofpath=folder/'PROOF.json';proof=savej(proofpath,dict(completed=True,parent_row_id=parent,constructed=z['constructed'],geometry=z['geometry'],recipe=z.get('recipe'),proof=z.get('proof'),original_input=r,original_RGB_PNG_byte_copy_equal=True,training_admission=False,native_point_DE_target=-1))
   cb=save(folder/'CARD.png',card(r,before,z));cards.append(dict(parent_row_id=parent,nominal=r['normalized_offset']==[0,0],source_role=r['source_role'],card=cb))
   attempts.append(dict(parent_row_id=parent,paired_id=FAMILY+'__'+parent,source_role=r['source_role'],constructed=z['constructed'],status=z['status'],geometry=z['geometry'],recipe=z.get('recipe'),proof=proof,images=images,alpha_images=alpha,valid96=vb,target_union=tb,card=cb,parent_metadata=r,training_admission=False))
   for var in('BEFORE','COVER','SHAM'):
    available=var in images;eligible=z['constructed'];label=(0 if var!='COVER'else 1)if eligible or var=='BEFORE'else None
    variants.append(dict(row_id=parent+'__'+var,parent_row_id=parent,paired_id=FAMILY+'__'+parent,family=FAMILY,variant=var,status='PRESENT_CONSTRUCTED'if eligible else'PARENT_BEFORE_ONLY'if var=='BEFORE'else'NOT_CONSTRUCTED_NO_LABEL',component_id=r['component_id'],source_role=r['source_role'],original_nominal_row_id=r['nominal_row_id'],native_slot=r['native_slot'],query_xy=r['generated_query_xy'],L=r['L'],query_available=r['generated_query_available'],q8_xy=r['q8_xy'],q8_available=r['q8_available'],q8_L=r['q8_L'],q8_native_slots=list(range(8)),condition32=r['condition32'],valid96_RLE=r['valid96_RLE'],RGB96=images.get(var)if(var=='BEFORE'or eligible)else None,input_available=bool(var=='BEFORE'or eligible),target_known=label is not None,failure_reason=None if eligible else z['status'],valid96_binding=vb,weak_finite_part_target_D0_E1=label,native_point_DE_target=-1,authority=AUTHORITY,exact_target_union_RLE=r['exact_union_RLE_rowmajor96'],ROI_by_observer=r['ROI96_by_observer'],parent_RGB96_binding=r['RGB96_binding'],parent_source_metadata_binding=plan['D48_source_metadata'],paired_proof=proof,constructed=eligible,proposed_CF_fit_eligible=bool(eligible and r['source_role']=='SOURCE_FIT'),all_same_query_L_native_q8_condition32_valid=True,training_admission=False,GT_derived_training_query=True,ROI_is_supervision_only=True))
  ledger=save(OUT/'ALL144_VARIANT_ROWS.jsonl',b''.join((json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode()for v in variants));ab=savej(OUT/'ALL48_ATTEMPTS.json',dict(completed=True,attempts48=attempts,all_input_parents_retained=True))
  pages=[]
  for nominal,title in [(True,'NOMINAL19'),(False,'JITTER29')]:
   rr=[c for c in cards if c['nominal']==nominal];need(len(rr)==(19 if nominal else 29),'Fixed full coverage')
   for off in range(0,len(rr),4):
    batch=rr[off:off+4];sheet=Image.new('RGB',(900,420*len(batch)),'white')
    for i,c in enumerate(batch):
     with Image.open(io.BytesIO(rb(c['card']['path'])))as im:sheet.paste(im,(0,i*420))
    f=io.BytesIO();sheet.save(f,format='PNG');pages.append(dict(kind=title,parents=[x['parent_row_id']for x in batch],image=save(OUT/(title+'_CONTACT_'+str(off//4).zfill(2)+'.png'),f.getvalue())))
  gallery=['# SOURCE expected finite-part matched controls','', 'All19 nominal parents; all29 jitter parents separately. Finite-part weak, not native-point gold. Failed attempts retained. No training.','']
  for c in pages:gallery+=['## '+c['kind']+' '+','.join(c['parents']),'','![]('+Path(c['image']['path']).name+')','']
  gb=save(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(gallery).encode());html='<meta charset="utf-8"><h1>'+FAMILY+'</h1><p>Weak finite-part only; native point target -1, no training.</p>'+''.join('<h2>'+c['kind']+'</h2><img style="max-width:100%" src="'+Path(c['image']['path']).name+'">'for c in pages);hb=save(OUT/'index.html',html.encode())
  counts=dict(parents48=48,nominal_parents19=19,jitter_parents29=29,attempts_constructed=sum(a['constructed']for a in attempts),failed_retained=sum(not a['constructed']for a in attempts),constructed_by_role=dict(Counter(a['source_role']for a in attempts if a['constructed'])),failed_by_status=dict(Counter(a['status']for a in attempts if not a['constructed'])),variant_rows144=len(variants),original_png_copies48=48,source_roles_fixed=dict(Counter(r['source_role']for r in rows)),new_native_point_gold=0,actual_training=0)
  role_manifests={}
  for role in('SOURCE_FIT','SOURCE_DEV'):
   rr=[v for v in variants if v['source_role']==role];cf=[v for v in rr if v['constructed']];bb=[v for v in rr if v['variant']=='BEFORE'];rgb={v['RGB96']['path']:v['RGB96']for v in rr if v['input_available']}
   role_manifests[role]=savej(OUT/(role+'_ROWS_AND_EXACT_PNG_WHITELIST.json'),dict(completed=True,PASS=True,source_role=role,successful_triplet_parent_ids=sorted({v['parent_row_id']for v in cf}),CF_variant_rows=cf,all_parent_BEFORE_rows=bb,exact_RGB96_whitelist=list(rgb.values()),trainer_DEV_payloads_allowed_before_two_new_states_freeze=role!='SOURCE_DEV',training_admission=False,native_point_gold=False))
  summary=savej(OUT/'SUMMARY.json',dict(completed=True,PASS=True,**counts,only_one_procedural_family=True,original_query9_not_used=True,new_checker_pattern_invention_claim=False,sourceDEV_generated_but_not_fit=True,counts_are_not_independent_frames=True))
  buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(variants[0]));writer.writeheader()
  for v in variants:writer.writerow({k:json.dumps(x,ensure_ascii=False,separators=(',',':'))if isinstance(x,(dict,list))else x for k,x in v.items()})
  cb=save(OUT/'ALL144_VARIANT_ROWS.csv',buf.getvalue().encode())
  image_rows=[]
  for a in attempts:
   image_rows.extend(a['images'].values());image_rows.extend(a['alpha_images'].values());image_rows.extend([a['valid96'],a['target_union'],a['card']])
  image_rows.extend(c['image']for c in pages)
  mb=savej(OUT/'IMAGE_MANIFEST.json',dict(completed=True,owned_PNG_bindings=image_rows,source_PNG_bindings=list(wl.values()),source_PNG_payloads_opened=48,all_owned_training_RGB_unmarked=True,display_resizes_only_in_cards=True))
  for path,b in wl.items():need(sha(rb(path))==b['sha256'],'Final original SOURCE PNG SHA unchanged')
  done=savej(OUT/'COMPLETE.json',dict(completed=True,PASS=True,entry=bind(entry_path),code=e['code'],protocol=e['protocol'],plan=e['plan'],source48_all_opaque_SHA_verified_before_any_decode=True,source48_final_SHA_unchanged=True,all48_parent_metadata_preserved=True,attempts=ab,variant_rows=ledger,CSV=cb,summary=summary,role_manifests=role_manifests,image_manifest=mb,gallery=gb,HTML=hb,**counts,original_query_L_native_q8_condition32_valid_unchanged=True,native_point_DE_targets_all_minus1=True,training_admission=False,original_GT_arrays_models_protected_evaluation_or_PnP=0,automatic_retry=False))
  print(json.dumps(dict(completed=True,PASS=True,COMPLETE=done,summary=summary,counts=counts),ensure_ascii=False))
 except BaseException as err:
  savej(OUT/'FAILURE_PRESERVED_NO_RETRY.json',dict(completed=False,PASS=False,error_type=type(err).__name__,error=str(err),automatic_retry=False));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--entry',required=True);run(a.parse_args().entry)
