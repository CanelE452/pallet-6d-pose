#!/usr/bin/env python3
"""Exact-OR post-RAW96 finite-part controls. Original native labels remain UNKNOWN."""
import argparse,copy,hashlib,io,json,os,stat
import numpy as np
from PIL import Image,ImageDraw
P='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
D=P+'/query_visibility_public_train_intended_part_query_targets_20261025_v218'
H=D+'/weak_part_pair_preparation_a';OUT=D+'/weak_part_pair_controls_v1'
PNG_PREFIX=P+'/query_visibility_public_train_visible_part_coverage_expansion_20261025_v215/fixed12_public_TRAIN_input_previews_v1/'
CASE_KEYS=((0,6),(2,0),(2,4),(4,0),(5,4),(5,7),(6,4),(6,5),(6,6))
PALETTE=np.array([[32,160,230],[230,64,160]],dtype=np.uint8)
def need(v,msg):
 if not v:raise ValueError(msg)
def opened(p,flags=os.O_RDONLY):
 need(isinstance(p,str) and 'v112' not in p.lower(),'Lexical forbidden before FS')
 need(os.path.isabs(p) and os.path.normpath(p)==p,'Normalized absolute path')
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for c in p.strip('/').split('/')[:-1]:
   n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  z=os.open(p.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
  if not stat.S_ISREG(os.fstat(z).st_mode):os.close(z);raise ValueError('Regular file only')
  return z
 finally:os.close(fd)
def rb(p):
 with os.fdopen(opened(p),'rb') as f:return f.read()
def sha(z):return hashlib.sha256(z).hexdigest()
def bind(p,z):return {'path':p,'sha256':sha(z),'size_bytes':len(z)}
def checked(b):
 need(isinstance(b.get('sha256'),str) and len(b['sha256'])==64,'Exact SHA before FS')
 z=rb(b['path']);need(sha(z)==b['sha256'],'Exact SHA')
 if b.get('size_bytes') is not None:need(len(z)==b['size_bytes'],'Exact optional size')
 return z
def enc(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def save(name,z):
 p=OUT+'/'+name
 with os.fdopen(opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(z)
 return bind(p,z)
def savej(name,x):return save(name,enc(x))
def mkdir_output():
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for c in D.strip('/').split('/'):
   n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  os.mkdir(OUT.rsplit('/',1)[1],dir_fd=fd)
 finally:os.close(fd)
def png(z):
 f=io.BytesIO();Image.fromarray(z).save(f,format='PNG');return f.getvalue()
def decode_RGB(z):
 with Image.open(io.BytesIO(z)) as im:
  need(im.format=='PNG' and im.mode=='RGB' and im.size==(96,96),'Exact original RGB96 PNG');return np.array(im,dtype=np.uint8)
def decode_valid(z):
 with Image.open(io.BytesIO(z)) as im:
  need(im.format=='PNG' and im.size==(96,96),'Exact original valid96 PNG')
  a=np.array(im)
  if a.ndim==3:
   need(a.shape==(96,96,3) and np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2]),'Equal channels binary validity only');a=a[:,:,0]
  need(a.shape==(96,96) and set(np.unique(a)).issubset({0,1,255}), 'Binary original valid values only')
  return np.ascontiguousarray(a!=0,dtype=np.bool_)
def target_mask(s):
 need(s['shape']==[96,96],'Original target96')
 m=np.zeros((96,96),dtype=np.bool_)
 for y,x0,x1 in s['positive_runs_y_x0_x1']:
  need(all(type(v) is int for v in (y,x0,x1)) and 0<=y<=95 and 0<=x0<=x1<=95,'Original exact OR runs');m[y,x0:x1+1]=True
 r=np.zeros_like(m)
 for a in s['original_ROIs_preserved']:
  need(len(a)==4 and all(type(v) is int for v in a) and 0<=a[0]<=a[2]<=95 and 0<=a[1]<=a[3]<=95,'Unchanged inclusive ROI')
  r[a[1]:a[3]+1,a[0]:a[2]+1]=True
 need(np.array_equal(m,r) and int(m.sum())==s['positive_pixels'] and m.any(),'Exact union not bbox/shrink');return m
def build_pair(before,valid,target):
 need(before.shape==(96,96,3) and before.dtype==np.uint8 and valid.shape==target.shape==(96,96) and valid.dtype==target.dtype==np.bool_,'Exact arrays')
 if not np.all(valid[target]):return {'status':'COVER_ALPHA_NOT_ALL_ORIGINAL_VALID','constructed':False}
 ys,xs=np.where(target);x0,x1=int(xs.min()),int(xs.max());y0,y1=int(ys.min()),int(ys.max());candidates=[]
 # Predeclared placement search uses geometry+original validity only, never RGB/model/labels.
 for dy in range(-y0,96-y1):
  for dx in range(-x0,96-x1):
   if np.any(target[ys+dy,xs+dx]) or not np.all(valid[ys+dy,xs+dx]):continue
   candidates.append((abs(dx)+abs(dy),y0+dy,x0+dx,dx,dy))
 if not candidates:return {'status':'NO_ALLVALID_DISJOINT_SHAM_NO_FORCING','constructed':False}
 _,_,_,dx,dy=min(candidates);sy,sx=ys+dy,xs+dx
 pattern=PALETTE[((xs-x0)//4+(ys-y0)//4)%2];cover=before.copy();sham=before.copy();cover[ys,xs]=pattern;sham[sy,sx]=pattern
 ca=target.copy();sa=np.zeros_like(target);sa[sy,sx]=True
 cd=np.any(cover!=before,axis=2);sd=np.any(sham!=before,axis=2)
 proof={'target_OR_pixels':int(target.sum()),'COVER_alpha_pixels':int(ca.sum()),'SHAM_alpha_pixels':int(sa.sum()),'COVER_target_whole_OR_opaque':bool(np.all(ca[target])),'both_alpha_all_original_valid':bool(np.all(valid[ca]) and np.all(valid[sa])),'alpha_same_mass_pattern_translation':bool(np.array_equal(ca[ys,xs],sa[sy,sx])),'RGB_pattern_exact_same':bool(np.array_equal(cover[ys,xs],sham[sy,sx])),'SHAM_alpha_disjoint_target_OR':not bool(np.any(sa&target)),'SHAM_target_RGB_byte_unchanged':bool(np.array_equal(sham[target],before[target])),'COVER_actual_changed_target_pixels':int(cd[target].sum()),'COVER_actual_changed_all_pixels':int(cd.sum()),'SHAM_actual_changed_target_pixels':int(sd[target].sum()),'SHAM_actual_changed_pixels':int(sd.sum()),'all_actual_changes_only_original_valid':not bool(np.any((cd|sd)&~valid)),'COVER_outside_alpha_byte_unchanged':bool(np.array_equal(cover[~ca],before[~ca])),'SHAM_outside_alpha_byte_unchanged':bool(np.array_equal(sham[~sa],before[~sa])),'SHAM_translation_xy':[dx,dy],'native_or_query_support_label_certified':False}
 ok=all(v for k,v in proof.items() if isinstance(v,bool) and k!='native_or_query_support_label_certified') and proof['COVER_actual_changed_target_pixels']>0 and proof['SHAM_actual_changed_pixels']>0
 if not ok:return {'status':'ACTUAL_RGB_OR_ALPHA_PROOF_FAILED_NO_FORCING','constructed':False,'proof':proof}
 return {'status':'CONSTRUCTED_FINITE_PART_WEAK_ONLY','constructed':True,'BEFORE':before.copy(),'COVER':cover,'SHAM':sham,'COVER_alpha':ca,'SHAM_alpha':sa,'proof':proof}
def variant_metadata(case,variant,constructed,image=None):
 o=copy.deepcopy(case['input']);known=variant=='BEFORE' or constructed
 return {'target_id':case['target_id'],'variant':variant,'input':o,'image':image,'stored_status':'PRESENT' if known else 'MISSING_FAILED_CONTROL_PRESERVED','finite_part_DE_weak_target_D0_E1':(1 if variant=='COVER' else 0) if known else -1,'authority':'POST_RAW96_FINITE_VISIBLE_PART_WITNESS_WEAK_NOT_NATIVE_D_E','native_DE_target':-1,'query_support_target':'NOT_INFERRED_FROM_CONTROL_VARIANT','ROI_is_target_only_not_deployment_input':True,'ROI_is_weak_bag_region_selector_not_pixelwise_material_gold':True,'outside_ROI_is_IGNORE_not_material_negative':True,'original_ROLE_preserved':True,'future_fit_role_eligible':o['internal_source_role']=='SOURCE_FIT','SOURCE_DEV_excluded_every_training_branch':o['internal_source_role']=='SOURCE_DEV','training_admission':False}
def card(case,before,result):
 im=Image.new('RGB',(1500,620),'white');d=ImageDraw.Draw(im);o=case['input']
 d.text((20,18),'SOURCE frame%d/native%d %s | FINITE PART ONLY; native UNKNOWN; q/L unchanged'%(o['position'],o['native_slot'],o['internal_source_role']),fill='black')
 for j,name in enumerate(('BEFORE','COVER','SHAM')):
  x=25+j*495;d.text((x,55),name+' | '+('weak D' if name!='COVER' else 'weak E') if result['constructed'] or name=='BEFORE' else name+' MISSING',fill='black')
  a=before if name=='BEFORE' else result.get(name)
  if a is None:continue
  im.paste(Image.fromarray(a).resize((384,384),Image.Resampling.NEAREST),(x,80));cx=x+191.5;cy=80+191.5
  d.line((cx-9,cy,cx+9,cy),fill='#00bfd7',width=2);d.line((cx,cy-9,cx,cy+9),fill='#00bfd7',width=2)
  for r in case['finite_target_support']['original_ROIs_preserved']:d.rectangle((x+4*r[0],80+4*r[1],x+4*(r[2]+1)-1,80+4*(r[3]+1)-1),outline='#ed2a47',width=2)
 d.text((20,490),'Cyan: original q47.5. Red: two unchanged weak witness target ROIs. Display overlays only.',fill='black')
 d.text((20,515),result['status'],fill='black');d.text((20,540),'Same fixed checker RGB/alpha pattern and mass; COVER exact OR, SHAM target untouched. Outside ROI is IGNORE.',fill='black')
 d.text((20,565),'Not natural occlusion / native visibility gold / q correction / RAW96 replay proof. No training admission.',fill='black')
 return png(np.array(im))
def run(ep):
 e=json.loads(rb(ep));need(e.get('approved_actual_finite_part_pair_images') is True and e['output_directory']==OUT and e['training_admission'] is False,'Root actual image approval before payload FS')
 need(e['code']['path']==os.path.abspath(__file__) and e['protocol']['path']==H+'/PROTOCOL_BEFORE_WEAK_PART_PAIR_PNG_READS.json','Exact code/protocol');checked(e['code']);checked(e['protocol'])
 plan=json.loads(checked(e['input_plan']));need([tuple(c['case_key']) for c in plan['cases9']]==list(CASE_KEYS),'Fixed9 no new selection')
 need(e['metadata_inputs']==plan['metadata_inputs'],'Exact saved-text closure')
 for b in e['metadata_inputs'].values():checked(b)
 payloads=plan['original_PNG_bindings18'];need(len(payloads)==18 and len({b['path'] for b in payloads})==18,'Exactly9originalclean+9valid')
 bypath={b['path']:b for b in payloads};used={c[k]['path'] for c in plan['cases9'] for k in ('clean96','valid96')};need(used==set(bypath),'Exact PNG closure')
 for c in plan['cases9']:
  need(c['input']['role']=='TRAIN' and c['input']['original_query_available'] and c['input']['internal_source_role'] in ('SOURCE_FIT','SOURCE_DEV'),'Original fixed source roles/availability')
  for k in ('clean96','valid96'):need(bypath[c[k]['path']]['sha256']==c[k]['sha256'],'Exact normalized path/SHA binding')
 for b in payloads:need(b['path'].startswith(PNG_PREFIX) and b['path'].endswith('.png'),'Declared original source preview PNGs only')
 data={b['path']:checked(b) for b in payloads};mkdir_output();opaque=savej('BEFORE_PNG_DECODE_SHA_QA.json',{'completed':True,'PASS':True,'verified_original_PNG18':payloads,'decoded_before_receipt':0})
 attempts=[];variants=[];cards=[];proofs=[]
 for c in plan['cases9']:
  o=c['input'];prefix='FRAME_%02d_NATIVE_%d'%(o['position'],o['native_slot']);before=decode_RGB(data[c['clean96']['path']]);valid=decode_valid(data[c['valid96']['path']])
  need(sha(valid.tobytes())==c['valid96_decoded_bool_sha256'],'Original decoded bool96 exact SHA')
  target=target_mask(c['finite_target_support']);result=build_pair(before,valid,target);images={}
  images['BEFORE']=save(prefix+'_BEFORE.png',data[c['clean96']['path']])
  if result['constructed']:
   for name in ('COVER','SHAM'):images[name]=save(prefix+'_'+name+'.png',png(result[name]))
   for name in ('COVER_alpha','SHAM_alpha'):images[name]=save(prefix+'_'+name+'.png',png(result[name].astype(np.uint8)*255))
  for name in ('BEFORE','COVER','SHAM'):variants.append(variant_metadata(c,name,result['constructed'],images.get(name)))
  cr=save(prefix+'_CARD.png',card(c,before,result));cards.append({'case_key':c['case_key'],'image':cr,'status':result['status']})
  attempts.append({'case_key':c['case_key'],'target_id':c['target_id'],'input':o,'status':result['status'],'constructed':result['constructed'],'original_target_support':c['finite_target_support'],'images':images,'card':cr,'training_admission':False})
  proofs.append({'case_key':c['case_key'],'status':result['status'],'proof':result.get('proof'),'original_clean96_byteexact':images['BEFORE']['sha256']==c['clean96']['sha256'],'original_valid_bitmap_binding':c['valid96'],'original_input_qL_native_U_roles_unchanged':True,'training_admission':False})
 rows=save('ALL27_ATTEMPTED_VARIANT_METADATA.jsonl',b''.join((json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode() for v in variants));a=savej('ALL9_ATTEMPTS.json',{'completed':True,'attempts9':attempts});p=savej('ALL9_ACTUAL_PIXEL_ALPHA_PROOFS.json',{'completed':True,'proofs9':proofs})
 md=['# 원RAW96의 finite local-part weak controls','', '원 q/L/native를 보존한 post-RAW96 counterfactual이다. D/E/D는 고정 finite witness의 bag/region 약관계이며 native visibility나 자연 가림 gold가 아니다. 사각형 OR에는 배경이 포함될 수 있으므로 모든 ROI 픽셀을 팔레트 material-positive로 학습하지 않는다. ROI는 타깃 selector 전용이고 바깥은 IGNORE다. SOURCE_DEV는 모든 fit에서 제외한다.','']
 for c in cards:md+=['## frame%d / native%d'%tuple(c['case_key']),'![]('+c['image']['path'].rsplit('/',1)[1]+')','']
 gallery=save('PORTABLE_GALLERY_KO.md','\n'.join(md).encode())
 done=savej('COMPLETE.json',{'completed':True,'PASS':True,'entry':bind(ep,rb(ep)),'code':e['code'],'protocol':e['protocol'],'input_plan':e['input_plan'],'metadata_inputs':e['metadata_inputs'],'opaque_PNG_QA':opaque,'attempts':a,'variants':rows,'proofs':p,'cards9':cards,'gallery':gallery,'attempted':9,'constructed':sum(r['constructed'] for r in attempts),'failed_retained':sum(not r['constructed'] for r in attempts),'original_query_or_native_labels_changed':0,'new_primary_native_D_E_labels':0,'image_validity_mode':'EXACT_ORIGINAL_VALID96_BITMAP','models_source_arrays_GT_payloads_PnP_reads_or_calls':0,'training_admission':False,'automatic_retry':False})
 print(json.dumps(done,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
