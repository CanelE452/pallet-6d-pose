"""Saved public image-only replay, independently written NumPy certificates."""
from collections import Counter,OrderedDict
import hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;V=HERE.parent;OUT=V/'corpus_build_v1'
EXPECTED_SEAL='0ef41a9fd9d35471acb286efc044aa4831fb6d209d2d916cae3bc343aec1689b'
CHECKS=Counter();FAIL=[]
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pixel(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def check(ok,name):
 CHECKS[name]+=1
 if not ok:FAIL.append(name)
def write(n,obj):
 with(HERE/n).open('x')as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def erosion(a):
 h,w=a.shape;p=np.pad(a,1)
 return np.logical_and.reduce([p[y:y+h,x:x+w]for y in range(3)for x in range(3)])
def protected(row,t):
 h,w=row['raw_hw'];result=np.zeros((h,w),bool)
 for region in t['original_regions']:
  x0,y0,x1,y1=region['bbox'];left,top=np.floor(np.array([x0,y0])+.5).astype(int)
  if region['kind']=='PALLOX_CLOSED_NATIVE_UNCERTAINTY':right,bottom=np.floor(np.array([x1,y1])+.5).astype(int)+1
  else:right,bottom=np.ceil(np.array([x1,y1])+.5).astype(int)
  assert 0<=left<right<=w and 0<=top<bottom<=h
  result[top:bottom,left:right]=True
 q,L=row['query_xy'],row['L'];cells=set()
 for sy in [-1,1]:
  for sx in [-1,1]:
   x,y=q[0]+sx*L/192,q[1]+sy*L/192;ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy
   for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
    if weight>0:cells.add((ix+ox,iy+oy));result[iy+oy,ix+ox]=True
 check([list(p)for p in sorted(cells)]==t['central4_native_stencil'],'Original_central4_stencil')
 return result
def crop_rgb(rgb,q,L):
 h,w=rgb.shape[:2];a=(np.arange(96,dtype=np.float64)+.5)/96-.5
 x,y=np.meshgrid(q[0]+a*L,q[1]+a*L);ix,iy=np.floor(x).astype(int),np.floor(y).astype(int);dx,dy=x-ix,y-iy
 out=np.zeros((96,96,3),np.float64)
 for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
  xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
  out[inside]+=rgb[yy[inside],xx[inside]].astype(np.float64)*weight[inside,None]
 valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
 out=np.floor(np.clip(out,0,255)+.5).astype(np.uint8);out[~valid]=0
 return out.transpose(2,0,1).copy(),valid[None].copy()
def mass96(alpha,q,L):
 h,w=alpha.shape;a=(np.arange(96,dtype=np.float64)+.5)/96-.5;x,y=np.meshgrid(q[0]+a*L,q[1]+a*L)
 ix,iy=np.floor(x).astype(int),np.floor(y).astype(int);dx,dy=x-ix,y-iy;out=np.zeros((96,96),np.float64)
 for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
  xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
  out[inside]+=alpha[yy[inside],xx[inside]].astype(np.float64)*weight[inside]
 valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5);out[~valid]=0
 return float(out.sum())

def main():
 assert sha(OUT/'CORPUS_OUTPUT_SEAL.json')==EXPECTED_SEAL
 seal=read(OUT/'CORPUS_OUTPUT_SEAL.json');assert seal['completed']and seal['PASS']and not seal['training_admission']and seal['model_forward_or_fit']==0 and not seal['current223']and not seal['V112']
 write('SCOPE.json',dict(expected_completed_corpus_seal=bind(OUT/'CORPUS_OUTPUT_SEAL.json'),scope='Root authorized saved public image/crop/mask numerical replay after terminal completion; no construction/model/fit/private/V112/PnP',no_teacher_admission=True))
 for b in seal['files']:check(sha(b['path'])==b['sha256'],'All_output_file_SHA')
 protocol=read(seal['protocol']['path']);check(sha(seal['protocol']['path'])==seal['protocol']['sha256'],'Protocol_binding')
 B=protocol['bindings'];base=rows(B['bases']['path']);targets=rows(B['targets']['path']);donors=rows(B['selected_donors']['path']);plan=rows(B['input_plan']['path']);contexts=rows(B['context_readiness']['path']);assign=rows(B['assignments']['path'])
 occ=rows(seal['occurrences']['path']);gt=rows(seal['GT']['path']);attempts=rows(seal['feasibility']['path']);donor_cert=rows(seal['donor_certificates']['path']);summary=read(seal['summary']['path'])
 check((len(base),len(donors),len(attempts),len(occ),len(gt))==(28,16,448,924,924),'Full_counts')
 check([a['original_base_row_index']for a in base]==sorted(a['original_base_row_index']for a in base),'Original_base_order')
 oldval=set(read(B['old_val_donor_RGB_shas']['path'])['RGB_SHA256'])
 check(not(oldval&{d['image_sha256']for d in donors}),'Donor_old_VAL_disjoint')
 images=OrderedDict();validated_files=set()
 def load_rgb(b,fmt):
  key=b['path']
  if key not in validated_files:check(sha(key)==b['sha256'],'Native_RGB_file_SHA');validated_files.add(key)
  if key not in images:
   image=np.load(key,allow_pickle=False)if fmt in('NPY','CANONICAL_RGB_UINT8_HWC_NPY')else np.array(Image.open(key).convert('RGB'))
   images[key]=image
   if len(images)>4:images.popitem(last=False)
  return images[key]
 def load_mask(b,hw):
  check(sha(b['path'])==b['sha256'],'Official_owner_mask_SHA');a=np.array(Image.open(b['path']))
  check(a.dtype==np.uint8 and a.shape==tuple(hw)and np.isin(a,[0,255]).all(),'Official_binary_owner_mask');return a>0
 native_donor={};cores={}
 for rank,(d,c)in enumerate(zip(donors,donor_cert)):
  check(d['donor_rank']==c['donor_rank']==rank and d['id']==c['donor_id']and d['object_id']==16 and d['scene_id']in(2,4)and d['split']=='TRAIN'and d['original_split']=='train','Explicit_donor16_TRAIN_scene2or4_lock')
  proof=np.load(c['native_ownership_proof']['path'],allow_pickle=False)
  sel=load_mask(d['own_visible_mask'],d['raw_hw']);others=np.zeros_like(sel)
  check(sha(d['controls']['scene_gt.json']['path'])==d['controls']['scene_gt.json']['sha256'],'Official_scene_GT_metadata_SHA')
  scene=read(d['controls']['scene_gt.json']['path'])[str(d['frame_id'])]
  owners=d['all_visible_owners'];check([a['instance_index']for a in owners]==list(range(len(scene)))and len(scene)==d['complete_scene_GT_owner_count'],'Complete_scene_owner_IDs')
  for owner in owners:
   check(owner['object_id']==scene[owner['instance_index']]['obj_id'],'Official_owner_object_ID')
   if owner['instance_index']!=d['instance_id']:others|=load_mask(owner['visible_mask'],d['raw_hw'])
  ex=sel&~others;core=erosion(ex)
  check(c['complete_owner_instance_set'] is True and c['actual_selected_visible_pixels']==int(sel.sum())==c['expected_visible_pixels'],'Complete_owner_and_visible_pixel_certificate')
  check(all(proof[k].dtype==bool for k in proof.files),'Saved_owner_proof_bool')
  for name,value in [('selected_visible',sel),('other_visible_union',others),('exclusive_alpha',ex),('erode1_exclusive_alpha',core)]:check(np.array_equal(proof[name],value),'Independent_native_ownership_'+name)
  check(not(ex&others).any()and not(core&others).any(),'No_other_owner_opaque_pixels')
  image=load_rgb(d['image'],'PNG');check(pixel(image)==c['decoded_RGB_sha256'],'Donor_decoded_RGB_SHA');native_donor[d['id']]=image;cores[d['id']]=core
  proof.close()
 with np.load(OUT/'RGB96_AND_VALID924_INPUT_ONLY.npz',allow_pickle=False)as z:
  crops=z['rgb96'];valids=z['crop_valid96']
 check(crops.shape==(924,3,96,96)and crops.dtype==np.uint8 and valids.shape==(924,1,96,96)and valids.dtype==bool,'All924_crop_schema')
 bysource=Counter();ready=0;mass_max_error=0.;crop_cache={};alpha_visual=[]
 for i,b in enumerate(base):
  row=b['original_row'];t=targets[i]['original_target'];pm=protected(row,t);s=b['source_lineage'];before=load_rgb(s['fullRGB_file_binding'],s['fullRGB_format'])
  check(pixel(before)==s['decoded_RGB_sha256'],'Original28_native_RGB_identity')
  actual_pm=np.load(OUT/'protected_masks'/f'BASE_{i:02d}.npz',allow_pickle=False)['protected_union']
  check(actual_pm.dtype==bool and np.array_equal(pm,actual_pm),'Independent28_protected_union_exact')
  check(pixel(crops[33*i])==contexts[i]['saved_RGB96_sha256']and pixel(valids[33*i])==contexts[i]['saved_valid96_sha256'],'All28_BEFORE_original_cached_input_hashes')
  for j in range(33):
   n=33*i+j;o,g,p=occ[n],gt[n],plan[n]
   check(all(o[key]==value for key,value in p.items()),'All924_original_plan_fields')
   check(g['occurrence_index']==o['occurrence_index']==n and g['occurrence_id']==o['occurrence_id'],'All924_GT_input_join')
   check(o['query_xy']==row['query_xy']and o['L']==row['L']and o['raw_split']==row['split']and o['native_slot_logical_only']==row['native_slot_logical_only']and o['frozen_context_box_xyxy']==contexts[i]['frozen_context_box_xyxy'],'All924_qL_rawfold_slot_context')
   check(g['native_mapping']=='UNKNOWN'and not g['human_gold']and not g['training_admission'],'No_gold_or_admission_promotion')
   rank=None if j==0 else(j-1)//2;expected_label=0 if j==0 else((1 if j%2 else 0)if attempts[16*i+rank]['pair_ready']else-1)
   check(g['candidate_aux_hidden_label_GT_only']==expected_label,'All924_labels_BEFORE0_ready10_failureUU')
   image=load_rgb(o['fullRGB'],o['format']);check(image.shape==tuple(row['raw_hw'])+(3,)and pixel(image)==o['decoded_RGB_sha256'],'All924_native_decoded_identity')
   key=(o['decoded_RGB_sha256'],tuple(row['query_xy']),row['L'])
   if key not in crop_cache:crop_cache[key]=crop_rgb(image,row['query_xy'],row['L'])
   a,av=crop_cache[key]
   check(np.array_equal(a,crops[n])and np.array_equal(av,valids[n]),'Independent_all924_RGB96_and_validity_exact')
   check(pixel(a)==o['rgb96_sha256']and pixel(av)==o['crop_valid96_sha256'],'All924_saved_crop_byte_hashes')
   if not o['full_rgb_changed']:check(np.array_equal(image,before)and o['fullRGB']==s['fullRGB_file_binding'],'Unchanged_reference_exact_reuse')
  for rank,d in enumerate(donors):
   index=16*i+rank;f=attempts[index];a=assign[index]
   check((f['attempt_index'],f['base_index'],f['donor_rank'],f['donor_id'])==(index,i,rank,d['id'])and a['donor_id']==d['id'],'All448_same16_assignments_order')
   check(f['original_q']==row['query_xy']and f['original_L']==row['L'],'All448_attempt_qL')
   cover_o,sham_o=occ[33*i+1+2*rank],occ[33*i+2+2*rank]
   cover,sham=load_rgb(cover_o['fullRGB'],cover_o['format']),load_rgb(sham_o['fullRGB'],sham_o['format'])
   if not f['pair_ready']:
    check(not cover_o['full_rgb_changed']and not sham_o['full_rgb_changed']and np.array_equal(cover,before)and np.array_equal(sham,before),'All386_failed_attempts_both_U_exact_before')
    continue
   ready+=1;bysource[b['source']]+=1
   with np.load(f['cutout_RGBA_file']['path'],allow_pickle=False)as z:cutrgb,alpha=z['rgb'],z['alpha']
   core=cores[d['id']];yy,xx=np.nonzero(core);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
   check(bbox==f['source_cutout_bbox_xyxy'],'Donor_original_eroded_bbox')
   x0,y0,x1,y1=bbox;h,w=f['resized_cutout_hw'];drgb=native_donor[d['id']]
   expected_rgb=np.asarray(Image.fromarray(drgb[y0:y1,x0:x1]).resize((w,h),Image.Resampling.BILINEAR));expected_alpha=np.asarray(Image.fromarray(core[y0:y1,x0:x1]).resize((w,h),Image.Resampling.NEAREST)).astype(bool)
   check(cutrgb.dtype==np.uint8 and alpha.dtype==bool and np.array_equal(cutrgb,expected_rgb)and np.array_equal(alpha,expected_alpha),'Independent_RGBA_transform_exact')
   maps=[]
   for image,xy in [(cover,f['cover_xy']),(sham,f['sham_xy'])]:
    left,top=xy;ah,aw=alpha.shape;check(0<=left and 0<=top and left+aw<=row['raw_hw'][1]and top+ah<=row['raw_hw'][0],'Unclipped_equal_size_RGBA_placements')
    amap=np.zeros(tuple(row['raw_hw']),bool);amap[top:top+ah,left:left+aw]=alpha;maps.append(amap)
    expected=before.copy();view=expected[top:top+ah,left:left+aw];view[alpha]=cutrgb[alpha]
    check(np.array_equal(expected,image),'Independent_native_compose_exact')
   ca,sa=maps
   check(ca[pm].all()and not sa[pm].any()and np.array_equal(sham[pm],before[pm]),'All62_protected_cover_ALL1_sham_ALL0_RGB_bitexact')
   q,L=row['query_xy'],row['L'];extent=L*47.5/96;ny,nx=np.mgrid[:row['raw_hw'][0],:row['raw_hw'][1]];inside=(nx>=q[0]-extent)&(nx<=q[0]+extent)&(ny>=q[1]-extent)&(ny<=q[1]+extent)
   total=[int(ca.sum()),int(sa.sum())];local=[int((ca&inside).sum()),int((sa&inside).sum())];bil=[mass96(ca,q,L),mass96(sa,q,L)];diff=abs(bil[0]-bil[1]);tol=1e-9+.01*np.mean(bil)
   check(total[0]>0 and total[0]==total[1]==int(alpha.sum())and total==local and min(bil)>0 and diff<=tol,'Independent_native_global_in_crop_and_RGB96_mass')
   cert=f['mass_recheck'];check(cert['native_global_alpha_cells']==total and cert['native_in_crop_alpha_cells']==local and cert['bilinear_RGB96_alpha_pixel_equivalents']==bil and cert['absolute_mass96_difference']==diff and cert['allowed_difference']==tol and cert['PASS'],'Saved_mass_certificate_exact_replay')
   mass_max_error=max(mass_max_error,diff)
 labels=Counter(str(g['candidate_aux_hidden_label_GT_only'])for g in gt);reason=Counter(f['reason']for f in attempts)
 check(ready==summary['accepted_pairs']==62 and 448-ready==summary['failures_U']==386,'Summary_attempt_counts')
 check(dict(labels)==summary['label_counts']=={'0':90,'1':62,'-1':772}and dict(reason)==summary['reasons'],'Summary_label_and_reason_counts')
 result=dict(completed=True,PASS=not FAIL,blockers=FAIL,corpus_seal=bind(OUT/'CORPUS_OUTPUT_SEAL.json'),review_code=bind(__file__),checks=dict(CHECKS),counts=dict(bases=28,donors=16,attempts=448,occurrences=924,accepted_pairs=ready,failed_U_pairs=448-ready,labels=dict(labels),accepted_by_source=dict(bysource),unique_crop_replays=len(crop_cache)),maximum_accepted_RGB96_mass_difference=mass_max_error,independent_math='Own NumPy full-source compose/protected union/central4/native exclusive-alpha/erode1/bilinear RGB96/alpha mass; Pillow exact declared transform. No production builder/kernel imported.',caveats=['This is numerical/provenance consistency, not physical ownership correctness, natural realism, human/native semantic gold or source training admission.','All failed rows, labels and unchanged pixels are replayed. The finite-placement exhaustion cause for failed geometry is retained as producer evidence; no second corpus construction/placement search was executed.','Boundary RGB interpolation may retain color contamination even with exclusive hard alpha; actual fixed image review remains necessary.'],operations=dict(saved_public_native_RGB_masks_and_crops=True,new_construction=0,model_forward_or_fit=0,current223=False,V112=False,PnP=0,source_output_mutations=0,training_admission=False))
 write('REVIEW.json',result);write('SEAL.json',dict(completed=True,PASS=not FAIL,review=bind(HERE/'REVIEW.json'),scope=bind(HERE/'SCOPE.json'),code=bind(__file__)))
 print(json.dumps(dict(PASS=not FAIL,blockers=FAIL,review=bind(HERE/'REVIEW.json'),seal=bind(HERE/'SEAL.json'),counts=result['counts'],checks=sum(CHECKS.values()))))

if __name__=='__main__':main()
