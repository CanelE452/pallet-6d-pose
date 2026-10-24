"""All-row source image construction replay; no model or current reader.
Own target/stencil/ranking/labels; intentionally reuse pinned V2/V1 image math.
"""
from pathlib import Path
from collections import defaultdict,Counter,OrderedDict
import json,hashlib,math,itertools,importlib.util,sys,time
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;B=V/'actual_query_anchored_image_intake';PREP=V/'query_anchored_pair_preparation'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def verify(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(a)for a in x]
 return x
def stencil(q,L):
 axes=[]
 for center in q:
  z=set()
  for a in [center-L/192.,center+L/192.]:
   i=math.floor(a);z.add(i)
   if a!=i:z.add(i+1)
  axes.append(z)
 return sorted(itertools.product(*axes))
def protected(row,regions):
 h,w=row['raw_hw'];mask=np.zeros((h,w),bool)
 for r in regions:
  x0,y0,x1,y1=r['bbox'];lo=(math.floor(x0+.5),math.floor(y0+.5))
  if r['kind']=='PALLOX_CLOSED_NATIVE_UNCERTAINTY':
   assert 0<=x0<=x1<w and 0<=y0<=y1<h;hi=(math.floor(x1+.5)+1,math.floor(y1+.5)+1)
  else:
   assert r['kind']in ['LOCO_HALFOPEN_SUPPORT','LOCO_HALFOPEN_UNCERTAINTY']
   assert -.5<=x0<x1<=w-.5 and -.5<=y0<y1<=h-.5;hi=(math.ceil(x1+.5),math.ceil(y1+.5))
  mask[lo[1]:hi[1],lo[0]:hi[0]]=True
 for x,y in stencil(row['query_xy'],row['L']):
  assert 0<=x<w and 0<=y<h;mask[y,x]=True
 return mask
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 assert not(H/'REVIEW.json').exists(),'No overwrite'
 seal=read(B/'IMAGE_ONLY_INTAKE_SEAL.json');assert bind(B/'IMAGE_ONLY_INTAKE_SEAL.json')['sha256']=='86a845cfacb56d8817cfca05502f58f5b76470b248b1a61ca7fededa695ef828'
 for f in seal['files']:verify(f)
 proto=read(verify(seal['protocol']));D=proto['bindings'];rows=lines(verify(D['rows960']));source=lines(verify(D['source960']));targets=lines(verify(D['target_GT']));assign=lines(verify(D['assignments']));pool=lines(verify(D['donor_pool']));plan=lines(verify(D['input_plan']))
 occs=lines(B/'OCCURRENCES31680_LINEAGE.jsonl');gt=lines(B/'OCCURRENCES31680_GT_ONLY.jsonl');facts=lines(B/'FEASIBILITY15360_GT_ONLY.jsonl');certs=lines(B/'SELECTED_DONOR_RGBA_CERTIFICATES.jsonl')
 inv=lines(V/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl');ip=read(V/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json');teachers={t['cue_view_index']:t for t in lines(verify(ip['inputs']['original_cue_views604']))};loco=lines(verify(ip['inputs']['loco_root160']))
 assert len(rows)==len(source)==len(targets)==len(assign)==960 and len(occs)==len(gt)==len(plan)==31680 and len(facts)==15360
 assert bind(B/'SOURCE960_DRIVER_LINEAGE.jsonl')['sha256']==D['source960']['sha256']
 donors={d['id']:d for d in pool};assert len(donors)==4200
 # Independently rank every base's16 donor choices; no outcome-driven replacement.
 expected_regions={};target_masks={};needed=set()
 for i,(r,t,a,x)in enumerate(zip(rows,targets,assign,inv)):
  assert r['slot_index']==t['base_row_index']==a['base_row_index']==x['row_index']==i
  assert t['query_anchored_label']==x['query_anchored_AUX_label_D0_E1_Uminus1'] and t['original_C1_V95_label']==x['original_C1_V95_label_D0_E1_Uminus1']
  assert t['original_q']==r['query_xy']==x['original_q'] and t['original_L']==r['L']==x['original_L'] and t['raw_split']==r['split']==x['raw_split']
  group=x['matched_certified_group_GT_only'];regions=[]
  if t['query_anchored_label']>=0:
   if i<160:regions=[dict(kind='LOCO_HALFOPEN_SUPPORT',bbox=loco[i]['support_bbox_xyxy']),dict(kind='LOCO_HALFOPEN_UNCERTAINTY',bbox=loco[i]['uncertainty_bbox_xyxy'])]
   else:
    ge=x['group_evidence_GT_only'][group];coverage=dict(zip(ge['views'],ge['original_coverage']))
    regions=[dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=teachers[k]['uncertainty_bbox_xyxy_half_open_unchanged'],cue_id=teachers[k]['cue_id'],cue_view_index=k,original_reference_xy=teachers[k]['reference_xy_unchanged'],original_coverage=coverage[k])for k in x['source_cue_view_indices']if teachers[k]['duplicate_group_link_only']==group]
  assert t['original_regions']==regions and t['matched_certified_group']==group
  assert t['construction_eligible']==(t['query_anchored_label']==0) and t['natural_E_before_only']==(t['query_anchored_label']==1)
  split='TRAIN'if r['split']in ['TRAIN','TRAIN_LAB']else 'VALIDATION';train=split=='TRAIN'
  eligible=[d for d in pool if d['split']==split and d['image_sha256']!=r['image_sha256'] and d['object_id']%2==int(train) and 1<=d['object_id']<=21 and ((0<=d['scene_id']<=7)if train else d['scene_id']in [8,9])]
  def key(d):return hashlib.sha256('|'.join(map(str,[20261023,'V96_QUERY_ANCHORED_CUE_K16_V1',r['input_id'],r['native_slot_logical_only'],d['id'],d['image_sha256']])).encode()).hexdigest()
  ranked=sorted(eligible,key=key)[:16]
  assert len({d['id']for d in ranked})==16
  for k,(d,a0)in enumerate(zip(ranked,a['ranks'])):
   assert a0['donor_rank']==k and a0['donor_id']==d['id'] and a0['ranking_SHA256']==key(d)
  if t['construction_eligible']:
   target_masks[i]=protected(r,regions);assert t['central4_native_stencil']==[list(p)for p in stencil(r['query_xy'],r['L'])]
   with np.load(B/'protected_masks'/f'base_{i:04d}.npz',allow_pickle=False)as z:assert np.array_equal(z['protected_union'],target_masks[i]),i
   needed.update(d['id']for d in ranked)
 assert len(target_masks)==61 and Counter(t['query_anchored_label']for t in targets)=={0:61,1:4,-1:895}
 # All identity/qL/rawfold and input/GT whitelist joins, including every U.
 forbidden={'candidate_aux_hidden_label_GT_only','query_anchored_before_label','original_C1_V95_label','known_candidate_GT_only','pair_ready'}
 for oi,(o,g,a)in enumerate(zip(occs,gt,plan)):
  i,j=divmod(oi,33);k=None if j==0 else(j-1)//2;v='BEFORE'if j==0 else'OPAQUE_COVER'if j%2 else'CUE_PRESERVING_SHAM'
  assert o['occurrence_index']==g['occurrence_index']==oi and o['base_row_index']==g['base_row_index']==o['slot_index']==g['slot_index']==i
  assert o['donor_rank']==g['donor_rank']==k and o['variant']==g['variant']==v and o['occurrence_id']==g['occurrence_id']==a['occurrence_id']
  for field,val in a.items():
   if field!='image_status':assert o[field]==val,(oi,field)
  assert not(forbidden&set(o)) and g['query_anchored_before_label']==targets[i]['query_anchored_label'] and g['original_C1_V95_label']==targets[i]['original_C1_V95_label']
  assert o['query_xy']==rows[i]['query_xy'] and o['L']==rows[i]['L'] and o['raw_split']==rows[i]['split']
  f=None if j==0 else facts[16*i+k];ready=False if j==0 else bool(f['pair_ready'])
  label=targets[i]['query_anchored_label']if j==0 else (1 if j%2 else 0)if ready else -1
  assert g['pair_ready']==ready and g['candidate_aux_hidden_label_GT_only']==label and g['known_candidate_GT_only']==(label>=0)
  assert g['native_expected_part_state']=='UNKNOWN'and not g['human_gold']and not g['training_admission']
  same=o['full_rgb_decoded_sha256']==source[i]['decoded_RGB_sha256'];assert o['full_rgb_changed']==(not same)
  if j==0 or not ready:assert same
  if targets[i]['query_anchored_label']==1:assert j==0 or (label==-1 and same)
 # Actual complete BOP owner sets and all saved alpha proofs independently replayed.
 owners=read(verify(D['complete_instances']))['frames'];frames={(f['scene_id'],f['frame_id']):f for f in owners};maskcache=OrderedDict();rgbcache=OrderedDict();verified=set();maskpaths=set()
 def mask(b):
  p=str(verify(b)) if b['path']not in verified else b['path'];verified.add(p);maskpaths.add(p)
  if p not in maskcache:
   z=np.array(Image.open(p));assert z.dtype==np.uint8 and np.isin(z,[0,255]).all();maskcache[p]=z>0
   if len(maskcache)>48:maskcache.popitem(last=False)
  else:maskcache.move_to_end(p)
  return maskcache[p]
 def rgb(b,fmt='PNG'):
  p=str(verify(b))if b['path']not in verified else b['path'];verified.add(p)
  if p not in rgbcache:
   a=np.load(p,allow_pickle=False)if fmt in ['CANONICAL_RGB_UINT8_HWC_NPY','NPY']else np.array(Image.open(p).convert('RGB'),copy=True)
   assert a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3;rgbcache[p]=a
   if len(rgbcache)>4:rgbcache.popitem(last=False)
  else:rgbcache.move_to_end(p)
  return rgbcache[p]
 for frame in owners:
  native=read(verify(frame['scene_GT']))[str(frame['frame_id'])]
  assert frame['available_instance_ids']==frame['expected_instance_ids']==list(range(len(native)))
  assert [m['instance_id']for m in frame['instance_masks']]==list(range(len(native))) and frame['complete']
  assert all(m['object_id']==native[m['instance_id']]['obj_id']for m in frame['instance_masks'])
 certby={c['id']:c for c in certs};assert set(certby)==needed
 def alpha(d):
  f=frames[d['scene_id'],d['frame_id']];selected=mask(d['visible_alpha_source']);other=np.zeros_like(selected)
  for m in f['instance_masks']:
   if m['instance_id']!=d['instance_id']:other|=mask(m['mask'])
  ex=selected&~other;pad=np.pad(ex,1);core=np.logical_and.reduce([pad[y:y+ex.shape[0],x:x+ex.shape[1]]for y in range(3)for x in range(3)])
  return selected,other,ex,core
 for id,c in certby.items():
  d=donors[id];sel,other,ex,core=alpha(d);assert sel.sum()==d['metadata_vis_pixels']==c['metadata_visible_pixels']
  with np.load(verify(c['native_alpha_proof']),allow_pickle=False)as z:
   for k,a in [('selected_visible',sel),('other_visible_union',other),('exclusive_alpha',ex),('erode1_exclusive_alpha',core)]:assert np.array_equal(z[k],a),(id,k)
  ec=c['exclusive_alpha'];assert ec['original_selected_visible_pixels']==int(sel.sum()) and ec['original_conflicting_pixels']==int((sel&other).sum()) and ec['excluded_conflicting_pixels']==int((sel&other).sum()) and ec['retained_exclusive_alpha_pixels']==int(ex.sum()) and ec['retained_erode1_pixels']==int(core.sum())
  assert ec['admitted']==bool(core.any())==c['foreign_opaque_cutout_admitted']and not(ex&other).any()and not(core&~sel).any()
 # Reuse the pin-verified image math, not B's protected/prepare/selection functions.
 kernel=module(verify(D['kernel']),'audit_pinned_math_access');V2,V1=kernel.V2,kernel.V1
 attempted=success=changed=0;accepted=[];reasons=Counter();begin=time.monotonic()
 for i,t in enumerate(targets):
  if not t['construction_eligible']:
   for k in range(16):assert facts[16*i+k]['reason']=='QUERY_ANCHORED_BEFORE_NOT_D_NO_EDIT'and not facts[16*i+k]['pair_ready']
   continue
  row=rows[i];pm=target_masks[i];native=rgb(source[i]['fullRGB_file_binding'],source[i]['fullRGB_format']);assert ash(native)==source[i]['decoded_RGB_sha256']
  for k,assigned in enumerate(assign[i]['ranks']):
   attempted+=1;d=donors[assigned['donor_id']];f=facts[16*i+k];sel,other,ex,core=alpha(d)
   assert f['base_row_index']==i and f['donor_rank']==k and f['assigned_donor_id']==d['id'] and f['original_q']==row['query_xy'] and f['original_L']==row['L']
   assert f['protected_pixels']==int(pm.sum()) and f['complete_original_regions_count']==len(t['original_regions'])
   assert f['protected_area_over_L_squared']==float(pm.sum()/row['L']**2)
   pair=None;cut=None
   if not core.any():reason='ASSIGNED_DONOR_METADATA_NOT_ADMITTED_U'
   else:
    drgb=rgb(d['image']);assert ash(drgb)==certby[d['id']]['decoded_RGB_sha256']
    cut,scale=V2.support_scaled_cutout(drgb,ex,pm,row['L']);assert clean(scale)==f['scale'],(i,k,'scale')
    if cut is None:reason=scale['reason']
    else:
     normalized=dict(row,split='TRAIN'if row['split']in ['TRAIN','TRAIN_LAB']else'VALIDATION')
     pair,mass=V2.core_aligned_pair(normalized,cut,pm);assert clean(mass)==f['mass'],(i,k,'mass')
     reason=mass['reason']if pair is None else'QUERY_ANCHORED_PAIR_READY'
   assert reason==f['reason'] and (pair is not None)==bool(f['pair_ready']),(i,k,reason,f['reason'])
   reasons[reason]+=1
   if pair is None:continue
   success+=1;(cxy,ca),(sxy,sa)=pair
   assert ca[pm].all()and not sa[pm].any()and ca.sum()==sa.sum()==cut['alpha'].sum()
   assert list(cxy)==f['cover_xy']and list(sxy)==f['sham_xy']
   for offset,xy,am in [(1,cxy,ca),(2,sxy,sa)]:
    oi=33*i+2*k+offset;o=occs[oi];expected=V1.compose(native,cut,xy,am);saved=rgb(o['edited_fullRGB_file'],o['edited_fullRGB_format'])
    assert o['full_rgb_changed']and np.array_equal(saved,expected)and ash(saved)==o['full_rgb_decoded_sha256'],(i,k,offset)
    assert np.array_equal(saved[~am],native[~am])
    if offset==2:assert np.array_equal(saved[pm],native[pm])
    changed+=1
   accepted.append(dict(base_row_index=i,donor_rank=k,source=t['source_family'],raw_split=row['split'],protected_pixels=int(pm.sum()),cover_alpha=int(ca.sum()),sham_alpha=int(sa.sum()),RGB_recompose_exact=True))
  print(json.dumps(dict(base=i,attempted=attempted,accepted=success,seconds=time.monotonic()-begin)),flush=True)
 assert attempted==976 and success==194 and changed==388
 stats=read(B/'IMAGE_ONLY_SUMMARY.json');assert stats['accepted_pairs']==success and stats['attempt_reasons']==dict(reasons)
 labels=Counter(str(g['candidate_aux_hidden_label_GT_only'])for g in gt);assert labels=={'0':255,'1':198,'-1':31227}==stats['all_GT_labels']
 report=dict(completed=True,PASS=True,blockers=[],intake_seal=bind(B/'IMAGE_ONLY_INTAKE_SEAL.json'),independent_code=bind(__file__),scope=bind(H/'SCOPE_BEFORE_ACTUAL_SAVED_REPLAY.json'),seal_file_bindings_verified=len(seal['files']),base_rows=960,all_occurrences=31680,all_rank_records=15360,eligible_rank_attempts=attempted,accepted_pairs=success,changed_RGB_recompositions_exact=changed,
 original_before_labels={'D':61,'E':4,'U':895},all_labels_D0_E1_Uminus1=dict(labels),donors=len(certs),complete_owner_frames=len(owners),actual_visible_mask_files_checked=len(maskpaths),protected_masks_and_stencils_exact=61,all960_rank_minSHA_selections_exact=True,no_rank_replacement=True,all31680_q_L_rawfold_order_candidate_unchanged=True,canonical_E4_BEFORE_only=True,GT_and_input_jobs_separate=True,
 by_source_raw_split={s+'|'+sp:sum(a['source']==s and a['raw_split']==sp for a in accepted)for s,sp in sorted({(t['source_family'],t['raw_split'])for t in targets})},attempt_reasons=dict(reasons),
 independence='Own source604-range/stencil union, SHA ranking, owner-mask subtraction and allrowlabel joins. Pinned unchanged V2 scale/placement/mass and V1 compositor reused explicitly. Did not call B prepare_rank/protected_mask/main.',
 limitations=['Native expectedpart/corner association UNKNOWN; conditional AI cue labels, not human gold.','Donor RGB bilinear boundary may mix nearby native pixels; exclusive native alpha guarantees known mask ownership, not calibrated optical opacity gold.','All accepted controls Pallox-only; flat-pallet controlled generalization remains untested.'],source_pixels_used_only_for_saved_construction_replay=True,new_model_feature_fit_current_PnP=0,training_admission=False)
 with(H/'ALL194_ACCEPTED_REPLAY.jsonl').open('x')as f:
  for a in accepted:f.write(json.dumps(a)+'\n')
 write(H/'REVIEW.json',report);write(H/'SEAL.json',dict(completed=True,PASS=True,files=[bind(p)for p in sorted(H.iterdir())if p.is_file()]))
 print(json.dumps(dict(PASS=True,report=bind(H/'REVIEW.json'),seal=bind(H/'SEAL.json'))))
if __name__=='__main__':main()
