from pathlib import Path
from collections import Counter
import json,hashlib,datetime,csv,sys
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_flat_pallet_cue_bridge_20261023_v92');O=R/'independent_actual_rgb96_audit';D=R/'rgb96_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def b(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def ck(x):assert sha(x['path'])==x['sha256'],x['path']
scope=read(O/'SCOPE_AND_FIXED_SAMPLE_BEFORE_PIXELS.json');[ck(x)for x in scope['bindings'].values()]
seal=read(D/'RGB96_REVIEW_SEAL.json');assert seal['completed']and seal['PASS'];assert sha(D/'RGB96_REVIEW_SEAL.json')=='56c0b7036bc1dc4d6c2a74c7db983e645c4229343279820093d1397830abc533'
for k in ('entry','policy','protocol','code'):ck(seal[k])
entry=read(seal['entry']['path']);assert entry['approved']and entry['source_only']and not any(entry[k]for k in ('model_forward_allowed','training_allowed','current_allowed','PnP_allowed'))
for k in ('policy','producer_protocol','code','generated','independent_review'):ck(entry[k])
assert read(entry['independent_review']['path'])['PASS']
assert entry['policy']==seal['policy']and entry['producer_protocol']==seal['protocol']and entry['code']==seal['code']
files=seal['files'];assert len(files)==24 and len({v['path']for v in files})==24
for f in files:ck(f)
assert {str(p.resolve())for p in D.rglob('*')if p.is_file()}=={v['path']for v in files}|{str((D/'RGB96_REVIEW_SEAL.json').resolve())}
bytes_excluding_seal=sum(Path(v['path']).stat().st_size for v in files);assert bytes_excluding_seal==seal['bytes_before_seal']
actualbytes=bytes_excluding_seal+(D/'RGB96_REVIEW_SEAL.json').stat().st_size;assert actualbytes<100000000
protocol=read(seal['protocol']['path']);frames=lines(protocol['input_manifest']['path']);assert len(frames)==20
rows=lines(D/'ROWS_INPUT_ONLY.jsonl');prov=read(D/'RGB_PROVENANCE_AND_FIGURES.json');assert len(rows)==160 and prov['completed']and len(prov['frames'])==20
with np.load(D/'INPUT_RGB96_ONLY.npz',allow_pickle=False)as z:a={k:z[k]for k in z.files}
expected={'rgb96':((160,3,96,96),'uint8'),'crop_valid96':((160,1,96,96),'bool'),'query_xy':((160,2),'float64'),'L':((160,),'float64'),'query_exists':((160,),'bool'),'query_continuous_valid':((160,),'bool'),'query_nearest_valid':((160,),'bool'),'query_center_valid':((160,),'bool'),'crop_anyvalid':((160,),'bool'),'native_slot':((160,),'int8'),'frame_index':((160,),'int32'),'input_id':((160,),None),'split':((160,),None)}
assert set(a)==set(expected)
for k,(shape,dtype)in expected.items():assert a[k].shape==shape and(dtype is None or str(a[k].dtype)==dtype),(k,a[k].shape,a[k].dtype)
assert a['input_id'].dtype.kind=='U'and a['split'].dtype.kind=='U'
counts=Counter();native_valid_points=0;zero_missing=0;bytebindings=[]
u=(np.arange(96,dtype=np.float64)+.5)/96-.5
for i,f in enumerate(frames):
 assert f['frame_index']==i;H,W=f['raw_hw'];p=prov['frames'][i]
 assert p['frame_index']==i and p['input_id']==f['input_id']and p['source_RGB']==f['decoded_RGB_input']and p['original_JPEG_SHA256']==f['image_sha256']and p['decoded_pixel_SHA256']==f['RGB_pixels_sha256']and p['selected_index']==f['selected_index']and p['q_L_unchanged']is True
 # Byte-only checks; these do not deserialize images or NPY pixels.
 ck(f['decoded_RGB_input']);assert sha(f['image'])==f['image_sha256'];ck(p['figure'])
 bytebindings.append({'frame_index':i,'original_JPEG_sha256':f['image_sha256'],'canonical_NPY':f['decoded_RGB_input'],'canonical_pixel_sha256_metadata':f['RGB_pixels_sha256']})
 for k in range(8):
  j=i*8+k;r=rows[j];missing=f['selected_index']is None
  assert int(a['frame_index'][j])==i and int(a['native_slot'][j])==k and str(a['input_id'][j])==f['input_id']and str(a['split'][j])==f['split']
  assert r['frame_index']==i and r['native_slot']==k and r['input_id']==f['input_id']and r['split']==f['split']and r['image_sha256']==f['image_sha256']
  assert r['semantic_state']=='UNLABELED_UNKNOWN'and r['native8_semantic_mapping']=='UNKNOWN'and r['new_training_label']is False
  assert bool(a['query_exists'][j])==r['query_exists']==(not missing)
  if missing:
   assert np.isnan(a['query_xy'][j]).all()and np.isnan(a['L'][j])and r['query_xy']is None and r['L']is None
   assert not a['rgb96'][j].any()and not a['crop_valid96'][j].any();zero_missing+=1
   cont=nn=cv=anyvalid=False;v=np.zeros((96,96),bool);assert r['input_status']=='NO_DETECTION_QUERY_U'
  else:
   q=np.asarray(f['query8_xy'][k],np.float64);L=float(f['L'])
   assert np.array_equal(a['query_xy'][j],q)and float(a['L'][j])==L and r['query_xy']==q.tolist()and r['L']==L and np.isfinite(q).all()
   x,y=q;rx,ry=np.floor(q+.5);cont=bool(0<=x<W and 0<=y<H);nn=bool(0<=rx<W and 0<=ry<H);cv=cont and nn
   xs=q[0]+u*L;ys=q[1]+u*L;v=(ys[:,None]>=-.5)&(ys[:,None]<H-.5)&(xs[None,:]>=-.5)&(xs[None,:]<W-.5);anyvalid=bool(v.any())
   assert r['input_status']==('SAVED_UNLABELED_QUERY'if cv else'OUTSIDE_OR_INVALID_QUERY_U')
  for key,val in [('query_continuous_valid',cont),('query_nearest_valid',nn),('query_center_valid',cv),('crop_anyvalid',anyvalid)]:assert bool(a[key][j])==val,(j,key)
  assert r['continuous_valid']==cont and r['nearest_valid']==nn and r['query_center_valid']==cv and r['crop_anyvalid']==anyvalid
  assert np.array_equal(a['crop_valid96'][j,0],v)and r['crop_valid_fraction']==float(v.mean())
  assert not a['rgb96'][j].transpose(1,2,0)[~v].any()
  counts[(f['split'],'NO_DETECTION'if missing else'CENTER_VALID'if cv else'CENTER_INVALID')]+=1;native_valid_points+=int(v.sum())
assert zero_missing==16 and a['query_exists'].sum()==144 and a['query_center_valid'].sum()==122
assert not np.isnan(a['query_xy'][a['query_exists']]).any()and np.isfinite(a['L'][a['query_exists']]).all()
# Recreate frozen min-SHA choice from metadata again, without consulting output pixels.
allslots=[]
for (split,state),num in counts.items():assert num==scope['all160_metadata_stratum_counts'][str((split,state))]
for i,f in enumerate(frames):
 for k in range(8):
  q=None if f['query8_xy']is None else f['query8_xy'][k];h,w=f['raw_hw']
  if q is None:state='NO_DETECTION'
  else:
   rx,ry=np.floor(np.asarray(q)+.5);state='CENTER_VALID'if 0<=q[0]<w and 0<=q[1]<h and 0<=rx<w and 0<=ry<h else'CENTER_INVALID'
  allslots.append({'frame_index':i,'input_id':f['input_id'],'native_slot':k,'split':f['split'],'input_state':state,'selection_sha256':hashlib.sha256(f"V92_ACTUAL_RGB96_REPLAY_V1\n{f['input_id']}\n{k}".encode()).hexdigest()})
groups={}
for s in allslots:groups.setdefault((s['split'],s['input_state']),[]).append(s)
ss=[min(groups[key],key=lambda s:s['selection_sha256'])for key in sorted(groups)]
for s in sorted(allslots,key=lambda s:s['selection_sha256']):
 if len(ss)>=8:break
 if s not in ss:ss.append(s)
assert ss==scope['fixed8']
# Independent scalar F64 bilinear; no production crop helper import.
def pixel(rgb,q,L,iy,ix):
 h,w=rgb.shape[:2];x=q[0]+((ix+.5)/96-.5)*L;y=q[1]+((iy+.5)/96-.5)*L
 if not(-.5<=x<w-.5 and-.5<=y<h-.5):return [0,0,0]
 x0=int(np.floor(x));y0=int(np.floor(y));dx=x-x0;dy=y-y0;acc=[0.,0.,0.]
 for ax,ay,wt in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  xx=x0+ax;yy=y0+ay
  if 0<=xx<w and 0<=yy<h:
   for ch in range(3):acc[ch]+=float(rgb[yy,xx,ch])*wt
 return [int(np.floor(min(255.,max(0.,x))+.5))for x in acc]
source_arrays={};replays=[];pixels=0
for sample in ss:
 i=sample['frame_index'];k=sample['native_slot'];j=i*8+k;f=frames[i]
 if sample['input_state']=='NO_DETECTION':
  assert not a['rgb96'][j].any();replays.append(dict(sample,source_pixels_read=False,RGB_mismatches=0,valid_mismatches=0));continue
 if i not in source_arrays:
  rgb=np.load(f['decoded_RGB_input']['path'],allow_pickle=False);assert rgb.dtype==np.uint8 and rgb.shape==tuple(f['raw_hw'])+(3,)and hashlib.sha256(rgb.tobytes()).hexdigest()==f['RGB_pixels_sha256'];source_arrays[i]=rgb
 rgb=source_arrays[i];q=np.asarray(f['query8_xy'][k],np.float64);L=float(f['L']);out=np.empty((96,96,3),np.uint8)
 for iy in range(96):
  for ix in range(96):out[iy,ix]=pixel(rgb,q,L,iy,ix)
 nm=int(np.count_nonzero(out!=a['rgb96'][j].transpose(1,2,0)));assert nm==0,(i,k,nm)
 pixels+=96*96;replays.append(dict(sample,source_pixels_read=True,RGB_mismatches=nm,valid_mismatches=0,RGB_pixel_positions=96*96,canonical_RGB_pixel_SHA256=f['RGB_pixels_sha256']))
with(O/'FIXED8_REPLAY.json').open('x')as f:json.dump({'PASS':True,'scope':b(O/'SCOPE_AND_FIXED_SAMPLE_BEFORE_PIXELS.json'),'fixed8':replays,'source_NPY_arrays_deserialized':len(source_arrays),'source_RGB_pixel_positions_replayed':pixels},f,ensure_ascii=False,indent=2);f.write('\n')
report={'schema':'V92_INDEPENDENT_ACTUAL_SAVED_RGB96_AUDIT_V1','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'bindings':{'scope':b(O/'SCOPE_AND_FIXED_SAMPLE_BEFORE_PIXELS.json'),'code':b(__file__),'producer_seal':b(D/'RGB96_REVIEW_SEAL.json'),'producer_protocol':seal['protocol'],'producer_code':seal['code'],'root_entry':seal['entry'],'input_manifest':protocol['input_manifest'],'pack':b(D/'INPUT_RGB96_ONLY.npz'),'rows':b(D/'ROWS_INPUT_ONLY.jsonl'),'provenance':b(D/'RGB_PROVENANCE_AND_FIGURES.json'),'fixed_replay':b(O/'FIXED8_REPLAY.json')},
 'full_saved_audit':{'frames':20,'slots':160,'selected_frames':18,'predicted_q_slots':144,'no_detection_frames':2,'no_detection_slots_U':16,'center_valid_q':122,'center_invalid_supplied_q_U':22,'split_frames':{'TRAIN':12,'VALIDATION':8},'split_slot_states':{str(k):v for k,v in counts.items()},'all_q_L_order_id_slot_split_exact':True,'all_NPZ_JSONL_flags_and_status_exact':True,'all_160_valid96_geometry_exact':True,'valid96_pixel_positions':native_valid_points,'total96_pixel_positions':160*96*96,'all_invalid_RGB_and_missing_placeholders_zero':True,'all_semantic_state_UNLABELED_UNKNOWN':True,'all_native8_mapping_UNKNOWN':True,'file_bindings_checked':len(files),'all_sealed_file_SHAs_and_fileinventory_exact':True,'bytes_before_seal':bytes_excluding_seal,'total_bytes_including_seal':actualbytes,'budget_bytes':100000000,'canonical_NPY_bytes_SHAs_checked':20,'original_JPEG_bytes_SHAs_checked_without_decode':20,'canonical_pixel_SHA_lineage_metadata_exact':20},
 'fixed_source_pixel_replay':{'frozen_sample_slots':8,'no_detection_zero_slots_no_source_pixels':2,'source_query_crops_replayed':6,'unique_public_canonical_NPY_arrays_deserialized':len(source_arrays),'RGB_pixel_positions_all3channels':pixels,'RGB_channel_values_compared':pixels*3,'RGB_uint8_mismatches':0,'valid_mask_mismatches':0,'crop_code_import_or_producer_rerun':False,'scalar_formula':'q+((j+.5)/96-.5)*L; F64 bilinear exterior0; floor(clip0..255+.5); halfopen[-.5,W-.5) valid'},
 '한국어_판정':'실제 저장 RGB96 pack 독립 감사 PASS. 160점의 q/L·순서·split·U·validmask와 NPZ/JSONL/lineage가 일치했습니다. 사전고정8점 중 검출된6점의 실제 canonical RGB96 전체를 독립 scalar 산술로 재생하여 차이0을 확인했습니다.',
 'limitations':['실제 source pixel 재생은 사전고정8점 중 query가 있는6점/6개 canonical NPY에 한정됩니다. 나머지 RGB crop 값의 실제 픽셀 재생을 전수 수행했다고 주장하지 않습니다.','20개 canonical NPY와 원JPEG raw-byteSHA는 전수 검사했지만 JPEG는 재decode하지 않았고 pixelSHA 실측은 고정6개만 수행했습니다.','저장된 figure/contact의 파일SHA만 검산했고 이 감사에서는 직접 시각 검토하지 않았습니다.','입력 정확성 감사이며 target-instance correspondence, physical D/E/S 또는 native8 의미 gold를 인증하지 않습니다.','TRAIN12/VAL8은 기존 subset split이며 촬영장 독립성이나 완전 목표 해결을 보증하지 않습니다.'],
 'producer_files_modified':0,'producer_rerun':0,'GT_bbox_mask_cue_labels_read':0,'P3_or_model_weights_read':0,'current_data_read':0,'model_or_head_forward':0,'training':0,'PnP':0,'model_training_admission':False,'fully_solved':False,'goal_status':'active'}
with(O/'REVIEW.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
with(O/'SEAL.json').open('x')as f:json.dump({'completed':True,'PASS':True,'review':b(O/'REVIEW.json'),'scope':b(O/'SCOPE_AND_FIXED_SAMPLE_BEFORE_PIXELS.json'),'code':b(__file__),'fixed_replay':b(O/'FIXED8_REPLAY.json'),'source_producer_seal':b(D/'RGB96_REVIEW_SEAL.json'),'producer_unchanged':True},f,indent=2);f.write('\n')
print(json.dumps({'PASS':True,'review':b(O/'REVIEW.json'),'seal':b(O/'SEAL.json'),'source_frames_pixel_replayed':len(source_arrays),'RGB_pixel_positions':pixels,'valid96_positions':native_valid_points,'bytes':actualbytes},ensure_ascii=False))
