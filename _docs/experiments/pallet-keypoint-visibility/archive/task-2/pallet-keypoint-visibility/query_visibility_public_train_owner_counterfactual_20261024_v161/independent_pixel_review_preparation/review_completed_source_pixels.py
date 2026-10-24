"""Independent V161 saved pixel/input review; no builder imports or learned calls.
--prepare-only reads code/metadata + invented arrays only. Actual saved/source
NPY/NPZ reads require --builder-completed-trigger root JSON, even for failures.
"""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, math, sys, traceback
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from independent_sampling_math import crop96,context224,center4_footprints,center4_from_footprints,generated_checks
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
BH=V/'pixel_builder_preparation';OUT=V/'source_pixels_v1';R=V/'independent_pixel_review'
DESIGN=V/'ROOT_SOURCE_GENERATION_DESIGN_BEFORE_PIXELS.json';DSHA='25f38eb0e8b098b681a8bb38e5a3f93086a1a24cdc9fdfa8f3300412382c332b'
CODE_SHA='835c2df24a1cad8bef97947bada0c28a2e8411b3115b757156bdaaa3d7a10875';PSHA='9d3341b69b63c375577f7e72967d5f35ef81e853953b3a95a2e73c71bc62f907'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM');BINDINGS=[]
def safe(p):
 p=Path(p).resolve();s=str(p).lower();assert p.is_relative_to(P) and not any(x in s for x in ('_v112','/private_evaluation/','current223','/current_evaluation/'));return p
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=safe(p);assert p.suffix not in ('.pt','.pth','.pkl');b=dict(path=str(p),sha256=sha(p));BINDINGS.append(b);return b
def verify(b):assert bind(b['path'])['sha256']==b['sha256'],b['path'];return safe(b['path'])
def read(p):return json.loads(safe(p).read_text())
def lines(p):return [json.loads(s) for s in safe(p).read_text().splitlines() if s]
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def erode(a):return np.lib.stride_tricks.sliding_window_view(np.pad(a,1,constant_values=False),(3,3)).all(axis=(-1,-2))
def transform(rgb,alpha,longside):
 h,w=alpha.shape;ratio=max(1,int(longside))/max(h,w);hh=max(1,int(math.floor(h*ratio+.5)));ww=max(1,int(math.floor(w*ratio+.5)))
 return np.asarray(Image.fromarray(rgb).resize((ww,hh),Image.Resampling.BILINEAR)).copy(),erode(np.asarray(Image.fromarray(alpha).resize((ww,hh),Image.Resampling.NEAREST)).astype(bool))
def geometry(q,L,alpha,hw):
 if not alpha.any():return None,{'reason':'TRANSFORMED_ALPHA_EMPTY'}
 distance=distance_transform_edt(np.pad(alpha,1,constant_values=False))[1:-1,1:-1];y,x=np.unravel_index(int(np.argmax(distance)),alpha.shape);anchor=[int(x),int(y)]
 cover=np.rint(np.asarray(q,np.float64)-anchor).astype(np.int64);direction=1 if q[0]<=(hw[1]-1)/2 else -1;delta=alpha.shape[1]+math.ceil(L/96)+4;sham=cover+[direction*delta,0]
 facts=dict(opaque_anchor_xy=anchor,cover_origin_xy=cover.tolist(),sham_origin_xy=sham.tolist(),horizontal_direction=direction,sham_shift_pixels=delta,transformed_hw=list(alpha.shape))
 if not all(0<=xy[0] and xy[0]+alpha.shape[1]<=hw[1] for xy in (cover,sham)):return None,dict(**facts,reason='X_BOX_NOT_FULLY_IN_BOUNDS')
 return (cover,sham),facts

def paste(original,rgb,alpha,origin):
 result=original.copy();mask=np.zeros(original.shape[:2],bool);x,y=map(int,origin);h,w=alpha.shape;assert 0<=x and x+w<=original.shape[1]
 # Independent indexed composition, including identical vertical clipping.
 yy,xx=np.nonzero(alpha);gy=yy+y;gx=xx+x;good=(gy>=0)&(gy<original.shape[0]);result[gy[good],gx[good]]=rgb[yy[good],xx[good]];mask[gy[good],gx[good]]=True
 return result,mask

def expected_views(rgb,frame):
 local=np.zeros((8,96,96,3),np.uint8);valid=np.zeros((8,96,96),bool)
 for k in range(8):
  if frame['feature_constructed'][k]:local[k],valid[k]=crop96(rgb,frame['query_xy'][k],frame['L'][k])
 target,tv=context224(rgb,frame['context_box_xyxy']);fractions=np.asarray([[np.float32(tv.mean()),np.float32(v.mean()),np.float32(v[32:64,32:64].mean())] for v in valid],np.float32)
 return dict(rgb96=local,valid96=valid,central32=local[:,32:64,32:64].copy(),target224=target,target_valid=tv,fractions=fractions,central4_available=valid[:,47:49,47:49].all((1,2)),context_available=np.full(8,tv.any(),bool))
def check_variant(z,rgb,alpha,fr,keys):
 assert set(z)==set(keys) and z['full_RGB'].dtype==np.uint8 and exact(z['full_RGB'],rgb) and exact(z['pasted_alpha'],alpha)
 want=expected_views(rgb,fr)
 for key,a in want.items():assert exact(z[key],a),key
 for key,field,dtype in [('query_xy','query_xy',np.float64),('L','L',np.float64),('native_slot','native_slot',np.int8),('crop_constructed','feature_constructed',bool),('native_continuous_valid','native_continuous_valid',bool),('nearest_valid','nearest_valid',bool)]:
  a=np.asarray(fr[field],dtype=dtype);assert z[key].dtype==a.dtype and z[key].shape==a.shape and np.array_equal(z[key],a,equal_nan=True),key
 assert z['query_id'].astype(str).tolist()==fr['query_id']
 for key in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):assert np.array_equal(z['original_'+key],[r[key] for r in fr['original_query_input_metadata']])
 for key in ('native_visibility_target','point_quality_target'):assert z[key].dtype==np.int8 and z[key].shape==(8,) and (z[key]==-1).all()
 return want

def execute(trigger):
 t=read(trigger);assert t['builder_completed'] is True and t['source_only'] is True;trigger_binding=bind(trigger)
 assert sha(DESIGN)==DSHA and sha(BH/'build_coherent_source_pixels.py')==CODE_SHA and sha(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json')==PSHA
 protocol=read(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json');bind(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json');verify(protocol['code'])
 done=read(OUT/'COMPLETE.json');assert done['completed'] and done['source_only'] and done['completion_PASS_means_all159_proofs_pass']
 assert done['protocol']==bind(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json') and done['assignments']==protocol['assignments'] and done['frame_inputs']==protocol['frame_inputs']
 assert done['training_admitted'] is False and not done['original_labels_read_or_changed'] and done['all_new_native_visibility_and_point_quality_UNKNOWN'] and done['encoder_model_fit_PnP_calls']==0 and not done['automatic_retry']
 entry=read(verify(done['entry']));assert entry['approved_actual_source_pixel_generation'] and entry['source_only'] and entry['code']==protocol['code'] and entry['protocol']==done['protocol']
 assert not any(entry[k] for k in ('training_allowed','encoder_allowed','model_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','automatic_retry'))
 assignments=lines(verify(protocol['assignments']));frames={r['frame_index']:r for r in read(verify(protocol['frame_inputs']))['frames']};proofs=lines(verify(done['proofs']));assert len(assignments)==len(proofs)==159 and len(frames)==46
 original_facts={r['global_index']:r for r in lines(verify(protocol['inputfacts']))};original_queries={r['query_id']:r for r in lines(verify(protocol['original_queries']))}
 for fr in frames.values():
  ff=[original_facts[i] for i in fr['global_index']];qq=[original_queries[q] for q in fr['query_id']]
  assert fr['facts']==ff and fr['original_query_input_metadata']==qq
  for key in ('global_index','query_id','native_slot','query_xy','L','feature_constructed','native_continuous_valid','nearest_valid','local_central4_available','context_available'):assert fr[key]==[f[key] for f in ff],key
  assert all(f['split']=='SOURCE_DEVELOPMENT' and f['frame_index']==fr['frame_index'] and f['context_box_xyxy']==fr['context_box_xyxy'] for f in ff)
  assert all(q['split']=='SOURCE_DEVELOPMENT' for q in qq)
 assert all(f['split']=='SOURCE_DEVELOPMENT' for f in frames.values()) and len(set(a['anchor_query_id'] for a in assignments))==53 and all(a['split']=='SOURCE_DEVELOPMENT' and a['donor_split']=='TRAIN' for a in assignments)
 dc=read(verify(protocol['donor_proof_complete']));donors={r['donor_id']:r for r in lines(verify(dc['records']))};assert dc['PASS'] and dc['passed']==59 and dc['failed']==0 and len(donors)==59
 assert dc['records']==protocol['donor_proof_records'] and done['donor_proof_complete']==protocol['donor_proof_complete']
 assert set(a['donor_id'] for a in assignments)==set(donors) and set(collections.Counter(a['donor_id'] for a in assignments).values())<={2,3}
 assert all(d['PASS'] and d['split']=='TRAIN' for d in donors.values())
 # Read only whitelisted selected source RGB/crops and proved compact donor NPZ,
 # plus this construction's saved variant/transformed NPZ. Never DINO features.
 allowed={str(safe(f[k]['path'])) for f in frames.values() for k in ('canonical_RGB_NPY','existing_factual96_packet')}|{str(safe(d['compact_RGBA_cutout']['path'])) for d in donors.values()}
 def load(b):
  p=safe(b['path']);assert p.suffix in ('.npy','.npz') and (str(p) in allowed or (p.is_relative_to(OUT) and p.name in {'TRANSFORMED_DONOR.npz',*[v+'_INPUTS.npz' for v in VARIANTS]}));verify(b)
  a=np.load(p,allow_pickle=False)
  if isinstance(a,np.ndarray):return a
  with a as z:
   assert 'features' not in z.files;return {k:z[k].copy() for k in z.files}
 cache={};donor_cache={};results=[]
 for a,r in zip(assignments,proofs):
  assert r==read(OUT/a['pair_id']/'PROOF.json');bind(OUT/a['pair_id']/'PROOF.json')
  for key in ('pair_id','pair_sha256','assignment_index','anchor_query_id','donor_id','native_slot','split'):assert r[key]==a[key],key
  assert r['original_q']==a['query_xy'] and r['original_L']==a['L'] and r['attempted_once'] and not r['replacement_or_retry'] and not r['training_admitted']
  assert r['new_native_visibility_target']==r['new_point_quality_target']==[-1]*8 and r['weak_owner_auxiliary_eligibility_only']==r['PASS']
  fr=frames[a['frame_index']];k=a['native_slot'];assert fr['query_xy'][k]==a['query_xy'] and fr['L'][k]==a['L'] and fr['query_id'][k]==a['anchor_query_id']
  if a['frame_index'] not in cache:
   original=load(fr['canonical_RGB_NPY']);saved=load(fr['existing_factual96_packet']);assert list(original.shape)==fr['raw_hw']+[3] and original.dtype==np.uint8 and ash(original)==fr['decoded_RGB_sha256'];cache[a['frame_index']]=(original,saved)
  original,saved=cache[a['frame_index']];variants={}
  for name,b in r['variants'].items():
   if 'path' in b:variants[name]=load(b)
   else:assert b=={'constructed':False,'reason':'FAILURE_RETAINED_NO_RETRY'} and not r['PASS']
  assert set(r['variants'])==set(VARIANTS)
  assert 'BEFORE' in variants, 'Unexpected pre-BEFORE execution failure requires explicit root diagnosis; not silently accepted'
  before=variants['BEFORE'];check_variant(before,original,np.zeros(original.shape[:2],bool),fr,protocol['output_schema']['variant_keys'])
  parity=dict(all8_RGB96_equal_saved_packet=exact(before['rgb96'],saved['rgb96']),all8_valid96_equal_saved_packet=exact(before['valid96'],saved['valid96']),all8_construct_equal_saved_packet=exact(before['crop_constructed'],saved['crop_constructed']),saved_global_indices_equal=np.array_equal(saved['global_index'],fr['global_index']),all8_RGB96_fact_hash=all(ash(before['rgb96'][i])==f['RGB96_HWC_sha256'] for i,f in enumerate(fr['facts'])),all8_valid96_fact_hash=all(ash(before['valid96'][i])==f['valid96_sha256'] for i,f in enumerate(fr['facts'])),target224_original_hash=all(ash(before['target224'])==f['target224_F32_HWC_sha256'] for f in fr['facts']),target_valid_original_hash=all(ash(before['target_valid'])==f['target_valid_sha256'] for f in fr['facts']),all8_fractions_original_exact=exact(before['fractions'],np.asarray([f['fractions'] for f in fr['facts']],np.float32)),selected_RGB96_anchor_hash=ash(before['rgb96'][k])==a['anchor']['existing_RGB96_array_sha256'],all8_center4_original_equal=np.array_equal(before['central4_available'],fr['local_central4_available']),all8_context_original_equal=np.array_equal(before['context_available'],fr['context_available']))
  assert parity==r['before_parity'] and all(parity.values())
  if a['donor_id'] not in donor_cache:
   d=donors[a['donor_id']];compact=load(d['compact_RGBA_cutout']);assert set(compact)=={'rgb','alpha'} and ash(compact['rgb'])==d['array_proof']['rgb']['array_sha256'] and ash(compact['alpha'])==d['array_proof']['alpha']['array_sha256'];donor_cache[a['donor_id']]=compact
  compact=donor_cache[a['donor_id']];expected_long=max(1,min(math.ceil(a['scale']*a['L']),math.floor(min(a['query_xy'][0]-4,fr['raw_hw'][1]-5-a['query_xy'][0])),math.floor((fr['raw_hw'][1]-12)/4)));assert expected_long==a['longside_pixels']
  cr,alpha=transform(compact['rgb'],compact['alpha'],expected_long);positions,g=geometry(a['query_xy'],a['L'],alpha,fr['raw_hw']);assert g==r['geometry']
  transformed=OUT/a['pair_id']/'TRANSFORMED_DONOR.npz';z=load(bind(transformed));assert exact(z['rgb'],cr) and exact(z['alpha'],alpha)
  if positions is None:
   assert not r['PASS'] and 'COVER' not in variants and 'ADJACENT_SHAM' not in variants;results.append(dict(pair_id=a['pair_id'],integrity_PASS=True,pixel_proof_PASS=False,retained_reason=g['reason']));continue
  assert 'COVER' in variants and 'ADJACENT_SHAM' in variants, 'Unexpected partial pair execution after valid geometry requires root diagnosis'
  co,ca=paste(original,cr,alpha,positions[0]);sh,sa=paste(original,cr,alpha,positions[1]);cover=variants['COVER'];sham=variants['ADJACENT_SHAM'];check_variant(cover,co,ca,fr,protocol['output_schema']['variant_keys']);check_variant(sham,sh,sa,fr,protocol['output_schema']['variant_keys'])
  footprint,cells=center4_footprints(a['query_xy'],a['L'],fr['raw_hw']);xy=[(x,y) for y,x in cells];inside=all(n['inside'] for f in footprint for n in f['neighbors'])
  for z,full in ((before,original),(cover,co),(sham,sh)):assert np.array_equal(center4_from_footprints(full,footprint),z['rgb96'][k,47:49,47:49])
  c,_=crop96(np.repeat(ca[:,:,None],3,2).astype(np.uint8)*255,a['query_xy'],a['L']);s,_=crop96(np.repeat(sa[:,:,None],3,2).astype(np.uint8)*255,a['query_xy'],a['L'])
  checks=dict(center4_footprint_inside=inside,all_contributing_raw_cover_alpha1=inside and all(ca[y,x] for x,y in xy),all_contributing_raw_sham_alpha0=inside and all(not sa[y,x] for x,y in xy),frozen_sampled_cover_alpha_center4_255=bool((c[47:49,47:49]==255).all()),frozen_sampled_sham_alpha_center4_0=bool((s[47:49,47:49]==0).all()),sham_center4_RGB_bit_equal_before=exact(sham['rgb96'][k,47:49,47:49],before['rgb96'][k,47:49,47:49]),sham_some_selected_local_donor_support=bool(s.any()),cover_sham_equal_pasted_alpha_mass=int(ca.sum())==int(sa.sum()),positive_pasted_mass=int(ca.sum())>0)
  for label,z in (('cover',cover),('sham',sham)):
   for key in ('valid96','target_valid','crop_constructed','native_continuous_valid','nearest_valid','central4_available','context_available','fractions','query_xy','L','native_slot'):checks[label+'_'+key+'_unchanged']=exact(z[key],before[key])
   for key in before:
    if key.startswith('original_') or key=='query_id':checks[label+'_'+key+'_unchanged']=exact(z[key],before[key])
  assert checks==r['proof'] and all(bool(x) for x in checks.values())==r['PASS']
  detail=dict(raw_footprint_xy=[list(p) for p in xy],raw_footprint_count=len(xy),cover_pasted_alpha_mass=int(ca.sum()),sham_pasted_alpha_mass=int(sa.sum()),cover_sampled_alpha96_sha256=ash(c.transpose(2,0,1).copy()),sham_sampled_alpha96_sha256=ash(s.transpose(2,0,1).copy()),sham_local_sampled_alpha_positive_cells=int((s[:,:,0]>0).sum()))
  assert detail==r['proof_detail']
  results.append(dict(pair_id=a['pair_id'],integrity_PASS=True,pixel_proof_PASS=r['PASS'],raw_footprint_count=len(xy),all8_views_coherent=True,before_parity=True,same_transform_mass=True,failed_predicates=[key for key,value in checks.items() if not value]))
 n=sum(r['pixel_proof_PASS'] for r in results);assert len(results)==159 and done['passed_pixel_input_proofs']==n and done['failed_attempts_retained']==159-n and done['all_proofs_pass']==done['PASS']==(n==159) and done['all159_attempts_retained']
 R.mkdir();record_path=R/'ALL159_INDEPENDENT_REVIEW.jsonl'
 with record_path.open('x') as f:
  for r in results:f.write(json.dumps(r,allow_nan=False)+'\n')
 final=dict(schema='V161_INDEPENDENT_POST_CONSTRUCTION_PIXEL_INPUT_REVIEW',completed=True,PASS=True,integrity_PASS=True,pair_attempts=159,anchor_queries=53,source_frames=46,fixed_TRAIN_donors=59,pixel_proof_passed=n,pixel_proof_failures_retained=159-n,all159_retained=True,review_code=bind(__file__),independent_math=bind(H/'independent_sampling_math.py'),root_completion_trigger=trigger_binding,source_complete=bind(OUT/'COMPLETE.json'),records=bind(record_path),checks=['all159 sealed assignments and original source identities','BEFORE all8 saved crop/valid/target224 hash parity','every saved variant coherent fullRGB/local96/central32/context224/fractions','independent raw bilinear center4 footprints and opacity/sham preservation','same frozen donor resize/EDT/rint placement/pasted mass with original q8/L/native/split/availability','original unavailable slots retained; no nativeE or pointquality conversion','independent proof/count arithmetic and preserved failures','59 donor proof receipt conformance without repeating448sourceasset proof'],bindings=list({(b['path'],b['sha256']):b for b in BINDINGS}.values()),no_model_feature_weight_encoder_fit_PnP_calls=True,donor448_asset_proof_not_repeated=True,held_RGB_or_validation_donor_reads=0,new_labels_or_admission=False,training_admitted=False,selected_model=None,fully_solved=False,semantics='Integrity and pixel/input proof for weak-owner auxiliary feasibility only; no measured opacity/nativegold/primary visibility or full-goal success claim.')
 write(R/'REVIEW_COMPLETE.json',final);print(json.dumps({'PASS':True,'pixel_passed':n,'retained_failed':159-n,'review':bind(R/'REVIEW_COMPLETE.json')}))

def prepare():
 assert sha(DESIGN)==DSHA and sha(BH/'build_coherent_source_pixels.py')==CODE_SHA and sha(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json')==PSHA
 p=read(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json');g=generated_checks()
 rgb=np.arange(21*31*3,dtype=np.uint16).reshape(21,31,3).astype(np.uint8);drgb=np.full((9,11,3),81,np.uint8);a=np.ones((9,11),bool);a[4,5]=False;cr,alpha=transform(drgb,a,9);co,ca=paste(rgb,cr,alpha,[2,-2]);sh,sa=paste(rgb,cr,alpha,[17,-2]);assert ca.sum()==sa.sum()>0 and np.array_equal(co[~ca],rgb[~ca]) and np.array_equal(sh[~sa],rgb[~sa]);assert geometry([0.,4.],6.,alpha,rgb.shape[:2])[0] is None
 assert geometry([10.,10.],6.,np.ones((4,4),bool),[100,100])[1]['opaque_anchor_xy']==[1,1]
 rules=dict(schema='V161_INDEPENDENT_PIXEL_REVIEW_PREPARATION',completed=True,PASS=True,prepared_only=True,code=bind(__file__),math=bind(H/'independent_sampling_math.py'),design=bind(DESIGN),builder_code=bind(BH/'build_coherent_source_pixels.py'),builder_protocol=bind(BH/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json'),invented_sampling_checks=bind(H/'GENERATED_INDEPENDENT_SAMPLING_CHECKS.json'),invented_same_transform_vertical_clipping_composition_and_EDT_checks_PASS=True,generated_output_or_source_pixel_bodies_read=0,required_root_trigger={'builder_completed':True,'source_only':True},no_idle_polling=True,planned_output=str(R/'REVIEW_COMPLETE.json'),scope='Only saved pixel/input integrity after completed-root trigger; no models/features/weights/encoder/fit/current/V112; no donor448 replay or newlabels/admission',numeric_policy='Exact bytes for saved image/mask/context arrays and original flags; strict weight>0 footprint contributors without epsilon/coordinate rounding; frozen integer/pixel sampler rules only.',PASS_meaning='Independent integrity/arithmetic agreement including retained failures; construction success counts reported separately, no training/native visibility success.',actual_review_executed=False)
 write(H/'PREPARATION_COMPLETE.json',rules);print(json.dumps({'prepared_only':True,'preparation':bind(H/'PREPARATION_COMPLETE.json')}))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--builder-completed-trigger',type=Path);args=parser.parse_args()
 if args.prepare_only and not args.builder_completed_trigger:prepare()
 elif args.builder_completed_trigger and not args.prepare_only:
  try:execute(args.builder_completed_trigger)
  except BaseException:
   destination=H/'FAILED_INDEPENDENT_PIXEL_REVIEW.json'
   if not destination.exists():write(destination,dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False))
   raise
 else:raise SystemExit('Use --prepare-only or explicit --builder-completed-trigger ROOT_JSON')
