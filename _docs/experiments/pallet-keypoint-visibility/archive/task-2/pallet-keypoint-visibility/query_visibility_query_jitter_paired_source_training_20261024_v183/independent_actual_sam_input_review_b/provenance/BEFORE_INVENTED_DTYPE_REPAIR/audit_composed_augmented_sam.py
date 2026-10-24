"""Independent V183 raw8->10 and SAM96 audit, saved sourceTRAIN only.
No producer/composer imports, GT target body, weights, original images or forward.
--prepare freezes metadata-only sample indices before composed SAM is opened.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,math,time,zipfile
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json';REFS=V/'augmentation_preparation_a/AUG65536_ROW_REFS_INPUT_ONLY.jsonl'
SELECT=H/'COMPOSER_SAMPLING_SELECTION_BEFORE_OUTPUTS.json'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available');RAW8=set(META)|{'rgb96','valid96'}
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','_v169','_v177','_v178','_v182','current223','private','synthetic_val','synth_val')) and p.suffix not in ('.pth','.pt','.ckpt')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(b):return [json.loads(s) for s in check(b).read_text().splitlines() if s]
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def member_sha(z,name):
 h=hashlib.sha256()
 with z.open(name+'.npy') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def scalar_point(logits,x,y):
 h,w=logits.shape;ix=math.floor(x);iy=math.floor(y);dx=x-ix;dy=y-iy;v=0.
 for xx,yy,weight in ((ix,iy,(1-dx)*(1-dy)),(ix+1,iy,dx*(1-dy)),(ix,iy+1,(1-dx)*dy),(ix+1,iy+1,dx*dy)):
  if 0<=xx<w and 0<=yy<h:v+=float(logits[yy,xx])*weight
 inside=(-.5<=x<w-.5 and -.5<=y<h-.5)
 return v>0 and inside,inside

def independent_sample(logits,q,L,pred_av):
 empty=np.zeros((96,96),bool)
 if not pred_av or q is None or np.shape(q)!=(2,) or not all(math.isfinite(float(v)) for v in q) or list(q)==[-1.,-1.] or L is None or not math.isfinite(float(L)) or L<=0:return empty,empty.copy()
 assert logits.dtype==np.float32 and logits.shape==(512,512)
 # Apply the scalar four-tap equation independently across the selected grid.
 u=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L;xx=np.broadcast_to(np.asarray(q[0]+u)[None,:],(96,96));yy=np.broadcast_to(np.asarray(q[1]+u)[:,None],(96,96))
 x0=np.floor(xx).astype(np.int64);y0=np.floor(yy).astype(np.int64);dx=xx-x0;dy=yy-y0;values=np.zeros((96,96),np.float64)
 for sx,sy,weight in ((x0,y0,(1-dx)*(1-dy)),(x0+1,y0,dx*(1-dy)),(x0,y0+1,(1-dx)*dy),(x0+1,y0+1,dx*dy)):
  good=(sx>=0)&(sx<512)&(sy>=0)&(sy<512);tap=np.zeros((96,96),np.float64);tap[good]=logits[sy[good],sx[good]];values+=tap*weight
 domain=(xx>=-.5)&(xx<511.5)&(yy>=-.5)&(yy<511.5)
 return (values>0)&domain,domain

def invented():
 log=np.arange(512*512,dtype=np.float32).reshape(512,512)-100000
 checks={}
 for q,L in (([2.3,4.7],96.),([0.,0.],3.),([511.3,511.1],4.),([-10.,16.],96.)):
  m,a=independent_sample(log,q,L,True)
  for row,col in ((0,0),(0,95),(47,47),(47,48),(48,47),(48,48),(95,95),(32,63)):
   x=q[0]+((col+.5)/96-.5)*L;y=q[1]+((row+.5)/96-.5)*L;s,sa=scalar_point(log,x,y);assert bool(m[row,col])==s and bool(a[row,col])==sa
 checks['own_4tap_grid_matches_Python_scalar_at_boundary_center_and_outside_points']=True
 z=np.zeros((512,512),np.float32);m,a=independent_sample(z,[2.,2.],3.,True);assert not m.any() and a.all();checks['strict_zero_negative_mask_with_available_input']=True
 for q,L in ((None,None),([-1.,-1.],96.),([1.,2.],None)):
  m,a=independent_sample(None,q,L,False);assert not m.any() and not a.any()
 checks['missing_U_no_fake_availability']=True
 return checks

def prepare():
 scope=read(SCOPE);source=lines(scope['development_source_manifest']);refs=lines(bind(REFS));assert len(source)==4096 and len(refs)==65536
 cells={}
 for r in source:
  assert r['split']=='TRAIN';f=r['factors'];key=(f['geometry_family'],f['material'],f['aspect'],f['foreign_family'],r['native_slot']);cells.setdefault(key,r['original_global_index'])
 chosen=set(cells.values());nominal=[r['original_global_index'] for r in refs if r['family']=='SELF_SCENE15' and r['state_index']==0];representative=[r['original_global_index'] for r in refs if r['family']=='SELF_SCENE15' and r['source_image_original_global_index'] in chosen];before=[r['original_global_index'] for r in refs if r['family']=='BEFORE_ANCHOR4']
 assert len(nominal)==4096 and len(representative)==15*len(chosen) and len(before)==4096
 selection=dict(completed=True,scope=bind(SCOPE),refs=bind(REFS),original_source_manifest=scope['development_source_manifest'],rule='Before output access: all SELF nominal4096 + all15states of first original row in each(geometry,material,aspect,foreignfamily,native) cell + all BEFORE4096. No label/score/pixel/outcome selection.',factor_native_cells=[dict(factor_native=list(k),original_global_index=v) for k,v in cells.items()],SELF_nominal_indices=nominal,SELF_fixed_factor_all15_indices=representative,BEFORE_all4096_indices=before,unique_sorted_indices=sorted(set(nominal+representative+before)))
 put(SELECT,selection);tests=invented();put(H/'COMPOSER_AUDIT_PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),selection=bind(SELECT),invented_checks=tests,actual_composed_input_or_raw_SAM_bodies_read=0,root_completion_notification_required=True));print(json.dumps(dict(prepared=True,selection_rows=len(selection['unique_sorted_indices']),cells=len(cells),receipt=bind(H/'COMPOSER_AUDIT_PREPARATION_COMPLETE.json'))))

def audit(cp):
 cp=Path(cp).resolve();assert cp.parent.parent==V and cp.name=='COMPLETE.json';selection=read(SELECT);check(selection['scope']);check(selection['refs']);source=lines(selection['original_source_manifest']);refs=lines(selection['refs'])
 c=read(cp);assert c['PASS'] and c['completed'] and c['rows']==65536 and c['domain']=='SYNTH_AUG65536' and c['target_body_reads']==0 and c['model_forward_fit_or_GTinput']==0 and c['source_TRAIN_only'] and c['fit_admission']
 e=read(check(c['entry']));assert e['approved_training_input_composition'] and e['code']==c['code'];check(c['code']);raw=read(check(c['raw_complete']));old=read(check(c['original_SAM_complete']));before=read(check(c['before_SAM_complete']));assert raw['PASS'] and raw['original_nominal4096_raw_valid_byteparity'] and raw['rows']==65536 and old['PASS'] and before['PASS']
 assert c['targets']==raw['targets'] and c['metadata']==raw['metadata'] and c['scene_metadata']==raw['scene_metadata']
 # Entire eight raw .npy members are decompressed and streamed, never all loaded.
 member_proofs={}
 with zipfile.ZipFile(check(raw['inputs'])) as za,zipfile.ZipFile(check(c['inputs'])) as zb:
  assert set(za.namelist())=={k+'.npy' for k in RAW8} and set(zb.namelist())=={k+'.npy' for k in RAW8|{'sam_mask96','sam_available96'}}
  for k in sorted(RAW8):
   a,b=member_sha(za,k),member_sha(zb,k);assert a==b,k;member_proofs[k]=a
 with np.load(c['inputs']['path'],allow_pickle=False) as z:
  numeric={k:z[k] for k in META};mask=z['sam_mask96'];available=z['sam_available96']
 assert mask.dtype==available.dtype==bool and mask.shape==available.shape==(65536,96,96)
 assert numeric['original_global_index'].dtype==np.int64 and numeric['native_slot'].dtype==np.int8 and numeric['query_xy'].dtype==numeric['L'].dtype==np.float64 and numeric['original_query_available'].dtype==bool
 assert numeric['row_id'].dtype.kind=='U' and numeric['query_xy'].shape==(65536,2) and numeric['L'].shape==(65536,)
 metadata=lines(c['metadata']);assert len(metadata)==len(refs)==65536 and metadata==refs
 states=read(SCOPE)['training_synthetic']['states15']
 for i,r in enumerate(metadata):
  assert r['original_global_index']==i and r['split']=='TRAIN' and str(numeric['row_id'][i])==r['row_id'] and int(numeric['original_global_index'][i])==i and int(numeric['native_slot'][i])==r['native_slot'] and bool(numeric['original_query_available'][i])==r['original_query_available']
  q=np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);L=np.float64(np.nan if r['L'] is None else r['L']);assert same(numeric['query_xy'][i],q) and same(np.asarray(numeric['L'][i]),np.asarray(L))
  im=source[r['source_image_original_global_index']];an=source[r['anchor_original_global_index']];assert im['scene_index']==r['scene_index'] and im['native_slot']==an['native_slot']==r['native_slot'] and r['original_query_available']==an['original_input_flags']['query_available']
  if r['family']=='SELF_SCENE15':
   s=states[r['state_index']];oq=an['original_query_xy'];ol=an['original_L'];assert r['query_xy']==(None if oq is None else [oq[0]+s['dx'],oq[1]+s['dy']]) and r['L']==(None if ol is None else ol*s['scale'])
  else:assert r['family']=='BEFORE_ANCHOR4' and an['variant']=='BEFORE' and r['query_xy']==an['original_query_xy'] and r['L']==an['original_L']
 assert int((~numeric['original_query_available']).sum())==c['original_aug_inputU']==1264
 for start in range(0,65536,256):
  end=min(start+256,65536);assert not (mask[start:end]&~available[start:end]).any();u=~numeric['original_query_available'][start:end];assert not mask[start:end][u].any() and not available[start:end][u].any()
 oldgroups={s['scene_index']:s for s in old['scenes']};newgroups={(s['base_index'],s['variant']):s for s in before['groups']};lineage=lines(c['SAM_lineage']);assert len(lineage)==65536
 for i,(r,f) in enumerate(zip(metadata,lineage)):
  g=oldgroups[r['scene_index']] if r['family']=='SELF_SCENE15' else newgroups[r['base_index'],r['variant']]
  assert f['row_id']==r['row_id'] and f['family']==r['family'] and f['SAM_output']==g['outputs'] and f['source_image_original_global_index']==r['source_image_original_global_index'] and f['anchor_original_global_index']==r['anchor_original_global_index']
  assert f['prompt_unchanged_by_local_crop_jitter']==(r['family']=='SELF_SCENE15') and f['sam_mask96_count']==int(mask[i].sum()) and f['sam_available96_count']==int(available[i].sum())
 firstgroup=None;last_output=None;packet=None;verified=0;scalar_points=0;starttime=time.monotonic();packet_bindings={}
 for n,i in enumerate(selection['unique_sorted_indices']):
  r=metadata[i];g=oldgroups[r['scene_index']] if r['family']=='SELF_SCENE15' else newgroups[r['base_index'],r['variant']];b=g['outputs'];k=r['native_slot'];an=source[r['anchor_original_global_index']]
  if b is None:logits=None;pred_av=False
  else:
   if b!=last_output:
    with np.load(check(b),allow_pickle=False) as z:packet={key:z[key] for key in ('image_logits','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available')}
    last_output=b;packet_bindings[b['path']]=b
   expected_id=an['original_global_index'] if r['family']=='SELF_SCENE15' else i;expected_occ=an['original_occurrence_id'] if r['family']=='SELF_SCENE15' else r['row_id']
   assert int(packet['original_global_index'][k])==expected_id and str(packet['original_occurrence_id'][k])==expected_occ and int(packet['native_slot'][k])==k
   assert np.array_equal(packet['box_xyxy'],np.asarray(g['box_xyxy'],np.float64)) and np.array_equal(packet['original_query_xy'][k],np.asarray([np.nan,np.nan] if an['original_query_xy'] is None else an['original_query_xy'],np.float64),equal_nan=True)
   assert np.array_equal(packet['original_L'][k],np.float64(np.nan if an['original_L'] is None else an['original_L']),equal_nan=True) and bool(packet['original_query_available'][k])==r['original_query_available']
   logits=packet['image_logits'];pred_av=bool(packet['prediction_available'][k]);assert logits.dtype==np.float32 and logits.shape==(512,512) and np.isfinite(logits).all()
  expectedmask,expectedavailable=independent_sample(logits,r['query_xy'],r['L'],pred_av);assert same(mask[i],expectedmask) and same(available[i],expectedavailable),i;verified+=1
  if pred_av and r['query_xy'] is not None and r['L'] is not None:
   for row,col in ((0,0),(0,95),(47,47),(47,48),(48,47),(48,48),(95,95),(32,63)):
    x=r['query_xy'][0]+((col+.5)/96-.5)*r['L'];y=r['query_xy'][1]+((row+.5)/96-.5)*r['L'];p,a=scalar_point(logits,x,y);assert bool(mask[i,row,col])==p and bool(available[i,row,col])==a;scalar_points+=1
  if (n+1)%1024==0:print(json.dumps(dict(checked=n+1,total=len(selection['unique_sorted_indices']),seconds=time.monotonic()-starttime)),flush=True)
 result=dict(completed=True,PASS=True,code=bind(__file__),composed_complete=bind(cp),root_entry=c['entry'],selection=bind(SELECT),original_raw8_members_all_bytes_equal_SHA=member_proofs,numeric_metadata_rows65536=True,lineage_rows65536=True,original_inputU1264_all_SAM_mask_and_availability_zero=True,mask_is_subset_of_availability_all_rows=True,SELF_nominal4096_rows_verified=True,BEFORE_all4096_rows_verified=True,all15_states_fixed_factor_native_cells_verified=True,unique_SAM96_rows_recomputed=verified,independent_Python_scalar_points_checked=scalar_points,used_packet_count=len(packet_bindings),method='Entire raw8 .npy bytes streamed unchanged, numeric identities65536 and lineage counts65536 checked. Independent four-tap FP64 scalar equation applied across each selected96grid; separate Python scalarpoint loop cross-checks boundaries/central4. No helper/producer/composer import.',GT_target_bodies_read=0,new_original_RGB_decode_sampling_model_forward_fit_or_selection=0,limitation='SAM96 selection covers all4096nominal, all4096BEFORE and fixedfactor/native all15states; remaining augmented SAM96 pixels were not independently resampled. No physical native/end truth or real transfer is certified.',training_admission='Only under V183 root composition scope/entry; standalone unlabeled SAM component grants no admission. V182 remains forbiddenfit.',selected_model=None,fully_solved=False)
 put(H/'COMPOSER_SAM_REVIEW.json',result);print(json.dumps(dict(PASS=True,review=bind(H/'COMPOSER_SAM_REVIEW.json'))))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--complete',type=Path);a=ap.parse_args()
 if a.prepare:assert a.complete is None;prepare()
 else:assert a.complete is not None;audit(a.complete)
if __name__=='__main__':main()
