"""V184 original real input-only adapter. --prepare uses invented arrays only.

Actual root entry + both final heads frozen precede input metadata/raw8 bodies.
Never read annotations, predictions, features/P3, RGB fullframes or checkpoints.
No crop generation: only copy original saved RGB96/valid96 to HWC raw8 packets.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,json,os,stat,tempfile,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json';SCOPE_SHA='05c628ad960a5715fedd96fb4e3cde7b2a67ee3ce188e49eb644867f4e3e0b25'
CLARIFY=V/'ROOT_REFERENCE_AUTHORITY_CLARIFICATION_BEFORE_REAL_BODIES.json'
T=P/'query_visibility_frozen_public_threshold_current_benchmark_20261024_v169'
U=P/'query_visibility_public_held_blind_center_owner_review_20261024_v177'
CURRENT_PROTOCOL=T/'crop_review_preparation/PROTOCOL_BEFORE_CURRENT_CROP_BODIES.json'
HELD_READINESS=U/'metadata_readiness_c/REVIEW.json'
CURRENT_SAMPLER=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
PUBLIC_SAMPLER=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
PUBLIC_PRODUCER=P/'query_visibility_mr6d_full_public_source_20261024_v153/initializer_feature_preparation/encode_full230.py'
PROTOCOL=H/'PROTOCOL_BEFORE_REAL_INPUT_METADATA_OR_RAW8_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'original_real_raw8_inputs_v1'
DOMAINS={'CURRENT223_1784':1784,'PUBLIC_HELD45_360':360}
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
CURRENT_KEYS=('rgb96','crop_valid96','query_xy','side_L');PUBLIC_KEYS=('rgb96','valid96','global_index','crop_constructed')
ARMS={'C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE'}


def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def safe_text(path):
 require('_v112' not in str(path).lower(),'Forbidden V112 path before filesystem access');return Path(path)
def sha(path):
 path=safe_text(path);h=hashlib.sha256()
 with path.open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(path):return dict(path=str(Path(path).absolute()),sha256=sha(path))
def checked(b):require(bind(b['path'])=={k:b[k] for k in ['path','sha256']},'Exact binding '+b['path']);return safe_text(b['path'])
def read(path):
 path=safe_text(path)
 with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as f:return json.load(f)
def lines(path):return [json.loads(s) for s in safe_text(path).read_text().splitlines() if s]
def clean(v):
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 return v

def write(path,v):
 with Path(path).open('x') as f:json.dump(clean(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_lines(path,rows):
 with path.open('x') as f:
  for r in rows:f.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):
 a=np.asarray(a);b=np.asarray(b);return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def qarray(q):return np.array([np.nan,np.nan] if q is None else q,np.float64)
def get_raw(binding,keys,np_module=np):
 # No .files enumeration: mixed original NPZ may contain P3/other fields.
 with np_module.load(checked(binding),allow_pickle=False) as z:return {k:z[k] for k in keys}
def domain96(q,L,hw):
 u=(np.arange(96,dtype=np.float64)+.5)/96-.5;x,y=np.meshgrid(q[0]+u*L,q[1]+u*L);h,w=hw
 return (x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
def cast_grid_drift(q,L):
 if not np.isfinite(q).all() or not np.isfinite(L):return None
 u=(np.arange(96,dtype=np.float64)+.5)/96-.5;q32=q.astype(np.float32).astype(np.float64);L32=float(np.float32(L))
 dx=q32[0]+u*L32-(q[0]+u*L);dy=q32[1]+u*L32-(q[1]+u*L)
 return dict(max_abs_axis_px=float(max(np.abs(dx).max(),np.abs(dy).max())),max_euclidean_px=float(np.hypot(np.abs(dx).max(),np.abs(dy).max())),FP32_alias_used_for_sampling=False)
def box_prompt(q,available):
 require(q.shape==(8,2) and available.shape==(8,),'Original q8 prompt shapes')
 good=available&np.isfinite(q).all(1)&~(q==[-1.,-1.]).all(1)
 if not good.any():return None,'NO_FINITE_AVAILABLE_INITIAL_Q'
 lo=q[good].min(0);hi=q[good].max(0)
 if not (hi>lo).all():return None,'DEGENERATE_INITIAL_Q8_MINMAX'
 return np.r_[lo,hi],'ORIGINAL_FINITE_AVAILABLE_Q8_UNCLIPPED_UNEXPANDED'
def new_arrays(n):
 return dict(rgb96=np.zeros((n,96,96,3),np.uint8),valid96=np.zeros((n,96,96),bool),row_id=np.empty(n,dtype='<U512'),original_global_index=np.empty(n,np.int64),native_slot=np.empty(n,np.int8),query_xy=np.empty((n,2),np.float64),L=np.empty(n,np.float64),original_query_available=np.empty(n,bool))
def validate_raw8(x,n,expected_ids,expected_U):
 require(set(x)==set(META)|{'rgb96','valid96'},'Exact target-free raw8 output')
 for k,dt,shape in [('rgb96',np.uint8,(n,96,96,3)),('valid96',np.bool_,(n,96,96)),('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('original_query_available',np.bool_,(n,))]:require(x[k].dtype==dt and x[k].shape==shape,'Raw8 schema '+k)
 require(x['row_id'].dtype.kind=='U' and x['row_id'].shape==(n,) and len(set(x['row_id'].tolist()))==n,'Unique exact row identities')
 require(np.array_equal(x['original_global_index'],expected_ids) and int((~x['original_query_available']).sum())==expected_U,'Original domain IDs/U never concatenated or resurrected')
 require(np.isfinite(x['query_xy'][x['original_query_available']]).all() and np.isfinite(x['L'][x['original_query_available']]).all() and (x['L'][x['original_query_available']]>0).all(),'Available q/L finitepositive')


def sources():
 c=read(CURRENT_PROTOCOL);u=read(HELD_READINESS)
 return dict(current={k:c[k] for k in ['original_cache','prepared_frames','feature_metadata','C11_frame_facts','parent_alias_renderer','parent_alias_protocol','fullframe_decoder']},current_prior_input_rows=read(P/'query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/evaluation_preparation/PROTOCOL_BEFORE_SAVED_CURRENT_SCORE_OR_LABEL_BODIES.json')['prior'],public_manifest=u['output_manifest'],current_raw_sampler=bind(CURRENT_SAMPLER),public_raw_sampler=bind(PUBLIC_SAMPLER),public_raw_producer=bind(PUBLIC_PRODUCER))
def row_record(domain,group,fi,id,g,k,q,L,av,rawbinding,rawrow,constructed,working_q=None,working_L=None):
 return dict(domain=domain,domain_index=rawrow if domain=='CURRENT223_1784' else g-584,row_id=id+':native'+str(k),row_id_authority='Original frame id + native slot; derived occurrence spelling only' if domain=='CURRENT223_1784' else 'Original public query_id',id=id,frame_index=fi,group_id=group,original_global_index=g,native_slot=k,query_xy=q,L=L,original_query_available=bool(av),actual_crop_grid_query_xy=q,actual_crop_grid_L=L,grid_precision='ORIGINAL_FLOAT64',cached_working_query_xy_FP32=working_q,cached_working_L_FP32=working_L,FP32_alias_grid_drift_diagnostic=cast_grid_drift(q,L),raw_packet=rawbinding,raw_row=rawrow,crop_constructed=bool(constructed),no_annotation_or_reference_or_geometry_input=True)
def group_record(domain,gid,fi,id,rr,source_rgb,hw,decoded):
 q=np.array([r['query_xy'] for r in rr],np.float64);av=np.array([r['original_query_available'] for r in rr],bool);box,reason=box_prompt(q,av)
 return dict(domain=domain,group_id=gid,frame_index=fi,id=id,raw_hw=hw,source_fullRGB=source_rgb,decoded_RGB_sha256=decoded,original_input_metadata8=rr,original_query_xy8=q,actual_crop_grid_query_xy8=q,original_L8=np.array([r['L'] for r in rr],np.float64),actual_crop_grid_L8=np.array([r['L'] for r in rr],np.float64),original_query_available8=av,native_slot=list(range(8)),box_xyxy=box,prompt_available=box is not None,prompt_reason=reason,prompt_GT_or_clip_or_expansion=False,fullRGB_not_read_by_this_producer=True)


def current_domain(src):
 s=src['current'];raw=get_raw(s['original_cache'],CURRENT_KEYS)
 require(raw['rgb96'].dtype==np.uint8 and raw['rgb96'].shape==(1784,3,96,96) and raw['crop_valid96'].dtype==bool and raw['crop_valid96'].shape==(1784,1,96,96),'Original current RGB/valid schema')
 require(raw['query_xy'].dtype==raw['side_L'].dtype==np.float32 and raw['query_xy'].shape==(1784,2) and raw['side_L'].shape==(1784,),'Original working FP32 aliases')
 # Existing inference-input descriptors only. Project immediately onto explicit
 # input fields; never copy pose/geometry/reference/annotation fields.
 frames=[{k:r[k] for k in ['id','image','image_sha256','raw_hw','initial_points']} for r in read(checked(s['prepared_frames']))]
 meta=[{k:r[k] for k in ['id','side']} for r in read(checked(s['feature_metadata']))]
 prior=[{k:r[k] for k in ['id','native_corner','query_x','query_y','roi_side','valid','network_callable','image_sha256']} for r in lines(checked(src['current_prior_input_rows']))]
 facts=[{k:r[k] for k in ['frame_index','id','image','raw_hw','decoded_RGB_sha256','RGB96_original_cached_sha256','valid96_original_cached_sha256']} for r in lines(checked(s['C11_frame_facts']))]
 require(len(frames)==len(meta)==len(facts)==223 and len(prior)==1784,'All current metadata223/1784')
 x=new_arrays(1784);x['rgb96']=np.ascontiguousarray(raw['rgb96'].transpose(0,2,3,1));x['valid96']=np.ascontiguousarray(raw['crop_valid96'][:,0]);rows=[];groups=[]
 for fi,(f,m,proof) in enumerate(zip(frames,meta,facts)):
  require(f['id']==m['id']==proof['id'] and proof['frame_index']==fi and proof['image']==dict(path=f['image'],sha256=f['image_sha256']) and proof['raw_hw']==f['raw_hw'],'Exact original frame/image metadata')
  q=np.array(f['initial_points'],np.float64)[:8];L=float(m['side']);sl=slice(fi*8,fi*8+8);require(q.shape==(8,2) and np.isfinite(L) and L>0,'Original q8/L')
  require(same(raw['query_xy'][sl],q.astype(np.float32)) and same(raw['side_L'][sl],np.full(8,np.float32(L))),'WorkingFP32 aliases match originalF64 casts, not replacement geometry')
  require(ash(raw['rgb96'][sl])==proof['RGB96_original_cached_sha256'] and ash(raw['crop_valid96'][sl])==proof['valid96_original_cached_sha256'],'Original cached frame8 pixel/valid SHA')
  gid='CURRENT_FRAME_%04d'%fi;rr=[]
  for k in range(8):
   i=fi*8+k;p=prior[i];require(p['id']==f['id'] and p['native_corner']==k and p['roi_side']==L and p['image_sha256']==f['image_sha256'],'Original prior id/native/L/image')
   pq=qarray([p['query_x'],p['query_y']]);require(same(q[k],pq),'Original q metadata exact')
   constructed=p['network_callable'];require(isinstance(p['valid'],bool) and isinstance(constructed,bool),'Factual availability flags')
   if constructed:require(same(x['valid96'][i],domain96(q[k],L,f['raw_hw'])),'Actual F64 crop grid matches saved validity')
   else:require(not x['valid96'][i].any() and not x['rgb96'][i].any(),'Original nonconstructed placeholders retained')
   r=row_record('CURRENT223_1784',gid,fi,f['id'],i,k,q[k],L,p['valid'],s['original_cache'],i,constructed,raw['query_xy'][i],raw['side_L'][i]);rr.append(r)
   for key in META:x[key][i]=r[key]
  rows.extend(rr);safe_text(f['image']);groups.append(group_record('CURRENT223_1784',gid,fi,f['id'],rr,dict(kind='ORIGINAL_ENCODED_RGB_SAFE_ALIAS',binding=dict(path=f['image'],sha256=f['image_sha256']),safe_alias_loader_source=s['parent_alias_renderer'],decoder=s['fullframe_decoder']),f['raw_hw'],proof['decoded_RGB_sha256']))
 x['row_id']=np.asarray([r['row_id'] for r in rows]);validate_raw8(x,1784,np.arange(1784,dtype=np.int64),45);return x,rows,groups


def public_domain(src):
 source=lines(checked(src['public_manifest']));require(len(source)==360,'All publicheld360')
 x=new_arrays(360);rows=[];groups=[];used={}
 for fi in range(73,118):
  eight=source[(fi-73)*8:(fi-72)*8];require([r['native_slot'] for r in eight]==list(range(8)) and all(r['frame_index']==fi and r['split']=='SOURCE_VALIDATION' for r in eight),'Original held frame/native/split')
  f=eight[0];binding=f['RGB96_packet'];raw=get_raw(binding,PUBLIC_KEYS);used[binding['path']]=binding
  require(raw['rgb96'].dtype==np.uint8 and raw['rgb96'].shape==(8,96,96,3) and raw['valid96'].dtype==bool and raw['valid96'].shape==(8,96,96) and raw['global_index'].dtype==np.int32 and np.array_equal(raw['global_index'],np.arange(fi*8,fi*8+8,dtype=np.int32)) and raw['crop_constructed'].dtype==bool and raw['crop_constructed'].shape==(8,),'Original public packet4 schema')
  rr=[];gid='PUBLIC_HELD_FRAME_%04d'%fi
  for k,r in enumerate(eight):
   g=fi*8+k;i=g-584;q=qarray(r['query_xy']);L=np.nan if r['L'] is None else float(r['L']);require(r['original_global_index']==g and r['query_id']==r['frame_id']+':native'+str(k) and r['packet_native_row']==k and r['RGB96_packet']==binding and r['canonical_RGB']==f['canonical_RGB'] and r['decoded_RGB_sha256']==f['decoded_RGB_sha256'],'Exact original public row/input authority')
   require(bool(raw['crop_constructed'][k])==r['feature_constructed'],'Original constructionflag')
   hashes=r['original_array_hashes_inherited_not_recomputed'];require(ash(raw['rgb96'][k])==hashes['RGB96_HWC_sha256'] and ash(raw['valid96'][k])==hashes['valid96_sha256'],'Original public raw/valid decoded array hashes')
   if r['feature_constructed']:require(same(raw['valid96'][k],domain96(q,L,r['raw_hw'])),'Public original F64 grid/domain parity')
   else:require(not raw['valid96'][k].any() and not raw['rgb96'][k].any(),'Public missing crop placeholders remain')
   out=row_record('PUBLIC_HELD45_360',gid,fi,r['frame_id'],g,k,q,L,r['original_query_available'],binding,k,r['feature_constructed']);require(out['row_id']==r['query_id'],'Preserved original public queryid');out['source_id']=r['source_id'];out['split']=r['split'];rr.append(out)
   x['rgb96'][i]=raw['rgb96'][k];x['valid96'][i]=raw['valid96'][k]
   for key in META:x[key][i]=out[key]
  rows.extend(rr);groups.append(group_record('PUBLIC_HELD45_360',gid,fi,f['frame_id'],rr,dict(kind='CANONICAL_RGB_NPY',binding=f['canonical_RGB'],expected_dtype='uint8',expected_shape=f['raw_hw']+[3],original_encoded_RGB=f['original_RGB']),f['raw_hw'],f['decoded_RGB_sha256']))
 x['row_id']=np.asarray([r['row_id'] for r in rows]);validate_raw8(x,360,np.arange(584,944,dtype=np.int64),108);return x,rows,groups,used


def invented(src):
 checks=[]
 def passed(name,ok):require(ok,name);checks.append(dict(name=name,PASS=True))
 def rejects(fn):
  try:fn()
  except (AssertionError,ValueError):return True
  return False
 # Execute original sampler functions only on invented pixels to prove F64 math.
 ns=dict(np=np);tree=ast.parse(CURRENT_SAMPLER.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop96');exec(compile(ast.Module(body=[node],type_ignores=[]),str(CURRENT_SAMPLER),'exec'),ns)
 pub=dict(np=np);tree=ast.parse(PUBLIC_SAMPLER.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop');exec(compile(ast.Module(body=[node],type_ignores=[]),str(PUBLIC_SAMPLER),'exec'),pub)
 rgb=np.arange(9*11*3,dtype=np.uint8).reshape(9,11,3);copy=rgb.copy();q=np.array([4.123456789,3.987654321],np.float64);L=5.987654321;a,v=ns['crop96'](rgb,q,L);c,w=pub['crop'](rgb,q,L,96)
 passed('Original currentV72 and publicV90 F64 samplers agree on invented RGB and grid domain',np.array_equal(a,c) and np.array_equal(v,w) and np.array_equal(v[0],domain96(q,L,[9,11])) and np.array_equal(rgb,copy))
 drift=cast_grid_drift(q,L);passed('FP32 aliases have measured nonzero drift but never replace F64 actualgrid',drift['max_abs_axis_px']>0 and drift['FP32_alias_used_for_sampling'] is False)
 data=dict(rgb96=np.arange(2*3*96*96,dtype=np.uint8).reshape(2,3,96,96),crop_valid96=np.ones((2,1,96,96),bool),query_xy=np.array([[1.,2.],[3.,4.]],np.float32),side_L=np.array([4.,4.],np.float32));access=[]
 class Archive:
  def __enter__(self):return self
  def __exit__(self,*args):return False
  def __getitem__(self,k):require(k in CURRENT_KEYS,'Forbidden mixedpacket read');access.append(k);return data[k]
  @property
  def files(self):raise AssertionError('No inventory of mixed NPZ')
 class FakeNP:
  def load(self,*args,**kwargs):return Archive()
 # Override the file binding check only around invented virtual archive access.
 global checked
 oldcheck=checked
 try:
  checked=lambda b:Path('<invented>');got=get_raw({'path':'<invented>','sha256':'invented'},CURRENT_KEYS,FakeNP())
 finally:checked=oldcheck
 passed('Current mixedNPZ whitelist reads exactly4 keys, never P3/features/annotations',access==list(CURRENT_KEYS) and set(got)==set(CURRENT_KEYS))
 hwc=np.ascontiguousarray(data['rgb96'].transpose(0,2,3,1));passed('CHWtoHWC raw copy is reversible byteexact and valid singleton removal exact',np.array_equal(hwc.transpose(0,3,1,2),data['rgb96']) and data['crop_valid96'][:,0].shape==(2,96,96))
 toy=new_arrays(8);toy['row_id'][:]=['i:native'+str(k) for k in range(8)];toy['original_global_index'][:]=np.arange(584,592);toy['native_slot'][:]=np.arange(8);toy['query_xy'][:]=1;toy['L'][:]=4;toy['original_query_available'][:]=[1,1,1,1,1,1,1,0];validate_raw8(toy,8,np.arange(584,592),1)
 passed('Perdomain original globalIDs retained; falseconcat IDs and extra target keys rejected',rejects(lambda:validate_raw8(toy,8,np.arange(8),1)) and rejects(lambda:validate_raw8(dict(toy,owner_target=np.zeros(8)),8,np.arange(584,592),1)))
 qq=np.array([[0.,0.],[2.,3.],[np.nan,np.nan],[-1,-1],[4,5],[6,7],[8,9],[10,11]],np.float64);av=np.array([1,1,0,0,0,0,0,0],bool);box,reason=box_prompt(qq,av)
 passed('Prompt finiteavailable originalq8 no GT/clip/expand; unavailableq doesnot expand box',np.array_equal(box,[0.,0.,2.,3.]))
 passed('Missing prompt remains separate from inputU; no synthetic successful mask',box_prompt(qq,np.zeros(8,bool))[0] is None and box_prompt(np.ones((8,2)),np.ones(8,bool))[0] is None and int((~toy['original_query_available']).sum())==1)
 # Exact existing alias helper on invented filesystem objects only.
 path=Path(src['current']['parent_alias_renderer']['path']);tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='safe_original_image_path');env=dict(Path=Path,os=os,stat=stat,sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest());exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
 with tempfile.TemporaryDirectory(prefix='v184_invented_alias_') as temp:
  target=Path(temp)/'safe';target.write_bytes(b'invented only');alias=Path(temp)/'alias';alias.symlink_to(target);binding=dict(path=str(alias),sha256=env['sha'](target));ok=env['safe_original_image_path'](binding)==target
  bad=Path(temp)/'bad';bad.symlink_to(Path(temp)/'forbidden_v112'/'never-created')
  passed('Original alias decoder verifies content and rejects forbidden expansion before target access',ok and rejects(lambda:env['safe_original_image_path'](dict(path=str(bad),sha256='0'*64))))
 return checks


def prepare():
 require(sha(SCOPE)==SCOPE_SHA,'Frozen prospective scope');src=sources();checks=invented(src)
 write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),checks=checks,real_metadata_raw_RGB_NPZ_weight_score_annotation_body_or_hash_reads=0,actual_crop_generation_or_model_forward=0))
 p=dict(completed=True,prepared_only=True,scope=bind(SCOPE),authority_clarification=bind(CLARIFY),code=bind(__file__),checks=bind(CHECKS),sources=src,domains=DOMAINS,groups268=True,output=str(OUT),
 npz_whitelists=dict(current=list(CURRENT_KEYS),public=list(PUBLIC_KEYS)),raw8_keys=['rgb96','valid96']+META,
 q_precision='Original identityF64 ==actualRGBcropgridF64 for BOTH domains: V72adapt_frame obtains qF64/Lfloat, passes directly to crop96; cacheFP32q/L aliases saved afterwards. V153 qdataF64 passes to V90crop. SAMmustuseoriginalF64 not cacheFP32. Castdrift diagnostic only, never corrective.',
 source_lines=dict(current=dict(path=str(CURRENT_SAMPLER),crop96=[89,107],F64input_and_alias=[132,157]),public=dict(path=str(PUBLIC_PRODUCER),F64q_crop_and_savedfacts=[139,167]),public_sampler=dict(path=str(PUBLIC_SAMPLER),crop=[74,85])),
 fullRGB='Only provenance is recorded; producer never opens encoded/current image aliases or public canonical NPY. Future SAMstage uses pinned safealias/Pillow12.2 forcurrent and exactcanonicalNPY/decodedSHA forpublic arbitraryHW.',
 prompt='Original finiteavailableq8 minmax; no clipping,expansion,GT/reference/contextbbox substitution. Missingprompt separate from originalqueryavailability.',
 output_contract='Separate domain INPUTS raw8/ROWS and all268GROUPS JSONL. originalIDs current0..1783/public584..943 preserved. No concatenatedfakeglobalIDs; current occurrence row_id explicitlyderived original frameID+native, publicquery_id unchanged. No targets/labels/SAM arrays here.',
 originalU=dict(CURRENT223_1784=45,PUBLIC_HELD45_360=108),no_new_crop_sampler_in_actual=True,original_human_annotation_body_reads=0,after_both_newheads_freeze_and_root_entry_only=True,selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p);template=H/'ROOT_ACTUAL_INPUT_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_original_real_raw8_input=False,all_two_models_frozen=False,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),fit_complete=None))
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_input_or_annotation_body_reads=0,actual_crop_or_forward=0))
 print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))


def run(entry):
 e=read(entry);require(e.get('approved_original_real_raw8_input') is True and e.get('all_two_models_frozen') is True,'Root entry after both final freeze')
 for key,path in [('code',Path(__file__)),('protocol',PROTOCOL),('scope',SCOPE),('authority_clarification',CLARIFY)]:require(e[key]==bind(path),'Frozen entry '+key)
 p=read(PROTOCOL);require(p['checks']==bind(CHECKS) and sha(SCOPE)==SCOPE_SHA and p['sources']==sources(),'Frozen preparation/source metadata')
 fit=read(checked(e['fit_complete']));require(fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_evaluation_access'] and set(fit['arms'])==ARMS and fit['updates_each']==8000 and fit['total_updates']==16000,'Both final8000 models already frozen, checkpoint bodies not read')
 freeze=read(checked(fit['all2_freeze']));require(freeze['completed'] and not freeze['saved_outputs_emitted'] and not freeze['validation_cache_or_targets_read'] and not freeze['common_diagnostic_inputs_or_targets_read'],'Actual twohead freeze barrier')
 for arm in ARMS:require(fit['arms'][arm]['updates']==8000 and fit['arms'][arm]['checkpoint']==freeze['checkpoints'][arm],'Finalhead binding metadata only')
 require(not OUT.exists(),'Preserve attempts, no automatic overwrite/retry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),annotation_or_prediction_or_fullRGB_or_weights_read=False))
 try:
  src=p['sources'];cx,cr,cg=current_domain(src);ux,ur,ug,used=public_domain(src);domains={}
  for name,x,rows in [('CURRENT223_1784',cx,cr),('PUBLIC_HELD45_360',ux,ur)]:
   packet=OUT/(name+'_INPUTS.npz');np.savez_compressed(packet,**x);rp=OUT/(name+'_ROWS.jsonl');write_lines(rp,rows)
   domains[name]=dict(rows=len(rows),inputs=bind(packet),metadata=bind(rp),original_inputU=int((~x['original_query_available']).sum()),actual_raw_array_SHA=dict(rgb96=ash(x['rgb96']),valid96=ash(x['valid96'])),no_target_or_new_crop=True)
  groups=cg+ug;require(len(groups)==268,'All sourcegroups268');gp=OUT/'ALL268_GROUPS_INPUT_ONLY.jsonl';write_lines(gp,groups)
  checked(src['current']['original_cache'])
  for x in used.values():checked(x)
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,evaluation_inputs_only=True,scope=bind(SCOPE),authority_clarification=bind(CLARIFY),code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),fit_complete=e['fit_complete'],domains=domains,groups=bind(gp),groups268=268,rows2144=2144,original_current_U45_public_U108=True,original_ID_qL_native_pixels_preserved=True,identity_F64_and_actual_crop_grid_F64_same=True,FP32_aliases_preserved_as_metadata_only=True,SAM_arrays_present=False,fullRGB_decodes_or_reads=0,annotation_prediction_checkpoint_body_reads=0,feature_or_P3_arrays_deserialized=0,mixed_raw_packet_encoded_SHA_verified=True,new_crop_sampling_initializer_model_forward_fit_or_labels=0,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False))
  print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.entry)
