"""V213 code-only preparation; root-entry-only both-final spatial-q8 prediction.

Reuse exact V184 original raw10 cache. No SAM, initializer or crop rerun, labels,
references, metrics, baseline outputs or model construction during preparation.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,stat,sys,traceback
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184'
FITPHASE=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'
SCOPE=V/'ROOT_PROSPECTIVE_DENSE_POINT_HEADS_FROZEN_BENCHMARK_SCOPE.json'
INHERITED_PROTOCOL=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203/prediction_preparation_a/PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'
DECODER_SPEC=P/'query_visibility_direct_end_paired_source_training_20261024_v190'/'ROOT_DIRECT_LOGIT_DECODER_NUMERICAL_SPEC_BEFORE_FIT.json'
RAW10=OLD/'real_raw10_SAM96_inputs_v1/COMPLETE.json'
RAW8=OLD/'original_real_raw8_inputs_v1/COMPLETE.json'
SOURCE=OLD/'frozen_prediction_preparation_c/compose_SAM96_and_infer_both_frozen_heads.py'
FIT_PROTOCOL=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'frozen_C62_C63_real_predictions_v1'
ARMS=(('C62_DENSE_POINT_RAW_Q8',False,True),('C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',False,True))
ARM_WEAK={ARMS[0][0]:(False,0.),ARMS[1][0]:(True,.25)}
DOMAINS={'CURRENT223_1784':(1784,0,45),'PUBLIC_HELD45_360':(360,584,108)}
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
INPUTS=set(META)|{'rgb96','valid96','sam_mask96','sam_available96'}
DIRECT_ORDER=['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'

def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def safe(path):
 p=Path(path).absolute();require(P in p.parents and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Exact project path/name before access');return p
def opened(path):
 p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 require(stat.S_ISREG(os.fstat(last).st_mode),'Regular file/no symlink');return os.fdopen(last,'rb')
def sha(path):
 h=hashlib.sha256()
 with opened(path) as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(safe(p)),'sha256':sha(p)}
def checked(b):require(bind(b['path'])=={k:b[k] for k in ('path','sha256')},'Exact frozen binding');return safe(b['path'])
def read(p):
 with opened(p) as f:return json.load(f)
def lines(p):
 with opened(p) as f:return [json.loads(s) for s in f if s.strip()]
def write(p,d):
 safe(p)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arrays(b):
 with opened(checked(b)) as f:
  with np.load(f,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fingerprints(x):return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=ash(v)) for k,v in x.items()}
def qarray(q):return np.array([np.nan,np.nan] if q is None else q,np.float64)
def decode(support,owner,direct,available):
 n=len(available);require(available.dtype==bool and available.shape==(n,),'Original availability bool')
 require(support.dtype==owner.dtype==direct.dtype==np.float32 and support.shape==(n,2) and owner.shape==(n,3) and direct.shape==(n,),'Exact FP32 raw output shapes')
 require(np.isfinite(support[available]).all() and np.isfinite(owner[available]).all(),'Finite old diagnostic heads')
 require(np.isnan(support[~available]).all() and np.isnan(owner[~available]).all() and np.isnan(direct[~available]).all(),'Factual inputU rawNaN first')
 sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);dp=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
 sp[available]=(support[available]>0).astype(np.int8);op[available]=owner[available].argmax(1).astype(np.int8)
 good=available&np.isfinite(direct);dp[good]=(direct[good]>=0).astype(np.int8)
 proxy[available]=np.where(sp[available,1]==1,1,np.where(sp[available,0]==1,0,-1)).astype(np.int8)
 return sp,op,dp,proxy

def validate_input(x,rows,domain,record):
 n,start,U=DOMAINS[domain];require(set(x)==INPUTS and len(rows)==n,'Exact target-free original raw10/allrows')
 for k,dt,sh in [('rgb96',np.uint8,(n,96,96,3)),('valid96',np.bool_,(n,96,96)),('sam_mask96',np.bool_,(n,96,96)),('sam_available96',np.bool_,(n,96,96)),('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('original_query_available',np.bool_,(n,))]:require(x[k].dtype==dt and x[k].shape==sh,'Raw10 exact schema '+k)
 require(x['row_id'].dtype.kind=='U' and x['row_id'].shape==(n,) and len(set(x['row_id'].tolist()))==n,'Unique original row IDs')
 require(np.array_equal(x['original_global_index'],np.arange(start,start+n,dtype=np.int64)) and int((~x['original_query_available']).sum())==U,'Original domain IDs/U, no concat alias')
 require(np.array_equal(x['native_slot'],np.arange(n,dtype=np.int64)%8),'Original all8 native slot order')
 av=x['original_query_available'];require(np.isfinite(x['query_xy'][av]).all() and np.isfinite(x['L'][av]).all() and (x['L'][av]>0).all(),'Original finite available qL')
 require(not x['sam_available96'][~av].any() and not x['sam_mask96'][~av].any(),'SAM channels do not resurrect inputU')
 require(not (x['sam_available96']&~x['valid96']).any() and not (x['sam_mask96']&~x['sam_available96']).any(),'Frozen SAM input-domain pixels')
 require(fingerprints({k:x[k] for k in record['original_array_fingerprints']})==record['original_array_fingerprints'],'All original raw8 bytes/f64 preserved')
 for i,r in enumerate(rows):
  require(r['domain']==domain and r['domain_index']==i and r['row_id']==str(x['row_id'][i]) and r['original_global_index']==int(x['original_global_index'][i]) and r['native_slot']==int(x['native_slot'][i]) and r['original_query_available']==bool(av[i]),'Exact original metadata join')
  require(np.array_equal(qarray(r['query_xy']),x['query_xy'][i],equal_nan=True) and np.array_equal(np.asarray(np.nan if r['L'] is None else r['L']),x['L'][i],equal_nan=True),'F64 qL identity')
  require(r['grid_precision']=='ORIGINAL_FLOAT64' and r['actual_crop_grid_query_xy']==r['query_xy'] and r['actual_crop_grid_L']==r['L'],'Original grid, no FP32alias/GTcrop')

def available_batches(available,size=192):
 ids=np.flatnonzero(available);return [ids[i:i+size] for i in range(0,len(ids),size)]
def source_module(binding,name):
 path=checked(binding);spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def object_sha(value):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):
   a=x.detach().cpu().contiguous();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
  elif isinstance(x,(list,tuple)):
   for a in x:visit(a)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()

def native_indices(a):
 require(a.dtype in (np.dtype(np.int8),np.dtype(np.int64)) and a.ndim==1 and ((a>=0)&(a<8)).all(),'Lossless stored native0..7 only')
 return a.astype(np.int64,copy=False)

def context_kernel(binding):
 # Extract only the frozen arithmetic functions. Never call its source loader.
 names={'native_index','bool_flag','q64','build_context_vectors'}
 with opened(checked(binding)) as f:tree=ast.parse(f.read().decode())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
 require({n.name for n in nodes}==names,'Frozen context arithmetic coverage')
 ns=dict(np=np,require=require);exec(compile(ast.Module(body=nodes,type_ignores=[]),binding['path']+'::PURE_CONTEXT','exec'),ns)
 return ns['build_context_vectors']

def build_real_context(x,rows,kernel):
 # Input-only original groups; no annotation/reference/GT or nested features.
 n=len(rows);require(n==len(x['row_id'])and n%8==0,'Full original all8 row census')
 mask=np.zeros((n,8),np.float32);xy=np.zeros((n,16),np.float32);onehot=np.zeros((n,8),np.float32)
 groups={}
 for i,r in enumerate(rows):
  require(isinstance(r['group_id'],str)and r['group_id']and type(r['frame_index'])is int and isinstance(r['id'],str),'Explicit input-only original group identity')
  identity=(r['domain'],r['group_id'],r['frame_index'],r['id'])
  groups.setdefault(identity,[]).append(i)
 for identity,ix in groups.items():
  require(len(ix)==8 and [int(x['native_slot'][i])for i in ix]==list(range(8)),'Exact ordered original native8 membership; no mate manufacture')
  require(len({r['original_global_index']for r in [rows[i]for i in ix]})==8,'Distinct original8 occurrence IDs')
  qq=[rows[i]['query_xy']for i in ix];ll=[rows[i]['L']for i in ix];aa=[rows[i]['original_query_available']for i in ix]
  for i in ix:
   r=rows[i];k=int(x['native_slot'][i]);require(r['row_id']==identity[3]+':native'+str(k),'Original group id/native lineage')
   require(np.array_equal(qarray(r['query_xy']),x['query_xy'][i],equal_nan=True)and np.array_equal(np.asarray(np.nan if r['L']is None else r['L'],np.float64),x['L'][i],equal_nan=True)and r['original_query_available']==bool(x['original_query_available'][i]),'Context same original META qL/U as raw packet')
   z=kernel(r['query_xy'],r['L'],k,r['original_query_available'],qq,ll,aa)
   mask[i]=z['other_mask8'];xy[i]=z['other_dxdy16'];onehot[i]=z['native_onehot8']
 require(len(groups)*8==n and np.array_equal(onehot,np.eye(8,dtype=np.float32)[native_indices(x['native_slot'])]),'Original native unchanged in context')
 require(not mask[~x['original_query_available']].any()and not xy[~x['original_query_available']].any(),'InputU spatial context zero and remains U')
 return dict(other_mask=mask,other_dxdy=xy),dict(groups=len(groups),rows=n,original_input_U=int((~x['original_query_available']).sum()),all8_original_groups_exact=True,original_metadata_fields_only=['domain','group_id','frame_index','id',*META],context_fingerprints=fingerprints(dict(other_mask=mask,other_dxdy=xy)),GT_or_target_or_reference_fields_used=False)

def infer(model,x,use_sam,head,context,device='cuda'):
 n=len(x['row_id']);av=x['original_query_available'];support=np.full((n,2),np.nan,np.float32);owner=np.full((n,3),np.nan,np.float32);direct=np.full(n,np.nan,np.float32)
 before=object_sha(model.state_dict());fp=fingerprints(x);cfp=fingerprints(context);batches=available_batches(av)
 with torch.inference_mode():
  for take in batches:
   rgb=torch.as_tensor(x['rgb96'][take],device=device);valid=torch.as_tensor(x['valid96'][take],device=device)
   extra=dict(sam_mask96=torch.as_tensor(x['sam_mask96'][take],device=device),sam_available=torch.as_tensor(x['sam_available96'][take],device=device)) if use_sam else {}
   inp=head.assemble_local_input(rgb,valid,use_sam=use_sam,**extra);native=torch.as_tensor(native_indices(x['native_slot'][take]),device=device);mask=torch.as_tensor(context['other_mask'][take],device=device);xy=torch.as_tensor(context['other_dxdy'][take],device=device);a=model(inp,native,mask,xy);b=model(inp,native,mask,xy)
   require(set(a)=={'support_logits','owner_logits','direct_end_external_logit'} and all(equal(a[k].detach().cpu().numpy(),b[k].detach().cpu().numpy()) for k in a),'Every fixed192 batch repeat2 exact')
   support[take]=a['support_logits'].float().cpu().numpy();owner[take]=a['owner_logits'].float().cpu().numpy();direct[take]=a['direct_end_external_logit'].float().cpu().numpy()
 require(object_sha(model.state_dict())==before and fingerprints(x)==fp and fingerprints(context)==cfp,'Final state/input/context bytes unchanged')
 sp,op,dp,proxy=decode(support,owner,direct,av)
 packet=dict(raw_support_logits=support,raw_owner_logits=owner,raw_direct_end_external_logit=direct,support_prediction=sp,owner_prediction=op,direct_end_prediction_D0_E1_Uminus1=dp,regional_support_proxy_D0_E1_Uminus1=proxy,support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']),direct_end_order=np.array(DIRECT_ORDER),**{k:x[k] for k in META})
 return packet,dict(batches=len(batches),head_calls=2*len(batches),all_batch_repeat2_exact=True,original_U_not_forwarded=int((~av).sum()),available_rows_forwarded=int(av.sum()),available_direct_nonfinite=int((av&~np.isfinite(direct)).sum()),state_before=before,state_after=before,input_fingerprints=fp,context_fingerprints=cfp)

def generated_checks():
 checks=[]
 def test(name,ok):require(ok,name);checks.append(dict(name=name,PASS=True))
 tiny=np.nextafter(np.float32(0),np.float32(1));d=np.array([-tiny,-0.,0.,tiny,np.inf,np.nan,np.nan],np.float32);av=np.array([1,1,1,1,1,1,0],bool)
 s=np.zeros((7,2),np.float32);o=np.zeros((7,3),np.float32);s[-1]=np.nan;o[-1]=np.nan
 sp,op,dp,proxy=decode(s,o,d,av)
 test('Direct_raw_negative_subnormal_D_signed_zero_and_positive_E_nonfinite_U',dp.tolist()==[0,1,1,1,-1,-1,-1])
 test('Old_support_strict_gt0_NONE_and_owner_firsttie_do_not_gate_direct',sp[:6].tolist()==[[0,0]]*6 and op.tolist()==[0]*6+[-1] and proxy.tolist()==[-1]*7)
 s[0]=[1,1];o[0]=[0,1,1];sp,op,dp,proxy=decode(s,o,d,av)
 test('Support_BOTH_owner_firsttie_and_separate_direct',sp[0].tolist()==[1,1] and op[0]==1 and dp[0]==0 and proxy[0]==1)
 bad=d.copy();bad[-1]=0
 try:decode(s,o,bad,av);ok=False
 except AssertionError:ok=True
 test('Unavailable_rawNaN_required_no_U_resurrection',ok)
 a=np.array([i%7!=0 for i in range(401)],bool);bs=available_batches(a);test('Fixed192_batches_cover_available_once_keep_U_unforwarded',np.array_equal(np.concatenate(bs),np.flatnonzero(a)) and max(map(len,bs))==192)
 test('Input_F64_identity_not_FP32_alias',np.float64(1.+2.**-30)!=np.float64(np.float32(1.+2.**-30)))
 # Execute the actual schema/identity join on invented arrays, no model.
 domain='INVENTED8';DOMAINS[domain]=(8,0,2)
 x=dict(rgb96=np.zeros((8,96,96,3),np.uint8),valid96=np.ones((8,96,96),bool),sam_mask96=np.zeros((8,96,96),bool),sam_available96=np.zeros((8,96,96),bool),row_id=np.array(['I:native'+str(k) for k in range(8)]),original_global_index=np.arange(8,dtype=np.int64),native_slot=np.arange(8,dtype=np.int8),query_xy=np.ones((8,2),np.float64),L=np.ones(8,np.float64),original_query_available=np.array([1]*6+[0]*2,bool))
 x['query_xy'][-1]=np.nan;x['L'][-1]=np.nan
 rows=[]
 for i in range(8):
  q=None if i==7 else x['query_xy'][i].tolist();L=None if i==7 else float(x['L'][i])
  rows.append(dict(domain=domain,domain_index=i,row_id=str(x['row_id'][i]),original_global_index=i,native_slot=i,original_query_available=bool(x['original_query_available'][i]),query_xy=q,L=L,grid_precision='ORIGINAL_FLOAT64',actual_crop_grid_query_xy=q,actual_crop_grid_L=L,group_id='INVENTED_FRAME_0000',frame_index=0,id='I'))
 rec=dict(original_array_fingerprints=fingerprints({k:x[k] for k in META+('rgb96','valid96')}));validate_input(x,rows,domain,rec)
 test('Actual_input_validator_NULL_qL_U_and_original_META6_fingerprints',True)
 for name,xx,rr in [('F32_q',dict(x,query_xy=x['query_xy'].astype(np.float32)),rows),('row_order',x,[rows[1],rows[0]]+rows[2:]),('SAM_U_resurrection',dict(x,sam_mask96=np.ones((8,96,96),bool)),rows)]:
  try:validate_input(xx,rr,domain,rec);ok=False
  except AssertionError:ok=True
  test(name+'_rejected',ok)
 del DOMAINS[domain]
 # Exercise the real infer batch path on an authored CPU model; never an actual head or checkpoint.
 global torch
 import torch
 class ToyModel(torch.nn.Module):
  def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.));self.seen=[]
  def forward(self,im,native,mask,xy):
   require(native.dtype==torch.int64 and native.ndim==1 and mask.dtype==xy.dtype==torch.float32,'Toy exact native/context argument');self.seen.append(native.tolist());require(mask.shape==(len(native),8)and xy.shape==(len(native),16),'Toy aligned context shapes')
   require(torch.equal(torch.round(im[:,0,0,0]*255).to(torch.int64),native),'Native same take as RGB')
   return dict(support_logits=torch.zeros((len(native),2)),owner_logits=torch.zeros((len(native),3)),direct_end_external_logit=native.float()-2.5+self.w*0)
 class ToyHead:
  @staticmethod
  def assemble_local_input(rgb,valid,use_sam,**extra):
   require(use_sam is False and not extra,'Both RAW and no SAM input');z=torch.zeros((len(rgb),8,96,96));z[:,:3]=rgb.permute(0,3,1,2).float()/255;return z
 for k in range(8):x['rgb96'][k,:,:,0]=k
 kernel=context_kernel(read(FIT_PROTOCOL)['context_adapter']);context,cproof=build_real_context(x,rows,kernel)
 model=ToyModel();packet,receipt=infer(model,x,False,ToyHead,context,device='cpu')
 test('Actual_infer_CPU_toy_same_take_native_int64_repeat2_and_U_not_forwarded',model.seen==[list(range(6))]*2 and packet['direct_end_prediction_D0_E1_Uminus1'].tolist()==[0,0,0,1,1,1,-1,-1] and receipt['head_calls']==2 and receipt['original_U_not_forwarded']==2 and all(equal(packet[k],x[k])for k in META))
 for bad in [np.array([1.],np.float32),np.array([-1],np.int8),np.array([8],np.int64)]:
  try:native_indices(bad);ok=False
  except AssertionError:ok=True
  test('Stored_native_invalid_'+str(bad.tolist())+'_rejected',ok)
 q8=[[float(j*100),float(j)]for j in range(8)];L8=[1.]*8;av8=[True]*8;av8[6]=False
 z=kernel(q8[0],L8[0],0,True,q8,L8,av8)
 test('Actual_V202_F64_clip_then_FP32_self_and_finite_unavailable_mask',z['other_dxdy16'].dtype==np.float32 and z['other_dxdy16'][2]==8 and not z['other_mask8_bool'][0]and not z['other_mask8_bool'][6]and not z['other_dxdy16'][12:14].any())
 q8=[[1.+2.**-30,0.],[1.+2.**-29,0.]]+[[0.,0.]]*6
 z=kernel(q8[0],1.,0,True,q8,[1.]*8,[True]*8)
 test('Actual_context_subtracts_F64_before_FP32',z['other_dxdy16'][2]==np.float32(2.**-30))
 badrows=[dict(r)for r in rows];badrows[7]['group_id']='DIFFERENT'
 try:build_real_context(x,badrows,kernel);ok=False
 except AssertionError:ok=True
 test('Actual_context_rejects_crossgroup_native_mate',ok)
 test('No_actual_model_weight_or_forward_in_preparation',True)
 return dict(completed=True,PASS=True,invented_only=True,checks=checks,count=len(checks),actual_real_array_header_hash_or_body_reads=0,actual_model_construction_or_forward=0,invented_CPU_toy_forward_calls=2)

def resolve_source_bindings(e,p):
 p=dict(p)
 for ek,pk in [('fit_complete','source_fit_complete'),('independent_source_fit_review','independent_source_fit_review'),('source_fit_protocol','source_fit_protocol'),('source_fit_trainer','source_fit_trainer'),('source_both_final_freeze','source_both_final_freeze')]:
  b=e.get(ek);require(isinstance(b,dict)and set(b)=={'path','sha256'}and isinstance(b['sha256'],str)and len(b['sha256'])==64,'Concrete root source binding '+ek)
  require(FITPHASE in safe(b['path']).parents,'Current V212 source closure only');p[pk]=b
 require(p['source_fit_complete']['path']==p['source_fit_complete_path']and p['source_fit_trainer']['path']==p['source_fit_trainer_path'],'Exact prospective V212 source paths')
 return p

def validate_source_metadata(fit,review,e,p,source_entry,barrier):
 require(fit['completed']and fit['PASS']and fit['source_only']and fit['from_scratch']and fit['all2_frozen_before_any_SOURCE_outputs'],'Both source finals before source outputs')
 require(review['completed']and review['PASS']and review['fit_complete']==e['fit_complete'],'Independent PASS binds exact current fit')
 require(fit['code']==source_entry['code']==p['source_fit_trainer']and fit['entry']==e['source_fit_entry']and source_entry['protocol']==fit['protocol']==p['source_fit_protocol'],'Exact source trainer/entry/protocol')
 require(fit['head']==p['head']and fit['candidate']==p['candidate']and fit['spatial_head']==p['spatial_head']and fit['context_adapter']==p['source_context_adapter'],'Same fixed C61 architecture/context')
 require(fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000 and fit['protected_real_COMMON_VAL_access']==0 and set(fit['arms'])=={a for a,_,_ in ARMS},'Both fixed RAW source arms4000')
 require(fit['all2_freeze']==p['source_both_final_freeze']and barrier['completed']and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a,_,_ in ARMS},'Both final checkpoint barrier exact')
 for a,_,_ in ARMS:
  r=fit['arms'][a];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is True and type(r['weak_bridge'])is bool and r['weak_bridge']is ARM_WEAK[a][0]and r['weak_loss_weight']==ARM_WEAK[a][1],'Fixed source target-only/weak-bridge roles')
 return True


def prepare():
 root=read(SCOPE);require(root['approved_actual_forward_or_evaluation']is False and root['status']=='PROSPECTIVE_CODE_PREPARATION_ONLY','Code-only new scope')
 inherited=read(INHERITED_PROTOCOL);fitp=read(FIT_PROTOCOL)
 write(CHECKS,generated_checks())
 p=dict(inherited)
 p.update(completed=True,prepared_only=True,code=bind(Path(__file__)),scope=bind(SCOPE),inherited_protocol=bind(INHERITED_PROTOCOL),checks=bind(CHECKS),
   arms=[dict(name=a,use_sam=s,spatial_q8=n,weak_bridge=ARM_WEAK[a][0],weak_loss_weight=ARM_WEAK[a][1])for a,s,n in ARMS],
   source_fit_protocol=None,source_fit_trainer=None,source_fit_complete=None,source_both_final_freeze=None,independent_source_fit_review=None,
   source_fit_complete_path=str(FITPHASE/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/COMPLETE.json'),
   source_fit_trainer_path=str(FITPHASE/'fit_preparation_a/train_dense_point_weak_bridge.py'),
   source_factory_authority=bind(FIT_PROTOCOL),head=fitp['source_bindings']['existing_head'],candidate=fitp['source_bindings']['candidate'],spatial_head=fitp['spatial_head'],
   source_context_adapter=fitp['context_adapter'],source_context_protocol=fitp['context_protocol'],output=str(OUT),parameters_each=118359,steps_each=4000,
   primary_a_priori='C63',ablation_a_priori='C62',required_source_fit_review_flags=[],
   original_input_metadata_bodies_read_during_prepare=False,existing_input_bindings_copied_without_body_or_fresh_asset_hash=True,
   context_contract='Both same native/mask/actualXY16, original input-only ROWS group8; V202 F64 subtract/divide clip[-8,8] then FP32; self/U zero. No GT/reference/point/role feature.',
   dynamic_source_binding_rule='Root actual entry after V212 final READY, source terminal and independent exact-fit PASS fills all five source bindings; nulls fail before real input/weight bodies.',
   selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p)
 template=H/'ROOT_ACTUAL_PREDICTION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
 write(template,dict(approved_both_frozen_direct_prediction=False,all_two_final4000_frozen=False,labels_old_scores_metrics_or_fit_allowed=False,
   code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],fit_complete=None,independent_source_fit_review=None,
   source_fit_protocol=None,source_fit_trainer=None,source_both_final_freeze=None,source_fit_entry=None,source_terminal_evidence=None,root_confirmed_source_terminal_exit0=False))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],entry_template=bind(template),actual_array_header_asset_hash_weight_or_output_reads=0,actual_forward=0,invented_CPU_toy_forward_only=True)
 write(H/'PREDICTION_PREPARATION_COMPLETE.json',ready);print(json.dumps(ready))

def run(entry_path):
 e=read(entry_path);require(e['approved_both_frozen_direct_prediction'] is True and e['all_two_final4000_frozen'] is True and e['labels_old_scores_metrics_or_fit_allowed'] is False,'Root frozen prediction entry')
 p=read(checked(e['protocol']));require(e['code']==bind(Path(__file__))==p['code'] and e['scope']==bind(SCOPE)==p['scope'] and e['protocol']==bind(PROTOCOL),'Frozen code/protocol/root scope')
 p=resolve_source_bindings(e,p)
 for key in ['decoder_spec','source_fit_protocol','source_fit_trainer','head','candidate','spatial_head','source_context_adapter','source_context_protocol','input_complete','original_raw_complete','checks']:checked(p[key])
 require(e['fit_complete']['path']==p['source_fit_complete_path'],'Exact completed source run path')
 require(e['fit_complete']==p['source_fit_complete']and e['independent_source_fit_review']==p['independent_source_fit_review'],'Exact source completion and independent review binding');fit=read(checked(e['fit_complete']));review=read(checked(e['independent_source_fit_review']))
 require(e['root_confirmed_source_terminal_exit0'] is True,'Root confirmed source terminal exit0 before real inference');checked(e['source_terminal_evidence'])
 source_entry=read(checked(e['source_fit_entry']));barrier=read(checked(p['source_both_final_freeze']))
 validate_source_metadata(fit,review,e,p,source_entry,barrier)
 source=read(RAW10);require(source['domains']==p['domains'] and not OUT.exists(),'Frozen original cache/new output only');OUT.mkdir()
 write(OUT/'STARTED_BEFORE_INPUT_WEIGHT_BODIES.json',dict(entry=bind(entry_path),fit=e['fit_complete'],independent_source_fit_review=e['independent_source_fit_review'],labels_old_scores_read=False))
 try:
  global torch
  import torch
  require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA runtime')
  require(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Pinned CUBLAS config')
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;require(not torch.is_autocast_enabled(),'FP32 no AMP')
  caches={};input_metadata={};contexts={};context_receipts={};kernel=context_kernel(p['source_context_adapter'])
  for domain,record in p['domains'].items():
   x=arrays(record['inputs']);rows=lines(checked(record['metadata']));validate_input(x,rows,domain,record);caches[domain]=x;input_metadata[domain]=rows;contexts[domain],context_receipts[domain]=build_real_context(x,rows,kernel)
  head=source_module(p['head'],'v200_original_head');candidate=source_module(p['candidate'],'v200_direct_candidate');spatial_factory=source_module(p['spatial_head'],'v203_spatial_residual');cls=spatial_factory.make_spatial_q8_head_class(head,candidate);arms={}
  for arm,use_sam,spatial_q8 in ARMS:
   f=fit['arms'][arm];require(f['updates']==4000 and f['use_sam']is False and type(f['spatial_q8'])is bool and f['spatial_q8']is spatial_q8,'Fixed RAW/native arm')
   payload=torch.load(checked(f['checkpoint']),map_location='cpu',weights_only=True);require(payload['update']==4000 and payload['seed']==1810 and payload['arm']==arm and payload['use_sam']is False and type(payload['spatial_q8'])is bool and payload['spatial_q8']is spatial_q8 and payload['protocol']==p['source_fit_protocol']and payload['weak_bridge']is ARM_WEAK[arm][0]and payload['weak_loss_weight']==ARM_WEAK[arm][1],'Exact final checkpoint metadata')
   model=cls(spatial_q8=spatial_q8).cuda().float().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True);del payload
   require(sum(v.numel() for v in model.parameters())==118359 and object_sha(model.state_dict())==f['final_state_sha'],'Exact source-final118359 state')
   require(len(model.state_dict())==36,'Exact36 source state tensors');directory=OUT/arm;directory.mkdir();outputs={};receipts={}
   for domain,x in caches.items():
    packet,proof=infer(model,x,use_sam,head,contexts[domain]);path=directory/('FINAL_'+domain+'_OUTPUTS.npz')
    with path.open('xb') as h:np.savez_compressed(h,**packet)
    outputs[domain]=bind(path);receipts[domain]=proof
   require(object_sha(model.state_dict())==f['final_state_sha'] and all(not t.requires_grad for t in model.parameters()),'Final frozen state unchanged');arms[arm]=dict(checkpoint=f['checkpoint'],updates=4000,source_final_state_sha=f['final_state_sha'],use_sam=use_sam,spatial_q8=spatial_q8,weak_bridge=ARM_WEAK[arm][0],weak_loss_weight=ARM_WEAK[arm][1],outputs=outputs,inference=receipts);del model;checked(f['checkpoint'])
  for record in p['domains'].values():checked(record['inputs']);checked(record['metadata'])
  for k in ('source_context_adapter','spatial_head','source_context_protocol'):checked(p[k])
  write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry_path),scope=p['scope'],decoder_spec=p['decoder_spec'],spatial_head=p['spatial_head'],source_context_adapter=p['source_context_adapter'],context_group_receipts=context_receipts,source_fit_protocol=p['source_fit_protocol'],source_fit_trainer=p['source_fit_trainer'],source_both_final_freeze=p['source_both_final_freeze'],source_fit_complete=e['fit_complete'],source_fit_integrity_review=e['independent_source_fit_review'],source_fit_entry=e['source_fit_entry'],source_terminal_evidence=e['source_terminal_evidence'],input_complete=p['input_complete'],original_raw_complete=p['original_raw_complete'],arms=arms,original_rows2144=2144,model_domain_rows4288=4288,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_currentU45_publicU108=True,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,new_encoder_initializer_SAM_forward_crop_fit_threshold_or_qL_changes=0,selected_model=None,adopted=False,fully_solved=False,requires_separate_saved_evaluation_entry=True,automatic_retry=False))
  print(json.dumps(dict(completed=True,PASS=True,prediction_seal=bind(OUT/'PREDICTION_SEAL.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
