"""One authorized PUBLIC-source frozen-head hybrid-feature probe; no encoder/fit."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,itertools,json,math,time,traceback
import numpy as np
import torch
import torch.nn.functional as F

H=Path(__file__).resolve().parent
POLICY=H/'POLICY_BEFORE_ARRAYS_AND_MODEL.json'
POLICY_SHA='cf66b693823a5b6a7f0fc85b84212aab631811ad263af06ed4cb8b887262f6ca'
GROUPS=('T','Q','F','C','P')

def bind(p):
 p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(n,x):
 with(H/n).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
def state_sha(model):
 h=hashlib.sha256()
 for k,t in sorted(model.state_dict().items()):
  t=t.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()
def shapley(values):
 assert values.shape[-1]==32
 phi=np.zeros(values.shape[:-1]+(5,),np.float64)
 for i in range(5):
  for mask in range(32):
   if mask&(1<<i):continue
   k=bin(mask).count('1');w=math.factorial(k)*math.factorial(4-k)/math.factorial(5)
   phi[...,i]+=w*(values[...,mask|(1<<i)].astype(np.float64)-values[...,mask].astype(np.float64))
 return phi
def generated():
 tests=[]
 for kind in ('additive','pair_interaction','all_interaction'):
  v=np.zeros(32,np.float64)
  for mask in range(32):
   if kind=='additive':v[mask]=sum(i+1 for i in range(5)if mask&(1<<i))
   elif kind=='pair_interaction':v[mask]=6 if mask&3==3 else 0
   else:v[mask]=10 if mask==31 else 0
  want={'additive':[1,2,3,4,5],'pair_interaction':[3,3,0,0,0],'all_interaction':[2]*5}[kind]
  p=shapley(v);assert np.max(np.abs(p-want))<1e-12 and abs(p.sum()-(v[31]-v[0]))<1e-12
  tests.append(dict(case=kind,phi=p.tolist()))
 write('GENERATED_FORMULA_TESTS.json',dict(PASS=True,cases=tests,source_arrays_or_parameters_read=0,code=bind(__file__)))

def summary(phi,z,ix):
 p=phi[ix];v=z[ix];assert len(p)
 absolute=np.abs(p).sum((0,1));den=float(absolute.sum())
 return dict(triplets=len(p),mean_before_to_cover_logit=float((v[:,0,31]-v[:,0,0]).astype(np.float64).mean()),mean_before_to_sham_logit=float((v[:,1,31]-v[:,1,0]).astype(np.float64).mean()),
  transitions={stage:dict(mean_signed=dict(zip(GROUPS,p[:,j].mean(0).tolist())),mean_absolute=dict(zip(GROUPS,np.abs(p[:,j]).mean(0).tolist())),absolute_share=dict(zip(GROUPS,(np.abs(p[:,j]).sum(0)/np.abs(p[:,j]).sum()).tolist()))if np.abs(p[:,j]).sum()else None,largest_absolute_player_counts={g:int((np.abs(p[:,j]).argmax(1)==k).sum())for k,g in enumerate(GROUPS)})for j,stage in enumerate(('BEFORE_TO_COVER','BEFORE_TO_SHAM'))},both_transitions_absolute_share=dict(zip(GROUPS,(absolute/den).tolist()))if den else None)

def run():
 assert bind(POLICY)['sha256']==POLICY_SHA
 assert not(H/'RUN_STARTED.json').exists(),'No repeated source diagnostic.'
 policy=read(POLICY);b=policy['bindings']
 for val in b.values():check(val)
 generated()
 write('RUN_STARTED.json',dict(policy=bind(POLICY),code=bind(__file__),generated=bind(H/'GENERATED_FORMULA_TESTS.json'),all_bound_files_verified=True,public_only=True,head_calls_completed=0))
 # Only the explicit NPZ member whitelist is deserialized. No image/GT/encoder/private reader.
 olds=rows(b['old194']['path']);news=rows(b['new62']['path']);lock=read(b['fixed_three_roles']['path'])
 assert len(olds)==194 and len(news)==62 and len(lock['roles'])==3
 meta=olds+news;trip=np.array([r['triple_global_indices']for r in meta],np.int64)
 assert trip.shape==(256,3)and len(set(map(tuple,trip)))==256
 with np.load(b['selected_scores']['path'],allow_pickle=False)as f:
  score={k:np.array(f[k],copy=True)for k in policy['model_input_whitelist']['score_members']}
 with np.load(b['old_features']['path'],allow_pickle=False)as f:
  old={k:np.array(f[k],copy=True)for k in policy['model_input_whitelist']['old_features_members']}
 with np.load(b['new_features']['path'],allow_pickle=False)as f:
  new={k:np.array(f[k],copy=True)for k in policy['model_input_whitelist']['new_features_members']}
 assert old['features'].shape==(11652,1928)and new['features'].shape==(152,1928)
 assert old['features'].dtype==new['features'].dtype==np.float32
 assert np.array_equal(trip[:194],score['old_accepted_triplets'])
 assert np.array_equal(trip[194:,1:],score['samewood_accepted_cover_sham_pairs'])
 assert (trip[:194]<31680).all()and(trip[194:]>=45184).all()and(trip[194:]<46108).all()
 original=np.empty((256,3,1928),np.float32);units=np.empty((256,3),np.int32)
 for i,r in enumerate(meta):
  for j,global_ix in enumerate(trip[i]):
   if i<194:local=int(global_ix);packet=old;u=int(packet['occurrence_unit'][local]);assert u==score['input_unit_index'][global_ix]
   else:local=int(global_ix)-45184;packet=new;u=int(packet['occurrence_unit'][local]);assert u+14852==score['input_unit_index'][global_ix]
   assert packet['query_available'][local]and score['query_available'][global_ix]
   original[i,j]=packet['features'][u];units[i,j]=u
  assert np.array_equal(original[i,0,1924:],original[i,1,1924:])and np.array_equal(original[i,0,1924:],original[i,2,1924:]),('geometric_scalars_differ',i)
  assert np.array_equal(np.array([r['C13'][s]for s in('BEFORE','COVER','SHAM')],np.float32),score['query_logit'][trip[i]])
 assert np.isfinite(original).all()
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 spec=importlib.util.spec_from_file_location('probe_exact_C11_head',b['head_class']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 model=mod.SpatialTargetQueryHead().cuda()
 payload=torch.load(b['checkpoint']['path'],map_location='cpu',weights_only=True)
 assert payload['update']==1000 and payload['threshold_logit']==0 and payload['head_class_code']==b['head_class']
 assert payload['trainer']==read(b['freeze']['path'])['trainer']and payload['design']==b['source_design']
 model.load_state_dict(payload['state_dict'],strict=True);model.eval();model.requires_grad_(False)
 before_state=state_sha(model);assert sum(p.numel()for p in model.parameters())==221825 and not any(p.requires_grad for p in model.parameters())
 x=torch.as_tensor(original,device='cuda');a=x[:,0];after=x[:,1:]
 hybrid=a[:,None,None,:].expand(256,2,32,1928).clone()
 for k in range(5):
  ids=[mask for mask in range(32)if mask&(1<<k)]
  hybrid[:,:,ids,384*k:384*(k+1)]=after[:,:,None,384*k:384*(k+1)]
 flat=hybrid.reshape(-1,1928)
 for k,(i,j)in enumerate(((0,3),(0,2),(1,3),(1,4))):flat[:,1920+k]=F.cosine_similarity(flat[:,384*i:384*(i+1)],flat[:,384*j:384*(j+1)],dim=1,eps=1e-12)
 hybrid[:,:,0,:]=a[:,None,:];hybrid[:,:,31,:]=after
 assert torch.equal(hybrid[:,:,0,:],a[:,None,:].expand(-1,2,-1))and torch.equal(hybrid[:,:,31,:],after)
 assert torch.equal(hybrid[:,:,:,1924:],a[:,None,None,1924:].expand(-1,2,32,-1))
 hybrid_sha=hashlib.sha256(hybrid.detach().cpu().numpy().tobytes()).hexdigest()
 t=time.monotonic()
 with torch.inference_mode():v=model(hybrid.reshape(-1,1928)).reshape(256,2,32).cpu().numpy()
 seconds=time.monotonic()-t
 assert v.dtype==np.float32 and np.isfinite(v).all()
 expected=score['query_logit'][trip];endpoint=np.stack((v[:,0,0],v[:,0,31],v[:,1,31]),1)
 endpoint_error=np.abs(endpoint.astype(np.float64)-expected.astype(np.float64));assert endpoint_error.max()<=policy['method']['endpoint_selected_logit_absolute_tolerance'],float(endpoint_error.max())
 assert np.array_equal(v[:,0,0],v[:,1,0])
 phi=shapley(v);delta=v[:,:,31].astype(np.float64)-v[:,:,0].astype(np.float64);eff=np.abs(phi.sum(-1)-delta)
 assert eff.max()<=policy['method']['efficiency_absolute_tolerance'],float(eff.max())
 assert state_sha(model)==before_state
 for val in b.values():check(val)
 np.savez_compressed(H/'ALL256_COALITIONS_AND_SHAPLEY.npz',coalition_logits=v,shapley_phi=phi,triple_global_indices=trip,packet_unit_indices=units,original_endpoint_logits=expected,probe_endpoint_logits=endpoint,endpoint_absolute_error=endpoint_error,efficiency_absolute_error=eff,player_names=np.array(GROUPS),transition_names=np.array(['BEFORE_TO_COVER','BEFORE_TO_SHAM']),coalition_masks=np.arange(32,dtype=np.uint8),original_geometric_scalars=original[:,0,1924:])
 cases=[]
 for i,r in enumerate(meta):
  fields={k:r[k]for k in('attempt_index','base_row_index','input_id','original_query_id','base_image_sha256','split','raw_split','source','source_family','donor_rank','donor_id','donor_object_id','donor_scene_id','donor_frame_id','donor_instance_id','donor_source_family')if k in r}
  fields.update(dict(probe_triplet_index=i,population='OLD194'if i<194 else'NEW62_TRAIN_ONLY',triple_global_indices=trip[i].tolist(),saved_C13_logits=expected[i].tolist(),probe_endpoint_logits=endpoint[i].tolist(),original_geometric_scalars=original[i,0,1924:].tolist(),query_available_all3=True,
   attribution={stage:dict(delta_logit=float(delta[i,j]),phi=dict(zip(GROUPS,phi[i,j].tolist())),efficiency_error=float(eff[i,j]),largest_absolute_player=GROUPS[int(np.abs(phi[i,j]).argmax())])for j,stage in enumerate(('BEFORE_TO_COVER','BEFORE_TO_SHAM'))}))
  cases.append(fields)
 with(H/'ALL256_PAIR_ATTRIBUTIONS.jsonl').open('x')as f:
  for r in cases:f.write(json.dumps(r,allow_nan=False)+'\n')
 summaries={}
 def group(key,ix):summaries[key]=summary(phi,v,np.array(ix,np.int64))
 group('ALL256',range(256));group('OLD194',range(194));group('NEW62_TRAIN_ONLY',range(194,256))
 for population,start,end in(('OLD194',0,194),('NEW62_TRAIN_ONLY',194,256)):
  for split in sorted({r['split']for r in meta[start:end]}):group(population+'/'+split,[i for i in range(start,end)if meta[i]['split']==split])
  for obj in sorted({r['donor_object_id']for r in meta[start:end]}):
   group(population+'/object_'+str(obj),[i for i in range(start,end)if meta[i]['donor_object_id']==obj])
   for split in sorted({r['split']for r in meta[start:end]if r['donor_object_id']==obj}):group(population+'/'+split+'/object_'+str(obj),[i for i in range(start,end)if meta[i]['donor_object_id']==obj and meta[i]['split']==split])
 fixed=[]
 for role in lock['roles']:
  i=role['original_ALL194_row_index'];assert meta[i]['triple_global_indices']==role['original_record']['triple_global_indices']
  fixed.append(dict(role=role['role'],selected_before_current_by_bound_lock=True,attribution=cases[i]))
 result=dict(completed=True,PASS=True,policy=bind(POLICY),code=bind(__file__),generated=bind(H/'GENERATED_FORMULA_TESTS.json'),bindings=b,triplets256=True,OLD194=194,NEW62_TRAIN_only=62,transitions512=True,coalitions_each=32,head_call_count=1,head_batch_rows=16384,encoder_calls=0,optimizer_updates=0,new_RGB_or_teacher=0,current_V112_V117_PnP=0,checkpoint_and_state_unchanged=True,head_state_sha256=before_state,hybrid_model_input_sha256=hybrid_sha,endpoint_max_absolute_error=float(endpoint_error.max()),endpoint_tolerance=1e-4,efficiency_max_absolute_error=float(eff.max()),efficiency_tolerance=1e-8,head_seconds=seconds,geometric_scalars_identical_within_all_triples=True,endpoints_original_feature_vectors_exact=True,summary=summaries,fixed_three_roles=fixed,interpretation_limits=policy['limits'],new_model_selection_or_training_authorization=False)
 write('RESULT.json',result)
 write('SEAL.json',dict(completed=True,PASS=True,files=[bind(H/n)for n in('ROOT_AUTHORIZATION.json','POLICY_BEFORE_ARRAYS_AND_MODEL.json','run_frozen_feature_shapley.py','GENERATED_FORMULA_TESTS.json','RUN_STARTED.json','ALL256_COALITIONS_AND_SHAPLEY.npz','ALL256_PAIR_ATTRIBUTIONS.jsonl','RESULT.json')]))
 print(json.dumps(dict(result=bind(H/'RESULT.json'),seal=bind(H/'SEAL.json'),endpoint_max_absolute_error=result['endpoint_max_absolute_error'],efficiency_max_absolute_error=result['efficiency_max_absolute_error'],summary={k:result['summary'][k]for k in('OLD194/TRAIN','OLD194/VALIDATION','NEW62_TRAIN_ONLY')})),flush=True)

if __name__=='__main__':
 try:run()
 except BaseException:
  if not(H/'FAILED_EXECUTION.json').exists():write('FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),automatic_retry=False,model_or_policy_not_changed=True))
  raise
