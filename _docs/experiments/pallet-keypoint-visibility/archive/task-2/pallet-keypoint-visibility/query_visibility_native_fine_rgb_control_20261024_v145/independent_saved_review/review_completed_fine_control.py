"""Saved-only independent row, sampler, arithmetic and CPU-state review; no model objects/forward."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,time
import numpy as np
import torch
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_native_fine_rgb_control_20261024_v145';H=V/'independent_saved_review';RUN=V/'runs/SHARP_BLURRED_NATIVE_FIXED2000';DATA=V/'source_inputs_v1'
ARMS=('SHARP_NATIVE_CENTRAL32','BLURRED_NATIVE_CENTRAL32');N=26624;OLD=6144
META=('state','target','split','boundary','bands','variant','material','aspect','geometry_family','foreign_family')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path'];assert sha(b['path'])==b['sha256'];return Path(b['path'])
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def statehash(state):return hashlib.sha256(b''.join(k.encode()+t.contiguous().numpy().tobytes()for k,t in state.items())).hexdigest()
def rec_equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and eq(a.numpy(),b.numpy())
 if isinstance(a,dict):return set(a)==set(b)and all(rec_equal(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(rec_equal(x,y)for x,y in zip(a,b))
 return a==b
start=time.monotonic();control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed']and control['PASS']and sha(RUN/'CONTROL_COMPLETE.json')=='3ea937cf8280a647cb63075ffde608a09e1271a60d205364cc8682ab2ebf7f07'
assert control['selected_model']is None and control['updates_each']==2000 and control['all26624_preserved']and control['AUX29_unmeasured']and control['original64_unchanged_not_claimed']
protocol=read(check(control['protocol']));check(control['code']);seal=read(check(protocol['input_seal']))
with np.load(check(seal['input_only']),allow_pickle=False)as f:inp={k:f[k].copy()for k in f.files if k not in('rgb32','valid32')}
with np.load(check(seal['GT_only']),allow_pickle=False)as f:gt={k:f[k].copy()for k in f.files}
limits=read(check(seal['native35_limits']))['limits'];assert len(limits)==35
old={k:gt['old_'+k]for k in META};new={k:gt['new_'+k]for k in META}
assert inp['available'].sum()==26440 and inp['constructed'].sum()==26512

def groups(meta,kind):
 n=len(meta['state']);out={}
 for fold in('ALL','TRAIN','VALIDATION'):
  f=np.ones(n,bool)if fold=='ALL'else meta['split']==fold;out[fold]=f
  if kind=='old':
   out[fold+'::NONBOUNDARY']=f&~meta['boundary'];out[fold+'::BOUNDARY']=f&meta['boundary'];out[fold+'::NONBOUNDARY::SAME_TARGET_MATERIAL']=f&~meta['boundary']&(meta['foreign_family']=='SAME_TARGET_MATERIAL')
  else:out[fold+'::historical_area_agreement']=f&~meta['boundary'];out[fold+'::historical_area_disagreement']=f&meta['boundary']
  for band in('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY'):
   out[fold+'::'+band]=f&(meta['bands']==band)
   if kind=='old':out[fold+'::NONBOUNDARY::'+band]=f&~meta['boundary']&(meta['bands']==band)
  for key in('variant','material','aspect','geometry_family','foreign_family'):
   for val in np.unique(meta[key][f]):out[fold+'::'+key+'::'+str(val)]=f&(meta[key]==val)
  if kind=='new':
   for mat in ('PROCEDURAL_WOOD','DARK_POLYMER'):
    for aspect in('RECTANGULAR','SQUARE'):out[fold+'::material_aspect::'+mat+'::'+aspect]=f&(meta['material']==mat)&(meta['aspect']==aspect)
 return out

def metric(meta,pred,mask):
 y=meta['target'];s=meta['state'];cm=np.array([[int((mask&(y==i)&(pred==j)).sum())for j in(0,1,-1)]for i in(0,1)],np.int64);ratio=lambda a,b:float(a/b)if b else None
 return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),prediction_U_known=int(cm[:,2].sum()),prediction_U_rate_known=ratio(cm[:,2].sum(),cm.sum()),all_original_rows=int(mask.sum()),known_DE=int(cm.sum()),teacher_S_outside_external=int((mask&(s=='S')).sum()),teacher_U_outside_external=int((mask&(s=='U')).sum()),teacher_S_binary_responses={str(p):int((mask&(s=='S')&(pred==p)).sum())for p in(0,1,-1)},teacher_U_binary_responses={str(p):int((mask&(s=='U')&(pred==p)).sum())for p in(0,1,-1)})
def reports(raw):
 pred=np.where(inp['available'],(raw>=0).astype(np.int8),-1)
 return ({name:metric(old,pred[:OLD],mask)for name,mask in groups(old,'old').items()},{name:metric(new,pred[OLD:],mask)for name,mask in groups(new,'new').items()})
def native35(orpt,nrpt):
 out={}
 for name,ref in limits.items():
  prefix='NEW_NATIVE::'if name.startswith('NEW_NATIVE::')else'NATIVE::';group,state=name[len(prefix):].rsplit('::',1);met=(nrpt if prefix.startswith('NEW')else orpt)[group];value=met[state+'_recall'];limit=ref['limit'];support=sum(met['CM_D_E_rows_D_E_U_columns'][0 if state=='D' else 1]);out[name]=dict(value=value,limit=limit,N=support,PASS=bool(support>0 and value is not None and limit is not None and value>=limit))
 return out
# Replay default layer initializer RNG as raw CPU tensor operations only.
# This is independent of model construction, training or any feature forward.
def initial_tensors(final_state):
 torch.manual_seed(1450);layers={}
 for name,shape in [('conv1',(16,4,3,3)),('conv2',(16,16,3,3)),('conv3',(16,16,3,3)),('film',(32,42)),('hidden',(32,1024)),('delta',(1,32))]:
  w=torch.empty(shape);torch.nn.init.kaiming_uniform_(w,a=math.sqrt(5));fan=math.prod(shape[1:]);b=torch.empty(shape[0]);torch.nn.init.uniform_(b,-1/math.sqrt(fan),1/math.sqrt(fan));layers[name+'.weight']=w;layers[name+'.bias']=b
 for name in('film','delta'):layers[name+'.weight'].zero_();layers[name+'.bias'].zero_()
 a=torch.arange(-2,3,dtype=torch.float32);g=torch.exp(-a.square()/2);g=g/g.sum();layers['gaussian5']=torch.outer(g,g)[None,None]
 return {k:layers[k]for k in final_state},torch.get_rng_state().clone()

def draw(pair,pools,pg,ug):
 pi=torch.randint(len(pair),(16,),generator=pg).numpy();d=pools[0][torch.randint(len(pools[0]),(16,),generator=ug).numpy()];e=pools[1][torch.randint(len(pools[1]),(16,),generator=ug).numpy()];return np.r_[pair[pi,0],pair[pi,1],d,e].astype(np.int64),pi
recalc={};checkpoints={};drawref=None;exposureref=None;metakeys=('available','constructed','global_index','source_occurrence_index','occurrence_id','corpus','scene_index','base_index','native_slot','query_xy','L')
for arm in ARMS:
 d=RUN/arm;fit=read(check(control['arms'][arm]['fit']));freeze=read(check(fit['freeze']));saved=read(check(fit['source_results']));update0=read(d/'UPDATE0_FROZEN_C32_PARITY.json')
 assert fit['completed']and fit['optimizer_updates']==2000 and saved['fixed_update']==2000 and saved['selected_model']is None and freeze['current_allowed']is False and freeze['selected_model']is None
 with np.load(check(fit['scores']),allow_pickle=False)as f:a={k:f[k].copy()for k in f.files}
 for k in metakeys:assert eq(a[k],inp[k]),(arm,k)
 raw=a['raw_head_logit_before_U'];delta=a['raw_residual_delta'];base=a['raw_frozen_C32_base_logit']
 assert raw.dtype==delta.dtype==base.dtype==np.float32 and raw.shape==delta.shape==base.shape==(N,)
 assert eq(base,inp['base_raw_logit'])and eq(raw,base+delta)
 assert np.array_equal(np.isfinite(raw),inp['constructed'])and np.array_equal(np.isfinite(delta),inp['constructed'])
 assert eq(a['query_logit'],np.where(inp['available'],raw,0).astype(np.float32))and np.array_equal(a['prediction'],np.where(inp['available'],(raw>=0).astype(np.int8),-1))and(a['prediction']==-1).sum()==184
 orpt,nrpt=reports(raw);gates=native35(orpt,nrpt)
 assert orpt==saved['metrics']['old_registered_native']and nrpt==saved['metrics']['continuous_registered_native']and gates==saved['metrics']['native35']
 assert sum(g['PASS']for g in gates.values())==control['arms'][arm]['native35_passed']and all(g['PASS']for g in gates.values())==freeze['native35_all_pass']
 pg,ug=torch.Generator().manual_seed(1334),torch.Generator().manual_seed(1332);epg,eug=torch.Generator().manual_seed(1374),torch.Generator().manual_seed(1372);dh=hashlib.sha256();exposure=np.zeros(N,np.int64);logs=0
 with Path(protocol['source_bindings']['base_draws']['path']).open()as oldlog,(d/'ALL2000_DRAWS_AND_LOSSES.jsonl').open()as logfile:
  for step,line in enumerate(logfile,1):
   r=json.loads(line);before=json.loads(next(oldlog));assert step==r['update']==before['update']
   ni,pi=draw(gt['old_pairs'],[gt['old_D'],gt['old_E']],pg,ug);ei,epi=draw(gt['new_pairs'],[gt['new_D'],gt['new_E']],epg,eug)
   for values,key,prior in [(ni,'old_native_occurrences','native_occurrences'),(pi,'old_pair_pool_positions','native_pair_pool_positions'),(ei,'new_native_occurrences','extra_native_occurrences'),(epi,'new_pair_pool_positions','extra_pair_pool_positions')]:assert np.array_equal(values,r[key])and np.array_equal(values,before[prior]);dh.update(values.astype('<i8').tobytes())
   ix=np.r_[ni,OLD+ei];np.add.at(exposure,ix,1)
   assert(old['split'][ni]=='TRAIN').all()and(new['split'][ei]=='TRAIN').all()and inp['available'][ix].all()
   assert np.array_equal(old['target'][ni],[1]*16+[0]*32+[1]*16)and np.array_equal(new['target'][ei],old['target'][ni])
   assert float(np.float32(np.float32(r['old_BCE'])+np.float32(r['new_BCE'])))==r['loss']and all(math.isfinite(r[k])for k in('loss','old_BCE','new_BCE','gradient_before_clip'))and r['gradient_before_clip']>=0;logs+=1
 assert logs==2000 and dh.hexdigest()==fit['draw_sha256']==control['draw_sha256']
 with np.load(d/'SAMPLING_EXPOSURE.npz',allow_pickle=False)as f:assert eq(exposure,f['occurrence_count'])
 assert exposure[:OLD].sum()==exposure[OLD:].sum()==128000
 if drawref is None:drawref=dh.hexdigest();exposureref=exposure
 else:assert drawref==dh.hexdigest()and eq(exposure,exposureref)
 cp=torch.load(check(freeze['checkpoint']),map_location='cpu',weights_only=True);checkpoints[arm]=cp
 assert cp['update']==2000 and cp['parameters']==39441 and cp['blurred']==(arm==ARMS[1])and cp['selected_model']is None and cp['input_seal']==protocol['input_seal']
 initial,rng=initial_tensors(cp['state_dict']);assert statehash(initial)==update0['initial_state_sha256']==saved['initial_state_sha256']
 assert update0['PASS']and update0['all26624_raw_byte_exact']and update0['all184_inputU_preserved']
 assert eq(rng.numpy(),cp['RNG']['torch_cpu'].numpy())
 paramstates=[t for k,t in cp['state_dict'].items()if k!='gaussian5'];assert sum(t.numel()for t in paramstates)==39441 and all(t.dtype==torch.float32 and torch.isfinite(t).all()for t in paramstates)
 assert eq(initial['gaussian5'].numpy(),cp['state_dict']['gaussian5'].numpy())
 optim=cp['optimizer_state'];assert len(optim['param_groups'])==1 and optim['param_groups'][0]['lr']==1e-4 and optim['param_groups'][0]['weight_decay']==.001 and all(float(x['step'])==2000 for x in optim['state'].values())
 for k,gen in [('native_pair',pg),('native_uniform',ug),('new_pair',epg),('new_uniform',eug)]:assert eq(cp['RNG'][k].numpy(),gen.get_state().numpy())
 recalc[arm]=dict(old_registered_native=orpt,continuous_registered_native=nrpt,native35=gates,native35_passed=sum(g['PASS']for g in gates.values()),raw_base_plus_delta_all26624_bytes_exact=True,metadata_and_original184U_exact=True,sampler_first2000_C32_and_arms_exact=True,exposure_old_new=[128000,128000],initial_state_CPU_tensor_seed_replay_exact=True,final_state_sha256=statehash(cp['state_dict']),optimizer_final_step2000=True)
assert rec_equal(checkpoints[ARMS[0]]['RNG'],checkpoints[ARMS[1]]['RNG'])
baseold,basenew=reports(inp['base_raw_logit']);baseline=dict(old_registered_native=baseold,continuous_registered_native=basenew,native35=native35(baseold,basenew))
result=dict(completed=True,PASS=True,control_complete=bind(RUN/'CONTROL_COMPLETE.json'),arms=recalc,C32_base=baseline,all26624_kept=True,selected_model=None,source_native_probe_not_admitted=True,AUX29_unmeasured_original64_not_claimed=True,all_new_performance_read_after_both_control_complete=True,limitations=['Saved loss sum is independently checked FP32 arithmetic, not a replay of intermediate neural logits/backward.','Final raw/delta repeated-pass exactness is recorded by producer receipt; this review rechecks all saved identities without a new model forward.'],actual_model_objects_or_forward_fit_encoder_baseline_RGB_current223_V112_PnP=0,CPU_checkpoint_state_and_raw_tensor_init_replay_only=True)
f=H/'INDEPENDENT_RECALCULATED_RESULTS.json'
with f.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
r=dict(completed=True,PASS=True,code=bind(Path(__file__)),control_complete=bind(RUN/'CONTROL_COMPLETE.json'),recalculated=bind(f),all26624_raw_base_plus_delta_and_metadata_bytes_exact=True,same2000_draws_exposures_sourceTRAIN_only=True,initial_seed_tensor_state_replay_and_final2000_optimizer_steps_exact=True,all_group_source_reports_and_native35_recalculated=True,original184U_SU_denominators_retained=True,source_probe_unadopted_selectednull=True,actual_model_object_forward_or_fit_calls=0,current223_V112_RGB_encoder_baseline_PnP=0,seconds=time.monotonic()-start)
f=H/'REVIEW.json'
with f.open('x')as stream:json.dump(r,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(f),arms={k:dict(native35=v['native35_passed'],oldVAL=v['old_registered_native']['VALIDATION'],newVAL=v['continuous_registered_native']['VALIDATION'])for k,v in recalc.items()})),flush=True)
