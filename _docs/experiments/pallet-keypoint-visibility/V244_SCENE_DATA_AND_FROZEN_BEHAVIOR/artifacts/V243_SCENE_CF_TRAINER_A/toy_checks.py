"""Pure invented SOURCE-shaped tensors; no actual sample, feature, weight or RGB reads."""
import ast,hashlib,json,os,sys,tempfile,types
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).absolute().parent
source=H/'train_scene_cf_data.py'
sys.dont_write_bytecode=True
m=types.ModuleType('v243_toy_trainer');m.__file__=str(source);exec(compile(source.read_text(),str(source),'exec'),m.__dict__)
old=Path(m.ORIGINAL_CODE_PIN[0]);oldraw=old.read_bytes();assert hashlib.sha256(oldraw).hexdigest()==m.ORIGINAL_CODE_PIN[1]
headpath=Path(m.MODULE_PIN[0]);raw=headpath.read_bytes();assert hashlib.sha256(raw).hexdigest()==m.MODULE_PIN[1]
h=types.ModuleType('v243_toy_head');h.__file__=str(headpath);exec(compile(raw,str(headpath),'exec'),h.__dict__)
checks=[]
# 1. Literal zero/UNKNOWN never becomes a successful pair. No chooser or RNG in step_plan.
n=32;family=np.array([0]*16+[1]*16,np.int8);truth=np.tile(np.array([0]*8+[1]*8,np.int8),2)
data={'family':family,'target':truth,'fit':np.ones(n,bool),'available':np.ones(n,bool)}
lookup=np.full(n,-1,np.int64);parentD=np.flatnonzero(truth==0);lookup[parentD]=np.arange(16)
proof=np.array([True,False]*8,bool);indices=np.arange(n,dtype=np.int64)
plan=m.step_plan(indices,data,lookup,proof)
for f in (0,1):
 for j in range(4):
  d=f*16+4+j;e=f*16+12+j;p=lookup[d]
  if proof[p]:assert plan[0][d]==plan[0][e]==d and plan[1][d]==plan[1][e]==p and plan[2][d]==1 and plan[2][e]==0
  else:assert plan[0][d]==d and plan[0][e]==e and plan[1][d]==plan[1][e]==-1 and plan[2][d]==plan[2][e]==-1
 for j in list(range(4))+list(range(8,12)):assert plan[0][f*16+j]==f*16+j and plan[2][f*16+j]==-1
assert all(np.array_equal(a,b) for a,b in zip(plan,m.step_plan(indices,data,lookup,proof)))
assert m.plan_digest(indices[None],[plan])==m.plan_digest(indices[None],[m.step_plan(indices,data,lookup,proof)])
checks.append('matched_last4_D_E_exact_same_parent; failed_both_natural; first4_unchanged; deterministic_no_new_RNG')
# 2. Random vectors prove branch wiring and one original geometry/condition for C/S.
rng=np.random.default_rng(241);groups=np.arange(n)%4;slots=np.arange(n)%8
features={'ORIGINAL_ALL_FEATURES.npy':rng.normal(size=(n,1186)).astype(np.float32),'NODE_CLS.npy':rng.normal(size=(4,8,384)).astype(np.float32),'CONTEXT_CLS.npy':rng.normal(size=(4,384)).astype(np.float32),'SAMPLE_GROUP_INDEX.npy':groups,'SAMPLE_NODE_INDEX.npy':slots,'RELATIVE_XY8.npy':rng.uniform(-8,8,(n,8,2)).astype(np.float32)}
geometry={'original_available8':np.ones((4,8),bool),'node_available':np.ones((4,8),bool),'context_fraction':np.ones(4,np.float32)}
cf={'RELATIVE_XY8.npy':features['RELATIVE_XY8.npy'][parentD].copy()}
for k,role in enumerate(m.ROLES):
 cf[role+'_BASE1186.npy']=np.full((16,1186),3+k,np.float32)
 cf[role+'_NODE_CLS.npy']=np.full((16,8,384),13+k,np.float32)
 cf[role+'_CONTEXT_CLS.npy']=np.full((16,384),23+k,np.float32)
values,target,fam=m.make_batch(indices,plan,data,features,cf,geometry,'cpu')
for pos,(pa,pi,ro) in enumerate(zip(*plan)):
 if ro<0:assert torch.equal(values[0][pos],torch.from_numpy(features['ORIGINAL_ALL_FEATURES.npy'][pos]))
 else:
  assert values[0][pos,0]==3+ro and values[1][pos,0,0]==13+ro and values[2][pos,0]==23+ro
  assert torch.equal(values[7][pos],torch.from_numpy(cf['RELATIVE_XY8.npy'][pi]))
 assert values[3][pos]==slots[pa]
assert torch.equal(target,torch.from_numpy(truth)) and torch.equal(fam,torch.from_numpy(family))
# family0 uses only output0, family1 only output1; BCE balanced mean fixed.
z=torch.randn(32,2,requires_grad=True);loss,parts=m.family_loss(z,target,fam)
assert torch.equal(loss,.5*(parts[0]+parts[1]));loss.backward()
assert torch.count_nonzero(z.grad[:16,1])==0 and torch.count_nonzero(z.grad[16:,0])==0
bad=target.clone();bad[0]=1
try:m.family_loss(z,bad,fam)
except ValueError:pass
else:raise AssertionError('Quota change not rejected')
checks.append('invented32batch_CFfeature_branch_and_original_geometry; fixed_two_referent_family_quota_and_gradient_routing')
# 3. Same original seed17 full model, exactly86054 parameters; single random toy step.
m.configure_runtime();torch.manual_seed(17);head=h.RelationalQueryHead().float()
assert sum(p.numel() for p in head.parameters())==86054
assert m.state_sha(head.state_dict())==m.INITIAL_FULL_SHA
assert m.state_sha(head.base.state_dict())==m.INITIAL_BASE_SHA
assert torch.count_nonzero(head.relational.readout[-1].weight)==0
optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);assert not optimizer.state
logits,_=head(*values);loss,_=m.family_loss(logits,target,fam);loss.backward();optimizer.step()
assert len(optimizer.state)==10 and all(float(v['step'])==1 for v in optimizer.state.values())
assert all(torch.isfinite(p).all() for p in head.parameters())
checks.append('random_only86054_original_initialSHA; zero_residual; one_toy_optimizerstep10states; no_actual_model_or_source')
# 4. Copied numerical helpers remain exact AST; false entry aborts before payload/code lookup.
names=('validate_geometry','relative_xy','paired_changes','state_sha','cpu_copy','validate_samples','fixed_draws','family_loss','reduce_rows','source_gate','byte_equal','global_feature_QA','context_args','configure_runtime')
a={v.name:ast.dump(v,include_attributes=False) for v in ast.parse(oldraw).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))}
b={v.name:ast.dump(v,include_attributes=False) for v in ast.parse(source.read_text()).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))}
assert all(a[k]==b[k] for k in names)
with tempfile.TemporaryDirectory(prefix='v243_invented_') as temp:
 p=Path(temp)/'false_entry.json';p.write_text(json.dumps({'approved_scene_CF_source_training':False,'source_only':True,'automatic_retry':False}))
 try:m.run(p)
 except PermissionError:pass
 else:raise AssertionError('False entry ran')
 p=Path(temp)/'blob.bin';p.write_bytes(b'invented artifact only')
 bb=m.binding(p);out=Path(temp)/'copy.bin';copied=m.copy_bound_opaque(bb,[bb],out)
 assert copied['sha256']==bb['sha256'] and out.read_bytes()==p.read_bytes()
for token in ('V112','__pycache__','opaque.pyc'):
 try:m.safe('/nonexistent/'+token+'/leaf')
 except ValueError:pass
 else:raise AssertionError('Lexical reject failed')
checks.append('14_original_helpers_ASTexact; falseentry_before_payload; invented_opaque_copy; forbidden_lexical_reject')
result={'completed':True,'PASS':True,'checks':checks,'groups':len(checks),'actual_SOURCE_payload_reads':0,'actual_trained_checkpoint_loads':0,'actual_DINO_instantiations':0,'actual_training_updates':0,'invented_optimizer_updates':1,'invented_source_features_only':True,'inherited_helpers_ASTexact':list(names)}
with (H/'TOY_CHECKS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
