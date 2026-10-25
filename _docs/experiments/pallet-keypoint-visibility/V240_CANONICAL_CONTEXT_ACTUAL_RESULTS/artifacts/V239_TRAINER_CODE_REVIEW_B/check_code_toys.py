from pathlib import Path
import ast,hashlib,json
import numpy as np
import torch
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=Path(__file__).parent
p=R/'V239_CANONICAL_PAIRED_TRAINER_A/train_canonical_pair.py';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419'
n={'__name__':'v239_invented','__file__':str(p)};exec(compile(raw,str(p),'exec'),n)
p=R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py';raw_module=p.read_bytes();assert hashlib.sha256(raw_module).hexdigest()=='dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9'
m={'__name__':'random_module'};exec(compile(raw_module,str(p),'exec'),m)
torch.set_num_threads(1);torch.manual_seed(17);control=m['CropFamilyHead']();torch.manual_seed(17);treat=m['RelationalQueryHead']()
assert n['state_sha'](control.state_dict())==n['state_sha'](treat.base.state_dict())==n['INITIAL_BASE_SHA']
assert [sum(p.numel() for p in x.parameters()) for x in (control,treat)]==[76098,86054]
x=torch.randn(32,1186);nodes=torch.randn(32,8,384);ctx=torch.randn(32,384);slots=torch.arange(32)%8;av=torch.ones(32,8,dtype=torch.bool)
z,_=treat(x,nodes,ctx,slots,av,av,torch.ones(32,dtype=torch.bool),torch.zeros(32,8,2));assert torch.equal(z,control(x))
leaf=torch.randn(32,2,requires_grad=True);family=torch.repeat_interleave(torch.arange(2),16);target=torch.tensor(([0]*8+[1]*8)*2)
loss,_=n['family_loss'](leaf,target,family);loss.backward();assert torch.isfinite(leaf.grad).all()
assert not leaf.grad[:16,1].count_nonzero() and not leaf.grad[16:,0].count_nonzero() and leaf.grad[:16,0].abs().sum()>0 and leaf.grad[16:,1].abs().sum()>0
# Small invented canonical inputs exercise byte sharing/condition/fractions with no payload reads.
rng=np.random.default_rng(17);group=np.array([0,0,1],np.int64);slot=np.array([0,2,5],np.int64);q=np.asarray([[[10*j,5*j] for j in range(8)]]*2,np.float64)
av=np.ones((2,8),bool);g={'q8':q,'L8':np.full((2,8),100.,np.float64),'original_available8':av,'node_available':av.copy(),'node_fraction':np.ones((2,8),np.float32),'context_fraction':np.array([1.,0.],np.float32),'box_xyxy':np.array([[0,0,100,100]]*2,np.float64)}
rel=n['relative_xy'](g,group,slot);cond=np.zeros((3,32),np.float32)
for i in (0,1):cond[i,slot[i]]=1;cond[i,8:16]=1;cond[i,8+slot[i]]=0;cond[i,16:]=rel[i].ravel()
node=rng.normal(size=(2,8,384)).astype(np.float32);context=np.ones((2,384),np.float32);context[1]=0
xx=np.zeros((3,1186),np.float32);xx[:,:384]=node[group,slot];xx[:,1152:1154]=1;xx[:,-32:]=cond
data={'fit':np.array([1,1,0],bool),'available':np.ones(3,bool),'condition':cond,'fractions':np.ones((3,2),np.float32),'family':np.array([0,0,1],np.int8)}
f={'ORIGINAL_ALL_FEATURES.npy':xx,'NODE_CLS.npy':node,'CONTEXT_CLS.npy':context,'SAMPLE_GROUP_INDEX.npy':group,'SAMPLE_NODE_INDEX.npy':slot,'RELATIVE_XY8.npy':rel}
assert n['global_feature_QA'](f,data,g)['checked']==3
xx[0,0]=np.nextafter(xx[0,0],np.float32(np.inf))
try:n['global_feature_QA'](f,data,g)
except ValueError:pass
else:raise AssertionError('One-ULP feature drift accepted')
xx[0,0]=node[0,0,0];xx[0,-1]=1
try:n['global_feature_QA'](f,data,g)
except ValueError:pass
else:raise AssertionError('Condition drift accepted')
# Draws must never sample a DEV sentinel and preserve each family8D8E.
d={'fit':np.tile(np.array([1,1,1,0],bool),4),'available':np.ones(16,bool),'family':np.repeat(np.array([0,1],np.int8),8),'target':np.tile(np.repeat(np.array([0,1],np.int8),4),2)}
draws,eligible=n['fixed_draws'](d);assert draws.shape==(1000,32) and eligible[draws].all()
assert np.array_equal(draws,n['fixed_draws'](d)[0])
for batch in draws:
 assert all(sum((d['family'][batch]==a)&(d['target'][batch]==b))==8 for a in (0,1) for b in (0,1))
a={f'family{f}_DEV':{k:{'recall':.5} for k in ('D','E')} for f in (0,1)};b={f'family{f}_DEV':{k:{'recall':.95} for k in ('D','E')} for f in (0,1)}
b['family0_DEV']['D']['recall']=.89;assert not n['source_gate'](a,b)['PASS'];b['family0_DEV']['D']['recall']=.9;assert n['source_gate'](a,b)['PASS'];b['family1_DEV']['E']['recall']=.49;assert not n['source_gate'](a,b)['PASS']
# Static chronology independent of any actual feature/state.
t=ast.parse(raw);run=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='run');tr=next(x for x in run.body if isinstance(x,ast.Try))
text=ast.get_source_segment(raw.decode(),run);assert text.index("global_QA=write_json")<text.index("module=load_code")<text.index("train_arm(arm,head")
assert text.index("for arm,head in ((CONTROL,control),(TREAT,treat))")<text.index("freeze=write_json")<text.index("available=np.flatnonzero(data['available'])")
assert not any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr in ('load','forward_features') and isinstance(c.func.value,ast.Name) and c.func.value.id in ('torch','model') for c in ast.walk(t))
result={'completed':True,'PASS':True,'fixture_groups':4,'seed17_base_SHA_exact':n['INITIAL_BASE_SHA'],'76098_86054_and_zeroresidual_output_exact':True,'family_gradient_routes_two_referents_exact':True,'canonical_byte_and_condition_drift_rejected':True,'PCG17_FIT_only1000_family8D8E':True,'SYN90_and_YCB50_fixed_gates_checked':True,'global_QA_before_optimizers_and_bothfreeze_before_DEV_CODE_checked':True,'actual_SOURCE_data_weights_images_features_fit':0,'random_leaf_backward_calls':1,'optimizer_steps':0}
p=H/'INDEPENDENT_TOY_CHECKS.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
