"""Independent V223 random CPU fixtures only. No source payload or saved weights."""
from pathlib import Path
import os,sys,types,json,hashlib,ast,copy
import numpy as np
import torch
from torch.nn import functional as F
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';H=R/'V223_SPLIT_TARGET_TRAINER_QA_A';B=R/'V223_SPLIT_TARGET_TRAINING_B';O=R/'V222_PAIRED_BRIDGE_TRAINING_B_V2'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def need(x,m):
 if not bool(x):raise AssertionError(m)
def rb(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'lexical before FS')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(leaf,'rb')as f:return f.read()
def sha(b):return hashlib.sha256(b).hexdigest()
def load(b,name):
 need(Path(b['path']).suffix=='.py','code only');raw=rb(b['path']);need(sha(raw)==b['sha256'],'pinned code')
 m=types.ModuleType(name);m.__file__=b['path'];sys.modules[name]=m;exec(compile(raw,b['path'],'exec'),m.__dict__);return m
def nodes(raw):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(raw.decode()).body if isinstance(n,ast.FunctionDef)}
def clone_state(m):return {k:v.detach().clone()for k,v in m.state_dict().items()}
def point_params(m):return [(k,p)for k,p in m.named_parameters()if k.startswith(('direct_end_head.','spatial_residual.'))]
def part_params(m):return [(k,p)for k,p in m.named_parameters()if k.startswith(('part_direct_end_head.','part_spatial_residual.'))]

ready=json.loads(rb(B/'FINAL_READY.json'));need(sha(rb(B/'FINAL_READY.json'))=='fe30a2dc021108c46d77ab1411a7d1865c9c7e0449c46f06de0cd4c1f0c08140','READY pin');bindings=ready['bindings']
for b in bindings.values():need(sha(rb(b['path']))==b['sha256'],'complete ready graph text SHA')
proto=json.loads(rb(bindings['PROTOCOL_BEFORE_ANY_V223_SOURCE_PAYLOAD_OR_FIT.json']['path']))
contract_raw=rb(bindings['source_training_contract.py']['path']);runner_raw=rb(bindings['train_split_point_part_source.py']['path'])
oldc=nodes(rb(O/'source_training_contract_v2.py'));oldr=nodes(rb(O/'train_C65_continued_finite_part_CF_v2.py'));newc=nodes(contract_raw);newr=nodes(runner_raw)
samec=[k for k in oldc if oldc[k]==newc[k]];samer=[k for k in oldr if oldr[k]==newr[k]];need(len(samec)==17 and len(samer)==14,'17/14 inherited AST exact')
C=load(bindings['source_training_contract.py'],'a_v223_contract');base=load(proto['kernels']['base_head'],'a_v223_base');candidate=load(proto['kernels']['candidate'],'a_v223_candidate');spatial=load(proto['kernels']['spatial_head'],'a_v223_spatial');split=load(bindings['split_point_part_head.py'],'a_v223_split')
old_factory=spatial.make_spatial_q8_head_class(base,candidate);new_factory=split.make_split_target_head_class(spatial)
torch.manual_seed(1810);original=old_factory(spatial_q8=True).float()
# Simulate a nonzero trained-state shape using random tensors, never a saved C65 payload.
with torch.no_grad():
 for p in list(original.direct_end_head.parameters())+list(original.spatial_residual.parameters()):p.normal_(0,.03)
state=clone_state(original);models=[]
for _ in range(2):
 torch.manual_seed(1810);loaded=old_factory(spatial_q8=True).float();loaded.load_state_dict(state,strict=True);before=torch.random.get_rng_state().clone();m=new_factory(loaded).float();need(torch.equal(before,torch.random.get_rng_state()),'clone consumes no RNG');need(sum(p.numel()for p in m.parameters())==121561 and len(m.state_dict())==42,'expanded shape');need(all(torch.equal(m.state_dict()[k],v)for k,v in state.items()),'all original36 entries exact')
 need(len(point_params(m))==len(part_params(m))==6 and all(torch.equal(a,b)and a.data_ptr()!=b.data_ptr()for (_,a),(_,b)in zip(point_params(m),part_params(m))),'6 exact independent clones');models.append(m)
need(all(torch.equal(models[0].state_dict()[k],v)for k,v in models[1].state_dict().items()),'both arms complete initial state equal')
rgb=torch.randint(0,256,(2,96,96,3),dtype=torch.uint8);valid=torch.ones((2,96,96),dtype=torch.bool);x=base.assemble_local_input(rgb,valid,use_sam=False);native=torch.tensor([0,7]);mask=torch.zeros((2,8));xy=torch.zeros((2,16))
counts={'encoder':0};hook=models[0].encoder.register_forward_hook(lambda *args:counts.__setitem__('encoder',counts['encoder']+1))
with torch.no_grad():a=original(x,native,mask,xy);b=models[0](x,native,mask,xy)
hook.remove();need(counts['encoder']==1 and torch.equal(a['direct_end_external_logit'],b['point_logit'])and torch.equal(b['point_logit'],b['part_logit'])and torch.equal(a['support_logits'],b['support_logits'])and torch.equal(a['owner_logits'],b['owner_logits']),'one feature call and initial base/point/part logits exact')
# Actual production routed objective, with independent leaf logits to verify every slice.
old_tree=ast.parse(rb(proto['kernels']['old_trainer']['path']).decode());ns=dict(torch=torch,F=F,require=need);n=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef)and n.name=='five_losses');exec(compile(ast.Module(body=[n],type_ignores=[]),'<old five losses>','exec'),ns)
point=torch.tensor([0]*16+[1]*16);natural=torch.tensor([0]*8+[1]*8)
for separate in(False,True):
 out={'support_logits':torch.randn(252,2,requires_grad=True),'owner_logits':torch.randn(252,3,requires_grad=True),'point_logit':torch.randn(252,requires_grad=True),'part_logit':torch.randn(252,requires_grad=True)}
 loss,parts,strong,local,cf=C.routed_objective(out,point,natural,separate,ns['five_losses']);z=out['part_logit'][240:].reshape(4,3);expected_cf=(F.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.,0.]).expand(4,3),reduction='none')*torch.tensor([.25,.5,.25])).sum(1).mean();need(torch.equal(cf,expected_cf),'parent balanced BCE formula');need(torch.allclose(loss,torch.stack(parts).mean()+strong+.25*local+.25*expected_cf,atol=1e-7,rtol=0),'full loss formula');loss.backward()
 if separate:
  need(out['point_logit'].grad[:192].count_nonzero()==0 and out['point_logit'].grad[192:224].count_nonzero()==32 and out['point_logit'].grad[224:].count_nonzero()==0,'strong POINT only');need(out['part_logit'].grad[:224].count_nonzero()==0 and out['part_logit'].grad[224:].count_nonzero()==28,'weakCF PART only')
 else:need(out['point_logit'].grad is None and out['part_logit'].grad[:192].count_nonzero()==0 and out['part_logit'].grad[192:].count_nonzero()==60,'control all direct losses PART')
# Real random encoder/head graph: direct objectives keep encoder gradients, optimizer skips nuisance POINT.
optimizer_counts=[]
for separate,m in zip((False,True),models):
 m.zero_grad(set_to_none=True);o=m(x,native,mask,xy);direct=o['point_logit'if separate else'part_logit'].square().mean();direct.backward();need(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.encoder.parameters()),'shared encoder direct gradient not detached')
 m.zero_grad(set_to_none=True);o=m(x,native,mask,xy);loss=o['part_logit'].square().mean()+o['support_logits'].square().mean()+o['owner_logits'].square().mean()
 if separate:loss=loss+o['point_logit'].square().mean()
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);need(not opt.state,'reset AdamW');prior={k:p.detach().clone()for k,p in point_params(m)};loss.backward()
 need(all(p.grad is not None for _,p in part_params(m))and all((p.grad is not None)==separate for _,p in point_params(m)),'actual head gradient routing')
 opt.step();expected=42 if separate else 36;need(len(opt.state)==expected,'optimizer state used-only36/42');need(all(float(s['step'])==1 for s in opt.state.values()),'invented one-step counters')
 if not separate:need(all(p not in opt.state and torch.equal(p,prior[k])for k,p in point_params(m)),'control unused POINT unchanged/no AdamWstate')
 optimizer_counts.append(len(opt.state))
# Both-arm CF True must advance the same four independent generators.
drawnode=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef)and n.name=='draw_step');drawns=dict(np=np,require=need);exec(compile(ast.Module(body=[drawnode],type_ignores=[]),'<old draw step>','exec'),drawns)
pools={'support':[[np.arange(20),np.arange(20,40)]for _ in range(2)],'conditional':[np.full(40,1/40)for _ in range(3)],'public':[np.arange(16*i,16*(i+1))for i in range(3)],'CF':[np.arange(16*i,16*(i+1))for i in range(2)]};balanced=[np.arange(20),np.arange(20,40)];draws=[]
for _ in range(2):
 rngs=[np.random.default_rng(seed)for seed in(1810,1811,1812,1813)];draws.append([C.draw_common_and_cf(drawns['draw_step'],*rngs,pools,balanced,balanced,39,True)for _ in range(32)])
for a,b in zip(*draws):need(all(np.array_equal(a[0][k],b[0][k])for k in a[0])and all(np.array_equal(a[i],b[i])for i in(1,2,3))and sum(map(len,a[0].values()))+len(a[1])+len(a[2])+3*len(a[3])==252,'all four RNGs and252 layout same')
finger=next(n for n in ast.parse(runner_raw.decode()).body if isinstance(n,ast.FunctionDef)and n.name=='input_fingerprint');fn=dict(C=C,np=np,hashlib=hashlib);exec(compile(ast.Module(body=[finger],type_ignores=[]),'<production fingerprint>','exec'),fn);r=rgb[0].numpy();v=valid[0].numpy();c=np.zeros(32,np.float32);c[0]=1;f=fn['input_fingerprint'](r,v,c);r2=r.copy();r2[0,0,0]^=1;need(f==fn['input_fingerprint'](r.copy(),v.copy(),c.copy())and f!=fn['input_fingerprint'](r2,v,c),'effective input fingerprint exact/sensitive')
source=runner_raw.decode();need(source.index("freeze=write(out/'ALL2_FINAL_FROZEN")<source.index("dev_manifest=js(e['CF_DEV_manifest']")<source.index("fingerprint=write(out/'SOURCE_INPUT_FINGERPRINTS.json'"),'seal before DEV decode/fingerprint');need("use_cf=True"in source and "steps==1100"in source and "FIXED_SPLIT_TARGET_FIT"in source and "THROUGHPUT_PROBE"not in source,'both CF/fixed1100/no probe')
result={'completed':True,'PASS':True,'ready_sha256':sha(rb(B/'FINAL_READY.json')),'parameters':121561,'state_tensors':42,'clone_tensors':6,'clone_nonalias_rng_unchanged_original36_preserved':True,'initial_C65_POINT_PART_logits_exact':True,'encoder_calls_per_forward':1,'actual_objective_leaf252_slice_and_BCE_gradient_exact':True,'shared_encoder_gradient_preserved':True,'invented_optimizer_state_counts_control_treatment':optimizer_counts,'control_POINT_grad_None_no_state_and_unchanged':True,'both_CF_true_four_RNG32step_and252_batch_exact':True,'fingerprint_actual_function_exact_and_sensitive':True,'seal_before_DEV_and_fingerprint_code_order':True,'inherited_contract17_and_runner14_AST_exact':True,'actual_SOURCE_payload_or_savedweights_reads':0,'CPU_random_model_only':True,'invented_optimizer_steps':2,'actual_SOURCE_training_updates':0}
p=H/'INDEPENDENT_CPU_CHECKS.json'
with p.open('x')as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
print(json.dumps(result,ensure_ascii=False));print('CHECKS_SHA256',sha(rb(p)))
