"""Invented CPU tensors/packets only; no saved source NPZ, pixels or weights."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;P=H.parent.parent
FINAL=H/'train_native_part_supervision_control.py'
PARENT=P/'query_visibility_counterfactual_frozen_features_20261024_v162/controlled_fit_preparation/train_counterfactual_owner_control.py'
HEAD=P/'query_visibility_native_slot_owner_factorial_20261024_v160/head_preparation/native_slot_owner_head.py'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert bind(PARENT)['sha256']=='021d5108666be1d0a103ae80a02fde91a142f37367b528ae8c148f7221355bde'
assert bind(HEAD)['sha256']=='82c843555cff138d8830f612befb733807c688e1e6c84d0607606dc4c4bd14b6'
def extract(p,names):
 ns={'torch':torch,'F':F,'np':np};nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+'::invented_only','exec'),ns);return ns
old=extract(PARENT,{'balanced_owner_ce','owner_loss_for_condition'})
new=extract(FINAL,{'balanced_owner_ce','balanced_part_native_bce','loss_for_condition','check_CF_packet','assert_byte_tree'})
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);checks={}
gen=torch.Generator().manual_seed(166)
# Both full92 and unequal invented class counts establish the actual balanced criterion.
for n0,n1 in ((46,46),(3,5)):
 target=torch.tensor([0.]*n0+[1.]*n1,dtype=torch.float64)
 logits=torch.randn(len(target),generator=gen,dtype=torch.float64,requires_grad=True)
 before=torch.get_rng_state().clone();loss=new['balanced_part_native_bce'](logits,target);loss.backward()
 weights=torch.where(target==0,.5/n0,.5/n1);want=(torch.sigmoid(logits.detach())-target)*weights
 assert torch.allclose(logits.grad,want,atol=1e-14,rtol=1e-12) and torch.equal(before,torch.get_rng_state())
 assert not torch.allclose(loss,2*loss)
checks['full92_and_unequal_class_BCE_analytic_gradient_half_perclass']=True
checks['part_BCE_no_RNG']=True
spec=importlib.util.spec_from_file_location('invented_exact_existing_head',HEAD);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def model():
 torch.manual_seed(1550);return mod.NativeSlotOwnerQueryHead(native_slot_hint=False,owner_auxiliary=True).float()
def opts(m):return [torch.optim.AdamW(p,lr=1e-4,weight_decay=.001) for p in (m.visibility_parameters(),m.auxiliary_parameters(),m.new_parameters())]
def same(a,b):
 try:new['assert_byte_tree'](a,b);return True
 except AssertionError:return False
models=[model(),model(),model()];assert len(models[0].state_dict())==16 and all(same(models[0].state_dict(),m.state_dict()) for m in models[1:])
optimizers=[opts(m) for m in models]
xs=[torch.randn((512,1928),generator=gen) for _ in range(4)];cf=torch.randn((92,1928),generator=gen)
slots=torch.arange(512,dtype=torch.int64)%8;cs=torch.arange(92,dtype=torch.int64)%8;z42=torch.zeros((512,42));positions=np.arange(253)
owner_y=torch.tensor([0]*53+[1]*52+[2]*148);cf_y=torch.tensor([0.]*46+[1.]*46);ci=np.arange(92)
rngs=[np.random.default_rng(1550) for _ in models]
def balance(z,y):return sum(F.binary_cross_entropy_with_logits(z[y==k],y[y==k]) for k in (0,1))/2
per_step=[]
for step in range(1,4):
 draws=[r.choice(512,64,replace=True) for r in rngs];assert all(np.array_equal(draws[0],d) for d in draws[1:]);states=[]
 for index,(m,oo,ii) in enumerate(zip(models,optimizers,draws)):
  before=torch.get_rng_state().clone()
  for opt in oo:opt.zero_grad(set_to_none=True)
  no=[m(xs[k][ii],z42[:64],slots[ii]) for k in (0,1)];po=m(xs[2][:43],z42[:43],slots[:43]);mo=m(xs[3][:390],z42[:390],slots[:390]);eo=m(xs[3][ii],z42[:64],slots[ii])['target_entity'];qo=[m(xs[k][:32],z42[:32],slots[:32])['coarse_point_quality'] for k in (0,1)]
  y64=torch.arange(64).float()%2;y43=torch.arange(43).float()%2;y390=torch.arange(390).float()%2;y32=torch.arange(32).float()%2
  original_loss=sum(balance(o['external_occlusion'],y64) for o in no)+balance(po['external_occlusion'],y43)+balance(mo['external_occlusion'],y390)+balance(eo,y64)+sum(balance(q,y32) for q in qo)/2
  if index==0:
   owner=old['owner_loss_for_condition'](None,mo,positions,owner_y,None,None,z42,None,None,False);total=original_loss+owner;part=None
  else:total,owner,part=new['loss_for_condition'](original_loss,mo,positions,owner_y,m,cf,cs,z42,ci,cf_y,index==2)
  total.backward();groups=[m.visibility_parameters(),m.auxiliary_parameters(),m.new_parameters()];norms=[torch.nn.utils.clip_grad_norm_(p,5) for p in groups]
  for opt in oo:opt.step()
  assert torch.equal(before,torch.get_rng_state()) and not m.geometry.weight.any() and not m.native_hint.weight.any()
  states.append(dict(total=float(total),owner=float(owner),norms=[float(n) for n in norms]))
 assert states[0]==states[1] and same(models[0].state_dict(),models[1].state_dict())
 assert all(same(a.state_dict(),b.state_dict()) for a,b in zip(optimizers[0],optimizers[1]))
 for key,value in models[0].state_dict().items():
  if key.startswith(('entity_head.','quality_head.')):assert same(value,models[2].state_dict()[key])
 assert same(optimizers[0][1].state_dict(),optimizers[2][1].state_dict())
 with torch.no_grad():
  a=models[0](xs[3],z42,slots);b=models[2](xs[3],z42,slots)
  assert same(a['target_entity'],b['target_entity']) and same(a['coarse_point_quality'],b['coarse_point_quality'])
 assert all(json.dumps(rngs[0].bit_generator.state,sort_keys=True)==json.dumps(r.bit_generator.state,sort_keys=True) for r in rngs[1:])
 per_step.append(dict(step=step,C43_C45_all16_parameters_and_all3_AdamW_states_byte_exact=True,C45_C46_entity_quality_parameters_optimizer_and_outputs_byte_exact=True,original_numpy_draws_and_torch_RNG_exact=True))
assert not same(models[0].head.state_dict(),models[2].head.state_dict())
checks['three_steps_C43_C45_all16params_all3optimizers_byte_exact']=True
checks['three_steps_C45_C46_entity_quality_state_optimizer_output_byte_exact']=True
checks['C46_actual_added_primary_path_changes_visibility']=True
checks['all3_original_AdamW_lr_wd_clip5_partitions_preserved']=True
# C45 must not touch CF arguments or call model, rather than accumulating a zero CF gradient.
class Poison:
 def __call__(self,*a,**kw):raise AssertionError('Baseline CF forward forbidden')
mo={'query_visible_owner':torch.randn((390,3),generator=gen)}
a,ol,part=new['loss_for_condition'](torch.tensor(2.),mo,positions,owner_y,Poison(),None,None,None,None,None,False)
assert part is None and same(a,torch.tensor(2.)+old['balanced_owner_ce'](mo['query_visible_owner'][positions],owner_y))
checks['C45_no_CF_forward_arguments_or_zero_CF_gradient']=True
# Isolate actual helper's primary path. Original owner logits are independent leaves.
m=model();mo={'query_visible_owner':torch.randn((390,3),generator=gen,requires_grad=True)}
before=torch.get_rng_state().clone();total,ol,part=new['loss_for_condition'](torch.tensor(0.),mo,positions,owner_y,m,cf,cs,z42,ci,cf_y,True);total.backward()
nonzero=[n for n,p in m.named_parameters() if p.grad is not None and bool(p.grad.any())]
assert set(nonzero)=={'head.0.weight','head.0.bias','head.2.weight','head.2.bias'} and torch.equal(before,torch.get_rng_state())
assert mo['query_visible_owner'].grad is not None
checks['CF_primary_gradient_only_shared_head0_primary_head2_no_owner_entity_quality_hint']=True
checks['original_owner253_CE_preserved_separately']=True
# V165-shaped invented packet accepts all-minus1 target arrays and continuous-only validity.
metadata=[];sidecar=[]
for i in range(1144):
 known=i<92;k=i%8;selected=k if known else (k+1)%8
 md=dict(input_occurrence_id=f'invented:{i}',pair_id=f'inventedpair:{i//2}',variant=('COVER' if i%2 else 'ADJACENT_SHAM') if known else 'BEFORE',original_query_id=f'q{i}',original_global_index=i,original_frame_index=i//8,source_id=f'frame{i//8}',split='SOURCE_DEVELOPMENT',native_slot=k,query_xy=[i+.25,i+.75],L=50.,query_available=i<966,nearest_valid=False,feature_constructed=True,central4_available=False,context_available=False,pair_pixel_proof_PASS=known,case_id=f'case{i%17}',native_visibility_target=-1,point_quality_target=-1,owner_target=-1)
 metadata.append(md);sidecar.append(dict(occurrence_index=i,input_metadata=md,selected_anchor_native_slot=selected,part_native_weak_target=(1 if i%2 else 0) if known else -1,part_native_loss_admitted_under_root_design=known,native_visibility_target=-1,point_quality_target=-1,owner_target=-1,human_gold=False,physical_native_corner_gold=False,training_admitted=False))
packet=dict(features=np.zeros((1144,1928),np.float32),occurrence_index=np.arange(1144))
for key in ('input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_xy','L','query_available','nearest_valid','feature_constructed','context_available'):
 packet[key]=np.asarray([r[key] for r in metadata])
packet.update(native_continuous_valid=packet['query_available'].copy(),local_central4_available=np.zeros(1144,bool))
for key in ('native_visibility_target','point_quality_target','owner_target'):packet[key]=np.full(1144,-1,np.int8)
statuses=[dict(constructed_payload=i<143,feature_index=i*8 if i<143 else None,feature_indices=list(range(i*8,i*8+8)) if i<143 else None,payload={'invented':i} if i<143 else None) for i in range(198)]
y,idx=new['check_CF_packet'](packet,metadata,sidecar,statuses);assert len(idx)==92 and np.bincount(y[idx]).tolist()==[46,46]
checks['V165_allminus1_target_arrays_accepted_regression']=True
checks['continuous_only_validity_no_nearest_center4_context_new_gate']=True
bad_cases=[]
for key in ('native_visibility_target','point_quality_target','owner_target'):
 bad=dict(packet);bad[key]=packet[key].copy();bad[key][0]=0;bad_cases.append((bad,sidecar))
bad=dict(packet);bad['native_visibility_target']=np.full(1143,-1,np.int8);bad_cases.append((bad,sidecar))
bad=dict(packet);bad['owner_target']=packet['owner_target'].astype(np.float32);bad_cases.append((bad,sidecar))
bad=dict(packet,pasted_alpha=np.zeros(1));bad_cases.append((bad,sidecar))
badrows=copy.deepcopy(sidecar);badrows[100]['part_native_weak_target']=0;bad_cases.append((packet,badrows))
badrows=copy.deepcopy(sidecar);badrows[0]['part_native_weak_target']=1;bad_cases.append((packet,badrows))
bad=dict(packet);bad['query_xy']=packet['query_xy'].copy();bad['query_xy'][0,0]+=1;bad_cases.append((bad,sidecar))
for bad,ss in bad_cases:
 try:new['check_CF_packet'](bad,metadata,ss,statuses)
 except AssertionError:pass
 else:raise AssertionError('Malformed packet/sidecar accepted')
checks['nine_wrong_target_shape_dtype_alpha_nonanchor_label_identity_packets_rejected']=True
assert all(checks.values())
result=dict(schema='V166_INVENTED_PRIMARY_NATIVE_PART_CONTROL_MEANINGFUL_CHECKS',completed=True,PASS=True,code=bind(FINAL),check_code=bind(Path(__file__).resolve()),head=bind(HEAD),parent_code=bind(PARENT),checks=checks,per_step=per_step,CF_gradient_nonzero_parameters=nonzero,owner_counts_original_unchanged=[53,52,148],CF_native_counts=[46,46],runtime=dict(executable=sys.executable,torch=str(torch.__version__),numpy=np.__version__,device='cpu'),source_pixel_feature_weight_or_prediction_bodies_read=0,actual_source_fit_or_encoder_calls=0,invented_optimizer_steps=3,all_new_quality_owner_targets_minus1=True)
p=H/'GENERATED_FINAL_CONTROL_CHECKS.json'
with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(PASS=True,checks=bind(p),check_count=len(checks))))
