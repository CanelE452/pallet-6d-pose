"""Independent CPU invented model and AST-extracted pure trainer checks only."""
import sys,os,types,ast,copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.dont_write_bytecode=True;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
H=Path(__file__).absolute().parent;V=H.parent;B=V/'fit_preparation_b';READS={};checks=[]
def need(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def rb(p,sha=None):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'literal safe code path')
 for q in(p,*p.parents):need(not q.is_symlink(),'no symlink code')
 z=p.read_bytes();digest=hashlib.sha256(z).hexdigest()
 if sha:need(digest==sha,'code binding')
 READS[str(p)]=digest;return z

def mod(name,binding):
 z=rb(binding['path'],binding['sha256']);m=types.ModuleType(name);m.__file__=binding['path'];sys.modules[name]=m;exec(compile(z,m.__file__,'exec'),m.__dict__);return m
p=json.loads(rb(B/'PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json','8df60dc146b5f026563341a1d959ecff20ef501333a14694c3b0794a2c0e07d8'))
base=mod('C_original_head',p['source_bindings']['existing_head']);candidate=mod('C_direct_candidate',p['source_bindings']['candidate']);new=mod('C_spatial_head',p['spatial_head']);adapter=mod('C_component_mask',p['mask_adapter'])
source=rb(p['trainer_code']['path'],p['trainer_code']['sha256']);tree=ast.parse(source)
ns=dict(np=np,torch=torch,hashlib=hashlib,c=SimpleNamespace(need=need,META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available'),qarray=lambda q:np.array([np.nan,np.nan]if q is None else q,np.float64)))
for name in ('native_indices','gather_native','nominal_fit_masks','gather_context','context_tensors','decode','cohort_counts','objective'):
 node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(B/'train_spatial_q8_nominal_source.py'),'exec'),ns)
def ok(name,test):need(test,name);checks.append({'name':name,'PASS':True})
Head=new.make_spatial_q8_head_class(base,candidate);Base=candidate.make_candidate_head_class(base)
torch.manual_seed(1810);original=Base().float();torch.manual_seed(1810);control=Head(spatial_q8=False).float();torch.manual_seed(1810);spatial=Head(spatial_q8=True).float()
ok('matched118359_and_original115222',sum(x.numel()for x in original.parameters())==115222 and sum(x.numel()for x in control.parameters())==sum(x.numel()for x in spatial.parameters())==118359)
ok('all_original_parameter_bytes_retained',all(torch.equal(t,control.state_dict()[k])and torch.equal(t,spatial.state_dict()[k])for k,t in original.state_dict().items()))
ok('paired_full_initial_state_equal',all(torch.equal(t,spatial.state_dict()[k])for k,t in control.state_dict().items()))
ok('final_residual_exactzero',all(torch.count_nonzero(x)==0 for x in spatial.spatial_residual[-1].parameters()))
rgb=torch.randint(0,256,(2,96,96,3),dtype=torch.uint8);valid=torch.ones((2,96,96),dtype=torch.bool);x=base.assemble_local_input(rgb,valid,use_sam=False)
native=torch.tensor([0,3],dtype=torch.int64);mask=torch.ones(2,8);mask[torch.arange(2),native]=0;xy=torch.linspace(-1,1,32).reshape(2,8,2);xy[~mask.bool()]=0;xy=xy.reshape(2,16).contiguous()
ok('RAW_SAM_channels_zero_originalXY',x.dtype==torch.float32 and torch.count_nonzero(x[:,6:])==0 and tuple(x.shape)==(2,8,96,96))
with torch.no_grad():
 encoded=spatial.encoder(x);pooled=spatial.pool(encoded);z=spatial.shared(pooled);a=original(x);b=control(x,native,mask,xy);d=spatial(x,native,mask,xy)
ok('encoder12_pool3x3_shared64',tuple(encoded.shape)==(2,64,12,12)and tuple(pooled.shape)==(2,64,3,3)and tuple(z.shape)==(2,64)and all(m.kernel_size==(3,3)for m in spatial.encoder if isinstance(m,torch.nn.Conv2d)))
ok('zero_residual_all_original_logits_exact',all(torch.equal(a[k],b[k])and torch.equal(a[k],d[k])for k in a))
record={}
def hook(name):return lambda module,args:record.setdefault(name,args[0].detach().clone()) and None
# Explicit hook bodies avoid returning an accidental replacement input.
def h0(module,args):record['control']=args[0].detach().clone()
def h1(module,args):record['spatial']=args[0].detach().clone()
hook0=control.spatial_residual[0].register_forward_pre_hook(h0);hook1=spatial.spatial_residual[0].register_forward_pre_hook(h1)
control(x,native,mask,xy);spatial(x,native,mask,xy);hook0.remove();hook1.remove()
ok('control_zero_only_xy16_native_mask_same',torch.equal(record['control'][:,:80],record['spatial'][:,:80])and torch.count_nonzero(record['control'][:,80:])==0 and torch.equal(record['spatial'][:,80:],xy)and torch.equal(record['spatial'][:,64:72],torch.nn.functional.one_hot(native,8).float())and torch.equal(record['spatial'][:,72:80],mask))
# Open the zero residual in invented states only to inspect the declared gradient path; no optimizer steps.
with torch.no_grad():
 for model in (control,spatial):model.spatial_residual[-1].weight.fill_(.125);model.spatial_residual[-1].bias.zero_()
for model,label in ((control,'control'),(spatial,'spatial')):
 model.zero_grad(set_to_none=True);q=xy.clone().requires_grad_(True);out=model(x,native,mask,q);out['direct_end_external_logit'].sum().backward();record[label+'_xygrad']=None if q.grad is None else q.grad.detach().clone();record[label+'_encodergrad']=model.encoder[0].weight.grad.detach().clone();record[label+'_direct']=out['direct_end_external_logit'].detach()
ok('control_xy_no_gradient_actual_xy_has_gradient',record['control_xygrad']is None and record['spatial_xygrad']is not None and torch.count_nonzero(record['spatial_xygrad'])>0)
ok('direct_residual_reaches_shared_backbone',torch.count_nonzero(record['control_encodergrad'])>0 and torch.count_nonzero(record['spatial_encodergrad'])>0 and not torch.equal(record['control_encodergrad'],record['spatial_encodergrad']))
with torch.no_grad():
 d1=spatial(x,native,mask,xy);d2=spatial(x,native,mask,-xy);c1=control(x,native,mask,xy);c2=control(x,native,mask,-xy)
ok('q8_direct_only_at_fixed_state',torch.equal(d1['support_logits'],d2['support_logits'])and torch.equal(d1['owner_logits'],d2['owner_logits'])and not torch.equal(d1['direct_end_external_logit'],d2['direct_end_external_logit'])and torch.equal(c1['direct_end_external_logit'],c2['direct_end_external_logit']))
for name,bad_native,bad_mask,bad_xy in [('self_nonzero',native,torch.ones_like(mask),xy),('mask_nonbinary',native,mask*.5,xy),('native_float',native.float(),mask,xy),('nan',native,mask,xy*float('nan')),('unclipped',native,mask,xy*100.)]:
 try:spatial(x,bad_native,bad_mask,bad_xy);passed=False
 except AssertionError:passed=True
 ok('invalid_'+name+'_rejected',passed)
roles={'SYNTH_AUG65536':np.array([True,True,False,True,False]),'PUBLIC_TRAIN1480':np.array([True,False,True]),'CF3816':np.array([True,False,True]),'DIRECT_CONTROL':np.array([True,False,True])};before={d:a.copy()for d,a in roles.items()};contexts={d:{'fit_nominal_eligible':np.ones(a.shape,bool)}for d,a in roles.items()};contexts['SYNTH_AUG65536']['fit_nominal_eligible']=np.array([True,False,True,True,True]);fit=ns['nominal_fit_masks'](roles,contexts)
ok('no_role_mask_mutation_or_TRAIN_to_HOLD_promotion',all(np.array_equal(a,before[d])for d,a in roles.items())and np.array_equal(fit['SYNTH_AUG65536'],[True,False,False,True,False])and np.array_equal(np.flatnonzero(~roles['SYNTH_AUG65536']),[2,4]))
ok('fitmask_new_array_not_role_alias',not np.shares_memory(fit['SYNTH_AUG65536'],roles['SYNTH_AUG65536']))
cache={'x':{'original_query_available':np.ones(5,bool),'native_slot':np.arange(5,dtype=np.int8)},'y':{}}
co=ns['cohort_counts'](cache,roles['SYNTH_AUG65536']);ok('saved_cohort_count_uses_original_role',co['TRAIN']['rows']==3 and co['HOLDOUT']['rows']==2 and fit['SYNTH_AUG65536'].sum()==2)
try:
 ns['nominal_fit_masks']({'PUBLIC_TRAIN1480':roles['PUBLIC_TRAIN1480']},{'PUBLIC_TRAIN1480':{'fit_nominal_eligible':np.array([True,False,True])}});passed=False
except AssertionError:passed=True
ok('non_synthetic_eligibility_drift_rejected',passed)
# Same ordered gather into all six losses; fabricated arrays contain row tokens rather than pixels.
D=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL');ctx={d:{'native_slot':np.arange(5,dtype=np.int64),'other_mask8':np.full((5,8),i,np.float32),'other_dxdy16':np.full((5,16),i+10,np.float32)}for i,d in enumerate(D)}
draw={k:np.array([i%5],np.int64)for i,k in enumerate(('visible','foreign','synthetic_owner','public_owner','CF_owner'))};dd=np.array([4],np.int64);g=ns['gather_context'](ctx,draw,dd)
ok('all6gather_domains_order',g['native_slot'].tolist()==[0,1,2,3,4,4]and g['other_mask8'][:,0].tolist()==[0,0,0,1,2,3]and g['other_dxdy16'][:,0].tolist()==[10,10,10,11,12,13])
av=np.array([True,True,True,False,True]);support=np.array([[0.,0.],[1.,1.],[-1.,1.],[1.,0.],[0.,1.]],np.float32);owner=np.ones((5,3),np.float32);direct=np.array([-1e-7,0.,1e-7,1.,np.nan],np.float32);dec=ns['decode'](support,owner,direct,av)
ok('strict_support_and_ge0_direct_with_U',dec[0].tolist()==[[0,0],[1,1],[0,1],[-1,-1],[0,1]]and dec[1].tolist()==[0,0,0,-1,0]and dec[2].tolist()==[0,1,1,-1,-1]and dec[3].tolist()==[-1,1,1,-1,1])
# Existing objective pure function is byte-AST identical to fixed parent; no source tensors or loss-body read.
old=ast.parse(rb(p['source_bindings']['previous_trainer']['path'],p['source_bindings']['previous_trainer']['sha256']));oldmap={f.name:ast.dump(f,include_attributes=False)for f in old.body if isinstance(f,ast.FunctionDef)};newmap={f.name:ast.dump(f,include_attributes=False)for f in tree.body if isinstance(f,ast.FunctionDef)}
names=['metadata_matches','load_old','load_direct','gather','native_indices','gather_native','train_native_class_priors','assemble','decode','objective','join_direct_split','split_masks','cohort_counts','fresh_run']
for name in names:
 if name=='train_native_class_priors':
  x=copy.deepcopy(next(f for f in old.body if isinstance(f,ast.FunctionDef)and f.name==name));y=copy.deepcopy(next(f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name==name))
  for node in (x,y):
   need(isinstance(node.body[0],ast.Expr)and isinstance(node.body[0].value,ast.Constant)and isinstance(node.body[0].value.value,str),'prior function has explicit docstring')
   node.body=node.body[1:]
  ok('parent_operational_AST_'+name,ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False))
 else:ok('parent_AST_'+name,oldmap[name]==newmap[name])
# Freeze control-flow order is inspected structurally, not by a production invocation.
fitnode=next(f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name=='fit');fittext=ast.unparse(fitnode)
ok('source_prediction_after_both_final_barrier',fittext.index("for arm, use_sam, spatial_q8 in ARMS")<fittext.index("ALL2_FINAL4000_FROZEN_BEFORE_SOURCE_TRAIN_OR_HOLDOUT_OUTPUTS.json")<fittext.index('for arm, r in pending.items()')<fittext.index('save_outputs('))
ok('4000_seed1810_direct1811_NO_SAM_fixed',"range(1, 4001)"in fittext and'np.random.default_rng(1810)'in fittext and'np.random.default_rng(1811)'in fittext and"False, False"in ast.unparse(tree)and"False, True"in ast.unparse(tree))
report={'completed':True,'PASS':True,'CPU_invented_only':True,'checks':checks,'check_count':len(checks),'optimizer_steps':0,'actual_source_arrays_headers_pixels_weights_or_model_checkpoint_reads':0,'production_model_or_fit':False,'code_bindings_read':READS,'gradient_fixture_note':'Only toy in-memory random heads; final residual weights manually made nonzero equally to test declared paths. This is not the initial-state gradient or source learning evidence.'}
b=(json.dumps(report,indent=2)+'\n').encode()
with(H/'B_CPU_AND_TRAINER_INVENTED_CHECKS.json').open('xb')as f:f.write(b)
print(json.dumps({'PASS':True,'checks':len(checks),'sha256':hashlib.sha256(b).hexdigest()}))
