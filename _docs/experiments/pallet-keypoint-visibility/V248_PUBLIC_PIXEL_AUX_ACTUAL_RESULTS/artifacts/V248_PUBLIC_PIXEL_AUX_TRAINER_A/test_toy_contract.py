"""Invented CPU tensors only: no SOURCE vectors/checkpoint/image/teacher payload."""
from pathlib import Path
import types,sys,json,hashlib,ast
H=Path(__file__).absolute().parent
s=(H/'train_public_pixel_aux.py').read_bytes()
m=types.ModuleType('v248_toy_trainer');m.__file__=str(H/'train_public_pixel_aux.py');m.__name__='v248_toy_trainer'
exec(compile(s,m.__file__,'exec'),m.__dict__)
np,torch=m.np,m.torch
torch.set_num_threads(1);torch.manual_seed(5);torch.use_deterministic_algorithms(True)
module=m.load_code({'path':m.MODULE_PIN[0],'sha256':m.MODULE_PIN[1]},[{'path':m.MODULE_PIN[0],'sha256':m.MODULE_PIN[1]}],'v248_toy_head',m.MODULE_PIN)
checks=[]
core=module.RelationalQueryHead().float();fake={'arm':m.OLD_ARM,'updates':1000,'head_parameters':86054,'use_lora':False,'draws_sha256':m.DRAW_SHA,'head_state_dict':m.cpu_copy(core.state_dict())}
models,warm_sha=m.make_models(module,fake,'cpu')
assert m.state_sha(models[m.CONTROL].state_dict())==m.state_sha(models[m.TREAT].state_dict())
assert sum(p.numel() for p in models[m.CONTROL].parameters())==86119
x=torch.randn(32,1186);node=torch.randn(32,8,384);ctx=torch.randn(32,384);active=torch.arange(32,dtype=torch.int64)%8;avail=torch.ones(32,8,dtype=torch.bool);support=avail.clone();cgood=torch.ones(32,dtype=torch.bool);rel=torch.zeros(32,8,2)
args=(node,ctx,active,avail,support,cgood,rel)
assert torch.equal(models[m.CONTROL].forward_main(x,*args),core(x,*args)[0])
assert torch.equal(models[m.CONTROL].forward_main(x,*args),models[m.TREAT].forward_main(x,*args))
checks.append({'name':'invented_warm_strict_load_and_main_forward_parity','PASS':True,'invented_core_state_only':True,'params':86119})
pixels=torch.randn(32,1154);targets=torch.tensor([0]*16+[1]*16,dtype=torch.int8)
probe=m.aux_gradient_probe(models[m.TREAT],pixels,targets)
assert probe['PASS'] and m.state_sha(models[m.CONTROL].state_dict())==m.state_sha(models[m.TREAT].state_dict())
checks.append({'name':'AUX_only_shared_visual_gradient_no_state_or_optimizer_change','PASS':True,'probe':probe})
models[m.CONTROL].zero_grad(set_to_none=True)
y=torch.tensor(([0]*8+[1]*8)*2,dtype=torch.int8);family=torch.tensor([0]*16+[1]*16,dtype=torch.int8)
loss,_=m.family_loss(models[m.CONTROL].forward_main(x,*args),y,family);loss.backward()
assert all(p.grad is None for p in models[m.CONTROL].public_pixel.parameters())
checks.append({'name':'CONTROL_main_only_public_gradNone','PASS':True,'optimizer_steps':0})
roles={'invented_fit':'AUX_FIT','invented_dev':'AUX_DEV'}
ad={'target':np.array([0,1,0,1],np.int8),'fit':np.array([1,1,0,0],bool),'available':np.ones(4,bool),'group_index':np.array([0,1,2,3],np.int64),'row_id':np.array(['toy0','toy1','toy2','toy3']),'query_xy':np.full((4,2),40.,np.float64),'L':np.full(4,64.,np.float64),'pair_id':np.array(['invented_fit']*2+['invented_dev']*2),'view':np.array(['A','B','A','B']),'source_role':np.array(['AUX_FIT']*2+['AUX_DEV']*2),'original_source_split':np.array(['TRAIN_LAB']*4)}
qa=m.validate_aux(ad,np.ones((4,1154),np.float32),roles);draws,eligible=m.aux_draws(ad)
assert qa['PASS'] and draws.shape==(1000,32) and np.array_equal(draws,m.aux_draws(ad)[0]) and (draws[:,:16]==0).all() and (draws[:,16:]==1).all()
bad={k:v.copy() for k,v in ad.items()};bad['fit'][2]=True
try:m.validate_aux(bad,np.ones((4,1154),np.float32),roles)
except ValueError:pass
else:raise AssertionError('DEV role promotion accepted')
checks.append({'name':'invented_aux_shape_roles_and_separate_PCG17_draws','PASS':True,'DEV_draws':0})
for p in ['/tmp/V112_ANY','/tmp/XCaChE/file','/tmp/x.PYC','/tmp/current223/a','/tmp/HOLD/a','/tmp/../safe']:
 try:m.safe(p)
 except ValueError:pass
 else:raise AssertionError('Forbidden lexical path accepted')
false=H/'TOY_FALSE_ENTRY.json'
false.write_text(json.dumps({'approved_source_public_pixel_aux_training':False}))
try:m.run(false)
except PermissionError:pass
else:raise AssertionError('False entry executed')
checks.append({'name':'cold_path_and_false_entry_before_payload','PASS':True,'actual_payload':0})
def metric(recall):return {f'family{f}_DEV':{k:{'recall':recall} for k in ('D','E')} for f in (0,1)}
b,c,t=metric(.90),metric(.92),metric(.94)
aux={'AUX_DEV':{'V':{'recall':.8},'E':{'recall':.8}}}
assert m.combined_gate(b,c,t,aux)['PASS']
aux['AUX_DEV']['E']['recall']=None
assert not m.combined_gate(b,c,t,aux)['PASS']
checks.append({'name':'fixed_three_gate_composition_and_empty_class_failure','PASS':True})
report={'completed':True,'PASS':True,'checks':checks,'random_CPU_tensors_only':True,'source_checkpoint_or_array_or_RGB_reads':0,'actual_model_or_encoder_instantiated':0,'random_small_head_instantiated':True,'optimizer_steps':0,'actual_training':0,'code_sha256':hashlib.sha256(s).hexdigest(),'head_CODE':{'path':m.MODULE_PIN[0],'sha256':m.MODULE_PIN[1]}}
(H/'TOY_CHECKS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'PASS':True,'toy_groups':len(checks),'actual_training':0}))
