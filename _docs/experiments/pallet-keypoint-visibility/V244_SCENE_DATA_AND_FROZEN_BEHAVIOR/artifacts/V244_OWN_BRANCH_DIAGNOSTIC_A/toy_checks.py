"""V244 invented tensor-only branch tests. Actual SOURCE/weights/vectors0."""
from pathlib import Path
import ast,hashlib,json,sys,tempfile,types
import numpy as np
import torch
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent
p=H/'diagnose_own_branches.py';m=types.ModuleType('v244_toy');m.__file__=str(p);exec(compile(p.read_text(),str(p),'exec'),m.__dict__)
hp=Path(m.MODULE_PIN[0]);raw=m.read(hp);assert hashlib.sha256(raw).hexdigest()==m.MODULE_PIN[1]
h=types.ModuleType('v244_toy_frozen_class');exec(compile(raw,str(hp),'exec'),h.__dict__)
torch.set_num_threads(1);torch.manual_seed(170)
model=h.RelationalQueryHead().eval().requires_grad_(False)
# Invent nonzero branch outputs; no saved checkpoint or optimizer is present.
with torch.no_grad():
 model.base.head[-1].weight.copy_(torch.randn_like(model.base.head[-1].weight)*.02)
 model.relational.readout[-1].weight.copy_(torch.randn_like(model.relational.readout[-1].weight)*.02)
x=torch.randn(8,1186);node=torch.randn(8,8,384);context=torch.randn(8,384);slots=torch.arange(8,dtype=torch.int64);orig=torch.ones(8,8,dtype=torch.bool);support=torch.rand(8,8)>.2;cs=torch.ones(8,dtype=torch.bool);rel=torch.rand(8,8,2)*16-8
args=(node,context,slots,orig,support,cs,rel)
b,r,t,details=m.own_branches(model,x,args)
assert torch.equal(t,b+r) and torch.equal(t,model(x,*args)[0])
assert not torch.equal(b,t) and torch.count_nonzero(r)
assert sum(p.numel() for p in model.parameters())==86054
checks=['same_random_frozen_checkpoint_ownbase_plus_ownresidual_byteequals_forward; no_independent_CONTROL']
# Each family chooses its own scalar logit at0; argmax would be a different problem.
base=np.array([[-2,50],[2,50],[-2,-50],[2,-50],[-50,-2],[50,2],[-50,-2],[50,2]],np.float32)
res=np.array([[3,0],[-3,0],[-.1,0],[.1,0],[0,3],[0,-3],[0,-.1],[0,.1]],np.float32)
total=base+res;family=np.array([0]*4+[1]*4,np.int8);target=np.array([1,0,0,1,0,1,0,1],np.int8)
data={'family':family,'target':target,'fit':np.array([True,True,False,False]*2),'available':np.ones(8,bool),'row_id':np.array(['invented'+str(i) for i in range(8)])}
saved=[{'selected_family_logit':float(total[i,family[i]]),'prediction':'E' if total[i,family[i]]>=0 else 'D'} for i in range(8)]
rows=m.rows_from_branches(data,saved,base,res,total)
assert [row['target_effect'] for row in rows]==['RECOVERED','RECOVERED','UNCHANGED_CORRECT','UNCHANGED_CORRECT','DAMAGED','DAMAGED','UNCHANGED_CORRECT','UNCHANGED_CORRECT']
assert rows[0]['threshold_crossing']=='D_TO_E' and rows[1]['threshold_crossing']=='E_TO_D'
assert rows[0]['target_aligned_residual']==3 and rows[4]['target_aligned_residual']==-3
assert rows[1]['own_total_prediction']=='D' and np.argmax(total[1])==1
metrics,strata=m.summarize(rows);assert sum(item['rows'] for item in strata)==8
assert metrics['own_total']['family0_FIT']['CM_D_E_by_D_E_U']==[[1,0,0],[0,1,0]]
checks.append('invented_family_scalar_zero_threshold_NOT_argmax; target_aligned_residual_and_recovery_damage_crossings; full_strata_retention')
# Fixed replay tolerances and sign invariant are independent checks.
expected=total.copy();actual=total.copy()+np.float32(1e-6)
assert np.allclose(actual,expected,atol=m.ATOL,rtol=m.RTOL)
assert np.array_equal(actual[np.arange(8),family]>=0,expected[np.arange(8),family]>=0)
wrong=total.copy();wrong[0,0]+=.1;assert not np.allclose(wrong,expected,atol=m.ATOL,rtol=m.RTOL)
checks.append('same_existing1e-5_absolute_relative_replay_boundary; exact_predicted_sign_guard')
with tempfile.TemporaryDirectory(prefix='v244_invented_') as tmp:
 p=Path(tmp)/'false.json';p.write_text(json.dumps({'approved_frozen_SOURCE_own_branch_diagnosis':False,'source_only':True,'automatic_retry':False}))
 try:m.run(p)
 except PermissionError:pass
 else:raise AssertionError('False entry executed')
for token in ('v112','V112','__pycache__','blob.pyc'):
 try:m.safe('/nonexistent/'+token+'/test')
 except ValueError:pass
 else:raise AssertionError('Lexical reject failed')
oldp=Path(m.R/'V239_CANONICAL_PAIRED_TRAINER_A/train_canonical_pair.py');old=m.read(oldp)
names=('safe','read','identity','verify','binding','write_json','load_code','validate_geometry','relative_xy','state_sha','validate_samples','byte_equal','global_feature_QA','context_args','reduce_rows')
a={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
c={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((H/'diagnose_own_branches.py').read_text()).body if isinstance(n,ast.FunctionDef)}
assert all(a[name]==c[name] for name in names)
checks.append('falseentry_aborts_before_payload; forbidden_lexical_reject;15_original_V239_helpers_ASTexact')
result={'completed':True,'PASS':True,'groups':4,'checks':checks,'actual_SOURCE_payload_reads':0,'actual_trained_checkpoint_loads':0,'actual_DINO_or_encoder':0,'actual_or_invented_optimizer_updates':0,'pure_random_head_and_invented_array_only':True,'inherited_ASTexact':list(names)}
with (H/'TOY_CHECKS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
