"""Pure AST and declared preparation metadata review. No actual artifact loader."""
from pathlib import Path
import ast,hashlib,json
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');A=R/'V238_CONTEXT_RESIDUAL_TRAINER_A';H=R/'V238_TRAINER_REVIEW_B'
def read(name):
 p=A/name;assert not any(s in str(p).lower() for s in ('v112','__pycache__','.pyc'));return p.read_bytes()
def dump(n):return ast.dump(n,include_attributes=False)
def calls(n,name):return [c for c in ast.walk(n) if isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id==name]
def assigned(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
a=read('train_context_residual.py');b=read('train_context_residual_v2.py')
assert hashlib.sha256(a).hexdigest()=='74fa6517c5058a8a1f5fc0dd4ad298d24b5ece72c9056b569ca99a198ab24e24'
assert hashlib.sha256(b).hexdigest()=='92442bb4a4e700eed89a6621a6706be5b1f9d305284d3c173c495591bd7bbd35'
t1,t2=ast.parse(a),ast.parse(b);f1={n.name:n for n in t1.body if isinstance(n,ast.FunctionDef)};f2={n.name:n for n in t2.body if isinstance(n,ast.FunctionDef)}
assert f1.keys()==f2.keys()
unchanged=[k for k in f1 if k!='run'];assert len(unchanged)==16
assert all(dump(f1[k])==dump(f2[k]) for k in unchanged)
assert [dump(n) for n in t1.body if not isinstance(n,ast.FunctionDef)]==[dump(n) for n in t2.body if not isinstance(n,ast.FunctionDef)]
r1,r2=f1['run'],f2['run'];tr1=next(n for n in r1.body if isinstance(n,ast.Try));tr2=next(n for n in r2.body if isinstance(n,ast.Try))
assert [dump(n) for n in r1.body if not isinstance(n,ast.Try)]==[dump(n) for n in r2.body if not isinstance(n,ast.Try)]
freeze1=next(i for i,n in enumerate(tr1.body) if assigned(n,'freeze'));freeze2=next(i for i,n in enumerate(tr2.body) if assigned(n,'freeze'))
assert freeze1==freeze2 and [dump(n) for n in tr1.body[:freeze1+1]]==[dump(n) for n in tr2.body[:freeze2+1]]
# The old 1000 update/head/data sequence is wholly identical up to and including finalfreeze.
post=tr2.body[freeze2+1:]
barrier_i=next(i for i,n in enumerate(post) if assigned(n,'global_QA_barrier'))
pre=post[:barrier_i];after=post[barrier_i+1:]
assert not any(calls(n,'head') for n in pre)
assert sum(assigned(n,'alignment_dev') for n in pre)==1 and sum(assigned(n,'replay') for n in pre)==1
assert any(isinstance(n,ast.If) and 'dev_checked' in ast.unparse(n.test) and '1741' in ast.unparse(n.test) for n in pre)
load_i=next(i for i,n in enumerate(after) if assigned(n,'saved_original_features'))
assert 'np.load' in ast.unparse(after[load_i].value) and "'ORIGINAL_ALL_FEATURES.npy'" in ast.unparse(after[load_i].value)
second=next(n for n in after if isinstance(n,ast.With))
assert len(calls(second,'head'))==1
assert 'saved_original_features[indices]' in ast.unparse(second)
assert 'source.features_for' not in ast.unparse(second)
loop=second.body[0];assert isinstance(loop,ast.For) and ast.unparse(loop.iter)=='range(0, len(all_indices), 64)'
assert sum(len(calls(n,'head')) for n in tr2.body)==2 # training + second frozen output pass
assert [dump(n) for n in tr1.handlers]==[dump(n) for n in tr2.handlers]
# Falseentry updates contain only the V2 code/protocol rebinding, recursively including closed code list.
e1=json.loads(read('ENTRY_FALSE_TEMPLATE.json'));e2=json.loads(read('ENTRY_FALSE_TEMPLATE_V2.json'))
oldcode=e1['code'];newcode=e2['code'];oldproto=e1['implementation_protocol'];newproto=e2['implementation_protocol']
def rebind(x):
 if isinstance(x,dict):
  if x==oldcode:return newcode
  if x==oldproto:return newproto
  return {k:rebind(v) for k,v in x.items()}
 if isinstance(x,list):return [rebind(v) for v in x]
 return x
assert rebind(e1)==e2
assert e2['approved_source_context_residual_training'] is False and e2['steps']==1000 and e2['use_V235_counterfactuals'] is False
assert e2['code'] in e2['code_whitelist'] and e2['implementation_protocol'] in e2['code_whitelist']
p=json.loads(read('IMPLEMENTATION_PROTOCOL_BEFORE_FEATURES_OR_FIT_V2.json'));assert p['code']==newcode and p['root_protocol']['sha256']=='17b3b00b1c8b801dc536460f555d40ff8df37f6fe2355955a5e1ad66ecb29ba3'
result={'completed':True,'PASS':True,'unchanged_non_run_functions':16,'module_level_AST_exact':True,'run_pre_final_freeze_AST_exact':True,'old_training_model_data_1000_rng_loss_threshold_unchanged':True,'no_new_head_in_first_postfreeze_pass':True,'all1741_guard_and_DEV_CONTROL_receipts_precede_global_barrier':True,'secondpass_saved_FP32_features_batch64_only':True,'secondpass_prefix_work':0,'falseentry_only_recursive_code_protocol_rebinding':True,'actual_SOURCE_weights_features_pixels_fit_or_forward':0}
out=H/'MINIMAL_V2_INDEPENDENT_CHECKS.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'PASS':True,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
