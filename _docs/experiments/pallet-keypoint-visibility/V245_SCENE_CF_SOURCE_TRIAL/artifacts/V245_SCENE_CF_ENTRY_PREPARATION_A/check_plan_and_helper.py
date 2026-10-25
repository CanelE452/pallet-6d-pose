"""Metadata/list-only plan and assembler checks; no numeric payload libraries."""
import ast,hashlib,json,struct,tempfile,types
from pathlib import Path
H=Path(__file__).absolute().parent
p=H/'assemble_training_entry.py';m=types.ModuleType('v245_helper_toy');m.__file__=str(p);exec(compile(p.read_text(),str(p),'exec'),m.__dict__)
plan=m.load(m.PLAN);raw=m.checked(m.PLAN_LEDGER);qa=m.verify_sealed_plan(plan,raw);rows=[json.loads(line) for line in raw.splitlines()]
# Execute ONLY the frozen pure V243 plan_digest function against Python-list wrappers.
source=m.R/'V243_SCENE_CF_TRAINER_A/train_scene_cf_data_v2.py';code=m.read(source)
assert hashlib.sha256(code).hexdigest()=='e18e6e601742dd2ad52cfbd417b5209852375d8a3d163687c7c2f0eb7b2d110d'
node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='plan_digest')
space={'hashlib':hashlib};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),space)
class ListBytes:
 def __init__(self,values,dtype='<i8'):self.values=values;self.dtype=dtype
 def astype(self,dtype):return ListBytes(self.values,dtype)
 def tobytes(self):
  def flat(x):
   for v in x:
    if isinstance(v,list):yield from flat(v)
    else:yield v
  items=list(flat(self.values));return struct.pack('<'+str(len(items))+('b' if self.dtype=='i1' else 'q'),*items)
plans=[(ListBytes(v['source_parent_indices']),ListBytes(v['pair_indices']),ListBytes(v['source_roles'],'i1')) for v in rows]
from_actual_function=space['plan_digest'](ListBytes([v['sample_indices'] for v in rows]),plans)
assert from_actual_function==m.PLAN_SHA==qa['source_plan_sha256']
# Exact actual ledger serialization, including dict insertion order, default spaces, newline.
replay=b''.join((json.dumps({'update':v['update'],'sample_indices':v['sample_indices'],'source_parent_indices':v['source_parent_indices'],'pair_indices':v['pair_indices'],'source_roles':v['source_roles'],'draws_sha256':m.DRAW_SHA,'source_plan_sha256':m.PLAN_SHA,'all_fit_available_known':True})+'\n').encode() for v in rows)
assert replay==raw
# Every failure and first4 position is explicitly checked in verify_sealed_plan.
mutated=[dict(v) for v in rows];mutated[0]=dict(mutated[0]);mutated[0]['source_roles']=list(mutated[0]['source_roles']);mutated[0]['source_roles'][0]=1
try:m.verify_sealed_plan(plan,b'\n'.join(json.dumps(v).encode() for v in mutated))
except ValueError:pass
else:raise AssertionError('Invalid first4 substitution accepted')
checks=['stdlib_struct_digest_matches_frozen_V243_plan_digest_on_full1000; no_NumPy_or_RNG','whole_expected_actual_ledger_serialization_byteexact; first4_and_failureguard_negativefixture']
# Late feature bindings are only lexical metadata identities, never file reads/stats.
b={'path':'/not_existing_frozen_SOURCE/feature.npy','sha256':'a'*64,'size_bytes':123,'shape':[5406,1186],'dtype':'float32'}
assert m.normalize(b)=={k:b[k] for k in ('path','sha256','size_bytes')}
assert len(m.unique_bindings([b,dict(b)]))==1
for s in ('/tmp/v112/a','/tmp/V112/a','/tmp/__pycache__/a','/tmp/a.pyc','/tmp/../a'):
 try:m.lexical(s)
 except ValueError:pass
 else:raise AssertionError('Forbidden path')
with tempfile.TemporaryDirectory(prefix='v245_fake_config_') as tmp:
 q=Path(tmp)/'config.json';q.write_text('{"approved_to_write_actual_training_entry":false}')
 try:m.prepare(q)
 except PermissionError:pass
 else:raise AssertionError('False preparation approval passed')
checks.append('late_feature_binding_annotations_normalized_without_payload_FS; falseconfig_rejects; lexical_path_guards')
# Assembler imports only stdlib, and has no training/process launcher or numerical loader.
imports={n.names[0].name for n in ast.parse(p.read_text()).body if isinstance(n,ast.Import)}
assert not imports&{'numpy','torch','subprocess'}
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('system','load_state_dict','backward','step') for n in ast.walk(ast.parse(p.read_text())))
result={'completed':True,'PASS':True,'groups':3,'checks':checks,'Root_expected_source_plan_sha256':m.PLAN_SHA,'expected_whole_ledger_SHA':hashlib.sha256(raw).hexdigest(),'actual_SOURCE_vectors_samples_weights_or_DEV_reads':0,'new_RNG_calls':0,'new_feature_or_training_execution':0,'trainer_modified':False}
with (H/'PLAN_AND_HELPER_CHECKS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
