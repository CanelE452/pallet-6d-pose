"""V2 CODE and invented memory arrays only; never any real SOURCE payload."""
from pathlib import Path
import ast,hashlib,json,tempfile
import numpy as np

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=R/'V242_SCENE_DRIVER_CODE_REVIEW_A';B=R/'V242_SCENE_CF_DRIVER_B'
def code(p,sha):
    assert all(t not in str(p).casefold() for t in ('v112','__pycache__','.pyc'))
    for a in [*reversed(p.parents),p]:assert not a.is_symlink()
    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;return raw
v1=code(B/'build_scene_pairs.py','db125ecb8158146e872393925af66c6d74b32718641ab6434f003923de78e3c9')
v2=code(B/'build_scene_pairs_v2.py','83a84fb22e177b0ea012b04f5c622e0ac71f9c5bdd612b9b63e4f1bae76538ca')
bad=b'48c865bc539f70d6c387f31f6fe8257ef43db2a50394161ef6caee166589a25c2';good=bad[:-1]
assert v1.replace(bad,good)==v2
def signatures(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert signatures(v1)==signatures(v2)
ns={'__name__':'invented_driver_only','__file__':str(B/'build_scene_pairs_v2.py')};exec(compile(v2,str(B/'build_scene_pairs_v2.py'),'exec'),ns)
assert len(ns['SCOPE']['sha256'])==64
try:ns['identity']({'path':'/tmp/V242_INVENTED_SCOPE.json','sha256':bad.decode()})
except ValueError:pass
else:raise AssertionError('65char SHA accepted')
checks=['V1 extra-character defect reproduced; V2 exact one-literal repair and all19 function/class AST unchanged']

arrays={k:np.zeros(shape,dtype) for k,(dtype,shape) in {
 'PAIR_TARGETS.npy':('int8',(3,2)), 'CONSTRUCTED.npy':('bool',(3,)), 'PROOF_PASS.npy':('bool',(3,)),
 'CF_NODE_RGB96.npy':('uint8',(3,2,8,3,96,96)), 'CF_CONTEXT_RGB224_F32.npy':('float32',(3,2,3,224,224)),
 'PAIR_LAYER_RGB96.npy':('uint8',(3,2,3,96,96)), 'PAIR_ALPHA_PACKBITS.npy':('uint8',(3,2,1152))}.items()}
pack={'NODE_RGB96':np.full((8,3,96,96),17,np.uint8),'CONTEXT_RGB224_F32':np.full((3,224,224),19,np.float32)}
layers=np.full((2,3,96,96),23,np.uint8);alphas=np.zeros((2,96,96),bool);alphas[:,10:14,20:25]=True
unknown={'constructed':True,'proof':{'PASS':False},'inputs':{r:pack for r in ns['ROLES']}}
ns['store_children'](arrays,0,unknown,layers,alphas)
assert arrays['CONSTRUCTED.npy'][0] and not arrays['PROOF_PASS.npy'][0]
assert (arrays['PAIR_TARGETS.npy'][0]==-1).all() and (arrays['CF_NODE_RGB96.npy'][0]==17).all() and (arrays['CF_CONTEXT_RGB224_F32.npy'][0]==19).all()
assert np.array_equal(np.unpackbits(arrays['PAIR_ALPHA_PACKBITS.npy'][0],axis=1,bitorder='big').reshape(2,96,96),alphas)
ns['store_children'](arrays,1,{'constructed':False,'proof':None})
assert not arrays['CF_NODE_RGB96.npy'][1].any() and not arrays['CF_CONTEXT_RGB224_F32.npy'][1].any() and (arrays['PAIR_TARGETS.npy'][1]==-1).all()
ns['store_children'](arrays,2,{**unknown,'proof':{'PASS':True}},layers,alphas)
assert arrays['PAIR_TARGETS.npy'][2].tolist()==[1,0]
checks.append('Production store_children: constructed UNKNOWN inputs+carrier retained/target-1; unconstructed zero; verified C/S labels1/0 and big-bitorder roundtrip')

with tempfile.TemporaryDirectory(prefix='toy_refusal_',dir=H) as tmp:
    p=Path(tmp)/'FALSE_ENTRY.json';p.write_text(json.dumps({'execution_authorized':False}))
    ns['file_binding']=lambda *a:(_ for _ in ()).throw(AssertionError('No code/scope/payload binding may run before false-entry refusal'))
    try:ns['run'](p)
    except PermissionError:pass
    else:raise AssertionError('False entry accepted')
for p in ('/never/V112/a','/never/__pycache__/a','/never/file.pyc'):
    try:ns['safe'](p)
    except ValueError:pass
    else:raise AssertionError('Forbidden path reached filesystem helper')
checks.append('False actual-entry refusal before any code/scope/data binding; lexical forbidden names rejected')
result={'completed':True,'PASS':True,'groups':len(checks),'checks':checks,'actual_SOURCE_pixel_array_GT_weight_or_model_reads':0,'actual_driver_runs':0,'invented_only':True}
p=H/'INDEPENDENT_DRIVER_CHECKS.json'
with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'groups':len(checks),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
