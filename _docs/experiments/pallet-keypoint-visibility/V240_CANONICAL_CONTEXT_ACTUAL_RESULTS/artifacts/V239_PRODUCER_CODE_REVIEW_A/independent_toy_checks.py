"""Independent invented tensors only. Extract named functions; no module run/model assets."""
import ast
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
from torch.nn import functional as F

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V239_PRODUCER_CODE_REVIEW_A'
P=R.parent
seen={}

def source(path):
    assert all(x not in str(path).casefold() for x in ('v112','__pycache__','.pyc'))
    for q in [*reversed(path.parents),path]:assert not q.is_symlink()
    raw=path.read_bytes();seen[str(path)]=hashlib.sha256(raw).hexdigest();return raw.decode()

def functions(path,names,extra=None):
    text=source(path);tree=ast.parse(text)
    ns={'np':np,'torch':torch,'F':F,'hashlib':hashlib,'json':json,'Path':Path,
        'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)}
    ns.update(extra or {})
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    return types.SimpleNamespace(**ns)

producer_path=R/'V239_CANONICAL_FEATURE_PRODUCER_B/build_canonical_source_features.py'
producer=functions(producer_path,['extract_canonical','node_pixel_batch','assemble_known','relative_xy','join_samples','lexical'],{'__file__':str(producer_path)})
producer.extract_canonical.__globals__['node_pixel_batch']=producer.node_pixel_batch
crop=functions(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py',['crop_features','fixed_center_q_read'])
pre=functions(P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py',['preprocess_views'])
ctx=functions(P/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py',['preprocess_target'])
trainer=functions(R/'V239_CANONICAL_PAIRED_TRAINER_A/train_canonical_pair.py',['byte_equal','global_feature_QA','relative_xy'])
trainer.global_feature_QA.__globals__.update({'byte_equal':trainer.byte_equal,'relative_xy':trainer.relative_xy})
checks=[]

class MockEncoder:
    """Not a DINO/model instance; invented deterministic arithmetic output schema."""
    def __init__(self):self.calls=[]
    def forward_features(self,x):
        self.calls.append((tuple(x.shape),x.detach().clone()))
        a=x.mean((1,2,3))[:,None]+torch.arange(384,dtype=torch.float32)[None,:]/384
        b=a[:,None,:]+torch.arange(256,dtype=torch.float32)[None,:,None]/256
        return {'x_norm_clstoken':a,'x_norm_patchtokens':b,'masks':None}

rng=np.random.Generator(np.random.PCG64(9));count=3
rgb=rng.integers(0,256,(count,8,3,96,96),dtype=np.uint8)
valid=np.zeros((count,8,96,96),bool);group=np.array([0,1,2],np.int64);slot=np.array([0,3,7],np.int64)
valid[group,slot]=True
context_rgb=rng.uniform(0,255,(count,3,224,224)).astype(np.float32)
context_valid=np.ones((count,224,224),bool);context_valid[2]=False
q=np.stack([np.arange(8)*7.,np.arange(8)*2.],1)[None,:,:]+np.arange(count)[:,None,None]
g={'q8':q.astype(np.float64),'L8':np.full((count,8),96.,np.float64),
   'original_available8':np.ones((count,8),bool),'node_available':valid[:,:,47:49,47:49].all((2,3))&valid.any((2,3)),
   'node_fraction':valid.mean((2,3)).astype(np.float32),'context_fraction':context_valid.mean((1,2)).astype(np.float32),
   'box_xyxy':np.tile(np.array([0.,0.,100.,100.]),(count,1))}
mock=MockEncoder()
pixel,context,node4,nodes,contexts=producer.extract_canonical(mock,rgb,valid,context_rgb,context_valid,g,pre.preprocess_views,ctx.preprocess_target,crop,'cpu',2)
assert pixel.shape==(3,8,1154) and context.shape==(3,384) and nodes==3 and contexts==2
assert len(mock.calls)==4 and all(s[1:]==(3,224,224) for s,_ in mock.calls)
assert torch.equal(mock.calls[0][1],mock.calls[1][1])
assert np.count_nonzero(pixel[~g['node_available']])==0 and np.count_nonzero(context[2])==0
checks.append('Actual pure preprocess/crop functions + invented encoder: full/centre split, first-batch repeat, support-only extraction, unsupported zero')

relative=producer.relative_xy(g,group,slot)
condition=np.zeros((3,32),np.float32)
for i in (0,2):
    condition[i,slot[i]]=1
    condition[i,8:16]=1;condition[i,8+slot[i]]=0
    condition[i,16:]=relative[i].reshape(-1)
data={'target':np.array([0,1,0],np.int8),'family':np.array([0,1,0],np.int8),
      'fit':np.array([True,True,False]),'available':np.ones(3,bool),'central4':np.ones(3,bool),
      'fractions':pixel[group,slot,1152:1154].copy(),'condition':condition,'row_id':np.array(['a','b','c'])}
base,node=producer.assemble_known(pixel,data,group,slot,g,node4)
assert base[:,:384].tobytes()==node[group,slot].tobytes()
assert base[:,1154:].tobytes()==condition.tobytes()
features={'ORIGINAL_ALL_FEATURES.npy':base,'NODE_CLS.npy':node,'CONTEXT_CLS.npy':context,
          'SAMPLE_GROUP_INDEX.npy':group,'SAMPLE_NODE_INDEX.npy':slot,'RELATIVE_XY8.npy':relative}
qa=trainer.global_feature_QA(features,data,g)
assert qa['PASS'] and qa['checked']==3 and qa['tolerance_used']==0
bad=data.copy();bad['fractions']=data['fractions'].copy();bad['fractions'][0,0]=.5
try:producer.assemble_known(pixel,bad,group,slot,g,node4)
except ValueError:pass
else:raise AssertionError('Original fraction mismatch accepted')
badfeatures=features.copy();badfeatures['ORIGINAL_ALL_FEATURES.npy']=base.copy();badfeatures['ORIGINAL_ALL_FEATURES.npy'][0,0]=np.nextafter(base[0,0],np.float32(np.inf))
try:trainer.global_feature_QA(badfeatures,data,g)
except ValueError:pass
else:raise AssertionError('One ULP active mismatch accepted')
checks.append('Producer packet -> exact trainer global_feature_QA, byte sharing/condition/fractions; fraction and one-ULP corruption fail')

rows=[]
for i,(gi,si) in enumerate(zip(group,slot)):
    rows.append({'row_id':str(data['row_id'][i]),'group_index':int(gi),'active_slot':int(si),'fit':bool(data['fit'][i]),
                 'q':g['q8'][gi,si].tolist(),'L':float(g['L8'][gi,si]),'original_query_available':True,'crop_available':True,
                 'sample_index':i,'target':int(data['target'][i]),'family_code':int(data['family'][i])})
rows.append({'row_id':'masked','group_index':0,'active_slot':2,'fit':True,'q':g['q8'][0,2].tolist(),
             'L':96.,'original_query_available':True,'crop_available':False,'sample_index':None,'target':-1,'family_code':0})
gg,ss,masked=producer.join_samples(rows,data,g)
assert np.array_equal(gg,group) and np.array_equal(ss,slot) and masked==1
badrows=[dict(r) for r in rows];badrows[-1]['fit']=False
try:producer.join_samples(badrows,data,g)
except ValueError:pass
else:raise AssertionError('Group FIT/DEV mixing accepted')
changed=data.copy();changed['target']=1-data['target']
samebase,_=producer.assemble_known(pixel,changed,group,slot,g,node4)
assert samebase.tobytes()==base.tobytes()
assert producer.relative_xy(g,group,slot).tobytes()==trainer.relative_xy(g,group,slot).tobytes()
checks.append('Original identity/group roles, masked target-1 retention, labels not in descriptors, identical F64->F32 relative geometry')

for path in ('/tmp/V112/data','/tmp/v112_data','/tmp/__pycache__/thing','/tmp/bad.pyc','/tmp/a/../b'):
    try:producer.lexical(path)
    except ValueError:pass
    else:raise AssertionError('Forbidden lexical path accepted')
checks.append('Lexical forbidden/traversal guards reject without invoking filesystem helper')

result={'completed':True,'PASS':True,'groups':len(checks),'checks':checks,'code_bindings':seen,
        'actual_SOURCE_payload_reads':0,'actual_DINO_or_weight_loads':0,'optimizer_steps':0,
        'toy_limit':'Invented CPU arithmetic encoder only; not GPU/vendor or actual SOURCE feature validation.'}
out=H/'INDEPENDENT_TOY_RESULT.json'
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'groups':len(checks),'result':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
