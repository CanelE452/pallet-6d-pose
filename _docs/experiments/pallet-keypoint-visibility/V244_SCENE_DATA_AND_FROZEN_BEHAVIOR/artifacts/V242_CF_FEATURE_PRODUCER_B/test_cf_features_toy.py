"""Mock encoder and invented arrays only; no SOURCE/weights/official DINO."""
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy,json,numpy as np,torch
from torch import nn
H=Path(__file__).parent
m=runpy.run_path(str(H/'build_cf_features.py'),run_name='toy_cf_features')
with TemporaryDirectory() as td:
 p=Path(td)/'entry.json';p.write_text('{"execution_authorized":false}')
 try:m['run'](str(p))
 except PermissionError:pass
 else:raise AssertionError('False entry accepted')
for bad in ('/toy/v112/array.npy','/toy/__pycache__/x','/toy/x.pyc'):
 try:m['lexical'](bad)
 except ValueError:pass
 else:raise AssertionError('Forbidden lexical input')
can=m['load_canonical'](m['CANONICAL'],[m['CANONICAL']])
code=[{'path':p,'sha256':s} for p,s in can.PINS.values()]
crop=can.load_code(code[1],code,'crop_modules',can.PINS['crop_module'])
pre=can.load_preprocessor(code[2],code,'preprocess_views',can.PINS['preprocess_CODE'])
prec=can.load_preprocessor(code[3],code,'preprocess_target',can.PINS['context_preprocess_CODE'])
class Mock(nn.Module):
 def __init__(self):super().__init__();self.register_buffer('direction',torch.randn(384));self.calls=0
 def forward_features(self,x):
  self.calls+=1;c=x.mean((1,2,3))[:,None]+self.direction[None,:]
  return {'x_norm_clstoken':c,'x_norm_patchtokens':c[:,None]+torch.arange(256)[None,:,None]/1000}
class RepeatedRows:
 def __init__(self,a):self.a=a
 def __getitem__(self,key):
  k=key if isinstance(key,tuple) else (key,)
  return self.a[(np.asarray(k[0])%2,)+k[1:]]
torch.set_num_threads(1);torch.manual_seed(17);rng=np.random.default_rng(242)
rgb=rng.integers(0,256,(2,2,8,3,96,96),dtype=np.uint8);valid=np.ones((2,8,96,96),bool)
crgb=rng.uniform(0,255,(2,2,3,224,224)).astype(np.float32);cv=np.ones((2,224,224),bool);cv[1]=False
av=np.ones((2,8),bool);av[0,7]=False;q=np.asarray([[[10*j,j+20] for j in range(8)]]*2,np.float64);q[0,7]=np.nan
g={'q8':q,'L8':np.full((2,8),100.,np.float64),'original_available8':av,'node_available':av.copy(),'node_fraction':valid.mean((2,3)).astype(np.float32),'context_fraction':cv.mean((1,2)).astype(np.float32),'box_xyxy':np.array([[0,0,100,100]]*2,np.float64)}
arrays={'node_rgb':RepeatedRows(rgb),'context_rgb':RepeatedRows(crgb),'node_valid':valid,'context_valid':cv}
cache=np.arange(5406,dtype=np.int64)%2;slot=np.zeros(5406,np.int64);slot[3]=5;constructed=np.zeros(5406,bool);constructed[[0,3]]=True
data={'condition':np.zeros((5406,32),np.float32),'fractions':np.ones((5406,2),np.float32),'central4':np.ones(5406,bool),'available':np.ones(5406,bool)}
model=Mock().eval();state=can.state_sha(model.state_dict())
b,node,ctx,nodes,contexts=m['extract_role'](can,model,arrays,g,cache,slot,constructed,data,0,pre,prec,crop,'cpu')
assert nodes==15 and contexts==1 and model.calls==3
assert b[[0,3],:384].tobytes()==node[[0,3],slot[[0,3]]].tobytes() and not node[0,7].any() and not ctx[3].any()
assert not b[~constructed].any() and not node[~constructed].any() and not ctx[~constructed].any()
assert b[[0,3],1154:].tobytes()==data['condition'][[0,3]].tobytes() and can.state_sha(model.state_dict())==state
# Role extraction does not take a proof/target argument: the same constructed
# array is processed even if its teacher sidecar later has proof=false/target-1.
assert 'proof' not in __import__('inspect').signature(m['extract_role']).parameters
relative=can.relative_xy(m['parent_geometry'](g,cache),np.arange(5406,dtype=np.int64),slot)
assert not relative[np.arange(5406),slot].any() and not relative[0,7].any()
result={'completed':True,'PASS':True,'groups':3,'false_entry_before_bindings':True,'indexed_shared_valid_role_arrays':True,'canonical_V239_math_reused_unchanged':True,'active_node_byte_sharing':True,'condition_fractions_support_preserved':True,'constructed_UNKNOWN_not_filtered_by_proof':True,'unconstructed_zero':True,'actual_payloads_or_weights':0,'official_encoder_or_feature_run':0,'mock_forward_calls':model.calls}
with (H/'TOY_CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
