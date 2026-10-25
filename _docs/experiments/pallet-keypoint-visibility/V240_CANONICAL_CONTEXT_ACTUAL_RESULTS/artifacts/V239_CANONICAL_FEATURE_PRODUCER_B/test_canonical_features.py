"""Invented pixels/random mock only; never constructs DINO or opens SOURCE."""
from pathlib import Path
import ast,hashlib,json,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
H=Path(__file__).absolute().parent;R=H.parent
ns={'__name__':'v239_invented_only','__file__':str(H/'build_canonical_source_features.py')}
exec(compile((H/'build_canonical_source_features.py').read_bytes(),str(H/'build_canonical_source_features.py'),'exec'),ns)
def pure(path,name):
 source=path.read_bytes();nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
 env={'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned pure preprocessing>','exec'),env);return env[name]
crop={'__name__':'v239_pure_crop'}
p=R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88';exec(compile(b,str(p),'exec'),crop);crop=types.SimpleNamespace(**crop)
pre=pure(R.parent/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py','preprocess_views')
prectx=pure(R.parent/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py','preprocess_target')
class MockFrozenEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('direction',torch.randn(384));self.calls=[]
 def forward_features(self,x):
  mean=x.mean((1,2,3));cls=mean[:,None]+self.direction[None,:]
  patch=cls[:,None,:]+torch.arange(256,dtype=torch.float32)[None,:,None]/1000
  out={'x_norm_clstoken':cls,'x_norm_patchtokens':patch};self.calls.append({k:v.clone() for k,v in out.items()});return out

torch.set_num_threads(1);torch.manual_seed(17);rng=np.random.default_rng(17)
rgb=rng.integers(0,256,(2,8,3,96,96),dtype=np.uint8);valid=np.ones((2,8,96,96),bool)
av=np.ones((2,8),bool);av[0,7]=False;valid[0,7]=False
crgb=rng.uniform(0,255,(2,3,224,224)).astype(np.float32);cv=np.ones((2,224,224),bool);cv[1]=False
q=np.asarray([[[10*j,20+j] for j in range(8)]]*2,np.float64);q[0,7]=np.nan;L=np.full((2,8),100.,np.float64)
g={'q8':q,'L8':L,'box_xyxy':np.array([[0,0,100,100]]*2,np.float64),'original_available8':av,'node_available':av.copy(),'node_fraction':valid.mean((2,3)).astype(np.float32),'context_fraction':cv.mean((1,2)).astype(np.float32)}
ns['validate_geometry'](g);model=MockFrozenEncoder().eval();before=ns['state_sha'](model.state_dict())
pixel,context,node4,supported,contexts=ns['extract_canonical'](model,rgb,valid,crgb,cv,g,pre,prectx,crop,'cpu',32)
assert supported==15 and contexts==1 and len(model.calls)==3 # one node batch, samebatch repeat, one context batch
assert ns['state_sha'](model.state_dict())==before and not np.count_nonzero(pixel[0,7]) and not np.count_nonzero(context[1])
full=model.calls[0]['x_norm_clstoken'][:15];patch=model.calls[0]['x_norm_patchtokens'][:15].reshape(15,16,16,384)
a=F.normalize(full,p=2,dim=1,eps=1e-12).numpy();selected=np.argwhere(av)
assert pixel[selected[:,0],selected[:,1],:384].tobytes()==a.tobytes()
r=torch.zeros_like(patch[:,7,7]);r=r+patch[:,7,7]*.25;r=r+patch[:,7,8]*.25;r=r+patch[:,8,7]*.25;r=r+patch[:,8,8]*.25
assert pixel[selected[:,0],selected[:,1],384:768].tobytes()==F.normalize(r,p=2,dim=1,eps=1e-12).numpy().tobytes()
known=[(0,0),(0,3),(1,5)];group=np.array([a for a,b in known],np.int64);slot=np.array([b for a,b in known],np.int64)
relative=ns['relative_xy'](g,group,slot);condition=np.zeros((3,32),np.float32)
for i in (0,1):
 condition[i,slot[i]]=1;condition[i,8:16]=av[group[i]];condition[i,8+slot[i]]=0;condition[i,16:]=relative[i].ravel()
d={'full_view':np.array([0,2,4]),'central_view':np.array([1,3,5]),'fractions':np.ones((3,2),np.float32),'central4':np.ones(3,bool),'condition':condition,'target':np.array([0,1,0],np.int8),'family':np.array([0,0,1],np.int8),'fit':np.array([True,True,False]),'available':np.ones(3,bool),'row_id':np.array(['a','b','c'])}
ns['validate_samples'](d)
rows=[];lookup={v:i for i,v in enumerate(known)}
for gi in range(2):
 for si in range(8):
  i=lookup.get((gi,si));rows.append({'row_id':str(d['row_id'][i]) if i is not None else f'U{gi}-{si}','group_index':gi,'active_slot':si,'fit':gi==0,'q':None if not av[gi,si] else q[gi,si].tolist(),'L':float(L[gi,si]),'original_query_available':bool(av[gi,si]),'crop_available':bool(av[gi,si]),'sample_index':i,'target':-1 if i is None else int(d['target'][i]),'family_code':0 if i is None else int(d['family'][i])})
group2,slot2,masked=ns['join_samples'](rows,d,g);assert np.array_equal(group,group2) and np.array_equal(slot,slot2) and masked==13
base,node=ns['assemble_known'](pixel,d,group,slot,g,node4)
assert base.shape==(3,1186) and base[:,:384].tobytes()==node[group,slot].tobytes()
assert base[:,1154:].tobytes()==condition.tobytes() and base[:,1152:1154].tobytes()==d['fractions'].tobytes()
assert base[2,1154:].sum()==0 and not relative[0,7].any() and not relative[np.arange(3),slot].any()
# Reject drift rather than repairing/copying an observed mismatch.
bad={k:v.copy() for k,v in d.items()};bad['fractions'][0,0]=0
try:ns['assemble_known'](pixel,bad,group,slot,g,node4)
except ValueError:pass
else:raise AssertionError('Metadata drift accepted')
# False actual entry rejects before whitelist, payload or model loader access.
read_orig=ns['read'];ns['read']=lambda p:b'{"approved_canonical_SOURCE_feature_build":false}'
try:ns['run']('/invented/entry.json')
except ValueError:pass
else:raise AssertionError('False actual entry accepted')
ns['read']=read_orig
for bad in ('/invented/v112/file','/invented/__pycache__/file'):
 try:ns['safe'](bad)
 except ValueError:pass
 else:raise AssertionError('Forbidden path reached FS')
result={'completed':True,'PASS':True,'fixture_groups':5,'same_full_centre_output_forward_and_fixedQ_formula':True,'firstbatch_repeat_exact_and_encoder_unchanged':True,'known_base_node_first384_byte_exact':True,'original_condition_fractions_qL_role_U_preserved_and_drift_rejected':True,'input_support_only_unsupported_zero_and_no_target_mask':True,'falseentry_before_payload_and_lexicalguard':True,'actual_SOURCE_DINO_weights_or_feature_reads':0,'toy_mock_forward_calls':3,'classifier_or_optimizer':0}
p=H/'INVENTED_CHECKS.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
