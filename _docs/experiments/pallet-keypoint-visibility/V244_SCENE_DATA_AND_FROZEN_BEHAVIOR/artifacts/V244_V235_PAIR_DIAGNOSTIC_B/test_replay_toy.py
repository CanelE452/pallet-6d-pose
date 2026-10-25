"""Invented CPU/mock and count checks only; no actual SOURCE/weights/model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy,json,numpy as np,torch
H=Path(__file__).parent;R=H.parent
m=runpy.run_path(str(H/'replay_v235_pairs.py'),run_name='v244_invented')
with TemporaryDirectory() as td:
 p=Path(td)/'false.json';p.write_text('{"approved_frozen_V235_pair_replay":false}')
 try:m['run'](str(p))
 except PermissionError:pass
 else:raise AssertionError('False entry accepted')
for bad in ('/tmp/v112/a','/tmp/__pycache__/a','/tmp/a.pyc'):
 try:m['lexical'](bad)
 except ValueError:pass
 else:raise AssertionError('Forbidden path accepted')
old=json.loads((R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/ACTUAL_DATA_ONLY_HEAD_FIT_ENTRY.json').read_text())
utility=m['load_utility'](m['OLD_TRAINER'],[m['OLD_TRAINER']])
crop=utility.load_code(old['crop_module'],old['code_whitelist'],'crop_modules',utility.PINS['crop_module']);pre=utility.load_preprocess(old)
class Mock:
 def __init__(self):self.calls=0;self.views=None
 def forward_features(self,x):
  self.calls+=1;self.views=len(x);v=torch.arange(384,dtype=torch.float32)[None,:]+x.mean((1,2,3))[:,None]
  return {'x_norm_clstoken':v,'x_norm_patchtokens':v[:,None,:]+torch.arange(256,dtype=torch.float32)[None,:,None]/1000}
torch.set_num_threads(1);rng=np.random.default_rng(244)
pixels=torch.from_numpy(rng.integers(0,256,(4,3,96,96),dtype=np.uint8));support=torch.ones((4,1,96,96),dtype=torch.bool)
fractions=np.ones((4,2),np.float32);central=np.ones(4,bool);condition=np.zeros((4,32),np.float32);condition[:2,0]=1
model=Mock();x=m['cf_batch_features'](model,crop,pre,pixels,support,fractions,central,condition,'cpu')
assert model.calls==1 and model.views==8 and x.shape==(4,1186) and x.dtype==torch.float32
assert x[:,1154:].numpy().tobytes()==condition.tobytes() and x[:,1152:1154].numpy().tobytes()==fractions.tobytes()
drift=fractions.copy();drift[0,0]=0
try:m['cf_batch_features'](model,crop,pre,pixels,support,drift,central,condition,'cpu')
except ValueError:pass
else:raise AssertionError('Support drift accepted')
rows=[{'success':True,'family':0,'COVER_prediction':'E','SHAM_prediction':'D','COVER_minus_SHAM':2.,'original_D_prediction':'D'},
 {'success':True,'family':1,'COVER_prediction':'D','SHAM_prediction':'E','COVER_minus_SHAM':-2.,'original_D_prediction':'D'},
 {'success':True,'family':1,'COVER_prediction':'E','SHAM_prediction':'E','COVER_minus_SHAM':0.,'original_D_prediction':'E'},
 {'success':False,'family':1,'COVER_prediction':'U','SHAM_prediction':'U','COVER_minus_SHAM':None,'original_D_prediction':None}]
s=m['summarize'](rows)['ALL'];assert s['successful_pairs']==3 and s['COVER_E']==2 and s['SHAM_D']==1 and s['COVER_E_and_SHAM_D']==1
assert (s['COVER_gt_SHAM'],s['COVER_eq_SHAM'],s['COVER_lt_SHAM'])==(1,1,1) and s['SHAM_preservation_denominator']==2
result={'completed':True,'PASS':True,'fixture_groups':3,'false_entry_before_input_bindings':True,'old_preprocess_fullcentre_single_forward':True,'original_condition_fractions_and_support_guard':True,'two_family_binary_counts_and_direction_tie':True,'actual_SOURCE_weight_DINO_or_checkpoint':0,'mock_encoder_calls':1,'optimizer_or_training':0}
with (H/'TOY_CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
