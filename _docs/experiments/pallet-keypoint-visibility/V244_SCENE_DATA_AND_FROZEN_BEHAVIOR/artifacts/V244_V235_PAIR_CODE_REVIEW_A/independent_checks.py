"""Independent CODE/metadata and invented-tensor V235 replay audit. No payload."""
from pathlib import Path
import ast,hashlib,json,stat,sys,tempfile,types
import numpy as np
import torch
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=R/'V244_V235_PAIR_CODE_REVIEW_A';B=R/'V244_V235_PAIR_DIAGNOSTIC_B'
sys.dont_write_bytecode=True
def read(p,pin=None):
 assert not any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc'))
 for a in reversed((p,*p.parents)):assert not stat.S_ISLNK(a.lstat().st_mode)
 raw=p.read_bytes()
 if pin:assert hashlib.sha256(raw).hexdigest()==pin
 return raw
raw=read(B/'replay_v235_pairs.py','c913a1594deff7cdb010a115f2b1571c2a80086f80542c5822c8a16f922b2e3b');m=types.ModuleType('v244_B_independent_A');m.__file__=str(B/'replay_v235_pairs.py');exec(compile(raw,m.__file__,'exec'),m.__dict__)
false=json.loads(read(B/'ENTRY_FALSE_TEMPLATE.json'))
actualpath=R/'V244_ROOT_FROZEN_BEHAVIOR/ACTUAL_OLD_V235_PAIR_REPLAY_ENTRY.json';actual=json.loads(read(actualpath,'f933fcb32293f83c79ac0f9d5287612062e5efc133aecf9c0a0c58a442f6f808'))
ident=lambda b:(b['path'],b['sha256'])
assert not {ident(x) for x in false['model_loader_config']['code_whitelist']}<={ident(x) for x in false['code_whitelist']}
assert {ident(x) for x in actual['model_loader_config']['code_whitelist']}<={ident(x) for x in actual['code_whitelist']}
assert actual['model_loader_config']==false['model_loader_config']
checks=['false_template_codeclosure_gap_reproduced; Rootactual18_union_resolves_cfg13_subset; oldconfig_unchanged']
# Use only literal CODE loaders, never official loader/model/weights or run(actual).
u=m.load_utility(m.OLD_TRAINER,[m.OLD_TRAINER]);crop=u.load_code(false['crop_module'],false['code_whitelist'],'crop_modules',u.PINS['crop_module']);pre=u.load_preprocess(false)
class MockEncoder:
 def __init__(self):self.calls=0;self.shapes=[]
 def forward_features(self,views):
  self.calls+=1;self.shapes.append(tuple(views.shape))
  v=views.mean((1,2,3))[:,None]+torch.arange(384,dtype=torch.float32)[None,:]/100
  patch=v[:,None,:]+torch.arange(256,dtype=torch.float32)[None,:,None]/1000
  return {'x_norm_clstoken':v,'x_norm_patchtokens':patch}
torch.set_num_threads(1);torch.manual_seed(244)
pixels=torch.randint(0,256,(32,3,96,96),dtype=torch.uint8);support=torch.ones((32,1,96,96),dtype=torch.bool);support[::3,:,0:12,:]=False
fractions=np.stack([support.numpy().mean((1,2,3)),support.numpy()[:,:,32:64,32:64].mean((1,2,3))],1).astype(np.float32)
central=support.numpy()[:,0,47:49,47:49].all((1,2));condition=np.zeros((32,32),np.float32);condition[:16,0]=1
mock=MockEncoder();got=m.cf_batch_features(mock,crop,pre,pixels,support,fractions,central,condition,'cpu')
# Independent transcription of the exact original V235 lines240–245.
views,ff=pre(pixels,support);reference_encoder=MockEncoder();out=reference_encoder.forward_features(views);n=len(pixels)
full={k:v[:n] for k,v in out.items() if isinstance(v,torch.Tensor)};centre={k:v[n:] for k,v in out.items() if isinstance(v,torch.Tensor)}
want=crop.crop_features(full,centre,torch.from_numpy(fractions),torch.from_numpy(central),torch.from_numpy(condition),None)
assert torch.equal(got,want) and mock.calls==1 and mock.shapes==[(64,3,224,224)] and got.shape==(32,1186) and got.dtype==torch.float32
assert got[:,1154:].numpy().tobytes()==condition.tobytes() and got[:,1152:1154].numpy().tobytes()==fractions.tobytes()
checks.append('invented32_U8_CHW96_bool_valid_to64views_single_mockforward; old_feature_formula_byteexact; centre32_fraction_condition_unchanged')
attempts=np.array([0,2,4],np.int64);aa=np.repeat(attempts,2);vv=np.tile(np.array([0,1]),len(attempts));flat=np.stack([aa,vv],1);compact=flat.reshape(3,2,2)
assert compact.tolist()==[[[0,0],[0,1]],[[2,0],[2,1]],[[4,0],[4,1]]]
rows=[{'success':True,'family':0,'COVER_prediction':'E','SHAM_prediction':'D','COVER_minus_SHAM':2.,'original_D_prediction':'D'}, {'success':True,'family':1,'COVER_prediction':'D','SHAM_prediction':'E','COVER_minus_SHAM':-2.,'original_D_prediction':'D'}, {'success':True,'family':1,'COVER_prediction':'E','SHAM_prediction':'E','COVER_minus_SHAM':0.,'original_D_prediction':'E'}, {'success':False,'family':1,'COVER_prediction':'U','SHAM_prediction':'U','COVER_minus_SHAM':None,'original_D_prediction':None}]
s=m.summarize(rows);assert s['ALL']['successful_pairs']==3 and s['ALL']['COVER_E_and_SHAM_D']==1
assert (s['ALL']['COVER_gt_SHAM'],s['ALL']['COVER_eq_SHAM'],s['ALL']['COVER_lt_SHAM'])==(1,1,1)
assert s['ALL']['SHAM_preservation_denominator']==2 and s['SYN_MAIN']['successful_pairs']==1 and s['YCB_ASSOCIATED_R_AUX']['successful_pairs']==2
checks.append('success_compact_repeat_COVER_SHAM_order_and_reshape; failedUNKNOWN_omitted_from_scored_denominator; independent_family_counts_ties_crossings')
with tempfile.TemporaryDirectory(prefix='v244_A_invented_') as temp:
 p=Path(temp)/'false.json';p.write_text('{"approved_frozen_V235_pair_replay":false}')
 try:m.run(p)
 except PermissionError:pass
 else:raise AssertionError('False approval accepted')
for p in ('/tmp/v112/x','/tmp/V112/x','/tmp/__pycache__/x','/tmp/item.pyc','/tmp/../x'):
 try:m.lexical(p)
 except ValueError:pass
 else:raise AssertionError('Lexical rejection failed')
# No optimizer/backward operations anywhere in the sealed producer.
for node in ast.walk(ast.parse(raw)):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('backward','step')
checks.append('falseentry_before_payload; caseinsensitive_v112_cache_traversal_lexical_rejection; no_optimizer_backward_or_step_calls')
result={'completed':True,'PASS':True,'groups':len(checks),'checks':checks,'actual_SOURCE_payload_reads':0,'actual_official_or_trained_weights_reads':0,'actual_DINO_instantiations_or_forwards':0,'actual_training_or_optimizer_updates':0,'mock_encoder_calls':1,'Root_actual_entry_metadata_only':True,'actual_entry_executed':False}
with (H/'INDEPENDENT_CHECKS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
