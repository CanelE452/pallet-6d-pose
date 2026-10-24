"""Independent AST-only generated cohort/admission tests; no model/source I/O."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent

def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def pure(path,name):
 fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 scope=dict(np=np,SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL'));scope['same']=lambda a,b:np.array_equal(a,b,equal_nan=True)if a.dtype.kind in 'fc'else np.array_equal(a,b);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope);return scope[name]
trainer=V/'train_source_once.py';metadata=pure(trainer,'metadata_from_compact');admit=pure(P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/train_mixed_once.py','admission');pair=pure(P/'query_visibility_controlled_pairs_20261023_v77/train_paired_once.py','pair_eligibility')
n=5203;a=4799
original=dict(id=np.array([f'GENERATED:{i}'for i in range(n)]),source=np.array(['SYNTHETIC']*1003+['YCB_PUBLIC_REAL']*4200),split=np.array(['train'if i%2==0 else'val'for i in range(n)]),label8=np.zeros((n,8),np.int8),native_GTref8=np.zeros((n,8,2),float),convex_self_proxy8_separate=np.ones((n,8),np.int8),near_boundary8=np.zeros((n,8),bool),rule8=np.full((n,8),'STRONG_3X3_COVERED'))
original['label8'][:a,:2]=1
orig=np.arange(a);after={k:v[:a].copy()for k,v in original.items()};after['id']=np.array([x+'::V77_OPAQUE_AFTER'for x in original['id'][:a]]);after['label8'][:,0]=0
after.update(original_row_index=orig,after_row_index=np.arange(a),anchor_k=np.zeros(a,np.int8),pair_anchor_success=np.ones(a,bool),anchor_training_target=np.zeros(a,np.int8),original_primary8=np.ones((a,8),bool),query_valid8=np.ones((a,8),bool),reference_in_local_square8=np.ones((a,8),bool))
d=dict(frame_index=np.arange(n+a),domain=np.r_[np.zeros(n,np.int8),np.ones(a,np.int8)],original_row_index=np.r_[np.arange(n),orig],query8_xy=np.zeros((n+a,8,2)),L=np.ones(n+a),context42=np.zeros((n+a,8,42),np.float32),query_center_valid=np.ones((n+a,8),bool),ROI_anyvalid=np.ones((n+a,8),bool),localany=np.ones((n+a,8),bool),targetany=np.ones(n+a,bool))
for k in ['id','source','split']:d[k]=np.concatenate([original[k],after[k]])
paired=SimpleNamespace(pair_eligibility=pair);base=SimpleNamespace(admission=admit)
m=metadata(d,original,after,paired,base);assert m['N']==80016 and m['pair_good'].all()and m['primary'].all()and len(m['pair_good'])==4799
# A failed control with an actual COVERED label remains excluded from pairs.
f={k:v.copy()for k,v in after.items()};f['pair_anchor_success'][0]=False;f['anchor_training_target'][0]=-1;mm=metadata(d,original,f,paired,base);assert not mm['pair_good'][0]and mm['pair_good'][1:].all()
# Unknown after labels lose admission while original cohorts remain unchanged.
u={k:v.copy()for k,v in after.items()};u['label8'][1,1]=-1;mu=metadata(d,original,u,paired,base);assert not mu['primary'][(5203+1)*8+1]and np.array_equal(mu['primary'][:41624],m['primary'][:41624])
rejections=[]
for field in ['query8_xy','context42','L']:
 bad={k:v.copy()for k,v in d.items()};bad[field][5203]=bad[field][5203]+1
 try:metadata(bad,original,after,paired,base)
 except AssertionError:rejections.append(field)
 else:raise AssertionError('Changed pair input accepted:'+field)
bad={k:v.copy()for k,v in after.items()};bad['native_GTref8'][0,0,0]=1
try:metadata(d,original,bad,paired,base)
except AssertionError:rejections.append('native_GTref8')
else:raise AssertionError('Changed fixed reference accepted')
out=R/'GENERATED_INDEPENDENT_TRAINER_METADATA_TESTS.json';assert not out.exists();out.write_text(json.dumps(dict(PASS=True,test=bind(Path(__file__)),trainer=bind(trainer),full_generated_frames=10002,full_generated_query_slots=80016,primary_and_anchor_reconstruction=True,failed_control_actualC_excluded=True,unknown_after_loss_admission_zero=True,changed_inputs_or_GT_rejected=rejections,source_GT_features_RGB_P3_weights_current_read=0,model_forward_fit_PnP=0),indent=2)+'\n');print(json.dumps(dict(PASS=True,report=bind(out))))
