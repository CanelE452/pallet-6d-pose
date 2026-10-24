"""Independent C11 producer code/generated review; no source/model/weight IO."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,subprocess
import numpy as np
import torch
O=Path(__file__).resolve().parent;V=O.parent;H=V/'feature_preparation'
CODE=H/'prepare_spatial_query_features.py'
CODE_SHA='cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert bind(CODE)['sha256']==CODE_SHA
subprocess.run([sys.executable,str(H/'test_generated_spatial_features.py'),'--output',str(O/'PRODUCER_GENERATED_RERUN.json')],check=True)
assert bind(O/'PRODUCER_GENERATED_RERUN.json')['sha256']==bind(H/'GENERATED_FEATURE_TESTS.json')['sha256']
s=importlib.util.spec_from_file_location('independent_c11_producer',CODE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1);gen=torch.Generator().manual_seed(1047)
# All256 exact patch centres on a non-square native context box. No model involved.
field=torch.randn((16,16,384),generator=gen)
box=np.array([10.,20.,170.,340.]);yy,xx=np.meshgrid(np.arange(16),np.arange(16),indexing='ij')
q=np.stack((10+(xx.ravel()+.5)*10,20+(yy.ravel()+.5)*20),1)
patch=field.reshape(1,256,384).repeat(256,1,1)
actual,inside=m.sample_Q_raw(patch,q,np.tile(box,(256,1)),np.ones(256,bool))
assert inside.all() and torch.equal(actual,field.reshape(256,384))
del patch
# Independent scalar bilinear reconstruction of arbitrary generated token values.
u=np.random.default_rng(1047).uniform(0,1,(24,2));qs=box[:2]+u*(box[2:]-box[:2])
actual,inside=m.sample_Q_raw(field.reshape(1,256,384).repeat(24,1,1),qs,np.tile(box,(24,1)),np.ones(24,bool))
expected=[];f=field.numpy()
for point in qs:
    grid=(2*(point-box[:2])/(box[2:]-box[:2])-1).astype(np.float32)
    index=np.clip((grid+np.float32(1))*np.float32(8)-np.float32(.5),0,15)
    x,y=map(float,index);x0,y0=int(np.floor(x)),int(np.floor(y));x1,y1=min(x0+1,15),min(y0+1,15);dx,dy=x-x0,y-y0
    expected.append((f[y0,x0]*(1-dx)*(1-dy)+f[y0,x1]*dx*(1-dy)+f[y1,x0]*(1-dx)*dy+f[y1,x1]*dx*dy).astype(np.float32))
error=float(np.max(np.abs(actual.numpy()-np.stack(expected))))
assert error<2e-5 and inside.all()
# A translated/scaled box preserves u and resulting query token sample.
actual2,inside2=m.sample_Q_raw(field.reshape(1,256,384).repeat(24,1,1),qs*2+np.array([40.,-30.]),np.tile(box*2+np.tile([40.,-30.],2),(24,1)),np.ones(24,bool))
assert np.array_equal(inside,inside2) and torch.equal(actual,actual2)
# Explicit raw-pool-before-L2 oracle using unequal-norm vectors, with fractional support.
N=3;cls=torch.randn((N,384),generator=gen);fp=torch.zeros(N,16,16,384)
values=torch.tensor([[2.,1.,0.],[0.,9.,0.],[0.,0.,3.],[1.,4.,2.]])
fp[:,7:9,7:9,:3]=values.reshape(1,2,2,3);fra=np.array([[1,.5,.25],[0,.3,.2],[1,.8,.1]],np.float32)
z,flag=m.assemble(cls,field.reshape(1,256,384).repeat(N,1,1),cls+1,fp.reshape(N,256,384),cls-1,np.tile([90.,180.],(N,1)),np.tile(box,(N,1)),fra,np.array([True,True,False]))
v=values.mean(0).numpy();v=v/np.sqrt(np.sum(v*v));assert np.max(np.abs(z[0,1536:1539].numpy()-v))<1e-7
assert not z[1,:768].any() and not z[2,1536:1920].any()
assert torch.equal(z[:,1924:1927],torch.tensor(fra))
assert bind(CODE)['sha256']==CODE_SHA
result=dict(completed=True,PASS=True,blockers=[],producer=bind(CODE),protocol=bind(H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'),generated=bind(H/'GENERATED_FEATURE_TESTS.json'),generated_rerun=bind(O/'PRODUCER_GENERATED_RERUN.json'),review_code=bind(Path(__file__)),runtime=torch.__version__,independent_math=dict(all256_native_patchcentre_samples_bitexact=True,random24_scalar_bilinear_max_abs_error=error,scalar_generated_bound=2e-5,native_box_translation_scale_invariance_bitexact=True,raw_central4_mean_then_L2=True,descriptor_T_Q_F_C_P_and8scalar_order=True,support_zero_rules=True),static_review=['Exact C8 preprocess_views and C9 preprocess_target are imported under pinned code bindings, preserving mean-fill/antialiased bilinear224/full+central order.','Old8unit batches form target/full/central groups24; new1context batches form target1/full8/central8 groups17, with aligned query descriptors.','Old RGB96/valid refs retain1348+2304 unit order; new1000x8 input rows retain3652+i*8+k. Native crop is unchanged pure V90AST, all q/L preserved.','sample_Q_raw uses F64 native normalized coordinates then F32 grid, alignFalse/border and closed-box support flag before F32rounding.','P pools raw full96 x_norm_patchtokens[7:9,7:9] before descriptor normalization and literal central4 support zero.','C10 NPZ opens metadata keys only; exact41984mapping/factual/context/combined availability are copied unchanged; Qoutside only zeros Q/flag.','No labels/masks/depth/GT reference/native class enter feature extraction. Real source execution is root-entry gated.'],limits=['Generated CPU interpolation/static control-flow review only; no actual DINO/source batch, source pixels or saved feature values read.','Native q spatial sampling is the prospective feature convention, not proof of correct target/cue association.'],source_pixels_features_weights_encoder_constructor_or_forward_fit_current_PnP_reads=0)
with (O/'REVIEW.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(bind(O/'REVIEW.json')))
