"""Independent invented checks of frozen A pure APIs; no metadata loader call."""
import sys,importlib.util,json,hashlib
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent;V=H.parent
p=V/'context_adapter_a/original_q8_context_adapter.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='b4c3134970004fcfb9351f35b16bc5a81c75ebd39cad9ee98784fca7059de5b6'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=module('context_A',p);C=module('scalar_C',H/'scalar_contract_oracle.py')
def prohibited(*a,**k):raise AssertionError('metadata access forbidden in invented pure check')
A.read_bound=prohibited
checks=[]
def ok(name,condition):
 assert condition,name;checks.append({'name':name,'PASS':True})
def run(q,flags,active,L,k=3):return A.build_context_vectors(active,L,k,flags[k],q,[L]*8,flags)
q=[[10.*i,-5.*i]for i in range(8)];flags=[True]*8
z=run(q,flags,[30.,-15.],10.)
ref=C.scalar_context(q,flags,[30.,-15.],10.,3)
for key,refkey in [('native_onehot8','native8'),('other_mask8','mask8'),('other_dxdy16','xy16')]:ok('scalar_oracle_'+key,np.array_equal(z[key],np.asarray(ref[refkey],np.float32)))
ok('all_features_FP32_finite',all(z[k].dtype==np.float32 and np.isfinite(z[k]).all()for k in ('native_onehot8','other_mask8','other_dxdy16')))
ok('self_zero_xy_and_mask',not z['other_mask8_bool'][3]and np.array_equal(z['other_dxdy16'][6:8],[0.,0.]))
q2=[r[:]for r in q];q2[3]=[999.,-999.]
ok('no_original_active_anchor_exposed',all(np.array_equal(z[k],run(q2,flags,[30.,-15.],10.)[k])for k in z))
uq=[r[:]for r in q];uq[1]=[float('nan'),float('nan')];uf=flags[:];uf[1]=False
u=run(uq,uf,[30.,-15.],10.)
ok('unavailable_nan_xy0_mask0',not u['other_mask8_bool'][1]and not u['other_dxdy16'][2:4].any())
uf[3]=False
allU=run(uq,uf,None,None)
ok('activeU_keeps_native_but_zero_context',np.array_equal(allU['native_onehot8'],z['native_onehot8'])and not allU['other_mask8'].any()and not allU['other_dxdy16'].any())
large=2.**30;q3=[[large+.25*i,large-.25*i]for i in range(8)]
f64=run(q3,flags,[large,large],.5)
ok('F64_before_FP32_critical_fixture',f64['other_dxdy16'][2]==np.float32(.5)and f64['other_dxdy16'][3]==np.float32(-.5))
clip=run(q,flags,[0.,0.],.125)
ok('clip_both_signs_exact8',np.array_equal(clip['other_dxdy16'][2:4],[8.,-8.]))
shift=run([[x+7,y-9]for x,y in q],flags,[37.,-24.],10.)
scale=run([[x*2,y*2]for x,y in q],flags,[60.,-30.],20.)
ok('common_translation_scale_invariance',all(np.array_equal(z[k],shift[k])and np.array_equal(z[k],scale[k])for k in z))
for i in range(15):ok('SELF_state_'+str(i),A.fit_nominal_eligible('SYNTH_AUG65536',dict(family='SELF_SCENE15',state_index=i,state_name='NOMINAL'if i==0 else'JITTER'))==(i==0))
ok('BEFORE_all_retained',A.fit_nominal_eligible('SYNTH_AUG65536',dict(family='BEFORE_ANCHOR4',state_index=0,state_name='NOMINAL')))
for domain in ('PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL'):ok(domain+'_unchanged_eligibility',A.fit_nominal_eligible(domain,{}))
for name,fn in [('flag_drift',lambda:A.build_context_vectors([0.,0.],10.,3,False,q,[10.]*8,flags)),('bool_native',lambda:run(q,flags,[0.,0.],10.,True)),('invalidL',lambda:run(q,flags,[0.,0.],0.)),('bad_before_state',lambda:A.fit_nominal_eligible('SYNTH_AUG65536',dict(family='BEFORE_ANCHOR4',state_index=1,state_name='JITTER'))),('bad_self_name',lambda:A.fit_nominal_eligible('SYNTH_AUG65536',dict(family='SELF_SCENE15',state_index=1,state_name='NOMINAL')))]:
 try:fn();passed=False
 except ValueError:passed=True
 ok(name+'_rejected',passed)
report={'completed':True,'PASS':True,'invented_only':True,'A_code_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checks':checks,'check_count':len(checks),'actual_metadata_loader_calls':0,'actual_source_NPZ_headers_or_model_reads':0}
b=(json.dumps(report,indent=2)+'\n').encode()
with(H/'A_PURE_API_INVENTED_CHECKS.json').open('xb')as f:f.write(b)
print(json.dumps({'PASS':True,'checks':len(checks),'sha256':hashlib.sha256(b).hexdigest()}))
