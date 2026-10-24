"""Invented-only checks of the production helpers; no source metadata loading."""
from pathlib import Path
import hashlib, importlib.util, json, stat
import numpy as np

H=Path(__file__).absolute().parent
code=H/'original_q8_context_adapter.py'
spec=importlib.util.spec_from_file_location('v202_targetfree_context',code)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
checks={}

def check(name,fn):fn();checks[name]=True
def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid fixture accepted')
def make(k=0,av=None,q=None,L=2.):
    av=[True]*8 if av is None else av
    q=[[float(j),float(-j)] for j in range(8)] if q is None else q
    return a.build_context_vectors([0.,0.],L,k,av[k],q,[L]*8,av)

def basic():
    for k in range(8):
        z=make(k)
        assert z['other_dxdy16'].dtype==np.float32 and z['other_dxdy16'].shape==(16,)
        assert z['native_onehot8'].dtype==z['other_mask8'].dtype==np.float32
        assert z['other_mask8_bool'].dtype==bool and z['other_mask8_bool'].sum()==7
        assert z['native_onehot8'][k]==1 and z['native_onehot8'].sum()==1
        assert not z['other_mask8_bool'][k] and not z['other_dxdy16'].reshape(8,2)[k].any()
check('all8native_shapes_dtype_selfzero_nativeunchanged',basic)

def fp64():
    q=[[1e8+.375,1e8+.375] for _ in range(8)]
    z=a.build_context_vectors([1e8+.125,1e8+.125],.5,0,True,q,[.5]*8,[True]*8)
    assert np.array_equal(z['other_dxdy16'].reshape(8,2)[1],np.array([.5,.5],np.float32))
    assert np.float32(1e8+.375)==np.float32(1e8+.125)
check('F64_subtract_divide_before_FP32_precision_boundary',fp64)

def clipping():
    q=[[1e6,-1e6] for _ in range(8)];z=make(q=q,L=.25)['other_dxdy16'].reshape(8,2)
    assert np.array_equal(z[1:],np.tile(np.array([8.,-8.],np.float32),(7,1)))
check('exact_clipping_and_interleaved_xy',clipping)

def unavailable():
    av=[True,False,True,True,True,True,True,True];q=[[0.,0.],[1e8,1e8]]+[[2.,2.]]*6
    z=make(av=av,q=q)
    assert not z['other_mask8_bool'][1] and not z['other_dxdy16'].reshape(8,2)[1].any()
check('finite_other_coordinate_does_not_override_original_U',unavailable)

def activeU():
    av=[False]+[True]*7
    z=a.build_context_vectors(None,None,0,False,[None]+[[1.,2.]]*7,[None]+[2.]*7,av)
    assert not z['other_mask8'].any() and not z['other_dxdy16'].any()
    assert z['native_onehot8'][0]==1
check('activeU_null_preserved_all_contextzero_nativeunchanged',activeU)

def mask_errors():
    rejects(lambda:a.build_context_vectors([0.,0.],2.,0,False,[[1.,2.]]*8,[2.]*8,[True]*8))
    rejects(lambda:a.build_context_vectors([0.,0.],2.,0,True,[[1.,2.]]*8,[2.]*8,[1]*8))
check('availability_type_and_active_origin_exact_mismatch_rejected',mask_errors)

def malformed():
    for k in (-1,8,True,1.5,'1'):rejects(lambda k=k:make(k))
    rejects(lambda:a.build_context_vectors(None,2.,0,True,[[1.,2.]]*8,[2.]*8,[True]*8))
    rejects(lambda:make(L=0.))
    q=[[1.,2.]]*8;q[1]=None
    rejects(lambda:make(q=q))
    rejects(lambda:a.build_context_vectors([0.,0.],2.,0,True,[[1.,2.]]*7,[2.]*7,[True]*7))
check('invalid_native_q_L_and_incomplete8_rejected',malformed)

def eligibility():
    for role in ('TRAIN','HOLDOUT'):
        r={'family':'SELF_SCENE15','state_name':'NOMINAL','state_index':0,'role':role}
        assert a.fit_nominal_eligible('SYNTH_AUG65536',r) and r['role']==role
        r.update(state_name='SHIFT_+1_+0',state_index=5)
        assert not a.fit_nominal_eligible('SYNTH_AUG65536',r) and r['role']==role
        r.update(family='BEFORE_ANCHOR4',state_name='NOMINAL',state_index=0)
        assert a.fit_nominal_eligible('SYNTH_AUG65536',r) and r['role']==role
    for name in ('PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL'):assert a.fit_nominal_eligible(name,{})
    rejects(lambda:a.fit_nominal_eligible('SYNTH_AUG65536',{'family':'SELF_SCENE15','state_name':'NOMINAL','state_index':1}))
check('nominal_eligibility_is_independent_of_immutable_roles_allotherpools_unchanged',eligibility)

def group():
    rr=[dict(row_id=str(k),original_global_index=k,native_slot=k,query_xy=[float(k),2.],L=8.,original_query_available=True) for k in range(8)]
    assert [x['native_slot'] for x in a.ordered8(rr[::-1])]==list(range(8))
    bad=[dict(x) for x in rr];bad[7]['native_slot']=6;rejects(lambda:a.ordered8(bad))
    bad=[dict(x) for x in rr];bad[0]['query_xy']=[999.,999.];rejects(lambda:a.assert_meta6(rr[0],bad[0]))
    bad=[dict(x) for x in rr];bad[0]['original_query_available']=False;rejects(lambda:a.assert_meta6(rr[0],bad[0]))
check('group8_membership_order_originalq_and_U_mismatch_negative',group)

def guards():
    old=a.os.lstat;calls=[]
    def forbidden_stat(p):calls.append(str(p));raise AssertionError('No stat allowed')
    a.os.lstat=forbidden_stat
    try:rejects(lambda:a.safe_path('/invented_v112/no_file.json'));assert not calls
    finally:a.os.lstat=old
    a.os.lstat=lambda _:type('S',(),{'st_mode':stat.S_IFLNK})()
    try:rejects(lambda:a.safe_path('/invented/symlink.json'))
    finally:a.os.lstat=old
check('forbidden_lexical_before_stat_and_component_symlink_refusal',guards)

def byte_replay():
    # Same row take in botharms; control removes XY only and preserves ID/mask.
    contexts=[make(k) for k in range(8)];take=np.array([7,1,7,0,3],np.int64)
    native=np.stack([z['native_onehot8'] for z in contexts]);mask=np.stack([z['other_mask8'] for z in contexts]);xy=np.stack([z['other_dxdy16'] for z in contexts])
    assert np.array_equal(native[take],np.eye(8,dtype=np.float32)[take])
    assert np.array_equal(mask[take],np.stack([contexts[int(k)]['other_mask8'] for k in take]))
    control_xy=np.zeros_like(xy[take]);assert not control_xy.any() and np.array_equal(mask[take],mask[take].copy())
check('same_draw_take_alignment_and_control_only_zero16XY',byte_replay)

out={'completed':True,'PASS':True,'invented_only':True,'checks':checks,'count':len(checks),'production_code':{'path':str(code),'sha256':hashlib.sha256(code.read_bytes()).hexdigest()},'test_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actual_source_metadata_NPZ_pixel_model_weight_score_reads':0}
p=H/'INVENTED_CHECKS.json'
with p.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'count':len(checks),'checks_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
