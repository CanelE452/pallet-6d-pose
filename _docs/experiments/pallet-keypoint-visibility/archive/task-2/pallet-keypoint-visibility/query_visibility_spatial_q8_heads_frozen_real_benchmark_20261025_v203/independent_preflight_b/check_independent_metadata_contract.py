"""Independent scalar metadata/guard fixtures. No arrays/weights/private data."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,stat
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PRED=V/'prediction_preparation_a/infer_both_frozen_spatial_q8_heads.py'
EVAL=V/'evaluation_preparation_a/evaluate_saved_spatial_q8_heads_real.py'

def safe(p):
    p=Path(p);assert p.is_absolute()and'v112'not in str(p).casefold()and'..'not in p.parts
    cur=Path(p.anchor)
    for q in p.parts[1:]:cur/=q;assert not stat.S_ISLNK(cur.lstat().st_mode)
    return p

def module(p,name):
    p=safe(p);sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def rejects(f):
    try:f()
    except(AssertionError,ValueError):return True
    return False

def run():
    p=module(PRED,'independent_spatial_predictor_toy');e=module(EVAL,'independent_spatial_evaluator_toy');checks={}
    source=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/context_adapter_a/original_q8_context_adapter.py';binding=dict(path=str(source),sha256='b4c3134970004fcfb9351f35b16bc5a81c75ebd39cad9ee98784fca7059de5b6');kernel=p.context_kernel(binding)
    n=16;rows=[];q=[];Ls=[];av=[]
    for i in range(n):
        k=i%8;g=i//8;yes=k not in(5,7);point=None if k==7 else[1.+2.**-30+k*(.25 if g==0 else 100.),10.-k]
        q.append([np.nan,np.nan]if point is None else point);Ls.append(np.nan if k==7 else(1. if g==0 else .5));av.append(yes)
        rows.append(dict(domain='AUTHORED',group_id='GROUP%d'%g,frame_index=g,id='FRAME%d'%g,row_id='FRAME%d:native%d'%(g,k),original_global_index=i,native_slot=k,query_xy=point,L=None if k==7 else Ls[-1],original_query_available=yes))
    x=dict(row_id=np.asarray([r['row_id']for r in rows]),native_slot=np.arange(n,dtype=np.int8)%8,query_xy=np.asarray(q,np.float64),L=np.asarray(Ls,np.float64),original_query_available=np.asarray(av,bool))
    ctx,proof=p.build_real_context(x,rows,kernel);mask=np.zeros((n,8),np.float32);xy=np.zeros((n,16),np.float32)
    for i,r in enumerate(rows):
        if not av[i]:continue
        base=(i//8)*8
        for j in range(8):
            if j==i%8 or not av[base+j]:continue
            mask[i,j]=1
            for axis in range(2):xy[i,2*j+axis]=np.float32(max(-8.,min(8.,(float(q[base+j][axis])-float(q[i][axis]))/float(Ls[i]))))
    checks['two_original_groups_scalar_F64_clip_mask_xy_exact']=np.array_equal(mask,ctx['other_mask'])and np.array_equal(xy,ctx['other_dxdy'])and proof['groups']==2
    checks['self_unavailable_activeU_zero_despite_finite_otherUcoords']=not ctx['other_mask'][~np.asarray(av)].any()and not ctx['other_dxdy'][~np.asarray(av)].any()and not ctx['other_mask'][:,5].any()and all(ctx['other_mask'][i,i%8]==0 for i in range(n))
    def mutated(key,value,i=7):
        rr=[dict(r)for r in rows];rr[i][key]=value;return rr
    checks['crossgroup_native_rowid_F64orU_mismatch_failclosed']=all(rejects(lambda rr=rr:p.build_real_context(x,rr,kernel))for rr in[mutated('group_id','OTHER'),mutated('row_id','WRONG'),mutated('query_xy',[1.,2.],0),mutated('original_query_available',True,5)])
    tiny=np.nextafter(np.float32(0),np.float32(1));raw=np.array([-tiny,-0.,0.,tiny,np.inf,np.nan,np.nan],np.float32);available=np.array([1,1,1,1,1,1,0],bool);ss=np.zeros((7,2),np.float32);oo=np.zeros((7,3),np.float32);ss[-1]=oo[-1]=np.nan
    a=p.decode(ss,oo,raw,available);checks['exact_zero_subnormal_nonfinite_U_same_prediction_evaluation']=a[2].tolist()==[0,1,1,1,-1,-1,-1]and np.array_equal(a[2],e.direct_decode(raw,available))
    b=dict(path='/authored/metadata.json',sha256='a'*64);pp=dict(scope=b,decoder_spec=b,spatial_head=b,source_context_adapter=b,input_complete=b,original_raw_complete=b,source_fit_complete_path=b['path'],prediction_producer_code=b,prediction_producer_protocol=b)
    seal=dict(completed=True,PASS=True,scope=b,decoder_spec=b,spatial_head=b,source_context_adapter=b,input_complete=b,original_raw_complete=b,source_fit_complete=b,code=b,protocol=b,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,original_rows2144=2144,model_domain_rows4288=4288,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,selected_model=None,adopted=False,arms={a:dict(updates=4000,use_sam=False,spatial_q8=e.ARM_SPATIAL[a],outputs={d:b for d in e.DOMAINS})for a in e.ARMS})
    assert set(e.validate_prediction_seal(seal,pp))==set(e.TAGS)
    wrong=[]
    wrong.append(dict(seal,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False));wrong.append(dict(seal,arms={e.ARMS[0]:seal['arms'][e.ARMS[0]]}))
    for k,v in [('updates',8000),('use_sam',True),('spatial_q8',False)]:
        arms={a:dict(r)for a,r in seal['arms'].items()};arms[e.ARMS[1]][k]=v;wrong.append(dict(seal,arms=arms))
    checks['actual_both_final4000_seal_rejects_exposure_missingarm_SAM_spatial_stepdrift']=all(rejects(lambda s=s:e.validate_prediction_seal(s,pp))for s in wrong)
    old=dict(prior_native_prediction_code=b,prior_native_prediction_protocol=b,input_complete=b,original_raw_complete=b);native=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,code=b,protocol=b,input_complete=b,original_raw_complete=b,arms={a:dict(updates=8000,use_sam=False,native_condition=bool(j),outputs={d:b for d in e.DOMAINS})for j,a in enumerate(e.PRIOR_NATIVE_ARMS)})
    checks['old_V200_two_saved_native_routes_only']=set(e.prior_native_outputs(native,old))==set(e.PRIOR_NATIVE_TAGS)
    checks['current12_public11_old10_primaryroute_counts']=len(e.MODELS)==12 and len(e.BASELINES+e.OLD_TAGS)==10 and len(e.ALLTAGS)+1==11 and len(e.DIRECT_TAGS)==8
    # Exact actual SELF guard AST: no annotations, all authored rows only.
    tree=ast.parse(safe(EVAL).read_text());guard=next(n for n in ast.walk(tree)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='require'and len(n.value.args)==2 and isinstance(n.value.args[1],ast.Constant)and n.value.args[1].value=='Original self all136 and pairedflag21 are separate from primary D/E1237')
    class Base:
        S='SELF_OCCLUDED'
        @staticmethod
        def yes(r,k):return r.get(k)=='True'
    fixture=[dict(human_state=Base.S,paired_same_query_RGB_D_vs_E='False',paired_same_query_RGB_D_vs_ES=str(i<21))for i in range(136)];code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'authored_actual_SELF_guard','exec');ns=dict(require=e.require,base=Base,state_inventory={Base.S:136},truth=fixture);exec(code,ns)
    checks['actual_SELF_DvsES136_21_guard_not_primaryDE']=rejects(lambda:exec(code,dict(ns,truth=[dict(r,paired_same_query_RGB_D_vs_ES='False',paired_same_query_RGB_D_vs_E=str(i<21))for i,r in enumerate(fixture)])))
    # Actual old10 guard with strict authored CSV equality, rejects prior C59 drift.
    node=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='prior_ten_exact'for t in n.targets));guard=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'authored_actual_old10_guard','exec')
    def exact(a,b):
        if a!=b:raise ValueError('Authored strict historical drift')
        return True
    new=[dict(model=m,N='1',TP='1')for m in e.MODELS];oldrows=[r for r in new if r['model']in e.BASELINES+e.OLD_TAGS];ns=dict(tables=dict(metrics=new),prior_ten=oldrows,BASELINES=e.BASELINES,OLD_TAGS=e.OLD_TAGS,exact=exact);exec(guard,ns)
    checks['actual_old10_guard_accept_equal_reject_C59_drift']=rejects(lambda:exec(guard,dict(ns,prior_ten=[dict(r,TP='0')if r['model']=='C59'else dict(r)for r in oldrows])))
    # Lexical before any filesystem call in both readers, and negative actual entry first gate.
    for mod in (p,e):
        oldopen=mod.os.open;events=[]
        try:
            mod.os.open=lambda *a,**k:events.append(a)or (_ for _ in ()).throw(RuntimeError('Premature filesystem'))
            assert rejects(lambda:mod.opened(H/'FORBIDDEN_V112'/'input.json'))and not events
        finally:mod.os.open=oldopen
    checks['both_NOFOLLOW_readers_lexical_forbidden_before_open']=True
    predtree=ast.parse(safe(PRED).read_text());body=ast.unparse(next(n for n in predtree.body if isinstance(n,ast.FunctionDef)and n.name=='run'));ebody=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run'))
    checks['source_final_CQA_checkpoint_gate_before_real_arrays_and_labels_afterseal']=body.index('source_entry = read')<body.index('x = arrays')and body.index('barrier = read')<body.index('x = arrays')and ebody.index('outputs = validate_prediction_seal')<ebody.index("check(p['current']['annotation_authority'])")
    # Main metric kernels inherited verbatim; newtags/old10 routing handled separately.
    oldcode=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200/evaluation_preparation_a/evaluate_saved_native_heads_real.py'
    funcs=lambda path:{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(safe(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    prev,now=funcs(oldcode),funcs(EVAL);same=('require','safe','opened','sha','bind','check','read','lines','write','write_lines','equal','extract','kernels','decode','direct_decode','joint','qarray','packet_schema','load_packet','current_associations','prior_outputs','prior_direct_outputs','prior_grouped_outputs')
    checks['23_V200_metric_schema_decoder_path_helpers_AST_exact']=all(prev[k]==now[k]for k in same)
    assert all(checks.values()),checks
    return {k:bool(v)for k,v in checks.items()}
