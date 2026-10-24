"""Authored metadata fixtures only; no model, arrays, weights, scores or GT."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,stat,sys
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent
P=H.parent.parent
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def read_bytes(p):
    p=Path(p).absolute()
    require(P in p.parents and '..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Lexical project guard')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(n).st_mode),'Regular code only')
    with os.fdopen(n,'rb')as f:return f.read()
def load(name,path):
    source=read_bytes(path);m=type(sys)(name);m.__file__=str(path);exec(compile(source,str(path),'exec'),m.__dict__);return m
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(read_bytes(p)).hexdigest())
def main():
    pred=load('authored_prediction',H/'infer_both_frozen_dense_point_heads.py')
    ev=load('authored_evaluation',H/'evaluate_saved_dense_point_heads_real.py')
    passed=[]
    def test(n,ok=True):require(ok,n);passed.append(n)
    def rejects(n,fn):
        try:fn()
        except (ValueError,AssertionError,KeyError):test(n)
        else:raise AssertionError('Expected rejection: '+n)
    b=lambda p:dict(path=str(p),sha256='a'*64)
    e=dict(fit_complete=b(pred.FITPHASE/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/COMPLETE.json'),independent_source_fit_review=b(pred.FITPHASE/'independent_saved_fit_review_c/REVIEW.json'),source_fit_protocol=b(pred.FITPHASE/'fit_preparation_a/PROTOCOL.json'),source_fit_trainer=b(pred.FITPHASE/'fit_preparation_a/train_dense_point_weak_bridge.py'),source_both_final_freeze=b(pred.FITPHASE/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/ALL2_FINAL4000_FROZEN_BEFORE_ANY_SOURCE_OUTPUTS.json'),source_fit_entry=b(pred.FITPHASE/'ROOT_ACTUAL_FIT_ENTRY.json'))
    p=dict(source_fit_complete_path=e['fit_complete']['path'],source_fit_trainer_path=e['source_fit_trainer']['path'])
    p=pred.resolve_source_bindings(e,p)
    for k in ('head','candidate','spatial_head','source_context_adapter'):p[k]=b(pred.FITPHASE/('authored_'+k+'.py'))
    test('Concrete_future_source_bindings_resolve_without_body_access')
    rejects('Missing_future_source_binding_rejected',lambda:pred.resolve_source_bindings(dict(e,source_fit_protocol=None),p))
    rejects('Crossphase_source_binding_rejected',lambda:pred.resolve_source_bindings(dict(e,source_fit_protocol=b(H/'invented.json')),p))
    fit=dict(completed=True,PASS=True,source_only=True,from_scratch=True,all2_frozen_before_any_SOURCE_outputs=True,code=p['source_fit_trainer'],entry=e['source_fit_entry'],protocol=p['source_fit_protocol'],head=p['head'],candidate=p['candidate'],spatial_head=p['spatial_head'],context_adapter=p['source_context_adapter'],parameters_each=118359,steps_each=4000,total_updates=8000,protected_real_COMMON_VAL_access=0,all2_freeze=p['source_both_final_freeze'],arms={a:dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=pred.ARM_WEAK[a][0],weak_loss_weight=pred.ARM_WEAK[a][1],checkpoint=b(pred.FITPHASE/(a+'.pt')))for a,_,_ in pred.ARMS})
    review=dict(completed=True,PASS=True,fit_complete=e['fit_complete'])
    se=dict(code=p['source_fit_trainer'],protocol=p['source_fit_protocol'])
    barrier=dict(completed=True,checkpoints={a:r['checkpoint']for a,r in fit['arms'].items()})
    test('Exact_current_fit_review_and_two_final_barrier',pred.validate_source_metadata(fit,review,e,p,se,barrier))
    rejects('Review_PASS_for_different_fit_rejected',lambda:pred.validate_source_metadata(fit,dict(review,fit_complete=b(H/'wrong.json')),e,p,se,barrier))
    bad=copy.deepcopy(fit);bad['arms'][pred.ARMS[1][0]]['weak_loss_weight']=0.
    rejects('Apriori_weak_bridge_weight_drift_rejected',lambda:pred.validate_source_metadata(bad,review,e,p,se,barrier))
    bad=copy.deepcopy(barrier);bad['checkpoints'][pred.ARMS[0][0]]=b(H/'wrong.pt')
    rejects('Checkpoint_not_in_both_final_barrier_rejected',lambda:pred.validate_source_metadata(fit,review,e,p,se,bad))
    bad=copy.deepcopy(fit);bad['protected_real_COMMON_VAL_access']=1
    rejects('Protected_access_claim_nonzero_rejected',lambda:pred.validate_source_metadata(bad,review,e,p,se,barrier))
    bb=b(H/'invented_binding.json')
    pp={k:bb for k in ('scope','decoder_spec','spatial_head','source_context_adapter','input_complete','original_raw_complete','prediction_producer_code','prediction_producer_protocol')};pp['source_fit_complete_path']=bb['path']
    ss=dict(completed=True,PASS=True,scope=bb,decoder_spec=bb,spatial_head=bb,source_context_adapter=bb,input_complete=bb,original_raw_complete=bb,source_fit_complete=bb,code=bb,protocol=bb,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,original_rows2144=2144,model_domain_rows4288=4288,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,selected_model=None,adopted=False,arms={a:dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=ev.ARM_WEAK[a][0],weak_loss_weight=ev.ARM_WEAK[a][1],outputs={d:bb for d in ev.DOMAINS})for a in ev.ARMS})
    test('Both_new_spatial_RAW_arms_seal_exact_roles',set(ev.validate_prediction_seal(ss,pp))==set(ev.TAGS))
    bad=copy.deepcopy(ss);bad['arms'][ev.ARMS[1]]['weak_bridge']=False
    rejects('New_seal_wrong_weak_role_rejected',lambda:ev.validate_prediction_seal(bad,pp))
    oldp=dict(prior_spatial_prediction_code=bb,prior_spatial_prediction_protocol=bb,input_complete=bb,original_raw_complete=bb)
    olds=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,code=bb,protocol=bb,input_complete=bb,original_raw_complete=bb,arms={a:dict(updates=4000,use_sam=False,spatial_q8=bool(i),outputs={d:bb for d in ev.DOMAINS})for i,a in enumerate(ev.PRIOR_SPATIAL_ARMS)})
    test('Old_C60_C61_saved_routes_preserve_old_ablation_flags',set(ev.prior_spatial_outputs(olds,oldp))==set(ev.PRIOR_SPATIAL_TAGS))
    bad=copy.deepcopy(olds);bad['arms'][ev.PRIOR_SPATIAL_ARMS[0]]['spatial_q8']=True
    rejects('Old_C60_context_flag_drift_rejected',lambda:ev.prior_spatial_outputs(bad,oldp))
    tree=ast.parse(read_bytes(H/'evaluate_saved_dense_point_heads_real.py'))
    guard=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='prior_twelve_exact'for t in n.targets))
    code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'authored_old12_guard','exec')
    rows=[dict(model=m,N='1')for m in ev.MODELS];old=[r.copy()for r in rows if r['model']in ev.BASELINES+ev.OLD_TAGS]
    def exact(x,y):require(x==y,'Exact authored old12');return True
    ns=dict(tables=dict(metrics=rows),prior_twelve=old,BASELINES=ev.BASELINES,OLD_TAGS=ev.OLD_TAGS,exact=exact)
    exec(code,ns);test('Production_old12_metric_AST_includes_C60_C61')
    drift=[dict(r,N='2')if r['model']=='C61'else r.copy()for r in old]
    rejects('Old_C61_metric_drift_rejected',lambda:exec(code,dict(ns,prior_twelve=drift)))
    parity={}
    old= P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203'
    for current,parent in [(H/'infer_both_frozen_dense_point_heads.py',old/'prediction_preparation_a/infer_both_frozen_spatial_q8_heads.py'),(H/'evaluate_saved_dense_point_heads_real.py',old/'evaluation_preparation_a/evaluate_saved_spatial_q8_heads_real.py')]:
        functions=lambda raw:{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
        aa=functions(read_bytes(current));zz=functions(read_bytes(parent));same=[n for n in aa if n in zz and aa[n]==zz[n]]
        parity[current.name]=dict(parent=bind(parent),unchanged_function_names=same,count=len(same))
    require(all(n in parity['infer_both_frozen_dense_point_heads.py']['unchanged_function_names']for n in ('build_real_context','infer','validate_input')),'Original F64 context/input/inference AST')
    require(all(n in parity['evaluate_saved_dense_point_heads_real.py']['unchanged_function_names']for n in ('packet_schema','load_packet','kernels')),'Original saved-packet and scalar kernel AST')
    test('Original_input_context_inference_and_metric_kernel_AST_unchanged')
    out=dict(completed=True,PASS=True,invented_only=True,count=len(passed),checks=passed,code_bindings={n:bind(H/n)for n in ('infer_both_frozen_dense_point_heads.py','evaluate_saved_dense_point_heads_real.py')},parent_function_parity=parity,actual_source_or_real_payload_headers_hashes_or_bodies=0,models_or_torch_imports_or_forwards=0)
    p=H/'NEW_SOURCE_AND_SAVED_ROUTES_CHECKS.json'
    with p.open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,count=len(passed),receipt=bind(p))))
if __name__=='__main__':main()
