"""Bounded TRAIN fit-capacity probe; frozen features, no current or encoder."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,ast,hashlib,json,time,traceback
from pathlib import Path
import numpy as np

V=Path(__file__).resolve().parents[1]
OUT=V/'capacity_probe/FULL_BATCH_LBFGS_MAX400'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
class CapacityBudgetStop(Exception):pass

def main(entry):
    e=read(entry);assert e['approved_public_TRAIN_capacity_probe']
    assert all(not e[k]for k in ('current_allowed','V112_allowed','PnP_allowed','encoder_allowed'))
    assert e['code']==bind(__file__)
    policy=read(check(e['policy']));d=read(check(e['design']))
    for k in ('wide_features','wide_input_occurrences','wide_GT_occurrences','head_class_code'):check(d[k])
    check(e['parent_final_checkpoint']);check(e['parent_final_scores'])
    assert policy['max_closure_calls']==400 and policy['max_LBFGS_iterations']==300
    assert not OUT.exists(),'Preserve any previous attempt; no automatic restart'
    import torch
    assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(223);torch.cuda.manual_seed_all(223)
    r=rows(d['wide_input_occurrences']['path']);gt=rows(d['wide_GT_occurrences']['path'])
    with np.load(d['wide_features']['path'],allow_pickle=False)as f:
        x=f['features'].copy();um=f['occurrence_unit'].copy();available=f['query_available'].copy()
    y=np.array([g['candidate_aux_hidden_label_GT_only']for g in gt],np.int8)
    split=np.array([z['split']for z in r]);variant=np.array([z['variant']for z in r])
    alpha=np.array([-1 if z['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is None else int(z['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for z in gt])
    train=(split=='TRAIN')&available&np.isin(y,[0,1])
    assert x.shape==(4428,1928)and x.dtype==np.float32 and train.sum()==795
    masks=[('E_qOUT',(y==1)&(alpha==0),.25),('E_qIN',(y==1)&(alpha==1),.25),
           ('SHAM_D_qIN',(y==0)&(variant=='CUE_PRESERVING_SHAM')&(alpha==1),.25),
           ('SHAM_D_qOUT',(y==0)&(variant=='CUE_PRESERVING_SHAM')&(alpha==0),.125),
           ('BEFORE_D',(y==0)&(variant=='BEFORE'),.125)]
    weights=np.zeros(4896,np.float64);pool_counts={}
    for name,m,mass in masks:
        m=m&train;n=int(m.sum());assert n>0
        pool_counts[name]=n;weights[m]=mass/n
    assert pool_counts=={'E_qOUT':156,'E_qIN':216,'SHAM_D_qIN':77,'SHAM_D_qOUT':295,'BEFORE_D':51}
    assert np.array_equal(weights>0,train)and np.isclose(weights.sum(),1)
    assert np.isclose(weights[y==0].sum(),.5)and np.isclose(weights[y==1].sum(),.5)
    tree=ast.parse(Path(d['head_class_code']['path']).read_text());cls=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='SpatialTargetQueryHead'];assert len(cls)==1
    ns={'torch':torch,'__name__':'fixed_feature_capacity_class'}
    exec(compile(ast.Module(body=cls,type_ignores=[]),d['head_class_code']['path']+'::class_only','exec'),ns)
    model=ns['SpatialTargetQueryHead']().cuda().float()
    payload=torch.load(e['parent_final_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert payload['update']==1000 and payload['head_class_code']==d['head_class_code']
    model.load_state_dict(payload['state_dict'],strict=True);del payload
    xg=torch.as_tensor(x,device='cuda');ig=torch.as_tensor(um[train],device='cuda',dtype=torch.long)
    xt=xg[ig];yt=torch.as_tensor(y[train],device='cuda',dtype=torch.float32)
    wt=torch.as_tensor(weights[train],device='cuda',dtype=torch.float32)
    with torch.no_grad():initial=model.eval()(xg).cpu().numpy()[um]
    initial=np.where(available,initial,0).astype(np.float32)
    with np.load(e['parent_final_scores']['path'],allow_pickle=False)as f:
        assert np.array_equal(initial,f['query_logit'][51004:])and np.array_equal(available,f['query_available'][51004:]),'Preserve warmstart drift; no tolerance relaxation'
    OUT.mkdir(parents=True);start=time.monotonic()
    write(OUT/'STARTED.json',dict(entry=bind(entry),train_examples=795,pool_counts=pool_counts,initial_logits_exact_C18_final1000=True,validation_used_for_optimizer_or_selection=False))
    calls=0;last_finite_state=None;reason='optimizer_returned'
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=300,max_eval=400,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-10)
    try:
        def closure():
            nonlocal calls,last_finite_state
            if calls>=400:raise CapacityBudgetStop()
            opt.zero_grad();z=model.train()(xt)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yt,reduction='none')*wt).sum()
            assert torch.isfinite(loss),'Numeric failure is not proof of capacity failure'
            loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters())
            calls+=1;last_finite_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
            accuracy=float(((z>=0)==(yt>=.5)).float().mean())
            with(OUT/'TRAIN_OPTIMIZATION_TRACE.jsonl').open('a')as f:
                f.write(json.dumps(dict(closure=calls,weighted_TRAIN_BCE=float(loss.detach()),TRAIN_correct=int(((z>=0)==(yt>=.5)).sum()),TRAIN_N=795,seconds=time.monotonic()-start))+'\n')
            if calls%50==0:print(json.dumps(dict(closure=calls,TRAIN_BCE=float(loss.detach()),TRAIN_accuracy=accuracy)),flush=True)
            return loss
        try:opt.step(closure)
        except CapacityBudgetStop:
            reason='fixed400_closure_budget_last_finite_TRAIN_state';model.load_state_dict(last_finite_state)
        assert calls<=400
        with torch.no_grad():
            raw=model.eval()(xg);again=model(xg);assert torch.equal(raw,again)
        z=np.where(available,raw.cpu().numpy()[um],0).astype(np.float32)
        pred=np.where(available,(z>=0).astype(np.int8),-1).astype(np.int8)
        np.savez_compressed(OUT/'FINAL_PUBLIC_WIDE_SCORES.npz',query_logit=z,prediction=pred,query_available=available,global_occurrence_index=np.arange(4896))
        def metric(mask):
            cm=[[int((mask&(y==a)&(pred==b)).sum())for b in (0,1,-1)]for a in (0,1)]
            known=mask&np.isin(y,[0,1]);valid=known&available;correct=int((valid&(pred==y)).sum())
            zz=z[valid].astype(np.float64);bce=np.logaddexp(0,zz)-y[valid]*zz
            return dict(CM_rows_D_E_cols_D_E_U=cm,known_planned=int(known.sum()),known_available=int(valid.sum()),correct_available=correct,available_accuracy=correct/int(valid.sum())if valid.any()else None,mean_available_BCE=float(bce.mean())if valid.any()else None)
        groups={}
        for name,m in [('TRAIN',split=='TRAIN'),('LOCO_VAL',np.array([z['source']=='LOCO'and z['split']=='VALIDATION'for z in r])),('TOMIE_DEV',np.array([z['source']=='TOMIE'for z in r])),('VALIDATION_COMBINED',split=='VALIDATION')]:
            groups[name]={'all':metric(m),'strata':{k:metric(m&mk)for k,mk,_ in masks}}
        torch.save(dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},head_class_code=d['head_class_code'],closure_calls=calls,optimizer='LBFGS',parent_checkpoint=e['parent_final_checkpoint'],threshold_logit=0,resume_state_saved=False),OUT/'FINAL_DIAGNOSTIC_HEAD.pt')
        write(OUT/'RESULTS.json',dict(completed=True,reason=reason,closure_calls=calls,max_closure_calls=400,max_iterations=300,head_parameters=221825,encoder_calls=0,source_only=True,current_V112_PnP=0,TRAIN_only_optimizer_and_budget=True,VAL_labels_read_for_partition_audit_before_fit_but_never_used_in_optimizer=True,VAL_metrics_computed_only_after_fit=True,VAL_not_used_for_checkpoint_threshold_or_arm_selection=True,groups=groups,threshold_logit=0,initial_exact_parent=True,pool_counts=pool_counts,optimizer='LBFGS strong_wolfe; fixed stratum-weighted full TRAIN batch; no regularization for capacity diagnostic',elapsed_seconds=time.monotonic()-start,scores=bind(OUT/'FINAL_PUBLIC_WIDE_SCORES.npz'),checkpoint=bind(OUT/'FINAL_DIAGNOSTIC_HEAD.pt'),limits=['TRAIN fitability and public generalization diagnostic only, not current real solution.','This solver change is separate from the C17/C18 matched1000 oldloss control.','Failure to fit under this finite budget does not prove representational impossibility.','Weak cue identity/labels remain conditional AUX, not native/human/pixelgold.']))
        print(json.dumps(read(OUT/'RESULTS.json')),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_RESTART.json',dict(closure_calls=calls,traceback=traceback.format_exc(),no_automatic_restart=True));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True,type=Path)
    main(ap.parse_args().entry)
