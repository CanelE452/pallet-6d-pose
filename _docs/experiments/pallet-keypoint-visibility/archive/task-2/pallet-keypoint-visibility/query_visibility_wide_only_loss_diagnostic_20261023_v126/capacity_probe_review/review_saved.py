"""Saved-score/GT arithmetic only. No Torch, model, checkpoint or feature reads."""
from pathlib import Path
import hashlib,json
import numpy as np

V=Path(__file__).resolve().parents[1]
O=V/'capacity_probe/FULL_BATCH_LBFGS_MAX400'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
entry_path=V/'ROOT_ACTUAL_CAPACITY_PROBE_ENTRY.json';e=read(entry_path)
policy=read(e['policy']['path']);design=read(e['design']['path']);res=read(O/'RESULTS.json');started=read(O/'STARTED.json')
assert res['completed']and e['code']==bind(Path(e['code']['path']))and e['policy']==bind(Path(e['policy']['path']))and e['design']==bind(Path(e['design']['path']))
assert e['parent_final_checkpoint']==policy['parent_fixed1000']
r=rows(design['wide_input_occurrences']['path']);g=rows(design['wide_GT_occurrences']['path'])
assert len(r)==len(g)==4896
y=np.array([x['candidate_aux_hidden_label_GT_only']for x in g]);split=np.array([x['split']for x in r]);variant=np.array([x['variant']for x in r]);alpha=np.array([-1 if x['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is None else int(x['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for x in g])
with np.load(res['scores']['path'],allow_pickle=False)as f:
    z=f['query_logit'];pred=f['prediction'];a=f['query_available'];order=f['global_occurrence_index']
with np.load(e['parent_final_scores']['path'],allow_pickle=False)as f:
    parentz=f['query_logit'][51004:];parenta=f['query_available'][51004:]
assert z.shape==pred.shape==a.shape==order.shape==(4896,)and np.array_equal(order,np.arange(4896))
assert z.dtype==np.float32 and pred.dtype==np.int8 and a.dtype==bool and np.isfinite(z).all()
assert np.array_equal(pred,np.where(a,(z>=0).astype(np.int8),-1))and np.array_equal(a,parenta)and not z[~a].any()
assert np.array_equal(a,np.array([x['query_available']for x in r]))
train=(split=='TRAIN')&a&np.isin(y,[0,1]);assert train.sum()==795 and(~a).sum()==51
masks=[('E_qOUT',(y==1)&(alpha==0),.25),('E_qIN',(y==1)&(alpha==1),.25),('SHAM_D_qIN',(y==0)&(variant=='CUE_PRESERVING_SHAM')&(alpha==1),.25),('SHAM_D_qOUT',(y==0)&(variant=='CUE_PRESERVING_SHAM')&(alpha==0),.125),('BEFORE_D',(y==0)&(variant=='BEFORE'),.125)]
w=np.zeros(4896,np.float64);counts={}
for k,m,mass in masks:
    n=int((m&train).sum());counts[k]=n;w[m&train]=mass/n
assert counts==res['pool_counts']==started['pool_counts']==dict(E_qOUT=156,E_qIN=216,SHAM_D_qIN=77,SHAM_D_qOUT=295,BEFORE_D=51)
assert np.array_equal(w>0,train)and abs(w.sum()-1)<1e-14
assert abs(w[y==0].sum()-.5)<1e-14 and abs(w[y==1].sum()-.5)<1e-14
def metric(mask):
    cm=[[int((mask&(y==aa)&(pred==b)).sum())for b in(0,1,-1)]for aa in(0,1)]
    known=mask&np.isin(y,[0,1]);valid=known&a;correct=int((valid&(pred==y)).sum());zz=z[valid].astype(np.float64);bce=np.logaddexp(0,zz)-y[valid]*zz
    return dict(CM_rows_D_E_cols_D_E_U=cm,known_planned=int(known.sum()),known_available=int(valid.sum()),correct_available=correct,available_accuracy=correct/int(valid.sum())if valid.any()else None,mean_available_BCE=float(bce.mean())if valid.any()else None)
groups={}
for name,m in [('TRAIN',split=='TRAIN'),('LOCO_VAL',np.array([x['source']=='LOCO'and x['split']=='VALIDATION'for x in r])),('TOMIE_DEV',np.array([x['source']=='TOMIE'for x in r])),('VALIDATION_COMBINED',split=='VALIDATION')]:
    groups[name]={'all':metric(m),'strata':{k:metric(m&mk)for k,mk,_ in masks}}
assert groups==res['groups'],'Exact independent saved-score arithmetic mismatch'
trace=rows(O/'TRAIN_OPTIMIZATION_TRACE.jsonl');assert len(trace)==res['closure_calls']==52<=policy['max_closure_calls']==400
assert [x['closure']for x in trace]==list(range(1,53))and all(x['TRAIN_N']==795 and np.isfinite(x['weighted_TRAIN_BCE'])and 0<=x['TRAIN_correct']<=795 for x in trace)
assert all(x['seconds']<=yy['seconds']for x,yy in zip(trace,trace[1:]))
def objective(zz):
    zz=zz[train].astype(np.float64);return float(((np.logaddexp(0,zz)-y[train]*zz)*w[train]).sum())
first_delta=trace[0]['weighted_TRAIN_BCE']-objective(parentz)
last_delta=trace[-1]['weighted_TRAIN_BCE']-objective(z)
assert abs(first_delta)<1e-6 and abs(last_delta)<1e-6
assert trace[0]['TRAIN_correct']==int(((parentz[train]>=0)==y[train]).sum())==671
assert trace[-1]['TRAIN_correct']==int((pred[train]==y[train]).sum())==795
assert started['initial_logits_exact_C18_final1000']and res['initial_exact_parent']
assert res['max_iterations']==policy['max_LBFGS_iterations']==300 and res['reason']=='optimizer_returned'
report=dict(completed=True,PASS=True,blockers=[],scope='Independent static code/policy + completed saved public-score/GT/trace arithmetic; no optimizer/model/encoder execution, no feature/parameter arrays or current/V112 reads.',
    code_contract=dict(same_1928_features_and_221825_head=True,initial_checkpoint='C18 fixed final1000, binding same as prospective policy; no parameter deserialize by reviewer',optimizer='Fresh full-batch LBFGS lr1/history50/strong_wolfe/tolerance_grad1e-7/tolerance_change1e-10; max_iter300 and explicit max400 closure stop',loss='TRAIN-only weighted BCE with stratum masses .25/.25/.25/.125/.125 = expected16/16/16/8/8 draws. No regularizer, no old192 loss.',VAL_boundary='GT sidecar is read before optimization to construct/audit partitions. Only TRAIN known+available rows index xt/yt/wt in closure. Initial all-row forward is parent parity, not a VAL metric/selection step; VAL metrics occur after opt.step stops.',fixed_state='Normal optimizer return, or last finite TRAIN closure state if400 cap; no VAL checkpoint/threshold/arm selection.',no_silent_restart=True),
    metadata=dict(all_rows=4896,teacher_D=int((y==0).sum()),teacher_E=int((y==1).sum()),teacher_U=int((y<0).sum()),factual_input_U=51,known_TRAIN_planned=846,known_TRAIN_available=795,TRAIN_D_available=int((train&(y==0)).sum()),TRAIN_E_available=int((train&(y==1)).sum()),stratum_pools=counts,each_stratum_expected_mass={k:mass for k,_,mass in masks},teacher_U_and_input_U_separate=True,raw0_prediction_all4896_exact=True,original_order_and_parent_availability_exact=True),
    independent_saved_groups=groups,all_group_counts_CMs_accuracies_BCE_match_exactly=True,
    trace=dict(calls=52,budget_max400=True,all_sequential_finite_and_TRAIN_N795=True,initial_TRAIN_correct=671,final_TRAIN_correct=795,initial_weighted_BCE_F64=objective(parentz),final_weighted_BCE_F64=objective(z),first_trace_F32_vs_F64_delta=first_delta,last_trace_F32_vs_F64_delta=last_delta,arithmetic_tolerance='Only float32 producer weighted BCE vs independent float64 objective:1e-6; actual deltas <5e-9. No label/count/CM tolerance. This does not replay hidden optimizer internals.'),
    evidence_limits=['All795 TRAIN examples fit under the fixed frozen-feature/head representation and this solver. This contradicts an absolute claim that these TRAIN labels cannot be represented by this head on these features.','100% TRAIN fit does not establish natural-real transfer, native cue identity, or completion of the user goal. LOCO_VAL/TOMIE are reused development cohorts, not fresh blind performance.','This optimizer probe is a separate multi-change diagnostic from the controlled C17/C18 old-loss coefficient comparison.','Initial per-row bitexact parameter/score parity is a producer assertion plus independently corroborated first weighted loss/correct count; initial full logits were not separately saved by this probe. Checkpoints were not loaded or rehashed by reviewer.','Finite-budget failure would not prove representational impossibility; this run instead succeeded on TRAIN.'],
    interpretation_KO='고정된1928특징과같은head가LBFGS52closure에서TRAIN795/795를맞췄다. 따라서이TRAIN집합의절대적표현불가능을주장할근거는없다. 그러나VAL은E-qOUT325/483,SHAM_D-qIN197/394로불완전하며,이는실사문제해결이나feature가충분하다는일반명제를증명하지않는다. old-loss계수대조와optimizer변경probe는구분해야한다.',
    bindings=dict(code=e['code'],policy=e['policy'],entry=bind(entry_path),design=e['design'],results=bind(O/'RESULTS.json'),started=bind(O/'STARTED.json'),scores=bind(Path(res['scores']['path'])),trace=bind(O/'TRAIN_OPTIMIZATION_TRACE.jsonl'),input_occurrences=design['wide_input_occurrences'],GT_occurrences=design['wide_GT_occurrences'],parent_final_scores=bind(Path(e['parent_final_scores']['path'])),parent_checkpoint_descriptor_only=e['parent_final_checkpoint'],audit_code=bind(Path(__file__))),
    actual_new_forward_optimizer_encoder_parameter_or_feature_array_reads_current_V112_PnP=0,source_outputs_or_labels_modified=False,new_training_or_promotion_decision=False,fully_solved=False)
with(Path(__file__).parent/'REVIEW.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(Path(__file__).parent/'REVIEW.json'),TRAIN_correct=795,TRAIN_available=795,closures=52),ensure_ascii=False))
