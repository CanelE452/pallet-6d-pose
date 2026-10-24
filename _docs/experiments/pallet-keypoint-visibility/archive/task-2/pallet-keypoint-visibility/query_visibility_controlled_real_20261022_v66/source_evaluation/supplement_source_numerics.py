"""Additive source-only CPU/GPU rounding and original strict-SAM tie audit.

Never edits frozen evaluator/gates/results or performs model inference.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p,keys):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in keys}
def module():
    p=ROOT/'contract/evaluate_partial_source.py';s=importlib.util.spec_from_file_location('frozen_eval_numerical_pure',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def raw_sam():
    out=ROOT/'source_evaluation/RAW_SAM_STRICT_ZERO_TIE_DIAGNOSTIC.json';assert not out.exists()
    cache=ROOT/'joined_source_cache/CONTROLLED_ONLY_VAL.npz'
    d=load(cache,['sam','source','dense_valid','dense_allowed','dense_exact_state','dense_radius2_state'])
    m=module();valid=d['dense_valid'][:,4,4];s=d['sam'][:,:,4,4].astype(float)
    gt=np.stack([d['dense_exact_state'][:,4,4],d['dense_radius2_state'][:,4,4]],1)
    result=[]
    for source in m.SOURCES:
        src=d['source'].astype(str)==source
        for k,name in enumerate(['exact','radius2']):
            ix=src&valid;known=ix&(gt[:,k]>=0);zero=ix&(s[:,k]==0)
            incl=s[:,k]>=0;strict=s[:,k]>0
            pred=np.where(strict,0,3);table=np.zeros((3,4),np.int64);np.add.at(table,(gt[known,k],pred[known]),1)
            result.append(dict(source=source,support=name,native_valid=int(ix.sum()),zero_normalized_logit=int(zero.sum()),zero_on_known_GT=int((zero&known).sum()),
                strict_vs_inclusive_decision_flips=int((ix&(incl!=strict)).sum()),
                strict_known_3x4_confusion=table.tolist(),GT_rows=['V','E','O'],prediction_cols=['V','E','O','U'],
                strict_binary_visible=m.binary_metrics(gt[known,k]==0,strict[known]),
                inclusive_binary_visible=m.binary_metrics(gt[known,k]==0,incl[known])))
    report=dict(PASS=True,schema='ORIGINAL_RAW_SAM_STRICT_ZERO_TIE_SOURCE_DIAGNOSTIC_V1',cache=ref(cache),code=ref(__file__),
        original_frozen_SAM_mask_rule='raw logit>0 (strict); zero is not positive',
        frozen_current_source_evaluator_rule='sigmoid(saved raw logit)>=.5 (inclusive); its already sealed report remains unchanged',
        actual_all_native_valid_tie_count=sum(r['zero_normalized_logit']for r in result),
        all_source_native_valid_decisions_identical_strict_vs_inclusive=not any(r['strict_vs_inclusive_decision_flips']for r in result),
        per_source=result,head_thresholds_or_weights_changed=False,source_selection_changed=False,
        model_inference_GPU_training_private_PnP_calls=0)
    out.write_text(json.dumps(report,indent=2)+'\n');print(out,sha(out),'ties',report['actual_all_native_valid_tie_count'],flush=True)

def cpu_gpu(name):
    out=ROOT/'source_evaluation'/name/'CPU_GPU_NUMERICAL_DIAGNOSTIC.json';assert not out.exists()
    cp=out.parent/'SOURCE_OUTPUTS.npz';rp=ROOT/'runs'/name/'VALIDATION_SOURCE_OUTPUTS.npz'
    c=load(cp,['joint_logits','joint_probability','source','sample_id','GT_state','center_native_valid'])
    r=load(rp,['joint_logits','sample_id']);lookup={str(s):i for i,s in enumerate(r['sample_id'])}
    assert len(lookup)==len(c['sample_id'])and set(lookup)==set(c['sample_id'].astype(str))
    rz=r['joint_logits'][[lookup[str(s)]for s in c['sample_id']]];m=module();p=c['joint_probability'];q=m.probabilities(rz)
    cstate,rstate=m.confident_states(p),m.confident_states(q);cm,rm=m.marginal(p),m.marginal(q)
    classflip=np.any(cstate!=rstate,axis=1);binflip=np.any((cm[:,:,0]>=.5)!=(rm[:,:,0]>=.5),axis=1)
    ids=np.flatnonzero(classflip|binflip);cases=[]
    for i in ids:
        cases.append(dict(sample_id=str(c['sample_id'][i]),source=str(c['source'][i]),native_valid=bool(c['center_native_valid'][i]),
            GT_exact_radius2=c['GT_state'][i].tolist(),CPU_joint_max=float(p[i].max()),GPU_joint_max=float(q[i].max()),
            CPU_joint_MAP=int(p[i].argmax()),GPU_joint_MAP=int(q[i].argmax()),CPU_confident_state=cstate[i].tolist(),GPU_confident_state=rstate[i].tolist(),
            CPU_pV=cm[i,:,0].tolist(),GPU_pV=rm[i,:,0].tolist()))
    diff=np.abs(c['joint_logits'].astype(float)-rz.astype(float));pd=np.abs(p-q)
    freeze=json.loads((ROOT/'runs'/name/'MODEL_FREEZE.json').read_text());metrics=json.loads((out.parent/'SOURCE_METRICS.json').read_text())
    CPU=metrics['fixed_selection_statistic']['common4domain_mean_informative_center_partial_NLL'];GPU=freeze['best_source_partial_NLL']
    report=dict(PASS=True,schema='SOURCE_CPU_GPU_NUMERICAL_DIAGNOSTIC_V1',code=ref(__file__),CPU_outputs=ref(cp),GPU_outputs=ref(rp),
        evaluator=ref(ROOT/'contract/evaluate_partial_source.py'),rows=len(p),
        logits_abs=dict(max=float(diff.max()),mean=float(diff.mean()),p99=float(np.quantile(diff,.99))),
        probability_abs=dict(max=float(pd.max()),mean=float(pd.mean()),p99=float(np.quantile(pd,.99))),
        confident_state_cell_flips=int(np.count_nonzero(cstate!=rstate)),confident_state_row_flips=int(classflip.sum()),
        binary_visible_cell_flips=int(np.count_nonzero((cm[:,:,0]>=.5)!=(rm[:,:,0]>=.5))),binary_visible_row_flips=int(binflip.sum()),
        CPU_common_partial_NLL=CPU,canonical_GPU_common_partial_NLL=GPU,CPU_minus_GPU_partial_NLL=CPU-GPU,flip_cases=cases,
        interpretation='Numerical runtime comparison only. CPU/GPU kernels can differ. Both fixed.5 thresholds remain unchanged, and canonical GPU source checkpoint/arm selection is preserved.',
        source_selection_changed=False,weights_or_thresholds_changed=False,model_inference_GPU_training_private_PnP_calls=0)
    out.write_text(json.dumps(report,indent=2)+'\n');print(out,sha(out),'maxlogit',diff.max(),'prob',pd.max(),'state',report['confident_state_cell_flips'],'binary',report['binary_visible_cell_flips'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-sam',action='store_true');p.add_argument('--model');a=p.parse_args()
    if a.raw_sam:raw_sam()
    if a.model:cpu_gpu(a.model)
