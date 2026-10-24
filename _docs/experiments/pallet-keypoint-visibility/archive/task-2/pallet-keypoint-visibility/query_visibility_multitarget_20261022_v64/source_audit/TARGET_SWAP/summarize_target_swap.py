#!/usr/bin/env python3
"""Descriptive summaries from already saved public-only counterfactual outputs."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import target_identity_swap as audit

OUT=Path(__file__).resolve().parent

def confusion(y,p):
    matrix=np.zeros((3,4),int);np.add.at(matrix,(y,np.where(p<0,3,p)),1)
    rows=[]
    for c in range(3):
        actual=int(matrix[c].sum());pred=int(matrix[:,c].sum());tp=int(matrix[c,c]);decided=int(matrix[c,:3].sum())
        rows.append(dict(state=c,N=actual,precision=tp/pred if pred else None,recall_all_GT=tp/actual if actual else None,recall_decided_GT=tp/decided if decided else None))
    return dict(columns=['V','E','O','U'],confusion=matrix.tolist(),per_class=rows,U_rate=float((p<0).mean()))

labels=audit.loadnp(OUT/'PAIR_LABELS.npz');old=labels['classes_original'][:,:,4,4];new=labels['classes_other'][:,:,4,4]
summary=dict(protocol_sha256=audit.sha(OUT/'PROTOCOL.json'),source_only=True,private_data_read=False,new_model_inference=False,training=False,source_box_is_annotation_derived=True,
             interpretation=['CONTEXT_ONLY is intentionally inconsistent conditioning, not deployment accuracy.',
                             'FULL_CONDITION simultaneously changes SAM, box and context; these results do not isolate SAM versus box contributions.',
                             'Other-target exact truth is outside2791/3200. Changed-ownership subsets are primary; overall accuracy alone is misleading.',
                             'The per-E-only-slice precision field in raw REPORT.json is tautological because every slice member has E ground truth. Ignore it. Precision below uses all V/E/O GT rows.',
                             'Official YCB source masks remain rendered/depth proxy truth; no human physical visibility confirmation is introduced.'],
             exact_transition_matrix_rows_original_cols_other=np.bincount(old[:,0]*3+new[:,0],minlength=9).reshape(3,3).tolist(),
             radius2_transition_matrix_rows_original_cols_other=np.bincount(old[:,1]*3+new[:,1],minlength=9).reshape(3,3).tolist(),models={})
flat=[]
for name in ['MIXED','SYNTH']:
    model=f'{name}_MULTITARGET_UNWEIGHTED';directory=OUT/f'EVAL_{model}';p=audit.loadnp(directory/'OUTPUTS.npz');report=json.loads((directory/'REPORT.json').read_text())
    r=dict(checkpoint_sha256=report['checkpoint_sha256'],raw_report_sha256=audit.sha(directory/'REPORT.json'),arms={},paired_jointMAP_confidence={})
    for arm in ['ORIGINAL','CONTEXT_ONLY_SWAP','FULL_CONDITION_SWAP']:
        pp=p[arm];truth=old if arm=='ORIGINAL'else new;sp,v,f,hard=audit.marginals(pp);conf=pp.max(1)>=.5;mapstates=audit.PAIRS[pp.argmax(1)];with_u=np.where(conf[:,None],mapstates,-1)
        r['arms'][arm]={scale:{'logical':confusion(truth[:,s],hard[:,s]),'confidence_qualified_jointMAP':confusion(truth[:,s],with_u[:,s])}for s,scale in enumerate(['exact','radius2'])}
    for arm in ['CONTEXT_ONLY_SWAP','FULL_CONDITION_SWAP']:
        p0=p['ORIGINAL'];p1=p[arm];h0=audit.PAIRS[p0.argmax(1)];h1=audit.PAIRS[p1.argmax(1)];conf=(p0.max(1)>=.5)&(p1.max(1)>=.5)
        rr={}
        for s,scale in enumerate(['exact','radius2']):
            changed=old[:,s]!=new[:,s];both=(h0[:,s]==old[:,s])&(h1[:,s]==new[:,s]);qualified=both&conf
            rr[scale]=dict(changed_N=int(changed.sum()),paired_confident_changed_N=int((changed&conf).sum()),both_correct_jointMAP_changed=float(both[changed].mean()),both_correct_confident_jointMAP_all_changed_GT=float(qualified[changed].mean()),both_correct_confident_jointMAP_within_decided=float(both[changed&conf].mean())if(changed&conf).any()else None)
            raw=report['metrics'][arm]['scales'][scale]
            flat.append(dict(model=model,arm=arm,scale=scale,changed_N=int(changed.sum()),class_correct_direction=raw['class_correct_direction_rate'],both_classes_correct_logical=raw['both_targets_correct_changed'],visible_correct_direction=raw['visible']['correct_direction_rate'],both_visible_correct=raw['visible']['both_targets_binary_correct_changed'],both_jointMAP_correct=float(both[changed].mean()),both_confident_jointMAP_correct_all_changed_GT=float(qualified[changed].mean()),jointly_confident_coverage_on_changed=float(conf[changed].mean())))
        r['paired_jointMAP_confidence'][arm]=rr
    if name=='MIXED':
        # CPU versus saved GPU source-output numerical check, same fixed checkpoint.
        earlier=audit.loadnp(audit.ROOT/f'runs/{model}/VAL_SOURCE_OUTPUTS.npz',['source','joint_probability']);prior=earlier['joint_probability'][earlier['source']=='YCB'];now=p['ORIGINAL']
        assert prior.shape==now.shape
        r['CPU_vs_saved_GPU_source_replay']=dict(max_probability_abs_difference=float(np.max(np.abs(prior-now))),joint_MAP_mismatches=int((prior.argmax(1)!=now.argmax(1)).sum()),logical_state_mismatches=int((audit.marginals(prior)[3]!=audit.marginals(now)[3]).sum()),interpretation='CPU/GPU floating-point convolution difference; current paired arms all evaluated on same CPU path, no threshold adjustment')
    summary['models'][model]=r
destination=OUT/'FIXED_PROBE_SUMMARY.json';assert not destination.exists();destination.write_text(json.dumps(summary,indent=2)+'\n')
with(OUT/'FIXED_PROBE_METRICS.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps({'summary':str(destination),'models':{name:{'original_E_confident_metrics':r['arms']['ORIGINAL']['exact']['confidence_qualified_jointMAP']['per_class'][1],'fullswap_E_confident_metrics':r['arms']['FULL_CONDITION_SWAP']['exact']['confidence_qualified_jointMAP']['per_class'][1],**({'CPU_vs_saved_GPU_source_replay':r['CPU_vs_saved_GPU_source_replay']}if'CPU_vs_saved_GPU_source_replay'in r else{})}for name,r in summary['models'].items()}},indent=2))
