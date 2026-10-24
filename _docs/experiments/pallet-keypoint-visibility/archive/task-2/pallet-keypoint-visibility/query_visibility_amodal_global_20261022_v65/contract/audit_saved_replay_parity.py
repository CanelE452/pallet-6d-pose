"""Compare saved GPU-fit VAL and frozen cross-source replay; no inference."""
import argparse,json
from pathlib import Path
import numpy as np
import saved_evaluate_frozen_source as base
from source_support_contract import confident_joint_map,hierarchical_states,joint_marginals


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-val',required=True,type=Path)
    ap.add_argument('--cross-source',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    with np.load(a.fit_val,allow_pickle=False)as z:fit={k:z[k]for k in z.files}
    with np.load(a.cross_source,allow_pickle=False)as z:cross={k:z[k]for k in z.files}
    lookup={s:i for i,s in enumerate(cross['sample_id'])}
    if len(lookup)!=len(cross['sample_id']):raise AssertionError('Duplicate cross-source query')
    ix=np.array([lookup[s]for s in fit['sample_id']])
    for key in ['truth_joint','truth_classes','source','sample_id','image_sha256']:
        if not np.array_equal(fit[key],cross[key][ix]):raise AssertionError('Saved replay label/identity mismatch: '+key)
    # target_index is local to the containing cache, so compare IDs/hashes above.
    p,q=base.softmax(fit['joint_logits']),base.softmax(cross['joint_logits'][ix])
    cm1,cm2=confident_joint_map(p),confident_joint_map(q)
    marginal1,marginal2=joint_marginals(p),joint_marginals(q)
    binary1,binary2=marginal1[:,:,0]>=.5,marginal2[:,:,0]>=.5
    logical1,logical2=hierarchical_states(p),hierarchical_states(q)
    changes=(cm1['joint']!=cm2['joint'])
    rows=[]
    for i in np.flatnonzero(changes|np.any(binary1!=binary2,axis=1)|np.any(logical1!=logical2,axis=1)):
        rows.append(dict(sample_id=str(fit['sample_id'][i]),fit_joint_or_U=int(cm1['joint'][i]),cross_joint_or_U=int(cm2['joint'][i]),
                         fit_maxjoint=float(p[i].max()),cross_maxjoint=float(q[i].max()),
                         fit_pV=marginal1[i,:,0].tolist(),cross_pV=marginal2[i,:,0].tolist()))
    result=dict(schema='V65_SAVED_CPU_GPU_REPLAY_NUMERIC_PARITY_V1',n=len(p),all_query_labels_and_identities_equal=True,
                max_absolute_logit_difference=float(np.abs(fit['joint_logits']-cross['joint_logits'][ix]).max()),
                max_absolute_joint_probability_difference=float(np.abs(p-q).max()),
                confident_joint_or_U_changes=int(changes.sum()),binary_visible_changes_per_scale=(binary1!=binary2).sum(0).tolist(),
                logical_support_changes_per_scale=(logical1!=logical2).sum(0).tolist(),
                minimum_fit_maxjoint_half_margin=float(np.abs(p.max(1)-.5).min()),
                changed_queries=rows,
                caveat='Numerical replay parity is reported without changing checkpoint, decoder or thresholds. Cross-source replay uses the same device procedure for all candidates.',
                no_model_RGB_mask_private_GPU_training_PnP_reads_or_calls=True,
                hashes={str(path):base.sha(path)for path in [Path(__file__),a.fit_val,a.cross_source]})
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
