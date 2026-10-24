"""Fixed saved-only C1/C2selected0/C2final2000 comparison, no torch/producer."""
from pathlib import Path
import json
import hashlib
import math
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'independent_saved_C1_baseline_audit'))
from audit_baseline import scores,audit_identity,all_metrics,compare,read,lines,bind,check,write


def main():
    c1=read(V/'source_baselines/C1_FROZEN_ON_V95/BASELINE_COMPLETE.json')
    c2=read(V/'runs/C2_SAME_Q_COUNTERFACTUAL_SOURCE_V1/FIT_COMPLETE.json')
    final=read(V/'source_diagnosis/FINAL2000_FIXED_NOT_SELECTOR/DIAGNOSIS_COMPLETE.json')
    assert c1['completed']and c2['completed']and final['completed']
    c2scores=c2['scores'];finalscores=next(x for x in final.values()if isinstance(x,dict)and x.get('path','').endswith('ALL2880_FINAL_SOURCE_SCORES.npz'))
    files={'C1_FROZEN':c1['scores'],'C2_SELECTED0_AUTHORITATIVE':c2scores,'C2_FINAL2000_DIAGNOSTIC_ONLY':finalscores}
    for d in files.values():check(d)
    z={k:scores(d['path'])for k,d in files.items()}
    for x in z.values():audit_identity(x)
    keys=['occurrence_id','occurrence_index','base_row_index','source','split','original_split','variant','input_id','native_slot_logical_only',
          'image_sha256','original_image_sha256','query_xy_raw','L_raw','scalar_target_D0_E1_Uminus1','candidate_known','accepted_pair','query_available']
    first=z['C1_FROZEN']
    for name,x in z.items():
        for key in keys:
            equal=np.array_equal(first[key],x[key],equal_nan=True)if first[key].dtype.kind=='f'else np.array_equal(first[key],x[key])
            assert equal,(name,key)
    assert c2['selected_update']==0
    meta=read(c2['MODEL_FREEZE']['path']);assert meta['selected_update']==0
    variants=['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM'];truth=[0,1,0]
    feasibility=lines(V/'source_pair_intake/image_only_v2_exclusive_alpha/FEASIBILITY960_GT_ONLY.jsonl')
    rows=lines(V/'source_pair_intake/ALL960_NORMALIZED_INPUT_ONLY.jsonl')
    accepted=[i for i,x in enumerate(first['accepted_pair'])if x];assert accepted==[198,237,260,262,284,385,421,422,423,625,708,711,717,828,926]
    records=[]
    for i in accepted:
        row=rows[i];f=feasibility[i];q=np.asarray(row['query_xy'],np.float64);ranges=f['protected_regions_GT_only']
        dist=[];inside=[]
        for region in ranges:
            box=region.get('bbox');assert region['kind']=='PALLOX_CLOSED_NATIVE_UNCERTAINTY'
            dx=max(float(box[0])-q[0],0.,q[0]-float(box[2]));dy=max(float(box[1])-q[1],0.,q[1]-float(box[3]))
            dist.append(float(math.hypot(dx,dy)));inside.append(bool(box[0]<=q[0]<=box[2]and box[1]<=q[1]<=box[3]))
        models={}
        for name,x in z.items():
            ix=[i*3+j for j in range(3)];pred=[int(x['prediction'][j])for j in ix];logits=[float(x['query_logit'][j])for j in ix]
            models[name]=dict(before_cover_sham_logits=logits,before_cover_sham_predictions_D0_E1_Uminus1=pred,
                             before_cover_sham_correct=[p==y for p,y in zip(pred,truth)],all3_correct=pred==truth,
                             cover_minus_before_logit=logits[1]-logits[0],sham_minus_before_logit=logits[2]-logits[0])
        records.append(dict(base_row_index=i,input_id=row['input_id'],logical_slot=row['native_slot_logical_only'],original_split=row['split'],
                            original_q=row['query_xy'],original_L=row['L'],truth_before_cover_sham=truth,
                            protected_range_count=len(ranges),protected_original_ranges=ranges,
                            q_inside_any_protected_range=any(inside),q_inside_all_protected_ranges=all(inside),
                            q_distance_to_each_range_px=dist,q_min_distance_to_union_px=min(dist),q_max_distance_to_range_px=max(dist),
                            q_min_distance_to_union_over_L=min(dist)/row['L'],models=models))
    summary={}
    for name in z:
        summary[name]={}
        for split in ['TRAIN_LAB','VAL_HALL']:
            rr=[x for x in records if x['original_split']==split]
            summary[name][split]=dict(pairs=len(rr),all3_correct=sum(x['models'][name]['all3_correct']for x in rr),
                     before_correct=sum(x['models'][name]['before_cover_sham_correct'][0]for x in rr),
                     cover_correct=sum(x['models'][name]['before_cover_sham_correct'][1]for x in rr),
                     sham_correct=sum(x['models'][name]['before_cover_sham_correct'][2]for x in rr))
    assert summary['C1_FROZEN']['TRAIN_LAB']['all3_correct']==0 and summary['C1_FROZEN']['VAL_HALL']['all3_correct']==0
    assert summary['C2_SELECTED0_AUTHORITATIVE']['TRAIN_LAB']['all3_correct']==0 and summary['C2_SELECTED0_AUTHORITATIVE']['VAL_HALL']['all3_correct']==0
    assert summary['C2_FINAL2000_DIAGNOSTIC_ONLY']['TRAIN_LAB']['all3_correct']==9 and summary['C2_FINAL2000_DIAGNOSTIC_ONLY']['VAL_HALL']['all3_correct']==1
    output=dict(completed=True,PASS=True,comparison='ALL15 original fixedaccepted pairs; no replacement or model-score based selection',
       order=accepted,variant_order=variants,decision='z<0 D0; z>=0 auxiliary hiddenE1; unavailableU-1',records=records,summary=summary,
       authoritative_C2_selected_update=0,fixed_final2000_is_diagnostic_only=True,final_used_for_selection_or_threshold=False,
       conclusion='C1 missed all15 controlled covers. Selected C2 update0 is the authoritative zero-logit baseline, predicting E for available inputs, not a trained winner. Fixed final2000 fits all9 TRAIN triplets but only1/6 heldout VAL triplets. This is a narrow conditional-source training/validation gap, not proof of a single failure cause or native semantic performance.',
       geometry='Distances are diagnostic to all previously fixed protected ranges; minimum distance does not select or rebind a cue. Originalq/L/ranges unchanged.',
       bindings={'C1_complete':bind(V/'source_baselines/C1_FROZEN_ON_V95/BASELINE_COMPLETE.json'),
                 'C2_complete':bind(V/'runs/C2_SAME_Q_COUNTERFACTUAL_SOURCE_V1/FIT_COMPLETE.json'),
                 'FINAL2000_diagnosis_complete':bind(V/'source_diagnosis/FINAL2000_FIXED_NOT_SELECTOR/DIAGNOSIS_COMPLETE.json'),
                 'scores':files},native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False,
       model_forward_optimizer_current_private_PnP=0,goal_complete=False)
    write(HERE/'ALL15_PAIR_COMPARISON.json',output)
    write(HERE/'SEAL.json',dict(completed=True,PASS=True,comparison=bind(HERE/'ALL15_PAIR_COMPARISON.json'),code=bind(__file__),bindings=output['bindings'],new_model_fit_current_PnP=0))
    print(json.dumps(dict(comparison=bind(HERE/'ALL15_PAIR_COMPARISON.json'),summary=summary),indent=2))


if __name__=='__main__':main()
