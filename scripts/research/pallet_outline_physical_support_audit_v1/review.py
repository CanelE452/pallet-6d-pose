"""Join explicit model visual review with unchanged saved sample costs."""
from collections import Counter
from pathlib import Path
import argparse
import json
import math
import shutil
import time

import numpy as np

from scripts.research.pallet_selection_score_diagnosis_v1.common import read, write, write_csv, sha, csv_rows
from .bindings import load_cases
from .audit_guard import SupportAuditOnly

CATEGORIES=('PALLET_OUTER_SUPPORT','OTHER_OR_CONSTRUCTED_BOUNDARY','UNRESOLVED')


def run(source_root,score_private,state_private,root,output,decisions):
    start,cpu=time.monotonic(),time.process_time()
    root,output,source_root=Path(root),Path(output),Path(source_root)
    output.mkdir(parents=True,exist_ok=True)
    decision=read(decisions);byid={d['id']:d for d in decision['cases']}
    reviews=[];samples=[];summaries=[];segments=[];directions=[];assessments=[]
    maxima=dict(witness_dense=0.,sample_direction_sum=0.,category_sum=0.,outline_sum=0.,component_delta=0.)
    with SupportAuditOnly() as guard:
        cases=load_cases(source_root,score_private,state_private,include_dense=True)
        assert [c['id'] for c in cases]==[d['id'] for d in decision['cases']]
        for case in cases:
            ident=case['id'];review=byid[ident]
            lineage=read(root/'mask_lineage_v2'/(ident.replace(':','__')+'.json'))
            parents=lineage['used_segments'];n=len(lineage['observed_boundary_samples_xy'])
            assert set(review['category_by_segment_id'])=={str(p['segment_id']) for p in parents}
            obs=[]
            for role,dense in zip(('selected','comparison'),case['dense']):
                oo={o['direction']:o for o in dense['per_observation'] if o['loss_family']=='outline'}
                assert set(oo)=={'observed_to_model','model_to_observed'}
                for direction,o in oo.items():
                    error=abs(math.fsum(o['weighted_sample_contributions'])-o['weighted_contribution'])
                    maxima['sample_direction_sum']=max(maxima['sample_direction_sum'],error);assert error<=1e-9
                    directions.append(dict(id=ident,ordinal=case['ordinal'],candidate_role=role,candidate_id=dense['candidate_id'],
                        stage=dense['stage'],family=dense['family'],direction=direction,sample_count=o['count'],
                        original_normalizer=o['normalizer'],original_normalizer_type=o['normalizer_type'],
                        weighted_contribution=o['weighted_contribution'],factor_weight_already_included=o['factor_weight'],
                        direction_half_already_included=o['direction_weight'],raw_residual_mean_px=o['residual_mean_px'],
                        support_category_for_model_samples='NOT_ASSIGNED_MODEL_SAMPLE_INDICES_ARE_POSE_DEPENDENT' if direction=='model_to_observed' else 'SEE_SHARED_OBSERVED_SAMPLE_REVIEW',
                        sample_indices_paired_across_candidates=direction=='observed_to_model'))
                error=abs(sum(o['weighted_contribution'] for o in oo.values())-dense['weighted_components']['outline'])
                maxima['outline_sum']=max(maxima['outline_sum'],error);assert error<=1e-9
                obs.append(oo['observed_to_model'])
            assert np.array_equal(obs[0]['sample_xy'],obs[1]['sample_xy'])
            assert np.array_equal(obs[0]['sample_xy'],lineage['observed_boundary_samples_xy'])
            old_csv=source_root/'_docs/experiments/pallet_selection_score_diagnosis_v1/gallery/outline_witness'/f'witness_{case["ordinal"]:02d}_samples.csv.gz'
            old=list(csv_rows(old_csv));assert len(old)==n
            xy=np.asarray(obs[0]['sample_xy']);a=np.asarray(obs[0]['weighted_sample_contributions']);b=np.asarray(obs[1]['weighted_sample_contributions']);delta=b-a
            for i,row in enumerate(old):
                assert int(row['sample_index'])==i and float(row['x_px'])==xy[i,0] and float(row['y_px'])==xy[i,1]
                errors=[abs(float(row[k])-value) for k,value in [('selected_contribution',a[i]),('comparison_contribution',b[i]),('Delta',delta[i])]]
                maxima['witness_dense']=max(maxima['witness_dense'],*errors);assert max(errors)<=1e-9
            assigned={};category_indices={c:[] for c in CATEGORIES};case_reviews=[]
            for p in parents:
                sid=p['segment_id'];category=review['category_by_segment_id'][str(sid)];assert category in CATEGORIES
                ix=p['sample_indices']
                assert ix and not(set(ix)&set(assigned))
                for i in ix:assigned[i]=(sid,category)
                category_indices[category].extend(ix)
                evidence=review['support_evidence'] if category==CATEGORIES[0] else review.get('other_evidence','') if category==CATEGORIES[1] else review.get('unknown_evidence','')
                r=dict(id=ident,ordinal=case['ordinal'],segment_id=sid,original_hull_edge_index=p['original_hull_edge_index'],
                    segment_xy_original_px=p['xy'],full_parent_xy_original_px=p['full_parent_xy'],sample_indices=ix,
                    sample_count=len(ix),support_category=category,reviewed=True,
                    observation_subtype='ORIGINAL_CLIPPED_CONVEX_ENVELOPE_SEGMENT',
                    physical_edge_hypothesis='Visible pallet outer envelope; exact physical corner owner unverified' if category==CATEGORIES[0] else 'Non-pallet cone support or constructed connection, as visually described; not every pixel classified' if category==CATEGORIES[1] else 'UNRESOLVED_MATERIAL_VS_SHADOW_OR_CONTACT',
                    evidence=evidence,counterargument='Convex construction itself does not imply an invalid cuboid envelope. Parent label is not per-pixel ground truth.',
                    image_sha256=case['image_sha256'],mask_file_sha256=case['mask_file_sha256'],
                    reviewer_kind='model',review_time_UTC=review['reviewed_at_UTC'],
                    visual_evidence_status='MODEL_VISUAL_INTERPRETATION_NOT_HUMAN_VALIDATED',
                    geometry_generation_status='CONFIRMED_BY_CODE_EXACT_MASK_TO_HULL_PARITY',
                    A_mask_support=review['case_A'],B_cuboid_correspondence=review['case_B'],C_registration_reference=review['case_C'],
                    unresolved_question=review['question'],diagnostic_only_not_GT_or_training_target=True)
                reviews.append(r);case_reviews.append(r)
                segments.append(dict(id=ident,ordinal=case['ordinal'],segment_id=sid,support_category=category,sample_count=len(ix),
                    selected_contribution=math.fsum(a[ix]),comparison_contribution=math.fsum(b[ix]),
                    signed_Delta=math.fsum(delta[ix]),absolute_sample_Delta_sum=math.fsum(abs(delta[ix])),
                    coordinate_xy_original_px=p['xy'],sample_indices=ix,reviewer_kind='model'))
            assert sorted(assigned)==list(range(n))
            for i in range(n):
                sid,category=assigned[i]
                samples.append(dict(id=ident,ordinal=case['ordinal'],sample_index=i,x_px=xy[i,0],y_px=xy[i,1],
                    segment_id=sid,support_category=category,reviewed=True,selected_contribution=a[i],comparison_contribution=b[i],Delta=delta[i]))
            for category in CATEGORIES:
                ix=category_indices[category]
                summaries.append(dict(id=ident,ordinal=case['ordinal'],original_group=case['group'],support_category=category,
                    used_parent_segment_count=sum(r['support_category']==category for r in case_reviews),
                    observed_sample_count=len(ix),reviewed_observed_sample_count=len(ix),unreviewed_observed_sample_count=0,
                    denominator_observed_samples=n,denominator_used_parent_segments=len(parents),
                    selected_contribution=math.fsum(a[ix]),comparison_contribution=math.fsum(b[ix]),
                    signed_Delta=math.fsum(delta[ix]),absolute_sample_Delta_sum=math.fsum(abs(delta[ix])),
                    meaning_of_positive_Delta='FAVORS_REFERENCE_FAILURE_SELECTED' if case['group']!='FINAL_SUCCESS_CONTROL' else 'PROTECTS_REFERENCE_SUCCESS_SELECTED',
                    diagnostic_parent_categories_not_pixel_accuracy=True))
            expected=obs[1]['weighted_contribution']-obs[0]['weighted_contribution']
            actual=math.fsum(s['signed_Delta'] for s in summaries if s['id']==ident)
            error=abs(actual-expected);maxima['category_sum']=max(maxima['category_sum'],error);assert error<=1e-9
            dd={k:case['dense'][1]['weighted_components'][k]-case['dense'][0]['weighted_components'][k] for k in ('full_BI','partial_normal','outline','prior')}
            error=abs(math.fsum(dd.values())-(case['dense'][1]['objective']-case['dense'][0]['objective']))
            maxima['component_delta']=max(maxima['component_delta'],error);assert error<=1e-9
            assessments.append(dict(id=ident,ordinal=case['ordinal'],group=case['group'],A=review['case_A'],B=review['case_B'],C=review['case_C'],
                question=review['question'],reviewer_kind='model',counts=dict(Counter(r['support_category'] for r in case_reviews)),
                total_parent_segments=len(parents),total_observed_samples=n,unreviewed_samples=0,
                unresolvable_samples=len(category_indices['UNRESOLVED']),Delta_observed_to_model=expected,
                Delta_model_to_observed=dd['outline']-expected,Delta_outline=dd['outline'],Delta_components=dd,
                selected_candidate_id=case['panels'][0]['candidate_id'],comparison_candidate_id=case['panels'][1]['candidate_id']))
    assert len(reviews)==127 and len(samples)==3030
    with (output/'SUPPORT_REVIEW.jsonl').open('x',encoding='utf-8') as stream:
        for row in reviews:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    write_csv(output/'SUPPORT_REVIEW.csv',reviews)
    write_csv(output/'COST_SUPPORT_SUMMARY.csv',summaries)
    write_csv(output/'SEGMENT_COSTS.csv',segments)
    write_csv(output/'OBSERVED_SAMPLE_REVIEW.csv.gz',samples)
    write_csv(output/'MODEL_DIRECTION_COSTS.csv',directions)
    write(output/'CASE_ASSESSMENTS.json',dict(cases=assessments,new_performance_computed=False,
        reference_role='unchanged geometric reconstruction; not physical ground truth',scope_six_only=True,
        no_observation_deleted_or_reweighted=True,no_new_ranking_or_selection=True))
    shutil.copyfile(decisions,output/'SUPPORT_DECISIONS.json')
    receipt=dict(passed=True,original_order_preserved=True,selected_fixed_candidates=12,
        parent_segments=127,shared_observed_samples=3030,reviewed_parent_segments=127,
        unreviewed_observed_samples=0,unresolved_parent_segments=sum(r['support_category']=='UNRESOLVED' for r in reviews),
        category_sum_preserves_whole_denominator=True,model_samples_never_paired_across_poses=True,
        maximum_absolute_differences=maxima,atol=1e-9,rtol=0,reviewer_kind='model',
        initial_review_input_sha256=sha(decisions),runtime_guard=guard.receipt(),
        new_cost_evaluations=0,new_NN_PnP_optimizer_candidate_calls=0,new_performance_computed=False,
        elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    write(root/'REVIEW_AGGREGATION_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-root','score-private','state-private','root','output','decisions'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(run(a.source_root,a.score_private,a.state_private,a.root,a.output,a.decisions))
