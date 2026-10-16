"""Independent stored-cost/support-ledger arithmetic, without review imports.

Model visual decisions are inputs, not verified semantic ground truth.  No
cost/pose evaluator, mask transform, detector or neural model is imported.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

CATEGORIES=('PALLET_OUTER_SUPPORT','OTHER_OR_CONSTRUCTED_BOUNDARY','UNRESOLVED')
ATOL=1e-9


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    path=Path(path)
    with (gzip.open(path,'rt',newline='') if path.suffix=='.gz' else path.open(newline='')) as stream:
        return list(csv.DictReader(stream))


def assert_close(left,right,description,maxima):
    difference=abs(float(left)-float(right));maxima[description]=max(maxima[description],difference)
    if difference>ATOL:raise ValueError(description+' mismatch '+str(difference))


def verify(source_root,private,score_private,public,output):
    started,cpu=time.monotonic(),time.process_time()
    source_root,private,score_private,public,output=map(Path,(source_root,private,score_private,public,output))
    old=source_root/'_docs/experiments/pallet_selection_score_diagnosis_v1'
    raw=private/'MASK_FIRST_SUPPORT_DECISIONS.json';decision=read(raw)
    if sha(raw)!=sha(public/'SUPPORT_DECISIONS.json'):raise ValueError('Raw/public model decision bytes differ')
    if decision['independent_blind_review'] is not False or decision['previous_cases_and_costs_already_known_from_prior_task'] is not True:
        raise ValueError('Prior familiarity/nonblind limitation must be preserved')
    protocol=read(private/'REVIEW_PROTOCOL.json');initial=read(private/'RAW_RGB_INITIAL_REVIEW.json')
    if initial['reviewer_kind']!='model' or initial['independent_blind_review'] is not False:raise ValueError('Initial review provenance changed')
    ids=protocol['six_ids_in_original_order'];dc={d['id']:d for d in decision['cases']}
    if ids!=[d['id'] for d in decision['cases']]:raise ValueError('Six case order differs')
    chronology=[]
    for d in decision['cases']:
        at=datetime.fromisoformat(d['reviewed_at_UTC'])
        if at.tzinfo is None or not protocol['recorded_epoch']<=initial['review_time_epoch']<=at.timestamp():
            raise ValueError('Review log timestamp order differs')
        if d['reviewer_kind']!='model':raise ValueError('Model review relabeled as human')
        chronology.append(dict(id=d['id'],protocol_epoch=protocol['recorded_epoch'],raw_review_epoch=initial['review_time_epoch'],mask_first_review_UTC=d['reviewed_at_UTC'],timestamps_ordered=True))
    with (public/'SUPPORT_REVIEW.jsonl').open() as stream:reviews=[json.loads(line) for line in stream if line.strip()]
    review_by_key={(r['id'],r['segment_id']):r for r in reviews}
    if len(reviews)!=127 or len(review_by_key)!=127:raise ValueError('127 distinct original parents required')
    csv_review=csv_rows(public/'SUPPORT_REVIEW.csv')
    if len(csv_review)!=127:raise ValueError('CSV review denominator differs')
    for line,row in zip(csv_review,reviews):
        for key,value in row.items():
            expected=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')) if isinstance(value,(dict,list)) else str(value)
            if line[key]!=expected:raise ValueError('JSONL/CSV review mismatch '+key)
    sample_rows=csv_rows(public/'OBSERVED_SAMPLE_REVIEW.csv.gz');sample_by_key={(r['id'],int(r['sample_index'])):r for r in sample_rows}
    segment_rows=csv_rows(public/'SEGMENT_COSTS.csv');segments={(r['id'],int(r['segment_id'])):r for r in segment_rows}
    summaries=csv_rows(public/'COST_SUPPORT_SUMMARY.csv');summary_by_key={(r['id'],r['support_category']):r for r in summaries}
    directions=csv_rows(public/'MODEL_DIRECTION_COSTS.csv');direction_by_key={(r['id'],r['candidate_role'],r['direction']):r for r in directions}
    assessments={r['id']:r for r in read(public/'CASE_ASSESSMENTS.json')['cases']}
    if len(sample_rows)!=3030 or len(sample_by_key)!=3030 or len(segments)!=127 or len(summary_by_key)!=18 or len(direction_by_key)!=24:
        raise ValueError('A sample/parent/category/direction denominator or identity differs')
    witness=read(old/'gallery/outline_witness/WITNESS_RECEIPT.json')['examples'];maxima=Counter();category_counts=Counter();unknown_samples=0;case_results=[]
    dense_seal=read(score_private/'cost_v1/COST_SEAL.json');dense_manifest={r['path']:r for r in dense_seal['dense_frame_journals']}
    for number,(ident,wr) in enumerate(zip(ids,witness),1):
        if wr['id']!=ident:raise ValueError('Original witness order changed')
        lineage=read(private/'mask_lineage_v2'/(ident.replace(':','__')+'.json'));parents=lineage['used_segments'];labels=dc[ident]['category_by_segment_id']
        if set(labels)!={str(p['segment_id']) for p in parents}:raise ValueError('Missing or additional parent classification')
        samples=csv_rows(old/f'gallery/outline_witness/witness_{number:02d}_samples.csv.gz');mapping={}
        for parent in parents:
            sid=parent['segment_id'];category=labels[str(sid)]
            if category not in CATEGORIES:raise ValueError('Unknown category')
            category_counts[category]+=1
            for index in parent['sample_indices']:
                if index in mapping:raise ValueError('Overlapping parent sample indices')
                mapping[index]=(sid,category)
            review=review_by_key[(ident,sid)]
            required=dict(sample_indices=parent['sample_indices'],sample_count=len(parent['sample_indices']),support_category=category,
                reviewer_kind='model',review_time_UTC=dc[ident]['reviewed_at_UTC'],reviewed=True,
                segment_xy_original_px=parent['xy'],full_parent_xy_original_px=parent['full_parent_xy'],
                visual_evidence_status='MODEL_VISUAL_INTERPRETATION_NOT_HUMAN_VALIDATED',diagnostic_only_not_GT_or_training_target=True)
            if any(review[k]!=value for k,value in required.items()):raise ValueError('Parent review does not preserve model/input provenance')
        if sorted(mapping)!=list(range(len(samples))):raise ValueError('Unreviewed/missing sample removed from denominator')
        for index,row in enumerate(samples):
            actual=sample_by_key[(ident,index)];sid,category=mapping[index]
            if int(actual['segment_id'])!=sid or actual['support_category']!=category or actual['reviewed']!='True':raise ValueError('Sample category mapping differs')
            for key in ('x_px','y_px','selected_contribution','comparison_contribution','Delta'):
                assert_close(actual[key],row[key],'original_sample_values',maxima)
        for sid,parent in enumerate(parents):
            row=segments[(ident,sid)];ix=parent['sample_indices']
            if int(row['sample_count'])!=len(ix) or json.loads(row['sample_indices'])!=ix or row['support_category']!=labels[str(sid)]:raise ValueError('Segment sample membership differs')
            for key,field,absolute in [('selected_contribution','selected_contribution',False),('comparison_contribution','comparison_contribution',False),('signed_Delta','Delta',False),('absolute_sample_Delta_sum','Delta',True)]:
                value=math.fsum(abs(float(samples[i][field])) if absolute else float(samples[i][field]) for i in ix)
                assert_close(row[key],value,'parent_'+key,maxima)
        for category in CATEGORIES:
            indices=[i for i,(_,label) in mapping.items() if label==category];row=summary_by_key[(ident,category)]
            expected=dict(used_parent_segment_count=sum(label==category for label in labels.values()),observed_sample_count=len(indices),reviewed_observed_sample_count=len(indices),unreviewed_observed_sample_count=0,denominator_observed_samples=len(samples),denominator_used_parent_segments=len(parents))
            if any(int(row[k])!=value for k,value in expected.items()):raise ValueError('Category zero/unknown/whole denominator differs')
            for key,field,absolute in [('selected_contribution','selected_contribution',False),('comparison_contribution','comparison_contribution',False),('signed_Delta','Delta',False),('absolute_sample_Delta_sum','Delta',True)]:
                value=math.fsum(abs(float(samples[i][field])) if absolute else float(samples[i][field]) for i in indices)
                assert_close(row[key],value,'category_'+key,maxima)
            if category=='UNRESOLVED':unknown_samples+=len(indices)
        dense_rows=[]
        for role,pointer in zip(('selected','comparison'),wr['original_dense_pointers']):
            path=score_private/'cost_v1'/pointer['path']
            if sha(path)!=dense_manifest[pointer['path']]['sha256']:raise ValueError('Dense selected frame SHA differs')
            with gzip.open(path,'rt') as stream:line=next(line for i,line in enumerate(stream,1) if i==pointer['line_number'])
            dense=json.loads(line);dense_rows.append(dense)
            if (dense['id'],dense['candidate_id'])!=(ident,pointer['candidate_id']):raise ValueError('Original candidate pointer differs')
            for observation in dense['per_observation']:
                if observation['loss_family']!='outline':continue
                direction=observation['direction'];row=direction_by_key[(ident,role,direction)]
                for key,value in [('candidate_id',dense['candidate_id']),('stage',dense['stage']),('family',dense['family'])]:
                    if row[key]!=value:raise ValueError('Candidate stage/family direction changed')
                if int(row['sample_count'])!=observation['count']:raise ValueError('Own model/observation sample count differs')
                assert_close(row['original_normalizer'],observation['normalizer'],'direction_normalizer',maxima)
                assert_close(row['weighted_contribution'],observation['weighted_contribution'],'direction_weighted_value',maxima)
                assert_close(math.fsum(observation['weighted_sample_contributions']),row['weighted_contribution'],'saved_direction_sample_sum',maxima)
                if row['sample_indices_paired_across_candidates']!=str(direction=='observed_to_model'):raise ValueError('Model sample indices incorrectly paired')
                if direction=='observed_to_model':
                    expected=[[float(r['x_px']),float(r['y_px'])] for r in samples]
                    if observation['sample_xy']!=expected:raise ValueError('Original shared observed sample indices changed')
        expected_obs=math.fsum(float(r['Delta']) for r in samples)
        assert_close(expected_obs,wr['Delta_observed_to_model'],'original_witness_direction_sum',maxima)
        assert_close(math.fsum(float(summary_by_key[(ident,k)]['signed_Delta']) for k in CATEGORIES),expected_obs,'category_signed_total',maxima)
        assert_close(math.fsum(float(summary_by_key[(ident,k)]['absolute_sample_Delta_sum']) for k in CATEGORIES),math.fsum(abs(float(r['Delta'])) for r in samples),'category_absolute_total',maxima)
        case=assessments[ident]
        if case['reviewer_kind']!='model' or case['total_observed_samples']!=len(samples) or case['unreviewed_samples']!=0 or case['unresolvable_samples']!=sum(label=='UNRESOLVED' for _,label in mapping.values()):raise ValueError('Case assessment denominator/provenance differs')
        if Counter(case['counts'])!=Counter(labels.values()):raise ValueError('Assessment category parent counts differ')
        assert_close(case['Delta_observed_to_model'],expected_obs,'assessment_observed_direction',maxima)
        for name in ('full_BI','partial_normal','outline','prior'):
            expected=dense_rows[1]['weighted_components'][name]-dense_rows[0]['weighted_components'][name]
            assert_close(case['Delta_components'][name],expected,'assessment_original_component',maxima)
        case_results.append(dict(id=ident,parents=len(parents),observed_samples=len(samples),category_parent_counts=dict(Counter(labels.values())),unresolved_samples=case['unresolvable_samples'],signed_Delta=expected_obs))
    if category_counts['UNRESOLVED']!=25 or unknown_samples!=686:raise ValueError('Original unresolved parents/samples were lost')
    source_code=(source_root/'scripts/research/pallet_outline_physical_support_audit_v1/review.py').read_text()
    source_flags=all(text in source_code for text in ("reviewer_kind='model'","diagnostic_only_not_GT_or_training_target=True","model_samples_never_paired_across_poses=True"))
    if not source_flags:raise ValueError('Model-review-only/source direction protections absent')
    result=dict(schema='INDEPENDENT_PHYSICAL_SUPPORT_REAGGREGATION_V1',passed=True,parent_rows=127,observed_rows=3030,
        category_rows=18,fixed_candidates=12,direction_rows=24,category_parent_counts=dict(category_counts),unresolved_samples=unknown_samples,
        duplicate_or_missing_sample_rows=0,zero_count_categories_preserved=True,all_original_sample_values_preserved=True,
        max_absolute_differences=dict(maxima),cases=case_results,model_visual_input_public_bytes_exact=True,
        model_visual_labels_verified_as_physical_GT=False,reviewer_kind='model',independent_blind_review=False,chronology=chronology,
        source_model_only_flags_present=source_flags,root_review_module_imports=0,new_cost_evaluations=0,new_pose_solves=0,
        input_sha256={str(p):sha(p) for p in (raw,public/'SUPPORT_DECISIONS.json',public/'SUPPORT_REVIEW.jsonl',public/'SUPPORT_REVIEW.csv',
            public/'COST_SUPPORT_SUMMARY.csv',public/'SEGMENT_COSTS.csv',public/'OBSERVED_SAMPLE_REVIEW.csv.gz',public/'MODEL_DIRECTION_COSTS.csv',public/'CASE_ASSESSMENTS.json',Path(__file__))},
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('source-root','private','score-private','public','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();value=verify(args.source_root,args.private,args.score_private,args.public,args.output)
    print({k:value[k] for k in ('passed','parent_rows','observed_rows','unresolved_samples','elapsed_seconds','process_cpu_seconds')})
