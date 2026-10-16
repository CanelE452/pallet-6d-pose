"""Independent discrete-policy and exact-rational influence reaggregation.

Imports no repository selector, influence, cost, scorer, NumPy, or solver.
Already sealed fixed-candidate metric scalars are inputs, not re-fitted poses.
"""
import argparse
from collections import defaultdict,Counter
import csv
from decimal import Decimal
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

FAMILIES=('ORIGINAL_WD','SWAPPED_WD')
TRANSITIONS=('both_success','recovery','damage','both_failure')


def load(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def assert_pose_equal(first,second):
    def same(a,b):
        if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if a is None or b is None:return a is b
        return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9
    for key in ('R_cf','centroid','R_physical','physical_registration_Q'):
        if not same(first[key],second[key]):raise ValueError('Original fallback pose mismatch '+key)


def choose(pool,components,omitted=None):
    if not pool:return None,None
    def score(candidate):
        value=float(candidate['original_record']['objective'])
        if omitted is not None:value-=float(components[(candidate['id'],candidate['candidate_id'])][omitted])
        return value
    # A single tuple minimum reproduces the original stable comparator.
    # The original pre-sort outcome order supplies exact-tie stability.
    def key(candidate):
        value=score(candidate);order=candidate['original_outcome_order']
        return (value,order) if candidate['solver_schema']=='statewise_u_region_solver_v4_5' else (value,candidate['original_record']['initialization'],order)
    winner=min(pool,key=key)
    return winner,score(winner)


def independent_choice(frame,pools,components,path,omitted=None):
    family=frame['selected_family']
    if path=='G10_ONLY':
        eligible=[]
        for index,name in enumerate(FAMILIES):
            metadata=frame['stages']['G10']['families'][index]
            if metadata['accepted'] is not True:continue
            winner,score=choose(pools[(frame['id'],'G10',name)],components,omitted)
            if winner is None:raise ValueError('Accepted G10 family has no eligible retained candidate')
            eligible.append((score,index,name))
        family=min(eligible)[2] if eligible else FAMILIES[0]
        cid=frame['stages']['G13']['families'][FAMILIES.index(family)]['winner_candidate_id']
    elif path=='G13_INTERNAL_ONLY':
        winner,_=choose(pools[(frame['id'],'G13',family)],components,omitted)
        cid=None if winner is None else winner['candidate_id']
    else:raise ValueError('Unexpected stage influence path')
    return family,cid


def verify(root,parent,influence,output):
    started,cpu=time.monotonic(),time.process_time();root,parent,influence,output=map(Path,(root,parent,influence,output))
    protocol=load(root/'INFLUENCE_PROTOCOL.json');frames=list(rows(root/'selection_v1/FRAME_SELECTIONS.jsonl.gz'))
    records=list(rows(influence/'SCORE_INFLUENCE_ROWS.jsonl.gz'));summary=load(influence/'INFLUENCE_SUMMARY.json')['summary']
    pools=defaultdict(list);candidate_by_key={}
    for row in rows(root/'selection_v1/RETAINED_CANDIDATES.jsonl.gz'):
        pools[(row['id'],row['stage'],row['family'])].append(row);candidate_by_key[(row['id'],row['candidate_id'])]=row
    components={(r['id'],r['candidate_id']):r['weighted_components'] for r in rows(root/'cost_v1/CANDIDATE_COSTS.jsonl.gz')}
    metrics={(r['id'],r['candidate_id']):r['metric'] for r in rows(root/'metrics_v2/CANDIDATE_METRICS.jsonl.gz')}
    with (parent/'score_v2/ROW_METRICS.csv').open(newline='') as stream:
        baseline={r['id']:r for r in csv.DictReader(stream) if r['arm']=='E0'}
    if len(candidate_by_key)!=11603 or set(components)!=set(candidate_by_key) or set(metrics)!=set(candidate_by_key):
        raise ValueError('Exact original candidate/cost/metric identity required')
    by_frame={f['id']:f for f in frames};expected_variants=[p+'__without_'+k for p in protocol['paths'] for k in protocol['actual_additive_families']]
    saved={(r['id'],r['variant']):r for r in records}
    if len(records)!=1428 or len(saved)!=1428 or set(saved)!={(f['id'],v) for f in frames for v in expected_variants}:
        raise ValueError('Influence full 238 by six population differs')
    baseline_checks=[];checked=[]
    for frame in frames:
        for stage in ('G10','G13'):
            for index,family in enumerate(FAMILIES):
                metadata=frame['stages'][stage]['families'][index]
                winner,value=choose(pools[(frame['id'],stage,family)],components)
                expected=metadata['winner_candidate_id']
                actual=None if winner is None else winner['candidate_id']
                if actual!=expected:raise ValueError('Original stage/family winner mismatch')
                if expected is not None and value!=metadata['original_objective']:
                    if abs(value-metadata['original_objective'])>1e-9:raise ValueError('Original family scalar mismatch')
        for path in protocol['paths']:
            original_family,original_id=independent_choice(frame,pools,components,path)
            if (original_family,original_id)!=(frame['selected_family'],frame['selected_candidate_id']):
                raise ValueError('Original nonomitted e selection mismatch')
            baseline_checks.append(dict(id=frame['id'],path=path,exact_family_candidate_selection=True))
            for omitted in protocol['actual_additive_families']:
                family,cid=independent_choice(frame,pools,components,path,omitted)
                variant=path+'__without_'+omitted;row=saved[(frame['id'],variant)];base=baseline[frame['id']]
                if row['new_family']!=family or row['new_candidate_id']!=cid:raise ValueError('Single omission selection mismatch '+frame['id']+'/'+variant)
                if cid is None:
                    pose=frame['stages']['G13']['families'][FAMILIES.index(family)]['published_actual_pose']
                    assert_pose_equal(pose,frame['original_e_actual_pose'])
                    score=dict(translation_cm=float(base['translation_cm']),rotation_deg=float(base['rotation_deg']),success_le5=base['success_le5']=='True')
                else:score=metrics[(frame['id'],cid)]
                base_good=base['success_le5']=='True';new_good=score['success_le5'] is True
                transition=TRANSITIONS[0] if base_good and new_good else TRANSITIONS[2] if base_good else TRANSITIONS[1] if new_good else TRANSITIONS[3]
                expected=dict(base_family=frame['selected_family'],new_family=family,
                    base_candidate_id=frame['selected_candidate_id'],new_candidate_id=cid,
                    base_success=base_good,new_success=new_good,transition=transition,
                    family_changed=family!=frame['selected_family'],candidate_changed=cid!=frame['selected_candidate_id'],
                    base_translation_cm=float(base['translation_cm']),base_rotation_deg=float(base['rotation_deg']),
                    new_translation_cm=score['translation_cm'],new_rotation_deg=score['rotation_deg'])
                for key,value in expected.items():
                    if row[key]!=value:
                        if isinstance(value,(int,float)) and not isinstance(value,bool) and abs(row[key]-value)<=1e-9:continue
                        raise ValueError('Recorded endpoint/transition mismatch '+frame['id']+'/'+variant+'/'+key)
                if new_good!=(score['translation_cm']<=5 and score['rotation_deg']<=5):raise ValueError('Success is not same-candidate T/R joint event')
                checked.append(row)
    checks=[]
    for item in summary:
        cohort=item['cohort'];variant=item['variant']
        cohort_rows=[r for r in checked if r['variant']==variant and (r['partition']!='CAL' if cohort=='PRIMARY' else r['partition']==cohort)]
        transitions=Counter(r['transition'] for r in cohort_rows);base_success=sum(r['base_success'] for r in cohort_rows);success=sum(r['new_success'] for r in cohort_rows)
        values=dict(N=len(cohort_rows),base_success=base_success,new_success=success,
                    recovery=transitions['recovery'],damage=transitions['damage'],both_success=transitions['both_success'],both_failure=transitions['both_failure'])
        if any(item[key]!=value for key,value in values.items()):raise ValueError('Whole cohort binary reaggregation mismatch')
        if values['recovery']-values['damage']!=success-base_success:raise ValueError('Recovery damage identity failed')
        continuous={}
        for name in ('translation_cm','rotation_deg'):
            available=[r for r in cohort_rows if r['new_'+name] is not None and r['base_'+name] is not None]
            changes=[Fraction.from_float(r['new_'+name])-Fraction.from_float(r['base_'+name]) for r in available]
            total=sum(changes,Fraction());groups={group:sum((change for change,row in zip(changes,available) if row['transition']==group),Fraction()) for group in TRANSITIONS}
            direct=sum((Fraction.from_float(r['new_'+name]) for r in available),Fraction())-sum((Fraction.from_float(r['base_'+name]) for r in available),Fraction())
            if total!=sum(groups.values(),Fraction()) or total!=direct:raise ValueError('Exact rational continuous identity failed')
            source=item['paired_delta'][name]
            if Fraction(Decimal(source['sum_delta']))!=total:raise ValueError('Reported exact continuous sum differs')
            if any(Fraction(Decimal(source['group_sums'][group]))!=value for group,value in groups.items()):raise ValueError('Reported group exact sums differ')
            new_values=[r['new_'+name] for r in available];new_mean=float(sum(map(Fraction.from_float,new_values),Fraction())/len(new_values)) if new_values else None
            if item['continuous'][name]['mean']!=new_mean:raise ValueError('Recorded mean is not exact-rational rounded mean')
            if source['mean_delta']!=(float(total/len(available)) if available else None):raise ValueError('Recorded paired mean differs')
            continuous[name]=dict(exact_rational_group_sum_identity=True,exact_rational_new_minus_base_identity=True,
                                  recorded_sum_exact=True,recorded_means_exact=True,common_available=len(available))
        checks.append(dict(cohort=cohort,variant=variant,**values,continuous=continuous))
    if len(summary)!=24:raise ValueError('Four full cohorts by six variants required')
    sources=[root/'INFLUENCE_PROTOCOL.json',root/'selection_v1/SELECTION_SEAL.json',root/'selection_v1/RETAINED_CANDIDATES.jsonl.gz',
             root/'selection_v1/FRAME_SELECTIONS.jsonl.gz',root/'cost_v1/COST_SEAL.json',root/'cost_v1/CANDIDATE_COSTS.jsonl.gz',
             root/'metrics_v2/CANDIDATE_METRICS.jsonl.gz',influence/'INFLUENCE_SEAL.json',influence/'INFLUENCE_SUMMARY.json',influence/'SCORE_INFLUENCE_ROWS.jsonl.gz',
             parent/'score_v2/ROW_METRICS.csv',Path(__file__)]
    receipt=dict(schema='INDEPENDENT_SINGLE_TERM_INFLUENCE_CHECK_V1',passed=True,original_stage_family_winner_checks=238*4,
        original_non_omitted_e_policy_checks=len(baseline_checks),single_omission_selection_checks=len(checked),
        cohort_variant_summary_checks=len(checks),cohort_results=checks,candidate_pool_rows=11603,full_population_rows=1428,
        no_influence_selector_cost_scorer_import=True,ranking_implementation='independent tuple minimum; original pre-sort outcome order as stable exact tie',
        score_arithmetic='original stored double total minus one original weighted additive family',
        continuous_implementation='Fraction.from_float exact rational arithmetic; independent of source Decimal aggregation',
        new_pose_solves=0,new_model_forwards=0,new_cost_evaluations=0,new_candidates=0,
        source_sha256={str(path):sha(path) for path in sources},elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
    return {k:receipt[k] for k in ('passed','original_stage_family_winner_checks','original_non_omitted_e_policy_checks','single_omission_selection_checks','cohort_variant_summary_checks','elapsed_seconds','process_cpu_seconds')}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','parent','influence','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(verify(a.root,a.parent,a.influence,a.output),indent=2))
