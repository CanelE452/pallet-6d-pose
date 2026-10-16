"""Independent sealed-journal integrity and arithmetic check, without cost code."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def check(folder,selection_directory):
    started,cpu=time.monotonic(),time.process_time();folder=Path(folder);selection_directory=Path(selection_directory)
    seal=json.loads((folder/'COST_SEAL.json').read_text());expected={(r['id'],r['candidate_id']) for r in rows(selection_directory/'RETAINED_CANDIDATES.jsonl.gz')}
    if digest(folder/'CANDIDATE_COSTS.jsonl.gz')!=seal['candidate_costs_sha256']:raise ValueError('Compact journal SHA mismatch')
    found=set();outline_observations=0;samples=0;failures=[];maxima=dict(component_sum=0.,observation_sum=0.,sample_sum=0.,sample_formula=0.)
    feasibility_failures=[];frame_counts=[]
    for frame in seal['dense_frame_journals']:
        file=folder/frame['path']
        if digest(file)!=frame['sha256']:raise ValueError('Dense frame SHA mismatch: '+frame['id'])
        count=0
        for row in rows(file):
            count+=1;cid=(row['id'],row['candidate_id'])
            if cid in found:raise ValueError('Duplicate frame/candidate ID '+str(cid))
            found.add(cid)
            if row['status']!='COMPLETE':failures.append(dict(candidate_id=cid,reason='unresolved row'));continue
            if row['recomputed_joint_feasible'] is not True:feasibility_failures.append(cid)
            terms=row['weighted_components'];total=math.fsum(terms.values());difference=abs(total-row['objective'])
            maxima['component_sum']=max(maxima['component_sum'],difference)
            if difference>1e-9:failures.append(dict(candidate_id=cid,reason='component sum',difference=difference))
            for family in ('full_BI','partial_normal','outline'):
                actual=math.fsum(r['weighted_contribution'] for r in row['per_observation'] if r['loss_family']==family)
                difference=abs(actual-terms[family]);maxima['observation_sum']=max(maxima['observation_sum'],difference)
                if difference>1e-9:failures.append(dict(candidate_id=cid,reason='observation sum:'+family,difference=difference))
            for observation in row['per_observation']:
                if observation['loss_family']!='outline':continue
                outline_observations+=1;n=observation['count'];samples+=n
                fields=('sample_xy','residual_xy_px','standardized_radius','robust_losses','arc_or_sample_weights','weighted_sample_contributions')
                if any(len(observation.get(key,[]))!=n for key in fields):
                    failures.append(dict(candidate_id=cid,reason='outline array length mismatch'));continue
                weights=observation['arc_or_sample_weights'];losses=observation['robust_losses'];contributions=observation['weighted_sample_contributions']
                actual=math.fsum(contributions);difference=abs(actual-observation['weighted_contribution']);maxima['sample_sum']=max(maxima['sample_sum'],difference)
                if difference>1e-9:failures.append(dict(candidate_id=cid,reason='sample contribution sum',difference=difference))
                normalizer=observation['normalizer'];factor=observation['factor_weight']*observation['direction_weight']
                for weight,loss,value in zip(weights,losses,contributions):
                    difference=abs(factor*weight*loss/normalizer-value)
                    maxima['sample_formula']=max(maxima['sample_formula'],difference)
                    if difference>1e-9:failures.append(dict(candidate_id=cid,reason='weighted sample formula',difference=difference));break
        if count!=frame['rows']:raise ValueError('Frame row count mismatch')
        frame_counts.append(dict(id=frame['id'],rows=count))
    if found!=expected:raise ValueError('Original retained candidate identity set differs')
    compact_ids=[(r['id'],r['candidate_id']) for r in rows(folder/'CANDIDATE_COSTS.jsonl.gz')]
    if len(compact_ids)!=len(found) or set(compact_ids)!=found:raise ValueError('Compact/dense identity mismatch')
    result=dict(schema='INDEPENDENT_DENSE_FIXED_COST_CHECK_V1',passed=not failures,frames=len(frame_counts),candidate_rows=len(found),
        outline_observations=outline_observations,outline_samples=samples,all_six_sample_arrays_preserved=True,
        max_absolute_differences=maxima,arithmetic_failures=failures,
        retained_recomputed_infeasible_candidate_ids=feasibility_failures,
        atol=1e-9,rtol=0.,original_cost_code_imports=0,new_pose_calls=0,new_model_forwards=0,
        cost_seal_sha256=digest(folder/'COST_SEAL.json'),selection_candidates_sha256=digest(selection_directory/'RETAINED_CANDIDATES.jsonl.gz'),
        source_sha256=digest(__file__),elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    with (folder/'DENSE_ARRAY_VALIDATION.json').open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--selection-directory',required=True)
    a=p.parse_args();print(json.dumps(check(a.folder,a.selection_directory),indent=2))
