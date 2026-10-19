"""Predeclared paired contrasts on complete, independently verified scores."""
from pathlib import Path
import argparse
import time
from . import s1_union as U, independent_check as I
from .evaluation import POPULATIONS, EXPECTED
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import paired as historical_paired


def default_plan():
    pairs=[('S2_C0','S2_CP','covariance_floor'),('S2_CP','S2_CB','role_calibration'),
           ('S1_UNION','S3_COMMON','common_material_proposals'),('S1_UNION','S3_BI','role_material_proposals'),
           ('S3_COMMON','S3_BI','S3_role_policy'),('S1_UNION','S4_PL','PL_proposals'),
           ('S4_PL','S4_CONTOUR','contour_minus_PL'),('S4_CONTOUR','S4_IMAGE','image_minus_contour'),
           ('S4_PL','S4_IMAGE','image_minus_PL'),('F_COMMON','F_BI','final_matched_BI_effect')]
    return dict(schema='PREDECLARED_WITHIN_CAMPAIGN_PAIRED_CONTRASTS_V1',
        contrasts=[dict(name=label+'_s'+str(seed),before=before+'_s'+str(seed),after=after+'_s'+str(seed))
                   for before,after,label in pairs for seed in (20261018,20261019,20261020)],
        primary_use='reused development results, no independent test claim; no winner/threshold/seed selection by this analysis',
        populations=EXPECTED,bootstrap_resamples=10000,recording_clusters_PRIMARY=6,
        reporting='all paired recovery/damage IDs, all4populations, recording/session bootstrap and leave-one-cluster sensitivity; no tail trimming')


def run(manifest,plan,output):
    manifest,plan,output=map(Path,(manifest,plan,output));start=time.monotonic()
    inputs=U.read(manifest);contrast_plan=U.read(plan);contrasts=contrast_plan['contrasts'];merged={};bindings=[];seen_arms=set();reference_sha=None
    if contrast_plan.get('bootstrap_resamples',10000)!=10000:raise ValueError('Original10000-draw paired bootstrap required')
    for source in inputs['sources']:
        ev=Path(source['evaluation_dir']);ind=Path(source['independent_dir'])
        verification=U.read(ind/'INDEPENDENT_VERIFICATION.json')
        if not verification['PASS']:raise ValueError('Independent verification must pass')
        required={'INDEPENDENT_ROWS.jsonl.gz','INDEPENDENT_VERIFICATION.json','INDEPENDENT_SUMMARY.json','INDEPENDENT_PAIRED.json'}
        if not required.issubset(source['independent_artifacts']):raise ValueError('Bind the complete independent verification bundle')
        for name,expected in source['independent_artifacts'].items():
            if U.sha(ind/name)!=expected:raise ValueError('Independently verified score output changed')
        es=U.read(ev/'EVALUATION_SEAL.json')
        if reference_sha is None:reference_sha=es['reference_sha256']
        elif reference_sha!=es['reference_sha256']:raise ValueError('Paired evaluations used different canonical references')
        if U.sha(ev/'EVALUATION_SEAL.json')!=verification['canonical_evaluation_seal_sha256']:
            raise ValueError('Independent/canonical evaluation ancestry differs')
        if U.sha(ev/'SCORED_ROWS.jsonl.gz')!=es['artifacts']['SCORED_ROWS.jsonl.gz']:
            raise ValueError('Canonical scored rows changed')
        for binding in es['prediction_bindings']:
            if U.sha(binding['path'])!=binding['sha256']or U.sha(Path(binding['path']).parent/'PREDICTION_SEAL.json')!=binding['seal_sha256']:
                raise ValueError('Original sealed predictions changed')
        rows=list(U.rows(ind/'INDEPENDENT_ROWS.jsonl.gz'))
        if len(rows)!=238 or len({r['id']for r in rows})!=238:raise ValueError('Whole real238 required')
        seen_arms.update(es['arms'])
        for row in rows:
            if row['id']not in merged:merged[row['id']]=row
            else:
                old=merged[row['id']]
                if any(old[k]!=row[k]for k in ('partition','session','cluster_id')):raise ValueError('Metadata population mismatch')
                for field in ('metrics','status','poses'):
                    for arm,value in row[field].items():
                        if arm in old[field]and old[field][arm]!=value:raise ValueError('Duplicated baseline independent outputs disagree')
                        old[field][arm]=value
        bindings.append(dict(evaluation_seal_sha256=U.sha(ev/'EVALUATION_SEAL.json'),independent_artifacts=source['independent_artifacts']))
    if len(merged)!=238:raise ValueError('Canonical global population differs')
    for pair in contrasts:
        if pair['before']not in seen_arms or pair['after']not in seen_arms:raise ValueError('Pending contrast arm cannot be omitted or zero filled')
    result={};canonical={};errors=[];maxima={}
    for pop,parts in POPULATIONS.items():
        rows=[r for r in merged.values()if r['partition']in parts]
        if len(rows)!=EXPECTED[pop]:raise ValueError('Full denominator lost')
        if pop=='PRIMARY192'and len({r['cluster_id']for r in rows})!=6:raise ValueError('Original six recording clusters required')
        result[pop]={};canonical[pop]={}
        for pair in contrasts:
            before,after=pair['before'],pair['after']
            result[pop][pair['name']]=I.paired(rows,before,after)
            canonical[pop][pair['name']]=historical_paired(rows,before,after)
            ee,mm=I.compare(canonical[pop][pair['name']],result[pop][pair['name']],pop+'/'+pair['name'])
            errors+=ee;maxima.update(mm)
    if errors:raise AssertionError(errors[:10])
    output.mkdir(parents=True,exist_ok=True);U.write(output/'INDEPENDENT_PAIRED_CONTRASTS.json',result)
    U.write(output/'CANONICAL_PAIRED_CONTRASTS.json',canonical)
    summary=dict(complete=True,PASS=True,source_sha256=U.sha(__file__),manifest_sha256=U.sha(manifest),plan_sha256=U.sha(plan),
        input_bindings=bindings,reference_sha256=reference_sha,contrasts=len(contrasts),populations=EXPECTED,compared_numeric_fields=len(maxima),
        maximum_absolute_difference=max(maxima.values(),default=0),independent_real_test_n=0,new_pose_calls=0,new_fits=0,
        wall_seconds=time.monotonic()-start,artifacts={n:U.sha(output/n)for n in ('INDEPENDENT_PAIRED_CONTRASTS.json','CANONICAL_PAIRED_CONTRASTS.json')})
    U.write(output/'CONTRASTS_SEAL.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-plan',type=Path)
    for key in ('manifest','plan','output'):parser.add_argument('--'+key,type=Path)
    a=parser.parse_args()
    if a.write_plan:U.write(a.write_plan,default_plan());print('Paired contrast plan fixed')
    else:print(run(a.manifest,a.plan,a.output))
