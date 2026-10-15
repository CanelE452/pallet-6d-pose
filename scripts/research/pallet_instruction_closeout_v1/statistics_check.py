"""Recount published saved errors: mean, ddof1/ddof0 variance, SD and maximum.

Only sealed score rows/summaries and matching published Git blobs are opened.
There are no reference/annotation/image/model/prediction/solver imports or IO.
Continuous summaries retain every finite saved error, including failed cases.
Unavailable values retain their full denominator and explicit missing count.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

ARMS=('R0','R1','N3_F','N3_BI','SUBPIX_F','SUBPIX_BI','N0','N1')
CONTEXT_ARM='E0'
METRICS=('translation_cm','rotation_deg','ADDsym_cm')
POPULATIONS={'PRIMARY':192,'CAL':46,'TEST':49,'REMAINDER':143}
SINGLE_DOC='_docs/experiments/pallet_single_component_bi_v1'
GUARDED_DOC='_docs/experiments/pallet_guarded_bi_final_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Checks:
    def __init__(self):self.n=0;self.failures=[]
    def check(self,ok,name,detail=None):
        self.n+=1
        if not bool(ok):self.failures.append(dict(check=name,detail=detail))
    def number(self,a,b,name):
        self.check(a is None and b is None or a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-9),name,dict(recount=a,saved=b))


def verify_score(folder,expected_sha,checks,prefix):
    folder=Path(folder);seal=read(folder/'SCORE_SEAL.json')
    checks.check(sha(folder/'SCORE_SEAL.json')==expected_sha,prefix+'/score seal')
    checks.check(seal['rows']==238 and seal['all_rows_kept'] is True,prefix+'/all238')
    for name,h in seal['artifacts'].items():checks.check(sha(folder/name)==h,prefix+'/artifact '+name)
    data=read(folder/'SCORED_ROWS.json');summary=read(folder/'RESULTS.json')
    checks.check(len(data['rows'])==len({r['id'] for r in data['rows']})==238,prefix+'/238 unique saved rows')
    checks.check([r['partition'] for r in data['rows']]==['CAL']*46+['TEST']*49+['REMAINDER']*143,prefix+'/same ordered partitions')
    return data,summary,seal


def published_blobs(repo,commit,single,guarded,single_summary,guarded_summary,checks):
    """Read immutable published Git objects, independent of dirty worktree files."""
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)
    resolved=git('rev-parse',commit+'^{commit}').decode().strip()
    checks.check(resolved==commit,'publication/full commit identity')
    inventory={}
    for prefix,data,summary in [(SINGLE_DOC,single,single_summary),(GUARDED_DOC,guarded,guarded_summary)]:
        for name,target,compressed in [('metrics/SCORED_ROWS.json.gz',data,True),('metrics/RESULTS.json',summary,False)]:
            relative=prefix+'/'+name;blob=git('show',commit+':'+relative)
            decoded=json.loads(gzip.decompress(blob) if compressed else blob)
            checks.check(decoded==target,'publication/exact saved numeric artifact '+relative)
            inventory[relative]=dict(git_blob_sha256=hashlib.sha256(blob).hexdigest(),numeric_content_identical=True)
    remote=git('ls-remote','origin','refs/heads/main').decode().strip().split()[0]
    # During this closeout the named publication is the current remote main.
    checks.check(remote==commit,'publication/current remote main equals declared source',dict(remote_main=remote,source_commit=commit))
    return dict(source_commit=resolved,remote_main_at_verification=remote,artifacts=inventory)


def error_statistics(values,full_n,checks,name):
    values=list(values);available=np.asarray([float(v) for v in values if v is not None],dtype=np.float64)
    checks.check(len(values)==full_n,name+'/full denominator')
    checks.check(bool(np.isfinite(available).all()),name+'/all saved finite values retained')
    n=len(available)
    mean=float(np.mean(available)) if n else None
    sample_variance=float(np.var(available,ddof=1)) if n>1 else None
    population_variance=float(np.var(available,ddof=0)) if n else None
    sample_sd=float(np.std(available,ddof=1)) if n>1 else None
    maximum=float(np.max(available)) if n else None
    if n>1:
        checks.number(sample_variance,sample_sd**2,name+'/sample variance equals sample SD squared')
        checks.number(population_variance,sample_variance*(n-1)/n,name+'/ddof0 equals ddof1 times(n-1)/n')
        explicit=float(np.sum((available-mean)**2)/(n-1))
        checks.number(sample_variance,explicit,name+'/direct centered squared-error sum')
    return dict(full_n=full_n,available_n=n,unavailable_n=full_n-n,mean=mean,sample_variance_ddof1=sample_variance,
                population_variance_ddof0=population_variance,sample_SD_ddof1=sample_sd,max=maximum,
                sample_variance_ddof=1,population_variance_ddof=0,no_trimming=True,
                continuous_population='all finite saved errors, including accuracy failures; unavailable values counted explicitly')


def run(*,single_score,guarded_score,repo,published_commit,output,single_score_seal_sha256,guarded_score_seal_sha256):
    start=time.monotonic();cpu=time.process_time();output=Path(output);repo=Path(repo)
    if output.exists():raise FileExistsError('Preserve every earlier statistics attempt')
    checks=Checks();single,single_summary,sseal=verify_score(single_score,single_score_seal_sha256,checks,'single')
    guarded,guarded_summary,gseal=verify_score(guarded_score,guarded_score_seal_sha256,checks,'guarded')
    checks.check(single['arm_rows']==1904 and guarded['arm_rows']==2856,'same complete score inventories')
    a=single['rows'];b=guarded['rows']
    checks.check([r['id'] for r in a]==[r['id'] for r in b],'contextE0/exact same238 IDs and order')
    for x,y in zip(a,b):
        key=x['id'];checks.check(all(x[k]==y[k] for k in ('id','partition','session','cluster_id')),key+'/context same paired metadata')
        checks.check(set(x['metrics'])==set(ARMS),key+'/all8 exact main arms')
        checks.check(CONTEXT_ARM in y['metrics'],key+'/frozen context E0 present')
        for arm in ('R0','R1','N0','N1'):checks.check(x['metrics'][arm]==y['metrics'][arm],key+'/'+arm+'/shared old scores exact')
    publication=published_blobs(repo,published_commit,single,guarded,single_summary,guarded_summary,checks)
    if checks.failures:raise ValueError('Saved/publication source gate failed: '+repr(checks.failures[:3]))
    joined=[dict(id=x['id'],partition=x['partition'],session=x['session'],cluster_id=x['cluster_id'],
                 metrics={**x['metrics'],CONTEXT_ARM:y['metrics'][CONTEXT_ARM]},
                 prediction_status={**x['prediction_status'],CONTEXT_ARM:y['prediction_status'][CONTEXT_ARM]}) for x,y in zip(a,b)]
    result={};flat=[]
    for pop,expected_n in POPULATIONS.items():
        population=[r for r in joined if r['partition']!='CAL'] if pop=='PRIMARY' else [r for r in joined if r['partition']==pop]
        checks.check(len(population)==expected_n,pop+'/fixed full count')
        arms={}
        for arm in (*ARMS,CONTEXT_ARM):
            values=[r['metrics'][arm] for r in population];statuses=[r['prediction_status'][arm] for r in population]
            counts=dict(full_n=expected_n,pose_available_n=sum(v['pose_available'] is True for v in values),
                no_pose_n=sum(v['pose_available'] is False for v in values),successes_joint_le5_n=sum(v['success_le5'] is True for v in values),
                accuracy_failure_n=sum(v['success_le5'] is False for v in values),reference_unresolved_n=sum(v['success_le5'] is None for v in values),
                fallback_n=sum(v['fallback'] is True for v in statuses),fallback_status_known_n=sum(v['fallback'] is not None for v in statuses),
                accepted_n=sum(v['accepted'] is True for v in statuses),acceptance_status_known_n=sum(v['accepted'] is not None for v in statuses))
            checks.check(counts['successes_joint_le5_n']+counts['accuracy_failure_n']+counts['reference_unresolved_n']==expected_n,pop+'/'+arm+'/all success failure unknown retained')
            source=single_summary if arm!=CONTEXT_ARM else guarded_summary
            saved=source['populations'][pop]['arms'][arm]
            checks.check(counts['no_pose_n']==saved['no_pose_n'] and counts['successes_joint_le5_n']==saved['successes_le5'] and counts['reference_unresolved_n']==saved['unresolved_n'],pop+'/'+arm+'/saved count agreement')
            metrics={}
            for metric in METRICS:
                name=pop+'/'+arm+'/'+metric;stat=error_statistics([v[metric] for v in values],expected_n,checks,name)
                old=saved['errors'][metric]
                checks.number(stat['mean'],old['mean'],name+'/saved mean')
                checks.number(stat['sample_SD_ddof1'],old['sample_std'],name+'/saved sample SD')
                checks.number(stat['max'],old['max'],name+'/saved maximum')
                checks.check(stat['available_n']==old['n'] and stat['unavailable_n']==old['missing'] and old['denominator']==expected_n,name+'/saved available full counts')
                unit='degree' if metric=='rotation_deg' else 'cm';stat.update(mean_SD_max_unit=unit,variance_unit=unit+'^2')
                metrics[metric]=stat;flat.append(dict(population=pop,arm=arm,context_only=arm==CONTEXT_ARM,metric=metric,**{**counts,**stat}))
            arms[arm]=dict(context_only=arm==CONTEXT_ARM,counts=counts,errors=metrics)
        result[pop]=dict(rows=expected_n,recording_clusters=len({r['cluster_id'] for r in population}),arms=arms)
    output.mkdir(parents=True)
    write(output/'STATISTICS.json',dict(schema='FIXED_SCORE_ROW_MEAN_VARIANCE_RECOUNT_V1',populations=result,
        sample_variance_definition='sum((x-mean)^2)/(n-1), ddof=1',population_variance_definition='sum((x-mean)^2)/n, ddof=0',
        variance_not_SD=True,no_trimming=True,no_pose_and_reference_unresolved_counts_retained=True,
        E0='context only, exactly same238 IDs from the separately frozen E baseline; never silently substituted for a main arm'))
    with (output/'STATISTICS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    receipt=dict(schema='INDEPENDENT_FIXED_EVALUATION_VARIANCE_CHECK_V1',checks=checks.n,failure_count=len(checks.failures),failures=checks.failures,
        input_rows_per_source=238,main_arms=8,context_arms=1,metric_summary_rows=len(flat),populations=POPULATIONS,
        single_score_seal_sha256=single_score_seal_sha256,guarded_score_seal_sha256=guarded_score_seal_sha256,
        published_source=publication,source_sha256=sha(__file__),
        numeric_inputs='saved SCORED_ROWS and summary outputs only',reference_or_annotation_or_image_or_model_opens=0,
        scorer_or_pose_or_optimizer_imports=0,new_inference_or_PnP_or_fit_calls=0,
        original_sealed_files_modified=False,variance_sample_SD_unit_distinction_explicit=True,
        validation_absolute_tolerance=1e-9,relative_tolerance=0,
        resources=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,processes=1))
    write(output/'RECEIPT.json',receipt)
    write(output/'SEAL.json',dict(artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    if checks.failures:raise ValueError('Statistics mismatches; full attempt preserved')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('single-score','guarded-score','repo','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('published-commit','single-score-seal-sha256','guarded-score-seal-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(single_score=a.single_score,guarded_score=a.guarded_score,repo=a.repo,published_commit=a.published_commit,
        output=a.output,single_score_seal_sha256=a.single_score_seal_sha256,guarded_score_seal_sha256=a.guarded_score_seal_sha256),indent=2))
