"""Independent postseal quaternion/atan2 metrics and descriptive aggregation.

Does not import the campaign or historical metric, aggregation, paired or
bootstrap functions. Real C2 is exactly I/Y(pi); original saved matrices and
centers enter the calculation. This is repeated development-data checking.
"""
from __future__ import annotations
import argparse,math,statistics,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import s1_union as U

POPS={'PRIMARY192':('TEST','REMAINDER'),'TEST49':('TEST',),'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
EXPECTED={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
SEED=20260917

def metric(pose,ref):
    if not pose or not pose.get('available'):return dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False)
    r=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
    if (r.shape!=(3,3)or t.shape!=(3,)or not np.isfinite(r).all()or not np.isfinite(t).all()
        or not np.allclose(r.T@r,np.eye(3),atol=1e-7,rtol=0)or not math.isclose(float(np.linalg.det(r)),1.,abs_tol=1e-7)):
        return dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False)
    qr=Rotation.from_matrix(r).as_quat();qg=Rotation.from_matrix(ref['R']).as_quat()
    if ref['order']!=2:raise ValueError('Original real C2 required; no new symmetry')
    # Right multiplication by the exact physical Y(pi) quaternion.
    variants=(qg,np.array([-qg[2],qg[3],qg[0],-qg[1]]))
    angles=[]
    for q in variants:
        v=q[3]*qr[:3]-qr[3]*q[:3]-np.cross(q[:3],qr[:3])
        w=math.fsum(float(x*y)for x,y in zip(q,qr))
        angles.append(math.degrees(2*math.atan2(math.hypot(*v),abs(w))))
    distance=math.hypot(*(float(a)-float(b)for a,b in zip(t,ref['t'])))*100
    angle=min(angles)
    return dict(pose_available=True,translation_cm=distance,rotation_deg=angle,success=bool(distance<=5 and angle<=5))

def quantile(values,p):
    a=sorted(values);position=(len(a)-1)*p;lo=math.floor(position);hi=math.ceil(position)
    return a[lo]+(a[hi]-a[lo])*(position-lo)
def distribution(values,n):
    a=[float(x)for x in values if x is not None];mean=math.fsum(a)/len(a)if a else None
    variance=math.fsum((x-mean)**2 for x in a)/(len(a)-1)if len(a)>1 else None
    return dict(n=len(a),full_denominator=n,missing=n-len(a),mean=mean,
        sample_std=math.sqrt(variance)if variance is not None else None,sample_variance=variance,
        median=statistics.median(a)if a else None,P90=quantile(a,.9)if a else None,P95=quantile(a,.95)if a else None,
        max=max(a)if a else None,ddof=1,available_only=True,tail_trimming=False)
def status(entry):
    original=entry.get('prediction_status');out=dict(original)if isinstance(original,dict)else{}
    for k in ('accepted','fallback','technical_failure','solver_rejected','timeout','status','reason'):
        if k not in out:out[k]=entry.get(k)
    if isinstance(original,str):out['status']=original
    return out
def aggregate(data,arms):
    result={}
    for arm in arms:
        ms=[r['metrics'][arm]for r in data];ss=[r['status'][arm]for r in data];n=len(data)
        counts={k:dict(true_n=sum(r.get(k)is True for r in ss),false_n=sum(r.get(k)is False for r in ss),unknown_n=sum(r.get(k)is None for r in ss))for k in ('accepted','fallback','technical_failure','solver_rejected','timeout')}
        counts['raw_status']=dict(Counter(str(r.get('status'))for r in ss))
        result[arm]=dict(denominator=n,successes=sum(m['success']for m in ms),success_rate=sum(m['success']for m in ms)/n,
            no_pose=sum(not m['pose_available']for m in ms),translation_cm=distribution([m['translation_cm']for m in ms],n),rotation_deg=distribution([m['rotation_deg']for m in ms],n),
            T_gt50cm=sum(m['translation_cm']is not None and m['translation_cm']>50 for m in ms),
            T_gt100cm=sum(m['translation_cm']is not None and m['translation_cm']>100 for m in ms),
            R_gt45deg=sum(m['rotation_deg']is not None and m['rotation_deg']>45 for m in ms),status_counts=counts,
            failures=[r['id']for r in data if not r['metrics'][arm]['success']])
    return result

def cluster_ci(values,rows,unit):
    labels=sorted({r[unit]for r in rows})
    if len(labels)<2:return dict(CI95=None,reason='fewer than two original clusters',cluster_count=len(labels))
    sums=[];counts=[]
    for label in labels:
        v=[float(v)for v,r in zip(values,rows)if r[unit]==label and v is not None];sums.append(math.fsum(v));counts.append(len(v))
    weights=np.random.default_rng(SEED).multinomial(len(labels),[1/len(labels)]*len(labels),10000)
    # Separate implementation: explicit cluster coefficient summation, no
    # original weighted-matrix helper or resampled-frame replication.
    numer=sum(weights[:,j]*sums[j]for j in range(len(labels)));denom=sum(weights[:,j]*counts[j]for j in range(len(labels)))
    draws=(numer[denom>0]/denom[denom>0]).tolist()
    return dict(CI95=[quantile(draws,.025),quantile(draws,.975)]if draws else None,cluster_count=len(labels),clusters=labels,
        cluster_sizes=dict(Counter(r[unit]for r in rows)),valid_resamples=len(draws),empty_resamples=int(sum(denom==0)),
        seed=SEED,resamples=10000,unit=unit,method='independent cluster coefficient sums, frame-weighted paired percentile bootstrap')

def paired(data,before,after):
    groups={k:[]for k in ('BOTH_SUCCESS','RECOVERY','DAMAGE','BOTH_FAILURE')};ds=[];dt=[];dr=[]
    for r in data:
        a,b=r['metrics'][before],r['metrics'][after];code=(a['success'],b['success'])
        category={(True,True):'BOTH_SUCCESS',(False,True):'RECOVERY',(True,False):'DAMAGE',(False,False):'BOTH_FAILURE'}[code]
        groups[category].append(r['id']);ds.append(int(b['success'])-int(a['success']))
        dt.append(b['translation_cm']-a['translation_cm']if a['translation_cm']is not None and b['translation_cm']is not None else None)
        dr.append(b['rotation_deg']-a['rotation_deg']if a['rotation_deg']is not None and b['rotation_deg']is not None else None)
    details={}
    for name,v in (('success_rate',ds),('translation_cm',dt),('rotation_deg',dr)):
        a=[x for x in v if x is not None];details[name]=dict(mean=math.fsum(a)/len(a)if a else None,common_available_n=len(a),
            improved=sum(x>0 if name=='success_rate'else x<0 for x in a),worsened=sum(x<0 if name=='success_rate'else x>0 for x in a),
            recording_cluster_CI=cluster_ci(v,data,'cluster_id'),named_session_cluster_sensitivity_CI=cluster_ci(v,data,'session'))
    percluster={};leaveone={}
    for cluster in sorted({r['cluster_id']for r in data}):
        indices=[j for j,r in enumerate(data)if r['cluster_id']==cluster];keep=[j for j,r in enumerate(data)if r['cluster_id']!=cluster]
        percluster[cluster]=dict(n=len(indices),before=sum(data[j]['metrics'][before]['success']for j in indices),after=sum(data[j]['metrics'][after]['success']for j in indices),
            recovery=[data[j]['id']for j in indices if ds[j]>0],damage=[data[j]['id']for j in indices if ds[j]<0],success_difference_count=sum(ds[j]for j in indices),
            delta_T_mean_cm=math.fsum(dt[j]for j in indices if dt[j]is not None)/sum(dt[j]is not None for j in indices)if any(dt[j]is not None for j in indices)else None,
            delta_R_mean_deg=math.fsum(dr[j]for j in indices if dr[j]is not None)/sum(dr[j]is not None for j in indices)if any(dr[j]is not None for j in indices)else None)
        leaveone[cluster]=dict(n=len(keep),success_difference_count=sum(ds[j]for j in keep),success_difference_percentage_points=100*sum(ds[j]for j in keep)/len(keep)if keep else None,
            delta_T_mean_cm=math.fsum(dt[j]for j in keep if dt[j]is not None)/sum(dt[j]is not None for j in keep)if any(dt[j]is not None for j in keep)else None,
            delta_R_mean_deg=math.fsum(dr[j]for j in keep if dr[j]is not None)/sum(dr[j]is not None for j in keep)if any(dr[j]is not None for j in keep)else None)
    return dict(before=before,after=after,denominator=len(data),counts={k:len(v)for k,v in groups.items()},ids=groups,
        success_difference_count=sum(ds),success_difference_percentage_points=100*sum(ds)/len(data),deltas=details,recording_clusters=percluster,leave_one_cluster_out=leaveone)

def compare(a,b,path='',errors=None,maxima=None):
    if errors is None:errors=[];maxima={}
    if isinstance(a,dict):
        for k,v in a.items():
            if k in ('method','metric_error','both_failure_tail'):continue
            if k not in b:errors.append(path+'/'+k+': missing')
            else:compare(v,b[k],path+'/'+k,errors,maxima)
    elif isinstance(a,list):
        if len(a)!=len(b):errors.append(path+': length')
        else:
            for j,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'/'+str(j),errors,maxima)
    elif isinstance(a,(float,int))and not isinstance(a,bool)and isinstance(b,(float,int)):
        d=abs(a-b);maxima[path]=d
        if isinstance(a,int):
            if a!=b:errors.append(path+': integer mismatch')
        elif d>1e-6:errors.append(path+': numeric delta '+str(d))
    elif a!=b:errors.append(path+': value mismatch')
    return errors,maxima

def verify(evaluation_dir,reference_dir,output):
    started=time.monotonic();evaluation_dir,reference_dir,output=map(Path,(evaluation_dir,reference_dir,output));output.mkdir(parents=True,exist_ok=True)
    es=U.read(evaluation_dir/'EVALUATION_SEAL.json');arms=es['arms'];merged={};bindings=[]
    # All prediction bytes/seals are validated BEFORE decoding references.
    for binding in es['prediction_bindings']:
        path=Path(binding['path']);seal=U.read(path.parent/'PREDICTION_SEAL.json')
        if U.sha(path)!=binding['sha256']or U.sha(path.parent/'PREDICTION_SEAL.json')!=binding['seal_sha256']:raise ValueError('Prediction binding changed')
        if not seal.get('complete'):raise ValueError('Prediction incomplete')
        for row in U.rows(path):
            selected={a:p for a,p in row['predictions'].items()if a in arms}
            if row['id']not in merged:merged[row['id']]=dict(row,predictions=selected)
            else:merged[row['id']]['predictions'].update(selected)
        bindings.append(binding)
    if len(merged)!=238 or any(set(r['predictions'])!=set(arms)for r in merged.values()):raise ValueError('Exact238 all-arm coverage required')
    refpath=reference_dir/'FIXED_REFERENCES.json';refseal=U.read(reference_dir/'SCORE_SEAL.json')
    if U.sha(refpath)!=es['reference_sha256']or U.sha(refpath)!=refseal['artifacts'][refpath.name]:raise ValueError('Canonical reference differs')
    refs=U.read(refpath)['references'];records=[]
    for row in merged.values():
        ref=refs[row['partition']][row['id']];records.append(dict(**{k:row[k]for k in ('id','partition','session','cluster_id')},
            poses={a:row['predictions'][a]['actual_pose']for a in arms},metrics={a:metric(row['predictions'][a]['actual_pose'],ref)for a in arms},
            status={a:status(row['predictions'][a])for a in arms}))
    summary={};pairs={}
    for pop,parts in POPS.items():
        subset=[r for r in records if r['partition']in parts]
        if len(subset)!=EXPECTED[pop]:raise ValueError('Full population changed')
        summary[pop]=aggregate(subset,arms);pairs[pop]={}
        for after in arms:
            if after.startswith(('D1_','POSEFIX_'))or after in ('N0','E0'):continue
            seed=after.rsplit('_s',1)[-1];before='D1_H1_MLP_s'+seed
            if before not in arms:before='D1_H1_MLP_s20261018'
            pairs[pop][after+'_minus_'+before]=paired(subset,before,after)
    errors,maximum=compare(U.read(evaluation_dir/'SUMMARY.json'),summary)
    pe,pm=compare(U.read(evaluation_dir/'PAIRED.json'),pairs);errors+=pe;maximum.update(pm)
    byid={r['id']:r for r in records};rowmax=Counter();row_mismatches=[]
    for expected in U.rows(evaluation_dir/'SCORED_ROWS.jsonl.gz'):
        actual=byid[expected['id']]
        for arm in arms:
            a,b=expected['metrics'][arm],actual['metrics'][arm]
            if a['success']!=b['success']or a['pose_available']!=b['pose_available']:row_mismatches.append(dict(id=expected['id'],arm=arm))
            for axis in ('translation_cm','rotation_deg'):
                if a[axis]is not None and b[axis]is not None:rowmax[axis]=max(rowmax[axis],abs(a[axis]-b[axis]))
    if rowmax['translation_cm']>1e-9 or rowmax['rotation_deg']>1e-6:errors.append('Row numerical tolerance exceeded')
    errors+=['Row classification mismatch: '+str(r)for r in row_mismatches]
    U.save(output/'INDEPENDENT_ROWS.jsonl.gz',records);U.write(output/'INDEPENDENT_SUMMARY.json',summary);U.write(output/'INDEPENDENT_PAIRED.json',pairs)
    result=dict(PASS=not errors,errors=errors,rows=238,arms=arms,populations=EXPECTED,
        row_max_abs=dict(rowmax),success_classifications_exact=not row_mismatches,summary_numeric_max_abs=max(maximum.values(),default=0),
        compared_numeric_fields=len(maximum),metric='SciPy matrix->quaternion; Hamilton relative product;2atan2(norm(v),abs(w));math.hypot translation',
        aggregation='independent math.fsum/samplevariance/sorted linear quantiles; separate paired clusters and10000draw seed20260917',
        reference_sha256=U.sha(refpath),prediction_bindings=bindings,canonical_evaluation_seal_sha256=U.sha(evaluation_dir/'EVALUATION_SEAL.json'),
        source_sha256=U.sha(__file__),wall_seconds=time.monotonic()-started,independent_real_test_n=0)
    U.write(output/'INDEPENDENT_VERIFICATION.json',result)
    if errors:raise AssertionError(errors[:10])
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ('evaluation-dir','reference-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=verify(a.evaluation_dir,a.reference_dir,a.output);print(U.canonical({k:r[k]for k in ('PASS','rows','row_max_abs','summary_numeric_max_abs','compared_numeric_fields','wall_seconds')}),flush=True)
if __name__=='__main__':main()
