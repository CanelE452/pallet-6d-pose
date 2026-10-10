"""Independent public verification of numeric rows, tables, CIs and PNG hashes.

Does not import summarize.py, private inputs, models or pose solvers. Aggregate
arithmetic uses Python statistics/math, linear quantiles are implemented here,
and bootstrap pooling is a separate session-loop implementation. NumPy is
used only to reproduce the published fixed random multinomial draw contract.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import struct
from collections import Counter
from pathlib import Path

METHODS=('BASE','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')
COMPARATORS=('N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC')
JOINT_METHODS=('JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')
METRICS=('corner_px','translation_cm','rotation_deg','ADDsym_cm')
SETS=('ALL','clean','moderate','severe')
ROOT=Path(__file__).resolve().parents[3]
DEFAULT_DOC=ROOT/'_docs/experiments/pallet_feature_gradient_joint_20261010'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b,label):
    if a is None or b is None:
        assert a is b,(label,a,b)
    else:
        assert math.isfinite(float(a)) and math.isfinite(float(b)),label
        assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-9),(label,a,b)


def quantile(values,p):
    a=sorted(values)
    if not a:
        return None
    position=(len(a)-1)*p
    lower=math.floor(position)
    upper=math.ceil(position)
    return a[lower]+(a[upper]-a[lower])*(position-lower)


def stats(values):
    assert all(math.isfinite(v) for v in values)
    n=len(values)
    return dict(n=n,mean=statistics.fmean(values) if n else None,
        sample_variance=statistics.variance(values) if n>1 else None,
        sample_std=statistics.stdev(values) if n>1 else None,
        median=statistics.median(values) if n else None,P90=quantile(values,.9),
        min=min(values) if n else None,max=max(values) if n else None)


def compare_stats(actual,values,label):
    independent=stats(values)
    for key,val in independent.items():
        close(actual[key],val,label+'/'+key)
    assert actual['ddof']==1,label
    if independent['sample_std'] is not None:
        close(actual['sample_std']**2,actual['sample_variance'],label+'/sd_squared')
    return independent


def point_valid(point):
    return len(point)==2 and all(v is not None and math.isfinite(v) for v in point) and point!=[-1,-1]


def same_point(a,b,label):
    assert len(a)==len(b)==2,label
    for x,y in zip(a,b):
        if x is None or y is None:
            assert x is y,label
        else:
            close(x,y,label)


def dist(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def values(row,metric):
    if metric=='corner_px':
        return [row['corner']['canonical_errors'][k] for k in range(8) if row['canonical_observed'][k]]
    if not row['pose']['available']:
        return []
    field='ADDsym_m' if metric=='ADDsym_cm' else metric
    return [row['pose'][field]*(100. if metric=='ADDsym_cm' else 1.)]


def paired_values(new,base,metric):
    if metric=='corner_px':
        return [new['corner']['canonical_errors'][k]-base['corner']['canonical_errors'][k]
                for k in range(8) if new['canonical_observed'][k] and base['canonical_observed'][k]]
    if not new['pose']['available'] or not base['pose']['available']:
        return []
    field='ADDsym_m' if metric=='ADDsym_cm' else metric
    return [(new['pose'][field]-base['pose'][field])*(100. if metric=='ADDsym_cm' else 1.)]


class IndependentBootstrap:
    def __init__(self,master):
        import numpy as np
        self.sessions=sorted({r['session'] for r in master})
        assert len(self.sessions)==13
        self.index={s:i for i,s in enumerate(self.sessions)}
        raw=np.random.default_rng(20260917).multinomial(13,np.full(13,1/13),size=10000).astype('uint16')
        self.draw_hash=hashlib.sha256(raw.tobytes()).hexdigest()
        self.draws=raw.tolist()
        self.count_cache={}

    def calculate(self,rows,per_row):
        lists=[[] for _ in self.sessions]
        for r,vals in zip(rows,per_row):
            lists[self.index[r['session']]].extend(vals)
        totals=[math.fsum(v) for v in lists]
        counts=tuple(len(v) for v in lists)
        if counts not in self.count_cache:
            self.count_cache[counts]=[sum(w*n for w,n in zip(draw,counts)) for draw in self.draws]
        denominators=self.count_cache[counts]
        samples=[]
        for draw,denom in zip(self.draws,denominators):
            if denom:
                samples.append(math.fsum(w*t for w,t in zip(draw,totals))/denom)
        eligible=sum(n>0 for n in counts)
        return dict(CI95=[quantile(samples,.025),quantile(samples,.975)] if samples and eligible>1 else None,
                    valid_resamples=len(samples),empty_resamples=10000-len(samples),
                    eligible_sessions=eligible)


def average_seed_rows(groups,method):
    """Independent scalar-error construction for seed-mean verification."""
    output=[]
    for triple in zip(*(groups[str(s)][method] for s in (1,2,3))):
        first=triple[0]
        c=first['corner']
        valid=c.get('canonical_valid',[False]*8)
        error=[statistics.fmean(r['corner']['canonical_errors'][k] for r in triple) if valid[k] else None for k in range(8)]
        observed=first['canonical_observed']
        corner=dict(evaluable=c['evaluable'],matched=c['matched'],detected=c['detected'],
                    canonical_valid=valid,canonical_errors=error,
                    errors=[v for v in error if v is not None],
                    observed_errors=[error[k] for k in range(8) if observed[k]],
                    frame_mean_px=statistics.fmean(r['corner']['frame_mean_px'] for r in triple) if c['evaluable'] else None)
        available=all(r['pose']['available'] for r in triple)
        pose=dict(available=available)
        if available:
            for field in ('translation_cm','rotation_deg','ADDsym_m','ADDsym_normalized'):
                pose[field]=statistics.fmean(r['pose'][field] for r in triple)
        output.append(dict(id=first['id'],session=first['session'],grade=first['grade'],corner=corner,
                           canonical_observed=observed,pose=pose))
    return output


def check_rows(rows,lock):
    required=('id','session','grade','seed','method','corner','pose','actual_pose','final_hypothesis',
              'canonical_observed','q0','qN','qS','qFinal','prediction_support','raw_hw','fixed_metadata',
              'evaluation_reference_points','evaluation_reference_valid','evaluation_permutation',
              'evaluation_reference_used_in_inference')
    expected_ids=lock['population']['frame_ids']
    inputs={v['id']:v for v in lock['input_manifest']}
    groups={str(s):{m:{} for m in METHODS} for s in (1,2,3)}
    reconstructed_corners=cap_checks=metadata_checks=0
    max_n3_legacy_overshoot=0.
    for r in rows:
        assert all(k in r for k in required),(r.get('id'),'required_fields')
        assert r['seed'] in (1,2,3) and r['method'] in METHODS
        assert r['id'] not in groups[str(r['seed'])][r['method']]
        groups[str(r['seed'])][r['method']][r['id']]=r
        bound=inputs[r['id']]
        assert r['session']==bound['session'] and r['grade']==bound['grade']
        assert r['raw_hw']==bound['raw_hw'] and r['prediction_support']==bound['prediction_support']
        for k in range(9):
            same_point(r['q0'][k],bound['q0'][k],r['id']+'/original_q0')
        md=r['fixed_metadata']
        assert md['preserved'] is True
        for a,b in (('selected_index','selected_index'),('K','K'),('dimensions_pnp_WH_D_m','dimensions_pnp_WH_D_m'),
                    ('canonical_symmetry_order','canonical_symmetry_order')):
            assert md[a]==bound[b],(r['id'],a)
        assert md['candidate_metadata']['candidate_index']==md['selected_index']
        assert md['candidate_metadata']==bound['candidates_metadata'][md['selected_index']],(r['id'],'candidate_scores_boxes_confidences')
        metadata_checks+=1
        assert r['evaluation_reference_used_in_inference'] is False
        for name in ('q0','qN','qS','qFinal'):
            assert len(r[name])==9 and all(len(p)==2 for p in r[name])
            same_point(r[name][8],r['q0'][8],r['id']+'/'+name+'/center8')
            for k,supported in enumerate(r['prediction_support']):
                if not supported:
                    same_point(r[name][k],r['q0'][k],r['id']+'/'+name+'/unsupported')
        h,w=r['raw_hw']; cap=.01*math.hypot(h,w)
        usable=[k for k in range(8) if r['prediction_support'][k] and point_valid(r['q0'][k])]
        if r['method']=='BASE':
            for k in range(9):
                same_point(r['qFinal'][k],r['q0'][k],r['id']+'/BASE')
        if r['method']=='N3_DIM_SYM':
            for k in usable:
                max_n3_legacy_overshoot=max(max_n3_legacy_overshoot,dist(r['qFinal'][k],r['q0'][k])-cap)
        if r['method'] in ('SUBPIX','N3_THEN_SUBPIX',*JOINT_METHODS):
            close(r['correction']['cap_px'],cap,r['id']+'/cap_radius')
            for k in usable:
                before=dist(r['qS'][k],r['q0'][k]); final=dist(r['qFinal'][k],r['q0'][k])
                assert final<=cap+1e-10,(r['id'],k,'strict_total_cap')
                scale=min(1.,cap/before) if before else 1.
                expected=[r['q0'][k][j]+(r['qS'][k][j]-r['q0'][k][j])*scale for j in (0,1)]
                same_point(r['qFinal'][k],expected,r['id']+'/single_BASE_anchor_projection')
                correction=r['correction']
                close(correction.get('total_before_cap_px8',correction.get('total_before_cap_px'))[k],before,r['id']+'/before_cap')
                close(correction.get('total_final_px8',correction.get('total_final_px'))[k],final,r['id']+'/final_motion')
                assert correction.get('cap_active8',correction.get('cap_active'))[k]==(before>cap)
                if r['method'] not in JOINT_METHODS:
                    start=r['qN'][k] if r['method']=='N3_THEN_SUBPIX' else r['q0'][k]
                    close(correction['additional_subpix_px8'][k],dist(r['qS'][k],start),r['id']+'/additional')
                cap_checks+=1
            starting=r['qN'] if r['method']!='SUBPIX' else r['q0']
            for detail in r['correction']['diagnostics']['corner_records']:
                fallback=detail.get('fallback_reason') if r['method'] in JOINT_METHODS else detail['status']!='refined'
                if fallback:
                    k=detail['corner']
                    same_point(r['qS'][k],starting[k],r['id']+'/correction_fallback_preserves_start')
        if r['method'] in JOINT_METHODS:
            check_joint_normal_equation(r)
        c=r['corner']; observed=r['canonical_observed']
        assert len(observed)==8
        if c['evaluable']:
            assert len(c['canonical_errors'])==len(c['canonical_valid'])==8
            assert c['canonical_valid']==r['evaluation_reference_valid'][:8]
            assert not any(obs and not valid for obs,valid in zip(observed,c['canonical_valid']))
            reconstructed=[None]*8; reconstructed_obs=[False]*8
            perm=r['evaluation_permutation']
            assert sorted(perm)==list(range(9)) and perm[8]==8
            for native in range(8):
                canonical=perm[native]
                if c['canonical_valid'][canonical]:
                    supported=point_valid(r['qFinal'][native]) and c['matched']
                    expected=dist(r['qFinal'][native],r['evaluation_reference_points'][canonical]) if supported else math.hypot(h,w)
                    reconstructed[canonical]=expected
                    reconstructed_obs[canonical]=bool(supported)
                    close(c['canonical_errors'][canonical],expected,r['id']+'/reconstructed_corner')
                    reconstructed_corners+=1
                else:
                    assert c['canonical_errors'][canonical] is None
            assert reconstructed_obs==observed,(r['id'],'canonical_observed')
            for a,b in zip(sorted(c['errors']),sorted(v for v in reconstructed if v is not None)):
                close(a,b,r['id']+'/full_reference_errors')
            obs_values=[reconstructed[k] for k in range(8) if observed[k]]
            assert len(obs_values)==len(c['observed_errors'])
            for a,b in zip(sorted(obs_values),sorted(c['observed_errors'])):
                close(a,b,r['id']+'/observed_errors')
            close(c['frame_mean_px'],statistics.fmean(c['errors']),r['id']+'/frame_corner_mean')
        else:
            assert not any(observed)
        if r['pose']['available']:
            assert r['actual_pose'] is not None,(r['id'],'actual_final_pose_missing')
            for metric in METRICS[1:]:
                assert all(v>=0 and math.isfinite(v) for v in values(r,metric))
        else:
            assert not values(r,'translation_cm')
    ordered={}
    for seed,methods in groups.items():
        ordered[seed]={}
        for name,lookup in methods.items():
            assert set(lookup)==set(expected_ids) and len(lookup)==319
            ordered[seed][name]=[lookup[i] for i in expected_ids]
            assert sum(len(r['corner'].get('errors',[])) for r in lookup.values())==2499
            assert sum(sum(r['canonical_observed']) for r in lookup.values())==2445
            assert sum(bool(r['corner']['matched']) for r in lookup.values())==311
    assert len(rows)==5742
    master=ordered['1']['BASE']
    assert Counter(r['grade'] for r in master)=={'clean':153,'moderate':92,'severe':74}
    assert len(set(r['session'] for r in master))==13
    for methods in ordered.values():
        for rr in methods.values():
            for r,b in zip(rr,master):
                for key in ('evaluation_reference_points','evaluation_reference_valid'):
                    assert r[key]==b[key],(r['id'],'same_evaluation_reference',key)
    for seed in ('2','3'):
        for method in ('BASE','SUBPIX'):
            for a,b in zip(ordered[seed][method],ordered['1'][method]):
                assert a['reused_from_seed1'] is True and a['F_attempt'] is False
                for key in ('q0','qN','qS','qFinal','actual_pose','pose','corner','final_hypothesis','fixed_metadata'):
                    assert a[key]==b[key],(a['id'],method,key,'shared_BASE_SUBPIX')
    frozen=ROOT/'_docs/experiments/pallet_n3_subpix_final_20261010/PREDICTIONS.jsonl.gz'
    with gzip.open(frozen,'rt',encoding='utf-8') as stream:
        old_rows=[json.loads(line) for line in stream]
    assert len(old_rows)==3828
    for old in old_rows:
        actual=groups[str(old['seed'])][old['method']][old['id']]
        for key,value in old.items():
            assert actual[key]==value,(old['id'],old['method'],'frozen_baseline_changed',key)
    return ordered,dict(rows=5742,images=319,sessions=13,reconstructed_corner_distances=reconstructed_corners,
                         strict_cap_corner_checks=cap_checks,metadata_checks=metadata_checks,
                         N3_cached_legacy_max_overshoot_px=max_n3_legacy_overshoot,
                         note='cached historical float32 N3 tiny cap excess retained; strict final combination checked',
                         actual_pose_parameters_saved=True,GT_fields_are_post_inference_evaluation_only=True)


def eigen2(matrix):
    assert len(matrix)==2 and all(len(row)==2 for row in matrix)
    a,b,c,d=matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]
    assert all(math.isfinite(v) for v in (a,b,c,d))
    close(b,c,'symmetric2x2')
    half=(a+d)/2; radius=math.hypot((a-d)/2,(b+c)/2)
    return half-radius,half+radius


def solve2(matrix,rhs):
    a,b=matrix[0];c,d=matrix[1]
    determinant=a*d-b*c
    assert determinant>0 and math.isfinite(determinant)
    return [(d*rhs[0]-b*rhs[1])/determinant,(a*rhs[1]-c*rhs[0])/determinant]


def check_joint_normal_equation(row):
    """Verify each saved successful solve independently via scalar2x2 algebra."""
    cap=row['correction']['cap_px']
    for record in row['correction']['diagnostics']['corner_records']:
        k=record['corner']
        covariance=record.get('C')
        sigma=record.get('Sigma')
        a=record.get('A'); b=record.get('b')
        if covariance is not None:
            lo,hi=eigen2(covariance)
            assert lo>=-1e-8 and hi>=lo,(row['id'],k,'C_positive_semidefinite')
            assert record['candidate_count']==222 and record['last_null_included'] is True
            close(record['probability_sum'],1.,row['id']+'/posterior_probability_sum')
            assert 0<=record['null_probability']<=1
            assert 0<=record['posterior_entropy']<=math.log(222)+1e-9
        if sigma is not None:
            lo,hi=eigen2(sigma)
            if row['method']=='JOINT_FIXED_ISOTROPIC':
                for i in (0,1):
                    for j in (0,1):
                        close(sigma[i][j],16. if i==j else 0.,row['id']+'/isotropic_only_Sigma')
            else:
                assert lo>=1.-1e-9 and hi<=cap*cap+1e-8,(row['id'],k,'Sigma_eigenvalue_bounds')
                if covariance is not None:
                    clo,chi=eigen2(covariance)
                    blo,bhi=clo+1.,chi+1.
                    flo,fhi=max(1.,min(cap*cap,blo)),max(1.,min(cap*cap,bhi))
                    if abs(bhi-blo)>1e-12:
                        alpha=(fhi-flo)/(bhi-blo); beta=flo-alpha*blo
                        expected=[[alpha*(covariance[i][j]+(1. if i==j else 0.))+(beta if i==j else 0.)
                                   for j in (0,1)] for i in (0,1)]
                    else:
                        expected=[[flo if i==j else 0. for j in (0,1)] for i in (0,1)]
                    for i in (0,1):
                        for j in (0,1):
                            close(sigma[i][j],expected[i][j],row['id']+'/C_plus_floor_eigen_clamp')
        if a is not None:
            lo,hi=eigen2(a)
            assert lo>=-1e-9 and hi>=lo,(row['id'],k,'A_positive_semidefinite')
            if record.get('S',0)>=1e-4:
                close(a[0][0]+a[1][1],1.,row['id']+'/gradient_trace1')
        if sigma is not None and a is not None and b is not None and not record.get('fallback_reason'):
            # Inversion by two scalar linear solves; inference itself uses np.linalg.solve.
            col0=solve2(sigma,[1.,0.]);col1=solve2(sigma,[0.,1.])
            precision=[[col0[0],col1[0]],[col0[1],col1[1]]]
            lhs=[[precision[i][j]+a[i][j]/16. for j in (0,1)] for i in (0,1)]
            mu=row['qN'][k]
            rhs=[math.fsum(precision[i][j]*mu[j] for j in (0,1))+b[i]/16. for i in (0,1)]
            expected=solve2(lhs,rhs)
            same_point(row['qS'][k],expected,row['id']+'/independent_joint_normal_equation')


def check_summary(actual,rows,bootstrap,label,seed_rows=None):
    assert actual['total_frames']==len(rows)
    for metric in METRICS:
        per_row=[values(r,metric) for r in rows]
        flat=[v for vs in per_row for v in vs]
        compare_stats(actual['metrics'][metric],flat,label+'/'+metric)
        independent=bootstrap.calculate(rows,per_row)
        for key in ('valid_resamples','eligible_sessions'):
            assert actual['metrics'][metric][key]==independent[key]
        ci=actual['metrics'][metric]['CI95']
        if ci is None:
            assert independent['CI95'] is None
        else:
            for a,b in zip(ci,independent['CI95']):
                close(a,b,label+'/'+metric+'/mean_CI95')
        assert actual['metrics'][metric]['draw_sha256']==bootstrap.draw_hash
        extremes=sorted([(r,statistics.fmean(values(r,metric))) for r in rows if values(r,metric)],
                        key=lambda pair:pair[1],reverse=True)[:10]
        records=actual['largest_errors'][metric]
        assert [r['id'] for r in records]==[r['id'] for r,_ in extremes],label+'/largest_error_IDs'
        for record,(row,error) in zip(records,extremes):
            close(record['error'],error,label+'/largest_error_value')
    full=[v for r in rows if r['corner']['evaluable'] for v in r['corner']['errors']]
    observed=[v for r in rows for v in values(r,'corner_px')]
    c=actual['corner']
    assert c['reference_corners']==len(full) and c['observed_corners']==len(observed)
    assert c['matched_frames']==sum(r['corner']['matched'] for r in rows)
    assert c['detected_frames']==sum(r['corner']['detected'] for r in rows)
    if seed_rows is None:
        thresholded=c
    else:
        thresholded=c['threshold_of_seed_mean_errors']
        seed_full=[[v for r in rr if r['corner']['evaluable'] for v in r['corner']['errors']] for rr in seed_rows]
        for threshold in (5,10,20):
            rates=[sum(v<=threshold for v in ff)/len(ff) for ff in seed_full]
            close(c['PCK'][str(threshold)],statistics.fmean(rates),label+'/arithmetic_mean_seed_PCK')
        close(c['gross20_count'],statistics.fmean(sum(v>20 for v in ff) for ff in seed_full),label+'/arithmetic_mean_seed_gross_count')
        close(c['gross20_rate'],statistics.fmean(sum(v>20 for v in ff)/len(ff) for ff in seed_full),label+'/arithmetic_mean_seed_gross_rate')
    for threshold in (5,10,20):
        close(thresholded['PCK'][str(threshold)],sum(v<=threshold for v in full)/len(full) if full else None,label+'/PCK_thresholded_errors')
    assert thresholded['gross20_count']==sum(v>20 for v in full)
    close(thresholded['gross20_rate'],sum(v>20 for v in full)/len(full) if full else None,label+'/gross20_thresholded_errors')
    compare_stats(c['full_reference_error_px'],full,label+'/full_reference')
    success=[r['id'] for r in rows if r['pose']['available']]
    failures=[r['id'] for r in rows if not r['pose']['available']]
    assert actual['pose']['available']==len(success) and actual['pose']['failures']==len(failures)
    assert actual['pose']['successful_ids']==success and actual['pose']['failure_ids']==failures
    meters=stats([r['pose']['ADDsym_m'] for r in rows if r['pose']['available']])
    for key,factor in (('mean',100.),('sample_std',100.),('sample_variance',10000.)):
        close(actual['metrics']['ADDsym_cm'][key],meters[key]*factor if meters[key] is not None else None,label+'/m_to_cm/'+key)
    correction=actual['correction_diagnostics']
    if seed_rows is None:
        statuses=Counter(); fallback=Counter(); fallback_ids=[]; calls=0
        for r in rows:
            detail=r.get('correction',{}).get('diagnostics',{})
            statuses.update(detail.get('status_counts',{})); fallback.update(detail.get('fallback_counts',{}))
            calls+=detail.get('algorithm_corner_calls',0)
            if any(detail.get('fallback_counts',{}).values()):
                fallback_ids.append(r['id'])
        assert correction['status_counts']==dict(statuses) and correction['fallback_counts']==dict(fallback)
        assert correction['fallback_ids']==fallback_ids and correction['fallback_frames']==len(fallback_ids)
        assert correction['fallback_corners']==sum(fallback.values()) and correction['algorithm_corner_calls']==calls
    else:
        assert correction['summary_only'] is True and correction['fallback_corners'] is None
    auc=actual['ADDsym_AUC']
    assert auc['status']=='EXISTING_CONTRACT' and auc['integration_points']==1001 and auc['max_normalized_fraction']==.1
    normalized=[r['pose']['ADDsym_normalized'] if r['pose']['available'] else math.inf for r in rows]
    finite=[v for v in normalized if math.isfinite(v)]
    expected=dict(full=independent_auc(normalized),conditional=independent_auc(finite))
    if seed_rows is None:
        target=auc
    else:
        target=auc['AUC_of_seed_mean_errors']
        for key in ('full','conditional'):
            per_seed=[]
            for rr in seed_rows:
                vals=[r['pose']['ADDsym_normalized'] if r['pose']['available'] else math.inf for r in rr]
                per_seed.append(independent_auc(vals if key=='full' else [v for v in vals if math.isfinite(v)]))
            close(auc[key],statistics.fmean(per_seed) if all(v is not None for v in per_seed) else None,label+'/seed_mean_AUC/'+key)
    for key,val in expected.items():
        close(target[key],val,label+'/AUC/'+key)
    assert actual['success_5cm_5deg']['status']=='NOT_CONFIRMED'


def independent_auc(values):
    if not values:
        return None
    assert all(v>=0 and not math.isnan(v) for v in values)
    accuracy=[sum(v<=.1*i/1000 for v in values)/len(values) for i in range(1001)]
    return math.fsum([accuracy[0]/2,*accuracy[1:-1],accuracy[-1]/2])/1000


def check_damage(actual,before,after,label):
    pairs=[(a['corner']['canonical_errors'][k],b['corner']['canonical_errors'][k])
           for a,b in zip(before,after) if a['corner']['evaluable']
           for k,v in enumerate(a['corner']['canonical_valid']) if v]
    expected=dict(corners=len(pairs),before_good5=sum(a<5 for a,b in pairs),before_bad20=sum(a>20 for a,b in pairs),
                  good5_to_bad10=sum(a<5 and b>10 for a,b in pairs),bad20_to_good10=sum(a>20 and b<=10 for a,b in pairs))
    for key,val in expected.items():
        assert actual[key]==val,(label,key,actual[key],val)
    assert len(actual['harmed_corner_records'])==expected['good5_to_bad10']
    assert len(actual['recovered_corner_records'])==expected['bad20_to_good10']


def check_paired(actual,new,base,bootstrap,label):
    for metric in METRICS:
        per_row=[paired_values(a,b,metric) for a,b in zip(new,base)]
        flat=[v for vs in per_row for v in vs]
        entry=actual['statistics'][metric]
        close(entry['mean_paired_difference'],statistics.fmean(flat) if flat else None,label+'/'+metric+'/mean')
        close(entry['median_paired_difference'],statistics.median(flat) if flat else None,label+'/'+metric+'/median')
        assert entry['common_eligible_observations']==len(flat)
        assert entry['common_eligible_frames']==sum(bool(v) for v in per_row)
        assert entry['excluded_frames']==sum(not v for v in per_row)
        independent=bootstrap.calculate(new,per_row)
        for key in ('eligible_sessions','valid_resamples','empty_resamples'):
            assert entry[key]==independent[key]
        if entry['CI95'] is None:
            assert independent['CI95'] is None
        else:
            for a,b in zip(entry['CI95'],independent['CI95']):
                close(a,b,label+'/'+metric+'/paired_CI95')
        assert entry['draw_sha256']==bootstrap.draw_hash
        marginal=actual['difference_of_marginal_statistics'][metric]
        new_stats=stats([v for r in new for v in values(r,metric)])
        base_stats=stats([v for r in base for v in values(r,metric)])
        for key in ('mean','median','P90','sample_variance','sample_std'):
            expected=new_stats[key]-base_stats[key] if new_stats[key] is not None and base_stats[key] is not None else None
            close(marginal[key],expected,label+'/'+metric+'/marginal_'+key)
    check_damage(actual['canonical_damage'],base,new,label+'/damage')
    coverage=actual['coverage']
    expected=dict(full_frames=len(new),common_success=sum(a['pose']['available'] and b['pose']['available'] for a,b in zip(new,base)),
        new_failures=sum(not r['pose']['available'] for r in new),base_failures=sum(not r['pose']['available'] for r in base),
        new_only_success=sum(a['pose']['available'] and not b['pose']['available'] for a,b in zip(new,base)),
        base_only_success=sum(not a['pose']['available'] and b['pose']['available'] for a,b in zip(new,base)),
        both_failed=sum(not a['pose']['available'] and not b['pose']['available'] for a,b in zip(new,base)))
    assert coverage==expected,label+'/coverage'


def check_diagnostics(metrics,groups):
    for seed,methods in groups.items():
        d=metrics['diagnostics']['by_seed'][seed]
        for method in METHODS:
            check_damage(d['damage_BASE'][method],methods['BASE'],methods[method],seed+'/'+method+'/BASE_damage')
            check_damage(d['damage_N3'][method],methods['N3_DIM_SYM'],methods[method],seed+'/'+method+'/N3_damage')
        caps=0; capped=[]; additional=[]; totals=[]; pre=[]; calls=0
        statuses=Counter(); fallback=Counter()
        for row in methods['N3_THEN_SUBPIX']:
            cap=.01*math.hypot(*row['raw_hw'])
            active=0
            for k in range(8):
                if row['prediction_support'][k] and point_valid(row['q0'][k]):
                    before=dist(row['qS'][k],row['q0'][k]); pre.append(before)
                    totals.append(dist(row['qFinal'][k],row['q0'][k]))
                    additional.append(dist(row['qS'][k],row['qN'][k]))
                    active+=before>cap
            caps+=active
            if active:
                capped.append(row['id'])
            diag=row['correction']['diagnostics']
            calls+=diag['algorithm_corner_calls']; statuses.update(diag['status_counts']); fallback.update(diag['fallback_counts'])
        motion=d['sequential_motion']
        assert motion['cap_active_corners']==caps and motion['cap_active_frames']==len(capped)
        assert motion['cap_active_ids']==capped and motion['OpenCV_corner_calls']==calls
        assert motion['status_counts']==dict(statuses) and motion['fallback_counts']==dict(fallback)
        compare_stats(motion['SUBPIX_additional_move_from_N3_px'],additional,seed+'/additional_motion')
        compare_stats(motion['total_move_from_BASE_px'],totals,seed+'/total_motion')
        compare_stats(motion['before_cap_total_move_from_BASE_px'],pre,seed+'/pre_cap_motion')
        for method in JOINT_METHODS:
            check_joint_diagnostics(d['joint'][method],methods[method],seed+'/'+method)
        for posterior,isotropic in zip(methods['FG_JOINT_POSTERIOR'],methods['JOINT_FIXED_ISOTROPIC']):
            for field in ('q0','qN','prediction_support','fixed_metadata'):
                assert posterior[field]==isotropic[field],(posterior['id'],'ablation_same_inputs',field)
            for p,i in zip(posterior['correction']['diagnostics']['corner_records'],
                           isotropic['correction']['diagnostics']['corner_records']):
                for field in ('corner','prediction_support','point_support','candidate_count','last_null_included',
                              'probability_sum','null_probability','posterior_entropy','candidate_mean_native_px',
                              'C','C_eigenvalues_px2','A','b','S','A_eigenvalues','fixed_window_center'):
                    assert p.get(field)==i.get(field),(posterior['id'],'ablation_only_Sigma_changes',field)
        for method in METHODS:
            check_damage(d['damage_SEQUENTIAL'][method],methods['N3_THEN_SUBPIX'],methods[method],seed+'/'+method+'/SEQUENTIAL_damage')
        for base in COMPARATORS:
            switches=[]; stable=[]; unavailable=[]
            for a,b in zip(methods['FG_JOINT_POSTERIOR'],methods[base]):
                if a['pose']['available'] and b['pose']['available'] and a['final_hypothesis'] and b['final_hypothesis']:
                    (switches if a['final_hypothesis']!=b['final_hypothesis'] else stable).append(a['id'])
                else:
                    unavailable.append(a['id'])
            h=d['hypothesis'][base]
            assert h['switch_ids']==switches and h['no_switch_ids']==stable and h['unavailable_ids']==unavailable
            assert h['switch_frames']==len(switches) and h['no_switch_frames']==len(stable) and h['unavailable_frames']==len(unavailable)
            lookup={r['id']:r for r in methods[base]}
            newlookup={r['id']:r for r in methods['FG_JOINT_POSTERIOR']}
            assert [r['id'] for r in h['switch_records']]==switches
            for record in h['switch_records']:
                a=newlookup[record['id']]; b=lookup[record['id']]
                close(record['translation_delta_cm'],a['pose']['translation_cm']-b['pose']['translation_cm'],seed+'/switch_delta_T')
                close(record['rotation_delta_deg'],a['pose']['rotation_deg']-b['pose']['rotation_deg'],seed+'/switch_delta_R')
    all_diag=metrics['diagnostics']['overall']
    assert all_diag['repeated_seed_frame_executions']==957 and all_diag['unique_images']==319
    for key,val in all_diag['sequential_motion'].items():
        assert val==sum(metrics['diagnostics']['by_seed'][str(s)]['sequential_motion'][key] for s in (1,2,3))
    for method in JOINT_METHODS:
        for key,val in all_diag['joint'][method].items():
            assert val==sum(metrics['diagnostics']['by_seed'][str(s)]['joint'][method][key] for s in (1,2,3))
    for subset in SETS:
        for method in METHODS:
            for metric in METRICS:
                for key,value in metrics['arithmetic_mean_of_seed_metrics'][subset][method][metric].items():
                    expected=[metrics['by_seed'][str(s)][subset][method]['metrics'][metric][key] for s in (1,2,3)]
                    close(value,statistics.fmean(expected) if all(v is not None for v in expected) else None,
                          subset+'/'+method+'/'+metric+'/arithmetic_seed_mean_'+key)


def check_joint_diagnostics(actual,rows,label):
    movements=[]; total=[]; before=[]; capped=[]; count=0
    matrices={k:dict(minimum=[],maximum=[],trace=[],missing=0) for k in ('C','Sigma','A')}
    strengths=[]; statuses=Counter(); fallback=Counter()
    for row in rows:
        active=0
        for k in range(8):
            if row['prediction_support'][k] and point_valid(row['q0'][k]):
                pre=dist(row['qS'][k],row['q0'][k]); before.append(pre)
                total.append(dist(row['qFinal'][k],row['q0'][k]))
                movements.append(dist(row['qS'][k],row['qN'][k]))
                active+=pre>row['correction']['cap_px']
        count+=active
        if active:
            capped.append(row['id'])
        detail=row['correction']['diagnostics'];statuses.update(detail['status_counts']);fallback.update(detail['fallback_counts'])
        for record in detail['corner_records']:
            for key,values_by_key in matrices.items():
                a=record.get(key)
                if a is None:
                    values_by_key['missing']+=1
                else:
                    lo,hi=eigen2(a)
                    values_by_key['minimum'].append(lo);values_by_key['maximum'].append(hi)
                    values_by_key['trace'].append(a[0][0]+a[1][1])
            if record.get('S') is not None:
                strengths.append(record['S'])
    assert actual['cap_active_corners']==count and actual['cap_active_ids']==capped and actual['cap_active_frames']==len(capped)
    assert actual['status_counts']==dict(statuses) and actual['fallback_counts']==dict(fallback)
    compare_stats(actual['unconstrained_move_from_N3_px'],movements,label+'/unconstrained_move')
    compare_stats(actual['total_move_from_BASE_px'],total,label+'/total_move')
    compare_stats(actual['before_cap_total_move_from_BASE_px'],before,label+'/before_cap')
    compare_stats(actual['gradient_strength_S'],strengths,label+'/S')
    for key,field in (('C','posterior_C_eigen_px2'),('Sigma','stabilized_Sigma_eigen_px2'),('A','gradient_A_eigen')):
        assert actual['matrix_missing_records'][key]==matrices[key]['missing']
        for component in ('minimum','maximum','trace'):
            compare_stats(actual[field][component],matrices[key][component],label+'/'+key+'/'+component)


def check_csv(path,metrics):
    rows=list(csv.DictReader(Path(path).open(encoding='utf-8')))
    assert len(rows)==384
    seen=set()
    for row in rows:
        key=(row['seed'],row['method'],row['set'],row['metric'])
        assert key not in seen; seen.add(key)
        groups=metrics['seed_mean'] if row['seed']=='seed_mean' else metrics['by_seed'][row['seed']]
        actual=groups[row['set']][row['method']]['metrics'][row['metric']]
        assert int(row['n'])==actual['n'] and row['unit']==actual['unit']
        for csv_field,json_field in (('mean','mean'),('variance','sample_variance'),('std','sample_std'),('median','median'),('P90','P90'),('max','max')):
            close(float(row[csv_field]) if row[csv_field] else None,actual[json_field],str(key)+'/'+csv_field)
        for csv_field,i in (('CI95_low',0),('CI95_high',1)):
            close(float(row[csv_field]) if row[csv_field] else None,actual['CI95'][i] if actual['CI95'] else None,str(key)+'/'+csv_field)
    return len(rows)


def check_figures(doc,require):
    path=doc/'FIGURE_INDEX.json'
    if not path.exists():
        assert not require,'FIGURE_INDEX.json required but missing'
        return dict(status='PENDING',reason='figures not generated yet')
    index=read(path)
    entries=index['figures']
    if isinstance(entries,dict):
        entries=[dict(path=k,**v) for k,v in entries.items()]
    verified=[]
    for entry in entries:
        relative=entry['path']
        p=Path(relative)
        assert not p.is_absolute() and '..' not in p.parts,relative
        artifact=doc/p
        assert sha(artifact)==entry['sha256'],relative+'/hash'
        assert artifact.stat().st_size==entry['bytes'],relative+'/bytes'
        for evidence in entry.get('evidence',[]):
            source=Path(evidence['path'])
            assert not source.is_absolute() and '..' not in source.parts
            assert sha(doc/source)==evidence['sha256'],relative+'/evidence/'+str(source)
        assert entry.get('rights_status')=='OWN_GENERATED_COORDINATE_OR_AGGREGATE_CHART; NO_RGB',relative+'/public_rights'
        content=artifact.read_bytes()
        assert content[:8]==b'\x89PNG\r\n\x1a\n',relative+'/PNG_signature'
        assert content[12:16]==b'IHDR'
        width,height=struct.unpack('>II',content[16:24])
        assert width>=400 and height>=200,(relative,width,height)
        # Decode the full PNG using the plotting dependency; validates CRC/data.
        import matplotlib.image as mpimg
        pixels=mpimg.imread(artifact)
        assert pixels.shape[:2]==(height,width)
        assert float(pixels.max())>float(pixels.min()),relative+'/empty_image'
        verified.append(dict(path=relative,sha256=entry['sha256'],width=width,height=height))
    assert len(verified)>=5,'at least five figures required'
    return dict(status='PASS',count=len(verified),files=verified,
                limitation='numeric table/hash and PNG decoding verified here; visible glyphs and layout require separate image inspection')


def check_seal(doc,groups):
    seal=read(doc/'COORDINATES_SEAL.json')
    assert seal['status']=='PASS' and seal['rows']==1914
    assert sha(doc/'NEW_COORDINATES_SEALED.jsonl.gz')==seal['coordinates_sha256']
    assert sha(doc/'INPUT_LOCK.json')==seal['input_lock_sha256']
    assert sha(doc/'FUSION_METHOD_LOCK.json')==seal['fusion_method_lock_sha256']
    with gzip.open(doc/'NEW_COORDINATES_SEALED.jsonl.gz','rt',encoding='utf-8') as stream:
        coordinates=[json.loads(line) for line in stream]
    assert len(coordinates)==1914
    seen=set()
    for coordinate in coordinates:
        key=(coordinate['seed'],coordinate['method'],coordinate['id'])
        assert key not in seen and coordinate['method'] in JOINT_METHODS
        seen.add(key)
        row=groups[str(coordinate['seed'])][coordinate['method']]
        row=next(r for r in row if r['id']==coordinate['id'])
        assert coordinate['GT_inference_inputs'] is False
        for field in ('q0','qN','qS','qFinal','prediction_support','raw_hw','correction'):
            assert row[field]==coordinate[field],(key,'sealed_inference_coordinate_changed',field)
        assert row['coordinate_file_sha256']==seal['coordinates_sha256']
        assert row['current_experiment_F_attempt'] is True and row['frozen_baseline_reused'] is False
    ledger=read(doc/'EXECUTION_LEDGER.json')
    assert ledger['new_F_calls']==1914 and ledger['baseline_F_calls']==0
    assert ledger['new_training']==ledger['optimizer_updates']==ledger['new_synthetic_images']==0
    assert ledger['FG_joint_cornerSubPix_calls']==0
    return dict(status='PASS',rows=1914,coordinates_sha256=seal['coordinates_sha256'],
                same_saved_coordinates_entered_actual_F=True,new_actual_F_calls=1914,
                baseline_actual_F_calls=0,new_training=0,new_synthetic_images=0)


def check_failures(doc,metrics,groups):
    failures=read(doc/'FAILURES.json')
    for seed,methods in groups.items():
        for method,rr in methods.items():
            f=failures['by_seed'][seed][method]; s=metrics['by_seed'][seed]['ALL'][method]
            assert f['no_pose_ids']==[r['id'] for r in rr if not r['pose']['available']]
            assert f['total_frames']==319 and f['available_poses']==s['pose']['available']
            assert f['correction_fallback']==s['correction_diagnostics']
            expected=[dict(id=r['id'],corner=d['corner'],reason=d.get('fallback_reason',d.get('status')))
                for r in rr for d in r.get('correction',{}).get('diagnostics',{}).get('corner_records',[])
                if d.get('fallback_reason')]
            assert f['fallback_records']==expected
            for base,label in (('BASE','BASE'),('N3_DIM_SYM','N3'),('N3_THEN_SUBPIX','SEQUENTIAL')):
                check_damage(f['damage_vs_'+label],methods[base],rr,seed+'/'+method+'/failure_damage_'+label)
            assert f['largest_finite_errors']==s['largest_errors']
    return dict(status='PASS',groups=18,pose_failure_and_corner_fallback_distinguished=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc',type=Path,default=DEFAULT_DOC)
    parser.add_argument('--require-figures',action='store_true')
    args=parser.parse_args(); doc=args.doc
    with gzip.open(doc/'PREDICTIONS.jsonl.gz','rt',encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    metrics=read(doc/'METRICS.json'); paired=read(doc/'PAIRED.json')
    lock=read(doc/'INPUT_LOCK.json')
    groups,row_checks=check_rows(rows,lock)
    master=groups['1']['BASE']; bootstrap=IndependentBootstrap(master)
    assert metrics['bootstrap']['draw_sha256']==paired['bootstrap']['draw_sha256']==bootstrap.draw_hash
    assert metrics['predictions_sha256']==paired['predictions_sha256']==sha(doc/'PREDICTIONS.jsonl.gz')
    summary_count=pair_count=0
    mean_methods={m:average_seed_rows(groups,m) for m in METHODS}
    for seed,methods in [*groups.items(),('seed_mean',mean_methods)]:
        dest=metrics['seed_mean'] if seed=='seed_mean' else metrics['by_seed'][seed]
        for subset in SETS:
            for method,rr in methods.items():
                selected=[r for r in rr if subset=='ALL' or r['grade']==subset]
                seed_rows=[[r for r in groups[str(s)][method] if subset=='ALL' or r['grade']==subset] for s in (1,2,3)] if seed=='seed_mean' else None
                check_summary(dest[subset][method],selected,bootstrap,seed+'/'+subset+'/'+method,seed_rows)
                summary_count+=1
        pdest=paired['seed_mean'] if seed=='seed_mean' else paired['by_seed'][seed]
        for subset in SETS:
            selected={m:[r for r in rr if subset=='ALL' or r['grade']==subset] for m,rr in methods.items()}
            for base in COMPARATORS:
                key='FG_JOINT_POSTERIOR_minus_'+base
                check_paired(pdest[subset][key],selected['FG_JOINT_POSTERIOR'],selected[base],bootstrap,seed+'/'+subset+'/'+key)
                pair_count+=1
    check_diagnostics(metrics,groups)
    seals=check_seal(doc,groups)
    failures=check_failures(doc,metrics,groups)
    csv_count=check_csv(doc/'METRICS.csv',metrics)
    figures=check_figures(doc,args.require_figures)
    payload=dict(schema='pallet_feature_gradient_joint_independent_verification_v1',status='PASS',
        row_contract=row_checks,distribution_summaries=summary_count,metric_distributions=summary_count*4,
        paired_comparisons=pair_count,paired_metric_CIs=pair_count*4,csv_rows=csv_count,
        fixed_draws_sha256=bootstrap.draw_hash,
        independent_arithmetic='Python statistics.fmean/variance/stdev/median and math; independently implemented linear P90',
        independent_bootstrap='separate Python session lists/totals and per-draw math.fsum; NumPy only fixed RNG draw reproduction',
        unit_conversion='ADDsym mean/SD m-to-cm x100; sample variance x10000',
        no_failures_replaced_with_finite_errors=True,seed_mean_does_not_count_957_independent_frames=True,
        figures=figures,coordinate_seal=seals,failures=failures,
        artifacts={p:sha(doc/p) for p in ('PREDICTIONS.jsonl.gz','METRICS.json','PAIRED.json','METRICS.csv','FAILURES.json','INPUT_LOCK.json','COORDINATES_SEAL.json')},
        source_code=dict(summarize_sha256=sha(Path(__file__).with_name('summarize.py')),verify_sha256=sha(Path(__file__)),
            existing_ADDsym_AUC_contract_path='scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py',
            existing_ADDsym_AUC_contract_sha256=sha(ROOT/'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py')),
        limitations='reference-coordinate distances and scalar metric aggregates verified; independent physical metrology GT unavailable; actual pose solver execution and no training are recorded separately')
    (doc/'VERIFICATION.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',summaries=summary_count,paired=pair_count,csv_rows=csv_count,figures=figures['status'])))


if __name__=='__main__':
    main()
