"""Raw-frame operational, new-pose and common-set statistics with fixed session draws."""
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import numpy as np
from . import common as C

METRICS={'translation_cm':('translation_cm',1.,'cm'),
         'rotation_deg':('rotation_deg',1.,'degree'),'ADDsym_cm':('ADDsym_m',100.,'cm')}
DRAW_PATH=C.ROOT/'_docs/experiments/pallet_n3_static_closeout_v1/evidence/pallet_n3_static_closeout_v1__BOOTSTRAP_SESSION_COUNTS.npy.gz'

def distribution(values,unit):
    a=np.asarray(values,float);assert np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        sample_variance=float(a.var(ddof=1)) if len(a)>1 else None,
        sample_std=float(a.std(ddof=1)) if len(a)>1 else None,
        median=float(np.median(a)) if len(a) else None,P90=float(np.quantile(a,.9)) if len(a) else None,
        max=float(a.max()) if len(a) else None,unit=unit,ddof=1)

def pose_auc(rows):
    if not rows:return None
    values=np.asarray([r['pose']['ADDsym_normalized'] if r['pose']['available'] else np.inf for r in rows])
    ts=np.linspace(0,.1,1001);curve=np.asarray([(values<=t).mean() for t in ts])
    return float(np.trapz(curve,ts)/.1) if len(rows) else None

def summary(rows):
    available=[r for r in rows if r['pose']['available']]
    new=[r for r in rows if r.get('new_pose_estimated') and r['pose']['available']]
    scopes={}
    for scope,rr in [('operational',available),('new_pose',new)]:
        scopes[scope]={m:distribution([r['pose'][field]*factor for r in rr],unit) for m,(field,factor,unit) in METRICS.items()}
    return dict(total_frames=len(rows),pose_available=len(available),new_pose_estimated=len(new),
        fallback_used=sum(r.get('fallback_used',False) for r in rows),no_pose=len(rows)-len(available),
        hidden_reprojected=sum(r.get('hidden_reprojected',False) for r in rows),
        available_ids=[r['id'] for r in available],new_pose_ids=[r['id'] for r in new],
        fallback_ids=[r['id'] for r in rows if r.get('fallback_used')],no_pose_ids=[r['id'] for r in rows if not r['pose']['available']],
        metrics=scopes,ADDsym_AUC_full=pose_auc(rows),full_AUC_denominator=len(rows),
        failure_reasons=dict(Counter(r.get('solver',{}).get('reason','historical_fixed') for r in rows if not r.get('new_pose_estimated'))))

def bootstrap_setup(ids,sessions):
    units,inverse=np.unique(sessions,return_inverse=True)
    with gzip.open(DRAW_PATH,'rb') as stream:draws=np.load(stream).astype('uint16')
    expected=np.random.default_rng(20260917).multinomial(len(units),np.full(len(units),1/len(units)),size=10000).astype('uint16')
    assert np.array_equal(draws,expected), 'Existing session draw contract changed'
    return units,inverse,draws

def paired(new,base,ids,inverse,draws,fresh):
    a={r['id']:r for r in new};b={r['id']:r for r in base}
    eligible=[i for i,fid in enumerate(ids) if a[fid]['pose']['available'] and b[fid]['pose']['available'] and
              (not fresh or (a[fid].get('new_pose_estimated') and (b[fid].get('new_pose_estimated') or b[fid].get('output_status')=='HISTORICAL_FIXED_CONTROL')))]
    result=dict(denominator=319,common_frames=len(eligible),common_ids=[ids[i] for i in eligible],
        excluded_ids=[fid for i,fid in enumerate(ids) if i not in set(eligible)],scope='common_new_pose' if fresh else 'common_operational',metrics={})
    sessions=len(draws[0]);counts=np.bincount(inverse[eligible],minlength=sessions)
    denominators=draws@counts;keep=denominators>0
    for m,(field,factor,unit) in METRICS.items():
        delta=np.asarray([(a[ids[i]]['pose'][field]-b[ids[i]]['pose'][field])*factor for i in eligible])
        totals=np.bincount(inverse[eligible],weights=delta,minlength=sessions)
        sampled=(draws@totals)[keep]/denominators[keep]
        result['metrics'][m]=dict(distribution(delta,unit),CI95=np.quantile(sampled,[.025,.975]).tolist() if len(sampled) else None,
            improved_frames=int((delta< -1e-9).sum()),worsened_frames=int((delta>1e-9).sum()),unchanged_frames=int((abs(delta)<=1e-9).sum()),
            interpretation='paired mean error delta, new minus comparator; repeated-development population; no multiplicity correction')
    result['new_marginals']=summary([a[ids[i]] for i in eligible])['metrics']['operational']
    result['base_marginals']=summary([b[ids[i]] for i in eligible])['metrics']['operational']
    return result

def masking_result(rows):
    outcome=Counter();counts=Counter();geometry=defaultdict(list);remaining=Counter()
    for r in rows:
        if 'mask_audit' not in r:continue
        audit=r['mask_audit'];wrong=audit['mask_wrong_on_known'];pose=r['pose'];old=r['baseline_pose']
        label='wrong_mask' if wrong else 'correct_mask_on_known'
        counts[label]+=1
        if not r['new_pose_estimated']:outcome[label+':'+r['output_status']]+=1;continue
        if pose['available'] and old['available']:
            T=pose['translation_cm']-old['translation_cm'];R=pose['rotation_deg']-old['rotation_deg']
            state='both_improved' if T< -1e-9 and R< -1e-9 else 'both_worsened' if T>1e-9 and R>1e-9 else 'mixed_or_equal'
            outcome[label+':'+state]+=1
        s=r['solver'];key=f"U{len(s['used'])}_inlier{len(s['inliers'])}_visible{audit['remaining_human_direct_visible']}"
        remaining[key]+=1
        j=s.get('geometry',{}).get('jacobian',{}).get('condition_number')
        if j is not None:geometry[label].append(j)
    return dict(frames=len(rows),mask_counts=dict(counts),pose_outcomes=dict(outcome),
        remaining_pool_inlier_visible_histogram=dict(remaining),
        normalized_Jacobian_condition={k:distribution(v,'dimensionless') for k,v in geometry.items()},
        limitation='mask correctness restricted to known human states and fixed oracle phase; visible != accurate coordinate')

def visibility_damage(rows):
    groups=defaultdict(list)
    for r in rows:
        if 'baseline_corner' not in r or not r['corner']['evaluable']:continue
        # Canonical GT identity and existing whole-object branch metric retained.
        old=r['baseline_corner'];new=r['corner'];states=r['mask_audit']['human_states_native']
        perm=r.get('oracle_permutation')
        # Human labels are directly queried in canonical identity below.
        labels=visibility_damage.labels
        for k,valid in enumerate(old['canonical_valid']):
            if valid:
                a=old['canonical_errors'][k];b=new['canonical_errors'][k]
                groups[labels.get((r['id'],k),'UNANNOTATED')].append((a,b))
    out={}
    for state,pairs in groups.items():
        a,b=np.asarray(pairs).T
        out[state]=dict(corners=len(a),before=distribution(a,'px'),after=distribution(b,'px'),
            good5_to_bad10=int(((a<5)&(b>10)).sum()),bad20_to_good10=int(((a>20)&(b<=10)).sum()),
            improved=int((b<a-1e-9).sum()),worsened=int((b>a+1e-9).sum()))
    return out

def run():
    rows=list(C.iter_rows(C.DOC/'PREDICTIONS.jsonl.gz'));controls=list(C.iter_rows(C.DOC/'FIXED_CONTROLS.jsonl.gz'))
    if (C.DOC/'LEARNED_PREDICTIONS.jsonl.gz').exists():rows+=list(C.iter_rows(C.DOC/'LEARNED_PREDICTIONS.jsonl.gz'))
    methods=defaultdict(list)
    for r in controls+rows:methods[r['method']].append(r)
    bounds=C.read(C.DOC/'INPUTS.json')['frames'];ids=[r['id'] for r in bounds]
    for m,rr in methods.items():
        assert len(rr)==319 and set(r['id'] for r in rr)==set(ids),m
        lookup={r['id']:r for r in rr};methods[m]=[lookup[i] for i in ids]
    units,inverse,draws=bootstrap_setup(ids,[r['session'] for r in bounds])
    summaries={m:summary(rr) for m,rr in methods.items()}
    C.write(C.DOC/'METRICS.json',dict(methods=summaries,reference='Existing geometry-reconstructed DEV319 reference; not independent physical GT',
        uncertainty='sample SD is error dispersion, not standard error, seed spread or CI',
        bootstrap=dict(seed=20260917,resamples=10000,sessions=len(units),draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
            existing_draw_file=C.binding(DRAW_PATH),existing_draw_values_exact=True,group_sizes=dict(Counter(r['session'] for r in bounds)))))
    with (C.DOC/'METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','scope','metric','n','mean','sample_variance','sample_std','median','P90','max','unit','ddof','total_frames','new_pose_estimated','fallback_used','no_pose'])
        writer.writeheader()
        for m,s in summaries.items():
            for scope,mm in s['metrics'].items():
                for metric,v in mm.items():writer.writerow(dict(method=m,scope=scope,metric=metric,**v,**{k:s[k] for k in ('total_frames','new_pose_estimated','fallback_used','no_pose')}))
    contrasts=[]
    for arm in ('BASE','N3_SUBPIX'):
        for suffix,base in [('NO_MASK_STANDARD',arm),('NO_MASK_ROBUST',arm+'_NO_MASK_STANDARD'),
            ('GEOM_NOSELF_STANDARD',arm+'_NO_MASK_STANDARD'),('GEOM_NOSELF_ROBUST',arm+'_GEOM_NOSELF_STANDARD'),
            ('GEOM_NOSELF_ROBUST',arm+'_NO_MASK_ROBUST'),('GEOM_NOSELF_ROBUST',arm),
            ('ORACLE_NOSELF_ROBUST',arm+'_GEOM_NOSELF_ROBUST'),('ORACLE_VISIBLE_ROBUST',arm+'_GEOM_NOSELF_ROBUST')]:
            contrasts.append((arm+'_'+suffix,base))
    contrasts += [('N3_SUBPIX_NO_MASK_ROBUST','BASE_NO_MASK_ROBUST'),('N3_SUBPIX_GEOM_NOSELF_ROBUST','BASE_GEOM_NOSELF_ROBUST'),
                  ('SHARED_BOUNDARY_GEOM_ROBUST','BASE_GEOM_NOSELF_ROBUST')]
    for m in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'):
        if m in methods:contrasts += [(m,'N3_SUBPIX_GEOM_NOSELF_ROBUST'),(m,'SHARED_BOUNDARY_GEOM_ROBUST'),
            (m,'BASE'),(m,'N3_SUBPIX'),(m,'BASE_NO_MASK_ROBUST'),(m,'N3_SUBPIX_NO_MASK_ROBUST')]
    if 'IMAGE_ROLE' in methods:
        contrasts += [('IMAGE_ROLE','GEOMETRY_ONLY'),('IMAGE_ROLE','IMAGE_NO_ROLE')]
        for m in ('IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE'):
            if m in methods:contrasts.extend([(m,'IMAGE_ROLE'),(m,'BASE'),(m,'N3_SUBPIX')])
    paired_results={}
    for a,b in contrasts:
        paired_results[a+'_minus_'+b]={scope:paired(methods[a],methods[b],ids,inverse,draws,fresh)
            for scope,fresh in [('operational',False),('new_pose_common',True)]}
    C.write(C.DOC/'PAIRED_COMPARISONS.json',dict(contrasts=paired_results,
        draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),confidence='13 sessions,10000 shared draws; repeated use DEV319',
        full_operations='Fixed fallback applied before scoring. Means of available pose errors, full319 failure denominator separately.'))
    C.write(C.DOC/'MASK_POSE_RELATION.json',{m:masking_result(rr) for m,rr in methods.items() if 'GEOM' in m or m in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE','IMAGE_ROLE_POINT_LINE')})
    vis=C.read(C.ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')['rows']
    visibility_damage.labels={(r['frame_id'],r['corner_id']):r['category'] for r in vis if r['population']=='DEV319'}
    C.write(C.DOC/'VISIBILITY_DAMAGE.json',{m:visibility_damage(rr) for m,rr in methods.items() if m not in C.CONTROLS})
    stress=list(C.iter_rows(C.DOC/'REAL_MASK_STRESS.jsonl.gz'));ss=defaultdict(list)
    for r in stress:ss[r['condition']].append(r)
    C.write(C.DOC/'REAL_MASK_STRESS_SUMMARY.json',{m:dict(summary(rr),transformed=sum(r['transformation_possible'] for r in rr),
        unavailable_transform=sum(not r['transformation_possible'] for r in rr),
        remaining_direct_visible_counts=dict(Counter(len(r['remaining_direct_visible_ids']) for r in rr))) for m,rr in ss.items()})
    with (C.DOC/'MASK_STRESS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['condition','id','injected_id','transformation_possible','pool_count','remaining_direct_visible','final_inliers','new_pose','fallback','no_pose','translation_cm','rotation_deg','ADDsym_m','condition_number'])
        writer.writeheader()
        for r in stress:
            s=r['solver'];p=r['pose'];writer.writerow(dict(condition=r['condition'],id=r['id'],injected_id=r['injected_id'],transformation_possible=r['transformation_possible'],
                pool_count=len(s['used']),remaining_direct_visible=len(r['remaining_direct_visible_ids']),final_inliers=len(s['inliers']),
                new_pose=r['new_pose_estimated'],fallback=r['fallback_used'],no_pose=r['no_pose'],
                translation_cm=p.get('translation_cm'),rotation_deg=p.get('rotation_deg'),ADDsym_m=p.get('ADDsym_m'),
                condition_number=s.get('geometry',{}).get('jacobian',{}).get('condition_number')))
    print('STATISTICS',len(methods),'methods',flush=True)

if __name__=='__main__':run()
