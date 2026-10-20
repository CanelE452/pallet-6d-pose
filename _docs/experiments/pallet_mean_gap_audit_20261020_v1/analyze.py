"""Read-only saved-pose audit. No prediction, fitting, tuning or data exclusion.

Run with the existing pallet-pose Python environment. Public outputs contain
derived errors/IDs/hashes, never numeric reference or physical pose tensors.
"""
import argparse, csv, gzip, hashlib, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np

SEEDS = (20261018, 20261019, 20261020)
PF = ('POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
POPS = {'PRIMARY192': ('TEST', 'REMAINDER'), 'TEST49': ('TEST',),
        'REMAINDER143': ('REMAINDER',), 'CAL46': ('CAL',)}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p, 'rt') as f: return [json.loads(s) for s in f if s.strip()]
def index(rs):
    result = {r['id']: r for r in rs}
    assert len(result) == len(rs), 'Duplicate ID'
    return result
def write(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, ensure_ascii=False, allow_nan=False)
def csvwrite(p, rs):
    assert rs
    with Path(p).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rs for k in r)), lineterminator='\n')
        w.writeheader(); w.writerows(rs)

def ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def angle(R, G, phases=(0, math.pi)):
    return min(math.degrees(math.acos(float(np.clip((np.trace((G @ ry(p)).T @ R)-1)/2,-1,1)))) for p in phases)

def metric(pose, ref):
    assert ref['order'] == 2
    if not pose['available']:
        return dict(pose_available=False, translation_cm=None, rotation_deg=None, success=False)
    R, t, G = np.array(pose['R_physical']), np.array(pose['centroid']), np.array(ref['R'])
    assert R.shape == (3,3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all()
    assert np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) and abs(np.linalg.det(R)-1)<1e-7
    T = float(np.linalg.norm(t-np.array(ref['t']))*100)
    A = angle(R,G)
    return dict(pose_available=True, translation_cm=T, rotation_deg=A, success=T<=5 and A<=5)

def summary(rs, arm):
    ms=[r['metrics'][arm] for r in rs]
    out=dict(arm=arm, denominator=len(ms), successes=sum(m['success'] for m in ms),
             no_pose=sum(not m['pose_available'] for m in ms), metric_errors=sum(bool(m.get('metric_error')) for m in ms))
    for tag, key, unit in [('T','translation_cm','cm'),('R','rotation_deg','deg')]:
        v=np.array([m[key] for m in ms if m[key] is not None],float)
        out.update({tag+'_n':len(v),tag+'_missing':len(ms)-len(v)})
        for name, value in [('mean',np.mean(v)),('median',np.median(v)),('P90',np.quantile(v,.9)),
                            ('P95',np.quantile(v,.95)),('max',max(v)),('sample_variance',np.var(v,ddof=1))]:
            out[f'{tag}_{name}_{unit}']=float(value)
    out.update(T_gt50cm=sum(m['translation_cm'] is not None and m['translation_cm']>50 for m in ms),
               T_gt100cm=sum(m['translation_cm'] is not None and m['translation_cm']>100 for m in ms),
               R_gt45deg=sum(m['rotation_deg'] is not None and m['rotation_deg']>45 for m in ms))
    return out

def transition(a,b):
    return 'BOTH_SUCCESS' if a and b else 'RECOVER' if b else 'DAMAGE' if a else 'BOTH_FAILURE'

def run(data, repo, out):
    start=time.monotonic(); out.mkdir(parents=True, exist_ok=False)
    B=data/'pallet-bi-full-execution-20261019-v1'
    D=data/'pallet-boundary-action-repair-20261018-v1'
    C=data/'pallet-accuracy-closeout-20261017-v1'
    REF=data/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2'
    source_paths={}; private_paths={}
    def bind(alias,p):
        source_paths[alias]=dict(sha256=sha(p),bytes=p.stat().st_size,basename=p.name)
        private_paths[alias]=str(p)
        return p
    refpath=bind('fixed_reference',REF/'FIXED_REFERENCES.json')
    refs=read(refpath)['references']
    assert sha(refpath)==read(bind('reference_seal',REF/'SCORE_SEAL.json'))['artifacts'][refpath.name]
    upstream=index(rows(bind('D1_upstream',D/'d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz')))
    pf_inputs=index(rows(bind('PF_inputs',C/'posefix_inputs_v1/POSEFIX_INPUTS238.jsonl.gz')))
    fixed=index(rows(bind('fixed_bank_coverage',B/'P0_FIXED_SETS/BANK_COVERAGE_ROWS.jsonl.gz')))
    raw_preds=index(rows(bind('original_predictions',D/'PREDICTIONS.jsonl.gz')))
    features=rows(bind('original_HEAD_bank',D/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz'))
    fidx=index(features)
    official=list(csv.DictReader(bind('official40',repo/'_docs/experiments/pallet_bi_full_execution_v1/tables/PUBLIC_METRICS40/ALL_POPULATION_METRICS.csv').open()))
    sources=[('baseline',B/'P0_BASELINE_REAGGREGATION'),('union',B/'S1_EVALUATION'),('normal',B/'S2_EVALUATION'),
             ('extensions',B/'S3_S4_EVALUATION'),('final',B/'F_EVALUATION'),
             ('risk',data/'pallet-risk-information-support-20261018-v1/evaluation_v1'),
             ('observation',data/'pallet-observation-contract-h1-20261018-v1/evaluation_v1')]
    merged={}; overlaps=[]; validation=[]
    for name,folder in sources:
        p=bind(name+'_scored',folder/'SCORED_ROWS.jsonl.gz'); rr=rows(p); idx=index(rr)
        assert len(idx)==238 and set(idx)==set(upstream)
        seal=read(bind(name+'_seal',folder/'EVALUATION_SEAL.json'))
        assert seal.get('complete') is True or seal.get('status') == 'COMPLETE'
        if p.name in seal.get('artifacts',{}): assert sha(p)==seal['artifacts'][p.name]
        if seal.get('reference_sha256'): assert seal['reference_sha256']==sha(refpath)
        for r in rr:
            ident=r['id']; u=upstream[ident]
            assert all(r[k]==u[k] for k in ('id','partition','session','cluster_id'))
            if ident not in merged: merged[ident]=dict(id=ident,partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],metrics={},poses={})
            m=merged[ident]
            for arm,value in r['metrics'].items():
                if arm in m['metrics']:
                    old=m['metrics'][arm]
                    dt=abs(value['translation_cm']-old['translation_cm']) if value['translation_cm'] is not None else 0
                    dr=abs(value['rotation_deg']-old['rotation_deg']) if value['rotation_deg'] is not None else 0
                    assert dt<1e-8 and dr<1e-8 and value['success']==old['success']
                    overlaps.append((dt,dr))
                else:
                    m['metrics'][arm]=value; m['poses'][arm]=r['poses'][arm]
                # Independent remeasurement of every source scalar, not original scorer import.
                calc=metric(r['poses'][arm],refs[r['partition']][ident])
                assert calc['success']==value['success'] and calc['pose_available']==value['pose_available']
                dt=abs(calc['translation_cm']-value['translation_cm']) if calc['pose_available'] else 0
                dr=abs(calc['rotation_deg']-value['rotation_deg']) if calc['pose_available'] else 0
                validation.append((dt,dr)); assert dt<1e-8 and dr<1e-8
    ms=list(merged.values()); arms=list(ms[0]['metrics']); assert all(set(r['metrics'])==set(arms) for r in ms)
    ids=set(merged); assert ids==set(pf_inputs)==set(fixed)==set(fidx)==set(raw_preds)
    contracts=[]
    for ident,u in upstream.items():
        p=pf_inputs[ident]; ref=refs[u['partition']][ident]
        assert u['image_sha256']==p['image_sha256'] and u['raw_hw']==p['raw_hw']
        assert u['partition']==p['partition'] and u['session']==p['session']
        assert np.array_equal(u['K'],p['K'])
        assert np.allclose(u['known_physical_dimensions_xyz_m'],p['xyz'],rtol=0,atol=1e-12)
        assert np.allclose(p['xyz'],ref['xyz'],rtol=0,atol=1e-12)
        image=Path(u['image']); assert sha(image)==u['image_sha256']
        bind('RGB_'+ident,image)
        for arm in [f'D1_H1_MLP_s{s}' for s in SEEDS]+list(PF):
            po=raw_preds[ident]['predictions'][arm]['actual_pose']; new=merged[ident]['poses'][arm]
            assert po['available']==new['available'] and np.array_equal(po['R_physical'],new['R_physical']) and np.array_equal(po['centroid'],new['centroid'])
        contracts.append(dict(id=ident,image_sha256=u['image_sha256'],partition=u['partition'],session=u['session'],recording=u['cluster_id'],reference_sha256=sha(refpath),symmetry='C2_I_Y180',translation_unit='cm',rotation_unit='degree',same_K=True,same_dimensions=True,raw_hw=str(u['raw_hw']),same_saved_returned_pose=True))
    csvwrite(out/'identity_contract.csv',contracts)
    primary=[r for r in ms if r['partition']!='CAL']; assert len(primary)==192
    allsum=[]; smap={}
    for pop,parts in POPS.items():
        rr=[r for r in ms if r['partition'] in parts]
        for arm in arms:
            v=dict(population=pop,**summary(rr,arm)); allsum.append(v);smap[pop,arm]=v
    csvwrite(out/'all_saved_methods_summary.csv',allsum)
    official_checks=[]
    for r in official:
        s=smap[r['population'],r['arm']]
        for k in ['denominator','successes','no_pose','T_n','T_missing','R_n','R_missing','T_gt50cm','T_gt100cm','R_gt45deg','T_mean_cm','R_mean_deg','T_median_cm','R_median_deg','T_P90_cm','R_P90_deg','T_P95_cm','R_P95_deg','T_max_cm','R_max_deg']:
            d=abs(float(s[k])-float(r[k])); assert d<1e-8
            official_checks.append(dict(population=r['population'],arm=r['arm'],field=k,official=float(r[k]),reproduced=s[k],abs_difference=d))
    csvwrite(out/'baseline_official_checks.csv',official_checks)
    # Exact banks are deliberately separate. No coordinate-wise oracle combinations.
    bank=[]; causes=[]
    for i,f in enumerate(features):
        ident=f['id']; r=merged[ident]; ref=refs[r['partition']][ident]
        head=[(c,metric(c['actual_pose'],ref)) for c in f['candidates']]
        eligible=[(c,m) for c,m in head if c['eligible']]
        rawp=D/f'd1_real_candidates_v1/frames/{i:04d}/E_RUN.jsonl.gz'
        bind('raw_bank_'+ident,rawp); raw=rows(rawp)[0]
        em=[metric(dict(available=True,R_physical=c['R_physical'],centroid=c['centroid']),ref) for c in raw['candidates']]
        rawgood=any(m['success'] for m in em); headgood=any(m['success'] for c,m in eligible)
        assert rawgood==fixed[ident]['generated_good'] and headgood==fixed[ident]['eligible_good']
        minimax=min(eligible,key=lambda cm:max(cm[1]['translation_cm']/5,cm[1]['rotation_deg']/5))
        bank.append(dict(id=ident,partition=r['partition'],recording=r['cluster_id'],raw_generated_n=len(em),raw_generated_good=rawgood,HEAD_eligible_n=len(eligible),HEAD_eligible_good=headgood,PF_unadmitted_union_good=fixed[ident]['union_good'],oracle_minimax_candidate_id=minimax[0]['candidate_id'],oracle_same_pose_T_cm=minimax[1]['translation_cm'],oracle_same_pose_R_deg=minimax[1]['rotation_deg'],oracle_is_realized=False))
        for seed in SEEDS:
            a=f'D1_H1_MLP_s{seed}'; met=r['metrics'][a]; po=r['poses'][a]
            err=np.array(po['centroid'])-np.array(ref['t']); norm=float(np.linalg.norm(err))
            refdir=np.array(ref['t'])/np.linalg.norm(ref['t']); radial=float(err@refdir)
            C4=angle(np.array(po['R_physical']),np.array(ref['R']),phases=(0,math.pi/2,math.pi,3*math.pi/2))
            selected=raw_preds[ident]['predictions'][a]
            cause='SUCCESS_CONTROL' if met['success'] else 'GOOD_CANDIDATE_NOT_SELECTED' if headgood else 'NO_SUCCESS_CANDIDATE_IN_EXACT_BANK'
            causes.append(dict(id=ident,partition=r['partition'],recording=r['cluster_id'],seed=seed,success=met['success'],candidate_coverage_class=cause,selected_candidate_id=selected['selected_candidate_id'],rotation90_compatible_diagnostic=met['rotation_deg']>45 and C4<=5,C4_diagnostic_angle_deg=C4,depth_error_energy_fraction=float(err[2]**2/norm**2) if norm else 0,radial_error_energy_fraction=float(radial**2/norm**2) if norm else 0,signed_depth_error_cm=float(err[2]*100),physical_owner='UNKNOWN_UNVERIFIED',reference_physical_accuracy='UNKNOWN_RECONSTRUCTED_REFERENCE',causal_scale_depth='UNKNOWN',causal_owner_mismatch='UNKNOWN',causal_rotation90='UNPROVEN_GEOMETRIC_COMPATIBILITY_ONLY'))
    csvwrite(out/'exact_bank_coverage.csv',bank);csvwrite(out/'failure_coverage.csv',causes)
    # No tuning: all diagnostics below summarize fixed outputs only.
    pairrows=[]; pairs=[]; concentration=[]; recording=[]; session=[]; tailgroups=[]
    for seed in SEEDS:
        a=f'D1_H1_MLP_s{seed}'
        for b in PF:
            pp=[]
            for r in primary:
                ma,mb=r['metrics'][a],r['metrics'][b]
                common=ma['pose_available'] and mb['pose_available'] and ma['translation_cm'] is not None and mb['translation_cm'] is not None and ma['rotation_deg'] is not None and mb['rotation_deg'] is not None
                x=dict(id=r['id'],image_sha256=upstream[r['id']]['image_sha256'],recording=r['cluster_id'],session=r['session'],partition=r['partition'],seed=seed,method=a,comparator=b,method_available=ma['pose_available'],comparator_available=mb['pose_available'],common_available=common,method_success=ma['success'],comparator_success=mb['success'],transition=transition(mb['success'],ma['success']),method_T_cm=ma['translation_cm'],comparator_T_cm=mb['translation_cm'],method_R_deg=ma['rotation_deg'],comparator_R_deg=mb['rotation_deg'],delta_T_cm=ma['translation_cm']-mb['translation_cm'] if common else None,delta_R_deg=ma['rotation_deg']-mb['rotation_deg'] if common else None)
                pp.append(x);pairrows.append(x)
            both=[x for x in pp if x['common_available']]
            pairs.append(dict(seed=seed,method=a,comparator=b,full_denominator=192,method_available=sum(x['method_available'] for x in pp),comparator_available=sum(x['comparator_available'] for x in pp),common_available=len(both),paired_delta_T_cm=float(np.mean([x['delta_T_cm'] for x in both])),paired_delta_R_deg=float(np.mean([x['delta_R_deg'] for x in both])),**Counter(x['transition'] for x in pp)))
            for tag in ('T','R'):
                key=f'delta_{tag}_'+('cm' if tag=='T' else 'deg')
                vals=sorted(both,key=lambda x:(-x[key],x['id']));pos=math.fsum(max(0,x[key]) for x in vals);neg=math.fsum(min(0,x[key]) for x in vals);net=pos+neg
                for k in (5,10,math.ceil(.1*len(both))):
                    top=math.fsum(max(0,x[key]) for x in vals[:k])
                    concentration.append(dict(seed=seed,comparator=b,metric=tag,common_n=len(both),top_k=k,positive_sum=pos,negative_sum=neg,net_sum=net,net_mean=net/len(both),worse_n=sum(x[key]>0 for x in vals),better_n=sum(x[key]<0 for x in vals),equal_n=sum(x[key]==0 for x in vals),top_positive_sum=top,top_fraction_of_positive=top/pos if pos else None,top_fraction_of_net=top/net if net else None,remainder_signed_sum=math.fsum(x[key] for x in vals[k:]),diagnostic_exclusion_only_no_trimmed_performance=True))
            for gkey,target in [('recording',recording),('session',session),('transition',tailgroups)]:
                for group in sorted({x[gkey] for x in pp}):
                    xx=[x for x in pp if x[gkey]==group];n=len(xx)
                    target.append(dict(seed=seed,comparator=b,group=group,n=n,method_successes=sum(x['method_success'] for x in xx),comparator_successes=sum(x['comparator_success'] for x in xx),recover=sum(x['transition']=='RECOVER' for x in xx),damage=sum(x['transition']=='DAMAGE' for x in xx),delta_T_sum=math.fsum(x['delta_T_cm'] for x in xx),delta_R_sum=math.fsum(x['delta_R_deg'] for x in xx),delta_T_mean=float(np.mean([x['delta_T_cm'] for x in xx])),delta_R_mean=float(np.mean([x['delta_R_deg'] for x in xx])),method_T_mean=float(np.mean([x['method_T_cm'] for x in xx])),comparator_T_mean=float(np.mean([x['comparator_T_cm'] for x in xx])),method_R_mean=float(np.mean([x['method_R_deg'] for x in xx])),comparator_R_mean=float(np.mean([x['comparator_R_deg'] for x in xx]))))
    csvwrite(out/'image_paired.csv',pairrows);csvwrite(out/'paired_seed_summary.csv',pairs);csvwrite(out/'gap_concentration.csv',concentration)
    csvwrite(out/'recording_summary.csv',recording);csvwrite(out/'session_summary.csv',session);csvwrite(out/'success_failure_strata.csv',tailgroups)
    # Prior trials stay explicitly historical, with the identical reference/population.
    prior=[]
    for a in arms:
        seed=next((s for s in SEEDS if a.endswith(str(s))),None)
        if seed is None or a.startswith('D1_H1_MLP_'): continue
        orig=f'D1_H1_MLP_s{seed}'; mm=smap['PRIMARY192',a]; oo=smap['PRIMARY192',orig]
        trans=Counter(transition(r['metrics'][orig]['success'],r['metrics'][a]['success']) for r in primary)
        prior.append(dict(arm=a,seed=seed,successes=mm['successes'],delta_successes=mm['successes']-oo['successes'],recover=trans['RECOVER'],damage=trans['DAMAGE'],T_mean_cm=mm['T_mean_cm'],R_mean_deg=mm['R_mean_deg'],delta_T_vs_D1=mm['T_mean_cm']-oo['T_mean_cm'],delta_R_vs_D1=mm['R_mean_deg']-oo['R_mean_deg'],T_mean_below_both_PF=all(mm['T_mean_cm']<smap['PRIMARY192',p]['T_mean_cm'] for p in PF),R_mean_below_both_PF=all(mm['R_mean_deg']<smap['PRIMARY192',p]['R_mean_deg'] for p in PF),historical_not_new_experiment=True))
    csvwrite(out/'prior_trials_reaggregated.csv',prior)
    checks=dict(status='PASS',images=238,primary_n=192,duplicate_IDs=0,distinct_image_hashes=len({u['image_sha256'] for u in upstream.values()}),image_hash_checks=238,metadata_contract_checks=238,all_source_scalar_remeasurements=len(validation),maximum_T_remeasurement_difference_cm=max(d[0] for d in validation),maximum_R_remeasurement_difference_deg=max(d[1] for d in validation),shared_arm_source_checks=len(overlaps),maximum_shared_T_difference=max(d[0] for d in overlaps),maximum_shared_R_difference=max(d[1] for d in overlaps),official_scalar_checks=len(official_checks),official_max_difference=max(r['abs_difference'] for r in official_checks),all_method_count=len(arms),reference_sha256=sha(refpath),binary_denominator='full; unavailable is failure',continuous_denominator='available only; paired separately',no_tuning=True,independent_real_test_n=0,wall_seconds=time.monotonic()-start)
    write(out/'VALIDATION.json',checks);write(out/'SOURCE_MANIFEST.json',source_paths)
    # Absolute locations are local provenance only, excluded from publication.
    write(out.parent/(out.name+'_PRIVATE_PATHS.json'),private_paths)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--source-repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data_root,a.source_repo,a.out)
