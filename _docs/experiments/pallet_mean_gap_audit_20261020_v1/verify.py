"""Independent scalar/figure conservation and geometry check.

Uses stdlib math/statistics for published CSV checks, and independently checks
native-family versus canonical physical-cuboid vertex sets for every frame.
No model/production-module imports or inference.
"""
import argparse,csv,json,math,statistics,hashlib,re
from pathlib import Path
from collections import Counter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rr(p):return list(csv.DictReader(Path(p).open()))
def close(a,b,tol=1e-8):assert abs(float(a)-float(b))<=tol,(a,b)
def truth(x):assert x in ('True','False');return x=='True'
def percentile(v,p):
    x=sorted(v);q=(len(x)-1)*p;i=math.floor(q);j=math.ceil(q);return x[i]+(x[j]-x[i])*(q-i)

def verify(public,data,output):
    p=public/'tables';pair=rr(p/'image_paired.csv');summ=rr(p/'paired_seed_summary.csv');count=Counter()
    assert len(pair)==1152
    for r in summ:
        ss=[x for x in pair if x['seed']==r['seed'] and x['comparator']==r['comparator']]
        assert len(ss)==192 and len({x['id'] for x in ss})==192
        cc=Counter(x['transition'] for x in ss)
        for k in ('BOTH_SUCCESS','RECOVER','DAMAGE','BOTH_FAILURE'):assert cc[k]==int(r[k]);count['paired_counts']+=1
        for tag,unit in [('T','cm'),('R','deg')]:
            v=[float(x[f'delta_{tag}_{unit}']) for x in ss if truth(x['common_available'])]
            close(math.fsum(v)/len(v),r[f'paired_delta_{tag}_{unit}']);count['paired_means']+=1
            for x in ss:
                close(x[f'delta_{tag}_{unit}'],float(x[f'method_{tag}_{unit}'])-float(x[f'comparator_{tag}_{unit}']));count['paired_deltas']+=1
        assert cc['RECOVER']-cc['DAMAGE']==sum(truth(x['method_success']) for x in ss)-sum(truth(x['comparator_success']) for x in ss)
    for r in rr(p/'gap_concentration.csv'):
        key='delta_'+r['metric']+('_cm' if r['metric']=='T' else '_deg');ss=[x for x in pair if x['seed']==r['seed'] and x['comparator']==r['comparator']];v=sorted([float(x[key]) for x in ss],reverse=True);k=int(r['top_k'])
        pos=math.fsum(max(0,x) for x in v);neg=math.fsum(min(0,x) for x in v)
        for z,k2 in [(pos,'positive_sum'),(neg,'negative_sum'),(pos+neg,'net_sum'),(math.fsum(v[k:]),'remainder_signed_sum'),(math.fsum(max(0,x) for x in v[:k]),'top_positive_sum')]:close(z,r[k2]);count['concentration_conservation']+=1
    for r in rr(p/'recording_summary.csv'):
        ss=[x for x in pair if x['seed']==r['seed'] and x['comparator']==r['comparator'] and x['recording']==r['group']]
        assert len(ss)==int(r['n'])
        close(math.fsum(float(x['delta_T_cm']) for x in ss),r['delta_T_sum']);close(math.fsum(float(x['delta_R_deg']) for x in ss),r['delta_R_sum']);count['recording_conservation']+=2
    panel=rr(public/'figures/PANEL_VALUES.csv');manifest=json.loads((public/'figures/PANEL_MANIFEST.json').read_text())
    for r in panel:
        # All panel numbers are also revalidated against primary paired rows.
        x=next(x for x in pair if x['id']==r['id'] and x['seed']=='20261018' and (x['comparator']==r['arm'] or x['method']==r['arm']))
        prefix='method' if x['method']==r['arm'] else 'comparator'
        close(r['T_cm'],x[prefix+'_T_cm']);close(r['R_deg'],x[prefix+'_R_deg']);assert r['success']==x[prefix+'_success'];count['figure_scalar_matches']+=3
    from PIL import Image
    for r in manifest['cases']:
        q=public/'figures'/r['file'];assert sha(q)==r['sha256']
        with Image.open(q) as im:assert list(im.size)==[r['whole_figure_width'],r['whole_figure_height']]
        count['figure_hash_and_canvas']+=1
    checks=rr(p/'baseline_official_checks.csv')
    for r in checks:close(r['official'],r['reproduced']);close(abs(float(r['official'])-float(r['reproduced'])),r['abs_difference']);count['official_values']+=1
    # Report scalar claims are checked independently against their actual tables.
    official={(r['population'],r['arm']):r for r in rr(p/'all_saved_methods_summary.csv')}
    for r in rr(public/'REPORT_VALUES.csv'):
        name=r['name'];matches=[(arm,key) for arm in {x['arm'] for x in official.values()} for key in ('successes','T_mean_cm','R_mean_deg') if name==arm+'_'+key]
        if matches:
            arm,key=matches[0];close(r['value'],official['PRIMARY192',arm][key]);count['report_scalar_matches']+=1
    # Verify projection geometry without using the visualization implementation.
    import gzip,numpy as np
    def gz(q):
        with gzip.open(q,'rt') as f:return [json.loads(s) for s in f]
    D=data/'pallet-boundary-action-repair-20261018-v1';ups=gz(D/'d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz');geommax=0
    for i,u in enumerate(ups):
        raw=gz(D/f'd1_real_candidates_v1/frames/{i:04d}/E_RUN.jsonl.gz')[0]
        d=np.array(u['known_physical_dimensions_xyz_m'])/2
        canonical=np.array([[x*d[0],y*d[1],z*d[2]] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for fam in raw['families'].values():
            xyz=np.array(fam['native_xyz']);Q=np.array(fam['physical_registration_Q']);phys=xyz@Q
            distance=np.linalg.norm(phys[:,None]-canonical[None,:],axis=2)
            err=max(float(distance.min(axis=1).max()),float(distance.min(axis=0).max()));assert err<1e-6;geommax=max(geommax,err);count['native_physical_geometry']+=1
    # Paper addition: validate complete backbone scalar populations independently.
    paper=public/'paper_tables';tr=rr(paper/'TABLE_03_ROW_METRICS.csv')
    pops={'PRIMARY192':{'TEST','REMAINDER'},'TEST49':{'TEST'},'REMAINDER143':{'REMAINDER'},'CAL46':{'CAL'}}
    for r in rr(paper/'TABLE_03_ALL_POPULATION_METRICS.csv'):
        ss=[x for x in tr if x['backbone']==r['backbone'] and x['arm']==r['arm'] and x['partition'] in pops[r['population']]]
        assert len(ss)==int(r['denominator']) and sum(truth(x['success']) for x in ss)==int(r['successes']) and sum(not truth(x['pose_available']) for x in ss)==int(r['no_pose'])
        for prefix,key in [('T_cm','translation_cm'),('R_deg','rotation_deg')]:
            v=[float(x[key]) for x in ss if x[key] not in ('','None','null')];assert len(v)==int(r[prefix+'_n'])
            for name,z in [('mean',statistics.mean(v)),('median',statistics.median(v)),('P90',percentile(v,.9)),('P95',percentile(v,.95)),('sample_variance',statistics.variance(v)),('max',max(v))]:close(z,r[prefix+'_'+name]);count['paper_backbone_scalar']+=1
    # Paired COMMON/BI and no-B effects have distinct saved source contracts.
    source_cache={}
    for r in rr(paper/'TABLE_05_PAIRED_CONTRASTS.csv'):
        if r['after'].startswith('S3_'): loc=data/'pallet-bi-full-execution-20261019-v1/S3_S4_EVALUATION/SCORED_ROWS.jsonl.gz'
        elif r['after'].startswith('F_'): loc=data/'pallet-bi-full-execution-20261019-v1/F_EVALUATION/SCORED_ROWS.jsonl.gz'
        else: loc=data/'pallet-observation-contract-h1-20261018-v1/evaluation_v1/SCORED_ROWS.jsonl.gz'
        if loc not in source_cache:source_cache[loc]=gz(loc)
        ss=[x for x in source_cache[loc] if x['partition'] in pops[r['population']]]
        aa=[x['metrics'][r['after']] for x in ss];bb=[x['metrics'][r['before']] for x in ss]
        assert sum(not y['success'] and x['success'] for x,y in zip(aa,bb))==int(r['recover'])
        assert sum(y['success'] and not x['success'] for x,y in zip(aa,bb))==int(r['damage'])
        close(math.fsum(x['translation_cm']-y['translation_cm'] for x,y in zip(aa,bb))/len(ss),r['delta_T_mean'])
        close(math.fsum(x['rotation_deg']-y['rotation_deg'] for x,y in zip(aa,bb))/len(ss),r['delta_R_mean'])
        count['distinct_ablation_pair_checks']+=4
    # Prevent accidental publishing of tensors, credentials or local metadata.
    forbidden={'FIXED_REFERENCES.json','PREDICTIONS.jsonl.gz','HEAD_FEATURES.jsonl.gz','PREFLIGHT.json','CHECKPOINTS_PRESERVE.json'}
    for q in public.rglob('*'):
        if not q.is_file():continue
        assert q.name not in forbidden and q.suffix.lower() not in ('.pt','.pth','.npz','.npy','.mp4','.zip','.pkl')
        assert q.stat().st_size<8*1024*1024
    # Every local relative document link resolves (anchors stripped).
    for q in public.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',q.read_text()):
            if link.startswith(('http:','https:','#','../')):continue
            assert (q.parent/link.split('#')[0]).exists(),(q,link)
            count['local_document_links']+=1
    result=dict(status='PASS',checks=dict(count),maximum_native_physical_vertex_difference_m=geommax,independent_implementation='stdlib fsum/statistics/linear quantiles; no production scorer imports',all192_retained=True,new_fits=0,new_inference=0)
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.public,a.data_root,a.output)
