"""Post-freeze official scoring and fixed-mapping mechanism diagnostics."""
import argparse
from collections import Counter
import numpy as np
from . import common as C
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1 import evaluate as V
from scripts.research.pallet_posefix_heatmap_diversity_v1.diagnose import inspect_channel
B=C.B

def transition(rows,arm,reference):
    left=np.array([r['errors'][reference]<=10 for r in rows],dtype=bool)
    right=np.array([r['errors'][arm]<=10 for r in rows],dtype=bool)
    out=dict(GG=int((left&right).sum()),GB=int((left&~right).sum()),BG=int((~left&right).sum()),BB=int((~left&~right).sum()))
    assert int(right.sum()-left.sum())==out['BG']-out['GB'];return out

def fixed_inputs():
    lock=C.read(C.DOC/'INPUT_LOCK.json');freeze=C.read(C.DOC/'PREDICTION_LOCK.json');assert freeze['all_frozen_before_scoring']
    for a,r in freeze['arms'].items():C.verify(r['predictions'])
    return lock,freeze

def score():
    lock,freeze=fixed_inputs();C.save(C.DOC/'SCORING_START.json',dict(predictions=C.bind(C.DOC/'PREDICTION_LOCK.json'),code=C.bind(C.HERE/'evaluate.py')))
    truth=C.read(B.E.V.RAW/'E1_FRAME_METRICS.json')['truth_for_display_only'];old=C.read(C.F.RAW/'FRAME_METRICS.json');metrics={a:old[a] for a in ('R0','BASE','N2')}
    sym={r['object_type']:r['permutations'] for r in C.read(C.ROOT/lock['symmetry']['path'])['objects']}
    raw=C.read(B.E.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0'];preds={a:C.read(C.ROOT/x['predictions']['path'])['predictions'] for a,x in freeze['arms'].items()}
    for a in C.ARMS:
        metrics[a]={}
        for r in lock['eval_records']:
            fid=r['id'];p=preds[a][fid];B.E.assert_preserved(raw[fid],p);cand=B.E.P.top(p);t=truth[fid];base=old['R0'][fid]
            q=np.full((9,2),np.nan) if cand is None else cand['keypoints_xy']
            m=B.E.V.M.measure(q,t['gt'],t['valid'],sym[r['object_type']],[480,640],base['matched'],base['detected']);metrics[a][fid]=dict(id=fid,session=r['session'],**m)
            if a=='A':assert metrics[a][fid]==old['FULL'][fid]
    rows=C.read(C.F.RAW/'CORNER_ROWS.json')
    for r in rows:
        for a in C.ARMS:r['errors'][a]=metrics[a][r['frame_id']]['canonical_errors'][r['corner_id']]
    summary={};trans={};bands={};session={}
    for pop,ids in lock['populations'].items():
        rr=[r for r in rows if r['frame_id'] in set(ids)];summary[pop]={};trans[pop]={};bands[pop]={}
        for a in C.ARMS:
            s=B.E.V.summarize([metrics[a][i] for i in ids],[metrics['R0'][i] for i in ids])
            s.update(correct10=sum(r['errors'][a]<=10 for r in rr),correct20=sum(r['errors'][a]<=20 for r in rr),gross20_count=sum(r['errors'][a]>20 for r in rr))
            assert s['corners']==len(rr);assert s['PCK']['10']==s['correct10']/len(rr);summary[pop][a]=s
            trans[pop][a]={ref:transition(rr,a,ref) for ref in ('A','BASE','N2')}
            trans[pop][a]['R0_good5_damage']=sum(r['errors']['R0']<5 and r['errors'][a]>10 for r in rr)
            if a=='A':
                for k,v in C.read(C.F.DOC/'REAL_RESULTS.json')['summary'][pop]['FULL'].items():assert s[k]==v,(pop,k)
        for band in ('B0','B1','B2','B3','B4','B5_MATCH_FAILURE'):
            sub=[r for r in rr if r['band']==band];bands[pop][band]=dict(n=len(sub),models={})
            for a in C.ARMS:
                e=np.array([r['errors'][a] for r in sub]);base=np.array([r['errors']['A'] for r in sub])
                bands[pop][band]['models'][a]=dict(correct10=int((e<=10).sum()),correct20=int((e<=20).sum()),mean=float(e.mean()) if len(e) else None,
                    median=float(np.median(e)) if len(e) else None,improved_vs_A=int((e<base-1e-9).sum()),worsened_vs_A=int((e>base+1e-9).sum()),
                    transitions={ref:transition(sub,a,ref) for ref in ('A','BASE','N2')})
        session[pop]={}
        for sn in sorted({r['session'] for r in rr}):
            sr=[r for r in rr if r['session']==sn];session[pop][sn]=dict(frames=len({r['frame_id'] for r in sr}),corners=len(sr),
                correct10={a:sum(r['errors'][a]<=10 for r in sr) for a in C.ARMS},A_C=transition(sr,'C','A'))
    subsets=C.read(C.DOC/'SUBSET_LOCK.json')['groups'];groups={};old29=[];rec={r['id']:r for r in lock['eval_records']}
    for name,ss in subsets.items():
        groups[name]=dict(n=len(ss),models={})
        for a in C.ARMS:
            e=np.array([metrics[a][r['id']]['canonical_errors'][r['canonical']] for r in ss])
            groups[name]['models'][a]=dict(correct10=int((e<=10).sum()),correct20=int((e<=20).sum()),median=float(np.median(e)) if len(e) else None)
    for r in subsets['U29']:
        fid=r['id'];j=r['canonical'];values={}
        for a in C.ARMS:
            branch=metrics[a][fid]['branch'];ch=sym[rec[fid]['object_type']][branch].index(j)
            xy=B.E.P.top(preds[a][fid])['keypoints_xy'][ch]
            values[a]=dict(error=metrics[a][fid]['canonical_errors'][j],xy=xy,native_channel=ch,branch=branch)
            if a in ('C','D') and r in subsets['U15']:
                cache=np.load(C.ROOT/freeze['arms'][a]['heatmaps'][fid]['path'])
                if cache['valid'][ch]:assert values[a]['error']>10,'Impossible support success: inspect transform/mapping'
        old29.append(dict(id=fid,canonical=j,reference=r['reference'],old_support=r['C0'],new_support=r['C1'],models=values,
            coordinate_provenance='existing legacy reference; independent coordinate review pending',visibility='REVIEW_PENDING'))
    C.save(C.RAW/'FRAME_METRICS.json',metrics);C.save(C.RAW/'CORNER_ROWS.json',rows);C.save(C.RAW/'OLD29_ROWS.json',old29)
    C.save(C.DOC/'REAL_RESULTS.json',dict(summary=summary,subgroups=groups,transitions=trans,bands=bands,sessions=session,
        identity='official per-model whole-object permutation; mechanism uses fixed FP mapping separately',denominator_unchanged=True))
    print('SCORED', {a:summary['PRIMARY_OCC96'][a]['correct10'] for a in C.ARMS},flush=True)

def peak_detail(cache,ch,target):
    z=cache['logits'][ch];m=cache['matrix'];v=inspect_channel(z,m,target,cache['expectation'][ch])
    yy,xx=np.mgrid[:96,:72];g=np.array(v['GT_crop']);dist=np.linalg.norm(np.stack([xx*4,yy*4],-1)-g,axis=-1)/m[0,0]
    area=int((dist<=10).sum());fraction=area/z.size
    local=np.array(v['top5_crop']);heights=np.array([z[int(y/4),int(x/4)] for x,y in local])
    v.update(neighborhood_grid_count=area,uniform_expected_mass=fraction,probability_mass_over_uniform=v['GT_radius10_mass']/fraction if fraction else None,
        normalized_entropy=v['entropy']/np.log(z.size),peak1_peak2_ratio=float(np.exp(np.clip(heights[0]-heights[1],-80,80))) if len(heights)>1 else None,
        in_output_support=v['nearest_output_error']<=1e-9)
    return v

def heatmaps():
    lock,freeze=fixed_inputs();metric=C.read(C.RAW/'FRAME_METRICS.json');fp=C.read(C.F.RAW/'FRAME_METRICS.json')['FULL_PRESERVE']
    truth=C.read(B.E.V.RAW/'E1_FRAME_METRICS.json')['truth_for_display_only'];syms={r['object_type']:r['permutations'] for r in C.read(C.ROOT/lock['symmetry']['path'])['objects']}
    primary=set(lock['populations']['PRIMARY_OCC96']);rec={r['id']:r for r in lock['eval_records']}
    fixed={(r['id'],r['canonical']) for r in C.read(C.DOC/'SUBSET_LOCK.json')['groups']['H163']}
    remaining={(fid,j) for fid in primary if metric['R0'][fid]['matched'] for j,v in enumerate(metric['R0'][fid]['canonical_valid']) if v and metric['R0'][fid]['canonical_errors'][j]>20 and metric['C'][fid]['canonical_errors'][j]>20}
    keys=sorted(fixed|remaining);rows=[]
    for a in C.ARMS:
        for fid in sorted({k[0] for k in keys}):
            b=freeze['arms'][a]['heatmaps'][fid];C.verify(b);cache=np.load(C.ROOT/b['path']);perms=syms[rec[fid]['object_type']];fixedperm=perms[fp[fid]['branch']];officialperm=perms[metric[a][fid]['branch']]
            for _,j in [key for key in keys if key[0]==fid]:
                ch=fixedperm.index(j);target=truth[fid]['gt'][j];v=peak_detail(cache,ch,target)
                if not cache['valid'][ch]:category='PASSTHROUGH_NOT_DECODER'
                elif v['unreachable10']:category='SUPPORT_FAIL'
                elif v['expected_error']<=10:category='DECODED_CORRECT'
                elif v['top5_nearest_error']<=10:category='PEAK_PRESENT_DECODE_WRONG'
                else:category='NO_TESTED_PEAK'
                other=[]
                for c in range(8):
                    if c==ch:continue
                    z=inspect_channel(cache['logits'][c],cache['matrix'],target,cache['expectation'][c])
                    if z['top5_nearest_error']<=10:
                        allowed=[i for i,p in enumerate(perms) if p[c]==j]
                        tag='CHANNEL_CONFUSION' if not allowed else 'LEGIT_SYMMETRY_EQUIVALENT' if officialperm[c]==j else 'UNKNOWN'
                        other.append(dict(channel=c,error=z['top5_nearest_error'],allowed_whole_branches=allowed,classification=tag,
                            basis='official model whole-object branch agrees' if tag=='LEGIT_SYMMETRY_EQUIVALENT' else 'not evidence of a deployable swap'))
                # Fixed channel error can differ from official reselected branch error; never mix them.
                rows.append(dict(model=a,id=fid,canonical=j,fixed_native_channel=ch,fixed_branch=fp[fid]['branch'],official_branch=metric[a][fid]['branch'],
                    official_error=metric[a][fid]['canonical_errors'][j],fixed_set=(fid,j) in fixed,remaining_set=(fid,j) in remaining,
                    decoder_applied=bool(cache['valid'][ch]),category=category,identity_suspect=category=='NO_TESTED_PEAK' and bool(other),other_channels=other,**v))
        print('HEATMAP_AUDIT',a,flush=True)
    summary={}
    for name,flag in [('H163','fixed_set'),('C_REMAINING_HARD','remaining_set')]:
        summary[name]={}
        for a in C.ARMS:
            rr=[r for r in rows if r['model']==a and r[flag]];reach=[r for r in rr if not r['unreachable10'] and r['decoder_applied']]
            counts=dict(Counter(r['category'] for r in rr));assert sum(counts.values())==len(rr)
            rescue=sum(r['expected_error']>10 and r['top5_nearest_error']<=10 for r in reach)
            summary[name][a]=dict(n=len(rr),categories=counts,reachable=len(reach),own_top5_present=sum(r['top5_nearest_error']<=10 for r in reach),
                top5_oracle_rescue=rescue,top5_rescue_fraction_reachable=rescue/len(reach) if reach else None,
                argmax_rescue=sum(r['expected_error']>10 and r['argmax_error']<=10 for r in reach),identity_suspect=sum(r['identity_suspect'] for r in rr),
                raw_mass_median=float(np.median([r['GT_radius10_mass'] for r in rr])),normalized_entropy_median=float(np.median([r['normalized_entropy'] for r in rr])),
                branch_changed_vs_fixed=sum(r['official_branch']!=r['fixed_branch'] for r in rr))
    maps={a:{(r['id'],r['canonical']):r for r in rows if r['model']==a and r['fixed_set']} for a in C.ARMS}
    transitions=dict(Counter(maps['A'][key]['category']+' -> '+maps['C'][key]['category'] for key in fixed))
    identity={a:dict(Counter(q['classification'] for r in rows if r['model']==a and r['fixed_set'] and r['identity_suspect'] for q in r['other_channels'])) for a in C.ARMS}
    C.save(C.RAW/'HEATMAP_DIAGNOSTICS.json',rows)
    C.save(C.DOC/'HEATMAP_SUMMARY.json',dict(summary=summary,A_C_transitions=transitions,identity_candidate_tags=identity,
        mapping='fixed historical FULL_PRESERVE whole-object native↔canonical for each frame, frozen before training; official performance uses per-model branches',
        peaks='unchanged 5x5 cv2.dilate local maxima, stable logit-descending top5 incl borders; not a learned selector',
        oracle_only=True,pointwise_remapping=False,physical_identity_confirmed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['score','heatmaps']);a=p.parse_args();globals()[a.stage]()
