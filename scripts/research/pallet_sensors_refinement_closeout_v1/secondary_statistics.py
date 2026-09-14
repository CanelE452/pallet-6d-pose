"""Exploratory paired tail/PCK/pose intervals, shared draws across seed panel."""
import numpy as np
from env import *
from analysis_panel import load
def run():
    stores,poses,summaries,_=load(True);ag=old('aggregate_results');keys=[r['frame_id'] for r in stores['R0']]
    sessions=[r['session_id'] for r in stores['R0']];names=sorted(set(sessions));group=np.array([names.index(s) for s in sessions]);lookup=dict(zip(keys,group))
    results={};frame_changes={}
    for left,right in [('P','R0'),('P','D'),('L','R0')]:
        labels=[f'{left}{s}' for s in (1,2,3)]+(['R0'] if right=='R0' else [f'{right}{s}' for s in (1,2,3)])
        out={}
        for metric in ['keypoint_location_p90_px','gross20','ALL_GT_PCK10',*ag.POSE_METRICS,'yaw_median_deg']:
            if metric in [*ag.POSE_METRICS,'yaw_median_deg'] and any(set(poses[name])!=set(keys) for name in labels):
                out[metric]=dict(status='NOT_ESTIMATED_SUPPORT_MISMATCH',reason='Do not silently intersect pose failures',excluded={name:sorted(set(keys)-set(poses[name])) for name in labels},available_metrics_in='UNIFIED_DEV_RESULTS.methods.<name>.pose',exploratory=True)
                continue
            series=[]
            for name in labels:
                if metric in ['gross20','ALL_GT_PCK10']:
                    rr=stores[name];num=np.array([int((r['errors']>20).sum()) if metric=='gross20' else int((r['errors']<=10).sum()) for r in rr]);den=np.array([len(r['errors']) if metric=='gross20' else r['gt'] for r in rr]);series.append((num,den))
                elif metric=='keypoint_location_p90_px':series.append(ag.prepare_series({r['frame_id']:r for r in stores[name]},keys,metric,lookup))
                else:
                    field='yaw_error_deg' if metric=='yaw_median_deg' else ag.POSE_FIELDS[metric]
                    vals=np.array([poses[name][k][field] for k in keys]);diam=np.array([poses[name][k]['diameter_m'] for k in keys]);series.append((vals,group,diam))
            def statistics(counts):
                if metric in ['gross20','ALL_GT_PCK10']:
                    ww=counts[:,group];v=np.stack([(ww*n[None]).sum(-1)/(ww*d[None]).sum(-1) for n,d in series])
                else:v=np.stack([ag.weighted_statistic(v,g,counts,metric,d) for v,g,d in series])
                return v[:3].mean(0)-(v[3] if right=='R0' else v[3:].mean(0))
            rng=np.random.default_rng(20260914);draws=[]
            for start in range(0,10000,128):draws.extend(statistics(rng.multinomial(len(names),np.full(len(names),1/len(names)),size=min(128,10000-start))))
            out[metric]=dict(delta=float(statistics(np.ones((1,len(names)),int))[0]),low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),sessions=len(names),frames=len(keys),draws=10000,seed=20260914,exploratory=True)
        results[f'{left}-{right}']=out;print('SECONDARY_STATS',left,right,flush=True)
    for name,rr in stores.items():
        if name=='R0':continue
        diff=np.array([r['errors'].mean()-b['errors'].mean() for r,b in zip(rr,stores['R0']) if len(b['errors'])])
        frame_changes[name]=dict(improved=int((diff<0).sum()),worsened=int((diff>0).sum()),unchanged=int((diff==0).sum()),metric='matched frame mean error; descriptive only')
    write(DOC/'EXPLORATORY_PAIRED.json',dict(comparisons=results,frame_changes=frame_changes,confidence='paired session 95%; no multiplicity adjustment, not confirmatory tests'))
if __name__=='__main__':run()
