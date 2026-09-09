"""Fixed H-vs-P primary comparison; no best-arm or best-seed selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .cache_io import write_json,sha256
from .metrics import paired_session_bootstrap


def main():
    p=argparse.ArgumentParser()
    for arm in ('P','S','H','HA'):p.add_argument('--'+arm,nargs=3,required=True,metavar=('SEED1','SEED2','SEED3'))
    p.add_argument('--output',required=True);args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('Do not overwrite the registered verdict')
    paths={a:getattr(args,a) for a in ('P','S','H','HA')}
    if len({str(Path(x).resolve()) for v in paths.values() for x in v})!=12:
        raise ValueError('Require 12 distinct result files, not repeated seeds')
    data={a:[json.loads(Path(x).read_text(encoding='utf8')) for x in paths[a]] for a in paths}
    all_rows=sum(data.values(),[]);ref=all_rows[0]
    if any(r.get('stage')!='main' for r in all_rows):raise ValueError('Smoke results may never establish performance')
    if any(r['population_fingerprint']!=ref['population_fingerprint'] or r['role']!=ref['role'] for r in all_rows):
        raise ValueError('Population, predictions or supervision masks differ')
    ids=[r['frame_id'] for r in ref['frame_rows']];sessions=[r['session_id'] for r in ref['frame_rows']]
    for a,entries in data.items():
        if any(r['arm']!=a for r in entries):raise ValueError('Arm mismatch')
        if [r['seed'] for r in entries]!=[1,2,3]:raise ValueError('Supply seeds 1,2,3 in that order')
        if len({r['checkpoint_sha256'] for r in entries})!=3:raise ValueError('Three checkpoint identities are required')
    for j in range(3):
        for key in ('initial_state_sha256','plan_sha256','protocol_sha256'):
            if len({data[a][j][key] for a in data})!=1:raise ValueError(f'Unmatched {key} at seed{j+1}')
    if any([r['frame_id'] for r in entry['frame_rows']]!=ids or [r['session_id'] for r in entry['frame_rows']]!=sessions for entry in all_rows):
        raise ValueError('Different paired frame order/session assignment')
    values={a:np.asarray([[np.nan if f['new_normalized'] is None else f['new_normalized'] for f in r['frame_rows']] for r in data[a]],float) for a in data}
    baseline=np.asarray([np.nan if f['base_normalized'] is None else f['base_normalized'] for f in ref['frame_rows']],float)
    mask=np.isfinite(baseline)
    if any(not np.array_equal(np.isfinite(v),np.broadcast_to(mask,v.shape)) for v in values.values()):
        raise ValueError('Coverage differs; do not silently intersect away failures')
    if not mask.any():raise ValueError('No common observed frame')
    per_seed={a:np.mean(v[:,mask],axis=1) for a,v in values.items()}
    base_mean=float(baseline[mask].mean())
    checks=[]
    for j in range(3):
        h,point=data['H'][j]['result_8'],data['P'][j]['result_8'];base=data['H'][j]['baseline_8']
        checks.append(dict(seed_slot=j+1,head_beats_baseline_1pct=bool(per_seed['H'][j]<=.99*base_mean),
            head_beats_P_1pct=bool(per_seed['H'][j]<=.99*per_seed['P'][j]),
            median_nonworse=bool(h['median_px']<=min(base['median_px'],point['median_px'])+1e-9),
            p90_nonworse=bool(h['p90_px']<=min(base['p90_px'],point['p90_px'])+1e-9),
            good_point_damage_safe=bool(h['good_damage_rate']<=.01)))
    numerical_pass=all(all(v for k,v in r.items() if k!='seed_slot') for r in checks)
    contrasts={};real=ref['role']=='real_dev';s=np.asarray(sessions)[mask]
    means={a:v[:,mask].mean(0) for a,v in values.items()}
    pairs={'H_minus_P':('H','P'),'H_minus_S':('H','S'),'S_minus_P':('S','P'),'H_minus_HA':('H','HA')}
    for name,(a,b) in pairs.items():
        diff=means[a]-means[b]
        contrasts[name]=paired_session_bootstrap(diff,s,alpha=.05 if name=='H_minus_P' else .05/3) if real else {'mean':float(diff.mean()),'ci':None,'reason':'synthetic diagnostic; no independent-session inference claimed'}
    base_diff=means['H']-baseline[mask]
    contrasts['H_minus_baseline']=paired_session_bootstrap(base_diff,s) if real else {'mean':float(base_diff.mean()),'ci':None}
    real_pass=(numerical_pass and contrasts['H_minus_P']['upper']<0 and contrasts['H_minus_baseline']['upper']<0) if real else None
    write_json(args.output,dict(schema='pointline_v4_comparison_1',role=ref['role'],fixed_primary='H versus P and unchanged common backbone output',
        source_sha256={x:sha256(x) for v in paths.values() for x in v},baseline_frame_normalized=base_mean,
        per_seed_frame_normalized={a:v.tolist() for a,v in per_seed.items()},checks=checks,contrasts=contrasts,
        synthetic_progression_signal=numerical_pass if not real else None,real_cached_head_signal=real_pass,
        overall_end_to_end_accuracy_improved=None,official_metric_parity_checked=all(r.get('official_metric_parity_checked') is True for r in all_rows),
        scope='Three HEAD seeds on a fixed joint backbone. P/S are not Hough-free. Real result is reused DEV; no FINAL claim.',
        caution='CLI must separately verify seed IDs, initial/batch trace matching and canonical metric parity; numeric gate is not a launch authorization.'))

if __name__=='__main__':main()
