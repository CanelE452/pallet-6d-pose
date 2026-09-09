"""Synthetic-only margin calibration or fixed-policy evaluation.

Scoring has already finished in score_cache.py. Only THIS stage opens GT.
Output contains both eight-corner and legacy-compatible nine-point summaries.
Repository canonical metric parity still has to be checked by the adapter.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from .cache_io import ExportDataset,sha256,write_json
from .policy import select_layout
from .metrics import error_summary,safety_counts

MARGINS=(None,1.,.5,.25,.1,.05,.025,0.)


def digest_arrays(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def compact(summary):
    return {k:v for k,v in summary.items() if not isinstance(v,np.ndarray)}


def number(value):return float(value) if np.isfinite(value) else None


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('calibrate','evaluate'))
    p.add_argument('--manifest',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True)
    p.add_argument('--policy');p.add_argument('--allow-smoke',action='store_true');args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Do not overwrite a frozen policy/result')
    data=ExportDataset(args.manifest,verify_targets=True);scores=json.loads(Path(args.scores).read_text(encoding='utf8'))
    if scores['manifest_sha256']!=sha256(args.manifest):raise ValueError('Scores refer to another population')
    if scores['stage']!='main' and not args.allow_smoke:raise ValueError('Smoke scores are not experiment evidence')
    records=data.records
    if len(records)!=len(scores['rows']):raise ValueError('Missing or extra scored frame')
    observations=[];targets=[]
    for i,(r,s) in enumerate(zip(records,scores['rows'])):
        if r['frame_id']!=s['frame_id'] or r['observation_sha256']!=s['observation_sha256']:
            raise ValueError('Frame/order/input binding mismatch')
        o=data.observation(i);t=data.supervision(i)
        if o.candidate_valid[0].tolist()!=s['candidate_valid']:raise ValueError('Candidate validity changed after scoring')
        observations.append({k:getattr(o,k).numpy()[0] for k in ('layouts','baseline','point_valid','diagonal','candidate_valid')})
        targets.append({k:v.numpy()[0] for k,v in t.items()})
    obs={k:np.stack([r[k] for r in observations]) for k in observations[0]}
    gt={k:np.stack([r[k] for r in targets]) for k in targets[0]}
    cost=torch.tensor([r['cost'] for r in scores['rows']],dtype=torch.float32)
    layouts=torch.from_numpy(obs['layouts']);valid=torch.from_numpy(obs['candidate_valid'])
    def summary(pred,corners_only=True):
        return error_summary(pred,gt['points'],gt['supervised'],obs['point_valid'],gt['matched'],obs['diagonal'],corners_only=corners_only)
    baseline=summary(obs['baseline']);base_primary=float(np.nanmean(baseline['frame_normalized']))
    if not np.isfinite(base_primary):raise ValueError('No observed GT for evaluation')
    def evaluate(margin):
        pred,ix=select_layout(cost,layouts,valid,margin);pred=pred.numpy();s=summary(pred)
        primary=float(np.nanmean(s['frame_normalized']));safety=safety_counts(baseline['errors'],s['errors'],baseline['observed'])
        if not np.array_equal(s['observed'],baseline['observed']):raise ValueError('Coverage/mask changed')
        eligible=(primary<=base_primary+1e-12 and s['median_px']<=baseline['median_px']+1e-9 and
                  s['p90_px']<=baseline['p90_px']+1e-9 and safety['good_damage_rate']<=.01)
        return dict(margin=margin,primary_frame_normalized=primary,eligible=bool(eligible),changed_frames=int((ix.numpy()!=0).sum()),
                    **compact(s),**safety),s,pred,ix.numpy()
    if args.mode=='calibrate':
        if data.document.get('role')!='calibration':raise ValueError('Use declared synthetic calibration split')
        if not args.allow_smoke and any(r['origin']!='source_synthetic' for r in records):
            raise ValueError('Never calibrate on real images or generated tests')
        rows=[evaluate(m)[0] for m in MARGINS]
        # Identity is first; exact objective ties retain the more conservative earlier rule.
        chosen=rows[0]
        for r in rows[1:]:
            if r['eligible'] and r['primary_frame_normalized']<chosen['primary_frame_normalized']-1e-12:chosen=r
        write_json(output,dict(schema='pointline_v4_policy_1',stage=scores['stage'],arm=scores['arm'],checkpoint_sha256=scores['checkpoint_sha256'],
            calibration_manifest_sha256=sha256(args.manifest),scores_sha256=sha256(args.scores),margin=chosen['margin'],
            margin_units='difference of predicted log1p normalized-error costs; not calibrated probabilities',
            baseline=compact(baseline),candidates=rows,chosen=chosen))
    else:
        if not args.policy:raise ValueError('--policy is required for evaluate')
        policy=json.loads(Path(args.policy).read_text(encoding='utf8'))
        if policy['checkpoint_sha256']!=scores['checkpoint_sha256']:raise ValueError('Policy/checkpoint mismatch')
        if data.document.get('role') not in ('synth_val','real_dev','generated_test'):
            raise ValueError('Declare synth_val or real_dev; FINAL is not authorized here')
        row,s,pred,ix=evaluate(policy['margin']);s9=summary(pred,corners_only=False);b9=summary(obs['baseline'],corners_only=False)
        frame_rows=[]
        for i,r in enumerate(records):
            frame_rows.append(dict(frame_id=r['frame_id'],session_id=r['session_id'],chosen=int(ix[i]),
                base_normalized=number(baseline['frame_normalized'][i]),new_normalized=number(s['frame_normalized'][i]),
                base_mean_px=number(baseline['frame_mean_px'][i]),new_mean_px=number(s['frame_mean_px'][i])))
        write_json(output,dict(schema='pointline_v4_evaluation_1',stage=scores['stage'],arm=scores['arm'],role=data.document['role'],
            seed=scores['seed'],initial_state_sha256=scores['initial_state_sha256'],plan_sha256=scores['plan_sha256'],protocol_sha256=scores['protocol_sha256'],
            manifest_sha256=sha256(args.manifest),scores_sha256=sha256(args.scores),policy_sha256=sha256(args.policy),
            checkpoint_sha256=scores['checkpoint_sha256'],population_fingerprint=digest_arrays(obs['baseline'],gt['points'],gt['supervised'],obs['point_valid'],gt['matched'],obs['diagonal']),
            baseline_8=compact(baseline),result_8=row,baseline_9=compact(b9),result_9=compact(s9),
            frame_rows=frame_rows,selected_coordinates=pred.tolist(),official_metric_parity_checked=False))

if __name__=='__main__':main()
