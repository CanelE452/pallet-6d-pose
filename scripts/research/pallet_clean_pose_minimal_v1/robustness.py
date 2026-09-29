"""Descriptive paired recording robustness; never tunes or selects predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import common as C
from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M

KEYS = ('translation_cm', 'rotation_deg')


def cluster_interval(before, after, rows, repeats=2000, seed=20260929):
    """Resample recording clusters, not 99 purportedly independent frames."""
    groups = {rec: [r['id'] for r in rows if r['recording'] == rec]
              for rec in sorted({r['recording'] for r in rows})}
    rng = np.random.default_rng(seed)
    names = list(groups)
    samples = {k: [] for k in KEYS}
    undefined = {k: 0 for k in KEYS}
    for _ in range(repeats):
        ids = [fid for index in rng.integers(0, len(names), len(names)) for fid in groups[names[index]]]
        for key in KEYS:
            medians = [M.extended_quantile([data[fid][key] if data[fid]['available'] else float('inf')
                                            for fid in ids], .5) for data in (before, after)]
            if not all(np.isfinite(v) for v in medians):
                undefined[key] += 1
                continue
            samples[key].append(medians[1] - medians[0])
    return dict(repeats=repeats, seed=seed, recording_count=len(names),
        recording_sizes={k:len(v) for k,v in groups.items()},
        intervals={k:dict(percentile95=np.quantile(v,[.025,.975]).tolist() if v else None,
                           finite_resamples=len(v), undefined_resamples=undefined[k]) for k,v in samples.items()},
        limitation='Small, unequal, reused DEV recording clusters; descriptive uncertainty, not independent confirmation or an acceptance gate.')


def select_cases(before, after, rows, count=2):
    common = [r['id'] for r in rows if before[r['id']]['available'] and after[r['id']]['available']]
    def delta(fid, key):
        return after[fid][key] - before[fid][key]
    improved = [fid for fid in common if all(delta(fid,k)<0 for k in KEYS)]
    worsened = [fid for fid in common if all(delta(fid,k)>0 for k in KEYS)]
    selected = dict(
        both_improved=sorted(improved,key=lambda fid:(delta(fid,KEYS[0]),fid))[:count],
        both_worsened=sorted(worsened,key=lambda fid:(-delta(fid,KEYS[0]),fid))[:count],
        largest_final_T=sorted(common,key=lambda fid:(-after[fid][KEYS[0]],fid))[:count],
        largest_final_R=sorted(common,key=lambda fid:(-after[fid][KEYS[1]],fid))[:count])
    metadata = {r['id']:r for r in rows}
    output = {}
    for group, ids in selected.items():
        output[group] = [dict(id=fid,recording=metadata[fid]['recording'],severity=metadata[fid]['severity'],
            before={k:before[fid][k] for k in KEYS}, after={k:after[fid][k] for k in KEYS},
            delta={k:delta(fid,k) for k in KEYS}) for fid in ids]
    output['failure_frames'] = [dict(id=r['id'],before_valid=before[r['id']]['available'],
                                      after_valid=after[r['id']]['available']) for r in rows
                               if not before[r['id']]['available'] or not after[r['id']]['available']]
    return output


def analyze(metrics, rows, pairs):
    natural = [r for r in rows if r['severity'] != 'CLEAN']
    assert len(natural) == 99 and len(rows) == 128
    result = {}
    for before, after in pairs:
        a,b = metrics[before],metrics[after]
        assert set(a) == set(b) == {r['id'] for r in rows}
        contrast = after + '-minus-' + before
        result[contrast] = dict(before=before,after=after,
            NATURAL99=M.paired(a,b,[r['id'] for r in natural]),
            cluster_bootstrap=cluster_interval(a,b,natural),
            leave_one_recording_out={rec:M.paired(a,b,[r['id'] for r in natural if r['recording']!=rec])
                for rec in sorted({r['recording'] for r in natural})},
            natural_cases=select_cases(a,b,natural),
            clean_cases=select_cases(a,b,[r for r in rows if r['severity']=='CLEAN']))
    return result


def run(metric_file, metadata_file, pairs_file, output):
    output = C.DOC / output
    if output.exists():
        for binding in C.read(output)['inputs']:
            C.verify(binding)
        print('ROBUSTNESS_REUSED',output.name)
        return
    metrics,rows,pairs = [C.read(p) for p in (metric_file,metadata_file,pairs_file)]
    result = analyze(metrics,rows,pairs)
    value = dict(created_at=C.now(),analysis_scope='Posthoc reused DEV; no new predictions or selection',
        comparisons=result,inputs=[C.bind(p) for p in (metric_file,metadata_file,pairs_file,Path(__file__),C.DOC/'ANALYSIS_LOCK.md')],
        new_fits=0,GPU_seconds=0)
    C.save(output,M.clean(value),True)
    print('ROBUSTNESS_COMPLETE',output.name,len(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--output',default='ROBUSTNESS.json')
    args=parser.parse_args()
    run(args.metrics,args.metadata,args.pairs,args.output)
