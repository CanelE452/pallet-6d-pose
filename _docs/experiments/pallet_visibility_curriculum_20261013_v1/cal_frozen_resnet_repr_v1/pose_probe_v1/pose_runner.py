"""Pose inference uses sealed RGB/frozen features, then CAL-only scoring."""
import argparse
import collections
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from code_snapshot import common as C
from code_snapshot.curriculum import solve_curriculum
from code_snapshot.run import no_annotations
from code_snapshot.scoring import pose_metric

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(C.finite(value),indent=2,allow_nan=False)+'\n')


def rows(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def infer():
    p=read(HERE/'PROTOCOL.json');assert not (HERE/'PREDICTIONS.jsonl.gz').exists()
    for path,digest in ((p['inputs'],p['inputs_sha256']),(p['baseline_predictions'],p['baseline_predictions_sha256']),
                        (p['head_probabilities'],p['head_probabilities_sha256'])):
        assert sha(path)==digest
    for model in p['model_bindings']:assert sha(model['path'])==model['sha256']
    for name,digest in p['source_sha256'].items():assert sha(HERE/'code_snapshot'/name)==digest
    frames=read(p['inputs'])['frames'];assert len(frames)==76 and all(f['partition']=='CAL' for f in frames)
    base={(r['id'],r['method']):r for r in rows(p['baseline_predictions'])}
    heads={(r['id'],r['corner'],r['family']):r for r in read(p['head_probabilities'])}
    write(HERE/'INFERENCE_CODE_BINDING.json',dict(runner_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.json'),before_pose_inference=True))
    result=[]
    with no_annotations():
        for frame in frames:
            ident=frame['id'];initial=frame['initial_poses']['N3_SUBPIX'];old=base[ident,'SEGMENTED_RGB_CURRICULUM']
            q=np.asarray(old['original_points'],float);H=np.asarray(old['self_hidden'],bool)
            xyz,K=C.cuboid(*initial['cf_extents']),np.asarray(frame['K'])
            R,t=np.asarray(initial['R_cf']),np.asarray(initial['centroid']);Q=R.T@np.asarray(initial['R_physical'])
            assert np.array_equal(H,C.self_hidden(xyz,R,t))
            for arm in p['arms']:
                family='RGB26_PLUS_F4_PCA32' if arm['family']=='F4' else 'RGB26_PLUS_F4F8_PCA32'
                app=deepcopy(old['appearance']);threshold=arm['threshold'];visible_ids=[];scores={}
                for k in range(8):
                    record=heads.get((ident,k,family))
                    if record is None:
                        continue
                    assert not H[k] and app['states'][k] not in ('SELF_OCCLUDED','OUT_OF_FRAME')
                    probability=record['full_CAL_probability'] if arm['style']=='FULL_CAL' else record['heldout_session_probability']
                    app['visibility_score'][k]=probability;scores[str(k)]=probability
                    if probability>=threshold:
                        app['states'][k]='VISIBLE_BOUNDARY' if app['boundary_score'][k]>=.35 else 'VISIBLE_INTERIOR'
                        app['local_accuracy_score'][k]=max(app['local_accuracy_score'][k],.12);visible_ids.append(k)
                    else:
                        app['states'][k]='UNKNOWN'
                config=dict(p['curriculum_base'],visible_threshold=threshold,easy_visibility=max(.45,threshold))
                solution=solve_curriculum(q,xyz,K,R,t,H,app,config)
                predicted=dict(R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),
                    centroid=solution['t'],cf_extents=initial['cf_extents'])
                result.append(dict(id=ident,session=frame['session'],arm=arm['name'],style=arm['style'],threshold=threshold,
                    initial_self_hidden=H.tolist(),original_points=q.tolist(),fitted_measurements=q.tolist(),
                    head_probabilities=scores,head_admitted_ids=visible_ids,solution=solution,actual_pose=predicted,
                    provenance=dict(H_unchanged=True,coordinates_unchanged=True,boundary_accuracy_source='frozen_SEGMENTED_RGB_CURRICULUM',
                     head_RGB_source='sealed original_RGB26_with_ImageNet_context',projected_points_used_for_fit=False,
                     GT_input=False,TEST_accessed=False,FINAL_accessed=False)))
    with gzip.open(HERE/'PREDICTIONS.jsonl.gz','wt') as stream:
        for row in result:stream.write(json.dumps(C.finite(row),separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_pose_scoring=True,GT_read_canary='PASS',rows=len(result),
        sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),protocol_sha256=sha(HERE/'PROTOCOL.json'),runner_sha256=sha(__file__),
        TEST_accessed=False,FINAL_accessed=False))
    print('REPR_POSE_SEALED',len(result),flush=True)


def distribution(value):
    x=np.asarray(value,float);return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),P90=float(np.quantile(x,.9)))


def score():
    p,seal=read(HERE/'PROTOCOL.json'),read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_pose_scoring'] and seal['GT_read_canary']=='PASS'
    assert seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz') and seal['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    oracle=read(HERE.parent.parent/'ORACLE_CAL_INPUTS.json')['frames'];assert len(oracle)==76 and all(f['partition']=='CAL' for f in oracle)
    oracle={f['id']:f for f in oracle};base={(r['id'],r['method']):r for r in rows(p['baseline_predictions'])}
    scored=rows(HERE/'PREDICTIONS.jsonl.gz');groups=collections.defaultdict(list)
    for row in scored:
        truth=oracle[row['id']]['target']['truth'];row['metric']=pose_metric(row['actual_pose'],truth)
        row['baseline_metric']=pose_metric(base[row['id'],'SEGMENTED_RGB_CURRICULUM']['actual_pose'],truth)
        row['frozen_initial_metric']=pose_metric(base[row['id'],'INITIAL']['actual_pose'],truth)
        groups[row['arm']].append(row)
    rng=np.random.default_rng(20261013);sessions=sorted({r['session'] for r in scored});draws=rng.integers(0,len(sessions),(10000,len(sessions)))
    summary={}
    for arm,value in groups.items():
        report=dict(frames=len(value),accepted=sum(r['solution']['accepted'] for r in value),
            metrics={key:distribution([r['metric'][key] for r in value]) for key in ('translation_cm','rotation_deg','ADDsym_cm')},
            success_5cm_5deg=float(np.mean([r['metric']['success_5cm_5deg'] for r in value])),
            reasons=dict(collections.Counter(r['solution']['reason'] for r in value)),paired_vs_SEGMENTED={})
        for key in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
            delta=np.asarray([float(r['metric'][key])-float(r['baseline_metric'][key]) for r in value])
            sums=np.asarray([sum(d for d,r in zip(delta,value) if r['session']==s) for s in sessions]);counts=np.asarray([sum(r['session']==s for r in value) for s in sessions])
            boot=sums[draws].sum(1)/counts[draws].sum(1)
            report['paired_vs_SEGMENTED'][key]=dict(mean_difference=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),
                improved=int((delta>0).sum()) if key.startswith('success') else int((delta<0).sum()),
                worsened=int((delta<0).sum()) if key.startswith('success') else int((delta>0).sum()))
        summary[arm]=report
        print(arm,'accepted',report['accepted'],'T',report['metrics']['translation_cm'],'R',report['metrics']['rotation_deg'],
              'success',report['success_5cm_5deg'],'paired',report['paired_vs_SEGMENTED'],flush=True)
    with gzip.open(HERE/'SCORED_ROWS.jsonl.gz','wt') as stream:
        for row in scored:stream.write(json.dumps(C.finite(row),separators=(',',':'))+'\n')
    write(HERE/'RESULTS.json',dict(schema='CAL_FROZEN_REPR_POSE_V1',CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        frames=76,summary=summary,baseline_SEGMENTED=dict(translation_cm=116.29147334738165,rotation_deg=19.290009317834865,success_5cm_5deg=37/76),
        predictions_sha256=seal['sha256'],limitations=p['limits']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['infer','score']);args=parser.parse_args()
    infer() if args.command=='infer' else score()
