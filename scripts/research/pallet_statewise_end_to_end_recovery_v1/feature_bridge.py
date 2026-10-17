"""Lossless feature packaging and physical-pose source-alias preservation.

Deduplication keeps one actual pose, which can originate from G13 AND E0,
or G10 AND N0. The first display ID must not erase the original baseline
role in the predeclared eligibility policy. No score, pose or feature scalar
is changed, and no supervision is read by this adapter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from .common import read,rows,save_rows,sha,write


def same_pose(candidate,source,scale):
    pose=candidate.get('actual_pose',candidate)
    return bool(np.max(np.abs(np.asarray(pose['R_physical'])-np.asarray(source['R_physical'])))<=1e-7
                and np.max(np.abs(np.asarray(pose['centroid'])-np.asarray(source['centroid'])))<=1e-7*scale)


def run(inputs,features,output):
    before=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frames={f['id']:f for f in rows(inputs)}
    values=[];alias_repairs=[];changed_scalars=0
    for row in rows(features):
        frame=frames[row['id']]
        scale=max(float(np.linalg.norm(np.ptp(np.asarray(f['native_xyz']),axis=0)))for f in frame['families'].values())
        baselines=[p for p in frame['candidates']if p.get('source')in('N0','E0')]
        row['object_scale_m']=scale
        for c in row['candidates']:
            aliases=[p['source']for p in baselines if same_pose(c,p,scale)]
            c['source_baseline_aliases']=sorted(set(aliases))
            old=bool(c.get('eligible',True))
            if aliases:
                c['eligible']=True
                c['baseline_alternative_not_joint_certificate']=True
            if bool(c.get('eligible',True))!=old:
                alias_repairs.append(dict(id=row['id'],candidate_id=c['candidate_id'],aliases=aliases))
        values.append(row)
    save_rows(out/'HEAD_FEATURES.jsonl.gz',values)
    receipt=dict(schema='LOSSLESS_FEATURE_SOURCE_ALIAS_BRIDGE_V1',rows=len(values),
                 input_frames_sha256=sha(inputs),original_features_sha256=sha(features),
                 head_features_sha256=sha(out/'HEAD_FEATURES.jsonl.gz'),source_sha256=sha(__file__),
                 feature_scalars_changed=changed_scalars,actual_poses_changed=0,
                 preserved_baseline_eligibility_repairs=alias_repairs,
                 alias_match_tolerance='same strict original numerical dedup: R max1e-7, t max1e-7*object_scale_m; no C2 phase collapse',
                 reference_reads=0,before_training_label_reads=True,before_PRIMARY_scoring=True,
                 elapsed_seconds=time.monotonic()-before)
    write(out/'FEATURE_SEAL.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(run(a.inputs,a.features,a.output))
