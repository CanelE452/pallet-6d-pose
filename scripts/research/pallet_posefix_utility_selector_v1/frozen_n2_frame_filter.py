"""Keep/drop whole examples of frozen N2; NEVER modify predicted coordinates."""
import argparse
import copy
import json

import cv2
import numpy as np

from . import core as P
from scripts.self_training_yolo.pseudo_label_filters import geometry_scores
from scripts.research.pallet_posefix_corner_gate_v1.evaluate import top
from scripts.research.pallet_real_refiner_twostage_v1.run import stage1

DOC = P.DOC/'frozen_n2_frame_filter'
RAW = P.RAW/'frozen_n2_frame_filter'
FILTER = P.ROOT/'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json'


def keep_unchanged(prediction, confidence_pass, loo, threshold):
    """Whole prediction passes unchanged or is absent; no fallback candidate."""
    keep = bool(confidence_pass and loo is not None and np.isfinite(loo) and loo <= threshold)
    return copy.deepcopy(prediction) if keep else None


def apply():
    P.verify()
    source = P.N.RAW/'PREDICTIONS.json'
    split = P.ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json'
    snapshot = P.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
    registry_path = P.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    thresholds = P.read(FILTER)
    protocol = dict(purpose='N2 output fixed; whole-image confidence + historical median LOO keep/drop',
        source=P.bound(source),metadata=[P.bound(p) for p in (split,snapshot,registry_path,FILTER)],
        code=[P.bound(__file__),P.bound(P.ROOT/'scripts/self_training_yolo/pseudo_label_filters.py'),
              P.bound(P.ROOT/'scripts/research/pallet_real_refiner_twostage_v1/run.py')],
        threshold=thresholds['geometry_thresholds']['tau_remove'],
        confidence='Original raw R0 stage1: score>=0.85 and >=6/8 finite corners with kpconf>=0.5',
        LOO='Original geometry_scores: median held-out residual / projected cuboid diagonal; min across registered W/D hypotheses',
        flip_used=False,not_identical_to_historical_LOO_plus_flip=True,
        coordinates='Exact saved N2 prediction; rejected images omitted entirely. No learned selector, Replay, per-corner replacement, or N2 fallback.',
        model_training=False,pseudo_label_training_export=False,evaluation_used_only_for_audit=True,
        decisions_use_GT=False,threshold_tuning=False,independent_test=False,
        runtime='Stored GPU N2 predictions; CPU OpenCV geometry only')
    P.freeze(DOC/'PROTOCOL.json',protocol)
    if (DOC/'OUTPUTS_LOCK.json').exists():
        for binding in P.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:
            P.verify_binding(binding)
        print('FROZEN_N2_FRAME_SELECTION_ALREADY_COMPLETE',flush=True)
        return
    meta={'DEV72':{r['id']:r for r in P.read(split)['evaluation']},
          'GREEN150':{r['id']:r for r in P.read(snapshot)['records']}}
    registry={r['object_type']:r['physical_dimensions_m'] for r in P.read(registry_path)['objects']}
    decisions,accepted = {},{}
    for ds,rows in P.read(source).items():
        decisions[ds],accepted[ds]=[],[]
        for row in rows:
            pred=row['predictions']['A_N2']
            original_json=json.dumps(pred,sort_keys=True,allow_nan=False)
            raw=top(row['predictions']['R0'])
            confidence_pass=stage1({'top1':None if raw is None else dict(
                box_conf=raw['score'],keypoints_xy=raw['keypoints_xy'],keypoints_conf=raw['keypoints_conf'])},thresholds)
            loo,error=None,None
            if confidence_pass:
                a=top(pred); q=np.asarray(a['keypoints_xy'],float)
                valid=(np.asarray(raw['keypoints_conf'])>=.5)&np.isfinite(q).all(-1)
                assert not np.any(valid[:8] & (q[:8]==-1).all(-1))
                m=meta[ds][row['id']]
                K=np.asarray(m['K'] if ds=='DEV72' else m['source_K'],float)
                kind=m['object_type'] if ds=='DEV72' else 'plastic_standard_110x110x15'
                try:
                    value=geometry_scores(q,valid,K,registry[kind])['s_remove']
                    loo=float(value) if np.isfinite(value) else None
                except cv2.error as exc:
                    error=str(exc)
            unchanged=keep_unchanged(pred,confidence_pass,loo,protocol['threshold'])
            assert json.dumps(pred,sort_keys=True,allow_nan=False)==original_json
            kept=unchanged is not None
            if kept:
                assert json.dumps(unchanged,sort_keys=True,allow_nan=False)==original_json
                accepted[ds].append(dict(id=row['id'],image=row['image'],prediction=unchanged))
            decisions[ds].append(dict(id=row['id'],confidence_pass=confidence_pass,s_remove=loo,
                keep=kept,reason='accepted' if kept else ('raw_confidence' if not confidence_pass else 'LOO'),solver_error=error))
        print(ds,'whole images kept',len(accepted[ds]),'/',len(rows),flush=True)
    P.freeze(RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json',accepted)
    P.freeze(RAW/'DECISIONS.json',decisions)
    P.freeze(DOC/'OUTPUTS_LOCK.json',dict(complete=True,all222_decisions_fixed_before_scoring=True,
        protocol=P.bound(DOC/'PROTOCOL.json'),artifacts=[P.bound(RAW/n) for n in ('ACCEPTED_UNCHANGED_PREDICTIONS.json','DECISIONS.json')]))


def subset_summary(rows):
    errors=np.asarray([e for r in rows for e in r['errors']],float)
    observed=[e for r in rows for e in r['observed_errors']]
    return dict(frames=len(rows),corners=len(errors),correct10=int((errors<=10).sum()),
        PCK10=float((errors<=10).mean()) if len(errors) else None,
        matched_frames=sum(r['matched'] for r in rows),
        matched_mean_px=float(np.mean(observed)) if observed else None,
        matched_P90_px=float(np.percentile(observed,90)) if observed else None)


def score():
    lock=P.read(DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']+[lock['protocol']]:P.verify_binding(b)
    protocol=P.read(DOC/'PROTOCOL.json')
    for b in [protocol['source'],*protocol['metadata'],*protocol['code']]:P.verify_binding(b)
    # GT-derived saved metrics are first accessed only AFTER whole-image choices freeze.
    metric_path=P.N.RAW/'PER_FRAME_METRICS.json'
    metrics=P.read(metric_path);decisions=P.read(RAW/'DECISIONS.json')
    results={}
    for mode in ('DEV72','GREEN150_MANUAL'):
        ds='DEV72' if mode=='DEV72' else 'GREEN150'
        by_id={r['id']:r for r in metrics[mode]['A_N2']}
        keep_ids=[r['id'] for r in decisions[ds] if r['keep']]
        reject_ids=[r['id'] for r in decisions[ds] if not r['keep']]
        selected=[by_id[i] for i in keep_ids]
        results[mode]=dict(all=subset_summary(list(by_id.values())),accepted=subset_summary(selected),
            rejected=subset_summary([by_id[i] for i in reject_ids]),
            retention=len(keep_ids)/len(by_id),confidence_pass=sum(r['confidence_pass'] for r in decisions[ds]),
            accepted_ids=keep_ids,rejected_ids=reject_ids,
            frames_all_evaluated_corners_within10=sum(all(e<=10 for e in r['errors']) for r in selected),
            same_accepted_images_R0=subset_summary([r for r in metrics[mode]['R0'] if r['id'] in set(keep_ids)]))
    output=dict(complete=True,results=results,metrics_source=P.bound(metric_path),
        output_lock=P.bound(DOC/'OUTPUTS_LOCK.json'),coordinate_changes=0,
        accepted_PCK_is_conditional_not_full_population=True,self_training=False)
    P.freeze(DOC/'RESULTS.json',output)
    print(json.dumps({k:{n:v for n,v in r.items() if not n.endswith('_ids')} for k,r in results.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['apply','score'])
    args=parser.parse_args();apply() if args.stage=='apply' else score()
