"""Frozen Replay -> all-eight-corner median LOO -> whole-image keep/drop."""
import argparse
import json

import cv2
import numpy as np

from . import frozen_n2_frame_filter as N
from . import core as P

DOC=P.DOC/'all8_replay_frame_filter'
RAW=P.RAW/'all8_replay_frame_filter'


def apply():
    P.verify()
    cv2.setNumThreads(1)
    source=P.N.RAW/'PREDICTIONS.json'
    split=P.ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json'
    green=P.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
    registry_path=P.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    thresholds=P.read(N.FILTER)
    protocol=dict(arm='POSEFIX_RAW / frozen Replay',
        policy='Raw R0 stage1 confidence -> all8 finite Replay corners regardless of their confidence -> original median LOO<=0.05 -> keep/drop whole example',
        excluded='No learned selector, N2 replacement, coordinate change, flip test, threshold sweep or retraining',
        invalid='Any missing/nonfinite corner among first8 rejects whole image',
        threshold=thresholds['geometry_thresholds']['tau_remove'],
        source=P.bound(source),metadata=[P.bound(p) for p in (split,green,registry_path,N.FILTER)],
        code=[P.bound(__file__),P.bound(N.__file__),
              P.bound(P.ROOT/'scripts/self_training_yolo/pseudo_label_filters.py')],
        evaluation_reused=True,GT_used_for_selection=False,model_modified=False,
        runtime='Cached GPU predictions; CPU OpenCV only')
    P.freeze(DOC/'PROTOCOL.json',protocol)
    if (DOC/'OUTPUTS_LOCK.json').exists():
        for b in P.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:P.verify_binding(b)
        print('ALL8_REPLAY_FRAME_OUTPUTS_ALREADY_FROZEN',flush=True)
        return
    meta={'DEV72':{r['id']:r for r in P.read(split)['evaluation']},
          'GREEN150':{r['id']:r for r in P.read(green)['records']}}
    registry={r['object_type']:r['physical_dimensions_m'] for r in P.read(registry_path)['objects']}
    accepted,decisions={},{}
    for ds,rows in P.read(source).items():
        accepted[ds],decisions[ds]=[],[]
        for row in rows:
            pred=row['predictions']['POSEFIX_RAW'];before=json.dumps(pred,sort_keys=True,allow_nan=False)
            raw=N.top(row['predictions']['R0'])
            confidence=N.stage1({'top1':None if raw is None else dict(box_conf=raw['score'],
                keypoints_xy=raw['keypoints_xy'],keypoints_conf=raw['keypoints_conf'])},thresholds)
            q=np.asarray(N.top(pred)['keypoints_xy'],float)
            valid=np.isfinite(q).all(-1)&~(q==-1).all(-1)
            finite8=bool(valid[:8].all());loo,error=None,None
            if confidence and finite8:
                m=meta[ds][row['id']]
                K=np.asarray(m['K'] if ds=='DEV72' else m['source_K'],float)
                kind=m['object_type'] if ds=='DEV72' else 'plastic_standard_110x110x15'
                try:
                    value=N.geometry_scores(q,valid,K,registry[kind])['s_remove']
                    loo=float(value) if np.isfinite(value) else None
                except cv2.error as exc:error=str(exc)
            unchanged=N.keep_unchanged(pred,confidence and finite8,loo,protocol['threshold'])
            assert json.dumps(pred,sort_keys=True,allow_nan=False)==before
            kept=unchanged is not None
            if kept:
                assert json.dumps(unchanged,sort_keys=True,allow_nan=False)==before
                accepted[ds].append(dict(id=row['id'],image=row['image'],prediction=unchanged))
            decisions[ds].append(dict(id=row['id'],confidence_pass=confidence,finite8=finite8,
                s_remove_all8=loo,keep=kept,reason='accepted' if kept else
                ('raw_confidence' if not confidence else ('missing_corner' if not finite8 else 'all8_LOO')),solver_error=error))
        print('ALL8_REPLAY_WHOLE_IMAGE',ds,len(accepted[ds]),'/',len(rows),flush=True)
    assert sum(map(len,decisions.values()))==222
    P.freeze(RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json',accepted)
    P.freeze(RAW/'DECISIONS.json',decisions)
    P.freeze(DOC/'OUTPUTS_LOCK.json',dict(complete=True,decisions_frozen_before_GT_scoring=True,
        protocol=P.bound(DOC/'PROTOCOL.json'),artifacts=[P.bound(RAW/n) for n in ('ACCEPTED_UNCHANGED_PREDICTIONS.json','DECISIONS.json')]))


def score():
    lock=P.read(DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']+[lock['protocol']]:P.verify_binding(b)
    protocol=P.read(DOC/'PROTOCOL.json')
    for b in [protocol['source'],*protocol['metadata'],*protocol['code']]:P.verify_binding(b)
    metric_path=P.N.RAW/'PER_FRAME_METRICS.json'
    metrics=P.read(metric_path);decisions=P.read(RAW/'DECISIONS.json')
    summaries={}
    for mode in ('DEV72','GREEN150_MANUAL'):
        ds='DEV72' if mode=='DEV72' else 'GREEN150'
        rows=decisions[ds];ids={r['id'] for r in rows if r['keep']}
        confidence_ids={r['id'] for r in rows if r['confidence_pass']}
        replay=metrics[mode]['POSEFIX_RAW']
        assert {r['id'] for r in replay}=={r['id'] for r in rows}
        summaries[mode]=dict(total_images=len(rows),confidence_pass=len(confidence_ids),
            all8_additional_rejected=len(confidence_ids)-len(ids),kept_images=len(ids),
            total_rejected=len(rows)-len(ids),retention=len(ids)/len(rows),
            all_Replay=N.subset_summary(replay),
            confidence_only_Replay=N.subset_summary([r for r in replay if r['id'] in confidence_ids]),
            accepted_Replay=N.subset_summary([r for r in replay if r['id'] in ids]),
            rejected_Replay=N.subset_summary([r for r in replay if r['id'] not in ids]),
            same_accepted_N2=N.subset_summary([r for r in metrics[mode]['A_N2'] if r['id'] in ids]),
            same_accepted_R0=N.subset_summary([r for r in metrics[mode]['R0'] if r['id'] in ids]))
    result=dict(complete=True,summary=summaries,metrics_source=P.bound(metric_path),
        output_lock=P.bound(DOC/'OUTPUTS_LOCK.json'),coordinates_changed=0,
        conditional_subset_accuracy=True,thresholds_tuned=False,self_training=False)
    P.freeze(DOC/'RESULTS.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['apply','score'])
    args=parser.parse_args();apply() if args.stage=='apply' else score()
