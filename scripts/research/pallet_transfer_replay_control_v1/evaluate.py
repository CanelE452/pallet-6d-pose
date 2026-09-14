"""Evaluate the fixed R0+12 panel on target, negatives, and source retention."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics import YOLO

from contracts import pck_counts
from runtime import ROOT,RAW,DOC,R0,atomic_json,sha

EVAL_CODE=ROOT/'scripts/research/pallet_active_learning_v1/simulation_evaluate.py'
spec=importlib.util.spec_from_file_location('replay_old_eval',EVAL_CODE)
OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
P=OLD.P; E=P.E
OLD.RAW=RAW; OLD.DOC=DOC
NAMES=['R0',*[f'{arm}_seed{seed}' for seed in (1,2,3) for arm in ('T8_FULL','T8_QUARTER','REPLAY','T32_COMPUTE')]]


def read(path):return json.loads(Path(path).read_text())


def pair():
    full=P.population(); split=read(ROOT/'_docs/experiments/pallet_active_learning_v1/retrospective_v1/SPLIT.json')
    ids={r['frame_id'] for r in split['evaluation']}
    result=SimpleNamespace(ready=True,positive=SimpleNamespace(items=[i for i in full.positive.items if i.frame_id in ids]),negative=full.negative)
    assert len(result.positive.items)==145 and len(result.negative.items)==2689
    return result,split


def checkpoint(name):return R0 if name=='R0' else RAW/'runs'/name/'last.pt'


def cache_predictions(name,population):
    out=RAW/'evaluation'/name; out.mkdir(parents=True,exist_ok=True); path=out/'PREDICTIONS.json'
    ck=checkpoint(name)
    if path.exists():
        value=read(path);assert value['complete'] and value['weights_sha256']==sha(ck);return path
    if name=='R0':
        original=read(ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json')
        assert original['complete'] and original['weights_sha256']==sha(R0)
        keys={P.canonical_key(item.image) for item in [*population.positive.items,*population.negative.items]}
        frames={key:original['frames'][key] for key in keys}
        provenance=dict(reused_canonical_cache=True,source_sha256=sha(ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'))
    else:
        predictor=E._UltralyticsPredictor(ck,'0');frames={}
        items=[*population.positive.items,*population.negative.items]
        for index,item in enumerate(items):
            values=predictor.predict(ROOT/item.image)
            frames[P.canonical_key(item.image)]=[dict(score=float(s),box_xyxy=b.tolist(),keypoints_xy=k.tolist() if k is not None else None) for s,b,k in values]
            if (index+1)%500==0:print(name,'target/negative inference',index+1,'/',len(items),flush=True)
        del predictor;torch.cuda.empty_cache();provenance=dict(reused_canonical_cache=False)
    atomic_json(path,dict(schema_version='paper_cached_predictions_v1',complete=True,model=name,
        weights_sha256=sha(ck),recipe=dict(reflect101_pad_px=100,imgsz=640,confidence_floor=.001,
        selection='highest score, no GT',coordinates='original unpadded image'),frames=frames,**provenance))
    return path


def target_metrics(name,population,split,cache):
    targets={i.frame_id:E._legacy_forbidden_target(i) for i in population.positive.items}
    collected=E._collect_predictions(population,E._CachedPredictor(cache),validated_targets=targets)
    _,candidates,top=collected; canonical=E._evaluate_2d_collected(population,*collected)
    sessions={r['frame_id']:r['capture_session'] for r in split['evaluation']}
    rows={}; total={str(t):0 for t in (5.,10.,20.)};denominator=0;matched_count=0;padding=0
    positive_scores=[];negative_scores=[]
    for item in population.positive.items:
        target=targets[item.frame_id]; pred=top.get(item.frame_id)
        matched=bool(pred is not None and pred.target_iou is not None and pred.target_iou>=.5 and
            pred.keypoints_xy is not None and pred.keypoints_xy.shape==(9,2) and np.isfinite(pred.keypoints_xy).all())
        counts=pck_counts(target.keypoints_xy,target.keypoint_supervision_mask,pred.keypoints_xy if pred is not None else None,matched)
        denominator+=counts['denominator']; matched_count+=matched
        for key,value in counts['numerators'].items():total[key]+=value
        errors=(np.linalg.norm(pred.keypoints_xy-target.keypoints_xy,axis=1)[target.keypoint_supervision_mask].tolist() if matched else [])
        score=float(pred.score) if pred is not None else 0.; positive_scores.append(score)
        image=ROOT/item.image
        from PIL import Image
        width,height=Image.open(image).size
        padding_only=bool(pred is not None and (pred.box_xyxy[2]<=0 or pred.box_xyxy[3]<=0 or pred.box_xyxy[0]>=width or pred.box_xyxy[1]>=height))
        padding+=padding_only
        rows[item.frame_id]=dict(kind='positive',image=item.image,session=sessions[item.frame_id],
            denominator=counts['denominator'],numerators=counts['numerators'],matched=matched,
            score=score,target_iou=float(pred.target_iou) if pred is not None else None,errors_px=errors,
            padding_only_top1=padding_only,pose_frame_id=None)
    for item in population.negative.items:
        pred=top.get(item.frame_id);score=float(pred.score) if pred is not None else 0.;negative_scores.append(score)
        rows[item.frame_id]=dict(kind='negative',image=item.image,session=None,denominator=0,numerators={str(t):0 for t in (5.,10.,20.)},
            matched=False,score=score,target_iou=None,errors_px=[],padding_only_top1=None,pose_frame_id=None)
    ranking=OLD.ranking(np.asarray(positive_scores),np.asarray(negative_scores))
    fixed={str(t):dict(count=int(np.sum(np.asarray(negative_scores)>=t)),fraction=float(np.mean(np.asarray(negative_scores)>=t))) for t in (.001,.25,.5,.85)}
    geometry_errors=np.concatenate([np.asarray(r['errors_px']) for r in rows.values() if r['kind']=='positive' and r['errors_px']])
    frame_means=[np.mean(r['errors_px']) for r in rows.values() if r['kind']=='positive' and r['errors_px']]
    metrics=dict(name=name,positive_frames=145,negative_frames=2689,supervised_points=denominator,
        ALL_GT_PCK={key:total[key]/denominator for key in total},ALL_GT_numerators=total,
        detection_match_rate=matched_count/145,matched_frames=matched_count,padding_only_top1=padding,
        common_candidate_metrics_pending=True,individual_matched=dict(points=len(geometry_errors),median_px=float(np.median(geometry_errors)),
            p90_px=float(np.quantile(geometry_errors,.9)),frame_mean_px=float(np.mean(frame_means)),gross20=float(np.mean(geometry_errors>20))),
        canonical_2d=canonical,negative=dict(**ranking,fixed_thresholds=fixed))
    return metrics,rows,top


def pose_metrics(name,frames):
    # The old wrapper is reused with globals redirected into this new experiment.
    result=OLD.pose_evaluation(RAW/'evaluation'/name,frames,name)
    binding=read(RAW/'evaluation_inputs'/'FRAME_ID_BINDING.json')
    mapping={r['evaluation_frame_id']:r['pose_frame_id'] for r in binding}
    return result,mapping


def source_target(row):
    assert len(row['targets'])==1
    target=row['targets'][0];h,w=row['prepared_shape_hw'];box=np.asarray(target['box_xywh_normalized'],float)
    cx,cy,bw,bh=box*np.asarray([w,h,w,h]);xyxy=np.asarray([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2])
    kpt=np.asarray(target['keypoints_normalized'],float);xy=kpt[:,:2]*np.asarray([w,h]);mask=kpt[:,2]!=0
    return xyxy,xy,mask


def source_predictions(name):
    out=RAW/'source_evaluation'/name;out.mkdir(parents=True,exist_ok=True);cache=out/'PREDICTIONS.json';ck=checkpoint(name)
    if cache.exists():value=read(cache);assert value['weights_sha256']==sha(ck);return value
    records=read(DOC/'SOURCE_RETENTION_SPLIT.json')['records'];model=YOLO(str(ck),task='pose');frames={}
    for index,row in enumerate(records):
        result=model.predict(str(ROOT/row['image']),conf=.001,imgsz=640,device='0',verbose=False)[0]
        values=[]
        if result.boxes is not None:
            scores=result.boxes.conf.detach().cpu().numpy();boxes=result.boxes.xyxy.detach().cpu().numpy()
            points=result.keypoints.xy.detach().cpu().numpy() if result.keypoints is not None else None
            values=[dict(score=float(scores[j]),box_xyxy=boxes[j].tolist(),keypoints_xy=points[j].tolist() if points is not None else None) for j in range(len(scores))]
        frames[row['id']]=values
        if (index+1)%128==0:print(name,'source inference',index+1,'/',len(records),flush=True)
    del model;torch.cuda.empty_cache();value=dict(complete=True,model=name,weights_sha256=sha(ck),
        recipe=dict(input='already prepared reflect-padded source image',additional_padding=0,imgsz=640,confidence_floor=.001,
                    coordinates='prepared image pixels'),frames=frames);atomic_json(cache,value);return value


def source_metrics(name,predictions):
    records=read(DOC/'SOURCE_RETENTION_SPLIT.json')['records'];rows={}; candidates=[];total={str(t):0 for t in (5.,10.,20.)};den=matched=0
    manifest=read(ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')
    target_by_id={r['id']:r for r in manifest['records']}
    for row in records:
        bound=target_by_id[row['id']]
        assert bound['image_sha256']==row['image_sha256'] and bound['label_sha256']==row['label_sha256']
        target_box,target_xy,mask=source_target(bound); values=predictions['frames'][row['id']];top=max(values,key=lambda v:v['score']) if values else None
        iou=E._box_iou(np.asarray(top['box_xyxy']),target_box) if top else None
        points=np.asarray(top['keypoints_xy']) if top and top['keypoints_xy'] is not None else None
        good=bool(top and iou>=.5 and points is not None and points.shape==(9,2) and np.isfinite(points).all())
        counts=pck_counts(target_xy,mask,points,good);den+=counts['denominator'];matched+=good
        for key,value in counts['numerators'].items():total[key]+=value
        errors=np.linalg.norm(points-target_xy,axis=1)[mask].tolist() if good else []
        rows[row['id']]=dict(scenario=row['scenario_id'],denominator=counts['denominator'],numerators=counts['numerators'],
            matched=good,score=float(top['score']) if top else 0.,target_iou=float(iou) if iou is not None else None,errors_px=errors)
        for value in values:candidates.append(E.DetectionCandidate(row['id'],True,float(value['score']),np.asarray(value['box_xyxy']),
            np.asarray(value['keypoints_xy']) if value['keypoints_xy'] is not None else None,E._box_iou(np.asarray(value['box_xyxy']),target_box)))
    aps={f'{t:.2f}':E._average_precision_at_iou(candidates,len(records),float(t)) for t in np.arange(.5,.951,.05)}
    errors=np.concatenate([np.asarray(r['errors_px']) for r in rows.values() if r['errors_px']]);means=[np.mean(r['errors_px']) for r in rows.values() if r['errors_px']]
    metrics=dict(name=name,frames=len(records),supervised_points=den,ALL_GT_PCK={k:v/den for k,v in total.items()},ALL_GT_numerators=total,
        detection_match_rate=matched/len(records),matched_frames=matched,box_ap50=aps['0.50'],box_ap50_95=float(np.mean(list(aps.values()))),box_ap_by_iou=aps,
        individual_matched=dict(points=len(errors),median_px=float(np.median(errors)),p90_px=float(np.quantile(errors,.9)),frame_mean_px=float(np.mean(means)),gross20=float(np.mean(errors>20))))
    return metrics,rows


def evaluate_name(name):
    population,split=pair();cache=cache_predictions(name,population);target,rows,top=target_metrics(name,population,split,cache)
    frames=read(cache)['frames'];pose,mapping=pose_metrics(name,frames)
    for fid,pid in mapping.items():rows[fid]['pose_frame_id']=pid
    source,source_rows=source_metrics(name,source_predictions(name))
    atomic_json(RAW/'evaluation'/name/'TARGET_PER_FRAME.json',rows);atomic_json(RAW/'source_evaluation'/name/'SOURCE_PER_FRAME.json',source_rows)
    atomic_json(RAW/'evaluation'/name/'RESULT.json',dict(target=target,pose=pose,source=source,
        target_cache_sha256=sha(cache),checkpoint_sha256=sha(checkpoint(name))))
    print(name,'EVALUATION_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=NAMES);args=parser.parse_args()
    assert read(DOC/'TRAINING_COMPLETE.json')['status']=='PASS'
    # Copy exact historical split only into the isolated experiment for the reused pose wrapper.
    split=read(ROOT/'_docs/experiments/pallet_active_learning_v1/retrospective_v1/SPLIT.json')
    if not (DOC/'SPLIT.json').exists():atomic_json(DOC/'SPLIT.json',split)
    for name in ([args.name] if args.name else NAMES):evaluate_name(name)
    print('ALL13_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
