"""Frozen evaluation subset only, after all12 acquisition students finish."""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
spec=importlib.util.spec_from_file_location('al_sim',Path(__file__).with_name('simulation.py'))
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
ROOT,RAW,DOC,P,write,sha=S.ROOT,S.RAW,S.DOC,S.P,S.write,S.sha
read=S.read
from scripts.paper.framing_closure_v1.static_missing_stat_audit import ranking

def training_audit():
    torch.set_num_threads(4)
    audits={};exposure={}
    baseline=S.load_model();expected_init=S.tensor_sha(baseline.state_dict())
    expected_params=sum(p.numel() for p in baseline.parameters())
    for seed in (1,2,3):
        for method in S.METHODS:
            name=f'{method}_seed{seed}';d=RAW/'runs'/name
            a=read(d/'TRAINING_AUDIT.json');assert a['optimizer_updates']==300
            assert sha(d/'last.pt')==a['checkpoint_sha256']
            assert a['init_state_sha256']==expected_init
            ck=torch.load(d/'last.pt',map_location='cpu');model=ck['model']
            assert ck['train_args']['task']=='pose' and model.args['task']=='pose'
            assert sum(p.numel() for p in model.parameters())==expected_params
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            audits[name]=a;exposure[name]=read(d/'EXPOSURE.json');assert len(exposure[name])==300
    assert len({a['init_state_sha256'] for a in audits.values()})==1
    for seed in (1,2,3):
        for step in range(300):assert len({exposure[f'{m}_seed{seed}'][step]['synthetic'] for m in S.METHODS})==1
    write(DOC/'TRAINING_AUDIT.json',dict(status='PASS',fits=12,optimizer_updates=3600,init_parity=True,
        actual_synthetic_augmented_input_parity=True,real_label_count_per_method=30,
        real_exposures_equal=True,real_image_identity_intentionally_differs=True,
        saved_checkpoint_finiteness_task_and_parameter_count=True,audits=audits))

def restricted_pair():
    full=P.population();ids={r['frame_id'] for r in read(DOC/'SPLIT.json')['evaluation']}
    return SimpleNamespace(ready=True,positive=SimpleNamespace(items=[i for i in full.positive.items if i.frame_id in ids]),negative=full.negative)

def geometry(rows,keys):
    pools=[np.asarray(rows[k]['errors_px'],float) for k in sorted(keys)]
    errors=np.concatenate(pools) if pools else np.array([])
    assert len(errors),'No common supervised geometry'
    return dict(frames=len(keys),corners=len(errors),median_px=float(np.median(errors)),
        p90_px=float(np.quantile(errors,.9)),gross20=float(np.mean(errors>20)))

def join_pose_manifest(evaluation,manifest_rows):
    """Join distinct frame-ID namespaces by canonical image identity, never ID spelling."""
    by_image={P.canonical_key(r['image_path']):r for r in evaluation}
    assert len(by_image)==len(evaluation),'Duplicate evaluation image'
    selected=[r for r in manifest_rows if P.canonical_key(r['image']) in by_image]
    assert len(selected)==len(by_image),'Missing or duplicate pose image'
    assert {P.canonical_key(r['image']) for r in selected}==set(by_image)
    assert len({r['frame_id'] for r in selected})==len(selected),'Duplicate pose ID'
    binding=[dict(evaluation_frame_id=by_image[P.canonical_key(r['image'])]['frame_id'],
        pose_frame_id=r['frame_id'],image=P.canonical_key(r['image']),
        image_sha256=by_image[P.canonical_key(r['image'])]['image_sha256']) for r in selected]
    return selected,binding

def pose_evaluation(d,frames,name):
    folder=RAW/'evaluation_inputs';evaluation=read(DOC/'SPLIT.json')['evaluation']
    ids={r['frame_id'] for r in evaluation}
    manifest=read(P.POSE/'AXIS_REVIEW_MANIFEST.json')
    manifest['frames_list'],binding=join_pose_manifest(evaluation,manifest['frames_list'])
    for r in binding:assert sha(ROOT/r['image'])==r['image_sha256']
    pose_ids={r['pose_frame_id'] for r in binding}
    write(folder/'FRAME_ID_BINDING.json',binding)
    original=read(P.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')
    original['frames']={k:v for k,v in original['frames'].items() if k in pose_ids}
    assert set(original['frames'])==pose_ids
    write(folder/'GEOMETRY_RESOLVED_POSE_GT.json',original)
    write(folder/'AXIS_REVIEW_MANIFEST.json',manifest)
    pred={}
    for r in manifest['frames_list']:
        choices=frames[P.canonical_key(r['image'])]
        if not choices:pred[r['frame_id']]=dict(status='NO_DETECTION');continue
        top=max(choices,key=lambda c:c['score'])
        pred[r['frame_id']]=dict(status='OK',keypoints_xy=top['keypoints_xy'],box_xyxy=top['box_xyxy'],box_conf=top['score'])
    write(d/f'predictions/{name}.json',dict(frames=pred))
    module=S.importlib.import_module('run_pose_evaluation')
    path=d/f'POSE_EVALUATION_{name}.json'
    if not path.exists():
        old={k:getattr(module,k) for k in ('OUT_DIR','GT_PATH','MANIFEST','PREDICTIONS')};argv=sys.argv
        module.OUT_DIR=d;module.GT_PATH=folder/'GEOMETRY_RESOLVED_POSE_GT.json';module.MANIFEST=folder/'AXIS_REVIEW_MANIFEST.json';module.PREDICTIONS=d/'predictions'
        sys.argv=[module.__file__,'--pose-object-contract',str(P.POSE/'POSE_EVAL_OBJECT_CONTRACT.json'),'--arm',name]
        try:assert module.main()==0
        finally:
            sys.argv=argv
            for k,v in old.items():setattr(module,k,v)
    result=read(path)
    write(d/'ACTUAL_POSE_BINDING.json',dict(actual_positive_denominator=len(ids),actual_ids=sorted(ids),
        actual_pose_ids=sorted(pose_ids),frame_id_binding_sha256=sha(folder/'FRAME_ID_BINDING.json'),
        prediction_sha256=sha(d/'PREDICTIONS.json'),filtered_GT_sha256=sha(folder/'GEOMETRY_RESOLVED_POSE_GT.json'),
        canonical_evaluator_sha256=sha(Path(module.__file__)),
        historical_metadata_warning='Unchanged evaluator contains literal PAPER_EVAL319/new_training0 prose; use this binding. Actual input and numerical denominator are the filtered evaluation subset.',
        interpretation='MAIN only; geometry-reconstructed reference; no independent confirmation'))
    return result['paths']['MAIN']

def evaluate():
    training_audit();pair=restricted_pair();torch.set_num_threads(4)
    targets={i.frame_id:P.E._legacy_forbidden_target(i) for i in pair.positive.items}
    eval_ids=set(targets)
    assert not eval_ids & set(read(DOC/'LABEL_REVEAL_AUDIT.json')['revealed_frame_ids'])
    names=['R0',*[f'{m}_seed{s}' for s in (1,2,3) for m in S.METHODS]]
    for name in names:
        d=RAW/'evaluation'/name;cache=d/'PREDICTIONS.json'
        ck=S.R0 if name=='R0' else RAW/'runs'/name/'last.pt'
        if not cache.exists():
            if name=='R0':
                source=read(ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json')['frames']
                frames={P.canonical_key(i.image):source[P.canonical_key(i.image)] for i in [*pair.positive.items,*pair.negative.items]}
            else:
                S.snapshot()
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
                foreign=[line for line in lines if line.split(',')[0].strip()!=str(os.getpid())]
                if foreign:
                    write(DOC/'EVALUATION_RESOURCE_UNAVAILABLE.json',dict(status='NOT_RUN',next=name,processes=foreign))
                    raise SystemExit('No waiting or foreign process mutation')
                predictor=P.E._UltralyticsPredictor(ck,'0');frames={}
                for i,item in enumerate([*pair.positive.items,*pair.negative.items]):
                    values=predictor.predict(ROOT/item.image)
                    frames[P.canonical_key(item.image)]=[dict(score=float(s),box_xyxy=b.tolist(),keypoints_xy=k.tolist() if k is not None else None) for s,b,k in values]
                    if (i+1)%1000==0:print(name,'inference',i+1,flush=True)
                del predictor;torch.cuda.empty_cache()
            write(cache,dict(schema_version='paper_cached_predictions_v1',complete=True,model=name,weights_sha256=sha(ck),frames=frames))
        frames=read(cache)['frames']
        collected=P.E._collect_predictions(pair,P.E._CachedPredictor(cache),validated_targets=targets)
        _,candidates,top=collected;m=P.E._evaluate_2d_collected(pair,*collected)
        rows={};scores={True:[],False:[]}
        for positive,items in ((True,pair.positive.items),(False,pair.negative.items)):
            for item in items:
                pred=top.get(item.frame_id);score=pred.score if pred is not None else 0.;scores[positive].append(score)
                if not positive:continue
                t=targets[item.frame_id];errors=[];matched=bool(pred is not None and pred.target_iou>=.5)
                if pred is not None and pred.keypoints_xy is not None:
                    errors=np.linalg.norm(pred.keypoints_xy-t.keypoints_xy,axis=1)[t.keypoint_supervision_mask].tolist()
                rows[item.frame_id]=dict(image=item.image,session=next(r['capture_session'] for r in read(DOC/'SPLIT.json')['evaluation'] if r['frame_id']==item.frame_id),
                    matched=matched,errors_px=errors,score=score)
        keys=[k for k,r in rows.items() if r['matched'] and r['errors_px']]
        gm=geometry(rows,keys)
        assert abs(gm['median_px']-m['keypoint_location_median_px'])<1e-10
        assert abs(gm['p90_px']-m['keypoint_location_p90_px'])<1e-10
        pose=pose_evaluation(d,frames,name)
        write(d/'RESULT.json',dict(name=name,actual_evaluation_positive=len(targets),negative_count=len(pair.negative.items),
            metrics=m,geometry=gm,Det=len(keys)/len(targets),**ranking(np.array(scores[True]),np.array(scores[False])),pose=pose))
        write(d/'PER_FRAME.json',rows)
        print(name,'evaluation complete',flush=True)

if __name__=='__main__':evaluate()
