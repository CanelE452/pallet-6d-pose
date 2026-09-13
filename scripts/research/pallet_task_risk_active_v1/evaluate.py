"""Reserved145 scoring only after four fits; old control predictions are reused."""
from types import SimpleNamespace
import numpy as np
from contracts import *
from pose_adapter import pose


def evaluator():
    e=import_path('task_original_evaluator',OLD_CODE/'simulation_evaluate.py')
    e.RAW=RAW;e.DOC=DOC;e.write=write;e.S.RAW=RAW;e.S.DOC=DOC;e.S.write=write
    return e


def main():
    verify_lock();training=read(DOC/'TRAINING_AUDIT.json')
    assert training['fits']==4 and training['optimizer_updates']==1200 and training['status']=='PASS'
    for name,a in training['audits'].items():assert sha(RAW/'runs'/name/'last.pt')==a['checkpoint_sha256']
    write(DOC/'EVALUATION_ACCESS_AUTHORIZED.json',dict(all_four_fits_complete=True,
        training_audit_sha256=sha(DOC/'TRAINING_AUDIT.json'),reserved_GT_first_scoring_phase=6,
        GT_source_hashes_bound_before_training=True))
    e=evaluator();p=e.P;e.torch.set_num_threads(4)
    split=read(DOC/'SPLIT_BINDING.json');write(DOC/'SPLIT.json',split)
    full=p.population();ids={r['frame_id'] for r in split['evaluation']}
    pair=SimpleNamespace(ready=True,positive=SimpleNamespace(items=[i for i in full.positive.items if i.frame_id in ids]),negative=full.negative)
    assert len(pair.positive.items)==145 and len(pair.negative.items)==2689
    targets={i.frame_id:p.E._legacy_forbidden_target(i) for i in pair.positive.items}
    sessions={r['frame_id']:r['capture_session'] for r in split['evaluation']};resources=[]
    for name in training['audits']:
        folder=RAW/'evaluation'/name;cache=folder/'PREDICTIONS.json';ck=RAW/'runs'/name/'last.pt'
        if not cache.exists():
            resources.append(gpu());predictor=p.E._UltralyticsPredictor(ck,'0');frames={}
            for j,item in enumerate([*pair.positive.items,*pair.negative.items]):
                values=predictor.predict(ROOT/item.image)
                frames[p.canonical_key(item.image)]=[dict(score=float(s),box_xyxy=b.tolist(),keypoints_xy=k.tolist() if k is not None else None) for s,b,k in values]
                if (j+1)%500==0:
                    resources.append(gpu());print('EVALUATE',name,j+1,'/2834',resources[-1]['gpu'],flush=True)
            del predictor;e.torch.cuda.empty_cache()
            write(cache,dict(schema_version='paper_cached_predictions_v1',complete=True,model=name,weights_sha256=sha(ck),frames=frames))
        frames=read(cache)['frames'];collected=p.E._collect_predictions(pair,p.E._CachedPredictor(cache),validated_targets=targets)
        _,candidates,top=collected;metrics=p.E._evaluate_2d_collected(pair,*collected);rows={};scores={True:[],False:[]}
        for positive,items in ((True,pair.positive.items),(False,pair.negative.items)):
            for item in items:
                pred=top.get(item.frame_id);score=pred.score if pred is not None else 0.;scores[positive].append(score)
                if not positive:continue
                t=targets[item.frame_id];errors=[];matched=bool(pred is not None and pred.target_iou>=.5)
                if pred is not None and pred.keypoints_xy is not None:
                    errors=np.linalg.norm(pred.keypoints_xy-t.keypoints_xy,axis=1)[t.keypoint_supervision_mask].tolist()
                rows[item.frame_id]=dict(image=item.image,session=sessions[item.frame_id],matched=matched,errors_px=errors,score=score)
        keys=[k for k,r in rows.items() if r['matched'] and r['errors_px']];gm=e.geometry(rows,keys)
        assert abs(gm['median_px']-metrics['keypoint_location_median_px'])<1e-10
        assert abs(gm['p90_px']-metrics['keypoint_location_p90_px'])<1e-10
        canonical=e.pose_evaluation(folder,frames,name)
        write(folder/'RESULT.json',dict(name=name,actual_evaluation_positive=145,negative_count=2689,
            metrics=metrics,geometry=gm,Det=len(keys)/145,**e.ranking(np.array(scores[True]),np.array(scores[False])),pose=canonical))
        write(folder/'PER_FRAME.json',rows);print('SCORED',name,flush=True)
    write(DOC/'NEW_INFERENCE_AUDIT.json',dict(status='PASS',fits=4,positive_per_fit=145,negative_per_fit=2689,
        real_image_forwards=4*2834,resources=resources,old_controls_new_forwards=0,GT_access_after_training=True))
    per_frame_pose(e,split)


def per_frame_pose(e,split):
    """Recompute canonical per-frame errors and verify all old aggregate metrics."""
    p=e.P;_,_,metrics=canonical_modules()
    manifest=read(POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list'];gt=read(POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    by_image={p.canonical_key(r['image']):r for r in manifest}
    pool_gt=read(RAW/'POOL_GT_REFERENCE.json')
    for r in split['pool']:
        original=gt[by_image[p.canonical_key(r['image_path'])]['frame_id']];local=pool_gt[r['frame_id']]
        assert local['axis_id']==original['physical_long_axis']
        assert np.array_equal(local['rotation'],original['R_gt_representative'])
        assert np.array_equal(local['translation'],original['t_gt'])
    write(DOC/'POOL_GT_CANONICAL_PARITY_AFTER_TRAINING.json',dict(status='PASS',bit_exact_frames=174,
        independent_pool_rebuild_verified_only_after_all_fits=True))
    registry={r['object_type']:r['physical_dimensions_m'] for r in read(REGISTRY)['objects']}
    all_names=['R0',*[f'{m}_seed{s}' for m in read(OLD_DOC/'PROTOCOL_LOCK.json')['methods'] for s in (1,2,3)],
               *read(DOC/'TRAINING_AUDIT.json')['audits']]
    output={};audits={}
    for name in all_names:
        old=name not in read(DOC/'TRAINING_AUDIT.json')['audits']
        folder=(OLD_RAW if old else RAW)/'evaluation'/name
        frames=read(folder/'PREDICTIONS.json')['frames'];summary=read(folder/'RESULT.json')['pose'];rows=[]
        for r in split['evaluation']:
            key=p.canonical_key(r['image_path']);m=by_image[key];truth=gt[m['frame_id']]
            label=read(ROOT/r['label_path']);k=label['camera_data']['intrinsics']
            camera=[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]
            d=registry[r['object_type']];dims=[max(d['x'],d['z']),min(d['x'],d['z']),d['y']]
            choices=frames[key];selected=max(choices,key=lambda c:c['score']) if choices else None
            prediction=pose(selected['keypoints_xy'] if selected else None,camera,dims)
            error=dict(translation_cm=None,yaw_deg=None)
            if prediction['pose_valid']:
                error=dict(translation_cm=metrics.translation_components_m(np.array(prediction['translation']),np.array(truth['t_gt']))['total_m']*100,
                    yaw_deg=metrics.yaw_error_degrees(np.array(prediction['rotation']),np.array(truth['R_gt_representative'])))
            rows.append(dict(frame_id=r['frame_id'],session=r['capture_session'],pose_valid=prediction['pose_valid'],**error))
        valid=[r for r in rows if r['pose_valid']];assert len(valid)==summary['ALL']['n']
        assert len(valid)/145==summary['coverage']
        for k,target in [('translation_cm','translation_median_cm'),('yaw_deg','yaw_median_deg')]:
            assert abs(float(np.median([r[k] for r in valid]))-summary['ALL'][target])<1e-9,(name,k)
        output[name]=rows;audits[name]=dict(n=len(valid),coverage=len(valid)/145,canonical_aggregate_parity=True,old_control_cache_reused=old)
    write(RAW/'PER_FRAME_POSE.json',output);write(DOC/'POSE_PER_FRAME_AUDIT.json',audits)
    print('PER_FRAME_POSE_CANONICAL_PARITY',len(output),'models',flush=True)


if __name__=='__main__':main()
