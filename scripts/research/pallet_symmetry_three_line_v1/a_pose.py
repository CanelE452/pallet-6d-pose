"""Same frozen prediction-only PnP; separate reference construction and full coverage."""
import csv,importlib,time
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np,torch
import env as E
from a_data import records
from a_evaluate import ARMS,targets
from b_pose import infer,error
from preflight import rotations

def infer_square(points,camera,xyz):
    """C4 square has identical W/D geometries: one representative, not a false failure.

    Uses known registered dimensions, never GT points/pose. Same SQPnP/LM solver.
    """
    assert abs(xyz[0]-xyz[2])<1e-9
    if points is None:return dict(available=False)
    points=np.asarray(points,float);usable=np.isfinite(points[:8]).all(-1)
    if usable.sum()<6:return dict(available=False)
    module=importlib.import_module('run_pose_evaluation');model=module.cuboid(*xyz)
    try:solved=module.solve(model,points[:8],camera,usable)
    except (cv2.error,ValueError):return dict(available=False)
    if solved is None:return dict(available=False)
    R,t,residual=solved
    if not np.isfinite(R).all() or not np.isfinite(t).all() or t[2]<=0:return dict(available=False)
    return dict(available=True,R_cf=R.tolist(),R_physical=R.tolist(),centroid=t.tolist(),cf_extents=list(xyz),
      reprojection_px=residual,selected_hypothesis='C4_IDENTICAL_WD_GEOMETRY_REPRESENTATIVE',GT_input=False)

def metadata_and_references():
    meta={k:{} for k in ['RECT_SYNTH_VAL','RECT_DEV','SQUARE_DEV']};truth={k:{} for k in meta};unresolved=[];reflected=[];R_error=[]
    g=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'))
    idx={str(v):i for i,v in enumerate(g['stems'])}
    for r in records('RECT','val'):
        fid=r['id'];i=idx[fid];fx,fy,cx,cy=g['K'][i]
        K=np.array([[fx,0,cx-g['pad'][i]],[0,fy,cy-g['pad'][i]],[0,0,1]])
        meta['RECT_SYNTH_VAL'][fid]=(K,g['dims'][i]);X=g['Xcf'][i]
        Q=np.stack([X[1]-X[0],X[3]-X[0],X[4]-X[0]],axis=1);dims=np.linalg.norm(Q,axis=0);Q/=dims
        assert np.max(abs(Q.T@Q-np.eye(3)))<1e-8
        assert np.max(abs(np.ptp(X,axis=0)-g['dims'][i]))<1e-7
        orth=float(np.max(abs(g['R'][i].T@g['R'][i]-np.eye(3))))
        determinant=abs(float(np.linalg.det(g['R'][i]))-1)
        R_error.append(max(orth,determinant))
        # Same 1e-6 rotation tolerance as existing oriented-IoU/pose metric code.
        assert np.max(abs(X.mean(0)))<1e-7 and max(orth,determinant)<1e-6
        if np.linalg.det(Q)<0:reflected.append(fid)
        # Use the renderer's proper physical axes for the reference body box.
        # A signed-axis permutation, even reflective CF indexing, does not change its volume.
        # Orientation GT remains the untouched proper renderer R, never R@reflection.
        truth['RECT_SYNTH_VAL'][fid]=dict(R_physical=g['R'][i],R_cf=g['R'][i],centroid=g['t'][i],cf_extents=g['dims'][i],order=r['group_order'])
    import pose_evaluation_paths as paths
    contract=paths.load_pose_object_contract(str(E.C.POSE/'POSE_EVAL_OBJECT_CONTRACT.json'))
    oldtruth=E.read(E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    pe=E.C.old('paper_evaluation')
    id_by_image={pe.canonical_key(r.image):r.frame_id for r in pe.population().positive.items}
    for r in E.read(E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']:
        fid=id_by_image[pe.canonical_key(r['image'])];a=E.read(E.ROOT/r['annotation'])['camera_data']['intrinsics'];spec=paths.object_spec(contract,r['object_type'])
        K=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float)
        meta['RECT_DEV'][fid]=(K,np.array([spec['long_m'],spec['height_m'],spec['short_m']]))
        t=oldtruth[r['frame_id']];Q=np.eye(3) if t['physical_long_axis']=='CF_WIDTH' else rotations(4)[1];d=t['physical_dimensions_m']
        truth['RECT_DEV'][fid]=dict(R_physical=np.array(t['R_gt_representative'])@Q,R_cf=np.array(t['R_gt_representative']),
          centroid=np.array(t['t_gt']),cf_extents=[d['across'],d['height'],d['along']],order=2)
    # Square reference is reconstructed ONLY from the corrected, approved target corners.
    # Stored pose_transform can refer to a different numbering/origin; never silently reuse it.
    module=importlib.import_module('run_pose_evaluation');square=targets()['SQUARE_DEV'];xyz=np.array([1.1,.15,1.1]);model=module.cuboid(*xyz)
    for r in records('SQUARE','val'):
        fid=r['id'];group,frame=fid.split('__',1)
        ann=E.ROOT/'challenge/data/01_real/live_capture_gt'/group/(frame+'.json');a=E.read(ann)['camera_data']['intrinsics']
        K=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float);meta['SQUARE_DEV'][fid]=(K,xyz)
        t=square[fid];usable=t['valid'][:8]&np.isfinite(t['points'][:8]).all(-1)
        if usable.sum()<6:unresolved.append(dict(id=fid,reason='fewer than six annotated corners'));continue
        try:solved=module.solve(model,t['points'][:8],K,usable)
        except (cv2.error,ValueError):solved=None
        if solved is None:unresolved.append(dict(id=fid,reason='reference PnP failed'));continue
        R,translation,residual=solved
        truth['SQUARE_DEV'][fid]=dict(R_physical=R,R_cf=R,centroid=translation,cf_extents=xyz,order=4,reference_reprojection_px=residual)
    E.write(E.DOC/'A/SOURCE_POSE_FRAME_AUDIT.json',dict(source_val_frames=4020,reflected_CF_index_frames=reflected,
      physical_reference_rotation_proper=True,CF_corner_extent_equals_physical_extent=True,
      existing_rotation_numeric_tolerance=1e-6,max_renderer_rotation_error=max(R_error),
      source_reference_body_frame='renderer physical R and dimensions; same body box as signed-permuted CF axes',
      reflected_frame_excluded=False,reflection_added_to_equivalence_group=False,source_targets_or_training_modified=False,
      correction='Initial A pose setup stopped before any pose inference because its inherited B matched-subset assertion assumed all CF index bases had determinant +1. Full val has one determinant -1. A second setup-only check used an unnecessarily strict 1e-7 rotation tolerance; observed renderer float32 error max 2.384e-7. Existing metric tolerance 1e-6 is used without changing stored R. Neither setup attempt performed pose inference. No neural inference or training rerun.'))
    return meta,truth,unresolved

def score_record(item):
    fid,pred,gt=item
    return dict(id=fid,**error(pred,gt))

def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert E.read(E.DOC/'A/INFERENCE_COMPLETE.json')['complete']
    meta,truth,unresolved=metadata_and_references();summary={};allrows={};flat=[];calls=0
    for split in meta:
        ids={r['id'] for r in E.read(E.RAW/f'A/predictions/{split}/R0.json')['records']}
        assert ids==set(meta[split]) and set(truth[split])<=ids
    E.freeze(E.DOC/'A/POSE_ADAPTER_LOCK.json',dict(
      RECT='Unchanged B prediction-only W/D selector and SQPnP/RefineLM.',
      SQUARE='Known x=z=1.1m, task C4. Collapse geometrically identical W/D hypotheses before same SQPnP/RefineLM; identity representative, no GT phase.',
      reason='Existing rectangular selector returns no selected_hypothesis on exact W/D ties; applying that to a square would falsely turn equivalent hypotheses into missing poses.',
      changed_reference_or_metric_for_model_selection=False,GT_for_prediction=False,
      fixed_before_any_A_pose_performance=True))
    for split in meta:
        summary[split]={};allrows[split]={}
        for arm in ARMS:
            rows=E.read(E.RAW/f'A/predictions/{split}/{arm}.json')['records'];dst=E.RAW/f'A/pose_predictions/{split}/{arm}.json'
            if dst.exists():predictions=E.read(dst)['records']
            else:
                predictions={};start=time.monotonic()
                for i,r in enumerate(rows):
                    c=max(r['candidates'],key=lambda c:c['score']) if r['candidates'] else None
                    fn=infer_square if split=='SQUARE_DEV' else infer
                    predictions[r['id']]=fn(None if c is None else c['keypoints_xy'],*meta[split][r['id']]);calls+=1
                E.write(dst,dict(complete=True,records=predictions,GT_pose_for_prediction=False,GT_match_gate=False,
                  prediction_source=E.bound(E.RAW/f'A/predictions/{split}/{arm}.json')))
                print('A POSE INFER',split,arm,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
            with ProcessPoolExecutor(max_workers=4) as pool:
                scores=list(pool.map(score_record,[(fid,predictions[fid],gt) for fid,gt in truth[split].items()],chunksize=64))
            available=[r for r in scores if r['available']]
            summary[split][arm]=dict(frames=len(rows),reference_available=len(scores),reference_missing=len(rows)-len(scores),
              prediction_available=sum(p['available'] for p in predictions.values()),paired_available=len(available),
              full_population_prediction_coverage=sum(p['available'] for p in predictions.values())/len(rows),
              reference_denominator_pose_coverage=len(available)/len(scores),
              **{f:dict(median=float(np.median([r[f] for r in available])),P90=float(np.quantile([r[f] for r in available],.9))) if available else dict(median=None,P90=None)
                 for f in ['centroid_translation_cm','equivalent_rotation_deg','body_extent_IoU']},
              centroid_within10cm_full_reference_rate=sum(r['available'] and r['centroid_translation_cm']<=10 for r in scores)/len(scores),
              pose_10cm_10deg_full_reference_rate=sum(r['available'] and r['centroid_translation_cm']<=10 and r['equivalent_rotation_deg']<=10 for r in scores)/len(scores))
            allrows[split][arm]=scores
            for r in scores:flat.append(dict(split=split,arm=arm,id=r['id'],available=r['available'],centroid_translation_cm=r.get('centroid_translation_cm'),
              equivalent_rotation_deg=r.get('equivalent_rotation_deg'),body_extent_IoU=r.get('body_extent_IoU')))
            print('A POSE SCORE',split,arm,len(available),'/',len(scores),flush=True)
    with (E.DOC/'A/pose_per_frame_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    E.write(E.RAW/'A/POSE_FULL_METRICS.json',allrows)
    E.write(E.DOC/'A/POSE_SECONDARY.json',dict(complete=True,summary=summary,
      expected_pose_frame_evaluations=7*sum(len(v) for v in meta.values()),new_pose_inference_this_execution=calls,
      DEV_ID_join='Exact canonical image path maps current colon-style IDs to historical pose-manifest IDs; no fuzzy matching or dropped frames',
      square_unresolved_references=unresolved,
      prediction='RECT same B prediction-only W/D selector. SQUARE merges geometrically identical W/D hypotheses; all use same SQPnP/RefineLM, no GT orbit/match gate.',
      origins='all model points centroid-centred; centroid=R*mean(X)+t=t',
      square_reference='new sidecar from approved corrected annotated corners and registered 1.1x.15x1.1m; NOT independent physical metrology',
      DEV_reference='unchanged geometry-reconstructed reference; not independent metrology',
      source_reference='renderer physical pose; CF-to-physical phase rule same as B, source C1 front can remain unidentifiable',
      metrics='median/P90 conditional on available pair; coverage and threshold success use full reference denominator',
      primary_selection_or_model_changes=False))
if __name__=='__main__':main()
