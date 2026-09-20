"""Plastic194-only before/after paper metrics; never mutate previous protocols.

Run negative inference first, then score. Positive raw predictions are reused,
not filtered or corrected at evaluation time. All outputs are exclusive.
"""
import argparse
import time
import numpy as np
from . import common as C
from .pseudo import top

RAW = C.RAW / 'paper_metrics_plastic'
DOC = C.DOC / 'paper_metrics_plastic'
POS = C.ROOT / 'challenge/real_gt_v2/manifests/PAPER_EVAL_PLASTIC_POS.json'
NEG = C.ROOT / 'challenge/real_gt_v2/manifests/DEV_NEG2689.json'
OLD = C.ROOT / 'data/pallet/results/paper_pose_metric_closure_v1'


def positive(arm):
    payload = C.read(C.RAW / f'EVAL_PREDICTIONS_{arm}.json')
    C.verify(payload['checkpoint'])
    return payload, {r['id']: top(r['prediction']) for r in payload['records'] if r['kind'] == 'PLASTIC'}


def prepare():
    from challenge.evaluation_v2 import paper_real_eval as E
    from scripts.self_training_yolo import filter_quality_m4 as F, evaluate_arms as A
    from scripts.paper.pose_metric_closure_v1 import run_pose_evaluation as P
    from scripts.paper.pose_metric_closure_v1 import symmetry_aware_pose_metrics as S
    from challenge.evaluation_v2 import pnp_selector, oriented_iou3d
    sources = [POS, NEG, F.LOCK, F.OUT, OLD/'POSE_EVALUATION_R0.json',
               OLD/'GEOMETRY_RESOLVED_POSE_GT.json', OLD/'AXIS_REVIEW_MANIFEST.json',
               OLD/'POSE_EVAL_OBJECT_CONTRACT.json', E.__file__, F.__file__, A.__file__,
               P.__file__, S.__file__, pnp_selector.__file__, oriented_iou3d.__file__, __file__]
    sources += [C.RAW/f'EVAL_PREDICTIONS_{a}.json' for a in ('R0', 'PLASTIC')]
    C.freeze(DOC/'PROTOCOL.json', dict(
        population='Ordinary plastic194 only; fixed NEG2689 shared by both arms',
        arms=['R0', 'PLASTIC'], positive_predictions='Existing all-candidate raw student/R0 cache; no evaluation-time refiner or filter',
        model_selection='Previously frozen final last.pt; no training or selection on these metrics',
        f0='Original M4 F0 eligibility and supervised nine-keypoint fixed-index errors; correct iff all supervised errors <=20px',
        detection='Original paper_real_eval with plastic-only positive manifest, all negative candidates; original frame-score AUROC/FPR95',
        pose='Original closure MAIN frozen prediction-only axis selector + SQPnP/RefineLM; geometry-reconstructed reference, not externally measured 6D GT',
        scope_correction='User screenshots AP/pose R0 are ALL319, not plastic194. Recalculate both arms on plastic194.',
        sources=[C.bound(p) for p in sources]))


def infer_negative(arm):
    import torch
    from challenge.evaluation_v2 import paper_real_eval as E
    from scripts.self_training_yolo.v3 import true_ignore_trainer  # noqa: F401
    C.N.setup(); torch.set_num_interop_threads(1)
    for b in C.read(DOC/'PROTOCOL.json')['sources']: C.verify(b)
    payload, _ = positive(arm)
    destination = RAW/f'NEG_{arm}.json'
    if destination.exists():
        assert C.read(destination)['checkpoint'] == payload['checkpoint']
        return
    C.N.E.gpu(); assert torch.cuda.is_available()
    torch.backends.cudnn.allow_tf32 = True
    predictor = E._UltralyticsPredictor(C.ROOT/payload['checkpoint']['path'], '0')
    records = {}; start = time.monotonic()
    for i, item in enumerate(C.read(NEG)['items']):
        records[item['image']] = [dict(score=s, box_xyxy=b.tolist(), keypoints_xy=k.tolist() if k is not None else None)
                                  for s,b,k in predictor.predict(C.ROOT/item['image'])]
        if (i+1) % 250 == 0:
            print('NEG_EVAL', arm, i+1, '/2689', round(time.monotonic()-start,1), C.N.E.gpu(), flush=True)
    assert len(records) == 2689
    C.freeze(destination, dict(arm=arm, checkpoint=payload['checkpoint'], frames=records,
                              protocol=C.bound(DOC/'PROTOCOL.json'), seconds=time.monotonic()-start))
    print('NEG_COMPLETE', arm, flush=True)


def f0(arm):
    from scripts.self_training_yolo import filter_quality_m4 as F
    _, predictions = positive(arm); lock = C.read(F.LOCK); rows=[]
    for item in C.read(POS)['items']:
        p = predictions[item['frame_id']]
        ann = C.read(C.ROOT/item['gt_v2_path'])['objects'][0]['keypoint_annotations']
        gt = np.asarray([a['xy'] for a in ann], float)
        supervised = np.asarray([bool(a.get('visibility',0)) and a.get('in_frame',True) for a in ann])
        errors = np.linalg.norm(np.asarray(p['keypoints_xy'])-gt,axis=1)[supervised] if p else np.array([])
        row = dict(frame_id=item['frame_id'], detected=p is not None,
                   valid_corners=int((np.asarray(p['keypoints_conf'][:8]) >= lock['keypoint_validity']['kp_conf_threshold']).sum()) if p else 0,
                   errors_px=errors.tolist(), gross_keypoints=int((errors>20).sum()) if p else None)
        row['pass'] = F.passes(row,'F0_NAIVE',lock)
        row['correct'] = row['detected'] and row['gross_keypoints']==0
        rows.append(row)
    accepted=[r for r in rows if r['pass']]; rejected=[r for r in rows if not r['pass']]
    tp=sum(r['correct'] for r in accepted); total=sum(r['correct'] for r in rows)
    summary=dict(accepted=len(accepted), retention=len(accepted)/len(rows),
                 **{'pass':F.pooled(accepted),'reject':F.pooled(rejected)}, TP=tp,
                 precision=tp/len(accepted) if accepted else None, recall=tp/total if total else None)
    C.freeze(RAW/f'F0_{arm}.json',dict(summary=summary, records=rows))
    return summary


def pose(arm):
    from scripts.paper.pose_metric_closure_v1 import run_pose_evaluation as P
    from scripts.paper.pose_metric_closure_v1 import symmetry_aware_pose_metrics as S
    from scripts.paper.pose_metric_closure_v1.pose_evaluation_paths import load_pose_object_contract, object_spec
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    _, predictions=positive(arm)
    truth=C.read(OLD/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    frames={r['frame_id']:r for r in C.read(OLD/'AXIS_REVIEW_MANIFEST.json')['frames_list']}
    contract=load_pose_object_contract(OLD/'POSE_EVAL_OBJECT_CONTRACT.json'); rows=[]; failures=[]
    for frame_id,p in predictions.items():
        if not p: failures.append(dict(id=frame_id,reason='no_prediction')); continue
        frame=frames[frame_id]
        k=C.read(C.ROOT/frame['annotation'])['camera_data']['intrinsics']
        camera=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],float)
        spec=object_spec(contract,frame['object_type']); long,short,height=(spec[k] for k in ('long_m','short_m','height_m'))
        points=np.asarray(p['keypoints_xy'],float); usable=np.isfinite(points[:8]).all(1)
        chosen=None
        if usable.sum()<6: failures.append(dict(id=frame_id,reason='usable<6')); continue
        try:
            selection=select_pnp_hypotheses(points,camera,{'x':long,'y':height,'z':short},None)
            for h in selection.hypotheses:
                if h.name==selection.selected_hypothesis and h.success:
                    chosen=P.CF_WIDTH if abs(float(h.camera_facing_dimensions.as_dict()['width'])-long)<1e-6 else P.CF_DEPTH
        except Exception as exc:
            failures.append(dict(id=frame_id,reason=repr(exc))); continue
        models={P.CF_WIDTH:P.cuboid(long,height,short),P.CF_DEPTH:P.cuboid(short,height,long)}
        fits={key:P.solve(m,points[:8],camera,usable) for key,m in models.items()}
        if chosen is None or any(f is None for f in fits.values()):
            failures.append(dict(id=frame_id,reason='selector_or_solve_failed')); continue
        # The prediction/axis decision is finished before reference values are accessed.
        g=truth[frame_id]; gr=np.asarray(g['R_gt_representative']); gt=np.asarray(g['t_gt'])
        dims=g['physical_dimensions_m']; extents=tuple(dims[k] for k in ('across','height','along'))
        model=S.cuboid_model_points(extents); r,t,_=fits[chosen]
        parts=S.translation_components_m(t,gt)
        rows.append(dict(frame_id=frame_id,axis_correct=chosen==g['physical_long_axis'],
            rotation_error_deg=S.rotation_error_degrees(r,gr), yaw_error_deg=S.yaw_error_degrees(r,gr),
            translation_error_cm=parts['total_m']*100, lateral_error_cm=parts['lateral_m']*100,depth_error_cm=parts['depth_m']*100,
            iou3d=oriented_iou_3d(r,t,extents,gr,gt,extents),add_sym_m=S.symmetry_aware_add_m(model,r,t,gr,gt),diameter_m=S.model_diameter_m(model)))
    arr=lambda k:np.array([r[k] for r in rows],float)
    summary=dict(n=len(rows),coverage=len(rows)/194,axis_accuracy=float(arr('axis_correct').mean()),
        rotation_median_deg=float(np.median(arr('rotation_error_deg'))),yaw_median_deg=float(np.median(arr('yaw_error_deg'))),
        translation_median_cm=float(np.median(arr('translation_error_cm'))),lateral_median_cm=float(np.median(arr('lateral_error_cm'))),
        depth_median_cm=float(np.median(arr('depth_error_cm'))),iou3d_median=float(np.median(arr('iou3d'))),
        add_sym_auc=S.pose_auc(arr('add_sym_m'),float(np.median(arr('diameter_m')))))
    C.freeze(RAW/f'POSE_{arm}.json',dict(path='MAIN',summary=summary,records=rows,failures=failures))
    return summary


def detection(arm):
    from challenge.evaluation_v2 import paper_real_eval as E
    from scripts.self_training_yolo.evaluate_arms import ranking
    payload, _=positive(arm); negative=C.read(RAW/f'NEG_{arm}.json')
    assert negative['checkpoint']==payload['checkpoint']
    meta={r['id']:r for r in C.read(C.DOC/'EVAL_PROTOCOL.json')['records']}
    frames=dict(negative['frames']); positive_scores=[]
    for row in payload['records']:
        if row['kind']!='PLASTIC': continue
        candidates=row['prediction']['candidates']
        frames[meta[row['id']]['image']['path']]=candidates
        positive_scores.append(max((p['score'] for p in candidates),default=0.))
    assert len(frames)==194+2689
    cache=RAW/f'PAPER_CACHE_{arm}.json'
    C.freeze(cache,dict(schema_version='paper_cached_predictions_v1',model=arm,
        weights_sha256=payload['checkpoint']['sha256'],recipe='Original reflect100/imgsz640/conf.001 all candidates; cached positive194 + new negative2689',frames=frames))
    output=RAW/f'PAPER_2D_{arm}.json'
    if not output.exists():
        args=E.build_parser().parse_args(['--positive-manifest',str(POS),'--negative-manifest',str(NEG),
            '--population-role','DEV','--weights',str(C.ROOT/payload['checkpoint']['path']),
            '--predictions',str(cache),'--migration-gate',str(C.ROOT/'challenge/real_gt_v2/MIGRATION_GATE.json'),
            '--symmetry-contract',str(C.ROOT/'challenge/real_gt_v2/SYMMETRY_CONTRACT.json'),'--out',str(output),
            '--per-frame-out',str(RAW/f'PAPER_2D_{arm}.csv'),'--report-out',str(RAW/f'PAPER_2D_{arm}.md')])
        E.run(args)
    report=C.read(output)
    ranks=ranking(np.array(positive_scores),np.array([max((p['score'] for p in v),default=0.) for v in negative['frames'].values()]))
    return dict(report=report,ranking=ranks)


def score():
    for b in C.read(DOC/'PROTOCOL.json')['sources']: C.verify(b)
    output={}
    for arm in ('R0','PLASTIC'):
        output[arm]=dict(f0=f0(arm),pose=pose(arm),detection=detection(arm))
        print('SCORED',arm,output[arm]['f0'],output[arm]['pose'],output[arm]['detection']['ranking'],flush=True)
    old_f0=C.read(C.ROOT/'data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json')['filters']['F0_NAIVE']
    old_pose=C.read(OLD/'POSE_EVALUATION_R0.json')['paths']['MAIN']['plastic']
    for key in ('accepted','retention','precision','recall'): assert np.isclose(output['R0']['f0'][key],old_f0[key],atol=1e-8)
    assert abs(output['R0']['f0']['pass']['median_px']-old_f0['pass']['median_px'])<.001
    for key,value in old_pose.items(): assert np.isclose(output['R0']['pose'][key],value,rtol=0,atol=1e-4),(key,output['R0']['pose'][key],value)
    C.freeze(RAW/'SUMMARY.json',dict(population='PLASTIC194 + NEG2689',arms=output,baseline_f0_pose_parity='PASS',protocol=C.bound(DOC/'PROTOCOL.json')))
    print('COMPLETE_BASELINE_PARITY_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['prepare','negative','score']);parser.add_argument('--arm',choices=['R0','PLASTIC'])
    args=parser.parse_args()
    if args.phase=='prepare': prepare()
    elif args.phase=='negative': infer_negative(args.arm)
    else: score()
