"""Learn to accept/reject a fixed R0 pose; prediction-only features, fixed splits."""
import argparse
import importlib
import math
import numpy as np
import torch
from PIL import Image
from experiment import S,E,ROOT,RAW,DOC,P,read,write,sha,SEEDS

BASE=RAW/'selective';OUT=DOC/'selective'
POSE=importlib.import_module('run_pose_evaluation')
from pose_evaluation_paths import load_pose_object_contract,object_spec
from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
from symmetry_aware_pose_metrics import yaw_error_degrees,translation_components_m

def prediction_features(pred,camera,dimensions,width,height):
    """No target, true axis, filename or session can enter this interface."""
    empty=dict(features=[0.]*38,valid=False,confidence=0.,reprojection=1e6,R=None,t=None,axis=None)
    if pred is None or pred.get('keypoints_xy') is None:return empty
    xy=np.asarray(pred['keypoints_xy'],float)
    if xy.shape!=(9,2) or not np.isfinite(xy).all():return empty
    long_m,short_m,height_m=dimensions
    models={POSE.CF_WIDTH:POSE.cuboid(long_m,height_m,short_m),POSE.CF_DEPTH:POSE.cuboid(short_m,height_m,long_m)}
    try:
        fits={k:POSE.solve(v,xy[:8],camera,np.ones(8,bool)) for k,v in models.items()}
        if any(v is None for v in fits.values()):return empty
        result=select_pnp_hypotheses(xy,camera,{'x':long_m,'y':height_m,'z':short_m},None)
        chosen=None
        for h in result.hypotheses:
            if h.name==result.selected_hypothesis and h.success:
                chosen=POSE.CF_WIDTH if abs(float(h.camera_facing_dimensions.as_dict()['width'])-long_m)<1e-6 else POSE.CF_DEPTH
        if chosen is None:return empty
        rot,t,res=fits[chosen]
        if not np.isfinite(rot).all() or not np.isfinite(t).all() or t[2]<=0:return empty
        b=np.asarray(pred['box_xyxy']);bw=max(b[2]-b[0],0);bh=max(b[3]-b[1],0);diag=np.hypot(width,height)
        inside=np.mean((xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height))
        values=[float(pred['score']),1.,bw*bh/(width*height),bw/width,bh/height,inside,
            res/diag,abs(fits[POSE.CF_WIDTH][2]-fits[POSE.CF_DEPTH][2])/diag,
            t[0]/t[2],t[1]/t[2],np.log1p(t[2]),*rot.reshape(-1),*(xy/np.array([width,height])).reshape(-1)]
        assert len(values)==38 and np.isfinite(values).all()
        return dict(features=np.clip(values,-20,20).tolist(),valid=True,confidence=float(pred['score']),
            reprojection=float(res),R=rot.tolist(),t=t.tolist(),axis=chosen)
    except (ValueError,RuntimeError,POSE.cv2.error):return empty

def extract():
    split=read(DOC/'SPLIT.json');allrows=sum(split.values(),[])
    manifest=read(P.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']
    selected,mapping=E.join_pose_manifest(allrows,manifest)
    by_image={P.canonical_key(r['image']):r for r in selected}
    contract=load_pose_object_contract(P.POSE/'POSE_EVAL_OBJECT_CONTRACT.json')
    source=read(ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json')
    assert source['complete'] and source['weights_sha256']==sha(S.R0)
    output={}
    for r in allrows:
        image=ROOT/r['image_path'];assert sha(image)==r['image_sha256']
        f=by_image[P.canonical_key(r['image_path'])]
        # Only intrinsic metadata is passed out of the annotation payload.
        intr=read(ROOT/f['annotation'])['camera_data']['intrinsics']
        camera=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float)
        spec=object_spec(contract,f['object_type'])
        with Image.open(image) as im:width,height=im.size
        choices=source['frames'][P.canonical_key(r['image_path'])];pred=max(choices,key=lambda c:c['score']) if choices else None
        out=prediction_features(pred,camera,(spec['long_m'],spec['short_m'],spec['height_m']),width,height)
        output[r['frame_id']]=dict(**out,pose_frame_id=f['frame_id'],image_sha256=r['image_sha256'])
    write(BASE/'FEATURES.json',output)
    write(OUT/'FEATURE_AUDIT.json',dict(status='PASS',frames=len(output),dimension=38,R0_unchanged=sha(S.R0)==S.R0_SHA,
        feature_input_GT_errors=False,frame_ID_and_session_features=False,valid=sum(v['valid'] for v in output.values()),
        source_cache_sha256=sha(ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'),
        feature_sha256=sha(BASE/'FEATURES.json'),frame_id_binding=mapping,
        note='Camera intrinsics read from existing annotation container; coordinate/pose truth not passed to feature function. No new neural inference.'))

def labels(records):
    """Compute labels only for the explicitly supplied split members."""
    features=read(BASE/'FEATURES.json');gt=read(P.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    source=read(ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json')['frames']
    members={i.frame_id:i for i in P.population().positive.items};out={}
    for r in records:
        fid=r['frame_id'];f=features[fid];unsafe=True;t_err=yaw=None;matched=False
        if f['valid']:
            target=P.E._legacy_forbidden_target(members[fid]);choices=source[P.canonical_key(r['image_path'])]
            pred=max(choices,key=lambda c:c['score'])
            a=np.asarray(pred['box_xyxy']);b=target.box_xyxy
            intersect=np.prod(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0))
            union=np.prod(np.maximum(a[2:]-a[:2],0))+np.prod(np.maximum(b[2:]-b[:2],0))-intersect
            matched=bool(intersect/max(float(union),1e-12)>=.5)
            truth=gt[f['pose_frame_id']]
            t_err=translation_components_m(np.asarray(f['t']),np.asarray(truth['t_gt']))['total_m']*100
            yaw=yaw_error_degrees(np.asarray(f['R']),np.asarray(truth['R_gt_representative']))
            unsafe=not(matched and t_err<=10 and yaw<=5)
        out[fid]=dict(unsafe=bool(unsafe),translation_cm=t_err,yaw_deg=yaw,matched=matched,valid=f['valid'])
    return out

def select_threshold(scores,bad,valid,max_risk=.1,min_count=10):
    scores=np.asarray(scores,float);bad=np.asarray(bad,bool);valid=np.asarray(valid,bool)
    threshold=None;count=0;risk=None
    for value in np.unique(scores[valid]):
        accepted=valid&(scores<=value);n=int(accepted.sum());r=float(np.mean(bad[accepted]))
        if n>=min_count and r<=max_risk and n>count:threshold=float(value);count=n;risk=r
    return dict(threshold=threshold,accepted=count,risk=risk,coverage=count/len(scores))

def curve(scores,bad,valid):
    """Tie-invariant expected AURC; invalid predictions always rank last."""
    scores=np.where(valid,np.asarray(scores,float),np.inf);bad=np.asarray(bad,float)
    count=0;errors=0.;area=0.;points=[]
    for value in np.unique(scores):
        mask=scores==value;n=int(mask.sum());total=float(bad[mask].sum())
        for k in range(1,n+1):area+=(errors+k*total/n)/(count+k)
        count+=n;errors+=total
        points.append(dict(coverage=count/len(bad),risk=errors/count,accepted=count,
            score=None if not np.isfinite(value) else float(value)))
    return dict(aurc=area/len(bad),curve=points)

def gate_model(kind):
    return torch.nn.Linear(38,1) if kind=='logistic' else torch.nn.Sequential(torch.nn.Linear(38,32),torch.nn.ReLU(),torch.nn.Linear(32,1))

def fit():
    torch.set_num_threads(4);split=read(DOC/'SPLIT.json');features=read(BASE/'FEATURES.json')
    train_ids=[r['frame_id'] for r in split['train']];cal_ids=[r['frame_id'] for r in split['meta_calibration']]
    truth=labels(split['train']+split['meta_calibration'])
    assert not set(truth)&{r['frame_id'] for r in split['evaluation']}
    x=np.array([features[k]['features'] for k in train_ids],np.float32)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    write(OUT/'STANDARDIZATION.json',dict(train_ids=train_ids,mean=mean.tolist(),scale=scale.tolist()))
    X=torch.tensor((x-mean)/scale);Y=torch.tensor([truth[k]['unsafe'] for k in train_ids],dtype=torch.float32)
    C=torch.tensor((np.array([features[k]['features'] for k in cal_ids],np.float32)-mean)/scale)
    scores={'confidence':[-features[k]['confidence'] for k in cal_ids],
        'negative_reprojection':[features[k]['reprojection'] for k in cal_ids]}
    for name in ('logistic','mlp_seed1','mlp_seed2','mlp_seed3'):
        path=BASE/f'{name}.pt'
        assert not path.exists(),f'Never repeat gate fit {name}'
        seed=1 if name=='logistic' else int(name[-1]);torch.manual_seed(seed)
        m=gate_model('logistic' if name=='logistic' else 'mlp');opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=.001)
        for step in range(1000):
            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(X).flatten(),Y)
            assert torch.isfinite(loss);loss.backward();opt.step()
        with torch.no_grad():scores[name]=m(C).flatten().tolist()
        path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(state_dict=m.state_dict(),name=name,steps=1000,mean=mean,scale=scale),path)
        write(OUT/f'{name}_TRAIN.json',dict(status='PASS',updates=1000,seed=seed,loss=float(loss.detach()),checkpoint_sha256=sha(path)))
    y=[truth[k]['unsafe'] for k in cal_ids];valid=[features[k]['valid'] for k in cal_ids]
    thresholds={name:select_threshold(s,y,valid) for name,s in scores.items()}
    write(BASE/'TRAIN_CAL_LABELS.json',truth)
    write(OUT/'THRESHOLD_LOCK.json',dict(status='LOCKED_BEFORE_TEST_SCORING',calibration_ids=cal_ids,thresholds=thresholds,
        calibration_scores=scores,calibration_unsafe=y,calibration_valid=valid,
        label_note='Shared GT JSON parsed to retrieve train/cal entries; no evaluation-entry error or label computed during fit. Historical eval was already used in prior work.'))
    print('4 gate fits x1000 complete; thresholds locked',flush=True)

def evaluate():
    split=read(DOC/'SPLIT.json');features=read(BASE/'FEATURES.json');lock=read(OUT/'THRESHOLD_LOCK.json')
    ids=[r['frame_id'] for r in split['evaluation']];truth=labels(split['evaluation'])
    valid=np.array([features[k]['valid'] for k in ids]);y=np.array([truth[k]['unsafe'] for k in ids])
    scores={'confidence':[-features[k]['confidence'] for k in ids],
        'negative_reprojection':[features[k]['reprojection'] for k in ids]}
    for name in ('logistic','mlp_seed1','mlp_seed2','mlp_seed3'):
        ck=torch.load(BASE/f'{name}.pt',map_location='cpu');m=gate_model('logistic' if name=='logistic' else 'mlp');m.load_state_dict(ck['state_dict']);m.eval()
        x=(np.array([features[k]['features'] for k in ids],np.float32)-ck['mean'])/ck['scale']
        with torch.no_grad():scores[name]=m(torch.tensor(x)).flatten().tolist()
    result={}
    for name,s in scores.items():
        threshold=lock['thresholds'][name]['threshold'];keep=np.zeros(len(ids),bool) if threshold is None else valid&(np.asarray(s)<=threshold)
        result[name]=dict(**curve(s,y,valid),threshold=threshold,accepted=int(keep.sum()),coverage=float(keep.mean()),
            unsafe_risk=float(y[keep].mean()) if keep.any() else None,unsafe_accepted=int(y[keep].sum()),
            accepted_ids=[k for k,v in zip(ids,keep) if v])
    gates={}
    for control in ('confidence','logistic'):
        values=[result[f'mlp_seed{s}']['aurc'] for s in SEEDS]
        gates[control]=dict(mean_aurc_better=float(np.mean(values))<result[control]['aurc'],
            two_of_three=sum(v<result[control]['aurc'] for v in values)>=2)
    gates['operating_point']={f'seed{s}':result[f'mlp_seed{s}']['coverage']>=.3 and result[f'mlp_seed{s}']['unsafe_risk'] is not None and result[f'mlp_seed{s}']['unsafe_risk']<=.1 for s in SEEDS}
    passed=all(v for d in gates.values() for v in d.values())
    # Recomputed 6D labels must agree with the unchanged canonical R0 evaluation.
    reference=read(S.RAW/'evaluation/R0/RESULT.json')['pose']['ALL']
    assert abs(float(np.median([r['translation_cm'] for r in truth.values() if r['valid']]))-reference['translation_median_cm'])<1e-10
    assert abs(float(np.median([r['yaw_deg'] for r in truth.values() if r['valid']]))-reference['yaw_median_deg'])<1e-10
    write(BASE/'TEST_LABELS.json',truth);write(BASE/'TEST_SCORES.json',scores)
    write(OUT/'RESULTS.json',dict(results=result,gates=gates,total=len(ids),unsafe_total=int(y.sum()),
        always_accept_risk=float(y.mean()),independent_confirmation=False,
        verdict='SELECTIVE_POSE_SIGNAL' if passed else 'SELECTIVE_POSE_NO_SIGNAL',canonical_R0_pose_parity=True,
        accepted_risk_is_empirical_not_certified=True))
    print('Selective evaluation', 'PASS' if passed else 'NO_SIGNAL',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['extract','fit','evaluate']);a=ap.parse_args();globals()[a.phase]()
