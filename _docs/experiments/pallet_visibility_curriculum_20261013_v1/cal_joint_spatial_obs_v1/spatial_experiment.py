"""Sealed layer2 spatial features, joint usability and independent RGB ridge."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from code_snapshot.calibration import features
from code_snapshot import common as C
from code_snapshot.run import no_annotations

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(C.finite(value),indent=2,allow_nan=False)+'\n')


def rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(s) for s in stream if s.strip()]


def crop(image,point,width):
    x0,y0=np.floor(np.asarray(point)-width/2).astype(int);x1,y1=x0+width,y0+width;h,w=image.shape[:2]
    tile=image[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]
    tile=cv2.copyMakeBorder(tile,max(0,-y0),max(0,y1-h),max(0,-x0),max(0,x1-w),cv2.BORDER_REFLECT_101)
    assert tile.shape[:2]==(width,width)
    return cv2.resize(tile,(128,128),interpolation=cv2.INTER_LINEAR)


def extract():
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18
    p=read(HERE/'PROTOCOL.json');assert not (HERE/'FEATURES.npz').exists()
    for file,digest in ((p['image_inputs'],p['image_inputs_sha256']),(p['cached_predictions'],p['cached_predictions_sha256']),
                        (p['pretrained_weights'],p['pretrained_weights_sha256'])):assert sha(file)==digest
    inputs=read(p['image_inputs']);frames=inputs['frames'];assert len(frames)==76 and all(f['partition']=='CAL' for f in frames)
    cached={r['id']:r for r in rows(p['cached_predictions']) if r['method']=='INITIAL'}
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    original=resnet18(weights=None);original.load_state_dict(torch.load(p['pretrained_weights'],map_location='cpu',weights_only=True))
    backbone=nn.Sequential(original.conv1,original.bn1,original.relu,original.maxpool,original.layer1,original.layer2).eval().requires_grad_(False).to(device)
    mean=torch.tensor(p['input_normalization']['mean']).reshape(1,3,1,1);std=torch.tensor(p['input_normalization']['std']).reshape(1,3,1,1)
    write(HERE/'EXTRACTION_CODE_BINDING.json',dict(runner_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.json'),before_extraction=True))
    keys=[];rgb=[];residual=[];f4=[];f8=[];radii=[];start=time.monotonic()
    with no_annotations(),torch.inference_mode():
        for i,frame in enumerate(frames):
            row=cached[frame['id']];image_path=Path(inputs['source_root'])/frame['image'];assert sha(image_path)==frame['image_sha256']
            image=cv2.imread(str(image_path));q=np.asarray(row['original_points'],float);H=np.asarray(row['self_hidden'],bool)
            h,w=image.shape[:2];valid=(np.isfinite(q).all(1)&~(q==-1).all(1)&~H&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h))
            initial=frame['initial_poses']['N3_SUBPIX'];xyz=C.cuboid(*initial['cf_extents']);projected=C.project(xyz,initial['R_cf'],initial['centroid'],np.asarray(frame['K']))
            r=int(row['appearance']['diagnostics']['patch_radius_px']);xx=features(row['appearance']);ids=np.flatnonzero(valid);patches=[]
            for k in ids:patches.extend([crop(image,q[k],4*r),crop(image,q[k],8*r)])
            if patches:
                bgr=np.stack(patches);tensor=torch.from_numpy(np.ascontiguousarray(bgr[...,::-1].transpose(0,3,1,2))).float().div_(255)
                spatial=backbone(((tensor-mean)/std).to(device));assert spatial.shape==(2*len(ids),128,16,16)
                four=F.adaptive_avg_pool2d(spatial[::2],(4,4)).flatten(1).cpu().numpy()
                eight=F.adaptive_avg_pool2d(spatial[1::2],(2,2)).flatten(1).cpu().numpy()
                for j,k in enumerate(ids):
                    error=(projected[k]-q[k])/r
                    keys.append(dict(id=frame['id'],corner=int(k),session=frame['session']))
                    rgb.append(xx[k]);residual.append([*np.clip(error,-8,8),min(8,float(np.linalg.norm(error)))])
                    f4.append(four[j]);f8.append(eight[j]);radii.append(r)
            if (i+1)%10==0:print('SPATIAL_FEATURES',i+1,'/76',round(time.monotonic()-start,2),flush=True)
    np.savez_compressed(HERE/'FEATURES.npz',RGB26=np.asarray(rgb,float),RESIDUAL3=np.asarray(residual,float),
        F4=np.asarray(f4,np.float32),F8=np.asarray(f8,np.float32),RADIUS=np.asarray(radii,float))
    write(HERE/'FEATURE_KEYS.json',keys)
    write(HERE/'FEATURE_SEAL.json',dict(complete=True,before_CAL_labels=True,GT_read_canary='PASS',
        features_sha256=sha(HERE/'FEATURES.npz'),keys_sha256=sha(HERE/'FEATURE_KEYS.json'),protocol_sha256=sha(HERE/'PROTOCOL.json'),
        runner_sha256=sha(__file__),points=len(keys),frames=76,torch_version=torch.__version__,torchvision_version=torchvision.__version__,
        weights_sha256=p['pretrained_weights_sha256'],trainable_backbone_parameters=0,
        extraction_seconds=time.monotonic()-start,TEST_accessed=False,FINAL_accessed=False))
    print('SPATIAL_SEALED',len(keys),flush=True)


def confusion(y,p,threshold):
    from sklearn.metrics import roc_auc_score
    pred=p>=threshold;tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    return dict(n=len(y),usable=int(y.sum()),unusable=int((~y).sum()),AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        threshold=threshold,TP=tp,FP=fp,FN=fn,TN=tn,usable_recall=tp/max(1,tp+fn),unusable_recall=tn/max(1,tn+fp),
        precision=tp/max(1,tp+fp),balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)))


def bounded_delta(normalized,radius):
    value=np.clip(normalized*radius[:,None],-16,16);length=np.linalg.norm(value,axis=1)
    return value*np.minimum(1,16/np.maximum(length,1e-12))[:,None]


def fit():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    p,seal=read(HERE/'PROTOCOL.json'),read(HERE/'FEATURE_SEAL.json')
    assert seal['before_CAL_labels'] and seal['features_sha256']==sha(HERE/'FEATURES.npz')
    assert seal['keys_sha256']==sha(HERE/'FEATURE_KEYS.json') and seal['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    keys=read(HERE/'FEATURE_KEYS.json');array=np.load(HERE/'FEATURES.npz');rgb=array['RGB26'];residual=array['RESIDUAL3'];radius=array['RADIUS']
    # Annotation use begins after the prediction-only cache seal.
    oracle=read(HERE.parent/'ORACLE_CAL_INPUTS.json')['frames'];assert len(oracle)==76 and all(f['partition']=='CAL' for f in oracle)
    oracle={f['id']:f for f in oracle};cached={r['id']:r for r in rows(p['cached_predictions']) if r['method']=='INITIAL'}
    groups=np.asarray([k['session'] for k in keys]);joint=np.full(len(keys),-1,int);reg=np.zeros(len(keys),bool)
    target=np.zeros((len(keys),2));error=np.full(len(keys),np.nan);label_records=[];reasons=collections.Counter()
    for i,key in enumerate(keys):
        frame=oracle[key['id']];meta=frame['scoring_metadata'];mapping=meta['native_to_reference_corner_ids'];k=key['corner']
        state=meta['human_states_native'][k];gt=np.asarray(frame['target']['gt'],float)[mapping[k]]
        q=np.asarray(cached[key['id']]['original_points'][k]);finite=np.isfinite(gt).all() and not (gt==-1).all()
        reason='excluded_unknown_semantic_state'
        if state in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED'):
            if state!='DIRECT_VISIBLE':joint[i]=0;reason='known_non_direct_unusable'
            elif not finite:reason='direct_invalid_native_GT_excluded'
            else:
                target[i]=gt-q;error[i]=np.linalg.norm(target[i]);joint[i]=int(error[i]<=8);reg[i]=error[i]<=16
                reason='direct_in16_regression_target' if reg[i] else 'direct_outside16_usability_negative_no_regression'
        reasons[reason]+=1
        label_records.append(dict(**key,human_state=state,usable_label=int(joint[i]),coordinate_error_px=error[i],
            regression_target_eligible=bool(reg[i]),reason=reason,native_reference_corner=int(mapping[k]),
            normalized_regression_target=(target[i]/radius[i]).tolist() if reg[i] else None))
    selected=np.flatnonzero(joint>=0);y=joint[selected].astype(bool);train_groups=groups[selected]
    splits=list(LeaveOneGroupOut().split(rgb[selected],y,train_groups));assert len(splits)==5
    families=dict(RGB26=(None,None),RGB26_RESIDUAL3=(None,None),RGB26_F4_PCA32=(array['F4'],32),
        RGB26_F4_PCA64=(array['F4'],64),RGB26_F4F8_PCA32=(np.column_stack([array['F4'],array['F8']]),32),
        RGB26_F4F8_PCA64=(np.column_stack([array['F4'],array['F8']]),64))
    summaries={};all_records=[];folds=[]
    for family,(spatial,components) in families.items():
        transforms=[]
        for train_local,test_local in splits:
            heldout=str(train_groups[test_local][0]);train=np.flatnonzero(groups!=heldout);test=np.flatnonzero(groups==heldout)
            if spatial is None:
                x=np.column_stack([rgb,residual]) if family=='RGB26_RESIDUAL3' else rgb
            else:
                bs=StandardScaler().fit(spatial[train]);pc=PCA(n_components=components,svd_solver='full').fit(bs.transform(spatial[train]))
                x=np.column_stack([rgb,pc.transform(bs.transform(spatial))])
            class_train=selected[train_local];class_test=selected[test_local]
            scaler=StandardScaler().fit(x[class_train]);zx=scaler.transform(x)
            transforms.append(dict(heldout=heldout,train=train,test=test,class_train=class_train,class_test=class_test,x=x,zx=zx))
            folds.append(dict(family=family,heldout=heldout,feature_training_sessions=sorted(set(groups[train])),
                class_training_points=len(class_train),regression_training_points=int(reg[train].sum()),
                PCA_and_scalers_trainfold_only=True))
        candidates=[]
        for strength in p['classification_C_grid']:
            prob=np.zeros(len(keys))
            for tr in transforms:
                head=LogisticRegression(C=strength,class_weight='balanced',max_iter=3000,random_state=20261013).fit(tr['zx'][tr['class_train']],joint[tr['class_train']])
                prob[tr['test']]=head.predict_proba(tr['zx'][tr['test']])[:,1]
            candidates.append(dict(C=strength,AUC=float(roc_auc_score(y,prob[selected])),probabilities=prob))
        best=max(candidates,key=lambda r:(r['AUC'],-r['C']));prob=best['probabilities']
        threshold=max(p['threshold_grid'],key=lambda t:(confusion(y,prob[selected],t)['balanced_accuracy'],-abs(t-.5),-t))
        summary=dict(C=best['C'],OOF=confusion(y,prob[selected],threshold),
            per_session={s:confusion(y[train_groups==s],prob[selected][train_groups==s],threshold) for s in sorted(set(groups))},
            class_candidates=[dict(C=c['C'],AUC=c['AUC']) for c in candidates])
        # Coordinate regressors use the RGB-only spatial families. The geometry
        # residual baseline never becomes a manufactured RGB observation.
        delta=np.zeros((len(keys),2));reg_summary=None
        if spatial is not None:
            options=[]
            for alpha in p['ridge_alpha_grid']:
                prediction=np.zeros((len(keys),2))
                for tr in transforms:
                    ids=tr['train'][reg[tr['train']]];scaler=StandardScaler().fit(tr['x'][ids])
                    ridge=Ridge(alpha=alpha).fit(scaler.transform(tr['x'][ids]),target[ids]/radius[ids,None])
                    prediction[tr['test']]=ridge.predict(scaler.transform(tr['x'][tr['test']]))
                moved=bounded_delta(prediction,radius);corrected_error=np.linalg.norm(target[reg]-moved[reg],axis=1)
                options.append(dict(alpha=alpha,mean_error=float(corrected_error.mean()),delta=moved))
            chosen=min(options,key=lambda r:(r['mean_error'],-r['alpha']));delta=chosen['delta'];after=np.linalg.norm(target[reg]-delta[reg],axis=1)
            reg_summary=dict(alpha=chosen['alpha'],eligible_DIRECT_in16=int(reg.sum()),
                original_mean_px=float(error[reg].mean()),original_median_px=float(np.median(error[reg])),
                corrected_mean_px=float(after.mean()),corrected_median_px=float(np.median(after)),
                component_MAE_px=float(np.abs(target[reg]-delta[reg]).mean()),improved=int((after<error[reg]).sum()),worsened=int((after>error[reg]).sum()),
                candidates=[dict(alpha=c['alpha'],mean_error=c['mean_error']) for c in options])
            summary['RGB_regression']=reg_summary
        summaries[family]=summary
        for i,key in enumerate(keys):all_records.append(dict(**key,family=family,usable_label=int(joint[i]),OOF_usable_probability=float(prob[i]),
            cutoff=threshold,RGB_delta_px=delta[i].tolist(),RGB_regressor_available=spatial is not None,
            regression_target_eligible=bool(reg[i]),original_GT_error_px=error[i]))
        print('JOINT_SPATIAL_CV',family,'usable',summary['OOF'],'REG',reg_summary,flush=True)
    write(HERE/'LABEL_AUDIT.json',dict(records=label_records,reasons=reasons,known_usable_labels=len(selected),
        positive=int(y.sum()),negative=int((~y).sum()),regression_targets=int(reg.sum()),GT_coords_are_targets_only=True))
    write(HERE/'FOLDS.json',folds);write(HERE/'OOF_OBSERVATIONS.json',all_records)
    # Deployment/pose replay consumes predictions only; label/error diagnostics
    # remain in distinct artifacts and never enter the pose inference reader.
    predicted=[{k:r[k] for k in ('id','corner','session','family','OOF_usable_probability',
                                  'cutoff','RGB_delta_px','RGB_regressor_available')} for r in all_records]
    write(HERE/'PREDICTED_OBSERVATIONS.json',predicted)
    write(HERE/'RESULTS.json',dict(schema='CAL_JOINT_SPATIAL_OBSERVATION_V1',CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        feature_seal_sha256=sha(HERE/'FEATURE_SEAL.json'),summary=summaries,limitations=p['limits']))
    write(HERE/'OOF_OBSERVATION_SEAL.json',dict(complete=True,before_pose_inference_and_scoring=True,
        observations_sha256=sha(HERE/'PREDICTED_OBSERVATIONS.json'),prediction_artifact='PREDICTED_OBSERVATIONS.json',
        feature_seal_sha256=sha(HERE/'FEATURE_SEAL.json'),
        TEST_accessed=False,FINAL_accessed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['extract','fit']);args=parser.parse_args()
    extract() if args.command=='extract' else fit()
