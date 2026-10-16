"""Small RGB visibility head trained on CAL labels with session-held-out selection.

Absolute position, keypoint ID, image/patch size and session ID are deliberately
excluded. Inference consumes only the local RGB evidence already extracted by
appearance.py; fitted coefficients are ordinary immutable JSON parameters.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import common as C

SCALARS=('inward_material_fraction','outward_material_fraction','patch_material_fraction',
         'nearby_inward_material_fraction','soft_inward_color_support','soft_nearby_color_support',
         'corner_structure','edge_energy','patch_chroma_mean','patch_chroma_std')
VECTORS=('patch_lab_mean','patch_lab_median','patch_lab_std','inward_lab_mean','inward_lab_std')


def features(appearance):
    output=[]
    corners=appearance['diagnostics']['corners']
    radius=max(1,appearance['diagnostics']['patch_radius_px'])
    for row in corners:
        x=[]
        for key in SCALARS:
            value=row.get(key)
            x.append(float(value) if value is not None else 0.)
        for key in VECTORS:
            vector=row.get(key)
            x.extend([0.,0.,0.] if vector is None else vector)
        x.append(min(4,float(row.get('edge_distance_px') or 0)/radius))
        output.append(x)
    return np.asarray(output,float)


def predict(appearance, model):
    from copy import deepcopy
    result=deepcopy(appearance)
    x=features(appearance)
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    logit=z@np.asarray(model['coefficients'])+model['intercept']
    probability=1/(1+np.exp(-np.clip(logit,-60,60)))
    threshold=model['visible_threshold']
    for index,state in enumerate(result['states']):
        if state in ('OUT_OF_FRAME','SELF_OCCLUDED'):
            continue
        if result['diagnostics']['corners'][index].get('reason')!='rgb_evidence_only':
            continue
        result['visibility_score'][index]=float(probability[index])
        if probability[index]>=threshold:
            result['states'][index]='VISIBLE_BOUNDARY' if result['boundary_score'][index]>=model['boundary_threshold'] else 'VISIBLE_INTERIOR'
            # RGB membership and exact coordinate precision are distinct: this
            # floor only allows credible interior observations to enter at low weight.
            result['local_accuracy_score'][index]=max(result['local_accuracy_score'][index],.12)
        else:
            result['states'][index]='UNKNOWN'
    result['diagnostics']['calibration']=dict(schema=model['schema'],ground_truth_input=False,
        training_partition='CAL_ONLY',visibility_probabilities=probability.tolist())
    return result


def train(output, destination):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    protocol=C.read(output/'PROTOCOL.json')
    assert protocol['partition']=='calibration','Only CAL may train state parameters'
    annotation=C.read(C.DOC/'SCORING_METADATA.json')
    labels={r['id']:r for r in annotation['frames'] if r['partition']=='CAL'}
    raw=[r for r in C.rows(output/'PREDICTIONS.jsonl.gz') if r['method']=='RGB_CURRICULUM']
    X=[];y=[];groups=[];keys=[]
    for row in raw:
        xx=features(row['appearance'])
        for index, state in enumerate(labels[row['id']]['human_states_native']):
            if state not in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
                continue
            if row['appearance']['states'][index] in ('SELF_OCCLUDED','OUT_OF_FRAME'):
                continue
            X.append(xx[index]);y.append(state=='DIRECT_VISIBLE');groups.append(row['session']);keys.append([row['id'],index])
    X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups)
    candidates=[]
    for strength in (.01,.1,1.,10.):
        probability=np.zeros(len(y))
        folds=[]
        for train_ids,test_ids in LeaveOneGroupOut().split(X,y,groups):
            scaler=StandardScaler().fit(X[train_ids])
            head=LogisticRegression(C=strength,class_weight='balanced',max_iter=2000,random_state=20261013)
            head.fit(scaler.transform(X[train_ids]),y[train_ids])
            probability[test_ids]=head.predict_proba(scaler.transform(X[test_ids]))[:,1]
            folds.append(dict(session=str(groups[test_ids][0]),points=len(test_ids),
                              AUC=float(roc_auc_score(y[test_ids],probability[test_ids])) if len(np.unique(y[test_ids]))==2 else None))
        auc=float(roc_auc_score(y,probability))
        candidates.append(dict(C=strength,AUC=auc,folds=folds,probabilities=probability.tolist()))
    best=max(candidates,key=lambda r:r['AUC'])
    p=np.asarray(best['probabilities'])
    # Use session-held-out predictions, never a training-set ROC, to pick cutoff.
    thresholds=np.arange(.2,.81,.05)
    threshold=max(thresholds,key=lambda value:(balanced_accuracy_score(y,p>=value),-abs(value-.5)))
    scaler=StandardScaler().fit(X)
    head=LogisticRegression(C=best['C'],class_weight='balanced',max_iter=2000,random_state=20261013)
    head.fit(scaler.transform(X),y)
    model=dict(schema='local_rgb_logistic_visibility_v1',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        coefficients=head.coef_[0].tolist(),intercept=float(head.intercept_[0]),
        visible_threshold=float(threshold),boundary_threshold=.35,feature_scalars=SCALARS,feature_vectors=VECTORS,
        training_frames=len(raw),training_points=len(y),training_sessions=sorted(set(groups)),
        visible_points=int(y.sum()),external_points=int((~y).sum()),C=best['C'],
        crossfit_AUC=best['AUC'],crossfit_balanced_accuracy=float(balanced_accuracy_score(y,p>=threshold)),
        training_predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
        TEST_accessed=False,FINAL_accessed=False,excluded_features=['corner_ID','coordinates','patch_size','session_ID'])
    C.write(destination,model)
    C.write(destination.with_suffix('.training.json'),dict(candidates=candidates,keys=keys,labels=y.tolist(),
        model_sha256=C.sha(destination),selection='CAL session leave-one-out AUC and balanced accuracy'))
    print('FIT',len(y),'points',len(set(groups)),'sessions','C',best['C'],'CV_AUC',round(best['AUC'],4),
          'threshold',round(threshold,2),'balanced_accuracy',round(model['crossfit_balanced_accuracy'],4),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    train(args.input,args.output)
