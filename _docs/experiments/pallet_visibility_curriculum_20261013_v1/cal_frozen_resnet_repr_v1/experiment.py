"""CAL-only frozen patch representations, sealed before label access."""
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
from code_snapshot.run import no_annotations

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def crop(image, point, width):
    """Point-centred integer crop with reflect101 padding and a fixed width."""
    x0, y0 = np.floor(np.asarray(point) - width / 2).astype(int)
    x1, y1 = x0 + width, y0 + width
    h, w = image.shape[:2]
    selected = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    selected = cv2.copyMakeBorder(selected, max(0, -y0), max(0, y1-h),
        max(0, -x0), max(0, x1-w), cv2.BORDER_REFLECT_101)
    if selected.shape[:2] != (width, width):
        raise AssertionError('Incorrect point crop shape')
    return cv2.resize(selected, (128, 128), interpolation=cv2.INTER_LINEAR)


def extract():
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import resnet18
    p = read(HERE / 'PROTOCOL.json')
    assert not (HERE / 'FEATURES.npz').exists()
    assert sha(p['pretrained_weights']) == p['pretrained_weights_sha256']
    assert sha(p['cached_predictions']) == p['cached_predictions_sha256']
    assert sha(p['image_inputs']) == p['image_inputs_sha256']
    for name, digest in p['code_sha256'].items():
        assert sha(HERE / 'code_snapshot' / name) == digest
    inputs = read(p['image_inputs']); frames = inputs['frames']
    assert len(frames) == 76 and all(f['partition'] == 'CAL' for f in frames)
    with gzip.open(p['cached_predictions'], 'rt') as stream:
        cached = {r['id']: r for line in stream if (r := json.loads(line))['method'] == 'INITIAL'}
    assert set(cached) == {f['id'] for f in frames}
    write(HERE / 'EXTRACTION_CODE_BINDING.json', dict(runner_sha256=sha(__file__),
        protocol_sha256=sha(HERE / 'PROTOCOL.json'), before_feature_extraction=True))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(p['pretrained_weights'], map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Identity()
    model = model.to(device).eval().requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    mean = torch.tensor(p['normalization']['mean'], dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(p['normalization']['std'], dtype=torch.float32).reshape(1, 3, 1, 1)
    keys, rgb, f4, f8 = [], [], [], []
    start = time.monotonic()
    with no_annotations(), torch.inference_mode():
        for index, frame in enumerate(frames):
            row = cached[frame['id']]
            image_path = Path(inputs['source_root']) / frame['image']
            assert sha(image_path) == frame['image_sha256']
            image = cv2.imread(str(image_path))
            q = np.asarray(row['original_points'], float)
            hidden = np.asarray(row['self_hidden'], bool)
            h, w = image.shape[:2]
            valid = (np.isfinite(q).all(1) & ~(q == -1).all(1) & ~hidden
                     & (q[:,0] >= 0) & (q[:,0] < w) & (q[:,1] >= 0) & (q[:,1] < h))
            radius = int(row['appearance']['diagnostics']['patch_radius_px'])
            xx = features(row['appearance'])
            ids = np.flatnonzero(valid)
            patches = []
            for k in ids:
                patches.extend([crop(image, q[k], 4*radius), crop(image, q[k], 8*radius)])
            if patches:
                bgr = np.stack(patches)
                tensor = torch.from_numpy(np.ascontiguousarray(bgr[...,::-1].transpose(0,3,1,2))).float().div_(255)
                representation = model(((tensor - mean) / std).to(device)).cpu().numpy()
                assert representation.shape == (2*len(ids), 512)
                for j, k in enumerate(ids):
                    keys.append(dict(id=frame['id'], corner=int(k), session=frame['session']))
                    rgb.append(xx[k]); f4.append(representation[2*j]); f8.append(representation[2*j+1])
            if (index + 1) % 10 == 0:
                print('FROZEN_FEATURE_EXTRACT', index+1, '/76', round(time.monotonic()-start,1), 'seconds', flush=True)
    np.savez_compressed(HERE / 'FEATURES.npz', RGB26=np.asarray(rgb, np.float64),
        F4=np.asarray(f4,np.float32), F8=np.asarray(f8,np.float32))
    write(HERE / 'FEATURE_KEYS.json', keys)
    write(HERE / 'FEATURE_SEAL.json', dict(complete=True, before_CAL_label_access=True,
        GT_read_canary='PASS', TEST_accessed=False, FINAL_accessed=False,
        features_sha256=sha(HERE/'FEATURES.npz'), keys_sha256=sha(HERE/'FEATURE_KEYS.json'),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), runner_sha256=sha(__file__),
        records=len(keys), frames=76, device=device, extraction_seconds=time.monotonic()-start,
        representation_shapes=dict(RGB26=[len(keys),26],F4=[len(keys),512],F8=[len(keys),512]),
        torch_version=torch.__version__, torchvision_version=torchvision.__version__,
        trainable_backbone_parameters=0))
    print('FEATURES_SEALED', len(keys), 'points', flush=True)


def metrics(y, p, threshold):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    predicted = p >= threshold
    tp, fp = int((y & predicted).sum()), int((~y & predicted).sum())
    fn, tn = int((y & ~predicted).sum()), int((~y & ~predicted).sum())
    return dict(points=len(y), visible=int(y.sum()), external=int((~y).sum()),
        AUC=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
        balanced_accuracy=float(balanced_accuracy_score(y,predicted)),
        TP=tp,FP=fp,FN=fn,TN=tn,visible_recall=tp/max(1,tp+fn),external_recall=tn/max(1,tn+fp),
        visible_precision=tp/max(1,tp+fp),threshold=float(threshold))


def fit():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    p, seal = read(HERE/'PROTOCOL.json'), read(HERE/'FEATURE_SEAL.json')
    assert seal['before_CAL_label_access'] and seal['GT_read_canary'] == 'PASS'
    assert seal['features_sha256'] == sha(HERE/'FEATURES.npz')
    assert seal['keys_sha256'] == sha(HERE/'FEATURE_KEYS.json')
    assert seal['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
    keys = read(HERE/'FEATURE_KEYS.json')
    # This is the first CAL-label access for this representation experiment.
    oracle = read(HERE.parent/'ORACLE_CAL_INPUTS.json')['frames']
    assert len(oracle) == 76 and all(f['partition'] == 'CAL' for f in oracle)
    oracle = {f['id']:f for f in oracle}
    selected, labels, groups = [], [], []
    for i, key in enumerate(keys):
        state = oracle[key['id']]['scoring_metadata']['human_states_native'][key['corner']]
        if state in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
            selected.append(i); labels.append(state=='DIRECT_VISIBLE'); groups.append(key['session'])
    array=np.load(HERE/'FEATURES.npz'); rgb=array['RGB26'][selected]; f4=array['F4'][selected]; f8=array['F8'][selected]
    y=np.asarray(labels,bool); groups=np.asarray(groups); selected_keys=[keys[i] for i in selected]
    splits=list(LeaveOneGroupOut().split(rgb,y,groups)); assert len(splits)==5
    frozen_sets = dict(RGB26_BASELINE=None,RGB26_PLUS_F4_PCA32=f4,
                      RGB26_PLUS_F4F8_PCA32=np.concatenate([f4,f8],axis=1))
    summaries={}; all_predictions=[]; folds=[]
    for family, frozen in frozen_sets.items():
        transformed=[]
        for train_ids, test_ids in splits:
            if frozen is None:
                train_x,test_x=rgb[train_ids],rgb[test_ids]; explained=None
            else:
                backbone_scaler=StandardScaler().fit(frozen[train_ids])
                pca=PCA(n_components=32,svd_solver='full',whiten=False).fit(backbone_scaler.transform(frozen[train_ids]))
                train_x=np.column_stack([rgb[train_ids],pca.transform(backbone_scaler.transform(frozen[train_ids]))])
                test_x=np.column_stack([rgb[test_ids],pca.transform(backbone_scaler.transform(frozen[test_ids]))])
                explained=float(pca.explained_variance_ratio_.sum())
            scaler=StandardScaler().fit(train_x)
            transformed.append((train_ids,test_ids,scaler.transform(train_x),scaler.transform(test_x)))
            train_frames=sorted({selected_keys[i]['id'] for i in train_ids}); test_frames=sorted({selected_keys[i]['id'] for i in test_ids})
            assert not set(train_frames)&set(test_frames)
            folds.append(dict(family=family,heldout_session=str(groups[test_ids][0]),train_frames=train_frames,
                test_frames=test_frames,train_points=len(train_ids),test_points=len(test_ids),
                PCA_explained_variance=explained,PCA_fitted_on_trainfold_only=True,
                all_scalers_fitted_on_trainfold_only=True))
        candidates=[]
        for strength in p['logistic']['C_grid']:
            probability=np.zeros(len(y))
            for train_ids,test_ids,train_x,test_x in transformed:
                head=LogisticRegression(C=strength,class_weight='balanced',max_iter=3000,random_state=20261013)
                head.fit(train_x,y[train_ids]);probability[test_ids]=head.predict_proba(test_x)[:,1]
            auc=float(roc_auc_score(y,probability))
            candidates.append(dict(C=strength,AUC=auc,probabilities=probability.tolist()))
        best=max(candidates,key=lambda r:(r['AUC'],-r['C'])); probability=np.asarray(best['probabilities'])
        threshold=max(p['threshold_grid'],key=lambda t:(balanced_accuracy_score(y,probability>=t),-abs(t-.5),-t))
        session_metrics={s:metrics(y[groups==s],probability[groups==s],threshold) for s in sorted(set(groups))}
        defined_auc=[r['AUC'] for r in session_metrics.values() if r['AUC'] is not None]
        summaries[family]=dict(C=best['C'],OOF=metrics(y,probability,threshold),per_session=session_metrics,
            macro_AUC_two_class_sessions=float(np.mean(defined_auc)),
            candidates=[{k:v for k,v in c.items() if k!='probabilities'} for c in candidates])
        for key,label,value in zip(selected_keys,y,probability):
            all_predictions.append(dict(**key,family=family,label=int(label),OOF_probability=float(value),threshold=threshold))
        # Fit the final CAL-only parameter export after model/threshold selection.
        representation=dict(PCA_components=None)
        if frozen is None:
            full_x=rgb
        else:
            backbone_scaler=StandardScaler().fit(frozen)
            pca=PCA(n_components=32,svd_solver='full',whiten=False).fit(backbone_scaler.transform(frozen))
            full_x=np.column_stack([rgb,pca.transform(backbone_scaler.transform(frozen))])
            representation=dict(backbone_mean=backbone_scaler.mean_.tolist(),backbone_scale=backbone_scaler.scale_.tolist(),
                PCA_mean=pca.mean_.tolist(),PCA_components=pca.components_.tolist())
        scaler=StandardScaler().fit(full_x)
        head=LogisticRegression(C=best['C'],class_weight='balanced',max_iter=3000,random_state=20261013).fit(scaler.transform(full_x),y)
        write(HERE/f'{family}_MODEL.json',dict(schema='CAL_FROZEN_PATCH_LOGISTIC_V1',family=family,
            frozen_backbone_weights_sha256=p['pretrained_weights_sha256'],feature_seal_sha256=sha(HERE/'FEATURE_SEAL.json'),
            representation=representation,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
            coefficients=head.coef_[0].tolist(),intercept=float(head.intercept_[0]),C=best['C'],threshold=threshold,
            CAL_only=True,TEST_accessed=False,FINAL_accessed=False,training_points=len(y),
            selected_OOF_AUC=best['AUC'],training_AUC=float(roc_auc_score(y,head.predict_proba(scaler.transform(full_x))[:,1]))))
        print('FROZEN_REPR_CV',family,'C',best['C'],'OOF_AUC',round(best['AUC'],4),
              'threshold',threshold,'metrics',summaries[family]['OOF'],flush=True)
        for session,m in session_metrics.items():
            print('SESSION',session,m,flush=True)
    write(HERE/'FOLDS.json',dict(folds=folds,frame_overlap_between_train_and_test=False))
    write(HERE/'OOF_PREDICTIONS.json',all_predictions)
    write(HERE/'RESULTS.json',dict(schema='CAL_FROZEN_RESNET_REPR_V1',CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        scorable_points=len(y),visible=int(y.sum()),external=int((~y).sum()),summary=summaries,
        features_sha256=seal['features_sha256'],limitations=p['declared_limitations']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['extract','fit']);args=parser.parse_args()
    extract() if args.command=='extract' else fit()
