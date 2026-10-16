"""Strict CAL46 raw-feature subset, fixed LOGO diagnostics, fresh full export."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT=Path('/home/minjae/Documents/github/pallet-pose');sys.path.insert(0,str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1.joint_observation import load_joint_model,predict_feature_rows
HERE=Path(__file__).resolve().parent
COHORT=HERE.parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def key(row):return dict(id=row['id'],corner=row['corner'],session=row['session'])
def metric(y,p):
    admitted=p>=.45;tp=int((y&admitted).sum());fn=int((y&~admitted).sum());fp=int((~y&admitted).sum());tn=int((~y&~admitted).sum())
    return dict(rows=len(y),usable=int(y.sum()),unusable=int((~y).sum()),AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        threshold=.45,TP=tp,FN=fn,FP=fp,TN=tn,usable_recall=tp/max(1,tp+fn),unusable_recall=tn/max(1,tn+fp),
        precision=tp/max(1,tp+fp),balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)))
def bound(delta,radius):
    result=np.clip(delta*radius[:,None],-16,16)
    return result*np.minimum(1,16/np.maximum(np.linalg.norm(result,axis=1),1e-12))[:,None]
def reg_metric(error,target,delta,mask):
    before=error[mask];after=np.linalg.norm(target[mask]-delta[mask],axis=1)
    return dict(rows=int(mask.sum()),before_mean_px=float(before.mean()),before_median_px=float(np.median(before)),
        after_mean_px=float(after.mean()),after_median_px=float(np.median(after)),component_MAE_px=float(np.abs(target[mask]-delta[mask]).mean()),
        improved=int((after<before).sum()),worsened=int((after>before).sum()))


def main():
    p=read(HERE/'PROTOCOL.json');assert not(HERE/'MODEL.json').exists()
    assert sha(COHORT/'COHORT_PROTOCOL.json')==p['cohort_protocol_sha256']
    assert sha(COHORT/'CAL_INPUTS.json')==p['CAL_inputs_sha256']
    assert sha(COHORT/'CAL_SCORING_METADATA.json')==p['CAL_metadata_sha256']
    source=Path(p['source_cache'])
    for name,digest in p['source_cache_bindings'].items():assert sha(source/name)==digest
    frames=read(COHORT/'CAL_INPUTS.json')['frames'];allowed={f['id'] for f in frames}
    cohort=read(COHORT/'COHORT_PROTOCOL.json');excluded=set(cohort['excluded_severe_ids'])
    assert len(allowed)==46 and allowed==set(p['train_frame_ids']) and not allowed&excluded
    assert all(f['partition']=='CAL' for f in frames)
    fullkeys=read(source/'FEATURE_KEYS.json');chosen=np.array([i for i,k in enumerate(fullkeys) if k['id'] in allowed],int)
    keys=[key(fullkeys[i]) for i in chosen];assert {k['id'] for k in keys}==allowed
    # Subsetting occurs before any statistics, PCA or supervised artifact use.
    with np.load(source/'FEATURES.npz') as a:
        arrays={name:a[name][chosen].copy() for name in ('RGB26','F4','F8','RADIUS')}
    np.savez_compressed(HERE/'FILTERED_FEATURES.npz',**arrays)
    write(HERE/'FEATURE_KEYS.json',keys)
    write(HERE/'FILTERED_FEATURE_SEAL.json',dict(before_supervised_labels_or_PCA=True,severe_rows=0,TEST_rows=0,FINAL_rows=0,
        training_frame_ids=sorted(allowed),feature_rows=len(keys),features_sha256=sha(HERE/'FILTERED_FEATURES.npz'),
        keys_sha256=sha(HERE/'FEATURE_KEYS.json'),source_feature_seal_sha256=sha(source/'FEATURE_SEAL.json'),protocol_sha256=sha(HERE/'PROTOCOL.json')))
    metadata={r['id']:r for r in read(COHORT/'CAL_SCORING_METADATA.json')['frames']}
    # Consume only exact selected targets from the previously audited native
    # label cache. No inherited fitted coefficients or severe target enters fit.
    selected_labels={(r['id'],r['corner']):r for r in read(source/'LABEL_AUDIT.json')['records'] if r['id'] in allowed}
    labels=[selected_labels[k['id'],k['corner']] for k in keys]
    for label in labels:
        m=metadata[label['id']];i=label['corner']
        assert label['human_state']==m['human_states_native'][i]
        assert label['native_reference_corner']==m['native_to_reference_corner_ids'][i]
    y=np.asarray([r['usable_label'] for r in labels]);known=y>=0;reg=np.asarray([r['regression_target_eligible'] for r in labels])
    error=np.asarray([r['coordinate_error_px'] if r['coordinate_error_px'] is not None else np.nan for r in labels])
    radius=arrays['RADIUS'];normalized_target=np.asarray([r['normalized_regression_target'] if r['regression_target_eligible'] else [0.,0.] for r in labels])
    target=normalized_target*radius[:,None];rgb=arrays['RGB26'];spatial=np.column_stack([arrays['F4'],arrays['F8']])
    groups=np.asarray([k['session'] for k in keys]);sessions=sorted(set(groups));assert len(sessions)==4
    assert (rgb[:,4:6]==0).all()
    write(HERE/'TRAINING_LABELS.json',dict(records=labels,known=int(known.sum()),positive=int(y[known].sum()),
        unknown_excluded=int((~known).sum()),regression_DIRECT_in16=int(reg.sum()),severe_target_rows=0,
        source='OnlyselectedCAL46 priorauditednativeGTlabelrows; coefficients neverreused'))
    folds=[];oofp=np.zeros(len(keys));oofdelta=np.zeros((len(keys),2))
    for session in sessions:
        train=groups!=session;test=~train;ct=train&known;rt=train&reg
        bs=StandardScaler().fit(spatial[train]);pc=PCA(n_components=32,svd_solver='full').fit(bs.transform(spatial[train]))
        x=np.column_stack([rgb,pc.transform(bs.transform(spatial))])
        cs=StandardScaler().fit(x[ct]);head=LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=20261013).fit(cs.transform(x[ct]),y[ct])
        rs=StandardScaler().fit(x[rt]);ridge=Ridge(alpha=100.).fit(rs.transform(x[rt]),normalized_target[rt])
        oofp[test]=head.predict_proba(cs.transform(x[test]))[:,1];oofdelta[test]=bound(ridge.predict(rs.transform(x[test])),radius[test])
        folds.append(dict(heldout_session=session,feature_training_keys=[k for k,t in zip(keys,train) if t],
            class_training_keys=[k for k,t in zip(keys,ct) if t],ridge_training_keys=[k for k,t in zip(keys,rt) if t],
            heldout_keys=[k for k,t in zip(keys,test) if t],class_metric=metric(y[test&known].astype(bool),oofp[test&known]),
            reg_metric=reg_metric(error,target,oofdelta,test&reg),severe_training_rows=0))
    direct=np.asarray([l['human_state']=='DIRECT_VISIBLE' and l['coordinate_error_px'] is not None for l in labels])
    # For out16 heldout corners target was never a regression training target,
    # but native errors alone lack dxdy in the selected audit. Report regression
    # accuracy only on explicitly declared in16 subset, not fabricated out16.
    diagnostic=dict(schema='NONSEVERE_FIXED_LOGO_CAL46_V1',sessions=sessions,feature_rows=len(keys),frame_count=46,
        known=int(known.sum()),positive=int(y[known].sum()),negative=int((y[known]==0).sum()),
        human_state_counts=dict(collections.Counter(l['human_state'] for l in labels)),
        class_OOF=metric(y[known].astype(bool),oofp[known]),reg_OOF=reg_metric(error,target,oofdelta,reg),
        gated_reg_OOF=reg_metric(error,target,oofdelta,reg&(oofp>=.45)),
        fixed_hyperparameters=p['fixed_hyperparameters'],hyperparameter_retuning=0,TEST_rows_used=0,FINAL_rows_used=0,
        severe_rows_for_any_fit=0,folds=folds,limits=p['limits'])
    write(HERE/'OOF_DIAGNOSTICS.json',diagnostic)
    write(HERE/'OOF_PREDICTIONS.json',[dict(**k,P_usable=float(oofp[i]),RGB_delta_px=oofdelta[i].tolist()) for i,k in enumerate(keys)])
    write(HERE/'OOF_SEAL.json',dict(before_FULL_fit=True,sha256=sha(HERE/'OOF_PREDICTIONS.json'),diagnostics_sha256=sha(HERE/'OOF_DIAGNOSTICS.json'),
        protocol_sha256=sha(HERE/'PROTOCOL.json'),fixed_parameters=True))
    print('NONSEVERE_OOF',diagnostic['class_OOF'],diagnostic['reg_OOF'],flush=True)
    # Every fitted statistic below is freshly derived from filtered CAL46.
    bs=StandardScaler().fit(spatial);pc=PCA(n_components=32,svd_solver='full').fit(bs.transform(spatial))
    x=np.column_stack([rgb,pc.transform(bs.transform(spatial))]);cs=StandardScaler().fit(x[known])
    head=LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=20261013).fit(cs.transform(x[known]),y[known])
    rs=StandardScaler().fit(x[reg]);ridge=Ridge(alpha=100.).fit(rs.transform(x[reg]),normalized_target[reg])
    np.savez_compressed(HERE/'PARAMETERS.npz',spatial_mean=bs.mean_,spatial_scale=bs.scale_,pca_mean=pc.mean_,pca_components=pc.components_,
        class_mean=cs.mean_,class_scale=cs.scale_,class_coefficients=head.coef_,class_intercept=head.intercept_,
        ridge_mean=rs.mean_,ridge_scale=rs.scale_,ridge_coefficients=ridge.coef_,ridge_intercept=ridge.intercept_)
    snapshot=HERE/'code_snapshot';snapshot.mkdir()
    sources=ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1'
    for name in ('joint_observation.py','appearance.py'):shutil.copy2(sources/name,snapshot/name)
    desc=dict(schema='JOINT_OBSERVATION_FROZEN_V1',family='RGB26_F4F8_PCA32',usable_threshold=.45,C=.01,ridge_alpha=100.,
        arrays_file='PARAMETERS.npz',arrays_sha256=sha(HERE/'PARAMETERS.npz'),feature_version='legacy_v1',legacy_zero_columns=[4,5],
        correction_bound_px=16.,geometry_features=False,
        train_scope=dict(partition='CAL',severity_scope='CLEAN_OR_MODERATE_ONLY',allowed_severity=cohort['allowed_severity'],
            training_frame_ids=sorted(allowed),training_frame_count=46,PCA_training_keys=keys,
            classifier_training_keys=[k for k,t in zip(keys,known) if t],ridge_training_keys=[k for k,t in zip(keys,reg) if t],
            feature_rows=len(keys),class_known=int(known.sum()),class_positive=int(y[known].sum()),class_negative=int((y[known]==0).sum()),
            class_unknown_excluded=int((~known).sum()),ridge_DIRECT_in16=int(reg.sum()),
            severe_rows_in_feature_basis=0,severe_rows_in_class_or_regression=0,TEST_rows=0,FINAL_rows=0,
            native_phase='Auditednative_to_referencecorner mapping; GTtargets only'),
        backbone=dict(architecture='ImageNet ResNet18 frozen stem+layer1+layer2',
            weights_repo_relative='data/pallet/results/pallet_resnet18_refiner_20261001_v1/pretrained/resnet18-f37072fd.pth',
            weights_sha256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',trainable_parameters=0,
            normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),
            crop='4r/8r floororigins reflect101 padding linear128 resize',pooling=dict(F4='layer2avg4x4flatten2048',F8='layer2avg2x2flatten512')),
        provenance=dict(protocol_sha256=sha(HERE/'PROTOCOL.json'),cohort_protocol_sha256=p['cohort_protocol_sha256'],
            filtered_feature_seal_sha256=sha(HERE/'FILTERED_FEATURE_SEAL.json'),OOF_diagnostics_sha256=sha(HERE/'OOF_DIAGNOSTICS.json'),
            build_source_sha256=sha(__file__),source_sha256={f.name:sha(f) for f in snapshot.glob('*.py')},
            fitted_coefficients_inherited=False,hyperparameters_inherited=True,hyperparameter_retuning=0,
            classifier_geometry_features=False,regression_geometry_features=False,GT_in_inference=False),limitations=p['limits'])
    write(HERE/'MODEL.json',desc)
    probability=head.predict_proba(cs.transform(x))[:,1];delta=bound(ridge.predict(rs.transform(x)),radius)
    write(HERE/'FULL_PREDICTION_CACHE.json',[dict(**k,P_usable=float(probability[i]),RGB_delta_px=delta[i].tolist()) for i,k in enumerate(keys)])
    write(HERE/'MODEL_FREEZE.json',dict(complete=True,before_nonsevere_TEST_inference=True,before_FULL_CAL_score=True,
        model_sha256=sha(HERE/'MODEL.json'),parameters_sha256=sha(HERE/'PARAMETERS.npz'),protocol_sha256=sha(HERE/'PROTOCOL.json'),
        full_prediction_cache_sha256=sha(HERE/'FULL_PREDICTION_CACHE.json'),build_source_sha256=sha(__file__),
        severe_rows_for_any_fit=0,TEST_rows_for_fit=0,FINAL_rows_used=0))
    model=load_joint_model(HERE/'MODEL.json',device='cpu');pure_p,pure_delta=predict_feature_rows(rgb,spatial,radius,model)
    assert np.max(np.abs(pure_p-probability))<1e-6 and np.max(np.abs(pure_delta-delta))<1e-5
    assert np.array_equal(pure_p>=.45,probability>=.45)
    write(HERE/'PARAMETER_QA.json',dict(rows=len(keys),probability_max_abs=float(np.abs(pure_p-probability).max()),
        delta_max_abs_px=float(np.abs(pure_delta-delta).max()),same_threshold_decisions=True,
        training_frames46=True,no_severe_feature_or_target_rows=True,TEST_rows_read=0,FINAL_rows_read=0))
    print('NONSEVERE_MODEL_FROZEN',sha(HERE/'MODEL.json'),'pureparity',np.abs(pure_p-probability).max(),flush=True)


if __name__=='__main__':main()
