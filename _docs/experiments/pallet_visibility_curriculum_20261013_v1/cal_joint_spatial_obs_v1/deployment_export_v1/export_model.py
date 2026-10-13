"""Fit the predeclared selected model once; seal before any FULL-CAL score."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/minjae/Documents/github/pallet-pose/scripts/research/pallet_visibility_curriculum_20261013_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = read(HERE/'PROTOCOL.json')
    assert not (HERE/'MODEL.json').exists() and not (HERE/'PARAMETERS.npz').exists()
    for binding in p['files'].values():
        assert sha(binding['path']) == binding['sha256']
    seal = read(p['files']['FEATURE_SEAL.json']['path'])
    assert seal['before_CAL_labels'] and seal['GT_read_canary'] == 'PASS'
    keys = read(p['files']['FEATURE_KEYS.json']['path'])
    audit = read(p['files']['LABEL_AUDIT.json']['path'])
    labels = audit['records']
    assert len(labels) == len(keys) == 485
    assert all((a['id'], a['corner']) == (b['id'], b['corner']) for a, b in zip(keys, labels))
    with np.load(p['files']['FEATURES.npz']['path']) as archive:
        rgb = archive['RGB26'].copy()
        spatial = np.column_stack([archive['F4'], archive['F8']])
        radius = archive['RADIUS'].copy()
    assert (rgb[:, 4:6] == 0).all()
    y = np.asarray([r['usable_label'] for r in labels])
    known = y >= 0
    reg = np.asarray([r['regression_target_eligible'] for r in labels])
    target = np.asarray([r['normalized_regression_target'] if r['regression_target_eligible'] else [0, 0] for r in labels])
    assert known.sum() == 475 and y[known].sum() == 188 and reg.sum() == 302
    assert all(r['human_state'] == 'DIRECT_VISIBLE' and r['coordinate_error_px'] <= 16 for r in np.asarray(labels,dtype=object)[reg])
    code = HERE/'code_snapshot'
    code.mkdir(exist_ok=False)
    for name in ('joint_observation.py', 'appearance.py'):
        shutil.copy2(SOURCE/name, code/name)
    shutil.copy2(HERE.parent/'code_snapshot/calibration.py', code/'legacy_calibration.py')
    write(HERE/'FIT_CODE_BINDING.json', dict(export_runner_sha256=sha(__file__),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), before_FULL_CAL_fit=True,
        source_sha256={file.name:sha(file) for file in code.glob('*.py')}))
    # The shared RGB representation sees all prediction-eligible image rows,
    # including unannotated rows, exactly as each LOGO training partition did.
    spatial_scaler = StandardScaler().fit(spatial)
    pca = PCA(n_components=32, svd_solver='full').fit(spatial_scaler.transform(spatial))
    compressed = pca.transform(spatial_scaler.transform(spatial))
    x = np.column_stack([rgb, compressed])
    class_scaler = StandardScaler().fit(x[known])
    classifier = LogisticRegression(C=p['selected_C'], class_weight='balanced', max_iter=3000,
        random_state=20261013).fit(class_scaler.transform(x[known]), y[known])
    ridge_scaler = StandardScaler().fit(x[reg])
    ridge = Ridge(alpha=p['selected_ridge_alpha']).fit(ridge_scaler.transform(x[reg]), target[reg])
    np.savez_compressed(HERE/'PARAMETERS.npz', spatial_mean=spatial_scaler.mean_, spatial_scale=spatial_scaler.scale_,
        pca_mean=pca.mean_, pca_components=pca.components_,
        class_mean=class_scaler.mean_, class_scale=class_scaler.scale_,
        class_coefficients=classifier.coef_, class_intercept=classifier.intercept_,
        ridge_mean=ridge_scaler.mean_, ridge_scale=ridge_scaler.scale_,
        ridge_coefficients=ridge.coef_, ridge_intercept=ridge.intercept_)
    desc = dict(schema='JOINT_OBSERVATION_FROZEN_V1', family=p['selected_family'],
        usable_threshold=p['selected_usable_threshold'], C=p['selected_C'], ridge_alpha=p['selected_ridge_alpha'],
        arrays_file='PARAMETERS.npz', arrays_sha256=sha(HERE/'PARAMETERS.npz'),
        feature_version='legacy_v1', legacy_zero_columns=[4,5],
        feature_transform='spatial2560 float32 trainbasis StandardScaler -> PCA32 -> concatenate RGB26 -> head-specific float64 StandardScaler',
        classifier_target='DIRECT_VISIBLE and raw Euclidean coordinate error<=8 pixels',
        regression_target='(gt_native-q)/raw_appearance_patch_radius; DIRECT_VISIBLE original_error<=16 pixels only',
        correction_bound_px=16., geometry_features=False,
        train_scope=dict(partition='CAL', feature_rows=485, class_known=475, class_positive=188,
            class_negative=287, class_unknown_excluded=10, ridge_DIRECT_in16=302,
            native_phase='CAL native_to_reference_corner_ids used only for training targets',
            PCA_train_basis='all485 prediction-eligible', classifier_scaler_train_basis='known475', ridge_scaler_train_basis='DIRECT_in16_302'),
        backbone=dict(architecture='ImageNet ResNet18 frozen stem+layer1+layer2',
            weights_repo_relative='data/pallet/results/pallet_resnet18_refiner_20261001_v1/pretrained/resnet18-f37072fd.pth',
            weights_sha256=p['backbone_weights_sha256'], trainable_parameters=0,
            normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),
            crop='4r and8r integer floor origins; reflect101 padding; cv2 linear128 resize',
            pooling=dict(F4='layer2[128,16,16] average4x4 flatten2048',F8='layer2 average2x2 flatten512')),
        provenance=dict(export_protocol_sha256=sha(HERE/'PROTOCOL.json'),feature_seal_sha256=p['files']['FEATURE_SEAL.json']['sha256'],
            export_runner_sha256=sha(__file__),source_sha256={file.name:sha(file) for file in code.glob('*.py')},
            classifier_geometry_features=False, regression_geometry_features=False,
            GT_in_inference=False, TEST_reads=0, FINAL_reads=0), limitations=p['caveats'])
    write(HERE/'MODEL.json',desc)
    # Cache predictions from sklearn itself for independent exported-forward
    # testing; labels/errors/targets are excluded from this inference artifact.
    probability = classifier.predict_proba(class_scaler.transform(x))[:,1]
    delta = np.clip(ridge.predict(ridge_scaler.transform(x))*radius[:,None],-16,16)
    delta *= np.minimum(1,16/np.maximum(np.linalg.norm(delta,axis=1),1e-12))[:,None]
    records = [dict(**k, P_usable=float(probability[i]), RGB_delta_px=delta[i].tolist()) for i,k in enumerate(keys)]
    write(HERE/'FULL_PREDICTION_CACHE.json',records)
    write(HERE/'MODEL_FREEZE.json',dict(complete=True,before_FULL_CAL_pose_or_scoring=True,
        protocol_sha256=sha(HERE/'PROTOCOL.json'),model_sha256=sha(HERE/'MODEL.json'),parameters_sha256=sha(HERE/'PARAMETERS.npz'),
        full_prediction_cache_sha256=sha(HERE/'FULL_PREDICTION_CACHE.json'),export_runner_sha256=sha(__file__),
        hyperparameter_retuning=0,TEST_reads=0,FINAL_reads=0))
    print('JOINT_MODEL_FROZEN',sha(HERE/'MODEL.json'),'rows',len(records),flush=True)


if __name__ == '__main__':
    main()
