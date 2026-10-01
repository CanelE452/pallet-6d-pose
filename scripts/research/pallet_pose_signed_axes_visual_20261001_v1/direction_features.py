"""Fixed input-only direction18 construction; importing installs no read hook.

Projection and tolerance are exactly the completed TRAIN input audit's pure
operators. Runtime adds no padding, PnP, inference, reference label or new norm.
The artifact helpers are called only after an evaluator verifies certified fits.
"""
from pathlib import Path
import numpy as np
from . import common as C
from scripts.research.pallet_pose_residual_direction_audit_20261001_v1 import direction_features as O

MODELS = O.MODELS
HYP = O.HYP
RULE = O.RULE
DIM = 18
NORMALIZATION = 'R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_anchor'
project_direction = O.project_direction
parity = O.parity
array_sha = O.array_sha


def validate_normalization(mean, std, expected_sha=None):
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    assert mean.shape == std.shape == (DIM,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std >= np.float32(1e-6)).all()
    digest = array_sha(np.stack([mean, std]))
    if expected_sha is not None:
        assert digest == expected_sha, 'DIRECTION_NORMALIZATION_DRIFT'
    return mean, std


def direction_difference(direction18, valid, anchor_index, mean18, std18):
    raw, valid, anchor = map(np.asarray, (direction18, valid, anchor_index))
    assert raw.dtype == np.float32 and valid.dtype == bool and valid.ndim == 2
    assert raw.shape == (*valid.shape, DIM) and valid.shape[1] in (2, 4)
    assert anchor.shape == (len(valid),) and np.issubdtype(anchor.dtype, np.integer)
    assert np.isin(anchor, [-1, 0, 1]).all()
    present = valid.any(1)
    assert np.array_equal(anchor >= 0, present), 'DIRECTION_ANCHOR_SUPPORT_CONTRACT'
    rows = np.flatnonzero(present)
    assert valid[rows, anchor[rows]].all()
    assert np.isfinite(raw[valid]).all()
    mean, std = validate_normalization(mean18, std18)
    normalized = np.zeros(raw.shape, np.float64)
    normalized[valid] = ((raw[valid] - mean) / std).astype(np.float64)
    result = np.zeros_like(normalized)
    i, j = np.nonzero(valid)
    result[i, j] = normalized[i, j] - normalized[i, anchor[i]]
    assert np.isfinite(result).all() and not result[~valid].any()
    assert not result[rows, anchor[rows]].any()
    return result


def normalization_receipt(protocol):
    binding = protocol['inputs']['direction_receipt']
    assert binding == C.bind(C.DIRECTION_DOC / 'FEATURE_AUDIT.json')
    receipt = C.read(C.ROOT / binding['path'])
    assert receipt['complete'] and receipt['PASS'] and receipt['source_TRAIN_only']
    assert receipt['frames'] == 2598 and receipt['direction_dim'] == DIM and receipt['direction_rule'] == RULE
    norm = receipt['normalization']
    assert norm['valid_candidate_count'] == 5194 and norm['source'] == 'R0_eligible_train_valid_candidates'
    assert norm['std_floor'] == 1e-6 and norm['dtype'] == 'float32'
    mean, std = validate_normalization(norm['mean18'], norm['std18'], norm['array_sha'])
    return binding, norm['array_sha'], mean, std


def verify_checkpoint(checkpoint, protocol):
    binding, digest, mean, std = normalization_receipt(protocol)
    assert checkpoint['direction_receipt_binding'] == binding
    assert checkpoint['direction_normalization_sha'] == digest
    actual_mean, actual_std = validate_normalization(checkpoint['direction_mean'], checkpoint['direction_std'], digest)
    np.testing.assert_array_equal(actual_mean, mean)
    np.testing.assert_array_equal(actual_std, std)
    return dict(direction_receipt_binding=binding, direction_normalization_sha=digest)


def verify_training_audit(protocol_binding):
    """Independent certification is required before any new VAL input extraction."""
    path = C.DOC / 'TRAIN_CONVERGENCE.json'
    receipt = C.read(path)
    assert receipt['complete'] and receipt['PASS'] and receipt['protocol'] == protocol_binding
    assert set(receipt['models']) == set(C.MODEL_NAMES)
    # The independent audit binds the exact fit inputs/outputs; verify each
    # embedded binding without importing its TRAIN label reader.
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                C.verify(value)
            else:
                for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(receipt)
    return C.bind(path)


def extract_candidate_pair(prediction, record, K, features94, valid, feature_names):
    """Pure two-hypothesis projection with unchanged validity and fixed parity."""
    features94, valid = np.asarray(features94), np.asarray(valid)
    assert features94.dtype == np.float32 and features94.shape == (2, 94)
    assert valid.dtype == bool and valid.shape == (2,)
    assert len(feature_names) == len(set(feature_names)) == 94
    px = [feature_names.index(f'residual{k}_px') for k in range(9)]
    norm = [feature_names.index(f'residual{k}_bboxnorm') for k in range(9)]
    result = np.zeros((2, DIM), np.float32)
    audits = []
    hypotheses = {h['name']: h for h in record['hypotheses']}
    assert len(hypotheses) == len(record['hypotheses']) and set(hypotheses) <= set(HYP)
    if valid.any():
        selected = prediction['selected_index']
        assert selected is not None and 0 <= selected < len(prediction['candidates'])
        candidate = prediction['candidates'][selected]
        points = np.asarray(candidate['keypoints_xy'], np.float64)
        assert points.shape == (9, 2) and np.isfinite(points).all()
        assert not (points == -1).all(1).any()
    for j, name in enumerate(HYP):
        if not valid[j]:
            continue
        hypothesis = hypotheses[name]
        assert hypothesis['pose']['available']
        output = project_direction(hypothesis['pose'], points, candidate['box_xyxy'], K)
        checks = dict(raw94_px=parity(output['norm_px'], features94[j, px]),
            raw94_bboxnorm=parity(output['norm_px']/output['diagonal'], features94[j, norm], diagonal=output['diagonal']),
            frozen_corner8_px=parity(output['norm_px'][:8], hypothesis['inference_cues']['corner8_residual_px']))
        result[j] = output['direction18']
        audits.append(checks)
    assert not result[~valid].any()
    return result, audits


def _paths(scope):
    assert scope in ('SOURCE_VAL', 'REAL')
    return C.RAW / f'{scope}_DIRECTIONS.npz', C.DOC / f'{scope}_DIRECTION_INPUTS.json'


def load_frozen(scope, rows, indices, anchor, masks, protocol, inputs):
    path, receipt_path = _paths(scope)
    receipt = C.read(receipt_path)
    assert receipt['complete'] and receipt['PASS'] and receipt['scope'] == scope
    assert receipt['frames'] == len(rows) and receipt['inputs'] == inputs
    assert receipt['protocol'] == protocol and receipt['directions'] == C.bind(path)
    assert receipt['direction_receipt_binding'] == inputs['direction_receipt'] == C.bind(C.DIRECTION_DOC/'FEATURE_AUDIT.json')
    fixed=C.read(C.ROOT/inputs['direction_receipt']['path'])
    assert fixed['complete'] and fixed['PASS']
    assert receipt['direction_normalization_sha'] == fixed['normalization']['array_sha']
    assert receipt['rule'] == RULE and receipt['parity'] == dict(atol_px=O.ATOL_PX, rtol=O.RTOL)
    assert receipt['additional_padding'] == 0 and receipt['source_label_values_read'] is False
    assert receipt['real_reference_values_read'] is False and receipt['normalization_recomputed'] is False
    assert receipt['operators'] == [C.bind(Path(__file__)), C.bind(Path(O.__file__))]
    assert receipt['training_verification'] == inputs['training_verification'] == C.bind(C.DOC/'TRAIN_CONVERGENCE.json')
    with np.load(path, allow_pickle=False) as stored:
        expected = {'ids', 'source_index', 'anchor_index'} | {m+s for m in MODELS for s in ('_direction18', '_valid')}
        assert set(stored.files) == expected
        np.testing.assert_array_equal(stored['ids'], [r['id'] for r in rows])
        np.testing.assert_array_equal(stored['source_index'], np.asarray(indices, np.int64))
        np.testing.assert_array_equal(stored['anchor_index'], anchor)
        directions = {}
        for model in MODELS:
            values = stored[model+'_direction18']
            assert values.dtype == np.float32 and values.shape == (len(rows), 2, DIM)
            np.testing.assert_array_equal(stored[model+'_valid'], masks[model])
            assert np.isfinite(values).all() and not values[~masks[model]].any()
            assert array_sha(values) == receipt['models'][model]['direction_sha']
            assert array_sha(masks[model]) == receipt['models'][model]['valid_sha']
            directions[model] = values.copy()
    return directions, dict(direction_inputs=C.bind(receipt_path), directions=C.bind(path),
        direction_normalization_sha=receipt['direction_normalization_sha'],
        direction_receipt_binding=receipt['direction_receipt_binding'])


def freeze_inputs(scope, rows, indices, anchor, features, masks, poses, names,
                  prediction_for, protocol_binding, training_protocol, inputs):
    """No target access: freeze 18D numeric inputs before learned routing."""
    path, receipt_path = _paths(scope)
    failed = C.DOC / f'{scope}_DIRECTION_INPUTS_FAILED.json'
    assert not failed.exists(), 'Frozen parity failure prohibits retries or tolerance changes.'
    norm_binding, norm_sha, _, _ = normalization_receipt(training_protocol)
    train_audit = verify_training_audit(C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))
    if receipt_path.exists():
        directions, bindings = load_frozen(scope, rows, indices, anchor, masks, protocol_binding, inputs)
        assert bindings['direction_receipt_binding'] == norm_binding and bindings['direction_normalization_sha'] == norm_sha
        return directions, bindings
    assert not path.exists(), 'Partial artifact must not be silently overwritten.'
    try:
        saved = dict(ids=np.asarray([r['id'] for r in rows]), source_index=np.asarray(indices, np.int64), anchor_index=np.asarray(anchor, np.int64))
        summaries = {}
        for model in MODELS:
            values = np.zeros((len(rows), 2, DIM), np.float32)
            maxima = {key: dict(max_absolute=0., max_fraction_of_tolerance=0.) for key in ('raw94_px','raw94_bboxnorm','frozen_corner8_px')}
            prediction_bindings = []
            for i, row in enumerate(rows):
                prediction, binding = prediction_for(model, i, row)
                prediction_bindings.append(binding)
                values[i], checks = extract_candidate_pair(prediction, poses[model][row['id']], row['K'], features[model][i], masks[model][i], names)
                for check in checks:
                    for key, result in check.items():
                        for stat, value in result.items(): maxima[key][stat] = max(maxima[key][stat], value)
            saved[model+'_direction18'], saved[model+'_valid'] = values, masks[model].copy()
            summaries[model] = dict(valid_candidates=int(masks[model].sum()), allinvalid_rows=int((~masks[model].any(1)).sum()),
                direction_sha=array_sha(values), valid_sha=array_sha(masks[model]), raw94_sha=array_sha(features[model]),
                prediction_bindings=prediction_bindings, parity=maxima)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle: np.savez_compressed(handle, **saved)
        C.save(receipt_path, dict(complete=True, PASS=True, created_at=C.now(), scope=scope, frames=len(rows),
            rule=RULE, dimension=DIM, protocol=protocol_binding, inputs=inputs, directions=C.bind(path), models=summaries,
            direction_receipt_binding=norm_binding, direction_normalization_sha=norm_sha, training_verification=train_audit,
            operators=[C.bind(Path(__file__)), C.bind(Path(O.__file__))], parity=dict(atol_px=O.ATOL_PX, rtol=O.RTOL),
            source_label_values_read=False, real_reference_values_read=False, normalization_recomputed=False,
            original_validity_preserved=True, additional_padding=0, new_PnP_solves=0, image_forwards=0, fits_executed=0,
            coordinate_contract='Use cached cf_extents/R_cf/centroid, observed q9 and K without extra padding; physical dimension ordering is not substituted.',
            container_disclosure='Source containers can include TRAIN and VAL; only the requested frozen source_index prediction files/pose rows are used.'))
    except Exception as error:
        if not failed.exists():
            C.save(failed, dict(complete=False, PASS=False, scope=scope, error_type=type(error).__name__, error=str(error),
                parity=dict(atol_px=O.ATOL_PX, rtol=O.RTOL), no_tolerance_relaxation=True,
                source_label_values_read=False, real_reference_values_read=False, fits_executed=0))
        raise
    return load_frozen(scope, rows, indices, anchor, masks, protocol_binding, inputs)


def selfcheck():
    pose = dict(available=True, cf_extents=[1.2,.8,.2], R_cf=np.eye(3).tolist(), centroid=[.1,-.2,3.])
    K = np.array([[700.,0.,320.],[0.,710.,240.],[0.,0.,1.]])
    box = np.array([100.,100.,500.,400.])
    q = np.zeros((9,2))
    base = project_direction(pose,q,box,K)
    delta = np.arange(18).reshape(9,2)/17.
    q = base['projected']-delta
    output = project_direction(pose,q,box,K)
    np.testing.assert_allclose(output['direction64'],(delta/500.).ravel(),atol=1e-15,rtol=0)
    padded_K=K.copy();padded_K[:2,2]+=100.
    np.testing.assert_allclose(project_direction(pose,q+100.,box+100.,padded_K)['direction18'],output['direction18'],rtol=0,atol=1e-9)
    names=[f'residual{k}_{suffix}' for suffix in ('px','bboxnorm') for k in range(9)]+[f'other{k}' for k in range(76)]
    features=np.zeros((2,94),np.float32)
    features[:,:9]=output['norm_px'];features[:,9:18]=output['norm_px']/output['diagonal']
    record=dict(hypotheses=[dict(name=name,pose=pose,inference_cues=dict(corner8_residual_px=output['norm_px'][:8].tolist())) for name in HYP])
    prediction=dict(selected_index=0,candidates=[dict(keypoints_xy=q.tolist(),box_xyxy=box.tolist())])
    pair,checks=extract_candidate_pair(prediction,record,K,features,np.array([True,False]),names)
    np.testing.assert_array_equal(pair[0],output['direction18']);assert not pair[1].any() and len(checks)==1
    bad=features.copy();bad[0,0]+=1.
    try:extract_candidate_pair(prediction,record,K,bad,np.array([True,False]),names)
    except AssertionError:pass
    else:raise AssertionError('PARITY_MISMATCH_MUST_STOP_WITHOUT_MASK_CHANGE')
    raw=np.zeros((3,4,18),np.float32);raw[0]=np.arange(72,dtype=np.float32).reshape(4,18);raw[2]=np.nan
    valid=np.array([[True]*4,[False,True,True,False],[False]*4]);anchor=np.array([0,1,-1])
    mean=np.arange(18,dtype=np.float32)/10;std=np.arange(18,dtype=np.float32)+1
    result=direction_difference(raw,valid,anchor,mean,std)
    expected=np.zeros_like(result)
    for i,j in zip(*np.nonzero(valid)):
        for k in range(18):
            expected[i,j,k]=float(np.float32((raw[i,j,k]-mean[k])/std[k]))-float(np.float32((raw[i,anchor[i],k]-mean[k])/std[k]))
    np.testing.assert_array_equal(result,expected)
    assert not result[~valid].any() and not result[[0,1],[0,1]].any()
    print('DIRECTION_RUNTIME_PURE_SELFCHECK_PASS')


if __name__ == '__main__':
    selfcheck()
