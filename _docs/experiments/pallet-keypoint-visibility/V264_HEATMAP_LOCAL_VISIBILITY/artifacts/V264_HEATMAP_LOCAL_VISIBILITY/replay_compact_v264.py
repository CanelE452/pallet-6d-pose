"""Replay V264 compact descriptors and final classifiers on CPU, without fitting.

The published package can be moved: serialized absolute artifact paths are never
followed. Each bound file is mapped explicitly to a fixed path below --root.
No DINO tokens, source RGB, backbone, spatial head, or optimizer are needed.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import types

import numpy as np
import torch


MODEL_SHA256 = 'd412ed01b844ad8e8374d423c3015f20bdcc42a778855aad6d721492613c8c7b'
ARMS = ('GLOBAL_MATCHED', 'HEATMAP_LOCAL')
TOLERANCE = 1e-5


def safe(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(
            word in str(p).lower() for word in ('v112', 'cache', '.pyc')):
        raise ValueError('Rejected artifact path: ' + str(p))
    for ancestor in reversed((p, *p.parents)):
        if ancestor.is_symlink():
            raise ValueError('Symlink artifact ancestor: ' + str(ancestor))
    return p


def binding(path):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(),
            'size_bytes': p.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).absolute().parent)
    parser.add_argument('--output', default='VERIFY_COMPACT.json',
                        help='New output filename below --root; existing files are never overwritten.')
    args = parser.parse_args()
    root = safe(args.root.absolute())
    output_name = Path(args.output)
    assert not output_name.is_absolute() and len(output_name.parts) == 1
    output = safe(root / output_name)
    assert not output.exists(), 'Refusing to overwrite: ' + str(output)
    verified_files = {}

    def mapped(bound, relative):
        """Only caller-declared filenames are opened; bound path is inert."""
        rel = Path(relative)
        assert not rel.is_absolute() and '..' not in rel.parts
        p = safe(root / rel)
        assert Path(bound['path']).name == p.name, (relative, bound['path'])
        actual = binding(p)
        assert actual['sha256'] == bound['sha256'], ('SHA mismatch', relative)
        assert actual['size_bytes'] == bound['size_bytes'], ('Size mismatch', relative)
        verified_files[str(rel)] = actual
        return p

    def json_file(path):
        return json.loads(safe(path).read_bytes())

    # The configuration seals final checkpoints before any current annotation
    # access. Its own integrity is also covered by the publication manifest.
    freeze_path = safe(root / 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json')
    freeze = json_file(freeze_path)
    assert freeze['completed'] and freeze['PASS']
    assert freeze['threshold_E_logit'] == 0 and tuple(freeze['arms']) == ARMS
    model_path = mapped(freeze['model_code'], 'v264_model.py')
    assert freeze['model_code']['sha256'] == MODEL_SHA256
    module = types.ModuleType('v264_compact_bound_model')
    module.__file__ = str(model_path)
    sys.modules[module.__name__] = module
    exec(compile(model_path.read_bytes(), str(model_path), 'exec'), module.__dict__)
    assert tuple(module.ARMS) == ARMS
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    protocol_path = mapped(freeze['protocol'], 'PROTOCOL_BEFORE_FIT.json')
    complete_path = mapped(freeze['training_complete'], 'syn_training_v1/COMPLETE.json')
    complete = json_file(complete_path)
    assert complete['completed'] and complete['PASS']
    mapped(complete['protocol'], 'PROTOCOL_BEFORE_FIT.json')
    assert complete['current_or_real_fit_reads'] == 0
    assert complete['new_optimizer_updates'] == 2000
    outputs = complete['outputs']

    pooled_path = mapped(outputs['SYN_POOLED_FEATURES.npz'],
                         'syn_training_v1/SYN_POOLED_FEATURES.npz')
    with np.load(pooled_path, allow_pickle=False) as archive:
        syn = {key: np.array(archive[key], copy=True) for key in archive.files}
    fit = syn['fit'].astype(bool)
    y = syn['y'].astype(np.int64)
    assert fit.shape == y.shape == (7040,) and fit.sum() == 5504
    assert np.isin(y, [-1, 0, 1]).all()
    assert syn['global_features'].shape == syn['local_features'].shape == (7040, 384)
    assert syn['auxiliary'].shape == (7040, 12)
    assert syn['native'].shape == (7040,) and np.isin(syn['native'], range(8)).all()
    draws_path = mapped(outputs['FIT_DRAWS1000x32.npy'],
                        'syn_training_v1/FIT_DRAWS1000x32.npy')
    draws = np.load(draws_path, allow_pickle=False)
    assert draws.shape == (1000, 32) and np.issubdtype(draws.dtype, np.integer)
    assert ((draws >= 0) & (draws < len(y))).all()
    assert fit[draws].all()
    assert ((y[draws] == 0).sum(1) == 16).all()
    assert ((y[draws] == 1).sum(1) == 16).all()
    draws_sha = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()

    barrier_path = mapped(outputs['BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json'],
                          'syn_training_v1/BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json')
    barrier = json_file(barrier_path)
    assert barrier['completed'] and barrier['current_reads'] == 0
    assert barrier['new_spatial_updates'] == 0
    assert barrier['SYN_DEV_classifier_forwards'] == 0
    assert barrier['frozen_spatial_state_sha256'] == complete['frozen_spatial_state_sha256']
    syn_report_path = mapped(outputs['SYN_DEV_RESULTS.json'],
                            'syn_training_v1/SYN_DEV_RESULTS.json')
    saved_syn_report = json_file(syn_report_path)
    assert saved_syn_report['completed'] and saved_syn_report['fixed_threshold'] == 0
    models = {}
    proofs = complete['arms']
    initial_hashes = []
    parameters = []
    syn_results = {}
    indices = np.flatnonzero(~fit)

    def assert_close(a, b, label):
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.shape == b.shape, (label, a.shape, b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(), label
        delta = float(np.max(np.abs(a - b))) if a.size else 0.0
        assert delta <= TOLERANCE, (label, delta, TOLERANCE)
        return delta

    for arm in ARMS:
        proof = proofs[arm]
        assert proof == barrier['arms'][arm]
        assert proof['steps'] == 1000 and proof['draws_sha256'] == draws_sha
        initial_hashes.append(proof['initial_state_sha256'])
        torch.manual_seed(17)
        classifier = module.VisibilityReadout().cpu()
        assert module.state_sha(classifier.state_dict()) == proof['initial_state_sha256']
        count = sum(parameter.numel() for parameter in classifier.parameters())
        assert count == proof['parameters'] == 25025
        parameters.append(count)
        name = arm + '_FINAL1000.pt'
        cp_path = mapped(freeze['checkpoints'][arm], 'syn_training_v1/' + name)
        mapped(outputs[name], 'syn_training_v1/' + name)
        mapped(proof['final'], 'syn_training_v1/' + name)
        checkpoint = torch.load(io.BytesIO(cp_path.read_bytes()), map_location='cpu')
        assert checkpoint['arm'] == arm and checkpoint['steps'] == 1000
        assert checkpoint['threshold_E_logit'] == 0 and checkpoint['input_dimension'] == 780
        classifier.load_state_dict(checkpoint['state_dict'], strict=True)
        assert module.state_sha(classifier.state_dict()) == proof['final_state_sha256']
        classifier.eval()
        models[arm] = classifier
        log_name = arm + '_TRAIN1000.jsonl'
        log_path = mapped(outputs[log_name], 'syn_training_v1/' + log_name)
        logs = [json.loads(line) for line in log_path.read_bytes().splitlines()]
        assert len(logs) == 1000 and [entry['step'] for entry in logs] == list(range(1, 1001))
        assert all(np.isfinite(entry['BCE_balanced_batch']) for entry in logs)
        pred_name = arm + '_SYN_DEV_PREDICTIONS.npz'
        saved_path = mapped(outputs[pred_name], 'syn_training_v1/' + pred_name)
        with np.load(saved_path, allow_pickle=False) as archive:
            saved = {key: np.array(archive[key], copy=True) for key in archive.files}
        assert np.array_equal(saved['source_index'], indices)
        assert np.array_equal(saved['y'], y[indices])
        # Source-invalid rows have no token support; no GT field is needed to
        # recover the exact availability that was used for DEV inference.
        available = syn['available'][indices].astype(bool)
        assert np.array_equal(saved['available'], available)
        logits, pred = module.infer(classifier, module.design(syn, arm)[indices], available, 'cpu')
        maximum = assert_close(logits, saved['E_logit'], arm + ' SYN DEV logits')
        assert np.array_equal(pred, saved['pred']), arm + ' SYN DEV decisions'
        metrics = module.summary(y[indices], logits, pred)
        assert metrics == saved_syn_report['results'][arm]
        assert metrics == complete['SYN_DEV_results'][arm]
        syn_results[arm] = {'max_abs_logit_difference': maximum,
                            'predictions_equal': True, 'metrics': metrics}

    assert len(set(initial_hashes)) == len(set(parameters)) == 1
    # Current inference must already be sealed; this replay never launches it.
    current_freeze_path = safe(root / 'CURRENT_PREDICTIONS_FROZEN.json')
    current_freeze = json_file(current_freeze_path)
    assert current_freeze['completed'] and current_freeze['PASS']
    mapped(current_freeze['configuration'], 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json')
    assert current_freeze['current_GT_or_annotation_reads'] == 0
    assert current_freeze['new_optimizer_updates'] == 0
    current_features_path = mapped(current_freeze['feature_packet'],
                                   'current_inference_v1/CURRENT_POOLED_FEATURES.npz')
    with np.load(current_features_path, allow_pickle=False) as archive:
        current = {key: np.array(archive[key], copy=True) for key in archive.files}
    current_predictions_path = mapped(current_freeze['predictions'],
                                      'CURRENT_ALL1784_PREDICTIONS.jsonl')
    records = [json.loads(line) for line in current_predictions_path.read_bytes().splitlines()]
    assert len(records) == current_freeze['rows'] == 1784
    assert all(record['global_index'] == index for index, record in enumerate(records))
    original_available = np.array([record['available'] for record in records], bool)
    token_available = current['available'].astype(bool)
    assert token_available.shape == original_available.shape == (1784,)
    assert (token_available | ~original_available).all()
    available = token_available & original_available
    assert available.sum() == current_freeze['supported'] == 1739
    code = {-1: 'U', 0: 'D', 1: 'E'}
    current_results = {}
    for arm in ARMS:
        logits, pred = module.infer(models[arm], module.design(current, arm), available, 'cpu')
        saved_logits = np.array([record[arm]['E_logit'] for record in records], np.float64)
        maximum = assert_close(logits, saved_logits, arm + ' current logits')
        assert [code[int(value)] for value in pred] == [record[arm]['pred'] for record in records]
        current_results[arm] = {
            'max_abs_logit_difference': maximum, 'predictions_equal': True,
            'prediction_counts': {name: int((pred == value).sum()) for value, name in code.items()}}

    result = {
        'completed': True, 'PASS': True, 'mode': 'CPU compact descriptor replay',
        'absolute_logit_tolerance': TOLERANCE,
        'new_optimizer_updates': 0, 'new_backbone_forwards': 0,
        'new_spatial_head_forwards': 0, 'RGB_or_full_DINO_token_reads': 0,
        'current_GT_or_annotation_reads': 0,
        'proofs': {'same_initial_weights': True, 'same_draws': True,
                   'same_parameters': parameters[0], 'steps_each': 1000,
                   'FIT_only_draws': True, 'each_batch_D16_E16': True,
                   'draws_sha256': draws_sha,
                   'initial_state_sha256': initial_hashes[0]},
        'SYN_DEV': syn_results, 'current_predictions': current_results,
        'current_metrics_scope': 'Prediction replay only; no current annotation or confusion-matrix join.',
        'configuration': binding(freeze_path), 'protocol': binding(protocol_path),
        'training_complete': binding(complete_path),
        'current_prediction_freeze': binding(current_freeze_path),
        'verified_files': verified_files,
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu'}}
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'output': binding(output),
                      'SYN_DEV': syn_results, 'current_predictions': current_results},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
