"""Apply the three original frozen D1 H1 heads; never fit or select a new head."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time


def predict(features, feature_seal, training_dir, output):
    import torch
    from scripts.research.pallet_boundary_action_repair_v1.action_data import (
        bound, read, rows, save, sha, write, validate_frame)
    from scripts.research.pallet_boundary_action_repair_v1.action_models import infer, make_model
    from scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head import prediction_no_refs

    features, feature_seal, training_dir, output = map(Path, (features, feature_seal, training_dir, output))
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    digest = bound(features, feature_seal)
    fs = read(feature_seal)
    frames = list(rows(features))
    ids = [r['id'] for r in frames]
    if len(ids) != 238 or len(set(ids)) != 238:
        raise ValueError('All 238 unique frames must be retained, including no-pose frames')
    if Counter(r['partition'] for r in frames) != Counter(CAL=46, TEST=49, REMAINDER=143):
        raise ValueError('Exact original populations are required')
    if ids != fs['ordered_ids']:
        raise ValueError('Sealed feature IDs/order changed')
    models, checkpoints, forwards = {}, {}, Counter()
    with prediction_no_refs():
        manifest = read(training_dir / 'TRAINING_MANIFEST.json')
        runs = [r for r in manifest['runs'] if r['arm'] == 'D1_H1_MLP']
        if sorted(r['seed'] for r in runs) != [20261018, 20261019, 20261020]:
            raise ValueError('Exactly the three original D1 heads are required')
        normalizers = []
        for run in runs:
            path = Path(run['checkpoint'])
            if sha(path) != run['checkpoint_sha256']:
                raise ValueError('Original H1 checkpoint changed')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = make_model('MLP')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval().requires_grad_(False)
            name = 'D1_H1_MLP_s' + str(run['seed'])
            def hook(module, args, result, arm=name):
                forwards[arm] += 1
            model.register_forward_hook(hook)
            normalizer = checkpoint['normalizer']
            normalizers.append(normalizer)
            normalizer_sha = hashlib.sha256(json.dumps(normalizer, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            models[name] = model, normalizer
            checkpoints[name] = dict(checkpoint_sha256=sha(path), normalizer_sha256=normalizer_sha,
                reused_original_weights=True, own_backbone_fit=False, architecture='188_64_64_1', parameter_count=sum(p.numel() for p in model.parameters()))
        if any(n != normalizers[0] for n in normalizers[1:]):
            raise ValueError('Original three heads must retain their common original normalizer')
        result, roles = [], Counter()
        for frame in frames:
            validate_frame(frame)
            predictions = {name: infer(frame, model, norm, 'H1', False)
                           for name, (model, norm) in models.items()}
            for name, value in predictions.items():
                roles[name + '/' + value['action_role']] += 1
            result.append(dict(id=frame['id'], split=frame.get('split'),
                **{k: frame[k] for k in ('partition', 'session', 'cluster_id')},
                predictions=predictions, reference_inputs=False))
        save(output / 'PREDICTIONS.jsonl.gz', result)
    seal = dict(schema='FROZEN_D1_THREE_BACKBONE_HEAD_PREDICTION_SEAL_V1', complete=True,
        rows=len(result), ordered_ids=ids, predictions_sha256=sha(output / 'PREDICTIONS.jsonl.gz'),
        features_sha256=digest, feature_seal_sha256=sha(feature_seal),
        training_manifest_sha256=sha(training_dir / 'TRAINING_MANIFEST.json'), checkpoints=checkpoints,
        same_original_H1_and_train_normalizer_for_every_backbone=True, own_backbone_fit=False,
        actual_tiny_head_forward_calls=dict(forwards), total_actual_tiny_head_forward_calls=sum(forwards.values()),
        no_pose_frames_do_not_call_MLP=True, action_role_counts=dict(roles),
        new_training_steps=0, new_backbone_or_N3_forwards=0, reference_reads=0, before_reference_decode=True,
        resource=dict(wall_seconds=time.monotonic()-start, CPU_seconds=time.process_time()-cpu,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), source_sha256=sha(__file__))
    write(output / 'PREDICTION_SEAL.json', seal)
    print(json.dumps(dict(stage='FROZEN_D1_H1', complete=True, rows=len(result), forwards=sum(forwards.values()))), flush=True)
    return seal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'feature-seal', 'training-dir', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    predict(args.features, args.feature_seal, args.training_dir, args.output)
