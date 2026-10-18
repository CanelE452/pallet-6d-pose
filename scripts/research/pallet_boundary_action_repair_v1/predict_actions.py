"""All2x2 models consume the exact same real feature bytes, without references."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import time
import torch
from .action_data import bound, read, rows, save, sha, write, validate_frame
from .action_models import ARMS, infer, make_model
from .train_actions import costs
from ..pallet_statewise_end_to_end_recovery_v1.learned_head import prediction_no_refs


def predict(features, feature_seal, training_dir, output, arms=None, metadata=None):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); start, cpu = time.monotonic(), time.process_time()
    output.mkdir(parents=True, exist_ok=False); digest = bound(features, feature_seal); manifest = read(training_dir / 'TRAINING_MANIFEST.json')
    models = {}; records = {}; fs = read(feature_seal)
    with prediction_no_refs():
        meta = {r['id']: {k: r.get(k) for k in ('partition', 'session', 'cluster_id')} for r in rows(metadata)} if metadata else {}
        for run in manifest['runs']:
            arm, seed = run['arm'], run['seed']
            if arms and arm not in arms: continue
            path = Path(run['checkpoint'])
            if sha(path) != run['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            domain, objective, architecture, pair = ARMS[arm]
            if pair and not fs.get('e0_independently_executed', False): raise ValueError('Real pair bank lacks independent E0 execution proof')
            model = make_model(architecture); model.load_state_dict(ck['model_state_dict']); model.eval(); model.requires_grad_(False)
            name = arm + '_s' + str(seed); models[name] = model, ck['normalizer'], objective, pair
            records[name] = dict(checkpoint_sha256=sha(path), reused_checkpoint=run.get('checkpoint_reused', False), training_domain=domain, objective=objective, architecture=architecture, pair=pair)
        if not models: raise ValueError('No frozen trained/reused models')
        result = []; counts = Counter()
        for frame in rows(features):
            validate_frame(frame); predictions = {name: infer(frame, *m) for name, m in models.items()}
            for name, p in predictions.items(): counts[name + '/' + p['action_role']] += 1
            fields = {k: frame.get(k) for k in ('partition', 'session', 'cluster_id')}
            if metadata:
                if frame['id'] not in meta: raise ValueError('Canonical inference metadata ID missing')
                for k, value in fields.items():
                    if value is not None and value != meta[frame['id']][k]: raise ValueError('Canonical metadata disagreement: ' + k)
                fields = meta[frame['id']]
            result.append(dict(id=frame['id'], split=frame.get('split'), **fields, predictions=predictions, reference_inputs=False))
        save(output / 'PREDICTIONS.jsonl.gz', result)
    write(output / 'PREDICTION_SEAL.json', dict(schema='COMMON_FEATURE_ACTION_PREDICTIONS_V1', complete=True, rows=len(result),
        predictions_sha256=sha(output / 'PREDICTIONS.jsonl.gz'), features_sha256=digest, feature_seal_sha256=sha(feature_seal),
        training_manifest_sha256=sha(training_dir / 'TRAINING_MANIFEST.json'), checkpoints=records, before_reference_decode=True, reference_reads=0,
        all_2x2_same_common_real_features=True, ordered_ids=[r['id'] for r in result], action_source_counts=dict(counts),
        tiny_head_forward_calls=len(result) * len(models), new_backbone_forwards=0,
        canonical_metadata_sha256=sha(metadata) if metadata else None, cost=costs(start, cpu), source_sha256=sha(__file__)))
    print('ACTION_PREDICTIONS_SEALED', len(result), len(models), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--features', type=Path, required=True); p.add_argument('--feature-seal', type=Path, required=True)
    p.add_argument('--training-dir', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--arms', nargs='*', choices=list(ARMS))
    p.add_argument('--metadata', type=Path, help='Canonical reference-free real input238; only ID/partition/session/cluster_id read'); a = p.parse_args()
    predict(a.features, a.feature_seal, a.training_dir, a.output, a.arms, a.metadata)


if __name__ == '__main__': main()
