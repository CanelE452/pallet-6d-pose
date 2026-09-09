"""Frozen-scale DEV evaluation; persist GT-free predictions before GT scoring."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import cache as C
from . import evaluation as S
from . import train as T

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('baseline', 'no_hough', 'hough')


def sources():
    return {str(Path(p).resolve()): C.sha(p) for p in
        (__file__, C.__file__, S.__file__, T.__file__, Path(__file__).with_name('model.py'),
         ROOT/'scripts/research/pallet_dht_structured_v2/infer.py')}


def load_selection(run, arm, seed=1):
    run = Path(run).resolve()
    protocol, bindings = T.verify(run)
    if arm not in protocol['model']['arms'] or seed not in protocol['training']['seeds']:
        raise ValueError('Unregistered local arm/seed')
    path = run/f'SELECTION_{arm}_seed{seed}.json'
    selection = C.read(path)
    checkpoint = run/'runs'/f'{arm}_seed{seed}'/'checkpoint_final.pth'
    if not (selection['complete'] and selection['PASS'] and selection['arm'] == arm
            and selection['seed'] == seed and selection['bindings'] == bindings
            and not selection['real_GT_used'] and not selection['validation_used_for_selection']):
        raise ValueError('Frozen synthetic-only local scale required')
    if selection['scale'] not in protocol['calibration']['output_scale_grid']:
        raise ValueError('Unregistered output scale')
    bound = {str(path): C.sha(path), str(checkpoint): selection['checkpoint_sha256'],
        str(run/'evaluation'/f'{arm}_seed{seed}'/'calibration.npz'): selection['calibration_archive_sha256']}
    C.verify_hashes(bound)
    return selection, bound


def apply_scale(baseline, predicted, point_valid, scale):
    """Same FP64 residual scaling as synthetic metrics; invalid/centre copied."""
    base, pred = np.asarray(baseline, np.float64), np.asarray(predicted, np.float64)
    valid = np.asarray(point_valid, bool)
    if base.shape != pred.shape or base.shape[-2:] != (9, 2) or valid.shape != base.shape[:-1]:
        raise ValueError('Expected matching original nine-point predictions/masks')
    if not np.isfinite(scale) or not 0 <= scale <= 1 or not np.isfinite(base).all() or not np.isfinite(pred).all():
        raise ValueError('Finite predictions and registered [0,1] scale required')
    output = base.copy() if scale == 0 else base + float(scale)*(pred-base)
    output[~valid] = base[~valid]
    output[..., 8, :] = base[..., 8, :]
    return output


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def predict_real(run, seed=1, device='cuda:0', batch_size=16):
    """Read no real annotation file and require the preregistered synthetic gate."""
    run = Path(run).resolve()
    protocol, bindings = T.verify(run)
    gate_path = run/'evaluation'/f'hough_seed{seed}'/'SYNTHETIC_RESULTS.json'
    gate = C.read(gate_path)
    if not (gate['complete'] and gate['PASS'] and gate['arm'] == 'hough' and gate['seed'] == seed
            and gate['bindings'] == bindings and gate['advancement']['advance']):
        raise ValueError('Hough model has not passed the registered synthetic advancement gate')
    bound = {str(run/'PROTOCOL.json'): C.sha(run/'PROTOCOL.json'),
        str(run/'CACHE_COMPLETION.json'): C.sha(run/'CACHE_COMPLETION.json'),
        str(gate_path): C.sha(gate_path),
        str(run/f'SELECTION_hough_seed{seed}.json'): gate['selection_sha256'],
        str(run/'evaluation'/f'hough_seed{seed}'/'synth_val.npz'): gate['validation_archive_sha256']}
    if tuple(protocol['model']['arms']) != ARMS[1:] or batch_size < 1:
        raise ValueError('Expected both fixed arms and positive batch size')
    for arm in ARMS[1:]:
        _, selection_binding = load_selection(run, arm, seed)
        bound.update(selection_binding)
    C.verify_hashes(bound)
    source = sources()
    output_path = run/f'REAL_PREDICTIONS_seed{seed}.json'
    if output_path.exists():
        saved = C.read(output_path)
        if not (saved['complete'] and saved['PASS'] and saved['input_sha256'] == bound
                and saved['source_sha256'] == source and saved['seed'] == seed):
            raise ValueError('Existing frozen real predictions differ')
        C.verify_hashes(saved['evidence_sha256'])
        return output_path
    data = C.LocalData(run)
    indices = data.populations['real_dev']
    if len(indices) != 319:
        raise ValueError('Canonical319 real records required')
    rows = [dict(index=index, id=data.records[index]['id'], image=data.records[index]['image'],
        image_sha256=data.records[index]['image_sha256'], session_id=data.records[index]['session_id'],
        population='real_dev', baseline=data.records[index]['baseline'], arms={}) for index in indices]
    evidence_hash = {}
    from scripts.research.pallet_dht_structured_v2.infer import precision
    for arm in ARMS[1:]:
        selection, _ = load_selection(run, arm, seed)
        model = T.load_trained(run, arm, seed, device).float().eval()
        stored = {}
        with precision(False), torch.inference_mode():
            for first in range(0, len(indices), batch_size):
                ix = indices[first:first+batch_size]
                inputs, targets = data.batch(ix, device)
                if any(bool(value.any()) for value in targets.values()):
                    raise ValueError('Real cached targets are not zero')
                pred, diag = model(inputs)
                if not torch.isfinite(pred).all() or not torch.equal(pred[:, 8], inputs['baseline_points'][:, 8]):
                    raise ValueError('Nonfinite output or changed centroid')
                raw_pred = pred.detach().cpu().numpy()
                original = np.array([rows[first+j]['baseline']['points'] for j in range(len(ix))], np.float64)
                valid = inputs['point_valid'].cpu().numpy()
                scaled = apply_scale(original, raw_pred, valid, selection['scale'])
                for name, value in diag.items():
                    if torch.is_tensor(value):
                        array = value.detach().cpu().numpy()
                        if not np.isfinite(array).all():
                            raise ValueError(f'Nonfinite diagnostic: {name}')
                        if len(array) != len(ix):
                            raise ValueError('Diagnostic batch axis differs')
                        stored.setdefault(name, []).append(array)
                for j in range(len(ix)):
                    compact = {key: plain(value[j]) for key, value in diag.items()
                        if torch.is_tensor(value) and key not in
                        ('probability', 'candidate_scores', 'candidate_valid', 'candidate_coverage')}
                    rows[first+j]['arms'][arm] = dict(points=scaled[j].tolist(),
                        point_valid=valid[j].tolist(), unscaled_points=raw_pred[j].tolist(),
                        scale=selection['scale'], diagnostics=compact)
        path = run/'REAL_LOCAL_EVIDENCE'/f'{arm}_seed{seed}.npz'
        if path.exists():
            raise ValueError('Preserve incomplete evidence before explicit retry')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, indices=np.asarray(indices, np.int64),
            ids=np.asarray([r['id'] for r in rows]),
            **{key: np.concatenate(values) for key, values in stored.items()})
        temporary.replace(path)
        evidence_hash[str(path)] = C.sha(path)
        del model
    C.verify_hashes(bound)
    C.verify_hashes(source)
    result = dict(schema='pallet_dht_local_real_predictions_v3', complete=True, PASS=True,
        seed=seed, records=rows, input_sha256=bound, source_sha256=source,
        evidence_sha256=evidence_hash, selection_uses_GT=False,
        real_annotation_files_opened=0, no_new_backbone_forwards=True,
        evidence_semantics='Actual local Hough posterior/line estimates; not attention or calibrated correctness.')
    C.freeze(output_path, result)
    return output_path


def evaluate_saved(run, seed=1):
    """Use only predictions already persisted before opening original GT files."""
    from scripts.research.pallet_dht_decoder_probe_v1 import evaluate as METRIC
    from scripts.research.pallet_dht_global_layout_v1.evaluation import iou
    run = Path(run).resolve()
    protocol = C.read(run/'PROTOCOL.json')
    pred_path = run/f'REAL_PREDICTIONS_seed{seed}.json'
    predictions = C.read(pred_path)
    if not (predictions['complete'] and predictions['PASS'] and predictions['seed'] == seed
            and not predictions['selection_uses_GT'] and predictions['real_annotation_files_opened'] == 0):
        raise ValueError('Completed GT-free saved real predictions required before GT reads')
    for key in ('input_sha256', 'source_sha256', 'evidence_sha256'):
        C.verify_hashes(predictions[key])
    base = Path(protocol['input_run'])
    if C.sha(base/'PROTOCOL.json') != protocol['input_protocol_sha256']:
        raise ValueError('Original v2 protocol changed')
    lineage = C.read(base/'PROTOCOL.json')
    snapshot_path = Path(lineage['input_snapshot']['path'])
    if C.sha(snapshot_path) != lineage['input_snapshot']['sha256']:
        raise ValueError('Original annotation/source snapshot changed')
    snapshot = C.read(snapshot_path)['sha256']
    bound = {str(pred_path): C.sha(pred_path), str(snapshot_path): C.sha(snapshot_path),
             str(base/'PROTOCOL.json'): C.sha(base/'PROTOCOL.json')}
    def frozen_read(path):
        path = Path(path).resolve()
        digest = snapshot[str(path)]
        if C.sha(path) != digest:
            raise ValueError(f'Original evaluation source changed: {path}')
        bound[str(path)] = digest
        return C.read(path)
    previous = frozen_read(Path(lineage['input_run'])/'FRAME_METRICS.json')
    old = {r['id']: r for r in previous['records'] if r['population'] == 'real_dev'}
    manifest = frozen_read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    items = {r['frame_id']: r for r in manifest['items']}
    if manifest['role'] != 'DEV' or len(items) != 319 or METRIC.BOOTSTRAP_SEED != protocol['evaluation']['bootstrap_seed']:
        raise ValueError('Canonical DEV/seed contract differs')
    rows = []
    for record in predictions['records']:
        fid = record['id']; item = items[fid]; prior = old[fid]
        if Path(record['image']).resolve() != (ROOT/item['image_path']).resolve() or record['session_id'] != prior['session_id']:
            raise ValueError('Canonical image/session changed')
        aa = frozen_read(ROOT/item['gt_v2_path'])['objects'][0]['keypoint_annotations']
        xy = np.array([a['xy'] for a in aa], float)
        gtvalid = np.array([a['visibility'] > 0 for a in aa])
        base_points, valid = METRIC.finite_points(record['baseline']['points'], record['baseline']['point_valid'])
        gtbox = np.r_[xy[:8].min(0), xy[:8].max(0)]
        box = record['baseline']['box_xyxy']
        matched = box is not None and iou(np.asarray(box), gtbox) >= .5
        if not (np.array_equal(xy, prior['gt_points']) and np.array_equal(gtvalid, prior['gt_supervised'])
                and matched == prior['baseline_match_iou50'] and np.array_equal(base_points, prior['arms']['baseline']['points'])):
            raise ValueError('Original GT, mask, IoU or baseline differs')
        row = dict(id=fid, population='real_dev', session_id=record['session_id'], gt_points=xy.tolist(),
            gt_supervised=gtvalid.tolist(), baseline_match_iou50=matched, arms={})
        for arm in ARMS:
            payload = record['baseline'] if arm == 'baseline' else record['arms'][arm]
            points, mask = METRIC.finite_points(payload['points'], payload['point_valid'])
            if not np.array_equal(mask, valid) or not np.array_equal(points[8], base_points[8]):
                raise ValueError('Coverage or centroid changed')
            observed = gtvalid & valid & matched
            distance = np.linalg.norm(points-xy, axis=-1)
            errors = [float(v) if ok else None for v, ok in zip(distance, observed)]
            if arm == 'baseline' and errors != prior['arms']['baseline']['errors_px']:
                raise ValueError('Original baseline numerical replay differs')
            move = np.linalg.norm(points-base_points, axis=-1)
            row['arms'][arm] = dict(points=payload['points'], errors_px=errors,
                move_px=[float(v) if ok else None for v, ok in zip(move, valid)])
        rows.append(row)
    if len(rows) != 319 or len({r['id'] for r in rows}) != 319 or len({r['session_id'] for r in rows}) != 13:
        raise ValueError('Canonical319/13 denominator differs')
    summaries = [dict(arm=arm, **METRIC.summarize(rows, arm)) for arm in ARMS]
    if summaries[0]['n_observed_points'] != 2738 or abs(summaries[0]['p90_px']-41.48732863482036) > 1e-12:
        raise ValueError('Canonical baseline2738/P90 replay failed')
    result = dict(schema='pallet_dht_local_real_results_v3', complete=True, PASS=True, seed=seed,
        summaries=summaries,
        corner_only_summaries=[dict(arm=arm, **METRIC.summarize(rows, arm, tuple(range(8)))) for arm in ARMS],
        difficulty=[row for arm in ARMS for row in METRIC.point_difficulty(rows, arm)],
        paired=[METRIC.paired(rows, 'hough', right, resamples=protocol['evaluation']['bootstrap_resamples'])
                for right in ('baseline', 'no_hough')],
        input_sha256=bound, source_sha256={str(Path(__file__).resolve()): C.sha(__file__),
            str(Path(METRIC.__file__).resolve()): C.sha(METRIC.__file__),
            str(ROOT/'scripts/research/pallet_dht_global_layout_v1/evaluation.py'):
                C.sha(ROOT/'scripts/research/pallet_dht_global_layout_v1/evaluation.py')},
        GT_unchanged=True, same_ID=True, real_training_or_scale_tuning=False,
        independent_final=False, stable_gain_claim=False, active_goal_complete=False,
        metric_scope='Reused DEV319/13 sessions; original nine-point IDs, original IoU matches and missing-point denominators. Local coordinate correction only; not C4 repair.')
    C.verify_hashes(bound)
    C.freeze(run/f'REAL_FRAME_METRICS_seed{seed}.json', dict(complete=True, records=rows))
    C.freeze(run/f'REAL_RESULTS_seed{seed}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--phase', choices=('predict', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase in ('predict', 'all'):
        predict_real(args.run_dir, args.seed, args.device)
    if args.phase in ('evaluate', 'all'):
        result = evaluate_saved(args.run_dir, args.seed)
        print([{key: row[key] for key in ('arm', 'median_px', 'p90_px', 'mean_px')} for row in result['summaries']])
