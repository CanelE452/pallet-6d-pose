"""Posthoc frozen-calibration baseline comparison; no new selection or inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_dir):
    run = Path(run_dir).resolve()
    bindings = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None:
            assert digest == expected, f'Frozen input changed: {path}'
        bindings[str(path)] = digest
        return json.loads(data)

    protocol = read(run/'PROTOCOL.json')
    snapshot = read(run/'INPUT_SNAPSHOT.json')
    selection = read(run/'CALIBRATION_SELECTION.json')
    assert selection['complete'] and selection['PASS'] and selection['no_real_GT_used']
    assert selection['protocol_sha256'] == sha(run/'PROTOCOL.json')
    assert selection['input_snapshot_sha256'] == sha(run/'INPUT_SNAPSHOT.json')
    frame_scores = read(run/'CALIBRATION_FRAME_SCORES.json', selection['frame_scores_sha256'])
    old = ROOT/protocol['input_run']
    manifest = read(old/'MANIFEST.json', snapshot['sha256'][str(old/'MANIFEST.json')])
    cache = read(old/'CACHE_RECORDS.json', snapshot['sha256'][str(old/'CACHE_RECORDS.json')])
    cached = {r['id']: r for r in cache['records']}
    source = {r['id']: r for r in manifest['records']}
    pool = [r for r in cache['records'] if r['population'] == 'synth_train']
    cal = protocol['calibration']
    ordered = sorted(pool, key=lambda r: hashlib.sha256((cal['sha256_sort_prefix']+r['id']).encode()).hexdigest())[:256]
    ids = [r['id'] for r in ordered]
    assert ids == selection['calibration_ids'] and len(ids) == len(set(ids)) == 256
    rows, normalized_errors, raw_errors = [], [], []
    for fid in ids:
        rec, sr = cached[fid], source[fid]['source_record']
        assert rec['population'] == 'synth_train' and sr['source_split'] == 'train'
        target = np.asarray(sr['targets'][0]['keypoints_normalized'], dtype=float)
        h, w = sr['prepared_shape_hw']
        xy = target[:8, :2]*[w, h]-sr['reflect_pad_px']
        pred = np.asarray(rec['baseline']['points'], dtype=float)[:8]
        valid = (target[:8, 2] > 0) & np.asarray(rec['baseline']['point_valid'][:8], bool) & bool(rec['loss_matched'])
        diagonal = math.hypot(rec['width'], rec['height'])
        distances = np.linalg.norm(pred-xy, axis=1)
        errors = distances[valid]
        normalized = errors/diagonal
        raw_errors.extend(errors.tolist())
        normalized_errors.extend(normalized.tolist())
        rows.append(dict(id=fid, source=source[fid]['source'], source_index=source[fid]['source_index'],
            observed_corners=int(valid.sum()), mask=valid.tolist(), raw_image_diagonal_px=diagonal,
            raw_corner_errors_px=[float(v) if ok else None for v, ok in zip(distances, valid)],
            normalized_corner_errors=[float(v/diagonal) if ok else None for v, ok in zip(distances, valid)],
            normalized_error_sum=float(normalized.sum()), raw_error_sum=float(errors.sum()),
            image_sha256=source[fid]['image_sha256'], label_sha256=sr['label_sha256']))
    assert len(normalized_errors) == 2022 and sum(r['observed_corners'] for r in rows) == 2022
    identity = float(np.mean(normalized_errors))
    grid = []
    for config in selection['grid']:
        points = [r for r in frame_scores['records'] if r['w_point'] == config['w_point'] and r['w_geom'] == config['w_geom']]
        assert len(points) == 256 and {r['id'] for r in points} == set(ids)
        assert all(r['observed_corners'] == next(b['observed_corners'] for b in rows if b['id'] == r['id']) for r in points)
        denominator = sum(r['observed_corners'] for r in points)
        replay = sum(r['sum_normalized_error'] for r in points)/denominator
        assert denominator == config['n_observed_corners'] == 2022
        assert abs(replay-config['mean_normalized_corner_error']) <= 1e-15
        grid.append({**config, 'objective_replayed_from_saved_sums': replay,
            'delta_vs_unconditional_identity': config['mean_normalized_corner_error']-identity,
            'ratio_to_unconditional_identity': config['mean_normalized_corner_error']/identity})
    selected = next(r for r in grid if all(r[k] == selection['selected'][k] for k in ('w_point', 'w_geom')))
    assert selected['mean_normalized_corner_error'] == selection['selected_objective']
    by_source = {}
    for name in sorted({r['source'] for r in rows}):
        rr = [r for r in rows if r['source'] == name]
        n = sum(r['observed_corners'] for r in rr)
        by_source[name] = dict(n_frames=len(rr), n_observed_corners=n,
            baseline_mean_normalized_corner_error=sum(r['normalized_error_sum'] for r in rr)/n,
            baseline_mean_raw_corner_error_px=sum(r['raw_error_sum'] for r in rr)/n)
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in bindings.items():
        assert sha(path) == digest
    out = dict(schema='pallet_global_layout_calibration_baseline_diagnostic_v1', complete=True, audit_integrity_PASS=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(), input_sha256=bindings,
        posthoc=True, new_model_forwards=0, new_training_updates=0, selection_changed=False, GT_modified=False,
        calibration_ids=ids, n_frames=256, n_observed_corners=2022,
        mask='Same registered source keypoint visibility>0, original predicted availability and original score-only selected-instance IoU match; eight corners only.',
        ranking='Frozen first256 SHA-ranked records from all2048 synth_train; no added eligibility filter.',
        baseline_unconditional_identity=dict(mean_normalized_corner_error=identity,
            mean_raw_corner_error_px=float(np.mean(raw_errors)),
            median_raw_corner_error_px=float(np.median(raw_errors)),
            p90_raw_corner_error_px=float(np.quantile(raw_errors, .9))),
        selected_config=selected, registered_grid=grid, by_source=by_source, per_frame=rows,
        interpretation='The registered grid selected its best configuration, not a demonstrated improvement over keeping the original points unchanged.',
        identity_option_scope='The exact baseline layout was available within every frame candidate bank. A policy that ALWAYS returns baseline regardless of layout score was NOT one of the nine coefficient configurations.',
        limitations=['This additional comparison was calculated after the original calibration and evaluation; it does not replace CALIBRATION_SELECTION or RESULTS.',
            'No new coefficient, expanded grid, threshold or model was selected.',
            'Synthetic calibration shares the frozen backbone training population; this is not a new backbone holdout or real-image performance measurement.'])
    path = run/'CALIBRATION_BASELINE_DIAGNOSTIC.json'
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(path)
    print(json.dumps(dict(path=str(path), sha256=sha(path), baseline=out['baseline_unconditional_identity'],
        selected=selected, counts=[256, 2022]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    main(parser.parse_args().run_dir)
