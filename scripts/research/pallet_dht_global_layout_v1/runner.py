"""Synthetic-only calibration then saved-cache whole-layout selection.

No network is loaded. Geometry receives an explicit GT-free input whitelist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FRAME_KEYS = ('logits', 'theta', 'rho', 'lattice_valid', 'feature_shape_hw', 'input_shape_hw', 'raw_to_input_affine')
ARMS = ('baseline', 'independent', 'global', 'global_shared_only')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Cache:
    def __init__(self, run_dir):
        self.run = Path(run_dir).resolve()
        self.protocol = read(self.run/'PROTOCOL.json')
        self.protocol_sha = sha(self.run/'PROTOCOL.json')
        self.snapshot = read(self.run/'INPUT_SNAPSHOT.json')
        require(self.snapshot['complete'] and self.snapshot['PASS'] and self.snapshot['protocol_sha256'] == self.protocol_sha, 'Completed protocol-bound immutable input preflight required')
        self.expected = self.snapshot['sha256']
        self.source = ROOT/self.protocol['input_run']
        self.consumed = {}
        self.manifest = self.read(self.source/'MANIFEST.json')
        self.completion = self.read(self.source/'CACHE_COMPLETION.json')
        self.extraction = self.read(self.source/'EXTRACTION_COMPLETE.json')
        cached = self.read(self.source/'CACHE_RECORDS.json')
        require(self.manifest['complete'] and self.completion['complete'] and self.completion['PASS'] and self.extraction['complete'] and self.extraction['PASS'], 'Completed frozen feature cache required')
        self.records = {r['id']: r for r in cached['records']}
        self.records_by_index = {r['index']: r for r in cached['records']}
        self.source_records = {r['id']: r for r in self.manifest['records']}
        require(len(self.records) == len(self.records_by_index) == 2879, 'All2879 source rows required')
        require({k: len(v) for k, v in self.manifest['populations'].items()} == dict(synth_train=2048, synth_val=512, real_dev=319), 'Population sizes changed')
        cal = self.protocol['calibration']
        pool = [r for r in cached['records'] if r['population'] == 'synth_train']
        self.calibration_records = sorted(pool, key=lambda r: hashlib.sha256((cal['sha256_sort_prefix']+r['id']).encode()).hexdigest())[:cal['n_frames']]
        # "clean_only" means no injected perturbation; there is NO visibility filter.
        require(len(self.calibration_records) == 256, 'Prespecified256 calibration frames required')
        self.evaluation_records = [self.records_by_index[i] for pop in ('synth_val', 'real_dev') for i in self.manifest['populations'][pop]]
        require(not ({r['id'] for r in self.calibration_records} & {r['id'] for r in self.evaluation_records}), 'Calibration/evaluation overlap')

    def bytes(self, path):
        path = Path(path).resolve()
        require(str(path) in self.expected, f'Input absent from completed snapshot: {path}')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(digest == self.expected[str(path)], f'Frozen input changed: {path}')
        self.consumed[str(path)] = digest
        return data

    def read(self, path):
        return json.loads(self.bytes(path))

    def load(self, record):
        frame_path = self.source/'frames'/f"{record['index']:06d}.npz"
        evidence_path = Path(record['candidate_evidence_npz'])
        require(self.extraction['frame_sha256'][str(frame_path.resolve())] == self.expected[str(frame_path.resolve())], 'Raw frame completion binding differs')
        require(self.completion['candidate_evidence_sha256'][str(evidence_path.resolve())] == self.expected[str(evidence_path.resolve())], 'Candidate evidence completion binding differs')
        with np.load(io.BytesIO(self.bytes(frame_path)), allow_pickle=False) as z:
            frame = {k: z[k].copy() for k in FRAME_KEYS}
        with np.load(io.BytesIO(self.bytes(evidence_path)), allow_pickle=False) as z:
            evidence = {k: z[k].copy() for k in z.files if k.startswith('line_fusion__')}
        public = dict(id=record['id'], width=record['width'], height=record['height'],
            baseline={k: record['baseline'][k] for k in ('points', 'point_valid', 'point_conf', 'detected')})
        require(not any('gt' in k.lower() for k in frame), 'GT key in inference frame inputs')
        return public, frame, evidence

    def bindings(self):
        return dict(protocol_sha256=self.protocol_sha, input_snapshot_sha256=sha(self.run/'INPUT_SNAPSHOT.json'),
            consumed_input_sha256=dict(self.consumed), source_sha256=source_hashes())


def source_hashes():
    paths = [HERE/n for n in ('runner.py', 'evaluation.py', 'geometry.py')]
    paths += [HERE.parent/'pallet_dht_decoder_probe_v1'/n for n in ('geometry.py', 'evaluate.py')]
    return {str(p.resolve()): sha(p) for p in paths}


def verify_code_freeze(run):
    path = Path(run)/'SOURCE_FREEZE.json'
    require(path.exists(), 'SOURCE_FREEZE.json must precede calibration')
    freeze = read(path)
    mapping = freeze.get('sha256', freeze.get('source_sha256', {}))
    for path, digest in source_hashes().items():
        require(mapping.get(path, mapping.get(str(Path(path).relative_to(ROOT)))) == digest, f'Actual consumed code is not frozen: {path}')


def preflight(cache):
    result = dict(schema='pallet_global_layout_runner_preparation_v1', complete=True, PASS=True,
        calibration_ids=[r['id'] for r in cache.calibration_records],
        calibration_indices=[r['index'] for r in cache.calibration_records],
        calibration_ranking='First256 SHA-ranked records from ALL2048 synthetic train; no visibility or matching eligibility filtering.',
        calibration_GT_used_only_for_objective=True, evaluation_count=831,
        no_new_CNN_forwards=True, no_new_training=True, bindings=cache.bindings())
    write(cache.run/'RUNNER_PREPARATION.json', result)
    return result


def geometry_call(cache, record):
    from . import geometry as G
    public, frame, evidence = cache.load(record)
    prepared = G.prepare(public, frame, evidence)
    return G, prepared, frame, evidence


def calibrate(cache):
    verify_code_freeze(cache.run)
    target = cache.run/'CALIBRATION_SELECTION.json'
    if target.exists():
        saved = read(target)
        require(saved['complete'] and saved['PASS'] and saved['protocol_sha256'] == cache.protocol_sha
            and saved['source_sha256'] == source_hashes()
            and saved['calibration_ids'] == [r['id'] for r in cache.calibration_records], 'Existing calibration provenance differs')
        return saved
    cal = cache.protocol['calibration']
    combinations = [(float(p), float(g)) for p in cal['w_point'] for g in cal['w_geometry']]
    values = {c: [] for c in combinations}
    observations = []
    n_mask_frames = 0
    start = time.monotonic()
    for j, rec in enumerate(cache.calibration_records):
        G, prepared, _, _ = geometry_call(cache, rec)
        # Supervised target is deliberately separate from every geometry call.
        sr = cache.source_records[rec['id']]['source_record']
        normalized = np.asarray(sr['targets'][0]['keypoints_normalized'], float)
        h, w = sr['prepared_shape_hw']
        gt = normalized[:8, :2]*[w, h]-sr['reflect_pad_px']
        valid = (normalized[:8, 2] > 0) & np.asarray(rec['baseline']['point_valid'][:8], bool) & bool(rec['loss_matched'])
        n_mask_frames += int(valid.any())
        diagonal = math.hypot(rec['width'], rec['height'])
        for point_weight in cal['w_point']:
            bank = G.build_layouts(prepared, w_point=float(point_weight))
            for geometry_weight in cal['w_geometry']:
                chosen = G.select_layout(prepared, bank, w_geom=float(geometry_weight), top_n=3)
                pp = np.asarray(chosen['points_xy'], float)
                require(pp.shape == (9, 2) and np.isfinite(pp).all(), 'Nonfinite calibration output')
                error = np.linalg.norm(pp[:8]-gt, axis=1)[valid]/diagonal
                values[float(point_weight), float(geometry_weight)].extend(error.tolist())
                observations.append(dict(id=rec['id'], w_point=float(point_weight), w_geom=float(geometry_weight),
                    observed_corners=int(valid.sum()), mean_normalized_error=float(error.mean()) if len(error) else None,
                    sum_normalized_error=float(error.sum()), selected_index=int(chosen.get('selected_index', 0))))
        if (j+1) % 16 == 0:
            print(f'Calibration {j+1}/256 elapsed={time.monotonic()-start:.1f}s', flush=True)
    grid = []
    for (p, g), errors in values.items():
        require(len(errors) > 0, 'Empty calibration objective')
        grid.append(dict(w_point=p, w_geom=g, n_observed_corners=len(errors), mean_normalized_corner_error=float(np.mean(errors))))
    minimum = min(g['mean_normalized_corner_error'] for g in grid)
    tied = [g for g in grid if g['mean_normalized_corner_error'] <= minimum+cal['tie_absolute_tolerance']]
    selected = sorted(tied, key=lambda g: (-g['w_point'], g['w_geom']))[0]
    verify_code_freeze(cache.run)
    write(cache.run/'CALIBRATION_FRAME_SCORES.json', dict(complete=True, records=observations))
    result = dict(schema='pallet_global_layout_calibration_selection_v1', complete=True, PASS=True,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=cache.protocol_sha,
        input_snapshot_sha256=sha(cache.run/'INPUT_SNAPSHOT.json'), source_sha256=source_hashes(),
        no_real_GT_used=True, no_real_predictions_computed_before_freeze=True,
        calibration_ids=[r['id'] for r in cache.calibration_records], n_calibration_frames=256,
        n_frames_with_supervised_corners=n_mask_frames, no_GT_based_frame_filter=True,
        objective='Pooled mean of matched supervised original-ID corner Euclidean error divided by each raw-image diagonal.',
        grid=grid, selected={k: selected[k] for k in ('w_point', 'w_geom')},
        selected_objective=selected['mean_normalized_corner_error'], tie_absolute_tolerance=cal['tie_absolute_tolerance'],
        tie_order='stronger point prior, then lower geometry weight', elapsed_seconds=time.monotonic()-start,
        frame_scores_sha256=sha(cache.run/'CALIBRATION_FRAME_SCORES.json'), bindings=cache.bindings())
    write(target, result)
    print(f'Calibration FROZEN: {result["selected"]}', flush=True)
    return result


def normalize_hypothesis(value, w_point, w_geom):
    out = plain(value)
    if 'points_xy' not in out:
        for k in ('hypothesis_xy', 'points'):
            if k in out:
                out['points_xy'] = out[k]
                break
    if 'indices' not in out and 'candidate_indices' in out:
        out['indices'] = out['candidate_indices']
    if 'total' not in out:
        for k in ('score_total', 'total_score', 'total_cost', 'score'):
            if k in out:
                out['total'] = out[k]
                break
    if 'term_dict' not in out:
        out['term_dict'] = out.get('raw_terms', {k: out[k] for k in ('shared_line_cost', 'independent_line_cost', 'point_prior_cost', 'geometry_cost') if k in out})
    out['selected_config'] = dict(w_point=w_point, w_geom=w_geom)
    return out


def normalized_selection(chosen, w_point, w_geom):
    selected = normalize_hypothesis(chosen.get('selected', chosen), w_point, w_geom)
    top = [normalize_hypothesis(v, w_point, w_geom) for v in chosen.get('top_hypotheses', [])]
    pp = np.asarray(chosen['points_xy'], float)
    valid = np.asarray(chosen['point_valid'], bool)
    require(pp.shape == (9, 2) and valid.shape == (9,) and np.isfinite(pp).all(), 'Invalid selected layout')
    return dict(points=pp.tolist(), point_valid=valid.tolist(),
        selected_candidate_indices=selected.get('indices', plain(chosen.get('candidate_indices', []))),
        selected_hypothesis_index=int(chosen.get('selected_index', 0)),
        score_total=selected.get('total'), score_terms=selected.get('term_dict', {}),
        top_hypotheses=top, selected=selected,
        explicit_hypotheses={k: normalize_hypothesis(v, w_point, w_geom) for k, v in chosen.get('explicit_hypotheses', {}).items()},
        n_hypotheses=chosen.get('n_hypotheses'), n_geometry_valid=chosen.get('n_geometry_valid'),
        score_gap=chosen.get('score_gap'), baseline_selected=chosen.get('baseline_selected'),
        baseline_geometry_exception=chosen.get('baseline_geometry_exception', False),
        baseline_exception_selected=chosen.get('baseline_exception_selected', False),
        identity_forced_by_input_failure=chosen.get('identity_forced_by_input_failure', False),
        geometry_penalty_enabled=chosen.get('geometry_penalty_enabled', False),
        geometry_safety_guard_retained=chosen.get('geometry_safety_guard_retained', False),
        uses_gt=chosen.get('uses_gt', False),
        fallback_reason=chosen.get('fallback_reason'),
        selection_uses_GT=False)


def predict(cache, selection):
    from . import geometry as G
    verify_code_freeze(cache.run)
    path = cache.run/'PREDICTIONS.json'
    selection_sha = sha(cache.run/'CALIBRATION_SELECTION.json')
    if path.exists():
        saved = read(path)
        require(saved['complete'] and saved['PASS'] and saved['calibration_selection_sha256'] == selection_sha
            and saved['source_sha256'] == source_hashes(), 'Existing prediction provenance differs')
        return saved
    point_weight, geometry_weight = selection['selected']['w_point'], selection['selected']['w_geom']
    records = []
    start = time.monotonic()
    for j, rec in enumerate(cache.evaluation_records):
        _, prepared, frame, evidence = geometry_call(cache, rec)
        bank = G.build_layouts(prepared, w_point=point_weight)
        independent = G.independent_select(prepared, w_point=point_weight)
        global_selection = G.select_layout(prepared, bank, w_geom=geometry_weight, top_n=3)
        shared = G.select_layout(prepared, bank, w_geom=0., top_n=3)
        arms = dict(independent=normalized_selection(independent, point_weight, 0.),
            global_shared_only=normalized_selection(shared, point_weight, 0.))
        arms['global'] = normalized_selection(global_selection, point_weight, geometry_weight)
        baseline = rec['baseline']
        for arm in arms.values():
            require(np.array_equal(arm['points'][8], baseline['points'][8]), 'Centroid changed')
            require(arm['point_valid'] == baseline['point_valid'], 'Availability changed')
        # Selected bank/candidate evidence is GT-free. Full original48 are kept by
        # reference for the evaluator's later, explicitly labelled oracle audit.
        known_keys = ('candidate_xy', 'candidate_valid', 'candidate_source_slot', 'candidates_xy', 'line_peaks_h_raw', 'line_peak_valid', 'edges', 'sigma', 'lines', 'line_weights', 'available_line_roles')
        info = {k: plain(prepared[k]) for k in known_keys if k in prepared}
        for k in ('line_peaks_h_raw', 'line_peak_valid'):
            if k not in info:
                info[k] = plain(evidence['line_fusion__'+k])
        info.update(candidate_evidence_path=rec['candidate_evidence_npz'],
            candidate_evidence_sha256=cache.completion['candidate_evidence_sha256'][rec['candidate_evidence_npz']],
            raw_frame_path=str((cache.source/'frames'/f"{rec['index']:06d}.npz").resolve()),
            raw_to_input_affine=plain(frame['raw_to_input_affine']), input_shape_hw=plain(frame['input_shape_hw']))
        info['edges'] = plain(G.EDGES)
        # Every named explicit whole-layout seed survives, including aliases.
        info['explicit_seed_hypotheses'] = arms['global']['explicit_hypotheses']
        row = dict(id=rec['id'], index=rec['index'], population=rec['population'], image=rec['image'],
            image_sha256=rec['image_sha256'], image_key=rec['image_key'], width=rec['width'], height=rec['height'],
            session_id=rec['session_id'], baseline=baseline, arms=arms, evidence=info)
        if rec['population'] == 'synth_val':
            row['synthetic_loss_matched'] = rec['loss_matched']
        records.append(row)
        if (j+1) % 32 == 0:
            print(f'Layout prediction {j+1}/831 elapsed={time.monotonic()-start:.1f}s', flush=True)
    verify_code_freeze(cache.run)
    result = dict(schema='pallet_dht_global_layout_predictions_v1', complete=True, PASS=True,
        protocol_sha256=cache.protocol_sha, calibration_selection_sha256=selection_sha, source_sha256=source_hashes(),
        selection_uses_GT=False, no_new_CNN_forward=True, n_records=len(records), records=records,
        score_definition=dict(direction='lower_is_better', point_weight=point_weight, geometry_weight=geometry_weight,
            independent='mean8(mean3 incident line-mode mixture cost) + w_point*mean8 point prior',
            global_score='mean12 shared-mode endpoint cost + w_point*mean8 point prior + w_geom*projective residual cost',
            geometry='Unrestricted projective cuboid fit; selected points are NOT replaced by fitted reprojections.',
            search='Four fixed corner orders, beam64; approximate restricted hypothesis bank plus explicit whole-layout seeds.'),
        elapsed_seconds=time.monotonic()-start, bindings=cache.bindings())
    write(path, result)
    return result


def main(run_dir, phase):
    cache = Cache(run_dir)
    if phase in ('preflight', 'all'):
        preflight(cache)
    if phase in ('calibrate', 'all'):
        calibrate(cache)
    if phase in ('predict', 'all'):
        selection = read(cache.run/'CALIBRATION_SELECTION.json')
        require(selection['complete'] and selection['PASS'] and selection['no_real_GT_used'], 'Freeze calibration before any evaluation selection')
        predict(cache, selection)
    if phase in ('evaluate', 'all'):
        from .evaluation import evaluate
        results = evaluate(cache.run)
        print(json.dumps(dict(complete=True, continuation=results['continuation']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--phase', choices=('preflight', 'calibrate', 'predict', 'evaluate', 'all'), default='preflight')
    args = parser.parse_args()
    main(args.run_dir, args.phase)
