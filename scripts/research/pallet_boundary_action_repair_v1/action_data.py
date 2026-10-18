"""Prediction-only action construction, and a separate synthetic-label command.

The builder never accepts a reference. H0 keeps historical candidate order and
none semantics; H1 exposes each eligible physical pose once, including N0.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np

SCALARS = 31
FEATURE_DIM = 62
INPUT_DIM = 188
SEEDS = (20261018, 20261019, 20261020)


def read(path):
    with open(path) as f:
        return json.load(f)


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def save(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt') as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')


def bound(path, sealpath):
    seal = read(sealpath); digest = sha(path)
    values = [seal.get(k) for k in ('head_features_sha256', 'features_sha256', 'predictions_sha256', 'sha256', 'output_sha256', 'labels_sha256')]
    artifacts = seal.get('artifacts', {})
    item = artifacts.get(Path(path).name)
    if isinstance(item, dict): item = item.get('sha256')
    values.append(item)
    if digest not in values: raise ValueError('Sealed bytes do not match: ' + str(path))
    return digest


def available(pose):
    if not pose or not pose.get('available'): return False
    try:
        R = np.asarray(pose['R_physical'], float); t = np.asarray(pose['centroid'], float)
        return R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all()
    except (KeyError, TypeError, ValueError): return False


def same_pose(a, b, scale):
    return bool(available(a) and available(b)
                and np.max(np.abs(np.asarray(a['R_physical']) - np.asarray(b['R_physical']))) <= 1e-7
                and np.max(np.abs(np.asarray(a['centroid']) - np.asarray(b['centroid']))) <= 1e-7 * max(scale, 1e-9))


def pose_key(pose):
    """ID-independent, exact numerical order; no physical symmetry collapse."""
    if not available(pose): return ()
    return tuple(np.asarray(pose['R_physical']).ravel()) + tuple(pose['centroid'])


def source_aliases(candidate):
    aliases = set(candidate.get('source_baseline_aliases', [])) | set(candidate.get('source_aliases', []))
    if candidate.get('candidate_id') in ('N0', 'E0', 'E0_REGENERATED'): aliases.add(candidate['candidate_id'])
    if 'E0_REGENERATED' in aliases: aliases.add('E0')
    return aliases


def validate_frame(frame):
    ids = [c['candidate_id'] for c in frame['candidates']]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate candidate ID')
    for c in frame['candidates']:
        x = np.asarray(c['features'], float)
        if x.shape != (FEATURE_DIM,) or not np.isfinite(x).all() or not np.isin(x[SCALARS:], [0, 1]).all():
            raise ValueError('Expected31 signedlog scalars and31 validity bits')
        if not isinstance(c.get('eligible'), bool): raise ValueError('Explicit eligibility required')
    if any(k in frame for k in ('reference', 'R_fixed', 't_gt', 'candidate_metrics', 'target_distribution')):
        raise ValueError('Prediction feature row contains reference/label fields')


def actions(frame, pair=False):
    """Return unique actual eligible actions, N0 first, independent of ID names.

    A missing baseline feature row gets explicit invalid features, never a made
    up energy. Baseline role aliases survive deduplication. Ineligible original
    candidates remain in the input journal, but cannot enter loss or pooling.
    """
    scale = float(frame.get('object_scale_m', 1.0)); candidates = []
    fallback = frame.get('fallback', {})
    for c in frame['candidates']:
        aliases = source_aliases(c)
        is_n0 = same_pose(c['actual_pose'], fallback, scale)
        if is_n0: aliases.add('N0')
        if available(c['actual_pose']) and (c['eligible'] or is_n0):
            x = deepcopy(c); x['source_baseline_aliases'] = sorted(aliases); candidates.append(x)
    if available(fallback) and not any('N0' in source_aliases(c) for c in candidates):
        candidates.append(dict(candidate_id='N0', actual_pose=deepcopy(fallback), eligible=True,
            features=[0.] * FEATURE_DIM, source_baseline_aliases=['N0'], feature_absence='N0 was not in original feature bank'))
    # A D1 producer must expose independently executed E0 even if feature export
    # accidentally omitted it. Missing energies remain explicitly invalid.
    e0 = frame.get('e0_regenerated')
    if available(e0) and not any(same_pose(c['actual_pose'], e0, scale) for c in candidates):
        candidates.append(dict(candidate_id='E0_REGENERATED', actual_pose=deepcopy(e0), eligible=True,
            features=[0.] * FEATURE_DIM, source_baseline_aliases=['E0'], feature_absence='E0 omitted from original feature bank'))
    for c in candidates:
        if available(e0) and same_pose(c['actual_pose'], e0, scale):
            c['source_baseline_aliases'] = sorted(source_aliases(c) | {'E0'})
    candidates.sort(key=lambda c: (0 if 'N0' in source_aliases(c) else 1, pose_key(c['actual_pose'])))
    result = []
    for c in candidates:
        match = next((x for x in result if same_pose(x['actual_pose'], c['actual_pose'], scale)), None)
        if match is not None:
            match['source_baseline_aliases'] = sorted(source_aliases(match) | source_aliases(c))
            match['candidate_aliases'] = sorted(set(match.get('candidate_aliases', [match['candidate_id']])) | {c['candidate_id']})
        else:
            c['is_N0'] = 'N0' in source_aliases(c); c['candidate_aliases'] = [c['candidate_id']]; result.append(c)
    if pair: result = [c for c in result if source_aliases(c) & {'N0', 'E0'}]
    for c in result:
        c['action_role'] = 'N0' if c['is_N0'] else ('E0' if 'E0' in source_aliases(c) else 'OTHER')
    return result


def fit_normalizer(frames):
    values = [np.asarray(c['features'], float) for r in frames for c in r['candidates'] if c['eligible']]
    if not values: raise ValueError('No eligible training features')
    array = np.asarray(values); mean = np.zeros(SCALARS); scale = np.ones(SCALARS); counts = np.zeros(SCALARS, int)
    for j in range(SCALARS):
        x = array[array[:, j + SCALARS] == 1, j]; counts[j] = len(x)
        if len(x):
            mean[j] = x.mean(); s = x.std(ddof=0); scale[j] = s if s > 1e-12 else 1.
    return dict(mean=mean.tolist(), scale=scale.tolist(), valid_train_counts=counts.tolist(),
        fit_scope='train eligible candidates only; shared by H0/H1 for same D', training_frame_ids=[r['id'] for r in frames],
        validity_bits_unchanged=True, scalar_transform='existing signed log1p, unchanged')


def inputs(frame, normalizer, objective, pair=False):
    if objective == 'H0': selected = [c for c in frame['candidates'] if c['eligible']]; entries = selected + [None]
    elif objective == 'H1': selected = actions(frame, pair); entries = selected
    else: raise ValueError(objective)
    mean = np.asarray(normalizer['mean']); scale = np.asarray(normalizer['scale']); seen = np.asarray(normalizer['valid_train_counts']) > 0
    if selected:
        raw = np.asarray([c['features'] for c in selected]); valid = raw[:, SCALARS:] > 0
        normalized = np.where(valid & seen[None, :], (raw[:, :SCALARS] - mean) / scale, 0.)
        x = np.concatenate([normalized, raw[:, SCALARS:]], 1).astype(np.float32)
        pm, px = x.mean(0), x.max(0)
    else: x = np.zeros((0, FEATURE_DIM), np.float32); pm = px = np.zeros(FEATURE_DIM, np.float32)
    out = np.zeros((len(entries), INPUT_DIM), np.float32)
    if selected: out[:len(selected), :FEATURE_DIM] = x
    out[:, FEATURE_DIM:2 * FEATURE_DIM] = pm; out[:, 2 * FEATURE_DIM:3 * FEATURE_DIM] = px
    out[:, 3 * FEATURE_DIM] = np.log1p(len(selected))
    if objective == 'H0': out[-1, -1] = 1.
    elif selected: out[:, -1] = [float(c['is_N0']) for c in selected]
    return out, entries


def pose_error(pose, target):
    if not available(pose): return dict(translation_cm=None, rotation_deg=None, success_5cm5deg=False)
    R = np.asarray(pose['R_physical'], float); t = np.asarray(pose['centroid'], float)
    G = np.asarray(target['R_fixed'], float); gt = np.asarray(target['t'], float); angles = []
    for k in range(int(target['symmetry_order'])):
        theta = k * 2 * np.pi / int(target['symmetry_order'])
        S = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace((G @ S).T @ R) - 1) / 2, -1, 1)))))
    tc, rc = float(np.linalg.norm(t - gt) * 100), min(angles)
    return dict(translation_cm=tc, rotation_deg=rc, success_5cm5deg=bool(tc <= 5 and rc <= 5))


def feature_profile(frames):
    """The same no-reference summaries for D0, D1 and common real inputs."""
    array = [c['features'] for r in frames for c in r['candidates'] if c['eligible']]
    values = np.asarray(array, float).reshape((-1, FEATURE_DIM)); names = frames[0].get('feature_names', []) if frames else []
    stats = []
    for j in range(SCALARS):
        valid = values[:, j + SCALARS] == 1; v = values[valid, j]
        stats.append(dict(index=j, name=names[j] if j < len(names) else str(j), eligible_valid=int(valid.sum()), eligible_missing=int((~valid).sum()),
            signedlog_mean=float(v.mean()) if len(v) else None, signedlog_SD=float(v.std()) if len(v) else None,
            signedlog_min=float(v.min()) if len(v) else None, signedlog_max=float(v.max()) if len(v) else None))
    sources = Counter(); states = Counter(); state_available = 0; covariance_available = 0
    for r in frames:
        if 'state_counts' in r:
            states.update(r['state_counts']); state_available += 1
        covariance_available += int('covariance_summary' in r)
        for c in r['candidates']:
            sources['source_prefix/' + c['candidate_id'].split('/')[0]] += 1
            for alias in source_aliases(c): sources['baseline_role/' + alias] += 1
    return dict(schema='SHARED_REFERENCE_FREE_FEATURE_PROFILE_V1', frames=len(frames), eligible_candidates=len(values), features=stats,
        candidates_per_frame=dict(Counter(str(len(r['candidates'])) for r in frames)),
        eligible_per_frame=dict(Counter(str(sum(c['eligible'] for c in r['candidates'])) for r in frames)), sources=dict(sources),
        state_counts=dict(states), state_counts_frames_available=state_available, covariance_summary_frames_available=covariance_available,
        absent_state_and_covariance_policy='not inferred from missing fields; N/A. Signedlog covariance feature distributions retained if originally exported')


def label_frame(frame, target):
    old = []
    for c in frame['candidates']: old.append(dict(candidate_id=c['candidate_id'], eligible=c['eligible'], **pose_error(c['actual_pose'], target)))
    h1 = []
    for c in actions(frame): h1.append(dict(candidate_id=c['candidate_id'], source_baseline_aliases=c['source_baseline_aliases'], action_role=c['action_role'], **pose_error(c['actual_pose'], target)))
    positives = [i for i, m in enumerate(old) if m['eligible'] and m['success_5cm5deg']]
    h0 = np.zeros(len(old) + 1)
    if positives: h0[positives] = 1 / len(positives)
    else: h0[-1] = 1.
    return dict(id=frame['id'], split=target['split'], h0_target_distribution=h0.tolist(), candidate_metrics=old,
        action_metrics=h1, n0_metric=pose_error(frame.get('fallback'), target), reference_symmetry_order=target['symmetry_order'],
        all_zero_H1=not any(x['success_5cm5deg'] for x in h1), zero_actions_H1=len(h1) == 0,
        reference_kind=target.get('reference_kind', 'BLENDER_EXISTING_REFERENCE'))


def prepare(features, feature_seal, targets, target_seal, split_lock, output, domain):
    start, cpu = time.monotonic(), time.process_time()
    output = Path(output); output.mkdir(parents=True, exist_ok=False); fdigest = bound(features, feature_seal)
    # Synthetic label seal uses its original exact artifact map; no GT conversion.
    tdigest = sha(targets); ts = read(target_seal)
    if tdigest not in json.dumps(ts): raise ValueError('Synthetic target bytes are not bound by original target seal')
    frames = list(rows(features)); gt = {r['id']: r for r in rows(targets)}; lock = read(split_lock)
    expected = {r['id']: r['split'] for r in lock['all_rows']}
    ids = [r['id'] for r in frames]
    if len(ids) != len(set(ids)) or set(ids) != set(expected): raise ValueError('Exact original1003 IDs required')
    for r in frames:
        validate_frame(r)
        if r.get('split') != expected[r['id']] or gt[r['id']]['split'] != expected[r['id']]: raise ValueError('Original asset split changed')
    labels = [label_frame(r, gt[r['id']]) for r in frames]; save(output / 'ACTION_TARGETS.jsonl.gz', labels)
    normalizer = fit_normalizer([r for r in frames if r['split'] == 'train']); write(output / 'TRAIN_ONLY_NORMALIZER.json', normalizer)
    write(output / 'FEATURE_PROFILE.json', feature_profile(frames))
    counts = Counter()
    for r in labels:
        sp = r['split']; counts[sp + '/frames'] += 1; counts[sp + '/with_good_action'] += int(not r['all_zero_H1'])
        counts[sp + '/all_zero_targets'] += int(r['all_zero_H1']); counts[sp + '/zero_actions'] += int(r['zero_actions_H1'])
    gate = dict(train_minimum=32, validation_minimum=16, train_actual=counts['train/with_good_action'], validation_actual=counts['validation/with_good_action'])
    gate['passed'] = gate['train_actual'] >= 32 and gate['validation_actual'] >= 16
    write(output / 'LABEL_SEAL.json', dict(schema='ACTUAL_ACTION_TARGETS_V1', complete=True, domain=domain, features_path=str(features),
        features_sha256=fdigest, features_seal_sha256=sha(feature_seal), targets_sha256=tdigest, target_seal_sha256=sha(target_seal),
        split_lock_sha256=sha(split_lock), labels_sha256=sha(output / 'ACTION_TARGETS.jsonl.gz'), normalizer_sha256=sha(output / 'TRAIN_ONLY_NORMALIZER.json'),
        rows=len(labels), counts=dict(counts), label_support_gate=gate, no_real_training=True, source_sha256=sha(__file__),
        cost=dict(wall_seconds=time.monotonic() - start, CPU_seconds=time.process_time() - cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, device='CPU', threads=1, new_training_runs=0)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--features', type=Path, required=True); p.add_argument('--feature-seal', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True); p.add_argument('--target-seal', type=Path, required=True); p.add_argument('--split-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--domain', choices=['D0', 'D1'], required=True); a = p.parse_args()
    prepare(a.features, a.feature_seal, a.targets, a.target_seal, a.split_lock, a.output, a.domain)


if __name__ == '__main__': main()
