"""Independent action/split arithmetic audit; never reads real references.

No production action, input, loss, label or model functions are imported. The
auditor reads sealed synthetic labels and their renderer reference only. It may
inspect reference-free real predictions after those have been sealed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

SEEDS = [20261018, 20261019, 20261020]
GROUPS = ['D0_H0_MLP', 'D0_H1_MLP', 'D1_H0_MLP', 'D1_H1_MLP', 'D1_H1_ROLE_AFFINE', 'D1_H1_PAIR_MLP', 'D1_H1_PAIR_ROLE_AFFINE']
FEATURE_NAMES = ['fixed_energy', 'A_energy', 'B_energy', 'AB_energy', 'edge_energy', 'full_energy', 'partial_energy', 'outline_energy', 'A_owner_entropy', 'A_none_mean', 'AB_owner_entropy', 'complete_n', 'partial_n', 'B_supported_n', 'B_profile_n', 'B_entropy_mean', 'fallback_observation_fraction', 'point_covariance_sigma_mean', 'point_residual_mean_px', 'point_residual_p90_px', 'partial_normal_mean_sigma', 'N0_translation_fraction', 'N0_rotation_radians', 'min_constraint_slack', 'positive_depth_fraction', 'B_consumed_fraction', 'score_gap_fixed', 'score_gap_A', 'score_gap_B', 'score_gap_AB', 'score_gap_edge']
# Before the first actual audit: float64 dot/acos roundoff gives a worst angular
# endpoint deviation on the order sqrt(eps) radians. This1e-5degree bound is for
# independent arithmetic, never a change to the exact5degree success threshold.
T_TOL_CM = 1e-8
R_TOL_DEG = 1e-5
NORM_TOL = 1e-10


def read(path):
    with open(path) as f: return json.load(f)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def records(path):
    fn = gzip.open if str(path).endswith('.gz') else open
    with fn(path, 'rt') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def good_pose(p):
    if not isinstance(p, dict) or p.get('available') is not True: return False
    try:
        a, t = np.array(p['R_physical'], dtype=float), np.array(p['centroid'], dtype=float)
        return a.shape == (3, 3) and t.shape == (3,) and bool(np.isfinite(a).all() and np.isfinite(t).all())
    except (KeyError, TypeError, ValueError): return False


def equal_pose(p, q, scale):
    if not good_pose(p) or not good_pose(q): return False
    return (all(abs(p['R_physical'][i][j] - q['R_physical'][i][j]) <= 1e-7 for i in range(3) for j in range(3))
            and all(abs(p['centroid'][i] - q['centroid'][i]) <= 1e-7 * max(float(scale), 1e-9) for i in range(3)))


def aliases(c):
    result = set(c.get('source_baseline_aliases', [])) | set(c.get('source_aliases', []))
    if c.get('candidate_id') in ('N0', 'E0', 'E0_REGENERATED'): result.add(c['candidate_id'])
    if 'E0_REGENERATED' in result: result.add('E0')
    return result


def independently_measure(p, reference):
    if not good_pose(p): return dict(translation_cm=None, rotation_deg=None, success_5cm5deg=False)
    delta = np.asarray(p['centroid'], dtype=float) - np.asarray(reference['t'], dtype=float)
    translation = 100. * math.sqrt(math.fsum(float(v) ** 2 for v in delta))
    original_expression_translation = math.sqrt(float(np.dot(delta, delta))) * 100.
    # Frobenius inner product tr((G*S)^T R), explicitly expanded. The original
    # renderer matrix is used as stored; no SVD/orthogonalization/correction.
    G, R = reference['R_fixed'], p['R_physical']; angles = []
    for phase in range(int(reference['symmetry_order'])):
        a = 2 * math.pi * phase / int(reference['symmetry_order']); c, s = math.cos(a), math.sin(a)
        GS = [[G[i][0] * c - G[i][2] * s, G[i][1], G[i][0] * s + G[i][2] * c] for i in range(3)]
        inner = math.fsum(GS[i][j] * R[i][j] for i in range(3) for j in range(3))
        angles.append(math.degrees(math.acos(min(1., max(-1., (inner - 1.) / 2.)))))
    rotation = min(angles)
    return dict(translation_cm=translation, original_expression_translation_cm=original_expression_translation,
        rotation_deg=rotation, success_5cm5deg=translation <= 5. and rotation <= 5.)


class Audit:
    def __init__(self):
        self.checks = Counter(); self.failures = []; self.maxima = Counter(); self.pending = []; self.representation_exceptions = []

    def check(self, condition, kind, context=''):
        self.checks[kind] += 1
        if not condition: self.failures.append(dict(kind=kind, context=context))

    def close(self, actual, expected, tolerance, kind, context=''):
        if actual is None or expected is None: self.check(actual is None and expected is None, kind, context); return
        delta = abs(float(actual) - float(expected)); self.maxima[kind] = max(self.maxima[kind], delta)
        self.check(delta <= tolerance, kind, dict(context=context, actual=actual, independent=expected, absolute_difference=delta, tolerance=tolerance))

    def metrics(self, reported, independent, context):
        actual, stable = reported.get('translation_cm'), independent['translation_cm']
        if actual is not None and stable is not None:
            difference = abs(actual - stable)
            self.maxima['independent_fsum_translation_absolute_difference'] = max(self.maxima['independent_fsum_translation_absolute_difference'], difference)
            if difference > T_TOL_CM:
                # Do not widen the original fixed absolute tolerance. Preserve
                # every such exception and demand exact original-expression
                # reproduction instead. The v1 FAIL is retained separately.
                self.representation_exceptions.append(dict(context=context, original=actual, independent_fsum=stable,
                    original_expression=independent['original_expression_translation_cm'], absolute_difference=difference,
                    original_fixed_absolute_tolerance=T_TOL_CM, original_ULP=math.ulp(actual), difference_ULPs=difference / math.ulp(actual)))
                self.check(independent['original_expression_translation_cm'] == actual, 'absolute_bound_exception_exactly_reproduced', context)
        self.close(actual, independent.get('original_expression_translation_cm'), 0., 'original_expression_translation_bitwise', context)
        self.close(reported.get('rotation_deg'), independent['rotation_deg'], R_TOL_DEG, 'rotation_arithmetic', context)
        self.check(reported.get('success_5cm5deg') == independent['success_5cm5deg'], 'strict_joint_success', context)


def inspect_feature_frame(audit, frame):
    fid = frame['id']; audit.check(frame.get('feature_names') == FEATURE_NAMES, 'exact31_feature_name_whitelist', fid)
    banned = {'R_fixed', 't_gt', 'reference', 'candidate_metrics', 'target_distribution', 'h0_target_distribution', 'action_metrics', 'translation_cm', 'rotation_deg', 'success_5cm5deg'}
    audit.check(not (set(frame) & banned), 'no_top_level_label_fields', fid)
    ids = []
    for c in frame['candidates']:
        ids.append(c['candidate_id']); x = np.asarray(c['features'], float)
        audit.check(x.shape == (62,) and bool(np.isfinite(x).all()) and bool(np.isin(x[31:], [0., 1.]).all()), 'numeric62_validity_whitelist', fid + '/' + c['candidate_id'])
        audit.check(type(c.get('eligible')) is bool, 'explicit_eligibility', fid)
        audit.check(not (set(c) & banned), 'no_candidate_label_fields', fid + '/' + c['candidate_id'])
    audit.check(len(ids) == len(set(ids)), 'unique_candidate_ID', fid)


def expected_action_coverage(audit, frame, action_rows, action_pose_map):
    """Check coverage/uniqueness directly, without rebuilding production order."""
    fid = frame['id']; scale = frame.get('object_scale_m', 1.); fallback = frame.get('fallback', {})
    poses = [action_pose_map[r['candidate_id']] for r in action_rows]
    audit.check(len(action_rows) == len({r['candidate_id'] for r in action_rows}), 'unique_action_ID', fid)
    for i, p in enumerate(poses):
        audit.check(good_pose(p), 'actual_action_pose_available', fid)
        for j in range(i): audit.check(not equal_pose(p, poses[j], scale), 'no_duplicate_actual_actions', fid)
    n0 = [i for i, p in enumerate(poses) if equal_pose(p, fallback, scale)]
    audit.check(len(n0) == int(good_pose(fallback)), 'explicit_available_N0_once', fid)
    for index in n0: audit.check('N0' in action_rows[index].get('source_baseline_aliases', []), 'N0_alias_preserved', fid)
    e0 = frame.get('e0_regenerated', {})
    if good_pose(e0):
        matches = [i for i, p in enumerate(poses) if equal_pose(p, e0, scale)]
        audit.check(len(matches) == 1, 'available_E0_once', fid)
        for index in matches: audit.check('E0' in action_rows[index].get('source_baseline_aliases', []), 'E0_alias_preserved', fid)
    for c in frame['candidates']:
        include = good_pose(c['actual_pose']) and (c['eligible'] or equal_pose(c['actual_pose'], fallback, scale))
        matches = [i for i, p in enumerate(poses) if equal_pose(p, c['actual_pose'], scale)]
        if include:
            audit.check(bool(matches), 'eligible_candidate_action_coverage', fid + '/' + c['candidate_id'])
            preserved = set().union(*(set(action_rows[i].get('source_baseline_aliases', [])) for i in matches))
            audit.check(aliases(c) <= preserved, 'source_alias_coverage', fid + '/' + c['candidate_id'])
    for r in action_rows:
        audit.check(r['candidate_id'] != 'H0_NONE', 'no_H1_virtual_none', fid)
        role = r.get('action_role'); aa = set(r.get('source_baseline_aliases', []))
        audit.check(role == ('N0' if 'N0' in aa else 'E0' if 'E0' in aa else 'OTHER'), 'action_role_matches_aliases', fid)
        if 'N0' in aa: audit.check(equal_pose(action_pose_map[r['candidate_id']], fallback, scale), 'N0_role_is_actual_fallback_pose', fid)
        if 'E0' in aa and good_pose(e0): audit.check(equal_pose(action_pose_map[r['candidate_id']], e0, scale), 'E0_role_is_actual_executed_pose', fid)
    pair = [p for r, p in zip(action_rows, poses) if set(r.get('source_baseline_aliases', [])) & {'N0', 'E0'}]
    audit.check(len(pair) <= 2, 'pair_at_most_two_actual_actions', fid)
    for i, p in enumerate(pair):
        for q in pair[:i]: audit.check(not equal_pose(p, q, scale), 'pair_actual_pose_unique', fid)


def inspect_domain(audit, domain, features_path, feature_seal, labels_dir, targets, split_lock, output):
    features = list(records(features_path)); labelpath = labels_dir / 'ACTION_TARGETS.jsonl.gz'; labs = {r['id']: r for r in records(labelpath)}
    seal = read(labels_dir / 'LABEL_SEAL.json'); featureseal = read(feature_seal); normalizer = read(labels_dir / 'TRAIN_ONLY_NORMALIZER.json')
    audit.check(seal['features_sha256'] == digest(features_path), 'domain_feature_hash', domain)
    audit.check(seal['features_seal_sha256'] == digest(feature_seal), 'feature_seal_hash', domain)
    audit.check(seal['labels_sha256'] == digest(labelpath), 'label_hash', domain)
    audit.check(seal['normalizer_sha256'] == digest(labels_dir / 'TRAIN_ONLY_NORMALIZER.json'), 'normalizer_hash', domain)
    audit.check(digest(features_path) in json.dumps(featureseal), 'producer_feature_hash', domain)
    locked = {r['id']: r for r in split_lock['all_rows']}; ids = [r['id'] for r in features]
    audit.check(len(ids) == 1003 and len(set(ids)) == 1003 and set(ids) == set(labs) == set(locked), 'exact_original1003_IDs', domain)
    summary = Counter(); ledger = []
    for frame in features:
        fid = frame['id']; inspect_feature_frame(audit, frame); lab = labs[fid]; reference = targets[fid]; sp = locked[fid]['split']
        audit.check(frame['split'] == lab['split'] == reference['split'] == sp, 'same_original_split', domain + '/' + fid)
        audit.check(reference['source_asset'] == locked[fid]['source_asset'], 'same_original_asset', fid)
        allowed = split_lock['train_assets'] if sp == 'train' else split_lock['validation_assets']
        audit.check(reference['source_asset'] in allowed, 'asset_holdout_membership', fid)
        audit.check(lab['reference_symmetry_order'] == reference['symmetry_order'], 'original_per_row_symmetry', fid)
        candidates = {c['candidate_id']: c for c in frame['candidates']}; expected = []; oldmetric = {r['candidate_id']: r for r in lab['candidate_metrics']}
        audit.check(set(oldmetric) == set(candidates), 'H0_all_candidate_metric_coverage', fid)
        for c in frame['candidates']:
            m = independently_measure(c['actual_pose'], reference); audit.metrics(oldmetric[c['candidate_id']], m, fid + '/' + c['candidate_id'])
            expected.append(bool(c['eligible'] and m['success_5cm5deg']))
        count = sum(expected); h0 = [float(g) / count if count else 0. for g in expected] + [float(count == 0)]
        audit.check(len(h0) == len(lab['h0_target_distribution']), 'H0_target_width', fid)
        if len(h0) == len(lab['h0_target_distribution']):
            for j, v in enumerate(h0): audit.close(lab['h0_target_distribution'][j], v, 2e-15, 'H0_uniform_positive_else_none', fid)
        actionmap = {k: c['actual_pose'] for k, c in candidates.items()}
        actionmap.setdefault('N0', frame.get('fallback', dict(available=False)))
        actionmap.setdefault('E0_REGENERATED', frame.get('e0_regenerated', dict(available=False)))
        act = lab['action_metrics']; expected_action_coverage(audit, frame, act, actionmap); good = 0
        for r in act:
            m = independently_measure(actionmap[r['candidate_id']], reference); audit.metrics(r, m, 'H1/' + fid + '/' + r['candidate_id']); good += int(m['success_5cm5deg'])
        audit.metrics(lab['n0_metric'], independently_measure(frame.get('fallback'), reference), 'N0/' + fid)
        audit.check(lab['all_zero_H1'] == (good == 0), 'all_zero_labels_preserved', fid)
        audit.check(lab['zero_actions_H1'] == (len(act) == 0), 'zero_action_separate_from_allzero', fid)
        summary[sp + '/frames'] += 1; summary[sp + '/with_good_actions'] += int(good > 0); summary[sp + '/all_zero'] += int(good == 0); summary[sp + '/zero_actions'] += int(len(act) == 0)
        ledger.append(dict(domain=domain, id=fid, split=sp, source_asset=reference['source_asset'], symmetry_order=reference['symmetry_order'],
            archived_candidates=len(candidates), H1_actual_actions=len(act), H1_good_actions=good, H1_all_zero=good == 0, H1_zero_actions=len(act) == 0))
    audit.check(summary['train/frames'] == 768 and summary['validation/frames'] == 235, 'exact768_235_split', domain)
    trainframes = [r for r in features if r['split'] == 'train']; audit.check(normalizer['training_frame_ids'] == [r['id'] for r in trainframes], 'normalizer_train_only_exact_ID_order', domain)
    values = [c['features'] for r in trainframes for c in r['candidates'] if c['eligible']]
    for j in range(31):
        x = [v[j] for v in values if v[j + 31] == 1.]; n = len(x)
        mu = math.fsum(x) / n if n else 0.; sigma = math.sqrt(math.fsum((v - mu) ** 2 for v in x) / n) if n else 1.
        sigma = sigma if sigma > 1e-12 else 1.
        audit.check(normalizer['valid_train_counts'][j] == n, 'normalizer_valid_train_counts', domain)
        audit.close(normalizer['mean'][j], mu, NORM_TOL, 'normalizer_train_mean', domain + '/' + str(j))
        audit.close(normalizer['scale'][j], sigma, NORM_TOL, 'normalizer_train_scale', domain + '/' + str(j))
    csvpath = output / f'{domain}_INDEPENDENT_ACTION_ROWS.csv'
    with open(csvpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
    return dict(rows=len(features), counts=dict(summary), feature_sha256=digest(features_path), label_sha256=digest(labelpath),
        label_seal_sha256=digest(labels_dir / 'LABEL_SEAL.json'), row_ledger_sha256=digest(csvpath),
        e0_independently_executed=featureseal.get('e0_independently_executed', False), normalizer=normalizer)


def inspect_training(audit, training_dir, domains):
    import torch
    torch.set_num_threads(1)
    manifest = read(training_dir / 'TRAINING_MANIFEST.json'); ledger_path = training_dir / 'TRAINING_ATTEMPTS.jsonl'
    attempts = list(records(ledger_path)) if ledger_path.exists() else []; runs = manifest['runs']; seen = Counter()
    audit.check(len(attempts) <= 18, 'actual_optimizer_attempt_cap18')
    audit.check(len(attempts) == manifest['actual_new_training_attempts'], 'manifest_attempt_count')
    summaries = []
    for r in runs:
        arm, seed = r['arm'], r['seed']; seen[arm] += 1
        audit.check(arm in GROUPS and seed in SEEDS, 'registered_group_seed', arm)
        audit.check(r['representative'] == (seed == SEEDS[0]), 'fixed_representative_seed', arm)
        cp = Path(r['checkpoint']); audit.check(digest(cp) == r['checkpoint_sha256'], 'checkpoint_exact_bytes', arm)
        checkpoint = torch.load(cp, map_location='cpu', weights_only=False); state = checkpoint['model_state_dict']
        if arm.endswith('ROLE_AFFINE'):
            shapes = {'n0.weight': [1, 187], 'n0.bias': [1], 'other.weight': [1, 187], 'other.bias': [1]}; nparam = 376
        else:
            shapes = {'0.weight': [64, 188], '0.bias': [64], '2.weight': [64, 64], '2.bias': [64], '4.weight': [1, 64], '4.bias': [1]}; nparam = 16321
        audit.check(set(state) == set(shapes) and all(list(state[k].shape) == v for k, v in shapes.items()), 'architecture_and_information_dimensions', arm)
        audit.check(sum(v.numel() for v in state.values()) == nparam == r['parameters'], 'exact_parameter_count', arm)
        audit.check(checkpoint['seed'] == seed, 'checkpoint_seed_matches_run', arm)
        domain = arm[:2]
        if domain in domains:
            norm = domains[domain]['normalizer']; ck_norm = checkpoint['normalizer']
            for k in ('mean', 'scale', 'valid_train_counts', 'training_frame_ids'): audit.check(ck_norm[k] == norm[k], 'sameD_train_only_normalizer_H0_H1_pair', arm + '/' + k)
        else: audit.pending.append(arm + ': independent domain audit not available')
        if r.get('checkpoint_reused'):
            audit.check(arm == 'D0_H0_MLP' and r.get('new_training_runs') == 0, 'only_D0_H0_reused_no_training', arm)
            audit.check(checkpoint.get('kind') == 'L_HEAD' and checkpoint.get('input_dim') == 188, 'legacy_H0_compatibility', arm)
        else:
            folder = cp.parent; audit.check(digest(folder / 'last.pt') == r['last_checkpoint_sha256'], 'last_checkpoint_preserved', arm)
            audit.check(digest(folder / 'HISTORY.json') == r['history_sha256'], 'all_history_preserved', arm)
            history = read(folder / 'HISTORY.json'); audit.check(1 <= len(history) <= 50 and len(history) == r['epochs'], 'epoch_cap_and_count', arm)
            independent_best = min(range(len(history)), key=lambda i: history[i]['validation']['loss']) + 1
            audit.check(independent_best == r['best_epoch'] == checkpoint['epoch'], 'own_validation_loss_checkpoint_selection', arm)
            cfg = checkpoint['config']
            for k, v in dict(learning_rate=.001, weight_decay=.0001, batch_frames=32, max_epochs=50, patience=5).items(): audit.check(cfg[k] == v, 'fixed_optimizer_protocol', arm + '/' + k)
            for step in history:
                for split in ('train', 'validation'):
                    audit.check(step[split]['frames'] == (768 if split == 'train' else 235), 'epoch_pose_metrics_full_split', arm)
                    audit.check(all(k in step[split] for k in ('success_5cm5deg', 'translation_cm', 'rotation_deg', 'tails', 'selected_action_source', 'frame_balanced_action_accuracy', 'frame_balanced_Brier', 'good_action_absent')), 'epoch_required_pose_action_diagnostics', arm)
            audit.check(any(x['arm'] == arm and x['seed'] == seed for x in attempts), 'optimizer_run_reserved_in_global_ledger', arm)
        summaries.append(dict(arm=arm, seed=seed, checkpoint_sha256=r['checkpoint_sha256'], parameters=nparam, checkpoint_reused=r.get('checkpoint_reused', False)))
    for arm in GROUPS:
        seeds = sorted(r['seed'] for r in runs if r['arm'] == arm)
        audit.check(len(seeds) == len(set(seeds)) and len(seeds) <= 3, 'no_duplicate_seed_or_extra_architecture', arm)
        if seeds != SEEDS: audit.pending.append(arm + ': expected3 checkpoints, available' + str(len(seeds)))
    audit.check(manifest['complete'] == all(seen[arm] == 3 for arm in GROUPS), 'manifest_completeness_truthful')
    return dict(available_checkpoints=len(runs), planned_checkpoints=21, actual_new_training_attempts=len(attempts), actual_new_successful_runs=sum(not r.get('checkpoint_reused') for r in runs),
        reused_checkpoints=sum(bool(r.get('checkpoint_reused')) for r in runs), manifest_sha256=digest(training_dir / 'TRAINING_MANIFEST.json'), runs=summaries)


def inspect_real(audit, features_path, feature_seal, predictions_dir, training, training_dir):
    """No real GT/score/metric access: verify source poses and common input hash."""
    fs = read(feature_seal); fdigest = digest(features_path); frames = {r['id']: r for r in records(features_path)}
    for frame in frames.values(): inspect_feature_frame(audit, frame)
    audit.check(len(frames) == 238, 'common_real_all238_feature_coverage')
    if not predictions_dir:
        audit.pending.append('common real predictions not yet supplied'); return dict(feature_rows=len(frames), features_sha256=fdigest, prediction_check='PENDING')
    seal = read(predictions_dir / 'PREDICTION_SEAL.json'); path = predictions_dir / 'PREDICTIONS.jsonl.gz'; predictions = list(records(path))
    audit.check(seal['features_sha256'] == fdigest and fdigest in json.dumps(fs), 'all2x2_exact_same_common_real_hash')
    audit.check(seal['predictions_sha256'] == digest(path), 'prediction_exact_bytes')
    audit.check(seal.get('reference_reads') == 0, 'declared_prediction_reference_reads_zero')
    audit.check({r['id'] for r in predictions} == set(frames) and len(predictions) == 238, 'real_prediction_all238_coverage')
    expected = {r['arm'] + '_s' + str(r['seed']) for r in training['runs']}
    import torch
    checkpoint_rows = read(training_dir / 'TRAINING_MANIFEST.json')['runs']; checkpoints = {}
    for r in checkpoint_rows:
        checkpoints[r['arm'] + '_s' + str(r['seed'])] = torch.load(r['checkpoint'], map_location='cpu', weights_only=False)
    forward_count = 0
    for row in predictions:
        f = frames[row['id']]; scale = f.get('object_scale_m', 1.); candidates = {c['candidate_id']: c for c in f['candidates']}
        audit.check(set(row['predictions']) == expected, 'all_available_checkpoints_same_frame', row['id'])
        for arm, p in row['predictions'].items():
            if p.get('prediction_status') == 'NO_POSE':
                audit.check(not good_pose(p['actual_pose']), 'NO_POSE_truthful', row['id'])
                usable = [c for c in f['candidates'] if c['eligible'] and good_pose(c['actual_pose'])]
                audit.check(not good_pose(f.get('fallback')) and not good_pose(f.get('e0_regenerated')) and (not usable or '_PAIR_' in arm), 'NO_POSE_only_when_no_available_actions', row['id'])
                continue
            catalog = p['actions']; scores = [c['score'] for c in catalog]; best = max(scores); maxima = [i for i, score in enumerate(scores) if score == best]
            own_actual = []
            for entry in catalog:
                cid = entry['candidate_id']
                source = candidates.get(cid)
                if cid == 'H0_NONE': own_actual.append(None); continue
                if source is None and cid in ('N0', 'E0_REGENERATED'):
                    source = dict(features=[0.] * 62, actual_pose=f.get('fallback') if cid == 'N0' else f.get('e0_regenerated'), candidate_id=cid)
                audit.check(source is not None, 'catalog_action_has_actual_input_source', row['id'] + '/' + arm + '/' + cid)
                own_actual.append(source)
            if '_H0_' in arm:
                audit.check([e['candidate_id'] for e in catalog] == [c['candidate_id'] for c in f['candidates'] if c['eligible']] + ['H0_NONE'], 'H0_exact_eligible_order_plus_none', row['id'])
            else:
                active = [source for source in own_actual if source is not None]
                physical_order = [(0 if 'N0' in entry['source_aliases'] else 1,
                    tuple(np.asarray(source['actual_pose']['R_physical'],float).ravel()) + tuple(source['actual_pose']['centroid']))
                    for entry,source in zip(catalog,own_actual) if source is not None and good_pose(source['actual_pose'])]
                audit.check(len(physical_order)==len(catalog) and physical_order==sorted(physical_order),
                    'real_H1_catalog_N0_first_then_physical_numeric_order', row['id']+'/'+arm)
                for j, source in enumerate(active):
                    audit.check(source.get('eligible', False) or equal_pose(source['actual_pose'], f.get('fallback'), scale) or equal_pose(source['actual_pose'], f.get('e0_regenerated'), scale), 'real_H1_no_ineligible_extra_actions', row['id'])
                    for previous in active[:j]: audit.check(not equal_pose(source['actual_pose'], previous['actual_pose'], scale), 'real_H1_physical_unique', row['id'])
                for entry, source in zip(catalog, own_actual):
                    if 'N0' in entry['source_aliases']: audit.check(equal_pose(source['actual_pose'], f.get('fallback'), scale), 'real_N0_bit_is_actual_runtime_N0', row['id'])
                    if 'E0' in entry['source_aliases']: audit.check(equal_pose(source['actual_pose'], f.get('e0_regenerated'), scale), 'real_E0_alias_is_actual_executed_E0', row['id'])
                if '_PAIR_' in arm:
                    baseline_poses = [f.get('fallback'), f.get('e0_regenerated')]
                    for baseline_pose in baseline_poses:
                        if good_pose(baseline_pose): audit.check(any(equal_pose(source['actual_pose'], baseline_pose, scale) for source in active), 'pair_available_baseline_action_coverage', row['id'])
                else:
                    for source in f['candidates']:
                        if good_pose(source['actual_pose']) and (source['eligible'] or equal_pose(source['actual_pose'], f.get('fallback'), scale)):
                            audit.check(any(equal_pose(x['actual_pose'], source['actual_pose'], scale) for x in active), 'real_H1_eligible_action_coverage', row['id'])
            # Independently assemble the188 whitelist inputs from the exact
            # source features. Reference values, IDs and sessions never enter.
            ck = checkpoints[arm]; normalizer = ck['normalizer']; m = sum(source is not None for source in own_actual)
            raw = np.asarray([source['features'] for source in own_actual if source is not None], float).reshape((m, 62))
            mu = np.asarray(normalizer['mean']); sigma = np.asarray(normalizer['scale']); observed = np.asarray(normalizer['valid_train_counts']) > 0
            numeric = np.where((raw[:, 31:] == 1) & observed[None], (raw[:, :31] - mu) / sigma, 0.)
            own62 = np.concatenate([numeric, raw[:, 31:]], axis=1).astype(np.float32)
            xx = np.zeros((len(catalog), 188), np.float32)
            if m:
                xx[:m, :62] = own62; xx[:, 62:124] = own62.mean(axis=0); xx[:, 124:186] = own62.max(axis=0)
            xx[:, 186] = np.log1p(m)
            if '_H0_' in arm: xx[-1, 187] = 1.
            else: xx[:, 187] = [float('N0' in entry['source_aliases']) for entry in catalog]
            tensor = torch.from_numpy(xx); state = ck['model_state_dict']
            with torch.inference_mode():
                if arm.endswith('ROLE_AFFINE_s20261018') or 'ROLE_AFFINE_s' in arm:
                    z0 = torch.nn.functional.linear(tensor[:, :187], state['n0.weight'], state['n0.bias'])
                    z1 = torch.nn.functional.linear(tensor[:, :187], state['other.weight'], state['other.bias'])
                    own_logits = torch.where(tensor[:, 187:] > .5, z0, z1).flatten()
                else:
                    z = torch.relu(torch.nn.functional.linear(tensor, state['0.weight'], state['0.bias']))
                    z = torch.relu(torch.nn.functional.linear(z, state['2.weight'], state['2.bias']))
                    own_logits = torch.nn.functional.linear(z, state['4.weight'], state['4.bias']).flatten()
                own_probabilities = own_logits.softmax(0) if '_H0_' in arm else own_logits.sigmoid()
            forward_count += 1
            for j, entry in enumerate(catalog):
                audit.close(entry['score'], float(own_logits[j]), 0., 'whitelist188_manual_network_logit_exact', row['id'] + '/' + arm)
                audit.close(entry['sigmoid_or_softmax'], float(own_probabilities[j]), 0., 'manual_network_score_transform_exact', row['id'] + '/' + arm)
            if '_H0_' in arm: chosen = maxima[0]
            else:
                n0_maxima = [i for i in maxima if 'N0' in catalog[i]['source_aliases']]; chosen = n0_maxima[0] if n0_maxima else maxima[0]
            selected = catalog[chosen]
            audit.close(p.get('selected_score'), scores[chosen], 0., 'selected_exact_argmax_score', row['id'] + '/' + arm)
            is_none = selected['candidate_id'] == 'H0_NONE'; cid = 'N0' if is_none else selected['candidate_id']
            audit.check(p['selected_candidate_id'] == cid and p['none_selected'] == is_none, 'selected_actual_argmax_identity', row['id'] + '/' + arm)
            source_pose = f.get('fallback') if is_none else candidates.get(cid, {}).get('actual_pose')
            if source_pose is None and cid == 'N0': source_pose = f.get('fallback')
            if source_pose is None and cid == 'E0_REGENERATED': source_pose = f.get('e0_regenerated')
            audit.check(equal_pose(source_pose, p['actual_pose'], scale) or (not good_pose(source_pose) and not good_pose(p['actual_pose'])), 'selected_pose_is_one_actual_source', row['id'] + '/' + arm)
            if '_H1_' in arm:
                audit.check(not any(c['candidate_id'] == 'H0_NONE' for c in catalog), 'real_H1_no_virtual_none', row['id'])
                if '_PAIR_' in arm:
                    audit.check(fs.get('e0_independently_executed') is True and len(catalog) <= 2 and all(set(c['source_aliases']) & {'N0', 'E0'} for c in catalog), 'actual_independent_E0_pair_contract', row['id'])
    return dict(feature_rows=238, prediction_rows=238, checkpoint_count=len(expected), features_sha256=fdigest, predictions_sha256=digest(path),
        independent_manual_tiny_head_frame_forwards=forward_count, reference_reads=0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--targets', type=Path, required=True); p.add_argument('--target-seal', type=Path, required=True); p.add_argument('--split-lock', type=Path, required=True)
    for domain in ('d0', 'd1'):
        for suffix in ('features', 'feature-seal', 'labels-dir'): p.add_argument('--' + domain + '-' + suffix, type=Path, required=domain == 'd0')
    p.add_argument('--training-dir', type=Path, required=True); p.add_argument('--common-real-features', type=Path); p.add_argument('--common-real-seal', type=Path); p.add_argument('--predictions-dir', type=Path)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args(); start = time.monotonic(); args.output.mkdir(parents=True, exist_ok=False)
    audit = Audit(); target_seal = read(args.target_seal); audit.check(digest(args.targets) in json.dumps(target_seal), 'original_synthetic_reference_hash')
    audit.check(digest(args.split_lock) in json.dumps(target_seal), 'original_split_lock_hash')
    targets = {r['id']: r for r in records(args.targets)}; split_lock = read(args.split_lock); domains = {}
    for domain in ('d0', 'd1'):
        features, seal, labels = [getattr(args, domain + '_' + k) for k in ('features', 'feature_seal', 'labels_dir')]
        if features and seal and labels: domains[domain.upper()] = inspect_domain(audit, domain.upper(), features, seal, labels, targets, split_lock, args.output)
        else: audit.pending.append(domain.upper() + ': domain inputs not yet supplied')
    training = inspect_training(audit, args.training_dir, domains); real = None
    if args.common_real_features and args.common_real_seal: real = inspect_real(audit, args.common_real_features, args.common_real_seal, args.predictions_dir, training, args.training_dir)
    else: audit.pending.append('common real features and predictions not yet supplied')
    pending = sorted(set(audit.pending)); result = dict(schema='INDEPENDENT_ACTION_SPLIT_AUDIT_V2',
        status='FAIL' if audit.failures else ('PASS_COMPLETED_SCOPE_WITH_PENDING_CHECKS' if pending else 'PASS_WITH_DOCUMENTED_REPRESENTATION_EXCEPTIONS' if audit.representation_exceptions else 'PASS'),
        complete_scope=not bool(pending), checks=dict(audit.checks), failure_count=len(audit.failures), failures=audit.failures,
        maximum_absolute_differences=dict(audit.maxima), pending=pending,
        tolerances_preregistered=dict(translation_cm=T_TOL_CM, rotation_deg=R_TOL_DEG, normalizer=NORM_TOL, success='strict unchanged5cm5deg; no tolerance'),
        representation_exceptions=audit.representation_exceptions,
        representation_policy='v1 FAIL preserved. Fixed absolute tolerance not widened. Stable-fsum exceptions enumerated; original sqrt(dot(delta,delta))*100 expression must reproduce all saved T values exactly. Labels and training unchanged.',
        domains={k: {kk: vv for kk, vv in v.items() if kk != 'normalizer'} for k, v in domains.items()}, training=training, real=real,
        references_scope='original1003 renderer synthetic only; NO real/CAL/PRIMARY references or scored rows read',
        production_action_input_loss_label_model_function_imports=0, source_sha256=digest(__file__), wall_seconds=time.monotonic() - start)
    write(args.output / 'INDEPENDENT_ACTION_AUDIT.json', result); print(json.dumps(dict(status=result['status'], checks=sum(audit.checks.values()), failures=len(audit.failures), pending=len(pending), wall_seconds=result['wall_seconds'])))
    if audit.failures: raise SystemExit(1)


if __name__ == '__main__': main()
