"""Independent direct arithmetic for the fixed visible-boundary pilot.

Imports only the Python standard library and NumPy. In particular this checker
never imports the pilot kernel/replay, the old selector, a solver or an NN
runtime. A separate monotone-chain hull, per-segment finite distances and
length/midpoint quadrature verify every retained candidate's fixed score.
Semantic annotation quality remains model-assisted judgement, not ground truth.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
ATOL = 1e-9
STAGES = ('G10', 'G13')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
VARIANTS = ('BASELINE_E', 'HULL_ONEWAY', 'HULL_ONEWAY_MASS', 'VISIBLE_SUPPORT')
IDS = ('plastic_night_01:038630', 'wood_day_01:005286',
       'wood_night_01:030698', 'plastic_day_01:020955',
       'plastic_day_01:014704', 'wood_night_01:030898')
DENIED = {'reference','reference_success','references','oracle','oracles',
          'metric','metrics','group','translation_cm','rotation_deg',
          'success_le5','final_reference_success','SUPPORT_DECISIONS',
          'representative_success_candidate_id',
          'reference_success_candidates_g13_all',
          'reference_success_candidates_g13_selected_family',
          'family_reference_diagnosis','g10_reference_success_candidate_ids'}

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]

def equal(a, b, label, maximum):
    """Nested exact metadata; arrays/floats use the declared absolute tolerance."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), label+' keys')
        for key in a:
            equal(a[key], b[key], label+'/'+key, maximum)
    elif isinstance(a, (list, tuple)):
        require(isinstance(b, (list, tuple)) and len(a) == len(b), label+' cardinality')
        for index, (left, right) in enumerate(zip(a, b)):
            equal(left, right, label+'/'+str(index), maximum)
    elif isinstance(a, float) or isinstance(b, float):
        require(isinstance(a, (float, int)) and isinstance(b, (float, int)), label+' numeric type')
        error = abs(float(a)-float(b))
        maximum[0] = max(maximum[0], error)
        require(math.isfinite(error) and error <= ATOL, label+' absolute difference '+str(error))
    else:
        require(type(a) is type(b) and a == b, label+' exact metadata')

def non_neutral(value):
    if isinstance(value, dict):
        return [key for key in value if key in DENIED] + [k for child in value.values() for k in non_neutral(child)]
    if isinstance(value, list):
        return [key for child in value for key in non_neutral(child)]
    return []

def bound(package_path, entry):
    path = Path(entry['path'])
    if not path.is_absolute():
        path = Path(package_path).parent/path
    require(digest(path) == entry['sha256'], 'Neutral input byte binding')
    return path

def select(frame, candidates, values, variant):
    pools = defaultdict(list)
    for c in candidates:
        pools[c['stage'], c['family']].append(c)
    winners, ranking, family_scores = {}, {}, []
    for stage in STAGES:
        winners[stage], ranking[stage] = {}, {}
        require([m['family'] for m in frame['stages'][stage]['families']] == list(FAMILIES), 'Original family order')
        for family_index, family in enumerate(FAMILIES):
            # Original insertion order is restored first, then Python stable
            # sorting preserves it when the schema comparator has an exact tie.
            ordered = sorted(pools[stage, family], key=lambda c: c['original_outcome_order'])
            def key(c):
                score = values[stage, family, c['candidate_id']]['scores'][variant]
                require(math.isfinite(score), 'Finite original/new scalar')
                return ((score,) if c['solver_schema'] == 'statewise_u_region_solver_v4_5'
                        else (score, c['original_record']['initialization']))
            ordered = sorted(ordered, key=key)
            metadata = frame['stages'][stage]['families'][family_index]
            require(bool(ordered) == metadata['accepted'], 'Unchanged original accepted pool')
            require(all(c['retained'] for c in ordered), 'Original retained only')
            require(len({c['solver_schema'] for c in ordered}) <= 1, 'One solver schema per pool')
            ranking[stage][family] = [c['candidate_id'] for c in ordered]
            winners[stage][family] = ordered[0]['candidate_id'] if ordered else None
            if stage == 'G10' and ordered:
                family_scores.append((key(ordered[0])[0], family_index, family))
    family = min(family_scores)[2] if family_scores else FAMILIES[0]
    cid = winners['G13'][family]
    metadata = frame['stages']['G13']['families'][FAMILIES.index(family)]
    pose = deepcopy(metadata['published_actual_pose'])
    selected = next((c for c in pools['G13', family] if c['candidate_id'] == cid), None)
    if selected is not None:
        pose.update(R_cf=selected['pose']['R_cf'], centroid=selected['pose']['centroid'],
            R_physical=selected['pose']['R_physical'], physical_registration_Q=selected['physical_registration_Q'],
            cf_extents=selected['cf_extents'])
        if 'native_t' in pose:
            pose['native_t'] = selected['pose']['native_t']
        require(np.allclose(np.asarray(pose['R_cf'])@pose['physical_registration_Q'], pose['R_physical'],
                            atol=ATOL, rtol=0), 'Physical same-family Q')
    return dict(id=frame['id'], variant=variant, selected_family=family, candidate_id=cid,
        accepted=metadata['accepted'], fallback=not metadata['accepted'], pose=pose,
        original_constraint_pass=None if selected is None else selected['dense']['original_joint_feasible'],
        recomputed_constraint_pass=None if selected is None else selected['dense']['recomputed_joint_feasible'],
        minimum_constraint_slack=None if selected is None else selected['dense']['minimum_constraint_slack'],
        original_fallback_status=metadata['status'], stage_winner_ids=winners, ranking=ranking,
        same_family_registration_preserved=True, compares_G10_with_G13_objectives=False,
        original_fallback_preserved=True, new_pose_fits=0)

def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))

def csv_value(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    return value

def planar_hull(points):
    """Andrew's monotone chain; no SciPy/OpenCV hull dependency."""
    values = sorted(set(tuple(map(float, p)) for p in points))
    require(len(values) >= 3, 'At least three projected model vertices')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1]+upper[:-1], dtype=float)
    require(len(hull) >= 3, 'Nondegenerate original model projection')
    return hull


def finite_distances(points, polygon):
    """Independent loop over finite model segments; clamp to either endpoint."""
    p = np.asarray(points, dtype=float)
    require(p.ndim == 2 and p.shape[1] == 2, 'Point matrix Nx2')
    best = np.full(len(p), np.inf)
    for j in range(len(polygon)):
        first, last = polygon[j], polygon[(j+1) % len(polygon)]
        vector = last-first
        denominator = math.fsum(float(x)*float(x) for x in vector)
        require(denominator > 0, 'Nonzero finite model segment')
        alpha = np.einsum('ij,j->i', p-first, vector)/denominator
        foot = first+np.minimum(1., np.maximum(0., alpha))[:, None]*vector
        residual = p-foot
        best = np.minimum(best, np.einsum('ij,ij->i', residual, residual))
    return np.sqrt(best)


def pseudo_huber(distances, sigma, delta):
    require(sigma > 0 and delta > 0, 'Original positive robust scales')
    radii = np.asarray(distances, dtype=float)/sigma
    return delta*delta*(np.sqrt(1.+(radii/delta)**2)-1.)


def geometry(candidate, frame):
    xyz, R, t = (np.asarray(candidate[k], dtype=float) for k in ('native_xyz', 'R', 't'))
    require(xyz.shape == (8, 3) and R.shape == (3, 3) and t.shape == (3,), 'Unchanged native cuboid pose')
    require(np.array_equal(candidate['R'], candidate['pose']['R_cf']), 'Same original native R')
    require(np.array_equal(candidate['t'], candidate['pose']['native_t']), 'Same original native t')
    require(np.allclose(R@candidate['physical_registration_Q'], candidate['pose']['R_physical'], atol=ATOL, rtol=0),
            'Own native-to-physical Q product')
    require(np.allclose(R@xyz.mean(axis=0)+t, candidate['pose']['centroid'], atol=ATOL, rtol=0),
            'Own registered physical centroid')
    camera = xyz@R.T+t
    require(np.min(camera[:, 2]) > 1e-8, 'Original candidate positive projected depth')
    homogeneous = camera@np.asarray(frame['K'], dtype=float).T
    projected = homogeneous[:, :2]/homogeneous[:, 2, None]
    require(np.isfinite(projected).all(), 'Finite original model projection')
    return projected, planar_hull(projected)


def polylines(entry):
    result = entry.get('polylines', entry.get('polylines_xy', []))
    return [np.asarray(line.get('points_xy', line.get('xy', [])) if isinstance(line, dict) else line, dtype=float)
            for line in result]


def check_input(entry, frame):
    require(entry['id'] == frame['id'], 'Input exact frame ID')
    require(entry['image_sha256'] == frame['image_sha256'], 'Input original RGB byte binding')
    require(entry['image_hw'] == frame['image_hw'], 'Input original canvas shape')
    h, w = frame['image_hw']
    seen, pairs = set(), []
    segment_count, length = 0, 0.
    for line in polylines(entry):
        require(line.ndim == 2 and line.shape[1] == 2 and len(line) >= 2, 'Polyline Nx2 at least two vertices')
        require(np.isfinite(line).all(), 'Finite original pixel coordinates')
        require((line[:, 0] >= 0).all() and (line[:, 0] <= w-1).all()
                and (line[:, 1] >= 0).all() and (line[:, 1] <= h-1).all(), 'Raw full RGB coordinate bounds')
        require(not np.array_equal(line[0], line[-1]), 'Open polyline without closing endpoint')
        for first, last in zip(line, line[1:]):
            l = math.hypot(*(last-first))
            require(l > 0, 'No zero-length visible segment')
            key = tuple(sorted((tuple(first), tuple(last))))
            require(key not in seen, 'No exact duplicate raw segment length')
            seen.add(key)
            # Positive collinear overlap is also duplicate raw boundary support.
            # Endpoint adjacency has zero overlap and is allowed.
            for a, b in pairs:
                direction = last-first
                cross_a = direction[0]*(a[1]-first[1])-direction[1]*(a[0]-first[0])
                cross_b = direction[0]*(b[1]-first[1])-direction[1]*(b[0]-first[0])
                if abs(cross_a) <= 1e-9 and abs(cross_b) <= 1e-9:
                    axis = int(abs(direction[1]) > abs(direction[0]))
                    low = max(min(first[axis], last[axis]), min(a[axis], b[axis]))
                    high = min(max(first[axis], last[axis]), max(a[axis], b[axis]))
                    require(high-low <= 1e-9, 'No collinear duplicate raw support length')
            pairs.append((first, last))
            segment_count += 1
            length += l
    return dict(polylines=len(polylines(entry)), segments=segment_count, total_length_px=length,
                raw_full_canvas=True, open_polylines=True, duplicate_segments=0, zero_length_segments=0)


def sample_input(entry, spacing):
    samples, masses, segment_index = [], [], []
    segment_id = 0
    for line in polylines(entry):
        for first, last in zip(line, line[1:]):
            delta = last-first
            length = math.hypot(*delta)
            count = max(1, math.ceil(length/spacing))
            mass = (length/spacing)/count
            for index in range(count):
                samples.append(first+((index+.5)/count)*delta)
                masses.append(mass)
                segment_index.append(segment_id)
            segment_id += 1
    return np.asarray(samples, dtype=float).reshape(-1, 2), np.asarray(masses), segment_index


def cost(candidate, frame, entry):
    outline = frame['outlines'][candidate['stage']][candidate['family']]
    dense = candidate['dense']
    objective = float(candidate['original_record']['objective'])
    require(abs(objective-float(dense['objective'])) <= ATOL, 'Original total objective binding')
    components = dense['weighted_components']
    C = math.fsum(float(components[k]) for k in ('full_BI', 'partial_normal', 'prior'))
    old_outline = float(components['outline'])
    require(abs(C+old_outline-objective) <= ATOL, 'C preserves full/partial/prior and exactly replaces outline')
    source = [d for d in dense['per_observation'] if d['loss_family'] == 'outline']
    active = bool(dense['outline_active'])
    require(active == bool(source), 'Original active outline directional rows')
    if not active:
        return dict(scores={v: objective for v in VARIANTS}, identity_shortcut=True,
                    original_objective=objective, C=C, original_outline_cost=old_outline,
                    treatment_active=False, Z_E=0., Z_H=0., r=0., visible_sample_count=0)
    require(outline['supported'], 'Original outline package supported')
    require(sorted(d['direction'] for d in source) == ['model_to_observed', 'observed_to_model'],
            'Both original directions exactly once')
    cfg = outline['configuration']
    sigma, delta, factor, spacing = (float(cfg[k]) for k in
        ('normal_sigma_px', 'huber_delta', 'outline_factor_weight', 'observed_sample_spacing_px'))
    require(factor > 0, 'Original active outline positive factor')
    observed = next(d for d in source if d['direction'] == 'observed_to_model')
    model = next(d for d in source if d['direction'] == 'model_to_observed')
    for row in source:
        require(abs(math.fsum(row['arc_or_sample_weights'])-float(row['normalizer'])) <= ATOL,
                'Original positive directional mass normalizer')
        require(abs(math.fsum(row['weighted_sample_contributions'])-float(row['weighted_contribution'])) <= ATOL,
                'Original weighted directional contribution sum')
        require(row['factor_weight'] == factor and row['direction_weight'] == .5,
                'Original lambda and half included exactly once')
    A_H_weighted, B_H_weighted = float(observed['weighted_contribution']), float(model['weighted_contribution'])
    require(abs(A_H_weighted+B_H_weighted-old_outline) <= ATOL, 'Original outline directional sum')
    _, hull = geometry(candidate, frame)
    original_distances = finite_distances(observed['sample_xy'], hull)
    saved_distances = np.linalg.norm(observed['residual_xy_px'], axis=1)
    require(np.allclose(original_distances, saved_distances, atol=ATOL, rtol=0), 'Own model hull finite distance vs old observed residual')
    old_losses = pseudo_huber(original_distances, sigma, delta)
    require(np.allclose(old_losses, observed['robust_losses'], atol=ATOL, rtol=0), 'Own pseudo-Huber against original direction')
    require(abs(.5*factor*math.fsum(float(a)*float(v) for a,v in zip(observed['arc_or_sample_weights'],old_losses))/float(observed['normalizer'])-A_H_weighted) <= ATOL,
            'No original lambda/half double weighting')
    if entry is None:
        samples, masses, sample_segment_ids = np.empty((0, 2)), np.empty(0), []
    else:
        samples, masses, sample_segment_ids = sample_input(entry, spacing)
    Z_E, Z_H = math.fsum(masses), float(observed['normalizer'])
    r = min(1., Z_E/Z_H)
    available = len(samples) > 0
    result = dict(original_objective=objective, C=C, original_outline_cost=old_outline,
        original_observed_weighted_cost=A_H_weighted, original_model_weighted_cost=B_H_weighted,
        Z_E=Z_E, Z_H=Z_H, r=r, visible_sample_count=len(samples),
        visible_sample_xy=samples.tolist(), visible_sample_masses=masses.tolist(),
        visible_segment_ids=sample_segment_ids, treatment_active=available, identity_shortcut=not available)
    if not available:
        result['scores'] = {v: objective for v in VARIANTS}
        return result
    distances = finite_distances(samples, hull)
    losses = pseudo_huber(distances, sigma, delta)
    A_E = math.fsum(float(a)*float(v) for a, v in zip(masses, losses))/Z_E
    visible_weighted = .5*factor*r*A_E
    result.update(A_E=A_E, visible_weighted_cost=visible_weighted,
        visible_distance_px=distances.tolist(), visible_robust_losses=losses.tolist(),
        visible_distance_sum_px=math.fsum(distances), visible_distance_mean_px=float(np.mean(distances)),
        visible_distance_max_px=float(np.max(distances)),
        visible_mass_weighted_distance_mean_px=math.fsum(float(a)*float(v) for a,v in zip(masses,distances))/Z_E,
        scores=dict(BASELINE_E=objective, HULL_ONEWAY=math.fsum((C, A_H_weighted)),
            HULL_ONEWAY_MASS=math.fsum((C, r*A_H_weighted)), VISIBLE_SUPPORT=math.fsum((C, visible_weighted))))
    return result
def verify_candidate(value, actual, candidate, maximum):
    """Compare meanings, not unrelated old/new score column names."""
    for own_name, actual_name in (
        ('original_objective', 'old_objective'), ('C', 'non_outline_cost'),
        ('original_outline_cost', 'original_outline_cost'),
        ('original_observed_weighted_cost', 'original_observed_weighted_cost'),
        ('original_model_weighted_cost', 'original_model_weighted_cost'),
        ('treatment_active', 'treatment_active'), ('identity_shortcut', 'identity_shortcut'),
        ('scores', 'scores')):
        if own_name in value:
            equal(value[own_name], actual[actual_name], 'candidate/'+actual_name, maximum)
    equal(candidate['dense']['weighted_components'], actual['original_weighted_components'],
          'Original full/partial/prior/outline components unchanged', maximum)
    require(actual['original_hard_constraints_preserved'] is True, 'Original hard constraints remain hard')
    require(actual['original_full_partial_prior_preserved'] is True, 'Original full/partial/prior preserved')
    require(actual['observed_hull_created'] is False and actual['model_to_visible_reverse_term_created'] is False,
            'Open observations unclosed; no new reverse model-to-visible penalty')
    require(all(actual[k] == 0 for k in ('new_pose_fits', 'new_candidate_calls', 'new_model_forward_calls')),
            'No new pose fit/candidate/forward')
    if value['identity_shortcut']:
        require(actual['visible_evidence'] is None, 'No fabricated missing evidence')
        require(all(x == value['original_objective'] for x in actual['scores'].values()), 'No-input scalar exact identity')
        return
    for own_name, actual_name in (('Z_E','visible_mass_Z_E'), ('Z_H','old_observed_mass_Z_H'), ('r','r')):
        equal(value[own_name], actual[actual_name], 'input mass/'+actual_name, maximum)
    evidence = actual['visible_evidence']
    for own_name, actual_name in (
        ('visible_sample_count','sample_count'), ('visible_sample_xy','sample_xy'),
        ('visible_sample_masses','sample_masses'), ('visible_distance_px','raw_distance_px'),
        ('visible_robust_losses','robust_losses'), ('A_E','robust_mean_A_E'),
        ('visible_distance_sum_px','raw_distance_sum_px'),
        ('visible_distance_mean_px','raw_distance_mean_px'),
        ('visible_distance_max_px','raw_distance_max_px'),
        ('visible_mass_weighted_distance_mean_px','mass_weighted_distance_mean_px')):
        equal(value[own_name], evidence[actual_name], 'Visible support/'+actual_name, maximum)
    require(evidence['endpoints_used_as_keypoints'] is False and evidence['gaps_joined'] is False,
            'Polyline endpoints not corners; gaps not interpolated')
    require(evidence['samples_are_independent_measurements'] is False, 'Quadrature not independent evidence count')
    require(evidence['reverse_direction_present'] is False and evidence['direction_weight'] == .5,
            'Original half and one-way visible term')
    contributions = evidence['weighted_sample_contributions']
    equal(value['visible_weighted_cost'], math.fsum(contributions), 'Independent visible contribution total', maximum)
    equal(value['visible_weighted_cost'], actual['replacement_outline_costs']['VISIBLE_SUPPORT'],
          'Independent replacement term', maximum)


def verify_run(package_path, input_manifest, run_path):
    package, run_path = read(package_path), Path(run_path)
    frames = rows(bound(package_path, package['frames']))
    candidates = rows(bound(package_path, package['candidates']))
    require([f['id'] for f in frames] == list(IDS), 'Exact six ID order')
    require(len(candidates) == 351, 'All original351 retained candidates')
    counts = Counter(c['id'] for c in candidates)
    require([counts[fid] for fid in IDS] == [22,94,42,130,41,22], 'Original per-image retained denominator')
    require(not non_neutral(frames) and not non_neutral(candidates), 'Neutral bank contains no evaluation or old audit labels')
    entries = {fid: None for fid in IDS} if input_manifest is None else {e['id']: e for e in rows(input_manifest)}
    if input_manifest is not None:
        require([e['id'] for e in rows(input_manifest)] == list(IDS), 'Exactly six ordered model input sheets')
    require(set(entries) == set(IDS), 'No missing or invented frame input')
    inputs = {}
    for frame in frames:
        inputs[frame['id']] = (None if entries[frame['id']] is None
                                else check_input(entries[frame['id']], frame))
    seal = read(run_path/'PREDICTION_SEAL.json')
    require(seal['neutral_package_sha256'] == digest(package_path), 'Fixed input package bound by actual prediction seal')
    require(seal['input_manifest_sha256'] == (None if input_manifest is None else digest(input_manifest)),
            'Frozen visible-input byte binding')
    for name, expected in seal['artifacts'].items():
        require(digest(run_path/name) == expected, 'Prediction output seal '+name)
    actual_scores = rows(run_path/'CANDIDATE_SCORES.jsonl.gz')
    key = lambda r: tuple(r[k] for k in ('id','stage','family','candidate_id'))
    actual_by_key = {key(c): c for c in actual_scores}
    require(len(actual_scores) == len(actual_by_key) == 351, 'Exactly351 unique candidate score rows')
    require(set(actual_by_key) == {key(c) for c in candidates}, 'Every original retained candidate scored')
    predictions = rows(run_path/'PREDICTIONS.jsonl.gz')
    require([(p['id'],p['variant']) for p in predictions] == [(fid,v) for fid in IDS for v in VARIANTS],
            'All24 final outputs in fixed order')
    actual_predictions = {(p['id'],p['variant']):p for p in predictions}
    maximum, checks = [0.], []
    for frame in frames:
        own = [c for c in candidates if c['id'] == frame['id']]
        values, identities = {}, {}
        treatment_count = 0
        for candidate in own:
            local_key = tuple(candidate[k] for k in ('stage','family','candidate_id'))
            value = cost(candidate, frame, entries[frame['id']])
            identity = cost(candidate, frame, None)
            verify_candidate(value, actual_by_key[key(candidate)], candidate, maximum)
            values[local_key], identities[local_key] = value, identity
            require(all(identity['scores'][v] == candidate['original_record']['objective'] for v in VARIANTS),
                    'No-input allfour original scalar exact')
            treatment_count += value['treatment_active']
            if value['treatment_active']:
                require(actual_by_key[key(candidate)]['r'] == min(1., value['Z_E']/value['Z_H']), 'Candidate-independent capped length ratio')
                projected, hull = geometry(candidate, frame)
                equal(projected.tolist(), actual_by_key[key(candidate)]['visible_evidence']['model_projected_vertices_xy'],
                      'Own projected native XYZ/R/t/K', maximum)
                own_distances = finite_distances(value['visible_sample_xy'], hull)
                actual_segments = actual_by_key[key(candidate)]['visible_evidence']['model_finite_segments_xy']
                # Hull ordering may differ by cyclic shift. Set-valued finite
                # distance is invariant; no arc indices are paired across poses.
                actual_polygon = np.asarray([segment[0] for segment in actual_segments])
                equal(own_distances.tolist(), finite_distances(value['visible_sample_xy'], actual_polygon).tolist(),
                      'Own monotone-chain vs original finite hull distance', maximum)
        baseline = select(frame, own, values, 'BASELINE_E')
        require(baseline['candidate_id'] == frame['selected_candidate_id'] and baseline['selected_family'] == frame['selected_family'],
                'Original e candidate/family exact')
        require(baseline['accepted'] == frame['original_e_accepted'], 'Original acceptance unchanged')
        equal(frame['original_e_actual_pose'], baseline['pose'], 'Original e full pose contract', maximum)
        for variant in VARIANTS:
            prediction = select(frame, own, values, variant)
            actual = actual_predictions[frame['id'], variant]
            for field, expected in prediction.items():
                equal(expected, actual[field], 'Independent hierarchy/'+field, maximum)
            identity = select(frame, own, identities, variant)
            equal({k:v for k,v in baseline.items() if k != 'variant'},
                  {k:v for k,v in identity.items() if k != 'variant'}, 'No-input entire ranking/pose exact identity', maximum)
        checks.append(dict(id=frame['id'], all_retained_candidates=len(own), treated_candidate_count=treatment_count,
            new_input=inputs[frame['id']], baseline_reproduced=True, empty_allthree_newarms_identity=True,
            full_G10_G13_rankings_and24_final_predictions_checked=True))
    grouped_r = defaultdict(set)
    for row in actual_scores:
        if row['treatment_active']:
            grouped_r[(row['id'],row['stage'],row['family'])].add(row['r'])
    require(all(len(value) == 1 for value in grouped_r.values()), 'Length ratio fixed within each stage/family, independent of candidate pose')
    for control_id in IDS[4:]:
        subset = [c for c in actual_scores if c['id'] == control_id]
        if input_manifest is not None:
            require(any(c['treatment_active'] for c in subset), 'Success control actually processed with new visible input')
            require(any(c['scores']['VISIBLE_SUPPORT'] != c['scores']['BASELINE_E'] for c in subset),
                    'Success control new evidence changes actual candidate costs')
    return dict(passed=True, frames=6, original_candidates=351, evaluated_scores=351*4,
        independently_selected_predictions=24, maximum_absolute_arithmetic_difference=maximum[0],
        no_new_reference_reads_in_arithmetic=True, original_constraints_Q_fallback_preserved=True,
        own_numpy_model_projection=True, own_monotone_chain_model_hull=True,
        own_loop_over_finite_segments=True, own_midpoint_quadrature=True,
        source_imports_only_stdlib_and_numpy=True, semantic_quality_is_not_GT_certified=True,
        frame_checks=checks, neutral_package_sha256=digest(package_path),
        input_manifest_sha256=None if input_manifest is None else digest(input_manifest),
        prediction_seal_sha256=digest(run_path/'PREDICTION_SEAL.json'))
def own_metric(pose, reference):
    if not pose['available']:
        return dict(translation_cm=None, rotation_deg=None, success_le5=False,
                    pose_available=False, reference_rotation_available=reference['R'] is not None)
    offset = np.asarray(pose['centroid'],float)-np.asarray(reference['t'],float)
    translation = 100.*math.sqrt(math.fsum(float(x)*float(x) for x in offset))
    require(reference['order'] == 2, 'Original C2 symmetry only')
    angle = None
    if reference['R'] is not None:
        R, G = np.asarray(pose['R_physical'],float), np.asarray(reference['R'],float)
        half_turn = np.diag([-1.,1.,-1.])
        cosines = [(math.fsum(float(a)*float(b) for a,b in zip((G@S).ravel(),R.ravel()))-1.)/2.
                   for S in (np.eye(3),half_turn)]
        angle = math.degrees(math.acos(max(-1.,min(1.,max(cosines)))))
    return dict(translation_cm=translation, rotation_deg=angle,
        success_le5=None if angle is None else translation <= 5. and angle <= 5.,
        pose_available=True, reference_rotation_available=reference['R'] is not None)


def transition(before, after):
    b,s = before['success_le5'], after['success_le5']
    return 'RECOVERY' if s and not b else 'DAMAGE' if b and not s else 'PRESERVED_SUCCESS' if s else 'PERSISTENT_FAILURE'


def metric_stats(values):
    mean = math.fsum(values)/len(values)
    sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1))
    return mean,sd,max(values)


def csv_cast(value, expected):
    if value == '': return None
    if type(expected) is bool: return value == 'True'
    if type(expected) is int: return int(value)
    if isinstance(expected,float): return float(value)
    return value


def evaluation_check(run_path, package_path, evaluation_path, reference_path, reference_seal):
    """This function reads reference values only after verifying output seals."""
    run_path,evaluation_path = Path(run_path),Path(evaluation_path)
    seal = read(run_path/'PREDICTION_SEAL.json')
    for name,expected in seal['artifacts'].items():
        require(digest(run_path/name) == expected, 'Prediction seal before reference opening')
    reference_binding = read(reference_seal)
    require(digest(reference_path) == reference_binding['artifacts']['FIXED_REFERENCES.json'], 'Original fixed references unchanged')
    refs = read(reference_path)['references']
    package = read(package_path)
    frames = {f['id']:f for f in rows(bound(package_path,package['frames']))}
    candidates = rows(bound(package_path,package['candidates']))
    predictions = rows(run_path/'PREDICTIONS.jsonl.gz')
    actual = csv_rows(evaluation_path/'METRICS.csv')
    require(len(actual) == len(predictions) == 24, 'All24 actual same-candidate errors')
    maximum,own = [0.],[]
    for prediction,stored in zip(predictions,actual):
        fid,variant = prediction['id'],prediction['variant']
        frame = frames[fid]
        value = own_metric(prediction['pose'],refs[frame['partition']][fid])
        row = dict(id=fid,ordinal=frame['ordinal'],partition=frame['partition'],variant=variant,
            candidate_id=prediction['candidate_id'],family=prediction['selected_family'],
            accepted=prediction['accepted'],fallback=prediction['fallback'],**value)
        for field,expected in row.items():
            equal(expected,csv_cast(stored[field],expected),'Independent24metrics/'+field,maximum)
        own.append(row)
    by_key = {(r['id'],r['variant']):r for r in own}
    summary={}
    all_transitions=[]
    for variant in VARIANTS:
        current = [r for r in own if r['variant'] == variant]
        counts=Counter()
        changed_candidates=changed_families=0
        if variant != 'BASELINE_E':
            for fid in IDS:
                before,after = by_key[fid,'BASELINE_E'],by_key[fid,variant]
                kind=transition(before,after)
                counts[kind]+=1
                changed_candidates += before['candidate_id'] != after['candidate_id']
                changed_families += before['family'] != after['family']
                all_transitions.append(dict(id=fid,ordinal=frames[fid]['ordinal'],variant=variant,
                    baseline_candidate_id=before['candidate_id'],new_candidate_id=after['candidate_id'],
                    candidate_changed=before['candidate_id'] != after['candidate_id'],
                    family_changed=before['family'] != after['family'],baseline_success=before['success_le5'],
                    new_success=after['success_le5'],transition=kind,
                    translation_delta_cm=after['translation_cm']-before['translation_cm'],
                    rotation_delta_deg=after['rotation_deg']-before['rotation_deg']))
            require(sum(counts.values()) == 6,'Complete6 paired transition denominator')
        tm,ts,tx=metric_stats([r['translation_cm'] for r in current])
        rm,rs,rx=metric_stats([r['rotation_deg'] for r in current])
        successes=sum(r['success_le5'] is True for r in current)
        require(successes-sum(by_key[fid,'BASELINE_E']['success_le5'] is True for fid in IDS)
                == counts['RECOVERY']-counts['DAMAGE'],'Recovery damage success arithmetic')
        summary[variant]=dict(denominator=6,successes=successes,recoveries=counts['RECOVERY'],damages=counts['DAMAGE'],
            changed_candidates=changed_candidates,changed_families=changed_families,
            translation_mean_cm=tm,translation_sample_SD_cm=ts,translation_max_cm=tx,
            rotation_mean_deg=rm,rotation_sample_SD_deg=rs,rotation_max_deg=rx,
            translation_gt50cm=sum(r['translation_cm'] > 50 for r in current),
            translation_gt100cm=sum(r['translation_cm'] > 100 for r in current))
    stored_transitions = csv_rows(evaluation_path/'TRANSITIONS.csv')
    require(len(stored_transitions) == 18, 'Complete18 baseline pair transitions')
    stored_by_key = {(r['id'],r['variant']):r for r in stored_transitions}
    require(len(stored_by_key) == 18, 'Transition keys unique')
    for expected in all_transitions:
        for field,value in expected.items():
            equal(value,csv_cast(stored_by_key[expected['id'],expected['variant']][field],value),
                  'Independent18transitions/'+field,maximum)
    stored_summary=read(evaluation_path/'SUMMARY.json')['metrics']
    for variant in VARIANTS:
        for field,value in summary[variant].items():
            equal(value,stored_summary[variant][field],'Independent summary/'+variant+'/'+field,maximum)
    mass_pairs=[]
    for fid in IDS:
        before,after=by_key[fid,'HULL_ONEWAY_MASS'],by_key[fid,'VISIBLE_SUPPORT']
        mass_pairs.append(dict(id=fid,from_variant='HULL_ONEWAY_MASS',to_variant='VISIBLE_SUPPORT',
            transition=transition(before,after),translation_delta_cm=after['translation_cm']-before['translation_cm'],
            rotation_delta_deg=after['rotation_deg']-before['rotation_deg'],
            candidate_changed=before['candidate_id'] != after['candidate_id'],family_changed=before['family'] != after['family']))
    success_controls=[]
    for fid in IDS[4:]:
        record=dict(id=fid)
        for variant in VARIANTS:
            record[variant]=by_key[fid,variant]['success_le5']
        success_controls.append(record)
    # Postseal candidate success membership only supports diagnostics. It never
    # changes costs, tie order, family choice or input observations.
    candidate_metrics = []
    for candidate in candidates:
        fid=candidate['id'];frame=frames[fid]
        candidate_metrics.append(dict(id=fid,stage=candidate['stage'],family=candidate['family'],
            candidate_id=candidate['candidate_id'],**own_metric(dict(available=True,**candidate['pose']),refs[frame['partition']][fid])))
    return dict(passed=True,independent_same_candidate_metrics=24,independent_baseline_pairs=18,
        independent_visible_vs_mass_pairs=mass_pairs,summary=summary,success_controls=success_controls,
        postseal_original_candidate_success_counts_by_frame={fid:sum(r['success_le5'] is True for r in candidate_metrics if r['id']==fid)
            for fid in IDS},maximum_absolute_error_difference=maximum[0],quarter_turn_equivalence_added=False,
        physical_reference_independent_truth_claimed=False,fixed_reference_sha256=digest(reference_path))


def auxiliary_check(independent_receipt, evaluation_path):
    """Compare the six saved matched-control pairs to already independent metrics.

    This small follow-up does not recompute any score or select another pose.
    """
    receipt=read(independent_receipt)
    require(receipt['passed'] and receipt['postseal_evaluation']['passed'], 'Prior direct arithmetic and metrics passed')
    expected=receipt['postseal_evaluation']['independent_visible_vs_mass_pairs']
    stored=csv_rows(Path(evaluation_path)/'VISIBLE_VS_MASS_TRANSITIONS.csv')
    require(len(stored)==6 and [r['id'] for r in stored]==list(IDS), 'Exact six visible versus mass paired denominator')
    maximum=[0.]
    for own,row in zip(expected,stored):
        require(row['comparison']=='VISIBLE_SUPPORT_vs_HULL_ONEWAY_MASS','Predeclared matched-control comparison')
        for field,value in own.items():
            if field not in ('from_variant','to_variant'):
                equal(value,csv_cast(row[field],value),'Independent matched-control CSV/'+field,maximum)
        before=receipt['postseal_evaluation']['summary']['HULL_ONEWAY_MASS']
        after=receipt['postseal_evaluation']['summary']['VISIBLE_SUPPORT']
    counts=Counter(r['transition'] for r in expected)
    require(after['successes']-before['successes']==counts['RECOVERY']-counts['DAMAGE'], 'Matched-control recovery minus damage identity')
    return dict(passed=True, direct_independent_receipt_sha256=digest(independent_receipt),
        matched_control_csv_sha256=digest(Path(evaluation_path)/'VISIBLE_VS_MASS_TRANSITIONS.csv'),
        compared_paired_rows=6,maximum_absolute_difference=maximum[0],
        recoveries=counts['RECOVERY'],damages=counts['DAMAGE'],
        new_score_evaluations=0,new_pose_selections=0,new_reference_reads=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neutral-package',required=True)
    parser.add_argument('--input-manifest')
    parser.add_argument('--run',required=True)
    parser.add_argument('--empty-run')
    parser.add_argument('--evaluation')
    parser.add_argument('--fixed-references')
    parser.add_argument('--reference-seal')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    started,cpu=time.monotonic(),time.process_time()
    result=dict(schema='VISIBLE_BOUNDARY_PILOT_INDEPENDENT_CHECK_V1',
        arithmetic_and_selection=verify_run(args.neutral_package,args.input_manifest,args.run))
    if args.empty_run:
        result['actual_empty_run']=verify_run(args.neutral_package,None,args.empty_run)
    if args.evaluation:
        require(args.fixed_references and args.reference_seal,'Postseal evaluation requires old reference binding')
        result['postseal_evaluation']=evaluation_check(args.run,args.neutral_package,args.evaluation,args.fixed_references,args.reference_seal)
    result.update(passed=True,new_NN_forward_calls=0,new_pose_fit_calls=0,new_candidate_generation_calls=0,
        checker_source_sha256=digest(__file__),elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu,maximum_resident_set_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
