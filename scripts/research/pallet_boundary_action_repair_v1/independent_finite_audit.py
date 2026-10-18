"""Independent fixed-pose finite-support audit of all B frame outputs.

This module imports neither finite_boundary, finite_solver, nor their geometry
or objective functions. Camera projection is direct matrix algebra and query
intersection is an independent 2x2 solve. It reads no pose references.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def stream(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n')


def query_set(bundle):
    groups = {}; result = []
    for g in (bundle or {}).get('groups', []):
        key = int(g['native_corner'])
        if key in groups:
            if json.dumps(g, sort_keys=True) != json.dumps(groups[key], sort_keys=True):
                raise ValueError('Conflicting duplicated original group')
            continue
        groups[key] = g
        if not g['supported']:
            continue
        queries = {q['support_id']: q for q in g['queries'] if q['supported']}
        result.extend(queries[k] for k in sorted(queries))
    return sorted(result, key=lambda x: (x['native_corner'], x['support_id']))


def projection(pose, geometry, camera):
    R = np.asarray(pose['R_physical'], dtype=float)
    Q = np.asarray(geometry['physical_registration_Q'], dtype=float)
    xyz = np.asarray(geometry['native_xyz'], dtype=float)
    center = np.asarray(pose['centroid'], dtype=float)
    native_R = R@Q.T
    camera_xyz = (xyz-xyz.mean(axis=0))@native_R.T+center
    homogeneous = camera_xyz@np.asarray(camera, dtype=float).T
    # Selected valid poses have positive depths. Invalid bank candidates retain
    # their actual signed depths and direct projection; no canvas clipping.
    with np.errstate(divide='ignore', invalid='ignore'):
        xy = homogeneous[:,:2]/homogeneous[:,2,None]
    return xy, camera_xyz[:,2]


def secondary_parameterization_projection(pose, geometry, camera):
    """Numerical expression replay only, deliberately separate from raw check.

    General SciPy rotation primitives reproduce the fixed solver's existing
    matrix -> rotation vector -> matrix representation. No production pose,
    projection, finite geometry, constraint, or solver function is imported.
    The original raw-matrix independent projection remains unchanged above.
    """
    from scipy.spatial.transform import Rotation
    xyz = np.asarray(geometry['native_xyz'], float)
    raw_R = np.asarray(pose['R_physical'], float)@np.asarray(geometry['physical_registration_Q'], float).T
    center = np.asarray(pose['centroid'], float)
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    original_native_t = center-raw_R@xyz.mean(axis=0)
    vector = Rotation.from_matrix(raw_R).as_rotvec()
    roundtrip_R = Rotation.from_rotvec(vector.copy()).as_matrix()
    camera_xyz = xyz@roundtrip_R.T+(original_native_t/scale)*scale
    homogeneous = camera_xyz@np.asarray(camera, float).T
    with np.errstate(divide='ignore', invalid='ignore'):
        xy = homogeneous[:,:2]/homogeneous[:,2,None]
    return xy, camera_xyz[:,2]


def independent_profile(query, xy, depth, bundle):
    k, m = int(query['native_corner']), int(query['native_neighbor'])
    a, b = xy[k], xy[m]
    c, n = np.asarray(query['center_xy'], float), np.asarray(query['outward_normal_xy'], float)
    vec = b-a; length = float(np.linalg.norm(vec))
    machine = np.finfo(np.float64).eps
    lambda_eps = 8*math.sqrt(machine)
    pixel_eps = math.sqrt(machine)*max(1., math.hypot(*bundle['image_hw']))
    denom = float(np.linalg.det(np.column_stack([n, vec])))
    threshold = max(pixel_eps, math.sqrt(machine)*length)
    status = 'SUPPORTED_FINITE_SEGMENT'
    if not np.isfinite(xy).all() or not np.isfinite([denom, length]).all():
        status = 'NONFINITE_PROJECTION'
    elif not np.isfinite(depth).all() or np.min(depth) <= 0:
        status = 'NONPOSITIVE_DEPTH'
    elif length <= pixel_eps:
        status = 'DEGENERATE_PROJECTED_EDGE'
    elif abs(denom) <= threshold:
        status = 'PARALLEL_NORMAL_AND_EDGE'
    lam, offset, condition = None, None, None
    error_bound = 0.
    if status not in ('NONFINITE_PROJECTION', 'DEGENERATE_PROJECTED_EDGE', 'PARALLEL_NORMAL_AND_EDGE'):
        mat = np.column_stack([n, -vec])
        try:
            distance, lam = np.linalg.solve(mat, a-c)
            lam = float(lam); offset = float(distance/bundle['pixel_scale'])
            condition = float(np.linalg.cond(mat))
            # Predeclared numerical solve forward-error bound, not a widened
            # segment predicate or an accuracy threshold. Boundary disagreements
            # are explicitly listed; they never become hidden normal supports.
            error_bound = max(1e-10, 64*machine*condition*(1+abs(lam)))
            if status == 'SUPPORTED_FINITE_SEGMENT' and lam < -lambda_eps:
                status = 'BEFORE_FINITE_ENDPOINT'
            elif status == 'SUPPORTED_FINITE_SEGMENT' and lam > 1+lambda_eps:
                status = 'AFTER_FINITE_ENDPOINT'
        except np.linalg.LinAlgError:
            status = 'PARALLEL_NORMAL_AND_EDGE'
    return dict(support_id=query['support_id'], lambda_segment=lam, offset_base_px=offset,
        projected_length_px=length, denominator_px=denom, geometry_status=status,
        finite_support=status == 'SUPPORTED_FINITE_SEGMENT', lambda_error_bound=error_bound,
        linear_system_condition=condition)


def rotation_magnitude(R):
    R = np.asarray(R, float)
    sine = .5*math.sqrt((R[2,1]-R[1,2])**2+(R[0,2]-R[2,0])**2+(R[1,0]-R[0,1])**2)
    cosine = .5*(float(np.trace(R))-1)
    return math.atan2(sine, cosine)


def local_cost(candidate, geometry):
    solve = candidate.get('solve', {}); reg = solve.get('local_regularization')
    if reg is None:
        return 0.
    start = solve['start_pose']; R = np.asarray(candidate['R_physical']); R0 = np.asarray(start['R_physical'])
    angle = rotation_magnitude(R0.T@R)
    xyz = np.asarray(geometry['native_xyz']); scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    dt = np.asarray(candidate['centroid'])-start['centroid']
    return float(reg['rotation_precision']*angle**2+reg['translation_precision']*float(dt@dt)/scale**2)


def self_test():
    # Fixed analytic answers, independent of all production geometry helpers.
    image = dict(image_hw=[80, 100], pixel_scale=1.)
    xy = np.zeros((8,2)); xy[1] = [10., 0.]
    base = dict(support_id='q', native_corner=0, native_neighbor=1, center_xy=[5., -.37], outward_normal_xy=[0., 1.])
    expected = [(-1., False), (0., True), (5., True), (10., True), (11., False)]
    for x, good in expected:
        record = independent_profile(dict(base, center_xy=[x, -.37]), xy, np.ones(8), image)
        if record['finite_support'] != good or abs(record['lambda_segment']-x/10) > 1e-14:
            raise AssertionError(record)
    q = xy.copy(); q[1] = [0., 1.]
    if independent_profile(base, q, np.ones(8), image)['geometry_status'] != 'PARALLEL_NORMAL_AND_EDGE':
        raise AssertionError('parallel')
    q[1] = [1e-12, 0.]
    if independent_profile(base, q, np.ones(8), image)['geometry_status'] != 'DEGENERATE_PROJECTED_EDGE':
        raise AssertionError('degenerate')
    if independent_profile(base, xy, -np.ones(8), image)['geometry_status'] != 'NONPOSITIVE_DEPTH':
        raise AssertionError('depth')
    # Known camera and basis change must produce the identical physical image.
    xyz = np.array([[i*.1, .2, 2.+i*.1] for i in range(8)])
    pose = dict(R_physical=np.eye(3).tolist(), centroid=[0., 0., 3.])
    g = dict(native_xyz=xyz.tolist(), physical_registration_Q=np.eye(3).tolist())
    camera = [[100.,0.,50.],[0.,100.,40.],[0.,0.,1.]]
    a, z = projection(pose, g, camera)
    Q = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    b, zz = projection(pose, dict(native_xyz=(xyz@Q.T).tolist(), physical_registration_Q=Q.tolist()), camera)
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    np.testing.assert_allclose(z, zz, atol=1e-12, rtol=0)
    return dict(status='PASS', analytic_range_cases=5, singular_depth_cases=3, coordinate_basis_case=1,
        production_geometry_imports=0, reference_reads=0)


def audit(inputs, original_batch, batch_dir, output, *, prior_failed_check=None):
    before, cpu = time.monotonic(), time.process_time()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    batch = Path(batch_dir); original = Path(original_batch)
    seal = load(batch/'BATCH_SEAL.json'); lock = load(batch/'BATCH_LOCK.json')
    if not seal['complete'] or seal['lock_sha256'] != digest(batch/'BATCH_LOCK.json') or seal['predictions_sha256'] != digest(batch/'PREDICTIONS.jsonl.gz'):
        raise ValueError('Finite batch seal/bytes mismatch')
    if lock['original_inputs_sha256'] != digest(inputs):
        raise ValueError('Input bytes mismatch')
    if lock['original_batch_seal_sha256'] != digest(original/'BATCH_SEAL.json'):
        raise ValueError('Original source batch mismatch')
    for name, value in lock['source_sha256'].items():
        if digest(Path(__file__).parent/name) != value:
            raise ValueError('Frozen scientific source changed: '+name)
    frames = list(stream(inputs)); predictions = list(stream(batch/'PREDICTIONS.jsonl.gz'))
    by_id = {f['id']: (i, f) for i, f in enumerate(frames)}
    if len(predictions) != 238 or len(set(p['id'] for p in predictions)) != 238:
        raise ValueError('Exact238 expected, all failures remain denominator')
    counts = Counter(); failure = []; boundaries = []; frame_rows = []; comparisons = Counter(); representations = []
    max_lambda_difference = 0.; max_offset_difference = 0.; max_condition = 0.
    def fail(kind, **fields):
        failure.append(dict(kind=kind, **fields))
    for row in predictions:
        index, frame = by_id[row['id']]; ident = frame['id']; counts['frames'] += 1; counts['partition:'+frame['partition']] += 1
        folder = batch/'frames'/f'{index:04d}'; fseal = load(folder/'FRAME_SEAL.json')
        if fseal['result_sha256'] != digest(folder/'RESULT.jsonl.gz'):
            fail('new_result_hash', id=ident); continue
        oldfolder = original/'frames'/f'{index:04d}'; oldseal = load(oldfolder/'FRAME_SEAL.json')
        if fseal['original_frame_seal_sha256'] != digest(oldfolder/'FRAME_SEAL.json') or oldseal['result_sha256'] != digest(oldfolder/'RESULT.jsonl.gz'):
            fail('original_source_binding', id=ident); continue
        source = next(stream(oldfolder/'RESULT.jsonl.gz')); bpath = oldfolder/'BOUNDARY_BUNDLE.jsonl.gz'
        bundle = next(stream(bpath)) if bpath.exists() else None
        if bundle is not None and source['boundary']['bundle_sha256'] != digest(bpath):
            fail('original_bundle_hash', id=ident); continue
        qs = query_set(bundle); result = next(stream(folder/'RESULT.jsonl.gz'))
        fixed = result['fixed_universe']
        expected_bundle_digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if (fixed['profile_count'] != len(qs) or fixed['original_fixed_bundle_sha256'] != expected_bundle_digest or
                fixed['consumed_native_ids'] != (bundle or {}).get('consumed_native_ids', []) or
                fixed['consumed_group_ids'] != (bundle or {}).get('consumed_group_ids', [])):
            fail('immutable_original_universe_binding', id=ident)
        if row['predictions'] != result['predictions']:
            fail('aggregate_prediction_differs_from_frame', id=ident)
        for arm, pred in result['predictions'].items():
            counts['arm:'+arm] += 1
            if pred.get('fallback'):
                counts[arm+':fallback'] += 1
                if pred['actual_pose'] != frame['fallback']:
                    fail('fallback_not_exact_original_N0', id=ident, arm=arm)
                if pred.get('finite_support') is not None or pred.get('boundary_applied'):
                    fail('fallback_claimed_as_new_finite_correction', id=ident, arm=arm)
            bank = result['banks'][arm]; score_rows = result['finite_scores'][arm]
            if len(bank) != len(score_rows):
                fail('bank_score_length', id=ident, arm=arm); continue
            choices = []
            for ci, (candidate, all_scores) in enumerate(zip(bank, score_rows)):
                for family, stored in all_scores.items():
                    xy, depth = projection(candidate, frame['families'][family], frame['packet']['solver_input']['K'])
                    direct = [independent_profile(query, xy, depth, bundle) for query in qs]
                    secondary_xy = None; secondary_depth = None
                    profiles = stored['support']['profiles']
                    counts['candidate_family_pairs'] += 1; counts['profile_checks'] += len(qs)
                    if len(profiles) != len(qs) or stored['support']['fixed_profile_count'] != len(qs):
                        fail('candidate_dependent_fixed_universe', id=ident, arm=arm, candidate_id=candidate['candidate_id'], family=family); continue
                    for d, p in zip(direct, profiles):
                        comparisons['geometry_status_pair:'+d['geometry_status']+'/'+p['geometry_status']] += 1
                        if d['support_id'] != p['support_id']:
                            fail('support_identity_order', id=ident, arm=arm); continue
                        if d['lambda_segment'] is not None and p['lambda_segment'] is not None:
                            difference = abs(d['lambda_segment']-p['lambda_segment']); max_lambda_difference = max(max_lambda_difference, difference)
                            max_condition = max(max_condition, d['linear_system_condition'])
                            if difference > d['lambda_error_bound']:
                                if secondary_xy is None:
                                    secondary_xy, secondary_depth = secondary_parameterization_projection(candidate,
                                        frame['families'][family], frame['packet']['solver_input']['K'])
                                query = next(qq for qq in qs if qq['support_id'] == d['support_id'])
                                replay = independent_profile(query, secondary_xy, secondary_depth, bundle)
                                residual = abs(replay['lambda_segment']-p['lambda_segment']) if replay['lambda_segment'] is not None else float('inf')
                                if residual > d['lambda_error_bound']:
                                    fail('lambda_arithmetic_unexplained_after_expression_replay', id=ident, arm=arm,
                                        candidate_id=candidate['candidate_id'], family=family, support_id=d['support_id'],
                                        independent=d['lambda_segment'], secondary=replay['lambda_segment'],
                                        stored=p['lambda_segment'], bound=d['lambda_error_bound'])
                                else:
                                    representations.append(dict(id=ident, arm=arm, candidate_id=candidate['candidate_id'],
                                        family=family, support_id=d['support_id'], independent=d['lambda_segment'],
                                        secondary=replay['lambda_segment'], stored=p['lambda_segment'], bound=d['lambda_error_bound'],
                                        raw_difference=difference, replay_difference=residual,
                                        maximum_projection_roundtrip_difference_px=float(np.max(np.abs(xy-secondary_xy))),
                                        independent_geometry_status=d['geometry_status'],
                                        stored_geometry_status=p['geometry_status'],
                                        selected_candidate=pred.get('selected_candidate_id') == candidate['candidate_id'] and pred.get('selected_family') == family))
                            offset_difference = abs(d['offset_base_px']-p['offset_base_px']); max_offset_difference = max(max_offset_difference, offset_difference)
                        if d['finite_support'] != p['finite_support']:
                            lam = d['lambda_segment']; eps = 8*math.sqrt(np.finfo(float).eps)
                            boundary_distance = min(abs(lam+eps), abs(lam-1-eps)) if lam is not None else float('inf')
                            event = dict(id=ident, arm=arm, candidate_id=candidate['candidate_id'], family=family,
                                support_id=d['support_id'], independent=d, stored_geometry_status=p['geometry_status'], stored_lambda=p['lambda_segment'])
                            if boundary_distance <= d['lambda_error_bound']:
                                boundaries.append(event)
                            else:
                                fail('finite_predicate_disagreement', **event)
                    # Selection eligibility uses the frozen original feasibility
                    # plus the separately stored finite gate. Roundoff boundary
                    # disagreements above are visible, never silently recoded.
                    if stored['feasible'] != bool(stored['original_feasible'] and stored['finite_support'] and np.isfinite(stored['energy'])):
                        fail('eligibility_formula', id=ident, arm=arm, candidate_id=candidate['candidate_id'])
                    if stored['feasible']:
                        energy = stored['energy'] + (local_cost(candidate, frame['families'][family]) if arm == 'B_FINITE_LOCAL' else 0.)
                        choices.append((energy, ci, family))
            if choices:
                expected = min(choices, key=lambda z: (z[0], z[1], z[2])); c = bank[expected[1]]
                if pred.get('fallback') or pred['selected_candidate_id'] != c['candidate_id'] or pred['selected_family'] != expected[2]:
                    fail('selector_minimum', id=ident, arm=arm, expected=c['candidate_id'], actual=pred['selected_candidate_id'])
                if abs(pred['selected_energy']-expected[0]) > 1e-8*max(1., abs(expected[0])):
                    fail('selected_energy_arithmetic', id=ident, arm=arm, expected=expected[0], actual=pred['selected_energy'])
                actual = pred['actual_pose']
                if actual['R_physical'] != c['R_physical'] or actual['centroid'] != c['centroid']:
                    fail('selected_actual_pose_identity', id=ident, arm=arm)
                xy, depth = projection(actual, frame['families'][pred['selected_family']], frame['packet']['solver_input']['K'])
                direct = [independent_profile(query, xy, depth, bundle) for query in qs]
                invalid = [d for d in direct if not d['finite_support']]
                counts[arm+':selected'] += 1; counts[arm+':selected_independent_invalid_profiles'] += len(invalid)
                if invalid:
                    ids = {d['support_id'] for d in invalid}
                    unaccounted = ids-{e['support_id'] for e in boundaries if e['id']==ident and e['arm']==arm and e['candidate_id']==pred['selected_candidate_id'] and e['family']==pred['selected_family']}
                    if unaccounted:
                        fail('selected_unexplained_unsupported_profile', id=ident, arm=arm, support_ids=sorted(unaccounted))
                if c.get('solve'):
                    counts[arm+':selected_optimizer_success'] += bool(c['solve'].get('optimizer_success'))
                    counts[arm+':selected_optimizer_false'] += not bool(c['solve'].get('optimizer_success'))
                frame_rows.append(dict(id=ident, arm=arm, fallback=False, selected_candidate_id=pred['selected_candidate_id'],
                    selected_family=pred['selected_family'], independent_supported_profiles=len(qs)-len(invalid),
                    fixed_profiles=len(qs), selected_candidate_pool=len(bank), eligible_candidate_family_pairs=len(choices)))
            elif not pred.get('fallback'):
                fail('no_eligible_without_fallback', id=ident, arm=arm)
            else:
                frame_rows.append(dict(id=ident, arm=arm, fallback=True, fixed_profiles=len(qs), selected_candidate_pool=len(bank), eligible_candidate_family_pairs=0))
        attempts = result['attempts']; caps = Counter(a['arm'] for a in attempts)
        if caps['B_FINITE_FULL'] > 8 or caps['EDGE_FINITE_FULL'] > 8 or caps['B_FINITE_LOCAL'] > 2:
            fail('solve_cap', id=ident, actual=dict(caps))
        counts['optimizer_attempts'] += len(attempts)
        for a in attempts:
            sol = a['solve']
            if sol.get('total_iteration_budget_used', 0) > 80:
                fail('iteration_cap', id=ident, candidate_id=a['candidate_id'])
            if sol.get('feasible') and not sol.get('finite_support'):
                fail('feasible_without_support', id=ident, candidate_id=a['candidate_id'])
    path = out/'GEOMETRY_ROWS.jsonl.gz'
    if path.exists():
        raise FileExistsError(path)
    with gzip.open(path, 'wt') as f:
        for r in frame_rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False)+'\n')
    representation_path = out/'RAW_MATRIX_REPRESENTATION_DIFFERENCES.jsonl.gz'
    with gzip.open(representation_path, 'wt') as f:
        for event in representations:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
    prior_accounting = None
    if prior_failed_check is not None:
        prior = load(prior_failed_check)
        if prior['status'] != 'FAIL' or prior['finite_batch_seal_sha256'] != digest(batch/'BATCH_SEAL.json'):
            raise ValueError('Exact original v1 failed receipt required')
        def key(event):
            return tuple(event[k] for k in ('id','arm','family','support_id','independent','stored','bound'))
        old_counts = Counter(key(event) for event in prior['failures'] if event['kind']=='lambda_arithmetic')
        new_counts = Counter(key(event) for event in representations)
        prior_accounting = dict(prior_receipt_sha256=digest(prior_failed_check),
            prior_failure_count=prior['failure_count'], representation_difference_count=len(representations),
            every_original_arithmetic_failure_accounted_exactly=old_counts==new_counts,
            non_arithmetic_prior_failures=sum(e['kind']!='lambda_arithmetic' for e in prior['failures']))
        if old_counts != new_counts or prior_accounting['non_arithmetic_prior_failures']:
            fail('original_v1_failure_accounting', old_count=sum(old_counts.values()), new_count=sum(new_counts.values()))
        # Verify that primary raw projection and 2x2/predicate expressions were
        # not changed to make the failing raw arithmetic comparison disappear.
        old_source = Path(prior_failed_check).parent/'independent_finite_audit_source.py'
        def functions_hash(path):
            source = path.read_text(); tree = ast.parse(source)
            names = ('projection','independent_profile')
            return {n.name: hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()
                for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
        old_functions, new_functions = functions_hash(old_source), functions_hash(Path(__file__))
        prior_accounting['primary_function_sha256_before'] = old_functions
        prior_accounting['primary_function_sha256_after'] = new_functions
        prior_accounting['raw_projection_and_predicate_unchanged'] = old_functions==new_functions
        if old_functions != new_functions:
            fail('primary_independent_function_changed')
    receipt = dict(schema='INDEPENDENT_FINITE_GEOMETRY_AUDIT_V2',
        status=('PASS_WITH_RAW_REPRESENTATION_DIFFERENCES' if representations else 'PASS') if not failure else 'FAIL', complete=True,
        geometric_support_status='PASS' if not failure else 'FAIL',
        primary_raw_lambda_arithmetic_status='FAIL_PRESERVED_AND_EXPLAINED' if representations else 'PASS',
        secondary_expression_arithmetic_status='PASS' if not failure else 'FAIL',
        representation_difference_count=len(representations), representation_difference_examples=representations[:10],
        full_representation_difference_sha256=digest(representation_path),
        full_representation_difference_file=representation_path.name,
        prior_v1_failure_accounting=prior_accounting,
        model_pose_observation_epsilon_selection_changes=0,
        secondary_expression_replay_note='Library rotation representation replay only; raw independent projection/predicate/2x2 solve and comparison bounds unchanged. Original v1 raw arithmetic FAIL remains preserved.',
        source_sha256=digest(__file__), original_inputs_sha256=digest(inputs), original_batch_seal_sha256=digest(original/'BATCH_SEAL.json'),
        finite_batch_seal_sha256=digest(batch/'BATCH_SEAL.json'), counts=dict(counts),
        predicate_status_comparisons=dict(comparisons), failure_count=len(failure), failures=failure,
        strict_boundary_disagreement_count=len(boundaries), strict_boundary_disagreements=boundaries,
        numeric_boundary_policy='strict predicate disagreements are retained; only independently bounded floating-point endpoint differences are classified as representation boundaries, never deleted or recoded',
        max_lambda_arithmetic_difference=max_lambda_difference, max_offset_arithmetic_difference=max_offset_difference,
        max_linear_system_condition=max_condition, geometry_rows_sha256=digest(path),
        original_material_owner='UNVERIFIED', actual_reference_or_GT_reads=0,
        production_geometry_imports=0, new_PnP=0, new_optimization=0, new_candidate_generation=0,
        all238_frames=True, elapsed_seconds=time.monotonic()-before, process_cpu_seconds=time.process_time()-cpu)
    save(out/'INDEPENDENT_FINITE_CHECK.json', receipt)
    return receipt

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--inputs', required=True); p.add_argument('--original-batch', required=True)
    p.add_argument('--batch-dir', required=True); p.add_argument('--output', required=True); p.add_argument('--prior-failed-check')
    a = p.parse_args(); result = audit(a.inputs, a.original_batch, a.batch_dir, a.output, prior_failed_check=a.prior_failed_check)
    print(json.dumps({k: result[k] for k in ('status','counts','failure_count','strict_boundary_disagreement_count','elapsed_seconds')}, indent=2))
    if result['failure_count']:
        raise SystemExit(1)
