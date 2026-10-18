"""Finite-support B solve and separate N0-local Tikhonov package.

Original conditional costs and seed proposals are preserved. Restoration uses
analytic linearized constraints, counts inside the same 80-iteration solve
budget, and never reads a reference. No extra multistart is introduced.
"""
from __future__ import annotations
from copy import deepcopy
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as OLD
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import boundary_model as OB
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import canonical, rows, read, sha, write
from . import finite_boundary as F

ARMS = ('B_FINITE_BANK', 'B_FINITE_FULL', 'B_FINITE_LOCAL', 'EDGE_FINITE_FULL')


def compile_models(frame, config, bundle=None):
    # Canonicalize only identical serialized groups. All originally unavailable
    # groups and their profiles remain in the bundle; conflicts are not hidden.
    if bundle is not None:
        unique = {}; groups = []
        for group in bundle['groups']:
            key = int(group['native_corner'])
            if key in unique:
                if canonical(group) != canonical(unique[key]):
                    raise ValueError('Conflicting duplicated original support group')
                continue
            unique[key] = group; groups.append(group)
        bundle = dict(bundle, groups=groups)
    return OLD.compile_frame(frame, config, bundle=bundle)


def projected(model, p):
    c = model.ctx
    return OLD.J.projection_and_jacobian(p, c.xyz, c.K, c.scale)


def finite_pair(model, p, signs=None):
    q, depth, jq, jd = projected(model, p)
    old, jo = OLD.constraints(model, p, 0, replace_B=True)
    new, jn = F.constraints_and_jacobian(model.bundle, q, jq, signs)
    return np.r_[old, new], np.concatenate([jo, jn]), q, depth


def support_score(candidate, model, mode, cached=None):
    p = OLD._native_pose(candidate, model.ctx)
    q, depth, _, _ = projected(model, p)
    support = F.support_audit(model.bundle, q, depth)
    slack = float(np.min(OLD.constraints(model, p, 0, replace_B=True)[0]))
    if cached is None:
        value, _, _ = OLD.objective(model, p, mode)
    else:
        value = cached['scores'][mode]
    old_feasible = slack >= -model.ctx.cfg['feasibility_tolerance']
    return dict(energy=float(value), feasible=bool(old_feasible and support['finite_support'] and np.isfinite(value)),
        original_feasible=bool(old_feasible), minimum_original_slack=slack,
        finite_support=support['finite_support'], support=support)


def local_penalty(p, p0, model, regularization):
    c = model.ctx
    R, t = OLD.V._pose(p, c.scale)
    R0, t0 = OLD.V._pose(p0, c.scale)
    mean = c.xyz.mean(0)
    r = Rotation.from_matrix(R0.T@R).as_rotvec()
    delta = (R@mean+t-R0@mean-t0)/c.scale
    rr, tt = regularization['rotation_precision'], regularization['translation_precision']
    g = np.zeros(6)
    g[:3] = 2*rr*OLD.J.so3_right_jacobian(p[:3]).T@r
    dc = np.zeros((3, 6)); dc[:, :3] = -R@OLD.J.skew(mean)@OLD.J.so3_right_jacobian(p[:3])/c.scale
    dc[:, 3:] = np.eye(3)
    g += 2*tt*dc.T@delta
    return float(rr*(r@r)+tt*(delta@delta)), g


def restore_feasibility(p0, pair, *, max_steps):
    """Damped linear feasibility corrections, with finite merit backtracking.

    Counts each correction as one of the fixed solve iterations. The damping
    follows sqrt(machine epsilon)*max(1, ||A||_2^2); correction norm is capped
    at one dimensionless parameter unit. No new optimizer invocation occurs.
    """
    p = p0.copy(); journal = []
    for _ in range(max_steps):
        g, j, _, _ = pair(p)
        bad = g < 0
        merit = float(np.linalg.norm(np.minimum(g, 0)))
        if not bad.any() or merit <= 1e-10:
            break
        a, b = j[bad], -g[bad]
        damping = np.sqrt(np.finfo(float).eps)*max(1., float(np.linalg.norm(a, 2)**2))
        step = np.linalg.solve(a.T@a+damping*np.eye(6), a.T@b)
        length = float(np.linalg.norm(step))
        if length > 1:
            step /= length
        accepted = False; trials = 0
        for exponent in range(10):
            trials += 1
            trial = p + step*(.5**exponent)
            try:
                gg, _, _, _ = pair(trial)
                merit2 = float(np.linalg.norm(np.minimum(gg, 0)))
            except (ValueError, FloatingPointError, OverflowError):
                merit2 = float('inf')
            if np.isfinite(merit2) and merit2 < merit:
                p, accepted = trial, True
                break
        journal.append(dict(violation_l2_before=merit, violation_l2_after=merit2 if np.isfinite(merit2) else None,
            accepted=accepted, correction_norm=float(np.linalg.norm(step)), backtracking_trials=trials))
        if not accepted:
            break
    return p, journal


def solve_one(model, start, mode, h, config, *, regularization=None):
    before = time.monotonic(); p0 = OLD._native_pose(start, model.ctx)
    q0 = projected(model, p0)[0]
    signs = F.constraint_orientation(model.bundle, q0) if model.bundle else np.empty(0)
    calls = dict(objective=0, constraint=0)
    def pair(p):
        calls['constraint'] += 1
        return finite_pair(model, p, signs)
    def fg(p):
        calls['objective'] += 1
        value, grad, _ = OLD.objective(model, p, mode)
        if regularization is not None:
            v, g = local_penalty(p, p0, model, regularization)
            value += v; grad += g
        return value, grad
    maxiter = int(config['max_iterations'])
    pstart, restoration = restore_feasibility(p0, pair, max_steps=min(20, maxiter//4))
    left = max(1, maxiter-len(restoration))
    result = minimize(fg, pstart, jac=True, method='SLSQP',
        constraints=[dict(type='ineq', fun=lambda p: pair(p)[0], jac=lambda p: pair(p)[1])],
        options=dict(maxiter=left, ftol=config['optimizer_ftol'], disp=False))
    values, cj, q, depth = pair(result.x)
    pose = OLD._actual_pose(result.x, model.ctx)
    audit = F.support_audit(model.bundle, q, depth)
    slack = float(np.min(values))
    numerical = bool(np.isfinite(result.x).all() and np.isfinite(result.fun) and
                     OLD.V._valid_rotation(np.asarray(pose['R_cf'])))
    feasible = bool(numerical and slack >= -model.ctx.cfg['feasibility_tolerance'] and audit['finite_support'])
    raw, grad = fg(result.x)
    R0 = np.asarray(start['R_physical']); R1 = np.asarray(pose['R_physical'])
    movement = dict(rotation_radians=float(Rotation.from_matrix(R1@R0.T).magnitude()),
        translation_m=float(np.linalg.norm(np.asarray(pose['centroid'])-start['centroid'])),
        translation_scale_fraction=float(np.linalg.norm(np.asarray(pose['centroid'])-start['centroid'])/model.ctx.scale),
        projection_rms_px=float(np.sqrt(np.mean((q-q0)**2))),
        projection_max_corner_px=float(np.max(np.linalg.norm(q-q0, axis=1))))
    return dict(candidate_id=f'NEW_FINITE/{mode}/{model.ctx.family_name}/h{h}/{start["candidate_id"]}',
        source='NEW_FINITE_LOCAL' if regularization else 'NEW_FINITE_'+mode,
        original_native_family=model.ctx.family_name, R_physical=pose['R_physical'], centroid=pose['centroid'],
        solve=dict(mode=mode, global_owner_h=0, proposal_native_yaw_index=h,
            start_id=start['candidate_id'], start_pose=deepcopy(start), actual_pose=pose,
            optimizer_success=bool(result.success), optimizer_status=int(result.status),
            optimizer_message=str(result.message), iterations=int(result.nit),
            restoration_iterations=len(restoration), restoration=restoration,
            total_iteration_budget_used=int(result.nit)+len(restoration), iteration_budget=maxiter,
            objective=float(result.fun), finite_support=audit['finite_support'], support=audit,
            minimum_constraint_slack=slack, feasible=feasible, numeric_pass=numerical,
            raw_gradient_norm=float(np.linalg.norm(grad)),
            stationarity_note='raw objective gradient is not a constrained KKT certificate',
            maximum_constraint_violation=float(max(0., -slack)),
            cumulative_SE3=movement, local_regularization=regularization,
            calls=calls, elapsed_seconds=time.monotonic()-before))


def _select(bank, models, frame, mode, *, cached_scores=None, local=False):
    all_scores = []; eligible = []
    for i, candidate in enumerate(bank):
        scores = {}
        for family, model in models.items():
            # Local owner/family is fixed per solve, never selected within it.
            # Cross-family aliases of the identical N0 remain two starts only.
            if local and candidate.get('original_native_family') != family:
                continue
            old = cached_scores[i].get(family) if cached_scores is not None else None
            scores[family] = support_score(candidate, model, mode, old)
            if scores[family]['feasible']:
                penalty = candidate.get('solve', {}).get('local_regularization')
                value = scores[family]['energy']
                if penalty is not None:
                    p0 = OLD._native_pose(candidate['solve']['start_pose'], model.ctx)
                    value += local_penalty(OLD._native_pose(candidate, model.ctx), p0, model, penalty)[0]
                eligible.append((value, i, family))
        all_scores.append(scores)
    if not eligible:
        return dict(actual_pose=deepcopy(frame['fallback']), fallback=True,
            prediction_status='NO_FINITE_FEASIBLE_OUTPUT_FALLBACK_N0', selected_candidate_id='N0',
            mode=mode, boundary_applied=False, finite_support=None,
            support_is_fallback_not_new_B_output=True), all_scores
    energy, i, family = min(eligible, key=lambda z: (z[0], z[1], z[2]))
    c = bank[i]; score = all_scores[i][family]
    return dict(actual_pose=dict(available=True, R_physical=deepcopy(c['R_physical']), centroid=deepcopy(c['centroid'])),
        fallback=False, prediction_status='SELECTED', selected_candidate_id=c['candidate_id'],
        selected_family=family, selected_energy=energy, mode=mode,
        boundary_applied=bool(models[family].bundle and models[family].bundle['consumed_native_ids']),
        finite_support=score['finite_support'], support=score['support'],
        selected_optimizer=c.get('solve'), selected_candidate_source=c.get('source')) , all_scores


def generate(frame, config, original_result, bundle, regularization):
    started = time.monotonic(); models = compile_models(frame, config, bundle=bundle)
    carried = original_result['carried']
    cached = original_result['carried_scores']
    predictions = {}; banks = {}; score_rows = {}; attempts = []
    predictions['B_FINITE_BANK'], score_rows['B_FINITE_BANK'] = _select(carried, models, frame, 'B', cached_scores=cached)
    banks['B_FINITE_BANK'] = carried
    by_id = {c['candidate_id']: c for c in carried}
    starts = original_result['starts']
    for arm, mode in [('B_FINITE_FULL', 'B'), ('EDGE_FINITE_FULL', 'edge')]:
        new = []
        for family, model in models.items():
            if len(starts[family]) > 4:
                raise ValueError('Original full proposal cap exceeded')
            for h, key in enumerate(starts[family]):
                start = OLD.owner_start_proposal(by_id[key], model, h%4)
                try:
                    result = solve_one(model, start, mode, h%4, config)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                    result = dict(candidate_id=f'FAILED_FINITE/{arm}/{family}/{h}', source='FAILED_TECHNICAL',
                        solve=dict(feasible=False, finite_support=False, optimizer_success=False,
                            exception=repr(exc), start_id=start['candidate_id']))
                result['arm'] = arm; attempts.append(result)
                if result['solve']['feasible']:
                    new.append(result)
        bank = OLD.unique_poses(carried+new, max(m.ctx.scale for m in models.values()))
        predictions[arm], score_rows[arm] = _select(bank, models, frame, mode,
            cached_scores=cached+[{} for _ in bank[len(carried):]])
        banks[arm] = bank
    local = []
    for family, model in models.items():
        start = dict(candidate_id='N0_LOCAL_SAME_PHYSICAL_POSE/'+family, source='N0',
            R_physical=deepcopy(frame['fallback']['R_physical']), centroid=deepcopy(frame['fallback']['centroid']))
        try:
            result = solve_one(model, start, 'B', 0, config, regularization=regularization)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            result = dict(candidate_id='FAILED_FINITE/LOCAL/'+family, source='FAILED_TECHNICAL',
                solve=dict(feasible=False, finite_support=False, optimizer_success=False, exception=repr(exc), start_id=start['candidate_id']))
        result['arm'] = 'B_FINITE_LOCAL'; attempts.append(result)
        if result['solve']['feasible']:
            local.append(result)
    banks['B_FINITE_LOCAL'] = local
    predictions['B_FINITE_LOCAL'], score_rows['B_FINITE_LOCAL'] = _select(local, models, frame, 'B', local=True)
    assert len(attempts) <= 18
    return dict(id=frame['id'], partition=frame['partition'], session=frame['session'], cluster_id=frame['cluster_id'],
        predictions=predictions, banks=banks, finite_scores=score_rows, attempts=attempts,
        starts=starts, fixed_universe=F.universe(bundle), actual_new_solve_attempts=len(attempts),
        original_B_RGB_reused=True, reference_inputs=False, elapsed_seconds=time.monotonic()-started)


def calibrate_regularization(inputs, old_config_path, original_batch, output):
    """One frozen train-only observed-information rule, no strength search."""
    started = time.monotonic(); config = read(old_config_path); values = []; ids = []
    counts = dict(complete_rows=0, partial_rows=0, B_profiles=0, B_unsupported_profiles=0, B_invalid_at_N0=0, no_information_frames=0)
    from pathlib import Path
    for index, frame in enumerate(rows(inputs)):
        if frame.get('split') != 'train':
            continue
        folder = Path(original_batch)/'frames'/f'{index:04d}'
        seal = read(folder/'FRAME_SEAL.json')
        if seal['id'] != frame['id']:
            raise ValueError('Train calibration original batch order mismatch')
        bundle_path = folder/'BOUNDARY_BUNDLE.jsonl.gz'
        bundle = next(rows(bundle_path)) if bundle_path.exists() else None
        models = compile_models(frame, config, bundle=bundle)
        model = models['ORIGINAL_WD']; c = model.ctx
        p = OLD._native_pose(dict(R_physical=frame['fallback']['R_physical'], centroid=frame['fallback']['centroid']), c)
        qxy, _, jq, _ = projected(model, p)
        consumed = set((bundle or {}).get('consumed_native_ids', []))
        revoked = {b['native_corner'] for b in c.boundary} | consumed
        js = []
        for m in c.measurements:
            if m['native_corner'] in revoked:
                continue
            cov = np.asarray(m['covariance_px2'])
            white = np.linalg.inv(np.linalg.cholesky(cov))
            reliability = float(m.get('fit_reliability_weight', 1.))
            residual = white@(qxy[m['native_corner']]-m['xy'])
            robust = 1/np.sqrt(1+float(residual@residual)/c.cfg['huber_scale']**2)
            js.extend(np.sqrt(reliability*robust)*white@jq[m['native_corner']])
            counts['complete_rows'] += 2
        for b in c.boundary:
            if b['native_corner'] in consumed:
                continue
            residual = [(np.asarray(line['normal_unit_xy'])@qxy[b['native_corner']]+line['offset_px'])/np.sqrt(line['normal_variance_px2']) for line in b['lines']]
            robust = 1/np.sqrt(1+float(np.dot(residual, residual))/c.cfg['huber_scale']**2)
            for line in b['lines']:
                js.append(np.sqrt(b['fit_reliability_weight']*robust)*np.asarray(line['normal_unit_xy'])@jq[b['native_corner']]/np.sqrt(line['normal_variance_px2']))
                counts['partial_rows'] += 1
        for group in F.fixed_groups(bundle):
            qs = group['queries']
            for query in qs:
                offset, qg = OB._intersection_offset_and_gradient(query, qxy, bundle['pixel_scale'])
                if offset is None:
                    counts['B_invalid_at_N0'] += 1
                    continue
                grid = np.asarray(query['offset_grid_base_px'])
                logp = np.asarray(query['offset_log_probability'])
                probability = np.asarray(query['offset_probability'])
                score_derivative = np.gradient(logp, grid)
                fisher = float(probability@(score_derivative**2))
                dJ = np.einsum('ni,nij->j', qg, jq)
                js.append(np.sqrt(group['group']['reliability_weight']*fisher/len(qs))*dJ)
                counts['B_profiles'] += 1
        counts['B_unsupported_profiles'] += sum(not qq['supported'] for gg in (bundle or {}).get('groups', []) for qq in gg['queries'])
        if js:
            j = np.asarray(js)
            curvature = 2*np.sum(j*j, axis=0)
            values.append([float(curvature[:3].mean()), float(curvature[3:].mean())])
        else:
            counts['no_information_frames'] += 1
        ids.append(frame['id'])
    if len(ids) != 768:
        raise ValueError('Exact original train768 required; validation/reference not read')
    a = np.asarray(values); medians = np.median(a, axis=0)
    result = dict(schema='B_LOCAL_TRAIN_ONLY_INFORMATION_V1', complete=True,
        rotation_precision=float(max(1., medians[0])), translation_precision=float(max(1., medians[1])),
        rule='median across train frames of mean block diagonal 2*J^T J; robust covariance-whitened complete/partial observations after B replacement; B discrete-profile Fisher information with original group reliability and fixed-query mean; lower bound unit precision',
        units='rotation radians; translation divided by fixed registered extent diagonal; energy is original B normalized energy',
        information_counts=counts, original_batch_manifest_sha256=sha(Path(original_batch)/'BATCH_SEAL.json'),
        train_frames=len(ids), information_frames=len(values), train_ids=ids,
        inputs_sha256=sha(inputs), original_config_sha256=sha(old_config_path),
        validation_fitted=False, real_fitted=False, target_or_reference_reads=0,
        strength_trials=1, elapsed_seconds=time.monotonic()-started)
    write(output, result); return result
