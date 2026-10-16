"""Controlled CPU checks for the fixed line-only C2 representation.

No dataset, model, PnP, ray or real-frame calls occur here. NumPy, SciPy and
OpenCV are needed for synthetic projection and the unchanged local optimizer.
The real evaluation's inline output adapter is executed from its AST so its
fallback/projection contract is checked without running its dataset driver.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
# Old common requires this variable during import; toy tests read no source.
os.environ.setdefault('PALLET_SOURCE_ROOT', str(REPO))

import cv2
import numpy as np

A = importlib.import_module('scripts.research.pallet_kp_supervision_gate_20261010_v1.all_lines')
P = A.P
from scripts.research.pallet_observation_refiner_20261009_v1 import common as C
from scripts.research.pallet_observation_refiner_20261009_v1.inference import hidden_mask
from scripts.research.pallet_observation_refiner_20261009_v1.solver import cuboid, project

DOC = REPO / '_docs/experiments/pallet_kp_supervision_gate_20261010_v1'


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def binding(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


class ForbiddenCoordinates:
    """Sentinel proves an irrelevant coordinate payload cannot be consumed."""
    def __iter__(self):
        raise AssertionError('hidden or derived 2D coordinates were iterated')

    def __array__(self, *args, **kwargs):
        raise AssertionError('hidden or derived 2D coordinates became an array')


def fixture(edges=(0, 8, 9)):
    dims = np.array([1.1, .14, 1.3])
    K = np.array([[600., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
    R = cv2.Rodrigues(np.array([.51, .38, .13]))[0]
    t = np.array([.08, -.04, 3.2])
    q = project(cuboid(*dims), R, t, K)
    lines = []
    for edge in edges:
        a, b = P.EDGES[edge]
        direction = q[b] - q[a]
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        lines.append(dict(edge=edge, normal=normal.tolist(), offset=float(normal @ q[a])))
    initial = dict(available=True, R_cf=R, R_physical=R, centroid=t,
                   cf_extents=dims, projected=q, selected_hypothesis='REGISTRY_WD')
    return dict(corners=[], lines=lines), K, dims, R, t, q, initial


def normalized_rank(matrix):
    matrix = np.asarray(matrix, float)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-300)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular > singular[0] * P.POLICY['rank_threshold'])), singular


def observed_jacobian(observations, K, dims, R, t):
    _, jacobian = cv2.projectPoints(cuboid(*dims), cv2.Rodrigues(R)[0], t, K, np.zeros(5))
    J = jacobian[:, :6].reshape(8, 2, 6)
    return np.asarray([np.asarray(line['normal']) @ J[endpoint] / np.sqrt(2)
                       for line in observations['lines'] for endpoint in P.EDGES[line['edge']]])


@contextmanager
def optimizer_accounting(ledger):
    original = P.least_squares

    def instrument(function, x0, *args, **kwargs):
        record = dict(residual_dimensions=int(len(function(x0))), loss=kwargs.get('loss'),
                      f_scale=kwargs.get('f_scale'), max_nfev=kwargs.get('max_nfev'),
                      actual_residual_evaluations=1)
        ledger.append(record)

        def residual(*a, **kw):
            record['actual_residual_evaluations'] += 1
            return function(*a, **kw)

        answer = original(residual, x0, *args, **kwargs)
        record.update(reported_nfev=int(answer.nfev), converged=bool(answer.success))
        return answer

    P.least_squares = instrument
    try:
        yield
    finally:
        P.least_squares = original


def check_geometry():
    obs, K, dims, R, t, _, _ = fixture()
    plane = A.model_geometry(obs, K, dims, R, t)
    require(plane['available'] and plane['rank'] == 6, 'three coplanar distinct edge constraints must be locally full rank')
    require(plane['scalar_residuals'] == 6, 'one edge supplies exactly two normalized scalar residuals')
    require(plane['local_observability_only'] and not plane['global_unique_pose_proven'], 'rank must not assert global uniqueness')
    # Hold the reference pose's modeled normals fixed during differentiation.
    fixed = copy.deepcopy(obs)
    z = np.r_[cv2.Rodrigues(R)[0].ravel(), t]

    def residual(parameters):
        q = project(cuboid(*dims), cv2.Rodrigues(parameters[:3])[0], parameters[3:], K)
        return np.asarray([(np.asarray(line['normal']) @ q[endpoint] - line['offset']) / np.sqrt(2)
                           for line in fixed['lines'] for endpoint in P.EDGES[line['edge']]])

    step = 1e-5
    fd = np.column_stack([(residual(z + np.eye(6)[axis] * step) - residual(z - np.eye(6)[axis] * step)) / (2 * step)
                          for axis in range(6)])
    analytic = np.asarray(plane['analytic_jacobian'])
    max_difference = float(np.max(np.abs(fd - analytic)))
    require(np.allclose(fd, analytic, rtol=1e-7, atol=1e-6), 'analytic projection Jacobian differs from fixed-normal FD')
    parallel, K, dims, R, t, _, _ = fixture((0, 2, 4))
    exact = A.model_geometry(parallel, K, dims, R, t)
    require(not exact['available'] and exact['rank'] == 5, 'three parallel edges retain a translation null direction')
    perturbed = copy.deepcopy(parallel)
    for line, angle in zip(perturbed['lines'], (.001, -.002, .003)):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        line['normal'] = (rotation @ np.asarray(line['normal'])).tolist()
    artificial_rank, artificial_singular = normalized_rank(observed_jacobian(perturbed, K, dims, R, t))
    guarded = A.model_geometry(perturbed, K, dims, R, t)
    require(artificial_rank == 6, 'control must demonstrate noisy-normal artificial rank6')
    require(guarded['rank'] == 5 and not guarded['available'], 'modeled-normal gate must reject artificial rank6')
    require(guarded['analytic_jacobian'] == exact['analytic_jacobian'], 'structural Jacobian must be independent of observed normal noise')
    # Directly exercise the new wrapper's acceptance guard independently of
    # whether a particular noisy local optimization happens to converge.
    # This is an explicitly constructed old-optimizer return, not an actual fit.
    original_solve = P.solve
    P.solve = lambda *args, **kwargs: dict(available=True, R_cf=R, centroid=t,
                                         cf_extents=dims, reason='controlled_old_candidate')
    try:
        wrapped = A.solve(perturbed, K, dims, dict(available=False), dict(available=False), [])
    finally:
        P.solve = original_solve
    require(not wrapped['available'] and wrapped['reason'] == 'model_line_geometry_rank_deficient',
            'accepted old candidate must be rejected by modeled rank5 wrapper')
    require(wrapped['model_geometry_observability']['rank'] == 5, 'wrapper must record the rejection evidence')
    return dict(three_coplanar_edges=plane, three_parallel_edges=exact,
                noisy_observed_normal_rank=artificial_rank, noisy_observed_normal_singular_values=artificial_singular,
                finite_difference_step=step, analytic_FD_max_absolute_difference=max_difference,
                controlled_old_candidate_rejected_by_wrapper=True,
                controlled_old_candidate_is_actual_optimizer_fit=False,
                four_factors_not_required_or_sufficient=True)


def check_unique_edges_and_policy():
    obs, K, dims, R, t, _, initial = fixture()
    duplicate = copy.deepcopy(obs)
    duplicate['lines'].append(copy.deepcopy(duplicate['lines'][0]))
    rejected = []
    for label, callback in (
        ('model_geometry', lambda: A.model_geometry(duplicate, K, dims, R, t)),
        ('solve', lambda: A.solve(duplicate, K, dims, initial, dict(available=False), [])),
    ):
        try:
            callback()
        except AssertionError:
            rejected.append(label)
    require(len(rejected) == 2, 'duplicate semantic edge must be rejected before optimization')
    keys = ('loss', 'f_scale_px', 'max_nfev', 'point_weight', 'line_weight', 'initial_pose_prior', 'rank_threshold')
    require(all(A.POLICY[key] == P.POLICY[key] for key in keys), 'representation repair must retain numerical policy')
    return dict(duplicate_rejected_by=rejected, unchanged_numeric_policy={key: P.POLICY[key] for key in keys})


def check_local_fit(ledger):
    obs, K, dims, R, t, q, initial = fixture()
    poisoned = dict(corners=ForbiddenCoordinates(), lines=obs['lines'])
    initial['raw_hidden_image_points'] = ForbiddenCoordinates()
    before = len(ledger)
    with C.no_truth_reads():
        answer = A.solve(poisoned, K, dims, initial, dict(available=False), [0, 3, 5, 6])
    require(answer['available'], 'exact local line fit must be available')
    require(answer['derived_corner_factors'] == 0 and answer['point_ids'] == [], 'derived corners must not enter the fit')
    require(answer['line_edges'] == [0, 8, 9] and answer['consumed_edges'] == [], 'all selected edges must survive exactly once')
    require(not answer['same_edge_point_line_double_count'] and not answer['initial_pose_prior'], 'no double count or residual prior')
    require(np.max(np.abs(np.asarray(answer['projected']) - q)) < 1e-6, 'exact fit must preserve exact projection')
    require(len(ledger) == before + 1 and ledger[-1]['residual_dimensions'] == 6, 'three lines must yield six scalars and one start')
    require(ledger[-1]['loss'] == 'soft_l1' and ledger[-1]['f_scale'] == 8 and ledger[-1]['max_nfev'] == 50,
            'optimizer settings changed')
    # A small local perturbation checks the unchanged optimizer's actual recovery.
    perturbed_start = dict(initial, centroid=t + np.array([.006, -.004, .008]))
    recovered = A.solve(obs, K, dims, perturbed_start, dict(available=False), [])
    require(recovered['available'], 'small local perturbation should be recovered in the full-rank control')
    require(np.max(np.abs(np.asarray(recovered['projected']) - q)) < 1e-4, 'local recovery projection is inaccurate')
    empty = A.solve(dict(corners=[], lines=[]), K, dims, initial, dict(available=False), [])
    no_start = A.solve(obs, K, dims, dict(available=False), dict(available=False), [])
    require(not empty['available'] and empty['reason'] == 'no_independent_factors', 'empty observations must remain unavailable')
    require(not no_start['available'] and no_start['reason'] == 'no_local_initialization', 'no initializer must not manufacture a pose')
    parallel, K, dims, R, t, _, initial = fixture((0, 2, 4))
    # Deliberately turn the observed-normal rank into six while preserving all
    # lines through their original edge midpoint. The model gate must reject it.
    projected = project(cuboid(*dims), R, t, K)
    for line, angle in zip(parallel['lines'], (.001, -.002, .003)):
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        n = rot @ np.asarray(line['normal'])
        midpoint = projected[list(P.EDGES[line['edge']])].mean(0)
        line.update(normal=n.tolist(), offset=float(n @ midpoint))
    guarded = A.solve(parallel, K, dims, initial, dict(available=False), [])
    require(not guarded['available'], 'noisy parallel-edge fit must never be a new pose')
    if 'model_geometry_observability' in guarded:
        require(guarded['model_geometry_observability']['rank'] == 5, 'final modeled geometry should retain exact rank5')
    return dict(exact_fit_available=True, locally_perturbed_fit_available=True,
                hidden_and_corner_coordinate_sentinels_unread=True,
                empty_reason=empty['reason'], no_start_reason=no_start['reason'],
                parallel_fit_available=guarded['available'], parallel_fit_reason=guarded['reason'],
                exact_projection_max_error=float(np.max(np.abs(np.asarray(answer['projected']) - q))),
                recovered_projection_max_error=float(np.max(np.abs(np.asarray(recovered['projected']) - q))))


def adapter_code():
    """Extract the actual sealed-output adapter, never the driver's IO/loops."""
    path = Path(A.__file__).with_name('evaluate_lines.py')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    loop = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple)
                and [name.id for name in node.target.elts if isinstance(name, ast.Name)] == ['i', 'observation'])
    start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'new' for target in node.targets))
    end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'result' for target in node.targets))
    selected = ast.Module(body=copy.deepcopy(loop.body[start:end + 1]), type_ignores=[])
    return compile(ast.fix_missing_locations(selected), str(path) + ':output_adapter', 'exec'), path


def check_actual_output_adapter():
    compiled, path = adapter_code()
    obs, _, _, _, _, q, initial = fixture()
    obs['raw_logits_sha256'] = 'controlled_fixture_no_logits'
    obs['corners'] = [dict(id=0, xy=(q[0] + [2., 1.]).tolist(), edges=[0, 8]),
                      dict(id=1, xy=(q[1] + [-1., 3.]).tolist(), edges=[0, 9])]
    original = np.vstack([q, [320., 240.]])
    H = [0, 3, 5]

    def apply(available, init, original_points):
        answer = dict(initial, available=available, selected_line_edges=[0, 8, 9])
        scope = dict(np=np, C=C, hidden_mask=hidden_mask, answer=answer,
                     init=init, H=H, original=original_points.copy(), observation=obs, diagnostic={})
        exec(compiled, scope)
        return scope['result']

    new = apply(True, initial, original)
    altered = original.copy()
    altered[H] += 1000.
    other = apply(True, initial, altered)
    require(new['output_status'] == 'NEW_POSE' and new['new_pose_estimated'] and not new['fallback_used'], 'new-pose status inconsistent')
    require(np.array_equal(new['native_points'][H], q[H]), 'hidden output must equal final actual projection')
    require(np.array_equal(other['native_points'][H], q[H]), 'hidden initial2D must not affect final hidden projection')
    require(np.array_equal(new['native_points'][8], original[8]), 'center must remain unchanged')
    require(np.array_equal(new['native_points'][1], obs['corners'][1]['xy']), 'selected visible output corner must retain observation')
    require(new['solver']['fit_input_ids'] == [] and new['solver']['final_inliers'] == [], 'line-only path must not invent point inliers')
    require(not new['reprojections_reused_as_observations'] and not new['initial_or_hidden_image_points_used_in_fit'], 'no circular or hidden fit inputs')
    fallback = apply(False, initial, altered)
    require(fallback['output_status'] == 'BASELINE_FALLBACK' and fallback['fallback_used'] and not fallback['new_pose_estimated'], 'baseline fallback status inconsistent')
    require(np.array_equal(fallback['native_points'], altered), 'fallback must return entire original coordinate output')
    require(fallback['actual_pose'] is initial and not fallback['hidden_reprojected'], 'fallback cannot claim newly fitted/reprojected pose')
    failure = apply(False, dict(available=False), original)
    require(failure['output_status'] == 'POSE_FAILURE' and failure['no_pose'] and not failure['fallback_used'], 'complete failure must be distinct from fallback')
    return dict(evaluation_adapter=binding(path), adapter_execution='actual AST statements, no driver execution',
                new_pose_hidden_projection=True, hidden_initial2D_independence=True,
                full_baseline_fallback=True, complete_failure_distinct=True,
                center_preserved=True, point_inliers_empty=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DOC / 'ALL_LINES_CHECKS.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Refusing to overwrite an existing artifact: ' + str(output))
    if output.suffix != '.json':
        parser.error('Output must be a new JSON receipt')
    cv2.setNumThreads(1)
    ledger = []
    checks = []
    with optimizer_accounting(ledger):
        for name, callback in (
            ('modeled_observability_and_analytic_derivative', check_geometry),
            ('unique_edge_and_unchanged_policy', check_unique_edges_and_policy),
            ('local_fit_no_corner_or_hidden2D_and_unavailable', lambda: check_local_fit(ledger)),
            ('actual_output_projection_fallback_failure_adapter', check_actual_output_adapter),
        ):
            try:
                details = callback()
                checks.append(dict(name=name, passed=True, details=details))
            except Exception as error:
                checks.append(dict(name=name, passed=False, error=type(error).__name__ + ': ' + str(error)))
    report = dict(schema='fixed_original_image_role_all_lines_toy_checks_v1',
                  passed=all(check['passed'] for check in checks), checks=checks,
                  code=[binding(Path(__file__)), binding(Path(A.__file__)), binding(Path(P.__file__))],
                  execution=dict(controlled_CPU_local_optimizer_starts=len(ledger), optimizer_calls=ledger,
                                 controlled_CPU_residual_evaluations=sum(row['actual_residual_evaluations'] for row in ledger),
                                 new_real_frame_fits=0, new_PnP_calls=0, new_model_forwards=0,
                                 new_training_updates=0, new_mesh_rays=0, GT_asset_reads=0),
                  limits=['Synthetic local correctness controls do not prove global unique pose.',
                          'No classifier accuracy, physical boundary veracity or real pose improvement is asserted.'],
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(finite(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=report['passed'], groups=len(checks),
                          controlled_optimizer_starts=len(ledger), receipt=str(output)), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
