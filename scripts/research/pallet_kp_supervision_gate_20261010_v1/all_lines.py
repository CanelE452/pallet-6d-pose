"""Duplicate-free line representation on unchanged sealed IMAGE_ROLE data.

The original local optimizer and scale are retained. A geometry Jacobian at
the final pose also checks observability independently of noisy line normals.
Initial/hidden image coordinates are never line factors or residual priors.
"""
import cv2
import numpy as np
from ..pallet_observation_refiner_20261009_v1 import point_line as P
from ..pallet_observation_refiner_20261009_v1.solver import cuboid

POLICY = dict(P.POLICY, representation='one line per selected edge; zero derived-corner factors',
              model_geometry_observability=True, physical_line_veracity_guaranteed=False)


def model_geometry(observations, K, dimensions, R, t):
    """Frozen modeled normals remove artificial rank caused by noisy TLS normals."""
    edges = [int(line['edge']) for line in observations['lines']]
    assert len(edges) == len(set(edges)), 'Each semantic edge is used once'
    X = cuboid(*np.asarray(dimensions, float))
    rvec = cv2.Rodrigues(np.asarray(R, float))[0]
    projected, jacobian = cv2.projectPoints(X, rvec, np.asarray(t, float),
                                         np.asarray(K, float), np.zeros(5))
    projected = projected.reshape(-1, 2)
    J = jacobian[:, :6].reshape(8, 2, 6)
    factors = []
    for edge in edges:
        a, b = P.EDGES[edge]
        direction = projected[b] - projected[a]
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length <= 1e-12:
            return dict(available=False, rank=0, reason='projected_model_edge_degenerate', edges=edges)
        normal = np.array([-direction[1], direction[0]]) / length
        factors.extend([normal @ J[a] / np.sqrt(2), normal @ J[b] / np.sqrt(2)])
    matrix = np.asarray(factors, float).reshape(-1, 6)
    if not len(matrix):
        return dict(available=False, rank=0, reason='no_line_factors', edges=edges)
    column_scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-300)
    singular = np.linalg.svd(matrix / column_scale, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * P.POLICY['rank_threshold']))
    return dict(available=rank == 6, rank=rank, edges=edges,
                scalar_residuals=2 * len(edges), singular_values=singular.tolist(),
                condition_number=float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
                analytic_jacobian=matrix.tolist(), normalized_column_scale=column_scale.tolist(),
                rank_threshold=P.POLICY['rank_threshold'],
                normals='computed from final modeled projection, held fixed for derivative',
                local_observability_only=True, global_unique_pose_proven=False)


def solve(observations, K, xyz, initial, point_result, hidden):
    """The signature retains old local starts; observations' corners are ignored."""
    lines = observations['lines']
    assert len(lines) == len({line['edge'] for line in lines})
    answer = P.solve(dict(corners=[], lines=lines), K, xyz, initial, point_result, hidden)
    answer['representation_policy'] = POLICY
    answer['derived_corner_factors'] = 0
    answer['selected_line_edges'] = [line['edge'] for line in lines]
    answer['same_edge_point_line_double_count'] = False
    if answer.get('available'):
        geometric = model_geometry(dict(lines=lines), K, answer['cf_extents'],
                                   answer['R_cf'], answer['centroid'])
        answer['model_geometry_observability'] = geometric
        if not geometric['available']:
            answer['available'] = False
            answer['reason'] = 'model_line_geometry_rank_deficient'
    return answer
