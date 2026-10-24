"""Generated camera/triangle oracles only. No source asset, image, or model loader."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent

def orient(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])

def canonical_triangle(uv, z):
    uv, z = np.asarray(uv, float).copy(), np.asarray(z, float).copy()
    if orient(uv[0], uv[1], uv[2]) < 0:
        uv[[1, 2]], z[[1, 2]] = uv[[2, 1]], z[[2, 1]]
    return uv, z

def top_left(a, b):
    # Y-down raster, positive signed screen area, integer-centred pixels.
    return b[1] < a[1] or (b[1] == a[1] and b[0] > a[0])

def scalar_inside(uv, p):
    uv, _ = canonical_triangle(uv, [1, 1, 1])
    for i in range(3):
        a, b = uv[i], uv[(i+1) % 3]
        e = orient(a, b, p)
        if not (e > 0 or (e == 0 and top_left(a, b))):
            return False
    return True

def reciprocal_depth(uv, z, p):
    area = orient(uv[0], uv[1], uv[2])
    weights = np.array([orient(uv[1], uv[2], p), orient(uv[2], uv[0], p), orient(uv[0], uv[1], p)]) / area
    return 1.0 / float(np.sum(weights / z)), weights

def direct_ray_plane_depth(vertices, p, f, centre):
    # Independent 3D intersection: ray is t*((u-cx)/f,(v-cy)/f,1).
    ray = np.array([(p[0]-centre)/f, (p[1]-centre)/f, 1.0])
    normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
    return float(np.dot(normal, vertices[0]) / np.dot(normal, ray))

def camera(axis):
    forward = -np.asarray(axis, float)
    up = np.array([0., 0., 1.]) if abs(forward[1]) == 1 else np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])

def nearest_owner(candidates, eps=1e-9):
    # Exact/declared near-tie diagnostic, independent of submission order.
    best_depth = min(x[0] for x in candidates)
    ties = [x for x in candidates if abs(x[0]-best_depth) <= eps]
    return min(x[2] for x in ties), len({x[1] for x in ties}) > 1

def main():
    f, centre = 128/math.tan(math.radians(25)), 127.5
    uv = np.array([[110., 110.], [145., 110.], [110., 145.]])
    z = np.array([2., 6., 4.])
    vertices = np.column_stack(((uv-centre)*z[:, None]/f, z))
    depths = []
    for p in [[120., 120.], [121., 115.], [112., 130.]]:
        reciprocal, weights = reciprocal_depth(uv, z, p)
        direct = direct_ray_plane_depth(vertices, p, f, centre)
        assert abs(reciprocal-direct) < 2e-14
        assert abs(float(weights @ z)-direct) > .1
        depths.append(dict(pixel=p, reciprocal_depth=reciprocal, direct_ray_depth=direct))
    square = [np.array([[1., 1.], [5., 1.], [5., 5.]]), np.array([[1., 1.], [5., 5.], [1., 5.]])]
    coverage = []
    for y in range(0, 7):
        for x in range(0, 7):
            count = sum(scalar_inside(t, [x, y]) for t in square)
            assert count == int(1 <= x < 5 and 1 <= y < 5)
            if count:
                coverage.append([x, y])
            for t in square:
                assert scalar_inside(t, [x, y]) == scalar_inside(t[[0, 2, 1]], [x, y])
    R = 2.0
    distance, near = 1.3*R/math.sin(math.radians(25)), R*1e-4
    # A generated bounding-sphere sample, not actual mesh coordinates.
    dirs = np.array([[x, y, z] for x in [-1., 0., 1.] for y in [-1., 0., 1.] for z in [-1., 0., 1.] if x or y or z])
    points = R*dirs/np.linalg.norm(dirs, axis=1)[:, None]
    camera_results = []
    for axis in np.concatenate([np.eye(3), -np.eye(3)]):
        basis = camera(axis)
        assert np.max(np.abs(basis @ basis.T-np.eye(3))) == 0
        assert abs(np.linalg.det(basis)-1) < 1e-14
        cam = (points-distance*axis) @ basis.T
        assert np.min(cam[:, 2]) > near
        projection = centre+f*cam[:, :2]/cam[:, 2:]
        assert np.min(projection) > -.5 and np.max(projection) < 255.5
        camera_results.append(dict(axis=axis.tolist(), min_Z=float(np.min(cam[:, 2])), min_uv=float(projection.min()), max_uv=float(projection.max())))
    assert .5*(distance+R) < distance-R
    for permutation in [[(4., 2, 20), (2., 1, 10)], [(2., 1, 10), (4., 2, 20)]]:
        assert nearest_owner(permutation) == (10, False)
    assert nearest_owner([(2., 1, 10), (2., 2, 5)]) == (5, True)
    assert nearest_owner([(2., 1, 10), (2., 1, 5)]) == (5, False)
    # PV base-copy reuse: an opaque foreground can remove but cannot create target ownership.
    base_owner = np.array([[1, 2], [-1, 1]])
    base_depth = np.array([[3., 4.], [np.inf, 3.]])
    owner, depth = base_owner.copy(), base_depth.copy()
    foreground = np.array([[True, True], [True, False]])
    front = foreground & (1. < depth)
    owner[front], depth[front] = 99, 1.
    assert np.all((owner == 1) <= (base_owner == 1))
    assert np.array_equal(base_owner, [[1, 2], [-1, 1]])
    code = Path(__file__)
    result = dict(completed=True, PASS=True, scope='Independent generated-only mathematical fixtures, not actual source rendering or production-code validation.',
                  code=dict(path=str(code), sha256=hashlib.sha256(code.read_bytes()).hexdigest()),
                  reciprocal_vs_direct_ray=depths, shared_edge_pixels=len(coverage), no_shared_edge_double_hits_or_cracks=True,
                  two_sided_winding_reversal_same_coverage=True, six_camera_results=camera_results,
                  all_positive_near_and_foreign_front_bound=True, depth_tie_order_independent_and_owner_U=True,
                  base_buffer_reuse_target_visibility_monotone=True, actual_mesh_or_pixels_models_current_fit_PnP=0)
    output = ROOT/'GENERATED_SCALAR_ORACLE_TESTS.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(PASS=True, output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))

if __name__ == '__main__':
    main()
