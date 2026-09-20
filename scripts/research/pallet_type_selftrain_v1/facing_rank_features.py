"""Calibrated parallel-edge features; no point edits, PnP or frame rejection."""
import numpy as np

X_EDGES = np.array([[0,1],[3,2],[4,5],[7,6]])
Z_EDGES = np.array([[0,4],[1,5],[2,6],[3,7]])


def camera_matrix(intrinsics):
    k = intrinsics
    out = np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]], float)
    if not np.isfinite(out).all() or min(out[0,0],out[1,1]) <= 0:
        raise ValueError('Invalid calibration')
    return out


def axis_direction(rays, edges):
    planes = np.cross(rays[edges[:,0]], rays[edges[:,1]])
    lengths = np.linalg.norm(planes, axis=1)
    if (lengths < 1e-10).any():
        raise ValueError('Coincident edge endpoints')
    planes /= lengths[:,None]
    _, singular, vt = np.linalg.svd(planes, full_matrices=False)
    if singular[1] < 1e-10:
        raise ValueError('Direction not constrained by independent edges')
    return vt[-1], float(singular[-1]/singular[1])


def describe(points, camera):
    q = np.asarray(points, float); k = np.asarray(camera, float)
    if q.shape != (9,2) or k.shape != (3,3):
        raise ValueError('Expected nine points and 3x3 camera')
    if not np.isfinite(k).all() or min(k[0,0],k[1,1]) <= 0 or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError('Invalid calibration')
    if not np.isfinite(q[:8]).all() or (q[:8] == -1).all(-1).any():
        return np.zeros(6, np.float32), dict(available=False, reason='invalid_points')
    ray = np.column_stack([q, np.ones(9)]) @ np.linalg.inv(k).T
    # The predicted center is not replaced by a GT center. Fallback is permutation invariant.
    center = ray[8] if np.isfinite(q[8]).all() and not (q[8] == -1).all() else ray[:8].mean(0)
    center = center/np.linalg.norm(center)
    try:
        dx, rx = axis_direction(ray, X_EDGES)
        dz, rz = axis_direction(ray, Z_EDGES)
    except (ValueError, np.linalg.LinAlgError) as e:
        return np.zeros(6, np.float32), dict(available=False, reason=str(e))
    # A cuboid side normal is parallel to the other horizontal axis.
    # Signs of homogeneous vanishing directions are intentionally irrelevant.
    front = abs(float(dz @ center)); side = abs(float(dx @ center))
    margin = front-side
    orth = abs(float(dx @ dz)); coherence = 1-max(rx,rz)
    x = np.array([-margin, -margin*coherence, -margin*(1-orth),
                  -margin*abs(margin), rx-rz, (rx-rz)*(1-orth)], np.float32)
    assert np.isfinite(x).all()
    return x, dict(available=True, front_facing=front, side_facing=side,
                  facing_margin=margin, axis_residual_x=rx, axis_residual_z=rz,
                  axis_nonorthogonality=orth)
