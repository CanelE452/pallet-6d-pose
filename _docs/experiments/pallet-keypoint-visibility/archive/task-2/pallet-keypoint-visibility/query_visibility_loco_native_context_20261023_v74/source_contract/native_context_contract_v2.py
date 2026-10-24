"""Source-only native-slot contract. Pure NumPy; no model, solver or file I/O.

The renderer's four candidate FRONTs are VERTICAL cuboid faces, not its top.
Stored renderer arrays are already camera-dynamic. Never re-permute them at
inference. The world/camera data below are GT-only audit inputs.
"""
import numpy as np

FIXED_ZUP_SIGNS = np.array([
    [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1],
    [-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]], dtype=np.int8)
VERTICAL_FIXED_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2]])
ALL_DYNAMIC_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2],
    [0, 1, 5, 4], [3, 2, 6, 7]])
NATIVE_NAMES = (
    'front_top_left', 'front_top_right', 'front_bottom_right',
    'front_bottom_left', 'rear_top_left', 'rear_top_right',
    'rear_bottom_right', 'rear_bottom_left', 'projected_3d_centroid')
PROXY_NAMES = ('D_CONVEX_UNBLOCKED_RAY_PROXY', 'E_EXTERNAL_RAY_GE_HALF_PROXY',
               'S_CONVEX_SELF_HIDDEN_PROXY', 'F_REFERENCE_OUTSIDE_PROXY',
               'U_MISSING_SOURCE_EVIDENCE')


def inverse_stored_permutation(dynamic, permutation):
    """perm[new_camera_dynamic_slot] = old_writer_fixed_slot."""
    a = np.asarray(dynamic)
    p = np.asarray(permutation)
    if p.shape != (8,) or sorted(p.tolist()) != list(range(8)) or a.shape[0] != 8:
        raise ValueError('Expected eight slots and a bijective dynamic-to-fixed map')
    fixed = np.empty_like(a)
    fixed[p] = a
    return fixed


def outward_normals(corners, faces):
    q = np.asarray(corners, np.float64)
    if q.shape != (8, 3) or not np.isfinite(q).all():
        raise ValueError('Finite eight world corners required')
    p = q[np.asarray(faces)]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    norm = np.linalg.norm(n, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError('Degenerate cuboid')
    n /= norm[:, None]
    n *= np.where(np.sum(n * (p.mean(1) - q.mean(0)), axis=1) < 0, -1, 1)[:, None]
    return n, p.mean(1)


def normal_facing_permutation(fixed_world, fixed_projected, camera_world):
    """Empirically verified renderer-equivalent rule, not recovered writer code.

    Camera direction originates at cuboid center. Choose max outward cosine
    among only four vertical sides. Then top/bottom uses world Z, and FRONT top
    endpoints use projected image X. Rear slots preserve each FRONT vertex
    correspondence across the depth edge; they are NOT independently X-sorted.
    Ties use the listed face order; writer tie policy remains uncertified.
    """
    q = np.asarray(fixed_world, np.float64)
    p = np.asarray(fixed_projected, np.float64)
    cam = np.asarray(camera_world, np.float64)
    if p.shape != (8, 2) or cam.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Eight projected points and camera position required')
    n, _ = outward_normals(q, VERTICAL_FIXED_FACES)
    ray = cam - q.mean(0)
    if np.linalg.norm(ray) < 1e-12:
        raise ValueError('Camera at cuboid center')
    cos = n @ ray / np.linalg.norm(ray)
    front = int(np.argmax(cos))
    def ordered(face):
        top = sorted(face, key=lambda i: q[i, 2], reverse=True)[:2]
        top = sorted(top, key=lambda i: p[i, 0])
        bottom = [i for i in face if i not in top]
        def paired(i):
            return min(bottom, key=lambda j: np.linalg.norm(q[i, :2] - q[j, :2]))
        return [top[0], top[1], paired(top[1]), paired(top[0])]
    front_slots = ordered(VERTICAL_FIXED_FACES[front])
    rear_indices = VERTICAL_FIXED_FACES[front ^ 1]
    rear_slots = [min(rear_indices, key=lambda j: np.linalg.norm(q[j] - q[i]))
                  for i in front_slots]
    perm = front_slots + rear_slots
    return np.asarray(perm, np.int8), cos


def convex_self_visible_proxy(dynamic_world, camera_world):
    """Any adjacent convex face camera-facing; not exact mesh visibility."""
    q = np.asarray(dynamic_world, np.float64)
    n, fc = outward_normals(q, ALL_DYNAMIC_FACES)
    facing = np.sum(n * (np.asarray(camera_world) - fc), axis=1) > 0
    visible = np.zeros(8, bool)
    for f in ALL_DYNAMIC_FACES[facing]:
        visible[f] = True
    return visible


def source_reference_proxy(projected8, camera_depth8, raw_hw, convex_visible8, fraction8):
    """GT-only weak source teacher, with F>S>E>D precedence, unknown=-1/4.

    Does not require the virtual cuboid point to be on the rendered mesh.
    Does not turn mask0 into semantic E/S and is not human semantic gold.
    """
    r = np.asarray(projected8, float); z = np.asarray(camera_depth8, float)
    v = np.asarray(convex_visible8, bool); f = np.asarray(fraction8, float)
    h, w = raw_hw
    if r.shape != (8, 2) or any(x.shape != (8,) for x in (z, v, f)):
        raise ValueError('Expected exactly eight reference slots')
    evidence = np.isfinite(r).all(1) & np.isfinite(z) & np.isfinite(f) & (f >= 0) & (f <= 1)
    inframe = evidence & (z > 0) & (r[:, 0] >= 0) & (r[:, 0] < w) & (r[:, 1] >= 0) & (r[:, 1] < h)
    state = np.full(8, 4, np.int8)
    state[evidence & ~inframe] = 3
    state[inframe & ~v] = 2
    state[inframe & v & (f >= .5)] = 1
    state[inframe & v & (f < .5)] = 0
    return state, inframe


def input_context_features(query8, native_index, box_xyxy, raw_hw):
    """Deployable 42-vector: slot onehot8, relative q8x2, valid8, box4,
    queryXY2 and spanXY2 and imageHW2. No GT coordinates/order permutation.

    The caller obtains slot k from the frozen initializer's channel, never
    from nearest GT correspondence. The full same-target query set is needed.
    """
    q = np.asarray(query8, np.float64); box = np.asarray(box_xyxy, np.float64)
    if q.shape != (8, 2) or box.shape != (4,) or native_index not in range(8):
        raise ValueError('Expected complete initializer slots0..7, k and predicted box')
    h, w = raw_hw
    finite = np.isfinite(q).all(1)
    valid = finite & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    safe = np.where(finite[:, None], q, 0)
    span = np.ptp(safe[finite], axis=0) if finite.any() else np.zeros(2)
    scale = max(float(span.max()), 1.)
    center = safe[native_index]
    relative = (safe - center) / scale
    relative[~finite] = 0
    geometry = np.r_[((box.reshape(2, 2) - center) / scale).ravel(),
                     center / np.asarray([max(w, 1), max(h, 1)]), span / scale,
                     np.asarray([h, w]) / scale]
    return np.r_[np.eye(8)[native_index], relative.ravel(), valid, geometry].astype(np.float32)


def source_canvas_to_raw(points_input, box_input, prepared_hw, input_hw, pad=100):
    """Inverse existing integer-pad LetterBox affine; no query adjustment."""
    h, w = prepared_hw; ih, iw = input_hw
    gain = min(640 / h, 640 / w)
    offset = np.asarray([round((iw - round(w * gain)) / 2 - .1),
                         round((ih - round(h * gain)) / 2 - .1)])
    if np.any(offset < 0):
        raise ValueError('Invalid stored letterbox frame')
    q = (np.asarray(points_input, np.float64) - offset) / gain - pad
    b = ((np.asarray(box_input, np.float64).reshape(2, 2) - offset) / gain - pad).reshape(4)
    return q, b, gain, offset


def source_pose_frames(cuboid_world, source_pose):
    """Source T acts on centered Y-up coordinates, not raw Z-up world.

    X_source = A @ (X_world-center), A=[[1,0,0],[0,0,1],[0,-1,0]].
    T_world_to_camera = T_source @ G_world_to_source. GT-only audit math.
    """
    q = np.asarray(cuboid_world, np.float64)
    T = np.asarray(source_pose, np.float64)
    if q.shape != (8, 3) or T.shape != (4, 4):
        raise ValueError('Eight world corners and source homogeneous pose needed')
    A = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
    G = np.eye(4); G[:3, :3] = A; G[:3, 3] = -A @ q.mean(0)
    return (q - q.mean(0)) @ A.T, T @ G
