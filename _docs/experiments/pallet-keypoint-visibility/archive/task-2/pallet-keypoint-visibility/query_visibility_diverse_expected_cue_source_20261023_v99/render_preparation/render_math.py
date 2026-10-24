"""Deterministic photometric operators. No source readers, GT or learned model."""
import hashlib
import numpy as np

STAGES = ('BEFORE', 'COVER', 'SHAM')

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def fixed_seed(namespace, split, base_index):
    text = f'V99_{namespace}|{split}|{int(base_index)}'
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')

def composite_premultiplied_rgba(rgba, background):
    """Resolved opaque MSAA RGB is coverage-premultiplied; do not multiply twice.

    This combines display-code values, not a radiometrically linear-light claim.
    The generated EGL probe binds this actual framebuffer convention before RGB.
    """
    assert rgba.dtype == background.dtype == np.uint8
    assert rgba.shape == background.shape[:2] + (4,) and background.shape[2] == 3
    a = rgba[..., 3:4].astype(np.float64) / 255.
    out = rgba[..., :3].astype(np.float64) + (1. - a) * background.astype(np.float64)
    return np.floor(np.clip(out, 0, 255) + .5).astype(np.uint8)

def gl_pose_from_cv(camera):
    R = np.asarray(camera['R'], np.float64)
    eye = np.asarray(camera['eye_world'], np.float64)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) > .999999999
    pose = np.eye(4)
    pose[:3, :3] = R.T @ np.diag([1., -1., -1.])
    pose[:3, 3] = eye
    return pose

def gl_intrinsics_from_integer_camera(camera):
    """OpenGL viewport physical centres=i+.5; native array centres=i.

    Shift the principal point by +.5 only in the GL API, preserving the declared
    camera and its integer-centre K exactly. V97 symmetric PerspectiveCamera has
    this same physical principal point320 for the declared native319.5.
    """
    K=np.asarray(camera['K'],np.float64)
    return dict(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]+.5),cy=float(K[1,2]+.5),
                znear=float(camera['near']),zfar=float(camera['far']))

def affine_vertices(xyz, matrix_world_row):
    """USD row-vector affine, applied once to the original true geometry."""
    xyz = np.asarray(xyz, np.float64)
    M = np.asarray(matrix_world_row, np.float64)
    out = np.c_[xyz, np.ones(len(xyz))] @ M
    assert np.allclose(out[:, 3], 1., atol=1e-12)
    assert np.linalg.det(M[:3, :3]) > 0
    return out[:, :3]

def project(camera, xyz):
    c = np.asarray(xyz, np.float64) @ np.asarray(camera['R'], np.float64).T + np.asarray(camera['t'], np.float64)
    uvh = c @ np.asarray(camera['K'], np.float64).T
    return uvh[:, :2] / uvh[:, 2:], c[:, 2]

def check_complete_clip(camera, target_xyz, foreign_cover=None, foreign_sham=None):
    arrays = [np.asarray(target_xyz)]
    arrays.extend(np.asarray(a) for a in (foreign_cover, foreign_sham) if a is not None)
    xyz = np.concatenate(arrays)
    _, z = project(camera, xyz)
    near, far = float(camera['near']), float(camera['far'])
    assert np.isfinite(z).all() and near > 0 and near < float(z.min()) and float(z.max()) < far
    return dict(vertex_count=len(z), min_camera_z=float(z.min()), max_camera_z=float(z.max()),
                near=near, far=far, near_clipped_vertices=0, far_clipped_vertices=0)
