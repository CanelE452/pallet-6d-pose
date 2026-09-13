"""GT-free adapter to the existing canonical MAIN selector and solver."""
import numpy as np
import cv2
from contracts import canonical_modules


def pose(points, camera, dimensions):
    empty = dict(pose_valid=False, axis_id=None, x_m=None, z_m=None,
                 yaw_rad=None, rotation=None, translation=None)
    if points is None:
        return empty
    evaluate, _, _ = canonical_modules()
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    points = np.asarray(points, np.float64)
    camera = np.asarray(camera, np.float64)
    long, short, height = dimensions
    usable = np.isfinite(points[:8]).all(axis=1)
    if usable.sum() < 6:
        return empty
    chosen = None
    try:
        selected = select_pnp_hypotheses(points, camera, dict(x=long,y=height,z=short), None)
        for h in selected.hypotheses:
            if h.name == selected.selected_hypothesis and h.success:
                chosen = 'CF_WIDTH' if abs(h.camera_facing_dimensions.as_dict()['width']-long)<1e-6 else 'CF_DEPTH'
        fits = {axis:evaluate.solve(evaluate.cuboid(a,height,b),points[:8],camera,usable)
                for axis,a,b in [('CF_WIDTH',long,short),('CF_DEPTH',short,long)]}
        if chosen is None or any(v is None for v in fits.values()):
            return empty
        rotation, translation, residual = fits[chosen]
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return empty
        return dict(pose_valid=True,axis_id=chosen,x_m=float(translation[0]),z_m=float(translation[2]),
            yaw_rad=float(np.arctan2(rotation[0,2],rotation[2,2])),rotation=rotation.tolist(),
            translation=translation.tolist(),reprojection_px=residual)
    except (cv2.error, ValueError, FloatingPointError):
        return empty


def pool_truth(points, camera, dimensions):
    """Caller supplies pool-only GT AFTER lock; no file access in this function."""
    _, build, _ = canonical_modules()
    long, short, height = dimensions
    points = np.asarray(points,np.float64)[:8]
    usable = np.isfinite(points).all(axis=1)
    assert usable.sum() >= 6
    fits = {axis:build.solve(build.cuboid(a,height,b),points,np.asarray(camera,np.float64),usable)
            for axis,a,b in [('CF_WIDTH',long,short),('CF_DEPTH',short,long)]}
    assert all(v is not None for v in fits.values())
    chosen = min(fits,key=lambda k:fits[k][2])
    r,t,error = fits[chosen]
    return dict(axis_id=chosen,rotation=r.tolist(),translation=t.tolist(),reprojection_px=error)
