"""Independent audit math, retained from A independent V182 audit; no IO or source operator import."""
import math
import numpy as np

def raw96(rgb, q, L):
    if q is None or L is None:
        return (np.zeros((96, 96, 3), np.uint8), np.zeros((96, 96), bool))
    pos = (np.arange(96, dtype=np.float64) + 0.5) / 96 - 0.5
    (x, y) = np.meshgrid(q[0] + pos * L, q[1] + pos * L)
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    fx = x - ix
    fy = y - iy
    acc = np.zeros((96, 96, 3), np.float64)
    (h, w) = rgb.shape[:2]
    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        xx = ix + dx
        yy = iy + dy
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        wt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
        acc[inside] += rgb[yy[inside], xx[inside]].astype(np.float64) * wt[inside, None]
    valid = (x >= -0.5) & (x < w - 0.5) & (y >= -0.5) & (y < h - 0.5)
    out = np.floor(np.clip(acc, 0, 255) + 0.5).astype(np.uint8)
    out[~valid] = 0
    return (out, valid)

def owner4(packet, q, L):
    if q is None or L is None:
        return np.full(5, np.nan)
    (h, w) = packet['part_id'].shape
    mass = [0.0] * 5
    for cy in [47, 48]:
        for cx in [47, 48]:
            x = q[0] + ((cx + 0.5) / 96 - 0.5) * L
            y = q[1] + ((cy + 0.5) / 96 - 0.5) * L
            if not (-0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5):
                mass[4] += 0.25
                continue
            i = math.floor(x)
            j = math.floor(y)
            fx = x - i
            fy = y - j
            for dx in [0, 1]:
                for dy in [0, 1]:
                    px = i + dx
                    py = j + dy
                    wt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * 0.25
                    if not (0 <= px < w and 0 <= py < h):
                        c = 4
                    elif packet['ambiguity'][py, px]:
                        c = 3
                    else:
                        p = int(packet['part_id'][py, px])
                        c = 2 if p < 0 else 1 if p == 999 else 0
                    mass[c] += wt
    return np.asarray(mass, np.float64)

def project_patch(g):
    """Independent ray-plane/2x2 solve, integer image centres; no source helper."""
    K = np.array(g['camera']['K'])
    R = np.array(g['camera']['R'])
    t = np.array(g['camera']['t'])
    inv = np.linalg.inv(K)
    dep = np.full((512, 512), np.inf)
    oof = False
    good = True
    for quad in g['quads_world']:
        f = np.asarray(quad) @ R.T + t
        if np.any(f[:, 2] <= g['camera'].get('near', 0.001)):
            good = False
            oof = True
            continue
        uv = f @ K.T
        uv = uv[:, :2] / uv[:, 2, None]
        oof |= bool(np.any(uv < 0) or np.any(uv > 511))
        e1 = f[1] - f[0]
        e2 = f[3] - f[0]
        n = np.cross(e1, e2)
        gram = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
        lo = np.maximum(np.floor(uv.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(uv.max(0)).astype(int), 511)
        for y in range(lo[1], hi[1] + 1):
            for x in range(lo[0], hi[0] + 1):
                ray = np.array([x, y, 1.0]) @ inv.T
                den = float(ray @ n)
                if den == 0:
                    continue
                hit = ray * (float(f[0] @ n) / den)
                if hit[2] <= g['camera'].get('near', 0.001):
                    continue
                d = hit - f[0]
                ab = np.linalg.solve(gram, [d @ e1, d @ e2])
                if np.all(ab >= 0) and np.all(ab <= 1):
                    dep[y, x] = min(dep[y, x], hit[2])
    return (dep, good, oof)

def end_classes(before, scene, depth, primary):
    cls = np.zeros((512, 512), np.int8)
    for (y, x) in np.argwhere(np.isfinite(depth)):
        e = float(depth[y, x])
        bp = int(before['part_id'][y, x])
        sp = int(scene['part_id'][y, x])
        bd = float(before['camera_depth'][y, x])
        sd = float(scene['camera_depth'][y, x])
        if before['ambiguity'][y, x] or scene['ambiguity'][y, x]:
            c = 4
        else:
            c = 5
            finite = math.isfinite(bd) and bd > 0
            tol = 64 * np.finfo(np.float64).eps * max(1.0, abs(bd) if finite else 0.0, abs(e))
            if bp >= 0 and bp != 999 and finite and (bd < e - tol):
                c = 3
            if bp == primary and finite and (abs(bd - e) <= tol):
                if sp == bp and before['global_triangle_id'][y, x] == scene['global_triangle_id'][y, x] and (bd == sd):
                    c = 1
                if sp == 999 and math.isfinite(sd) and (sd > 0) and (sd < bd):
                    c = 2
        cls[y, x] = c
    return cls

def mass96(cls, q, L):
    off = ((np.arange(96, dtype=np.float64) + 0.5) / 96 - 0.5) * L
    (xx, yy) = np.meshgrid(q[0] + off, q[1] + off)
    ix = np.floor(xx).astype(int)
    iy = np.floor(yy).astype(int)
    fx = xx - ix
    fy = yy - iy
    w = np.zeros((96, 96, 7))
    (a, b) = np.indices((96, 96))
    for dx in [0, 1]:
        for dy in [0, 1]:
            x = ix + dx
            y = iy + dy
            inside = (x >= 0) & (x < 512) & (y >= 0) & (y < 512)
            c = np.full((96, 96), 6)
            c[inside] = cls[y[inside], x[inside]]
            w[a, b, c] += (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
    invalid = (xx < -0.5) | (xx >= 511.5) | (yy < -0.5) | (yy >= 511.5)
    w[invalid] = 0
    w[invalid, 6] = 1
    return w.reshape(-1, 7).mean(0)

def target(m, available, good, oof):
    if not available or not good or oof:
        return [-1, -1]
    unknown = float(m[3] + m[4] + m[5])
    if m[1] + m[2] + unknown == 0:
        return [0, -1]
    return [1 if m[c] > 0 else -1 if unknown > 0 else 0 for c in [1, 2]]
