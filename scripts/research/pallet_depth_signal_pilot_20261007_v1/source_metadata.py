"""Registered SOURCE256 native-coordinate references; no F or neural calls.

The physical renderer R is the existing reference.  Production F(source=True)
already applies its fixed Rx(pi) basis, so the reference is not rotated again.
2D references use original YOLO label bytes, not a new cuboid projection label.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from . import common as C
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M

_CACHE = None
_RECEIPT = None
MANIFEST = C.DOC / 'SOURCE_EVAL_MANIFEST.json'
SIDECAR = C.ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.json'


def _bytes_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(Path(path).relative_to(C.ROOT)),
                                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def load():
    """Return (metadata rows, physical pose references, original-label 2D refs)."""
    global _CACHE, _RECEIPT
    if _CACHE is not None:
        return _CACHE
    manifest, manifest_binding = _bytes_json(MANIFEST)
    records = manifest['rows']; ids = [r['id'] for r in records]
    assert manifest['status'] == 'PASS' and manifest['locked_before_student_fit']
    assert len(ids) == len(set(ids)) == 256 and all(r['partition'] == 'heldout' for r in records)
    assert len({r['scenario_id'] for r in records}) == 249
    C.verify(manifest['geometry_side_table'])
    geometry_path = C.ROOT / manifest['geometry_side_table']['path']
    with np.load(geometry_path, allow_pickle=False) as handle:
        geometry = {key: handle[key] for key in ('stems', 'K', 'dims', 'R', 't', 'pad', 'Xcf', 'match_err')}
    index = {str(fid): i for i, fid in enumerate(geometry['stems'])}
    assert len(index) == len(geometry['stems'])
    side, side_binding = _bytes_json(SIDECAR)
    wanted = set(ids)
    side = {r['frame_id']: r for r in side['records'] if r['frame_id'] in wanted}
    assert set(side) == wanted
    metadata, pose_truth, two_truth = [], {}, {}
    projection_errors = []; declared_corners = 0; rounding_bounds = []
    for record in records:
        fid = record['id']; i = index[fid]; s = side[fid]
        if s.get('dimension_source_file') == manifest['geometry_side_table']['path']:
            assert s['dimension_source_sha256'] == manifest['geometry_side_table']['sha256']
        xyz = geometry['dims'][i]
        np.testing.assert_array_equal(np.asarray(s['canonical_WDH'])[[0, 2, 1]], xyz)
        order = int(s['symmetry_order'])
        permutations = np.asarray(s['allowed_permutations'], dtype=np.int64)
        assert order in (1, 2, 4) and permutations.shape == (order, 9)
        assert np.array_equal(permutations[0], np.arange(9)) and np.all(permutations[:, 8] == 8)
        assert all(np.array_equal(np.sort(p), np.arange(9)) for p in permutations)
        pad = float(geometry['pad'][i]); assert pad == record['reflect_pad_px']
        h, w = map(int, record['raw_shape_hw']); ph, pw = map(int, record['prepared_shape_hw'])
        assert [ph, pw] == [h + 2 * pad, w + 2 * pad]
        fx, fy, cx, cy = geometry['K'][i]
        K = np.array([[fx, 0., cx - pad], [0., fy, cy - pad], [0., 0., 1.]])
        metadata.append(dict(id=fid, scenario_id=record['scenario_id'], session=record['scenario_id'],
            image=record['image'], hw=[h, w], raw_shape_hw=[h, w], prepared_shape_hw=[ph, pw],
            reflect_pad_px=int(pad), K=K.tolist(), xyz=xyz.tolist(), source=True,
            source_kind=record['source'], symmetry_order=order))
        R, t = geometry['R'][i], geometry['t'][i]
        pose_truth[fid] = dict(R=R.tolist(), t=t.tolist(), xyz=xyz.tolist(),
                              body_R=R.tolist(), body_xyz=xyz.tolist(), order=order)
        label = C.ROOT / record['label']['path']
        label_bytes = label.read_bytes()
        assert len(label_bytes) == record['label']['bytes'] and hashlib.sha256(label_bytes).hexdigest() == record['label']['sha256']
        entries = [np.fromstring(line, sep=' ', dtype=np.float64)
                   for line in label_bytes.decode().splitlines() if line.strip()]
        assert len(entries) == 1 and entries[0].shape == (32,), ('SOURCE single-object label schema', fid)
        target = entries[0]; points = target[5:].reshape(9, 3)
        gt = points[:, :2] * [pw, ph] - pad
        valid = (points[:, 2] > 0) & np.isfinite(points[:, :2]).all(-1) & ~np.all(points[:, :2] == -1, axis=-1)
        box_center = target[1:3] * [pw, ph] - pad
        box_size = target[3:5] * [pw, ph]
        box = np.r_[box_center - box_size / 2, box_center + box_size / 2]
        two_truth[fid] = dict(id=fid, gt=gt.tolist(), valid=valid.tolist(), box=box.tolist(),
                             permutations=permutations.tolist(), hw=[h, w],
                             visibility_semantics='original source supervision visibility>0; not direct physical visibility')
        # Xcf carries the exact camera-facing corner ordering in physical axes.
        camera = geometry['Xcf'][i] @ R.T + t
        assert np.isfinite(camera).all() and np.min(camera[:, 2]) > 0
        projected = (camera @ K.T)[:, :2] / camera[:, 2, None]
        errors = np.linalg.norm(projected[valid[:8]] - gt[:8][valid[:8]], axis=-1)
        projection_errors.extend(errors.tolist())
        rounding_bounds.extend([.5e-6 * np.hypot(pw, ph)] * len(errors))
        declared_corners += int(valid[:8].sum())
    assert [r['id'] for r in metadata] == ids and set(pose_truth) == set(two_truth) == set(ids)
    errors = np.asarray(projection_errors); bounds = np.asarray(rounding_bounds)
    _RECEIPT = dict(status='PASS', frames=256, scenarios=249,
        proper_group_counts=dict(Counter(r['symmetry_order'] for r in metadata)),
        source_kind_counts=dict(Counter(r['source_kind'] for r in metadata)),
        native_K_and_labels_subtract_exact_existing_pad=True,
        source_F_requires_source_true=True, reference_extra_Rx_pi=False,
        source_R_basis='Existing GEOMETRY_SIDETABLE.R physical Y-up reference; F(source=True) alone applies fixed Rx(pi)',
        points_2D='original bound single-object YOLO label; prepared normalized xy decoded and pad100 subtracted',
        symmetry='exact frame-ID joined approved sidecar C1/C2/C4, unknown asset remains C1',
        full_supervised_corner_denominator=declared_corners,
        renderer_Xcf_projection_original_label_residual_px=dict(n=len(errors),median=float(np.median(errors)),
            P90=float(np.quantile(errors,.9)),max=float(np.max(errors)),
            max_6_decimal_xy_rounding_bound_px=float(np.max(bounds)),
            corners_beyond_label_rounding_bound=int((errors > bounds + 1e-9).sum()),
            use='coordinate/basis diagnostic only; original labels remain evaluation reference, no new label generation'),
        inputs=[manifest_binding, manifest['geometry_side_table'], side_binding],
        image_and_label_files_rewritten=0, new_F_calls=0, new_PnP_calls=0,new_NN_calls=0)
    _CACHE = (metadata, pose_truth, two_truth)
    return _CACHE


def validation_receipt():
    if _RECEIPT is None:
        load()
    return _RECEIPT


def measure_two_d(prediction, truth):
    """Existing whole-object metric, same highest-score candidate and IoU .5."""
    candidates = prediction['candidates']; selected = prediction['selected_index']
    assert selected == (int(np.argmax([r['score'] for r in candidates])) if candidates else None)
    candidate = None if selected is None else candidates[selected]
    matched = False
    if candidate is not None:
        a, b = np.asarray(candidate['box_xyxy'], float), np.asarray(truth['box'], float)
        intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
        union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - intersection
        matched = float(intersection / max(union, 1e-12)) >= .5
    points = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
    return dict(id=truth['id'], **M.measure(points, truth['gt'], truth['valid'], truth['permutations'],
                                         truth['hw'], matched=matched, detected=candidate is not None))
