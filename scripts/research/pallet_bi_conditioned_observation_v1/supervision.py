"""Separate synthetic virtual-corner quality targets for actual applied XY.

The physical cuboid vertices are authenticated by the P0 audit. One complete
C1/C2 correspondence is chosen using all available original N0 queries. This
module does not certify material edges and cannot read real references.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SIGMA_REFERENCE_PX = 2.
SCHEMA = 'BI_LOCAL_APPLIED_CORNER_QUALITY_TARGETS_V1'


def _array(value, shape):
    try:
        a = np.asarray(value, float)
    except (TypeError, ValueError):
        return None
    return a if a.shape == shape and np.isfinite(a).all() else None


def _project(body, R, t, K):
    camera = body @ R.T + t
    homogeneous = camera @ K.T
    good = np.isfinite(homogeneous).all(1) & (camera[:, 2] > 0)
    result = np.full((8, 2), np.nan)
    result[good] = homogeneous[good, :2] / homogeneous[good, 2:]
    good &= np.isfinite(result).all(1)
    return result, good


def align_reference(upstream, gt):
    """Preserve original renderer R; choose a single whole-object Y symmetry."""
    missing = dict(available=False, reason='MISSING_REFERENCE', chosen_symmetry_index=None,
        symmetry_costs=[], aligned_reference_xy8=[None]*8, reference_valid8=[False]*8,
        original_renderer_R=None, raw_renderer_R_modified=False,
        target_kind='VIRTUAL_REGISTERED_CUBOID_CORNER_REFERENCE')
    if not isinstance(gt, dict):
        return missing
    if gt.get('id') != upstream['id']:
        raise ValueError('Synthetic target and upstream frame identity differ')
    if gt.get('split') != upstream.get('split'):
        raise ValueError('Synthetic target and upstream split differ')
    R = _array(gt.get('R_fixed'), (3, 3))
    t = _array(gt.get('t'), (3,))
    K = _array(gt.get('K'), (3, 3))
    if R is None or t is None or K is None:
        return missing
    order = gt.get('symmetry_order')
    if order not in (1, 2):
        raise ValueError('Only original per-asset C1/C2 is permitted')
    if not np.array_equal(K, np.asarray(upstream['K'])):
        raise ValueError('Synthetic target camera changed')
    pose = upstream['initial_pose']
    native = _array(upstream.get('native_xyz'), (8, 3))
    Rcf = _array(pose.get('R_cf'), (3, 3))
    Rphysical = _array(pose.get('R_physical'), (3, 3))
    points = _array(upstream.get('initial_points'), (9, 2))
    # Individual missing queries remain unavailable without making zero targets.
    if points is None:
        try:
            points = np.asarray(upstream['initial_points'], float)
        except (TypeError, ValueError):
            points = np.full((9, 2), np.nan)
    if native is None or Rcf is None or Rphysical is None or points.shape != (9, 2):
        raise ValueError('Valid prediction-native geometry and N0 query shape required')
    Q = Rcf.T @ Rphysical
    body = native @ Q
    raw = points[:8]
    valid_query = np.isfinite(raw).all(1) & ~np.all(raw == [-1., -1.], axis=1)
    symmetries = [np.eye(3)] if order == 1 else [np.eye(3), np.diag([-1., 1., -1.])]
    projections, validity, costs = [], [], []
    for index, symmetry in enumerate(symmetries):
        xy, good = _project(body, R @ symmetry, t, K)
        used = good & valid_query
        cost = float(np.sum((xy[used]-raw[used])**2)) if used.any() else None
        projections.append(xy); validity.append(good)
        costs.append(dict(symmetry_index=index, sum_squared_error_px2=cost,
            compared_native_ids=np.flatnonzero(used).tolist(), compared_queries=int(used.sum())))
    eligible = [i for i, record in enumerate(costs) if record['sum_squared_error_px2'] is not None]
    if not eligible:
        return dict(missing, reason='NO_FINITE_N0_ALIGNMENT_QUERIES', symmetry_costs=costs,
                    original_renderer_R=R.tolist())
    chosen = min(eligible, key=lambda i: (costs[i]['sum_squared_error_px2'], i))
    xy, good = projections[chosen], validity[chosen]
    return dict(available=True, reason='ONE_WHOLE_OBJECT_CORRESPONDENCE',
        chosen_symmetry_index=chosen, symmetry_costs=costs,
        chosen_symmetry_matrix=symmetries[chosen].tolist(),
        aligned_reference_xy8=[xy[k].tolist() if good[k] else None for k in range(8)],
        reference_valid8=good.tolist(), original_renderer_R=R.tolist(),
        raw_renderer_R_modified=False, physical_registration_Q=Q.tolist(),
        correspondence='single original C1/C2 whole-object minimum sum-squared error against all finite original N0 first-eight queries; identity exact tie',
        target_kind='VIRTUAL_REGISTERED_CUBOID_CORNER_REFERENCE')


def targets_for_frame(upstream, feature_row, gt):
    """Build target rows only; no training input or feature value is changed."""
    if feature_row.get('id', feature_row.get('frame_id')) != upstream['id']:
        raise ValueError('Feature and upstream frame identity differ')
    if feature_row.get('split', upstream.get('split')) != upstream.get('split'):
        raise ValueError('Feature split differs from original synthetic split')
    alignment = align_reference(upstream, gt)
    result = []
    seen = set()
    for corner in feature_row['corner_rows']:
        k = corner['native_corner']
        if type(k) is not int or not 0 <= k < 8 or k in seen:
            raise ValueError('Unique native corner IDs in [0,7] required')
        seen.add(k)
        ids, applied = list(corner['candidate_ids']), corner['applied_xy']
        if len(ids) != len(set(ids)) or len(ids) != len(applied):
            raise ValueError('Exact unique candidate IDs and applied coordinates required')
        values, validities, reasons = [], [], []
        fixed_partial = bool(corner.get('partial_replaces_full_likelihood', False))
        for point in applied:
            xy = _array(point, (2,))
            if fixed_partial:
                value, valid, reason = None, False, 'PARTIAL_PATH_FIXED'
            elif not alignment['available']:
                value, valid, reason = None, False, alignment['reason']
            elif not alignment['reference_valid8'][k]:
                value, valid, reason = None, False, 'MISSING_CORNER_REFERENCE'
            elif xy is None:
                value, valid, reason = None, False, 'NONFINITE_APPLIED_COORDINATE'
            else:
                ref = alignment['aligned_reference_xy8'][k]
                error = math.hypot(float(xy[0]-ref[0]), float(xy[1]-ref[1]))
                value = math.log1p(error / SIGMA_REFERENCE_PX)
                if not math.isfinite(value):
                    raise ValueError('Unrepresentable finite-coordinate target error')
                valid, reason = True, 'APPLIED_XY_VIRTUAL_CORNER_QUALITY'
            values.append(value); validities.append(valid); reasons.append(reason)
        result.append(dict(native_corner=k, candidate_ids=ids, target_log1p_error=values,
            valid_target=validities, target_reasons=reasons,
            partial_replaces_full_likelihood=fixed_partial))
    return dict(schema=SCHEMA, id=upstream['id'], split=upstream['split'],
        corner_rows=result, alignment=alignment, sigma_reference_px=SIGMA_REFERENCE_PX,
        reference_kind='VIRTUAL_REGISTERED_CUBOID_POSITION_QUALITY_NOT_PHYSICAL_EDGE_LABEL',
        frame_retained_in_denominator=True)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def build_targets(*, upstream, features, gt, p0, output):
    """Build the exact P0-bound 1003-row target ledger in a separate process."""
    started, cpu = time.monotonic(), time.process_time()
    upstream, features, gt, p0, output = map(Path, (upstream,features,gt,p0,output))
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(p0.read_text())
    if report['status'] != 'PASS' or str(gt.resolve()) != report['synthetic_targets']['source']:
        raise ValueError('Exact P0 synthetic target source required')
    if _sha(gt) != report['synthetic_targets']['sha256']:
        raise ValueError('Synthetic target bytes changed')
    frames, feature_frames, targets = list(_rows(upstream)), list(_rows(features)), list(_rows(gt))
    inventory = report['synthetic']['inventory']
    ordered = [r['id'] for r in inventory]
    if len(ordered) != 1003 or any([r['id'] for r in data] != ordered for data in (frames,feature_frames,targets)):
        raise ValueError('Exact original ordered synthetic1003 required')
    if Counter(r['split'] for r in frames) != Counter(train=768,validation=235):
        raise ValueError('Exact original synthetic768/235 split required')
    values = []
    counts = Counter()
    for frame, feature, target, item in zip(frames,feature_frames,targets,inventory):
        for key in ('id','partition','split','image','image_sha256','raw_hw','initial_pose','K','native_xyz'):
            if frame[key] != item[key]:
                raise ValueError('P0 input binding changed: '+key)
        row = targets_for_frame(frame,feature,target)
        values.append(row)
        counts['frames'] += 1
        counts['frames_with_valid_targets'] += any(any(c['valid_target']) for c in row['corner_rows'])
        for corner in row['corner_rows']:
            counts['candidate_targets'] += len(corner['candidate_ids'])
            counts['valid_candidate_targets'] += sum(corner['valid_target'])
            counts['fixed_partial_candidates'] += sum(r == 'PARTIAL_PATH_FIXED' for r in corner['target_reasons'])
    output.mkdir(parents=True)
    destination = output/'TARGETS.jsonl.gz'
    with destination.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            for row in values:
                stream.write((json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    seal = dict(schema='BI_LOCAL_CORNER_TARGET_SEAL_V1',complete=True,rows=1003,
        split_counts=dict(Counter(r['split'] for r in values)),counts=dict(counts),
        ordered_ids=ordered, targets_sha256=_sha(destination),
        bindings={str(p):_sha(p) for p in (upstream,features,gt,p0)},
        target_source='P0-authenticated original renderer R,t and registered physical vertices',
        correspondence='whole-object C1/C2 against all finite original N0 first-eight coordinates',
        fixed_partial_path_has_neural_targets=False, normalization_or_model_fitting=False,
        real_reference_reads=0, newly_added_real_supervision=0, source_sha256=_sha(__file__),
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu))
    (output/'TARGET_SEAL.json').write_text(json.dumps(seal,indent=2,allow_nan=False)+'\n')
    return seal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('upstream','features','gt','p0','output'):
        parser.add_argument('--'+field,required=True)
    args = parser.parse_args()
    result = build_targets(**vars(args))
    print(json.dumps(dict(complete=result['complete'],counts=result['counts'],cost=result['cost'])))
