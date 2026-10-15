"""Read-only RGB response audit for frozen local B/I line observations.

This diagnostic neither supplies pose inputs nor reads reference coordinates.
The selected examples are an exploratory inspection of already evaluated data.
All original observations, sources, masks and pose outputs remain unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import face_aware_partial_bi_observations_v4_13d as D
from . import visible_bi_global_edge_correction_v4_9c as G


ROOT = Path(__file__).resolve().parents[3]
REMAINDER = ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1'
OUTPUT = ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/edge_response_censoring_v4_17'
CASE_IDS = ('wood_night_01:031426', 'wood_night_01:030743',
            'wood_night_01:030101', 'wood_night_01:030219')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): result.update(chunk)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def read_rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f: yield json.loads(line)


def local_grid_audit(ray, point, projection, cfg):
    """Reconstruct chosen grid coordinates without inspecting hidden raw slots."""
    k, mate = ray['native_corner'], ray['native_neighbor']
    a = np.asarray(point['xy'], float)
    guide = np.asarray(projection[mate], float) - np.asarray(projection[k], float)
    direction = guide / np.linalg.norm(guide)
    normal = np.array([-direction[1], direction[0]])
    observed = np.asarray(ray['observed_points'], float)
    offsets = (observed-a) @ normal - ray['hypothesis_source_shift_px']
    bound = cfg['global_hypothesis_probe_px']
    hits = np.isclose(np.abs(offsets), bound, atol=1e-7, rtol=0)
    return dict(native_corner=k, native_neighbor=mate,
                initial_guide_direction_xy=direction.tolist(),
                initial_guide_normal_xy=normal.tolist(),
                local_selected_offsets_px=offsets.tolist(),
                local_window_boundary_hit_count=int(hits.sum()),
                local_window_boundary_hit_fraction=float(hits.mean()),
                point_count=len(observed),
                archived_RMS_px=ray['inlier_rms_px'],
                archived_angle_change_deg=ray['angle_change_deg'],
                archived_global_shift_px=ray['hypothesis_source_shift_px'],
                near_zero_RMS=bool(ray['inlier_rms_px'] < 1e-8),
                complete_point_created=False)


def response_audit(image, mask, ray, projection, cfg):
    """Read a wider response profile; retain all eligible peaks, no point adoption."""
    k, mate = ray['native_corner'], ray['native_neighbor']
    guide = np.asarray(projection[mate], float)-np.asarray(projection[k], float)
    guide /= np.linalg.norm(guide)
    normal = np.array([-guide[1], guide[0]])
    old = np.asarray(ray['observed_points'], float)
    # Twice the old local half-window, within the already used 43px search.
    offsets = np.arange(-2*cfg['global_hypothesis_probe_px'],
                        2*cfg['global_hypothesis_probe_px']+.01,
                        cfg['normal_step_px']/2)
    lab = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32), (3, 3), .8)
    gradients = (cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)/8.,
                 cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)/8.)
    xy = old[:, None, :]+offsets[None, :, None]*normal
    internal = ray['mask_context_kind'] == 'TARGET_TARGET_INTERNAL'
    trace, avoidance = D._guarded_trace(lab, gradients, mask, xy, normal, offsets,
                                       internal, cfg, np.asarray(projection[mate], float))
    context = max(cfg['RGB_probe_px'], cfg['boundary_context_px'])
    safe = np.linalg.norm(xy-np.asarray(projection[mate], float), axis=-1) >= 2*context
    mp, mm = trace['mask_plus'], trace['mask_minus']
    occupancy = ((mp >= cfg['minimum_inside']) & (mm >= cfg['minimum_inside']) if internal else
                 (np.maximum(mp, mm) >= cfg['minimum_inside']) &
                 (np.minimum(mp, mm) <= cfg['maximum_outside']) &
                 (np.abs(mp-mm) >= cfg['minimum_transition']))
    h, w = mask.shape
    plus, minus = xy+normal*context, xy-normal*context
    inframe = ((plus[..., 0] >= 0) & (plus[..., 0] < w-1) &
               (plus[..., 1] >= 0) & (plus[..., 1] < h-1) &
               (minus[..., 0] >= 0) & (minus[..., 0] < w-1) &
               (minus[..., 1] >= 0) & (minus[..., 1] < h-1))
    valid = (safe & inframe & occupancy &
             (trace['energy'] >= cfg['minimum_gradient']) &
             (trace['alignment'] >= cfg['minimum_gradient_alignment']) &
             (trace['contrast'] >= cfg['minimum_color_transition']))
    response = trace['energy']*trace['alignment']
    scores = np.where(valid, response-.05*np.abs(offsets)[None, :], -np.inf)
    best = np.argmax(scores, axis=1)
    keep = np.isfinite(scores[np.arange(len(old)), best])
    zero = int(np.flatnonzero(np.isclose(offsets, 0))[0])
    records = []
    for i in range(len(old)):
        maximum = float(response[i, best[i]]) if keep[i] else None
        # A descriptive response plateau, never a confidence interval or gate.
        plateau = valid[i] & (response[i] >= .95*maximum) if maximum is not None else np.zeros(len(offsets), bool)
        records.append(dict(sample=i, old_xy=old[i].tolist(),
            response_energy_alignment=response[i].tolist(), eligible=valid[i].tolist(),
            selected_relative_offset_px=float(offsets[best[i]]) if keep[i] else None,
            selected_xy=xy[i, best[i]].tolist() if keep[i] else None,
            selected_energy=float(trace['energy'][i, best[i]]) if keep[i] else None,
            old_energy=float(trace['energy'][i, zero]),
            descriptive_95percent_peak_response_offsets_px=offsets[plateau].tolist()))
    selected = offsets[best[keep]]
    return dict(native_corner=k, native_neighbor=mate,
        offsets_relative_to_archived_line_px=offsets.tolist(), samples=records,
        supported_samples=int(keep.sum()),
        median_peak_displacement_px=float(np.median(selected)) if len(selected) else None,
        median_old_energy=float(np.median(trace['energy'][keep, zero])) if keep.any() else None,
        median_peak_energy=float(np.median(trace['energy'][np.arange(len(old))[keep], best[keep]])) if keep.any() else None,
        protected_unknown_mate_query_audit=avoidance,
        complete_point_created=False, pose_solver_called=False,
        response_is_conditional_on_guide_and_mask=True, predictive_coverage_claim=False)


def figure(image, ray, response, path, identifier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old = np.asarray(ray['observed_points'], float)
    observed = np.asarray([r['selected_xy'] for r in response['samples'] if r['selected_xy'] is not None])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.plot(old[:, 0], old[:, 1], 'o-', color='#ff9d00', ms=3, lw=1, label='Archived grid-selected points')
    if len(observed): ax.scatter(observed[:, 0], observed[:, 1], s=18, color='#00d4ff', label='Wider actual RGB response peaks')
    allxy = np.vstack([old, observed]) if len(observed) else old
    lo, hi = allxy.min(axis=0)-8, allxy.max(axis=0)+8
    ax.set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), title=f'{identifier} edge {ray["native_corner"]}-{ray["native_neighbor"]}')
    ax.legend(fontsize=8); ax.set_aspect('equal')
    offsets = response['offsets_relative_to_archived_line_px']
    for i in sorted(set([0, len(response['samples'])//2, len(response['samples'])-1])):
        rec = response['samples'][i]
        axes[1].plot(offsets, rec['response_energy_alignment'], label=f'sample {i}')
        eligible = np.asarray(rec['eligible'], bool)
        axes[1].scatter(np.asarray(offsets)[eligible], np.asarray(rec['response_energy_alignment'])[eligible], s=8)
    axes[1].axvline(0, color='#ff9d00', ls='--', label='Archived line')
    axes[1].set(xlabel='Normal displacement from archived line (px)', ylabel='Actual RGB gradient response', title='Points are response samples; no pose or KP adoption')
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior diagnostic output')
    source_obs = REMAINDER/'fresh_observations_v4_14/OBSERVATIONS.jsonl.gz'
    source_pred = REMAINDER/'pose_evaluation_v4_14/PREDICTIONS.jsonl.gz'
    before = [binding(p) for p in [Path(__file__), Path(D.__file__), Path(G.__file__), source_obs, source_pred]]
    pred = read_rows(source_pred)
    inventory = []
    selected = {}
    for row in pred:
        fo = row['partial_global_outline']['face_observation']
        points = {p['native_corner']: p for p in fo['visible_measurements']}
        geometry = row['partial_global_outline']['initial_geometry']
        projection = geometry['initial_projection_xy']
        hidden = geometry['initial_self_hidden']
        for ray in fo['edge_records']:
            if not ray.get('supported') or ray.get('whole_edge_search'): continue
            k, mate = ray['native_corner'], ray['native_neighbor']
            if hidden[k] or hidden[mate] or row['states'][k] not in ('B', 'I') or row['states'][mate] == 'S':
                raise AssertionError('Frozen local ray violates state-first gates')
            inventory.append(dict(id=row['id'], **local_grid_audit(ray, points[k], projection, fo['configuration'])))
        if row['id'] in CASE_IDS: selected[row['id']] = row
    obs = {r['id']: r for r in read_rows(source_obs) if r['id'] in CASE_IDS}
    args.output.mkdir(parents=True)
    results = []
    for identifier in CASE_IDS:
        row, original = selected[identifier], obs[identifier]
        image_path = ROOT/row['image']
        if sha(image_path) != row['image_sha256']: raise ValueError('Image binding mismatch')
        image = cv2.imread(str(image_path)); mask_path = REMAINDER/'fresh_observations_v4_14/masks'/original['mask_evidence']['mask_npz']
        if sha(mask_path) != original['mask_evidence']['mask_file_sha256']: raise ValueError('Mask binding mismatch')
        with np.load(mask_path) as data: mask = data['pallet_mask_clean'].copy()
        if image is None or mask.shape != image.shape[:2]: raise ValueError('Actual image/mask canvas mismatch')
        fo = row['partial_global_outline']['face_observation']
        geometry = row['partial_global_outline']['initial_geometry']
        for ray in fo['edge_records']:
            if not ray.get('supported') or ray.get('whole_edge_search'): continue
            response = response_audit(image, mask, ray, geometry['initial_projection_xy'], fo['configuration'])
            stem = identifier.replace(':', '__')+f'__edge{ray["native_corner"]}_{ray["native_neighbor"]}'
            figure(image, ray, response, args.output/(stem+'.png'), identifier)
            results.append(dict(id=identifier, image=binding(image_path), mask=binding(mask_path), figure=stem+'.png', **response))
    summary = dict(local_supported_rays=len(inventory), near_zero_RMS_rays=sum(x['near_zero_RMS'] for x in inventory),
                   all_samples_at_window_boundary=sum(x['local_window_boundary_hit_count']==x['point_count'] for x in inventory),
                   any_sample_at_window_boundary=sum(x['local_window_boundary_hit_count']>0 for x in inventory))
    report = dict(schema='EDGE_RESPONSE_CENSORING_DIAGNOSTIC_V4_17', source_bindings=before,
        sample_selection='Exploratory already-evaluated failure and successful-control cases; no thresholds fitted.',
        production_changes=0, NN_forwards=0, pose_fits=0, reference_coordinate_reads=0,
        summary=summary, inventory=inventory, actual_RGB_response_diagnostics=results,
        interpretation='Zero TLS residual can follow identical search-window boundary offsets; it is not independent angular precision. The wider profiles are diagnostics, not adopted points or probability intervals.')
    (args.output/'DIAGNOSTIC.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if before != [binding(ROOT/x['path']) for x in before]: raise RuntimeError('Bound input changed during diagnostic')
    assets = [binding(p) for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output/'DIAGNOSTIC_SEAL.json').write_text(json.dumps(dict(schema='EDGE_RESPONSE_CENSORING_SEAL_V4_17', source_bindings=before, artifacts=assets), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), summary=summary, figures=len(results))))


if __name__ == '__main__': main()
