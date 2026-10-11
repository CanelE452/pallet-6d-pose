"""Registered W/D axis correction using NumPy projection only; no pose fitting.

The original A3 ratio experiment remains immutable.  This separate supplement
restores the registered canonical W/D order, demonstrates its exact geometric
effect, and does not assign the old wood pose scores to the corrected geometry.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import common as C


GEOMETRY = 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
WD = '_docs/experiments/pallet_wd_hypothesis_diag_20261010'
NAMES = ('long-face-front', 'short-face-front')
SAMPLE_SHA = 'a3f75e6521c5718d20b07a3de5ff7412fd29acba281804812cb06ff92de745d1'
# The first ratio is canonical W/D.  The second is the original code's D/W.
CONDITIONS = (
    ('plastic_ratio', 1.1 / 1.3, 1.3 / 1.1),
    ('wood_ratio', .8 / .59, .8 / .59),
    ('square', 1., 1.),
)


def digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def project(points, rotation, translation, camera):
    """Native exact projection, including the unchanged cuboid center."""
    cam = np.vstack([points, np.zeros((1, 3))]) @ rotation.T + translation
    homog = cam @ camera.T
    assert np.all(homog[:, 2] > 0), 'Exact projection behind camera'
    return homog[:, :2] / homog[:, 2:]


def parity(points):
    width = float(np.linalg.norm(points[1] - points[0]))
    depth = float(np.linalg.norm(points[4] - points[0]))
    return None if abs(width - depth) < 1e-10 else NAMES[int(width < depth)]


def statistics(values):
    x = np.asarray([v for v in values if v is not None], float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=None, SD=None, median=None, P90=None, maximum=None)
    return dict(n=len(x), mean=float(np.mean(x)),
                SD=float(np.std(x, ddof=1)) if len(x) > 1 else None,
                median=float(np.median(x)), P90=float(np.quantile(x, .9)),
                maximum=float(np.max(x)))


def main():
    start = time.monotonic()
    code = Path(__file__).resolve()
    old_names = ('INPUT_CONTRACT.json', 'A3_GEOMETRY_DIAGNOSTIC.json', 'A3_RATIO_PROJECTION_ROWS.jsonl.gz',
                 'A3_PHYSICAL_ROUNDTRIP_AUDIT.json', 'GEOMETRY_CODE_LOCK.json',
                 'GEOMETRY_EXECUTION_AMENDMENT.json')
    source_paths = [C.SOURCE / GEOMETRY,
                    C.PUBLIC / WD / 'STAGE1_ROWS_SYNTH.jsonl.gz',
                    C.PUBLIC / WD / 'STAGE1_ROWS_REAL.jsonl.gz',
                    C.ROOT / 'scripts/research/pallet_wd_info_fast_20261011/geometry_perturbations.py']
    preserved_paths = source_paths + [C.DOC / name for name in old_names]
    before = [C.binding(path) for path in preserved_paths]
    original = C.read(C.DOC / 'A3_GEOMETRY_DIAGNOSTIC.json')
    ids = original['frame_ids']
    assert len(ids) == 16 and len(set(ids)) == 16 and digest(ids) == SAMPLE_SHA
    output_names = ('A3_RATIO_AXIS_CORRECTION_LOCK.json',
                    'A3_RATIO_AXIS_CORRECTION.json')
    assert all(not (C.DOC / name).exists() for name in output_names)
    # Write the fixed code, source, sample, and ratio contract before arithmetic.
    C.write(C.DOC / output_names[0], dict(
        status='LOCKED_BEFORE_PROJECTION_ARITHMETIC', code_binding=C.binding(code),
        source_bindings=before, sample_ids=ids, sample_sha256=SAMPLE_SHA,
        conditions=[dict(name=name, corrected_ratio_W_over_D=wd,
                         original_ratio_argument_D_over_W=old)
                    for name, wd, old in CONDITIONS],
        formula='solverXYZ=[sqrt(area*W_over_D), H, sqrt(area/W_over_D)]',
        fixed='Same16 sample, nativeK, physicalR/t, H, area, signed point order',
        condition_selection='Registered numeric axis order; no performance-based selection',
        new_PnP_calls=0, fits=0, model_inference=0, RGB_reads=0,
        new_RGB_generation=0, protected_REAL_processing=0,
        raw_reference_coordinates_public=False))
    records = {r['id']: r for r in C.rows(source_paths[1])
               if r['seed'] == 1 and r['method'] == 'N3_THEN_SUBPIX' and r['id'] in ids}
    assert set(records) == set(ids)
    # Only permitted registered dimension metadata is used from the cache;
    # protected REAL predictions, references, poses, and metrics are not used.
    allowed = set(C.read(C.DOC / 'INPUT_CONTRACT.json')['allowed_ids'])
    assert len(allowed) == 173
    registry = set()
    for row in C.rows(source_paths[2]):
        if row['id'] in allowed and row['seed'] == 1 and row['method'] == 'BASE':
            registry.add(tuple(row['fixed_metadata']['dimensions_pnp_WH_D_m']))
    assert registry == {(1.1, .11, 1.3), (.8, .14, .59)}, registry
    with np.load(source_paths[0], allow_pickle=False) as archive:
        required = {'stems', 'dims', 'Xcf', 'R', 't', 'K', 'pad'}
        assert required <= set(archive.files)
        index = {str(s): i for i, s in enumerate(archive['stems'])}
        geom = {fid: {key: archive[key][index[fid]] for key in required - {'stems'}}
                for fid in ids}
    old_rows = list(C.rows(C.DOC / 'A3_RATIO_PROJECTION_ROWS.jsonl.gz'))
    assert len(old_rows) == 48
    old_by = {(r['id'], r['ratio_condition']): r for r in old_rows}
    assert len(old_by) == 48 and {r['id'] for r in old_rows} == set(ids)
    rows = []
    for fid in ids:
        g = geom[fid]
        xyz = np.asarray(g['dims'], float)
        points = np.asarray(g['Xcf'], float)
        nativeK = np.asarray(records[fid]['fixed_metadata']['K'], float)
        fx, fy, cx, cy = g['K']
        pad = float(g['pad'])
        assert np.allclose(nativeK, [[fx, 0, cx-pad], [0, fy, cy-pad], [0, 0, 1]],
                           rtol=0, atol=1e-8)
        assert np.allclose(xyz, records[fid]['fixed_metadata']['dimensions_pnp_WH_D_m'],
                           rtol=0, atol=1e-8)
        assert np.allclose(np.abs(points / xyz), .5, rtol=0, atol=1e-10)
        assert parity(points) == records[fid]['hypOracle']
        area = float(xyz[0] * xyz[2])
        for name, ratioWD, oldRatio in CONDITIONS:
            fixedXYZ = np.array([math.sqrt(area * ratioWD), xyz[1],
                                 math.sqrt(area / ratioWD)])
            oldXYZ = np.array([math.sqrt(area / oldRatio), xyz[1],
                               math.sqrt(area * oldRatio)])
            scale = fixedXYZ / xyz
            assert np.all(scale > 0) and scale[1] == 1
            newPoints, oldPoints = points * scale, points * (oldXYZ / xyz)
            shape_error = float(np.max(np.abs(np.ptp(newPoints, axis=0) - fixedXYZ)))
            signed_error = float(np.max(np.abs(newPoints / fixedXYZ - points / xyz)))
            area_error = abs(float(fixedXYZ[0]*fixedXYZ[2]) - area)
            assert max(shape_error, signed_error, area_error) < 1e-10
            assert abs(fixedXYZ[0]/fixedXYZ[2]-ratioWD) < 1e-12
            assert float(np.linalg.det(np.diag(scale))) > 0
            old_parity, new_parity = parity(oldPoints), parity(newPoints)
            cached = old_by[(fid, name)]
            assert old_parity == cached['reference_parity']
            assert abs(cached['ratio'] - oldRatio) < 1e-12
            oldQ = project(oldPoints, g['R'], g['t'], nativeK)
            newQ = project(newPoints, g['R'], g['t'], nativeK)
            displacement = np.linalg.norm(newQ - oldQ, axis=1)
            assert displacement[8] == 0
            same_geometry = bool(np.allclose(oldXYZ, fixedXYZ, rtol=0, atol=1e-12))
            if name == 'wood_ratio':
                assert not same_geometry and new_parity != old_parity
                assert np.allclose(fixedXYZ[[0, 2]], oldXYZ[[2, 0]], rtol=0, atol=1e-12)
            else:
                assert same_geometry and new_parity == old_parity
            rows.append(dict(
                id=fid, ratio_condition=name,
                corrected_canonical_W_over_D=ratioWD,
                original_canonical_W_over_D=1/oldRatio,
                original_argument_meaning='D_over_W',
                original_and_corrected_geometry_identical=same_geometry,
                physical_long_axis_before=None if name=='square' else 'canonical_Z_depth',
                physical_long_axis_after=None if name=='square' else
                    ('canonical_X_width' if ratioWD > 1 else 'canonical_Z_depth'),
                old_reference_parity=old_parity, corrected_reference_parity=new_parity,
                nonsquare_parity_changed=name!='square' and new_parity!=old_parity,
                positive_axis_scale_determinant=float(np.linalg.det(np.diag(scale))),
                signed_normalized_corner_order_max_error=signed_error,
                corrected_coordinate_span_max_error_m=shape_error,
                footprint_area_max_error_m2=area_error,
                center_projection_displacement_px=float(displacement[8]),
                nine_point_projection_displacement_px=statistics(displacement),
                raw_reference_R_t_K_or_points_public=False,
                corrected_6D_scores='NOT_EVALUATED_NO_NEW_PNP'))
    groups = defaultdict(list)
    for row in rows:
        groups[row['ratio_condition']].append(row)
    summaries = []
    for name, ratioWD, oldRatio in CONDITIONS:
        corrected = groups[name]
        cached = [r for r in old_rows if r['ratio_condition'] == name]
        summaries.append(dict(
            ratio_condition=name, frames=len(corrected), registered_canonical_W_over_D=ratioWD,
            original_canonical_W_over_D=1/oldRatio,
            registered_axis_matches_old=name!='wood_ratio',
            geometry_unchanged_frames=sum(r['original_and_corrected_geometry_identical'] for r in corrected),
            nonsquare_parity_changed_frames=sum(r['nonsquare_parity_changed'] for r in corrected),
            mean_projection_displacement_per_frame_px=statistics(
                r['nine_point_projection_displacement_px']['mean'] for r in corrected),
            maximum_projection_displacement_px=max(
                r['nine_point_projection_displacement_px']['maximum'] for r in corrected),
            original_pose_metrics_reused=dict(
                source='A3_RATIO_PROJECTION_ROWS.jsonl.gz',
                semantic_geometry='Old D_over_W formula, before canonical axis correction',
                available=sum(bool(r['pose'].get('available')) for r in cached),
                success_rate=float(np.mean([r['success'] for r in cached])),
                parity_error_rate=None if name=='square' else float(np.mean([r['parity_wrong'] for r in cached])),
                T_cm=statistics(r['pose'].get('translation_cm') for r in cached),
                R_deg=statistics(r['pose'].get('rotation_deg') for r in cached)),
            corrected_pose_metrics='NOT_EVALUATED_NO_NEW_PNP'))
    audit = C.read(C.DOC / 'A3_PHYSICAL_ROUNDTRIP_AUDIT.json')
    assert audit['physical_pose_roundtrip_status'] == 'FAIL_2_OF_16'
    after = [C.binding(path) for path in preserved_paths]
    assert before == after, 'Existing source or original A3 artifact changed'
    result = dict(
        status='PASS_REGISTERED_CANONICAL_AXIS_PROJECTION_CONTRACT',
        original_pose_gate='FAIL_2_OF_16_UNRESOLVED',
        original_pose_failure_ids=audit['failure_ids'],
        interpretation='The wood canonical W/D order was inverted in the original optional ratio condition. '
                       'The corrected pure projection swaps its physical long axis without reflecting or reordering points. '
                       'This repairs the shape-condition definition only; signed C1 pose restoration remains unresolved.',
        scope='Optional A3 ratio supplement; original noise and ratio pose results remain immutable',
        sample_frames=16, sample_sha256=SAMPLE_SHA, conditions=3, derived_rows=len(rows),
        registry_dimensions_solverXYZ=[list(x) for x in sorted(registry)],
        square='Controlled C4 shape geometry; long/short parity is undefined. Original assets and approved groups unchanged.',
        unchanged_inputs=['nativeK', 'physicalR', 'physicalt', 'H', 'footprint_area', 'sample', 'signed corner order'],
        formula='solverXYZ=[sqrt(area*ratioWD), H, sqrt(area/ratioWD)]',
        no_reflection=True, new_corner_assignment=False,
        original_artifacts_preserved=True, preserved_source_bindings_after=after,
        corrected_6D_metrics_status='NOT_EVALUATED_NO_NEW_PNP',
        summaries=summaries, rows=rows,
        new_PnP_calls=0, fits=0, model_inference=0, RGB_reads=0,
        new_RGB_generation=0, protected_REAL_processing=0,
        raw_reference_coordinates_public=False, wall_seconds=time.monotonic()-start)
    C.write(C.DOC / output_names[1], result)
    print(json.dumps(dict(status=result['status'], rows=len(rows),
        summaries=[{key: value for key, value in r.items() if key != 'original_pose_metrics_reused'}
                   for r in summaries],
        physical_pose_gate=result['original_pose_gate'], new_PnP_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
