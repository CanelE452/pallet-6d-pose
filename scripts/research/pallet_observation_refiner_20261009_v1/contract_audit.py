"""Read-only coordinate, reference-phase and inference-boundary audit.

Human visibility is canonical-reference evidence. Its use in a native-index
mask additionally consumes the frozen evaluation phase and is always oracle.
No new mesh-edge ownership or independent physical ground truth is inferred.
"""
from __future__ import annotations

from collections import Counter
import argparse
import builtins
import inspect
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from .solver import HypothesisBank, cuboid, project, visibility

VIS_REL = Path('_docs/experiments/pallet_combined_closeout_20261003_v1/'
    'closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
TARGET_REL = Path('data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json')
SYM_REL = Path('_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
PROVENANCE_REL = Path('_docs/experiments/pallet_combined_closeout_20261003_v1/'
    'closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json')
REG_REL = Path('challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')
OLD_REL = Path('_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz')
PANEL_REL = Path('data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json')


def native_oracle_categories(target, baseline_row, labels):
    """Translate canonical human states with one already frozen object phase."""
    phase = baseline_row['corner']['branch']
    if phase is None:
        return ['UNANNOTATED'] * 8, None
    permutation = np.asarray(target['permutations'][phase], int)
    states = [labels.get((baseline_row['id'], int(k)), {}).get('category', 'UNANNOTATED')
              for k in permutation[:8]]
    return states, dict(status='ORACLE_MASK_AND_PHASE', branch=int(phase),
                       native_to_canonical=permutation[:8].tolist())


def _canary_self_test():
    methods = [lambda p: builtins.open(p, 'rb'), lambda p: p.read_bytes()]
    protected = [C.ROOT / TARGET_REL, C.ROOT / VIS_REL]
    caught = 0
    for method in methods:
        for path in protected:
            with C.no_truth_reads():
                try:
                    with method(path) if method is methods[0] else __import__('contextlib').nullcontext(method(path)):
                        raise AssertionError('GT read unexpectedly permitted')
                except AssertionError as error:
                    assert str(error).startswith('GT_CANARY:'), str(error)
                    caught += 1
    return dict(protected_read_attempts=caught, builtin_open_and_Path_read_bytes_blocked=caught == 4)


def repair_replay():
    """Validate the stronger read canary without altering any measured result."""
    import time
    from .inference import hidden_mask, finish
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
    E, _ = legacy()
    output = C.DOC / 'REPAIR_LOG.json'
    assert not output.exists(), 'Preserve previous repair validation attempt'
    lock = C.read(C.DOC / 'INFERENCE_CODE_LOCK.json')
    old_binding = next(b for b in lock['files'] if b['path'].endswith('/common.py'))
    new_binding = C.binding(Path(C.__file__))
    observations_path = C.DOC / 'OBSERVATIONS.jsonl.gz'
    old_observations = C.binding(observations_path)
    rows = list(C.iter_rows(observations_path))
    old = {(r['id'], r['method']): r for r in rows}
    inputs = C.read(C.DOC / 'INPUTS.json')['frames']
    shared = {r['id']: r for r in C.iter_rows(C.ROOT / '_docs/experiments/pallet_visible_boundary_20261009_v1/COORDINATES.jsonl.gz')
              if r['method'] == 'BASE_BOUNDARY_NATIVE'}
    ledger = Counter()
    repair = dict(schema='inference_canary_repair_v1', complete=False,
        cause='Python 3.10 pathlib retained an accessor-cached io.open; monkeypatching io.open alone did not block Path.read_text/read_bytes.',
        correction='common.no_truth_reads now wraps Path.open as well as builtins.open and io.open; restores all three.',
        old_common=old_binding, new_common=new_binding, original_observations=old_observations,
        performance_tuning=False, original_outputs_preserved=True, validation_scope='prediction-only full319 replay; no GT scoring/model forwards',
        self_test=_canary_self_test())
    C.write(output, repair)
    started = time.monotonic()
    comparisons = []
    names = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')
    originals = {name: getattr(cv2, name) for name in names}
    for name in names:
        def wrapped(*args, _name=name, **kwargs):
            ledger[_name] += 1
            return originals[_name](*args, **kwargs)
        setattr(cv2, name, wrapped)
    try:
        with C.no_truth_reads():
            for i, frame in enumerate(inputs):
                initial_base = None
                for arm in ('BASE', 'N3_SUBPIX'):
                    q = np.asarray(frame['points'][arm], float)
                    initial = E.POSE.infer(q, np.asarray(frame['K']), np.asarray(frame['xyz']), source=False)
                    ledger['initial_pose_calls'] += 1
                    if arm == 'BASE':
                        initial_base = initial
                    H, _ = hidden_mask(initial)
                    bank = HypothesisBank(q, frame['K'], frame['xyz'], image_size=tuple(frame['raw_hw'][::-1]))
                    for suffix in C.SUFFIXES[:4]:
                        hidden = H if suffix.startswith('GEOM') else []
                        new = finish(bank, q, initial, excluded=hidden, hidden=hidden, robust=suffix.endswith('ROBUST'))
                        ledger['logical_solver_calls'] += 1
                        reference = old[(frame['id'], arm + '_' + suffix)]
                        compare = {k: reference[k] for k in new}
                        assert C.digest(new) == C.digest(compare), (frame['id'], arm, suffix, 'numeric output drift')
                        comparisons.append(dict(id=frame['id'], method=arm + '_' + suffix,
                                                output_sha256=C.digest(new), exact_numeric_output_equal=True))
                raw = shared[frame['id']]
                q = np.asarray(raw['native_points'], float)
                selected = [r['corner'] for r in raw['correction']['diagnostics']['corner_records']
                            if r['status'] == 'refined' and len(r.get('selected_edges') or []) == 2]
                observed = q.copy()
                observed[[j for j in range(8) if j not in selected]] = np.nan
                H, _ = hidden_mask(initial_base)
                bank = HypothesisBank(observed, frame['K'], frame['xyz'], image_size=tuple(frame['raw_hw'][::-1]))
                new = finish(bank, q, initial_base, excluded=H, hidden=H, robust=True)
                ledger['logical_solver_calls'] += 1
                reference = old[(frame['id'], 'SHARED_BOUNDARY_GEOM_ROBUST')]
                compare = {k: reference[k] for k in new}
                assert C.digest(new) == C.digest(compare), (frame['id'], 'SHARED', 'numeric output drift')
                comparisons.append(dict(id=frame['id'], method='SHARED_BOUNDARY_GEOM_ROBUST',
                                        output_sha256=C.digest(new), exact_numeric_output_equal=True))
                if i % 50 == 0 or i == 318:
                    print('CANARY_REPLAY', i + 1, 'rows', len(comparisons), 'seconds', round(time.monotonic() - started, 2), flush=True)
        assert len(comparisons) == 319 * 9 and ledger['initial_pose_calls'] == 638
        assert C.binding(observations_path) == old_observations
        repair.update(complete=True, status='PASS', frames=319, exact_output_rows=len(comparisons),
            comparisons= comparisons, execution=dict(ledger, seconds=time.monotonic() - started,
                detector_calls=0, N3_calls=0, training_updates=0, GT_reads=0, GT_scoring_calls=0))
    except Exception as error:
        repair.update(status='FAILED', reason=dict(type=type(error).__name__, message=str(error)),
                      execution=dict(ledger, seconds=time.monotonic() - started))
        raise
    finally:
        for name, fn in originals.items():
            setattr(cv2, name, fn)
        C.write(output, repair)
    return repair


def _network_roundtrip(panel, input_by_id):
    """Use real frozen cache affines without loading RGB or running a model."""
    import torch
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
    E, _ = legacy()
    features = E.POSE.E.old('features')
    checked, max_error, float32_error, paths = [], 0., 0., []
    axis = C.read(C.ROOT / 'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json')['frames_list']
    image_to_axis = {r['image']: r['frame_id'] for r in axis}
    for record in panel:
        frame = input_by_id[record['frame_id']]
        path = C.ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/DEV_cache' / (image_to_axis[record['image_key']] + '.pt')
        captured = torch.load(path, map_location='cpu', weights_only=False)['captured']
        gain, offset = features.canvas_affine(captured['canvas_shape'], captured['input_shape'])
        pad = captured['added_border']
        index = captured['selected_index']
        if index is not None:
            candidate = captured['candidates'][index]
            native = np.asarray(candidate['keypoints_xy'], np.float64)
            np.testing.assert_array_equal(native, frame['points']['BASE'])
            network = (native + pad) * gain + offset
            restored = (network - offset) / gain - pad
            max_error = max(max_error, float(np.max(np.abs(restored - native))))
            np.testing.assert_allclose(restored, native, atol=1e-10, rtol=0)
            branch = features.branch_inputs(captured)
            float32_error = max(float32_error, float(np.max(np.abs(branch['points'] - network))))
            unchanged = features.restore_refinement(candidate, branch['points'], branch['points'].copy(), gain, lam=1.)
            np.testing.assert_array_equal(unchanged, native)
        checked.append(dict(id=frame['id'], reflected_border_px=int(pad), gain=float(gain),
                            letterbox_offset=offset.tolist(), input_shape=list(captured['input_shape'])))
        paths.append(C.binding(path))
    return dict(checked_frames=len(checked), exact_float64_max_abs_px=max_error,
                inherited_float32_network_rounding_max_px=float32_error,
                zero_displacement_native_identity=True, frames=checked), paths


def run():
    path = C.DOC / 'CONTRACT_AUDIT.json'
    assert not path.exists(), 'Preserve completed contract audit'
    inputs = C.read(C.DOC / 'INPUTS.json')['frames']
    input_by_id = {r['id']: r for r in inputs}
    all_targets = C.read(C.ROOT / TARGET_REL)
    targets = {fid: all_targets[fid] for fid in input_by_id}
    packet = C.read(C.ROOT / VIS_REL)
    labels = {(r['frame_id'], int(r['corner_id'])): r for r in packet['rows'] if r['population'] == 'DEV319'}
    assert len(labels) == sum(r['population'] == 'DEV319' for r in packet['rows'])
    categories = ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'EXTERNAL_OCCLUDED', 'OUT_OF_FRAME', 'UNKNOWN', 'UNANNOTATED')
    symmetry = C.read(C.ROOT / SYM_REL)
    objects = {r['object_type']: r for r in symmetry['objects']}
    registry = {r['object_type']: r for r in C.read(C.ROOT / REG_REL)['objects']}
    prior = {m: {} for m in C.CONTROLS}
    for row in C.iter_rows(C.ROOT / OLD_REL):
        prior[row['method']][row['id']] = row
    assert len(inputs) == len(input_by_id) == len(targets) == 319
    assert len({r['session'] for r in inputs}) == 13
    assert all(set(values) == set(input_by_id) for values in prior.values())
    checks, states, sources, annotation_sources = Counter(), Counter(), Counter(), Counter()
    oracle_rows = []
    for frame in inputs:
        fid, target = frame['id'], targets[frame['id']]
        xyz = np.asarray(frame['xyz'], np.float64)
        obj = objects[target['object']]
        dimensions = registry[target['object']]['physical_dimensions_m']
        np.testing.assert_array_equal(xyz, [dimensions[k] for k in ('x', 'y', 'z')])
        np.testing.assert_array_equal(xyz, target['xyz'])
        np.testing.assert_array_equal(np.asarray(target['dimensions_WDH'])[[0, 2, 1]], xyz)
        np.testing.assert_array_equal(frame['K'], target['K'])
        assert np.asarray(frame['K']).shape == (3, 3)
        assert frame['raw_hw'] == prior['BASE'][fid]['raw_hw']
        np.testing.assert_array_equal(frame['points']['BASE'], prior['BASE'][fid]['native_points'])
        np.testing.assert_array_equal(frame['points']['N3_SUBPIX'], prior['N3_SUBPIX'][fid]['native_points'])
        assert target['permutations'] == obj['permutations']
        assert target['truth']['order'] == obj['group_order']
        assert obj['group_order'] == 2, 'DEV319 rectangular proper C2 is frozen'
        X = cuboid(*xyz)
        np.testing.assert_allclose(X, np.asarray(obj['corners_centroid'])[:8], atol=1e-14, rtol=0)
        for Q, permutation in zip(obj['rotations'], obj['permutations']):
            Q, permutation = np.asarray(Q), np.asarray(permutation, int)
            assert permutation[8] == 8 and len(set(permutation)) == 9
            np.testing.assert_allclose(np.linalg.det(Q), 1., atol=1e-14)
            np.testing.assert_allclose((Q @ X.T).T, X[permutation[:8]], atol=1e-14, rtol=0)
        for method in ('BASE', 'N3_SUBPIX'):
            row = prior[method][fid]
            states_native, phase = native_oracle_categories(target, row, labels)
            if phase is not None:
                permutation = np.asarray(target['permutations'][phase['branch']], int)
                native = np.asarray(row['native_points'], float)
                canonical = np.empty_like(native)
                canonical[permutation] = native
                np.testing.assert_array_equal(canonical[permutation], native)
                checks['native_canonical_phase_roundtrips'] += 1
            oracle_rows.append(dict(id=fid, method=method, categories_native=states_native, phase=phase))
        for k in range(8):
            label = labels.get((fid, k))
            category = (label or {}).get('category', 'UNANNOTATED')
            assert category in categories
            states[category] += 1
            if label:
                np.testing.assert_allclose(label['reference_xy'], target['gt'][k], atol=1e-7, rtol=0)
                sources[label.get('reference_source', 'UNRECORDED')] += 1
                annotation_sources[label.get('label_evidence', {}).get('annotation_coordinate_source', 'UNRECORDED')] += 1
                checks['visibility_reference_xy_equal'] += 1
        checks['camera_dimension_input_rows_equal'] += 1
    panel = C.read(C.ROOT / PANEL_REL)['selected']
    assert len(panel) == 26 and len({r['session_id'] for r in panel}) == 13
    network, caches = _network_roundtrip(panel, input_by_id)
    canary = _canary_self_test()
    K = np.array([[610., 0., 320.], [0., 615., 240.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.3])
    R = cv2.Rodrigues(np.array([.5, .3, .1]))[0]
    t = np.array([.03, -.04, 3.])
    corners = project(cuboid(*xyz), R, t, K)
    q = np.vstack([corners, [320., 240.]])
    hidden = np.flatnonzero(visibility(cuboid(*xyz), R, t)[0]).tolist()
    with C.no_truth_reads():
        bank = HypothesisBank(q, K, xyz, image_size=(640, 480))
        result = bank.solve(excluded=hidden, robust=True, hidden=hidden)
    assert result['available'] and not set(hidden) & set(result['fit_input_ids'])
    np.testing.assert_array_equal(np.asarray(result['points_final'])[8], q[8])
    if hidden:
        np.testing.assert_allclose(np.asarray(result['points_final'])[hidden], np.asarray(result['projected'])[hidden], atol=1e-12, rtol=0)
    canary.update(synthetic_deployable_solver_executed=True, GT_reads=0,
                  solver_constructor_parameters=list(inspect.signature(HypothesisBank).parameters),
                  solver_parameters=list(inspect.signature(HypothesisBank.solve).parameters))
    result = dict(schema='observation_refiner_contract_audit_v1', status='PASS', complete=True,
        population=dict(frames=319, sessions=13, physical_corner_ids=list(range(8)), center_id=8,
                        visibility_state_total=319 * 8, label_counts=dict(states), reference_sources=dict(sources),
                        annotation_coordinate_sources=dict(annotation_sources)),
        coordinates=dict(image='raw original native pixels', units='registered geometry meters; translation cm in metrics',
                         K='exact frozen raw K', distortion='None, unchanged',
                         WDH_to_xyz=[0, 2, 1], self_occlusion_margin_deg=2., network=network),
        symmetry=dict(rectangular='fixed proper C2 only', quarter_turn_equivalence=False,
                      width_depth='two correspondence hypotheses; independent of allowed symmetry',
                      source_basis='REAL_DEV source=False; synthetic historical Rx(pi) outside this real audit'),
        human_mask=dict(status='ORACLE_MASK_AND_PHASE', native_mapping='native_state[k] = canonical_state[permutation[k]]',
                        permutation_origin='same-input frozen baseline whole-object evaluation branch',
                        unknown_unannotated_not_reclassified=True, reference_coordinates_unchanged=True,
                        rows=oracle_rows),
        physical_geometry=dict(cuboid_corners='registered virtual reference corners',
                               actual_mesh_boundary_ownership='not established by cuboid indexing; source supervision audited separately',
                               point_PnP_diagnostic_allowed=True, cuboid_mask_is_actual_mesh=False),
        reference=dict(kind='existing geometry-reconstructed reference', independent_physical_measurement=False,
                       ADDsym='minimum corresponding eight-corner ADD over approved proper rotation group'),
        inference_canary=canary, checks=dict(checks),
        input_bindings=[C.binding(C.ROOT / p) for p in (VIS_REL, TARGET_REL, SYM_REL, PROVENANCE_REL, REG_REL, OLD_REL, PANEL_REL)] + caches,
        execution=dict(new_detector_calls=0, new_head_calls=0, synthetic_solver_calls=1, new_training=0,
                       real_new_pose_calls=0, cache_reads=len(caches)))
    C.write(path, result)
    print('CONTRACT_AUDIT PASS', dict(checks), dict(states), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=('audit', 'repair_replay'), default='audit')
    args = parser.parse_args()
    {'audit': run, 'repair_replay': repair_replay}[args.stage]()
