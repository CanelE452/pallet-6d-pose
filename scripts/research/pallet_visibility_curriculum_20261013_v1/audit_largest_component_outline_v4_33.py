"""Independent pixel/count/outline replay for stored LCC8 observations.

The pure auditor never calls the new LCC producer or frozen observer. It uses
OpenCV's shared connected-component and hull primitives, independently counts
labels, clips hull segments and reconstructs midpoint samples. This certifies
the stored transformation, not semantic pallet identity or pose accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROFILE = 'LCC8_MAX_AREA_LOWEST_LABEL_FROZEN_V13_OUTLINE_V33'
BASELINE = 'ORIGINAL_UNION_FROZEN_V13_OUTLINE'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def clipped_segment(a, b, width, height, margin):
    """Liang--Barsky interval intersection, independently of the observer."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = b-a
    lower, upper = 0., 1.
    for coefficient, slack in ((-delta[0], a[0]-margin),
                               (delta[0], width-1-margin-a[0]),
                               (-delta[1], a[1]-margin),
                               (delta[1], height-1-margin-a[1])):
        if abs(coefficient) < 1e-14:
            if slack < 0:
                return None
            continue
        crossing = slack/coefficient
        if coefficient < 0:
            lower = max(lower, crossing)
        else:
            upper = min(upper, crossing)
        if lower >= upper:
            return None
    return a+lower*delta, a+upper*delta


def replay_outline(mask, configuration):
    values = np.asarray(mask, float)
    h, w = values.shape
    yy, xx = np.nonzero(values >= .5)
    if len(xx) < 3:
        return dict(supported=False, reason='insufficient_mask_foreground')
    polygon = cv2.convexHull(np.stack((xx, yy), axis=1).astype(np.float32)).reshape(-1, 2).astype(float)
    area = float(cv2.contourArea(polygon.astype(np.float32)))
    if len(polygon) < 3 or abs(area) < 1:
        return dict(supported=False, reason='degenerate_mask_envelope')
    segments, samples = [], []
    for index in range(len(polygon)):
        pair = clipped_segment(polygon[index], polygon[(index+1) % len(polygon)],
                               w, h, configuration['frame_margin_px'])
        if pair is None:
            continue
        first, last = pair
        segments.append([first.tolist(), last.tolist()])
        count = max(1, math.ceil(float(np.linalg.norm(last-first))/configuration['observed_sample_spacing_px']))
        for k in range(count):
            samples.append((first+(k+.5)/count*(last-first)).tolist())
    if len(samples) < 3:
        return dict(supported=False, reason='no_non_canvas_outline')
    return dict(supported=True, polygon_xy=polygon.tolist(),
                observed_boundary_segments_xy=segments,
                observed_boundary_samples_xy=samples, mask_area_px=len(xx),
                foreground_hull_area_px=area)


def audit_row(mask, row):
    """Pure check of one mask and its stored union/LCC observation pair."""
    original = np.asarray(mask)
    before = pixel_sha(original)
    values = np.asarray(mask, float)
    if (values.ndim != 2 or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError('Finite original global mask in [0,1] required')
    checks = []

    def check(condition, reason):
        checks.append(dict(passed=bool(condition), reason=reason))

    binary = np.ascontiguousarray(values >= .5, dtype=np.uint8)
    # Independently count the returned labels rather than trust producer stats.
    labels_count, labels, _, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S)
    areas = np.bincount(labels.ravel(), minlength=labels_count)[1:]
    selected_label = None if not len(areas) else int(np.argmax(areas)+1)
    selected = np.zeros(binary.shape, np.uint8) if selected_label is None else (labels == selected_label).astype(np.uint8)
    ties = [] if not len(areas) else (np.flatnonzero(areas == areas.max())+1).tolist()
    union, lcc = row['original_union'], row['largest_component']
    audit = lcc['foreground_component_selection']
    check(row['selection_profile'] == lcc['selection_profile'] == PROFILE,
          'uniform current LCC8 profile')
    check(row['baseline_profile'] == union['selection_profile'] == BASELINE,
          'original union control retained')
    check(row['row_retained'] is True and row['union_exact_match'] is True,
          'row and producer union comparison retained')
    bare_union = {key:value for key,value in union.items() if key != 'selection_profile'}
    check(row['stored_union_sha256'] == row['regenerated_union_sha256'] == digest(bare_union),
          'exact source union canonical digest')
    check(all(row['union_per_field_equal'].values()), 'all producer union fields compared')
    check(audit['selected_label'] == selected_label and audit['maximum_area_tied_labels'] == ties,
          'independently counted max-area component and exact-label tie')
    check(audit['foreground_component_count'] == len(areas), 'all foreground components counted')
    check(audit['original_foreground_area_px'] == int(binary.sum())
          and audit['retained_foreground_area_px'] == int(selected.sum())
          and audit['discarded_foreground_area_px'] == int(binary.sum()-selected.sum()),
          'independent original/retained/discarded pixel counts')
    check(audit['original_union_binary_pixel_sha256'] == pixel_sha(binary)
          and audit['selected_binary_pixel_sha256'] == pixel_sha(selected),
          'independent original/selected binary pixel hashes')
    check(audit['connectivity'] == 8 and audit['mask_threshold'] == .5
          and audit['holes_filled'] is False and audit['morphology_used'] is False
          and audit['convex_hull_used_in_component_selection'] is False,
          'one fixed preprocessing rule, no morphology/hull selection')
    components = {item['label']:item for item in audit['components']}
    check(set(components) == set(range(1, labels_count)), 'complete component metadata')
    for label in range(1, labels_count):
        yy, xx = np.nonzero(labels == label)
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)]
        item = components.get(label, {})
        check(item.get('area_px') == len(xx) and item.get('bbox_xywh') == bbox
              and item.get('retained') == (label == selected_label),
              'component area/bbox/retention '+str(label))
    check(union['configuration'] == lcc['configuration']
          and union['image_binding'] == lcc['image_binding']
          and union['image_hw'] == lcc['image_hw'] == list(values.shape),
          'same global image/canvas/configuration')
    maximum_difference = 0.
    for name, observation, array in (('union', union, values), ('LCC', lcc, selected.astype(np.float32))):
        check(observation['mask_array_sha256'] == pixel_sha(np.asarray(array, float)),
              name+' original float-mask hash')
        check(observation['global_mask_only'] is True
              and observation['S_local_queries'] == observation['U_local_queries'] == 0,
              name+' no state-local queries')
        expected = replay_outline(array, observation['configuration'])
        check(observation['supported'] == expected['supported'], name+' support/abstention')
        if not expected['supported']:
            check(observation.get('reason') == expected['reason'], name+' exact abstention reason')
            continue
        for key, value in expected.items():
            if key == 'supported':
                continue
            actual = np.asarray(observation[key]); want = np.asarray(value)
            same_shape = actual.shape == want.shape
            difference = float(np.max(np.abs(actual-want))) if same_shape and actual.size else 0.
            maximum_difference = max(maximum_difference, difference)
            check(same_shape and np.allclose(actual, want, atol=1e-9, rtol=1e-12),
                  name+' independent '+key)
    check(pixel_sha(original) == before, 'original mask array unchanged')
    return dict(checks=checks, failure_count=sum(not item['passed'] for item in checks),
                maximum_coordinate_difference=maximum_difference,
                expected_component_count=len(areas), selected_label=selected_label,
                original_foreground_area_px=int(binary.sum()), retained_foreground_area_px=int(selected.sum()),
                old_observer_or_new_LCC_producer_calls=0,
                physical_semantic_identity_or_accuracy_certified=False)


ROOT = Path(__file__).resolve().parents[3]


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def relative_path(value):
    path = (ROOT/value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Input must remain in the repository')
    return path


def verify_maps(*maps):
    combined = {}
    for mapping in maps:
        for name, expected in mapping.items():
            if name in combined and combined[name] != expected:
                raise ValueError('Conflicting binding: '+name)
            combined[name] = expected
    for name, expected in combined.items():
        if sha_file(relative_path(name)) != expected:
            raise ValueError('Frozen input changed: '+name)


def audit_CAL(*, folder, expected_seal_sha256, expected_central_sha256,
              root_CAL_observation_audit_GO=False):
    """Replay all46 masks after the separately sealed CAL producer finishes."""
    if root_CAL_observation_audit_GO is not True:
        raise PermissionError('ROOT all46 observation audit GO before IO')
    if not all(isinstance(value, str) and len(value) == 64
               and all(c in '0123456789abcdef' for c in value)
               for value in (expected_seal_sha256, expected_central_sha256)):
        raise ValueError('Explicit exact CAL-seal and central SHA before IO')
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError('CAL artifacts must remain in the repository')
    seal_path = folder/'CAL_OBSERVATION_SEAL.json'
    if sha_file(seal_path) != expected_seal_sha256:
        raise ValueError('Current exact CAL observation seal required')
    seal = read(seal_path); prepared = read(folder/'PREPARATION.json')
    central = prepared['central_plan']
    if (central['sha256'] != expected_central_sha256
            or sha_file(relative_path(central['path'])) != expected_central_sha256
            or seal['central_sha256'] != expected_central_sha256
            or seal['rows'] != 46 or len(seal['ordered_ids']) != 46
            or len(set(seal['ordered_ids'])) != 46
            or seal['selection_profile'] != PROFILE
            or prepared['ordered_CAL_ids'] != seal['ordered_ids']
            or seal['union_exact_match_count'] != 46
            or sha_file(folder/'PREPARATION.json') != seal['preparation_sha256']
            or sha_file(folder/'CAL_ROW_INDEX.json') != seal['row_index_sha256']
            or sha_file(folder/'CAL_OUTLINE_OBSERVATIONS.jsonl.gz') != seal['observations_sha256']):
        raise ValueError('Same complete46 preparation/currentcentral/profile/seal required')
    maps = [prepared[key] for key in ('source_sha256','input_sha256',
                                     'protected_user_sha256','original_CAL_file_sha256')]
    if len(maps[2]) != 6 or len(maps[3]) != 53:
        raise ValueError('Protected6 and immutable CAL53 required')
    verify_maps(*maps, seal['assets'])
    entries = read(folder/'CAL_ROW_INDEX.json')['entries']
    cases = prepared['cases']
    ids = seal['ordered_ids']
    if [entry['id'] for entry in entries] != ids or [case['id'] for case in cases] != ids:
        raise ValueError('Every ordered CAL46 row including empty cases required')
    dest = folder/'independent_mask_outline_audit_v1'
    if dest.exists():
        raise FileExistsError('Preserve every independent audit attempt')
    dest.mkdir()
    sources = {str(path.relative_to(ROOT)):sha_file(path) for path in (
        Path(__file__).resolve(), Path(__file__).with_name('test_audit_largest_component_outline_v4_33.py').resolve())}
    write(dest/'PLAN.json', dict(schema='INDEPENDENT_CAL46_LCC8_PIXEL_OUTLINE_AUDIT_V33',
        source_sha256=sources, CAL_observation_seal_sha256=expected_seal_sha256,
        central_sha256=expected_central_sha256, rows=46, ordered_ids=ids,
        new_LCC_producer_or_frozen_observer_calls=0,
        numeric_mask_or_row_decodes_before_plan=0,
        scope='Same cached mask components/pixels, independent clipping/sampling; no semantic/pose accuracy proof',
        opencv_version=cv2.__version__, numpy_version=np.__version__))
    results = []; failures = []; total_checks = 0; last_id = None
    try:
        for entry, case in zip(entries, cases):
            last_id = entry['id']; rowpath = relative_path(entry['path'])
            if sha_file(rowpath) != entry['sha256']:
                raise ValueError('Sealed observation row changed')
            row = read(rowpath)
            if (digest(row) != entry['canonical_row_sha256'] or row['id'] != last_id
                    or row['partition'] != 'CAL' or row['positive_packet_sha256'] != case['positive_packet_sha256']
                    or row['target_mask_binding'] != case['target_mask_binding']):
                raise ValueError('Canonical row/current input/mask identity changed')
            binding = row['target_mask_binding']; maskpath = relative_path(binding['path'])
            if sha_file(maskpath) != binding['sha256']:
                raise ValueError('Original cached mask bytes changed')
            with np.load(maskpath, allow_pickle=False) as arrays:
                mask = arrays['pallet_mask_clean'].copy()
            result = audit_row(mask, row)
            total_checks += len(result['checks'])
            failures.extend(dict(id=last_id, **check) for check in result['checks'] if not check['passed'])
            results.append(dict(id=last_id, **result))
            del mask, row
        verify_maps(*maps, seal['assets'], sources)
        if sha_file(seal_path) != expected_seal_sha256:
            raise ValueError('Current observation seal changed during replay')
    except BaseException as error:
        write(dest/'FAILURE.json', dict(rows_completed=len(results), attempted_id=last_id,
            exception=type(error).__name__, reason=str(error), attempt_preserved=True))
        raise
    report = dict(schema='INDEPENDENT_CAL46_LCC8_PIXEL_OUTLINE_AUDIT_V33', rows=len(results),
        checks=total_checks, failure_count=len(failures), failures=failures, row_results=results,
        maximum_coordinate_difference=max(item['maximum_coordinate_difference'] for item in results),
        CAL_observation_seal_sha256=expected_seal_sha256, central_sha256=expected_central_sha256,
        source_sha256=sources, new_LCC_producer_or_old_observer_calls=0,
        RGB_GT_reference_PnP_or_NN_inputs=0, source_CAL_or_mask_mutations=0,
        semantic_or_pose_accuracy_certification=False)
    write(dest/'AUDIT.json', report)
    write(dest/'AUDIT_SEAL.json', dict(report_sha256=sha_file(dest/'AUDIT.json'),
        plan_sha256=sha_file(dest/'PLAN.json'), source_sha256=sources,
        CAL_observation_seal_sha256=expected_seal_sha256))
    if failures:
        raise SystemExit(1)
    return dict(rows=len(results), checks=total_checks, failures=0,
                audit_sha256=sha_file(dest/'AUDIT.json'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--central-sha256', required=True)
    parser.add_argument('--root-CAL-observation-audit-GO', action='store_true')
    args = parser.parse_args()
    print(audit_CAL(folder=args.folder, expected_seal_sha256=args.seal_sha256,
          expected_central_sha256=args.central_sha256,
          root_CAL_observation_audit_GO=args.root_CAL_observation_audit_GO))
