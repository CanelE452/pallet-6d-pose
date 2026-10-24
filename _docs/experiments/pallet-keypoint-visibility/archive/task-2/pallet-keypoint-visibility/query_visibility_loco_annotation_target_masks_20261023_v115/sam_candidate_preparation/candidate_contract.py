"""Pure NumPy contracts; no files, models, torch, or source I/O at import."""
import math
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


ENTRY_FIELDS = {'schema', 'approved', 'allow_public_source_SAM_candidate_forward',
                'source_only', 'current223', 'V112', 'fit', 'PnP',
                'expected_frames', 'expected_request_slots', 'expected_primary_masks',
                'expected_repeat_calls', 'expected_set_image_calls', 'expected_query_slots',
                'output_directory', 'bindings'}


def validate_entry_structure(e, static_paths, output_directory, executable, expected_python):
    """No callback or I/O: reject authorization/whitelist/output issues first."""
    require(set(e) == ENTRY_FIELDS, 'Exact root entry field whitelist')
    require(e['schema'] == 'V115_ROOT_SAM_CANDIDATE_EXECUTION_ENTRY_V1', 'Entry schema')
    require(e['approved'] is True and e['allow_public_source_SAM_candidate_forward'] is True,
            'Separate approved root entry required')
    require(e['source_only'] is True and e['current223'] is False and e['V112'] is False
            and e['fit'] is False and type(e['PnP']) is int and e['PnP'] == 0, 'Source-only scope')
    expected = dict(expected_frames=20, expected_request_slots=40, expected_primary_masks=33,
                    expected_repeat_calls=1, expected_set_image_calls=20, expected_query_slots=160)
    require(all(type(e[k]) is int and e[k] == v for k, v in expected.items()), 'Fixed counts')
    require(Path(e['output_directory']).resolve() == Path(output_directory).resolve(), 'Fixed output path')
    require(Path(executable).absolute() == Path(expected_python).absolute(), 'Sealed isolated Python')
    require(set(e['bindings']) == set(static_paths), 'Exact binding whitelist')
    for k, path in static_paths.items():
        b = e['bindings'][k]
        require(set(b) == {'path', 'sha256'}, 'Strict path/SHA binding')
        require(Path(b['path']).resolve() == Path(path).resolve(), 'Unexpected binding path: ' + k)
        require(isinstance(b['sha256'], str) and len(b['sha256']) == 64
                and all(c in '0123456789abcdef' for c in b['sha256']), 'SHA256 syntax')


def validate_cohort(frames, slots, requests, entities):
    require(len(frames) == len(entities) == 20 and len(slots) == 160 and len(requests) == 40,
            'Full 20/160/40 retention')
    require(len({f['input_id'] for f in frames}) == 20, 'Duplicate frame ID')
    require(len({r['request_id'] for r in requests}) == 40, 'Duplicate request ID')
    require(sum(r['call_eligible_metadata'] for r in requests) == 33, '33 masks / 7 null no-calls')
    factual = {'NO_DETECTION_QUERY_U': 0, 'OUTSIDE_OR_INVALID_QUERY_U': 0, 'SAVED_UNLABELED_QUERY': 0}
    for i, f in enumerate(frames):
        require(f['frame_index'] == i and entities[i]['frame_index'] == i, 'Frame order')
        ent = entities[i]
        require(ent['input_id'] == f['input_id'] and ent['image_sha256'] == f['image_sha256']
                and ent['split'] == f['split'], 'Entity frame/fold identity')
        require(ent['existing_target_ann_id'] is not None, 'Existing target required; no substitution')
        for j, role in enumerate(('TARGET', 'COMPETITOR')):
            r = requests[2*i+j]
            require(r['request_index'] == 2*i+j and r['frame_index'] == i and r['role'] == role,
                    'Request order/role')
            require(r['input_id'] == f['input_id'] and r['split'] == f['split'], 'Request ID/fold')
            require(r['image_sha256'] == f['image_sha256'] and r['raw_hw'] == f['raw_hw'], 'Image identity')
            require(r['image'] == f['image'] and r['canonical_decoded_RGB'] == f['decoded_RGB_input']
                    and r['decoded_RGB_pixels_sha256'] == f['RGB_pixels_sha256'], 'Canonical NPY identity')
            require(r['point_coords'] is None and r['point_labels'] is None, 'No point prompt')
            require(r['executed'] is False and r['box_clipping_or_adjustment_applied'] is False,
                    'Original unexecuted raw prompt')
            if role == 'TARGET':
                a = dict(ann_id=ent['existing_target_ann_id'], bbox_xywh_raw=ent['bbox_xywh_raw'],
                         box_xyxy_raw=ent['box_xyxy_raw'])
            else:
                a = ent['competitor']
            require(r['call_eligible_metadata'] == (a is not None), 'Missing role stays no-call')
            if a is None:
                require(r['ann_id'] is None and r['bbox_xywh_raw'] is None and r['box_xyxy_raw'] is None
                        and r['status'] == 'NO_CALL_MISSING_COMPETITOR', 'Missing is null, not zero')
            else:
                require(r['status'] == 'PLANNED_BOX_REQUEST' and r['ann_id'] == a['ann_id']
                        and r['bbox_xywh_raw'] == a['bbox_xywh_raw'] and r['box_xyxy_raw'] == a['box_xyxy_raw'],
                        'Frozen ann_id/raw box changed')
                b = np.asarray(r['box_xyxy_raw'], np.float64)
                require(b.shape == (4,) and np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1],
                        'Finite positive native box')
        for k in range(8):
            s = slots[8*i+k]
            require((s['frame_index'], s['native_slot'], s['input_id']) == (i, k, f['input_id']),
                    'Original native order/ID')
            require(s['split'] == f['split'] and s['image_sha256'] == f['image_sha256'], 'Query fold/image')
            q = None if f['query8_xy'] is None else f['query8_xy'][k]
            require(s['query_xy'] == q and s['L'] == f['L'], 'Original q/L parity')
            if q is None:
                require(not s['query_exists'] and not s['query_center_valid']
                        and s['input_status'] == 'NO_DETECTION_QUERY_U', 'No query stays U')
            else:
                a = np.asarray(q, np.float64); h, w = f['raw_hw']
                require(a.shape == (2,) and np.isfinite(a).all() and math.isfinite(f['L'])
                        and f['L'] > 0 and s['query_exists'], 'Present finite q/L')
                continuous = bool(0 <= a[0] < w and 0 <= a[1] < h)
                # Original V92 factual bounds used floor(q+.5). Keep that field;
                # this pilot's descriptive nearest readout uses rint separately.
                nx, ny = np.floor(a + .5).astype(np.int64)
                nearest = bool(0 <= nx < w and 0 <= ny < h)
                require(s['continuous_valid'] == continuous and s['nearest_valid'] == nearest
                        and s['query_center_valid'] == (continuous and nearest), 'Original validity changed')
                require(s['input_status'] == ('SAVED_UNLABELED_QUERY' if s['query_center_valid']
                                              else 'OUTSIDE_OR_INVALID_QUERY_U'), 'Original factual U')
            require(s['input_status'] in factual, 'Original input status')
            factual[s['input_status']] += 1
    require(factual == dict(NO_DETECTION_QUERY_U=16, OUTSIDE_OR_INVALID_QUERY_U=22,
                            SAVED_UNLABELED_QUERY=122), 'Frozen U and eligible denominators')
    return factual


def single_native_mask(masks, quality, hw):
    """Exact V81 official F32 binary API conversion; no threshold/argmax."""
    a, s = np.asarray(masks), np.asarray(quality)
    h, w = hw
    require(a.shape == (1, h, w) and a.dtype == np.float32, 'One native F32 API mask')
    require(np.isfinite(a).all() and ((a == 0) | (a == 1)).all(), 'Exact finite binary 0/1')
    require(s.shape == (1,) and np.isfinite(s).all(), 'One finite diagnostic quality')
    return np.ascontiguousarray(a[0].astype(np.bool_)), float(s[0])


def query_sample(mask, original, hw):
    q = original['query_xy']; h, w = hw
    px = py = None
    if q is not None:
        a = np.asarray(q, np.float64)
        require(a.shape == (2,) and np.isfinite(a).all(), 'Finite descriptive q')
        px, py = map(int, np.rint(a))
    bounds = bool(px is not None and 0 <= px < w and 0 <= py < h)
    eligible = original['input_status'] == 'SAVED_UNLABELED_QUERY' and original['query_center_valid']
    result = dict(nearest_method='numpy.rint_ties_to_even', pixel_xy_rint=None if px is None else [px, py],
                  nearest_pixel_bounds_valid=bounds, original_factual_eligible=bool(eligible),
                  candidate_mask_present=mask is not None, mask_value=None,
                  raw_clipped_3x3_true_count=None, raw_clipped_3x3_denominator=None,
                  readout_status='FACTUAL_QUERY_U' if not eligible else 'NO_CANDIDATE_MASK')
    if not eligible or not bounds or mask is None:
        if eligible and not bounds:
            result['readout_status'] = 'RINT_NEAREST_OUTSIDE_U'
        return result
    require(mask.dtype == np.bool_ and mask.shape == (h, w), 'Native bool candidate mask')
    patch = mask[max(0, py-1):min(h, py+2), max(0, px-1):min(w, px+2)]
    result.update(mask_value=bool(mask[py, px]), raw_clipped_3x3_true_count=int(patch.sum()),
                  raw_clipped_3x3_denominator=int(patch.size), readout_status='DESCRIPTIVE_MASK_SAMPLE')
    return result


def mask_diagnostics(mask, box):
    h, w = mask.shape
    x0, y0, x1, y1 = np.asarray(box, np.float64)
    inside = ((np.arange(h)[:, None] >= y0) & (np.arange(h)[:, None] < y1)
              & (np.arange(w)[None, :] >= x0) & (np.arange(w)[None, :] < x1))
    return dict(native_true_area=int(mask.sum()), native_pixel_count=int(mask.size),
                raw_box_intersection_true_area=int((mask & inside).sum()),
                raw_box_outside_true_area=int((mask & ~inside).sum()),
                raw_box_raster_rule='Native integer pixel centres within raw half-open XYXY; no box clipping/prompt adjustment')


def pair_diagnostics(target, competitor):
    if competitor is None:
        return dict(competitor_present=False, intersection=None, union=None, mask_IoU=None,
                    distinct_physical_instance='UNKNOWN')
    require(target.shape == competitor.shape and target.dtype == competitor.dtype == np.bool_, 'Pair native mask shape')
    intersection = int((target & competitor).sum()); union = int((target | competitor).sum())
    return dict(competitor_present=True, intersection=intersection, union=union,
                mask_IoU=None if union == 0 else intersection/union, distinct_physical_instance='UNKNOWN')


def output_request(req, mask_binding, quality, diagnostics, empty, policy, entry):
    called = mask_binding is not None
    require(called == req['call_eligible_metadata'], 'Every eligible primary call/no-call retained')
    require(called or (quality is None and diagnostics is None and empty is None), 'Missing is null, never fake zero mask')
    return dict(req, metadata_request=req, input_metadata_executed=req['executed'],
                executed=called, full_execution_policy_fixed=True, root_execution_policy_fixed=True,
                root_execution_policy_required=False, root_policy=policy, root_entry=entry,
                mask_status='SAVED_CANDIDATE' if called else 'NO_CALL_MISSING_COMPETITOR',
                mask=mask_binding, predicted_quality=quality, selected_mask_index=0 if called else None,
                diagnostics=diagnostics, empty_mask=empty, semantic_D_E_S_created=False, training_admission=False)
