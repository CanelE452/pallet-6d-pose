"""Independent saved-only audit. No producer imports, RGB/weights, or model calls."""
import argparse
import collections
import hashlib
import json
import math
import traceback
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
V = R.parent
O = V / 'source_sam_candidates_v1'
EXPECTED = '714372e11d338730e5da6edb5075ab6a391aa30d4b700f47311f42949779c635'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def binding(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def check_binding(b, inside=None):
    p = Path(b['path']).resolve()
    if inside is not None:
        assert p.is_relative_to(inside.resolve()), 'Saved output escaped its directory'
    assert sha(p) == b['sha256'], str(p)
    return p


def write(p, o):
    with p.open('x') as f:
        json.dump(o, f, indent=2, allow_nan=False); f.write('\n')
    return binding(p)


def sample(mask, original, hw):
    h, w = hw; q = original['query_xy']
    px, py = (None, None) if q is None else (int(round(float(q[0]))), int(round(float(q[1]))))
    bounds = px is not None and 0 <= px < w and 0 <= py < h
    eligible = original['input_status'] == 'SAVED_UNLABELED_QUERY' and original['query_center_valid']
    out = dict(nearest_method='numpy.rint_ties_to_even', pixel_xy_rint=None if px is None else [px, py],
               nearest_pixel_bounds_valid=bool(bounds), original_factual_eligible=bool(eligible),
               candidate_mask_present=mask is not None, mask_value=None,
               raw_clipped_3x3_true_count=None, raw_clipped_3x3_denominator=None,
               readout_status='FACTUAL_QUERY_U' if not eligible else 'NO_CANDIDATE_MASK')
    if not eligible or not bounds or mask is None:
        if eligible and not bounds:
            out['readout_status'] = 'RINT_NEAREST_OUTSIDE_U'
        return out
    cells = [(py+dy, px+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if 0 <= py+dy < h and 0 <= px+dx < w]
    out.update(mask_value=bool(mask[py, px]), raw_clipped_3x3_true_count=sum(bool(mask[y, x]) for y, x in cells),
               raw_clipped_3x3_denominator=len(cells), readout_status='DESCRIPTIVE_MASK_SAMPLE')
    return out


def areas(mask, box):
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = box
    count = sum(x0 <= int(x) < x1 and y0 <= int(y) < y1 for y, x in zip(ys, xs))
    return dict(native_true_area=len(xs), native_pixel_count=mask.size,
                raw_box_intersection_true_area=count, raw_box_outside_true_area=len(xs)-count)


def load_mask(b, hw):
    p = check_binding(b, O)
    with np.load(p, allow_pickle=False) as z:
        assert z.files == ['visible_mask']
        a = z['visible_mask']
    assert a.dtype == np.bool_ and a.shape == tuple(hw)
    return a


def generated():
    a = np.zeros((3, 4), bool); a[0, 0] = True; a[1, 2] = True
    s = dict(query_xy=[0., 0.], input_status='SAVED_UNLABELED_QUERY', query_center_valid=True)
    r = sample(a, s, [3, 4]); assert r['raw_clipped_3x3_denominator'] == 4 and r['raw_clipped_3x3_true_count'] == 1
    s['query_xy'] = [2.5, .5]; assert sample(a, s, [3, 4])['pixel_xy_rint'] == [2, 0]
    assert sample(None, s, [3, 4])['mask_value'] is None
    s.update(query_xy=[-.1, 0.], input_status='OUTSIDE_OR_INVALID_QUERY_U', query_center_valid=False)
    assert sample(a, s, [3, 4])['mask_value'] is None
    assert areas(a, [0., 0., 1., 1.]) == dict(native_true_area=2, native_pixel_count=12, raw_box_intersection_true_area=1, raw_box_outside_true_area=1)
    return dict(PASS=True, fixtures=5, source_payload_reads=0, model_calls=0)


def audit(destination):
    assert not destination.exists(), 'Preserve old audit; no overwrite'
    destination.mkdir()
    try:
        sp = O / 'SAM_CANDIDATE_OUTPUT_SEAL.json'
        assert sha(sp) == EXPECTED
        seal = read(sp); assert seal['completed'] and seal['PASS']
        for b in seal['outputs']:
            check_binding(b, O)
        assert len({b['path'] for b in seal['outputs']}) == len(seal['outputs'])
        entry = read(check_binding(seal['root_entry']))
        assert entry['approved'] is True and entry['allow_public_source_SAM_candidate_forward'] is True
        assert entry['bindings'] == seal['bindings']
        assert not (O / 'FAILED.json').exists()
        # Do not recurse into checkpoint/NPY payload bindings.
        keys = ('root_policy', 'metadata_receipt', 'metadata_seal', 'request_plan', 'frame_inputs',
                'query_inputs', 'target_entities', 'witnesses', 'code', 'contract', 'generated', 'protocol', 'runtime')
        for k in keys:
            check_binding(seal['bindings'][k])
        plan = rows(seal['bindings']['request_plan']['path'])
        frames = rows(seal['bindings']['frame_inputs']['path'])
        original = rows(seal['bindings']['query_inputs']['path'])
        rr = rows(check_binding(seal['output_requests'], O))
        qq = rows(check_binding(seal['query_readouts'], O))
        ff = rows(O / 'DECODED_FRAME_LINEAGE20.jsonl')
        pp = rows(O / 'TARGET_COMPETITOR_DIAGNOSTICS20.jsonl')
        summary = read(check_binding(seal['summary'], O))
        assert len(plan) == len(rr) == 40 and len(original) == len(qq) == 160 and len(ff) == len(pp) == len(frames) == 20
        masks = {}; role_counts = collections.defaultdict(collections.Counter); pair_replay = []
        actual_calls = empty = 0
        for j, (r, req) in enumerate(zip(rr, plan)):
            assert req['request_index'] == r['request_index'] == j and r['metadata_request'] == req
            changed = {'executed', 'full_execution_policy_fixed', 'root_execution_policy_required'}
            assert all(r[k] == v for k, v in req.items() if k not in changed)
            called = req['call_eligible_metadata']
            assert r['executed'] is called and r['input_metadata_executed'] is False
            assert r['root_execution_policy_fixed'] is True and r['full_execution_policy_fixed'] is True
            assert r['root_execution_policy_required'] is False
            assert r['root_policy'] == seal['bindings']['root_policy'] and r['root_entry'] == seal['root_entry']
            assert r['point_coords'] is None and r['point_labels'] is None
            assert r['semantic_D_E_S_created'] is False and r['training_admission'] is False
            if not called:
                assert r['mask_status'] == 'NO_CALL_MISSING_COMPETITOR'
                assert all(r[k] is None for k in ['mask', 'predicted_quality', 'selected_mask_index', 'diagnostics', 'empty_mask'])
                masks[(req['frame_index'], req['role'])] = None
                continue
            a = load_mask(r['mask'], req['raw_hw']); masks[(req['frame_index'], req['role'])] = a
            actual_calls += 1; empty += not bool(a.any())
            assert r['mask_status'] == 'SAVED_CANDIDATE' and r['selected_mask_index'] == 0
            assert math.isfinite(r['predicted_quality']) and r['empty_mask'] == (not bool(a.any()))
            expected = areas(a, req['box_xyxy_raw'])
            assert all(r['diagnostics'][k] == v for k, v in expected.items())
            assert r['diagnostics']['mask_payload_bytes'] == Path(r['mask']['path']).stat().st_size
            assert math.isfinite(r['diagnostics']['predict_seconds']) and r['diagnostics']['predict_seconds'] >= 0
        assert actual_calls == 33 and sum(r['mask'] is None for r in rr) == 7
        for i, (f, lineage, p) in enumerate(zip(frames, ff, pp)):
            assert lineage['frame_index'] == i and lineage['set_image_call_index'] == i
            assert all(lineage[k] == f[k] for k in ['input_id', 'split', 'raw_hw'])
            assert lineage['canonical_decoded_RGB'] == f['decoded_RGB_input']
            assert lineage['verified_RGB_pixel_sha256'] == f['RGB_pixels_sha256']
            assert lineage['original_JPEG_sha256'] == f['image_sha256']
            assert math.isfinite(lineage['set_image_seconds']) and lineage['set_image_seconds'] >= 0
            a, b = masks[(i, 'TARGET')], masks[(i, 'COMPETITOR')]
            assert p['frame_index'] == i and p['input_id'] == f['input_id']
            if b is None:
                assert p['competitor_present'] is False and all(p[k] is None for k in ['intersection', 'union', 'mask_IoU'])
            else:
                intersection = int(np.count_nonzero(a & b)); union = int(np.count_nonzero(a | b))
                assert p['competitor_present'] is True and p['intersection'] == intersection and p['union'] == union
                assert p['mask_IoU'] == (intersection/union if union else None)
            assert p['distinct_physical_instance'] == p['initializer_target_equivalence'] == 'UNKNOWN'
            pair_replay.append(dict(frame_index=i, input_id=f['input_id'], split=f['split'],
                target_area=int(a.sum()), competitor_area=None if b is None else int(b.sum()),
                intersection=p['intersection'], union=p['union'], mask_IoU=p['mask_IoU']))
        factual = collections.Counter(); overlap = collections.Counter(); by_fold = collections.defaultdict(collections.Counter)
        for j, (r, s) in enumerate(zip(qq, original)):
            assert all(r[k] == v for k, v in s.items()), 'Original query metadata changed at %d' % j
            assert (s['frame_index'], s['native_slot']) == (j//8, j%8)
            assert r['semantic_D_E_S_created'] is False and r['native_expected_state'] == 'UNKNOWN'
            assert r['human_gold'] is False and r['training_admission'] is False
            f = frames[j//8]; factual[s['input_status']] += 1
            for role, key in [('TARGET', 'target_candidate'), ('COMPETITOR', 'competitor_candidate')]:
                expected = sample(masks[(j//8, role)], s, f['raw_hw'])
                assert r[key] == expected, 'Independent nearest/3x3 mismatch'
                if s['input_status'] == 'SAVED_UNLABELED_QUERY':
                    bucket = 'MASK_TRUE' if expected['mask_value'] is True else ('MASK_FALSE' if expected['mask_value'] is False else expected['readout_status'])
                    role_counts[role][bucket] += 1; by_fold[s['split']][role + ':' + bucket] += 1
            a = r['target_candidate']['mask_value']; b = r['competitor_candidate']['mask_value']
            both = None if a is None or b is None else a and b
            assert r['both_candidate_masks_true_at_q'] == both
            if s['input_status'] == 'SAVED_UNLABELED_QUERY':
                overlap['both_true' if both is True else ('both_defined_not_both_true' if both is False else 'competitor_or_sample_missing')] += 1
        assert factual == dict(NO_DETECTION_QUERY_U=16, OUTSIDE_OR_INVALID_QUERY_U=22, SAVED_UNLABELED_QUERY=122)
        rep = read(O / 'repeat_diagnostics/REPEAT_CHECK.json')
        first = next(r for r in rr if r['role'] == 'TARGET' and r['executed'])
        assert rep['request_id'] == first['request_id'] and rep['primary_mask'] == first['mask']
        ra = load_mask(rep['repeat_mask'], frames[first['frame_index']]['raw_hw'])
        pa = masks[(first['frame_index'], 'TARGET')]
        assert np.array_equal(ra, pa) and rep['mask_bitexact'] is True
        diff = abs(rep['primary_quality']-rep['repeat_quality'])
        assert rep['primary_quality'] == first['predicted_quality'] and diff == rep['quality_abs_difference'] and diff <= 1e-6
        assert rep['same_embedding'] is True and rep['replaces_primary'] is False
        assert summary['first_target_repeat'] == rep and summary['model_state_before'] == summary['model_state_after']
        assert summary['model_unchanged'] is True and summary['original_factual_counts'] == dict(factual)
        for k, val in dict(frames=20, request_slots=40, primary_masks=33, missing_competitor_no_calls=7,
                           repeat_predict_calls=1, total_predict_calls=34, set_image_calls=20, original_query_slots=160).items():
            assert summary[k] == val
        assert summary['empty_primary_masks'] == empty
        assert all(summary[k] is False for k in ['compile', 'postprocessing', 'TF32', 'multimask_output',
                                                'return_logits', 'point_prompts', 'quality_selection',
                                                'human_gold', 'semantic_D_E_S_created', 'training_admission', 'current223', 'V112', 'fit'])
        assert summary['weights'] == 'FP32' and summary['autocast'] == 'CUDA BF16' and summary['mask_threshold'] == 0. and summary['PnP'] == 0
        pair_binding = write(destination/'PAIR20_MASK_GEOMETRY_REPLAY.json', pair_replay)
        report = dict(schema='V115_INDEPENDENT_SAVED_CANDIDATE_MASK_AUDIT_V1', completed=True, PASS=True,
            output_seal=binding(sp), scope=binding(R/'AUDIT_SCOPE_BEFORE_OUTPUT_READS.json'), code=binding(Path(__file__)),
            counts=dict(frames=20, request_outputs=40, independently_read_bool_masks=33, null_no_calls=7,
                        query_records=160, factual_U=38, eligible_query_records=122, empty_masks=empty),
            factual_input_counts=dict(factual), eligible122_per_role_mask_readouts={k:dict(v) for k,v in role_counts.items()},
            eligible122_overlap_descriptions=dict(overlap), per_fold_descriptive_readouts={k:dict(v) for k,v in by_fold.items()},
            pair_mask_geometry_replay=pair_binding, frame_pairs_with_positive_mask_overlap=sum((x['intersection'] or 0)>0 for x in pair_replay),
            repeat=dict(mask_arrays_bitexact=True, NPZ_file_SHA_equal=first['mask']['sha256']==rep['repeat_mask']['sha256'],
                        quality_abs_difference=diff, primary_never_replaced=True),
            producer_bound_certificates=dict(state_unchanged=summary['model_state_before']==summary['model_state_after'],
                call_counts='20 set_image,33 primary+1 predetermined repeat; checked saved receipts, not re-executed',
                RGB_lineage='20 saved declarations match original bindings; RGB payloads not read/rehashed',
                official_API='Saved bool encodings checked; original F32 API tensors and model state not independently reconstructed'),
            independently_recomputed='All original metadata joins, mask shape/dtype/SHA/area/raw-box intersections, pair union/IoU, all160 nearest/3x3/missingness values and saved repeat arrays',
            semantics='Candidate annotation-entity mask support only; overlaps may reflect merged/duplicate entities or segmentation ambiguity. No accuracy, true ownership, external E, self S, coordinate-correctness or native-gold claim.',
            original_output_mutations=0, RGB_reads=0, checkpoint_reads=0, model_forwards=0, current_V112_reads=0,
            fit=0, PnP=0, training_admission=False, native_mapping='UNKNOWN', human_gold=False)
        rb = write(destination/'REVIEW.json', report)
        sb = write(destination/'SEAL.json', dict(completed=True, PASS=True, review=rb, output_seal=binding(sp),
                   code=binding(Path(__file__)), scope=binding(R/'AUDIT_SCOPE_BEFORE_OUTPUT_READS.json'),
                   pair_replay=pair_binding, model_calls=0, RGB_reads=0, training_admission=False))
        print(json.dumps(dict(PASS=True, review=rb, seal=sb, readouts=report['eligible122_per_role_mask_readouts'], overlap=dict(overlap)), indent=2))
    except BaseException as ex:
        write(destination/'FAILED.json', dict(PASS=False, error=str(ex), type=type(ex).__name__, traceback=traceback.format_exc(),
              code=binding(Path(__file__)), no_output_mutation=True))
        raise


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated', action='store_true');ap.add_argument('--output', type=Path)
    args=ap.parse_args()
    if args.generated:
        print(json.dumps(generated()))
    else:
        assert args.output is not None
        audit(args.output)
