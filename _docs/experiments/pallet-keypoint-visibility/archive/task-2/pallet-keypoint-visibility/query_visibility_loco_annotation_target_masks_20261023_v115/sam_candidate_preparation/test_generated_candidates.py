"""Invented arrays and metadata only. No source/model/weight payload access."""
import argparse
import ast
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import candidate_contract as c
import produce_sam_candidates_once as runner

R = Path(__file__).resolve().parent


def binding(p):
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def fixture():
    frames, slots, requests, entities = [], [], [], []
    bad = 22
    for i in range(20):
        absent = i < 2
        qq = None if absent else [[4.25, 5.25] for _ in range(8)]
        if qq is not None:
            for k in range(8):
                if bad:
                    qq[k] = [-1., 5.25]; bad -= 1
        f = dict(frame_index=i, input_id='GENERATED_%02d' % i, image='/invented/%02d.jpg' % i,
                 image_sha256=hashlib.sha256(str(i).encode()).hexdigest(), raw_hw=[10, 12],
                 decoded_RGB_input=dict(path='/invented/%02d.npy' % i, sha256='a'*64),
                 RGB_pixels_sha256='b'*64, split='TRAIN' if i % 2 else 'VALIDATION',
                 query8_xy=qq, L=None if absent else 8.)
        frames.append(f)
        b = [1., 1., 7., 7.]
        comp = None if i < 7 else dict(ann_id=2000+i, bbox_xywh_raw=b, box_xyxy_raw=[1., 1., 8., 8.])
        e = dict(frame_index=i, input_id=f['input_id'], image_sha256=f['image_sha256'], split=f['split'],
                 existing_target_ann_id=1000+i, bbox_xywh_raw=b, box_xyxy_raw=[1., 1., 8., 8.], competitor=comp)
        entities.append(e)
        for role, a in [('TARGET', dict(ann_id=1000+i, bbox_xywh_raw=b, box_xyxy_raw=[1., 1., 8., 8.])), ('COMPETITOR', comp)]:
            requests.append(dict(request_index=len(requests), request_id='G%02d_%s' % (i, role),
                frame_index=i, role=role, input_id=f['input_id'], split=f['split'], image=f['image'],
                image_sha256=f['image_sha256'], raw_hw=f['raw_hw'], canonical_decoded_RGB=f['decoded_RGB_input'],
                decoded_RGB_pixels_sha256=f['RGB_pixels_sha256'], executed=False,
                point_coords=None, point_labels=None, box_clipping_or_adjustment_applied=False,
                call_eligible_metadata=a is not None, ann_id=None if a is None else a['ann_id'],
                bbox_xywh_raw=None if a is None else a['bbox_xywh_raw'],
                box_xyxy_raw=None if a is None else a['box_xyxy_raw'],
                status='NO_CALL_MISSING_COMPETITOR' if a is None else 'PLANNED_BOX_REQUEST'))
        for k in range(8):
            q = None if qq is None else qq[k]
            ok = q is not None and q[0] >= 0
            slots.append(dict(frame_index=i, native_slot=k, input_id=f['input_id'], split=f['split'],
                image_sha256=f['image_sha256'], query_xy=q, L=f['L'], query_exists=q is not None,
                query_center_valid=ok, continuous_valid=ok, nearest_valid=ok,
                input_status='NO_DETECTION_QUERY_U' if q is None else ('SAVED_UNLABELED_QUERY' if ok else 'OUTSIDE_OR_INVALID_QUERY_U')))
    return frames, slots, requests, entities


def generated_entry():
    return dict(schema='V115_ROOT_SAM_CANDIDATE_EXECUTION_ENTRY_V1', approved=True,
        allow_public_source_SAM_candidate_forward=True, source_only=True, current223=False,
        V112=False, fit=False, PnP=0, expected_frames=20, expected_request_slots=40,
        expected_primary_masks=33, expected_repeat_calls=1, expected_set_image_calls=20,
        expected_query_slots=160, output_directory=str(runner.OUT),
        bindings={k:dict(path=str(p), sha256='a'*64) for k, p in runner.STATIC.items()})


def main(output):
    results = []
    def yes(name, condition=True):
        assert condition, name
        results.append(dict(name=name, PASS=True))
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, AssertionError):
            yes(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    f = fixture()
    yes('generated20_40_160_full_retention', c.validate_cohort(*f) == dict(NO_DETECTION_QUERY_U=16, OUTSIDE_OR_INVALID_QUERY_U=22, SAVED_UNLABELED_QUERY=122))
    changes = [
        ('duplicate_frame_ID', lambda a: a[0][1].update(input_id=a[0][0]['input_id'])),
        ('duplicate_request_ID', lambda a: a[2][1].update(request_id=a[2][0]['request_id'])),
        ('native_slot_order_change', lambda a: a[1][8].update(native_slot=2)),
        ('q_coordinate_change', lambda a: a[1][40].update(query_xy=[4.9, 5.25])),
        ('L_change', lambda a: a[1][40].update(L=9.)),
        ('fold_change', lambda a: a[1][40].update(split='OTHER')),
        ('request_order_change', lambda a: a[2][2].update(request_index=9)),
        ('prompt_box_change', lambda a: a[2][14].update(box_xyxy_raw=[0., 0., 8., 8.])),
        ('annotation_ID_change', lambda a: a[2][14].update(ann_id=999999)),
        ('added_point_prompt', lambda a: a[2][14].update(point_coords=[[4., 4.]])),
        ('missing_competitor_zero_box', lambda a: a[2][1].update(box_xyxy_raw=[0., 0., 0., 0.])),
        ('noquery_promoted_valid', lambda a: a[1][0].update(query_center_valid=True)),
        ('OOB_promoted_valid', lambda a: a[1][16].update(query_center_valid=True)),
        ('source_NPY_substitution', lambda a: a[2][14].update(canonical_decoded_RGB=dict(path='/other.npy', sha256='a'*64))),
    ]
    for name, change in changes:
        a = copy.deepcopy(f); change(a); rejects(name, lambda a=a: c.validate_cohort(*a))
    a = copy.deepcopy(f); a[1].pop(); rejects('dropped_query_rejected', lambda: c.validate_cohort(*a))
    api = np.zeros((1, 5, 7), np.float32); api[0, 0, 0] = 1; api[0, 2, 2] = 1
    mask, quality = c.single_native_mask(api, np.array([.25], np.float32), [5, 7])
    yes('exact_F32_01_to_bool', mask.dtype == np.bool_ and mask.sum() == 2 and quality == .25)
    for name, bad in [('wrong_shape', api[:, :4]), ('multiple_masks', np.repeat(api, 2, 0)),
                      ('F64_encoding', api.astype(np.float64)), ('bool_encoding', api.astype(bool))]:
        rejects(name, lambda bad=bad: c.single_native_mask(bad, np.array([.25], np.float32), [5, 7]))
    bad = api.copy(); bad[0, 0, 1] = .5
    rejects('nonbinary_rejected', lambda: c.single_native_mask(bad, np.array([.25]), [5, 7]))
    bad = api.copy(); bad[0, 0, 1] = np.nan
    rejects('NaN_mask_rejected', lambda: c.single_native_mask(bad, np.array([.25]), [5, 7]))
    rejects('quality_NaN_rejected', lambda: c.single_native_mask(api, np.array([np.nan]), [5, 7]))
    rejects('quality_wrong_shape_rejected', lambda: c.single_native_mask(api, np.array([.2, .3]), [5, 7]))
    s = dict(query_xy=[0., 0.], input_status='SAVED_UNLABELED_QUERY', query_center_valid=True)
    sm = c.query_sample(mask, s, [5, 7])
    yes('corner_clipped3x3_denominator4', sm['raw_clipped_3x3_denominator'] == 4 and sm['raw_clipped_3x3_true_count'] == 1 and sm['mask_value'] is True)
    s['query_xy'] = [2., 2.]; sm = c.query_sample(mask, s, [5, 7])
    yes('interior3x3_denominator9', sm['raw_clipped_3x3_denominator'] == 9 and sm['raw_clipped_3x3_true_count'] == 1)
    s['query_xy'] = [2.5, 2.5]; sm = c.query_sample(mask, s, [5, 7])
    yes('rint_even_tie2_not_floor3', sm['pixel_xy_rint'] == [2, 2])
    s['query_xy'] = [3.5, 1.5]; sm = c.query_sample(mask, s, [5, 7])
    yes('rint_even_tie4_2', sm['pixel_xy_rint'] == [4, 2])
    sm = c.query_sample(None, s, [5, 7])
    yes('missing_mask_null_not_negative', sm['mask_value'] is None and sm['raw_clipped_3x3_denominator'] is None and sm['readout_status'] == 'NO_CANDIDATE_MASK')
    s = dict(query_xy=[-.1, 0.], input_status='OUTSIDE_OR_INVALID_QUERY_U', query_center_valid=False)
    sm = c.query_sample(mask, s, [5, 7])
    yes('continuous_OOB_remains_U_even_rint_inside', sm['nearest_pixel_bounds_valid'] and sm['mask_value'] is None and sm['readout_status'] == 'FACTUAL_QUERY_U')
    s = dict(query_xy=None, input_status='NO_DETECTION_QUERY_U', query_center_valid=False)
    sm = c.query_sample(mask, s, [5, 7])
    yes('noquery_has_no_index_or_sample', sm['pixel_xy_rint'] is None and sm['mask_value'] is None)
    s = dict(query_xy=[6.8, 1.], input_status='SAVED_UNLABELED_QUERY', query_center_valid=True)
    sm = c.query_sample(mask, s, [5, 7])
    yes('rint_OOB_no_clipping', not sm['nearest_pixel_bounds_valid'] and sm['mask_value'] is None)
    pd = c.pair_diagnostics(mask, None)
    yes('missing_competitor_pair_is_null', pd['intersection'] is None and pd['union'] is None and pd['mask_IoU'] is None)
    pd = c.pair_diagnostics(mask, mask)
    yes('pair_intersection_union_exact', pd['intersection'] == pd['union'] == 2 and pd['mask_IoU'] == 1.)
    policy_binding = dict(path='/invented/policy.json', sha256='a'*64)
    oo = c.output_request(f[2][0], dict(path='/invented/mask.npz', sha256='b'*64), .2, {}, False, policy_binding, policy_binding)
    yes('actual_output_execution_and_original_metadata_separate', oo['executed'] is True
        and oo['root_execution_policy_fixed'] is True and oo['full_execution_policy_fixed'] is True
        and oo['metadata_request']['executed'] is False and oo['input_metadata_executed'] is False)
    oo = c.output_request(f[2][1], None, None, None, None, policy_binding, policy_binding)
    yes('missing_output_no_execution_or_fakezero', oo['executed'] is False and oo['mask'] is None
        and oo['empty_mask'] is None and oo['predicted_quality'] is None)
    diag = c.mask_diagnostics(mask, [0., 0., 1., 1.])
    yes('raw_halfopen_box_diag', diag['raw_box_intersection_true_area'] == 1 and diag['raw_box_outside_true_area'] == 1)
    e = generated_entry()
    c.validate_entry_structure(e, runner.STATIC, runner.OUT, runner.PYTHON, runner.PYTHON)
    yes('generated_root_entry_structure_accepts_exact_fields')
    for name, change in [
        ('unapproved_entry', lambda e: e.update(approved=False)),
        ('private_scope', lambda e: e.update(current223=True)),
        ('extra_binding', lambda e: e['bindings'].update(extra=dict(path='/bad', sha256='a'*64))),
        ('changed_static_path', lambda e: e['bindings']['frame_inputs'].update(path='/bad')),
        ('changed_call_count', lambda e: e.update(expected_primary_masks=34)),
        ('changed_output', lambda e: e.update(output_directory='/bad')),
    ]:
        ee = copy.deepcopy(e); change(ee)
        rejects(name, lambda ee=ee: c.validate_entry_structure(ee, runner.STATIC, runner.OUT, runner.PYTHON, runner.PYTHON))
    # Run entry rejection with a poison SHA/payload function: no model modules,
    # output directory, metadata hashing, NPY or checkpoint reads can occur.
    with tempfile.TemporaryDirectory(prefix='V115_generated_') as td:
        ep = Path(td) / 'UNAPPROVED.json'; ee = copy.deepcopy(e); ee['approved'] = False
        ep.write_text(json.dumps(ee))
        old_sha = runner.sha
        def forbidden_sha(*args, **kwargs):
            raise AssertionError('Payload/metadata SHA reached before root guard')
        runner.sha = forbidden_sha
        try:
            try:
                runner.run(ep)
            except ValueError as ex:
                yes('unapproved_run_fails_before_any_other_file_hash_or_payload', 'root entry' in str(ex))
            else:
                raise AssertionError('Unapproved run accepted')
        finally:
            runner.sha = old_sha
        ep = Path(td) / 'APPROVED_BUT_EXISTING_OUTPUT.json'
        ee = copy.deepcopy(e); ee['output_directory'] = td
        ep.write_text(json.dumps(ee))
        old_out = runner.OUT; runner.OUT = Path(td); runner.sha = forbidden_sha
        try:
            try:
                runner.run(ep)
            except ValueError as ex:
                yes('existing_output_fails_before_payload_or_metadata_hash', 'Output must be absent' in str(ex))
            else:
                raise AssertionError('Existing output accepted')
        finally:
            runner.OUT = old_out; runner.sha = old_sha
    yes('no_torch_or_SAM_imported', not any(k == 'torch' or k == 'sam2' or k.startswith('sam2.') for k in sys.modules))
    # AST confirms all model imports are function-local in the guarded run.
    tree = ast.parse((R/'produce_sam_candidates_once.py').read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    yes('module_import_inert', all(not any(x in ast.unparse(n) for x in ['torch', 'sam2', 'PIL']) for n in top_imports))
    out = dict(schema='V115_GENERATED_CANDIDATE_TESTS_V1', completed=True, PASS=True,
        test_count=len(results), tests=results, NumPy=np.__version__,
        bindings=dict(runner=binding(R/'produce_sam_candidates_once.py'), contract=binding(R/'candidate_contract.py'), tests=binding(Path(__file__))),
        actual_source_payload_reads=0, checkpoint_byte_reads=0, model_calls=0,
        private_current_V112_reads=0, fit=0, PnP=0)
    with Path(output).open('x') as ff:
        json.dump(out, ff, indent=2); ff.write('\n')
    print(json.dumps(dict(PASS=True, tests=len(results), output=str(output))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True)
    main(ap.parse_args().output)
