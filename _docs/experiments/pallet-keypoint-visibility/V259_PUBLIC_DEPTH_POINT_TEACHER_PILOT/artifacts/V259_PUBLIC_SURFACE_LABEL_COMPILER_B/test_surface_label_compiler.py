"""Analytic invented arrays only; no frame, file, pose or model input."""
import copy
import json
import numpy as np
from surface_label_compiler import (EXPECTED_POLICY, POLICY_SHA256, compile_surface_labels,
    emulated_query_records, reference_grid_records, reference_grid_xy, validate_policy)


def main():
    checks = []
    shape = (25, 25)
    one = np.ones(shape, np.float64)
    target = np.ones(shape, bool)
    background = np.zeros(shape, bool)
    def run(obs=one, ren=one, sam=target):
        return compile_surface_labels(obs, ren, sam, copy.deepcopy(EXPECTED_POLICY))
    def rejected(fn):
        try: fn()
        except ValueError: return
        raise AssertionError('Invalid contract accepted')
    d = run()
    assert d['labels'].dtype == np.int8 and d['counts']['labels'] == {'D':225, 'E':0, 'U':400}
    assert d['labels'][12,12] == 0 and d['labels'][4,12] == -1 and d['reasons']['image_edge'][4,12]
    checks.append('clean_D_full11_and_edge_U')
    e = run(obs=one*.75, sam=background)
    assert e['counts']['labels'] == {'D':0, 'E':225, 'U':400}
    assert e['raw_center_residual_m'][12,12] == -.25
    checks.append('closer_external_E_uncorrected_raw_residual')
    same = run(sam=background)
    assert (same['labels'] == -1).all() and same['reasons']['E_residual_not_closer'][12,12]
    assert not same['counts']['SELF_inferred']
    checks.append('same_depth_masknegative_U_never_SELF')
    for bad in (0., -1., float('nan'), float('inf')):
        obs = one.copy(); obs[12,12] = bad
        z = run(obs=obs)
        assert z['labels'][12,12] == -1 and z['reasons']['observed_invalid'][12,12]
    nan = run(obs=np.full(shape, np.nan))
    assert np.isnan(nan['raw_center_residual_m'][12,12])
    assert all(r['r_raw_observed_minus_render_m'] is None for r in reference_grid_records(nan, EXPECTED_POLICY))
    checks.append('missing_invalid_observed_U_and_raw_nonfinite_preserved')
    ren = one.copy(); ren[12,12] = 0
    z = run(ren=ren)
    assert z['labels'][12,12] == -1 and z['reasons']['render_invalid'][12,12]
    checks.append('render_missing_boundary_U')
    far = run(obs=one*1.25, sam=background)
    assert far['labels'][12,12] == -1 and far['reasons']['E_residual_not_closer'][12,12]
    checks.append('farther_background_U')
    sam = target.copy(); sam[:,12:] = False
    mixed = run(sam=sam)
    assert mixed['labels'][12,12] == -1 and mixed['reasons']['sam_window_mixed'][12,12]
    checks.append('SAM_boundary_U_fullwindow_requirement')
    obs = one.copy(); obs[12,12] = 1.125
    z = run(obs=obs)
    assert z['labels'][12,12] == -1 and z['reasons']['observed_discontinuity'][12,12]
    ren = one.copy(); ren[12,12] = 1.125
    z = run(ren=ren)
    assert z['labels'][12,12] == -1 and z['reasons']['render_discontinuity'][12,12]
    checks.append('observed_and_render_depth_discontinuities_U')
    obs = one.copy(); obs[12,12] = 1.03125
    assert run(obs=obs)['labels'][12,12] == 0
    obs[12,12] = 1.0625
    assert run(obs=obs)['labels'][12,12] == -1
    checks.append('raw_stability_and_residual_thresholds_no_bias_or_epsilon')
    sam = target.copy(); sam[12,18] = False
    distinct = run(sam=sam)
    assert distinct['labels'][12,12] == 0
    qrows = emulated_query_records(distinct, sam, EXPECTED_POLICY)
    row = next(q for q in qrows if q['r_xy'] == [12,12] and q['q_offset_xy'] == [6,0])
    assert row['r_derived_label'] == 0 and row['r_derived_state'] == 'D'
    assert row['q_xy'] == [18,12] and row['q_sam_target_present'] is False
    assert row['q_pixel_DE_gold'] is None and row['r_label_copied_as_query_label'] is False
    assert row['native_slot'] is None and row['SELF_state'] == 'UNKNOWN'
    assert row['q_is_public_emulation'] and not row['q_is_actual_initializer']
    checks.append('r_D_and_q_masknegative_are_distinct_no_query_gold')
    outside = next(q for q in qrows if q['r_xy'] == [4,4] and q['q_offset_xy'] == [-6,0])
    assert not outside['q_inside_image'] and outside['q_sam_target_present'] is None
    grid = reference_grid_xy(shape, EXPECTED_POLICY)
    assert grid.tolist() == [[4,4],[12,4],[20,4],[4,12],[12,12],[20,12],[4,20],[12,20],[20,20]]
    assert len(qrows) == len(grid)*9 and qrows[:9][0]['q_offset_xy'] == [0,0]
    checks.append('fixed_stride8_offset4_nine_offsets_ymajor_outside_unknown')
    tiny = compile_surface_labels(np.ones((5,5),np.float32), np.ones((5,5),np.float32), np.ones((5,5),bool), EXPECTED_POLICY)
    assert tiny['counts']['labels'] == {'D':0, 'E':0, 'U':25}
    checks.append('image_smaller_than_full11_all_U')
    changed = copy.deepcopy(EXPECTED_POLICY); changed['parameters']['radius_pixels'] = 5.0
    rejected(lambda: validate_policy(changed))
    changed = copy.deepcopy(EXPECTED_POLICY); changed['parameters']['unknown_override'] = True
    rejected(lambda: validate_policy(changed))
    rejected(lambda: compile_surface_labels(one.astype(np.uint16), one, target, EXPECTED_POLICY))
    rejected(lambda: compile_surface_labels(one, one, target.astype(np.uint8), EXPECTED_POLICY))
    checks.append('exact_typed_policy_and_depth_units_and_bool_contract')
    before = (one.tobytes(), target.tobytes())
    again = run()
    assert before == (one.tobytes(),target.tobytes()) and np.array_equal(d['labels'],again['labels'])
    assert d['counts'] == again['counts'] and all(np.array_equal(d['reasons'][k],again['reasons'][k]) for k in d['reasons'])
    json.dumps({'counts':nan['counts'],'reference_rows':reference_grid_records(nan,EXPECTED_POLICY),'q_rows':qrows},allow_nan=False)
    checks.append('deterministic_no_input_mutation_JSON_ready_counts_and_records')
    print(json.dumps({'completed':True,'PASS':True,'policy_sha256':POLICY_SHA256,
      'analytic_invented_groups':len(checks),'checks':checks,
      'actual_public_or_protected_frame_array_model_reads':0,'new_labels_applied_to_original_native':False,
      'optimizer_or_model_forward':0},sort_keys=True,indent=2))


if __name__ == '__main__': main()
