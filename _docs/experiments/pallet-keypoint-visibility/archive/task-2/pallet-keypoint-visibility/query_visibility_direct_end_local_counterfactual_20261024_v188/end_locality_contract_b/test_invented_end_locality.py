"""Invented CPU geometry only; exact V90 crop AST runs on a 2x2 toy RGB."""
from pathlib import Path
import ast
import copy
import json
import math
import numpy as np
import finite_end_locality_contract as m

H = Path(__file__).absolute().parent
P = H.parent.parent


def run():
    tests = []
    def check(name, condition):
        assert condition, name
        tests.append({'name': name, 'PASS': True})
    original = dict(development_role='SOURCE_TRAIN', native_slot=0, original_query_available=True,
                    query_xy=[47.5, 47.5], L=96., raw_hw=[96, 96], original_semantic_state='D',
                    original_row_id='INVENTED/native0', original_quality=-1)
    endpoint = dict(authority=m.REAL_AUTHORITY, native_slot=0, end_locality_id='INVENTED_DECK_END',
                    selected_instance_supported=True, intended_part_supported=True,
                    queried_end_localization_supported=True, all_plausible_candidates_enclosed=True,
                    before_end_visible_supported=True, candidate_enclosures_xyxy=[[29.25, 29.25, 30.25, 30.25]],
                    evidence_binding={'path':'INVENTED_NOT_OPENED','sha256':'a'*64},
                    surrounding_material_witnesses=[dict(raw_pixel_xy=[50, 50], material_observed=True,
                        same_selected_part_supported=True, evidence_binding={'invented':True})])
    placement = dict(cover_bbox_xyxy=[28, 28, 33, 33], sham_translation_xy=[12, 0])
    support, invalid = m.exact96_support(original['query_xy'], original['L'], original['raw_hw'])
    check('original_pixel_center_grid_integer_fixture',len(support)==9216 and math.isclose(sum(support.values()),1.,abs_tol=1e-12) and invalid==0)
    check('zero_weight_taps_excluded',m.rgb_taps([30.,30.],[96,96])==[((30,30),1.)])
    v90 = P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
    crop = next(n for n in ast.parse(v90.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop')
    ns={'np':np};exec(compile(ast.Module(body=[crop],type_ignores=[]),str(v90)+'::crop','exec'),ns)
    toy=np.repeat(np.array([[0,100],[200,255]],dtype=np.uint8)[...,None],3,axis=2)
    rgb,valid=ns['crop'](toy,[.25,.25],1.,1)
    acc=math.fsum(float(toy[y,x,0])*weight for (x,y),weight in m.rgb_taps([.25,.25],[2,2]))
    check('actual_V90_four_tap_uint8_rounding_not_nearest',int(rgb[0,0,0])==math.floor(acc+.5)==72 and valid.all() and m.nearest_diagnostic([.25,.25],[2,2])==(0,0))
    check('invalid_output_sample_entirely_zero_support',m.rgb_taps([-.51,0.],[2,2])==[(None,1.)])
    p,outside=m.candidate_footprint([[29.25,29.25,30.25,30.25]],[96,96])
    check('continuous_candidate_conservative_all_positive_taps',p=={(x,y) for x in (29,30,31) for y in (29,30,31)} and not outside)
    exact,_=m.candidate_footprint([[30.,30.,30.,30.]],[96,96])
    check('integer_point_no_spurious_neighbor_taps',exact=={(30,30)})
    plan=m.evaluate_plan(original,endpoint,placement)
    check('end_only_cover_geometry_with_surrounding_material_retained',plan['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING' and all(plan['geometry_controls'].values()))
    check('geometry_alone_never_creates_native_D_E_or_admission',set(plan['direct_end_weak_states'].values())=={'UNKNOWN'} and plan['new_native_visibility_target']==plan['new_quality_target']==-1 and not plan['training_admission'])
    unknown=copy.deepcopy(endpoint);unknown['all_plausible_candidates_enclosed']=False
    check('old_broad_coarse_box_without_endpoint_authority_stays_unknown',m.evaluate_plan(original,unknown,placement)['status']=='UNKNOWN')
    u=copy.deepcopy(original);u['original_query_available']=False
    check('original_input_U_retained',m.evaluate_plan(u,endpoint,placement)['reasons']==['ORIGINAL_INPUT_U'])
    far=copy.deepcopy(endpoint);far['candidate_enclosures_xyxy']=[[150.,150.,151.,151.]]
    result=m.evaluate_plan(original,far,placement)
    check('endpoint_outside_original_crop_is_UNKNOWN_not_E',result['status']=='UNKNOWN' and set(result['direct_end_weak_states'].values())=={'UNKNOWN'})
    alternatives=copy.deepcopy(endpoint);alternatives['candidate_enclosures_xyxy'].append([60.,60.,61.,61.])
    check('uncovered_alternative_not_removed_to_pass',m.evaluate_plan(original,alternatives,placement)['status']=='GEOMETRY_FAILURE_RETAINED')
    sham=copy.deepcopy(placement);sham['sham_translation_xy']=[0,0]
    check('SHAM_over_endpoint_fails_without_retry',m.evaluate_plan(original,endpoint,sham)['status']=='GEOMETRY_FAILURE_RETAINED')
    patch=copy.deepcopy(endpoint);patch['surrounding_material_witnesses'][0]['raw_pixel_xy']=[29,29]
    check('whole_patch_cover_without_retained_surrounding_witness_fails',m.evaluate_plan(original,patch,placement)['status']=='GEOMETRY_FAILURE_RETAINED')
    before=copy.deepcopy(original);m.evaluate_plan(original,endpoint,placement)
    check('original_q_L_native_semantic_quality_unchanged',original==before)
    keys=('original_q_L_native_flags_and_all8_inputs_unchanged','before_original_RGB96_valid96_target_context_exact',
          'every_candidate_positive_RGB_tap_cover_alpha_one','every_candidate_positive_RGB_tap_sham_alpha_zero',
          'distinct_foreign_material_supported','foreign_in_front_paste_composite_or_source_order',
          'same_transformed_foreign_RGB_alpha_cover_and_sham','equal_positive_alpha_pasted_mass',
          'background_outside_paste_exact','original_crop_cover_RGB_bytes_changed',
          'original_crop_sham_has_donor_RGB_support','certified_surrounding_material_RGB_unchanged_both')
    proof={k:True for k in keys};proof['evidence_bindings']=[{'invented':True}]
    closed=m.close_actual_proof(plan,proof)
    check('surrounding_patch_visible_but_end_COVER_state_separate',closed['direct_end_weak_states']['COVER']=='EXTERNALLY_BLOCKED_WEAK' and closed['broad_support_is_not_native_decoder'])
    shifted=copy.deepcopy(original);shifted['center_owner_diagnostic']='BACKGROUND'
    direct=m.close_actual_proof(m.evaluate_plan(shifted,endpoint,placement),proof)
    check('center_BG_or_SHAM_coordinate_offset_can_leave_end_visible',direct['direct_end_weak_states']['SHAM']=='VISIBLE_WEAK' and direct['original_input']['center_owner_diagnostic']=='BACKGROUND')
    incomplete=copy.deepcopy(proof);incomplete['every_candidate_positive_RGB_tap_cover_alpha_one']=False
    check('partial_alpha_cannot_label_end_external',m.close_actual_proof(plan,incomplete)['status']=='PIXEL_PROOF_FAILURE_RETAINED')
    quads=m.near_anchor_quads([1.,2.,3.],[1,-1,1],.02)
    check('synthetic_three_end_local_quads_fixed_nearest_witness_fraction',len(quads)==3 and all(q[0]==[1.,2.,3.] for q in quads) and math.isclose(max(abs(x-v) for q in quads for point in q for x,v in zip(point,[1.,2.,3.])),.004,abs_tol=1e-15))
    for native in range(8):
        r=copy.deepcopy(original);r['native_slot']=native;c=copy.deepcopy(endpoint);c['native_slot']=native
        assert m.evaluate_plan(r,c,placement)['original_input']['native_slot']==native
    check('all_native8_preserved_without_role_inference',True)
    return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,
                source_asset_RGB_NPZ_header_depth_mask_mesh_or_model_reads=0,actual_source_labels_or_training=0)


if __name__=='__main__':
    result=run()
    with (H/'GENERATED_INVENTED_END_LOCALITY_CHECKS.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
