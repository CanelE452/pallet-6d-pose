"""Invented pure scalar fixtures; pinned crop function AST on invented RGB only."""
from pathlib import Path
import ast,copy,importlib.util,json,math
import numpy as np
import finite_end_locality_contract as m

H=Path(__file__).absolute().parent
V90=H.parent.parent/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'

def scalar_crop(rgb,q,L,n=96):
    out=np.zeros((3,n,n),np.uint8);valid=np.zeros((1,n,n),bool)
    h,w=rgb.shape[:2]
    for j in range(n):
        y=q[1]+((j+.5)/n-.5)*L
        for i in range(n):
            x=q[0]+((i+.5)/n-.5)*L
            valid[0,j,i]=-.5<=x<w-.5 and -.5<=y<h-.5
            if not valid[0,j,i]:continue
            ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy
            for c in range(3):
                a=0.
                for ax,ay,z in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                    if 0<=ix+ax<w and 0<=iy+ay<h:a+=float(rgb[iy+ay,ix+ax,c])*z
                out[c,j,i]=math.floor(min(255.,max(0.,a))+.5)
    return out,valid


def fixtures():
    checks=[]
    def check(name,v):
        assert bool(v),name
        checks.append(dict(name=name,PASS=True))
    original=dict(development_role='SOURCE_DEVELOPMENT',native_slot=0,query_xy=[256.,256.],L=384.,raw_hw=[512,512],original_query_available=True)
    endpoint=dict(authority=m.REAL_AUTHORITY,native_slot=0,end_locality_id='invented-end',selected_instance_supported=True,intended_part_supported=True,queried_end_localization_supported=True,all_plausible_candidates_enclosed=True,before_end_visible_supported=True,candidate_enclosures_xyxy=[[255.,255.,261.,261.]],evidence_binding={'invented':True},surrounding_material_witnesses=[dict(raw_pixel_xy=[298,258],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':True})])
    placement=dict(cover_bbox_xyxy=[252,252,265,265],sham_translation_xy=[32,0])
    p=m.evaluate_plan(original,endpoint,placement)
    check('L384_tap_holes_do_not_reject_in_extent_sampled_endpoint',p['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING' and not p['end_candidate_fully_in_rgb96_support'] and p['unsampled_candidate_raw_pixels_retained_for_opacity']>0)
    check('exact_original_sample_centre_extent',m.sample96_extent([256.,256.],384.)==[66.,66.,446.,446.])
    check('per_candidate_positive_valid_tap_intersection',p['sampled_domain_relation']['candidates'][0]['positive_valid_RGB96_sample_intersections']>0)
    hole=copy.deepcopy(endpoint);hole['candidate_enclosures_xyxy']=[[256.,256.,256.,256.]]
    check('inside_extent_but_no_sampled_end_remains_UNKNOWN',m.evaluate_plan(original,hole,placement)['status']=='UNKNOWN')
    outside=copy.deepcopy(endpoint);outside['candidate_enclosures_xyxy']=[[447.,255.,447.,261.]]
    check('outside_continuous_extent_UNKNOWN_not_external',m.evaluate_plan(original,outside,placement)['status']=='UNKNOWN')
    alt=copy.deepcopy(endpoint);alt['candidate_enclosures_xyxy'].append([256.,256.,256.,256.])
    check('one_unsampled_alternative_cannot_be_dropped',m.evaluate_plan(original,alt,placement)['status']=='UNKNOWN')
    edge=m.sampled_candidate_relation([256.,256.],384.,[512,512],[[66.,66.,66.,66.]])
    check('closed_extent_boundary_and_zero_weight_taps',edge['all_enclosures_inside_original96_sample_extent'] and edge['every_enclosure_has_positive_valid_original96_sample'] and m.rgb_taps((66.,66.),[512,512])==[((66,66),1.)])
    frac=m.sampled_candidate_relation([256.25,256.25],384.,[512,512],[[258.5,258.5,258.6,258.6]])
    check('fractional_positive_four_tap_support',frac['every_enclosure_has_positive_valid_original96_sample'])
    badw=copy.deepcopy(endpoint);badw['surrounding_material_witnesses'][0]['raw_pixel_xy']=[300,260]
    check('surrounding_witness_raw_tap_hole_still_fails_no_replacement',m.evaluate_plan(original,badw,placement)['status']=='GEOMETRY_FAILURE_RETAINED')
    u=copy.deepcopy(original);u['original_query_available']=False
    check('original_U_preserved',m.evaluate_plan(u,endpoint,placement)['reasons']==['ORIGINAL_INPUT_U'])
    weak=copy.deepcopy(endpoint);weak['queried_end_localization_supported']=False
    check('bbox_crop_presence_without_end_authority_stays_UNKNOWN',m.evaluate_plan(original,weak,placement)['status']=='UNKNOWN')
    names=('original_q_L_native_flags_and_all8_inputs_unchanged','before_original_RGB96_valid96_target_context_exact','every_candidate_positive_RGB_tap_cover_alpha_one','every_candidate_positive_RGB_tap_sham_alpha_zero','distinct_foreign_material_supported','foreign_in_front_paste_composite_or_source_order','same_transformed_foreign_RGB_alpha_cover_and_sham','equal_positive_alpha_pasted_mass','background_outside_paste_exact','original_crop_cover_RGB_bytes_changed','original_crop_sham_has_donor_RGB_support','end_related_original96_COVER_bytes_changed','end_related_original96_SHAM_bytes_unchanged','certified_surrounding_material_RGB_unchanged_both')
    proof={k:True for k in names};proof['evidence_bindings']={'invented':True}
    check('full_opacity_even_unsampled_taps_mandatory',m.close_actual_proof(p,{**proof,'every_candidate_positive_RGB_tap_cover_alpha_one':False})['status']=='PIXEL_PROOF_FAILURE_RETAINED')
    check('SHAM_end_bytes_equality_mandatory',m.close_actual_proof(p,{**proof,'end_related_original96_SHAM_bytes_unchanged':False})['status']=='PIXEL_PROOF_FAILURE_RETAINED')
    check('opacity_without_COVER_end_RGB_change_stays_UNKNOWN',m.close_actual_proof(p,{**proof,'end_related_original96_COVER_bytes_changed':False})['status']=='PIXEL_PROOF_FAILURE_RETAINED')
    check('complete_proof_weak_only_native_quality_admission_unchanged',m.close_actual_proof(p,proof)['direct_end_weak_states']['COVER']=='EXTERNALLY_BLOCKED_WEAK' and m.close_actual_proof(p,proof)['new_native_visibility_target']==-1 and not m.close_actual_proof(p,proof)['training_admission'])
    tree=ast.parse(V90.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='crop');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_V90_crop_AST','exec'),ns)
    yy,xx=np.indices((41,53));rgb=np.stack(((xx*7+yy*3)%251,(xx*3+yy*11)%241,(xx*13+yy*2)%239),-1).astype(np.uint8)
    cases=[([26.,20.],32.),([26.25,20.75],64.),([-.25,-.25],12.),([52.25,40.25],24.),([26.,20.],384.)]
    for i,(q,L) in enumerate(cases):
        a,b=scalar_crop(rgb,q,L);c,d=ns['crop'](rgb,q,L,96)
        check('independent_scalar_byte_parity_original_V90_case_'+str(i),a.tobytes()==c.tobytes() and b.tobytes()==d.tobytes())
    check('original_metadata_enclosures_immutable',original['query_xy']==[256.,256.] and endpoint['candidate_enclosures_xyxy']==[[255.,255.,261.,261.]])
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_source_asset_or_header_or_fresh_asset_SHA_reads=0,model_fit_or_evaluation_reads=0)

if __name__=='__main__':print(json.dumps(fixtures(),ensure_ascii=False),flush=True)
