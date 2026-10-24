"""Independent pure V187 witness contract; invented inputs only in this phase.

No asset/model loaders. Sparse observation fields are proposed source metadata,
not generated truth. Native semantic state, center owner and reference positions
are deliberately not arguments to the support decision.
"""
import argparse
import copy
import json
import math
from pathlib import Path

CLASSES=('VISIBLE_SELECTED_PART','FOREIGN_FRONT_SELECTED_PART','UNKNOWN_OR_UNOBSERVED','INVALID_SUPPORT')
CORRESPONDENCE_AUTHORITIES={'CONSTRUCTIVE_SOURCE_DECLARED','RGB_SELECTED_PART_WEAK'}
COVERAGE_AUTHORITIES={'CONSTRUCTIVE_SOURCE_DECLARED','RGB_COMPLETE_SELECTED_PART_WEAK'}

def require(value,message):
    if not value:raise ValueError(message)

def exact96(q,L):
    require(isinstance(q,(list,tuple)) and len(q)==2 and all(math.isfinite(x) for x in q),'Original finite q')
    require(math.isfinite(L) and L>0,'Original finite positive L')
    offsets=[((i+.5)/96-.5)*L for i in range(96)]
    return [(q[0]+x,q[1]+y) for y in offsets for x in offsets]

def taps(point,hw):
    x,y=point;h,w=hw
    if not(-.5<=x<w-.5 and -.5<=y<h-.5):return [(None,None,1.)]
    ix,iy=math.floor(x),math.floor(y);fx,fy=x-ix,y-iy;result=[]
    for dx,wx in ((0,1-fx),(1,fx)):
        for dy,wy in ((0,1-fy),(1,fy)):
            weight=wx*wy
            if weight<=0:continue
            px,py=ix+dx,iy+dy
            result.append((px if 0<=px<w and 0<=py<h else None,py if 0<=px<w and 0<=py<h else None,weight))
    return result

def correspondence_supported(c,native):
    return (c.get('supported') is True and c.get('native_slot')==native
            and c.get('authority') in CORRESPONDENCE_AUTHORITIES and bool(c.get('selected_instance_id'))
            and bool(c.get('intended_part_id')) and c.get('part_scope')=='FIXED_INTENDED_END_REGION')

def classify_observation(o):
    authority=o.get('authority');kind=o.get('kind')
    if kind=='VISIBLE_PART' and authority=='RGB_SELECTED_PART_WEAK':
        return 0 if all(o.get(k) is True for k in ('material_observed','selected_instance_supported','selected_part_supported')) else 2
    if kind=='VISIBLE_PART' and authority=='CONSTRUCTIVE_SOURCE_DECLARED':
        return 0 if all(o.get(k) is True for k in ('before_selected_part_supported','scene_selected_part_supported','same_surface_identity_and_depth')) else 2
    if kind=='FOREIGN_FRONT' and authority=='CONSTRUCTIVE_SOURCE_DECLARED':
        before,scene=o.get('before_Z'),o.get('scene_Z')
        finite=(isinstance(before,(int,float)) and isinstance(scene,(int,float))
                and math.isfinite(before) and math.isfinite(scene) and 0<scene<before)
        return 1 if finite and all(o.get(k) is True for k in ('before_selected_part_supported','scene_foreign_supported','paired_same_raw_pixel')) else 2
    if kind=='FOREIGN_FRONT' and authority=='RGB_OCCLUSION_CONTINUATION_WEAK':
        return 1 if all(o.get(k) is True for k in ('expected_part_localization_supported','selected_edges_continue_behind_foreign','foreign_material_observed','foreground_order_observed')) else 2
    return 2  # bbox, other wood, owner, model, mask, sensor or unobserved alone

def evaluate_source_witness(record,witness):
    require(record['development_role'] in ('SOURCE_TRAIN','SOURCE_DEVELOPMENT'),'Only frozen source TRAIN role')
    require(type(record['native_slot']) is int and 0<=record['native_slot']<8,'Original native0..7')
    original=copy.deepcopy(record);c=witness['correspondence'];coverage=witness.get('coverage',{})
    result=dict(original_input=original,selected_part_correspondence=copy.deepcopy(c),visible_support_target=-1,foreign_front_support_target=-1,
        local96_witness_mass4=None,class_order=list(CLASSES),authority_tags=[],evidence_bindings=copy.deepcopy(witness.get('evidence_bindings',[])),
        coverage_evidence=copy.deepcopy(coverage),reason_codes=[],native_visibility_target=-1,point_quality_target=-1,training_admission=False)
    q,L=record.get('query_xy'),record.get('L')
    finite=(isinstance(q,(list,tuple)) and len(q)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in q)
            and isinstance(L,(int,float)) and math.isfinite(L) and L>0)
    if not finite:
        result['reason_codes']=['ORIGINAL_Q_L_MISSING'];return result
    grid=exact96(q,L);observed={};authorities=set()
    corr=correspondence_supported(c,record['native_slot'])
    for o in witness.get('observations',[]):
        xy=o['raw_pixel_xy'];require(len(xy)==2 and all(type(v) is int for v in xy),'Integer source pixel witness')
        cls=classify_observation(o) if corr else 2;key=tuple(xy)
        if key in observed and observed[key]!=cls:observed[key]=2
        else:observed[key]=cls
        if cls in (0,1):authorities.add(o['authority'])
    columns=[[],[],[],[]]
    for point in grid:
        local=[0.,0.,0.,0.]
        for x,y,weight in taps(point,record['raw_hw']):
            cls=3 if x is None else observed.get((x,y),2)
            local[cls]+=weight
        require(math.isclose(math.fsum(local),1.,rel_tol=0,abs_tol=2e-15),'Onehot+unknown+invalid mass preserved')
        for i,value in enumerate(local):columns[i].append(value)
    mass=[math.fsum(values)/9216 for values in columns]
    result['local96_witness_mass4']=mass;result['authority_tags']=sorted(authorities)
    if not record['original_query_available']:
        result['reason_codes']=['ORIGINAL_INPUT_U_UNCHANGED'];return result
    if not corr:
        result['reason_codes']=['SELECTED_INSTANCE_OR_INTENDED_PART_CORRESPONDENCE_UNKNOWN'];return result
    complete=(coverage.get('authority') in COVERAGE_AUTHORITIES and all(coverage.get(k) is True for k in
             ('candidate_set_exhaustive','all_candidates_localized','all_in_crop_candidates_observed')))
    no_v=complete and coverage.get('no_visible_selected_support') is True
    no_f=complete and coverage.get('no_foreign_front_selected_support') is True
    opportunity=(mass[0]>0 or mass[1]>0 or (complete and coverage.get('selected_part_opportunity_exists') is True))
    require(not(no_v and mass[0]>0) and not(no_f and mass[1]>0),'Contradictory positive/absence witness; preserve as failure, do not relabel')
    result['visible_support_target']=1 if mass[0]>0 else 0 if no_v else -1
    result['foreign_front_support_target']=1 if mass[1]>0 else 0 if no_f and opportunity else -1
    if mass[0]>0 and mass[1]>0:result['reason_codes'].append('NONEXCLUSIVE_BOTH_LOCAL_SUPPORTS')
    if result['visible_support_target']<0 or result['foreign_front_support_target']<0:result['reason_codes'].append('SPARSE_OR_UNCERTAIN_ABSENCE_REMAINS_UNKNOWN')
    if mass[3]>0:result['reason_codes'].append('INVALID_SUPPORT_MASS_RETAINED_NO_RENORMALIZATION')
    require(record==original,'Original input metadata mutation prohibited')
    return result

def observe_queried_end(record,correspondence,observation):
    """Direct source semantic end observation, separate from broad support2.

    The end locality is supported by constructive identity or RGB topology. It
    need not be a pixel-exact physical vertex. Surrounding patch material may
    remain visible while this particular queried end is externally blocked.
    """
    require(record['development_role'] in ('SOURCE_TRAIN','SOURCE_DEVELOPMENT'),'Only source TRAIN end observations')
    result=dict(original_input=copy.deepcopy(record),queried_end_semantic_weak_state='UNKNOWN',
        queried_end_locality_id=observation.get('end_locality_id'),authority=observation.get('authority'),
        evidence_binding=copy.deepcopy(observation.get('evidence_binding')),broad_support2_is_not_native_decoder=True,
        original_native_semantic_unchanged=True,native_visibility_target=-1,point_quality_target=-1,training_admission=False)
    supported=(correspondence_supported(correspondence,record['native_slot'])
               and bool(observation.get('end_locality_id')) and observation.get('end_localization_supported') is True
               and observation.get('local_crop_has_end_witness') is True)
    if not supported or not record['original_query_available']:return result
    authority=observation.get('authority');state=observation.get('state')
    if authority=='SOURCE_RGB_SEMANTIC_END_WEAK':
        if state=='VISIBLE' and observation.get('queried_end_material_or_topological_tip_observed') is True:
            result['queried_end_semantic_weak_state']='VISIBLE'
        elif state=='EXTERNALLY_BLOCKED' and all(observation.get(k) is True for k in
                ('foreign_front_of_queried_end_observed','selected_end_continuation_behind_foreign_supported')):
            result['queried_end_semantic_weak_state']='EXTERNALLY_BLOCKED'
        elif state=='SELF_HIDDEN' and observation.get('selected_own_geometry_hides_queried_end_observed') is True:
            result['queried_end_semantic_weak_state']='SELF_HIDDEN'
    elif authority=='CONSTRUCTIVE_SOURCE_FIXED_END_WITNESS':
        if state=='VISIBLE' and all(observation.get(k) is True for k in
                ('fixed_end_witness_supported_in_target_alone','same_end_witness_unchanged_and_visible')):
            result['queried_end_semantic_weak_state']='VISIBLE'
        elif state=='EXTERNALLY_BLOCKED' and all(observation.get(k) is True for k in
                ('fixed_end_witness_supported_in_target_alone','all_plausible_end_witness_footprint_covered','distinct_foreign_strictly_in_front')):
            result['queried_end_semantic_weak_state']='EXTERNALLY_BLOCKED'
    return result

def end_cover_and_sham_fixture(end_raw_pixels,cover_xyxy,sham_xyxy):
    """Invented finite witness control only; not an actual source raster proof."""
    def inside(p,b):return b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3]
    require(bool(end_raw_pixels),'Finite preregistered end witness required')
    return dict(end_fully_inside_cover=all(inside(p,cover_xyxy) for p in end_raw_pixels),
                end_disjoint_from_sham=all(not inside(p,sham_xyxy) for p in end_raw_pixels))

def self_tests():
    tests=[]
    r=dict(original_global_index=7,original_occurrence_id='INVENTED/native7',native_slot=7,query_xy=[47.5,47.5],L=96.,raw_hw=[96,96],original_query_available=True,development_role='SOURCE_TRAIN',original_semantic_state='D',center_owner='BACKGROUND',reference_xy=[1.,1.])
    c=dict(supported=True,native_slot=7,selected_instance_id='selected-toy',intended_part_id='deck-left-end',part_scope='FIXED_INTENDED_END_REGION',authority='RGB_SELECTED_PART_WEAK')
    visible=dict(raw_pixel_xy=[20,20],kind='VISIBLE_PART',authority='RGB_SELECTED_PART_WEAK',material_observed=True,selected_instance_supported=True,selected_part_supported=True)
    ext=dict(raw_pixel_xy=[30,30],kind='FOREIGN_FRONT',authority='CONSTRUCTIVE_SOURCE_DECLARED',before_selected_part_supported=True,scene_foreign_supported=True,paired_same_raw_pixel=True,before_Z=10.,scene_Z=5.)
    def e(observations,record=None,corr=None,coverage=None):return evaluate_source_witness(copy.deepcopy(record or r),dict(correspondence=copy.deepcopy(corr or c),observations=copy.deepcopy(observations),coverage=coverage or {}))
    grid=exact96(r['query_xy'],r['L']);require(grid[0]==(0.,0.) and grid[47*96+47]==(47.,47.) and grid[-1]==(95.,95.),'Exact original pixel-center96')
    tests.append('Exact96_pixel_center_grid_and_clean_original_q_L')
    v=e([visible]);require(v['visible_support_target']==1 and v['foreign_front_support_target']==-1 and math.isclose(v['local96_witness_mass4'][0],1/9216,rel_tol=0,abs_tol=2e-18),'Weak visible existence without pose certificate')
    tests.append('RGB_weak_visible_positive_needs_no_perfect_pose_registration')
    s=copy.deepcopy(r);s.update(original_semantic_state='E',center_owner='FOREIGN',reference_xy=[900.,900.])
    require(e([visible],s)['local96_witness_mass4']==v['local96_witness_mass4'],'Semantic/owner/ref do not enter support')
    tests.append('Native_semantics_center_owner_reference_mutation_target_invariant')
    coarse=dict(visible,authority='COARSE_BOUND_ONLY');require(e([coarse])['visible_support_target']==-1,'BBox intersection is not material support')
    tests.append('Positive_coarsebox_overlap_alone_cannot_admit_YES')
    wrong=dict(visible,selected_part_supported=False);require(e([wrong])['visible_support_target']==-1,'Other part not intended end')
    tests.append('Other_selected_wood_or_other_end_not_intended_part')
    f=e([ext]);require(f['foreign_front_support_target']==1 and f['visible_support_target']==-1,'Paired front positive')
    tests.append('Constructive_same_pixel_before_part_and_strict_foreign_front')
    for change in ({'before_selected_part_supported':False},{'scene_Z':12.},{'scene_Z':float('nan')},{'paired_same_raw_pixel':False}):
        bad=dict(ext,**change);require(e([bad])['foreign_front_support_target']==-1,'Foreign alone/depth invalid is not external witness')
    tests.append('Foreign_without_before_part_or_front_order_remains_UNKNOWN')
    weak=dict(raw_pixel_xy=[30,30],kind='FOREIGN_FRONT',authority='RGB_OCCLUSION_CONTINUATION_WEAK',expected_part_localization_supported=True,selected_edges_continue_behind_foreign=True,foreign_material_observed=True,foreground_order_observed=True)
    require(e([weak])['foreign_front_support_target']==1,'RGB structural weak external')
    tests.append('Localized_RGB_continuation_and_front_order_weak_external')
    both=e([visible,ext]);require((both['visible_support_target'],both['foreign_front_support_target'])==(1,1),'Nonexclusive existence')
    tests.append('Different_part_portions_allow_BOTH_without_native_relabel')
    unknown=dict(c,supported=False);require(e([visible,ext],corr=unknown)['visible_support_target']==-1,'Unknown correspondence')
    tests.append('Unknown_selected_instance_or_part_correspondence_forces_U')
    s=copy.deepcopy(r);s['original_query_available']=False;u=e([visible],s);require(u['visible_support_target']==u['foreign_front_support_target']==-1,'Original U')
    tests.append('Original_input_U_not_resurrected_by_visible_witness')
    s=copy.deepcopy(r);s.update(query_xy=[0.,0.],L=2.,raw_hw=[2,2]);vv=dict(visible,raw_pixel_xy=[0,0]);p=e([vv],s)
    require(p['visible_support_target']==1 and p['local96_witness_mass4'][3]>0 and math.isclose(sum(p['local96_witness_mass4']),1.,abs_tol=1e-14),'Partial invalid retained, positive witness survives')
    tests.append('Partial_invalid_mass_retained_without_extra_allknown_gate')
    coverage=dict(authority='RGB_COMPLETE_SELECTED_PART_WEAK',candidate_set_exhaustive=True,all_candidates_localized=True,all_in_crop_candidates_observed=True,no_visible_selected_support=True,no_foreign_front_selected_support=True,selected_part_opportunity_exists=False)
    n=e([],coverage=coverage);require((n['visible_support_target'],n['foreign_front_support_target'])==(0,-1),'No sampled end is not external negative/positive')
    tests.append('Certified_no_crop_opportunity_visible_NO_foreign_UNKNOWN')
    coverage['selected_part_opportunity_exists']=True;n=e([],coverage=coverage);require((n['visible_support_target'],n['foreign_front_support_target'])==(0,0),'Exhaustive known absence')
    tests.append('Explicit_complete_candidate_coverage_required_for_NO')
    coverage['candidate_set_exhaustive']=False;require(e([],coverage=coverage)['visible_support_target']==-1,'Sparse absence unknown')
    tests.append('Incomplete_enclosure_never_certifies_absence')
    raw_taps=taps((0.,0.),[2,2]);require(raw_taps==[(0,0,1.)],'Zero bilinear taps ignored')
    tests.append('Zero_weight_boundary_taps_not_witness_intersections')
    for native in range(8):
        rr=dict(r,native_slot=native);cc=dict(c,native_slot=native)
        x=e([visible],rr,cc);require(x['original_input']['native_slot']==native and x['native_visibility_target']==x['point_quality_target']==-1 and x['training_admission'] is False,'All native retained')
    tests.append('All_native8_and_new_native_quality_minus1_no_admission')
    snapshot=copy.deepcopy(r);evaluate_source_witness(r,dict(correspondence=c,observations=[visible]));require(r==snapshot,'Immutable original metadata')
    tests.append('Input_record_not_mutated_and_old_labels_untouched')
    end_base=dict(end_locality_id='selected-deck-left-tip',authority='SOURCE_RGB_SEMANTIC_END_WEAK',end_localization_supported=True,local_crop_has_end_witness=True)
    covered=dict(end_base,state='EXTERNALLY_BLOCKED',foreign_front_of_queried_end_observed=True,selected_end_continuation_behind_foreign_supported=True)
    queried=observe_queried_end(r,c,covered)
    require(v['visible_support_target']==1 and queried['queried_end_semantic_weak_state']=='EXTERNALLY_BLOCKED','Visible surrounding patch does not mean queried end D')
    tests.append('Surrounding_same_part_visible_but_queried_end_COVER_no_support_to_D_decode')
    end_visible=dict(end_base,state='VISIBLE',queried_end_material_or_topological_tip_observed=True)
    shifted=dict(r,center_owner='BACKGROUND',original_semantic_state='E')
    queried=observe_queried_end(shifted,c,end_visible)
    require(queried['queried_end_semantic_weak_state']=='VISIBLE' and queried['original_input']['center_owner']=='BACKGROUND','Center BG does not hide observed end')
    tests.append('SHAM_or_coordinate_offset_center_BG_but_queried_end_visible')
    controls=end_cover_and_sham_fixture([(30,30),(31,30),(30,31),(31,31)],[29,29,32,32],[40,29,43,32])
    require(all(controls.values()) and (r['query_xy'],r['L'])==([47.5,47.5],96.),'Fixed end footprint control, same original crop')
    tests.append('Fixed_source_end_witness_cover_sham_geometry_possible_without_q_change')
    unsafe=dict(end_base,authority='NOMINAL_RENDER_ONLY',state='VISIBLE',queried_end_material_or_topological_tip_observed=True)
    require(observe_queried_end(r,c,unsafe)['queried_end_semantic_weak_state']=='UNKNOWN','Nominal mask is not observed end gold')
    tests.append('Nominal_fullmask_or_sensor_authority_not_direct_end_observation')
    require(queried['native_visibility_target']==-1 and queried['training_admission'] is False,'Weak end observation not old native label overwrite')
    tests.append('Direct_end_weak_authority_separate_old_native_labels_and_admission')
    return dict(completed=True,PASS=True,invented_only=True,checks=tests,count=len(tests),source_asset_or_evaluation_body_reads=0,actual_source_witness_labels_generated=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true',required=True);p.add_argument('--report',type=Path);a=p.parse_args()
    result=self_tests();text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.report:
        with a.report.open('x',encoding='utf-8') as f:f.write(text)
    print(text,end='')
