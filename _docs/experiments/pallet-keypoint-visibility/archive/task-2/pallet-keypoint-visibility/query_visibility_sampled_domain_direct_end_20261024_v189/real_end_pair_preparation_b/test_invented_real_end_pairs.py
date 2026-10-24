"""V189 CPU invented arrays only; original pinned crop/transform operators on toys."""
from pathlib import Path
import copy,json,math,tempfile
import numpy as np
import build_real_end_local_pairs as m

H=Path(__file__).absolute().parent


def run():
    checks=[]
    def check(name,value):
        assert value,name
        checks.append({'name':name,'PASS':True})
    p={key:m.bind(path) for key,path in [('original_pixel_helper',m.HELPER),('original_RGB_sampler',m.LOCAL),('original_context_sampler',m.TARGET),('frozen_geometry',m.GEOMETRY)]}
    op=m.operators(p);geo=op['geometry']
    h,w=96,160;yy,xx=np.indices((h,w));rgb=np.stack(((xx*7+yy*3)%251,(xx*3+yy*11)%241,(xx*13+yy*2)%239),-1).astype(np.uint8)
    frame=dict(query_xy=[[80.,48.]]*8,L=[96.]*8,feature_constructed=[True]*8,context_box_xyxy=[40.,20.,120.,80.],native_slot=list(range(8)),native_continuous_valid=[True]*8,nearest_valid=[True]*8,global_index=list(range(8)),query_id=['INVENTED/native'+str(n) for n in range(8)],raw_hw=[h,w],split='SOURCE_DEVELOPMENT')
    original=m.original_record({**frame,'original_query_input_metadata':[{'invented':True}]*8},0)
    endpoint=dict(authority=geo.REAL_AUTHORITY,native_slot=0,end_locality_id='toy-deck-end',selected_instance_supported=True,intended_part_supported=True,queried_end_localization_supported=True,all_plausible_candidates_enclosed=True,before_end_visible_supported=True,candidate_enclosures_xyxy=[[70.25,40.25,72.25,42.25]],evidence_binding={'invented':True},surrounding_material_witnesses=[dict(raw_pixel_xy=[110,75],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':True})])
    pixels,reason=m.endpoint_preflight(original,endpoint,geo)
    check('finite_localized_end_preflight',reason is None and len(pixels)==16)
    dr=np.full((40,40,3),[251,31,71],np.uint8);da=np.ones((40,40),bool)
    longside,scale=m.fixed_transform_plan(pixels,da,[h,w],op);cut,alpha=op['transformed_cutout'](dr,da,longside)
    co,so,placement=m.fixed_placements(scale['end_support_bbox_xyxy'],alpha,[h,w],op)
    plan=geo.evaluate_plan(original,endpoint,placement)
    check('single_scale_placement_preserves_frozen_end_contract',plan['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING')
    check('SHAM_fullbbox_minimal_4px_gap_no_search',placement['cover_bbox_xyxy'][0]+placement['sham_translation_xy'][0]==scale['end_support_bbox_xyxy'][2]+4)
    before=m.materialize(rgb,frame,op);cr,ca=op['paste'](rgb,cut,alpha,co);sr,sa=op['paste'](rgb,cut,alpha,so)
    cover,sham=m.materialize(cr,frame,op),m.materialize(sr,frame,op)
    proof,detail=m.prove(before,cover,sham,ca,sa,pixels,endpoint,original,cut,alpha,(co,so),op)
    check('actual_toy_fullRGB_end_opacity_sham_surrounding_proofs',all(proof.values()))
    check('clean_all8_native_query_globals_preserved',m.exact(before['original_global_index'],np.arange(8,dtype=np.int32)) and all(m.exact(a['query_xy'],before['query_xy']) for a in (cover,sham)))
    check('same_donor_RGBA_translated_equal_mass',int(ca.sum())==int(sa.sum())>0 and proof['same_transformed_RGB_alpha_pasted_at_both_locations'])
    check('surrounding_samepart_can_stay_visible_while_end_covered',proof['end_every_candidate_raw_tap_COVER_alpha1'] and proof['chosen_surrounding_samepart_RGB_unchanged_in_both'])
    broken=ca.copy();x,y=next(iter(pixels));broken[y,x]=False
    bad,_=m.prove(before,cover,sham,broken,sa,pixels,endpoint,original,cut,alpha,(co,so),op)
    check('one_missing_positive_end_tap_rejects_pair',not bad['end_every_candidate_raw_tap_COVER_alpha1'])
    hole=da.copy();hole[20,20]=False;_,resized=op['transformed_cutout'](dr,hole,40)
    check('donor_holes_never_flatrectangle_filled',not resized[19:22,19:22].any())
    invisible=copy.deepcopy(endpoint);invisible['queried_end_localization_supported']=False
    check('prior_coarsebox_without_direct_end_not_upgraded',m.endpoint_preflight(original,invisible,geo)[0] is None)
    u=copy.deepcopy(original);u['original_query_available']=False
    check('original_U_preserved_before_asset_read',m.endpoint_preflight(u,endpoint,geo)[1]=='ORIGINAL_INPUT_U')
    outside=copy.deepcopy(endpoint);outside['candidate_enclosures_xyxy']=[[1.,1.,2.,2.]]
    check('outside_original96_is_UNKNOWN_not_E',m.endpoint_preflight(original,outside,geo)[0] is None)
    alt=copy.deepcopy(endpoint);alt['candidate_enclosures_xyxy'].append([125.,90.,126.,91.])
    more,_=m.endpoint_preflight(original,alt,geo)
    check('all_candidate_alternatives_enlarge_support_not_drop',more is not None and len(more)>len(pixels))
    badplace=copy.deepcopy(placement);badplace['cover_bbox_xyxy'][0]=-1
    check('out_of_bounds_pair_retained_no_reposition',geo.evaluate_plan(original,endpoint,badplace)['status']=='GEOMETRY_FAILURE_RETAINED')
    nochange,_=m.prove(before,before,sham,ca,sa,pixels,endpoint,original,cut,alpha,(co,so),op)
    check('opacity_without_actual_end_RGB96_change_fails',not nochange['end_support_actual_original96_bytes_changed_COVER'])
    altered=copy.deepcopy(sham);altered['query_xy'][0,0]+=.01
    bad,_=m.prove(before,cover,altered,ca,sa,pixels,endpoint,original,cut,alpha,(co,so),op)
    check('q_drift_detected_not_corrected',not bad['all8_input_q_L_flags_validity_and_ID_unchanged'])
    check('all8_native_quality_owner_stay_UNKNOWN',all((a[k]==-1).all() for a in (before,cover,sham) for k in ('native_visibility_target','point_quality_target','owner_target')))
    with tempfile.TemporaryDirectory(prefix='invented_end_pair_',dir=H) as folder:
        png=Path(folder)/'clean.png';b=m.save_clean_png(png,before['rgb96'][0]);check('saved_clean96_PNG_decoded_byteparity',b['decoded_array_sha256']==m.ash(before['rgb96'][0]))
        snapshot=before['rgb96'].copy();display=m.contact(Path(folder),'INVENTED_LAYOUT_ONLY',0,(before,cover,sham),endpoint,proof)
        with m.Image.open(display['path']) as image:check('invented_contact_fixed_readable_canvas',image.size==(1140,660))
        check('display_overlay_does_not_change_clean96',m.exact(snapshot,before['rgb96']))
        with (H/'INVENTED_CONTACT_LAYOUT.png').open('xb') as f:f.write(Path(display['path']).read_bytes())
    return dict(completed=True,PASS=True,invented_CPU_only=True,count=len(checks),checks=checks,source_RGB_NPZ_header_donor_body_reads=0,model_encoder_feature_fit_or_eval_reads=0)


if __name__=='__main__':
    result=run()
    with (H/'GENERATED_INVENTED_REAL_END_PAIR_CHECKS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
