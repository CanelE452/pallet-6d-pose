"""Sparse L>192 original96 integration on invented RGB/donor only."""
import copy,json
import numpy as np
import build_real_end_local_pairs as m

def run():
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(dict(name=name,PASS=True))
    p={key:m.bind(path) for key,path in [('original_pixel_helper',m.HELPER),('original_RGB_sampler',m.LOCAL),('original_context_sampler',m.TARGET),('frozen_geometry',m.GEOMETRY)]};op=m.operators(p);geo=op['geometry']
    yy,xx=np.indices((512,512));rgb=np.stack(((xx*7+yy*3)%251,(xx*3+yy*11)%241,(xx*13+yy*2)%239),-1).astype(np.uint8)
    f=dict(query_xy=[[256.,256.]]*8,L=[384.]*8,feature_constructed=[True]*8,context_box_xyxy=[128.,128.,384.,384.],native_slot=list(range(8)),native_continuous_valid=[True]*8,nearest_valid=[True]*8,global_index=list(range(8)),query_id=['INVENTED/native'+str(n) for n in range(8)],raw_hw=[512,512],split='SOURCE_DEVELOPMENT',original_query_input_metadata=[dict(no_detection=False,finite_non_sentinel=True,query_RGB96_centre_domain_valid=True,native_grid_any_valid=True)]*8)
    o=m.original_record(f,0);e=dict(authority=geo.REAL_AUTHORITY,native_slot=0,end_locality_id='invented-sparse-end',selected_instance_supported=True,intended_part_supported=True,queried_end_localization_supported=True,all_plausible_candidates_enclosed=True,before_end_visible_supported=True,candidate_enclosures_xyxy=[[255.,255.,261.,261.]],evidence_binding={'invented':True},surrounding_material_witnesses=[dict(raw_pixel_xy=[298,258],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':True})])
    pixels,reason=m.endpoint_preflight(o,e,geo);support,_=geo.exact96_support(o['query_xy'],o['L'],o['raw_hw'])
    check('real_preflight_accepts_holey_L384_without_q_change',reason is None and not pixels<=support.keys())
    dr=np.full((40,40,3),[251,31,71],np.uint8);da=np.ones((40,40),bool)
    longside,scale=m.fixed_transform_plan(pixels,da,[512,512],op);cut,alpha=op['transformed_cutout'](dr,da,longside);co,so,placement=m.fixed_placements(scale['end_support_bbox_xyxy'],alpha,[512,512],op)
    plan=geo.evaluate_plan(o,e,placement);before=m.materialize(rgb,f,op);cr,ca=op['paste'](rgb,cut,alpha,co);sr,sa=op['paste'](rgb,cut,alpha,so);cover,sham=m.materialize(cr,f,op),m.materialize(sr,f,op)
    proof,detail=m.prove(before,cover,sham,ca,sa,pixels,e,o,cut,alpha,(co,so),op)
    check('all_raw_candidates_opaque_even_unsampled',plan['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING' and all(proof.values()))
    check('actual_end_samples_cover_changed_sham_identical',proof['end_support_actual_original96_bytes_changed_COVER'] and proof['end_support_actual_original96_bytes_unchanged_SHAM'] and detail['end_related_RGB96_sample_count']>0)
    bad=copy.deepcopy(e);bad['candidate_enclosures_xyxy']=[[256.,256.,256.,256.]]
    check('no_sampled_intersection_real_preflight_UNKNOWN',m.endpoint_preflight(o,bad,geo)[0] is None)
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_source_asset_or_header_or_fresh_asset_SHA_reads=0)
if __name__=='__main__':print(json.dumps(run()),flush=True)
