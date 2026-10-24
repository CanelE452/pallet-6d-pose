"""Independent static/text-metadata preflight; no producer import or asset reads."""
from pathlib import Path
import ast,hashlib,json,math,os,stat

H=Path(__file__).absolute().parent;V=H.parent
D=V/'real_end_pair_preparation_b'
SCOPE=V/'ROOT_SCOPE_BEFORE_DIRECT_END_SOURCE_CODE_AND_METADATA_PREPARATION.json'

def content(path):
    p=Path(path)
    assert p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts)
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nf=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nf
        final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(final,'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        return f.read()

def binding(path):return dict(path=str(path),sha256=hashlib.sha256(content(path)).hexdigest())
def read(path):return json.loads(content(path))

def run():
    protocol=read(D/'PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json')
    preparation=read(D/'PREPARATION_COMPLETE.json')
    inherited=read(D/'INHERITED_SOURCE_DONOR_INPUT_BINDINGS_BEFORE_PIXELS.json')
    generated=read(D/'GENERATED_INVENTED_REAL_END_PAIR_CHECKS.json')
    source=content(D/'build_real_end_local_pairs.py').decode();tree=ast.parse(source)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    lines={n.name:[n.lineno,n.end_lineno] for n in functions.values()}
    helpers={key:content(protocol[key]['path']).decode() for key in ('original_pixel_helper','original_RGB_sampler','original_context_sampler','frozen_geometry')}
    checks={}
    def record(name,value):
        assert value,name;checks[name]=True
    record('scope_exact_and_preparation_final',protocol['root_scope']==binding(SCOPE) and preparation['FINAL_READY'] and preparation['freeze_after_READY'])
    for key,value in preparation['bindings'].items():record('frozen_binding_'+key,value==binding(D/key))
    for key in helpers:record('frozen_pure_helper_'+key,protocol[key]==binding(Path(protocol[key]['path'])))
    record('invented20_closed',generated['PASS'] and generated['invented_CPU_only'] and generated['count']==len(generated['checks'])==20 and all(x['PASS'] for x in generated['checks']))
    plan=read(protocol['source_plan']['path'])
    record('all22_66_21frames168slots_same37_train_donors',len(plan['attempts'])==66 and len({a['case_id'] for a in plan['attempts']})==22 and len(plan['frames'])==21 and sum(len(f['native_slot']) for f in plan['frames'])==168 and len({a['donor_id'] for a in plan['attempts']})==37 and all(f['split']=='SOURCE_DEVELOPMENT' for f in plan['frames']) and all(a['donor']['split']=='TRAIN' for a in plan['attempts']))
    record('inherited_census_and_prior_outcomes_not_upgraded',all(inherited[k]==v for k,v in [('cases22',22),('attempts66',66),('frames21',21),('original_all8_slots168',168),('same_assigned_donor_ids_used37',37)]) and inherited['old_broad_COVER_outputs_reused_as_end_controls'] is False and inherited['new_labels_or_admission'] is False)
    record('all8_input_order_metadata_preserved',all(f['native_slot']==list(range(8)) and len(f['global_index'])==len(f['query_xy'])==len(f['L'])==len(f['native_continuous_valid'])==8 for f in plan['frames']))
    # Independent formula fixtures: exact four positive taps and both minimal X gaps.
    x,y=.25,.75;weights=[(1-x)*(1-y),x*(1-y),(1-x)*y,x*y]
    record('independent_four_tap_fractional_fixture',weights==[.1875,.0625,.5625,.1875] and math.fsum(weights)==1 and all(w>0 for w in weights))
    for center,w,x0,x1,origin,cw in [(20,100,18,23,13,15),(80,100,77,83,69,17)]:
        dx=x1+4-origin if center<=(w-1)/2 else x0-4-(origin+cw)
        gap=(origin+dx)-x1 if dx>0 else x0-(origin+dx+cw)
        record('independent_minimal_fullbbox_gap4_'+str(center),gap==4 and dx!=0)
    record('single_transform_not_scale_search',len([n for n in ast.walk(functions['fixed_transform_plan']) if isinstance(n,(ast.For,ast.While))])==0 and 'radius-1.' in ast.unparse(functions['fixed_transform_plan']).replace(' ','') and 'np.argmax' in helpers['original_pixel_helper'])
    record('strict_U_unknown_outside_preflight_before_asset_load',source.index('end_pixels,reason=endpoint_preflight')<source.index('rgb=load_bound') and 'END_OUTSIDE_OR_NOT_FULLY_REPRESENTED_IN_ORIGINAL96_UNKNOWN_NOT_E' in source and 'DIRECT_END_WEAK_OBSERVATION_UNKNOWN' in source)
    record('original_V90_crop_function_unchanged',"load_functions(LOCAL,('crop',),ns)" in source and "np.floor(np.clip(acc,0,255)+.5)" in helpers['original_RGB_sampler'])
    record('all8_BEFORE_pixel_hash_and_packet_parity_is_gate',"require(all(parity.values()),'Original all8 BEFORE parity')" in source and 'all8_RGB96_equal_saved_packet' in helpers['original_pixel_helper'] and 'all8_valid96_fact_hash' in helpers['original_pixel_helper'])
    record('union_every_positive_tap_opacity_actual_change_and_surrounding_proofs',all(name in source for name in ['end_every_candidate_raw_tap_COVER_alpha1','end_every_candidate_raw_tap_SHAM_alpha0','end_support_actual_original96_bytes_changed_COVER','end_support_actual_original96_bytes_unchanged_SHAM','chosen_surrounding_samepart_RGB_unchanged_in_both','equal_positive_pasted_alpha_mass','same_transformed_RGB_alpha_pasted_at_both_locations']))
    record('all8_native_quality_owner_remain_UNKNOWN_and_no_old_native_decoder',"a['owner_target']=np.full(8,-1,np.int8)" in source and 'all8_new_native_quality_owner_unknown' in source and "'native_visibility_target': np.full(8, -1, np.int8)" in ast.unparse(ast.parse(helpers['original_pixel_helper'])))
    record('failure_unknown_no_retry_and_census_PASS_not_allpair_PASS',"r['status']='ATTEMPT_FAILURE_UNKNOWN_RETAINED'" in source and 'Cached donor failure, no retry' in source and 'PASS_means_attempt_census_and_invariants_not_all_pairs_supported=True' in source)
    forbidden_calls={'torch.load','torch.save','requests.get','urllib.request.urlopen','cv2.solvePnP','model.forward'}
    record('no_model_fit_forward_download_or_PnP_calls',not any(k in source for k in forbidden_calls))
    return dict(completed=True,PASS=True,review_type='INDEPENDENT_PREEXEC_STATIC_AND_METADATA_ONLY',reviewer_code=binding(Path(__file__).absolute()),code=protocol['code'],protocol=binding(D/'PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json'),preparation=binding(D/'PREPARATION_COMPLETE.json'),inherited_bindings=binding(D/'INHERITED_SOURCE_DONOR_INPUT_BINDINGS_BEFORE_PIXELS.json'),checks=checks,number_checks=len(checks),line_evidence=lines,blockers=[],limits=['Not actual pixel/opacity/semantic-end annotation verification.','Root actual entry must bind closed A all22 direct-end observations before generation.','Weak source semantic end plus 2D opaque foreground composite is not physical-depth/native gold.','Selected real cases lack native7 and real SQUARE coverage; no full-goal or transfer claim.'],actual_source_RGB_NPZ_header_mask_depth_weight_feature_or_eval_body_reads=0,actual_asset_hashes=0,producer_imports_or_execution=0,model_forward_fit=0,selected_model=None,fully_solved=False)

if __name__=='__main__':
    result=run()
    with (H/'REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,review=binding(H/'REVIEW.json'),checks=result['number_checks'])))
