"""Generated arrays only. Exact old lower math, explicit new donor-role guard."""
import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import build_image_corpus_once as m


def main(output):
    checks=[]
    def check(ok,name):
        assert ok,name;checks.append(name)
    k=m.load_math(m.KERNEL_SHA);before_kernel=m.sha(m.KERNEL)
    row=dict(slot_index=65,frame_index=8,input_id='GENERATED_ONLY',native_slot_logical_only=1,split='TRAIN_LAB',image_sha256='a'*64,raw_hw=[96,96],query_xy=[48.,48.],L=80.,selected_box_xyxy=[0,0,95,95],original_input_valid=True)
    target=dict(base_row_index=65,original_q=[48.,48.],original_L=80.,query_anchored_label=0,matched_certified_group='GENERATED_GROUP',source_family='GENERATED',central4_native_stencil=[list(p) for p in k.central_stencil([48.,48.],80.)],original_regions=[dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[44.,44.,50.,50.]),dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=[55.,44.,57.,46.])])
    donor=dict(id='GENERATED_OBJECT16',donor_rank=0,object_id=16,scene_id=2,split='TRAIN',original_split='train',image_sha256='b'*64)
    locked={donor['id']:donor};old_val={'c'*64};frozen=copy.deepcopy((row,target,donor))
    check(not k.donor_allowed(row,donor),'Immutable_V96_odd_ID_guard_still_rejects_object16_TRAIN')
    check(m.donor_allowed_v116(row,donor,locked,old_val),'Explicit_new_locked_object16_TRAIN_scene2_allowlist')
    for name,value in [('object_id',15),('scene_id',9),('split','VALIDATION'),('original_split','val'),('id','UNLOCKED'),('image_sha256','a'*64)]:
        bad=dict(donor,**{name:value})
        check(not m.donor_allowed_v116(row,bad,locked,old_val),'Reject_mutated_or_unlocked_'+name)
    check(not m.donor_allowed_v116(row,donor,locked,{'b'*64}),'Old_VALIDATION_RGB_disjoint_guard')
    native=np.full((96,96,3),13,np.uint8);drgb=np.full((24,24,3),219,np.uint8);selected=np.ones((24,24),bool);other=np.zeros_like(selected)
    got=m.prepare_pair(k,native,row,target,drgb,selected,other,True,donor,locked,old_val)
    check(got['facts']['pair_ready'] and got['labels']==(1,0),'Generated_controlled_coverE_shamD_pair')
    protected,_=k.protected_mask(row,target)
    check(protected[44,44] and protected[50,50] and protected[46,57] and not protected[45,52],'Original_disjoint_support_union_not_bbox_fill')
    check(all(protected[y,x] for x,y in k.central_stencil(row['query_xy'],row['L'])),'Exact_positive_weight_central4_stencil_protected')
    exclusive,_=k.alpha_rule.exclusive_alpha(selected,other,True)
    cut,scale=k.V2.support_scaled_cutout(drgb,exclusive,protected,row['L'])
    pair,mass=k.V2.core_aligned_pair(row,cut,protected)
    (cxy,ca),(sxy,sa)=pair
    expected=[k.V1.compose(native,cut,xy,a) for xy,a in [(cxy,ca),(sxy,sa)]]
    check(all(np.array_equal(a,b) for a,b in zip(got['images'],expected)),'Bitexact_pinned_lower_geometry_mass_compose_parity')
    check(ca[protected].all() and not sa[protected].any() and np.array_equal(got['images'][1][protected],native[protected]),'Full_cover_opacity_and_exact_sham_protected_RGB')
    check(got['facts']['mass_recheck']['PASS'] and ca.sum()==sa.sum()==cut['alpha'].sum(),'Identical_RGBA_global_native_and_RGB96_mass_certificate')
    check((row,target,donor)==frozen,'Original_qL_rawsplit_slot_support_and_real_donor_identity_unchanged')
    for complete,missing_rgb,name in [(False,False,'incomplete_owners'),(True,True,'missing_RGB')]:
        failure=m.prepare_pair(k,native,row,target,None if missing_rgb else drgb,selected,other,complete,donor,locked,old_val)
        check(not failure['facts']['pair_ready'] and failure['labels']==(-1,-1) and all(np.array_equal(a,native) for a in failure['images']),'Both_after_U_unchanged_without_replacement_'+name)
    conflict=m.prepare_pair(k,native,row,target,drgb,selected,selected,True,donor,locked,old_val)
    check(conflict['labels']==(-1,-1) and not conflict['facts']['pair_ready'],'Empty_exclusive_donor_stays_bothU')
    holes=np.zeros_like(selected);holes[9:15,9:15]=True
    ex,cert=k.alpha_rule.exclusive_alpha(selected,holes,True)
    check(not ex[9:15,9:15].any() and not (ex&holes).any() and cert['excluded_conflicting_pixels']==36,'Opaque_alpha_holes_and_other_owners_never_filled')
    large=copy.deepcopy(target);large['original_regions']=[dict(kind='LOCO_HALFOPEN_SUPPORT',bbox=[8.,8.,88.,88.])]
    failure=m.prepare_pair(k,native,row,large,drgb,selected,other,True,donor,locked,old_val)
    check(not failure['facts']['pair_ready'] and failure['labels']==(-1,-1) and failure['facts']['protected_pixels']==6561,'Oversized_protected_support_U_without_shrinking_or_rescale_search')
    displaced=np.zeros((96,96),bool);displaced[0:10,0:10]=True
    centered=np.zeros_like(displaced);centered[43:53,43:53]=True
    check(not k.V2.mass_certificate(row,centered,displaced)[0],'Equal_global_alpha_mass_outside_crop_is_rejected')
    crop,crop_ast=k.V1.frozen_rgb96_operator(m.CROP,m.CROP_SHA)
    old=m.P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
    oldcrop,_=k.V1.frozen_rgb96_operator(old,'802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66')
    ramp=np.random.default_rng(116).integers(0,256,(31,47,3),dtype=np.uint8)
    for q,L in [([23.5,15.5],32.),([-.25,.5],16.),([48.,30.],40.)]:
        a,av=crop(ramp,np.array(q),L,96);b,bv=oldcrop(ramp,np.array(q),L,96)
        check(np.array_equal(a,b) and np.array_equal(av,bv),'V90_exact_RGB96_AST_parity_with_old_operator_'+str(q))
    check(m.sha(m.KERNEL)==before_kernel and not k.donor_allowed(row,donor),'Old_kernel_bytes_and_guard_unchanged_after_newwrapper')
    tree=ast.parse(Path(m.__file__).read_text())
    attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    check(not set(attrs)&{'donor_allowed','fixed_donors','prepare_rank'},'Production_never_calls_old_eligibility_or_rank_API')
    try:m.entry_structure(dict(approved=False))
    except ValueError:pass
    else:raise AssertionError('Unapproved entry accepted')
    check(True,'Unapproved_entry_fails_before_source_IO')
    # Reimport with every Path content-reader disabled: the module import is inert.
    oldread,oldtext,oldopen=Path.read_bytes,Path.read_text,Path.open
    def forbidden(*args,**kwargs):raise AssertionError('Import attempted source/file I/O')
    Path.read_bytes=forbidden;Path.read_text=forbidden;Path.open=forbidden
    try:
        spec=importlib.util.spec_from_file_location('generated_inert_import',m.__file__);fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
    finally:Path.read_bytes=oldread;Path.read_text=oldtext;Path.open=oldopen
    check(True,'Producer_import_has_no_metadata_payload_or_weight_IO')
    proof=dict(PASS=True,completed=True,producer=m.bind(m.__file__),test_code=m.bind(__file__),kernel=m.bind(m.KERNEL),crop_source=m.bind(m.CROP),crop_AST_sha256=crop_ast,checks=checks,source_payload_reads=0,source_metadata_reads=0,model_forward_or_fit=0,current223=False,V112=False,PnP=0)
    m.write(output,proof);print(m.clean(dict(PASS=True,checks=len(checks),proof=m.bind(output))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();main(a.output)
