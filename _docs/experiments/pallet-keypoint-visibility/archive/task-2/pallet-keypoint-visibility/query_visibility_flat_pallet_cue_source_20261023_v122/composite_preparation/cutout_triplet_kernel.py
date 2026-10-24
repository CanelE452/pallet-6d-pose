"""Inert, array-only V122 wrapper over pinned real-cutout math. No I/O CLI."""
from pathlib import Path
import copy,hashlib,importlib.util
import numpy as np
P=Path(__file__).resolve().parents[2]
KERNEL=P/'query_visibility_query_anchored_cue_20261023_v96/query_anchored_pair_preparation/query_anchored_pair_kernel.py'
KERNEL_SHA='0299a3f6af75b3907875675499054b88bfc8f9a59203bd4ea4b58529f661b5f9'

def lower_math():
    assert hashlib.sha256(KERNEL.read_bytes()).hexdigest()==KERNEL_SHA
    s=importlib.util.spec_from_file_location('v122_pinned_cutout_math',KERNEL)
    k=importlib.util.module_from_spec(s);s.loader.exec_module(k);return k

def canonical_fold(split):
    return {'TRAIN':'TRAIN','TRAIN_LAB':'TRAIN','VALIDATION':'VALIDATION','VAL_HALL':'VALIDATION'}[split]

def source_emulated_input(target_bbox_xyxy,query_xy,raw_hw,anchor_uv,input_id,split,source_frame_group):
    """Caller supplies the prospectively fixed8 rectangle anchor UVs.

    This creates a NEW source tuple; no actual initializer q8 is accepted here.
    q0 is a carrier only, never a native identity. No cue-driven bbox expansion.
    """
    b=np.asarray(target_bbox_xyxy,np.float64);uv=np.asarray(anchor_uv,np.float64);q=np.asarray(query_xy,np.float64)
    assert b.shape==(4,)and np.isfinite(b).all()and(b[2:]>b[:2]).all()
    assert uv.shape==(8,2)and np.isfinite(uv).all()and((uv>=0)&(uv<=1)).all()and q.shape==(2,)and np.isfinite(q).all()
    points=b[:2]+uv*(b[2:]-b[:2]);points[0]=q
    lo=points.min(0);hi=points.max(0);assert(hi>lo).all()
    L=float(np.clip(.25*np.max(np.ptp(points,axis=0)),16,max(raw_hw)))
    return dict(input_id=input_id,split=canonical_fold(split),raw_split=split,raw_hw=list(raw_hw),query_xy=q.tolist(),L=L,
        initial_points=points.tolist(),context_box_xyxy=np.r_[lo,hi].tolist(),source_emulated_not_initializer=True,
        fixed_carrier_slot=0,carrier_is_native_identity=False,source_frame_group=source_frame_group,
        original_initializer_touched=False)

def check_partition_consistency(jobs):
    """Metadata only; group every derivative by source frame/episode and donor frame."""
    groups={};donors={};rgbfold={}
    for j in jobs:
        fold=canonical_fold(j['fixed_input']['split']);sg=j['fixed_input']['source_frame_group'];d=j['assigned_donor']
        assert groups.setdefault(sg,fold)==fold,'Source frame/pair/episode split leakage'
        frame=(d['scene_id'],d['frame_id']);assert donors.setdefault(frame,fold)==fold,'Donor frame split leakage'
        assert canonical_fold(d['split'])==fold
        assert rgbfold.setdefault(d['image_sha256'],fold)==fold,'Donor RGB split leakage'
    return True

def protected_region(k,cue,raw_hw):
    """Full original support+uncertainty regions plus original reference nearest cell.

    Region kinds are explicit; preserve original boxes including outside targetbbox.
    Never clips to targetbbox, shrinks uncertainty or needs a target SAM mask.
    """
    out=np.zeros(tuple(raw_hw),bool)
    if not cue.get('before_visible_weak_supported',False):return None,'BEFORE_CUE_NOT_WEAK_D_U'
    if not cue.get('original_regions'):return None,'MISSING_ORIGINAL_CUE_GEOMETRY_U'
    for r in cue['original_regions']:
        if r['kind']=='HALF_OPEN_PIXEL_EDGE':part=k.adapter.v3.pixel_edge_box_cells(r['bbox'],raw_hw)
        elif r['kind']=='CLOSED_NATIVE_UNCERTAINTY':part=k.adapter.closed_range_cells(r['bbox'],raw_hw)
        else:raise ValueError('Explicit original source bbox convention required')
        if part is None:return None,'ORIGINAL_COMPLETE_REGION_OOB_OR_INVALID_U'
        out|=part
    ref=np.asarray(cue['reference_xy'],np.float64)
    if ref.shape!=(2,)or not np.isfinite(ref).all():return None,'MISSING_ORIGINAL_REFERENCE_U'
    x,y=np.floor(ref+.5).astype(int);h,w=raw_hw
    if not(0<=x<w and 0<=y<h):return None,'ORIGINAL_REFERENCE_NEAREST_CELL_OOB_U'
    out[y,x]=True
    return out,'ORIGINAL_COMPLETE_CUE_AND_UNCERTAINTY'

def reconstruct(native_rgb,patch_rgb,patch_alpha,xy,k=None):
    """Same shared real RGBA patch can reconstruct COVER/SHAM transient fullRGB."""
    k=lower_math()if k is None else k
    assert patch_rgb.dtype==np.uint8 and patch_alpha.dtype==bool and patch_rgb.shape==(*patch_alpha.shape,3)
    cut=dict(rgb=patch_rgb,alpha=patch_alpha);alpha=k.V1.translated_alpha(native_rgb.shape[:2],cut,xy)
    if alpha is None:raise ValueError('Frozen patch placement would clip')
    return k.V1.compose(native_rgb,cut,tuple(xy),alpha)

def make_triplet(native_rgb,fixed_input,cue,donor_rgb,selected_visible,other_visible_union,complete_instances,donor_meta):
    """ONE preassigned donor/rank; all failures retain BEFORE and two typed-U afters.

    The root corpus driver supplies bound source/donor pixels later. This function
    never chooses donors, opens files, edits actualq8, computes features or fits.
    """
    k=lower_math();f=copy.deepcopy(fixed_input)
    assert f['source_emulated_not_initializer']and f['fixed_carrier_slot']==0 and not f['carrier_is_native_identity']
    assert native_rgb.dtype==np.uint8 and native_rgb.shape==(*f['raw_hw'],3)
    assert canonical_fold(f['split'])==canonical_fold(donor_meta['split'])
    protected,why=protected_region(k,cue,f['raw_hw']);before=0 if cue.get('before_visible_weak_supported',False)else-1
    facts=dict(pair_ready=False,reason=why,same_q_L_context=True,source_emulated_not_native=True,native_mapping='UNKNOWN',
        human_gold=False,protected_pixels=0 if protected is None else int(protected.sum()))
    def fail(reason):
        facts['reason']=reason
        return dict(images=(native_rgb,native_rgb,native_rgb),weak_labels=(before,-1,-1),fixed_inputs=(copy.deepcopy(f),copy.deepcopy(f),copy.deepcopy(f)),facts=facts,patch=None,protected=protected)
    if protected is None:return fail(why)
    if selected_visible is None or other_visible_union is None:return fail('MISSING_COMPLETE_DONOR_MASKS_U')
    exclusive,ownership=k.alpha_rule.exclusive_alpha(selected_visible,other_visible_union,complete_instances);facts['exclusive_alpha']=ownership
    if not ownership['admitted']:return fail(ownership['reason'])
    if donor_rgb is None:return fail('MISSING_PREASSIGNED_DONOR_RGB_U')
    cut,scale=k.V2.support_scaled_cutout(donor_rgb,exclusive,protected,f['L']);facts['scale']=scale
    if cut is None:return fail(scale['reason'])
    # Legacy field used solely as fixed placement-hash salt, not semantic identity.
    row=dict(input_id=f['input_id'],native_slot_logical_only=0,query_xy=f['query_xy'],L=f['L'],raw_hw=f['raw_hw'],split=canonical_fold(f['split']))
    pair,mass=k.V2.core_aligned_pair(row,cut,protected);facts['mass']=mass
    if pair is None:return fail(mass['reason'])
    (cxy,ca),(sxy,sa)=pair
    cover=k.V1.compose(native_rgb,cut,cxy,ca);sham=k.V1.compose(native_rgb,cut,sxy,sa)
    assert ca[protected].all()and not sa[protected].any()and np.array_equal(sham[protected],native_rgb[protected])
    passed,recheck=k.V2.mass_certificate(row,ca,sa);assert passed and ca.sum()==sa.sum()==cut['alpha'].sum()
    assert np.array_equal(reconstruct(native_rgb,cut['rgb'],cut['alpha'],cxy,k),cover)and np.array_equal(reconstruct(native_rgb,cut['rgb'],cut['alpha'],sxy,k),sham)
    facts.update(pair_ready=True,reason='REAL_CUTOUT_COMPLETE_CUE_COVER_AND_SHAM_READY',cover_xy=list(cxy),sham_xy=list(sxy),
        same_shared_RGB_alpha_scale=True,protected_cover_all1=True,protected_sham_all0=True,sham_protected_RGB_bitexact=True,mass_recheck=recheck,
        no_target_SAM_mask_used=True,no_scale_or_donor_reroll=True)
    patch=dict(rgb=cut['rgb'],alpha=cut['alpha'],cover_xy=list(cxy),sham_xy=list(sxy),donor_native_bbox_xyxy=cut['donor_native_bbox_xyxy'],resized_hw=cut['resized_hw'])
    return dict(images=(native_rgb,cover,sham),weak_labels=(before,1,0),fixed_inputs=(copy.deepcopy(f),copy.deepcopy(f),copy.deepcopy(f)),facts=facts,patch=patch,protected=protected)
