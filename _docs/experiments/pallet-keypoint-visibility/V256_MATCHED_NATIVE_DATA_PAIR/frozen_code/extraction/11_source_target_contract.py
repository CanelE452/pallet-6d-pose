"""Pure prospective loss eligibility and paired-batch contracts; no file access."""
import math
import numpy as np

META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')

def need(v,m):
    if not bool(v):raise AssertionError(m)

def finite_pair(q):
    return q is not None and len(q)==2 and all(math.isfinite(float(x))for x in q)

def point_loss_target(r):
    """Read saved own-image state. Keep states intact and return loss-target only."""
    reasons=[]
    if r.get('component_role')!='TRAIN':reasons.append('NOT_TRAIN')
    if r.get('mapping_status')!='EXACT_ORIGINAL_IMAGE_OWN_SCENE':reasons.append('MISSING_IMAGE_MAPPING')
    if r.get('family')not in('SELF_SCENE15','BEFORE_ANCHOR4')or r.get('state_name')!='NOMINAL'or r.get('state_index')!=0:reasons.append('NOT_NOMINAL_ELIGIBLE')
    if r.get('original_query_available')is not True:reasons.append('ORIGINAL_INPUT_U')
    q=r.get('query_xy');L=r.get('L');ref=r.get('saved_projected_reference_xy')
    finite_q=finite_pair(q)and L is not None and math.isfinite(float(L))and float(L)>0
    if not finite_q:reasons.append('MISSING_Q_OR_L')
    if r.get('point0_target_alone_state')!='D':reasons.append('TARGET_ALONE_NOT_D')
    state=r.get('point0_combined_state')
    if state not in('D','E'):reasons.append('COMBINED_NOT_DE')
    loc=r.get('reference_locality_for_actual_input',{})
    if not finite_pair(ref):reasons.append('MISSING_REFERENCE')
    if loc.get('inside_original_frame_canvas_extent')is not True:reasons.append('REFERENCE_OUTSIDE_FRAME')
    if loc.get('inside_crop_canvas_extent')is not True:reasons.append('REFERENCE_OUTSIDE_ACTUAL_INPUT_CROP')
    if finite_q and finite_pair(ref):
        uv=[47.5+96*(float(ref[j])-float(q[j]))/float(L)for j in range(2)]
        expected=all(-.5<=v<95.5 for v in uv)
        need(expected==loc.get('inside_crop_canvas_extent'),'Saved actual-input crop locality exact; no q correction')
        need(np.array_equal(np.asarray(uv),np.asarray(loc.get('raw96_reference_index_xy')),equal_nan=True),'Saved crop index algebra exact')
    return (-1 if reasons else int(state=='E')),reasons

def join_point_rows(cache_metadata,teacher_rows,immutable_train,nominal_eligible):
    """Use row identity then verify all META6 and source-image key; not stale indices."""
    n=len(cache_metadata);need(immutable_train.dtype==nominal_eligible.dtype==bool and immutable_train.shape==nominal_eligible.shape==(n,),'All-row source masks')
    byid={r['row_id']:i for i,r in enumerate(cache_metadata)};need(len(byid)==n,'Unique original rows')
    target=np.full(n,-1,np.int8);seen=set();selected=[];reasons=[]
    for r in teacher_rows:
        rid=r['row_id'];need(rid in byid and rid not in seen,'Exact unique teacher row');seen.add(rid);i=byid[rid];m=cache_metadata[i]
        for k in META:
            if k in('query_xy','L'):need(np.array_equal(np.asarray(m[k],object),np.asarray(r[k],object)),'Original q/L exact')
            else:need(m[k]==r[k],'Original META6 exact '+k)
        need(immutable_train[i]and nominal_eligible[i],'Teacher only original TRAIN/nominal rows')
        need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']],'Teacher is actual image scene, not anchor')
        need(r['source_image_original_global_index']==m['source_image_original_global_index']and r['anchor_original_global_index']==m['anchor_original_global_index'],'Image and q-anchor remain distinct')
        t,why=point_loss_target(r);target[i]=t;selected.append(i);reasons.append(dict(row_id=rid,loss_target=t,excluded_reasons=why))
    expected={cache_metadata[i]['row_id']for i in np.flatnonzero(immutable_train&nominal_eligible)}
    need(seen==expected,'All eligible7040 rows retained including S/U/outside/missing')
    need((target[~(immutable_train&nominal_eligible)]==-1).all(),'No non-TRAIN/jitter target')
    return target,np.asarray(selected,np.int64),reasons

def balanced_pools(target,available,mask):
    need(target.dtype==np.int8 and available.dtype==mask.dtype==bool and target.shape==available.shape==mask.shape,'Typed full-row target/masks')
    need(np.isin(target,[-1,0,1]).all()and(target[~available]==-1).all(),'S/U/unknown not external')
    p=[np.flatnonzero(mask&available&(target==j))for j in(0,1)]
    need(all(len(x)>0 for x in p),'Both prescribed classes required; no fallback')
    return p

def balanced_loss(logit,target):
    import torch
    from torch.nn import functional as F
    need(logit.ndim==target.ndim==1 and logit.shape==target.shape,'Scalar class-balanced BCE')
    need(((target==0)|(target==1)).all(),'Loss has known D/E only')
    return sum(F.binary_cross_entropy_with_logits(logit[target==j],target[target==j].to(logit.dtype))for j in(0,1))/2

def objective(a,point_target,weak_target,pure,weak_enabled):
    need(a['support_logits'].shape==(240 if weak_enabled else 224,2),'Fixed old192+point32+optionalweak16')
    base,components=pure['five_losses']({k:a[k][:192]for k in('support_logits','owner_logits')})
    point=balanced_loss(a['direct_end_external_logit'][192:224],point_target)
    weak=balanced_loss(a['direct_end_external_logit'][224:240],weak_target)if weak_enabled else point.new_zeros(())
    return base+point+.25*weak,base,components,point,weak
