"""Pure native-pixel controlled-occlusion certainty, no images/models/poses.

Primary assumption: original visible-mask negatives are source-known negatives.
Original full-only pixels still have UNKNOWN full membership for CONTROLLED GT.
An optional untrusted-negative audit mode propagates visibility uncertainty too;
it is never silently substituted for the registered primary label contract.
"""
import numpy as np

U,V,E,O=-1,0,1,2
CLASS_NAMES={U:'UNKNOWN',V:'VISIBLE',E:'NEW_CONTROLLED_EXTERNAL_SUPPORT',O:'OUTSIDE_TARGET_SUPPORT'}
JOINT_PAIRS=((V,V),(E,V),(E,E),(O,V),(O,E),(O,O))


def decode_public_masks(original_visible,original_full):
    visible,full=np.asarray(original_visible),np.asarray(original_full)
    if visible.ndim!=2 or full.shape!=visible.shape:raise ValueError('Aligned native single-channel masks required')
    for a in [visible,full]:
        if not set(np.unique(a))<={0,255}:raise ValueError('Registered public source masks must be0/255')
    visible,full=visible>0,full>0
    if (visible&~full).any():raise ValueError('Original visible is not subset of original full')
    return visible,full


def construct_certainty(original_visible,original_full,paste_mask,visible_negative_trusted=True):
    visible,full,paste=map(np.asarray,[original_visible,original_full,paste_mask])
    if visible.ndim!=2 or visible.shape!=full.shape or visible.shape!=paste.shape:
        raise ValueError('Aligned native2D masks required')
    if any(a.dtype!=np.bool_ for a in [visible,full,paste]):raise ValueError('Explicit Boolean masks required')
    if (visible&~full).any():raise ValueError('Original visible is not subset of full')
    trust=np.broadcast_to(np.asarray(visible_negative_trusted,bool),visible.shape)
    full_positive=visible.copy();full_negative=~full
    visible_positive=visible&~paste
    # Outside-full cannot be visible even in untrusted-negative audit mode.
    visible_negative=full_negative|(visible&paste)|(~visible&trust)
    if (visible_positive&visible_negative).any():raise AssertionError('Contradictory visibility certainty')
    exact=np.full(visible.shape,U,np.int8)
    exact[full_negative]=O
    exact[full_positive&visible_negative]=E
    exact[visible_positive]=V
    return dict(original_visible=visible.copy(),original_full=full.copy(),paste_mask=paste.copy(),
                visible_positive=visible_positive,visible_negative=visible_negative,
                visible_known=visible_positive|visible_negative,
                full_positive=full_positive,full_negative=full_negative,
                full_known=full_positive|full_negative,full_unknown=full&~visible,
                exact_state=exact,original_proxy_state=np.where(visible,V,np.where(full,E,O)).astype(np.int8),
                controlled_E_pixel=visible&paste,
                original_visible_negative_assumption=np.broadcast_to(trust,visible.shape).copy())


def _region(cert,selector):
    vp=cert['visible_positive'][selector]
    vn=cert['visible_negative'][selector]
    fp=cert['full_positive'][selector]
    fn=cert['full_negative'][selector]
    visible_status=1 if vp.any() else 0 if vn.all() else U
    full_status=1 if fp.any() else 0 if fn.all() else U
    if visible_status==1 and full_status==0:raise AssertionError('Visible outside full support')
    possible=[]
    if visible_status!=0 and full_status!=0:possible.append(V)
    if visible_status!=1 and full_status!=0:possible.append(E)
    if visible_status!=1 and full_status!=1:possible.append(O)
    if not possible:raise AssertionError('Inconsistent region facts')
    state=possible[0]if len(possible)==1 else U
    return dict(state=state,possible_states=possible,visible_status=visible_status,full_status=full_status,
                pixels=int(vp.size),known_visible_pixels=int(vp.sum()),
                known_nonvisible_pixels=int(vn.sum()),certified_full_positive_pixels=int(fp.sum()),
                certified_full_negative_pixels=int(fn.sum()),unknown_full_pixels=int((~fp&~fn).sum()))


def _combine(exact,other):
    if V in(exact,other):return V
    return E if E in(exact,other)else O


def query_truth(cert,query,radius=2):
    if radius!=2:raise ValueError('Only the fixed native radius2 joint task is registered')
    h,w=cert['exact_state'].shape;q=np.asarray(query,float)
    invalid=dict(valid=False,exact_state=U,radius2_state=U,joint_known=False,joint_state=U,
                 joint_compatible_codes=[],joint_compatible_mask=0,semantic_self=None,corner_semantic_truth=None)
    if q.shape!=(2,)or not np.isfinite(q).all():return invalid
    px,py=np.floor(q+.5).astype(np.int64)
    if not(0<=q[0]<w and 0<=q[1]<h and 0<=px<w and 0<=py<h):return invalid
    x0,x1=max(0,px-2),min(w,px+3);y0,y1=max(0,py-2),min(h,py+3)
    exact_selector=(slice(py,py+1),slice(px,px+1))
    window_selector=(slice(y0,y1),slice(x0,x1))
    window_bits={k:cert[k][window_selector]for k in ['visible_positive','visible_negative','full_positive','full_negative']}
    other_selector=np.ones((y1-y0,x1-x0),bool);other_selector[py-y0,px-x0]=False
    exact=_region(cert,exact_selector);near=_region(cert,window_selector);other=_region(window_bits,other_selector)
    possible={JOINT_PAIRS.index((a,_combine(a,b)))for a in exact['possible_states']for b in other['possible_states']}
    codes=sorted(possible)
    known=exact['state']!=U and near['state']!=U
    joint=JOINT_PAIRS.index((exact['state'],near['state']))if known else U
    if known and codes!=[joint]:raise AssertionError('Known states have inconsistent joint completions')
    strong=cert['controlled_E_pixel'][window_selector]
    return dict(valid=True,pixel=[int(px),int(py)],exact_state=exact['state'],radius2_state=near['state'],
                exact=exact,radius2=near,window_without_exact=other,
                joint_known=known,joint_state=joint,joint_compatible_codes=codes,
                joint_compatible_mask=int(sum(1<<j for j in codes)),
                exact_controlled_E=bool(cert['controlled_E_pixel'][py,px]),
                exact_original_hidden_proxy_unknown=bool(cert['full_unknown'][py,px]),
                radius2_contains_unknown_full=near['unknown_full_pixels']>0,
                radius2_E_uses_source_visible_negative_assumption=bool(near['state']==E and near['unknown_full_pixels']>0),
                radius2_strict_new_controlled_E_core=bool(strong.size==25 and strong.all()),
                radius2_strict_known_O_core=bool(near['pixels']==25 and near['certified_full_negative_pixels']==25),
                radius2_clipped_by_frame=near['pixels']!=25,
                semantic_self=None,corner_semantic_truth=None)


def label_queries(cert,queries):
    return [query_truth(cert,q)for q in np.asarray(queries)]
