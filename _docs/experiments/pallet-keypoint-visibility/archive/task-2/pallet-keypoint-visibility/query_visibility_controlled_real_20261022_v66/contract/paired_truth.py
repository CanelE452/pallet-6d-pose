"""Pure partially known native support labels for conditional paired-real evidence.

No image reader, model, renderer, pose, fitting, or private-evaluation dependency.
SAM masks are supplied teachers, never complete visible-negative annotations.
Full-negative is disabled by the registered paired-data construction.
"""
import numpy as np

U,V,E,O=-1,0,1,2
JOINT_PAIRS=((V,V),(E,V),(E,E),(O,V),(O,E),(O,O))
CLASS_NAMES={U:'UNKNOWN',V:'VISIBLE_TEACHER_SUPPORTED',E:'EXTERNAL_SUPPORT_TEACHER_DEPTH_SUPPORTED',O:'OUTSIDE_CERTIFIED_SUPPORT'}


def _bool2(a,shape=None):
    a=np.asarray(a)
    if a.ndim!=2 or a.dtype!=np.bool_ or (shape is not None and a.shape!=shape):
        raise ValueError('Aligned explicit Boolean native2D evidence required')
    return a


def construct_certainty(visible_positive,visible_negative,full_positive,full_negative):
    """Logically close positive-visible and negative-full facts; reject conflicts."""
    vp=_bool2(visible_positive).copy();shape=vp.shape
    vn=_bool2(visible_negative,shape).copy();fp=_bool2(full_positive,shape).copy();fn=_bool2(full_negative,shape).copy()
    fp|=vp;vn|=fn
    if (vp&vn).any() or (fp&fn).any():raise ValueError('Contradictory visible/full certainty')
    exact=np.full(shape,U,np.int8);exact[fn]=O;exact[fp&vn]=E;exact[vp]=V
    return dict(visible_positive=vp,visible_negative=vn,visible_known=vp|vn,
                full_positive=fp,full_negative=fn,full_known=fp|fn,
                full_unknown=~fp&~fn,exact_state=exact)


def teacher_interior(teacher_mask):
    """Fixed native erode2, a5x5 all-positive square, constant-false border."""
    a=_bool2(teacher_mask);p=np.pad(a,2,constant_values=False)
    return np.lib.stride_tricks.sliding_window_view(p,(5,5)).all(axis=(-2,-1))


def construct_symmetric_pair(teacher_a,teacher_b,depth_a_mm,depth_b_mm,mesh_depth_mm,*,pair_eligible,depth_gap_mm=50.0):
    """Frozen symmetric conditional TEACHER evidence; depth-gap sensitivity50/100/200 only.

    pair_eligible is an explicit metadata/geometry admission bit, not inferred here.
    Camera-Z/mm and common native RGB-depth grid are caller assumptions. NoO.
    The2*tau margin uses the VISIBLE target-reference view, not foreground depth.
    """
    ta=_bool2(teacher_a);tb=_bool2(teacher_b,ta.shape)
    da,db,m=[np.asarray(x,np.float64)for x in (depth_a_mm,depth_b_mm,mesh_depth_mm)]
    if any(x.shape!=ta.shape for x in (da,db,m)):raise ValueError('Aligned native depth required')
    if type(pair_eligible) not in (bool,np.bool_):raise ValueError('Explicit scalar pair eligibility required')
    if depth_gap_mm not in (50,100,200):raise ValueError('Only predeclared50/100/200mm depth-gap values')
    ia,ib=teacher_interior(ta),teacher_interior(tb)
    za=np.isfinite(da)&(da>0);zb=np.isfinite(db)&(db>0);zm=np.isfinite(m)&(m>0)
    taua=np.maximum(20.0,.01*da);taub=np.maximum(20.0,.01*db)
    agreea=za&zm&(np.abs(da-m)<=taua);agreeb=zb&zm&(np.abs(db-m)<=taub)
    va=bool(pair_eligible)&ia&agreea;vb=bool(pair_eligible)&ib&agreeb
    ea=vb&za&(db-da>=depth_gap_mm)&~va&(m-da>=np.maximum(50.0,2*taub))
    eb=va&zb&(da-db>=depth_gap_mm)&~vb&(m-db>=np.maximum(50.0,2*taua))
    if (va&ea).any()or(vb&eb).any()or(ea&eb).any():raise AssertionError('Symmetric evidence conflict')
    fp=va|vb;fn=np.zeros(ta.shape,bool)
    return dict(A=construct_certainty(va,ea,fp,fn),B=construct_certainty(vb,eb,fp,fn),
                evidence=dict(teacher_interior_a=ia,teacher_interior_b=ib,valid_depth_a=za,valid_depth_b=zb,
                              valid_mesh_depth=zm,tau_a_mm=taua,tau_b_mm=taub,mesh_agree_a=agreea,mesh_agree_b=agreeb,
                              V_A=va,V_B=vb,E_A=ea,E_B=eb,pair_eligible=bool(pair_eligible),depth_gap_mm=float(depth_gap_mm)),
                evidence_tier='CONDITIONAL_PAIRED_SAM_MESH_DEPTH_TEACHER',full_negative_enabled=False)


def region_truth(cert,selector):
    vp,vn,fp,fn=[cert[k][selector]for k in ('visible_positive','visible_negative','full_positive','full_negative')]
    visible=1 if vp.any()else 0 if vn.all()else U
    full=1 if fp.any()else 0 if fn.all()else U
    possible=[]
    if visible!=0 and full!=0:possible.append(V)
    if visible!=1 and full!=0:possible.append(E)
    if visible!=1 and full!=1:possible.append(O)
    if not possible:raise AssertionError('Inconsistent partial evidence')
    return dict(state=possible[0]if len(possible)==1 else U,possible_states=possible,
                visible_status=visible,full_status=full,pixels=int(vp.size),
                known_visible_pixels=int(vp.sum()),known_nonvisible_pixels=int(vn.sum()),
                known_full_positive_pixels=int(fp.sum()),known_full_negative_pixels=int(fn.sum()),
                unknown_visible_pixels=int((~vp&~vn).sum()),unknown_full_pixels=int((~fp&~fn).sum()))


def query_truth(cert,query):
    h,w=cert['exact_state'].shape;q=np.asarray(query,float)
    invalid=dict(valid=False,exact_state=U,radius2_state=U,joint_known=False,joint_state=U,
                 joint_compatible_codes=[],joint_compatible_mask=0,semantic_self=None,corner_semantic_truth=None)
    if q.shape!=(2,)or not np.isfinite(q).all():return invalid
    x,y=np.floor(q+.5).astype(np.int64)
    if not(0<=q[0]<w and 0<=q[1]<h and 0<=x<w and 0<=y<h):return invalid
    x0,x1=max(0,x-2),min(w,x+3);y0,y1=max(0,y-2),min(h,y+3)
    sel=(slice(y0,y1),slice(x0,x1));exact=region_truth(cert,(slice(y,y+1),slice(x,x+1)))
    near=region_truth(cert,sel);bits={k:cert[k][sel]for k in ('visible_positive','visible_negative','full_positive','full_negative')}
    other=np.ones((y1-y0,x1-x0),bool);other[y-y0,x-x0]=False;other_truth=region_truth(bits,other)
    def combine(a,b):return V if V in(a,b)else E if E in(a,b)else O
    codes=sorted({JOINT_PAIRS.index((a,combine(a,b)))for a in exact['possible_states']for b in other_truth['possible_states']})
    known=exact['state']!=U and near['state']!=U
    joint=JOINT_PAIRS.index((exact['state'],near['state']))if known else U
    if known and codes!=[joint]:raise AssertionError('Inconsistent exact/near completion')
    return dict(valid=True,pixel=[int(x),int(y)],exact_state=exact['state'],radius2_state=near['state'],
                exact=exact,radius2=near,window_without_exact=other_truth,joint_known=known,joint_state=joint,
                joint_compatible_codes=codes,joint_compatible_mask=int(sum(1<<c for c in codes)),
                radius2_strict_E_core=bool(near['pixels']==25 and (cert['exact_state'][sel]==E).all()),
                radius2_strict_O_core=bool(near['pixels']==25 and near['known_full_negative_pixels']==25),
                radius2_clipped_by_frame=near['pixels']!=25,semantic_self=None,corner_semantic_truth=None)


def label_queries(cert,queries):return [query_truth(cert,q)for q in np.asarray(queries)]
