"""Pure source pair API; real pixels require a later root prototype entry."""
from pathlib import Path
import hashlib
import importlib.util
import math
import numpy as np

P=Path(__file__).resolve().parents[2]
def module(path,name,expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
adapter=module(P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/protected_union_adapter.py','v96_bound_union_adapter','62a1ecdb7f945f5b12e8a2d9cf55648e16e1d8922000252babf0c5918800ef3b')
alpha_rule=module(P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/image_only_v2_exclusive_alpha/exclusive_alpha_rule.py','v96_bound_exclusive_alpha','145a2da15891f71a12475ff26f147d76e328e581c70f15ebf5ec187a1aaf6dd9')
V2=adapter.v3.v2;V1=adapter.v3.v1
K=16;SALT='V96_QUERY_ANCHORED_CUE_K16_V1';SEED=20261023

def canonical(split):return {'TRAIN':'TRAIN','VALIDATION':'VALIDATION','TRAIN_LAB':'TRAIN','VAL_HALL':'VALIDATION'}[split]
def rank_key(row,donor):
    return hashlib.sha256('|'.join(map(str,[SEED,SALT,row['input_id'],row['native_slot_logical_only'],donor['id'],donor['image_sha256']])).encode()).hexdigest()
def donor_allowed(row,d):
    train=canonical(row['split'])=='TRAIN'
    return d['split']==canonical(row['split']) and d['image_sha256']!=row['image_sha256'] and 1<=d['object_id']<=21 and d['object_id']%2==int(train) and ((0<=d['scene_id']<=7)if train else(8<=d['scene_id']<=9))
def fixed_donors(row,pool):
    eligible=[d for d in pool if donor_allowed(row,d)]
    assert len({d['id']for d in eligible})==len(eligible),'Duplicate donor IDs in frozen metadata pool'
    ranked=sorted(eligible,key=lambda d:rank_key(row,d));assert len(ranked)>=K
    return ranked[:K]
def central_stencil(q,L):
    cells=set()
    for sy in [-1,1]:
        for sx in [-1,1]:
            x=q[0]+sx*L/192;y=q[1]+sy*L/192;i=math.floor(x);j=math.floor(y);dx=x-i;dy=y-j
            for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
                if w>0:cells.add((i+ox,j+oy))
    return sorted(cells)
def protected_mask(row,target):
    """Whole original matched-group boxes plus stencil; no other-cue union."""
    adapter.v3.checked_row(row)
    assert row['query_xy']==target['original_q'] and row['L']==target['original_L']
    if target['query_anchored_label']!=0:return None,'NOT_QUERY_ANCHORED_D_U'
    if not target['matched_certified_group']or not target['original_regions']:return None,'MISSING_MATCHED_GROUP_GEOMETRY_U'
    q=row['query_xy'];L=row['L']
    if q is None or L is None or not np.isfinite(q).all()or not np.isfinite(L)or L<=0:return None,'MISSING_QUERY_U'
    stencil=central_stencil(q,L);assert [list(p)for p in stencil]==target['central4_native_stencil']
    h,w=row['raw_hw'];out=np.zeros((h,w),bool)
    for region in target['original_regions']:
        if region['kind']=='PALLOX_CLOSED_NATIVE_UNCERTAINTY':m=adapter.closed_range_cells(region['bbox'],(h,w))
        elif region['kind']in['LOCO_HALFOPEN_SUPPORT','LOCO_HALFOPEN_UNCERTAINTY']:m=adapter.v3.pixel_edge_box_cells(region['bbox'],(h,w))
        else:raise ValueError('Unfrozen source raster convention')
        if m is None:return None,'UNUSABLE_COMPLETE_MATCHED_GROUP_RASTER_U'
        out|=m
    for x,y in stencil:
        if not(0<=x<w and 0<=y<h):return None,'CENTRAL_STENCIL_OOB_U'
        out[y,x]=True
    return out,'FULL_MATCHED_CUE_GROUP_PLUS_QUERY_STENCIL'

def prepare_rank(native_rgb,row,target,donor_rgb,selected_visible,other_visible_union,complete,donor_meta):
    """Independent pair for one PREASSIGNED rank; never replaces/searches ranks."""
    assert native_rgb.dtype==np.uint8 and native_rgb.shape==(*row['raw_hw'],3)
    protected,reason=protected_mask(row,target)
    facts=dict(pair_ready=False,reason=reason,donor_id=None if donor_meta is None else donor_meta['id'],no_query_or_L_change=True,
        protected_pixels=0 if protected is None else int(protected.sum()),
        protected_area_over_L_squared=None if protected is None else float(protected.sum()/row['L']**2),
        complete_original_regions_count=len(target['original_regions']),source_family=target.get('source_family','GENERATED'))
    if protected is None:return dict(images=(native_rgb.copy(),native_rgb.copy()),after_labels=(-1,-1),facts=facts)
    if donor_meta is None or not donor_allowed(row,donor_meta)or donor_meta.get('foreign_opaque_cutout_admitted')is not True:
        facts['reason']='ASSIGNED_DONOR_METADATA_NOT_ADMITTED_U'
        return dict(images=(native_rgb.copy(),native_rgb.copy()),after_labels=(-1,-1),facts=facts)
    exclusive,ownership=alpha_rule.exclusive_alpha(selected_visible,other_visible_union,complete);facts['exclusive_alpha']=ownership
    if not ownership['admitted']:
        facts['reason']=ownership['reason'];return dict(images=(native_rgb.copy(),native_rgb.copy()),after_labels=(-1,-1),facts=facts)
    cut,scale=V2.support_scaled_cutout(donor_rgb,exclusive,protected,row['L']);facts['scale']=scale
    if cut is None:
        facts['reason']=scale['reason'];return dict(images=(native_rgb.copy(),native_rgb.copy()),after_labels=(-1,-1),facts=facts)
    normalized=dict(row,split=canonical(row['split']));pair,mass=V2.core_aligned_pair(normalized,cut,protected);facts['mass']=mass
    if pair is None:
        facts['reason']=mass['reason'];return dict(images=(native_rgb.copy(),native_rgb.copy()),after_labels=(-1,-1),facts=facts)
    (cxy,ca),(sxy,sa)=pair
    assert ca[protected].all()and not sa[protected].any()and ca.sum()==sa.sum()==cut['alpha'].sum()
    cover=V1.compose(native_rgb,cut,cxy,ca);sham=V1.compose(native_rgb,cut,sxy,sa)
    assert np.array_equal(sham[protected],native_rgb[protected])
    passed,recheck=V2.mass_certificate(normalized,ca,sa);assert passed
    facts.update(pair_ready=True,reason='QUERY_ANCHORED_PAIR_READY',protected_pixels=int(protected.sum()),
        cover_xy=list(cxy),sham_xy=list(sxy),same_full_RGBA_transform=True,protected_cover_all1=True,protected_sham_all0=True,mass_recheck=recheck)
    return dict(images=(cover,sham),after_labels=(1,0),facts=facts)

def occurrence_plan(rows,targets):
    assert len(rows)==len(targets)==960;out=[]
    for i,(row,target)in enumerate(zip(rows,targets)):
        assert row['slot_index']==target['base_row_index']==i
        for j in range(33):
            rank=None if j==0 else(j-1)//2;variant='BEFORE'if j==0 else'OPAQUE_COVER'if j%2 else'CUE_PRESERVING_SHAM'
            oid=f"{row['input_id']}::slot{row['native_slot_logical_only']}::V96::"+('BEFORE'if j==0 else f'RANK_{rank:02d}::{variant}')
            out.append(dict(occurrence_index=33*i+j,occurrence_id=oid,base_row_index=i,donor_rank=rank,variant=variant))
    assert len(out)==31680 and len({r['occurrence_id']for r in out})==31680
    return out
