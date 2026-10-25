"""Pure saved-pair checks; no file I/O, producer import, rendering or model code."""
import math
import numpy as np

def required_four_taps(query_xy,L,reference_xy):
    q=np.asarray(query_xy,dtype=np.float64);r=np.asarray(reference_xy,dtype=np.float64)
    assert q.shape==r.shape==(2,) and np.isfinite(q).all() and np.isfinite(r).all()
    assert math.isfinite(float(L)) and L>0
    centre=np.floor(r+.5).astype(np.int64)
    result=set()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            source=centre+np.array([dx,dy],np.int64)
            uv=(source-q)*96.0/float(L)+47.5
            lo=np.floor(uv).astype(np.int64)
            # Contract intentionally requires all four neighbours, even at an integer uv.
            for sy in (0,1):
                for sx in (0,1):result.add((int(lo[0]+sx),int(lo[1]+sy)))
    inside=all(0<=x<96 and 0<=y<96 for x,y in result)
    return sorted(result),inside

def taps_mask(taps):
    m=np.zeros((96,96),bool)
    for x,y in taps:
        assert 0<=x<96 and 0<=y<96
        m[y,x]=True
    return m

def shifted_mask_and_pixels(mask,rgb,offset_xy):
    assert mask.dtype==np.bool_ and mask.shape==(96,96)
    assert rgb.dtype==np.uint8 and rgb.shape==(96,96,3)
    dx,dy=offset_xy;assert isinstance(dx,int) and isinstance(dy,int)
    y,x=np.nonzero(mask);xx=x+dx;yy=y+dy
    assert ((xx>=0)&(xx<96)&(yy>=0)&(yy<96)).all(),'alpha clipping changes footprint'
    out=np.zeros_like(mask);out[yy,xx]=True
    pixels=np.zeros_like(rgb);pixels[yy,xx]=rgb[y,x]
    return out,pixels

def unchanged_input_metadata(original,variant):
    # Schema adapter must provide these explicit deployment fields from the sealed rows.
    keys=('parent_row_id','family','query_xy','L','native_slot','condition32','valid_sha256','original_query_available')
    assert all(k in original and k in variant for k in keys)
    for k in keys:assert original[k]==variant[k],k

def validate_constructed_pair(base,valid,cover,sham,cover_alpha,sham_alpha,taps,sham_offset_xy,parent,donor,variant_meta):
    for rgb in (base,cover,sham):assert rgb.dtype==np.uint8 and rgb.shape==(96,96,3)
    for m in (valid,cover_alpha,sham_alpha):assert m.dtype==np.bool_ and m.shape==(96,96)
    assert parent['internal_source_role']=='INTERNAL_SOURCE_FIT' and parent['target']==0 and parent['original_query_available'] is True
    assert donor['internal_source_role']=='INTERNAL_SOURCE_FIT' and donor['original_associated_r_state']=='D'
    if parent['family']=='YCB_AUX':assert donor['object_id']!=parent['object_id']
    required=taps_mask(taps);assert required.any() and valid[required].all()
    assert cover_alpha[required].all() and not sham_alpha[required].any()
    assert not cover_alpha[~valid].any() and not sham_alpha[~valid].any()
    assert np.array_equal(cover[~cover_alpha],base[~cover_alpha]),'COVER changes outside alpha'
    assert np.array_equal(sham[~sham_alpha],base[~sham_alpha]),'SHAM changes outside alpha'
    translated,translated_rgb=shifted_mask_and_pixels(cover_alpha,cover,sham_offset_xy)
    assert np.array_equal(translated,sham_alpha),'same donor/size alpha footprint mismatch'
    assert np.array_equal(translated_rgb[sham_alpha],sham[sham_alpha]),'opaque layer RGB differs between matched placements'
    assert int(sham_alpha.sum())>=64
    assert not np.array_equal(cover[cover_alpha],base[cover_alpha]),'no actual COVER pixel change'
    assert not np.array_equal(sham[sham_alpha],base[sham_alpha]),'no actual SHAM pixel change'
    for name,target in (('BEFORE',0),('COVER',1),('SHAM',0)):
        v=variant_meta[name];unchanged_input_metadata(parent,v)
        assert v['target']==target and v['input_available'] is True
    return {'required_taps':int(required.sum()),'cover_alpha_pixels':int(cover_alpha.sum()),'sham_alpha_pixels':int(sham_alpha.sum()),'cover_changed_pixels':int(np.any(cover!=base,axis=2).sum()),'sham_changed_pixels':int(np.any(sham!=base,axis=2).sum()),'same_opaque_layer_under_integer_translation':True}

def validate_failed_construction(parent,variants):
    assert parent['target']==0 and parent['original_query_available'] is True
    assert variants['BEFORE']['target']==0 and variants['BEFORE']['input_available'] is True
    for name in ('COVER','SHAM'):
        v=variants[name]
        assert v['constructed'] is False and v['target'] in (None,-1)
        assert v['input_available'] is False and v['RGB'] is None
    return True
