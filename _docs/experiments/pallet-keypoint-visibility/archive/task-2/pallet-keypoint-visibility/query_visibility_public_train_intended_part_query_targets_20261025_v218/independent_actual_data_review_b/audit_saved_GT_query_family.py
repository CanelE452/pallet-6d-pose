#!/usr/bin/env python3
"""All360 independent sampler QA from exact12 RGB PNG, never producer import."""
import argparse
import collections
import io
import json
import math
import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
from audit_saved_weak_pairs import rb, sha, bind, checked, put

H = Path(__file__).absolute().parent
V = H.parent


def independent_sampler(rgb, q, L):
    """Flattened coordinate/tap scatter, F64 accumulation in declared tap order."""
    n = 96*96
    accum = np.zeros((n,3), np.float64)
    support = np.zeros(n, np.float64)
    bad_positive_tap = np.zeros(n, bool)
    if q is None or not all(math.isfinite(float(x)) for x in [*q,L]) or L <= 0:
        return accum.reshape(96,96,3).astype(np.uint8), np.zeros((96,96),bool), support.reshape(96,96)
    step = float(L)/96.
    xaxis = float(q[0])+(np.arange(96,dtype=np.float64)-47.5)*step
    yaxis = float(q[1])+(np.arange(96,dtype=np.float64)-47.5)*step
    xx,yy = np.tile(xaxis,96),np.repeat(yaxis,96)
    lowx,lowy = np.floor(xx).astype(np.int64),np.floor(yy).astype(np.int64)
    fx,fy = xx-lowx,yy-lowy
    height,width = rgb.shape[:2]
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        weight = (fx if dx else 1-fx)*(fy if dy else 1-fy)
        ix,iy = lowx+dx,lowy+dy
        inside = (ix>=0)&(iy>=0)&(ix<width)&(iy<height)
        bad_positive_tap |= (weight>0)&~inside
        good = np.nonzero(inside)[0]
        support[good] += weight[good]
        accum[good] += weight[good,None]*rgb[iy[good],ix[good]].astype(np.float64)
    pixels = np.rint(np.clip(accum,0.,255.)).astype(np.uint8).reshape(96,96,3)
    return pixels,(~bad_positive_tap).reshape(96,96),support.reshape(96,96)


def decode_runs(runs):
    a = np.zeros(96*96,bool)
    previous_end = 0
    for start,length in runs:
        assert type(start) is int and type(length) is int and length>0 and previous_end <= start and start+length <= 9216
        assert not a[start:start+length].any()
        a[start:start+length] = True
        previous_end = start+length
    return a.reshape(96,96)


def in_frame(q, hw):
    return q is not None and len(q)==2 and all(math.isfinite(float(x)) for x in q) and -.5 <= q[0] < hw[1]-.5 and -.5 <= q[1] < hw[0]-.5


def direct_assertions():
    rgb = np.full((2,2,3),100,np.uint8)
    a,v,m = independent_sampler(rgb,[0.,0.],96.)
    assert np.array_equal(a[47,47],[25]*3) and m[47,47]==.25 and not v[47,47]
    assert np.array_equal(a[48,48],[100]*3) and m[48,48]==1 and v[48,48]
    src = np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8)
    a,v,m = independent_sampler(src,[47.5,47.5],96.)
    assert np.array_equal(a,src) and v.all() and (m==1).all()
    assert np.array_equal(decode_runs([[0,3],[8,2]]).reshape(-1).nonzero()[0],[0,1,2,8,9])
    return ['partial source is zero-padded without renormalization',
            'integer sample edge zero-weight taps do not invalidate',
            'row-major start/length RLE independent decode']


def audit(entry_path):
    e = json.loads(rb(entry_path))
    assert e['approved_independent_GT_family_pixel_QA'] is True and e['code']==bind(Path(__file__))
    small_checks = direct_assertions()
    done = json.loads(checked(e['COMPLETE']))
    scope = json.loads(checked(e['root_scope']))
    assert done['completed'] and done['PASS'] and done['training_admission'] is False
    assert scope['approved_PUBLIC_SOURCE_GT_QUERY_JITTER_V1'] and scope['approved_exact12_derived_RGB_PNG']
    assert done['root_actual_scope']==e['root_scope']
    plan = json.loads(checked(done['plan']))
    summary = json.loads(checked(done['summary']))
    ledger = [json.loads(x) for x in checked(done['all360']).decode().splitlines()]
    manifest = json.loads(checked(done['images']))
    assert len(ledger)==len(plan['rows360'])==360 and len(plan['groups45'])==45
    assert manifest['training_RGB360']==[r['RGB96'] for r in ledger]
    source12 = scope['exact_PNG12']
    assert plan['exact_source_RGB12']==source12 and len(source12)==12
    owned = manifest['training_RGB360']+manifest['nominal_contacts15']
    assert len(owned)==375 and len({b['path'] for b in owned})==375
    assert all(Path(b['path']).parent==V/'public_source_gt_query_jitter_v1' and Path(b['path']).suffix=='.png' for b in owned)
    # Every original/owned SHA closes before the first image decoder.
    raw = {b['path']:checked(b) for b in [*source12,*owned]}
    out = H/'actual_GT_v1'; os.mkdir(out)
    put(out/'OPAQUE_SHA_BEFORE_ANY_DECODE.json',{'completed':True,'PASS':True,
        'source12':source12,'owned375':owned,'decoder_calls_before_receipt':0,'direct_assertions':small_checks})
    images=[]
    for i,b in enumerate(source12):
        with Image.open(io.BytesIO(raw[b['path']])) as im:
            assert im.format=='PNG' and im.mode=='RGB'
            im.load(); images.append(np.array(im))
        assert list(images[-1].shape[:2])==next(r['source_hw'] for r in ledger if r['frame_position']==i)
    nominal = json.loads(checked(e['saved_nominal120']))['rows120']
    original96 = {}
    nominal_by_key={}
    for r in nominal:
        o=r['original_input']; original96[o['row_id']]=o
        nominal_by_key[(o['frame_position'],r['diagnostic_candidate_gt_list_index'],r['native_slot'])]=r
    assert len(original96)==96 and sum(not r['original_query_available'] for r in original96.values())==11
    assert {r['row_id']:r for r in plan['original_inputs96']}==original96
    groups=collections.defaultdict(list)
    max_mass_delta=0.; roles=collections.Counter(); pixel_rows=[]
    base_keys=set(plan['rows360'][0])
    for r,p in zip(ledger,plan['rows360']):
        assert {k:r[k] for k in base_keys}==p
        old=nominal_by_key[(r['frame_position'],r['candidate_gt_list_index'],r['native_slot'])]
        assert r['original_input']==old['original_input']
        assert r['nominal_reference_xy']==old['nominal_reference_xy'] and r['canonical_index']==old['canonical_index']
        assert r['anatomical_family']==('DECK' if r['canonical_index'] in (0,1,4,5) else 'BOTTOM_RUNNER')
        offset=r['normalized_offset'];assert offset in [[0.,0.],[.25,0.],[0.,.25]]
        ref,L=r['nominal_reference_xy'],r['L']
        expected_q=[float(ref[j])+offset[j]*L for j in range(2)] if ref is not None else None
        assert r['generated_query_xy']==expected_q
        assert r['generated_query_available']==in_frame(expected_q,r['source_hw'])
        assert r['nominal_target_in_frame']==in_frame(ref,r['source_hw'])
        assert r['source_role']==r['original_input']['internal_source_role']
        assert r['future_fit_role_eligible']==(r['source_role']=='SOURCE_FIT')
        assert r['training_admission'] is False and r['native_visibility_D_E_SELF_UNKNOWN']=='UNKNOWN'
        assert r['visible_intended_part']=='UNKNOWN' and r['q8_origin']=='GT_DERIVED_TRAINING_INITIALIZATION'
        assert r['annotation_markers_in_training_RGB'] is False
        predicted,valid,mass=independent_sampler(images[r['frame_position']],expected_q,L)
        with Image.open(io.BytesIO(raw[r['RGB96']['path']])) as im:
            assert im.format=='PNG' and im.mode=='RGB' and im.size==(96,96)
            saved=np.array(im)
        assert np.array_equal(predicted,saved),'Independent RGB96 mismatch '+r['row_id']
        assert np.array_equal(decode_runs(r['valid96_RLE']),valid)
        assert np.array_equal(decode_runs(r['any_positive_source_support_RLE']),mass>0)
        assert r['strict_valid_fraction']==float(valid.mean())
        assert r['any_positive_source_support_fraction']==float((mass>0).mean())
        delta=abs(r['mean_source_bilinear_mass']-float(mass.mean()));max_mass_delta=max(max_mass_delta,delta)
        assert delta<1e-14
        assert r['source_RGB_binding']==source12[r['frame_position']]
        roles[r['source_role']]+=1;groups[r['group_id']].append(r)
        pixel_rows.append({'row_id':r['row_id'],'strict_valid_samples':int(valid.sum()),
                           'positive_support_samples':int((mass>0).sum()),'mean_source_mass':float(mass.mean())})
    for g in plan['groups45']:
        rr=groups[g['group_id']];assert [r['native_slot'] for r in rr]==list(range(8))
        assert g['generated_q8']==[r['generated_query_xy'] for r in rr]
        assert g['availability8']==[r['generated_query_available'] for r in rr]
        assert g['L8']==[r['L'] for r in rr]
    assert roles=={'SOURCE_FIT':240,'SOURCE_DEV':120}
    assert summary['roles']==dict(roles)
    assert summary['generated_query_available']==sum(r['generated_query_available'] for r in ledger)
    assert summary['nominal_target_in_frame']==sum(r['nominal_target_in_frame'] for r in ledger)
    assert summary['all_strict_crop_pixels_valid_rows']==sum(r['strict_valid_samples']==9216 for r in pixel_rows)
    assert sum(not r['original_input']['original_query_available'] for r in ledger)==51
    for b in manifest['nominal_contacts15']:
        with Image.open(io.BytesIO(raw[b['path']])) as im:
            assert im.format=='PNG' and im.mode=='RGB' and im.size==(1600,1430)
    gallery=checked(done['gallery']).decode()
    targets=re.findall(r'\]\(([^)]+)\)',gallery)
    allowed={Path(b['path']).name for b in manifest['nominal_contacts15']}|{'SUMMARY.json','IMAGE_MANIFEST.json'}
    assert set(targets)<=allowed and len(targets)==17
    assert all(checked(b)==raw[b['path']] for b in [*source12,*owned])
    result={'completed':True,'PASS':True,'code':bind(Path(__file__)),'protocol':e['protocol'],'entry':bind(entry_path),
        'producer_COMPLETE':e['COMPLETE'],'root_scope':e['root_scope'],'summary':{
        'source_RGB_PNGs':12,'training_RGB96_PNGs':360,'contacts':15,'groups':45,'roles':dict(roles),
        'original_unique_rows':96,'original_inputU':11,'repeated_original_U_provenance_rows':51,
        'generated_query_available':summary['generated_query_available'],
        'nominal_target_in_frame':summary['nominal_target_in_frame'],
        'all_strict_crop_pixels_valid_rows':summary['all_strict_crop_pixels_valid_rows'],
        'independent_RGB96_byte_parity_rows':360,'independent_strict_and_support_RLE_rows':360,
        'max_mass_mean_delta':max_mass_delta,'gallery_links':len(targets),'native_labels_created':0,
        'training_admission':False,'actual_tool_image_views':0},'rows360':pixel_rows,
        'source_and_owned_final_SHA_unchanged':True,'producer_helpers_imported':False,
        'provenance':'The generated q8 itself derives from saved nominal SOURCE GT. No additional GT pose/ref/ROI/mask channel is provided by the deployment condition API. This family does not measure or correct original deployed query performance.',
        'limits':'Exact sampling/metadata only. All instance/part/native semantic visibility remains UNKNOWN here; no physical gold, natural occlusion, model performance/adoption claim.'}
    put(out/'REVIEW.json',result)
    print(json.dumps({'REVIEW':bind(out/'REVIEW.json'),'summary':result['summary']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True)
    audit(p.parse_args().entry)
