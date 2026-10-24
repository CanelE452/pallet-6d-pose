"""Independent source-array/ref/geometry audit. No producer/model imports."""
from pathlib import Path
import collections
import hashlib
import json
import math
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
PACK=V/'actual_source_pair_input_pack';BASE=V/'source_pair_intake';V2=BASE/'image_only_v2_exclusive_alpha'
FIELDS=('p3_roi','rgb96','crop_valid96')


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def canonical_crop(rgb,q,L):
    # Independent vectorized derivation; no frozen crop function is imported.
    t=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L
    xs=q[0]+t;ys=q[1]+t;x=np.broadcast_to(xs,(96,96));y=np.broadcast_to(ys[:,None],(96,96))
    x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64);fx=x-x0;fy=y-y0
    h,w=rgb.shape[:2];acc=np.zeros((96,96,3),np.float64)
    for ox,oy,wt in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        xx=x0+ox;yy=y0+oy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        acc[ok]+=rgb[yy[ok],xx[ok]].astype(np.float64)*wt[ok,None]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    out=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out.transpose(2,0,1).copy(),valid[None].copy()


def native_valid(q,L,hw):
    if q is None or L is None:return np.zeros((9,9),bool)
    t=(np.arange(9,dtype=np.float64)+.5)/9-.5
    gx=q[0]+t[None,:]*L;gy=q[1]+t[:,None]*L
    return np.broadcast_to((gx>=0)&(gx<hw[1]),(9,9))&np.broadcast_to((gy>=0)&(gy<hw[0]),(9,9))


def expected_affine(hw):
    ch,cw=[x+200 for x in hw];g=min(640/ch,640/cw);rw,rh=round(cw*g),round(ch*g)
    ih,iw=rh+(640-rh)%32,rw+(640-rw)%32
    return dict(added_border=100,canvas_shape=[ch,cw],input_shape=[ih,iw],actual_P3_hw=[ih//8,iw//8],gain=g,
                offset=[float(round((iw-rw)/2-.1)),float(round((ih-rh)/2-.1))],fullP3_stored=False)


def main():
    checked={}
    def check(d):
        key=(d['path'],d['sha256'])
        if key not in checked:assert sha(d['path'])==d['sha256'],d['path'];checked[key]=True
    seal=read(PACK/'INPUT_PACK_SEAL.json');assert seal['completed']and seal['PASS']
    for d in [seal['manifest'],seal['QA'],seal['lineage'],seal['original_natural_GT_only']]+seal['changed_chunks']:check(d)
    manifest=read(seal['manifest']['path']);entry=read(manifest['entry']['path']);protocol=read(manifest['protocol']['path'])
    check(manifest['entry']);check(manifest['protocol'])
    for d in protocol['code_bindings'].values():check(d)
    for d in protocol['assets'].values():check(d) # File hashes only, no weight deserialization.
    for k in ['input_rows','source_lineage','occurrences','construction_ledger','intake_seal']:check(entry[k])
    rows=lines(entry['input_rows']['path']);source=lines(entry['source_lineage']['path']);occs=lines(entry['occurrences']['path']);gt=lines(entry['construction_ledger']['path'])
    actual=lines(seal['lineage']['path']);natural=lines(seal['original_natural_GT_only']['path'])
    assert len(rows)==len(source)==len(gt)==len(natural)==960 and len(occs)==len(actual)==2880
    assert [r['slot_index']for r in rows]==list(range(960)) and [r['occurrence_index']for r in actual]==list(range(2880))
    cache={}
    def ref(d):
        check(d['file']);p=d['file']['path']
        if p not in cache:
            with np.load(p,allow_pickle=False)as z:cache[p]={k:z[k]for k in FIELDS}
        return np.array(cache[p][d['key']][d['index']],copy=True)
    changed=0;unchanged=0;valid9checks=0;rgbchecks=0;finite_invalid=0;split_counts=collections.Counter();natural_counts=collections.Counter()
    before_pixels={}
    # Candidate selection is compared to original source metadata, not re-run.
    v94=P/'query_visibility_matched_query_scalar_20261023_v94';v92=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92'
    loco_frames=lines(v92/'rgb96_preparation/FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl')
    pallox_frames=lines(v94/'pallox_actual_initializer/FIXED100/FROZEN_PROPOSALS_ALL100.jsonl')
    for i,(r,s,g,n)in enumerate(zip(rows,source,gt,natural)):
        assert r['slot_index']==s['slot_index']==g['base_row_index']==n['slot_index']==i
        assert r['query_xy']==s['query_xy']==g['original_q'] and r['L']==s['L']==g['original_L']
        f=(loco_frames if s['source']=='LOCO'else pallox_frames)[r['frame_index']]
        si=f['selected_index'];assert s['selected_candidate_index']==si
        box=f['selected_predictor_box_xyxy']if s['source']=='LOCO'else(None if si is None else f['candidates'][si]['box_xyxy'])
        assert r['selected_box_xyxy']==box
        assert n['original_natural_label_GT_only']==g['original_natural_label_GT_only']==n['before_final_label_must_keep_natural_E']
        natural_counts[g['original_natural_label_GT_only']]+=1
        base={k:ref(s['before_input_refs'][k])for k in FIELDS}
        for vi,variant in enumerate(['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']):
            j=3*i+vi;a=actual[j];o=occs[j]
            assert a['occurrence_index']==o['occurrence_index']==j and a['slot_index']==o['slot_index']==i
            for k in ['input_id','native_slot_logical_only','query_xy','L','split','original_input_valid']:assert a[k]==r[k],(i,k)
            assert a['original_split']==r['split'] and a['source']==s['source'] and a['variant']==o['variant']==variant
            assert a['occurrence_id']==o['occurrence_id'] and a['original_image_sha256']==r['image_sha256']
            assert a['full_rgb_changed']==o['full_rgb_changed'] and a['full_rgb_decoded_sha256']==o['full_rgb_decoded_sha256']
            split_counts[(s['source'],r['split'],variant)]+=1
            if a['input_ref']['kind']=='BASE_ROW':
                assert not o['full_rgb_changed'] and a['input_ref']['slot_index']==i
                assert a['input_ref']['base_lineage']==manifest['base_input_lineage']
                x=base;unchanged+=1
                assert a['image_sha256']==r['image_sha256'] and not a['input_facts']['new_feature_forward']
            else:
                assert a['input_ref']['kind']=='CHANGED_CHUNK' and o['full_rgb_changed']
                rr=a['input_ref'];x={k:ref(dict(file=rr['file'],key=k,index=rr['index']))for k in FIELDS};changed+=1
                check(o['edited_fullRGB_file']);rgb=np.asarray(Image.open(o['edited_fullRGB_file']['path']).convert('RGB')).copy()
                assert ash(rgb)==o['full_rgb_decoded_sha256'] and a['image_sha256']==o['edited_fullRGB_file']['sha256']
                expected,valid=canonical_crop(rgb,np.asarray(r['query_xy'],np.float64),r['L'])
                assert np.array_equal(expected,x['rgb96'])and np.array_equal(valid,x['crop_valid96']),(i,'RGB96')
                assert a['input_facts']['feature_provenance']==expected_affine(r['raw_hw'])
                assert a['input_facts']['new_feature_forward']==1 and not a['input_facts']['reused_before_tensors'];rgbchecks+=1
            assert set(x)==set(FIELDS) and x['p3_roi'].dtype==np.float16 and x['p3_roi'].shape==(65,9,9)
            assert x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(3,96,96)and x['crop_valid96'].dtype==bool and x['crop_valid96'].shape==(1,96,96)
            for k in FIELDS:assert ash(x[k])==a['tensor_value_sha256'][k]
            assert np.isfinite(x['p3_roi']).all()
            v=native_valid(r['query_xy'],r['L'],r['raw_hw']);assert np.array_equal(v,x['p3_roi'][64].astype(bool))
            assert np.isin(x['p3_roi'][64],[0,1]).all() and not x['p3_roi'][:64][:,~v].any();valid9checks+=1
            facts=a['input_facts'];assert facts['ROI_anyvalid']==bool(v.any())and facts['ROI_center_valid']==bool(v[4,4])
            assert facts['RGB_central4_valid']==bool(x['crop_valid96'][0,47:49,47:49].all())
            assert facts['operator_query_available']==bool(v[4,4]and x['crop_valid96'][0,47:49,47:49].all())
            if r['query_xy']is None:assert not any(z.any()for z in x.values())
            elif not r['original_input_valid']:finite_invalid+=1
    assert (unchanged,changed,rgbchecks,valid9checks)==(2850,30,30,2880)
    assert natural_counts=={-1:797,0:143,1:20}
    assert manifest['model_state_before']==manifest['model_state_after']and manifest['model_unchanged']
    assert manifest['source_predict_calls']==30 and manifest['first_repeat_calls']==manifest['blank_warmup_calls']==1
    assert manifest['current_allowed']is False and manifest['head_forward_training_PnP']==0
    result=dict(completed=True,PASS=True,bindings=dict(pack_seal=bind(PACK/'INPUT_PACK_SEAL.json'),manifest=seal['manifest'],
           original_rows=entry['input_rows'],source_lineage=entry['source_lineage'],occurrences=entry['occurrences'],construction=entry['construction_ledger']),
           rows=960,occurrences=2880,all_identity_q_L_original_split_candidate_exact=True,unchanged_base_triplet_refs_and_value_hashes=unchanged,
           changed_edited_fullRGB_SHA_RGB96_valid_exact=rgbchecks,independent_native_valid9_checks=valid9checks,
           finite_invalid_occurrences_preserved=finite_invalid,natural_before_counts={str(k):v for k,v in natural_counts.items()},
           source_split_variant_counts={ '|'.join(k):v for k,v in split_counts.items()},unique_file_SHA_bindings_checked=len(checked),
           producer_frozen_state_certificate=manifest['model_state_before'],producer_changed_source_forward_calls=30,
           producer_first_repeat_plus_blank_warmup=[1,1],fresh_P3_numeric_replay='separate fixedfirstchanged replay pending',
           audit_new_model_or_backbone_forward=0,checkpoint_parameter_loads=0,current_private_reads=0,fit_PnP=0)
    write(HERE/'SAVED_ARRAYS_REVIEW.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
