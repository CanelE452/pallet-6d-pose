"""Independent full saved-unit/occurrence audit; no producer/model imports."""
from pathlib import Path
from collections import Counter,OrderedDict
import hashlib,json,time
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
PACK=V/'actual_source_pair_input_pack';IMAGE=V/'actual_query_anchored_image_intake'
FIELDS=('p3_roi','rgb96','crop_valid96')
EXPECTED_SEAL='eb1607140fb1466a5aaad4ce839e4b9471c5514abd49288c7572a916123773f6'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def crop(rgb,q,L):
    t=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L
    x,y=np.meshgrid(q[0]+t,q[1]+t);ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);dx=x-ix;dy=y-iy
    h,w=rgb.shape[:2];out=np.zeros((96,96,3),np.float64)
    for ox,oy,wt in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        xx=ix+ox;yy=iy+oy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        out[ok]+=rgb[yy[ok],xx[ok]].astype(np.float64)*wt[ok,None]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    out=np.floor(np.clip(out,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out.transpose(2,0,1).copy(),valid[None].copy()
def valid9(q,L,hw):
    if q is None:return np.zeros((9,9),bool)
    t=(np.arange(9,dtype=np.float64)+.5)/9-.5;x,y=np.meshgrid(q[0]+t*L,q[1]+t*L)
    return (x>=0)&(x<hw[1])&(y>=0)&(y<hw[0])&not_missing(q)
def not_missing(q):return not np.all(np.asarray(q)==-1)
def affine(hw):
    ch,cw=[x+200 for x in hw];gain=min(640/ch,640/cw);rw,rh=round(cw*gain),round(ch*gain)
    ih,iw=rh+(640-rh)%32,rw+(640-rw)%32
    return dict(added_border=100,canvas_shape=[ch,cw],input_shape=[ih,iw],actual_P3_hw=[ih//8,iw//8],gain=gain,offset=[float(round((iw-rw)/2-.1)),float(round((ih-rh)/2-.1))],fullP3_stored=False)
def main():
    start=time.monotonic();checked=set()
    def check(b):
        key=(b['path'],b['sha256'])
        if key not in checked:assert sha(b['path'])==b['sha256'],b['path'];checked.add(key)
    assert sha(PACK/'INPUT_PACK_SEAL.json')==EXPECTED_SEAL
    seal=read(PACK/'INPUT_PACK_SEAL.json');assert seal['completed']and seal['PASS']
    for key in ['manifest','QA','lineage','input_units','original_before_GT_only']:check(seal[key])
    for b in seal['changed_chunks']:check(b)
    m=read(seal['manifest']['path']);check(m['entry']);check(m['protocol'])
    e=read(m['entry']['path']);p=read(m['protocol']['path']);assert e['approved_by_root']and e['source_only']
    for key in ['current_allowed','training_allowed','head_allowed','PnP_allowed']:assert e[key]is False
    for b in p['code_bindings'].values():check(b)
    for b in p['assets'].values():check(b) # SHA bytes only; no parameter load here.
    for key in ['input_rows','base_source_lineage','input_plan','source_lineage','occurrences','gt_only_occurrences','intake_seal','metadata_lock']:check(e[key])
    rows=lines(e['input_rows']['path']);baseoriginal=lines(e['base_source_lineage']['path']);source=lines(e['source_lineage']['path'])
    plan=lines(e['input_plan']['path']);jobs=lines(e['occurrences']['path']);gt=lines(e['gt_only_occurrences']['path'])
    actual=lines(seal['lineage']['path']);units=lines(seal['input_units']['path']);quarantine=read(seal['original_before_GT_only']['path'])
    assert quarantine['GT_only_occurrences']==e['gt_only_occurrences']and quarantine['body_read_by_feature_driver']is False and quarantine['GT_fields_may_forward']is False
    assert len(rows)==len(source)==len(baseoriginal)==960 and len(actual)==len(plan)==len(jobs)==len(gt)==31680 and len(units)==1348
    assert [u['input_unit_index']for u in units]==list(range(1348))
    assert source==baseoriginal
    assert sha(e['source_lineage']['path'])==e['base_source_lineage']['sha256']
    chunk_counts=[];all_chunk_indices=set()
    for c in m['chunks']:
        check(c['file'])
        with np.load(c['file']['path'],allow_pickle=False)as z:
            assert set(z.files)==set(FIELDS)
            for k,tail,dt in [('p3_roi',(65,9,9),np.float16),('rgb96',(3,96,96),np.uint8),('crop_valid96',(1,96,96),np.bool_)]:
                assert z[k].shape==(c['count'],)+tail and z[k].dtype==dt
        all_chunk_indices.update((c['file']['path'],j)for j in range(c['count']));chunk_counts.append(c['count'])
    assert chunk_counts==[128,128,128,4] and sum(chunk_counts)==388
    cache=OrderedDict()
    def get_npz(b):
        check(b);path=b['path']
        if path not in cache:
            with np.load(path,allow_pickle=False)as z:cache[path]={k:z[k]for k in FIELDS}
            if len(cache)>6:cache.popitem(last=False)
        else:cache.move_to_end(path)
        return cache[path]
    def ref(b):return get_npz(b['file'])[b['key']][b['index']]
    hashes=[];facts=[];changed_unit_refs=set();unit_geometry_count=0
    for ui,u in enumerate(units):
        i=u['slot_index'];r=rows[i];s=source[i];rr=u['input_ref']
        assert 0<=i<960
        if ui<960:
            assert i==ui and rr==dict(kind='BASE_ROW',slot_index=i,base_lineage=e['source_lineage'])
            x={k:ref(s['before_input_refs'][k])for k in FIELDS}
        else:
            assert rr['kind']=='CHANGED_CHUNK';key=(rr['file']['path'],rr['index']);assert key not in changed_unit_refs;changed_unit_refs.add(key)
            x={k:get_npz(rr['file'])[k][rr['index']]for k in FIELDS}
        assert x['p3_roi'].shape==(65,9,9)and x['p3_roi'].dtype==np.float16 and np.isfinite(x['p3_roi']).all()
        assert x['rgb96'].shape==(3,96,96)and x['rgb96'].dtype==np.uint8
        assert x['crop_valid96'].shape==(1,96,96)and x['crop_valid96'].dtype==np.bool_
        h={k:ash(x[k])for k in FIELDS};assert h==u['tensor_value_sha256'];hashes.append(h)
        v=valid9(r['query_xy'],r['L'],r['raw_hw']);assert np.array_equal(v,x['p3_roi'][64].astype(bool))
        assert np.isin(x['p3_roi'][64],[0,1]).all()and not x['p3_roi'][:64][:,~v].any();unit_geometry_count+=1
        ff=u['input_facts'];assert ff['ROI_anyvalid']==bool(v.any())and ff['ROI_center_valid']==bool(v[4,4])
        assert ff['RGB_central4_valid']==bool(x['crop_valid96'][0,47:49,47:49].all())
        assert ff['operator_query_available']==bool(v[4,4]and x['crop_valid96'][0,47:49,47:49].all())
        assert ff['reused_before_tensors']==(ui<960)and ff['new_feature_forward']==int(ui>=960)
        if r['query_xy']is None:assert not any(a.any()for a in x.values())
        facts.append(ff)
    assert changed_unit_refs==all_chunk_indices
    v92=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92';v94=P/'query_visibility_matched_query_scalar_20261023_v94'
    loco=lines(v92/'rgb96_preparation/FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl');pallox=lines(v94/'pallox_actual_initializer/FIXED100/FROZEN_PROPOSALS_ALL100.jsonl')
    for i,(r,s)in enumerate(zip(rows,source)):
        assert r['slot_index']==s['slot_index']==i and r['input_id']==s['input_id']and r['query_xy']==s['query_xy']and r['L']==s['L']
        frame=(loco if s['source']=='LOCO'else pallox)[r['frame_index']];si=frame['selected_index'];assert si==s['selected_candidate_index']
        box=frame['selected_predictor_box_xyxy']if s['source']=='LOCO'else(None if si is None else frame['candidates'][si]['box_xyxy'])
        assert r['selected_box_xyxy']==box
    forbidden={'candidate_aux_hidden_label_GT_only','query_anchored_before_label','original_C1_V95_label','pair_ready','teacher_label','target_ref','protected_regions_GT_only','mask','mask_vis','mask_full','pose','depth','donor'}
    changed=[];reused=0;counts=Counter();GTcounts=Counter();availability=Counter()
    for oi,(a,o,z,g)in enumerate(zip(actual,jobs,plan,gt)):
        i=oi//33;r=rows[i];s=source[i];off=oi%33;rank=None if off==0 else(off-1)//2;variant='BEFORE'if off==0 else['OPAQUE_COVER','CUE_PRESERVING_SHAM'][(off-1)%2]
        assert a['occurrence_index']==o['occurrence_index']==z['occurrence_index']==g['occurrence_index']==oi
        assert a['slot_index']==a['base_row_index']==o['base_row_index']==g['base_row_index']==i
        assert a['donor_rank']==o['donor_rank']==g['donor_rank']==rank and a['variant']==o['variant']==g['variant']==variant
        assert a['occurrence_id']==o['occurrence_id']==z['occurrence_id']==g['occurrence_id']
        for key in z:
            if key!='image_status':assert o[key]==z[key],(oi,key)
        assert not(set(a)&forbidden)and not(set(o)&forbidden)
        for key in ['input_id','query_xy','L','native_slot_logical_only','original_input_valid','selected_box_xyxy']:assert a[key]==r[key],(oi,key)
        assert a['split']==a['raw_split']==a['original_split']==r['split']and a['source']==s['source']
        assert a['selected_candidate_index']==s['selected_candidate_index']and a['selected_candidate_score']==s['selected_candidate_score']
        assert a['original_image_sha256']==r['image_sha256']and a['full_rgb_decoded_sha256']==o['full_rgb_decoded_sha256']
        assert a['full_rgb_changed']==o['full_rgb_changed']
        ui=a['input_unit_index'];assert a['input_ref']==units[ui]['input_ref']and units[ui]['slot_index']==i
        assert a['tensor_value_sha256']==hashes[ui]and a['input_facts']==facts[ui]
        if o['full_rgb_changed']:
            assert ui>=960 and a['image_sha256']==o['edited_fullRGB_file']['sha256'];changed.append(oi)
            check(o['edited_fullRGB_file']);rgb=np.array(Image.open(o['edited_fullRGB_file']['path']).convert('RGB'),copy=True)
            assert rgb.shape==tuple(r['raw_hw'])+(3,)and ash(rgb)==o['full_rgb_decoded_sha256']
            rr=a['input_ref'];x=get_npz(rr['file']);expected,valid=crop(rgb,np.asarray(r['query_xy'],np.float64),r['L'])
            assert np.array_equal(expected,x['rgb96'][rr['index']])and np.array_equal(valid,x['crop_valid96'][rr['index']]),(oi,'RGB96')
            assert a['input_facts']['feature_provenance']==affine(r['raw_hw'])
        else:
            assert ui==i and a['image_sha256']==r['image_sha256']and a['full_rgb_decoded_sha256']==s['decoded_RGB_sha256'];reused+=1
        if variant!='BEFORE'and g['query_anchored_before_label']==1:assert g['candidate_aux_hidden_label_GT_only']==-1 and not g['pair_ready']
        GTcounts[g['candidate_aux_hidden_label_GT_only']]+=1
        counts[(s['source'],r['split'],variant)]+=1;availability[(s['source'],r['split'],bool(a['input_facts']['operator_query_available']))]+=1
    assert len(changed)==388 and len(set(actual[j]['input_unit_index']for j in changed))==388 and reused==31292
    assert GTcounts=={-1:31227,0:255,1:198}
    assert m['model_state_before']==m['model_state_after']and m['model_unchanged']
    assert (m['source_predict_calls'],m['first_repeat_calls'],m['blank_warmup_calls'])==(388,1,1)
    assert m['unique_input_units']==1348 and m['GT_fields_may_forward']is False and m['current_allowed']is False and m['head_forward_training_PnP']==0
    result=dict(completed=True,PASS=True,code=bind(__file__),scope=bind(HERE/'SCOPE_BEFORE_PACK_AND_SOURCE_ARRAYS.json'),pack_seal=bind(PACK/'INPUT_PACK_SEAL.json'),manifest=seal['manifest'],
        bindings=dict(entry=m['entry'],protocol=m['protocol'],original_rows=e['input_rows'],source_lineage=e['source_lineage'],input_plan=e['input_plan'],image_jobs=e['occurrences'],GT_quarantine=e['gt_only_occurrences']),
        occurrences=31680,unique_input_units=1348,base_units=960,changed_units=388,base_reused_occurrences=31292,all_occurrence_ref_index_rank_rawsplit_qL_candidate_exact=True,
        base_triplet_exact_refs_value_SHA=960,all_unique_unit_shape_dtype_native65_and_factual_flags_exact=unit_geometry_count,all_changed_RGB96_and_valid_exact=388,
        storage_only_dedup=True,changed_units_each_fresh_and_distinct=True,GT_quarantine_body_not_forwarded_certificate=True,GT_counts={str(k):v for k,v in GTcounts.items()},natural_E4_after_U_verified=True,
        chunk_counts=chunk_counts,source_split_variant_counts={'|'.join(k):v for k,v in counts.items()},source_split_operator_availability={'|'.join(map(str,k)):v for k,v in availability.items()},
        file_SHA_bindings_checked=len(checked),producer_state_before_after_same=m['model_state_before'],producer_predict_repeat_warmup=[388,1,1],
        fixed_first_changed=dict(occurrence_index=changed[0],base_row_index=actual[changed[0]]['base_row_index'],donor_rank=actual[changed[0]]['donor_rank'],variant=actual[changed[0]]['variant']),
        audit_parameter_deserialization=0,audit_model_or_backbone_forward=0,head_optimizer_fit_private_current_PnP=0,seconds=time.monotonic()-start,
        limits=['388 P3 provenance/hash/geometry checks are not388 independent model reruns; one fixed first-changed numerical replay is separate','GT is conditional source auxiliary cue state, native expected-part UNKNOWN and human_gold false'])
    write(HERE/'SAVED_ARRAYS_REVIEW.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
