"""Root-invoked saved-only V82 SAM audit; scalar sample/formula replay.

No actual I/O at import. Root supplies the final completed SAM seal SHA.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import numpy as np

R=Path(__file__).resolve().parent;V=R.parent;OUT=V/'segmentation/outputs';SEAL=OUT/'SAM_OUTPUT_SEAL.json'
POLICY=R/'INDEPENDENT_SAM_AUDIT_POLICY_BEFORE_SEAL.json'
POLICY_SHA=None  # Policy itself is immutably bound by the pre-seal reader lock.
TARGET8={'input_id','image','image_sha256','raw_hw','initial_points','roi_side','query_box_xyxy','requested_category'}
SAM_EXTRA={'mask_status','mask_reason','mask','predicted_quality','selected_mask_index'}
READS=[]

def safe(path):
    p=Path(path)
    assert p.suffix.lower() in {'.json','.jsonl','.csv','.py','.npz'},'Metadata/prediction mask only, no RGB/weights'
    assert 'private_evaluation' not in p.parts and 'current223' not in str(p).lower()
    if p.suffix=='.npz':assert str(p).startswith(str(OUT/'masks')),'No source GT/cache NPZ reads'
    return p
def sha(path):p=safe(path);READS.append(str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def read(path):p=safe(path);READS.append(str(p));return json.loads(p.read_text())
def lines(path):p=safe(path);READS.append(str(p));return [json.loads(x) for x in p.read_text().splitlines() if x]
def value(s):return None if s=='' else float(s)

def main(seal_sha):
    assert len(seal_sha)==64 and all(c in '0123456789abcdef' for c in seal_sha)
    lock=read(R/'INDEPENDENT_SAM_READER_LOCK_BEFORE_SEAL.json')
    assert lock['reader']==binding(__file__) and lock['policy']==binding(POLICY)
    assert sha(SEAL)==seal_sha
    s=read(SEAL);assert s['completed'] is True and s['PASS'] is True and s['source_only'] is True
    assert s['current223'] is False and s['fit'] is False and s['PnP']==0
    for k in ['entrylock','decisions','detector_output_seal','associations','per_request','runtime','code','helper','legacy_helper','generated','detector_independent_audit','official_output_contract','output_masks','summary']:check(s[k])
    # Checkpoint identity is metadata only; never open/hash weight body here.
    assert s['checkpoint']['sha256']=='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
    entry=read(s['entrylock']['path']);assert entry['approved'] is True and entry['allow_source_SAM_forward'] is True
    assert entry['bindings']['detector_output_seal']==s['detector_output_seal'] and entry['bindings']['detector_independent_audit']==s['detector_independent_audit']
    da=read(s['detector_independent_audit']['path']);assert da['completed'] is True and da['PASS'] is True and da['detector_output_seal']==s['detector_output_seal']
    ds=read(s['detector_output_seal']['path']);assert ds['completed'] is True and ds['PASS'] is True
    assert ds['source_inputs']==s['source_inputs'] and ds['associations']==s['associations'] and ds['per_request']==s['per_request']
    for b in s['source_inputs'].values():check(b)
    inputs=lines(s['source_inputs']['manifest']['path']);requests=lines(s['source_inputs']['unique_manifest']['path']);mapping=lines(s['source_inputs']['target_to_request']['path'])
    assocs=lines(s['associations']['path']);reqrows=lines(s['per_request']['path']);rows=lines(s['output_masks']['path']);summary=read(s['summary']['path'])
    assert len(inputs)==len(requests)==len(mapping)==len(assocs)==len(reqrows)==len(rows)==1035
    assert len({r['input_id'] for r in rows})==1035
    reqby={r['request_id']:r for r in reqrows};originalreq={r['request_id']:r for r in requests};assert set(reqby)==set(originalreq) and len(reqby)==1035
    grouped={}
    for req in reqrows:
        assert all(req[k]==originalreq[req['request_id']][k] for k in originalreq[req['request_id']])
        assert req['caption']==req['requested_category']+'.'
        assert req['request_key_sha256']==hashlib.sha256((req['image_sha256']+'\n'+req['caption']).encode()).hexdigest()
        rgb=tuple(req[k] if k!='raw_hw' else tuple(req[k]) for k in ['image','image_sha256','raw_hw','decoded_RGB_SHA256'])
        if req['image_sha256'] in grouped:assert grouped[req['image_sha256']]==rgb
        grouped[req['image_sha256']]=rgb
    assert len(grouped)==435
    bindings={b['path']:b for b in s['outputs']};assert len(bindings)==len(s['outputs'])
    for b in s['outputs']:check(b)
    querypath=OUT/'QUERY_MASK_SAMPLES.csv';assert str(querypath) in bindings
    READS.append(str(querypath))
    with querypath.open() as f:qrows=list(csv.DictReader(f))
    assert len(qrows)==8280
    predictions=Counter();validity=Counter();holds=Counter();maskbindings=[];associatedRGB=set();qualities=[];zeros=0
    for i,(original,a,row,link) in enumerate(zip(inputs,assocs,rows,mapping)):
        assert set(original)==TARGET8 and set(row)==set(a)|SAM_EXTRA and {k:row[k] for k in a}==a
        assert {k:row[k] for k in TARGET8}==original
        assert link=={k:a[k] for k in ['input_id','request_id','image_sha256','requested_category','caption','request_key_sha256']}
        r=reqby[a['request_id']]
        assert all(a[k]==r[k] for k in ['image','image_sha256','raw_hw','requested_category','caption','request_key_sha256'])
        h,w=original['raw_hw'];known=a['status']=='ASSOCIATED';mask=None
        if known:
            assert row['mask_status']=='OK' and row['mask_reason']=='SINGLE_BOX_NATIVE_VISIBLE_MASK' and row['selected_mask_index']==0
            assert math.isfinite(row['predicted_quality']) and row['mask']==bindings[row['mask']['path']]
            READS.append(row['mask']['path'])
            with np.load(safe(row['mask']['path']),allow_pickle=False) as z:
                assert z.files==['visible_mask'];mask=z['visible_mask']
            assert mask.dtype==np.bool_ and mask.shape==(h,w)
            maskbindings.append(row['mask']);associatedRGB.add(original['image_sha256']);qualities.append(row['predicted_quality']);zeros+=int(not mask.any())
        else:
            assert a['status']=='U' and row['mask_status']=='U' and row['mask_reason']=='NO_ASSOCIATED_BOX:'+a['reason']
            assert row['mask'] is None and row['predicted_quality'] is None and row['selected_mask_index'] is None
            assert all(a[k] is None for k in ['selected_detection_index','selected_detector_query_index','selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt'])
            holds[a['reason']]+=1
        for k,q in enumerate(original['initial_points']):
            qr=qrows[8*i+k]
            assert int(qr['input_index'])==i and int(qr['native_corner'])==k and qr['input_id']==original['input_id'] and qr['mask_status']==row['mask_status']
            assert value(qr['query_x'])==q[0] and value(qr['query_y'])==q[1]
            finite=all(v is not None and math.isfinite(v) for v in q)
            px,py=(math.floor(q[0]+.5),math.floor(q[1]+.5)) if finite else (None,None)
            good=bool(finite and 0<=q[0]<w and 0<=q[1]<h and 0<=px<w and 0<=py<h)
            assert value(qr['pixel_x'])==px and value(qr['pixel_y'])==py and qr['native_query_valid']==str(good)
            expected='U' if not known or not good else ('VISIBLE_MASK' if mask[py,px] else 'NOT_VISIBLE_MASK')
            assert qr['mask_prediction']==expected and qr['visible_mask_value']==('' if expected=='U' else str(expected=='VISIBLE_MASK'))
            predictions[expected]+=1;validity[(a['status'],good)]+=1
    actualmask=len(maskbindings);actualU=1035-actualmask
    assert actualmask==da['counts']['associated']==790 and actualU==da['counts']['U']==245
    assert summary['associated']==actualmask and summary['unassociated']==actualU and summary['embedding_calls']==len(associatedRGB)
    assert summary['targets']==1035 and summary['unique_RGB']==435 and summary['query_slots']==8280
    assert summary['zero_visible_masks']==zeros and summary['query_predictions']==dict(predictions) and summary['state_before']==summary['state_after']
    assert summary['weights']=='FP32' and summary['autocast']=='CUDA BF16' and summary['mask_threshold']==0. and summary['selected_mask_index']==0
    assert all(summary[k] is False for k in ['multimask_output','return_logits','quality_selection','points','postprocessing','compile','current223','fit']) and summary['PnP']==0
    assert set(bindings)=={s['output_masks']['path'],s['summary']['path'],str(querypath)}|{b['path'] for b in maskbindings}
    # Successful code execution requires this exact assertion before set_image.
    code=Path(s['code']['path']).read_text()
    checktext="require(hashlib.sha256(image.tobytes()).hexdigest()==rgb['decoded_RGB_SHA256']"
    assert checktext in code and code.index(checktext)<code.index('predictor.set_image(image)')
    cert=R/'V82_SAM_DECODE_GUARD_CERTIFICATES.csv'
    with cert.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['image_sha256','raw_H','raw_W','detector_decoded_RGB_SHA256','successful_code_requires_same_SHA_before_set_image','independent_raw_RGB_recheck']);writer.writeheader()
        for sha_,rgb in grouped.items():
            if sha_ in associatedRGB:writer.writerow(dict(image_sha256=sha_,raw_H=rgb[2][0],raw_W=rgb[2][1],detector_decoded_RGB_SHA256=rgb[3],successful_code_requires_same_SHA_before_set_image=True,independent_raw_RGB_recheck=False))
    result=dict(completed=True,PASS=True,blockers=[],sam_output_seal=binding(SEAL),detector_output_seal=s['detector_output_seal'],audit_code=binding(__file__),policy=binding(POLICY),reader_lock=binding(R/'INDEPENDENT_SAM_READER_LOCK_BEFORE_SEAL.json'),detector_independent_audit=s['detector_independent_audit'],counts=dict(targets=1035,query_slots=8280,all_RGB_groups=435,actual_masks=actualmask,actual_U=actualU,actual_predict_calls_completed_run=actualmask,actual_associated_RGB_embedding_calls=len(associatedRGB),zero_masks=zeros),query_predictions=dict(predictions),query_validity_by_association={str(k):v for k,v in validity.items()},U_reasons=dict(holds),quality=dict(N=len(qualities),minimum=min(qualities),maximum=max(qualities),used_to_select=False),invariants=dict(all_original8_request_caption_key_order_exact=True,all_native_bool_mask_SHA_shape_bindings_exact=True,all8280_scalar_native_floor_plus_half_samples_exact=True,no_U_fake_zero_mask=True,all_associated_rows_retained_single_mask_index0=True,model_state_before_after_saved_identical=True,all_output_bindings_complete_unique=True,RGB_group_decode_metadata_consistent=True),decode_certificates=dict(csv=binding(cert),evidence='Successful bound SAM code asserts raw SHA/HW and detector decoded byte SHA before each set_image. Metadata/groups/counts and code order verified; raw RGB not independently reopened, and a separate postdecode SAM digest list was not persisted.'),phase_counts='Completed V82 SAM predict/embedding counts are real nonzero values derived from saved masks and associated RGB groups. No claim of whole-phase forward0; this audit performs no new forward.',semantics='Visible segmentation proxy and U only. NOT_VISIBLE_MASK is not semantic E/S; no GT performance metrics computed here.',scope=dict(new_forward=0,RGB_reads=0,GTmask_reads=0,weights_reads=0,model_constructors=0,current223_reads=0,fit=0,PnP=0,existing_artifacts_modified=0),saved_unique_files_read=len(set(READS)))
    path=R/'INDEPENDENT_SAM_OUTPUT_AUDIT.json'
    with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,path=str(path),sha256=sha(path),counts=result['counts'],query_predictions=dict(predictions))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal-sha',required=True);a=p.parse_args();main(a.seal_sha)
