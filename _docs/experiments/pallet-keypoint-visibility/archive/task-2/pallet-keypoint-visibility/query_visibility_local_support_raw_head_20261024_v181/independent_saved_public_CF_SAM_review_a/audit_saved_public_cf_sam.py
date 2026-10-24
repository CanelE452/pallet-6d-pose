"""Independent saved public TRAIN/CF SAM integrity audit; no model/helper imports."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;OUT=V/'frozen_sam_public_cf_inputs_v1';C=OUT/'COMPLETE.json'
def require(x,msg):
    if not bool(x):raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):require(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def numeric(rr):
    return dict(original_query_xy=np.array([r['original_query_xy'] for r in rr],np.float64),original_L=np.array([r['original_L'] for r in rr],np.float64),
        native_slot=np.array([r['native_slot'] for r in rr],np.int8),original_global_index=np.array([r['original_global_index'] for r in rr],np.int64),
        original_occurrence_id=np.array([r['original_occurrence_id'] for r in rr]),original_query_available=np.array([r['original_query_available'] for r in rr],bool))

def run():
    start=time.monotonic();c=read(C);require(c['completed'] and c['PASS'] and c['fatal'] is None,'Actual complete barrier')
    require(sha(C)=='f04e23360677dc24625fdb0252390abfcd606991a80e100b24625dc3b5366bf3','Root completed binding')
    groups=lines(check(c['groups_manifest']));rows=lines(check(c['rows_manifest']));status=lines(check(c['all_rows']))
    require(len(groups)==len(c['groups'])==662 and len(rows)==len(status)==5296,'Full original corpus')
    require(len(set(g['group_id'] for g in groups))==662 and len(set(r['input_occurrence_id'] for r in rows))==5296,'Unique original occurrences')
    checked_outputs={};source_bindings={};canonical={};seenRGB={};proofs=[];missing=0;reuse=0
    expected_keys={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}
    for j,(original,s) in enumerate(zip(groups,c['groups'])):
        require(all(s[k]==val for k,val in original.items()),'Full original group metadata preserved')
        rr=s['original_input_metadata8'];require(len(rr)==8 and [r['native_slot'] for r in rr]==list(range(8)),'Every original native8')
        require(rr==rows[j*8:j*8+8] and all(r['split']=='SOURCE_DEVELOPMENT' for r in rr),'TRAIN-only rows/order')
        meta=numeric(rr);q8=np.asarray(s['original_initial_points8'],np.float64)
        require(q8.shape==(8,2) and np.isfinite(q8).all() and np.array_equal(meta['original_query_xy'],q8),'Original own q8')
        box=np.r_[q8.min(0),q8.max(0)];require(box.dtype==np.float64 and box.tobytes()==np.asarray(s['box_xyxy'],np.float64).tobytes(),'Exact original q8 minmax, no clip/expand')
        require(s['prompt_available'] and s['runtime_available'] and s['outputs'] is not None,'All662 actual available groups')
        image=s['image'];path=check(image['binding']);source_bindings[str(path)]=image['binding']
        if s['domain']=='PUBLIC_ORIGINAL':rgb=np.load(path,allow_pickle=False)
        else:
            with np.load(path,allow_pickle=False) as z:
                rgb=z['full_RGB'].copy()
                for src,dst in [('query_xy','original_query_xy'),('L','original_L'),('native_slot','native_slot'),('query_id','original_occurrence_id')]:require(exact(z[src],meta[dst]),'CF original payload q/L/identity bytes')
        require(rgb.dtype==np.uint8 and rgb.shape==tuple(s['raw_hw'])+(3,),'Original full RGB exact dtype/shape')
        rgbsha=ash(rgb);require(rgbsha==s['decoded_RGB_sha256_observed'],'Actual source RGB SHA')
        if image['decoded_RGB_sha256'] is not None:require(rgbsha==image['decoded_RGB_sha256'],'Original public decoded RGB fact')
        mdsha={k:ash(v) for k,v in meta.items()};mdsha['box_xyxy']=ash(box)
        if s['reused']:
            reuse+=1;old=canonical[s['reused_from_group_id']]
            require((s['original_frame_index'],s['source_id'],rgbsha,rgb.shape,mdsha)==old['identity'],'Actual reuse same frame/RGB/q8/box/L/numeric metadata')
            require(s['outputs']==old['outputs'] and s['array_sha256']==old['array_sha256'],'Reuse points exact canonical saved packet')
            require(set(s['reuse_proof'])=={'same_original_frame_and_source_id','same_decoded_RGB_SHA_shape','same_q8_box_L_and_numeric_metadata_bytes','canonical_group_id'} and all(v is True for k,v in s['reuse_proof'].items() if k!='canonical_group_id') and s['reuse_proof']['canonical_group_id']==old['group_id'],'Reuse proof truthful')
        else:
            op=check(s['outputs'])
            with np.load(op,allow_pickle=False) as z:
                require(set(z.files)==expected_keys,'Exact twelve output keys');a={k:z[k].copy() for k in z.files}
            for key,shape in [('image_logits',tuple(s['raw_hw'])),('lowres_logits',(256,256)),('predicted_quality',(1,))]:require(a[key].dtype==np.float32 and a[key].shape==shape and np.isfinite(a[key]).all(),'Saved FP32 finite '+key)
            require(a['mask_logits_gt0'].dtype==bool and exact(a['mask_logits_gt0'],a['image_logits']>0),'Strict logit0 saved binary parity')
            for key,val in meta.items():require(exact(a[key],val),'Every saved metadata array '+key)
            require(exact(a['box_xyxy'],box),'Exact saved box')
            require(exact(a['prediction_available'],meta['original_query_available']),'Original U first, no new availability gate')
            arrsha={k:ash(val) for k,val in a.items()};require(arrsha==s['array_sha256'],'Every saved array actual SHA')
            canonical[s['group_id']]=dict(group_id=s['group_id'],identity=(s['original_frame_index'],s['source_id'],rgbsha,rgb.shape,mdsha),outputs=s['outputs'],array_sha256=arrsha)
            checked_outputs[str(op)]=s['outputs']
        for r,t in zip(rr,status[j*8:j*8+8]):
            require(all(t[k]==val for k,val in r.items()),'All original slot metadata preserved')
            available=bool(r['original_query_available']);missing+=not available
            require(t['prediction_available']==available and t['prompt_available'] and t['runtime_available'] and t['group_id']==s['group_id'] and t['group_output']==s['outputs'],'Saved status U-priority/group binding')
            require(t['prediction_U_reason']==(None if available else 'ORIGINAL_INPUT_U'),'Explicit original inputU reason')
        proofs.append(dict(group_id=s['group_id'],reused=s['reused'],reused_from_group_id=s['reused_from_group_id'],source_binding=image['binding'],decoded_RGB_sha256_actual=rgbsha,metadata_array_SHA_actual=mdsha,output=s['outputs']))
        if (j+1)%100==0:print(json.dumps(dict(groups_checked=j+1,total662=662,seconds=time.monotonic()-start)),flush=True)
    require(len(canonical)==503 and reuse==159 and missing==554,'All unique/reuse/originalU counts')
    repeat=c['first_group_repeat'];require(repeat['attempted'] and repeat['PASS'] and repeat['group_id']==groups[0]['group_id'],'First repeat metadata')
    with np.load(check(repeat['outputs']),allow_pickle=False) as z:
        require(set(z.files)=={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0'},'Repeat exact visual outputs only')
        with np.load(check(c['groups'][0]['outputs']),allow_pickle=False) as original:
            for k in z.files:require(exact(z[k],original[k]) and repeat['array_byte_equal'][k] is True,'Actual repeat bytes '+k)
    counters=c['actual_forward_call_counters'];require(counters['set_image_calls']==counters['predict_calls']==504 and len(counters['predict_warn_only_scopes'])==504,'503unique plus one fixed repeat counters')
    require(all(x['strict_before'] and x['warn_only_during'] and x['strict_restored_after'] for x in counters['predict_warn_only_scopes']),'Strict scoped policy receipt')
    require(c['model_state_byte_unchanged'] and c['model_state_before_sha256']==c['model_state_after_sha256'] and c['input_checkpoint_and_source_hashes_unchanged'],'State/file unchanged receipt consistency')
    source=read(V.parent/'query_visibility_frozen_sam_source_baseline_20261024_v180/frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json')
    require(c['checkpoint']==source['checkpoint'] and c['source_commit']==source['source_commit'] and c['model_state_before_sha256']==source['model_state_before_sha256'],'Same original frozen SAM provenance/state SHA')
    for b in source_bindings.values():check(b)
    for b in checked_outputs.values():check(b)
    for key in ['groups_manifest','rows_manifest','all_rows','code','protocol','scope','source_pin','cuda_cumsum_guard']:check(c[key])
    pf=H/'ALL662_INDEPENDENT_BINDINGS_AND_REUSE_CHECKS.jsonl'
    with pf.open('x') as f:
        for row in proofs:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    r=dict(completed=True,PASS=True,review_code=bind(__file__),actual_complete=bind(C),proofs=bind(pf),
        coverage=dict(groups662=662,slots5296=5296,public185=185,CF477=477,unique_saved_packets503=503,reused159=159,original_inputU554=554,metadata_rows_exact5296=True,all_raw_RGB_groups_verified662=True,all12_array_schema_metadata_mask_SHA_checked503=True,first_repeat_actual4arrays_byteequal=True),
        independent_method='No producer/helper imports. Original662 fullRGB bytes/hash and CFqLmetadata checked before comparing saved group reuse proofs. Every503unique savedpacket has actual12key metadata/finiteness/mask>0/arraySHA audit; all662groups/5296statuses exactjoin; first repeat4arrays directly bytecompared.',
        state_scope='Statebefore/after/checkpoint/commit consistency checked against sealed V180 metadata. No weights loaded or model reconstructed; not independent actual model-state tensor rehash or forward replay.',
        model_state_metadata_unchanged=True,no_new_SAM_or_model_or_fit_or_threshold=True,GT_or_private_or_publicheld_V112_reads=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    f=H/'REVIEW.json';f.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(f))))
if __name__=='__main__':run()
