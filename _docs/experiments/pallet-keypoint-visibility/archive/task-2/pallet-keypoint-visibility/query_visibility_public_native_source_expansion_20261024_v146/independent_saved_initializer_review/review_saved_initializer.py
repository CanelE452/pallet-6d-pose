"""Read-only independent saved arithmetic review; no producer/model imports.
No RGB/NPY image/checkpoint/GT/pose/K/mask bodies are read or re-hashed.
"""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;D=V/'actual_public_initializer_v1'
SEAL=D/'PREDICTION_SEAL.json';EXPECTED='4f4b4611038b2abe2b4be174f37ca557e4fd164136ada4a7f10ef2c667b1e11f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path'];assert sha(b['path'])==b['sha256'];return Path(b['path'])
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
assert sha(SEAL)==EXPECTED
seal=read(SEAL);assert seal['completed']and seal['PASS']and seal['frames']==64 and seal['rows']==512 and seal['GT_read']is False and seal['historical_predictions_read']is False
assert seal['actual_original_initializer']and not seal['source_emulation']and seal['all64_frames512_native_slots_preserved']
outputs={Path(b['path']).name:b for b in seal['outputs']}
assert len(outputs)==5
for b in outputs.values():check(b)
inputs=lines(check(seal['inputs']));transport=read(check(seal['canonical_transport_seal']));manifest=read(check(transport['manifest']))
assert inputs==manifest['rows']and len(inputs)==64 and len({r['id']for r in inputs})==64
assert transport['completed']and transport['PASS']and transport['frames']==64 and transport['GT_pose_K_depth_mask_label_body_reads']==0
props=lines(check(outputs['FROZEN_PROPOSALS_ALL64_INPUT_ONLY.jsonl']));queries=lines(check(outputs['ORIGINAL_NATIVE_QUERY512_INPUT_ONLY.jsonl']))
assert len(props)==64 and len(queries)==512
with np.load(check(outputs['ORIGINAL_NATIVE_QUERY512_AND_VALID9X9_INPUT_ONLY.npz']),allow_pickle=False)as f:q={k:f[k].copy()for k in f.files}
with np.load(check(outputs['ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz']),allow_pickle=False)as f:c={k:f[k].copy()for k in f.files}
assert set(c)=={'frame_offsets','selected_index','box_xyxy','keypoints_xy','keypoints_conf','score'}
assert c['frame_offsets'].shape==(65,)and c['selected_index'].shape==(64,)and c['frame_offsets'][0]==0 and c['frame_offsets'][-1]==236
assert(c['frame_offsets'][1:]>=c['frame_offsets'][:-1]).all()
assert c['keypoints_xy'].shape==(236,9,2)and c['keypoints_conf'].shape==(236,9)and c['box_xyxy'].shape==(236,4)and c['score'].shape==(236,)
for k in('score','keypoints_xy','keypoints_conf','box_xyxy'):assert c[k].dtype==np.float64 and np.isfinite(c[k]).all()
assert np.array_equal(q['global_index'],np.arange(512))and np.array_equal(q['frame_index'],np.repeat(np.arange(64),8))and np.array_equal(q['native_slot'],np.tile(np.arange(8),64))
expected={k:[]for k in('query_xy','L','finite_non_sentinel','continuous_valid','nearest_valid','RGBcentre_valid','native_grid_valid','no_detection')}
tie_frames=[];selected=[];counts_by_fold={};perframe=[]
for i,(row,p)in enumerate(zip(inputs,props)):
    assert row['frame_index']==p['frame_index']==i
    for k in row:assert row[k]==p[k],k
    assert p['actual_original_initializer']and p['source_emulation']is False
    lo,hi=map(int,c['frame_offsets'][i:i+2]);candidates=p['candidates'];assert len(candidates)==hi-lo
    for j,r in enumerate(candidates):
        assert r['candidate_index']==j
        for k in('box_xyxy','keypoints_xy','keypoints_conf','score'):assert np.array_equal(np.asarray(r[k],np.float64),c[k][lo+j]),k
    selected_index=int(np.argmax(c['score'][lo:hi]))if hi>lo else None
    assert p['selected_index']==selected_index and c['selected_index'][i]==(-1 if selected_index is None else selected_index)
    selected.append(selected_index)
    if hi>lo and(c['score'][lo:hi]==c['score'][lo+selected_index]).sum()>1:tie_frames.append(i)
    hw=row['raw_hw'];height,width=hw
    if selected_index is None:
        xy=np.full((8,2),np.nan,np.float64);L=np.full(8,np.nan,np.float64);finite=np.zeros(8,bool);continuous=nearest=centre=np.zeros(8,bool);gridvalid=np.zeros((8,9,9),bool)
    else:
        raw9=c['keypoints_xy'][lo+selected_index];xy=raw9[:8].copy()
        finite=np.isfinite(xy).all(1)&~np.all(xy==-1,axis=1)
        usable=xy[finite];side=float(np.clip(np.ptp(usable,axis=0).max()/4,16,max(hw)))if len(usable)>=2 else 16.
        L=np.full(8,side,np.float64)
        continuous=finite&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
        nearest=finite&(np.floor(xy[:,0]+.5)>=0)&(np.floor(xy[:,0]+.5)<width)&(np.floor(xy[:,1]+.5)>=0)&(np.floor(xy[:,1]+.5)<height)
        centre=finite&(xy[:,0]>=-.5)&(xy[:,0]<width-.5)&(xy[:,1]>=-.5)&(xy[:,1]<height-.5)
        offsets=((np.arange(9)+.5)/9-.5)*side
        xx,yy=np.meshgrid(offsets,offsets);grid=xy[:,None,None,:]+np.stack([xx,yy],-1)[None]
        gridvalid=finite[:,None,None]&(grid[...,0]>=0)&(grid[...,0]<width)&(grid[...,1]>=0)&(grid[...,1]<height)
    for k,a in dict(query_xy=xy,L=L,finite_non_sentinel=finite,continuous_valid=continuous,nearest_valid=nearest,RGBcentre_valid=centre,native_grid_valid=gridvalid,no_detection=np.repeat(selected_index is None,8)).items():expected[k].append(a)
    for k in range(8):
        j=i*8+k;r=queries[j]
        assert r['query_id']==row['id']+':native'+str(k)==q['query_id'][j]
        assert r['frame_id']==row['id']==q['frame_id'][j]and r['source_id']==row['source_id']and r['source']==row['source']
        assert r['frame_index']==i and r['native_slot']==k and r['split']==row['split']==q['split'][j]
        assert r['association_status']=='UNASSIGNED_BY_INPUT_PRODUCER'and r['actual_original_initializer']and r['source_emulation']is False
        assert np.array_equal(r['query_xy'],xy[k])and r['L']==L[k]
        for key,value in dict(no_detection=selected_index is None,finite_non_sentinel=finite[k],native_continuous_valid=continuous[k],query_nearest_pixel_valid=nearest[k],query_RGB96_centre_domain_valid=centre[k],native_grid_any_valid=gridvalid[k].any()).items():assert r[key]==bool(value),key
    perframe.append(dict(frame_index=i,id=row['id'],split=row['split'],candidates=hi-lo,selected_index=selected_index,continuous_valid=int(continuous.sum()),finite_OOB=int((finite&~continuous).sum()),L=None if selected_index is None else float(L[0])))
for k,parts in expected.items():assert eq(np.concatenate(parts,axis=0),q[k]),k
for fold in('TRAIN_LAB','VAL_HALL'):
    mask=q['split']==fold;assert mask.sum()==256
    counts_by_fold[fold]=dict(frames=int(mask.sum()/8),slots=int(mask.sum()),finite=int(q['finite_non_sentinel'][mask].sum()),continuous_valid=int(q['continuous_valid'][mask].sum()),nearest_valid=int(q['nearest_valid'][mask].sum()),OOB=int((q['finite_non_sentinel'][mask]&~q['continuous_valid'][mask]).sum()),noquery=int(q['no_detection'][mask].sum()),native_grid_any_valid=int(q['native_grid_valid'][mask].any((1,2)).sum()))
result=read(check(outputs['RESULT.json']));started=read(D/'STARTED.json')
assert result['completed']and result['PASS']and result['frames']==64 and result['native_slots']==512 and result['candidate_count']==236
assert int(q['finite_non_sentinel'].sum())==result['finite_non_sentinel_slots']==512 and int(q['continuous_valid'].sum())==result['continuous_valid_slots']==463
assert int(q['nearest_valid'].sum())==result['nearest_valid_slots']==463 and int((q['finite_non_sentinel']&~q['continuous_valid']).sum())==result['finite_OOB_slots']==49
assert int(q['no_detection'].sum())==result['no_detection_frames']==0
assert result['explicit_predict_calls']==66 and result['total_forward_hooks']==67 and result['generated_warmup_hooks']==2
assert result['warmup_state_SHA']==started['warmed_state_SHA']and result['after_state_unchanged']and result['checkpoint_unchanged']and result['repeat_first_serial_and_P3_exact']
assert result['GT_pose_K_reference_semantic_mask_or_historic_score_reads']==0 and result['visibility_classifier_fit_private_V112_PnP_q_L_correction_calls']==0
report=dict(completed=True,PASS=True,review_script=bind(__file__),prediction_seal=bind(SEAL),inputs=seal['inputs'],transport_seal=seal['canonical_transport_seal'],outputs=seal['outputs'],
    frames64_slots512_all_retained=True,all236_candidates_raw9_confidence_box_score_exact_JSON_NPZ=True,all64_firstargmax_and_selected_indices_recomputed=True,actual_equal_maximum_tie_frames=tie_frames,
    all512_q_L_recomputed_from_selected_original_first8=True,centroid_excluded_from_L=True,all512_finite_continuous_nearest_RGBcentre_native9x9_masks_recomputed=True,
    all512_queryJSON_NPZ_metadata_flags_ID_q_L_exact=True,all64_inputs_equal_completed_canonical_manifest=True,counts_by_fold=counts_by_fold,per_frame=perframe,
    source_model_repeat_state_checkpoint_sealed_receipt_consistency=True,model_repeat_or_checkpoint_state_independently_replayed=False,
    RGB_NPY_checkpoint_GT_pose_K_mask_label_current_V112_body_reads=0,model_calls=0,producer_function_imports=0,rerun=False,semantic_association_or_visibility_not_inferred=True)
with(H/'REVIEW.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'REVIEW.json'),counts=counts_by_fold)))
