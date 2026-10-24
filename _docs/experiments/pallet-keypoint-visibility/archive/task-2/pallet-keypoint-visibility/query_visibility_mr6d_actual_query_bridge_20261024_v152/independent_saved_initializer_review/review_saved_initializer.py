"""Independent saved-only V152 raw9 -> native80 arithmetic audit.
Never imports the producer, torch, models, feature/RGB/GT/checkpoint payloads.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def eq(a,b):
 a,b=np.asarray(a),np.asarray(b)
 same=np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else np.array_equal(a,b)
 assert same,(a.shape,b.shape)
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float):return x if np.isfinite(x)else None
 if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[clean(v)for v in x]
 return x

def direct(q,hw):
 h,w=hw
 if q is None:return dict(query_xy=np.full((8,2),np.nan),L=np.full(8,np.nan),finite_non_sentinel=np.zeros(8,bool),continuous_valid=np.zeros(8,bool),nearest_valid=np.zeros(8,bool),RGBcentre_valid=np.zeros(8,bool),native_grid_valid=np.zeros((8,9,9),bool))
 q=np.asarray(q,np.float64)[:8].copy(); good=np.isfinite(q).all(1)&~np.all(q==-1,axis=1)
 span=np.ptp(q[good],axis=0) if good.sum()>=2 else None
 L=float(np.clip(.25*max(span),16,max(hw))) if span is not None else 16.
 delta=((np.arange(9,dtype=float)+.5)/9-.5)*L
 xx,yy=np.meshgrid(delta,delta);grid=q[:,None,None,:]+np.stack((xx,yy),-1)[None]
 valid=(grid[...,0]>=0)&(grid[...,0]<w)&(grid[...,1]>=0)&(grid[...,1]<h)&good[:,None,None]
 continuous=good&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
 with np.errstate(invalid='ignore'):
  nearest=good&(np.floor(q[:,0]+.5)>=0)&(np.floor(q[:,0]+.5)<w)&(np.floor(q[:,1]+.5)>=0)&(np.floor(q[:,1]+.5)<h)
 centre=good&(q[:,0]>=-.5)&(q[:,0]<w-.5)&(q[:,1]>=-.5)&(q[:,1]<h-.5)
 assert np.array_equal(nearest,centre)
 return dict(query_xy=q,L=np.full(8,L),finite_non_sentinel=good,continuous_valid=continuous,nearest_valid=nearest,RGBcentre_valid=centre,native_grid_valid=valid)

def main(expected):
 out=V/'actual_mr6d_initializer_v1';sp=out/'PREDICTION_SEAL.json';assert sha(sp)==expected
 seal=read(sp);assert seal['completed']and seal['PASS']and seal['frames']==10 and seal['rows']==80 and seal['all10_frames80_native_slots_preserved']and seal['actual_original_initializer']and not seal['source_emulation']
 assert seal['GT_read']is False and seal['historical_predictions_read']is False
 for k in ('code','policy','protocol','entry','inputs','canonical_transport_seal'):ck(seal[k])
 entry=read(ck(seal['entry']));p=read(ck(seal['protocol']));assert entry['approved_actual_public_initializer']is True
 assert all(entry[k]is False for k in ('classifier_forward','fit','current223','V112','PnP','pose_optimization','q_L_correction','GT_input','GT_centered_crop','detection_reselection','historical_score_read','automatic_retry'))
 for k in ('code','policy','protocol','inputs','canonical_transport_seal'):assert entry[k]==seal[k]
 assert entry['generated']==p['generated'] and read(ck(p['generated']))['PASS']
 assert p['code']==seal['code'] and p['policy']==seal['policy'] and p['inputs']==seal['inputs'] and p['canonical_transport_seal']==seal['canonical_transport_seal']
 outputs={Path(x['path']).name:ck(x) for x in seal['outputs']};assert len(outputs)==5
 proposals=lines(outputs['FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl']);queries=lines(outputs['ORIGINAL_NATIVE_QUERY80_INPUT_ONLY.jsonl']);rows=lines(ck(seal['inputs']))
 qz=np.load(outputs['ORIGINAL_NATIVE_QUERY80_AND_VALID9X9_INPUT_ONLY.npz'],allow_pickle=False)
 cz=np.load(outputs['ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz'],allow_pickle=False)
 result=read(outputs['RESULT.json']);started=read(out/'STARTED.json')
 assert result['completed']and result['PASS']and len(proposals)==len(rows)==10 and len(queries)==80
 assert started['entry']==seal['entry'] and started['protocol']==seal['protocol'] and started['inputs']==seal['inputs'] and started['GT_read']is False
 assert started['source_forward_calls']==0 and started['warmed_state_SHA']==result['warmup_state_SHA']
 assert result['after_state_unchanged']and result['checkpoint_unchanged']and result['repeat_first_serial_and_P3_exact']
 assert result['all_candidates_raw9_and_firstargmax_unchanged']and result['centroid_excluded_from_native_q_and_L']and result['all10_and80_retained']
 assert result['explicit_predict_calls']==12 and result['total_forward_hooks']==result['generated_warmup_hooks']+11
 assert result['generated_warmup_hooks']==started['generated_blank_warmup_hooks']
 assert result['GT_pose_K_reference_semantic_mask_or_historic_score_reads']==result['visibility_classifier_fit_private_V112_PnP_q_L_correction_calls']==0
 assert result['association_not_inferred']
 eq(qz['global_index'],np.arange(80));eq(qz['frame_index'],np.repeat(np.arange(10),8));eq(qz['native_slot'],np.tile(np.arange(8),10))
 assert cz['frame_offsets'].shape==(11,)and cz['frame_offsets'][0]==0 and np.all(np.diff(cz['frame_offsets'])>=0)
 expectedall={k:[]for k in direct(None,[1,1])};facts=[];offset=0
 for i,(r,z) in enumerate(zip(rows,proposals)):
  assert all(z[k]==v for k,v in r.items()) and z['actual_original_initializer']and not z['source_emulation']
  cand=z['candidates'];n=len(cand);assert int(cz['frame_offsets'][i])==offset and int(cz['frame_offsets'][i+1])==offset+n
  scores=np.asarray(cz['score'][offset:offset+n]);selected=None if not n else int(np.argmax(scores))
  assert selected==z['selected_index'] and int(cz['selected_index'][i])==(-1 if selected is None else selected)
  for j,c in enumerate(cand):
   assert c['candidate_index']==j
   for key in ('score','box_xyxy','keypoints_xy','keypoints_conf'):
    assert clean(cz[key][offset+j])==c[key]
   assert np.asarray(c['keypoints_xy']).shape==(9,2) and np.asarray(c['keypoints_conf']).shape==(9,)
  q=None if selected is None else cz['keypoints_xy'][offset+selected]
  d=direct(q,r['raw_hw']);
  for k,v in d.items():expectedall[k].append(v)
  for k in range(8):
   t=queries[i*8+k]
   required=dict(query_id=r['id']+':native'+str(k),frame_index=i,frame_id=r['id'],source=r['source'],source_id=r['source_id'],split=r['split'],native_slot=k,query_xy=clean(d['query_xy'][k]),L=clean(d['L'][k]),no_detection=selected is None,finite_non_sentinel=bool(d['finite_non_sentinel'][k]),native_continuous_valid=bool(d['continuous_valid'][k]),query_nearest_pixel_valid=bool(d['nearest_valid'][k]),query_RGB96_centre_domain_valid=bool(d['RGBcentre_valid'][k]),native_grid_any_valid=bool(d['native_grid_valid'][k].any()),association_status='UNASSIGNED_BY_INPUT_PRODUCER',actual_original_initializer=True,source_emulation=False)
   assert t==required
  eq(qz['frame_id'][i*8:i*8+8],np.full(8,r['id']));eq(qz['split'][i*8:i*8+8],np.full(8,r['split']))
  eq(qz['no_detection'][i*8:i*8+8],np.full(8,selected is None))
  facts.append(dict(frame_index=i,id=r['id'],source_id=r['source_id'],candidates=n,firstargmax=selected,finite=int(d['finite_non_sentinel'].sum()),continuous=int(d['continuous_valid'].sum()),nearest=int(d['nearest_valid'].sum()),finite_OOB=int((d['finite_non_sentinel']&~d['continuous_valid']).sum())))
  offset+=n
 for k,v in expectedall.items():eq(qz[k],np.concatenate(v,axis=0))
 assert qz['query_id'].tolist()==[r['query_id']for r in queries]
 assert cz['keypoints_xy'].shape==(offset,9,2)and cz['keypoints_conf'].shape==(offset,9)and cz['box_xyxy'].shape==(offset,4)
 assert result['candidate_count']==offset==int(cz['frame_offsets'][-1])
 assert result['finite_non_sentinel_slots']==int(qz['finite_non_sentinel'].sum()) and result['continuous_valid_slots']==int(qz['continuous_valid'].sum())and result['nearest_valid_slots']==int(qz['nearest_valid'].sum())
 assert result['finite_OOB_slots']==int((qz['finite_non_sentinel']&~qz['continuous_valid']).sum())and result['no_detection_frames']==sum(x['firstargmax']is None for x in facts)
 report=dict(completed=True,PASS=True,code=bind(__file__),prediction_seal=bind(sp),entry=seal['entry'],protocol=seal['protocol'],frames=10,rows=80,candidate_count=offset,frame_facts=facts,all80_q_L_valid9x9_exact_from_saved_raw9=True,all_candidate_offsets_firstargmax_JSON_NPZ_exact=True,all_ID_namespace_noquery_OOB_slots_preserved=True,recorded_terminal_repeat_state_checkpoint_guards_PASS=True,model_or_RGB_feature_checkpoint_GT_body_reads_or_calls=0,current_private_V112_access=0,teacher_or_association_created=False,limitations=['State/repeat/checkpoint invariance is checked as a bound producer terminal receipt; this audit does not replay a model or independently reread model tensors.','Saved q and candidate arithmetic does not establish GT instance or dynamic-native role correspondence.'])
 rp=H/'REVIEW.json'
 with rp.open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=bind(rp),candidate_count=offset,frames=10,rows=80)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prediction-seal-sha',required=True);z=a.parse_args();main(z.prediction_seal_sha)
