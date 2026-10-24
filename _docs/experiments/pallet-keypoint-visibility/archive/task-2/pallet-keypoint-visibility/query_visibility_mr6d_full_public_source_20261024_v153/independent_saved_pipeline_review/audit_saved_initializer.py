"""Independent NumPy audit of sealed public R0 outputs; no model/helper import."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
O=V/'actual_full230_initializer_v1';SEAL=O/'PREDICTION_SEAL.json'
OLD=P/'query_visibility_mr6d_actual_query_bridge_20261024_v152/actual_mr6d_initializer_v1'
EXPECTED='e986a15f998974e6c1aabade48aff7cbf02565fbc6512339425bae175d272f5d'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def packet(p):
 with np.load(p,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def grid(q,hw):
 # Original native8 span rule and nine pixel-center offsets, expressed here
 # directly without importing the producer or V60 helper.
 h,w=hw;good=np.isfinite(q).all(1)&~(q==[-1,-1]).all(1);finite=q[good]
 side=float(np.clip(.25*np.max(np.ptp(finite,axis=0)),16,max(hw)))if len(finite)>=2 else 16.
 offsets=((np.arange(9)+.5)/9-.5)*side;xx,yy=np.meshgrid(offsets,offsets)
 xy=q[:,None,None,:]+np.stack((xx,yy),axis=-1)[None]
 valid=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&good[:,None,None]
 continuous=good&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
 with np.errstate(invalid='ignore'):
  nearest=good&(np.floor(q[:,0]+.5)>=0)&(np.floor(q[:,0]+.5)<w)&(np.floor(q[:,1]+.5)>=0)&(np.floor(q[:,1]+.5)<h)
 centre=good&(q[:,0]>=-.5)&(q[:,0]<w-.5)&(q[:,1]>=-.5)&(q[:,1]<h-.5)
 assert np.array_equal(nearest,centre)
 return dict(query_xy=q,L=np.full(8,side,np.float64),finite_non_sentinel=good,continuous_valid=continuous,nearest_valid=nearest,RGBcentre_valid=centre,native_grid_valid=valid)

def run():
 assert bind(SEAL)['sha256']==EXPECTED
 s=read(SEAL);assert s['completed']and s['PASS']and s['GT_read']is False and s['frames']==230 and s['rows']==1840
 for key in ('code','policy','protocol','entry','inputs','canonical_transport_seal'):check(s[key])
 files={Path(x['path']).name:check(x)for x in s['outputs']}
 props=lines(files['FROZEN_PROPOSALS_ALL230_INPUT_ONLY.jsonl']);rows=lines(files['ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'])
 q=packet(files['ORIGINAL_NATIVE_QUERY1840_AND_VALID9X9_INPUT_ONLY.npz']);c=packet(files['ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz']);result=read(files['RESULT.json'])
 inputs=lines(check(s['inputs']));manifest=read(check(read(check(s['canonical_transport_seal']))['RGB_manifest']))['rows']
 assert inputs==manifest and len(props)==230 and len(rows)==1840 and len({r['source_id']for r in props})==230
 assert Counter(r['split']for r in inputs)==Counter(SOURCE_DEVELOPMENT=185,SOURCE_VALIDATION=45)
 assert q['query_xy'].shape==(1840,2)and q['query_xy'].dtype==q['L'].dtype==np.float64 and q['native_grid_valid'].shape==(1840,9,9)
 assert exact(q['global_index'],np.arange(1840,dtype=np.int32))and exact(q['frame_index'],np.repeat(np.arange(230,dtype=np.int32),8))and exact(q['native_slot'],np.tile(np.arange(8,dtype=np.int8),230))
 for key in ('finite_non_sentinel','continuous_valid','nearest_valid','RGBcentre_valid','native_grid_valid','no_detection'):assert q[key].dtype==bool
 assert set(c)=={'frame_offsets','selected_index','score','keypoints_xy','keypoints_conf','box_xyxy'}
 assert c['frame_offsets'].shape==(231,)and c['frame_offsets'].dtype==c['selected_index'].dtype==np.int32 and c['frame_offsets'][0]==0
 n=int(c['frame_offsets'][-1]);assert c['keypoints_xy'].shape==(n,9,2)and c['keypoints_conf'].shape==(n,9)and c['box_xyxy'].shape==(n,4)and c['score'].shape==(n,)
 for key in ('score','keypoints_xy','keypoints_conf','box_xyxy'):assert c[key].dtype==np.float64
 noquery=[];offset=0;scope=Counter()
 for i,(p,inp)in enumerate(zip(props,inputs)):
  assert all(p[k]==value for k,value in inp.items())and p['frame_index']==i and p['actual_original_initializer']and p['source_emulation']is False
  assert p['split']==('SOURCE_VALIDATION'if p['source_id'].startswith('val/000001:')else'SOURCE_DEVELOPMENT')
  assert p['added_border']==100 and p['canvas_shape']==[p['raw_hw'][0]+200,p['raw_hw'][1]+200]
  assert c['frame_offsets'][i]==offset
  for j,candidate in enumerate(p['candidates']):
   assert candidate['candidate_index']==j
   for key in ('score','keypoints_xy','keypoints_conf','box_xyxy'):
    value=np.asarray(candidate[key],np.float64);assert exact(value,c[key][offset+j])
  end=offset+len(p['candidates']);assert c['frame_offsets'][i+1]==end
  selected=None if end==offset else int(np.argmax(c['score'][offset:end]));assert p['selected_index']==selected and c['selected_index'][i]==(-1 if selected is None else selected)
  sl=slice(i*8,i*8+8)
  if selected is None:
   noquery.append(p['source_id']);expected=dict(query_xy=np.full((8,2),np.nan,np.float64),L=np.full(8,np.nan,np.float64),finite_non_sentinel=np.zeros(8,bool),continuous_valid=np.zeros(8,bool),nearest_valid=np.zeros(8,bool),RGBcentre_valid=np.zeros(8,bool),native_grid_valid=np.zeros((8,9,9),bool))
  else:expected=grid(c['keypoints_xy'][offset+selected,:8].copy(),p['raw_hw'])
  for key,value in expected.items():assert exact(q[key][sl],value),(i,key)
  assert np.array_equal(q['no_detection'][sl],np.full(8,selected is None))
  for k,row in enumerate(rows[sl]):
   g=i*8+k;identity=dict(query_id=p['id']+':native'+str(k),frame_index=i,frame_id=p['id'],source='MR6D',source_id=p['source_id'],split=p['split'],native_slot=k)
   assert all(row[key]==value for key,value in identity.items())
   for key in ('query_id','frame_index','frame_id','split','native_slot'):assert q[key][g]==identity[key]
   assert np.array_equal(np.asarray(row['query_xy'],np.float64),q['query_xy'][g],equal_nan=True)and np.array_equal(np.asarray(row['L'],np.float64),q['L'][g],equal_nan=True)
   for jk,nk in [('no_detection','no_detection'),('finite_non_sentinel','finite_non_sentinel'),('native_continuous_valid','continuous_valid'),('query_nearest_pixel_valid','nearest_valid'),('query_RGB96_centre_domain_valid','RGBcentre_valid')]:assert row[jk]==bool(q[nk][g])
   assert row['native_grid_any_valid']==bool(q['native_grid_valid'][g].any())and row['association_status']=='UNASSIGNED_BY_INPUT_PRODUCER'
   scope[(p['split'],'rows')]+=1;scope[(p['split'],'continuous')]+=int(q['continuous_valid'][g]);scope[(p['split'],'nearest')]+=int(q['nearest_valid'][g])
  offset=end
 assert offset==n==532
 # Old10 parity independently rechecked, with source_id as the sole join.
 oldseal=read(OLD/'PREDICTION_SEAL.json');assert oldseal['completed']and oldseal['PASS'];oldfiles={Path(x['path']).name:check(x)for x in oldseal['outputs']}
 op=lines(oldfiles['FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl']);oq=packet(oldfiles['ORIGINAL_NATIVE_QUERY80_AND_VALID9X9_INPUT_ONLY.npz']);oc=packet(oldfiles['ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz'])
 new={r['source_id']:(i,r)for i,r in enumerate(props)};proof=[]
 for oi,old in enumerate(op):
  ni,newrow=new[old['source_id']]
  for key in ('source_id','raw_hw','image_sha256','decoded_RGB_sha256','candidates','selected_index','input_shape','canvas_shape','added_border','affine_gain','affine_offset'):assert newrow[key]==old[key]
  assert newrow['canonical_RGB_NPY']['sha256']==old['canonical_RGB_NPY']['sha256']
  for key in ('query_xy','L','finite_non_sentinel','continuous_valid','nearest_valid','RGBcentre_valid','native_grid_valid','native_slot','no_detection'):assert exact(q[key][ni*8:ni*8+8],oq[key][oi*8:oi*8+8])
  for key in ('score','keypoints_xy','keypoints_conf','box_xyxy'):assert exact(c[key][c['frame_offsets'][ni]:c['frame_offsets'][ni+1]],oc[key][oc['frame_offsets'][oi]:oc['frame_offsets'][oi+1]])
  proof.append(dict(source_id=old['source_id'],old_index=oi,new_index=ni,allcandidate_raw9_boxes_conf_scores_q8_L_and_masks_bitexact=True))
 assert len(proof)==10
 counts=dict(frames=230,rows=1840,candidates=n,no_detection_frames=len(noquery),noquery_slots=int(q['no_detection'].sum()),finite_non_sentinel=int(q['finite_non_sentinel'].sum()),continuous_valid=int(q['continuous_valid'].sum()),nearest_valid=int(q['nearest_valid'].sum()),finite_OOB=int((q['finite_non_sentinel']&~q['continuous_valid']).sum()),native9x9_any_valid=int(q['native_grid_valid'].any((1,2)).sum()))
 for key,rk in [('frames','frames'),('rows','native_slots'),('candidates','candidate_count'),('no_detection_frames','no_detection_frames'),('finite_non_sentinel','finite_non_sentinel_slots'),('continuous_valid','continuous_valid_slots'),('nearest_valid','nearest_valid_slots'),('finite_OOB','finite_OOB_slots')]:assert counts[key]==result[rk]
 assert result['explicit_predict_calls']==232 and result['total_forward_hooks']==230+1+result['generated_warmup_hooks']and result['generated_warmup_hooks']==2
 assert result['after_state_unchanged']and result['checkpoint_unchanged']and result['repeat_first_serial_and_P3_exact']and result['no_saved_fullP3_duplicate']
 recalculation=dict(completed=True,PASS=True,counts=counts,byfold={fold:{k:scope[fold,k]for k in ('rows','continuous','nearest')}for fold in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION')},noquery_source_ids=noquery,old10_parity=proof,dtypes={k:str(v.dtype)for k,v in q.items()},raw_candidates_dtypes={k:str(v.dtype)for k,v in c.items()})
 path=H/'RECALCULATED_INITIALIZER_FIDELITY.json';path.write_text(json.dumps(recalculation,indent=2)+'\n')
 review=dict(completed=True,PASS=True,prediction_seal=bind(SEAL),code=bind(__file__),recalculation=bind(path),counts=counts,all1840_and_all532_candidates_original9_preserved=True,all_frame_offsets_firstargmax_native8_L_and_9x9_masks_recomputed=True,old10_originalinput_hashes_and_all_saved_outputs_bitexact=True,sourcefolds_185_development_45_validation_unchanged=True,GT_reference_pose_mask_depth_or_semantic_reads=0,RGB_body_reads=0,learned_model_instantiation_or_forward=0,private_reads=0,limits=['State/checkpoint/repeatP3 guards are crosschecked sealed producer assertions; no model state was rerun or newly measured by this saved-output audit.','This audit certifies input fidelity and availability, not target association/native visibility correctness.'])
 out=H/'INITIALIZER_REVIEW.json';out.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(dict(review=bind(out),**counts)),flush=True)
if __name__=='__main__':run()
