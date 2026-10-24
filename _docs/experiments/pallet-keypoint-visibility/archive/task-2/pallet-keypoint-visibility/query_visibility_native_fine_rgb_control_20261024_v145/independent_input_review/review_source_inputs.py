"""Independent saved-source RGB/identity/geometry/base score/pool feasibility audit."""
from pathlib import Path
from collections import Counter
import hashlib,json,time
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_native_fine_rgb_control_20261024_v145';H=V/'independent_input_review'
C32=P/'query_visibility_native_target_gradient_control_20261024_v142/runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/C32_AUX_TO_NATIVE_GRAD0_FIXED8000'
POLICY=V/'ROOT_FINE_NATIVE_CONTROL_POLICY_BEFORE_FIT.json';GSEAL=P/'query_visibility_saved_query_reliability_20261024_v138/input_only_reliability_v1/INPUT_SEAL.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert '_v112'not in b['path'].lower();assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
start=time.monotonic();assert sha(POLICY)=='6e6bd9c7ebaff4f6dcd3b69b91867583957501d48fa23c3934c78f9e5066977b';policy=read(POLICY)
freeze=read(check(policy['base_fixed_before_fit']));assert freeze['arm']=='C32_AUX_TO_NATIVE_GRAD0_FIXED8000'and not freeze['source_eligible']and not freeze['current_allowed']
gseal=read(GSEAL);assert gseal['completed']and gseal['PASS']and gseal['GT_read']is False
GKEYS=('global_index','corpus','source_occurrence_index','occurrence_id','scene_index','base_index','native_slot','split','query_xy','L','G42','G42_constructed','query_available','crop_constructed','feature_constructed','finite_non_sentinel_query','native_continuous_valid','native_nearest_valid','context_geometry_valid','RGB_central4_available')
with np.load(check(gseal['table']),allow_pickle=False)as f:geom={k:f[k].copy()for k in GKEYS}
assert eq(geom['global_index'],np.arange(26624,dtype=np.int32))and geom['G42'].shape==(26624,42)and geom['G42'].dtype==np.float32 and np.isfinite(geom['G42']).all()
assert not geom['G42'][~geom['G42_constructed']].any()
configs=[('OLD_NATIVE','query_visibility_actual_initializer_source_20261024_v132',6144,128,125,(1869,131),'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz'),('CONTINUOUS_NATIVE','query_visibility_continuous_native_source_20261024_v136',20480,512,473,(6920,496),'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz')]
results={};offset=0;allchunks=[];pairpools={}
for corpus,dirname,N,trainbases,paircount,support,packetname in configs:
 root=P/dirname;cs=read(root/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json');assert cs['completed']and cs['PASS']and cs['GT_read']is False and cs['rows']==N
 facts=read(check(cs['input_crop_facts']));rows=facts['rows'];assert facts['completed']and facts['GT_read']is False and len(rows)==N
 idx=slice(offset,offset+N);g={k:v[idx]for k,v in geom.items()};assert (g['corpus']==corpus).all()and eq(g['source_occurrence_index'],np.arange(N,dtype=np.int32))
 with np.load(C32/packetname,allow_pickle=False)as f:base={k:f[k].copy()for k in f.files}
 assert base['raw_head_logit_before_U'].dtype==np.float32 and base['raw_head_logit_before_U'].shape==(N,)
 assert np.array_equal(base['global_index'],np.arange(N))and np.array_equal(base['occurrence_id'],g['occurrence_id'])and eq(base['query_xy'],g['query_xy'])and eq(base['L'],g['L'])and eq(base['query_available'],g['query_available'])and eq(base['head_forwarded'],g['feature_constructed'])and eq(base['split'],g['split'])
 assert eq(base['query_logit'],np.where(base['query_available'],base['raw_head_logit_before_U'],0).astype(np.float32))
 assert np.array_equal(base['prediction'],np.where(base['query_available'],(base['raw_head_logit_before_U']>=0).astype(np.int8),-1))
 assert np.array_equal(np.isfinite(base['raw_head_logit_before_U']),g['feature_constructed'])
 expectedfold=np.where(np.arange(N)//32<trainbases,'TRAIN','VALIDATION');assert np.array_equal(expectedfold,g['split'])
 diag=read(root/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json');assert diag['completed']and diag['PASS']
 ledgerpath=root/'saved_initializer_diagnosis_v1'/('ALL'+str(N)+'_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl')
 assert any(b['path']==str(ledgerpath)and sha(ledgerpath)==b['sha256']for b in diag['outputs'])
 primary=[]
 for j,line in enumerate(ledgerpath.open()):
  r=json.loads(line)
  assert r['convention']==('CF_PRIMARY_DECLARED'if j%2==0 else'V4_SENSITIVITY_ONLY')
  if j%2==0:primary.append(r)
 assert len(primary)==N
 for i,(r,crop)in enumerate(zip(primary,rows)):
  assert r['query_id']==crop['occurrence_id']==str(g['occurrence_id'][i])and r['scene_index']==crop['scene_index']==int(g['scene_index'][i])and r['base_index']==crop['base_index']==int(g['base_index'][i])and r['native_slot']==crop['native_slot']==int(g['native_slot'][i])and r['split']==crop['split']==str(g['split'][i])
  q=np.array(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan],np.float64);L=np.float64(r['roi_side']if r['roi_side']is not None else np.nan)
  assert np.array_equal(q,g['query_xy'][i],equal_nan=True)and (L==g['L'][i]or np.isnan(L)and np.isnan(g['L'][i]))
  assert r['teacher_state']==base['raw_teacher_state_CF'][i]and r['correspondence_boundary']==base['convention_boundary'][i]
  for k in('crop_constructed','feature_constructed','query_available','native_slot','split'):assert crop[k]==g[k][i]
 states=np.array([r['teacher_state']for r in primary]);target=np.where(states=='D',0,np.where(states=='E',1,-1)).astype(np.int8)
 assert eq(target,base['external_target_CF'])
 eligible=(g['split']=='TRAIN')&g['query_available']&np.isin(target,[0,1]);pools=[np.flatnonzero(eligible&(target==t))for t in(0,1)];assert tuple(map(len,pools))==support
 pairs=[]
 for b in range(trainbases):
  for slot in range(8):
   e,d=(b*4+1)*8+slot,(b*4+2)*8+slot
   if states[e]=='E'and states[d]=='D'and g['query_available'][e]and g['query_available'][d]:
    a,z=primary[e],primary[d];assert a['canonical_index']==z['canonical_index']and a['base_index']==z['base_index']==b and a['native_slot']==z['native_slot']==slot and a['split']==z['split']=='TRAIN'and a['variant']=='COVER'and z['variant']=='NEAR_SHAM'
    pairs.append((e,d))
 pairs=np.asarray(pairs,np.int64);assert pairs.shape==(paircount,2)and len(np.unique(pairs))==paircount*2 and eligible[pairs].all()
 pairpools[corpus]=pairs.tolist()
 seen=np.zeros(N,np.int8);centralfractions=[];readbytes=0
 for b in facts['crop_chunks']:
  path=check(b);allchunks.append(b);readbytes+=path.stat().st_size
  with np.load(path,allow_pickle=False)as f:
   assert set(f.files)=={'rgb96','valid96','crop_constructed','global_index'}
   rgb=f['rgb96'];valid=f['valid96'];constructed=f['crop_constructed'];ix=f['global_index']
   assert rgb.dtype==np.uint8 and rgb.shape==(512,96,96,3)and valid.dtype==np.bool_ and valid.shape==(512,96,96)and constructed.dtype==np.bool_ and constructed.shape==(512,)and ix.dtype==np.int32 and ix.shape==(512,)
   assert np.array_equal(ix,np.arange(ix[0],ix[0]+512))and not seen[ix].any();seen[ix]+=1
   assert np.array_equal(constructed,g['crop_constructed'][ix])
   for j,i in enumerate(ix):
    row=rows[int(i)];assert row['rgb96_chunk']==b and row['rgb96_row']==j
    assert hashlib.sha256(np.ascontiguousarray(rgb[j]).tobytes()).hexdigest()==row['rgb96_sha256']and hashlib.sha256(np.ascontiguousarray(valid[j]).tobytes()).hexdigest()==row['valid96_sha256']
   assert not rgb[~constructed].any()and not valid[~constructed].any()
   assert np.array_equal(valid[:,47:49,47:49].all((1,2)),np.array([rows[int(i)]['RGB_central4_available']for i in ix]))
   central=rgb[:,32:64,32:64,:];cv=valid[:,32:64,32:64];assert central.shape==(512,32,32,3)and cv.shape==(512,32,32)
   centralfractions.extend(cv.mean((1,2)).tolist())
 assert (seen==1).all()and len(facts['crop_chunks'])==N//512
 results[corpus]=dict(rows=N,source_scenes=N//8,source_bases=N//32,original_TRAIN_bases=trainbases,original_VAL_bases=N//32-trainbases,all_rows_preserved=True,crop_chunks=N//512,compressed_crop_bytes_hashed=readbytes,allRGB96_valid96_row_hashes_and_chunk_global_indices_exact=True,crop_format='uint8[N,96,96,3] HWC; bool[N,96,96]',central32_exact_slice='RGB[:,32:64,32:64,:];valid[:,32:64,32:64]; convert RGB NCHW for conv',feature_constructed=int(g['feature_constructed'].sum()),query_available=int(g['query_available'].sum()),input_U=int((~g['query_available']).sum()),raw_base_nonfinite=int((~np.isfinite(base['raw_head_logit_before_U'])).sum()),finite_raw_but_inputU=int((np.isfinite(base['raw_head_logit_before_U'])&~g['query_available']).sum()),all_C32_G42_crop_GT_ID_q_L_splits_availability_joins_exact=True,teacher_state_counts=dict(Counter(states.tolist())),eligible_TRAIN_D_E=list(map(len,pools)),TRAIN_available_DE_rows=int(eligible.sum()),excluded_TRAIN_teacherS_U=int(((g['split']=='TRAIN')&~np.isin(target,[0,1])).sum()),TRAIN_inputU_allstates=int(((g['split']=='TRAIN')&~g['query_available']).sum()),pair_pool_count=paircount,all_pairs_TRAIN_available_CF_samecanonical_COVER_E_NEAR_D=True,no_boundary_or_error_or_score_based_pair_filter=True,central32_valid_fraction_quantiles=np.quantile(np.array(centralfractions),[0,.25,.5,.75,1]).tolist(),C32_scores=bind(C32/packetname),crop_seal=bind(root/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'),GT_seal=bind(root/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json'),GT_primary_ledger=bind(ledgerpath))
 offset+=N
assert offset==26624 and len(allchunks)==52
source=read(C32/'SOURCE_RESULTS.json');gates=source['source_gates']if 'source_gates'in source else source['gates']
nativegates={k:v for k,v in gates.items()if k.startswith(('NATIVE::','NEW_NATIVE::'))};assert len(nativegates)==35
result=dict(completed=True,PASS=True,policy=bind(POLICY),code=bind(Path(__file__)),base_C32_freeze=policy['base_fixed_before_fit'],G42_input_seal=bind(GSEAL),G42_table=gseal['table'],corpora=results,total_rows=26624,total_crop_chunks=52,total_available=sum(r['query_available']for r in results.values()),total_inputU=sum(r['input_U']for r in results.values()),total_feature_constructed=sum(r['feature_constructed']for r in results.values()),native35_existing_gates=nativegates,source_input_feature_whitelist=['saved RGB96 central32','matching central32 valid96','existing input-only G42','frozen C32 raw_head_logit_before_U'],forbidden_model_features=['GTcoords','GTerror','teacherstate','sourceGTpose','split/factor/domain numericcode','newV138risks/confidence','current223'],loss_mask='TRAIN AND original query_available AND teacher_CF in D/E; teacherS/U and every inputU have zero BCE contribution',noquery_NaN_and_zero_placeholders='Preserve rawNaN/constructFalse rows and final U; do not make them fake measured negative crops.',new_probe_AUX29='unmeasured/inapplicable, do not claim original64PASS; originalC32sourcefail unchanged',native_sampler='old64/new64: pairedE16/pairedD16/uniformD16/uniformE16, original125/473 pools; same draw positions botharms',source_input_readiness_only=True,actual_fit_or_encoder_initializer_baseline_forward_calls=0,current223_V112_PnP_access=0,seconds=time.monotonic()-start)
with (H/'PAIR_POOLS_SOURCE_ONLY.json').open('x')as f:json.dump(pairpools,f,allow_nan=False);f.write('\n')
result['pair_pools']=bind(H/'PAIR_POOLS_SOURCE_ONLY.json')
with (H/'REVIEW.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),corpora=results,total_rows=26624,total_chunks=52)),flush=True)
