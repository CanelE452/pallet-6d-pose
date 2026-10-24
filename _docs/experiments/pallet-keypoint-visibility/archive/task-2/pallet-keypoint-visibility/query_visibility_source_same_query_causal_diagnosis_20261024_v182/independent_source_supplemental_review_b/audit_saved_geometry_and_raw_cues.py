"""V182 independent sourceTRAIN saved-crop geometry/raw-cue verification.
No producer/evaluator import, new sampling, original scene reads or model calls.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
GEOM=V/'saved_crop_geometry_sensitivity_v1/COMPLETE.json';CUE=V/'saved_raw_cue_audit_v1/COMPLETE.json'
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','private','_v169','_v177','_v178','synthetic_val','final_synth_val')) and p.suffix not in ('.pt','.pth','.ckpt')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(b):return [json.loads(s) for s in check(b).read_text().splitlines() if s]
def load(b,keys):
 with np.load(check(b),allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def qnumeric(q):return np.array([np.nan,np.nan] if q is None else q,np.float64)
def verify_decoder(a,av):
 assert a['raw_support_logits'].dtype==np.float32 and a['raw_support_logits'].shape==(4096,2) and a['support_prediction'].dtype==np.int8
 for i in range(4096):
  if av[i]:assert np.isfinite(a['raw_support_logits'][i]).all() and a['support_prediction'][i].tolist()==[int(v>0) for v in a['raw_support_logits'][i]]
  else:assert np.isnan(a['raw_support_logits'][i]).all() and (a['support_prediction'][i]==-1).all()
def compare(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:compare(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
 else:assert a==b,(a,b)
def main():
 assert not (H/'REVIEW.json').exists()
 g=read(GEOM);cue=read(CUE);assert g['PASS'] and cue['PASS'];check(g['code']);check(cue['code'])
 frozen=read(check(g['frozen_predictions']));assert frozen['PASS'] and frozen['predictions_sealed_before_target_read']
 source=read(check(frozen['common_input_complete']));protocol=read(check(frozen['protocol']));oldcache=read(check(protocol['original_source_cache_metadata']));old_domain=oldcache['domains']['SYNTH_TRAIN4096'];fit=read(check(frozen['frozen_V181_fit']));oldrows=lines(old_domain['metadata'])
 assert source['PASS'] and source['fit_admission'] is False and source['diagnostic_only']
 rr=lines(source['metadata']);assert len(rr)==4096 and all(r['split']=='TRAIN' and not r['fit_admission'] and not r['production_query_replay'] for r in rr)
 keys=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
 oldx=load(old_domain['inputs'],keys);oldy=load(old_domain['targets'],('support_target',))['support_target'];newy=load(source['targets'],('support_target',))['support_target'];newx=load(source['inputs'],keys)
 assert oldy.shape==newy.shape==(4096,2) and oldy.dtype==newy.dtype==np.int8
 assert int((~oldx['original_query_available']).sum())==80 and int((~newx['original_query_available']).sum())==96
 mapping={int(x):i for i,x in enumerate(oldx['original_global_index'])};assert len(mapping)==4096
 geomledger=lines(g['rows']);assert len(geomledger)==2048;ledger={(r['arm'],r['base_index'],r['native_slot']):r for r in geomledger};assert len(ledger)==2048
 summaries={};computed_rows=[];max_distance_error=0.
 for arm,model in frozen['arms'].items():
  new=load(model['outputs'],('raw_support_logits','support_prediction')+keys);old=load(fit['arms'][arm]['outputs']['SYNTH_TRAIN4096'],('raw_support_logits','support_prediction')+keys)
  for k in keys:assert same(new[k],newx[k]) and same(old[k],oldx[k])
  verify_decoder(new,newx['original_query_available']);verify_decoder(old,oldx['original_query_available'])
  armrows=[]
  for i,r in enumerate(rr):
   assert r['diagnostic_row_index']==i and str(newx['row_id'][i])==r['row_id'] and int(newx['native_slot'][i])==r['native_slot']
   assert same(newx['query_xy'][i],qnumeric(r['query_xy'])) and (math.isnan(newx['L'][i]) if r['L'] is None else float(newx['L'][i])==r['L'])
   if r['variant']!='BEFORE':continue
   j=mapping[r['source_image_original_global_index']]
   assert str(oldx['row_id'][j])==r['source_image_original_query_id'] and int(oldx['native_slot'][j])==r['native_slot']
   assert same(oldx['query_xy'][j],qnumeric(r['image_original_query_xy'])) and (math.isnan(oldx['L'][j]) if r['image_original_L'] is None else float(oldx['L'][j])==r['image_original_L'])
   assert r['source_RGB_binding']==r['source_image_original_metadata']['full_RGB_packet'] and r['source_image_original_metadata']['variant']=='BEFORE'
   assert r['source_image_original_metadata']==oldrows[j]['original'] and r['source_RGB_binding']==oldrows[j]['original']['full_RGB_packet']
   assert r['source_image_original_metadata']['raw_RGB_PNG']==oldrows[j]['original']['raw_RGB_PNG']
   avold=bool(oldx['original_query_available'][j]);avnew=bool(newx['original_query_available'][i]);same_truth=all(int(a)>=0 and int(b)>=0 and int(a)==int(b) for a,b in zip(oldy[j],newy[i]));valid=avold and avnew and same_truth
   qo,qn=r['image_original_query_xy'],r['query_xy'];dist=None if qo is None or qn is None else math.sqrt(sum((float(b)-float(a))**2 for a,b in zip(qo,qn)))
   dl=None if r['image_original_L'] is None or r['L'] is None else float(r['L'])-float(r['image_original_L'])
   row=dict(arm=arm,base_index=r['base_index'],native_slot=r['native_slot'],diagnostic_row_index=i,source_image_original_global_index=r['source_image_original_global_index'],anchor_original_global_index=r['anchor_original_global_index'],same_saved_BEFORE_RGB_binding=r['source_RGB_binding'],original_q=qo,common_q=qn,q_shift_image_px=dist,original_L=r['image_original_L'],common_L=r['L'],delta_L_image_px=dl,original_available=avold,common_available=avnew,both_support_truth_known_and_same=same_truth,original_truth=oldy[j].tolist(),common_truth=newy[i].tolist(),original_prediction=old['support_prediction'][j].tolist(),common_prediction=new['support_prediction'][i].tolist(),original_correct=all(int(a)==int(b) for a,b in zip(old['support_prediction'][j],oldy[j])) if valid else None,common_correct=all(int(a)==int(b) for a,b in zip(new['support_prediction'][i],newy[i])) if valid else None)
   reported=ledger[arm,r['base_index'],r['native_slot']];compare(row,reported)
   if dist is not None:max_distance_error=max(max_distance_error,abs(dist-reported['q_shift_image_px']))
   armrows.append(row);computed_rows.append(row)
  assert len(armrows)==1024;take=[r for r in armrows if r['original_correct'] is not None];lost=[r for r in take if r['original_correct'] and not r['common_correct']]
  summary=dict(BEFORE_rows=1024,same_known_truth_and_both_available=len(take),original_correct=sum(r['original_correct'] for r in take),common_correct=sum(r['common_correct'] for r in take),original_correct_became_wrong=len(lost),lost_with_q_shift_le1px=sum(r['q_shift_image_px'] is not None and r['q_shift_image_px']<=1 for r in lost),lost_with_q_shift_le3px=sum(r['q_shift_image_px'] is not None and r['q_shift_image_px']<=3 for r in lost),lost_cases=lost)
  compare(summary,g['arms'][arm]);summaries[arm]={k:v for k,v in summary.items() if k!='lost_cases'}
 # Exact raw-input collision and RGB-change counts use only saved96 arrays; never resample.
 x=load(source['inputs'],('rgb96','valid96')+keys);y=newy;assert x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(4096,96,96,3) and x['valid96'].dtype==bool and x['valid96'].shape==(4096,96,96)
 groups=defaultdict(list)
 for i in range(4096):
  if x['original_query_available'][i]:groups[hashlib.sha256(x['rgb96'][i].tobytes()+x['valid96'][i].tobytes()).hexdigest()].append(i)
 conflicts=[]
 for ii in groups.values():
  # A digest is only an index; verify actual bytes inside each exact-input group.
  for i in ii[1:]:assert same(x['rgb96'][i],x['rgb96'][ii[0]]) and same(x['valid96'][i],x['valid96'][ii[0]])
  for j in range(2):
   if len({int(y[i,j]) for i in ii if int(y[i,j])>=0})>1:conflicts.append(dict(rows=ii,support=j))
 by={(r['base_index'],r['variant'],r['native_slot']):i for i,r in enumerate(rr)};reported_cue_rows=lines(cue['rows']);assert len(reported_cue_rows)==1024
 pair_records=[];flip_counts={name:[] for name in ('visible_end','foreign_front')};available_pairs=identical=0
 for base in range(128):
  for native in range(8):
   a=by[base,'BEFORE',native];b=by[base,'COVER',native];different=0;absolute_sum=0;maximum=0
   # Channel deltas are exact integers. Scalar pixel loops independently count
   # changed spatial pixels; numpy reductions/producer helpers are not used.
   aa=x['rgb96'][a].reshape(-1,3);bb=x['rgb96'][b].reshape(-1,3)
   for pa,pb in zip(aa,bb):
    delta=[abs(int(v)-int(w)) for v,w in zip(pa,pb)];different+=any(delta);absolute_sum+=sum(delta);maximum=max(maximum,max(delta))
   mean=absolute_sum/(96*96*3);available=bool(x['original_query_available'][a]);available_pairs+=available;identical+=different==0
   row=dict(base_index=base,native_slot=native,before_diagnostic_index=a,cover_diagnostic_index=b,original_anchor_available=available,changed_RGB_pixels=different,before_support=y[a].tolist(),cover_support=y[b].tolist(),channel_MAE=mean,channel_max_difference=maximum);compare(row,reported_cue_rows[len(pair_records)]);pair_records.append(row)
   for j,name in enumerate(('visible_end','foreign_front')):
    if int(y[a,j])>=0 and int(y[b,j])>=0 and int(y[a,j])!=int(y[b,j]):flip_counts[name].append((different,mean))
 flips={}
 for name,vals in flip_counts.items():
  count=[v[0] for v in vals];means=[v[1] for v in vals]
  flips[name]=dict(known_truth_flip_pairs=len(vals),exact_same_RGB_with_known_truth_flip=sum(v==0 for v in count),changed_RGB_pixel_count=None if not vals else dict(min=min(count),median=float(np.median(count)),max=max(count)),whole_crop_channel_absolute_difference_mean=None if not vals else dict(min=min(means),median=float(np.median(means)),max=max(means)))
 expected=dict(diagnostic_rows=4096,available_rows=int(sum(bool(v) for v in x['original_query_available'])),distinct_available_RAW_inputs=len(groups),exact_RAW_input_conflicting_known_target_groups=len(conflicts),conflicts=conflicts,BEFORE_COVER_pairs=1024,available_pairs=available_pairs,identical_RGB_pairs=identical,support_truth_flips=flips)
 compare(expected,{k:cue[k] for k in expected})
 report=dict(completed=True,PASS=True,code=bind(__file__),geometry_complete=bind(GEOM),raw_cue_complete=bind(CUE),frozen_predictions=g['frozen_predictions'],all2048_geometry_rows_verified=True,all_old_and_new_score_decoders4096_perarm_verified=True,geometry_summary=summaries,max_q_shift_recompute_absolute_error=max_distance_error,all1024_RGB_change_rows_verified=True,raw_cue_summary=expected,method='No producer/evaluator import. Independent originalglobal-ID join, direct old/new FP32>0 decoder, exact known-target equality/availability and scalar counts. Q shift recomputed by scalar Euclidean formula;1e-12 comparison. RAW input byte groups verified; changed RGB pixels/absolute channel sum/max recomputed by independent scalar pixel loops on saved RGB96 only.',limits=['Geometry diagnostic changes original q and L together. C50 changes crop geometry only; C51 also changes SAM box/mask. Translation-only sensitivity is not isolated.','Zero exact RAW contradictory-known-target groups only rules out that literal contradiction in this saved corpus. All95 flips show whole-crop changes; no proof of intended-end contrast/recognizability follows.','SourceTRAIN only; physical native identity and real transfer remain unproved. No new sampling, scene RGB, models, weights, features, fit, VAL, held or private input.'],new_sampling_forward_fit_or_selection=0,training_admission=False,selected_model=None,fully_solved=False)
 with (H/'REVIEW.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'))))
if __name__=='__main__':main()
