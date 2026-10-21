"""TRAIN-only two-forecast-outlier headroom and measured cost, not a new method."""
import pathlib,sys,importlib.util,time,json,itertools,copy
import numpy as np
import cv2
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;s=importlib.util.spec_from_file_location('j24_export',T/'jackknife_bank_20261021_v24/experiment_v3.py');J=importlib.util.module_from_spec(s);s.loader.exec_module(J);G=J.G;U=J.R/'SYNTHETIC_v3';cv2.setNumThreads(1);torch.set_num_threads(1)
fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');up=G.index(G.rows(G.B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));gt=G.index(G.rows(G.B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'));labels=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));train=[(i,f)for i,f in enumerate(fs)if f['split']=='train'];absent=[(i,f)for i,f in train if not any(a['success_5cm5deg']for a in labels[f['id']]['action_metrics'])];good=[(i,f)for i,f in train if any(a['success_5cm5deg']for a in labels[f['id']]['action_metrics'])];diagnosis=[]
for i,f in train:
 u=up[f['id']];g=gt[f['id']];e=np.linalg.norm(np.asarray(u['initial_points'])[:8]-np.asarray(g['projected_GT_native_xy'])[:8],axis=1);diagnosis.append(dict(id=f['id'],old_bank_good=any(a['success_5cm5deg']for a in labels[f['id']]['action_metrics']),forecast_errors_sorted_px=sorted(e.tolist()),error_count_gt3px=int(sum(e>3)),error_count_le3px=int(sum(e<=3))))
G.write(R/'TRAIN_POINT_ERROR_DIAGNOSIS.json',dict(n=768,diagnostic_threshold_px=3,not_inference_rule=True,all_counts={str(n):sum(d['error_count_gt3px']==n for d in diagnosis)for n in range(9)},absent_counts={str(n):sum(not d['old_bank_good']and d['error_count_gt3px']==n for d in diagnosis)for n in range(9)},independent_real_owner_truth=False,rows=diagnosis));sample=[absent[i]for i in np.linspace(0,len(absent)-1,8,dtype=int)]+[good[i]for i in np.linspace(0,len(good)-1,2,dtype=int)];fn=J.P.load_functions();start=time.monotonic();ps=[];tt=[];c=torch.load(J.R/'s20261018/best.pt',weights_only=False,map_location='cpu');model=G.M.make_model('MLP');model.load_state_dict(c['model_state_dict']);model.eval().requires_grad_(False)
for i,f in sample:
 tick=time.monotonic();u=up[f['id']];xy=np.asarray(u['initial_points'])[:8];valid=np.isfinite(xy).all(-1)&~np.all(xy==-1,axis=1);X=np.asarray(u['native_xyz']);K=np.asarray(u['K']);Q=np.asarray(u['initial_pose']['R_cf']).T@u['initial_pose']['R_physical'];new=[];calls=0
 for dropped in itertools.combinations(np.flatnonzero(valid),2):
  take=valid.copy();take[list(dropped)]=False
  if take.sum()<6:continue
  calls+=1
  try:z=fn['solve'](X,xy,K,take)
  except cv2.error:continue
  if z is None:continue
  rot,t,_=z
  if not np.isfinite(rot).all()or not np.isfinite(t).all()or((X@rot.T+t)[:,2]<=0).any():continue
  new.append(dict(candidate_id='JK2_RAW/'+','.join(map(str,dropped)),source='TWO_FORECAST_OUTLIER',R_physical=(rot@Q).tolist(),centroid=(rot@X.mean(0)+t).tolist()))
 inp,m=J.packet(i,f,G.B/'d1_synthetic_candidates_v1');nf,added=J.export(f,inp,m,new)
 with torch.inference_mode():
  x,es=G.A.inputs(nf,c['normalizer'],'H1');k=int(model(torch.from_numpy(x)).flatten().argmax());bx,be=G.A.inputs(f,c['normalizer'],'H1');bk=int(model(torch.from_numpy(bx)).flatten().argmax())
 ps.append(dict(id=f['id'],split='train',targeted_absence_case=any(q['id']==f['id']and not q['old_bank_good']for q in diagnosis),old_selected=be[bk],frozen_H1_expanded_selected=es[k],new_candidates=added));tt.append(dict(id=f['id'],old_actions=len(be),new_proposals=len(new),new_eligible=len(es)-len(be),PnP_pairs=calls,seconds=time.monotonic()-tick));print(json.dumps(tt[-1]),flush=True);assert time.monotonic()-start<180,'pilot cap exceeded stop before fullrun'
G.save(R/'SEALED_TRAIN_PILOT_PREDICTIONS.jsonl.gz',ps);G.csvwrite(R/'PILOT_COST.csv',tt);G.write(R/'PREDICTION_SEAL.json',dict(status='SEALED_BEFORE_METRICS',sha256=G.sha(R/'SEALED_TRAIN_PILOT_PREDICTIONS.jsonl.gz'),pilot_seconds=time.monotonic()-start,new_PnP_pairs=sum(t['PnP_pairs']for t in tt),new_H1fits=0,newRGB=0,newGPU=0,subset_selection_uses_noGT=True,TRAIN_sample_selection_targeted_not_randomVAL=True))
# Reference metric evaluation only after proposal bank has been serialized/sealed.
out=[]
for p in ps:
 g=gt[p['id']];ref=dict(R=g['R_fixed'],t=g['t'],order=2);ms=[G.metric(a['actual_pose'],ref)for a in p['new_candidates']if a['eligible']];a=G.metric(p['old_selected']['actual_pose'],ref);b=G.metric(p['frozen_H1_expanded_selected']['actual_pose'],ref);out.append(dict(id=p['id'],targeted_absence=p['targeted_absence_case'],new_eligible_good=sum(m['success']for m in ms),new_bank_good=any(m['success']for m in ms),before_success=a['success'],frozen_distribution_shift_success=b['success'],before_T=a['translation_cm'],after_T=b['translation_cm'],before_R=a['rotation_deg'],after_R=b['rotation_deg']))
G.csvwrite(R/'PILOT_DIAGNOSTIC_METRICS.csv',out);estimate=float(np.mean([t['seconds']for t in tt]))*1003;G.write(R/'DECISION.json',dict(status='PENDING_LARGER_CPU_SCOPE'if estimate>900 else'PILOT_ONLY_NO_ADOPTION',pilot_seconds=sum(t['seconds']for t in tt),estimated_all1003_export_seconds=estimate,estimated_all1003_export_minutes=estimate/60,above_fixed900second_minimal_export_budget=estimate>900,absence8_new_good=sum(p['targeted_absence']and p['new_bank_good']for p in out),control2_damage=sum(not p['targeted_absence']and p['before_success']and not p['frozen_distribution_shift_success']for p in out),frozen_H1_sensitivity_not_retrained_system=True,new_full_training=0,new_VAL=0,new_CAL=0,new_PRIMARY=0,needed='scope decision for longer one-core export+same188 H1 fit only if pilotheadroom justified; do not launch automatically',sampling_limit='8 targetedTRAINabsence+2 goodTRAINcontrols; not fullVALperformance or coverage extrapolation'));print(G.read(R/'DECISION.json'))
