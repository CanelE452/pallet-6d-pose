"""Matched original H1 recipe on a properly re-exported forecast-jackknife union bank."""
import pathlib,sys,argparse,copy,json,time,gzip,math,os
import numpy as np
import torch
import cv2
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sys.path[:0]=[str(T/'geometric_influence_20261021_v19'),str(T/'bottleneck_repair_20261021_v14')];import experiment as G;import track2_gamma_proposal as P
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S
CONF=pathlib.Path('${DATA_ROOT}/pallet-statewise-end-to-end-recovery-v1/CONFIG_FROZEN.json')
RECONSTRUCTION_FALLBACKS=[]
def bound():
 p=G.read(R/'PROTOCOL_V3.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def bank0(f):
 z=[]
 for c in f['candidates']:
  p=c['actual_pose'];aliases=c.get('source_baseline_aliases',[]);z.append(dict(candidate_id=c['candidate_id'],source='N0'if'N0'in aliases else'E0'if'E0'in aliases else'OTHER',R_physical=p['R_physical'],centroid=p['centroid']))
 return z
def packet(i,f,root):
 folder=root/'frames'/f'{i:04d}';seal=G.read(folder/'FRAME_SEAL.json');assert G.sha(folder/'E_RUN.jsonl.gz')==seal['e_run_sha256']and G.sha(folder/'BOUNDARY_BUNDLE.jsonl.gz')==seal['boundary_bundle_sha256'];run=G.rows(folder/'E_RUN.jsonl.gz')[0];assert run['id']==f['id'];bundle=G.rows(folder/'BOUNDARY_BUNDLE.jsonl.gz')[0];inp=dict(packet=run['packet'],families=run['families'],fallback=f['fallback']);return inp,S.compile_frame(inp,G.read(CONF),bundle=bundle)
def proposals(u,fn):
 X=np.asarray(u['native_xyz']);K=np.asarray(u['K']);xy=np.asarray(u['initial_points'])[:8];valid=np.isfinite(xy).all(-1)&~np.all(xy==-1,axis=1);Q=np.asarray(u['initial_pose']['R_cf']).T@u['initial_pose']['R_physical'];new=[]
 for k in np.flatnonzero(valid):
  take=valid.copy();take[k]=False
  if take.sum()<6:continue
  try:z=fn['solve'](X,xy,K,take)
  except cv2.error:continue
  if z is None:continue
  rot,t,res=z
  if not np.isfinite(rot).all()or not np.isfinite(t).all()or ((X@rot.T+t)[:,2]<=0).any():continue
  new.append(dict(candidate_id='JK_RAW_FORECAST/'+str(int(k)),source='FORECAST_JACKKNIFE',R_physical=(rot@Q).tolist(),centroid=(rot@X.mean(0)+t).tolist()))
 return new
def inverse(x):return np.sign(x)*np.expm1(abs(x))
def old_minima(f,models):
 result={};exact=None
 for j,mode in enumerate(['fixed','A','B','AB','edge']):
  estimates=[inverse(c['features'][j])-inverse(c['features'][j+26])for c in f['candidates']if c['features'][31+j]and c['features'][57+j]]
  if estimates:
   spread=max(estimates)-min(estimates)
   if spread>1e-5*max(1,abs(np.median(estimates))):
    if exact is None:
     exact=[S.candidate_scores(c,models)for c in bank0(f)];RECONSTRUCTION_FALLBACKS.append(dict(id=f['id'],reason='signedlog_inverse_cancellation',trigger_mode=mode,spread=float(spread)));print(json.dumps(dict(stage='EXACT_MINIMUM_FALLBACK',**RECONSTRUCTION_FALLBACKS[-1])),flush=True)
    vv=[ss[family]['scores'][mode]for ss in exact for family in models if S._finite_feature(ss[family]['scores'][mode])is not None];result[mode]=min(vv)if vv else None
   else:result[mode]=float(np.median(estimates))
  else:result[mode]=None
 return result
def export(f,inp,models,new):
 old=bank0(f);scale=f['object_scale_m'];new=[c for c in S.unique_poses(old+new,scale)if c['candidate_id']not in {x['candidate_id']for x in old}];scores=[S.candidate_scores(c,models)for c in new];nf=S.features_for_bank(inp,models,new,scores);minima=old_minima(f,models)
 for j,mode in enumerate(['fixed','A','B','AB','edge']):
  vv=[s[family]['scores'][mode]for s in scores for family in models if S._finite_feature(s[family]['scores'][mode])is not None]
  if minima[mode]is not None:vv.append(minima[mode])
  minima[mode]=min(vv)if vv else None
 out=copy.deepcopy(f);out['candidates']+=nf
 for c in out['candidates']:
  for j,mode in enumerate(['fixed','A','B','AB','edge']):
   if minima[mode]is not None and c['features'][31+j]:
    gap=max(0.,float(inverse(c['features'][j])-minima[mode]));c['features'][j+26]=float(np.log1p(gap));c['features'][57+j]=1.
   else:c['features'][j+26]=0.;c['features'][57+j]=0.
 G.A.validate_frame(out);return out,nf
def pilot():
 bound();G.firewall();cv2.setNumThreads(1);fs=G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz');train=[i for i,f in enumerate(fs)if f['split']=='train'];ix=[train[j]for j in np.linspace(0,len(train)-1,8,dtype=int)];up=G.index(G.rows(G.B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));fn=P.load_functions();start=time.monotonic();rr=[]
 for i in ix:
  f=fs[i];tick=time.monotonic();inp,m=packet(i,f,G.B/'d1_synthetic_candidates_v1');old=bank0(f);scores=[S.candidate_scores(c,m)for c in old];full=S.features_for_bank(inp,m,old,scores);by={c['candidate_id']:c for c in full};errors=[]
  for c in f['candidates']:errors.append(float(np.max(abs(np.asarray(c['features'])-by[c['candidate_id']]['features']))));assert c['eligible']==by[c['candidate_id']]['eligible']
  maximum=max(errors);assert maximum<1e-5,('OLD_EXPORT_PARITY_FAILED',f['id'],maximum)
  nf,added=export(f,inp,m,proposals(up[f['id']],fn));rr.append(dict(id=f['id'],old_candidates=len(old),old_export_max_error=maximum,new_proposals=len(added),new_eligible=sum(c['eligible']for c in added),seconds=time.monotonic()-tick));print(json.dumps(rr[-1]),flush=True)
 G.csvwrite(R/'TECHNICAL_PILOT8.csv',rr);G.write(R/'TECHNICAL_PILOT.json',dict(status='PASS',n=8,seconds=time.monotonic()-start,estimated_all1003_seconds=float(np.mean([x['seconds']for x in rr]))*1003,new_H1fit=0,new_gt_reads=0,full_old_export_verified_frames=8,not_all1003_equivalence_proof=True));print(G.read(R/'TECHNICAL_PILOT.json'))
def features(pop):
 bound();assert G.read(R/'TECHNICAL_PILOT.json')['status']=='PASS';G.firewall();cv2.setNumThreads(1);root=G.B/('d1_synthetic_candidates_v1'if pop=='SYNTHETIC'else'd1_real_candidates_v1');fs=G.rows(root/'HEAD_FEATURES.jsonl.gz');up=G.index(G.rows(G.B/'d1_upstream_v1'/('SYNTHETIC1003_UPSTREAM.jsonl.gz'if pop=='SYNTHETIC'else'REAL238_UPSTREAM.jsonl.gz')));out=R/(pop+'_v3');out.mkdir(exist_ok=False);fn=P.load_functions();start=time.monotonic();results=[];timings=[];val=G.index(G.rows(T/'forecast_jackknife_20261021_v23/SYNTHETIC_VAL/PREDICTIONS.jsonl.gz'))if pop=='SYNTHETIC'else None;new_calls=0
 for i,f in enumerate(fs):
  if pop!='SYNTHETIC'and f['partition']!='CAL':continue
  tick=time.monotonic();inp,m=packet(i,f,root)
  if val is not None and f['id']in val:
   new=[dict(candidate_id='JK_RAW_FORECAST/'+c['candidate_id'].split('_')[-1],source='FORECAST_JACKKNIFE',R_physical=c['pose']['R_physical'],centroid=c['pose']['centroid'])for c in val[f['id']]['candidates'][2:]]
  else:new=proposals(up[f['id']],fn);new_calls+=8
  exported,added=export(f,inp,m,new);results.append(exported);timings.append(dict(id=f['id'],split=f['split'],original_actions=len(G.A.actions(f)),new_proposals=len(added),new_eligible=sum(c['eligible']for c in added),union_actions=len(G.A.actions(exported)),seconds=time.monotonic()-tick))
  assert time.monotonic()-start<900,'900second full-export cap'
  if len(results)%50==0:print(json.dumps(dict(stage='UNION_FEATURES',done=len(results),total=len(fs),seconds=time.monotonic()-start)),flush=True)
 G.save(out/'HEAD_FEATURES.jsonl.gz',results);G.csvwrite(out/'TIMING.csv',timings)
 if pop=='SYNTHETIC':G.write(out/'NORMALIZER.json',G.A.fit_normalizer([f for f in results if f['split']=='train']))
 G.write(out/'SEAL.json',dict(rows=len(results),features_sha256=G.sha(out/'HEAD_FEATURES.jsonl.gz'),seconds=time.monotonic()-start,new_forecast_PnP_pairs_upper_bound=new_calls,reused_VAL_proposals=True,new_RGB=0,new_SAM=0,full_bank_pooling_recomputed_at_model_input=True,old_minimum_reconstructed_from_signedlog_score_gap=True,exact_minimum_fallbacks=RECONSTRUCTION_FALLBACKS));print(dict(stage='FEATURES_SEALED',n=len(results),seconds=time.monotonic()-start))
def targets():
 start=time.monotonic();fs=G.rows(R/'SYNTHETIC_v3/HEAD_FEATURES.jsonl.gz');gt=G.index(G.rows(G.B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'));old=G.index(G.rows(G.V5/'DERIVED_C2_ACTION_TARGETS.jsonl.gz'));labs=[];coverage=[]
 for f in fs:
  g=gt[f['id']];ref=dict(R=g['R_fixed'],t=g['t'],order=2);ms=[]
  for c in G.A.actions(f):
   m=G.metric(c['actual_pose'],ref);ms.append(dict(candidate_id=c['candidate_id'],translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],success_5cm5deg=m['success']))
  labs.append(dict(id=f['id'],split=f['split'],action_metrics=ms));by={x['candidate_id']:x for x in ms};og=any(c['success_5cm5deg']for c in old[f['id']]['action_metrics']);good=any(x['success_5cm5deg']for x in ms);coverage.append(dict(id=f['id'],split=f['split'],original_bank_good=og,eligible_union_good=good,new_good_without_original=good and not og))
 G.save(R/'SYNTHETIC_v3/C2_TARGETS.jsonl.gz',labs);G.csvwrite(R/'SYNTHETIC_v3/COVERAGE.csv',coverage);counts={s:dict(n=sum(c['split']==s for c in coverage),original_good=sum(c['original_bank_good']for c in coverage if c['split']==s),union_good=sum(c['eligible_union_good']for c in coverage if c['split']==s),new_good=sum(c['new_good_without_original']for c in coverage if c['split']==s))for s in ['train','validation']};G.write(R/'SYNTHETIC_v3/TARGET_SEAL.json',dict(counts=counts,seconds=time.monotonic()-start,sha256=G.sha(R/'SYNTHETIC_v3/C2_TARGETS.jsonl.gz'),GT_used_for_target_only=True));print(json.dumps(counts,indent=2))
def cost():
 bound();G.firewall();cv2.setNumThreads(1);fs=G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz');by={f['id']:(i,f)for i,f in enumerate(fs)};ids=[x['id']for x in __import__('csv').DictReader((R/'TECHNICAL_PILOT8.csv').open())];up=G.index(G.rows(G.B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));fn=P.load_functions();rr=[]
 for ident in ids:
  i,f=by[ident];tick=time.monotonic();inp,m=packet(i,f,G.B/'d1_synthetic_candidates_v1');nf,added=export(f,inp,m,proposals(up[ident],fn));rr.append(dict(id=ident,seconds=time.monotonic()-tick,new_proposals=len(added),new_eligible=sum(c['eligible']for c in added)));print(json.dumps(rr[-1]),flush=True)
 G.csvwrite(R/'INCREMENTAL_COST8.csv',rr);G.write(R/'INCREMENTAL_COST.json',dict(status='PASS',timing_scope='actual repeated union compile/new8cost/export/PnP only; excludes one-time fulloldexport parity audit',estimated_all1003_seconds=float(np.mean([x['seconds']for x in rr]))*1003,CPU_threads=1,pilot_PnP_pairs_upper_bound=64,old_parity_not_rerun=True));print(G.read(R/'INCREMENTAL_COST.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['cost','features','targets']);p.add_argument('--population',default='SYNTHETIC');a=p.parse_args();cost()if a.stage=='cost'else features(a.population)if a.stage=='features'else targets()
