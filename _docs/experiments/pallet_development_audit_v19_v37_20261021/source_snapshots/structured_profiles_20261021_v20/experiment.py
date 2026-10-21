"""Retain existing ordered B-profile evidence in both original family contexts."""
import sys,pathlib,json,gzip,time,argparse
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T/'geometric_influence_20261021_v19'))
import experiment as G
G.R=R
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import boundary_model as BM
FAMILIES=G.FAMILIES;CHANNELS=['normal_edge_sine','intersection_offset_grid_fraction','outside_grid_fraction','original_profile_energy','signed_offset_from_global_peak_fraction'];D=2*8*3*4*len(CHANNELS)
def model(arm,initial=None):
 n=188 if arm=='CONTROL'else 188+2*D;m=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial is not None:
  state={k:v.clone()for k,v in initial.items()}
  if arm!='CONTROL':state['0.weight']=torch.cat([state['0.weight'],torch.zeros((64,2*D))],dim=1)
  m.load_state_dict(state)
 return m
def extend(frame,norm,extra,en,arm):
 x,es=G.A.inputs(frame,norm,'H1')
 if arm=='CONTROL':return x,es
 by={a['candidate_id']:a for a in extra['actions']};raw=np.full((len(es),D),np.nan)
 for i,e in enumerate(es):a=by[e['candidate_id']];raw[i,a['indices']]=a['values']
 valid=np.isfinite(raw);z=np.sign(raw)*np.log1p(abs(raw));v=np.where(valid&np.asarray(en['seen'])[None,:],(z-en['mean'])/en['scale'],0.);return np.concatenate([x,v,valid.astype(float)],axis=1).astype(np.float32),es
G.model=model;G.extend=extend
def ordered_queries(bundle,family):
 xyz=np.asarray(family['native_xyz']);along=list(bundle['config']['along_base_px']);assert len(along)==4;unique={};cfg=BM._config(bundle['config'])
 for group in bundle['groups']:
  if not group['supported']:continue
  for q in group['queries']:
   if not q['supported']:continue
   k,m=q['native_corner'],q['native_neighbor'];axis=int(np.argmax(abs(xyz[m]-xyz[k])));slot=min(range(4),key=lambda j:abs(along[j]-q['along_base_px']));assert abs(along[slot]-q['along_base_px'])<1e-6
   key=(k,axis,slot)
   if key in unique:assert unique[key]['support_id']==q['support_id']
   unique[key]=q
 return [(k,axis,s,q)for(k,axis,s),q in sorted(unique.items())],cfg
def feature_values(pose,run,bundle,scale,compiled):
 indices=[];values=[]
 for fi,family in enumerate(FAMILIES):
  qproj,_,_=G.project(pose,run['families'][family],np.asarray(run['packet']['solver_input']['K']),scale);queries,cfg=compiled[family]
  for k,axis,slot,q in queries:
   n=np.asarray(q['outward_normal_xy']);edge=qproj[q['native_neighbor']]-qproj[k];length=np.linalg.norm(edge);sine=abs(n[0]*edge[1]-n[1]*edge[0])/length if length>1e-10 else 0.;offset,_=BM._intersection_offset_and_gradient(q,qproj,bundle['pixel_scale']);energy,_=BM._profile_energy_and_derivative(q,offset,'region',cfg);grid=np.asarray(q['offset_grid_base_px']);extent=float(max(abs(grid)));peak=float(grid[int(np.argmax(q['offset_log_probability']))]);vv=[float(sine),None if offset is None else float(offset/extent),None if offset is None else float(abs(offset-np.clip(offset,grid[0],grid[-1]))/extent),float(energy),None if offset is None else float((offset-peak)/extent)]
   start=((((fi*8+k)*3+axis)*4+slot)*len(CHANNELS))
   for j,v in enumerate(vv):
    if v is not None:assert np.isfinite(v);indices.append(start+j);values.append(v)
 assert len(indices)==len(set(indices));return indices,values
def features(pop):
 G.bound();G.firewall();start=time.monotonic();syn=pop=='SYNTHETIC';root=G.B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1');fs=G.rows(root/'HEAD_FEATURES.jsonl.gz');out=R/pop;out.mkdir(exist_ok=False);zs=[];times=[];count=np.zeros(D,int);total=np.zeros(D);squared=np.zeros(D);actions=0;profile_frames=0
 for i,f in enumerate(fs):
  if not syn and f['partition']!='CAL':continue
  tick=time.monotonic();folder=root/'frames'/f'{i:04d}';seal=G.read(folder/'FRAME_SEAL.json');p=folder/'E_RUN.jsonl.gz';bp=folder/'BOUNDARY_BUNDLE.jsonl.gz';assert G.sha(p)==seal['e_run_sha256']and G.sha(bp)==seal['boundary_bundle_sha256'];run=G.rows(p)[0];bundle=G.rows(bp)[0];assert run['id']==f['id'];compiled={name:ordered_queries(bundle,run['families'][name])for name in FAMILIES};aa=[];profile_frames+=bool(compiled[FAMILIES[0]][0])
  for e in G.A.actions(f):
   ix,vs=feature_values(e['actual_pose'],run,bundle,f['object_scale_m'],compiled);aa.append(dict(candidate_id=e['candidate_id'],indices=ix,values=vs));actions+=1
   if syn and f['split']=='train':
    z=np.sign(vs)*np.log1p(np.abs(vs));count[ix]+=1;total[ix]+=z;squared[ix]+=z*z
  zs.append(dict(id=f['id'],split=f['split'],partition=f['partition'],actions=aa,e_run_sha256=seal['e_run_sha256'],boundary_bundle_sha256=seal['boundary_bundle_sha256']));times.append(dict(id=f['id'],candidates=len(aa),supported_profiles=len(compiled[FAMILIES[0]][0]),seconds=time.monotonic()-tick))
  if(i+1)%100==0:print(json.dumps(dict(stage='ORDERED_EXISTING_PROFILES',done=i+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
 G.save(out/'FEATURES.jsonl.gz',zs);G.csvwrite(out/'TIMING.csv',times)
 if syn:
  mu=np.divide(total,count,out=np.zeros(D),where=count>0);variance=np.maximum(0,np.divide(squared,count,out=np.zeros(D),where=count>0)-mu*mu);sd=np.sqrt(variance);sd[sd<1e-12]=1.;G.write(out/'NORMALIZER.json',dict(mean=mu.tolist(),scale=sd.tolist(),seen=(count>0).tolist(),counts=count.tolist(),scope='synthetic TRAIN eligible actions only, each fixed indexed profile channel; missing has validity0',dimension=D,index_order='family ORIGINAL/SWAPPED, native_corner0..7, native_incident_axis0..2, along four original config positions, channel five',channels=CHANNELS))
 G.write(out/'SEAL.json',dict(rows=len(zs),actions=actions,profile_frames=profile_frames,features_sha256=G.sha(out/'FEATURES.jsonl.gz'),source_sha256=G.sha(__file__),seconds=time.monotonic()-start,new_PnP=0,new_RGB_queries=0,new_SAM=0,new_model_forward=0,existing_profiles_reused=True,physical_owner='UNVERIFIED',new_scalar_dimensions=D));print(dict(stage='PROFILE_FEATURES_DONE',population=pop,n=len(zs),profile_frames=profile_frames,actions=actions,seconds=time.monotonic()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();features(a.population)if a.stage=='features'else G.train(a.seed,'INFLUENCE')if a.stage=='train'else(G.predict if a.stage=='predict'else G.evaluate)(a.population,a.seed)
