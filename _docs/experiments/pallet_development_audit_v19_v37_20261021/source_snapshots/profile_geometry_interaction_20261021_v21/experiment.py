"""Single prespecified joint profile/geometry evidence experiment, synthetic labels only."""
import pathlib,sys,importlib.util,argparse,numpy as np,torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
spec=importlib.util.spec_from_file_location('profile_runtime',T/'structured_profiles_20261021_v20/experiment.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);G=S.G;G.R=R
GEO=None;GN=None
def extend(frame,norm,extra,en,arm):
 global GEO,GN
 x,es=S.extend(frame,norm,extra,en,arm)
 if arm=='CONTROL':return x,es
 if GEO is None:GEO=G.index(G.rows(T/'geometric_influence_20261021_v19/SYNTHETIC/FEATURES.jsonl.gz'));GN=G.read(T/'geometric_influence_20261021_v19/SYNTHETIC/NORMALIZER.json')
 if frame['id']not in GEO:GEO.update(G.index(G.rows(T/'geometric_influence_20261021_v19/CAL46/FEATURES.jsonl.gz')))
 by={a['candidate_id']:a['values']for a in GEO[frame['id']]['actions']};raw=np.array([by[e['candidate_id']]for e in es],float);valid=np.isfinite(raw);z=np.sign(raw)*np.log1p(abs(raw));v=np.where(valid&np.array(GN['seen'])[None,:],(z-GN['mean'])/GN['scale'],0.);return np.concatenate([x,v,valid.astype(float)],axis=1).astype(np.float32),es
def model(arm,initial=None):
 n=188 if arm=='CONTROL'else 2156;m=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial is not None:
  state={k:v.clone()for k,v in initial.items()}
  if arm!='CONTROL':state['0.weight']=torch.cat([state['0.weight'],torch.zeros((64,1968))],dim=1)
  m.load_state_dict(state)
 return m
G.extend=extend;G.model=model
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC_VAL');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();G.train(a.seed,'INFLUENCE')if a.stage=='train'else(G.predict if a.stage=='predict'else G.evaluate)(a.population,a.seed)
