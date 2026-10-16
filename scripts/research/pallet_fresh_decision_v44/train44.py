from context44 import *
L=legacy()
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle

def fit44(data,mode,seed):
 X=data[mode];take=data['fit'];y=(data['gt']-data['anchor'])/data['diag'][:,None,None];xx=X[take].reshape(-1,X.shape[-1]);w=np.repeat(L.frame_weights(data['groups'],take),8);models=[]
 for j in (0,1):
  m=HistGradientBoostingRegressor(loss='absolute_error',random_state=44000+seed,**PARAMS);m.fit(xx,y[take,:,j].reshape(-1),sample_weight=w);models.append(m)
 return models

def predict44(models,X,anchor,diag):
 d=np.stack([m.predict(X.reshape(-1,X.shape[-1])).reshape(X.shape[:-1]) for m in models],-1);d=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True)/.15,1);return anchor+d*diag[:,None,None]

def main():
 from features44 import population44
 p=population44();old=L.population();(RUN/'models').mkdir(exist_ok=True)
 write(RUN/'TRAIN_SOURCE_BINDINGS.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),files={str(q.relative_to(ROOT)):sha(q) for q in (KIT/'context44.py',KIT/'features44.py',KIT/'train44.py')}));count=0
 for seed in SEEDS:
  fresh=dict(np.load(RUN/'features'/f'fresh_seed{seed}.npz'));reused=dict(np.load(RUN/'features'/f'reused_seed{seed}.npz'));fresh['diag']=p['diag'];reused['diag']=old['diag'];out={};bundle={}
  for arm in ARMS:
   origin,mode=arm.split('_');data=fresh if origin=='fresh' else reused;models=fit44(data,mode,seed);bundle[arm]=models;out['points_'+arm]=predict44(models,fresh[mode],fresh['anchor'],fresh['diag']);out['train_points_'+arm]=predict44(models,data[mode],data['anchor'],data['diag']);count+=2;print('FIT44',seed,arm,'models',count,flush=True)
  (RUN/'models'/f'seed{seed}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(RUN/'arrays'/f'corrections_seed{seed}.npz',**out)
 write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=count,boosting_rounds=count*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in [*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('corrections*'))]}))
if __name__=='__main__':main()
