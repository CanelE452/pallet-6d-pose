from features44 import *
METHODS=('base','strong','anchor','frozen_v41',*ARMS)
def masks44(p,hard):
 for split,frame0 in [('fit',p['train']),('calibration',p['calibration']),('check',p['check'])]:
  for cond in ['all',*sorted(set(p['condition']))]:
   frame=frame0&(True if cond=='all' else p['condition']==cond)
   for typ,M in [('all_detected',np.ones_like(hard)),('hard',hard),('easy',~hard),('outside',np.broadcast_to(p['g']['outside'],hard.shape)),('hidden',np.broadcast_to(p['g']['hidden'],hard.shape))]:
    m=M&frame[None,:,None]
    if (m.sum((1,2))>0).all():yield split+'/'+cond+'/'+typ,m

def main():
 assert read(RUN/'TRAINING_COMPLETE.json')['complete'];p=population44();data=[]
 for seed in SEEDS:
  q=dict(np.load(RUN/'arrays'/f'parents_seed{seed}.npz'));c=np.load(RUN/'arrays'/f'corrections_seed{seed}.npz');data.append({k:q[k] for k in METHODS[:4]}|{k:c['points_'+k] for k in ARMS})
 E={m:np.stack([np.linalg.norm(q[m]-p['g']['gt'],axis=-1) for q in data]) for m in METHODS};hard=E['anchor']>=10;results={};comparisons={}
 for key,M in masks44(p,hard):
  row={}
  for m in METHODS:
   z=L.metric(E[m],E['anchor'],M,p['groups']);z['harmed_over1px']=[int(((E[m][i]-E['anchor'][i]>1)&M[i]).sum()) for i in range(3)];z['helped_over1px']=[int(((E['anchor'][i]-E[m][i]>1)&M[i]).sum()) for i in range(3)];row[m]=z
  results[key]=row;comparisons[key]={a+'_vs_'+b:L.metric(E[a],E[b],M,p['groups']) for a,b in [('fresh_line','fresh_point'),('reused_line','reused_point'),('fresh_line','reused_line'),('fresh_point','reused_point')]}
 def gain(r):return min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0
 key='check/all/all_detected';frozen=dict(point_gain=gain(results[key]['frozen_v41']),preservation=all(min(v['frozen_v41']['gain_percent'])>=-1 for k,v in results.items() if k.startswith('check/')));frozen['passed']=all(frozen.values());new=dict(point_gain=gain(results[key][PRIMARY]),incremental_line_gain=gain(comparisons[key]['fresh_line_vs_fresh_point']),preservation=all(min(v[PRIMARY]['gain_percent'])>=-1 for k,v in results.items() if k.startswith('check/')));new['passed']=all(new.values())
 allrows=read(RUN/'cache/manifest.json')['rows'];coverage={}
 for split in ('train','calibration','check'):
  for cond in ['all',*sorted(set(p['condition']))]:
   rr=[r for r in allrows if r['split']==split and (cond=='all' or r['condition']==cond)];coverage[split+'/'+cond]=dict(requested=len(rr),detected=sum(r['detected'] for r in rr),groups=len(set(r['source_group'] for r in rr)))
 write(RUN/'SUMMARY.json',dict(results=results,comparisons=comparisons,coverage=coverage,gates=dict(frozen=frozen,parent_unseen=new),decision='PASS_SCREEN_REQUIRES_SEPARATE_REAL_PROTOCOL' if new['passed'] else 'CLOSE_CURRENT_FROZEN_FEATURE_CORRECTION_FAMILY',scope=read(RUN/'PROTOCOL.json')['scope']))
 write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in (RUN/'SUMMARY.json',KIT/'evaluate44.py')}));print('EVAL44',frozen,new,flush=True);print('POINT44',{k:round(v['mean_px'],4) for k,v in results[key].items()},flush=True)
if __name__=='__main__':main()
