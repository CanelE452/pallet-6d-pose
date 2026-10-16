from context36 import *
from analyze35 import ci,INC

def main():
    meta=read(V35.RUN/'POPULATION.json')['rows'];g=np.load(V35.RUN/'geometry.npz');ref=np.load(V35.RUN/'analysis_seed1.npz');roles=ref['selected_role'];supported=ref['supported'];N=len(meta);groups=np.array([r['source_group'] for r in meta]);train=np.array([r['split']=='train' for r in meta]);cond=np.array([r['condition'] for r in meta]);bb=np.array([r['box'] for r in meta]);wh=np.maximum(bb[:,2:]-bb[:,:2],1);diag=np.linalg.norm(wh,axis=-1);center=(bb[:,:2]+bb[:,2:])/2;lookup={r['id']:i for i,r in enumerate(meta)};distances={};best={};nulls={};reachable={};scores={}
    for seed in SEEDS:
        for arm in ARMS:
            pn=np.zeros((N,12,8,2));pr=np.zeros((N,12,8));ps=np.zeros_like(pr);attention=np.zeros((N,8,97))
            paths=[RUN/'training_predictions'/f'{arm}_seed{seed}.npz']+list((RUN/'predictions/render_check').glob(f'*/{arm}_seed{seed}.npz'))
            for path in paths:
                z=np.load(path);take=z['detected'] if 'detected' in z else np.ones(len(z['ids']),bool);ix=np.array([lookup[i] for i in z['ids'][take]]);pn[ix]=z['normal'][take].reshape(-1,12,8,2);pr[ix]=z['offset'][take].reshape(-1,12,8)*diag[ix,None,None]+(pn[ix]*center[ix,None,None]).sum(-1);ps[ix]=z['score'][take].reshape(-1,12,8);attention[ix]=z['attention'][take].reshape(-1,8,2,97).mean(2)
            d=np.abs((pn[:,:,:,None]*g['support'][:,:,None]).sum(-1)-pr[:,:,:,None]).mean(-1);chosen=ps.argmax(-1);top=d[np.arange(N)[:,None],np.arange(12)[None],chosen];distances[f'{arm}_{seed}']=top[np.arange(N)[:,None],roles];best[f'{arm}_{seed}']=d.min(-1)[np.arange(N)[:,None],roles];nulls[f'{arm}_{seed}']=attention[...,-1];scores[f'{arm}_{seed}']=ps.max(-1)
        # Reachability of frozen grid candidates under allowed one-bin refinement.
        cache=torch.load(RUN/'cache'/f'training_seed{seed}.pt',weights_only=True,map_location='cpu');sources=[(cache['ids'],cache['x'])]
        for c in sorted(set(cond)):
            data=torch.load(RUN/'cache/render_check'/c/f'seed{seed}.pt',weights_only=True,map_location='cpu');sources.append((data['ids'],data['x']))
        reach=np.zeros((N,12),bool)
        for ids,x in sources:
            for j,name in enumerate(ids):
                if name not in lookup:continue
                i=lookup[name];normal=g['normal'][i]*wh[i]/2;length=np.linalg.norm(normal,axis=-1);angle=np.arctan2(normal[:,1],normal[:,0]);rho=(g['rho'][i]-(g['normal'][i]*center[i]).sum(-1))/np.maximum(length,1e-8);theta=x['theta'][j].numpy().reshape(12,8);r=x['rho'][j].numpy().reshape(12,8);diff=angle[:,None]-theta;sign=np.where(np.cos(diff)<0,-1,1);da=(diff+np.pi/2)%np.pi-np.pi/2;dr=rho[:,None]*sign-r;reach[i]=((np.abs(da)<=np.pi/24)&(np.abs(dr)<=2*np.sqrt(2)/47)).any(-1)&(g['length'][i]>0)
        reachable[seed]=reach[np.arange(N)[:,None],roles]
    results={}
    for split,take in [('train',train),('check',~train)]:
      for c in ['all',*sorted(set(cond))]:
        mask=take[:,None]&(True if c=='all' else (cond==c)[:,None])&supported
        if not mask.any():continue
        row={}
        for arm in ARMS:
            values=np.stack([distances[f'{arm}_{s}'] for s in SEEDS]);optimal=np.stack([best[f'{arm}_{s}'] for s in SEEDS]);baseline=np.stack([distances[f'grid_{s}'] for s in SEEDS]);bestbase=np.stack([best[f'grid_{s}'] for s in SEEDS]);per=values[:,mask].mean(-1);pb=baseline[:,mask].mean(-1);perbest=optimal[:,mask].mean(-1);pbb=bestbase[:,mask].mean(-1);row[arm]=dict(mean_support_distance_px=float(per.mean()),per_seed_px=per.tolist(),per_seed_gain_percent=((1-per/pb)*100).tolist(),difference=ci(values-baseline,mask,groups),best8_distance_px=float(perbest.mean()),best8_per_seed_gain_percent=((1-perbest/pbb)*100).tolist(),best8_difference=ci(optimal-bestbase,mask,groups),selected_within5px=float((values[:,mask]<=5).mean()),null_attention=float(np.stack([nulls[f'{arm}_{s}'] for s in SEEDS])[:,mask].mean()))
        results[split+'/'+c]=dict(points=int(mask.sum()),reachable_fraction=float(np.stack([reachable[s] for s in SEEDS])[:,mask].mean()),methods=row)
    r=results['check/all']['methods']['both'];gate=min(r['per_seed_gain_percent'])>=10 and r['difference']['ci95'][1]<0;potential=min(r['best8_per_seed_gain_percent'])>=10 and r['best8_difference']['ci95'][1]<0;write(RUN/'LINE_SUMMARY.json',dict(complete=True,results=results,actual_line_improved=bool(gate),candidate_geometry_improved=bool(potential),scope='Matched longest visible physical-role support, same V35 supported points. GT used only scoring. Best8 is a physical-line oracle; not deployed selection.'));np.savez_compressed(RUN/'LINE_ERRORS.npz',supported=supported,roles=roles,**{'top_'+k:v for k,v in distances.items()},**{'best_'+k:v for k,v in best.items()} ,**{'null_'+k:v for k,v in nulls.items()} )
    point=read(RUN/'POINT_SUMMARY.json');write(RUN/'SUMMARY.json',dict(complete=True,models=12,updates=24000,line_improved=bool(gate),candidate_geometry_improved=bool(potential),point_stable=point['point_stable'],accepted=bool(gate and point['point_stable']),point_sha256=sha(RUN/'POINT_SUMMARY.json'),line_sha256=sha(RUN/'LINE_SUMMARY.json')));print('LINE36',gate,potential,'ACCEPTED',read(RUN/'SUMMARY.json')['accepted'],flush=True)
if __name__=='__main__':main()
