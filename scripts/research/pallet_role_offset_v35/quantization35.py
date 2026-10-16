"""Best representable Hough bin by GT support distance: geometry oracle only."""
from context35 import *
from analyze35 import INC,ci

def main():
    write(RUN/'QUANTIZATION_PROTOCOL.json',dict(exploratory=True,reason='Predicted physical support distance remained large with robust offsets. Separate finite24x48grid representation from learned candidate extraction.',choice='Enumerate all1152bins per observed physical role, minimize mean absolute distance of physical support endpoints. No virtual point GT used for bin choice. Uses GT physical lines/role; not inference.',source_sha256=sha(KIT/'quantization35.py')))
    rows=read(RUN/'POPULATION.json')['rows'];g=np.load(RUN/'geometry.npz');N=len(rows);bb=np.array([r['box'] for r in rows]);half=np.maximum(bb[:,2:]-bb[:,:2],1)/2;center=(bb[:,:2]+bb[:,2:])/2;diag=np.linalg.norm(half*2,axis=-1);n=np.zeros((N,12,2));rho=np.zeros((N,12));dist=np.zeros((N,12));index=np.zeros((N,12),int);theta=np.repeat(np.arange(24)*np.pi/24,48);rh=np.tile(np.arange(48)*2*np.sqrt(2)/47-np.sqrt(2),24)
    for i in range(N):
        raw=np.stack((np.cos(theta),np.sin(theta)),-1)/half[i];ll=np.linalg.norm(raw,axis=-1);norm=raw/ll[:,None];offset=rh/ll+norm@center[i]
        for r in np.where(g['length'][i]>0)[0]:
            d=np.abs(norm@g['support'][i,r].T-offset[:,None]).mean(-1);k=d.argmin();n[i,r]=norm[k];rho[i,r]=offset[k];dist[i,r]=d[k];index[i,r]=k
    models=read(RUN/'OFFSET_CALIBRATION.json')['models'];allerr=[]
    for seed in SEEDS:
        a=np.load(RUN/f'analysis_seed{seed}.npz');p=a['base'].copy();roles=a['selected_role'];mask=a['supported']
        for j in range(8):
            nn=n[np.arange(N),roles[:,j]];rr=rho[np.arange(N),roles[:,j]];b=np.array([models[f'{j}_{r}']['median'] for r in roles[:,j]])*diag;amount=rr-(nn*p[:,j]).sum(-1)+b;p[mask[:,j],j]+=nn[mask[:,j]]*amount[mask[:,j],None]
        error=np.linalg.norm(p-g['gt'],axis=-1);allerr.append(error);np.savez_compressed(RUN/f'quantization_seed{seed}.npz',points=p,error=error)
    allerr=np.array(allerr);baseerr=np.stack([np.load(RUN/f'analysis_seed{s}.npz')['errors'][0] for s in SEEDS]);train=np.array([r['split']=='train' for r in rows]);cond=np.array([r['condition'] for r in rows]);groups=np.array([r['source_group'] for r in rows]);selected=dist[np.arange(N)[:,None],roles];results={}
    for split,take in [('train',train),('check',~train)]:
      for c in ['all',*sorted(set(cond))]:
        frame=take&(True if c=='all' else cond==c);m=frame[:,None]&mask
        if not m.any():continue
        per=allerr[:,m].mean(-1);results[f'{split}/{c}']=dict(points=int(m.sum()),mean_px=float(per.mean()),per_seed_px=per.tolist(),difference=ci(allerr-baseerr,m,groups),support_distance_mean_px=float(selected[m].mean()),support_distance_median_px=float(np.median(selected[m])))
    np.savez_compressed(RUN/'quantization_geometry.npz',normal=n,rho=rho,index=index,distance=dist);write(RUN/'QUANTIZATION.json',dict(complete=True,exploratory=True,results=results,GT_line_oracle=True,files={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'quantization_geometry.npz',*RUN.glob('quantization_seed*.npz')]}));print('QUANTIZATION35',results['check/all'],flush=True)
if __name__=='__main__':main()
