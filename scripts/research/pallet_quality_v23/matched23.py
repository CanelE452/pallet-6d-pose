"""Posthoc point-ID and base-error-bin composition matching; never training or routing."""
from context23 import *
from train23 import training
from eval23 import population,prediction_path
from scipy.optimize import minimize_scalar

def stats(base,d,gt,weights):
    mask=weights>0;b=base[mask].astype(float);v=d[mask].astype(float);g=gt[mask].astype(float);w=weights[mask];before=np.linalg.norm(b-g,axis=-1);full=np.linalg.norm(b+v-g,axis=-1)
    def f(a):return float(np.average(np.linalg.norm(b+a*v-g,axis=-1),weights=w))
    z=minimize_scalar(f,bounds=(0,1),method='bounded');alpha=min((0.,1.,float(z.x)),key=f)
    return dict(points=int(mask.sum()),weight_sum=float(w.sum()),base_px=f(0),full_px=f(1),constant_optimal_alpha=alpha,constant_optimal_px=f(alpha),wrong_direction_fraction=float(np.average(((g-b)*v).sum(-1)<=0,weights=w)),full_harm_fraction=float(np.average(full>before+.1,weights=w)))
def main():
    setup();assert read(RUN/'SUMMARY.json')['complete'];rows=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=training(seed);b=r['base'].cpu().numpy();d=r['proposal_delta'].cpu().numpy();g=gt.cpu().numpy();v=valid.cpu().numpy();k=kind.cpu().numpy();del O,r,gt,valid,kind
        for c in ('crop20','crop40'):
            m,ts=population('real',c);p=np.load(prediction_path('real',c,f'unit_expand_seed{seed}'));rb=p['base'];rd=p['proposal_delta'];rg=np.array([t['points'] for t in ts]);rv=np.array([t['valid'] for t in ts])&p['detected'][:,None]&(C.P.target_type(ts).numpy()==2)
            tv=v&(k==2)&np.array([x.endswith('::'+c) for x in ids])[:,None];tbins=np.digitize(np.linalg.norm(b-g,axis=-1),[5,15],right=True);rbins=np.digitize(np.linalg.norm(rb-rg,axis=-1),[5,15],right=True)
            for match in ('point_id','point_id_and_error_bin','point_id_and_5px_bins'):
                edges=np.arange(5,101,5) if match=='point_id_and_5px_bins' else [5,15];tbins=np.digitize(np.linalg.norm(b-g,axis=-1),edges,right=True);rbins=np.digitize(np.linalg.norm(rb-rg,axis=-1),edges,right=True)
                tw=np.zeros_like(tv,dtype=float);rw=np.zeros_like(rv,dtype=float);cells=[]
                for point in range(8):
                    for bin0 in (range(len(edges)+1) if match!='point_id' else (None,)):
                        tm=tv[:,point]&(tbins[:,point]==bin0 if bin0 is not None else True);rm=rv[:,point]&(rbins[:,point]==bin0 if bin0 is not None else True);nt=int(tm.sum());nr=int(rm.sum());cells.append(dict(point=point,bin=bin0,train=nt,real=nr))
                        if nt and nr:tw[tm,point]=nr/nt;rw[rm,point]=1
                rows.append(dict(seed=seed,condition=c,matching=match,cells=cells,real_support_points=int(rv.sum()),matched_real_points=int((rw>0).sum()),training=stats(b,d,g,tw),real=stats(rb,rd,rg,rw)))
        print('MATCHED',seed,flush=True)
    write(RUN/'MATCHED_DIAGNOSIS.json',dict(rows=rows,scope='Posthoc real GT composition standardization for diagnosis only; no checkpoint, loss, input or inference was changed. Matches point ID and coarse or 5px error magnitude bins, not direction, pose, texture, scale, or exact error. Not sole-cause proof.'))
if __name__=='__main__':main()
