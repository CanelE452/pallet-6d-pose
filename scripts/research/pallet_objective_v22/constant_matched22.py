"""Posthoc train/real optimal constant policy with identical point-ID composition."""
from context22 import *
from scipy.optimize import minimize_scalar
from eval22 import population,prediction_path

def main():
    rows=[]
    for seed in SEEDS:
        view=torch.load(OLD.RUN/f'QUALITY_VIEWS_seed{seed}.pt',map_location='cpu',weights_only=True);base=view['bases'][0].numpy().astype(float);delta=view['deltas'][0].numpy().astype(float);ts=[];ids=[]
        for c in TRAIN_CONDITIONS:
            m=read(C.RUN/'cache/heldout'/c/'manifest.json');t=C.target_rows('heldout',c);ts.extend(x for x,r in zip(t,m['rows']) if r['detected']);ids.extend(r['id']+'::'+c for r in m['rows'] if r['detected'])
        assert ids==view['ids'];truth=np.array([x['points'] for x in ts]);valid=np.array([x['valid'] for x in ts]);outside=np.array([x['outside'] for x in ts]);conditions=np.array([x.rsplit('::',1)[1] for x in ids])
        for c in ('crop20','crop40'):
            m,rs=population('real',c);pred=np.load(prediction_path('real',c,f'unit_bce_seed{seed}'));det=pred['detected'];realgt=np.array([x['points'] for x in rs])[det];rv=(np.array([x['valid'] for x in rs])&np.array([x['outside'] for x in rs]))[det];weights=rv.sum(0)/rv.sum();sv=valid&outside&(conditions==c)[:,None];result={}
            for name,b,d,g,v in [('train',base,delta,truth,sv),('real',pred['base'][det].astype(float),pred['proposal_delta'][det].astype(float),realgt,rv)]:
                counts=v.sum(0);assert (counts[weights>0]>0).all();pointweights=weights/np.maximum(counts,1)
                def cost(alpha):return float((np.linalg.norm(b+alpha*d-g,axis=-1)*pointweights)[v].sum())
                opt=minimize_scalar(cost,bounds=(0,1),method='bounded');alpha=min((0.,float(opt.x),1.),key=cost);result[name]=dict(optimal_constant_alpha=alpha,optimal_constant_error_px=cost(alpha),base_error_px=cost(0),full_error_px=cost(1))
            rows.append(dict(seed=seed,condition=c,real_point_id_weights=weights.tolist(),matched=result))
    write(RUN/'CONSTANT_MATCHED_DIAGNOSIS.json',dict(status='PASS',rows=rows,scope='Posthoc GT diagnostic, never a deployed constant policy, training weight, or checkpoint selector. Point-ID matched but pose/scale/texture not matched.'));print('MATCHED CONSTANT PASS',flush=True)

if __name__=='__main__':main()
