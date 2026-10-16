from context43 import *
from experiment43 import observed,truth,route43

def main():
    p=population();part=partition(p);curves=read(RUN/'CURVES.json');assert curves['primary_sha256']==sha(RUN/'SUMMARY.json');count=0
    for seed in SEEDS:
        a,v,X=observed(p,seed);Y=truth(p,a,v);q=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'))
        for arm in ARMS:
          for kind in ROUTES:
            for split in ('fit','calibration','check'):
              take=part[split]
              for row in curves['rows'][str(seed)][arm+'_'+kind][split]:
                # Select independently from score/eligibility, then recompute point errors.
                prob=q['prob_'+arm];eligible=(prob>=row['threshold'])&a['valid']
                if kind=='probability':score=prob
                else:
                    amount=a['amount']/p['diag'][:,None,None];score=2*amount*v['mu_pred_segment_physical3_residual']-amount*amount*np.sum(a['normal']*a['normal'],axis=-1);eligible&=score>0
                idx=np.argmax(np.where(eligible,score,-np.inf),axis=-1);used=eligible.any(-1);pts=q['fallback'].copy();chosen=a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],idx];pts[used]=chosen[used];err=np.hypot(pts[...,0]-p['g']['gt'][...,0],pts[...,1]-p['g']['gt'][...,1]);delta=Y['base_error']-err;M=take[:,None];coverage=(eligible&Y['useful']).any(-1)&M
                assert abs(err[take].mean()-row['mean_px'])<1e-10
                assert int((coverage&q['hard']).sum())==row['hard_covered_points'];assert int(coverage.sum())==row['covered_points']
                assert int(((delta<-1)&M).sum())==row['harm_over1px'];assert int(((delta>1)&M).sum())==row['help_over1px']
                for c,z in row['conditions'].items():assert abs(err[take&(p['condition']==c)].mean()-z['mean_px'])<1e-10
                count+=1
    write(RUN/'DIAGNOSTIC_AUDIT.json',dict(status='PASS',fixed_grid_rows_independently_recalculated=count,primary_unchanged=True));print('DIAG AUDIT43',count,flush=True)
if __name__=='__main__':main()
