from context43 import *
from experiment43 import observed,truth,route43

def main():
    p=population();part=partition(p);summary=read(RUN/'SUMMARY.json');write(RUN/'DIAGNOSTIC_PROTOCOL.json',dict(primary_sha256=sha(RUN/'SUMMARY.json'),scope='Postprimary descriptive fixed-grid diagnostic, no fits or deployed policy changes. All18 prespecified thresholds reported, no bestcheck selection. Useful-target scores are not calibrated probabilities, especially weighted fits.',question='Which calibration requirements exclude each fixed threshold, and how do the same fixed thresholds behave by condition and useful opportunity on reusedcheck?'))
    rows={};oracle={}
    for seed in SEEDS:
        a,v,X=observed(p,seed);Y=truth(p,a,v);q=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));meta=read(RUN/'models'/f'seed{seed}.json');rows[str(seed)]={};oracle[str(seed)]={}
        for arm in ARMS:
          for kind in ROUTES:
            name=arm+'_'+kind;rows[str(seed)][name]={}
            for split in ('fit','calibration','check'):
                frame=part[split];w=frame_weights(p['groups'],frame)[:,None];rr=[]
                for th in np.round(np.arange(.1,.951,.05),2):
                    pts,sel,allow=route43(p,a,v,q['prob_'+arm],th,kind);err=np.linalg.norm(pts-p['g']['gt'],axis=-1);used=frame[:,None]&(sel>=0);idx=np.maximum(sel,0);ys=Y['valid'][np.arange(p['N'])[:,None],np.arange(8)[None],idx];sw=np.linalg.norm(pts-q['fallback'],axis=-1)>1e-8;coverage=(allow&Y['useful']).any(-1)&frame[:,None];gain=Y['base_error']-err;conds={}
                    for c in sorted(set(p['condition'][frame])):
                        mask=frame&(p['condition']==c);b=Y['base_error'][mask].mean();z=err[mask].mean();conds[c]=dict(mean_px=float(z),fallback_px=float(b),gain_percent=float(100*(1-z/b)),harm_over1px=int((gain[mask]<-1).sum()))
                    rr.append(dict(threshold=float(th),mean_px=float(err[frame].mean()),gain_px=float((gain[frame]*w).sum()/(8*w.sum())),precision=float(ys[used].mean()) if used.any() else None,switches=int((used&sw).sum()),help_over1px=int((used&(gain>1)).sum()),harm_over1px=int((used&(gain<-1)).sum()),hard_help_over1px=int((used&q['hard']&(gain>1)).sum()),hard_harm_over1px=int((used&q['hard']&(gain<-1)).sum()),covered_points=int(coverage.sum()),hard_covered_points=int((coverage&q['hard']).sum()),conditions=conds))
                rows[str(seed)][name][split]=rr
            # Exact reuse of stored calibration group-weighted pixel gain and selected precision.
            for old,new in zip(meta['thresholds'][name]['grid'],rows[str(seed)][name]['calibration']):
                assert abs(old['gain_px']-new['gain_px'])<1e-12
                assert old['precision']==(new['precision'] if new['precision'] is not None else 0.)
        for split in ('fit','calibration','check'):
            frame=part[split];allow=Y['useful']&a['valid'];candidateerror=np.where(allow,Y['candidate_error'],np.inf);best=np.minimum(Y['base_error'],candidateerror.min(-1));delta=Y['base_error']-best
            oracle[str(seed)][split]=dict(opportunity_points=int(((delta>1)&frame[:,None]).sum()),hard_opportunity_points=int(((delta>1)&frame[:,None]&q['hard']).sum()),best_mean_px=float(best[frame].mean()),fallback_mean_px=float(Y['base_error'][frame].mean()))
    write(RUN/'CURVES.json',dict(rows=rows,oracle=oracle,primary_sha256=sha(RUN/'SUMMARY.json')));print('DIAG43 COMPLETE',flush=True)
if __name__=='__main__':main()
